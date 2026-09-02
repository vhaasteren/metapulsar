"""Per-PTA vela-jax timing engine.

vela-jax evaluates Vela.jl's component chain as a JAX function of the timing
parameters, over arrays frozen from either timing package: PINT reads a PINT
leg, tempo2 reads a tempo2 leg (clocks, INCLUDE trees, ``TRACK -2`` pulse
numbers), and the delay physics is the same either way. That is why one impl
token serves both packages -- the leg's own ``timing_package`` picks the
reader, not the physics.

Unlike :class:`~metapulsar.engines.vela.VelaEngine`, which calls ``SPNTA``
per delta, this leg is differentiable: it exposes ``residual_delta_jax``, so a
composite built from vela-jax legs is a ``PulsarJaxTimingEngine`` and NUTS can
sample the timing parameters directly rather than through a libstempo
callback.

Enterprise/Discovery still own EFAC/EQUAD/ECORR and the GPs; the par handed to
vela-jax is delay-only, stripped by the same classifier
:mod:`metapulsar.parfile_lines` uses.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from typing import Any, Mapping, NamedTuple

import numpy as np
from nltiming.engine_support import LinearModel, is_exact_linear_param
from nltiming.protocols import GaugeProvenance

from nltiming.hybrid import (
    is_hybrid_engine_axis,
    resolve_hybrid_partition,
    validate_nonlinear_params,
)

#: A row is identified by three columns, not one. Site arrival alone cannot
#: separate simultaneous sub-band TOAs, and frequency alone cannot separate
#: epochs; together they pin a row. The time tolerance is far above one ulp of
#: an MJD in seconds (9.5e-7 s at 5e9 s) and far below any TOA spacing, so it
#: catches a reordering or a different TOA set and nothing else.
ROW_TIME_TOLERANCE_S = 1e-3
ROW_FREQ_RTOL = 1e-6  # PINT and tempo2 agree on the barycentric Doppler to ~1e-9
ROW_ERR_RTOL = 1e-9

#: A different timescale moves a reference value by ~1.5e-8 relative (the IFTE
#: rate); a different par moves it by more. 1e-12 separates the two by four
#: orders while leaving room for the last digits of two independent TCB->TDB
#: conversions to differ.
REFERENCE_RTOL = Decimal("1e-12")


class TOARows(NamedTuple):
    """The three columns that identify a row; ``psrdata`` will own this type."""

    stoas: np.ndarray
    freqs: np.ndarray
    toaerrs: np.ndarray


def check_row_alignment(engine: Any, rows: TOARows | None) -> None:
    """Refuse a leg whose engine rows are not the composite's rows.

    vela-jax never reorders TOAs -- row ``i`` is the timing package's row
    ``i`` -- so an engine built from this leg's own par/tim lines up with the
    composite's rows by construction. "By construction" is worth exactly as
    much as a check, though, because on this path the engine is built from a
    *second* read of the files, and a filter, a clock difference or a
    canonicalized-vs-release tim would quietly give it different rows.

    ``rows=None`` skips the check. That is the factory path, where the leg
    engine *is* the record's engine and there is no second read to disagree
    with.
    """
    if rows is None:
        return
    signature = getattr(engine, "toa_rows", None)
    if signature is None:
        raise ValueError("rows were given but the engine exposes no toa_rows()")
    own = signature()
    expected = np.asarray(rows.stoas, dtype=float).reshape(-1)
    if own.stoas.shape != expected.shape:
        raise ValueError(
            f"vela-jax read {own.stoas.size} TOAs for this leg but the composite "
            f"has {expected.size} rows: the engine's read of the par/tim did not "
            "reproduce the leg the pulsar was built from"
        )
    checks = (
        ("stoas", own.stoas, expected, ROW_TIME_TOLERANCE_S, False),
        ("freqs", own.freqs, np.asarray(rows.freqs, dtype=float), ROW_FREQ_RTOL, True),
        (
            "toaerrs",
            own.toaerrs,
            np.asarray(rows.toaerrs, dtype=float),
            ROW_ERR_RTOL,
            True,
        ),
    )
    for name, mine, theirs, tol, relative in checks:
        # An infinite observing frequency is a real value on both sides.
        both_inf = np.isinf(mine) & np.isinf(theirs)
        with np.errstate(invalid="ignore"):
            diff = np.abs(mine - theirs)
        bound = tol * np.abs(theirs) if relative else tol
        bad = ~both_inf & ~(diff <= bound)
        if bad.any():
            i = int(np.argmax(bad))
            raise ValueError(
                f"vela-jax's TOAs are not the composite's rows for this leg: "
                f"{name}[{i}] engine={mine[i]!r} composite={theirs[i]!r}. Same "
                "TOAs in a different order, or a different set; either way the "
                "leg's residuals would line up against the wrong rows of the "
                "design matrix."
            )


def check_reference_agreement(
    engine: Any,
    linear_model: LinearModel,
    param_mapping: Mapping[str, str],
    engine_fitpars: tuple[str, ...],
) -> None:
    """The composite's theta* and the engine's theta* must be the same par.

    Aligned rows are not enough: the two sides can read the same TOAs and
    still expand around different reference values. For a tempo2 leg both ran
    a TCB->TDB conversion independently -- MetaPulsar's on the retained par,
    tempo2's own transform inside vela-jax -- so the last digits may differ,
    and a disagreement bigger than that is a different par or a different
    timescale. Either one is a silent bias in every delta the sampler takes.
    """
    engine_reference = engine.reference_theta_exact()
    with localcontext() as ctx:
        ctx.prec = 40
        for name in engine_fitpars:
            key = param_mapping.get(name, name)
            mine = Decimal(linear_model.theta_exact[name])
            theirs = Decimal(engine_reference[key])
            if not mine or not theirs:
                # An exact zero has no scale to be relative to -- a zeroed
                # parameter (PHOFF, and everything `prepare_model` zeroes) is
                # the common case. The absolute floor is deliberately *not*
                # applied when both sides are non-zero: on a small parameter
                # like EPS1 ~ 5e-6 a 1e-12 absolute floor would sit above the
                # 1.5e-8 relative shift a wrong timescale causes, and let it
                # through.
                bound = REFERENCE_RTOL
            else:
                bound = REFERENCE_RTOL * max(abs(mine), abs(theirs))
            if abs(mine - theirs) > bound:
                raise ValueError(
                    f"reference theta disagrees for {name!r} (engine {key!r}): "
                    f"composite {mine} vs vela-jax {theirs}. The leg engine was "
                    "built from a different par, or in a different timescale."
                )


class VelaJaxEngine:
    """Native vela-jax residual-delta engine for one PTA leg.

    Nonlinear residual deltas come from the vela-jax ``Engine``; the design
    matrix and reference-theta metadata are served from the pulsar-derived
    ``LinearModel`` so the composite sees the same canonical columns as the
    pulsar design matrix. Fit parameters the engine cannot evaluate
    (another PTA's JUMPs, Offset gauge columns) go to the exact-linear path.

    ``nonlinear_params`` (``None`` | ``"binary"`` | ``"binary+"``) is the same
    hybrid residual mode every MetaPulsar engine family executes: under a
    hybrid mode only the binary axes (plus ``PX`` for ``"binary+"``) reach the
    engine and every other fitpar is evaluated through its design-matrix
    column (see :mod:`.hybrid`).
    """

    engine_name = "vela_jax"

    def __init__(
        self,
        *,
        engine: Any,
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
        engine_fitpars: tuple[str, ...] | None = None,
        exact_linear_fitpars: frozenset[str] | set[str] | None = None,
        nonlinear_params: str | None = None,
        rows: TOARows | None = None,
    ):
        check_row_alignment(engine, rows)
        self._engine = engine
        self._model = linear_model
        self._param_mapping = dict(param_mapping or {})
        self.fitpars = tuple(linear_model.fitpars)
        self.native_units = dict(linear_model.native_units)
        self.nonlinear_params = validate_nonlinear_params(nonlinear_params)

        engine_fitpars, exact_linear_fitpars = resolve_hybrid_partition(
            fitpars=self.fitpars,
            param_mapping=self._param_mapping,
            mode=self.nonlinear_params,
            engine_fitpars=engine_fitpars,
            exact_linear_fitpars=exact_linear_fitpars,
        )
        self._engine_fitpars = tuple(engine_fitpars)
        self._exact_linear_fitpars = frozenset(exact_linear_fitpars)
        if rows is not None:
            check_reference_agreement(
                engine, linear_model, self._param_mapping, self._engine_fitpars
            )

        engine_index = {name: i for i, name in enumerate(engine.param_names)}
        self._engine_fitpar_indices = np.array(
            [self.fitpars.index(name) for name in self._engine_fitpars], dtype=int
        )
        self._engine_indices = np.array(
            [
                engine_index[self._param_mapping.get(name, name)]
                for name in self._engine_fitpars
            ],
            dtype=int,
        )
        self._exact_linear_indices = np.array(
            sorted(self.fitpars.index(name) for name in self._exact_linear_fitpars),
            dtype=int,
        )

    @classmethod
    def from_files(
        cls,
        par_file,
        tim_file,
        *,
        timing_package: str = "pint",
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
        nonlinear_params: str | None = None,
        binary_conventions: str = "pint",
        engine_kwargs: Mapping[str, Any] | None = None,
        rows: TOARows | None = None,
    ) -> "VelaJaxEngine":
        """Build a leg engine straight from the retained par/tim inputs.

        ``timing_package`` is the leg's own timing package: ``"tempo2"`` legs are read by
        tempo2, ``"pint"`` legs by PINT. ``binary_conventions`` stays Vela's by
        default on both, so every leg of one pulsar uses one residual formula
        unless the caller deliberately asks otherwise.

        This is the **from-par/tim path**: it reads the par/tim itself, a
        *second* time, so it takes ``rows`` and runs both guards. The composite
        already holds a materialized pulsar for this leg; the two agree only if
        the second read produced the same TOAs in the same order, expanded
        around the same reference values -- and a filter, a clock difference or
        a canonicalized-vs-release tim would break either. Pass ``rows``, the
        leg's ``(stoas, freqs, toaerrs)`` in the composite's row order, and a
        mismatch becomes a refusal instead of a plausible, wrong likelihood.

        The factory path (``create_metapulsar(engines=...)``) builds the leg
        from the record's own engine and passes ``rows=None``: same object,
        same freeze, nothing to re-read. Both guards stay -- they are not dead
        code, they are this path's contract.
        """
        from vela_jax import Engine

        kwargs = dict(engine_kwargs or {})
        kwargs["binary_conventions"] = binary_conventions
        engine = Engine.from_files(
            str(par_file),
            str(tim_file),
            timing_package=timing_package,
            **kwargs,
        )
        return cls.from_engine(
            engine,
            linear_model=linear_model,
            param_mapping=param_mapping,
            nonlinear_params=nonlinear_params,
            rows=rows,
        )

    @classmethod
    def from_engine(
        cls,
        engine: Any,
        *,
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
        nonlinear_params: str | None = None,
        rows: TOARows | None = None,
    ) -> "VelaJaxEngine":
        """Partition the pulsar fitpars against an already-built vela-jax engine."""
        mapping = dict(param_mapping or {})
        mode = validate_nonlinear_params(nonlinear_params)
        settable = set(engine.param_names)

        engine_fitpars: list[str] = []
        exact_linear: list[str] = []
        for name in tuple(linear_model.fitpars):
            engine_param = mapping.get(name, name)
            if (
                is_exact_linear_param(engine_param)
                or engine_param not in settable
                or not is_hybrid_engine_axis(engine_param, mode)
            ):
                exact_linear.append(name)
                continue
            engine_fitpars.append(name)

        # A hybrid mode on a pulsar without binary axes legitimately degenerates
        # to the pure ``-M delta`` residual, as it does for every other family.
        if not engine_fitpars and mode is None:
            raise ValueError(
                "No vela-jax-evaluable fit parameters remain after filtering; "
                f"exact-linear candidates: {exact_linear}"
            )
        return cls(
            engine=engine,
            linear_model=linear_model,
            param_mapping=mapping,
            engine_fitpars=tuple(engine_fitpars),
            exact_linear_fitpars=frozenset(exact_linear),
            nonlinear_params=mode,
            rows=rows,
        )

    # --- residuals ---------------------------------------------------------

    def _check(self, delta_theta) -> np.ndarray:
        delta = np.asarray(delta_theta, dtype=float).reshape(-1)
        if delta.shape != (len(self.fitpars),):
            raise ValueError("delta_theta shape mismatch with fitpars")
        return delta

    def residual_delta(self, delta_theta) -> np.ndarray:
        delta = self._check(delta_theta)
        step = np.zeros(len(self._engine.param_names))
        step[self._engine_indices] = delta[self._engine_fitpar_indices]
        native = np.asarray(self._engine.residual_delta(step), dtype=float)
        return native + self._exact_linear_delta(delta)

    def residual_delta_jax(self, delta_theta):
        """The differentiable path: the leg's own JAX residual, plus -M delta."""
        import jax.numpy as jnp

        delta = jnp.asarray(delta_theta)
        step = jnp.zeros((len(self._engine.param_names),), dtype=delta.dtype)
        step = step.at[self._engine_indices].set(delta[self._engine_fitpar_indices])
        native = self._engine.residual_delta_jax(step)
        if not len(self._exact_linear_indices):
            return native
        columns = jnp.asarray(
            self._model.design[:, self._exact_linear_indices], dtype=delta.dtype
        )
        return native - columns @ delta[self._exact_linear_indices]

    def _exact_linear_delta(self, delta: np.ndarray) -> np.ndarray:
        if not len(self._exact_linear_indices):
            return np.zeros(self.design_matrix().shape[0], dtype=float)
        columns = np.asarray(
            self._model.design[:, self._exact_linear_indices], dtype=float
        )
        return -(columns @ delta[self._exact_linear_indices])

    def residual_jacobian(self) -> np.ndarray:
        import jax
        import jax.numpy as jnp

        zeros = jnp.zeros((len(self.fitpars),), dtype=jnp.float64)
        return np.asarray(jax.jacfwd(self.residual_delta_jax)(zeros), dtype=float)

    # --- metadata ----------------------------------------------------------

    def engine_design_block(self) -> tuple[tuple[str, ...], np.ndarray]:
        """``(fitpar names, the engine's own -J columns)`` for the axes it evaluates.

        What the composite consistency check compares against
        ``pulsar.Mmat``. Exact equality is the right bar, not a tolerance: on
        a leg the timing side emitted, the composite's columns were *copied*
        from the record the engine produced, so anything but equality means
        they came from somewhere else.
        """
        return self._engine_fitpars, np.asarray(
            self._engine.design_matrix()[:, self._engine_indices], dtype=float
        )

    def gauge_direction(self) -> np.ndarray:
        """The gauge direction of the engine underneath, for nltiming's check."""
        return np.asarray(self._engine.gauge_direction(), dtype=float)

    def exact_linear_fitpars(self) -> frozenset[str]:
        """Pulsar fitpars evaluated exactly via the design matrix."""
        return self._exact_linear_fitpars

    def identically_linear_fitpars(self) -> frozenset[str]:
        """Fitpars whose engine delay is affine in delta."""
        engine_linear = self._engine.identically_linear_params()
        native = {
            name
            for name in self._engine_fitpars
            if self._param_mapping.get(name, name) in engine_linear
        }
        return self._exact_linear_fitpars | native

    def precision_critical_fitpars(self) -> frozenset[str]:
        critical = self._engine.precision_critical_params()
        return frozenset(
            name
            for name in self._engine_fitpars
            if self._param_mapping.get(name, name) in critical
        )

    def binary_chart_capability(self, chart_family: str, suffix: str):
        from vela_jax.backend import VelaJaxTimingEngine

        return VelaJaxTimingEngine(self._engine).binary_chart_capability(
            chart_family, suffix
        )

    def reference_theta(self) -> np.ndarray:
        return self._model.reference_theta()

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self._model.theta_exact)

    def design_matrix(self, params=None) -> np.ndarray:
        return np.asarray(self._model.design, dtype=float)

    def gauge_provenance(self) -> GaugeProvenance:
        return GaugeProvenance(
            export="none",
            reference_mode="none",
            reporting_mode="mean",
            reporting_weighted=True,
        )

    @property
    def gauge_applied(self) -> bool:
        return self.gauge_provenance().export != "none"

    def __repr__(self) -> str:
        return (
            f"<VelaJaxEngine timing_package={self._engine.timing_package} "
            f"engine_fitpars={list(self._engine_fitpars)}>"
        )
