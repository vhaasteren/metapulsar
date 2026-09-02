"""Per-PTA JUG timing engine."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np

from nltiming.protocols import GaugeProvenance

from .delta import infer_jug_param_mapping
from nltiming.hybrid import hybrid_linearized_fitpars
from nltiming.engine_support import (
    LinearModel,
    LinearTimingEngine,
    is_exact_linear_param,
)

_ECLIPTIC_FITPARS = frozenset(
    {
        "ELONG",
        "ELAT",
        "PMELONG",
        "PMELAT",
        "LAMBDA",
        "BETA",
        "PMLAMBDA",
        "PMBETA",
    }
)


def _resolve_binary_facts(session: Any, fit_params: tuple[str, ...]):
    """Compute JUG's binary-chart facts from a session, degrading to ``None``
    (→ candidacy fallback) on any resolution failure — an unsupported or absent
    binary must never break engine construction. Unexpected failures are logged
    at debug level so genuine bugs are discoverable rather than silent."""
    try:
        from jug.fitting.binary_delay_plan import binary_chart_facts

        params = getattr(session, "params", None)
        if not params:
            return None
        return binary_chart_facts(params, list(fit_params))
    except Exception:  # never break engine construction; surface for debugging
        logging.getLogger(__name__).debug(
            "binary_chart_facts resolution failed; using the capability "
            "fallback for this engine",
            exc_info=True,
        )
        return None


def _reporting_from_compatibility(compatibility: str) -> tuple[str, bool | None]:
    """Host reporting convention (engine_conventions.py:303)."""
    if str(compatibility).lower().startswith("tempo2"):
        return "mean", False
    return "mean", True


def _provenance_from_frozen_model(state: Any) -> GaugeProvenance:
    """Translate JUG ``ReferenceGauge`` into backend-neutral provenance."""
    reference_gauge = getattr(state, "reference_gauge", None)
    compatibility = str(getattr(state, "compatibility", "pint"))
    reporting_mode, reporting_weighted = _reporting_from_compatibility(compatibility)
    if reference_gauge is None:
        return GaugeProvenance(
            export="none",
            reference_mode="none",
            reporting_mode=reporting_mode,
            reporting_weighted=reporting_weighted,
        )
    mode = str(getattr(reference_gauge, "mode", "none"))
    weights = getattr(reference_gauge, "weights", None)
    reference_weighted = None if mode != "mean" else (weights is not None)
    return GaugeProvenance(
        export="none",
        reference_mode=mode,  # type: ignore[arg-type]
        reference_weighted=reference_weighted,
        reporting_mode=reporting_mode,
        reporting_weighted=reporting_weighted,
    )


class JugEngine:
    """Native JUG engine with NumPy and pure-JAX residual-delta paths.

    The nonlinear residuals are evaluated by a frozen JUG ``FrozenResidualModel``.
    ``design_matrix`` and reference theta metadata are intentionally served from
    the pulsar-derived ``LinearModel`` so the pulsar timing engine uses the same
    canonical columns and analytically marginalized basis as ``MetaPulsar.Mmat``.

    Unit convention: this engine's entire external surface — ``design_matrix``,
    ``residual_jacobian``, ``residual_delta`` and ``residual_delta_jax`` —
    speaks the **pulsar fit-unit** ``delta_theta`` convention that ``MetaPulsar.Mmat``
    (and the libstempo/Velan engines) use, e.g. RAJ in hourangle and DECJ in
    degrees. The frozen model is internally **native** (RAJ/DECJ in radians), so
    incoming deltas are divided by the per-parameter fit/native factor before
    reaching the state, and JUG's native residual-Jacobian columns are divided by
    the same factor on the way out. With gauge-free residuals the fitter identity
    ``residual_delta(delta) ≈ -(design_matrix @ delta)`` holds in the linear
    regime under either ``derivative_method``.
    """

    engine_name = "jug"

    def __init__(
        self,
        *,
        state: Any,
        linear_model: LinearModel,
        precision_critical: frozenset[str] | set[str] | None = None,
    ):
        self._state = state
        self._model = linear_model
        self.fitpars = tuple(linear_model.fitpars)
        self.native_units = dict(linear_model.native_units)
        self._precision_critical = frozenset(precision_critical or frozenset())
        # Defaults for direct construction: every fitpar is JUG-evaluable and no
        # exact-linear column is delegated to the pulsar design matrix.
        self._exact_linear_fitpars: frozenset[str] = frozenset()
        self._exact_linear_indices: tuple[int, ...] = tuple()
        self._jug_fitpars: tuple[str, ...] = self.fitpars
        self._jug_indices: tuple[int, ...] = tuple(range(len(self.fitpars)))
        # Hybrid residual mode (``jug.fitting.nonlinear_params``) and the JUG
        # axes it moves onto the baked ``J @ δ`` path; both empty for direct
        # construction.
        self.nonlinear_params: str | None = getattr(state, "nonlinear_params", None)
        self._hybrid_linear_fitpars: frozenset[str] = frozenset()
        # Authoritative Kepler↔Laplace chart facts, resolved once from
        # the JUG session at from_contribution time; None for direct/test-double
        # construction (→ candidacy fallback).
        self._binary_facts: Any = None
        self.compatibility = str(getattr(state, "compatibility", "auto"))

    @classmethod
    def from_contribution(
        cls,
        session: Any,
        *,
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
        subtract_tzr: bool = True,
        nonlinear_params: str | None = None,
    ) -> "JugEngine":
        """Build a native JUG engine from an already-created JUG session.

        ``nonlinear_params`` (when provided) sets the session residual-
        linearization mode before export. When omitted, the session's existing
        ``nonlinear_params`` is left unchanged.
        """
        from jug.fitting.residual_model import export_frozen_residual_model

        if nonlinear_params is not None:
            from jug.fitting.nonlinear_params import validate_nonlinear_params

            session.nonlinear_params = validate_nonlinear_params(nonlinear_params)

        fitpars = tuple(linear_model.fitpars)
        mapping = dict(param_mapping or {})
        if not mapping:
            mapping = infer_jug_param_mapping(
                fitpars, set(getattr(session, "params", {}).keys())
            )

        from jug.model.parameter_spec import validate_fit_param

        jug_fitpars: list[str] = []
        exact_linear: list[str] = []
        for name in fitpars:
            engine_param = mapping.get(name, name)
            if is_exact_linear_param(engine_param):
                exact_linear.append(name)
                continue
            try:
                validate_fit_param(engine_param)
            except ValueError:
                exact_linear.append(name)
                continue
            jug_fitpars.append(name)

        if not jug_fitpars:
            raise ValueError(
                "No JUG-evaluable fit parameters remain after filtering; "
                f"exact-linear candidates: {exact_linear}"
            )

        state = export_frozen_residual_model(
            session,
            fit_params=tuple(jug_fitpars),
            subtract_tzr=subtract_tzr,
            param_mapping=mapping,
        )
        engine = cls(
            state=state,
            linear_model=linear_model,
            precision_critical=_canonical_high_precision(tuple(jug_fitpars), mapping),
        )
        engine._exact_linear_fitpars = frozenset(exact_linear)
        engine._exact_linear_indices = tuple(
            fitpars.index(name) for name in exact_linear
        )
        engine._jug_fitpars = tuple(jug_fitpars)
        engine._jug_indices = tuple(fitpars.index(name) for name in jug_fitpars)
        engine.compatibility = str(getattr(state, "compatibility", "auto"))
        engine.nonlinear_params = getattr(
            state, "nonlinear_params", getattr(session, "nonlinear_params", None)
        )
        # Under a hybrid mode JUG evaluates the non-binary (non-PX for
        # "binary+") axes as ``J @ δ`` — affine in delta — so report them as
        # identically linear, matching what the libstempo/Vela/PINT adapters
        # report for the same mode.
        engine._hybrid_linear_fitpars = hybrid_linearized_fitpars(
            tuple(jug_fitpars), mapping, engine.nonlinear_params
        )
        engine._binary_facts = _resolve_binary_facts(session, tuple(jug_fitpars))
        return engine

    def binary_chart_capability(self, chart_family: str, suffix: str):
        """Authoritative binary-chart capability, translated from JUG's binary facts
        (ownership split: JUG owns the physics, this is a thin
        translator). ``None`` (→ candidacy fallback) when the family is not ours
        or JUG could not resolve the binary. ``origin_certified`` /
        ``supports_domain`` are nltiming-owned, not backend facts.

        ``suffix`` is intentionally ignored here: a leaf engine describes the one
        binary of its single contribution, so per-suffix routing across a
        composite pulsar is the composite engine's job
        (``PulsarTimingEngine.binary_chart_capability``), not the leaf's.
        """
        if chart_family != "kepler_laplace":
            return None
        facts = getattr(self, "_binary_facts", None)
        if facts is None:
            return None
        from nltiming.protocols import BinaryChartCapability

        return BinaryChartCapability(
            kepler_convention=facts.convention_family,  # 'dd' | 'ell1' | 'other'
            epoch_shift_exact=bool(facts.epoch_shift_exact),
            secular_terms=tuple(facts.secular_terms),
            origin_certified=False,  # flip only via a passing origin-cert PR
            supports_domain=True,
        )

    def exact_linear_fitpars(self) -> frozenset[str]:
        """Pulsar fitpars evaluated exactly via the design matrix."""
        return getattr(self, "_exact_linear_fitpars", frozenset())

    def identically_linear_fitpars(self) -> frozenset[str]:
        """Fitpars whose engine delay is affine in delta.

        The exact-linear (design-matrix) axes, plus — under a hybrid
        ``nonlinear_params`` mode — the JUG axes the hybrid formula evaluates
        as ``J @ δ``.
        """
        return getattr(self, "_exact_linear_fitpars", frozenset()) | getattr(
            self, "_hybrid_linear_fitpars", frozenset()
        )

    @property
    def _native_scale(self) -> np.ndarray:
        """Per-JUG-fitpar fit-unit -> native-delta factor (see class docstring).

        Derived live from ``_jug_fitpars`` (and cached against it) so it stays
        correct even when the JUG partition is adjusted after construction.
        """
        key = self._jug_fitpars
        cached = self.__dict__.get("_native_scale_cache")
        if cached is not None and cached[0] == key:
            return cached[1]
        scale = _native_delta_scale(key)
        self.__dict__["_native_scale_cache"] = (key, scale)
        return scale

    def _jug_delta(self, delta_theta) -> np.ndarray:
        delta = np.asarray(delta_theta, dtype=float).reshape(-1)
        if delta.shape != (len(self.fitpars),):
            raise ValueError("delta_theta shape mismatch with fitpars")
        return delta[np.asarray(self._jug_indices, dtype=int)]

    def _jug_delta_native(self, delta_theta) -> np.ndarray:
        """JUG-evaluable delta in the state's native units (fit -> native)."""
        return self._jug_delta(delta_theta) / self._native_scale

    def reference_theta(self) -> np.ndarray:
        return self._model.reference_theta()

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self._model.theta_exact)

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        return np.asarray(self.residual_delta_jax(delta_theta), dtype=float)

    def design_matrix(self, params: Any | None = None) -> np.ndarray:
        return np.asarray(self._model.design, dtype=float)

    def residual_jacobian(self) -> np.ndarray:
        """J in pulsar fit units, gauge-free, cached on the frozen model."""
        cached = self.__dict__.get("_residual_jacobian_cache")
        if cached is not None:
            return cached

        J = -np.asarray(self._model.design, dtype=float).copy()
        jug_matrix = self._native_residual_jacobian()
        param_mapping = dict(getattr(self._state, "param_mapping", ()))
        jug_fitpars = getattr(self, "_jug_fitpars", self.fitpars)
        for local_col, model_col in enumerate(self._jug_indices):
            # Native residual Jacobian → fit units: divide by fit/native factor.
            J[:, model_col] = jug_matrix[:, local_col] / self._native_scale[local_col]
            canonical = jug_fitpars[local_col]
            engine = param_mapping.get(canonical, canonical)
            if (
                engine.upper() in _ECLIPTIC_FITPARS
                or canonical.upper() in _ECLIPTIC_FITPARS
            ):
                col_norm = float(np.linalg.norm(jug_matrix[:, local_col]))
                if col_norm < 1e-30:
                    raise ValueError(
                        f"JUG residual-jacobian column for {canonical!r} is "
                        "numerically zero; ecliptic sync may be broken. Use "
                        "derivative_method='analytic' or exact_linear for "
                        "ecliptic params until fixed."
                    )
        self.__dict__["_residual_jacobian_cache"] = J
        return J

    def _native_residual_jacobian(self) -> np.ndarray:
        fn = getattr(self._state, "residual_jacobian_native", None)
        if callable(fn):
            return np.asarray(fn(), dtype=float)
        # No design_matrix attribute fallback: that name means M, not J.
        # Test doubles and FrozenResidualModel must expose residual_jacobian_native.
        import jax
        import jax.numpy as jnp

        scale = jnp.asarray(self._native_scale, dtype=jnp.float64)
        zeros = jnp.zeros((len(self._jug_fitpars),), dtype=jnp.float64)

        def residual_of_native(native_delta):
            full = jnp.zeros((len(self.fitpars),), dtype=native_delta.dtype)
            full = full.at[jnp.asarray(self._jug_indices, dtype=int)].set(
                native_delta * scale
            )
            return self.residual_delta_jax(full)

        return np.asarray(jax.jacfwd(residual_of_native)(zeros), dtype=float)

    def residual_delta_jax(self, delta_theta: Any) -> Any:
        import jax.numpy as jnp

        delta = jnp.asarray(delta_theta)
        scale = jnp.asarray(self._native_scale, dtype=delta.dtype)
        jug_delta = delta[jnp.asarray(self._jug_indices, dtype=int)] / scale
        nonlinear = self._state.residual_delta_jax(jug_delta)

        indices = getattr(self, "_exact_linear_indices", tuple())
        if not indices:
            return nonlinear
        design = jnp.asarray(self._model.design[:, list(indices)], dtype=delta.dtype)
        fallback_delta = delta[jnp.asarray(indices, dtype=int)]
        return nonlinear - design @ fallback_delta

    def precision_critical_fitpars(self) -> frozenset[str]:
        return self._precision_critical

    def gauge_provenance(self) -> GaugeProvenance:
        return _provenance_from_frozen_model(self._state)

    @property
    def gauge_applied(self) -> bool:
        return self.gauge_provenance().export != "none"

    def verify_native_chain(self) -> None:
        fn = getattr(self._state, "verify_native_chain", None)
        if callable(fn):
            fn()


def _native_delta_scale(jug_fitpars: tuple[str, ...]) -> np.ndarray:
    """Per-fitpar factor converting pulsar fit-unit deltas to JUG native units.

    ``delta_native = delta_fit / factor``; the factor is
    ``jug.utils.units.native_to_fit_value(name, 1.0)`` — JUG's own authoritative
    conversion (``HOURANGLE_PER_RAD`` for RAJ, ``RAD_TO_DEG`` for DECJ, 1.0
    otherwise). Falls back to all-ones if JUG is unavailable (e.g. test doubles
    with a fake state), which leaves non-astrometry parameters unchanged.
    """
    try:
        from jug.utils.units import native_to_fit_value
    except Exception:
        return np.ones(len(jug_fitpars), dtype=float)
    return np.array(
        [float(native_to_fit_value(name, 1.0)) for name in jug_fitpars],
        dtype=float,
    )


def _canonical_high_precision(
    fitpars: tuple[str, ...], mapping: Mapping[str, str]
) -> frozenset[str]:
    """Return canonical fitpars JUG marks as high-precision."""
    try:
        from jug.model.parameter_spec import get_high_precision_params
    except Exception:
        return frozenset()

    backend_high_precision = set(get_high_precision_params())
    return frozenset(
        name for name in fitpars if mapping.get(name, name) in backend_high_precision
    )


def verify_jug_native_chain(engine: object) -> None:
    """Smoke-check tempo2 JUG contributions export native-chain JAX state."""
    verify = getattr(engine, "verify_native_chain", None)
    if callable(verify):
        verify()
        return
    contributions = getattr(engine, "contributions", None) or getattr(
        engine, "_contributions", ()
    )
    for contribution in contributions:
        leaf = contribution.engine
        leaf_verify = getattr(leaf, "verify_native_chain", None)
        if callable(leaf_verify):
            leaf_verify()


class LinearizedJugEngine(LinearTimingEngine):
    """Explicit linearized JUG test double with JAX-capable surface."""

    engine_name = "jug"

    def __init__(
        self,
        model: LinearModel,
        *,
        gauge_provenance: GaugeProvenance,
        compatibility: str = "auto",
        precision_critical: frozenset[str] | set[str] = frozenset(),
    ):
        super().__init__(model, gauge_provenance=gauge_provenance)
        self.compatibility = compatibility
        self._precision_critical = frozenset(precision_critical)

    @classmethod
    def from_linear_model(
        cls,
        model: LinearModel,
        *,
        gauge_provenance: GaugeProvenance | None = None,
        compatibility: str = "auto",
        precision_critical: frozenset[str] | set[str] = frozenset(),
    ) -> "LinearizedJugEngine":
        if gauge_provenance is None:
            reporting_mode, reporting_weighted = _reporting_from_compatibility(
                compatibility
            )
            # Export is gauge-free; reference gauge of the host path is unknown
            # for a bare LinearModel (do not claim reference_mode="none").
            gauge_provenance = GaugeProvenance(
                export="none",
                reference_mode="unknown",
                reporting_mode=reporting_mode,
                reporting_weighted=reporting_weighted,
            )
        return cls(
            model,
            gauge_provenance=gauge_provenance,
            compatibility=compatibility,
            precision_critical=precision_critical,
        )

    def residual_delta_jax(self, delta_theta: Any) -> Any:
        import jax.numpy as jnp

        design = jnp.asarray(self.design_matrix(), dtype=jnp.asarray(delta_theta).dtype)
        delta = jnp.asarray(delta_theta)
        return -(design @ delta)

    def precision_critical_fitpars(self) -> frozenset[str]:
        return self._precision_critical
