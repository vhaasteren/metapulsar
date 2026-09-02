"""Pulsar timing engine that assembles per-contribution engines in pulsar row order."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from typing import Any, Mapping

import numpy as np

from nltiming.protocols import GaugeProvenance, JaxTimingEngine, TimingEngine


@dataclass(frozen=True)
class PtaContribution:
    """One per-PTA contribution (row slice + engine) to a pulsar timing engine."""

    name: str
    row_indices: np.ndarray
    engine: TimingEngine
    exact_linear_fitpars: frozenset[str] = frozenset()
    fallback_reference_exact: Mapping[str, str] = field(default_factory=dict)


def _to_exact_str(value: str) -> str:
    with localcontext() as ctx:
        ctx.prec = 50
        return format(Decimal(value), "f")


class PulsarTimingEngine:
    """Canonical-row-order timing engine over per-contribution engines."""

    def __init__(
        self,
        *,
        fitpars: tuple[str, ...],
        nrows: int,
        contributions: list[PtaContribution],
        design_matrix: np.ndarray | None = None,
    ):
        self.fitpars = fitpars
        self.native_units = {name: "native" for name in fitpars}
        self._nrows = int(nrows)
        self._contributions = list(contributions)
        self._global_index = {name: i for i, name in enumerate(self.fitpars)}
        for contribution in self._contributions:
            unknown_exact_linear = [
                name
                for name in contribution.exact_linear_fitpars
                if name not in self._global_index
            ]
            if unknown_exact_linear:
                raise ValueError(
                    f"Contribution '{contribution.name}' declares exact-linear evaluation for "
                    f"unknown fitpars: {unknown_exact_linear}"
                )
        self._design_matrix = (
            None if design_matrix is None else np.asarray(design_matrix, dtype=float)
        )
        self._ref_exact = self._merge_reference_theta_exact()
        self.nonlinear_params = self._merge_nonlinear_params()

    def _merge_nonlinear_params(self) -> str | None:
        """The one hybrid residual mode every contribution honours.

        Leaves that carry no ``nonlinear_params`` attribute (linearized
        stand-ins, test doubles) count as native (``None``). A disagreement is
        a wiring bug, never something to average over.
        """
        modes = {
            contribution.name: getattr(contribution.engine, "nonlinear_params", None)
            for contribution in self._contributions
        }
        distinct = set(modes.values())
        if len(distinct) > 1:
            raise ValueError(
                "Contributions disagree on nonlinear_params: "
                + ", ".join(f"{k}={v!r}" for k, v in sorted(modes.items()))
            )
        return next(iter(distinct)) if distinct else None

    @property
    def contributions(self) -> list[PtaContribution]:
        """Per-PTA contributions in pulsar row order."""
        return list(self._contributions)

    def _leaf_identically_linear(self, contribution: PtaContribution) -> frozenset[str]:
        """What one contribution declares affine, leaf plus host-side columns.

        The leaf engine is the authority on its own waveform — including the
        axes a hybrid ``nonlinear_params`` mode moves onto the baked ``J @ δ``
        path, which are *not* in ``exact_linear_fitpars``. Names the host
        evaluates for the leaf (``-M δ``, the host-only exact-linear columns)
        are affine by construction.
        """
        declared = getattr(contribution.engine, "identically_linear_fitpars", None)
        leaf = frozenset(declared()) if declared is not None else frozenset()
        return (leaf | frozenset(contribution.exact_linear_fitpars)) & frozenset(
            self.fitpars
        )

    def identically_linear_fitpars(self) -> frozenset[str]:
        """Fitpars whose composite delay is affine in delta.

        A name qualifies only when *every* contribution that carries it is
        affine in it: the composite residual is the sum of the leaf blocks, so
        one nonlinear leg makes the composite axis nonlinear. Reported from the
        leaf engines rather than from ``PtaContribution.exact_linear_fitpars``,
        which is the residual-routing set (which columns the host evaluates)
        and misses hybrid-linearized JUG axes.
        """
        out: set[str] = set()
        for name in self.fitpars:
            carriers = [
                contribution
                for contribution in self._contributions
                if name in tuple(getattr(contribution.engine, "fitpars", ()))
                or name in contribution.exact_linear_fitpars
            ]
            if carriers and all(
                name in self._leaf_identically_linear(contribution)
                for contribution in carriers
            ):
                out.add(name)
        return frozenset(out)

    def binary_chart_capability(self, chart_family: str, suffix: str):
        """Forward the binary-chart capability to the contribution that owns
        this binary group (ownership split).

        Composite forwarding is nltiming-side: candidacy calls this on the whole
        pulsar engine, and we delegate to the leaf engine (``JugEngine`` /
        ``PintEngine``) of the contribution owning ``suffix``. Returns ``None``
        (→ candidacy uses its conservative pulsar/name-search fallback) when no
        contribution owns the group, the owner's leaf engine does not implement
        the query, or two contributions sharing an unsuffixed binary DISAGREE —
        we never guess across disagreeing owners. Leaf engines that lack the
        method (e.g. a JugEngine before its translator lands) therefore keep the
        whole group on the fallback, unchanged.
        """
        caps = []
        for contribution in self._contributions:
            cap_fn = getattr(contribution.engine, "binary_chart_capability", None)
            if cap_fn is None or not self._owns_binary_group(contribution, suffix):
                continue
            cap = cap_fn(chart_family, suffix)
            if cap is not None:
                caps.append(cap)
        if not caps:
            return None
        first = caps[0]
        if any(cap != first for cap in caps[1:]):
            return None  # shared-binary contributions disagree -> fall back
        return first

    @staticmethod
    def _owns_binary_group(contribution: PtaContribution, suffix: str) -> bool:
        fitpars = tuple(getattr(contribution.engine, "fitpars", ()))
        if suffix:
            return any(name.endswith(suffix) for name in fitpars)
        # Unsuffixed group: the contribution carries an unsuffixed binary
        # (Kepler or Laplace) coordinate. Multiple contributions may share it;
        # the disagreement guard above keeps that safe.
        binary = {"ECC", "OM", "T0", "EPS1", "EPS2", "TASC"}
        return any(name in binary for name in fitpars)

    def _merge_reference_theta_exact(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for contribution in self._contributions:
            ref = contribution.engine.reference_theta_exact()
            ref = dict(ref) | dict(contribution.fallback_reference_exact)
            for name in self.fitpars:
                if name not in ref:
                    continue
                exact = _to_exact_str(str(ref[name]))
                if name in merged and merged[name] != exact:
                    raise ValueError(
                        f"Shared fitpar '{name}' disagrees across contributions: "
                        f"{merged[name]} != {exact} (contribution={contribution.name})"
                    )
                merged[name] = exact
        for name in self.fitpars:
            if name not in merged:
                raise ValueError(
                    f"No contribution provides reference_theta_exact for '{name}'"
                )
        return merged

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self._ref_exact)

    def reference_theta(self) -> np.ndarray:
        return np.asarray(
            [float(self._ref_exact[name]) for name in self.fitpars], dtype=float
        )

    def _host_only_exact_linear_names(self, contribution: PtaContribution) -> list[str]:
        """Exact-linear axes owned by the host, not the leaf."""
        leaf_fitpars = set(contribution.engine.fitpars)
        return [
            name
            for name in contribution.exact_linear_fitpars
            if name not in leaf_fitpars
        ]

    def _host_only_exact_linear_numpy(
        self, contribution: PtaContribution, delta_theta: np.ndarray
    ) -> np.ndarray:
        host_only = [
            name
            for name in self._host_only_exact_linear_names(contribution)
            if delta_theta[self._global_index[name]] != 0.0
        ]
        rows = np.asarray(contribution.row_indices, dtype=int)
        if not host_only:
            return np.zeros(len(rows), dtype=float)
        if self._design_matrix is None:
            raise ValueError(
                f"Contribution '{contribution.name}' requires exact-linear evaluation for "
                f"{host_only}, but no pulsar design matrix was provided"
            )
        exact_delta = np.zeros(len(rows), dtype=float)
        for name in host_only:
            exact_delta -= (
                self._design_matrix[rows, self._global_index[name]]
                * delta_theta[self._global_index[name]]
            )
        return exact_delta

    def _contribution_local_delta(
        self, contribution: PtaContribution, delta_theta: np.ndarray
    ) -> np.ndarray:
        local = np.zeros(len(contribution.engine.fitpars), dtype=float)
        for i, name in enumerate(contribution.engine.fitpars):
            if name in self._global_index:
                local[i] = delta_theta[self._global_index[name]]
            else:
                raise ValueError(
                    f"Contribution '{contribution.name}' fitpar '{name}' is not a canonical pulsar fitpar"
                )
        return local

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta_theta, dtype=float)
        if delta.shape != (len(self.fitpars),):
            raise ValueError("delta_theta shape mismatch with fitpars")
        out = np.zeros(self._nrows, dtype=float)
        for contribution in self._contributions:
            local_delta = self._contribution_local_delta(contribution, delta)
            exact_linear = self._host_only_exact_linear_numpy(contribution, delta)
            block = (
                np.asarray(contribution.engine.residual_delta(local_delta), dtype=float)
                + exact_linear
            )
            out[np.asarray(contribution.row_indices, dtype=int)] = block
        return out

    def design_matrix(self, params: Any | None = None) -> np.ndarray:
        out = np.zeros((self._nrows, len(self.fitpars)), dtype=float)
        for contribution in self._contributions:
            block = np.asarray(
                contribution.engine.design_matrix(params=params), dtype=float
            )
            rows = np.asarray(contribution.row_indices, dtype=int)
            for local_j, name in enumerate(contribution.engine.fitpars):
                if name not in self._global_index:
                    raise ValueError(
                        f"Contribution '{contribution.name}' fitpar '{name}' is not a canonical pulsar fitpar"
                    )
                out[rows, self._global_index[name]] = block[:, local_j]
            if contribution.exact_linear_fitpars:
                host_only = self._host_only_exact_linear_names(contribution)
                if host_only and self._design_matrix is None:
                    raise ValueError(
                        f"Contribution '{contribution.name}' requires exact-linear evaluation but no "
                        "pulsar design matrix was provided"
                    )
                for name in host_only:
                    out[rows, self._global_index[name]] = self._design_matrix[
                        rows, self._global_index[name]
                    ]
        return out

    def gauge_provenance(self) -> GaugeProvenance:
        """Composites have no own provenance; readers use the context map."""
        raise AttributeError(
            "PulsarTimingEngine has no own gauge_provenance; use "
            "TimingSignal.gauge_provenance (one entry per contribution)"
        )

    @property
    def gauge_applied(self) -> bool:
        """OR over leaves — diagnostic only; never gates arithmetic."""
        return any(
            bool(getattr(c.engine, "gauge_applied", False)) for c in self._contributions
        )


class PulsarJaxTimingEngine(PulsarTimingEngine):
    """Pulsar timing engine with JAX-capable path and precision-critical union."""

    def __init__(
        self,
        *,
        fitpars: tuple[str, ...],
        nrows: int,
        contributions: list[PtaContribution],
        design_matrix: np.ndarray | None = None,
    ):
        super().__init__(
            fitpars=fitpars,
            nrows=nrows,
            contributions=contributions,
            design_matrix=design_matrix,
        )
        self._precision_union = frozenset().union(
            *[
                contribution.engine.precision_critical_fitpars()
                for contribution in contributions
                if isinstance(contribution.engine, JaxTimingEngine)
            ]
        )

    def residual_delta_jax(self, delta_theta):
        import jax.numpy as jnp

        delta = jnp.asarray(delta_theta)
        out = jnp.zeros((self._nrows,), dtype=delta.dtype)
        for contribution in self._contributions:
            if not isinstance(contribution.engine, JaxTimingEngine):
                raise ValueError(
                    f"Contribution '{contribution.name}' does not provide a JAX engine path"
                )
            local = jnp.zeros((len(contribution.engine.fitpars),), dtype=delta.dtype)
            for i, name in enumerate(contribution.engine.fitpars):
                if name in self._global_index:
                    local = local.at[i].set(delta[self._global_index[name]])
                else:
                    raise ValueError(
                        f"Contribution '{contribution.name}' missing global mapping for fitpar '{name}'"
                    )
            block = jnp.asarray(
                contribution.engine.residual_delta_jax(local), dtype=delta.dtype
            )
            host_only = self._host_only_exact_linear_names(contribution)
            if host_only:
                if self._design_matrix is None:
                    raise ValueError(
                        f"Contribution '{contribution.name}' requires exact-linear evaluation for "
                        f"{host_only}, but no pulsar design matrix was provided"
                    )
                rows = jnp.asarray(contribution.row_indices, dtype=int)
                exact = jnp.zeros((len(contribution.row_indices),), dtype=delta.dtype)
                design = jnp.asarray(self._design_matrix, dtype=delta.dtype)
                for name in host_only:
                    col = design[rows, self._global_index[name]]
                    exact = exact - col * delta[self._global_index[name]]
                block = block + exact
            out = out.at[jnp.asarray(contribution.row_indices, dtype=int)].set(block)
        return out

    def residual_jacobian(self) -> np.ndarray:
        """Cached ``jacfwd(residual_delta_jax)`` at the reference (zeros)."""
        cached = self.__dict__.get("_residual_jacobian_cache")
        if cached is not None:
            return cached
        import jax
        import jax.numpy as jnp

        zeros = jnp.zeros((len(self.fitpars),), dtype=jnp.float64)
        J = np.asarray(jax.jacfwd(self.residual_delta_jax)(zeros), dtype=float)
        self.__dict__["_residual_jacobian_cache"] = J
        return J

    def precision_critical_fitpars(self) -> frozenset[str]:
        return self._precision_union


def build_composite_engine(
    *,
    fitpars: tuple[str, ...],
    nrows: int,
    contributions: list[PtaContribution],
    design_matrix: np.ndarray | None = None,
) -> TimingEngine:
    """Return JAX-capable composite only when all contributions are JAX-capable."""
    all_jax = all(isinstance(s.engine, JaxTimingEngine) for s in contributions)
    cls = PulsarJaxTimingEngine if all_jax else PulsarTimingEngine
    return cls(
        fitpars=fitpars,
        nrows=nrows,
        contributions=contributions,
        design_matrix=design_matrix,
    )


def validate_composite_against_pulsar(engine, pulsar) -> None:
    """The composite consistency check: shapes, fitpars, zero delta, blocks.

    nltiming's ``validate_engine_against_pulsar`` demands
    ``engine.design_matrix() == pulsar.Mmat`` globally, which a composite
    cannot satisfy and, worse, satisfies *tautologically* when the composite's
    design matrix is the pulsar's own -- which is exactly what MetaPulsar was
    calling it with. It proved nothing.

    What can be checked, and is worth checking, is **block equality**: for
    every leg whose engine owns its columns, the pulsar's ``Mmat`` on that
    leg's rows and those columns must be the engine's own ``-J``, exactly. A
    leg served by a host matrix owns nothing to compare against and is skipped
    -- a composite may legitimately mix the two (SPEC III.4), and saying so is
    better than pretending the global check covers it.
    """
    from nltiming.engine_support import (
        validate_engine_shapes,
        validate_engine_zero_delta,
        validate_pulsar_surface,
    )

    validate_pulsar_surface(pulsar)
    validate_engine_shapes(engine)
    # 1e-9 s, the tolerance the call site this replaced already used. A
    # nonlinear leg re-runs its timing chain at delta = 0 rather than
    # returning a stored array, so `residual_delta(0)` is zero to the chain's
    # own reproducibility -- sub-nanosecond, not sub-picosecond. Tightening it
    # to 1e-12 refuses a correct JUG engine under `nonlinear_params="binary"`.
    validate_engine_zero_delta(engine, tol=1e-9)

    if tuple(pulsar.fitpars) != tuple(engine.fitpars):
        raise ValueError("Engine fitpars must match pulsar fitpars in canonical order")
    design = np.asarray(pulsar.Mmat, dtype=float)
    if design.shape[0] != len(pulsar.toas):
        raise ValueError("Engine row count must match pulsar rows")

    index = {name: i for i, name in enumerate(pulsar.fitpars)}
    for contribution in getattr(engine, "contributions", ()) or ():
        block = getattr(contribution.engine, "engine_design_block", None)
        if block is None:
            continue
        names, matrix = block()
        rows = np.asarray(contribution.row_indices, dtype=int)
        columns = [index[name] for name in names]
        if not np.array_equal(design[np.ix_(rows, columns)], matrix):
            raise ValueError(
                f"Contribution {contribution.name!r}: pulsar.Mmat is not the leg "
                f"engine's own -J on columns {list(names)}. The record and the "
                "engine disagree -- which is the thing building the leg on the "
                "timing side was supposed to make impossible."
            )
