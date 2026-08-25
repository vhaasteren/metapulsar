"""Build the flexfit timing block from an ``nltiming`` ``TimingSignal``.

The timing directions enter the joint basis in ``nltiming``'s prior-transformed
``z`` coordinates. This adapter splits the timing model exactly as ``nltiming``'s
own timing plan does:

* **analytically marginalized** columns become a single broad-prior block built
  from the column-normalized design matrix, held fixed at ``1e40`` and never
  updated — reproducing Discovery's ``makegp_improper`` timing marginalization;
* **sampled** columns become a learnable ``J_z`` block whose per-parameter
  variance is fixed at ``1e40`` on the first sweep and then inferred.

Sign convention (mandated finite-difference check in :func:`sign_check`):
timing blocks are built from the residual Jacobian ``J = -M``
(``J = ∂(Δr)/∂δ`` under the fitter contract
``r(θ+δ) ≈ r(θ) - M δ``). The block matrix is therefore
``J_z = dr/dz``; the fitted coefficient ``m`` still implies a physical step
``dz = -damping * m`` (Discovery's ``detres`` convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from ..basis import BasisBlock, VarianceGroup, rho_bounds_from_rms
from ..timing import TimingModel

# Sentinel: groups with this update_from_sweep are never touched by the M-step.
NEVER_UPDATE = 10**9

# Wide default induced-RMS bounds (seconds) for a learnable timing coordinate.
DEFAULT_TIMING_SIGMA_MIN = 1.0e-12
DEFAULT_TIMING_SIGMA_MAX = 1.0e0


def _normalized(matrix: np.ndarray) -> np.ndarray:
    from nltiming.whitening import normalized_basis

    return np.asarray(normalized_basis(matrix), dtype=float)


def _delta_from_z(space, z: np.ndarray) -> np.ndarray:
    return np.asarray(space.delta_from_z(np.asarray(z, dtype=float), np), dtype=float)


def _d_delta_d_z(space, z: np.ndarray) -> np.ndarray:
    return np.asarray(
        space.prior_bijector.jacobian_diag_delta_from_z(np.asarray(z, dtype=float), np),
        dtype=float,
    )


@dataclass(frozen=True)
class NltimingTimingModel(TimingModel):
    """A ``TimingModel`` backed by an ``nltiming`` ctx, anchored at ``z``."""

    pulsar: Any
    backend: Any
    space: Any
    sampled_names: tuple[str, ...]
    sampled_indices: tuple[int, ...]
    marg_indices: tuple[int, ...]
    design_matrix: np.ndarray  # M (fitter sign); not used as a residual Jacobian
    residual_jacobian: np.ndarray  # J = -M; used for blocks and sign_check
    base_residuals: np.ndarray
    z_anchor: np.ndarray
    initial_variance: float
    sample_update_from_sweep: int
    sample_sigma_min: float
    sample_sigma_max: float
    marg_name: str = "timing_marg"
    sampled_name: str = "timing"
    _evaluator: Any = field(default=None, repr=False)

    # --- TimingModel protocol ----------------------------------------------
    @property
    def sampled_block(self) -> str | None:
        return self.sampled_name if self.sampled_names else None

    def residuals(self) -> np.ndarray:
        if not np.any(self.z_anchor):
            return np.asarray(self.base_residuals, dtype=float)
        delta_full = self._full_delta(self.z_anchor)
        residual_delta = np.asarray(
            self.backend.residual_delta(delta_full), dtype=float
        )
        return np.asarray(self.base_residuals, dtype=float) + residual_delta

    def blocks(self) -> tuple[BasisBlock, ...]:
        out: list[BasisBlock] = []
        if self.marg_indices:
            out.append(self._marginalized_block())
        if self.sampled_names:
            out.append(self._sampled_block())
        if not out:
            raise ValueError(
                "timing model has neither sampled nor marginalized columns"
            )
        return tuple(out)

    def advance(
        self, sampled_mean: np.ndarray, *, damping: float
    ) -> "TimingModel | None":
        # Relinearization is only meaningful with an autodiff evaluator to rebuild
        # J_z at the new anchor; a linear single pass returns None.
        if self._evaluator is None or not self.sampled_names:
            return None
        dz = -float(damping) * np.asarray(sampled_mean, dtype=float)
        return replace(self, z_anchor=np.asarray(self.z_anchor, dtype=float) + dz)

    def summary(self, sampled_mean: np.ndarray) -> Mapping[str, object]:
        sampled_mean = np.asarray(sampled_mean, dtype=float)
        if sampled_mean.size == 0:
            z_est = np.asarray(self.z_anchor, dtype=float)
        else:
            # Apply the pending (un-committed) step so the estimate reflects the fit.
            z_est = np.asarray(self.z_anchor, dtype=float) - sampled_mean
        delta = _delta_from_z(self.space, z_est) if self.sampled_names else np.zeros(0)
        physical: dict[str, float] = {}
        if self.sampled_names:
            phys = self.space.to_physical(z_est[None, :], units="display", coord="z")
            physical = {name: float(phys[name][0]) for name in self.space.names}
        return {
            "sampled_names": self.sampled_names,
            "z": z_est,
            "z_anchor": np.asarray(self.z_anchor, dtype=float),
            "delta": delta,
            "physical": physical,
        }

    # --- block builders -----------------------------------------------------
    def _marginalized_block(self) -> BasisBlock:
        cols = np.asarray(self.residual_jacobian, dtype=float)[
            :, list(self.marg_indices)
        ]
        basis = _normalized(cols)
        names = tuple(f"marg_{i}" for i in self.marg_indices)
        groups = tuple(
            VarianceGroup(
                name=f"{self.marg_name}_{i}",
                indices=(col,),
                lower=self.initial_variance,
                upper=self.initial_variance,
                initial=self.initial_variance,
                update_from_sweep=NEVER_UPDATE,
            )
            for col, i in enumerate(self.marg_indices)
        )
        return BasisBlock(
            name=self.marg_name,
            matrix=basis,
            coefficient_names=names,
            groups=groups,
            kind="timing",
            metadata={"role": "analytically-marginalized", "n_col": basis.shape[1]},
        )

    def _sampled_block(self) -> BasisBlock:
        jz = self._jacobian_z(self.z_anchor)
        groups: list[VarianceGroup] = []
        for col, name in enumerate(self.sampled_names):
            lower, upper = rho_bounds_from_rms(
                jz,
                (col,),
                sigma_min=self.sample_sigma_min,
                sigma_max=self.sample_sigma_max,
            )
            groups.append(
                VarianceGroup(
                    name=f"{self.sampled_name}_{name}",
                    indices=(col,),
                    lower=lower,
                    upper=upper,
                    initial=self.initial_variance,
                    update_from_sweep=self.sample_update_from_sweep,
                )
            )
        return BasisBlock(
            name=self.sampled_name,
            matrix=jz,
            coefficient_names=self.sampled_names,
            groups=tuple(groups),
            kind="timing",
            metadata={"role": "sampled-z", "coordinate": "z"},
        )

    # --- jacobians ----------------------------------------------------------
    def _jacobian_z(self, z_anchor: np.ndarray) -> np.ndarray:
        if self._evaluator is not None:
            return np.asarray(
                self._evaluator.jacobian_z(self.space, z_anchor), dtype=float
            )
        # Linear approximation at the reference: J_z = J[:, sampled] * (dδ/dz).
        cols = np.asarray(self.residual_jacobian, dtype=float)[
            :, list(self.sampled_indices)
        ]
        return cols * _d_delta_d_z(self.space, z_anchor)[None, :]

    def _full_delta(self, z_anchor: np.ndarray) -> np.ndarray:
        n_fit = np.asarray(self.residual_jacobian).shape[1]
        delta = np.zeros(n_fit, dtype=float)
        if self.sampled_names:
            delta[list(self.sampled_indices)] = _delta_from_z(self.space, z_anchor)
        return delta


def timing_model(
    ctx,
    *,
    evaluator=None,
    initial_variance: float = 1.0e40,
    sample_update_from_sweep: int = 2,
    sample_sigma_min: float = DEFAULT_TIMING_SIGMA_MIN,
    sample_sigma_max: float = DEFAULT_TIMING_SIGMA_MAX,
    marginalize_all: bool = False,
) -> NltimingTimingModel:
    """Build a :class:`NltimingTimingModel` from an ``nltiming`` ``TimingSignal``.

    Parameters
    ----------
    ctx
        A bound ``nltiming`` timing model (``ntm.for_pulsar(pulsar)``).
    evaluator
        Optional ``nltiming.TimingEvaluator`` for autodiff ``J_z`` and nonlinear
        relinearization. When omitted, ``J_z`` is the linear design at the
        reference and ``advance`` is a no-op (single linear pass).
    initial_variance
        Broad first-sweep timing variance (``1e40``).
    sample_update_from_sweep
        Sweep at which sampled timing variances start updating (default 2).
    marginalize_all
        Treat *every* timing parameter as analytically marginalized (broad
        fixed prior, no sampled block). This reproduces the standard
        Enterprise/Discovery timing-marginalized GLS reconstruction and is the
        most robust choice for quick-look waveform plots.
    """
    pulsar = ctx.pulsar
    # Geometry-plan API: TimingParameterPlan on ctx.plan (PartitionResult removed).
    plan = ctx.plan
    design = np.asarray(ctx.design_matrix, dtype=float)  # M
    residual_jacobian = -design  # J = -M (fitter contract)
    n_fit = design.shape[1]

    if marginalize_all:
        sampled_names: tuple[str, ...] = ()
        sampled_indices: tuple[int, ...] = ()
        marg_indices: tuple[int, ...] = tuple(range(n_fit))
    else:
        sampled_names = tuple(plan.sampled)
        sampled_indices = tuple(int(i) for i in plan.idx_sampled)
        marg_indices = tuple(int(i) for i in plan.idx_analytically_marginalized)

    return NltimingTimingModel(
        pulsar=pulsar,
        backend=ctx.engine,
        space=ctx.space,
        sampled_names=sampled_names,
        sampled_indices=sampled_indices,
        marg_indices=marg_indices,
        design_matrix=design,
        residual_jacobian=residual_jacobian,
        base_residuals=np.asarray(pulsar.residuals, dtype=float),
        z_anchor=np.zeros(len(sampled_names), dtype=float),
        initial_variance=float(initial_variance),
        sample_update_from_sweep=int(sample_update_from_sweep),
        sample_sigma_min=float(sample_sigma_min),
        sample_sigma_max=float(sample_sigma_max),
        _evaluator=evaluator,
    )


def sign_check(
    ctx,
    *,
    eps: float = 1e-6,
    parameters: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Finite-difference validation of the residual-Jacobian sign convention.

    Compares each sampled column of ``J = -M`` against a central finite
    difference of ``backend.residual_delta``. Returns the maximum relative
    column error; a small value confirms both the magnitude and the sign of
    ``∂(Δr)/∂δ`` (and hence, via the analytic prior Jacobian, of ``J_z``).
    Raises if the convention is inconsistent.
    """
    plan = ctx.plan
    backend = ctx.engine
    residual_jacobian = -np.asarray(ctx.design_matrix, dtype=float)
    names = tuple(plan.sampled) if parameters is None else tuple(parameters)
    index_of = {name: int(i) for name, i in zip(plan.sampled, plan.idx_sampled)}
    n_fit = residual_jacobian.shape[1]

    errors: dict[str, float] = {}
    for name in names:
        col = index_of[name]
        delta_plus = np.zeros(n_fit)
        delta_plus[col] = eps
        delta_minus = np.zeros(n_fit)
        delta_minus[col] = -eps
        fd = (
            np.asarray(backend.residual_delta(delta_plus), dtype=float)
            - np.asarray(backend.residual_delta(delta_minus), dtype=float)
        ) / (2.0 * eps)
        analytic = residual_jacobian[:, col]
        scale = np.linalg.norm(analytic) or 1.0
        errors[name] = float(np.linalg.norm(fd - analytic) / scale)
    worst = max(errors.values()) if errors else 0.0
    if worst > 1e-3:
        raise ValueError(
            f"timing sign/scale check failed: max relative column error {worst:.2e} "
            "(J=-M does not match finite differences of residual_delta)"
        )
    return errors
