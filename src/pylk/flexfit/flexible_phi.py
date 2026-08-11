"""Joint conditional Gaussian solve with staged bounded variance updates.

This is Piece B: given a residual vector ``y``, a
fixed white-noise operator ``N``, an assembled joint basis ``T`` and a diagonal
``Phi`` prior structured into variance groups, compute the conditional
coefficient moments and iterate a small number of empirical-Bayes sweeps that
learn the group variances.

The solve is deterministic and small. It is a *quick-look empirical-Bayes*
estimator and an initializer for production inference, not a substitute for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import scipy.linalg as sl

from .basis import AssembledModel, VarianceGroup
from .noise import NoiseOperator


def _stable_cho_factor(precision: np.ndarray) -> tuple[tuple[np.ndarray, bool], float]:
    """Cholesky-factor a symmetric precision, adding minimal jitter if needed.

    A broad improper timing block (``Phi = 1e40`` -> precision ``1e-40``) that
    sits on a rank-deficient design leaves the precision numerically singular in
    its null directions. A tiny diagonal jitter -- a slightly less broad prior on
    those unconstrained directions -- restores positive-definiteness without
    changing the constrained solution. Returns the factor and the jitter used.
    """
    n = precision.shape[0]
    diag_mean = float(np.mean(np.abs(np.diag(precision)))) or 1.0
    jitter = 0.0
    for _ in range(9):
        try:
            factor = sl.cho_factor(precision + jitter * np.eye(n), lower=True)
            return factor, jitter
        except sl.LinAlgError:
            jitter = 1e-12 * diag_mean if jitter == 0.0 else jitter * 10.0
    raise np.linalg.LinAlgError(
        "conditional precision is not positive definite even with jitter; "
        "check the basis for exactly duplicated columns or invalid noise."
    )


def conditional_moments(
    y: np.ndarray,
    basis: np.ndarray,
    phi: np.ndarray,
    noise: NoiseOperator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(mean, covariance, second_moment)`` of the coefficient posterior.

    Solves ``Sigma = (T^T N^-1 T + Phi^-1)^-1`` and ``m = Sigma T^T N^-1 y`` via
    a Cholesky factorization of the precision, working with ``Phi^-1`` directly
    so that broad timing variances (``Phi = 1e40``) never form large entries. A
    minimal adaptive jitter keeps a rank-deficient improper timing block solvable.
    """
    ninv_y = noise.solve(y)
    ninv_t = noise.solve(basis)
    precision = basis.T @ ninv_t
    precision[np.diag_indices_from(precision)] += 1.0 / np.asarray(phi, dtype=float)
    factor, _ = _stable_cho_factor(precision)
    mean = sl.cho_solve(factor, basis.T @ ninv_y)
    covariance = sl.cho_solve(factor, np.eye(precision.shape[0]))
    second_moment = mean**2 + np.diag(covariance)
    return mean, covariance, second_moment


def bounded_variance_update(
    second_moment: np.ndarray, group: VarianceGroup
) -> tuple[float, bool]:
    """Deterministic EM / MAP update of one group's variance, clipped to bounds.

    Returns ``(rho, hit_bound)``. With no hyperprior (``alpha = beta = 0``) this
    is the maximum-likelihood EM update ``rho = mean(second_moment)``. With an
    inverse-gamma hyperprior ``p(rho) ~ rho^(-alpha-1) exp(-beta/rho)`` it is the
    posterior-mode MAP update ``(sum s + 2 beta) / (n + 2 alpha + 2)``.
    """
    s = np.asarray(second_moment, dtype=float)[list(group.indices)]
    total = float(np.sum(s))
    if group.alpha == 0.0 and group.beta == 0.0:
        rho = total / group.size  # maximum-likelihood EM
    else:
        rho = (total + 2.0 * group.beta) / (group.size + 2.0 * group.alpha + 2.0)
    clipped = float(np.clip(rho, group.lower, group.upper))
    return clipped, clipped != rho


@dataclass(frozen=True)
class FlexiblePhiResult:
    """Immutable result of a staged flexible-``Phi`` fit."""

    coefficient_mean: np.ndarray
    coefficient_covariance: np.ndarray
    phi_diagonal: np.ndarray
    second_moment: np.ndarray
    block_waveforms: Mapping[str, np.ndarray]
    block_spans: Mapping[str, slice]
    group_variances: Mapping[str, float]
    n_sweeps: int
    bound_hits: tuple[str, ...]
    phi_history: np.ndarray
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in (
            "coefficient_mean",
            "coefficient_covariance",
            "phi_diagonal",
            "second_moment",
            "phi_history",
        ):
            arr = np.array(getattr(self, name), dtype=float, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        object.__setattr__(
            self,
            "block_waveforms",
            MappingProxyType(
                {k: np.asarray(v, dtype=float) for k, v in self.block_waveforms.items()}
            ),
        )
        object.__setattr__(
            self, "group_variances", MappingProxyType(dict(self.group_variances))
        )
        object.__setattr__(
            self, "block_spans", MappingProxyType(dict(self.block_spans))
        )
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )

    def waveform(self, name: str) -> np.ndarray:
        """Conditional-mean waveform ``T_block @ m_block`` for one block."""
        return np.asarray(self.block_waveforms[name], dtype=float)

    def block_mean(self, name: str) -> np.ndarray:
        """Conditional coefficient mean for one block."""
        return np.asarray(self.coefficient_mean[self.block_spans[name]], dtype=float)

    def block_second_moment(self, name: str) -> np.ndarray:
        """Conditional coefficient second moments ``m^2 + diag(Sigma)`` for one block."""
        return np.asarray(self.second_moment[self.block_spans[name]], dtype=float)

    def total_waveform(self, *names: str) -> np.ndarray:
        """Sum of the named block waveforms (all blocks if none given)."""
        keys = names or tuple(self.block_waveforms)
        first = self.waveform(keys[0])
        out = np.zeros_like(first)
        for key in keys:
            out = out + self.waveform(key)
        return out


def _initial_phi(model: AssembledModel) -> np.ndarray:
    phi = np.empty(model.n_coef, dtype=float)
    for group in model.groups:
        rho = group.initial_rho()
        for idx in group.indices:
            phi[idx] = rho
    return phi


def solve_flexible_phi(
    y: np.ndarray,
    model: AssembledModel,
    noise: NoiseOperator,
    *,
    n_sweeps: int = 3,
    tolerance: float | None = None,
    initial_phi: np.ndarray | None = None,
) -> FlexiblePhiResult:
    """Run the staged empirical-Bayes flexible-``Phi`` fit.

    Sweep ``s`` runs an E-step (conditional moments at the current ``Phi``) then
    an M-step that updates every variance group with ``update_from_sweep <= s``.
    Timing groups default to ``update_from_sweep = 2`` and an initial variance of
    ``1e40``, so the first sweep effectively marginalizes the timing directions
    while the non-timing groups are learned. A final E-step at the converged
    ``Phi`` gives mutually consistent reported moments.

    ``tolerance`` may stop the sweeps early once the largest fractional change in
    any group variance falls below it, but never before the second sweep.
    """
    y = np.asarray(y, dtype=float)
    if y.shape != (model.n_obs,):
        raise ValueError(f"y must have shape ({model.n_obs},), got {y.shape}")
    if noise.n_obs != model.n_obs:
        raise ValueError("noise operator n_obs does not match the basis")
    n_sweeps = int(n_sweeps)
    if n_sweeps < 2:
        raise ValueError("n_sweeps must be >= 2 (the staged heuristic needs two)")

    phi = _initial_phi(model) if initial_phi is None else np.array(initial_phi, float)
    if phi.shape != (model.n_coef,):
        raise ValueError("initial_phi has the wrong length")

    phi_history: list[np.ndarray] = []
    bound_hits: set[str] = set()
    completed = 0
    for sweep in range(1, n_sweeps + 1):
        phi_history.append(phi.copy())
        _, _, second_moment = conditional_moments(y, model.matrix, phi, noise)
        previous = phi.copy()
        for group in model.groups:
            if group.update_from_sweep > sweep:
                continue
            rho, hit = bounded_variance_update(second_moment, group)
            if hit:
                bound_hits.add(group.name)
            for idx in group.indices:
                phi[idx] = rho
        completed = sweep
        if tolerance is not None and sweep >= 2:
            active = previous > 0.0
            rel = np.max(np.abs(phi[active] - previous[active]) / previous[active])
            if rel < tolerance:
                break

    # Final E-step so the reported moments are consistent with the final Phi.
    mean, covariance, second_moment = conditional_moments(y, model.matrix, phi, noise)
    phi_history.append(phi.copy())

    block_waveforms: dict[str, np.ndarray] = {}
    for name, span in model.block_spans.items():
        block_waveforms[name] = model.matrix[:, span] @ mean[span]

    group_variances = {
        group.name: float(phi[group.indices[0]]) for group in model.groups
    }

    precision_cond = float(np.linalg.cond(covariance)) if model.n_coef else 1.0
    diagnostics = {
        "converged_sweeps": completed,
        "requested_sweeps": n_sweeps,
        "covariance_condition_number": precision_cond,
        "noise_logdet": noise.logdet(),
    }
    return FlexiblePhiResult(
        coefficient_mean=mean,
        coefficient_covariance=covariance,
        phi_diagonal=phi,
        second_moment=second_moment,
        block_waveforms=block_waveforms,
        block_spans=dict(model.block_spans),
        group_variances=group_variances,
        n_sweeps=completed,
        bound_hits=tuple(sorted(bound_hits)),
        phi_history=np.array(phi_history, dtype=float),
        diagnostics=diagnostics,
    )
