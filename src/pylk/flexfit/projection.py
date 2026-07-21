"""Project a flexible (free-spectrum) fit onto a physical hyperparameter model.

After the staged solve produces per-coefficient second moments ``s_j``, a
physical spectral model ``Phi(theta)`` (power law, broken power law, free
spectrum) is fit by minimizing the expected Gaussian complete-data objective

    Q(theta) = 0.5 * sum_j [ log phi_j(theta) + s_j / phi_j(theta) ],

which uses only the coefficient summary and never re-touches the TOA residual
vector, so it is very fast. The spectral model family itself is supplied by the
caller (an adapter reusing Discovery/Enterprise conventions); this module only
owns the objective and the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class SpectrumProjection:
    """Result of projecting second moments onto a physical spectrum model."""

    theta: np.ndarray
    parameter_names: tuple[str, ...]
    phi_model: np.ndarray
    objective: float
    success: bool
    message: str

    @property
    def values(self) -> dict[str, float]:
        return dict(zip(self.parameter_names, map(float, self.theta), strict=True))


def spectrum_objective(
    theta: np.ndarray,
    second_moments: np.ndarray,
    spectrum_fn: Callable[[np.ndarray], np.ndarray],
) -> float:
    """Evaluate ``Q(theta)`` for the given spectrum model."""
    phi = np.asarray(spectrum_fn(np.asarray(theta, dtype=float)), dtype=float)
    s = np.asarray(second_moments, dtype=float)
    if phi.shape != s.shape:
        raise ValueError("spectrum_fn output must align with second_moments")
    if np.any(phi <= 0.0) or np.any(~np.isfinite(phi)):
        return np.inf
    return 0.5 * float(np.sum(np.log(phi) + s / phi))


def project_spectrum(
    second_moments: np.ndarray,
    spectrum_fn: Callable[[np.ndarray], np.ndarray],
    theta0: Sequence[float],
    *,
    parameter_names: Sequence[str],
    bounds: Sequence[tuple[float, float]] | None = None,
    method: str = "Nelder-Mead",
) -> SpectrumProjection:
    """Fit ``theta`` minimizing the complete-data spectrum objective.

    ``spectrum_fn(theta)`` must return the model variance ``phi_j`` for every
    coefficient covered by ``second_moments`` (tied sine/cosine pairs simply map
    to the same model value).
    """
    second_moments = np.asarray(second_moments, dtype=float)
    theta0 = np.asarray(theta0, dtype=float)
    parameter_names = tuple(parameter_names)
    if len(parameter_names) != theta0.size:
        raise ValueError("parameter_names must match theta0 length")

    result = minimize(
        spectrum_objective,
        theta0,
        args=(second_moments, spectrum_fn),
        method=method,
        bounds=bounds,
    )
    theta = np.asarray(result.x, dtype=float)
    return SpectrumProjection(
        theta=theta,
        parameter_names=parameter_names,
        phi_model=np.asarray(spectrum_fn(theta), dtype=float),
        objective=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
    )
