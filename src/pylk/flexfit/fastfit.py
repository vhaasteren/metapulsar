"""Top-level flexible-fit orchestration.

``fastfit`` combines an optional nonlinear timing model, a set of GP basis
blocks, and a fixed white-noise operator into one staged flexible-``Phi`` fit,
wrapping the conditional solve in a short trust-region/damped relinearization
loop for nonlinear timing. The result is an immutable, provenance-carrying
object suitable for quick-look waveform reconstruction and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .basis import BasisBlock, assemble
from .flexible_phi import FlexiblePhiResult, solve_flexible_phi
from .noise import NoiseOperator
from .timing import TimingModel


@dataclass(frozen=True)
class FastFitResult:
    """Immutable flexible-fit result, labelled *quick-look empirical Bayes*."""

    solve: FlexiblePhiResult
    residuals: np.ndarray
    outer_iterations: int
    timing_summary: Mapping[str, object]
    block_kinds: Mapping[str, str]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        arr = np.array(self.residuals, dtype=float, copy=True)
        arr.setflags(write=False)
        object.__setattr__(self, "residuals", arr)
        for name in ("timing_summary", "block_kinds", "provenance"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    # --- waveform accessors -------------------------------------------------
    def waveform(self, name: str) -> np.ndarray:
        """Conditional-mean waveform for one block (``T_block @ m_block``)."""
        return self.solve.waveform(name)

    @property
    def block_names(self) -> tuple[str, ...]:
        return tuple(self.solve.block_waveforms)

    def block_names_of(self, *kinds: str) -> tuple[str, ...]:
        """Block names whose kind is in ``kinds`` (e.g. ``"red", "dm"``)."""
        wanted = set(kinds)
        return tuple(n for n, k in self.block_kinds.items() if k in wanted)

    def noise_waveform(self) -> np.ndarray:
        """Sum of all non-timing (GP) block waveforms."""
        names = tuple(n for n, k in self.block_kinds.items() if k != "timing")
        if not names:
            return np.zeros_like(self.residuals)
        return self.solve.total_waveform(*names)

    def whitened_residuals(self) -> np.ndarray:
        """Residuals with every GP (non-timing) waveform subtracted."""
        return self.residuals - self.noise_waveform()

    def residuals_minus(self, *block_names: str) -> np.ndarray:
        """Residuals with the named block waveforms subtracted."""
        if not block_names:
            return np.asarray(self.residuals, dtype=float)
        return self.residuals - self.solve.total_waveform(*block_names)

    # --- convenience --------------------------------------------------------
    @property
    def group_variances(self) -> Mapping[str, float]:
        return self.solve.group_variances

    @property
    def bound_hits(self) -> tuple[str, ...]:
        return self.solve.bound_hits


def fastfit(
    *,
    noise: NoiseOperator,
    blocks: Sequence[BasisBlock] = (),
    timing: TimingModel | None = None,
    residuals: np.ndarray | None = None,
    n_sweeps: int = 3,
    max_timing_iterations: int = 1,
    damping: float = 1.0,
    step_tolerance: float = 1e-8,
    sweep_tolerance: float | None = None,
    warm_start: bool = True,
) -> FastFitResult:
    """Run the staged flexible-``Phi`` fit with optional nonlinear timing.

    Parameters
    ----------
    noise
        Fixed white-noise operator (applies ``N^-1``).
    blocks
        GP basis blocks (red, DM, chromatic, ECORR, custom) built by an adapter.
    timing
        Optional timing model. When given, the timing basis block(s) are placed
        first and the residual vector is taken from the timing anchor; otherwise
        ``residuals`` must be supplied.
    residuals
        Residual vector ``y`` when there is no timing model.
    n_sweeps
        Empirical-Bayes sweeps (``>= 2``; default 3).
    max_timing_iterations
        Outer relinearization iterations for nonlinear timing (default 1, i.e. a
        single linear pass; capped in practice at a few).
    damping
        Trust-region damping applied to each timing step in ``[0, 1]``.
    step_tolerance
        Stop the outer loop when the timing step norm falls below this.
    sweep_tolerance
        Optional inner convergence tolerance forwarded to the sweep loop.
    warm_start
        Reuse the previous iteration's ``Phi`` to initialize the next solve.
    """
    if max_timing_iterations < 1:
        raise ValueError("max_timing_iterations must be >= 1")
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must be in (0, 1]")
    if timing is None and residuals is None:
        raise ValueError(
            "provide either a timing model or an explicit residuals vector"
        )

    current_timing = timing
    initial_phi: np.ndarray | None = None
    outer = 0
    solve_result: FlexiblePhiResult | None = None
    y = np.asarray(residuals, dtype=float) if residuals is not None else None
    sampled_mean = np.zeros(0, dtype=float)

    for outer in range(1, max_timing_iterations + 1):
        if current_timing is not None:
            timing_blocks = tuple(current_timing.blocks())
            y = np.asarray(current_timing.residuals(), dtype=float)
        else:
            timing_blocks = ()
        model = assemble((*timing_blocks, *blocks))
        solve_result = solve_flexible_phi(
            y,
            model,
            noise,
            n_sweeps=n_sweeps,
            tolerance=sweep_tolerance,
            initial_phi=initial_phi if warm_start else None,
        )

        if current_timing is None or current_timing.sampled_block is None:
            break

        span = model.block_spans[current_timing.sampled_block]
        sampled_mean = solve_result.coefficient_mean[span]
        if outer >= max_timing_iterations:
            break
        advanced = current_timing.advance(sampled_mean, damping=damping)
        if advanced is None:
            break
        if float(np.linalg.norm(sampled_mean)) <= step_tolerance:
            break
        current_timing = advanced
        if warm_start:
            initial_phi = solve_result.phi_diagonal.copy()

    assert solve_result is not None
    timing_summary: Mapping[str, object] = {}
    if current_timing is not None:
        timing_summary = dict(current_timing.summary(sampled_mean))

    provenance = {
        "label": "quick-look empirical Bayes",
        "n_sweeps_requested": int(n_sweeps),
        "n_sweeps_completed": int(solve_result.n_sweeps),
        "outer_iterations": int(outer),
        "max_timing_iterations": int(max_timing_iterations),
        "damping": float(damping),
        "warm_start": bool(warm_start),
        "bound_hits": tuple(solve_result.bound_hits),
        "n_obs": int(noise.n_obs),
        "n_coef": int(solve_result.coefficient_mean.shape[0]),
        **{f"diag_{k}": v for k, v in solve_result.diagnostics.items()},
    }
    block_kinds = _block_kinds(timing, current_timing, blocks)
    return FastFitResult(
        solve=solve_result,
        residuals=np.asarray(y, dtype=float),
        outer_iterations=int(outer),
        timing_summary=timing_summary,
        block_kinds=block_kinds,
        provenance=provenance,
    )


def _block_kinds(timing, current_timing, blocks) -> dict[str, str]:
    kinds: dict[str, str] = {}
    source = current_timing if current_timing is not None else timing
    if source is not None:
        for block in source.blocks():
            kinds[block.name] = block.kind
    for block in blocks:
        kinds[block.name] = block.kind
    return kinds
