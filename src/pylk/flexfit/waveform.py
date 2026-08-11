"""Waveform analysis over a finished flexible-``Phi`` solve.

Groups basis blocks, subtracts / rebuilds waveforms in named stages, predicts
Fourier-GP bands on a time grid, and emits a fixed panel / figdata payload —
labelled *quick-look empirical Bayes*, not MCMC posterior draws.

Depends only on NumPy/SciPy at import time; ``pyarrow`` is imported
function-locally by the figdata I/O helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .basis import BasisBlock
from .flexible_phi import FlexiblePhiResult
from .noise import EpochKernelNoise, NoiseOperator, ShermanMorrisonNoise

SEC_TO_US = 1.0e6
FIGDATA_FORMAT = "pylk.flexfit.waveform_figdata/v1"
EB_LABEL = "quick-look empirical Bayes"


# --------------------------------------------------------------------------- #
# Module-level name / residual helpers (single implementation)
# --------------------------------------------------------------------------- #
def block_names_of(
    block_kinds: Mapping[str, str],
    block_waveforms: Mapping[str, Any],
    *kinds: str,
) -> tuple[str, ...]:
    """Names whose kind is in ``kinds``, in ``block_waveforms`` column order."""
    wanted = set(kinds)
    order = {n: i for i, n in enumerate(block_waveforms)}
    return tuple(
        sorted(
            (n for n, k in block_kinds.items() if k in wanted), key=order.__getitem__
        )
    )


def block_names_excluding(
    block_kinds: Mapping[str, str],
    block_waveforms: Mapping[str, Any],
    *kinds: str,
) -> tuple[str, ...]:
    """Names whose kind is *not* in ``kinds``, in ``block_waveforms`` column order.

    ``block_names_excluding(..., "timing")`` is the exact predicate live
    ``FastFitResult.noise_waveform`` / ``whitened_residuals`` use
    (``kind != "timing"``) — not an explicit PTA kind allow-list.
    """
    excluded = set(kinds)
    order = {n: i for i, n in enumerate(block_waveforms)}
    return tuple(
        sorted(
            (n for n, k in block_kinds.items() if k not in excluded),
            key=order.__getitem__,
        )
    )


def residuals_after(
    solve: FlexiblePhiResult,
    y: np.ndarray,
    names: Sequence[str],
    *,
    extra_waveforms: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """``y`` minus the named block waveforms; empty ``names`` → copy of ``y``."""
    y_arr = np.asarray(y, dtype=float)
    if not names:
        return y_arr.copy()
    out = y_arr.copy()
    extras = extra_waveforms or {}
    solve_names = [n for n in names if n not in extras]
    if solve_names:
        out = out - solve.total_waveform(*solve_names)
    for name in names:
        if name in extras:
            out = out - np.asarray(extras[name], dtype=float)
    return out


def frequencies_from_blocks(
    blocks: Sequence[BasisBlock],
) -> dict[str, np.ndarray]:
    """Extract Fourier frequencies from ``BasisBlock.metadata['frequencies']``."""
    out: dict[str, np.ndarray] = {}
    for block in blocks:
        if "frequencies" in block.metadata:
            out[block.name] = np.asarray(block.metadata["frequencies"], dtype=float)
    return out


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #
def _freeze_array(value: np.ndarray) -> np.ndarray:
    arr = np.array(value, dtype=float, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class StageSpec:
    """One named residual stage in a waveform analysis.

    Exactly one of ``subtract_names`` or ``subtract_kinds`` may be used to
    choose blocks; both may be empty (raw stage). ``normalize=True`` divides
    the stage residual by ``sqrt(variance)`` after subtraction (whitened / z).
    """

    name: str
    subtract_names: tuple[str, ...] = ()
    subtract_kinds: tuple[str, ...] = ()
    normalize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "subtract_names", tuple(self.subtract_names))
        object.__setattr__(self, "subtract_kinds", tuple(self.subtract_kinds))
        if self.subtract_names and self.subtract_kinds:
            raise ValueError(
                f"stage {self.name!r}: pass subtract_names or subtract_kinds, not both"
            )
        if not self.name:
            raise ValueError("stage name must be non-empty")


@dataclass(frozen=True)
class WaveformStage:
    name: str
    residuals: np.ndarray  # seconds, or unitless if normalized
    subtracted: tuple[str, ...]  # block names actually subtracted
    normalized: bool
    rms: float  # np.sqrt(mean(residuals**2)) on finite entries

    def __post_init__(self) -> None:
        object.__setattr__(self, "residuals", _freeze_array(self.residuals))
        object.__setattr__(self, "subtracted", tuple(self.subtracted))


@dataclass(frozen=True)
class GPBand:
    name: str  # block name
    kind: str  # block kind string
    t_grid: np.ndarray  # seconds
    mean: np.ndarray  # seconds
    std: np.ndarray  # seconds, 1σ from coefficient covariance

    def __post_init__(self) -> None:
        for name in ("t_grid", "mean", "std"):
            object.__setattr__(self, name, _freeze_array(getattr(self, name)))


@dataclass(frozen=True)
class WaveformPanelArrays:
    """Standard EB waveform panel arrays (quick-look empirical Bayes).

    TOA vectors are length ``n_toa``. Grid vectors are length ``n_grid``.
    Missing optional GP bands are zero-length arrays (shape ``(0,)``), never
    ``None``, so serializers stay uniform.
    """

    # --- TOA group ---
    mjd: np.ndarray
    freq_mhz: np.ndarray
    resid_us: np.ndarray
    after_timing_us: np.ndarray
    after_all_us: np.ndarray
    z: np.ndarray
    sigma_us: np.ndarray

    # --- grid group (Fourier GP bands) ---
    grid_mjd: np.ndarray
    red_mean_us: np.ndarray
    red_std_us: np.ndarray
    dm_mean_us: np.ndarray
    dm_std_us: np.ndarray

    # --- provenance ---
    label: str
    stage_rms_us: Mapping[str, float]

    def __post_init__(self) -> None:
        for name in (
            "mjd",
            "freq_mhz",
            "resid_us",
            "after_timing_us",
            "after_all_us",
            "z",
            "sigma_us",
            "grid_mjd",
            "red_mean_us",
            "red_std_us",
            "dm_mean_us",
            "dm_std_us",
        ):
            object.__setattr__(self, name, _freeze_array(getattr(self, name)))
        object.__setattr__(
            self, "stage_rms_us", MappingProxyType(dict(self.stage_rms_us))
        )


STANDARD_PTA_STAGES: tuple[StageSpec, ...] = (
    StageSpec("raw"),
    StageSpec("after_timing", subtract_kinds=("timing",)),
    StageSpec("after_red", subtract_kinds=("timing", "red")),
    StageSpec(
        "after_all",
        subtract_kinds=("timing", "red", "dm", "chromatic", "ecorr", "custom"),
    ),
    StageSpec(
        "whitened",
        subtract_kinds=("timing", "red", "dm", "chromatic", "ecorr", "custom"),
        normalize=True,
    ),
)


# --------------------------------------------------------------------------- #
# Algorithms
# --------------------------------------------------------------------------- #
def predict_fourier_gp(
    *,
    frequencies: np.ndarray,
    coefficient_mean: np.ndarray,
    coefficient_covariance: np.ndarray,
    t_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fourier-GP posterior mean and 1σ std on ``t_grid``.

    Basis columns are ``sin(2π f t)`` at even indices and ``cos(2π f t)`` at
    odd indices — identical to the former Discovery ``Reconstruction.predict_gp``.
    """
    f = np.asarray(frequencies, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    if f.ndim != 1 or f.size % 2 != 0:
        raise ValueError("frequencies must be 1-D with even length (sin/cos pairs)")
    if coefficient_mean.shape != f.shape:
        raise ValueError("coefficient_mean shape must match frequencies")
    if coefficient_covariance.shape != (f.size, f.size):
        raise ValueError(
            "coefficient_covariance must be (n, n) with n = len(frequencies)"
        )
    phase = 2.0 * np.pi * np.outer(t, f)
    basis = np.empty_like(phase)
    basis[:, 0::2] = np.sin(phase[:, 0::2])
    basis[:, 1::2] = np.cos(phase[:, 1::2])
    mean = basis @ np.asarray(coefficient_mean, dtype=float)
    cov = np.asarray(coefficient_covariance, dtype=float)
    var = np.einsum("gi,ij,gj->g", basis, cov, basis)
    return mean, np.sqrt(np.clip(var, 0.0, None))


def aggregate_bands(bands: Sequence[GPBand], *, name: str, kind: str) -> GPBand:
    """Sum GPBands sharing a grid: mean = Σ meanₖ, std = sqrt(Σ stdₖ²).

    Variances add as if the blocks' coefficient posteriors were independent —
    it ignores cross-block covariance (each band already used only its own
    ``coefficient_covariance[span, span]`` diagonal block). Empty input → a
    zero-length GPBand. All bands must share ``t_grid``.
    """
    if not bands:
        empty = np.zeros(0, dtype=float)
        return GPBand(name=name, kind=kind, t_grid=empty, mean=empty, std=empty)
    t0 = np.asarray(bands[0].t_grid, dtype=float)
    for band in bands[1:]:
        if not np.array_equal(np.asarray(band.t_grid, dtype=float), t0):
            raise ValueError("aggregate_bands requires all bands to share t_grid")
    mean = np.zeros_like(t0)
    var = np.zeros_like(t0)
    for band in bands:
        mean = mean + np.asarray(band.mean, dtype=float)
        std = np.asarray(band.std, dtype=float)
        var = var + std * std
    return GPBand(
        name=name,
        kind=kind,
        t_grid=t0,
        mean=mean,
        std=np.sqrt(np.clip(var, 0.0, None)),
    )


def _resolve_stage_names(
    spec: StageSpec,
    *,
    block_kinds: Mapping[str, str],
    block_waveforms: Mapping[str, Any],
) -> tuple[str, ...]:
    if spec.subtract_names:
        for name in spec.subtract_names:
            if name not in block_waveforms:
                raise KeyError(f"stage {spec.name!r}: unknown block {name!r}")
        return tuple(spec.subtract_names)
    if spec.subtract_kinds:
        return block_names_of(block_kinds, block_waveforms, *spec.subtract_kinds)
    return ()


def _stage_rms(residuals: np.ndarray) -> float:
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(finite))))


def _evaluate_stages(
    y: np.ndarray,
    variance: np.ndarray,
    solve: FlexiblePhiResult,
    block_kinds: Mapping[str, str],
    stages: Sequence[StageSpec],
    *,
    waveforms: Mapping[str, Any] | None = None,
    extra_waveforms: Mapping[str, np.ndarray] | None = None,
) -> tuple[WaveformStage, ...]:
    waves = waveforms if waveforms is not None else solve.block_waveforms
    out: list[WaveformStage] = []
    for spec in stages:
        names = _resolve_stage_names(
            spec,
            block_kinds=block_kinds,
            block_waveforms=waves,
        )
        r = residuals_after(solve, y, names, extra_waveforms=extra_waveforms)
        if spec.normalize:
            r = r / np.sqrt(variance)
        out.append(
            WaveformStage(
                name=spec.name,
                residuals=r,
                subtracted=names,
                normalized=bool(spec.normalize),
                rms=_stage_rms(r),
            )
        )
    return tuple(out)


@dataclass(frozen=True)
class WaveformAnalysis:
    """Immutable EB waveform analysis over one finished flexible-Phi solve."""

    y: np.ndarray
    variance: np.ndarray
    toas: np.ndarray
    toa_mjd: np.ndarray
    freqs_mhz: np.ndarray | None
    solve: FlexiblePhiResult
    block_kinds: Mapping[str, str]
    block_frequencies: Mapping[str, np.ndarray]
    stages: tuple[WaveformStage, ...]
    label: str = EB_LABEL
    extra_waveforms: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("y", "variance", "toas", "toa_mjd"):
            object.__setattr__(
                self, field_name, _freeze_array(getattr(self, field_name))
            )
        if self.freqs_mhz is not None:
            object.__setattr__(self, "freqs_mhz", _freeze_array(self.freqs_mhz))
        object.__setattr__(
            self, "block_kinds", MappingProxyType(dict(self.block_kinds))
        )
        object.__setattr__(
            self,
            "block_frequencies",
            MappingProxyType(
                {
                    k: np.asarray(v, dtype=float)
                    for k, v in self.block_frequencies.items()
                }
            ),
        )
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(
            self,
            "extra_waveforms",
            MappingProxyType(
                {
                    k: _freeze_array(np.asarray(v, dtype=float))
                    for k, v in dict(self.extra_waveforms).items()
                }
            ),
        )

    @property
    def waveforms(self) -> Mapping[str, np.ndarray]:
        """Solve block waveforms plus any analysis-layer extras (e.g. kernel ECORR)."""
        return {**self.solve.block_waveforms, **self.extra_waveforms}

    # --- accessors ----------------------------------------------------------
    def stage(self, name: str) -> WaveformStage:
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(f"unknown stage {name!r}")

    def residuals_after(self, *block_names: str) -> np.ndarray:
        return residuals_after(
            self.solve, self.y, block_names, extra_waveforms=self.extra_waveforms
        )

    def residuals_after_kinds(self, *kinds: str) -> np.ndarray:
        names = self.block_names_of(*kinds)
        return residuals_after(
            self.solve, self.y, names, extra_waveforms=self.extra_waveforms
        )

    def residuals_after_excluding_kinds(self, *kinds: str) -> np.ndarray:
        names = self.block_names_excluding(*kinds)
        return residuals_after(
            self.solve, self.y, names, extra_waveforms=self.extra_waveforms
        )

    def waveform(self, name: str) -> np.ndarray:
        if name in self.extra_waveforms:
            return np.asarray(self.extra_waveforms[name], dtype=float)
        return self.solve.waveform(name)

    def block_names_of(self, *kinds: str) -> tuple[str, ...]:
        return block_names_of(self.block_kinds, self.waveforms, *kinds)

    def block_names_excluding(self, *kinds: str) -> tuple[str, ...]:
        return block_names_excluding(self.block_kinds, self.waveforms, *kinds)

    def predict_gp(self, name: str, t_grid: np.ndarray) -> GPBand:
        if name not in self.block_frequencies:
            raise KeyError(
                f"block {name!r} has no Fourier frequencies (cannot predict on a "
                "grid; timing-model prediction would need a new design matrix)"
            )
        span = self.solve.block_spans[name]
        mean, std = predict_fourier_gp(
            frequencies=self.block_frequencies[name],
            coefficient_mean=self.solve.coefficient_mean[span],
            coefficient_covariance=self.solve.coefficient_covariance[span, span],
            t_grid=t_grid,
        )
        return GPBand(
            name=name,
            kind=self.block_kinds[name],
            t_grid=np.asarray(t_grid, dtype=float),
            mean=mean,
            std=std,
        )

    # --- Reconstruction dict-facade -----------------------------------------
    def __getitem__(self, name: str) -> np.ndarray:
        return self.waveform(name)

    def __iter__(self):
        return iter(self.waveforms)

    def __contains__(self, name: str) -> bool:
        return name in self.waveforms

    def keys(self):
        return self.waveforms.keys()

    def gp_bands(
        self,
        *,
        t_grid: np.ndarray | None = None,
        n_grid: int = 400,
        kinds: tuple[str, ...] = ("red", "dm", "chromatic"),
    ) -> tuple[GPBand, ...]:
        if t_grid is None:
            t_grid = np.linspace(float(self.toas.min()), float(self.toas.max()), n_grid)
        else:
            t_grid = np.asarray(t_grid, dtype=float)
        names = self.block_names_of(*kinds)
        return tuple(
            self.predict_gp(name, t_grid)
            for name in names
            if name in self.block_frequencies
        )

    def panel_arrays(
        self,
        *,
        t_grid: np.ndarray | None = None,
        n_grid: int = 400,
    ) -> WaveformPanelArrays:
        n = self.y.size
        timing_names = self.block_names_of("timing")
        non_timing_names = self.block_names_excluding("timing")
        after_timing = self.residuals_after(*timing_names)
        after_all = self.residuals_after(*(timing_names + non_timing_names))
        z = after_all / np.sqrt(self.variance)
        freq_mhz = (
            np.full(n, np.nan)
            if self.freqs_mhz is None
            else np.asarray(self.freqs_mhz, dtype=float)
        )

        if t_grid is None:
            t_grid_arr = np.linspace(
                float(self.toas.min()), float(self.toas.max()), n_grid
            )
            grid_mjd = np.linspace(
                float(self.toa_mjd.min()), float(self.toa_mjd.max()), n_grid
            )
        else:
            t_grid_arr = np.asarray(t_grid, dtype=float)
            order = np.argsort(self.toas)
            toas_sorted = np.asarray(self.toas, dtype=float)[order]
            mjd_sorted = np.asarray(self.toa_mjd, dtype=float)[order]
            grid_mjd = np.interp(t_grid_arr, toas_sorted, mjd_sorted)

        bands = self.gp_bands(
            t_grid=t_grid_arr, n_grid=n_grid, kinds=("red", "dm", "chromatic")
        )
        red = aggregate_bands(
            [b for b in bands if b.kind == "red"], name="red", kind="red"
        )
        dm = aggregate_bands(
            [b for b in bands if b.kind in ("dm", "chromatic")],
            name="dm",
            kind="dm",
        )

        stage_rms_us: dict[str, float] = {}
        for s in self.stages:
            if s.normalized:
                stage_rms_us[s.name] = float(s.rms)
            else:
                stage_rms_us[s.name] = float(s.rms * SEC_TO_US)

        empty = np.zeros(0, dtype=float)
        return WaveformPanelArrays(
            mjd=np.asarray(self.toa_mjd, dtype=float),
            freq_mhz=freq_mhz,
            resid_us=np.asarray(self.y, dtype=float) * SEC_TO_US,
            after_timing_us=after_timing * SEC_TO_US,
            after_all_us=after_all * SEC_TO_US,
            z=z,
            sigma_us=np.sqrt(self.variance) * SEC_TO_US,
            grid_mjd=grid_mjd,
            red_mean_us=(
                empty if red.mean.size == 0 else np.asarray(red.mean) * SEC_TO_US
            ),
            red_std_us=(
                empty if red.std.size == 0 else np.asarray(red.std) * SEC_TO_US
            ),
            dm_mean_us=(
                empty if dm.mean.size == 0 else np.asarray(dm.mean) * SEC_TO_US
            ),
            dm_std_us=empty if dm.std.size == 0 else np.asarray(dm.std) * SEC_TO_US,
            label=self.label,
            stage_rms_us=stage_rms_us,
        )

    def summary(self) -> dict[str, object]:
        blocks: dict[str, object] = {}
        for name in self.waveforms:
            span = self.solve.block_spans.get(name)
            n_coef = None if span is None else int(span.stop - span.start)
            blocks[name] = {
                "kind": self.block_kinds[name],
                "n_coef": n_coef,
                "toa_rms_us": float(
                    np.sqrt(np.mean(self.waveform(name) ** 2)) * SEC_TO_US
                ),
                "has_frequencies": name in self.block_frequencies,
            }
        return {
            "label": self.label,
            "n_toa": int(self.y.size),
            "blocks": blocks,
            "stages": {
                s.name: {
                    "subtracted": list(s.subtracted),
                    "normalized": s.normalized,
                    "rms": s.rms,
                    "rms_us": None if s.normalized else s.rms * SEC_TO_US,
                }
                for s in self.stages
            },
            "group_variances": dict(self.solve.group_variances),
            "bound_hits": list(self.solve.bound_hits),
        }


def _kernel_ecorr_waveform(
    y: np.ndarray, solve: FlexiblePhiResult, noise: NoiseOperator
) -> np.ndarray:
    """Epoch waveform ``E â`` for a kernel-ECORR operator (§5.6)."""
    r = y - solve.total_waveform()
    if isinstance(noise, EpochKernelNoise):
        a_hat = (
            noise.capacitance_scale
            * np.asarray(noise.indicator.T @ (r / noise.diagonal)).ravel()
        )
        return np.asarray(noise.indicator @ a_hat).ravel()
    if isinstance(noise, ShermanMorrisonNoise):
        from .fasttnt import _indicator

        e = _indicator(noise)
        dinv = 1.0 / noise.diagonal
        t = np.asarray(e.multiply(e).T @ dinv).ravel()
        s = 1.0 / (1.0 / noise.jitter + t)
        a_hat = s * np.asarray(e.T @ (dinv * r)).ravel()
        return np.asarray(e @ a_hat).ravel()
    raise TypeError(
        f"noise={type(noise)!r} is not a kernel-ECORR operator; "
        "pass EpochKernelNoise or ShermanMorrisonNoise, or omit noise="
    )


def analyze_waveforms(
    y: np.ndarray,
    variance: np.ndarray,
    solve: FlexiblePhiResult,
    *,
    toas: np.ndarray,
    toa_mjd: np.ndarray,
    block_kinds: Mapping[str, str],
    block_frequencies: Mapping[str, np.ndarray] | None = None,
    freqs_mhz: np.ndarray | None = None,
    stages: Sequence[StageSpec] | None = None,
    noise: NoiseOperator | None = None,
) -> WaveformAnalysis:
    """Build a WaveformAnalysis from a finished flexible-Phi solve.

    ``y`` is the residual vector the solve used (seconds). ``variance`` is the
    diagonal of ``D`` (seconds²) — the whitening denominator in both topologies
    after ECORR is subtracted as a waveform. ``toas`` / ``toa_mjd`` are per-TOA
    time coordinates. ``block_frequencies`` maps Fourier block name → interleaved
    frequency vector in Hz (usually from ``BasisBlock.metadata["frequencies"]``).

    When ``noise`` is a kernel-ECORR operator, an ``"ecorr_kernel"`` waveform is
    injected so ``after_all`` / ``whitened`` still subtract ECORR. Mode-1 scripts
    that pin ECORR in ``N`` must pass ``noise=``; omitting it leaves epoch power
    in those stages.
    """
    y_arr = np.asarray(y, dtype=float)
    var_arr = np.asarray(variance, dtype=float)
    toas_arr = np.asarray(toas, dtype=float)
    mjd_arr = np.asarray(toa_mjd, dtype=float)
    n = y_arr.shape
    if y_arr.ndim != 1:
        raise ValueError("y must be 1-D")
    for name, arr in (
        ("variance", var_arr),
        ("toas", toas_arr),
        ("toa_mjd", mjd_arr),
    ):
        if arr.shape != n:
            raise ValueError(f"{name} shape {arr.shape} does not match y shape {n}")
    freqs_arr = None if freqs_mhz is None else np.asarray(freqs_mhz, dtype=float)
    if freqs_arr is not None and freqs_arr.shape != n:
        raise ValueError(
            f"freqs_mhz shape {freqs_arr.shape} does not match y shape {n}"
        )

    for name in solve.block_waveforms:
        if name not in block_kinds:
            raise KeyError(f"block_kinds missing entry for solve block {name!r}")
    # Extra keys in block_kinds are ignored (kept mapping is solve-column order).
    kinds_kept = {name: block_kinds[name] for name in solve.block_waveforms}

    extras: dict[str, np.ndarray] = {}
    if noise is not None and isinstance(
        noise, (EpochKernelNoise, ShermanMorrisonNoise)
    ):
        extras["ecorr_kernel"] = _kernel_ecorr_waveform(y_arr, solve, noise)
        kinds_kept["ecorr_kernel"] = "ecorr"

    waveforms = {**solve.block_waveforms, **extras}
    freqs_map = {
        k: np.asarray(v, dtype=float) for k, v in (block_frequencies or {}).items()
    }
    stage_specs = tuple(stages) if stages is not None else STANDARD_PTA_STAGES
    evaluated = _evaluate_stages(
        y_arr,
        var_arr,
        solve,
        kinds_kept,
        stage_specs,
        waveforms=waveforms,
        extra_waveforms=extras,
    )
    return WaveformAnalysis(
        y=y_arr,
        variance=var_arr,
        toas=toas_arr,
        toa_mjd=mjd_arr,
        freqs_mhz=freqs_arr,
        solve=solve,
        block_kinds=kinds_kept,
        block_frequencies=freqs_map,
        stages=evaluated,
        label=EB_LABEL,
        extra_waveforms=extras,
    )


# --------------------------------------------------------------------------- #
# Figdata I/O (pyarrow imported function-locally)
# --------------------------------------------------------------------------- #
def write_waveform_figdata(
    analysis: WaveformAnalysis,
    path: str | Path,
    *,
    pulsar_name: str,
    n_grid: int = 400,
    json_sidecar: bool = False,
) -> Path:
    """Write WaveformPanelArrays to a feather file (overwrite). Returns path.

    Builds the panel arrays via ``analysis.panel_arrays(n_grid=n_grid)``, packs
    the TOA vectors as pyarrow columns and the grid/summary/label into the single
    ``{"json": ...}`` schema-metadata key, then
    ``pyarrow.feather.write_feather(table, path)``.
    """
    try:
        import pyarrow as pa
        from pyarrow import feather
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "write_waveform_figdata requires pyarrow; install with `pip install pyarrow`"
        ) from exc

    path = Path(path)
    panels = analysis.panel_arrays(n_grid=n_grid)
    pydict = {
        "mjd": np.asarray(panels.mjd, dtype=float),
        "freq_mhz": np.asarray(panels.freq_mhz, dtype=float),
        "resid_us": np.asarray(panels.resid_us, dtype=float),
        "after_timing_us": np.asarray(panels.after_timing_us, dtype=float),
        "after_all_us": np.asarray(panels.after_all_us, dtype=float),
        "z": np.asarray(panels.z, dtype=float),
        "sigma_us": np.asarray(panels.sigma_us, dtype=float),
    }
    meta = {
        "format": FIGDATA_FORMAT,
        "label": panels.label,
        "pulsar": pulsar_name,
        "summary": analysis.summary(),
        "stage_rms_us": dict(panels.stage_rms_us),
        "grid": {
            "grid_mjd": np.asarray(panels.grid_mjd, dtype=float).tolist(),
            "red_mean_us": np.asarray(panels.red_mean_us, dtype=float).tolist(),
            "red_std_us": np.asarray(panels.red_std_us, dtype=float).tolist(),
            "dm_mean_us": np.asarray(panels.dm_mean_us, dtype=float).tolist(),
            "dm_std_us": np.asarray(panels.dm_std_us, dtype=float).tolist(),
        },
    }
    table = pa.Table.from_pydict(pydict, metadata={"json": json.dumps(meta)})
    feather.write_feather(table, path)

    if json_sidecar:
        sidecar = path.with_name(f"{path.stem}_waveform.json")
        sidecar.write_text(
            json.dumps(
                {
                    "format": FIGDATA_FORMAT,
                    "pulsar": pulsar_name,
                    "label": panels.label,
                    "summary": meta["summary"],
                    "stage_rms_us": meta["stage_rms_us"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return path


def load_waveform_figdata(path: str | Path) -> WaveformPanelArrays:
    """Load panel arrays written by write_waveform_figdata.

    Reads the feather table, rebuilds TOA vectors from columns and grid vectors
    from ``json["grid"]``; ``label`` and ``stage_rms_us`` come from the JSON blob.
    The full ``summary`` sub-object is provenance only — it is **not** carried on
    the returned ``WaveformPanelArrays`` (which has no summary field); re-derive it
    from a live ``WaveformAnalysis`` if needed.
    """
    try:
        from pyarrow import feather
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "load_waveform_figdata requires pyarrow; install with `pip install pyarrow`"
        ) from exc

    table = feather.read_table(Path(path))
    meta = json.loads(table.schema.metadata[b"json"])
    if meta.get("format") != FIGDATA_FORMAT:
        raise ValueError(
            f"unsupported figdata format {meta.get('format')!r}; "
            f"expected {FIGDATA_FORMAT!r}"
        )
    grid = meta["grid"]
    return WaveformPanelArrays(
        mjd=table["mjd"].to_numpy(),
        freq_mhz=table["freq_mhz"].to_numpy(),
        resid_us=table["resid_us"].to_numpy(),
        after_timing_us=table["after_timing_us"].to_numpy(),
        after_all_us=table["after_all_us"].to_numpy(),
        z=table["z"].to_numpy(),
        sigma_us=table["sigma_us"].to_numpy(),
        grid_mjd=np.asarray(grid["grid_mjd"], dtype=float),
        red_mean_us=np.asarray(grid["red_mean_us"], dtype=float),
        red_std_us=np.asarray(grid["red_std_us"], dtype=float),
        dm_mean_us=np.asarray(grid["dm_mean_us"], dtype=float),
        dm_std_us=np.asarray(grid["dm_std_us"], dtype=float),
        label=str(meta["label"]),
        stage_rms_us=dict(meta["stage_rms_us"]),
    )
