# flexfit — fast flexible-Φ timing + GP quick-look fits

A deterministic empirical-Bayes fit over a joint timing-and-Gaussian-process
reduced-rank basis, following the Enterprise `T`/`Phi` formulation. It
reconstructs quick-look waveforms (red noise, DM, chromatic, …) and timing
estimates for diagnostics and initialization.

**Package:** `pylk.flexfit` (incubated in the MetaPulsar repo under `src/pylk/`).  
**Design:** flexible-Φ EM fit under `pylk.flexfit`.

> **Quick-look empirical Bayes**, not inference. The fit learns a data-driven
> width in the transformed timing coordinate, so the result is an
> initialization/visualization estimate — never present it as a posterior under
> the original production prior.

## Layout

```
pylk/flexfit/
  basis.py         # BasisBlock / VarianceGroup / assemble  (pure data)
  noise.py         # NoiseOperator: DiagonalNoise, EpochKernelNoise, …
  flexible_phi.py  # conditional solve + staged bounded EB sweeps
  fasttnt.py       # epoch factorization for TᵀN⁻¹T (opt-in Factorization)
  projection.py    # project a free spectrum onto a physical Phi(theta)
  timing.py        # TimingModel protocol (relinearization interface)
  fastfit.py       # top-level orchestration + relinearization loop
  whitenoise.py    # Gibbs/ECM per-backend EFAC/EQUAD estimation
  waveform.py      # EB waveform stages / figdata
  adapters/
    discovery.py   # red/DM/chromatic blocks + white noise (Discovery)
    enterprise.py  # planned Enterprise gp_bases / gp_priors adapter (stub)
    nltiming.py    # timing J_z block + finite-difference sign check
```

### Fast-TNT factorization (opt-in)

For large multi-PTA pulsars, pass a `Factorization` into `fastfit` /
`fit_white_noise`, or call `factorize(model, toas=..., freqs_mhz=...)` before
`solve_flexible_phi`. Default remains the dense path. For ECORR-heavy pulsars
prefer Topology B: `dx.white_noise(psr, noisedict, ecorr=True)` returns an
`EpochKernelNoise` (ECORR in `N`, not as basis columns). Pin ECORR with the
operator as-is (mode 2.1), or learn it via
`fit_white_noise(..., kernel_ecorr=..., learn_kernel_ecorr=True)` and read
amplitudes with `ecorr_from_kernel(result.kernel)`. See
`feature_flexfit_fasttnt.md`.

Ownership: the timing `z`-space Jacobian and prior transform belong to
`nltiming`. GP bases and `Phi` conventions belong to Discovery **or** Enterprise
(via adapters). `flexfit` composes both; the dependency arrow is always
`flexfit -> {nltiming, discovery|enterprise}`, never the reverse.

## Usage

```python
from pylk.flexfit import fastfit
from pylk.flexfit.adapters import discovery as dx, nltiming as nx

ctx = ntm.for_pulsar(pulsar)          # nltiming TimingContext
nx.sign_check(ctx)                    # verify the timing convention by finite differences

timing = nx.timing_model(ctx, marginalize_all=True)   # analytically marginalize timing
blocks = [dx.red_noise_block(pulsar, components=30),
          dx.dm_noise_block(pulsar, components=30)]
noise = dx.white_noise_from_variance(white_variance)

fit = fastfit(noise=noise, blocks=blocks, timing=timing, n_sweeps=3)
whitened = fit.whitened_residuals()
red_wave = fit.waveform("red")
```

## Tests

```bash
# core only (NumPy/SciPy):
pytest tests/pylk/flexfit/test_core.py tests/pylk/flexfit/test_white_noise.py

# adapters + real Discovery/nltiming/MetaPulsar (devcontainer):
pytest tests/pylk/flexfit/
```
