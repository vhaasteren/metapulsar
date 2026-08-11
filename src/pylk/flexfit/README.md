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
  noise.py         # NoiseOperator: DiagonalNoise, ShermanMorrisonNoise
  flexible_phi.py  # conditional solve + staged bounded EB sweeps
  projection.py    # project a free spectrum onto a physical Phi(theta)
  timing.py        # TimingModel protocol (relinearization interface)
  fastfit.py       # top-level orchestration + relinearization loop
  whitenoise.py    # Gibbs/ECM per-backend EFAC/EQUAD estimation
  adapters/
    discovery.py   # red/DM/chromatic blocks + white noise (Discovery)
    enterprise.py  # planned Enterprise gp_bases / gp_priors adapter (stub)
    nltiming.py    # timing J_z block + finite-difference sign check
```

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
