# Feature proposal: Fast flexible-Phi timing and waveform fits

**Status:** Incubating implementation under `pylk.flexfit` (research algorithm;
not a stable public API contract)
**Scope:** Algorithm, placement, and implementation status. Companion to
[`../feature_pylk.md`](../feature_pylk.md).
**Home (this repo):** `src/pylk/flexfit/` → import as `pylk.flexfit`
**Ownership decision:** the timing `z`-space optimizer belongs in `nltiming`; the
flexible-`Phi` GP fit does **not**. It lives under the Pylk umbrella
(`pylk.flexfit`), incubated in the MetaPulsar repository until Pylk is extracted.
Paper diagnostics, notebooks, and the future Pylk workspace all consume this
one headless module.

## Implementation status

### Built (as of migration into `src/pylk/flexfit`)

| Piece | Module | Notes |
|-------|--------|--------|
| `BasisBlock` / `VarianceGroup` / `assemble` | `basis.py` | Pure data; RMS bounds; Fourier pair / per-column groups |
| `NoiseOperator`, diagonal + Sherman–Morrison | `noise.py` | ECORR-in-`N` supported |
| Conditional moments + staged EB sweeps | `flexible_phi.py` | Cholesky solve; jitter for broad timing prior |
| Spectrum projection objective | `projection.py` | Caller supplies `Phi(theta)` |
| `TimingModel` protocol + `fastfit` outer loop | `timing.py`, `fastfit.py` | Damped relinearization hook |
| Gibbs/ECM white-noise MPE | `whitenoise.py` | Per-backend EFAC/EQUAD (tnequad) |
| Discovery adapter | `adapters/discovery.py` | Red/DM/chrom blocks, WN, ECORR blocks, power-law project; also notebook helpers (`reconstruct_waveforms`, `map_powerlaw_hypers`) |
| nltiming adapter | `adapters/nltiming.py` | Marg/sampled timing blocks, `sign_check` |
| Core + adapter tests | `tests/pylk/flexfit/` | Pure core always; Discovery/nltiming/MetaPulsar skip cleanly |

### Not built / incomplete

| Piece | Notes |
|-------|--------|
| **Enterprise adapter** | `adapters/enterprise.py` is a **stub** (`NotImplementedError`). Planned: `gp_bases` / `gp_priors` block builders + WN + `project_powerlaw`, mirroring Discovery. Core already accepts Enterprise-built arrays. |
| Full Piece A nltiming APIs | See §"nltiming changes required" — some `J_z` / z-fit surface still partial; adapter uses design × prior Jacobian and optional evaluator today |
| Pylk workspace integration | No UI, caching, or revision plumbing yet — solver only |
| Variable white noise inside EB sweeps as first-class `fastfit` option | Separate `fit_white_noise` path exists; not unified with outer timing loop |
| Multi-pulsar / common processes | Out of initial scope |
| Production posterior labelling / full Bayesian product | Explicitly out of scope (quick-look EB only) |

### Former location

Prototype lived at `paper/code/flexfit/`. That path is a redirect stub; do not
add new code there.

## Purpose

Provide a fast, deterministic, empirical-Bayes fit over a joint
timing-and-Gaussian-process basis. The immediate uses are quick-look waveform
reconstruction and diagnostics for the MetaPulsar paper, interactive analysis,
and the Pylk workspace. The result is an initialization and visualization
estimate, not a substitute for production posterior inference.

The model follows the Enterprise `Tmat`/`phi` formulation. Timing-model
directions, Fourier red noise, dispersion variations, chromatic processes,
basis ECORR, and later user-defined bases all enter one matrix:

\[
T = [J_z, F_{\rm red}, F_{\rm DM}, F_{\rm chrom}, U_{\rm ECORR}, \ldots].
\]

The corresponding low-level coefficients are

\[
b = [z, a_{\rm red}, a_{\rm DM}, a_{\rm chrom},
     a_{\rm ECORR}, \ldots],
\]

with a zero-mean Gaussian prior

\[
b \mid \Phi \sim N(0, \Phi).
\]

`Phi` is diagonal, but heterogeneous: each diagonal entry belongs to an
explicit variance group. Different groups may have different values and
bounds, and several diagonal entries may be tied to the same value.

## Placement and ownership

This feature is two separable pieces with different natural homes. The plan's
earlier draft fused them inside `nltiming`; that is corrected here.

### Piece A — timing `z`-space optimizer (owned by `nltiming`)

Everything about residuals as a function of transformed timing coordinates:
the timing Jacobian `J_z`, nonlinear Gauss-Newton / trust-region
relinearization, prior bounds, and immutable fit results. This is in
`nltiming`'s charter and already half-built in
`ref-packages/nltiming/src/nltiming/evaluator.py` (`TimingEvaluator.fit`, today
physical-`delta` coordinates with diagonal white errors) and
`bijectors.py` (`PriorBijector.jacobian_diag_delta_from_z`). The required
extensions are listed under "nltiming changes required" below.

### Piece B — flexible-`Phi` GP fit (NOT in `nltiming`)

The joint reduced-rank Gaussian-process solve over red/DM/chromatic/ECORR
bases, the variance-group empirical-Bayes updates, and the physical-spectrum
projection. This is a noise-model inference engine, and it stays out of
`nltiming` for four reasons:

- **Charter.** `nltiming`'s stated scope is nonlinear timing components *for*
  Discovery and Enterprise; its README explicitly leaves correlated-noise
  inference to those frontends. A reduced-rank GP noise fit is exactly what it
  disclaims.
- **Dependency inversion.** An in-`nltiming` implementation would add
  `frontends/discovery.py` / `frontends/enterprise.py` adapters that translate
  those frontends' noise models *into* `nltiming` arrays — the opposite of
  `nltiming` feeding the frontends.
- **Duplication.** Discovery already owns `fourierbasis` / `fourierbasis_dm` /
  `fourierbasis_chrom`, `powerlaw`, `freespectrum`, and assembles the
  `(N + T Phi T^T)` Woodbury structure this plan re-derives; Enterprise has the
  equivalent. Re-authoring `fref`, ECORR grouping, and power-law conventions in
  `nltiming` invites drift from the production stacks this fit is meant to
  initialize.
- **Semantic label.** Learning a data-driven width in `z` produces an object
  that is *not* the `nltiming` prior-transformed posterior (see "Timing
  coordinates"). Placing it inside the package whose identity is the correct
  timing prior transform invites exactly that confusion.

### Ownership table

| Layer | Owner |
|-------|-------|
| Timing `z`-space Jacobian, relinearization, bounded local fit | **nltiming** (Piece A) |
| Prior-transform Jacobian and supported-prior contract | **nltiming** |
| GP basis blocks (red/DM/chrom/ECORR), `Phi` variance groups | Discovery/Enterprise, adapted by **flexfit** |
| Joint conditional solve, staged sweeps, spectrum projection | **flexfit** (Piece B) |
| Interactive workspace, revisions, provenance, plotting | **Pylk** |

### Consumers and dependency direction

`flexfit` composes `nltiming` (for the timing block and prior transform) and
**Discovery and/or Enterprise** (for GP bases and white-noise conventions). The
paper diagnostics and Pylk both consume `pylk.flexfit`; neither requires it to
live in `nltiming`. The dependency arrow is always

```text
pylk.flexfit  →  {nltiming, discovery?, enterprise?}
```

never the reverse. Adapters are optional integrations; the NumPy/SciPy core
imports neither frontend. See [`../feature_pylk.md`](../feature_pylk.md)
§"Fast flexible GP model for visualization".

## Timing coordinates

The timing block is expressed in `nltiming`'s prior-transformed coordinates
`z`, before the optional standardization or whitening transform `x`:

\[
J_z = \frac{\partial r}{\partial z}
    = \frac{\partial r}{\partial \Delta\theta}
      \frac{\partial \Delta\theta}{\partial z}.
\]

This incorporates the physical prior transformation and gives a useful
coordinate system for timing parameters with different units, scales, and
bounds. The two factors have different owners:

- `d(delta_theta)/dz`, the **prior-transform Jacobian**, is owned by
  `nltiming`. For every supported *proper, continuous, scalar* prior it is
  available analytically (see below), so `nltiming` never needs to
  differentiate through a SciPy implementation.
- `dr/d(delta_theta)`, the **timing Jacobian**, is owned by the timing
  backend. JUG autodiff supplies it when available; other engines may provide a
  backend-native design matrix or a finite-difference fallback, each with an
  explicit capability and fidelity label.

`J_z` is their product. An analytic prior-transform Jacobian therefore does not
by itself guarantee an analytic `J_z`: the backend must still supply
`dr/d(delta_theta)` at the (possibly nonlinear) anchor. Conversely, an
autodiff-capable backend does not remove the requirement that the prior admit a
transform Jacobian at all. Both factors are Piece A: `nltiming` should expose
`J_z` directly (see "nltiming changes required"), so `flexfit` consumes a ready
timing block rather than re-assembling it.

### Prior-transform Jacobian

The transform is the probability-integral transform (PIT) from the standard
normal `z` to the physical increment `delta_theta`,

\[
\Delta\theta = F^{-1}\!\left(\Phi_{\rm N}(z)\right),
\]

where `F` is the physical prior CDF and `Phi_N` the standard-normal CDF.
Differentiating `F(delta_theta) = Phi_N(z)` gives a diagonal Jacobian in closed
form,

\[
\frac{d\Delta\theta}{dz} = \frac{\phi_{\rm N}(z)}{p(\Delta\theta)},
\]

with `p` the physical prior PDF and `phi_N` the standard-normal PDF. This needs
only the prior's `ppf` and `logpdf`, never a derivative of SciPy's `ppf`:

```python
from scipy.special import ndtr
from scipy.stats import norm

delta = prior.ppf(ndtr(z))
d_delta_d_z = np.exp(norm.logpdf(z) - prior.logpdf(delta))
```

The result is a diagonal (per-axis) Jacobian evaluated in NumPy/SciPy and
combined with the backend timing Jacobian outside any JAX trace:

```python
j_delta = timing_backend_jacobian_delta(...)      # dr / d(delta_theta)
transform_diag = prior_bijector.jacobian_diag_delta_from_z(z, np)
j_z = j_delta * transform_diag[None, :]
```

so an autodiff backend such as JUG never has to trace through the SciPy prior.
For the priors `nltiming` supports explicitly today — normal, uniform,
log-uniform, and truncated normal — this is already implemented analytically in
`ref-packages/nltiming/src/nltiming/bijectors.py`
(`PriorBijector.jacobian_diag_delta_from_z`).

### Supported-prior contract

The fast-fit path accepts a prior only if it can define this transform. The
minimal contract is a proper, continuous scalar distribution exposing `ppf` and
`logpdf`:

```python
class ScalarPrior(Protocol):
    def ppf(self, u: np.ndarray) -> np.ndarray: ...
    def logpdf(self, value: np.ndarray) -> np.ndarray: ...
```

Advanced priors may instead supply the transform and its derivative directly,
bypassing the PIT formula:

```python
class ScalarPriorTransform(Protocol):
    def delta_from_z(self, z: np.ndarray) -> np.ndarray: ...
    def derivative_delta_from_z(self, z: np.ndarray) -> np.ndarray: ...
```

Under either contract every accepted prior yields an analytic prior-transform
Jacobian. Priors that fall outside it must be rejected or replaced with a proper
override; the fast path must not silently substitute a default. The excluded
cases are:

- **Improper priors** (e.g. PINT's unbounded uniform) have no normalized CDF and
  hence no PIT transform; require a proper fallback or explicit override.
- **PDF-only priors** that expose `pdf`/`logpdf` but no reliable `cdf`/`ppf`
  cannot define the transform automatically.
- **Discrete, mixed, or point-mass priors** have no ordinary Jacobian.
- **Densities that vanish** on relevant regions produce a singular transform
  Jacobian.
- **Hard support boundaries** drive the derivative toward zero or infinity and
  need clipping or trust-region handling.
- **Correlated multivariate priors** require a full transport Jacobian, not the
  per-axis diagonal formula above.
- **Backends that cannot supply `dr/d(delta_theta)`** at an arbitrary nonlinear
  point block `J_z` regardless of the prior; an analytic prior Jacobian does not
  cover this factor.

### Effective prior and relinearization

Learning a timing variance `rho_i` in `z` changes the effective prior from the
standard `nltiming` prior `z_i ~ N(0, 1)` to `z_i ~ N(0, rho_i)`. That is a
deliberate feature of this flexible quick-look model, and the central reason
Piece B is labelled a visualization estimate rather than inference: the prior
transform still defines the physical coordinate map and its bounds, but the
result must not be presented as inference under the original production prior.

At a nonlinear timing anchor `z0`, use the affine approximation

\[
r(z) \simeq [r(z_0) - J_z(z_0) z_0] + J_z(z_0)z.
\]

Representing the timing coefficient by the absolute `z`, rather than only a
local increment, retains a zero-centered Gaussian coefficient prior after
relinearization. Each adapter must test its residual sign convention by finite
differences.

## Diagonal Phi and variance groups

A diagonal `Phi` does not require every diagonal entry to be independently
estimated. Recommended defaults are:

- one variance per transformed timing coordinate;
- one shared variance for each Fourier sine/cosine pair;
- configurable sharing for ECORR epochs, backends, or selections;
- explicit groups for other chromatic or user-provided bases.

Sine/cosine pairs should normally share a variance even in the flexible model.
Estimating them separately makes the result depend on the arbitrary Fourier
phase origin. Tying their diagonal entries retains phase-rotation invariance
without introducing off-diagonal covariance.

The joint basis is not orthogonal. In particular, spin timing parameters are
strongly covariant with the lowest-frequency Fourier modes. The proposed
algorithm does not claim that those correlations vanish. Instead, it uses a
short staged iteration that has worked well in practice and mirrors the useful
initial behavior of the Enterprise Extensions Gibbs sampler.

## Staged quick-look iteration

The number of user-facing low-level/hyperparameter sweeps is configurable:

```python
n_sweeps: int = 3
```

It must satisfy `n_sweeps >= 2`; the default is three.

### Sweep 1: effectively marginalize timing directions

On the first sweep, assign every timing-coordinate variance the fixed
Enterprise-style value

```python
INITIAL_TIMING_VARIANCE = 1e40
```

and exclude timing groups from the hyperparameter update. The implementation
should work with the corresponding precision, `1e-40`, so it does not need to
form large intermediate covariance entries. This is a very broad finite
Gaussian prior, conventionally used as an effective improper-prior
marginalization; it is not mathematically identical to an infinite prior.

With timing directions effectively marginalized, update the Fourier and other
non-timing coefficients and their bounded variance groups. Despite the strong
spin/low-frequency covariance, this first projected solve has produced useful
Fourier coefficient values in practice. It also parallels the first useful
stage of the Gibbs approach, with the important difference that the timing
basis here already includes the physical `nltiming` prior transformation.

### Sweeps 2 through n: infer timing Phi as well

Beginning with the second sweep, make the timing variance groups eligible for
the same bounded hyperparameter update as the other blocks. Recompute all
low-level coefficient moments jointly using the complete `T` matrix and the
updated heterogeneous `Phi`.

The default three-sweep sequence is therefore:

1. fixed `Phi_timing = 1e40`; update non-timing coefficients and variances;
2. update timing and non-timing coefficients and variance groups jointly;
3. repeat the joint update as a cheap stabilization sweep.

This fixed small sweep count is the primary quick-look control. An optional
convergence tolerance may stop after the second sweep, but must never reduce a
requested run to only one sweep. Advanced callers may request more sweeps for
diagnostics; a large sweep count should not turn this API into a production
sampler.

For nonlinear timing, the whole sweep sequence may sit inside a small outer
relinearization loop. After obtaining a proposed timing mean, apply damping or
a trust-region limit, evaluate the full nonlinear residual, rebuild `J_z`, and
repeat. The `J_z` rebuild is a Piece-A (`nltiming`) call; the outer loop is
owned by `flexfit`. The initial implementation should default to no more than
three outer iterations.

## Conditional low-level solve

Given residual vector `y`, base covariance `N`, joint basis `T`, and current
diagonal `Phi`, compute

\[
\Sigma = (T^T N^{-1}T + \Phi^{-1})^{-1},
\qquad
m = \Sigma T^T N^{-1}y.
\]

For block `k`, the quick-look waveform is its conditional mean

\[
w_k = T_k m_k.
\]

All blocks must be solved jointly. Solving red, DM, timing, or ECORR blocks
independently would discard the cross-block covariance that motivates the
staged initialization.

A minimal backend-neutral implementation is:

```python
def conditional_moments(y, basis, phi, noise):
    ninv_y = noise.solve(y)
    ninv_t = noise.solve(basis)

    precision = basis.T @ ninv_t + np.diag(1.0 / phi)
    factor = scipy.linalg.cho_factor(precision, lower=True)

    mean = scipy.linalg.cho_solve(factor, basis.T @ ninv_y)
    covariance = scipy.linalg.cho_solve(factor, np.eye(len(phi)))
    second_moment = mean**2 + np.diag(covariance)
    return mean, covariance, second_moment
```

The linear algebra itself is small; the weight is the basis and spectrum
semantics wrapped around it, which is why those live with Discovery/Enterprise
and `flexfit` rather than in `nltiming`. The reference implementation may form
the full coefficient covariance. A later optimized implementation should obtain
only the selected inverse diagonal when the basis becomes large.

## Bounded hyperparameter updates

The Enterprise Extensions update alternates a random low-level coefficient
draw with a bounded inverse-gamma variance draw. A deterministic reactive fit
must not insert only the conditional mean into that same update: weakly
constrained coefficients would then look artificially close to zero.

Use the conditional second moment

\[
s_j = E[b_j^2 \mid y, \Phi] = m_j^2 + \Sigma_{jj}.
\]

For group `g`, maximum-likelihood EM gives

\[
\rho_g^{\rm new} = \frac{1}{n_g}\sum_{j\in g}s_j.
\]

For an inverse-gamma prior

\[
p(\rho_g) \propto
\rho_g^{-\alpha_g-1}\exp(-\beta_g/\rho_g),
\]

the corresponding deterministic MAP update is

\[
\rho_g^{\rm new} =
\frac{\sum_{j\in g}s_j + 2\beta_g}
     {n_g + 2\alpha_g + 2}.
\]

Clip or solve this update within explicit lower and upper bounds. The API may
also expose an optional stochastic bounded inverse-gamma draw for Gibbs parity
tests, but deterministic second-moment updates should be the interactive
default.

## Variance bounds and basis scaling

Bounds should be configured in terms of induced residual RMS rather than raw
coefficient variance. For a variance group with basis `T_g`, define

\[
q_g = \frac{1}{n_{\rm TOA}}\operatorname{tr}(T_gT_g^T).
\]

Then an induced waveform range
`sigma_min <= sigma_g <= sigma_max` corresponds to

\[
\rho_{g,\min} = \frac{\sigma_{\min}^2}{q_g},
\qquad
\rho_{g,\max} = \frac{\sigma_{\max}^2}{q_g}.
\]

This definition applies to transformed timing columns, Fourier pairs,
chromatic bases, and ECORR bases while respecting their different native
scales. Keep canonical Enterprise/Discovery basis normalization and record the
RMS conversion as metadata. Internal numerical column scaling is permitted
only if it is exactly mapped back to canonical coefficient and `Phi` units.

## Projection onto a physical hyperparameter model

After estimating the flexible variance groups, fit a desired physical model
`Phi(theta)`. Raw unweighted least squares is not the preferred default because
the entries span many orders of magnitude and have unequal information.

With the flexible second moments held fixed, minimize

\[
Q(\theta) = \frac{1}{2}\sum_j
\left[\log\phi_j(\theta) +
\frac{s_j}{\phi_j(\theta)}\right].
\]

This expected Gaussian complete-data objective uses only the flexible
coefficient summary after the joint solve; it no longer touches the TOA
residual vector or covariance. It should therefore provide a very fast initial
estimate for power-law or other production hyperparameters. Least squares in
`log(phi)` may be retained as an initializer or diagnostic. The spectral model
family (power law, broken power law, free spectrum) should reuse
Discovery/Enterprise conventions rather than redefining them.

## Package structure

The solver lives under the **Pylk** package umbrella, never in `nltiming`.

### Current layout (committed)

```text
src/pylk/                          # Pylk incubation root (this repo)
  feature_pylk.md                  # workbench proposal
  __init__.py
  flexfit/                         # ← this module (import: pylk.flexfit)
    feature_flexible_fit.md        # this document
    basis.py                       # BasisBlock / VarianceGroup / assemble
    noise.py                       # DiagonalNoise, ShermanMorrisonNoise
    flexible_phi.py                # conditional solve + staged EB sweeps
    projection.py                  # Phi(theta) projection objective
    timing.py                      # TimingModel protocol
    fastfit.py                     # orchestration + relinearization loop
    whitenoise.py                  # Gibbs/ECM EFAC/EQUAD
    adapters/
      discovery.py                 # Discovery GP bases + WN (implemented)
      enterprise.py                # Enterprise adapter (stub / planned)
      nltiming.py                  # timing J_z + sign_check (implemented)
tests/pylk/flexfit/                # pytest (core + optional integration)
```

Paper validation under `paper/code/validation/` and notebooks import
`pylk.flexfit` from the installed package (`pip install -e .`); they must not
vendor a second solver. Per repository policy, `src/` never imports from
`paper/`.

### Extraction later

When Pylk is extracted from MetaPulsar, `pylk.flexfit` moves with it. The
import path stays `pylk.flexfit`. MetaPulsar remains a client (via shared
install), not the reverse owner of the algorithm.

- `basis.py` holds pure data containers. It does **not** contain spectral
  constructors such as `RedFourierBlock`; those live in adapters and defer to
  Discovery/Enterprise.
- `flexible_phi.py` implements the conditional Gaussian solve and staged
  bounded variance updates.
- `fastfit.py` combines the timing block, optional relinearization, joint
  basis fits, and immutable result objects.
- adapters translate Discovery **or** Enterprise models into common arrays.

The numerical core depends only on NumPy/SciPy; `nltiming`, Discovery,
Enterprise, JAX, and JUG remain optional integrations reached through adapters.

Suggested core types:

```python
@dataclass(frozen=True)
class VarianceGroup:
    name: str
    indices: np.ndarray
    lower: float
    upper: float
    alpha: float = 0.0
    beta: float = 0.0
    update_from_sweep: int = 1


@dataclass(frozen=True)
class BasisBlock:
    name: str
    matrix: np.ndarray
    coefficient_names: tuple[str, ...]
    groups: tuple[VarianceGroup, ...]
    kind: Literal["timing", "red", "dm", "chromatic", "ecorr", "custom"]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class FlexiblePhiResult:
    coefficient_mean: np.ndarray
    coefficient_covariance: np.ndarray
    phi_diagonal: np.ndarray
    block_waveforms: Mapping[str, np.ndarray]
    timing_z: np.ndarray | None
    timing_delta: np.ndarray | None
    n_sweeps: int
    outer_iterations: int
    bound_hits: tuple[str, ...]
    diagnostics: Mapping[str, object]
```

For timing groups, `update_from_sweep=2` and the initial variance is `1e40`.
For ordinary non-timing groups, `update_from_sweep=1`.

## Public API sketch

```python
from pylk.flexfit import fastfit
from pylk.flexfit.adapters import discovery as dx, nltiming as nx
# later: from pylk.flexfit.adapters import enterprise as ex

ctx = ntm.for_pulsar(pulsar)
timing = nx.timing_model(ctx, marginalize_all=True)
blocks = [
    dx.red_noise_block(pulsar, components=30),
    dx.dm_noise_block(pulsar, components=30),
]
noise = dx.white_noise_from_variance(white_variance)
fit = fastfit(noise=noise, blocks=blocks, timing=timing, n_sweeps=3)
```

Enterprise path (once `adapters/enterprise.py` is implemented) is the same
call shape with `ex.red_noise_block` / `ex.white_noise` / …; the core does not
branch on frontend. Hand-built `BasisBlock`s from
`enterprise.signals.gp_bases` already work without that adapter.

The API is still research-grade (quick-look EB, not a frozen contract).

## nltiming changes required

Piece B needs a small, in-charter set of additions to `nltiming` so it can
consume a ready timing block instead of re-deriving one. These are the only
`nltiming` changes this feature implies.

1. **Expose `J_z` (a z-frame timing Jacobian).** Today
   `TimingEvaluator.jacobian(frame=...)` returns `d residual_delta /
   d delta_theta` regardless of `frame`
   (`ref-packages/nltiming/src/nltiming/evaluator.py`). Add a genuine z-frame
   result that chains the backend delta-frame Jacobian with
   `PriorBijector.jacobian_diag_delta_from_z`, returned with a capability and
   fidelity label (autodiff / analytic design matrix / finite-difference).

2. **Extend the local fit to `z`-space and nonlinearity.** `TimingEvaluator.fit`
   currently hardcodes `frame="delta"`, diagonal white errors, and no bounds or
   prior penalty. Add (or add a sibling for) a `z`-coordinate Gauss-Newton /
   trust-region fit with parameter bounds and damping, still diagonal-white.
   This is the "transformed-space iterative least squares" that both this
   document and `feature_pylk.md` reference as the default fast optimizer.

3. **Attach a residual-evaluation fidelity tier to results.**
   `TimingEvaluation` and `TimingFitResult` carry no tier label. Add the T0-T4
   vocabulary from `ref-packages/jug/TEMPO2_NATIVE_MODES.md` so consumers (Pylk,
   `flexfit`) can enforce escalation-at-acceptance for large nonlinear moves.

4. **Publish the prior-transform contract.** Promote
   `jacobian_diag_delta_from_z` to a documented public surface, add the
   `ScalarPrior` / `ScalarPriorTransform` protocols above, and add validation
   that rejects improper, PDF-only, discrete, or singular priors with clear
   capability errors rather than silent substitution.

5. **Reinforce the charter (docs only).** Record in `nltiming`'s README /
   `UPSTREAM_INTEGRATION.md` that `nltiming` provides the timing block, the
   prior transform, and the `z`-space fit, and explicitly does **not** own GP
   bases, `Phi` inference, or spectra. This prevents the placement question from
   reopening.

## MetaPulsar paper validation

The paper validation lives under `paper/code/validation/` and imports
`pylk.flexfit` plus development `nltiming` (timing block) and Discovery
(bases). It must not ship a second copy of the solver.

The first validation should cover:

1. one pulsar and fixed white-noise hyperparameters;
2. selected timing coordinates in `z`;
3. red and DM Fourier bases;
4. optional basis ECORR;
5. timing variance fixed to `1e40` on sweep one;
6. timing and non-timing variance updates from sweep two;
7. the default three-sweep result and sensitivity to additional sweeps;
8. nonlinear residual re-evaluation after the timing update;
9. power-law projection from the flexible spectrum;
10. comparison with Discovery, Enterprise, the Enterprise Extensions Gibbs
    sampler, and full posterior waveform reconstructions;
11. phase-origin invariance from tying Fourier sine/cosine pairs;
12. warm-start and cold-start latency.

Special attention should go to covariance between spin parameters and the
lowest Fourier modes. Validation should document that the staged heuristic is
useful in practice, identify datasets where it fails, and report sensitivity
to the timing initialization, variance bounds, basis size, and sweep count.

## Pylk integration

Pylk calls the headless `pylk.flexfit` API asynchronously and caches results by
scientific revision and fit configuration. Pylk owns task cancellation,
invalidation, preview/accept state, plotting, and provenance; it does not own
the basis solver or hyperparameter mathematics. See
[`../feature_pylk.md`](../feature_pylk.md) §"Fast flexible GP model for
visualization" and §"Reactive computational model" for the surrounding
workspace behavior.

Every displayed result should be labelled **quick-look empirical Bayes**, and
should retain:

- timing anchor and residual-evaluation fidelity tier;
- all basis definitions and coefficient groupings;
- canonical basis units and normalization metadata;
- initial and final `Phi` values;
- variance bounds and bound hits;
- requested and completed sweep counts;
- nonlinear outer-iteration count;
- white-noise snapshot;
- condition and convergence diagnostics;
- per-block conditional means and uncertainties.

## Initial scope and later extensions

The first implementation should remain narrow: one pulsar, fixed white noise,
timing plus red and DM bases, optional basis ECORR, deterministic bounded
second-moment updates, and a power-law projection.

Later work may add variable white noise, free chromatic indices, solar-wind,
band and group noise, common multi-pulsar processes, non-diagonal `Phi`, sparse
or iterative linear algebra, stochastic Gibbs updates, and richer physical
hyperparameter models. Those extensions should follow validation of the small
joint solver rather than shape its first API.
