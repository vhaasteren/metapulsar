## Method: Direct combination (“MetaPulsar”)

### Problem statement and summary

Given multiple public PTA data sets for the **same** pulsar—each consisting of a **timing model** (a `.par` file) and **times of arrival** (a `.tim` file)—MetaPulsar constructs a single “metapulsar” that can be analyzed with standard PTA likelihoods without first manually re‑deriving a common timing solution. The procedure **does not modify the TOAs**; it only organizes the **deterministic timing model** across PTAs, and then builds the **combined design matrix** and metadata needed by Enterprise/Discovery.

After analytic marginalization over timing‑model parameters, the likelihood depends on the **column space** of the design matrix ( **M** ) rather than on the specific nominal parameter values ( β₀ ). Our procedure guarantees that the relevant column space is the same as in a traditional manual combination, so it is **statistically equivalent** to a full re‑timing while being vastly simpler and deterministic.

> **Terminology.** Throughout this method note, **design matrix** / **M** means
> the raw **fitter basis** in the PINT/tempo2 convention,
> \(r(\theta+\delta)\approx r(\theta)-M\delta\). That is the same object classical
> PTA packages call a design matrix. It is **not** the residual Jacobian
> \(J=\partial(\Delta r)/\partial\theta\) used by nonlinear timing engines; that
> object is named `residual_jacobian` and must not be called a design matrix.
> See [`design_matrix_terminology.md`](design_matrix_terminology.md).

### Inputs and conventions

For each PTA (p) that observed a given pulsar, MetaPulsar requires:

* a `.par` file specifying the **deterministic timing model** (astrometry, spin, binary, dispersion, and instrument/telescope‑specific deterministic delays such as **JUMPs**, **FD** coefficients, and overall **phase offsets**), and
* a `.tim` file with TOAs and their formal uncertainties.

Let ( **d**_p ) denote the vector of residuals for PTA (p) when linearized about its nominal model ( β_{0,p} ), and let ( **M**_p ) be the corresponding **design matrix** — the raw fitter basis with sign \(r(\theta+\delta)\approx r(\theta)-M\delta\), in public fit units. The full data vector is the concatenation ( **d** = ⨁_p **d**_p ). White‑ and red‑noise hyperparameters (EFAC/EQUAD/ECORR and RN/DM GP parameters) are **not** part of the deterministic timing model and are handled in the subsequent noise inference; MetaPulsar leaves them unchanged at this stage.

MetaPulsar uses **PINT** and **Tempo2/libstempo** to parse/realize timing models, and MetaPulsar-owned `_PtaTimingData` records to hold per-PTA arrays. The implementation provides two combination modes:

* **shared** (default): share selected astrophysical timing‑model components across PTAs while preserving detector‑specific timing‑model terms;
* **per_pta**: leave all `.par` files untouched and compose them as‑is (useful for diagnostics; everything remains PTA‑specific).

### Step 0: Aggregate switches and the common deterministic surface

For **multi‑PTA** combinations, before anything else, MetaPulsar normalizes what
the `.par` files are allowed to say:

1. Tempo2's aggregate `TEMPO1` line is expanded into the six explicit states it
   selects at once (`UNITS TDB`, `TIMEEPH FB90`, `DILATEFREQ N`,
   `PLANET_SHAPIRO N`, `T2CMETHOD TEMPO`, `CORRECT_TROPOSPHERE N`), then removed.
   Values already stated explicitly win.
2. Deterministic terms outside the common PINT/Tempo2 surface are **stripped by
   default**, with a warning naming the PTA and every removed key. Setting
   `AlignmentPolicy(unsupported="error")` turns the same finding into a
   `ValueError` that lists every offender instead.

The stripped families are, by engine set:

| Present engine | Removed |
|---|---|
| PINT | `EPHEM_FILE`/`EPH_FILE`, `EOP_FILE`, `CLK_CORR_CHAIN`, `NE_SW_SIN`, `NE_SW_IFUNC`, `DMMODEL` with `_DM`/`_CM`/`DMOFF` and its `CONSTRAIN DMMODEL*` entries, `SATJUMP`, second proper‑motion derivatives, `PMRV`, `DSHK`, `D_AOP`, telescope‑position terms |
| Tempo2 | `SWM 1` controls (`SWP`, `NE_SW1+`, `SWEPOCH`), `SWX*`, DMWaveX, `VLBIA*` |
| Both (mixed) | achromatic `WAVE*`/WaveX/`IFUNC*`, `CM` and `CM1+`, `CMX*`/`CHROMX*`/CMWaveX, deterministic chromatic Gaussian / shapelet / event terms, `DMASSPLANET*`, `DPHASEPLANET*`, glitches, piecewise spindown, exponential dips and events |

Matching is by exact name, anchored prefix, or anchored regular expression.
Noise hyperparameters (EFAC/EQUAD/ECORR, red noise, DM GP, ordinary chromatic
noise, `DMJUMP`) are **never** matched — there is no generic “starts with
`DM`/`CM`/`TN`” rule.

`IPM` and `SWM` are value‑dependent: `IPM 0` (Tempo2's interplanetary medium
off) and `SWM 1` (PINT's non‑constant solar wind) are violations, and under the
default policy they are normalized rather than merely dropped — see Step 3.

A **single‑PTA** pulsar has no second engine or reference to align against, so
Step 0 and the cross‑PTA parts of Step 3 are skipped and its native timescale is
preserved. The `shared` strategy can still perform its normal component and
dispersion cleanup (including DMX removal when dispersion is selected) and can
make Tempo2's implicit `NE_SW` explicit. Use `per_pta` when the release model
must remain byte-for-byte unchanged.

Users who need engine‑native terms preserved should use
`combination_strategy="per_pta"`, which performs no alignment at all.

### Step 1: Unit normalization

For the **shared** strategy MetaPulsar uses this timescale policy:

1. Mixed PINT/Tempo2 stacks are normalized to explicit TDB, including an
   all‑TCB input collection. PINT does not evaluate TCB models at all: by
   default (`allow_tcb=False`) it refuses a TCB par outright, and only with
   `allow_tcb=True` — which MetaPulsar passes — does it convert the model to
   TDB at read time. That conversion never appears in the serialized par,
   which is why the mixed stack must expose it explicitly.

   * For PINT models MetaPulsar re‑emits the model in TDB;
   * For Tempo2 models MetaPulsar calls the `transform tdb` plugin (both paths are implemented).

2. A genuinely mixed TCB/TDB multi‑PTA collection is normalized to TDB even
   when all PTAs use the same engine.

3. A homogeneous single‑engine collection keeps its common native timescale,
   as does a single PTA.

4. A Tempo2 par with no `UNITS` line counts as TCB, which is what Tempo2
   assumes, and `UNITS SI` is accepted as the Tempo2 spelling of TCB.

5. Converted text is re‑parsed and required to carry exactly one explicit
   `UNITS TDB`; duplicate active `UNITS` lines are rejected rather than silently
   resolved.

The target is one explicit, auditable unit convention. The `per_pta` strategy
performs no unit conversion at all.

**No TOA samples are modified** in this step or any subsequent step of this method.

### Step 2: Merge astrophysical timing‑model components

Within `ParameterManager._make_parameters_shared`, MetaPulsar merges selected **astrophysical** components across PTAs by **copying parameter values from the reference PTA** into the other `.par` files *for those components only*. The reference PTA is chosen from the (optionally user‑ordered) PTA list; by default it is the first PTA, and a convenience function can select the PTA with the longest timespan. The set of components is configurable and defaults to

* `astrometry` (RAJ/DECJ or ELONG/ELAT, proper motions, etc.),
* `spindown` (F0, F1, …),
* `binary` (Keplerian and post‑Keplerian parameters),
* `dispersion` (baseline DM and its low‑order derivatives).

Concretely:

* For each component, MetaPulsar discovers its parameters in each PTA’s timing model using PINT’s model metadata and a transparent alias resolver (e.g., `RAJ`/`ELONG`, `DECJ`/`ELAT`, `TASC`/`T0`, etc.).
* In non‑reference PTAs MetaPulsar **removes any existing values** for those component parameters and **inserts the reference PTA’s values**. This ensures that all PTAs linearize around the same astrophysical trajectory.

#### Dispersion special handling

To avoid PTA‑specific *DMX* implementations and make the deterministic part of the dispersion model uniform, MetaPulsar:

* removes **DMX** parameters if present,
* preserves each PTA's local **DM** reference value and marks it **free** (by default via `exclude_from_shared=("DM",)`),
* defines a fixed **DMEPOCH** (copied from the reference; frozen), and
* optionally inserts **DM1** and **DM2** (default: present and **free**, initialized at 0).

For dispersion, MetaPulsar removes DMX terms and keeps the Taylor DM evolution consistent, but `DM` itself is PTA-local by default. The `DM` value in a par file is the reference DM used when producing that PTA's TOAs, so the default `exclude_from_shared=("DM",)` preserves those reference values while exposing `DM_<pta>` fit parameters in the combined design matrix.

This choice keeps the deterministic dispersion expansion identical across PTAs while leaving the **stochastic DM process** (DM GP) to the noise model, as is standard practice.

> **Detector‑specific timing‑model parameters.**
> Terms that describe *deterministic* instrument/telescope‑dependent delays—e.g., **JUMPs**, **FD** coefficients, and overall **phase offsets**—are part of the timing model and are **not** made consistent. They remain **PTA/backend specific**. By contrast, **EFAC/EQUAD/ECORR** are **noise** hyperparameters (not timing‑model parameters) and are *never* touched here.

Terminology: in MetaPulsar we use the word consistent to describe model components or parameters that are common between data of different PTAs, and that we 'lock' together (i.e. they become the same parameters or model component).

### Step 3: Make timing‑model conventions consistent

Still inside `_make_parameters_shared`, MetaPulsar calls
`_apply_shared_convention_rules` before the shared `.par` strings are written.
The ordering is fixed: numerically transform ecliptic astrometry, align the
solar wind, resolve and apply the reference conventions, apply the explicit
profile, and only then merge astrophysical components (Step 2) and clean up
dispersion. Multi‑PTA alignment is skipped when there is only one PTA;
otherwise MetaPulsar discovers per‑PTA state and applies only the rules needed
for the observed timing engines.

**Prerequisites.** The `EPHEM` and clock realization must be resolvable: from
`AlignmentPolicy` if supplied, otherwise from the reference PTA's `EPHEM` and
`CLOCK`/`CLK`. The step raises if either cannot be resolved. A bare `TT(BIPM)`
is ambiguous across environments and raises unless `AlignmentPolicy.bipm_version`
pins a year; a dated policy clock that disagrees with `bipm_version` also raises.

**Astrometry guard.** If a parfile contains both equatorial and ecliptic astrometry parameters, `detect_astrometry_style` raises a `ValueError` (surfaced as `RuntimeError` by `_make_parameters_shared`).

For a **single‑PTA pulsar**, the par/tim rewrite path is still used when needed for DM model cleanup and pulse‑number support. The multi‑PTA convention rules below are skipped because there is no second PTA to align to: `ECL`, `T2CMETHOD`, `EPHEM`, and `CLOCK`/`CLK` are not changed by this step.

#### Ecliptic astrometry is transformed, never relabelled

When ecliptic frames have to change, MetaPulsar performs a **numeric coordinate
transformation** through PINT (`as_ICRS().as_ECL(ecl=...)`), copying back only
the transformed astrometry fields (`ELONG`, `ELAT`, `PMELONG`, `PMELAT`,
`POSEPOCH`, `ECL`, and `KOM` for DDK). The physical sky direction and its
proper‑motion propagation are preserved; only the frame the numbers are
expressed in changes. Rewriting the `ECL` label alone would move the implied
direction by ~10⁻⁴ arcsec between `IERS2003` and `IERS2010`, which is why it is
not done.

The target obliquity is chosen per stack:

- **PINT + Tempo2:** always transform every ecliptic PTA to `IERS2003`;
- **PINT‑only:** transform only when the `ECL` values differ, to the reference
  `ECL` (or `IERS2010` when the reference omits it);
- **Tempo2‑only:** transform only when the `ECL` values differ, to `IERS2003`.

Equatorial cross‑engine stacks have no active obliquity convention to align:
their `ECL` line is removed and a warning about the known few‑ns equatorial
parity floor is emitted.

#### Reference conventions

Every PTA adopts the resolved `EPHEM` and clock realization. The clock is
written with the receiving engine's native keyword: `CLOCK` for PINT and `CLK`
for Tempo2/libstempo.

#### The explicit mixed‑engine profile

When **both** PINT and Tempo2 are in the stack, MetaPulsar writes the validated
common profile explicitly rather than relying on either engine's defaults:

| Surface | Written value |
|---|---|
| `UNITS` | `TDB` |
| `T2CMETHOD` | `IAU2000B` |
| `TIMEEPH` | `FB90` |
| `DILATEFREQ` | `N` |
| `CORRECT_TROPOSPHERE` | `N` |
| `PLANET_SHAPIRO` | `N` |
| whole‑Solar‑System Shapiro | enabled: `NO_SS_SHAPIRO` removed |
| solar wind | `SWM 0` everywhere, `IPM 1` on Tempo2 pars, explicit constant `NE_SW` |
| ecliptic `ECL` | `IERS2003` (numerically transformed, above) |

`NO_SS_SHAPIRO` is handled independently of `PLANET_SHAPIRO`: the former
disables the whole Solar‑System Shapiro component including the Sun, the latter
only the planetary terms.

This profile is **not** applied to single‑engine stacks. In particular a
PINT‑only multi‑PTA combination keeps whatever troposphere and planetary‑Shapiro
settings its inputs declare; only `UNITS`, `EPHEM`, the clock, `ECL`, and
`NE_SW` are aligned there, exactly as before.

| Scenario | Full §4.1 forced profile | `T2CMETHOD TEMPO` | Ecliptic `ECL` |
|----------|--------------------------|-------------------|----------------|
| Single PTA | No | Keep | Unchanged |
| Multi PTA, tempo2-only | No | Align across PTAs; keep `TEMPO` if shared in reference | Transform to `IERS2003` if heterogeneous |
| Multi PTA, PINT-only | No | N/A | Transform to reference or `IERS2010` if heterogeneous |
| Multi PTA, PINT + tempo2 | Yes | Force `IAU2000B` | Transform to `IERS2003` |
| Equatorial + cross-engine | Yes | Force `IAU2000B` | Strip `ECL` + warning |

#### Solar-wind convention (`NE_SW`)

Tempo2 applies an implicit `NE_SW = 4` cm⁻³ when the line is absent; PINT applies none.
During consistent parfile rewriting MetaPulsar writes one explicit frozen `NE_SW`
line on every PTA, resolved in this order:

1. `AlignmentPolicy.ne_sw`, when supplied;
2. the reference par's explicit `NE_SW`, `NE1AU`, or `SOLARN0`;
3. `4` cm⁻³ when any PTA uses tempo2.

For PINT-only stacks with no reference `NE_SW`, no line is added (PINT effective zero).
Alias spellings are replaced by the canonical `NE_SW`, and conflicting explicit
values on non-reference PTAs are overwritten with a warning.

#### Orthometric Shapiro (`H3`/`STIG`/`H4`) in mixed stacks

`H4` and `STIG`/`STIGMA`/`VARSIGMA` are mutually exclusive
parameterizations. Tempo2 logs that it is ignoring the stigma parameter and
continues with the approximate `H3`+`H4` model; PINT rejects the combination.
Every `shared` invocation therefore raises a focused error before model
construction, independently of the unsupported-term strip policy. MetaPulsar
does not silently choose one physical model. `per_pta` retains native engine
behavior.

Two mechanisms are needed for ELL1H and `T2` binaries to agree across engines.
Both apply only to mixed PINT+Tempo2 `shared` stacks; PINT‑only, `per_pta`,
and single‑PTA paths keep PINT's published defaults.

* **`H3`+`STIG`.** PINT's default evaluates Freire & Wex (2010) Eq. 29, Tempo2's
  ELL1H mode 1 evaluates Eq. 28. MetaPulsar passes `ell1h_shapiro="absorbed"`
  to every PINT model build on the mixed‑engine path — factory materialization
  and the temporary models `ParameterManager` builds during alignment — so both
  engines evaluate the same delay for the same printed `(A1, EPS1, H3, STIG)`.
  This selects an evaluator; it does not rewrite the par or convert gauges.
* **`H3`+`H4`.** PINT floors `NHARMS` at 7 whenever `H4` is set, while Tempo2
  falls back to `nharm=4` when the keyword is absent. The two spellings are not
  aliases of each other, so the shared par carries **both** `NHARM` (Tempo2) and
  `NHARMS` (PINT) at the same value: the largest count any input declared,
  floored at 7. The count is written after the binary component merge, which is
  what decides each par's final orthometric model, and is removed entirely when
  that model turns out to be `H3`+`STIG`.

`tests/test_cross_engine_parity.py` measures both mechanisms by idealizing TOAs
with PINT and re‑timing them with Tempo2.

#### Gated ELL1-family → DD/DDH conversion (mixed engines)

When `"binary"` is shared on a mixed PINT+Tempo2 stack, MetaPulsar applies a
**scale gate** (not a residual floor)
`scale_s = a1_max·e_max² + ½·n_b·a1_max²·e_max` (§6.3). Above the default 1 ns
threshold it **losslessly rewrites** supported plain ELL1/T2-EPS binaries to
`BINARY DD` (via `pint.binaryconvert` plus the §7.6 δT0 and TASC→T0
re-referencing corrections) and supported ELL1H blocks with a complete
orthometric pair to `BINARY DDH` (gauge-correct first-party map). A mandatory
mean-removed delay-fidelity check (derived tolerances) guards the rewrite.
Unsupported families (ELL1k, FB series, H3-only under the default
`h3_only="error"`, H4 series tails above threshold, …) raise
`BinaryConversionError` with a remediation list, or proceed unconverted under
`unsupported_binary="keep"`. H3-only sources may convert under
`h3_only="sample_stigma"` with `stigma_central`/`stigma_provenance`; the
emitted STIGMA is a prior center (`required_sampling=("STIGMA",)` on the
conversion report / `MetaPulsar.conversion_metadata()`), never a measurement.
nltiming may then recondition fully delta-flat Kepler triples via
`MarginalBasisFrame` and enforce the STIGMA sampling contract. See
`feature_ell1h_truncation_fixw_nltiming.md`.

For cross-engine parity motivation and validation context, see Luo et al. 2021,
*ApJ* 911, 45, [doi:10.3847/1538-4357/abe62f](https://doi.org/10.3847/1538-4357/abe62f).
This method description intentionally does not reproduce the full paper
regression narrative.

### Step 4: Materialize PTA timing records and validate identity

For each PTA MetaPulsar materializes a validated `_PtaTimingData` record:

* PINT path: `materialize_pint(TimingModel, TOAs)` (requires `planets=True` TOAs).
* Tempo2 path: `materialize_tempo2(tempopulsar)` (zeros `DMASSPLANET*` and forms BATs).

PINT models are loaded with `allow_T2=True` (so a `T2` binary is resolved to the
family PINT can evaluate) and, on mixed PINT+Tempo2 `shared` stacks only,
with `ell1h_shapiro="absorbed"` (Step 3).

MetaPulsar validates that all PTAs refer to the **same sky position** (pairwise separation ≤ 10″ at J2000 ICRS) and compatible catalog letter-suffix usage. The public **`MetaPulsar.name`** is the B-preferred **catalog** string from parfile `PSRJ`/`PSR`/`PSRB` fields (not a truncated coordinate designator).

### Step 5: Parameter mapping (merged vs PTA‑specific)

MetaPulsar now defines the **meta‑parameters** that the combined design matrix will use.

* For any parameter that belongs to a consistent component and exists across PTAs, MetaPulsar exposes **one merged meta‑parameter** (e.g., `RAJ`, `F0`, `PB`, `DM1`), mapped to the corresponding parameter name in each PTA object.
* By default, `DM` is excluded from shared merging (`exclude_from_shared=("DM",)`) and is exposed as PTA-specific meta-parameters (`DM_<pta>`), because the par-file `DM` is each PTA's reference value for TOA production.
* All **detector‑specific** timing‑model parameters (e.g., `JUMP`, `FD*`, per‑backend offsets) are exposed as **PTA‑specific** meta‑parameters by suffixing with the PTA label (e.g., `JUMP_XXXX_epta`, `Offset_nanograv`).
* If a per‑dataset **phase offset** is implicit in a given timing package, MetaPulsar explicitly includes an **`Offset_<pta>`** meta‑parameter to reflect the standard constant phase term that is effectively fit in pulsar timing (this is not a noise parameter).
* NOTE: The `Offset_XXXX` parameter is effectively just a `JUMP_XXXX` parameter for that specific PTA. But the name `Offset` makes it clear it is _not_ an added parameter, but merely the mapped phase offset from a specific PTA.

This mapping is produced by `ParameterManager.build_parameter_mappings()` and recorded as `fitparameters` (free) and `setparameters` (present) in the `MetaPulsar` object. It is **deterministic** given the input `.par` files and the selected consistent components.

### Step 6: Concatenate TOAs and flags (no data edits)

MetaPulsar concatenates the per‑PTA arrays into combined vectors:

* TOAs, residuals, TOA errors, SSB frequencies, telescope codes, etc.
* Flags include `pta`, `pta_dataset`, and `timing_package` tags for each TOA, and may also include `mjd_jump_pta` when the release par has `JUMP MJD` windows. These are read back from the `.tim` files, not synthesized here: MetaPulsar always hands its timing engines a canonical `.tim` in which those flags are stamped (see “Canonical `.tim` artifacts” below).

Concatenation itself does not alter TOA values; any `TIME` baking already happened when each leg’s canonical `.tim` was written.

### Canonical `.tim` artifacts

Every PTA leg is loaded from a rewritten standalone Tempo2 `FORMAT 1` `.tim` rather than the release file, so the PTA identity of a TOA travels with the data instead of living only in memory. The rewrite is dual-engine-reloadable:

* `INCLUDE` directives are inlined in place. Cumulative `TIME` offsets are baked into TOA MJDs with exact decimal arithmetic — MetaPulsar applies Tempo2’s mathematical rule `sat += TIME / 86400`, evaluated exactly and rounded once to the emitted MJD token. It does **not** reproduce Tempo2’s intermediate `double`/`longdouble` rounding. `TIME` and `MODE` lines are never emitted. TOA names are rewritten to safe `toaNNNNN` tokens so PINT cannot classify lines as Princeton/Parkes/ITOA. Other directives (`T2EFAC`, …) and comments are preserved; existing flag pairs (including `-to`) are kept.
* Effective `MODE` is discovered from the **release** tim tree (including legacy `FORMAT 0` / untagged releases) before pulse-number rewriting, and transferred onto the engine-facing `.par` as a single appended `MODE` line that supersedes any `MODE`/`WEIGHT`. An absent tim `MODE` means no override: the engine par’s own mode is left unchanged.
* Source MJD tokens and baked epochs are range-checked (`[0, 1e6]`); `TIME` offsets are bounded at `|Δ| ≤ 1e9` s with an exponent floor that blocks memory-amplifying literals.
* `-pta`, `-pta_dataset`, and `-timing_package` are appended to every TOA. `-pta` and `-pta_dataset` carry the PTA key MetaPulsar knows the dataset by; `-timing_package` records which engine loaded it.
* When the release `.par` contains `JUMP MJD t1 t2 …` lines, selected TOAs also receive `-mjd_jump_pta {pta}_{k}` (one-based index of that PTA’s `JUMP MJD` lines). Selection follows the parsing leg on the **post-bake** FORMAT 1 MJD token: tempo2 half-open `[t1, t2)`, PINT closed `[t1, t2]`. Overlapping windows that select the same TOA are refused.
* A release that already uses one of those MetaPulsar-owned flag names (PPTA DR1/DR2 ships `-pta`) has its own value preserved as `-<name>_orig`.
* `-pn` pulse numbers are present when `use_pulse_numbers` asks for them; the rewrite itself is unconditional.
* `convert_jump_mjd=False` (default) leaves engine-par `JUMP MJD` lines intact while still stamping the tim flags. `convert_jump_mjd=True` rewrites each engine-par line to `JUMP -mjd_jump_pta {pta}_{k} …` with the same values; release `.par` files on disk are never mutated.

`TIME` ownership matches the parsing leg: tempo2 is file-local per `INCLUDE`; PINT shares one accumulator across includes. Flattening is refused (with an error, never a silent shift) if tempo2 stateful directives are live at an `INCLUDE` boundary. Passing `timfile_output_dir` exports the exact files the engines consumed, so a combination can be reproduced or handed to other tooling directly.

### Step 7: Construct the combined design matrix

Let ( **P** ) be the set of meta‑parameters (columns to be fit). For each meta‑parameter ( q ∈ **P** ):

1. For each PTA, locate the corresponding underlying parameter (using the mapping).
2. Copy the associated **design‑matrix column** from that PTA’s `_PtaTimingData` record into the appropriate rows of the combined design matrix.
3. Apply **unit matching** where PINT and Tempo2 differ (e.g., RA, DEC, ecliptic longitude/latitude in hourangle/deg vs radians); these conversions are explicit and limited to astrometric columns.

After assembly MetaPulsar performs a **non‑identifiability check**: any column whose absolute sum is numerically zero (no support in any rows) is dropped from the fit list. This avoids singular normal matrices and is reported via warnings (note: if a parameter has zero support, this indicates an error in the underlying data release. This happens in, e.g., IPTA-DR2 datasets).

### Step 8: Planetary and positional metadata

MetaPulsar adopts position vectors, SSB ephemerides, and related arrays directly from the materialized PTA timing records and copies them into the combined structure row‑wise. This is bookkeeping only and does not alter any physical quantity.

### Statistical equivalence to a manual combination (sketch)

For each PTA (p), linearize timing residuals about the (possibly different) nominal parameter vectors ( β_{0,p} ):

**r**_p(β) ≈ **n**_p - **M**_p ε,  where ε ≡ β - β₀,  and **n**_p ~ N(0, **C**_p).

Concatenate over PTAs: ( **r** = **n** - **M** ε ), ( **C** = diag(**C**_p) ), and let ( **M** ) contain **merged** columns for consistent parameters and **block‑diagonal** columns for PTA‑specific parameters (exactly what the construction above produces).

The Gaussian likelihood marginalized over ( ε ) with flat priors depends on the **projector**

**P** = **I** - **M** ( **M**^T **C**^{-1} **M** )^{-1} **M**^T **C**^{-1}.

Any re‑timing that yields the **same column space** of ( **M** ) produces the **same marginalized likelihood** (and therefore the same posteriors for noise and GW parameters and the same frequentist quadratic statistics). Our method to make model components consistent ensures that the astrophysical columns are **shared** across PTAs and detector‑specific columns remain **PTA‑local**, which is exactly the structure a manual combined global `.par` would produce. Differences in the **nominal** parameter values ( β_{0,p} ) do not affect the marginalized likelihood (beyond negligible second‑order effects), because only the **derivatives** (the columns of ( **M** )) enter ( **P** ). Hence, under the standard linear‑response assumptions used throughout PTA analyses, this direct combination is **not less accurate** than a manual global re‑fit.

### Practical options and safeguards

* **Choice of consistent components.** The default choice `{astrometry, spindown, binary, dispersion}` fits most pulsars. For problematic sources one can drop a component from the consistent set; all parameters of that component then remain PTA‑specific.
* **DM modeling.** Removing DMX in favor of {DM, DMEPOCH, DM1, DM2} makes the deterministic DM part uniform. Stochastic DM variations are handled entirely in the noise model (e.g., a DM GP) during inference.
* **Challenging timing models.** If a pulsar resides in a regime where ( **M** ) varies rapidly with ( β₀ ) (high‑order binary models, poorly constrained orbital evolution), manual inspection is recommended. For Pulsar Timing Array purposes this is typically not an important regime to take into account. Note that the factory allows a **composite** strategy (no merging model components) for such cases (aka: FrankenStat) that is slightly more forgiving in this regard.
* **Name handling.** Pulsar identity uses **10″ J2000 position matching** plus parfile catalog names. B‑ vs J‑name strings that refer to the same source are aliases of one group; truncated `JHHMM±DDMM` strings are not used for identity or lookup.
* **Determinism and provenance.** Given the set of `.par`/`.tim` inputs, the chosen reference PTA, and the list of consistent components, the output is deterministic. The code can optionally write the **shared** `.par` files it constructs (`parfile_output_dir`) and the canonical `.tim` files it feeds the engines (`timfile_output_dir`) for full auditability.
* **Single‑PTA behavior.** A single‑PTA pulsar still gets its `.par`/`.tim` artifacts rewritten (DM model cleanup, pulse numbers, canonical PTA flags). What is skipped is only the multi‑PTA consistency/alignment step, because there is no cross‑PTA reference relationship to enforce.

### Implementation details (reproducibility pointers)

* **Factory and orchestration.** `MetaPulsarFactory.create_metapulsar(...)` loads `.par` content, validates the single‑pulsar grouping by coordinates, selects/accepts the reference PTA, and (for the **shared** strategy) calls `ParameterManager.make_parfiles_shared()` to emit shared `.par` files (optionally to disk). That method runs: parse → unit convert → `_make_parameters_shared` (component merge, then shared convention rules) → write.
* **Parameter discovery and aliasing.** `ParameterManager` uses PINT’s model metadata plus a lightweight alias resolver to collect the parameter sets by *component type* and to resolve name differences between PINT and Tempo2.
* **Design‑matrix assembly.** `MetaPulsar` implements the Enterprise/Discovery pulsar surface by duck typing. It builds `fitparameters`/`setparameters` from the mapping, concatenates the per‑PTA arrays, and assembles the combined `designmatrix` column‑by‑column—applying explicit unit corrections for astrometric columns where PINT and Tempo2 differ. A zero‑information column cull prevents singularities.
* **Flags and metadata.** `metapulsar.tim_canonical.write_canonical_tim(...)` stamps `pta`, `pta_dataset`, and `timing_package` into the `.tim` each engine loads, and `-mjd_jump_pta` on TOAs selected by release `JUMP MJD` windows, so the combined flags are read back from the data; `MetaPulsar._combine_flags()` fills the PTA metadata flags in only for legs that lack them (a MetaPulsar built directly from pulsar objects rather than from files). Planetary and positional arrays are copied row‑wise from the materialized PTA timing records.

### What this method does **not** do

* It **does not** invent TOA uncertainties or backend flags. The canonical `.tim` rewrite bakes release `TIME` into MJDs, drops `TIME`/`MODE`, renames TOA filename tokens to `toaNNNNN`, adds `-pta`/`-pta_dataset`/`-timing_package` (and `-pn` when requested), may add `-mjd_jump_pta` for `JUMP MJD` windows, and renames a colliding MetaPulsar-owned release flag to `-<name>_orig`; every other flag is copied.
* It **does not** decide noise hyperparameters; EFAC/EQUAD/ECORR and the red/DM noise models are inferred in the usual way in Enterprise/Discovery after the metapulsar is constructed.
* It **does not** convert `DMMODEL` grids to DMX or to a Taylor expansion.
  Binary-family conversion is gated and limited to the supported ELL1/ELL1H
  sets above; unsupported families are refused (or kept under policy), never
  silently degraded. A `T2` model without EPS remains resolved by PINT's
  `allow_T2` guessing only.
* Outside the gated conversion path it **does not** rewrite `(A1, EPS1, …)`
  between the Freire & Wex Eq. 28 and Eq. 29 gauges — selecting which
  expression PINT evaluates (`ell1h_shapiro`) is separate from conversion.
* It **does not** refit after stripping unsupported terms, nor reconcile the
  external realizations (clock files, IERS tables, downloaded ephemerides) that
  each package resolves independently from the same `EPHEM`/`CLK` strings.
* It **does not** ingest already-combined IPTA data products; MetaPulsar
  combines per-PTA releases.

### Minimal algorithm (for reference)

1. **Align the orbital chart** and **parse** all PTAs with PINT; for multi‑PTA combinations **expand `TEMPO1`** and **strip (or reject) unsupported deterministic families**.
2. **Normalize units**: every par is written with an explicit `UNITS TDB`; TCB inputs are converted by their own timing package.
3. **Transform ecliptic astrometry** numerically onto one obliquity convention (never a relabel).
4. **Apply shared convention rules**: for single‑PTA pulsars, skip multi‑PTA alignment; for multi‑PTA pulsars, align `NE_SW`, the resolved `EPHEM` and dated clock, then the engine‑gated rules — the full explicit profile when both PINT and tempo2 are present, otherwise only heterogeneous single‑engine conventions.
5. **Make shared** selected components by copying reference PTA values; **leave detector‑specific timing‑model parameters as PTA‑local**; for dispersion: remove DMX, preserve each PTA's local DM value and mark it free, set DMEPOCH (frozen), add DM1/DM2 (free, 0). Then normalize the ELL1H harmonic count for the merged binary model, and — on mixed-engine stacks — apply the gated ELL1→DD / ELL1H→DDH conversion when the scale gate fires.
6. **Materialize** PTA timing records (PINT or Tempo2 path). Validate same pulsar by coordinates.
7. **Map parameters** into merged and PTA‑specific meta‑parameters (deterministic mapping).
8. **Concatenate** per‑PTA arrays (TOAs, flags, etc.) without modification.
9. **Assemble** the combined design matrix column‑by‑column using the mapping, with explicit unit conversions; drop zero‑information columns.
10. **Expose** a `MetaPulsar` object fully compatible with Enterprise/Discovery.

---

#### Notes

* **Unit conversions:** `ParameterManager._convert_units_if_needed`, `_convert_pint_to_tdb`, `_convert_tempo2_to_tdb`.
* **Component merge:** `_make_component_parameters_shared`, `_handle_dm_special_cases`.
* **Shared convention rules:** `ParameterManager._apply_shared_convention_rules` (called at the end of `_make_parameters_shared`). Astrometry style detection: `detect_astrometry_style` in `pint_helpers.py`.
* **NE_SW convention alignment:** `ParameterManager._align_ne_sw_convention`
* **Component discovery/aliasing:** `get_parameters_by_type_from_models`, `resolve_parameter_alias`, `check_component_available_in_model`.
* **Detector‑specific timing‑model parameters remain local:** anything not in the consistent component set becomes PTA‑suffixed in `_add_pta_specific_parameter`.
* **Phase offset exposure:** if `PHOFF` is absent MetaPulsar defines a meta‑parameter mapped to the canonical “Offset” column so that per‑dataset constant phase terms are explicit.
* **Combined design matrix:** `MetaPulsar._build_design_matrix` (with unit corrections in `_convert_design_matrix_units`) and the zero‑information cull in `_remove_nonidentifiable_parameters`.
* **Identity validation and naming:** `discover_pulsars_by_position`, `positions_within_tolerance`, and catalog suffix checks in `_validate_pulsar_consistency`; `MetaPulsar.name` from `preferred_group_name`.
* **No TOA edits:** `MetaPulsar._combine_timing_data` concatenates; there are no writes or transforms of TOAs.
