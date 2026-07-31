## Method: Direct combination (“MetaPulsar”)

### Problem statement and summary

Given multiple public PTA data sets for the **same** pulsar—each consisting of a **timing model** (a `.par` file) and **times of arrival** (a `.tim` file)—MetaPulsar constructs a single “metapulsar” that can be analyzed with standard PTA likelihoods without first manually re‑deriving a common timing solution. The procedure **does not modify the TOAs**; it only organizes the **deterministic timing model** across PTAs, and then builds the **combined design matrix** and metadata needed by Enterprise/Discovery.

After analytic marginalization over timing‑model parameters, the likelihood depends on the **column space** of the design matrix ( **M** ) rather than on the specific nominal parameter values ( β₀ ). Our procedure guarantees that the relevant column space is the same as in a traditional manual combination, so it is **statistically equivalent** to a full re‑timing while being vastly simpler and deterministic.

### Inputs and conventions

For each PTA (p) that observed a given pulsar, MetaPulsar requires:

* a `.par` file specifying the **deterministic timing model** (astrometry, spin, binary, dispersion, and instrument/telescope‑specific deterministic delays such as **JUMPs**, **FD** coefficients, and overall **phase offsets**), and
* a `.tim` file with TOAs and their formal uncertainties.

Let ( **d**_p ) denote the vector of residuals for PTA (p) when linearized about its nominal model ( β_{0,p} ), and let ( **M**_p ) be the corresponding design matrix (partial derivatives of the residuals with respect to timing‑model parameters). The full data vector is the concatenation ( **d** = ⨁_p **d**_p ). White‑ and red‑noise hyperparameters (EFAC/EQUAD/ECORR and RN/DM GP parameters) are **not** part of the deterministic timing model and are handled in the subsequent noise inference; MetaPulsar leaves them unchanged at this stage.

MetaPulsar uses **PINT** and **Tempo2/libstempo** to parse/realize timing models, and **Enterprise** classes to hold pulsar objects. The implementation provides two combination modes:

* **consistent** (default): make consistent astrophysical timing‑model components across PTAs while preserving detector‑specific timing‑model terms;
* **composite**: leave all `.par` files untouched and compose them as‑is (useful for diagnostics; everything remains PTA‑specific).

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
its native deterministic model is preserved exactly; none of Step 0 or Step 3
applies to it.

Users who need engine‑native terms preserved should use
`combination_strategy="composite"`, which performs no alignment at all.

### Step 1: Unit normalization

MetaPulsar then applies an engine-gated timescale policy:

1. A mixed PINT/Tempo2 multi-PTA stack is always materialized as explicit
   **TDB**, including a collection in which *all* inputs are TCB. PINT otherwise
   performs this conversion internally, hiding the representation it actually
   evaluates from the shared par files.

   * For PINT models MetaPulsar re‑emits the model in TDB;
   * For Tempo2 models MetaPulsar calls the `transform tdb` plugin (both paths are implemented).

2. A homogeneous PINT-only or Tempo2-only multi-PTA stack keeps its native
   timescale. A genuinely mixed TCB/TDB collection is normalized to TDB. A
   Tempo2 par with no `UNITS` line counts as TCB, which is what Tempo2 assumes.

3. A single PTA keeps its native timescale.

4. Converted text is re‑parsed and required to carry an explicit `UNITS TDB`.

The mixed-engine target is one explicit, auditable unit convention; the
single-engine paths do not rewrite a homogeneous convention unnecessarily.

**No TOA samples are modified** in this step or any subsequent step of this method.

### Step 2: Merge astrophysical timing‑model components

Within `ParameterManager._make_parameters_consistent`, MetaPulsar merges selected **astrophysical** components across PTAs by **copying parameter values from the reference PTA** into the other `.par` files *for those components only*. The reference PTA is chosen from the (optionally user‑ordered) PTA list; by default it is the first PTA, and a convenience function can select the PTA with the longest timespan. The set of components is configurable and defaults to

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
* preserves each PTA's local **DM** reference value and marks it **free** (by default via `exclude_from_consistent=("DM",)`),
* defines a fixed **DMEPOCH** (copied from the reference; frozen), and
* optionally inserts **DM1** and **DM2** (default: present and **free**, initialized at 0).

For dispersion, MetaPulsar removes DMX terms and keeps the Taylor DM evolution consistent, but `DM` itself is PTA-local by default. The `DM` value in a par file is the reference DM used when producing that PTA's TOAs, so the default `exclude_from_consistent=("DM",)` preserves those reference values while exposing `DM_<pta>` fit parameters in the combined design matrix.

This choice keeps the deterministic dispersion expansion identical across PTAs while leaving the **stochastic DM process** (DM GP) to the noise model, as is standard practice.

> **Detector‑specific timing‑model parameters.**
> Terms that describe *deterministic* instrument/telescope‑dependent delays—e.g., **JUMPs**, **FD** coefficients, and overall **phase offsets**—are part of the timing model and are **not** made consistent. They remain **PTA/backend specific**. By contrast, **EFAC/EQUAD/ECORR** are **noise** hyperparameters (not timing‑model parameters) and are *never* touched here.

Terminology: in MetaPulsar we use the word consistent to describe model components or parameters that are common between data of different PTAs, and that we 'lock' together (i.e. they become the same parameters or model component).

### Step 3: Make timing‑model conventions consistent

Still inside `_make_parameters_consistent`, MetaPulsar aligns the conventions
before the consistent `.par` strings are written. The ordering is fixed:
numerically transform ecliptic astrometry, align the solar wind, resolve and
apply the reference conventions, apply the explicit profile, and only then merge
astrophysical components (Step 2) and clean up dispersion.

**Prerequisites.** The `EPHEM` and clock realization must be resolvable: from
`AlignmentPolicy` if supplied, otherwise from the reference PTA's `EPHEM` and
`CLOCK`/`CLK`. The step raises if either cannot be resolved. A bare `TT(BIPM)`
is ambiguous across environments and raises unless `AlignmentPolicy.bipm_version`
pins a year; a dated policy clock that disagrees with `bipm_version` also raises.

**Astrometry guard.** If a parfile contains both equatorial and ecliptic astrometry parameters, `detect_astrometry_style` raises a `ValueError` (surfaced as `RuntimeError` by `_make_parameters_consistent`).

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

Every PTA adopts the resolved `EPHEM` and clock realization, written under
whichever alias that PTA already uses (`CLOCK` or `CLK`).

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
Every `consistent` invocation therefore raises a focused error before model
construction, independently of the unsupported-term strip policy. MetaPulsar
does not silently choose one physical model. `composite` retains native engine
behavior.

Two mechanisms are needed for ELL1H and `T2` binaries to agree across engines.
Both apply only to mixed PINT+Tempo2 `consistent` stacks; PINT‑only, composite,
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

For cross-engine parity motivation and validation context, see Luo et al. 2021,
*ApJ* 911, 45, [doi:10.3847/1538-4357/abe62f](https://doi.org/10.3847/1538-4357/abe62f).
This method description intentionally does not reproduce the full paper
regression narrative.

### Step 4: Build Enterprise pulsars and validate identity

For each PTA MetaPulsar builds an Enterprise pulsar object:

* PINT path: `ep.PintPulsar(TOAs, TimingModel, planets=True)`.
* Tempo2 path: `ep.Tempo2Pulsar(tempopulsar, planets=True)`.

PINT models are loaded with `allow_T2=True` (so a `T2` binary is resolved to the
family PINT can evaluate) and, on mixed PINT+Tempo2 `consistent` stacks only,
with `ell1h_shapiro="absorbed"` (Step 3).

MetaPulsar validates that all PTAs refer to the **same sky position** by converting names to a canonical **J‑name** derived from coordinates. “B‑vs‑J” selection is only for **display**—coordinate matching is authoritative.

### Step 5: Parameter mapping (merged vs PTA‑specific)

MetaPulsar now defines the **meta‑parameters** that the combined design matrix will use.

* For any parameter that belongs to a consistent component and exists across PTAs, MetaPulsar exposes **one merged meta‑parameter** (e.g., `RAJ`, `F0`, `PB`, `DM1`), mapped to the corresponding parameter name in each PTA object.
* By default, `DM` is excluded from consistent merging (`exclude_from_consistent=("DM",)`) and is exposed as PTA-specific meta-parameters (`DM_<pta>`), because the par-file `DM` is each PTA's reference value for TOA production.
* All **detector‑specific** timing‑model parameters (e.g., `JUMP`, `FD*`, per‑backend offsets) are exposed as **PTA‑specific** meta‑parameters by suffixing with the PTA label (e.g., `JUMP_XXXX_epta`, `Offset_nanograv`).
* If a per‑dataset **phase offset** is implicit in a given timing package, MetaPulsar explicitly includes an **`Offset_<pta>`** meta‑parameter to reflect the standard constant phase term that is effectively fit in pulsar timing (this is not a noise parameter).
* NOTE: The `Offset_XXXX` parameter is effectively just a `JUMP_XXXX` parameter for that specific PTA. But the name `Offset` makes it clear it is _not_ an added parameter, but merely the mapped phase offset from a specific PTA.

This mapping is produced by `ParameterManager.build_parameter_mappings()` and recorded as `fitparameters` (free) and `setparameters` (present) in the `MetaPulsar` object. It is **deterministic** given the input `.par` files and the selected consistent components.

### Step 6: Concatenate TOAs and flags (no data edits)

MetaPulsar concatenates the per‑PTA arrays into combined vectors:

* TOAs, residuals, TOA errors, SSB frequencies, telescope codes, etc.
* Flags include `pta`, `pta_dataset`, and `timing_package` tags for each TOA.

Again, **no TOA value is altered**; this is a pure concatenation with bookkeeping.

### Step 7: Construct the combined design matrix

Let ( **P** ) be the set of meta‑parameters (columns to be fit). For each meta‑parameter ( q ∈ **P** ):

1. For each PTA, locate the corresponding underlying parameter (using the mapping).
2. Copy the associated **design‑matrix column** from that PTA’s Enterprise object into the appropriate rows of the combined design matrix.
3. Apply **unit matching** where PINT and Tempo2 differ (e.g., RA, DEC, ecliptic longitude/latitude in hourangle/deg vs radians); these conversions are explicit and limited to astrometric columns.

After assembly MetaPulsar performs a **non‑identifiability check**: any column whose absolute sum is numerically zero (no support in any rows) is dropped from the fit list. This avoids singular normal matrices and is reported via warnings (note: if a parameter has zero support, this indicates an error in the underlying data release. This happens in, e.g., IPTA-DR2 datasets).

### Step 8: Planetary and positional metadata

MetaPulsar adopts position vectors, SSB ephemerides, and related arrays directly from the underlying Enterprise objects and copies them into the combined structure row‑wise. This is bookkeeping only and does not alter any physical quantity.

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
* **Name handling.** Pulsar identity is validated via **coordinates**. B‑ vs J‑name is a display convention only and does not enter any computation.
* **Determinism and provenance.** Given the set of `.par`/`.tim` inputs, the chosen reference PTA, and the list of consistent components, the output is deterministic. The code can optionally write the **consistent** `.par` files it constructs for full auditability.
* **Single‑PTA behavior.** A single‑PTA pulsar may still have its `.par`/`.tim` artifacts rewritten for DM model cleanup and pulse‑number handling. What is skipped is only the multi‑PTA consistency/alignment step, because there is no cross‑PTA reference relationship to enforce.

### Implementation details (reproducibility pointers)

* **Factory and orchestration.** `MetaPulsarFactory.create_metapulsar(...)` loads `.par` content, validates the single‑pulsar grouping by coordinates, selects/accepts the reference PTA, and (for the **consistent** strategy) calls `ParameterManager.make_parfiles_consistent()` to emit consistent `.par` files (optionally to disk). That method runs: parse → unit convert → `_make_parameters_consistent` (component merge, then consistent convention rules) → write.
* **Parameter discovery and aliasing.** `ParameterManager` uses PINT’s model metadata plus a lightweight alias resolver to collect the parameter sets by *component type* and to resolve name differences between PINT and Tempo2.
* **Design‑matrix assembly.** `MetaPulsar` (a subclass of `enterprise.pulsar.BasePulsar`) builds `fitparameters`/`setparameters` from the mapping, concatenates the per‑PTA arrays, and assembles the combined `designmatrix` column‑by‑column—applying explicit unit corrections for astrometric columns where PINT and Tempo2 differ. A zero‑information column cull prevents singularities.
* **Flags and metadata.** The combined flags include `pta`, `pta_dataset`, and `timing_package`. Planetary and positional arrays are copied row‑wise from the underlying Enterprise pulsars.

### What this method does **not** do

* It **does not** change TOAs, TOA uncertainties, or backend flags.
* It **does not** decide noise hyperparameters; EFAC/EQUAD/ECORR and the red/DM noise models are inferred in the usual way in Enterprise/Discovery after the metapulsar is constructed.
* It **does not** convert `DMMODEL` grids to DMX or to a Taylor expansion, and
  it does not convert between binary families: `BINARY` lines are left alone and
  a `T2` model is resolved by PINT's own `allow_T2` guessing.
* It **does not** rewrite `(A1, EPS1, …)` between the Freire & Wex Eq. 28 and
  Eq. 29 gauges. Selecting which expression PINT evaluates is in scope;
  converting the printed parameters is not.
* It **does not** refit after stripping unsupported terms, nor reconcile the
  external realizations (clock files, IERS tables, downloaded ephemerides) that
  each package resolves independently from the same `EPHEM`/`CLK` strings.
* It **does not** ingest already-combined IPTA data products; MetaPulsar
  combines per-PTA releases.

### Minimal algorithm (for reference)

1. **Parse** all PTAs with PINT; for multi‑PTA combinations **expand `TEMPO1`** and **strip (or reject) unsupported deterministic families**.
2. **Normalize units when required**: mixed PINT/tempo2 stacks use explicit TDB; genuinely mixed TCB/TDB multi-PTA stacks normalize to TDB; homogeneous single-engine and single-PTA inputs keep their native timescale.
3. **Transform ecliptic astrometry** numerically onto one obliquity convention (never a relabel).
4. **Apply consistent convention rules**: for single‑PTA pulsars, skip multi‑PTA alignment; for multi‑PTA pulsars, align `NE_SW`, the resolved `EPHEM` and dated clock, then the engine‑gated rules — the full explicit profile when both PINT and tempo2 are present, otherwise only heterogeneous single‑engine conventions.
5. **Make consistent** selected components by copying reference PTA values; **leave detector‑specific timing‑model parameters as PTA‑local**; for dispersion: remove DMX, preserve each PTA's local DM value and mark it free, set DMEPOCH (frozen), add DM1/DM2 (free, 0). Then normalize the ELL1H harmonic count for the merged binary model.
6. **Instantiate** Enterprise pulsars (PINT or Tempo2 path). Validate same pulsar by coordinates.
7. **Map parameters** into merged and PTA‑specific meta‑parameters (deterministic mapping).
8. **Concatenate** per‑PTA arrays (TOAs, flags, etc.) without modification.
9. **Assemble** the combined design matrix column‑by‑column using the mapping, with explicit unit conversions; drop zero‑information columns.
10. **Expose** a `MetaPulsar` object fully compatible with Enterprise/Discovery.

---

#### Notes

* **Unit conversions:** `ParameterManager._convert_units_if_needed`, `_convert_pint_to_tdb`, `_convert_tempo2_to_tdb`.
* **Component merge:** `_make_component_parameters_consistent`, `_handle_dm_special_cases`.
* **Consistent convention rules:** `ParameterManager._apply_consistent_convention_rules` (called at the end of `_make_parameters_consistent`). Astrometry style detection: `detect_astrometry_style` in `pint_helpers.py`.
* **NE_SW convention alignment:** `ParameterManager._align_ne_sw_convention`
* **Component discovery/aliasing:** `get_parameters_by_type_from_models`, `resolve_parameter_alias`, `check_component_available_in_model`.
* **Detector‑specific timing‑model parameters remain local:** anything not in the consistent component set becomes PTA‑suffixed in `_add_pta_specific_parameter`.
* **Phase offset exposure:** if `PHOFF` is absent MetaPulsar defines a meta‑parameter mapped to the canonical “Offset” column so that per‑dataset constant phase terms are explicit.
* **Combined design matrix:** `MetaPulsar._build_design_matrix` (with unit corrections in `_convert_design_matrix_units`) and the zero‑information cull in `_remove_nonidentifiable_parameters`.
* **Identity validation and naming:** `bj_name_from_pulsar` and coordinate‑based checks in `_validate_pulsar_consistency`.
* **No TOA edits:** `MetaPulsar._combine_timing_data` concatenates; there are no writes or transforms of TOAs.
