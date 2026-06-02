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

### Step 1: Unit normalization

MetaPulsar first ensures that all `.par` files are in the same time unit convention:

1. Read all `.par` files. If mixed `UNITS` are detected (TCB vs TDB), convert to **TDB**.

   * For PINT models MetaPulsar re‑emits the model in TDB;
   * For Tempo2 models MetaPulsar calls the `transform tdb` plugin (both paths are implemented).

If all inputs already share the same `UNITS`, no conversion is performed.

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
* ensures that **DM** is present and marked **free**,
* defines a fixed **DMEPOCH** (copied from the reference; frozen), and
* optionally inserts **DM1** and **DM2** (default: present and **free**, initialized at 0).

This choice keeps the deterministic dispersion expansion identical across PTAs while leaving the **stochastic DM process** (DM GP) to the noise model, as is standard practice.

> **Detector‑specific timing‑model parameters.**
> Terms that describe *deterministic* instrument/telescope‑dependent delays—e.g., **JUMPs**, **FD** coefficients, and overall **phase offsets**—are part of the timing model and are **not** made consistent. They remain **PTA/backend specific**. By contrast, **EFAC/EQUAD/ECORR** are **noise** hyperparameters (not timing‑model parameters) and are *never* touched here.

Terminology: in MetaPulsar we use the word consistent to describe model components or parameters that are common between data of different PTAs, and that we 'lock' together (i.e. they become the same parameters or model component).

### Step 3: Make timing‑model conventions consistent

Still inside `_make_parameters_consistent`, **after** the component merge (Step 2) and **before** the consistent `.par` strings are written, MetaPulsar calls `_apply_convention_harmonization`. The policy uses the pattern: discover per‑PTA state, compare PTAs/engines, and only act when heterogeneity matters.

**Prerequisites.** The reference PTA must contain `EPHEM` and either `CLOCK` or `CLK`; harmonization raises if either is missing.

**Astrometry guard.** If a parfile contains both equatorial and ecliptic astrometry parameters, `detect_astrometry_style` raises a `ValueError` (surfaced as `RuntimeError` by `_make_parameters_consistent`).

Layer A is always applied:

1. align `EPHEM` to the reference PTA,
2. align `CLOCK`/`CLK` to the reference PTA.

Layer B is applied only when both PINT and tempo2 PTAs are present
(cross-engine stack):

1. ecliptic astrometry: set `ECL IERS2003`,
2. remove active `T2CMETHOD TEMPO` (only when the first token is `TEMPO`),
3. equatorial astrometry: remove active `ECL` and emit a warning about the
   known residual few-ns parity floor for equatorial inputs.

Layer C is applied only for single-engine, multi-PTA stacks:

- **PINT-only:** if ecliptic `ECL` values differ (including missing vs present),
  align to the reference `ECL`; if the reference omits `ECL`, default to
  `IERS2010`.
- **Tempo2-only:** if ecliptic `ECL` values differ, align to `IERS2003`; if
  `T2CMETHOD` values differ, align to the reference PTA's `T2CMETHOD` (which
  preserves `TEMPO` when already shared in the reference).

No-op cases:

- single PTA: Layers B and C are skipped (Layer A still aligns `EPHEM` and
  `CLOCK`/`CLK`, which is usually already true),
- multi-PTA single-engine stacks that are already homogeneous for the relevant
  conventions.

`UNITS` normalization remains handled by `_convert_units_if_needed`; convention
harmonization does not force `UNITS`.

For cross-engine parity motivation and validation context, see Luo et al. 2021,
*ApJ* 911, 45, [doi:10.3847/1538-4357/abe62f](https://doi.org/10.3847/1538-4357/abe62f).
This method description intentionally does not reproduce the full paper
regression narrative.

| Scenario | `T2CMETHOD TEMPO` | Ecliptic `ECL` |
|----------|-------------------|----------------|
| Single PTA, tempo2-only | Keep | Unchanged |
| Multi PTA, tempo2-only | Align across PTAs; keep `TEMPO` if shared in reference | Align to `IERS2003` if heterogeneous |
| Multi PTA, PINT-only | N/A | Align to reference or `IERS2010` if heterogeneous |
| Multi PTA, PINT + tempo2 | Remove/neutralize | Force `IERS2003` |
| Equatorial + cross-engine | Remove + warning | Strip `ECL` |

### Step 4: Build Enterprise pulsars and validate identity

For each PTA MetaPulsar builds an Enterprise pulsar object:

* PINT path: `ep.PintPulsar(TOAs, TimingModel, planets=True)`.
* Tempo2 path: `ep.Tempo2Pulsar(tempopulsar, planets=True)`.

MetaPulsar validates that all PTAs refer to the **same sky position** by converting names to a canonical **J‑name** derived from coordinates. “B‑vs‑J” selection is only for **display**—coordinate matching is authoritative.

### Step 5: Parameter mapping (merged vs PTA‑specific)

MetaPulsar now defines the **meta‑parameters** that the combined design matrix will use.

* For any parameter that belongs to a consistent component and exists across PTAs, MetaPulsar exposes **one merged meta‑parameter** (e.g., `RAJ`, `F0`, `PB`, `DM`), mapped to the corresponding parameter name in each PTA object.
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

### Implementation details (reproducibility pointers)

* **Factory and orchestration.** `MetaPulsarFactory.create_metapulsar(...)` loads `.par` content, validates the single‑pulsar grouping by coordinates, selects/accepts the reference PTA, and (for the **consistent** strategy) calls `ParameterManager.make_parfiles_consistent()` to emit consistent `.par` files (optionally to disk). That method runs: parse → unit convert → `_make_parameters_consistent` (component merge, then convention harmonization) → write.
* **Parameter discovery and aliasing.** `ParameterManager` uses PINT’s model metadata plus a lightweight alias resolver to collect the parameter sets by *component type* and to resolve name differences between PINT and Tempo2.
* **Design‑matrix assembly.** `MetaPulsar` (a subclass of `enterprise.pulsar.BasePulsar`) builds `fitparameters`/`setparameters` from the mapping, concatenates the per‑PTA arrays, and assembles the combined `designmatrix` column‑by‑column—applying explicit unit corrections for astrometric columns where PINT and Tempo2 differ. A zero‑information column cull prevents singularities.
* **Flags and metadata.** The combined flags include `pta`, `pta_dataset`, and `timing_package`. Planetary and positional arrays are copied row‑wise from the underlying Enterprise pulsars.

### What this method does **not** do

* It **does not** change TOAs, TOA uncertainties, or backend flags.
* It **does not** decide noise hyperparameters; EFAC/EQUAD/ECORR and the red/DM noise models are inferred in the usual way in Enterprise/Discovery after the metapulsar is constructed.

### Minimal algorithm (for reference)

1. **Parse & normalize units** for all PTAs (`UNITS → TDB` only when needed).
2. **Make consistent** selected components by copying reference PTA values; **leave detector‑specific timing‑model parameters as PTA‑local**; for dispersion: remove DMX, set DM (free), set DMEPOCH (frozen), add DM1/DM2 (free, 0).
3. **Harmonize conventions** with gated rules: always align `EPHEM`, `CLOCK/CLK`; apply cross-engine parity rules only when both PINT and tempo2 are present; otherwise apply single-engine multi-PTA alignment only when conventions are heterogeneous.
4. **Instantiate** Enterprise pulsars (PINT or Tempo2 path). Validate same pulsar by coordinates.
5. **Map parameters** into merged and PTA‑specific meta‑parameters (deterministic mapping).
6. **Concatenate** per‑PTA arrays (TOAs, flags, etc.) without modification.
7. **Assemble** the combined design matrix column‑by‑column using the mapping, with explicit unit conversions; drop zero‑information columns.
8. **Expose** a `MetaPulsar` object fully compatible with Enterprise/Discovery.

---

#### Notes

* **Unit conversions:** `ParameterManager._convert_units_if_needed`, `_convert_pint_to_tdb`, `_convert_tempo2_to_tdb`.
* **Component merge:** `_make_component_parameters_consistent`, `_handle_dm_special_cases`.
* **Convention harmonization (Layers A/B/C):** `ParameterManager._apply_convention_harmonization` (called at the end of `_make_parameters_consistent`); legacy alias `_apply_parity_protocol`. Astrometry style detection: `detect_astrometry_style` in `pint_helpers.py`. Legacy mirror: `MetaParfiles.apply_parity_protocol` in `metapulsar.legacy`.
* **Component discovery/aliasing:** `get_parameters_by_type_from_models`, `resolve_parameter_alias`, `check_component_available_in_model`.
* **Detector‑specific timing‑model parameters remain local:** anything not in the consistent component set becomes PTA‑suffixed in `_add_pta_specific_parameter`.
* **Phase offset exposure:** if `PHOFF` is absent MetaPulsar defines a meta‑parameter mapped to the canonical “Offset” column so that per‑dataset constant phase terms are explicit.
* **Combined design matrix:** `MetaPulsar._build_design_matrix` (with unit corrections in `_convert_design_matrix_units`) and the zero‑information cull in `_remove_nonidentifiable_parameters`.
* **Identity validation and naming:** `bj_name_from_pulsar` and coordinate‑based checks in `_validate_pulsar_consistency`.
* **No TOA edits:** `MetaPulsar._combine_timing_data` concatenates; there are no writes or transforms of TOAs.
