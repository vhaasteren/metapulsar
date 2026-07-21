# Feature proposal: Pylk universal pulsar-timing workbench

**Status:** Revised initial design for review; **incubation home created**  
**Scope:** Product and architecture proposal; not an implementation plan or API commitment  
**Incubation home (this repo):** `src/pylk/` → import root `pylk`  
**Proposed long-term:** extractable package, frontend dependencies optional  
**Companion:** the fast flexible-`Phi` GP fit is specified in
[`flexfit/feature_flexible_fit.md`](flexfit/feature_flexible_fit.md)
(`pylk.flexfit`); this document consumes it and does not re-specify its mathematics.

## Implementation status (package tree)

| Area | Status |
|------|--------|
| `src/pylk/` package root | **Created** (thin `__init__`; version via metapulsar install) |
| `pylk.flexfit` headless EB solver | **Migrated** from `paper/code/flexfit` (see companion doc) |
| Discovery + nltiming adapters | **Implemented** |
| Enterprise flexfit adapter | **Stub only** (planned; core is frontend-agnostic) |
| Workspace / Revision / Command / Capabilities | **Not built** |
| Providers (PINT, Tempo2, JUG, Vela, …) | **Not built** |
| Jupyter / PySide views | **Not built** |
| Async flexfit caching / provenance UI | **Not built** |

This tree exists so flexfit (and later workspace code) can live on git under the
correct long-term name without waiting for a full workbench.

## Summary

Pylk should be revived as a scientific workspace for interactive pulsar timing, not as another backend-specific plotting GUI.

Its primary product is a fast loop for data exploration, physical-model editing, waveform inspection, and point optimization. MetaPulsar combinations are first-class documents rather than a later extension of a single-dataset GUI. A scientist should be able to move quickly from a residual feature to a candidate model change, optimize it, inspect the reconstructed deterministic and stochastic waveforms, and either accept or reject the result.

The workspace should let a user load a pulsar or a multi-PTA MetaPulsar dataset, inspect and edit scientifically meaningful state, evaluate that state with appropriate timing engines, run the operations each engine genuinely supports, and continue the same analysis from Python. PINT, Tempo2/libstempo, JUG, Vela, Discovery, MetaPulsar, and `nltiming` should remain recognizable rather than being forced behind a fake universal pulsar object.

Multiple timing engines remain valuable, but engine comparison is a diagnostic capability rather than the default scientific workflow. Pylk should distinguish:

- physical-model comparisons, which are central and should be easy;
- comparisons within a compatibility family, which may be low-friction;
- cross-family comparisons, which must be explicit, convention-aware diagnostics.

The natural compatibility families are Tempo2-compatible evaluation (Tempo2/libstempo and JUG-Tempo2) and PINT-compatible evaluation (PINT, JUG-PINT, and Vela). PINT and Tempo2 defaults are not generally compatible, and they need not implement identical model subsets. Cross-family overlays should therefore never appear as an unqualified default.

The initial architecture should remain deliberately small:

```text
Core
----
Workspace
Revision
Command
Capabilities

Providers
---------
Discovery/MetaPulsar
PINT
Tempo2/libstempo
JUG
Vela

Views
-----
Notebook/Jupyter view
PySide desktop view

Scientific engines
------------------
nltiming (timing evaluation + z-space fit)
Flexible-Phi GP fit (headless pylk.flexfit)
JUG/JAX
```

The two scientific-engine responsibilities are deliberately split: `nltiming`
owns backend-neutral timing evaluation and the transformed-space (`z`) timing
optimizer, while the fast flexible-`Phi` GP waveform fit is a separate headless
module (`pylk.flexfit`, under `src/pylk/flexfit/`) that the future Pylk
workspace consumes. GP bases may come from **Discovery or Enterprise** via
adapters; the solver does not live in either frontend. See
[`flexfit/feature_flexible_fit.md`](flexfit/feature_flexible_fit.md) for that
split.

This proposal does **not** require a full event-sourcing system, a large family of provider interfaces, a production MCMC runner, or a decision today that JupyterLab or Qt must be the only frontend.

## Motivation

The pulsar-timing ecosystem now contains several capable but differently shaped systems:

- PINT offers rich Python timing-model and fitting objects, plus the mature but backend-specific `pintk` workflow.
- Tempo2/libstempo provides a compact and familiar mutable interface around Tempo2.
- JUG provides a JAX-based timing engine, correlated-noise functionality, and a modern PySide/PyQtGraph GUI.
- Vela exposes Julia timing and Bayesian functionality through PyVela.
- MetaPulsar represents composite multi-PTA datasets whose parameters and TOAs may originate in different timing systems.
- `nltiming` now supplies backend-neutral immutable timing evaluation across PINT, libstempo, JUG, and Vela, including composite MetaPulsar sessions.

Each project has valuable functionality, but each GUI naturally grows around its own internal objects. Reimplementing the same selection, plotting, parameter, fitting, and file-management workflows in every timing package creates duplicated work and inconsistent behavior.

The opportunity is to build one workspace that treats timing packages as scientific capabilities, while giving Python users access to their native objects and semantics. The workspace should be optimized for the work scientists actually do interactively: inspect data, edit a timing or physical model, obtain a useful point estimate quickly, examine waveform reconstructions, and iterate.

## Lessons from the earlier pylk work

The abandoned pylk repository contains several ideas worth retaining:

- its original embedded IPython console expressed the right interactive-computing ambition;
- later work emphasized direct access to backend-native objects;
- unit preservation and coordinate conversion received serious attention;
- pixel-space point selection was more robust than comparing unlike physical axes;
- pre-fit/post-fit state, revert behavior, mock data, and interaction tests provide useful requirements;
- the unfinished model-chain work recognized the need for history.

Its central architectural problem was that scientific state, Qt state, plot caches, selection, fitting, and undo became interdependent. A nominally GUI-neutral `PulsarModel` inherited from `QObject`, emitted Qt signals, owned scientific mutations, coordinated fit history, and fed plot-specific payloads. Backends were then asked to mimic PINT's mutable `model`, `toas`, `fitter`, and `residuals` bundle.

The new design should reuse behavioral knowledge and tests, not preserve that object model.

Pintk and the JUG GUI should similarly be treated as requirements and UX references:

- pintk is the best inventory of classic interactive timing operations;
- JUG demonstrates a more modern plot, asynchronous workers, and valuable noise controls;
- neither GUI implementation should become the universal application's state model.

## Product vision

Pylk is an interactive editor and explorer for pulsar-timing computations.

A user should be able to:

1. Open par/tim files, a native timing object, or a MetaPulsar host.
2. Inspect timing conventions, clocks, ephemerides, parameters, TOAs, flags, and capabilities without having to understand composite storage for ordinary operations.
3. Select or exclude TOAs without destroying their identity.
4. Change scientifically meaningful timing-model state and move safely through its history.
5. Compare candidate physical models and their deterministic or stochastic waveform reconstructions.
6. Obtain fast point estimates for timing-model parameters, including nonlinear parameters represented in the transformed space used by `nltiming`.
7. Evaluate the same state with a compatible timing path, with residual fidelity and convention provenance visible.
8. Run opt-in engine diagnostics or launch a longer inference workflow without pretending either is the primary interactive loop.
9. Save or export results with explicit provenance.
10. Access the same workspace and native timing objects from Python.

Pylk should feel like a document editor with scientific computation, not a collection of buttons around a mutable global pulsar.

## Goals

### Scientific goals

- Make candidate physical models and waveform contributions easy to inspect and compare.
- Preserve backend-specific semantics and capabilities.
- Make engine differences inspectable when deliberately requested rather than casually over-plotting them.
- Support single-dataset and multi-PTA MetaPulsar documents through the same editing loop.
- Keep TOA identities stable through selection, exclusion, sorting, and engine evaluation.
- Integrate linear timing fits, nonlinear evaluation, and external inference without conflating them.
- Support fast transformed-space timing optimization using `nltiming` parameter transforms and JUG autodiff.
- Support a flexible, quickly estimated GP visualization model for red, DM, chromatic, and related processes.
- Record sufficient provenance to understand and reproduce displayed results.

### Workflow goals

- Provide a responsive residual and waveform workspace with a parameter inspector.
- Make quick iteration the default while keeping evaluation fidelity visible.
- Support undo and redo for semantic scientific changes.
- Prevent long or stale computations from blocking or overwriting newer work.
- Make the active workspace available from notebooks or an IPython console.
- Permit native backend experimentation through an explicit synchronization boundary.

### Engineering goals

- Keep scientific truth independent of Qt, Jupyter, and plotting libraries.
- Use small capability contracts rather than a universal `BasePulsar` hierarchy.
- Build on `nltiming.TimingEvaluator` instead of duplicating backend evaluation.
- Prefer Discovery/MetaPulsar with JUG/JAX as the default scientific path where supported.
- Keep the first implementation small enough to validate with a vertical slice.
- Test scientific state and providers headlessly.

## Non-goals for the first release

- Reproducing every pintk keyboard command or JUG noise-control feature.
- Providing identical fitting behavior across engines.
- Making unrestricted PINT-versus-Tempo2 residual overlays a default workflow.
- Treating full MCMC or production posterior inference as an interactive GUI operation.
- Producing authoritative GP hyperparameter estimates from the fast visualization model.
- Automatically detecting arbitrary mutation of every native backend object.
- Making every visual interaction part of undo history.
- Building a general workflow engine or persistent event-sourcing framework.
- Supporting collaborative multi-user editing.
- Replacing the Python APIs of PINT, libstempo, JUG, Vela, MetaPulsar, Discovery, or `nltiming`.
- Guaranteeing lossless round trips for every historical par/tim dialect in the first milestone.

## Central design problem: computational semantics

The hardest questions are semantic, not graphical:

- What does “fit” or “optimize” mean for each engine and parameter representation?
- Which model is authoritative before and after an optimization?
- What constitutes a parameter change?
- At which residual-evaluation tier was a candidate state evaluated, and is that tier adequate for how far it moved from its anchor?
- Which differences are numerical error, convention differences, approximation error, or genuinely different physical models?
- How are shared MetaPulsar parameters represented across PTA inputs?
- How do least-squares results, nonlinear scans, fast waveform estimates, and posterior inference relate?
- Which operations mutate source state, and which only produce derived results?

Pylk must answer these questions visibly rather than normalize them away.

### Operation vocabulary

The UI and Python API should avoid a context-free `fit()` action. Interactive optimization is the default concept; long-running inference is a launch-and-inspect workflow. Named profiles may include:

- linear timing-model least squares;
- `nltiming` transformed-space iterative least squares with JUG autodiff;
- PINT WLS or GLS;
- JUG correlated-noise optimization;
- fast hierarchical GP waveform estimation;
- `nltiming` nonlinear scans;
- Vela, Discovery, or Enterprise posterior configuration and job launch.

Results must retain their provider, configuration, input revision, residual-evaluation tier, and assumptions. Production MCMC should not shape the core GUI architecture.

## Minimal core model

### Workspace

`Workspace` is the user-facing root object. It owns one or more open scientific documents, the active document, and access to available providers and background operations.

It should be usable without a GUI:

```python
workspace = Workspace.open(par_file, tim_file)
doc = workspace.active

doc.current
doc.evaluate(engine="pint")
doc.compare_engines(["pint", "jug"])
```

The initial `Workspace` should not become a service locator for every future concern. It needs only enough responsibility to coordinate open documents and expose their operations.

### Revision

A `Revision` is an immutable snapshot of semantic scientific state.

The first revision type should remain intentionally narrow. Its authoritative state should be the timing-model parameter state plus committed TOA activity masks and enough fixed context to interpret those parameters.

Candidate contents include:

- timing-model parameter values, retaining exact textual values where available;
- parameter fit/frozen/sample/marginalize state;
- stable logical TOA keys and active/excluded state;
- immutable timing context such as `PEPOCH`, `EPHEM`, clocks, units, and source provenance;
- references to accepted optimization or transformation results.

The first milestone should not allow arbitrary edits to contextual quantities such as `PEPOCH`, `EPHEM`, clocks, or units. Supporting those safely requires explicit transformations, not ordinary scalar parameter edits.

Not every interaction creates a revision.

Revision-worthy changes include:

- changing a parameter value;
- changing a parameter's scientific role;
- excluding or restoring TOAs;
- adding or removing jumps or phase wraps;
- accepting an optimization result;
- later, applying an explicit validated timing-context transformation.

Transient view state should remain outside revision history:

- zoom and pan;
- plot colors;
- open panels;
- hover state;
- current selection, unless that selection is committed as a scientific exclusion or operation input;
- temporary parameter previews while dragging a control.

This keeps history scientifically meaningful.

### Command

A `Command` describes one semantic change and returns a new revision or a derived result.

Initial commands may include:

- `SetParameter`;
- `SetParameterRole`;
- `ExcludeTOAs`;
- `RestoreTOAs`;
- `AddJump`;
- `ApplyPhaseWrap`;
- `AcceptOptimizationResult`.

Context-changing commands such as `ChangeTimingConvention` should follow only after their scientific transformations and validation rules are defined.

The first implementation does not need a general reducer framework, command log replay, or serialized event stream. A small command protocol is sufficient if it gives GUI and notebook operations one validated mutation path.

### Capabilities

Providers advertise what they can do for a particular document. Capabilities are evaluated at runtime because support may depend on the loaded model, TOA type, installed packages, or selected convention.

Examples include:

- residual evaluation;
- analytic design matrix;
- autodiff Jacobian;
- narrowband or wideband support;
- TOA editing and export;
- classical fitting profiles;
- noise realization;
- Bayesian inference;
- native-object access.

Capabilities should include structured explanations when unavailable. A disabled action should be able to say, for example, that Vela is not installed, the model component is unsupported, or a Tempo2-native cache has not been primed.

## Reactive computational model

Pylk should be reactive, but it should begin with a small explicit dependency graph rather than a general-purpose reactive framework.

The conceptual graph is:

```text
Sources and raw observations
            │
            ▼
Canonical TOA identity and active mask
            │
            ▼
Timing parameters and conventions
            │
            ├──────────────┐
            ▼              ▼
 Evaluation mode/tier  Flexible GP model
            │              │
            └──────┬───────┘
                   ▼
 Residuals / Jacobian / waveform estimates
                   │
          ┌────────┼─────────┐
          ▼        ▼         ▼
      Plots   Optimization   Export/inference
```

Changing an input should invalidate only its downstream products. Derived results should be keyed by at least:

- input revision;
- provider and engine;
- operation configuration;
- relevant software and data provenance.

The initial implementation can express this with typed computation keys and explicit dependencies. It should not begin by introducing an elaborate DAG scheduler. The important behavior is that invalidation is derived from scientific dependencies instead of scattered `dirty` flags in widgets.

### Preview versus commit

Interactive controls may need fast previews. A parameter slider, for example, should not create dozens of revisions while it moves.

A preview operates against the current revision and produces ephemeral downstream results. Releasing or explicitly applying the control commits one command and one new revision. Cancelling returns to the current revision.

## Stable TOA identity

TOAs should not be physically deleted from the authoritative workspace representation during interactive editing.

Each observation needs a stable identity that survives:

- sorting;
- plotting filters;
- engine-specific row order;
- PTA session assembly;
- exclusion and restoration;
- background calculations.

At the GUI level, a MetaPulsar combination is one logical pulsar and ordinary selection operations should not require users to reason about included tim files or composite-session internals. Internally, however, every plotted row still needs a stable key so selection, masking, sorting, provider row order, and asynchronous results refer to the same observation. The first implementation should test whether a logical document identity plus an immutable original row index is sufficient. Source/session provenance remains metadata used by providers, filters, export, and diagnostics rather than the primary user-facing identity.

Providers must return explicit mappings between logical TOA keys and engine-native row order. Diagnostics must fail clearly when alignment is incomplete or ambiguous.

Exclusion changes an active mask. Export policy decides whether excluded TOAs are omitted, commented, or represented using a backend-native mechanism.

## Interactive optimization and waveform modeling

### Fast timing-model optimization

The first-class timing operation should be iterative least squares for timing-model parameters. Linear least-squares steps have historically converged quickly in timing workflows; modern gradient-based alternatives should be benchmarked, but they should earn their place through measured interactive latency and robustness.

For nonlinear timing parameters, Pylk should optimize in the same prior-transformed parameter space that `nltiming` constructs before standardization for NUTS. This permits arbitrary supported priors while retaining a least-squares-style local optimization. JUG-powered autodiff should provide Jacobians in that space. A small number of relinearize-and-solve iterations may give a fast, physically useful point estimate without turning the GUI into an inference runner.

Every result must record its anchor revision and evaluation tier. Interactive response should use the fastest tier already validated for the current regime; physical correctness is enforced at acceptance by escalating fidelity when the proposed move requires it. A quick T0/T1-style step can be useful near a mature solution, but a proposed state should be re-evaluated at the appropriate nonlinear tier before acceptance when astrometric, binary, phase-connection, or other nonlinear changes are large.

### Fast flexible GP model for visualization

Interactive waveform exploration needs a process model that is fast to estimate and flexible enough to avoid forcing a production noise analysis into every plotting action. The proposed default visualization model should combine configurable basis blocks such as:

- timing-model directions in the `nltiming` transformed space;
- achromatic red-noise modes;
- DM-variation modes;
- chromatic-noise modes;
- additional user-selected deterministic or stochastic bases.

Within each block, low-level coefficients receive zero-mean truncated-Gaussian priors with a shared width. Sensible lower and upper width bounds should be derived from the dataset in terms of induced residual variance, analogous in spirit to nonlinear-timing cheat priors. Conditional coefficient means provide the plotted waveform; width updates are obtained cheaply from the corresponding bounded inverse-gamma or gamma conditional, following the principle used by `enterprise_extensions` Gibbs updates but omitting the random coefficient draw when the immediate goal is a stable visualization estimate.

This is a research design, not yet a settled statistical contract. A prototype must derive the exact conditional distributions, truncation convention, identifiability behavior, and units, then validate reconstructed means and bounds on simulated and real PTA data. Its output must be labelled as an interactive waveform estimate rather than a production posterior. Users may instead build a full Enterprise/Discovery model and optimize its log likelihood or log posterior, but Pylk should make the extra cost visible.

The full algorithm, ownership, and package placement for this model are specified in the companion [`flexfit/feature_flexible_fit.md`](flexfit/feature_flexible_fit.md). In summary: it is **not** part of `nltiming` (which owns only the timing block and the prior transform it consumes); it is the headless module `pylk.flexfit` that both the paper diagnostics and Pylk call. Adapters target Discovery (implemented) and Enterprise (planned). Pylk owns the interactive, caching, and provenance layer around it, not the solver.

### Residual-evaluation fidelity

Pylk should expose the tier vocabulary documented in `ref-packages/jug/TEMPO2_NATIVE_MODES.md`:

- T0: fixed first-order design-matrix response;
- T1: stripped or lightweight retracing;
- T2: full barycentric-geometry retracing with clocks and ephemeris vectors frozen;
- T3: T2 plus full binary/TRACK-2 staging;
- T4: complete recomputation.

Cold-start residual construction is full, but later parameter updates may use a lower tier. That distinction is more useful than a generic engine label because optimization error depends on which parameters moved, how far they moved from the anchor, and what the path recomputed. The UI should display the active tier, recommend escalation when appropriate, and support lower-tier-versus-higher-tier diagnostics for the same physical-model state.

## Providers

The initial provider abstraction should be coarse to avoid boilerplate.

A provider can expose a single descriptor plus whichever optional methods it supports:

```python
class TimingProvider(Protocol):
    name: str

    def capabilities(self, document) -> Capabilities: ...
    def evaluate(self, revision, request): ...
```

Optional capability-specific protocols may be introduced only when a second real implementation demonstrates the boundary, such as:

- `SupportsFit`;
- `SupportsExport`;
- `SupportsNativeAccess`;
- `SupportsInference`.

This avoids both extremes: one universal base class and six mandatory provider classes per backend.

### nltiming's role

`nltiming.TimingEvaluator` should be the preferred engine-neutral evaluation path for:

- parameter metadata;
- reference and delta evaluation;
- residual scans;
- design matrices and Jacobians;
- immutable local fits;
- composite MetaPulsar evaluation.

Pylk should not move GUI, document editing, or revision concepts into `nltiming`. It should adapt workspace revisions into the frozen host/backend inputs that `nltiming` expects.

`nltiming` also defines the prior-transformed timing-parameter space and owns the transformed-space (`z`) timing optimizer that Pylk uses as its default fast timing fit. It remains responsible for timing evaluation and transformations, not GUI state and not the flexible GP visualization model. The companion [`flexfit/feature_flexible_fit.md`](flexfit/feature_flexible_fit.md) §"nltiming changes required" lists the small, in-charter additions this feature implies for `nltiming` — a z-frame timing Jacobian `J_z`, a `z`-space bounded local fit, a residual-fidelity tier on results, and a documented prior-transform contract — none of which add GP, spectrum, or noise-model machinery to `nltiming`.

### Discovery and the default scientific stack

The default scientific path should be Discovery/MetaPulsar objects with the JUG JAX-based evaluation engine where supported. This gives the workbench direct access to MetaPulsar combinations, flexible PTA model construction, JAX autodiff, and fast repeated evaluation. PINT, Tempo2/libstempo, and Vela providers remain essential native paths and validation tools; “default” is a product choice, not a claim that their semantics are interchangeable.

### PINT

PINT should supply the portable first fixture, while the preferred vertical slice wraps it through Discovery and uses JUG evaluation where supported. PINT contributes:

- native Python model and TOA objects;
- rich parameter metadata;
- narrowband and wideband residuals;
- several established fitters;
- par/tim parsing and export;
- pintk as a behavioral reference.

Pylk should not copy pintk's mutable `Pulsar` wrapper. The provider should translate between PINT objects and workspace state while retaining access to native PINT objects.

### Tempo2/libstempo

The Tempo2 provider should support native `tempopulsar` workflows and `nltiming` evaluation. Because Tempo2/libstempo can terminate or corrupt the hosting process for some inputs, robust workflows should use the available sandbox/subprocess boundary where practical.

The workspace must distinguish Tempo2-native parameter names, units, residual conventions, and row order from their canonical display forms.

### JUG

JUG's `TimingSession` is a strong provider boundary. Pylk should reuse its session API and asynchronous-computation lessons rather than its monolithic GUI window.

JUG should initially contribute:

- cached residual evaluation;
- JAX/autodiff Jacobians;
- comparison with PINT- and Tempo2-family conventions;
- named classical/correlated fit profiles where stable.

Noise controls and estimation can follow after the core state model is proven.

### Vela/PyVela

Vela should expose its `SPNTA`, `VelaFitter`, residual, whitening, and posterior capabilities without pretending they are PINT operations.

Julia initialization and long computations must occur behind the task boundary. Parameter transformations, priors, and marginalized parameters should remain visible in result metadata.

### MetaPulsar

MetaPulsar is both the incubation host for this feature and a first-class scientific document source.

Its provider must expose:

- per-PTA source sessions and timing packages;
- canonical fit parameters and their per-PTA aliases;
- shared versus PTA-specific parameters;
- canonical row order and per-session TOA mappings;
- `timing_backend()` and `timing()` through `nltiming`;
- nonlinear sample/marginalize configuration where appropriate.

The GUI must not flatten these distinctions into a single anonymous pulsar.

## Physical-model comparison and engine diagnostics

Comparing physical models is a signature interactive feature. A user should be able to hold the dataset and evaluation path fixed while comparing candidate timing components, basis choices, parameter roles, waveform reconstructions, and accepted versus proposed revisions. Useful products include residual changes, reconstructed process waveforms, weighted statistics, parameter shifts, and localized views of where one model improves or worsens the data.

Engine comparison instead serves validation and implementation diagnostics. Low-friction comparison is appropriate within a declared compatibility family:

- Tempo2 family: Tempo2/libstempo and JUG-Tempo2;
- PINT family: PINT, JUG-PINT, and Vela where components and conventions match.

Cross-family comparison requires an explicit user action and a compatibility report. MetaPulsar can construct compatible PINT/Tempo2 inputs only under the rules in `docs/METHOD_DESCRIPTION.md`, including units, `EPHEM`, clock, ecliptic convention, and `T2CMETHOD` handling. Even then, supported model subsets and residual conventions must be reported. Pylk should not present unmatched PINT and Tempo2 defaults as estimates of the same quantity.

Diagnostic products may include:

- within-family residual overlays and pairwise differences;
- lower-tier versus higher-tier residual differences after a proposed update;
- Jacobian or design-matrix column comparisons;
- parameter, unit, convention, component-support, and TOA-alignment reports;
- opt-in cross-family residual comparisons after compatibility checks.

```python
models = doc.compare_models([baseline, candidate])
models.waveform_difference(process="dm")

diagnostic = doc.compare_engines(
    ["pint", "jug-pint"],
    purpose="implementation-validation",
)
diagnostic.compare_tiers("jug-tempo2", lower="T1", higher="T4")
```

The diagnostic layer must identify relevant differences in physical model, residual tier, clocks, ephemerides, timescales, mean subtraction, tracking, parameter mapping, and component support; it must not merely label results “equal” or “different.”

## Native Python access

The native objects are part of the product, not merely a debugging escape hatch.

Read-only or detached access should be straightforward:

```python
pint_model = doc.native("pint", copy=True)
jug_session = doc.native("jug", copy=False)
```

Untracked in-place mutation cannot be guaranteed to update the workspace safely. Tracked mutation should therefore use an explicit synchronization boundary:

```python
with doc.native_edit("pint") as native:
    native.model.F0.quantity = new_f0
```

On successful exit, the provider should:

1. inspect or serialize the changed native state;
2. compute the supported canonical differences;
3. validate them;
4. commit one semantic revision;
5. invalidate downstream computations;
6. notify attached views.

If a native change cannot be represented safely, the context should fail without silently producing partial workspace state. This facility should be added after ordinary command-based editing works; it is not required for the first plot prototype.

## Background tasks and stale results

Residual evaluation, optimization, Julia startup, JAX compilation, and inference must not block the frontend.

Each task records its input revision and computation key. When it finishes:

- a derived result for an older revision may be cached and inspected;
- it must not silently replace the current plot or scientific state;
- an optimization result from an older revision may be explicitly reviewed and applied only through a validated command;
- cancellation should be supported when the backend permits it.

The first implementation needs a modest task abstraction, not a distributed scheduler.

## Frontend strategy

The core must be frontend-neutral. Two delivery modes should be prototyped before choosing a primary user experience.

### Notebook/Jupyter frontend

Advantages:

- the workspace and native objects naturally live in the Python kernel;
- notebooks, consoles, files, terminals, and rich output already exist;
- remote/HPC workflows are natural;
- a custom widget can show multiple synchronized views of one Python object.

Open concerns:

- a JupyterLab extension adds browser and TypeScript maintenance;
- browser applications do not feel like desktop applications to every user;
- kernel restarts and comm reconnection require clear behavior;
- large-array transport and frontend plotting need measurement.

The initial experiment should be a notebook-rendered widget, not a full JupyterLab application extension.

### PySide desktop frontend

Advantages:

- conventional launch and desktop interaction;
- strong local file-dialog and window behavior;
- JUG has already demonstrated PySide6 plus PyQtGraph for this domain;
- it may better serve users who do not live in notebooks.

Open concerns:

- embedding an in-process IPython kernel is poorly isolated and should not be the default architecture;
- a separate kernel complicates direct access to GUI-owned objects;
- Qt main-thread rules and arbitrary user code must not govern scientific truth.

A PySide frontend should therefore be a view of the same kernel-independent `Workspace`, not the owner of it. An external console or notebook bridge can be explored after the desktop vertical slice works.

### Initial frontend decision gate

Phase 0 should implement the same small workflow in both environments:

- open a Discovery/PINT dataset;
- render residuals;
- select TOAs;
- commit an exclusion;
- change one parameter;
- show history;
- access the workspace from Python.

The primary frontend should be selected from evidence about scientific usability, deployment, remote use, performance, and maintenance—not architecture preference alone. Both may remain supported if their shared-core boundary stays inexpensive.

This frontend decision is independent of the scientific default: both prototypes should use the Discovery/MetaPulsar plus JUG path where available, with a PINT-only fixture retained for portability and provider testing.

## Initial user interface

The first useful interface should contain four conceptual areas, regardless of frontend toolkit.

### Dataset and session browser

- open documents and pulsars;
- MetaPulsar PTA sessions;
- source timing package and active evaluation providers;
- clocks, ephemerides, timescales, and compatibility modes;
- capability and warning indicators.

### Central plot

- pre-fit, post-fit, and residual-delta views;
- MJD, year, frequency, orbital phase, DM, TOA error, and residual axes as supported;
- uncertainty bars;
- selection, zoom, inspection, and committed exclusion;
- color by observatory, backend, PTA, frequency, jump, or flag;
- candidate physical-model overlays and per-process waveform reconstructions;
- visible residual-evaluation tier and a higher-fidelity validation action;
- within-family or explicitly requested diagnostic overlays.

### Parameter inspector

- canonical and native names;
- source session or PTA;
- exact value, display value, uncertainty, and units;
- fit/frozen/sample/marginalize role;
- component/category;
- per-engine support;
- pre-fit, candidate, and accepted-result differences;
- transformed-space coordinates and priors where relevant to optimization.

### Results, tasks, and history

- running and completed computations;
- input revision and provider;
- fit or inference configuration;
- statistics and parameter differences;
- warnings and provenance;
- revision history and undo/redo;
- explicit acceptance or rejection of optimization results;
- fast GP basis configuration, waveform estimates, and clear non-posterior labeling.

## Incubation and package placement

The logical architecture should be extractable, but the initial implementation should incubate in the MetaPulsar repository to reduce packaging and coordination overhead and to reach real multi-PTA users early.

Proposed initial placement:

```text
src/metapulsar/
    pylk/
        core/
        providers/
        views/
```

The incubating import path should be `metapulsar.pylk`; this avoids claiming a stable standalone API too early.

Constraints during incubation:

- importing `metapulsar` must not import Qt, Jupyter, or frontend packages;
- frontend dependencies must be optional extras;
- core workspace tests must run headlessly;
- the core must not import MetaPulsar application internals unnecessarily;
- provider registration must allow later movement to a standalone distribution;
- `nltiming` must never depend on MetaPulsar or pylk.

Extraction to a standalone `pylk` package should occur once the incubating workbench is fully functional for its defined core workflow. Independent adoption, release cadence, packaging, and ownership remain supporting signals:

- PINT, JUG, Vela, or libstempo users adopt it independently of MetaPulsar;
- its release cadence materially diverges from MetaPulsar;
- frontend dependencies complicate MetaPulsar packaging or CI;
- external maintainers need ownership without contributing to MetaPulsar;
- the provider API is stable enough to version independently.

This approach prioritizes early users without coupling the scientific architecture permanently to MetaPulsar.

## Testing strategy

### Headless core tests

- command validation and revision creation;
- semantic versus transient state separation;
- undo/redo;
- stable TOA identity and mappings;
- reactive invalidation;
- stale task-result behavior;
- preview/commit/cancel behavior.

### Provider contract tests

- capability reporting;
- parameter and unit metadata;
- zero-delta behavior;
- TOA row alignment;
- residual and Jacobian shapes;
- source round trips where supported;
- clear unsupported-operation errors.

### Compatibility and fidelity diagnostic tests

- curated within-family residual comparisons;
- derivative/design-matrix and T0–T4 fidelity comparisons;
- matched clock, ephemeris, timescale, convention, and model-component fixtures;
- explicit incompatibility or expected differences for cross-family cases;
- MetaPulsar composite row and parameter mapping.

### Frontend tests

- a small set of critical interaction tests;
- selection and committed exclusion;
- notebook/GUI synchronization;
- stale-result display behavior;
- frontend reconnection or document reload;
- smoke tests for supported platforms.

Tests should target behavior and scientific invariants, not Qt signal counts or private widget structure.

## Delivery plan

### Phase 0: semantics, optimizer, and dual-frontend spike

- Define the minimum `Workspace`, `Revision`, `Command`, and `Capabilities` objects.
- Load one small Discovery/PINT dataset through a path that can later accept a MetaPulsar source.
- Establish stable logical TOA keys.
- Implement parameter change and committed TOA exclusion.
- Evaluate residuals through `nltiming`, with JUG autodiff where available.
- Implement one transformed-space iterative least-squares step and record its evaluation tier.
- Render the same workflow in a notebook widget and a minimal PySide window.
- Demonstrate Python access and explicit downstream invalidation.

**Exit criterion:** One scientific edit and optimization step work headlessly and through both views, with history, explicit fidelity, and no GUI-owned scientific truth.

### Phase 1: usable single-dataset workbench

- File and native-object loading, residual plots, parameter inspection, selection, masking, and semantic undo/redo.
- Named linear and transformed-space timing optimization profiles.
- Candidate physical-model comparison.
- A first fast flexible-GP waveform prototype for red and DM processes.
- Explicit result review and acceptance, provenance-aware export, and revision-keyed background work.

**Exit criterion:** A useful narrowband editing and waveform-exploration session is possible, with point estimates clearly distinguished from posterior inference.

### Phase 2: MetaPulsar and multi-session workflows

- MetaPulsar document loading and a PTA/session browser.
- Shared and PTA-specific parameter representation.
- Composite residual, physical-model, and waveform plots.
- Fast joint timing-model optimization across the combination.
- Per-session filtering, provenance, nonlinear scans, and parameter-role configuration.

**Exit criterion:** A combined pulsar can be edited and optimized through the same loop without losing per-PTA timing semantics.

### Phase 3: compatibility families and engine diagnostics

- JUG-PINT, JUG-Tempo2, Tempo2/libstempo, and Vela providers as available.
- Compatibility-family, convention, TOA-alignment, and residual-tier diagnostics.
- Within-family residual and Jacobian comparisons.
- Opt-in cross-family diagnostics gated by MetaPulsar compatibility checks.
- Subprocess isolation for unsafe native paths.

**Exit criterion:** Pylk validates compatible implementations without implying that arbitrary PINT and Tempo2 results are scientifically interchangeable.

### Phase 4: advanced modeling and inference integration

- Stable JUG and Vela optimization profiles.
- Expanded flexible-GP bases, whitening, and waveform views.
- Discovery/Enterprise inference configuration and job launch.
- Posterior and artifact inspection.
- Native edit contexts.

## Initial success criteria

The initial design is successful if:

- scientific state can be created, evaluated, and tested without a frontend;
- neither Qt nor Jupyter owns authoritative timing state;
- residuals update reactively after a semantic command;
- transient view state and uncommitted selection do not pollute history;
- excluded TOAs retain stable logical keys and can be restored;
- stale background work cannot overwrite a newer revision;
- the active workspace is inspectable from Python;
- a user can compare candidate physical models and reconstructed waveforms;
- transformed-space timing optimization converges quickly on a curated fixture and can be validated at an appropriate residual tier;
- two engines within one compatibility family can display a diagnostic difference;
- cross-family comparison is opt-in and blocked when compatibility metadata is insufficient;
- a MetaPulsar combination uses the same edit and optimization loop as a single dataset;
- MetaPulsar remains importable without GUI dependencies.

## Principal risks and mitigations

### Semantic normalization hides real differences

**Mitigation:** Preserve native names, units, conventions, provider metadata, and source provenance alongside canonical display forms. Prefer explicit incompatibility to a misleading conversion.

### Reactive machinery becomes a framework project

**Mitigation:** Begin with a fixed dependency vocabulary and computation keys. Introduce a general graph abstraction only after real workflows require dynamic graph construction.

### Revision history becomes noisy or expensive

**Mitigation:** Revise only semantic scientific state. Keep view state and previews ephemeral. Store structurally shared state or compact diffs only if measurement shows full snapshots are costly.

### Provider abstractions proliferate

**Mitigation:** Start with a coarse provider and optional capability protocols. Extract a new interface only after two concrete providers share it.

### Native objects bypass workspace invariants

**Mitigation:** Default to detached/read access. Add tracked native-edit contexts only for changes that providers can diff and validate.

### The frontend decision is made ideologically

**Mitigation:** Complete the Phase 0 notebook and PySide spikes against the same core, then choose from deployment and usability evidence.

### Incubation creates permanent MetaPulsar coupling

**Mitigation:** Keep frontend imports optional, keep core contracts generic, and document objective extraction criteria.

### Engine diagnostics produce false scientific conclusions

**Mitigation:** Make physical-model comparison the default. Restrict frictionless engine overlays to declared compatibility families, require aligned TOAs, and surface residual tier, clock, ephemeris, timescale, mean subtraction, tracking, unit, and model-support metadata. Gate cross-family diagnostics on an explicit action and compatibility report.

### The fast GP model is mistaken for production inference

**Mitigation:** Label its outputs as interactive waveform estimates, retain bounds and model configuration in provenance, and validate against full Discovery/Enterprise analyses. Never present its width estimates as posterior constraints.

### Fast optimization leaves the regime where its approximation is valid

**Mitigation:** Record the anchor and tier for every step, monitor parameter displacement and predicted residual change, and re-evaluate candidate states at T2/T3/T4 as appropriate before acceptance.

## Open design questions

1. Which timing-context fields must be frozen initially, and which later receive explicit transformation commands?
2. Is a logical document plus immutable original row index sufficient as the first internal TOA key?
3. How should parameter previews work without mutating the current revision?
4. What displacement or residual-error thresholds trigger T2, T3, or T4 validation before accepting an optimization?
5. Which parameters and priors can the first transformed-space least-squares profile support robustly?
6. What exact hierarchy, truncation, and data-driven bounds define the fast flexible-GP visualization model?
7. Which basis blocks are required first: red noise, DM variations, chromatic noise, timing directions, or a smaller subset?
8. What metadata and tests establish compatibility-family membership for a loaded model?
9. Which physical-model comparisons should be first-class commands?
10. Which frontend should become primary after the Phase 0 spike?
11. Which native changes, if any, justify `native_edit` after command-based editing is mature?
12. What exact “fully functional” checklist triggers extraction from `metapulsar.pylk` into a standalone package?

## Recommendation

Proceed with a small semantics-first Phase 0 inside `metapulsar.pylk`, using Discovery/MetaPulsar plus JUG as the preferred scientific path and retaining a portable PINT path.

Do not begin by porting pylk widgets, embedding an IPython kernel, building a general event-sourcing framework, implementing every backend, or embedding production MCMC. Define one scientific edit, one reactive evaluation path, one transformed-space optimization step, one revision transition, and one physical-model comparison on a curated fixture. Render that same core through minimal notebook and PySide views.

If that slice is clear and pleasant, add the fast flexible-GP waveform model, then make MetaPulsar combinations first-class before broad engine diagnostics. The larger vision remains one Python-accessible workspace in which the pulsar-timing ecosystem can be used together without erasing valuable backend differences—or mistaking backend agreement for the scientific purpose of the application.
