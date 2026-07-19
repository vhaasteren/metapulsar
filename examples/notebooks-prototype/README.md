# Prototype notebooks — new nonlinear-timing API

Runnable prototypes for the timing-coordinate-charts / geometry-certification
feature, written against the **new** `nltiming` API (typed `TimingInference`
plan, coordinate charts, `whitening=None` identity static layer, dynamic joint
transport, geometry certifier, pivot amplitude). They supersede the `sample=` /
`transform=` pedagogy of `examples/notebooks/nonlinear_timing.ipynb`, which still
uses the removed API and is migrated separately.

Run top-to-bottom in the MetaPulsar devcontainer (JUG, discovery, numpyro,
enterprise, PINT). Datasets are simulated with PINT — no external data.

- **`01_nonlinear_timing_charts.ipynb`** — the typed inference plan
  (`sample_all` / `default` / `groups`), one coordinate chart per sampled axis
  (`chart_summary`), delta-flat vs z-prior distinct records, a joint full-basis
  NUTS run with `dense_mass="auto"` block adaptation, physical decode via
  `jm.to_df`, chain-preserving diagnostics, and the Enterprise path (including
  `sample_z_coefficients=True`).
- **`02_geometry_certification_and_pivot.ipynb`** — fixed-expansion refinement
  at the conditional MPE (marginal objective), off-zero geometry certification
  over box hyper-probes with standalone JSON+NPZ report write/read, the
  transport-center report, and pivot-amplitude red noise
  (`make_powerlaw_pivot`, sensitivity-weighted pivot frequency, decode to
  1/yr, and wiring the pivot PSD into a real `makegp_fourier` GP).
  **Section 2b is the feature's headline demonstration**: the certifier catches a
  ~million-fold off-mode geometry defect (spin parameters sampled on wide uniform
  charts — the `F0` axis that broke the earlier decentering run), and the
  `identically_linear=` knob turns it into a clean, `Hessian ≈ identity` pass.

Two **validation notebooks** run the new-API workflow on the real IPTA-DR2
J1640+2224 data (they supersede the old-API `notebooks-dev/nlt_ipta_dr2_*`
prototypes, which are left intact):

- **`03_j1640_decentering_validation.ipynb`** — the joint full-basis decentering
  validation: charts read chart-type-first, the `identically_linear` geometry fix
  on real data (Hessian eigenvalues collapse toward 1; the report still doesn't
  fully pass because the binary parameters stay nonlinear — the honest residual),
  expansion refinement, certification + standalone report, block-dense-mass NUTS
  with each chain plotted separately, and pivoted vs 1/yr red-noise amplitude.
- **`04_j1640_marginalization_validation.ipynb`** — delta-flat vs z-prior
  marginalization as distinct first-class records (different GP, different
  log-likelihood, different fingerprint), and sampling the z-prior coefficients.

Requires the EPTA-DR2 J1640 par/tim under `data/ipta-dr2/`. Non-sampling API
calls have been smoke-tested end to end (incl. on real J1640); the NUTS/PTMCMC
cells follow the same patterns as `tests/test_joint_timing.py`. Outputs are not
committed — regenerate by running the notebooks.
