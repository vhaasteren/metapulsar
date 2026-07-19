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

Non-sampling API calls have been smoke-tested end to end; the NUTS/PTMCMC cells
follow the same patterns as `tests/test_joint_timing.py`. Outputs are not
committed — regenerate by running the notebooks.
