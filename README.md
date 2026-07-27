<p align="left">
  <img src="docs/logo/metapulsar_logo_notext.png" alt="MetaPulsar Logo" width="180" />
</p>

# MetaPulsar

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/metapulsar.svg)](https://pypi.org/project/metapulsar/)
[![DOI](https://zenodo.org/badge/727659043.svg)](https://doi.org/10.5281/zenodo.17626664)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-219%20passing-brightgreen)](https://github.com/metapulsar)

A framework for combining pulsar timing data from multiple PTA collaborations into unified "metapulsar" objects for gravitational wave detection analysis.

## Features

- **Multi-PTA Data Combination**: Combine data from EPTA, PPTA, NANOGrav, MPTA, and other PTAs
- **Enterprise Integration**: Full compatibility with the Enterprise pulsar timing analysis package
- **Nonlinear Timing**: `NonLinearTimingModel` component for Discovery/NumPyro and Enterprise samplers
- **Dual Timing Package Support**: Works with both PINT and libstempo/tempo2
- **Flexible Parameter Management**: Support for "consistent" and "composite" combination strategies

## Quick Start

### Installation

Install the latest release from PyPI:

```bash
pip install metapulsar

# With optional extras
pip install "metapulsar[dev,libstempo,timing]"
```

Or install from source for development:

```bash
git clone https://github.com/vhaasteren/metapulsar.git
cd metapulsar
pip install -e .

# With optional dependencies
pip install -e ".[dev,libstempo,timing]"
```

### Basic Usage

```python
from metapulsar import create_metapulsar

# Create MetaPulsar
metapulsar = create_metapulsar(
    file_data=pulsar_data,
    combination_strategy="shared",
    combine_components=["astrometry", "spindown", "binary", "dispersion"],
    add_dm_derivatives=True,
)

# Access combined data
print(f"Number of TOAs: {len(metapulsar.toas)}")
print(f"PTA names: {list(metapulsar._pulsars.keys())}")
```

### Pulse-number tracking (`use_pulse_numbers`)

When combining PTAs with `combination_strategy="shared"`, merged par files can
break per-PTA phase coherence. Pulse-number tracking preserves phase connectivity
via `-pn` flags and `TRACK -2` on the timing model.

Pass a **string** mode to `create_metapulsar` (default `"yes"`). Booleans are no
longer accepted.

| Mode | Behavior |
|------|----------|
| `"no"` | Ignore pulse numbers; no `TRACK -2` override (Tempo2). |
| `"yes"` | Reuse complete `-pn` on all TOAs; otherwise re-derive from original `par` + `tim`; warn on mixed partial `-pn`. |
| `"reuse"` | Same as `"yes"` when complete; warn and re-derive when `-pn` is missing or incomplete. |
| `"overwrite"` | Always re-derive `-pn` from the original coherent `par` + `tim`. |

Migration: replace `use_pulse_numbers=True` with `"yes"` and `False` with `"no"`.

### Nonlinear timing (`NonLinearTimingModel`)

Par files are aligned to PINT's orbital chart in `ParameterManager` before
merging (hybrid `PB + FBn` → free `FB0`); see
[`feature_orbital_chart_alignment.md`](feature_orbital_chart_alignment.md).

**Linear objects (locked names):** `design_matrix` / \(M\) is the delay tangent
in the PINT/tempo2 fitter sign (uncentered; what every other PTA package means
by “design matrix”). `residual_jacobian` / \(J=-M\) is
\(\partial(\Delta r)/\partial\theta\) from the **gauge-free** residual function
— never called a design matrix. The old `waveform_jacobian` noun is deleted:
the delay tangent *is* \(M\). Full vocabulary:
[`docs/design_matrix_terminology.md`](docs/design_matrix_terminology.md);
design notes: `ref-packages/jug/feature_phase_gauge.md`.

MetaPulsar exposes a live nonlinear-timing interface as well as an
Enterprise/Discovery-compatible frozen-pulsar view. Open its engine-independent
evaluator for parameter inspection, residual evaluation, scans, residual
Jacobians, and immutable local fits:

```python
from metapulsar import create_metapulsar

mp = create_metapulsar(...)
timing = mp.timing(
    engines={"pint": "jug", "tempo2": "jug"},
    derivative_method="autodiff",
)

print(timing.parameters.names)
shifted = timing.evaluate({"TASC": 1e-3}, frame="delta")
scan = timing.scan("TASC", [-0.5, 0.0, 0.5], scale="PB")
J = timing.jacobian(method="autodiff")
fit = timing.fit(["F0", "F1"])
```

The evaluator never mutates the pulsar, TOAs, per-PTA inputs, or par files.
Its residual convention is explicit: `residual_delta = r(theta) - r(theta_ref)`,
`residuals = mp.residuals + residual_delta`, and `delay = -residual_delta`.

For sampler-facing nonlinear timing, resolve a config-only nltiming model for the
same pulsar:

```python
import discovery as ds
from nltiming import NonLinearTimingModel, TimingInference, WhiteningConfig, sampling

# Name marginalized axes via TimingInference; whitening= selects the static layer.
# whitening=None is required for sampling.numpyro.joint_model (dynamic full-basis).
ntm = NonLinearTimingModel(
    engines="jug",
    inference=TimingInference.groups(delta_flat=["DM", "DM1"]),
    whitening=WhiteningConfig(),
)
ctx = ntm.for_pulsar(mp)

likelihood = ds.PulsarLikelihood([
    mp.residuals,
    ds.makenoise_measurement_simple(mp, noisedict),
    *ctx.discovery_signals(),
])

model = sampling.numpyro.model(likelihood, ctx, fixed=noisedict)
mcmc = sampling.numpyro.nuts(model, ctx)
```

nltiming exposes three model builders on the same `TimingPulsar`:
`sampling.numpyro.model` (static whitening, shown above),
`sampling.numpyro.joint_model` (full-basis dynamic transport, `whitening=None`),
and `sampling.numpyro.decentered_model` (marginalized dynamic decentering — the
small sampled timing block whitened against the live `C(η)`, `whitening=None`).
See the nltiming README and `examples/notebooks/03_j1640_decentering_validation.ipynb`
(the three-mode comparison) for choosing among them.

Install timing extras with `pip install "metapulsar[timing]"`.
The JUG extra currently requires Python 3.12. The package base and nltiming
require Python 3.11 or newer.

Introductory notebooks and the coordinate/geometry guide live in the
`nltiming` package (`ref-packages/nltiming/examples/notebooks/` and that
package’s README when using the MetaPulsar devcontainer checkout). MetaPulsar
is still required today as the `TimingPulsar` implementation those notebooks
bind to.

## Documentation

- **[Interactive Tutorial](examples/notebooks/using_metapulsar.ipynb)** - Complete usage guide with examples
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation
- **[Method Description](docs/METHOD_DESCRIPTION.md)** - Detailed description of the direct combination method
- **[Poster](docs/poster/metapulsar-poster-2025.pdf)** - MetaPulsar poster (2025)

## Examples

- **[Python Examples](examples/)** - Standalone Python examples

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/metapulsar
```

## Release Workflow (GitHub -> PyPI)

MetaPulsar uses tag-based versioning via `setuptools-scm`. The package version is
derived from the Git tag (for example, `v0.9.6` -> `0.9.6`), and PyPI publishing
is triggered automatically when a GitHub Release is published.

### One-time setup

- Ensure the repository secret `PYPI_API_TOKEN` is configured in GitHub settings.
- Use release tags in the format `vX.Y.Z` (for example, `v0.9.6`).

### Normal release process

1. Merge PRs into `main`.
2. In GitHub, create a new Release and create/select a new tag like `v0.9.6`
   from `main`.
3. Publish the release.
4. GitHub Actions runs the release workflow, builds distributions from that tag,
   and uploads them to PyPI.

### Important notes

- Do not manually edit a static version in `pyproject.toml`; version comes from
  the Git tag.
- Reusing an existing release tag/version will fail the PyPI publish step, which
  is intentional to prevent silent no-op releases.

## Troubleshooting

### Debug Mode

```python
import loguru
import sys
loguru.logger.remove()
loguru.logger.add(sys.stdout, level="DEBUG")
```

### CI import error with `pkg_resources`

If CI fails during test collection with an error like
`ImportError: cannot import name 'Requirement' from 'pkg_resources'`, this
usually comes from `enterprise-pulsar` importing the legacy `pkg_resources`
API via setuptools.

Current workaround in this repository:
- CI installs `setuptools<81` to preserve compatibility with current
  `enterprise-pulsar` releases.

Long-term plan:
- Upgrade to a future `enterprise-pulsar` release that removes the
  `pkg_resources` dependency, then remove the setuptools pin.

## Dependencies

- **Python 3.9+**
- **numpy** ≥ 1.20.0
- **astropy** ≥ 5.0.0
- **scipy** ≥ 1.7.0
- **pint-pulsar** with [nanograv/PINT#2023](https://github.com/nanograv/PINT/pull/2023) (hybrid `PB+FBn` → free `FB0`; pinned in `pyproject.toml` / `requirements.txt` until that lands in a release). Dev checkouts: `pip install -e ref-packages/PINT`. Diagnose installs with `pathlib.Path(pint.__file__).resolve()` and `hasattr(PulsarBinary, "_bridge_pb_to_fb0")` — never `pip show` alone (a user-site `pint` can shadow an editable install while reporting the right version).
- **enterprise-pulsar** ≥ 3.0.0


## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{metapulsar,
  title={MetaPulsar},
  author={van Haasteren, Rutger and Yu, Wang-Wei and Wright, David},
  year={2025},
  doi={10.5281/zenodo.17626664},
  url={https://github.com/vhaasteren/metapulsar},
  license={MIT}
}
```

## Authors

- **Rutger van Haasteren** - *Lead Developer* - [rutger@vhaasteren.com](mailto:rutger@vhaasteren.com)
- **Wang-Wei Yu** - *Co-Developer* - [wangwei.yu@aei.mpg.de](mailto:wangwei.yu@aei.mpg.de)
- **David Wright** - *Co-Developer* - [dcw3.dev@gmail.com](mailto:dcw3.dev@gmail.com)

## Support

- **Issues**: [GitHub Issues](https://github.com/vhaasteren/metapulsar/issues)
- **Email**: [rutger@vhaasteren.com](mailto:rutger@vhaasteren.com)
