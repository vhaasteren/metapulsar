<p align="left">
  <img src="docs/logo/metapulsar_logo_notext.png" alt="MetaPulsar Logo" width="180" />
</p>

# MetaPulsar

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/metapulsar.svg)](https://pypi.org/project/metapulsar/)
[![DOI](https://zenodo.org/badge/727659043.svg)](https://doi.org/10.5281/zenodo.17626664)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-219%20passing-brightgreen)](https://github.com/metapulsar)

A framework for combining pulsar timing data from multiple PTA collaborations into unified "metapulsar" objects for gravitational wave detection analysis.

## Features

- **Multi-PTA Data Combination**: Combine data from EPTA, PPTA, NANOGrav, MPTA, and other PTAs
- **Enterprise Integration**: Full compatibility with the Enterprise pulsar timing analysis package
- **Dual Timing Package Support**: Works with both PINT and libstempo/tempo2
- **Flexible Parameter Management**: Support for "consistent" and "composite" combination strategies

## Quick Start

### Installation

Install the latest release from PyPI:

```bash
pip install metapulsar

# With optional extras
pip install "metapulsar[dev,libstempo]"
```

Or install from source for development:

```bash
git clone https://github.com/vhaasteren/metapulsar.git
cd metapulsar
pip install -e .

# With optional dependencies
pip install -e ".[dev,libstempo]"
```

### Basic Usage

```python
from metapulsar import create_metapulsar

# Create MetaPulsar
metapulsar = create_metapulsar(
    file_data=pulsar_data,
    combination_strategy="consistent",
    combine_components=["astrometry", "spindown", "binary", "dispersion"],
    add_dm_derivatives=True,
)

# Access combined data
print(f"Number of TOAs: {len(metapulsar.toas)}")
print(f"PTA names: {list(metapulsar._pulsars.keys())}")
```

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
from loguru import logger
import sys

logger.add(sys.stdout, level="DEBUG")
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
- **pint-pulsar** ≥ 0.9.0
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
