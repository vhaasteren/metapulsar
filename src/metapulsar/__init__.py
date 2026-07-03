"""MetaPulsar - Multi-PTA pulsar timing data combination framework.

This package provides tools for combining pulsar timing data from multiple PTA
collaborations (EPTA, PPTA, NANOGrav, MPTA, etc.) into unified "metapulsar"
objects for gravitational wave detection.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("metapulsar")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Core classes
from .metapulsar import MetaPulsar
from .metapulsar_factory import (
    MetaPulsarFactory,
    reorder_ptas_for_pulsar,
    create_metapulsar,
    create_all_metapulsars,
    pta_summary,
)
from .file_discovery_service import (
    FileDiscoveryService,
    PTA_DATA_RELEASES,
    discover_files,
    get_pulsar_names_from_file_data,
    filter_file_data_by_pulsars,
)
from .layout_discovery_service import (
    LayoutDiscoveryService,
    discover_layout,
    combine_layouts,
)
from .parameter_manager import (
    ParameterManager,
    ParameterMapping,
    ParameterInconsistencyError,
)
from .mockpulsar import (
    MockLibstempo,
    MockParameter,
    create_mock_libstempo,
    create_mock_timing_data,
    create_mock_flags,
    validate_mock_data,
)
from .tim_file_analyzer import TimFileAnalyzer, TimMetadata
from .selection_utils import create_staggered_selection

# Exceptions
from .pint_helpers import PINTDiscoveryError

_TIMING_LAZY_EXPORTS = {
    "NonLinearTimingModel": (
        "metapulsar.timing.nonlinear_timing_model",
        "NonLinearTimingModel",
    ),
    "ParameterSpace": ("metapulsar.timing.space", "ParameterSpace"),
    "EnterprisePulsarLike": ("metapulsar.timing.protocols", "EnterprisePulsarLike"),
    "EphemerisExtras": ("metapulsar.timing.protocols", "EphemerisExtras"),
    "TimingBackend": ("metapulsar.timing.protocols", "TimingBackend"),
    "JaxTimingBackend": ("metapulsar.timing.protocols", "JaxTimingBackend"),
    "PulsarInterface": ("metapulsar.timing.protocols", "PulsarInterface"),
}


def __getattr__(name: str):
    if name in _TIMING_LAZY_EXPORTS:
        module_name, attr = _TIMING_LAZY_EXPORTS[name]
        return getattr(import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__author__ = "Rutger van Haasteren, Wangwei Yu, David Wright"
__email__ = "rutger@vhaasteren.com"

__all__ = [
    # Core classes
    "MetaPulsar",
    "MetaPulsarFactory",
    "FileDiscoveryService",
    "PTA_DATA_RELEASES",
    "LayoutDiscoveryService",
    "ParameterManager",
    "ParameterMapping",
    "ParameterInconsistencyError",
    "MockLibstempo",
    "MockParameter",
    "create_mock_libstempo",
    "create_mock_timing_data",
    "create_mock_flags",
    "validate_mock_data",
    "TimFileAnalyzer",
    "TimMetadata",
    "create_staggered_selection",
    "PINTDiscoveryError",
    # Convenience functions
    "discover_files",
    "discover_layout",
    "combine_layouts",
    "reorder_ptas_for_pulsar",
    "create_metapulsar",
    "create_all_metapulsars",
    "pta_summary",
    "get_pulsar_names_from_file_data",
    "filter_file_data_by_pulsars",
    # Nonlinear timing (lazy; see metapulsar.timing)
    "NonLinearTimingModel",
    "ParameterSpace",
    "EnterprisePulsarLike",
    "EphemerisExtras",
    "TimingBackend",
    "JaxTimingBackend",
    "PulsarInterface",
]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_TIMING_LAZY_EXPORTS))
