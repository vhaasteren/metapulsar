"""MetaPulsar - Multi-PTA pulsar timing data combination framework.

This package provides tools for combining pulsar timing data from multiple PTA
collaborations (EPTA, PPTA, NANOGrav, MPTA, etc.) into unified "metapulsar"
objects for gravitational wave detection.
"""

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
]
