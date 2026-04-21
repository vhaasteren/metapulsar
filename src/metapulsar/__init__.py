"""MetaPulsar - Multi-PTA pulsar timing data combination framework.

This package provides tools for combining pulsar timing data from multiple PTA
collaborations (EPTA, PPTA, NANOGrav, MPTA, etc.) into unified "metapulsar"
objects for gravitational wave detection.
"""

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
from .tim_file_analyzer import TimFileAnalyzer
from .selection_utils import create_staggered_selection
from .consistency import (
    TensionResult,
    FFTIntPosterior,
    subset_metapulsar,
    list_ptas,
    build_fftint_posterior,
    samples_to_mcsamples,
    hyper_tension,
    waveform_tension,
    timing_tension,
    combined_vs_single_tension,
    summarize,
)

# Exceptions
from .pint_helpers import PINTDiscoveryError


__version__ = "0.1.0"
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
    "create_staggered_selection",
    "PINTDiscoveryError",
    # Cross-PTA consistency checks
    "TensionResult",
    "FFTIntPosterior",
    "subset_metapulsar",
    "list_ptas",
    "build_fftint_posterior",
    "samples_to_mcsamples",
    "hyper_tension",
    "waveform_tension",
    "timing_tension",
    "combined_vs_single_tension",
    "summarize",
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
