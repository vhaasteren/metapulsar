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
from .metapulsar import MetaPulsar, normalize_combination_strategy
from .metapulsar_factory import (
    MetaPulsarFactory,
    reorder_ptas_for_pulsar,
    create_metapulsar,
    create_all_metapulsars,
    pta_summary,
)
from .file_discovery import (
    FileDiscovery,
    PTA_DATA_RELEASES,
    FileSelectionError,
    AmbiguousFileError,
    MissingOverrideError,
    discover_files,
    get_pulsar_names_from_file_data,
    filter_file_data_by_pulsars,
)
from .layout_discovery import (
    DataReleaseLayout,
    discover_layout,
    combine_layouts,
)
from .parameter_manager import (
    AlignmentPolicy,
    BinaryConversionMode,
    H3OnlyPolicy,
    ParameterManager,
    ParameterMapping,
    ParameterInconsistencyError,
    UnsupportedBinaryPolicy,
)
from .binary_family_convert import (
    BinaryConversionDecision,
    BinaryConversionError,
    BinaryConversionMetadata,
    BinaryConversionRecord,
    BinaryConversionReport,
    BinaryFidelityReport,
    BinaryPatch,
    BinaryScaleGate,
    apply_binary_patch,
    convert_shared_binary,
    decide_binary_conversion,
)
from .mockpulsar import (
    MockLibstempo,
    MockParameter,
    create_mock_libstempo,
    create_mock_timing_data,
    create_mock_flags,
    validate_mock_data,
)
from .position_helpers import discover_pulsars_by_position
from .tim_file_analyzer import TimFileAnalyzer, TimMetadata
from .selection_utils import create_staggered_selection

# Exceptions
from .pint_helpers import PINTDiscoveryError
from .parfile_header import (
    ensure_metapulsar_par_header,
    format_metapulsar_par_header,
)
from .parfile_update import (
    ParUpdateResult,
    apply_native_deltas,
    gls_update_and_write_par,
)

_TIMING_LAZY_EXPORTS = {
    "NonLinearTimingModel": (
        "nltiming.nonlinear_timing_model",
        "NonLinearTimingModel",
    ),
    "ParameterSpace": ("nltiming.space", "ParameterSpace"),
    "RunIOError": ("nltiming.run_io", "RunIOError"),
    "RunManifest": ("nltiming.run_io", "RunManifest"),
    "RunResults": ("nltiming.run_io", "RunResults"),
    "build_run_manifest": ("nltiming.run_io", "build_run_manifest"),
    "derived_param_name": (
        "nltiming.run_io",
        "derived_param_name",
    ),
    "decode_physical": (
        "nltiming.run_io",
        "decode_physical",
    ),
    "save_discovery_checkpoint": (
        "nltiming.run_io",
        "save_discovery_checkpoint",
    ),
    "load_run": ("nltiming.run_io", "load_run"),
    "EnterprisePulsarLike": ("nltiming.protocols", "EnterprisePulsarLike"),
    "EphemerisExtras": ("nltiming.protocols", "EphemerisExtras"),
    "TimingEngine": ("nltiming.protocols", "TimingEngine"),
    "JaxTimingEngine": ("nltiming.protocols", "JaxTimingEngine"),
    "TimingPulsar": ("nltiming.protocols", "TimingPulsar"),
    "PulsarData": ("nltiming.protocols", "PulsarData"),
    "TimingCapabilities": ("nltiming.evaluator", "TimingCapabilities"),
    "TimingEvaluation": ("nltiming.evaluator", "TimingEvaluation"),
    "TimingEvaluator": ("nltiming.evaluator", "TimingEvaluator"),
    "TimingFitResult": ("nltiming.evaluator", "TimingFitResult"),
    "TimingParameter": ("nltiming.evaluator", "TimingParameter"),
    "TimingParameters": ("nltiming.evaluator", "TimingParameters"),
    "TimingScan": ("nltiming.evaluator", "TimingScan"),
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
    "FileDiscovery",
    "PTA_DATA_RELEASES",
    "FileSelectionError",
    "AmbiguousFileError",
    "MissingOverrideError",
    "DataReleaseLayout",
    "AlignmentPolicy",
    "BinaryConversionMode",
    "H3OnlyPolicy",
    "UnsupportedBinaryPolicy",
    "ParameterManager",
    "ParameterMapping",
    "ParameterInconsistencyError",
    "BinaryConversionDecision",
    "BinaryConversionError",
    "BinaryConversionMetadata",
    "BinaryConversionRecord",
    "BinaryConversionReport",
    "BinaryFidelityReport",
    "BinaryPatch",
    "BinaryScaleGate",
    "apply_binary_patch",
    "convert_shared_binary",
    "decide_binary_conversion",
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
    "format_metapulsar_par_header",
    "ensure_metapulsar_par_header",
    "ParUpdateResult",
    "apply_native_deltas",
    "gls_update_and_write_par",
    # Convenience functions
    "discover_files",
    "discover_layout",
    "combine_layouts",
    "reorder_ptas_for_pulsar",
    "create_metapulsar",
    "create_all_metapulsars",
    "normalize_combination_strategy",
    "pta_summary",
    "get_pulsar_names_from_file_data",
    "filter_file_data_by_pulsars",
    "discover_pulsars_by_position",
    # Nonlinear timing (lazy; see nltiming)
    "NonLinearTimingModel",
    "ParameterSpace",
    "RunIOError",
    "RunManifest",
    "RunResults",
    "build_run_manifest",
    "derived_param_name",
    "decode_physical",
    "save_discovery_checkpoint",
    "load_run",
    "EnterprisePulsarLike",
    "EphemerisExtras",
    "TimingEngine",
    "JaxTimingEngine",
    "TimingPulsar",
    "PulsarData",
    "TimingCapabilities",
    "TimingEvaluation",
    "TimingEvaluator",
    "TimingFitResult",
    "TimingParameter",
    "TimingParameters",
    "TimingScan",
]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_TIMING_LAZY_EXPORTS))
