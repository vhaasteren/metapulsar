"""MetaPulsar nonlinear timing-model package."""

from .engines import (
    JugDeltaEngine,
    PintDeltaEngine,
    Tempo2DeltaEngine,
    TimingDeltaEngine,
    build_delta_engine,
    infer_jug_param_mapping,
)
from .partitioning import TimingPartition, compute_timing_partition
from .signal_builder import build_nonlinear_timing_signal
from .signal_builder_discovery import (
    DiscoveryNonlinearTimingComponents,
    build_discovery_nonlinear_timing_components,
    build_discovery_nonlinear_timing_likelihood,
)
from .transforms import AffineTransform, TransformRegistry

__all__ = [
    "AffineTransform",
    "TransformRegistry",
    "TimingPartition",
    "compute_timing_partition",
    "TimingDeltaEngine",
    "PintDeltaEngine",
    "Tempo2DeltaEngine",
    "JugDeltaEngine",
    "infer_jug_param_mapping",
    "build_delta_engine",
    "build_nonlinear_timing_signal",
    "DiscoveryNonlinearTimingComponents",
    "build_discovery_nonlinear_timing_components",
    "build_discovery_nonlinear_timing_likelihood",
]
