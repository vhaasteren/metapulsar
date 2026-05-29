"""MetaPulsar nonlinear timing-model package."""

from .engines import (
    JugDeltaEngine,
    PintDeltaEngine,
    Tempo2DeltaEngine,
    TimingDeltaEngine,
    build_delta_engine,
)
from .partitioning import TimingPartition, compute_timing_partition
from .signal_builder import build_nonlinear_timing_signal
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
    "build_delta_engine",
    "build_nonlinear_timing_signal",
]
