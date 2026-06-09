"""MetaPulsar nonlinear timing-model package."""

from .engines import (
    JugDeltaEngine,
    JaxTimingDeltaEngine,
    PintDeltaEngine,
    Tempo2DeltaEngine,
    TimingDeltaEngine,
    build_delta_engine,
    infer_jug_param_mapping,
)
from .jug_jax_engine import JugJaxTimingEngine, JugLinearizedTimingContext
from .parameter_space import SampledTimingParameterSpace
from .partitioning import TimingPartition, compute_timing_partition
from .signal_builder import build_nonlinear_timing_signal
from .signal_builder_discovery import (
    DiscoveryNonlinearTimingComponents,
    build_discovery_nonlinear_timing_components,
    build_discovery_nonlinear_timing_likelihood,
)
from .priors import (
    CHEAT_PRIOR_SIGMA_MULTIPLIER,
    PriorOverrideSpec,
    PriorPolicy,
    PintPriorAdapter,
    SampledTimingParameter,
    SampledTimingParameterRegistry,
    build_sampled_timing_parameter_registry,
)
from .transforms import AffineTransform, TransformRegistry
from .wls import wls_uncertainties

__all__ = [
    "AffineTransform",
    "TransformRegistry",
    "CHEAT_PRIOR_SIGMA_MULTIPLIER",
    "PriorOverrideSpec",
    "PriorPolicy",
    "PintPriorAdapter",
    "SampledTimingParameter",
    "SampledTimingParameterRegistry",
    "SampledTimingParameterSpace",
    "build_sampled_timing_parameter_registry",
    "wls_uncertainties",
    "TimingPartition",
    "compute_timing_partition",
    "TimingDeltaEngine",
    "JaxTimingDeltaEngine",
    "PintDeltaEngine",
    "Tempo2DeltaEngine",
    "JugDeltaEngine",
    "JugJaxTimingEngine",
    "JugLinearizedTimingContext",
    "infer_jug_param_mapping",
    "build_delta_engine",
    "build_nonlinear_timing_signal",
    "DiscoveryNonlinearTimingComponents",
    "build_discovery_nonlinear_timing_components",
    "build_discovery_nonlinear_timing_likelihood",
]
