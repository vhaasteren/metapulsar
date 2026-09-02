"""Timing-engine adapters owned by MetaPulsar."""

from __future__ import annotations

from nltiming.engine_config import (
    _ENGINE_CHOICES,
    _IMPL_FAMILY,
    normalize_engines,
)

from .composite import (
    validate_composite_against_pulsar,
    PtaContribution,
    PulsarJaxTimingEngine,
    PulsarTimingEngine,
    build_composite_engine,
)
from .delta import (
    JaxTimingDeltaEngine,
    JugDeltaEngine,
    PintDeltaEngine,
    Tempo2DeltaEngine,
    TimingDeltaEngine,
    build_delta_engine,
    infer_jug_param_mapping,
)
from nltiming.hybrid import (
    hybrid_linearized_fitpars,
    is_hybrid_engine_axis,
    validate_nonlinear_params,
)
from .jug import JugEngine, LinearizedJugEngine, verify_jug_native_chain
from .pint import LinearizedPintEngine, PintEngine
from .tempo2 import LinearizedLibstempoEngine, LibstempoEngine
from .vela import EmptyMaskParameterError, VelaDeltaEngine, VelaEngine
from .vela_jax import VelaJaxEngine


def build_engine(*, fitpars, nrows, contributions, design_matrix=None):
    return build_composite_engine(
        fitpars=fitpars,
        nrows=nrows,
        contributions=contributions,
        design_matrix=design_matrix,
    )


__all__ = [
    "validate_composite_against_pulsar",
    "_ENGINE_CHOICES",
    "_IMPL_FAMILY",
    "normalize_engines",
    "TimingDeltaEngine",
    "JaxTimingDeltaEngine",
    "PintDeltaEngine",
    "Tempo2DeltaEngine",
    "JugDeltaEngine",
    "infer_jug_param_mapping",
    "build_delta_engine",
    "PtaContribution",
    "PulsarTimingEngine",
    "PulsarJaxTimingEngine",
    "build_composite_engine",
    "build_engine",
    "PintEngine",
    "LinearizedPintEngine",
    "LibstempoEngine",
    "LinearizedLibstempoEngine",
    "JugEngine",
    "LinearizedJugEngine",
    "verify_jug_native_chain",
    "validate_nonlinear_params",
    "is_hybrid_engine_axis",
    "hybrid_linearized_fitpars",
    "VelaDeltaEngine",
    "VelaEngine",
    "VelaJaxEngine",
    "EmptyMaskParameterError",
]
