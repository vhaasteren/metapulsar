"""Public export surface for the timing package."""

import metapulsar
from metapulsar.timing import (
    EnterprisePulsarLike,
    EphemerisExtras,
    JaxTimingBackend,
    NonLinearTimingModel,
    ParameterSpace,
    TimingBackend,
    TimingHost,
)


def test_timing_subpackage_exports():
    assert NonLinearTimingModel is not None
    assert ParameterSpace is not None
    assert EnterprisePulsarLike is not None
    assert TimingHost is not None
    assert JaxTimingBackend is not None
    assert EphemerisExtras is not None
    assert TimingBackend is not None


def test_metapulsar_lazy_timing_exports():
    assert metapulsar.NonLinearTimingModel is NonLinearTimingModel
    assert metapulsar.ParameterSpace is ParameterSpace
    assert metapulsar.TimingHost is TimingHost
