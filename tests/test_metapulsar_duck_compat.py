"""Duck-typing compatibility checks for standalone MetaPulsar hosts."""

import enterprise.signals.parameter as parameter
from enterprise.signals import gp_signals, white_signals
import discovery.signals as discovery_signals

from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import create_mock_libstempo


def _build_host():
    pulsars = {
        "pta_a": create_mock_libstempo(
            n_toas=30, name="J1857+0943", telescope="pta_a", seed=10
        ),
        "pta_b": create_mock_libstempo(
            n_toas=30, name="J1857+0943", telescope="pta_b", seed=20
        ),
    }
    return MetaPulsar(pulsars, combination_strategy="composite")


def test_enterprise_signal_factories_accept_standalone_host():
    host = _build_host()

    measurement = white_signals.MeasurementNoise(efac=parameter.Constant(1.0))
    timing = gp_signals.TimingModel()

    measurement_signal = measurement(host)
    timing_signal = timing(host)

    assert measurement_signal is not None
    assert timing_signal is not None


def test_discovery_signal_factories_accept_standalone_host():
    host = _build_host()

    measurement = discovery_signals.makenoise_measurement(host)
    timing = discovery_signals.makegp_timing(host)

    assert measurement is not None
    assert timing is not None
