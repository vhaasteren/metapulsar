"""Enterprise smoke tests for standalone MetaPulsar duck compatibility."""

import pytest

pytest.importorskip("enterprise")
pytestmark = pytest.mark.requires_enterprise

import enterprise.signals.parameter as parameter  # noqa: E402
from enterprise.signals import gp_signals, signal_base, white_signals  # noqa: E402

from metapulsar.metapulsar import MetaPulsar  # noqa: E402
from metapulsar.mockpulsar import (  # noqa: E402
    create_mock_libstempo,
    write_mock_pta_files,
)


def _build_pulsar(directory):
    pulsars = {
        "pta_a": create_mock_libstempo(
            n_toas=30, name="J1857+0943", telescope="pta_a", seed=10
        ),
        "pta_b": create_mock_libstempo(
            n_toas=30, name="J1857+0943", telescope="pta_b", seed=20
        ),
    }
    return MetaPulsar(
        pulsars,
        combination_strategy="per_pta",
        pta_files=write_mock_pta_files(pulsars, directory),
    )


def test_enterprise_signal_factories_accept_standalone_pulsar(tmp_path):
    pulsar = _build_pulsar(tmp_path)

    measurement = white_signals.MeasurementNoise(efac=parameter.Constant(1.0))
    timing = gp_signals.TimingModel()

    measurement_signal = measurement(pulsar)
    timing_signal = timing(pulsar)

    assert measurement_signal is not None
    assert timing_signal is not None


def test_enterprise_pta_assembly_accepts_standalone_pulsar(tmp_path):
    pulsar = _build_pulsar(tmp_path)

    model = (
        white_signals.MeasurementNoise(efac=parameter.Constant(1.0))
        + gp_signals.TimingModel()
    )
    pta = signal_base.PTA([model(pulsar)])

    assert pta is not None
    assert pta.pulsars == [pulsar.name]
