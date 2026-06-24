"""Duck-typing compatibility checks for standalone MetaPulsar hosts."""

from pathlib import Path
import re

import enterprise.signals.parameter as parameter
from enterprise.signals import gp_signals, signal_base, white_signals
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


def test_enterprise_pta_assembly_accepts_standalone_host():
    host = _build_host()

    model = (
        white_signals.MeasurementNoise(efac=parameter.Constant(1.0))
        + gp_signals.TimingModel()
    )
    pta = signal_base.PTA([model(host)])

    assert pta is not None
    assert pta.pulsars == [host.name]


def test_discovery_signal_factories_accept_standalone_host():
    host = _build_host()

    measurement = discovery_signals.makenoise_measurement(host)
    timing = discovery_signals.makegp_timing(host)

    assert measurement is not None
    assert timing is not None


def test_slice_3a_upstream_audit_no_basepulsar_type_gates_in_signal_paths():
    repo_root = Path(__file__).resolve().parents[1]
    ref_packages = repo_root / "ref-packages"
    signal_roots = {
        "enterprise": ref_packages / "enterprise" / "enterprise" / "signals",
        "discovery": ref_packages / "discovery" / "src" / "discovery",
        "enterprise_extensions": ref_packages
        / "enterprise_extensions"
        / "enterprise_extensions",
    }

    basepulsar_hits: list[str] = []
    pintpulsar_hits: list[tuple[str, str]] = []
    for name, root in signal_roots.items():
        for py_file in root.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            if "BasePulsar" in text:
                basepulsar_hits.append(f"{name}:{py_file.relative_to(root)}")
            if "PintPulsar" in text:
                pintpulsar_hits.append((name, str(py_file.relative_to(root))))

    assert basepulsar_hits == []

    # One known enterprise_extensions guard exists for physical ephemeris and
    # is not part of the default factory paths exercised above.
    allowed_pint_hits = {
        ("enterprise_extensions", "dropout.py"),
    }
    assert set(pintpulsar_hits) <= allowed_pint_hits
    assert ("enterprise_extensions", "dropout.py") in set(pintpulsar_hits)

    dropout_text = (signal_roots["enterprise_extensions"] / "dropout.py").read_text(
        encoding="utf-8"
    )
    assert re.search(
        r"isinstance\(psr,\s*enterprise\.pulsar\.PintPulsar\)", dropout_text
    )
