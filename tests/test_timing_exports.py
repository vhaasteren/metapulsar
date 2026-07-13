"""Public export surface for the timing package."""

import pytest

import metapulsar
from metapulsar.timing import (
    EnterprisePulsarLike,
    EphemerisExtras,
    JaxTimingBackend,
    NonLinearTimingModel,
    ParameterSpace,
    TimingBackend,
    PulsarInterface,
)


def test_timing_subpackage_exports():
    assert NonLinearTimingModel is not None
    assert ParameterSpace is not None
    assert EnterprisePulsarLike is not None
    assert PulsarInterface is not None
    assert JaxTimingBackend is not None
    assert EphemerisExtras is not None
    assert TimingBackend is not None


def test_metapulsar_lazy_timing_exports():
    assert metapulsar.NonLinearTimingModel is NonLinearTimingModel
    assert metapulsar.ParameterSpace is ParameterSpace
    assert metapulsar.PulsarInterface is PulsarInterface


def test_timing_imports_and_constructs_without_jug():
    """JUG-free configs must import and construct with jug/jax uninstalled.

    Runs in a subprocess with ``jug`` and ``jax`` blocked from importing, so
    the check is meaningful even in environments where both are installed.
    """
    import subprocess
    import sys

    code = "; ".join(
        [
            "import sys",
            "sys.modules['jug'] = None",
            "sys.modules['jax'] = None",
            "import metapulsar.timing",
            "from metapulsar.timing import NonLinearTimingModel",
            "m = NonLinearTimingModel("
            "engines={'tempo2': 'libstempo', 'pint': 'pint'})",
            "assert m.tempo2_jug_options is None",
            "m.set_prior('F0', 'normal', mean=0.0, std=1.0)",
            "m2 = m.with_engines({'tempo2': 'libstempo', 'pint': 'pint'})",
            "assert m2.tempo2_jug_options is None",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_jug_config_still_resolves_tempo2_options():
    pytest.importorskip("jug")
    model = NonLinearTimingModel(engines="jug")
    options = model.tempo2_jug_options
    assert options is not None
    assert "iers_policy" in options
