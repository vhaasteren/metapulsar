"""Mixed-engine routing tests for MetaPulsar timing engines."""

import numpy as np

from metapulsar.metapulsar import MetaPulsar, PtaFiles
from nltiming.engines.composite import PulsarTimingEngine


class _SessionPulsar:
    def __init__(self, n_toa: int):
        self._toas = np.arange(n_toa, dtype=float)


def test_metapulsar_routes_engines_by_native_compatibility(tmp_path):
    par = tmp_path / "session.par"
    tim = tmp_path / "session.tim"
    par.write_text("F0 1\n", encoding="utf-8")
    tim.write_text("FORMAT 1\n", encoding="utf-8")

    pulsar = MetaPulsar.__new__(MetaPulsar)
    pulsar.name = "J0000+0000"
    pulsar._epulsars = {"epta": _SessionPulsar(2), "ng9": _SessionPulsar(2)}
    pulsar._pta_files = {
        "epta": PtaFiles(par_path=par, tim_path=tim, timing_package="tempo2"),
        "ng9": PtaFiles(par_path=par, tim_path=tim, timing_package="pint"),
    }
    pulsar._fitparameters = {"F0": {"epta": "F0", "ng9": "F0"}}
    pulsar.fitpars = ["F0"]
    pulsar._parfile_dicts = {"epta": {"F0": "1.0"}, "ng9": {"F0": "1.0"}}
    pulsar._designmatrix = np.ones((4, 1), dtype=float)
    pulsar._toas = np.arange(4, dtype=float)
    pulsar._residuals = np.zeros(4, dtype=float)
    pulsar._toaerrs = np.ones(4, dtype=float)
    pulsar._ssbfreqs = np.full(4, 1400.0, dtype=float)
    pulsar._backend_flags = np.array(["a", "a", "b", "b"])
    pulsar._flags = {"f": pulsar._backend_flags}
    pulsar._isort = slice(None, None, None)
    pulsar._timing_engine_cache = {}

    engine = pulsar.timing_engine(
        {"tempo2": "libstempo", "pint": "jug"}, linearized=True
    )

    assert isinstance(engine, PulsarTimingEngine)
    assert [session.engine.engine_name for session in engine._contributions] == [
        "tempo2",
        "jug",
    ]
    np.testing.assert_allclose(engine.residual_delta(np.zeros(1)), 0.0)
