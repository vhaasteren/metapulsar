"""Mixed-engine routing tests for MetaPulsar timing backends."""

import numpy as np

from metapulsar.metapulsar import MetaPulsar, SessionFiles
from nltiming.backends.composite import PulsarTimingBackend


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
    pulsar._session_files = {
        "epta": SessionFiles(par_path=par, tim_path=tim, timing_package="tempo2"),
        "ng9": SessionFiles(par_path=par, tim_path=tim, timing_package="pint"),
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
    pulsar._timing_backend_cache = {}

    backend = pulsar.timing_backend(
        {"tempo2": "libstempo", "pint": "jug"}, linearized=True
    )

    assert isinstance(backend, PulsarTimingBackend)
    assert [session.backend.backend_name for session in backend._sessions] == [
        "tempo2",
        "jug",
    ]
    np.testing.assert_allclose(backend.residual_delta(np.zeros(1)), 0.0)
