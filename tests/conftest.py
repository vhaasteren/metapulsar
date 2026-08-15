import itertools
from pathlib import Path

import pytest
import numpy as np


def pytest_addoption(parser):
    parser.addoption(
        "--sandbox-tempo2-debug",
        action="store_true",
        default=False,
        help=(
            "Enable DEBUG/INFO logs from metapulsar.sandbox_tempo2 during tests. "
            "By default these logs are suppressed (WARNING+ only)."
        ),
    )


@pytest.fixture(scope="session", autouse=True)
def suppress_sandbox_tempo2_debug_logs(request):
    """Reduce sandbox tempo2 logging noise during tests.

    Keep WARNING and above visible while suppressing DEBUG/INFO chatter.
    """
    if request.config.getoption("--sandbox-tempo2-debug"):
        # Explicit debug mode: keep full sandbox logger output.
        yield
        return

    import metapulsar.sandbox_tempo2 as sandbox_tempo2

    original_debug = sandbox_tempo2.logger.debug
    original_info = sandbox_tempo2.logger.info

    sandbox_tempo2.logger.debug = lambda *args, **kwargs: None
    sandbox_tempo2.logger.info = lambda *args, **kwargs: None
    try:
        yield
    finally:
        sandbox_tempo2.logger.debug = original_debug
        sandbox_tempo2.logger.info = original_info


@pytest.fixture
def mock_metapulsar(tmp_path):
    """Build a ``MetaPulsar`` from mock pulsars, with its retained pars on disk.

    ``MetaPulsar`` requires ``pta_files`` for every PTA -- it reads par text
    from those files and never re-serializes an engine object -- so mock-backed
    construction has to materialize the mocks' own pars first. Use this instead
    of calling ``MetaPulsar(mocks, ...)`` directly.
    """
    from metapulsar.metapulsar import MetaPulsar
    from metapulsar.mockpulsar import write_mock_pta_files

    counter = itertools.count()

    def _build(pulsars, **kwargs):
        directory = tmp_path / f"pta_files_{next(counter)}"
        kwargs.setdefault("pta_files", write_mock_pta_files(pulsars, directory))
        return MetaPulsar(pulsars, **kwargs)

    return _build


@pytest.fixture(scope="session")
def parfiles_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_parfiles"


@pytest.fixture
def load_parfile_text(parfiles_dir):
    def _load(name: str) -> str:
        p = (parfiles_dir / name).resolve()
        if not p.exists():
            # helpful debug if it ever happens again
            available_files = [f.name for f in parfiles_dir.iterdir()]
            raise FileNotFoundError(
                f"Missing test parfile: {p}. Available: {available_files}"
            )
        return p.read_text(encoding="utf-8")

    return _load


@pytest.fixture(scope="session")
def timing_fixtures_dir() -> Path:
    """Return the fixtures directory used by timing slice tests."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def timing_minimal_par_tim_paths(timing_fixtures_dir: Path) -> dict[str, Path]:
    """Provide a tiny par/tim pair for fast timing-slice tests."""
    return {
        "par": timing_fixtures_dir / "sample_parfiles" / "simple.par",
        "tim": timing_fixtures_dir / "sample_parfiles" / "simple.tim",
    }


class FakeTimingEngine:
    """Minimal engine stub for early timing unit tests."""

    def __init__(self, fitpars: list[str]):
        self.fitpars = tuple(fitpars)
        self.native_units = {name: "native" for name in self.fitpars}
        self._theta_exact = {name: "0.0" for name in self.fitpars}

    def reference_theta(self) -> np.ndarray:
        return np.zeros(len(self.fitpars), dtype=float)

    def reference_theta_exact(self) -> dict[str, str]:
        return dict(self._theta_exact)

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        return np.zeros(4, dtype=float)

    def design_matrix(self) -> np.ndarray:
        return np.zeros((4, len(self.fitpars)), dtype=float)


class FakeTimingPulsar:
    """Minimal pulsar stub implementing the expected timing pulsar shape."""

    def __init__(self) -> None:
        self.name = "FAKEPSR"
        self.fitpars = ["F0", "F1"]
        self._toas = np.array([3.0, 1.0, 4.0, 2.0], dtype=float)
        self._residuals = np.zeros(4, dtype=float)
        self._toaerrs = np.full(4, 1e-6, dtype=float)
        self._freqs = np.full(4, 1400.0, dtype=float)
        self._Mmat = np.zeros((4, 2), dtype=float)
        self._flags = {"pta": np.array(["fake"] * 4, dtype="U8")}
        self._backend_flags = np.array(["fake"] * 4, dtype="U8")

    @property
    def toas(self) -> np.ndarray:
        return self._toas

    @property
    def residuals(self) -> np.ndarray:
        return self._residuals

    @property
    def toaerrs(self) -> np.ndarray:
        return self._toaerrs

    @property
    def freqs(self) -> np.ndarray:
        return self._freqs

    @property
    def Mmat(self) -> np.ndarray:
        return self._Mmat

    @property
    def flags(self) -> dict[str, np.ndarray]:
        return self._flags

    @property
    def backend_flags(self) -> np.ndarray:
        return self._backend_flags

    def pint_model(self):  # pragma: no cover - placeholder for Slice 3b.
        return None

    def timing_engine(self, engines="jug") -> FakeTimingEngine:
        return FakeTimingEngine(self.fitpars)

    def can_use_engines(self, engines="jug") -> bool:
        return True

    def state_id(self) -> str:
        return "fake-pulsar-v1"


@pytest.fixture
def fake_timing_pulsar() -> FakeTimingPulsar:
    """Provide a deterministic fake pulsar interface for early timing tests."""
    return FakeTimingPulsar()
