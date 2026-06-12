from pathlib import Path

import pytest


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
