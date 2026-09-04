"""Run the nltiming README quickstart scripts end to end.

The scripts live in the nltiming checkout under ``ref-packages/`` (git-ignored)
and import ``metapulsar``, so nltiming's own CI cannot run them. They are
exercised here, in the devcontainer, when that checkout and the engines exist.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = (
    Path(__file__).resolve().parent.parent
    / "ref-packages"
    / "nltiming"
    / "examples"
    / "scripts"
)

pytestmark = pytest.mark.slow


def _run(script: str) -> None:
    path = SCRIPTS / script
    if not path.exists():
        pytest.skip(f"{path} not present")
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=path.parent,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    assert "sampled:" in proc.stdout


@pytest.mark.requires_jug
def test_quickstart_discovery():
    pytest.importorskip("jug")
    pytest.importorskip("discovery")
    pytest.importorskip("numpyro")
    _run("quickstart_discovery.py")


@pytest.mark.requires_libstempo
@pytest.mark.requires_enterprise
def test_quickstart_enterprise():
    pytest.importorskip("libstempo")
    pytest.importorskip("enterprise")
    pytest.importorskip("PTMCMCSampler")
    _run("quickstart_enterprise.py")
