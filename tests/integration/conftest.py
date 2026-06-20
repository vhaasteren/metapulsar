"""Pytest configuration for integration tests."""

import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def data_dir():
    """Return the data directory path."""
    return Path(__file__).parent.parent.parent / "data"


@pytest.fixture(scope="session")
def ipta_dr2_dir(data_dir):
    """Return the IPTA DR2 submodule directory."""
    return data_dir / "ipta-dr2"


@pytest.fixture(scope="session")
def ipta_dr3_dir(data_dir):
    """Return the IPTA DR3 directory."""
    return data_dir / "IPTA-DR3"


@pytest.fixture(scope="session")
def legacy_module():
    """Import and return the legacy module (skip if not installed)."""
    return pytest.importorskip("metapulsar.legacy.metapulsar")


@pytest.fixture(scope="session")
def new_module():
    """Import and return the new module."""
    from metapulsar import MetaPulsar, MetaPulsarFactory, FileDiscoveryService

    return {
        "MetaPulsar": MetaPulsar,
        "MetaPulsarFactory": MetaPulsarFactory,
        "FileDiscoveryService": FileDiscoveryService,
    }


@pytest.fixture(scope="session")
def test_pulsars():
    """Return a list of test pulsar names for integration testing."""
    return [
        "J0030+0451",  # Millisecond pulsar
        "J0437-4715",  # Bright millisecond pulsar
        "J1909-3744",  # Binary pulsar
        "J1713+0747",  # Well-timed pulsar
    ]


@pytest.fixture(scope="session")
def test_pta_data_releases():
    """Return test PTA data releases for integration testing."""
    return [
        "epta_dr1_v2_2",
        "ppta_dr2",
        "nanograv_9y",
        "epta_dr2",
        "inpta_dr1",
        "mpta_dr1",
        "nanograv_12y",
        "nanograv_15y",
    ]


@pytest.fixture(scope="session")
def available_data_sets(ipta_dr2_dir, ipta_dr3_dir):
    """Check which data sets are available and return their info."""
    available = {}

    # Check DR2 submodule
    if ipta_dr2_dir.exists():
        available["dr2"] = {
            "epta": ipta_dr2_dir / "EPTA_v2.2",
            "ppta": ipta_dr2_dir / "PPTA_dr1dr2",
            "nanograv": ipta_dr2_dir / "NANOGrav_9y",
        }

    # Check DR3 directory
    if ipta_dr3_dir.exists():
        available["dr3"] = {
            "epta": ipta_dr3_dir / "EPTA_DR2",
            "inpta": ipta_dr3_dir / "InPTA_DR1",
            "mpta": ipta_dr3_dir / "MPTA_DR1",
            "nanograv_12y": ipta_dr3_dir / "NANOGrav_12y",
            "nanograv_15y": ipta_dr3_dir / "NANOGrav_15y",
        }

    return available
