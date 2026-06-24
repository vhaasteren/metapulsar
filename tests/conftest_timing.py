"""Slice-0 shared fixtures for upcoming timing package tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


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


class FakeTimingBackend:
    """Minimal backend scaffold for early timing unit tests."""

    def __init__(self, fitpars: list[str]):
        self.fitpars = tuple(fitpars)

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        return np.zeros(4, dtype=float)

    def design_matrix(self) -> np.ndarray:
        return np.zeros((4, len(self.fitpars)), dtype=float)


class FakeTimingHost:
    """Minimal host scaffold implementing the expected timing host shape."""

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

    def timing_backend(self, name: str) -> FakeTimingBackend:
        return FakeTimingBackend(self.fitpars)

    def cache_token(self) -> str:
        return "fake-host-v1"


@pytest.fixture
def fake_timing_host() -> FakeTimingHost:
    """Provide a deterministic fake timing host for early timing tests."""
    return FakeTimingHost()
