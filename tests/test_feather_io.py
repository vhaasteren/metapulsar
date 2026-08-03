"""Enterprise/Discovery-compatible feather I/O for MetaPulsar.

Contract consumer: combine_five_pta_pulsars.py calls mpsr.to_feather(...).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from metapulsar.feather_io import read_pulsar_feather, save_pulsar_feather
from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import create_mock_libstempo


def _minimal_duck(*, n: int = 5, npar: int = 3, name: str = "J0000+0000"):
    rng = np.random.default_rng(0)
    return SimpleNamespace(
        name=name,
        toas=np.linspace(1.0e9, 1.0e9 + n, n, dtype=float),
        stoas=np.linspace(1.0e9, 1.0e9 + n, n, dtype=float) + 0.1,
        toaerrs=np.full(n, 1.0e-6, dtype=float),
        residuals=rng.normal(scale=1.0e-6, size=n),
        freqs=np.full(n, 1400.0, dtype=float),
        backend_flags=np.array([f"be{i % 2}" for i in range(n)], dtype="U32"),
        telescope=np.array(["GBT"] * n, dtype="U32"),
        Mmat=rng.normal(size=(n, npar)),
        sunssb=rng.normal(size=(n, 3)),
        pos_t=rng.normal(size=(n, 3)),
        planetssb=rng.normal(size=(n, 9, 6)),
        flags={"f": np.array([f"sys{i % 2}" for i in range(n)], dtype="U32")},
        dm=15.5,
        dmx=None,
        pdist=(1.0, 0.2),
        _pdist=(1.0, 0.2),
        pos=np.array([0.1, 0.2, 0.97], dtype=float),
        phi=0.3,
        theta=1.2,
        fitpars=[f"PAR{i}" for i in range(npar)],
        setpars=[f"PAR{i}" for i in range(npar)],
        noisedict={f"{name}_efac": 1.0, "OTHER_efac": 2.0},
    )


def _build_metapulsar(*, n_toas: int = 20) -> MetaPulsar:
    pulsars = {
        "pta_a": create_mock_libstempo(
            n_toas=n_toas, name="J1857+0943", telescope="pta_a", seed=11
        ),
        "pta_b": create_mock_libstempo(
            n_toas=n_toas, name="J1857+0943", telescope="pta_b", seed=22
        ),
    }
    return MetaPulsar(pulsars, combination_strategy="per_pta")


def test_feather_io_round_trip_minimal_duck(tmp_path):
    duck = _minimal_duck()
    path = tmp_path / "duck.feather"
    save_pulsar_feather(duck, path, noisedict=duck.noisedict)

    out = read_pulsar_feather(path)
    np.testing.assert_allclose(out.toas, duck.toas)
    np.testing.assert_allclose(out.stoas, duck.stoas)
    np.testing.assert_allclose(out.toaerrs, duck.toaerrs)
    np.testing.assert_allclose(out.residuals, duck.residuals)
    np.testing.assert_allclose(out.freqs, duck.freqs)
    np.testing.assert_allclose(out.Mmat, duck.Mmat)
    np.testing.assert_allclose(out.sunssb, duck.sunssb)
    np.testing.assert_allclose(out.pos_t, duck.pos_t)
    np.testing.assert_allclose(out.planetssb, duck.planetssb)
    assert list(out.telescope) == list(duck.telescope)
    assert list(out.backend_flags) == list(duck.backend_flags)
    assert set(out.flags) == {"f"}
    np.testing.assert_array_equal(out.flags["f"], duck.flags["f"])
    assert out.name == duck.name
    assert out.dm == duck.dm
    assert out.dmx is None
    assert out.pdist == list(duck.pdist)
    np.testing.assert_allclose(out.pos, duck.pos)
    assert out.phi == duck.phi
    assert out.theta == duck.theta
    assert out.fitpars == duck.fitpars
    assert out.noisedict == {f"{duck.name}_efac": 1.0}


def test_feather_io_mmat_column_index_ordering(tmp_path):
    """Pin Mmat_10 / Mmat_11 ordering (locale-safe index sort on read)."""
    duck = _minimal_duck(npar=12)
    path = tmp_path / "mmat12.feather"
    save_pulsar_feather(duck, path)
    out = read_pulsar_feather(path)
    assert out.Mmat.shape == (5, 12)
    np.testing.assert_allclose(out.Mmat, duck.Mmat)


def test_metapulsar_to_feather_round_trip(tmp_path):
    mp = _build_metapulsar()
    path = tmp_path / "mp.feather"
    mp.to_feather(path)

    out = read_pulsar_feather(path)
    assert len(out.toas) == len(mp.toas)
    assert out.Mmat.shape == mp.Mmat.shape
    np.testing.assert_allclose(out.toas, mp.toas)
    np.testing.assert_allclose(out.residuals, mp.residuals)
    np.testing.assert_allclose(out.Mmat, mp.Mmat)
    np.testing.assert_allclose(out.planetssb, mp.planetssb)
    np.testing.assert_allclose(out.sunssb, mp.sunssb)
    np.testing.assert_allclose(out.pos_t, mp.pos_t)
    assert out.name == mp.name
    assert out.phi == mp.phi
    assert out.theta == mp.theta
    np.testing.assert_allclose(out.pos, mp.pos)
    assert out.pdist == list(mp.pdist)
    assert isinstance(out.dm, float)
    assert out.dm == pytest.approx(mp.dm)


def test_metapulsar_dm_lazy_and_cache_invalidation():
    mp = _build_metapulsar()
    assert mp._dispersion_metadata_ready is False
    dm = mp.dm
    assert mp._dispersion_metadata_ready is True
    assert isinstance(dm, float)
    mp._invalidate_timing_caches()
    assert mp._dispersion_metadata_ready is False
    assert mp._dm is None
    assert mp.dm == pytest.approx(dm)


def test_to_feather_survives_unavailable_dispersion_metadata(tmp_path):
    """dm/dmx are best effort: a broken reference model must not lose the snapshot."""
    mp = _build_metapulsar()

    def _unparseable():
        raise ValueError("PINT cannot parse retained par content")

    mp.pint_model = _unparseable  # instance attribute shadows the method

    path = tmp_path / "nodm.feather"
    mp.to_feather(path)

    assert mp.dm is None
    assert mp.dmx is None
    out = read_pulsar_feather(path)
    assert len(out.toas) == len(mp.toas)
    np.testing.assert_allclose(out.Mmat, mp.Mmat)
    assert out.dm is None


def test_discovery_feather_round_trip(tmp_path):
    discovery = pytest.importorskip("discovery")
    mp = _build_metapulsar()
    path = tmp_path / "discovery.feather"
    mp.to_feather(path)

    psr = discovery.pulsar.Pulsar.read_feather(str(path))
    assert len(psr.toas) == len(mp.toas)
    assert psr.Mmat.shape[0] == len(mp.toas)
    np.testing.assert_allclose(psr.toas, mp.toas)
    np.testing.assert_allclose(psr.Mmat, mp.Mmat)


@pytest.mark.requires_enterprise
def test_enterprise_feather_round_trip(tmp_path):
    pytest.importorskip("enterprise")
    from enterprise.pulsar import FeatherPulsar

    mp = _build_metapulsar()
    path = tmp_path / "enterprise.feather"
    mp.to_feather(path)

    psr = FeatherPulsar.read_feather(str(path))
    assert hasattr(psr, "telescope")
    assert psr.telescope is not None
    np.testing.assert_allclose(psr.toas, mp.toas)
    np.testing.assert_allclose(psr.Mmat, mp.Mmat)
    assert len(psr.telescope) == len(mp.telescope)
