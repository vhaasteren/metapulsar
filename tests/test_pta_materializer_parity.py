"""Golden parity tests for MetaPulsar-owned PTA materializers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pint.models import get_model_and_toas

from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import MockParameter, create_mock_libstempo
from metapulsar.pta_data import (
    _PtaTimingData,
    _tempo2_is_ecliptic,
    materialize_pint,
    materialize_tempo2,
    pulsar_distance,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "enterprise_surface"
SAMPLE_DIR = Path(__file__).resolve().parent / "fixtures" / "sample_parfiles"
PULSE_DIR = Path(__file__).resolve().parent / "fixtures" / "pulse_tracking"

EXACT_ARRAYS = (
    "_toas",
    "_stoas",
    "_residuals",
    "_toaerrs",
    "_ssbfreqs",
    "_telescope",
    "_designmatrix",
    "_pos",
)
TOLERANCED_ARRAYS = ("_pos_t", "_planetssb", "_sunssb")
EPHEMERIS_RTOL = 5.0e-14
EPHEMERIS_ATOL = 5.0e-14

PUBLIC_EXACT = (
    "toas",
    "stoas",
    "residuals",
    "toaerrs",
    "freqs",
    "telescope",
    "Mmat",
    "backend_flags",
    "pos",
)
PUBLIC_TOLERANCED = ("pos_t", "planetssb", "sunssb")


def _load_fixture(name: str):
    path = FIXTURE_DIR / name
    assert path.is_file(), f"missing golden fixture {path}"
    return np.load(path, allow_pickle=False)


def _flag_keys(npz) -> list[str]:
    return sorted(key[len("flag__") :] for key in npz.files if key.startswith("flag__"))


def _assert_record_parity(record: _PtaTimingData, npz) -> None:
    assert record.name == str(npz["name"])
    assert record.timing_package == str(npz["timing_package"])
    np.testing.assert_array_equal(record.fitpars, npz["fitpars"])
    # PINT model.params order for set-only names is not stable across AbsPhase
    # setup; validate membership and multiplicity instead of positional order.
    np.testing.assert_array_equal(sorted(record.setpars), sorted(npz["setpars"]))
    assert record._raj == float(npz["_raj"])
    assert record._decj == float(npz["_decj"])
    np.testing.assert_array_equal(record._pdist, npz["_pdist"])

    for key in EXACT_ARRAYS:
        np.testing.assert_array_equal(getattr(record, key), npz[key], err_msg=key)

    for key in TOLERANCED_ARRAYS:
        got = getattr(record, key)
        expected = np.asarray(npz[key])
        # Enterprise PINT may emit a single (3,) direction for _pos_t; the
        # materializer expands that to (n, 3) for the record contract.
        if key == "_pos_t" and expected.shape == (3,) and got.ndim == 2:
            expected = np.broadcast_to(expected, got.shape)
        np.testing.assert_allclose(
            got,
            expected,
            rtol=EPHEMERIS_RTOL,
            atol=EPHEMERIS_ATOL,
            equal_nan=True,
            err_msg=key,
        )

    fixture_flags = _flag_keys(npz)
    assert sorted(record._flags) == fixture_flags
    for flag in fixture_flags:
        np.testing.assert_array_equal(
            record._flags[flag], npz[f"flag__{flag}"], err_msg=f"flag {flag}"
        )


def test_golden_fixtures_load_without_pickle():
    for name in (
        "pint_equatorial.npz",
        "pint_ecliptic.npz",
        "tempo2_mock_equatorial.npz",
        "metapulsar_tempo2_pair.npz",
    ):
        data = _load_fixture(name)
        assert len(data.files) > 0
    manifest = (FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8")
    assert "enterprise_version" in manifest


def test_pint_equatorial_parity():
    model, toas = get_model_and_toas(
        str(SAMPLE_DIR / "simple.par"),
        str(SAMPLE_DIR / "simple.tim"),
        planets=True,
    )
    record = materialize_pint(model, toas)
    _assert_record_parity(record, _load_fixture("pint_equatorial.npz"))


def test_pint_ecliptic_parity():
    model, toas = get_model_and_toas(
        str(PULSE_DIR / "nanograv_like.par"),
        str(PULSE_DIR / "nanograv_like.tim"),
        planets=True,
    )
    record = materialize_pint(model, toas)
    _assert_record_parity(record, _load_fixture("pint_ecliptic.npz"))


def test_tempo2_mock_equatorial_parity():
    mock = create_mock_libstempo(
        n_toas=30, name="J1857+0943", telescope="pta_a", seed=10
    )
    record = materialize_tempo2(mock)
    _assert_record_parity(record, _load_fixture("tempo2_mock_equatorial.npz"))


def test_metapulsar_public_surface_parity():
    pulsars = {
        "pta_a": create_mock_libstempo(
            n_toas=30, name="J1857+0943", telescope="pta_a", seed=10
        ),
        "pta_b": create_mock_libstempo(
            n_toas=30, name="J1857+0943", telescope="pta_b", seed=20
        ),
    }
    mp = MetaPulsar(pulsars, combination_strategy="per_pta")
    npz = _load_fixture("metapulsar_tempo2_pair.npz")

    assert mp.name == str(npz["name"])
    np.testing.assert_array_equal(mp.fitpars, npz["fitpars"])
    np.testing.assert_array_equal(mp.pdist, npz["pdist"])
    # Enterprise/Discovery feather metadata: restored with to_feather, lazily
    # derived from the reference-PTA PINT model (see tests/test_feather_io.py).
    assert "dm" in MetaPulsar.__dict__
    assert "dmx" in MetaPulsar.__dict__

    for key in PUBLIC_EXACT:
        np.testing.assert_array_equal(getattr(mp, key), npz[key], err_msg=key)
    for key in PUBLIC_TOLERANCED:
        np.testing.assert_allclose(
            getattr(mp, key),
            npz[key],
            rtol=EPHEMERIS_RTOL,
            atol=EPHEMERIS_ATOL,
            equal_nan=True,
            err_msg=key,
        )
    for flag in _flag_keys(npz):
        np.testing.assert_array_equal(
            mp.flags[flag], npz[f"flag__{flag}"], err_msg=f"flag {flag}"
        )


def test_pta_timing_data_rejects_bad_shapes():
    n = 3
    good = dict(
        name="J0000+0000",
        timing_package="tempo2",
        _toas=np.zeros(n),
        _stoas=np.zeros(n),
        _residuals=np.zeros(n),
        _toaerrs=np.zeros(n),
        _ssbfreqs=np.zeros(n),
        _telescope=np.array(["x"] * n),
        _designmatrix=np.zeros((n, 1)),
        _flags={"f": np.array(["a"] * n)},
        fitpars=["Offset"],
        setpars=[],
        _raj=0.0,
        _decj=0.0,
        _pos=np.zeros(3),
        _pos_t=np.zeros((n, 3)),
        _planetssb=np.zeros((n, 9, 6)),
        _sunssb=np.zeros((n, 6)),
        _pdist=(1.0, 0.2),
    )
    _PtaTimingData(**good)
    with pytest.raises(ValueError, match="_designmatrix"):
        _PtaTimingData(**{**good, "_designmatrix": np.zeros((n, 2))})
    with pytest.raises(ValueError, match="_pos_t"):
        _PtaTimingData(**{**good, "_pos_t": np.zeros((n, 2))})
    with pytest.raises(ValueError, match="flag 'f'"):
        _PtaTimingData(**{**good, "_flags": {"f": np.array(["a", "b"])}})


def test_pulsar_distance_lookup_policy():
    # B1855+09 is present in the Enterprise catalog copy.
    known = pulsar_distance("B1855+09")
    assert known == pulsar_distance("1855+09")
    assert isinstance(known[0], float) and isinstance(known[1], float)
    assert pulsar_distance("J0030+0451")[0] == pytest.approx(0.28)
    assert pulsar_distance("UNKNOWN_PSR_XYZ") == (1.0, 0.2)


def test_tempo2_elat_only_skips_ecliptic_rotation():
    """ELAT alone must not trigger ecliptic vector rotation (Enterprise bug fix)."""
    mock = create_mock_libstempo(n_toas=5, name="J1857+0943", telescope="pta_a", seed=3)
    mock._params["ELAT"] = MockParameter(0.1)
    mock._setpars = (*mock._setpars, "ELAT")
    assert not _tempo2_is_ecliptic(mock)
    original_pos = mock.psrPos.copy()
    record = materialize_tempo2(mock)
    np.testing.assert_array_equal(record._pos_t, original_pos)
