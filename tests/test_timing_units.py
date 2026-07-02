"""Slice-1 tests for timing units conversion helpers."""

import astropy.units as u
import numpy as np

from metapulsar.timing.units import (
    display_unit,
    native_physical_bounds,
    normalize_param_name,
    storage_unit,
    to_display,
    to_native,
)


class _FakeParam:
    def __init__(self, units):
        self.units = units


class _FakePintModel:
    def __init__(self, params):
        for name, param in params.items():
            setattr(self, name, param)


def _astro_model() -> _FakePintModel:
    return _FakePintModel(
        {
            "RAJ": _FakeParam(u.hourangle),
            "DECJ": _FakeParam(u.deg),
            "T0": _FakeParam(u.day),
        }
    )


def test_raj_hourangle_roundtrip():
    model = _astro_model()
    display_hours = np.array([0.0, 6.0, 12.0, 18.5], dtype=float)
    native = to_native("RAJ", display_hours, pint_model=model)
    roundtrip = to_display("RAJ", native, pint_model=model)
    np.testing.assert_allclose(roundtrip, display_hours)
    np.testing.assert_allclose(native, display_hours)
    assert display_unit("RAJ", model) == "hourangle"
    assert storage_unit("RAJ", model) == u.hourangle


def test_decj_degree_roundtrip():
    model = _astro_model()
    display_deg = np.array([-30.0, -1.5, 0.0, 12.25, 80.0], dtype=float)
    native = to_native("DECJ", display_deg, pint_model=model)
    roundtrip = to_display("DECJ", native, pint_model=model)
    np.testing.assert_allclose(roundtrip, display_deg)
    np.testing.assert_allclose(native, display_deg)
    assert display_unit("DECJ", model) == "deg"
    assert storage_unit("DECJ", model) == u.deg


def test_t0_identity_conversion():
    model = _astro_model()
    mjd = np.array([53000.125, 54000.0], dtype=float)
    native = to_native("T0", mjd, pint_model=model)
    roundtrip = to_display("T0", native, pint_model=model)
    np.testing.assert_allclose(roundtrip, mjd)
    assert display_unit("T0", model) == "MJD"


def test_unknown_param_without_pint_model():
    values = np.array([1.0, 2.0, 3.0], dtype=float)
    np.testing.assert_allclose(to_native("F0", values), values)
    np.testing.assert_allclose(to_display("F0", values), values)
    assert display_unit("F0") == "native"
    assert storage_unit("F0") is None


def test_pint_model_supplies_standard_parameter_units():
    model = _FakePintModel({"F0": _FakeParam(u.Hz), "PB": _FakeParam(u.day)})
    assert display_unit("F0", model) == "Hz"
    assert storage_unit("F0", model) == u.Hz
    assert display_unit("PB", model) == "MJD"


def test_native_physical_bounds():
    assert native_physical_bounds("ECC") == (0.0, 1.0)
    assert native_physical_bounds("SINI") == (0.0, 1.0)
    assert native_physical_bounds("KIN") == (0.0, 180.0)
    assert native_physical_bounds("M2") == (0.0, None)
    assert native_physical_bounds("A1") == (0.0, None)
    assert native_physical_bounds("J1640+2224_timing_M2") == (0.0, None)
    assert native_physical_bounds("RAJ") == (None, None)
    assert native_physical_bounds("F0") == (None, None)


def test_suffixed_metapulsar_names_use_canonical_units_and_bounds():
    model = _astro_model()
    assert display_unit("RAJ_ng5", model) == "hourangle"
    np.testing.assert_allclose(to_display("RAJ_ng5", 12.0, pint_model=model), 12.0)
    assert display_unit("DECJ_ng5", model) == "deg"
    assert native_physical_bounds("ECC_ng5") == (0.0, 1.0)
    assert native_physical_bounds("M2_ng5") == (0.0, None)


def test_prefixed_and_suffixed_names_prefer_timing_parameter_key():
    model = _astro_model()
    assert display_unit("J1640+2224_timing_RAJ_ng5", model) == "hourangle"
    assert native_physical_bounds("J1640+2224_timing_M2_ng5") == (0.0, None)
    assert normalize_param_name("J1640+2224_timing_RAJ_ng5") == "RAJ"
