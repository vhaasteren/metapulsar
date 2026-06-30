"""Slice-1 tests for timing units conversion helpers."""

import numpy as np

from metapulsar.timing.units import (
    display_unit,
    native_physical_bounds,
    to_display,
    to_native,
)


def test_raj_hourangle_roundtrip():
    display_hours = np.array([0.0, 6.0, 12.0, 18.5], dtype=float)
    native = to_native("RAJ", display_hours)
    roundtrip = to_display("RAJ", native)
    np.testing.assert_allclose(roundtrip, display_hours)
    assert display_unit("RAJ") == "hourangle"


def test_decj_degree_roundtrip():
    display_deg = np.array([-30.0, -1.5, 0.0, 12.25, 80.0], dtype=float)
    native = to_native("DECJ", display_deg)
    roundtrip = to_display("DECJ", native)
    np.testing.assert_allclose(roundtrip, display_deg)
    assert display_unit("DECJ") == "deg"


def test_t0_identity_conversion():
    mjd = np.array([53000.125, 54000.0], dtype=float)
    native = to_native("T0", mjd)
    roundtrip = to_display("T0", native)
    np.testing.assert_allclose(roundtrip, mjd)
    assert display_unit("T0") == "MJD"


def test_unknown_param_passthrough():
    values = np.array([1.0, 2.0, 3.0], dtype=float)
    np.testing.assert_allclose(to_native("F0", values), values)
    np.testing.assert_allclose(to_display("F0", values), values)
    assert display_unit("F0") == "native"


def test_native_physical_bounds():
    assert native_physical_bounds("ECC") == (0.0, 1.0)
    assert native_physical_bounds("SINI") == (0.0, 1.0)
    assert native_physical_bounds("KIN") == (0.0, 180.0)
    assert native_physical_bounds("M2") == (0.0, None)
    assert native_physical_bounds("A1") == (0.0, None)
    # Qualified names resolve to the trailing canonical token.
    assert native_physical_bounds("J1640+2224_timing_M2") == (0.0, None)
    # Unbounded parameters report no constraints.
    assert native_physical_bounds("RAJ") == (None, None)
    assert native_physical_bounds("F0") == (None, None)


def test_suffixed_metapulsar_names_use_canonical_units_and_bounds():
    assert display_unit("RAJ_ng5") == "hourangle"
    np.testing.assert_allclose(to_display("RAJ_ng5", np.pi), 12.0)
    assert display_unit("DECJ_ng5") == "deg"
    assert native_physical_bounds("ECC_ng5") == (0.0, 1.0)
    assert native_physical_bounds("M2_ng5") == (0.0, None)


def test_prefixed_and_suffixed_names_prefer_timing_parameter_key():
    assert display_unit("J1640+2224_timing_RAJ_ng5") == "hourangle"
    assert native_physical_bounds("J1640+2224_timing_M2_ng5") == (0.0, None)
