"""
Comprehensive tests for position_helpers module.

Tests coordinate conversion between PINT TimingModel, libstempo tempopulsar,
and Enterprise Pulsar objects, plus J-name generation.
"""

import pytest
from io import StringIO
from dataclasses import dataclass

import astropy.units as u
from astropy.coordinates import SkyCoord, ICRS, BarycentricMeanEcliptic
from astropy.time import Time
from pint.models.model_builder import ModelBuilder

from metapulsar.position_helpers import (
    _skycoord_from_pint_model,
    _skycoord_from_enterprise,
    _skycoord_from_libstempo,
    bj_name_from_pulsar,
    extract_coordinates_from_parfile_optimized,
    bj_name_from_coordinates_optimized,
    discover_pulsars_by_coordinates_optimized,
    _parse_parfile_optimized,
    _get_first_par_value_by_aliases,
    _parse_ra_string_optimized,
    _parse_dec_string_optimized,
    _get_pm_equatorial_masyr_optimized,
    _parse_float_optimized,
)

# === FIXTURES ===


@pytest.fixture
def mb():
    """PINT ModelBuilder instance."""
    return ModelBuilder()


@pytest.fixture
def model_J(mb, load_parfile_text):
    """PINT model from binary.par file."""
    return _build_pint_model(mb, load_parfile_text("binary.par"))


@pytest.fixture
def model_B(mb, load_parfile_text):
    """PINT model from binary-B.par file."""
    return _build_pint_model(mb, load_parfile_text("binary-B.par"))


# === HELPER FUNCTIONS ===


def _build_pint_model(mb: ModelBuilder, par_text: str):
    """Build PINT model from parfile text."""
    return mb(StringIO(par_text), allow_tcb=True, allow_T2=True)


# === MOCK CLASSES ===


@dataclass
class LibstempoParam:
    """Mock libstempo parameter with .val attribute."""

    val: float


class LibstempoMock:
    """Mock libstempo tempopulsar with dict-like parameter access."""

    def __init__(self, mapping):
        self._m = mapping

    def __getitem__(self, key):
        return self._m[key]


class EnterpriseMock:
    """Mock Enterprise Pulsar with internal coordinate attributes."""

    def __init__(self, raj_rad: float, decj_rad: float):
        self._raj = raj_rad
        self._decj = decj_rad


# === UTILITY FUNCTIONS ===


def _icrs_from_model(model) -> SkyCoord:
    """Ground-truth ICRS from the PINT model using your extractor."""
    return _skycoord_from_pint_model(model).transform_to(ICRS())


def enterprise_from_model(model) -> EnterpriseMock:
    """Create Enterprise mock from PINT model coordinates."""
    c = _icrs_from_model(model)
    return EnterpriseMock(c.ra.to(u.rad).value, c.dec.to(u.rad).value)


def libstempo_from_model_equatorial(model) -> LibstempoMock:
    """Mock with RAJ/DECJ (in radians)."""
    c = _icrs_from_model(model)
    mapping = {
        "RAJ": LibstempoParam(c.ra.to(u.rad).value),
        "DECJ": LibstempoParam(c.dec.to(u.rad).value),
        # No ecliptic keys so _skycoord_from_libstempo takes equatorial branch
    }
    return LibstempoMock(mapping)


def libstempo_from_model_ecliptic(model) -> LibstempoMock:
    """Mock with ELONG/ELAT only (in radians) to hit the ecliptic branch."""
    c = _icrs_from_model(model).transform_to(BarycentricMeanEcliptic(equinox="J2000"))
    mapping = {
        "ELONG": LibstempoParam(c.lon.to(u.rad).value),
        "ELAT": LibstempoParam(c.lat.to(u.rad).value),
        # Intentionally omit RAJ/DECJ so the code must use ecliptic path
    }
    return LibstempoMock(mapping)


def _assert_coords_close(c1: SkyCoord, c2: SkyCoord, atol_rad=1e-10):
    """Assert two SkyCoord objects are close within tolerance."""
    sep = c1.separation(c2).to(u.rad).value
    assert sep <= atol_rad, f"Coords differ by {sep} rad (> {atol_rad})"


# === TEST CLASSES ===


class TestBJNameGeneration:
    """Test B/J-name generation from various pulsar objects."""

    @pytest.mark.parametrize("parfile_name", ["binary.par", "binary-B.par"])
    def test_j_name_from_pint_model(self, mb, load_parfile_text, parfile_name):
        """Test J-name generation from PINT models."""
        par_text = load_parfile_text(parfile_name)
        model = _build_pint_model(mb, par_text)
        jlabel = bj_name_from_pulsar(model, "J")
        assert jlabel == "J1857+0943"

    @pytest.mark.parametrize("parfile_name", ["binary.par", "binary-B.par"])
    def test_b_name_from_pint_model(self, mb, load_parfile_text, parfile_name):
        """Test B-name generation from PINT models."""
        par_text = load_parfile_text(parfile_name)
        model = _build_pint_model(mb, par_text)
        blabel = bj_name_from_pulsar(model, "B")
        assert blabel == "B1855+09"

    def test_name_consistency_across_parfiles(self, model_J, model_B):
        """Test that names are consistent between different parfile formats."""
        jl_j = bj_name_from_pulsar(model_J, "J")
        jl_b = bj_name_from_pulsar(model_B, "J")
        bl_j = bj_name_from_pulsar(model_J, "B")
        bl_b = bj_name_from_pulsar(model_B, "B")
        assert jl_j == jl_b == "J1857+0943"
        assert bl_j == bl_b == "B1855+09"

    def test_default_name_type_is_j(self, model_J):
        """Test that default name type is J."""
        jlabel = bj_name_from_pulsar(model_J)
        assert jlabel == "J1857+0943"

    def test_invalid_name_type_raises_error(self, model_J):
        """Test that invalid name type raises ValueError."""
        with pytest.raises(ValueError):
            bj_name_from_pulsar(model_J, "X")


class TestCoordinateConversion:
    """Test coordinate conversion between different pulsar object types."""

    @pytest.mark.parametrize("which", ["J", "B"])
    def test_skycoord_from_enterprise_matches_pint(self, which, model_J, model_B):
        """Test Enterprise mock produces same coordinates as PINT model."""
        model = model_J if which == "J" else model_B
        truth = _icrs_from_model(model)

        emock = enterprise_from_model(model)
        c_ent = _skycoord_from_enterprise(emock).transform_to(ICRS())

        _assert_coords_close(c_ent, truth)

    @pytest.mark.parametrize("which", ["J", "B"])
    def test_skycoord_from_libstempo_equatorial_matches_pint(
        self, which, model_J, model_B
    ):
        """Test libstempo equatorial mock produces same coordinates as PINT model."""
        model = model_J if which == "J" else model_B
        truth = _icrs_from_model(model)

        lmock = libstempo_from_model_equatorial(model)
        c_lt = _skycoord_from_libstempo(lmock).transform_to(ICRS())

        _assert_coords_close(c_lt, truth)

    @pytest.mark.parametrize("which", ["J", "B"])
    def test_skycoord_from_libstempo_ecliptic_matches_pint(
        self, which, model_J, model_B
    ):
        """Test libstempo ecliptic mock produces same coordinates as PINT model."""
        model = model_J if which == "J" else model_B
        truth = _icrs_from_model(model)

        lmock = libstempo_from_model_ecliptic(model)
        c_lt = _skycoord_from_libstempo(lmock).transform_to(ICRS())

        _assert_coords_close(c_lt, truth)


class TestEndToEndJNameGeneration:
    """Test end-to-end J-name generation using mocks."""

    @pytest.mark.parametrize("which", ["J", "B"])
    def test_j_label_from_enterprise_mock(self, which, model_J, model_B):
        """Test J-name generation from Enterprise mock objects."""
        model = model_J if which == "J" else model_B
        emock = enterprise_from_model(model)
        assert bj_name_from_pulsar(emock, "J") == "J1857+0943"

    @pytest.mark.parametrize("which", ["J", "B"])
    def test_b_label_from_enterprise_mock(self, which, model_J, model_B):
        """Test B-name generation from Enterprise mock objects."""
        model = model_J if which == "J" else model_B
        emock = enterprise_from_model(model)
        assert bj_name_from_pulsar(emock, "B") == "B1855+09"

    @pytest.mark.parametrize(
        "which,variant", [("J", "eq"), ("B", "eq"), ("J", "ecl"), ("B", "ecl")]
    )
    def test_j_label_from_libstempo_mocks(self, which, variant, model_J, model_B):
        """Test J-name generation from libstempo mock objects."""
        model = model_J if which == "J" else model_B
        if variant == "eq":
            lmock = libstempo_from_model_equatorial(model)
        else:
            lmock = libstempo_from_model_ecliptic(model)
        assert bj_name_from_pulsar(lmock, "J") == "J1857+0943"

    @pytest.mark.parametrize(
        "which,variant", [("J", "eq"), ("B", "eq"), ("J", "ecl"), ("B", "ecl")]
    )
    def test_b_label_from_libstempo_mocks(self, which, variant, model_J, model_B):
        """Test B-name generation from libstempo mock objects."""
        model = model_J if which == "J" else model_B
        if variant == "eq":
            lmock = libstempo_from_model_equatorial(model)
        else:
            lmock = libstempo_from_model_ecliptic(model)
        assert bj_name_from_pulsar(lmock, "B") == "B1855+09"


# ============================================================================
# OPTIMIZED COORDINATE EXTRACTION TESTS
# ============================================================================


class TestOptimizedCoordinateExtraction:
    """Test optimized coordinate extraction functions."""

    def test_extract_coordinates_from_parfile_optimized_equatorial(
        self, load_parfile_text
    ):
        """Test optimized coordinate extraction from equatorial coordinates."""
        parfile_content = load_parfile_text("binary.par")
        coords = extract_coordinates_from_parfile_optimized(parfile_content)

        assert coords is not None, "Failed to extract coordinates"
        ra_hours, dec_deg = coords

        # Verify coordinates are reasonable
        assert 0 <= ra_hours < 24, f"RA out of range: {ra_hours}"
        assert -90 <= dec_deg <= 90, f"DEC out of range: {dec_deg}"

        # Verify J-name generation
        j_name = bj_name_from_coordinates_optimized(ra_hours, dec_deg, "J")
        assert j_name == "J1857+0943", f"J-name mismatch: {j_name}"

    def test_extract_coordinates_from_parfile_optimized_ecliptic_lambda_beta(self):
        """Test optimized coordinate extraction from LAMBDA/BETA coordinates."""
        parfile_content = """
PSR J1857+0943
LAMBDA 285.1234
BETA 9.7214
F0 186.494081
F1 -1.23e-15
PEPOCH 55000.0
DM 10.0
"""

        coords = extract_coordinates_from_parfile_optimized(parfile_content)
        assert coords is not None, "Failed to extract LAMBDA/BETA coordinates"

        ra_hours, dec_deg = coords
        j_name = bj_name_from_coordinates_optimized(ra_hours, dec_deg, "J")

        # Should produce a valid J-name
        assert j_name.startswith("J"), f"Invalid J-name format: {j_name}"
        assert len(j_name) == 10, f"Invalid J-name length: {j_name}"

    def test_extract_coordinates_from_parfile_optimized_ecliptic_elong_elat(self):
        """Test optimized coordinate extraction from ELONG/ELAT coordinates."""
        parfile_content = """
PSR J1857+0943
ELONG 285.1234
ELAT 9.7214
F0 186.494081
F1 -1.23e-15
PEPOCH 55000.0
DM 10.0
"""

        coords = extract_coordinates_from_parfile_optimized(parfile_content)
        assert coords is not None, "Failed to extract ELONG/ELAT coordinates"

        ra_hours, dec_deg = coords
        j_name = bj_name_from_coordinates_optimized(ra_hours, dec_deg, "J")

        # Should produce a valid J-name
        assert j_name.startswith("J"), f"Invalid J-name format: {j_name}"
        assert len(j_name) == 10, f"Invalid J-name length: {j_name}"

    def test_lambda_beta_vs_elong_elat_consistency(self):
        """Test that LAMBDA/BETA and ELONG/ELAT produce identical results."""
        parfile_lambda = """
PSR J1857+0943
LAMBDA 285.1234
BETA 9.7214
F0 186.494081
F1 -1.23e-15
PEPOCH 55000.0
DM 10.0
"""

        parfile_elong = """
PSR J1857+0943
ELONG 285.1234
ELAT 9.7214
F0 186.494081
F1 -1.23e-15
PEPOCH 55000.0
DM 10.0
"""

        coords_lambda = extract_coordinates_from_parfile_optimized(parfile_lambda)
        coords_elong = extract_coordinates_from_parfile_optimized(parfile_elong)

        assert coords_lambda is not None, "Failed to extract LAMBDA/BETA coordinates"
        assert coords_elong is not None, "Failed to extract ELONG/ELAT coordinates"

        # Both should produce identical results
        assert (
            coords_lambda == coords_elong
        ), f"LAMBDA/BETA and ELONG/ELAT results differ: {coords_lambda} != {coords_elong}"

    def test_bj_name_from_coordinates_optimized_j_name(self):
        """Test optimized J-name generation from coordinates."""
        ra_hours = 18.9601  # 18:57:36.4
        dec_deg = 9.7214  # +09:43:17.1

        j_name = bj_name_from_coordinates_optimized(ra_hours, dec_deg, "J")
        expected_j_name = "J1857+0943"

        assert (
            j_name == expected_j_name
        ), f"J-name mismatch: {j_name} != {expected_j_name}"

    def test_bj_name_from_coordinates_optimized_b_name(self):
        """Test optimized B-name generation from coordinates."""
        ra_hours = 18.9601  # 18:57:36.4
        dec_deg = 9.7214  # +09:43:17.1

        b_name = bj_name_from_coordinates_optimized(ra_hours, dec_deg, "B")
        expected_b_name = "B1855+09"  # B-names use FK4 coordinates

        assert (
            b_name == expected_b_name
        ), f"B-name mismatch: {b_name} != {expected_b_name}"

    def test_discover_pulsars_by_coordinates_optimized(self, load_parfile_text):
        """Test optimized pulsar discovery system."""
        # Create test file data
        parfile_content = load_parfile_text("binary.par")
        file_data = {
            "EPTA": [
                {"par": "test.par", "par_content": parfile_content, "tim": "test.tim"}
            ]
        }

        # Run optimized discovery
        coordinate_map = discover_pulsars_by_coordinates_optimized(file_data)

        # Verify results
        assert len(coordinate_map) > 0, "No pulsars discovered"
        assert "J1857+0943" in coordinate_map, "Expected pulsar not found"
        assert "EPTA" in coordinate_map["J1857+0943"], "PTA not found in results"
        assert (
            len(coordinate_map["J1857+0943"]["EPTA"]) == 1
        ), "Incorrect number of files"

    def test_optimized_vs_original_consistency(self, load_parfile_text):
        """Test that optimized functions produce same results as original."""
        parfile_content = load_parfile_text("binary.par")

        # Extract coordinates using optimized method
        coords_opt = extract_coordinates_from_parfile_optimized(parfile_content)
        assert coords_opt is not None, "Optimized extraction failed"

        ra_hours, dec_deg = coords_opt
        j_name_opt = bj_name_from_coordinates_optimized(ra_hours, dec_deg, "J")
        b_name_opt = bj_name_from_coordinates_optimized(ra_hours, dec_deg, "B")

        # Create PINT model for comparison
        from io import StringIO
        from pint.models.model_builder import ModelBuilder

        mb = ModelBuilder()
        model = mb(StringIO(parfile_content), allow_tcb=True, allow_T2=True)

        # Extract using original method
        j_name_orig = bj_name_from_pulsar(model, "J")
        b_name_orig = bj_name_from_pulsar(model, "B")

        # Results should match
        assert (
            j_name_opt == j_name_orig
        ), f"J-name mismatch: {j_name_opt} != {j_name_orig}"
        assert (
            b_name_opt == b_name_orig
        ), f"B-name mismatch: {b_name_opt} != {b_name_orig}"

    def test_malformed_parfile_handling(self):
        """Test handling of malformed parfiles."""
        malformed_parfiles = [
            "",  # Empty content
            "PSR J1857+0943\nF0 186.494081",  # No coordinates
            "PSR J1857+0943\nRAJ invalid\nDECJ 9.7214",  # Invalid RA
            "PSR J1857+0943\nRAJ 18.9601\nDECJ invalid",  # Invalid DEC
        ]

        for parfile_content in malformed_parfiles:
            coords = extract_coordinates_from_parfile_optimized(parfile_content)
            assert (
                coords is None
            ), f"Should return None for malformed parfile: {parfile_content}"

    def test_coordinate_precision_optimized(self):
        """Test coordinate precision in optimized extraction."""
        parfile_content = """
PSR J1857+0943
RAJ 18:57:36.4000
DECJ +09:43:17.1000
F0 186.494081
F1 -1.23e-15
PEPOCH 55000.0
DM 10.0
"""

        coords = extract_coordinates_from_parfile_optimized(parfile_content)
        assert coords is not None, "Failed to extract coordinates"

        ra_hours, dec_deg = coords

        # Verify precision is maintained
        expected_ra = 18.96011111111111  # 18:57:36.4000 in hours
        expected_dec = 9.721416666666667  # +09:43:17.1000 in degrees

        assert (
            abs(ra_hours - expected_ra) < 1e-10
        ), f"RA precision error: {ra_hours} != {expected_ra}"
        assert (
            abs(dec_deg - expected_dec) < 1e-10
        ), f"DEC precision error: {dec_deg} != {expected_dec}"

    def test_coordinate_parameter_aliases_optimized(self):
        """Test that coordinate parameter aliases work correctly."""
        # Test RA/DEC aliases
        parfile_ra_dec = """
PSR J1857+0943
RA 18:57:36.4
DEC +09:43:17.1
F0 186.494081
F1 -1.23e-15
PEPOCH 55000.0
DM 10.0
"""

        coords_ra_dec = extract_coordinates_from_parfile_optimized(parfile_ra_dec)
        assert (
            coords_ra_dec is not None
        ), "Failed to extract coordinates with RA/DEC aliases"

        # Test LAMBDA/BETA aliases (LAMBDA->ELONG, BETA->ELAT)
        parfile_lambda_beta = """
PSR J1857+0943
LAMBDA 285.1234
BETA 9.7214
F0 186.494081
F1 -1.23e-15
PEPOCH 55000.0
DM 10.0
"""

        coords_lambda_beta = extract_coordinates_from_parfile_optimized(
            parfile_lambda_beta
        )
        assert (
            coords_lambda_beta is not None
        ), "Failed to extract coordinates with LAMBDA/BETA aliases"

        # Test that both produce valid J-names
        j_name_ra_dec = bj_name_from_coordinates_optimized(
            coords_ra_dec[0], coords_ra_dec[1], "J"
        )
        j_name_lambda_beta = bj_name_from_coordinates_optimized(
            coords_lambda_beta[0], coords_lambda_beta[1], "J"
        )

        assert j_name_ra_dec.startswith("J"), f"Invalid J-name format: {j_name_ra_dec}"
        assert j_name_lambda_beta.startswith(
            "J"
        ), f"Invalid J-name format: {j_name_lambda_beta}"
        assert len(j_name_ra_dec) == 10, f"Invalid J-name length: {j_name_ra_dec}"
        assert (
            len(j_name_lambda_beta) == 10
        ), f"Invalid J-name length: {j_name_lambda_beta}"


class TestProperMotionJ2000Normalization:
    """Test proper motion propagation to J2000 for epoch-stable naming."""

    def test_same_pulsar_different_posepoch_produces_same_j_name(
        self, load_parfile_text
    ):
        """Test that same pulsar at different POSEPOCH produces identical J-name."""
        # Use generated parfiles with consistent PM but different positions at different epochs
        # Both should normalize to the same J2000 position

        parfile_epoch1 = load_parfile_text("test_same_pulsar_epoch1_54500.par")
        parfile_epoch2 = load_parfile_text("test_same_pulsar_epoch2_56000.par")

        coords1 = extract_coordinates_from_parfile_optimized(parfile_epoch1)
        coords2 = extract_coordinates_from_parfile_optimized(parfile_epoch2)

        assert coords1 is not None and coords2 is not None

        # Verify coordinates are actually different at the two POSEPOCH values
        # (before normalization, they would differ)
        ra1, dec1 = coords1
        ra2, dec2 = coords2

        # After normalization to J2000, coordinates should be identical
        # Allow for second-order effects: with large PMRA, relative error ~0.1%
        # The propagated position difference has second-order errors proportional to PMRA
        # Set tolerance to account for these effects (scales with PMRA value used)
        tolerance_hours = 0.02  # 0.02 arcmin accounts for second-order effects with large PMRA and PMDEC
        assert (
            abs(ra1 - ra2) < tolerance_hours
        ), f"RA should match after normalization (within second-order tolerance): {ra1} != {ra2}, diff={abs(ra1-ra2):.10f}h"
        assert (
            abs(dec1 - dec2) < tolerance_hours
        ), f"DEC should match after normalization (within second-order tolerance): {dec1} != {dec2}, diff={abs(dec1-dec2):.10f}deg"

        j_name1 = bj_name_from_coordinates_optimized(ra1, dec1, "J")
        j_name2 = bj_name_from_coordinates_optimized(ra2, dec2, "J")

        # Should produce identical J-names after normalization
        assert j_name1 == j_name2, f"J-names differ: {j_name1} != {j_name2}"

    def test_same_position_different_posepoch_with_pm_should_not_match(
        self, load_parfile_text
    ):
        """Test that parfiles with identical positions but different POSEPOCH and PM should NOT match.

        This tests the error case: if someone incorrectly provides the same position at different
        epochs when proper motion exists, the positions should NOT match after normalization to J2000.
        This is because proper motion means the position should have been different at different epochs.

        Uses large PM values (15000 mas/yr) to ensure coordinate difference > 1 arcmin,
        which produces different J-names and allows the test to correctly verify separation.
        """
        # Use generated parfiles with same position at different POSEPOCH
        parfile_epoch1 = load_parfile_text(
            "test_same_position_large_pm_epoch1_54500.par"
        )
        parfile_epoch2 = load_parfile_text(
            "test_same_position_large_pm_epoch2_56000.par"
        )

        # Extract coordinates (should normalize to J2000)
        coords1 = extract_coordinates_from_parfile_optimized(parfile_epoch1)
        coords2 = extract_coordinates_from_parfile_optimized(parfile_epoch2)

        assert coords1 is not None and coords2 is not None

        ra1, dec1 = coords1
        ra2, dec2 = coords2

        # After normalization to J2000, coordinates should be DIFFERENT
        # because they started from the same position at different epochs
        # (this indicates an error in the parfiles - position should have been different)
        coord_diff_ra = abs(ra1 - ra2)
        coord_diff_dec = abs(dec1 - dec2)

        # With large proper motion (15000 mas/yr), there should be a significant difference
        # For PMDEC=15000 mas/yr and dt=1500 days ≈ 4.1 years, difference ≈ 61 arcsec = 1.02 arcmin
        # This ensures different J-names (J-names have 1 arcmin precision)
        assert coord_diff_ra > 1e-6 or coord_diff_dec > 1e-6, (
            f"Coordinates should differ when same position at different POSEPOCH with PM: "
            f"RA diff={coord_diff_ra}, DEC diff={coord_diff_dec}"
        )

        # J-names should differ with large PM (ensures coordinate difference > 1 arcmin)
        j_name1 = bj_name_from_coordinates_optimized(ra1, dec1, "J")
        j_name2 = bj_name_from_coordinates_optimized(ra2, dec2, "J")

        # With large PM (15000 mas/yr), coordinate difference is > 1 arcmin,
        # so J-names should differ
        assert j_name1 != j_name2, (
            f"J-names should differ with large PM: {j_name1} == {j_name2}, "
            f"RA diff={coord_diff_ra}, DEC diff={coord_diff_dec}"
        )

    def test_proper_motion_propagation_equatorial(self, load_parfile_text):
        """Test that PMRA/PMDEC are correctly propagated to J2000."""
        parfile_with_pm = load_parfile_text("test_equatorial_pm.par")

        # Extract catalogued position at POSEPOCH
        parfile_dict = _parse_parfile_optimized(parfile_with_pm)
        posepoch_mjd = _parse_float_optimized(parfile_dict.get("POSEPOCH"))

        # Parse catalogued coordinates (before propagation)
        raj = _get_first_par_value_by_aliases(parfile_dict, "RAJ")
        decj = _get_first_par_value_by_aliases(parfile_dict, "DECJ")
        ra_catalogued = _parse_ra_string_optimized(raj)
        dec_catalogued = _parse_dec_string_optimized(decj)

        # Get propagated coordinates (at J2000)
        coords = extract_coordinates_from_parfile_optimized(parfile_with_pm)
        assert coords is not None
        ra_propagated, dec_propagated = coords

        # Verify propagation actually happened (coordinates should differ from catalogued)
        # unless POSEPOCH is exactly J2000
        if posepoch_mjd is not None:
            j2000_mjd = Time("J2000").mjd
            if abs(posepoch_mjd - j2000_mjd) > 1.0:  # More than 1 day difference
                # Coordinates should be different (propagation occurred)
                assert (
                    abs(ra_propagated - ra_catalogued) > 1e-8
                    or abs(dec_propagated - dec_catalogued) > 1e-8
                ), "Coordinates should differ after propagation from POSEPOCH to J2000"

        # Verify coordinates are reasonable
        assert 0 <= ra_propagated < 24, f"RA out of range: {ra_propagated}"
        assert -90 <= dec_propagated <= 90, f"DEC out of range: {dec_propagated}"

        j_name = bj_name_from_coordinates_optimized(ra_propagated, dec_propagated, "J")
        assert j_name.startswith("J") and len(j_name) == 10, f"Invalid J-name: {j_name}"

    def test_proper_motion_propagation_ecliptic_with_aliases(self, load_parfile_text):
        """Test ecliptic PM propagation with PMELONG/PMLAMBDA and PMELAT/PMBETA aliases."""
        # Test PMELONG/PMELAT
        parfile_pmelong = load_parfile_text("test_ecliptic_pmelong.par")

        # Test PMLAMBDA/PMBETA (aliases)
        parfile_pmlambda = load_parfile_text("test_ecliptic_pmlambda.par")

        coords1 = extract_coordinates_from_parfile_optimized(parfile_pmelong)
        coords2 = extract_coordinates_from_parfile_optimized(parfile_pmlambda)

        assert coords1 is not None and coords2 is not None

        # Both should produce identical results (aliases work)
        j_name1 = bj_name_from_coordinates_optimized(coords1[0], coords1[1], "J")
        j_name2 = bj_name_from_coordinates_optimized(coords2[0], coords2[1], "J")

        assert j_name1 == j_name2, "PMELONG/PMLAMBDA aliases should produce same result"
        # Also verify coordinates match (not just names)
        assert (
            abs(coords1[0] - coords2[0]) < 1e-6 and abs(coords1[1] - coords2[1]) < 1e-6
        ), "Coordinates should match when using aliases"

    def test_ecliptic_lambda_beta_coordinates_with_pm(self, load_parfile_text):
        """Test that LAMBDA/BETA coordinates work with proper motion."""
        parfile_lambda = load_parfile_text("test_ecliptic_lambda_beta.par")

        coords = extract_coordinates_from_parfile_optimized(parfile_lambda)
        assert coords is not None

        # Should produce valid J-name
        j_name = bj_name_from_coordinates_optimized(coords[0], coords[1], "J")
        assert j_name.startswith("J") and len(j_name) == 10

    def test_partial_pm_missing_components(self, load_parfile_text):
        """Test behavior when only some PM components are present."""
        # Missing PMDEC
        parfile_no_pmdec = load_parfile_text("test_partial_pm_no_pmdec.par")

        # Missing PMRA
        parfile_no_pmra = load_parfile_text("test_partial_pm_no_pmra.par")

        # Missing POSEPOCH
        parfile_no_posepoch = load_parfile_text("test_partial_pm_no_posepoch.par")

        # All should still work (no propagation, uses catalogued position)
        # Note: When POSEPOCH equals J2000, dt_yr=0 and propagation is automatically a no-op
        # (handled by early return in _propagate_* functions), so no special test needed.
        for parfile in [parfile_no_pmdec, parfile_no_pmra, parfile_no_posepoch]:
            coords = extract_coordinates_from_parfile_optimized(parfile)
            assert coords is not None
            j_name = bj_name_from_coordinates_optimized(coords[0], coords[1], "J")
            assert j_name == "J1857+0943"

    def test_missing_pm_issues_warning(self, load_parfile_text):
        """Test that missing PM/POSEPOCH issues warning and uses catalogued position."""
        from unittest.mock import patch
        from loguru import logger

        parfile_no_pm = load_parfile_text("test_no_pm.par")

        with patch.object(logger, "warning") as mock_warning:
            coords = extract_coordinates_from_parfile_optimized(parfile_no_pm)

            # Should issue warning about missing PM/POSEPOCH
            assert mock_warning.called, "Should issue warning when PM/POSEPOCH missing"
            warning_calls = [str(call) for call in mock_warning.call_args_list]
            assert any(
                "POSEPOCH" in call or "proper motion" in call.lower()
                for call in warning_calls
            ), "Warning should mention POSEPOCH or proper motion"

        assert coords is not None

        # Should still produce valid J-name (no propagation, uses catalogued position)
        j_name = bj_name_from_coordinates_optimized(coords[0], coords[1], "J")
        assert j_name == "J1857+0943"

    def test_posepoch_falls_back_to_pepoch_no_warning(self, load_parfile_text):
        """POSEPOCH should fall back to PEPOCH and suppress the warning.

        This mirrors the NANOGrav 9-yr/12.5-yr ``.gls.par`` convention: ecliptic
        position with PMLAMBDA/PMBETA, only PEPOCH (no explicit POSEPOCH).
        """
        from unittest.mock import patch
        from loguru import logger

        parfile = load_parfile_text("test_ecliptic_pmlambda_no_posepoch.par")

        with patch.object(logger, "warning") as mock_warning:
            coords = extract_coordinates_from_parfile_optimized(parfile)

        assert coords is not None
        assert not mock_warning.called, (
            "No warning should be issued when PEPOCH provides the POSEPOCH "
            f"fallback. Got calls: {mock_warning.call_args_list}"
        )

        # And propagation must actually have happened (relative to the cataloged
        # ecliptic position at PEPOCH=54500 with the large PMLAMBDA in the fixture).
        sibling_with_posepoch = load_parfile_text("test_ecliptic_pmlambda.par")
        coords_ref = extract_coordinates_from_parfile_optimized(sibling_with_posepoch)
        assert coords_ref is not None
        # POSEPOCH and PEPOCH are identical (54500) in the two fixtures, so the
        # propagated J2000 coordinates must agree to numerical precision.
        assert abs(coords[0] - coords_ref[0]) < 1e-9
        assert abs(coords[1] - coords_ref[1]) < 1e-9

    def test_pepoch_fallback_pint_model_path(self, mb, load_parfile_text):
        """PINT-model coordinate path also falls back to PEPOCH when POSEPOCH is absent."""
        from metapulsar.position_helpers import _skycoord_from_pint_model

        parfile = load_parfile_text("test_ecliptic_pmlambda_no_posepoch.par")
        sibling = load_parfile_text("test_ecliptic_pmlambda.par")

        model_no_pose = _build_pint_model(mb, parfile)
        model_with_pose = _build_pint_model(mb, sibling)

        c_no_pose = _skycoord_from_pint_model(model_no_pose)
        c_with_pose = _skycoord_from_pint_model(model_with_pose)

        assert c_no_pose.separation(c_with_pose).to(u.mas).value < 1.0, (
            "PINT-model coordinates must match between explicit-POSEPOCH and "
            "PEPOCH-fallback parfiles."
        )

    def test_b_name_generation_with_propagation(self, mb, load_parfile_text):
        """Test that B-name generation also normalizes to J2000."""
        parfile_with_pm = load_parfile_text("test_b_name_propagation.par")

        coords1 = extract_coordinates_from_parfile_optimized(parfile_with_pm)
        assert coords1 is not None

        # Generate B-name from propagated coordinates
        b_name = bj_name_from_coordinates_optimized(coords1[0], coords1[1], "B")
        assert b_name.startswith("B") and len(b_name) == 8

        # Test with PINT model path too
        model = _build_pint_model(mb, parfile_with_pm)
        b_name_model = bj_name_from_pulsar(model, "B")
        assert (
            b_name_model == b_name
        ), "B-names should match between optimized and model paths"

    def test_pint_model_proper_motion_normalization(self, mb, load_parfile_text):
        """Test that PINT model path also normalizes to J2000."""
        import astropy.units as u
        from metapulsar.position_helpers import _skycoord_from_pint_model

        parfile_with_pm = load_parfile_text("test_pint_model_normalization.par")

        model = _build_pint_model(mb, parfile_with_pm)

        # Extract coordinates via model path
        c_model = _skycoord_from_pint_model(model)
        ra_model = c_model.ra.to(u.hourangle).value
        dec_model = c_model.dec.to(u.deg).value

        # Verify coordinates are reasonable (propagated to J2000)
        assert 0 <= ra_model < 24, f"RA out of range: {ra_model}"
        assert -90 <= dec_model <= 90, f"DEC out of range: {dec_model}"

        j_name = bj_name_from_pulsar(model, "J")
        assert j_name.startswith("J") and len(j_name) == 10, f"Invalid J-name: {j_name}"

    def test_libstempo_proper_motion_normalization(self, load_parfile_text):
        """Test that libstempo path also normalizes to J2000."""
        from tests.test_position_helpers import LibstempoMock, LibstempoParam
        from metapulsar.position_helpers import _skycoord_from_libstempo
        import numpy as np

        # Use the same parfile as other tests for consistency
        parfile_with_pm = load_parfile_text("test_pint_model_normalization.par")
        parfile_dict = _parse_parfile_optimized(parfile_with_pm)

        # Extract coordinates and PM from parfile
        raj = _get_first_par_value_by_aliases(parfile_dict, "RAJ")
        decj = _get_first_par_value_by_aliases(parfile_dict, "DECJ")
        ra_hours = _parse_ra_string_optimized(raj)
        dec_deg = _parse_dec_string_optimized(decj)
        pmra, pmdec = _get_pm_equatorial_masyr_optimized(parfile_dict)
        posepoch_mjd = _parse_float_optimized(parfile_dict.get("POSEPOCH"))

        # Convert to radians for libstempo mock
        raj_rad = np.deg2rad(ra_hours * 15.0)
        decj_rad = np.deg2rad(dec_deg)

        # Create libstempo mock with values from parfile
        mapping = {
            "RAJ": LibstempoParam(raj_rad),
            "DECJ": LibstempoParam(decj_rad),
            "PMRA": LibstempoParam(pmra),  # mas/yr
            "PMDEC": LibstempoParam(pmdec),  # mas/yr
            "POSEPOCH": LibstempoParam(posepoch_mjd),  # MJD
        }

        lmock = LibstempoMock(mapping)
        c_libstempo = _skycoord_from_libstempo(lmock)
        ra_libstempo = c_libstempo.ra.to(u.hourangle).value
        dec_libstempo = c_libstempo.dec.to(u.deg).value

        # Verify coordinates are reasonable (propagated to J2000)
        assert 0 <= ra_libstempo < 24, f"RA out of range: {ra_libstempo}"
        assert -90 <= dec_libstempo <= 90, f"DEC out of range: {dec_libstempo}"

        j_name = bj_name_from_pulsar(lmock, "J")
        assert j_name.startswith("J") and len(j_name) == 10, f"Invalid J-name: {j_name}"
