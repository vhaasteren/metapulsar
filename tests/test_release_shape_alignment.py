"""End-to-end alignment over representative PTA release shapes.

Each case runs ``ParameterManager.make_parfiles_consistent()`` on compact
checked-in par files (``tests/fixtures/release_shapes/``) and checks that the
written par lands on the agreed common profile and still materializes through
its declared timing engine.
"""

from __future__ import annotations

import shutil
from io import StringIO
from pathlib import Path

import pytest
from pint.models.model_builder import parse_parfile

from metapulsar.parameter_manager import AlignmentPolicy, ParameterManager
from metapulsar.pint_helpers import create_pint_model

pytestmark = pytest.mark.slow

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "release_shapes"

TEMPO2_AVAILABLE = shutil.which("tempo2") is not None
needs_tempo2 = pytest.mark.skipif(
    not TEMPO2_AVAILABLE, reason="tempo2 binary not available"
)


def read_shape(name: str) -> str:
    return (FIXTURE_DIR / f"{name}.par").read_text(encoding="utf-8")


def build_file_data(*shapes: tuple[str, str, str]) -> dict:
    """Build ParameterManager file data from (pta_name, fixture, engine) triples."""
    return {
        pta_name: {
            "par": FIXTURE_DIR / f"{fixture}.par",
            "tim": FIXTURE_DIR / f"{fixture}.tim",
            "timing_package": engine,
            "par_content": read_shape(fixture),
        }
        for pta_name, fixture, engine in shapes
    }


def align(file_data: dict, tmp_path: Path, policy: AlignmentPolicy | None = None):
    """Run the consistent strategy and return {pta: parsed par dict}."""
    manager = ParameterManager(
        file_data=file_data,
        output_dir=tmp_path,
        pulsar_name="J1600-3053",
        alignment_policy=policy,
    )
    written = manager.make_parfiles_consistent()
    # parse_parfile returns a defaultdict(list); make it a plain dict so a
    # missing-key lookup in a test cannot silently create an empty entry.
    return {
        pta_name: dict(parse_parfile(StringIO(Path(path).read_text(encoding="utf-8"))))
        for pta_name, path in written.items()
    }, written


def first(value):
    return str(value[0]).split()[0]


def tempo2_roundtrip(par_path: Path, out_path: Path) -> str:
    """Round-trip a written par through tempo2 and return the result text."""
    import subprocess

    result = subprocess.run(
        ["tempo2", "-gr", "transform", str(par_path), str(out_path), "tdb"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return out_path.read_text(encoding="utf-8")


class TestMixedEngineReleaseShapes:
    """NANOGrav (PINT) + EPTA (Tempo2): the full section 4.1 profile."""

    @pytest.fixture
    def aligned(self, tmp_path):
        file_data = build_file_data(
            ("nanograv", "nanograv_style", "pint"),
            ("epta", "epta_style", "tempo2"),
        )
        return align(file_data, tmp_path)

    @needs_tempo2
    def test_every_global_convention_is_explicit(self, aligned):
        parsed, _ = aligned
        for pta_name, par in parsed.items():
            assert first(par["UNITS"]) == "TDB", pta_name
            assert first(par["T2CMETHOD"]) == "IAU2000B", pta_name
            assert first(par["TIMEEPH"]) == "FB90", pta_name
            assert first(par["DILATEFREQ"]) == "N", pta_name
            assert first(par["CORRECT_TROPOSPHERE"]) == "N", pta_name
            assert first(par["PLANET_SHAPIRO"]) == "N", pta_name
            assert first(par["SWM"]) == "0", pta_name
            assert "NO_SS_SHAPIRO" not in par, pta_name

    @needs_tempo2
    def test_reference_ephemeris_and_clock_win(self, aligned):
        parsed, _ = aligned
        for pta_name, par in parsed.items():
            assert first(par["EPHEM"]) == "DE436", pta_name
            clock = par.get("CLOCK") or par["CLK"]
            assert first(clock) == "TT(BIPM2017)", pta_name

    @needs_tempo2
    def test_ipm_is_a_tempo2_only_control(self, aligned):
        parsed, _ = aligned
        assert first(parsed["epta"]["IPM"]) == "1"
        assert "IPM" not in parsed["nanograv"]

    @needs_tempo2
    def test_solar_wind_amplitude_is_explicit_and_canonical(self, aligned):
        parsed, _ = aligned
        for pta_name, par in parsed.items():
            assert "SOLARN0" not in par, pta_name
            assert "NE1AU" not in par, pta_name
            # The reference PTA (NANOGrav) declares SOLARN0 0, which is explicit
            # and therefore beats tempo2's implicit 4 cm^-3.
            assert float(first(par["NE_SW"])) == pytest.approx(0.0)

    @needs_tempo2
    def test_ecliptic_frames_are_transformed_to_iers2003(self, aligned):
        parsed, _ = aligned
        for pta_name, par in parsed.items():
            assert first(par["ECL"]) == "IERS2003", pta_name
            assert "LAMBDA" not in par, pta_name
            assert "BETA" not in par, pta_name
            assert "ELONG" in par and "ELAT" in par, pta_name

    @needs_tempo2
    def test_merged_binary_without_h4_carries_no_harmonic_count(self, aligned):
        parsed, _ = aligned
        # The reference (NANOGrav) is a plain ELL1, so the binary merge replaces
        # EPTA's H3+H4 model; a leftover harmonic count would be stale.
        for pta_name, par in parsed.items():
            assert first(par["BINARY"]) == "ELL1", pta_name
            assert "H4" not in par, pta_name
            assert "NHARM" not in par, pta_name
            assert "NHARMS" not in par, pta_name

    @needs_tempo2
    def test_dispersion_cleanup_is_unchanged(self, aligned):
        parsed, _ = aligned
        for pta_name, par in parsed.items():
            assert not any(key.startswith("DMX") for key in par), pta_name
            assert par["DM1"] == ["0.0 1"], pta_name
            assert par["DM2"] == ["0.0 1"], pta_name
        # DM stays PTA-local by default.
        assert float(first(parsed["nanograv"]["DM"])) == pytest.approx(52.3)

    @needs_tempo2
    def test_detector_local_terms_survive(self, aligned):
        parsed, _ = aligned
        nanograv = parsed["nanograv"]
        assert nanograv["FD1"]
        assert nanograv["FD2"]
        assert len(nanograv["JUMP"]) == 2
        assert nanograv["TZRMJD"] and nanograv["TZRSITE"] and nanograv["TZRFRQ"]

        epta = parsed["epta"]
        assert epta["JUMP"]
        assert epta["FDDC"] and epta["FDDI"]

    @needs_tempo2
    def test_written_pars_load_in_pint(self, aligned):
        _, written = aligned
        for pta_name, path in written.items():
            model = create_pint_model(
                Path(path).read_text(encoding="utf-8"), ell1h_shapiro="absorbed"
            )
            assert model.PSR.value == "J1600-3053", pta_name

    @needs_tempo2
    def test_written_tempo2_par_is_accepted_by_tempo2(self, aligned, tmp_path):
        _, written = aligned
        roundtrip = tempo2_roundtrip(written["epta"], tmp_path / "roundtrip.par")
        assert "J1600-3053" in roundtrip


class TestAggregateAndUnsupportedShapes:
    """TEMPO1 expansion, incidental DMMODEL, and development PINT extensions."""

    @needs_tempo2
    def test_tempo1_and_dmmodel_shape(self, tmp_path):
        file_data = build_file_data(
            ("nanograv", "nanograv_style", "pint"),
            ("ppta", "ppta_style", "tempo2"),
        )
        parsed, _ = align(file_data, tmp_path)

        ppta = parsed["ppta"]
        # TEMPO1 is expanded, then normalized by the mixed-engine profile.
        assert "TEMPO1" not in ppta
        assert first(ppta["T2CMETHOD"]) == "IAU2000B"
        assert first(ppta["TIMEEPH"]) == "FB90"
        # DMMODEL and its grid are stripped; no basis conversion is attempted.
        assert "DMMODEL" not in ppta
        assert "_DM" not in ppta
        assert "CONSTRAIN" not in ppta
        assert first(ppta["UNITS"]) == "TDB"

    def test_development_pint_extensions_are_stripped(self, tmp_path):
        file_data = build_file_data(
            ("mpta", "mpta_style", "pint"),
            ("epta_pint", "pint_only_a", "tempo2"),
        )
        parsed, _ = align(file_data, tmp_path)

        mpta = parsed["mpta"]
        for key in (
            "SWP",
            "SWEPOCH",
            "NE_SW1",
            "SWXDM_0001",
            "SWXR1_0001",
            "SWXR2_0001",
            "DMWXEPOCH",
            "DMWXFREQ_0001",
            "DMWXSIN_0001",
            "DMWXCOS_0001",
            "WXEPOCH",
            "WXFREQ_0001",
            "WXSIN_0001",
            "WXCOS_0001",
            "CM",
            "CMEPOCH",
            "CMX_0001",
        ):
            assert key not in mpta, f"{key} survived"
        assert first(mpta["SWM"]) == "0"
        # Ordinary red-noise settings are untouched.
        assert mpta["TNREDAMP"] == ["-14.2"]
        assert mpta["TNREDGAM"] == ["3.1"]

    def test_error_policy_rejects_the_development_shape(self, tmp_path):
        file_data = build_file_data(
            ("mpta", "mpta_style", "pint"),
            ("epta_pint", "pint_only_a", "tempo2"),
        )
        with pytest.raises(ValueError, match="unsupported timing parameters"):
            align(file_data, tmp_path, AlignmentPolicy(unsupported="error"))


class TestPintOnlyReleaseShape:
    """A PINT-only multi-PTA stack keeps the thinner profile (section 4.2)."""

    @pytest.fixture
    def aligned(self, tmp_path):
        file_data = build_file_data(
            ("pta_a", "pint_only_a", "pint"),
            ("pta_b", "pint_only_b", "pint"),
        )
        return align(file_data, tmp_path)

    def test_troposphere_and_planet_shapiro_are_retained(self, aligned):
        parsed, _ = aligned
        for pta_name, par in parsed.items():
            assert first(par["CORRECT_TROPOSPHERE"]) == "Y", pta_name
            assert first(par["PLANET_SHAPIRO"]) == "Y", pta_name

    def test_mixed_engine_switches_are_not_written(self, aligned):
        parsed, _ = aligned
        for pta_name, par in parsed.items():
            assert "T2CMETHOD" not in par, pta_name
            assert "TIMEEPH" not in par, pta_name
            assert "IPM" not in par, pta_name

    def test_heterogeneous_ecl_is_aligned_to_the_reference(self, aligned):
        parsed, _ = aligned
        for pta_name, par in parsed.items():
            assert first(par["ECL"]) == "IERS2010", pta_name

    def test_no_solar_wind_line_is_invented(self, aligned):
        parsed, _ = aligned
        for pta_name, par in parsed.items():
            assert "NE_SW" not in par, pta_name


class TestBinaryShapes:
    """Binary families pass through their declared engine unchanged in family."""

    @needs_tempo2
    def test_ell1h_h3_stig_keeps_no_harmonic_count(self, tmp_path):
        file_data = build_file_data(
            ("pint_pta", "binary_ell1h_h3stig", "pint"),
            ("t2_pta", "epta_style", "tempo2"),
        )
        parsed, _ = align(file_data, tmp_path)

        # The reference PTA's binary values are copied across, so both pars end
        # up on H3+STIG; neither may gain a harmonic count.
        for pta_name, par in parsed.items():
            assert "STIG" in par or "STIGMA" in par, pta_name
            assert "NHARM" not in par, pta_name
            assert "NHARMS" not in par, pta_name

    @needs_tempo2
    def test_binary_family_names_are_not_rewritten(self, tmp_path):
        file_data = build_file_data(
            ("pint_pta", "binary_ell1h_h3stig", "pint"),
            ("t2_pta", "epta_style", "tempo2"),
        )
        parsed, _ = align(file_data, tmp_path)
        for pta_name, par in parsed.items():
            assert first(par["BINARY"]) == "ELL1H", pta_name

    @needs_tempo2
    def test_h3_h4_reference_gets_both_harmonic_spellings(self, tmp_path):
        # EPTA is the reference here, so its T2 + H3 + H4 orthometric model is
        # what every PTA ends up linearizing around.
        file_data = build_file_data(
            ("epta", "epta_style", "tempo2"),
            ("nanograv", "nanograv_style", "pint"),
        )
        parsed, written = align(file_data, tmp_path)

        for pta_name, par in parsed.items():
            assert "H4" in par, pta_name
            assert "STIG" not in par and "STIGMA" not in par, pta_name
            assert int(first(par["NHARM"])) == 7, pta_name
            assert int(first(par["NHARMS"])) == 7, pta_name

        # Tempo2 only reads NHARM, so it must survive into the on-disk par.
        roundtrip = tempo2_roundtrip(written["epta"], tmp_path / "roundtrip.par")
        assert "NHARM" in roundtrip

    @needs_tempo2
    def test_existing_harmonic_count_above_the_floor_is_preserved(self, tmp_path):
        raised = read_shape("epta_style").replace(
            "H4               6.0e-08 1",
            "H4               6.0e-08 1\nNHARM            9",
        )
        file_data = build_file_data(
            ("epta", "epta_style", "tempo2"),
            ("nanograv", "nanograv_style", "pint"),
        )
        file_data["epta"]["par_content"] = raised
        parsed, _ = align(file_data, tmp_path)

        for pta_name, par in parsed.items():
            assert int(first(par["NHARM"])) == 9, pta_name
            assert int(first(par["NHARMS"])) == 9, pta_name

    @needs_tempo2
    def test_ddk_kom_follows_the_ecliptic_transformation(self, tmp_path):
        file_data = build_file_data(
            ("t2_pta", "binary_ddk", "tempo2"),
            ("pint_pta", "nanograv_style", "pint"),
        )
        parsed, _ = align(file_data, tmp_path)

        ddk = parsed["t2_pta"]
        assert first(ddk["ECL"]) == "IERS2003"
        assert "KOM" in ddk
        assert float(first(ddk["KOM"])) == pytest.approx(76.0, abs=1e-2)
