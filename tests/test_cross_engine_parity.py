"""Cross-engine residual parity for the two orthometric-Shapiro mechanisms.

MetaPulsar's mixed-engine ``consistent`` profile makes two claims that only a
real PINT-vs-Tempo2 residual comparison can check:

* PINT must evaluate Freire & Wex (2010) Eq. 28 (``ell1h_shapiro="absorbed"``)
  so an ELL1H/``T2`` ``H3``+``STIG`` model matches Tempo2's mode 1;
* a shared ``H3``+``H4`` par must carry ``NHARM`` (Tempo2) as well as ``NHARMS``
  (PINT), because Tempo2 otherwise falls back to ``nharm=4``.

Protocol: idealize TOAs with PINT (residuals ~0 by construction), write them,
then re-time with Tempo2 through libstempo and measure the residual RMS.

Two deliberate choices keep the measurement sensitive to the binary model only:

* TOAs are generated at the solar-system barycentre. Observatory TOAs carry a
  ~270 ns floor from the two packages' independent clock-correction chains,
  which is an external-realization difference this feature does not address.
* ``DM 0`` removes a dispersion-versus-binary delay-ordering difference that
  otherwise dominates at ~400 ns per light-second of ``A1`` per 10 pc/cm3.

What remains is a ~5 ns control floor, so these tests assert *relative* parity
against a matched non-Shapiro control. The absolute ~1 ns figures quoted in
``consistency-pint-tempo2.md`` come from the round-trip suite in
``paper/code/round-trip/parity_confirm/``, which uses real release TOAs.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from metapulsar.parameter_manager import AlignmentPolicy, ParameterManager

libstempo = pytest.importorskip("libstempo")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_libstempo,
    pytest.mark.skipif(
        shutil.which("tempo2") is None, reason="tempo2 binary not available"
    ),
]

#: RMS of the matched control (no orthometric Shapiro term) in ns; see module
#: docstring for what sets it.
CONTROL_FLOOR_NS = 15.0

COMMON_BODY = """PSR J1600-3053
EPHEM DE440
CLOCK TT(BIPM2021)
UNITS TDB
ELONG 244.347677 1
ELAT -10.071873 1
ECL IERS2003
POSEPOCH 55000
F0 277.937 1
F1 -7.3e-16 1
PEPOCH 55000
DM 0.0
DMEPOCH 55000
"""

KEPLERIAN = """PB 14.348466 1
A1 8.801653 1
TASC 55000.0 1
EPS1 1.7e-05 1
EPS2 -9.1e-06 1
"""

CONTROL_PAR = COMMON_BODY + "BINARY ELL1\n" + KEPLERIAN
H3_STIG_PAR = COMMON_BODY + "BINARY ELL1H\n" + KEPLERIAN + "H3 1.2e-07 1\nSTIG 0.72 1\n"
H3_H4_PAR = COMMON_BODY + "BINARY ELL1H\n" + KEPLERIAN + "H3 1.2e-07 1\nH4 6.0e-08 1\n"


def tempo2_rms_ns(par_path: Path, tmp_path: Path, ell1h_shapiro: str) -> float:
    """Idealize TOAs with PINT, then measure Tempo2's residual RMS in ns."""
    import astropy.units as u
    from pint.models import get_model
    from pint.simulation import make_fake_toas_uniform

    model = get_model(str(par_path), allow_T2=True, ell1h_shapiro=ell1h_shapiro)
    toas = make_fake_toas_uniform(
        54500,
        56500,
        400,
        model,
        freq=1400 * u.MHz,
        obs="ssb",
        add_noise=False,
    )
    tim_path = tmp_path / f"{par_path.stem}_{ell1h_shapiro}.tim"
    toas.write_TOA_file(str(tim_path), format="tempo2")

    psr = libstempo.tempopulsar(
        parfile=str(par_path), timfile=str(tim_path), dofit=False
    )
    residuals = np.asarray(
        psr.residuals(updatebats=True, formresiduals=True, removemean=True),
        dtype=float,
    )
    return float(np.std(residuals)) * 1e9


def write_par(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / f"{name}.par"
    path.write_text(text, encoding="utf-8")
    return path


def align_mixed(tmp_path: Path, par_text: str) -> tuple[Path, str]:
    """Run the mixed-engine shared strategy on one par.

    ``ne_sw=0`` pins the solar wind off: these TOAs sit at the barycentre, where
    the observatory-Sun geometry that both engines' solar-wind terms need is
    degenerate. The binary model is what this module measures.

    Returns the tempo2 PTA's written par and the manager's ELL1H convention.
    """
    manager = ParameterManager(
        file_data={
            "t2": {"timing_package": "tempo2", "par_content": par_text},
            "pint": {"timing_package": "pint", "par_content": par_text},
        },
        output_dir=tmp_path / "aligned",
        pulsar_name="J1600-3053",
        # binary_conversion="off": these tests assert the ELL1H absorbed
        # evaluator and dual NHARM/NHARMS alignment, which conversion to DDH
        # would short-circuit. P1–P4 below exercise the conversion path.
        alignment_policy=AlignmentPolicy(ne_sw=0.0, binary_conversion="off"),
    )
    written = manager.make_parfiles_shared()
    return Path(written["t2"]), manager.ell1h_shapiro


@pytest.fixture(scope="module")
def control_rms(tmp_path_factory) -> float:
    tmp_path = tmp_path_factory.mktemp("parity_control")
    par = write_par(tmp_path, "control", CONTROL_PAR)
    return tempo2_rms_ns(par, tmp_path, "absorbed")


def test_control_floor_is_small(control_rms):
    """Sanity check: without an orthometric term the two engines agree."""
    assert control_rms < CONTROL_FLOOR_NS


class TestOrthometricShapiroEvaluator:
    """Section 7.8: absorbed vs full for H3+STIG."""

    def test_absorbed_matches_tempo2(self, tmp_path, control_rms):
        par = write_par(tmp_path, "h3stig", H3_STIG_PAR)
        rms = tempo2_rms_ns(par, tmp_path, "absorbed")
        # The Shapiro term adds nothing above the control floor.
        assert rms == pytest.approx(control_rms, abs=1.0)

    def test_pint_default_full_disagrees_with_tempo2(self, tmp_path, control_rms):
        par = write_par(tmp_path, "h3stig", H3_STIG_PAR)
        rms = tempo2_rms_ns(par, tmp_path, "full")
        # Guard: the previous test would pass trivially if the evaluator choice
        # did not matter.
        assert rms > 20 * max(control_rms, 1.0)

    def test_mixed_engine_alignment_selects_absorbed(self, tmp_path, control_rms):
        aligned, mode = align_mixed(tmp_path, H3_STIG_PAR)
        assert mode == "absorbed"
        rms = tempo2_rms_ns(aligned, tmp_path, mode)
        assert rms == pytest.approx(control_rms, abs=1.0)


class TestOrthometricHarmonicCount:
    """Section 7.7: dual NHARM + NHARMS >= 7 for H3+H4."""

    def test_missing_harmonic_count_leaves_tempo2_at_its_default(
        self, tmp_path, control_rms
    ):
        par = write_par(tmp_path, "h3h4_bare", H3_H4_PAR)
        rms = tempo2_rms_ns(par, tmp_path, "absorbed")
        assert rms > 3 * max(control_rms, 1.0)

    def test_nharm_four_is_as_bad_as_omitting_it(self, tmp_path):
        bare = tempo2_rms_ns(
            write_par(tmp_path, "h3h4_bare", H3_H4_PAR), tmp_path, "absorbed"
        )
        four = tempo2_rms_ns(
            write_par(tmp_path, "h3h4_four", H3_H4_PAR + "NHARM 4\nNHARMS 4\n"),
            tmp_path,
            "absorbed",
        )
        assert four == pytest.approx(bare, rel=1e-3)

    def test_seven_harmonics_restores_parity(self, tmp_path, control_rms):
        par = write_par(tmp_path, "h3h4_seven", H3_H4_PAR + "NHARM 7\nNHARMS 7\n")
        rms = tempo2_rms_ns(par, tmp_path, "absorbed")
        assert rms == pytest.approx(control_rms, abs=1.0)

    def test_mixed_engine_alignment_restores_parity(self, tmp_path, control_rms):
        aligned, mode = align_mixed(tmp_path, H3_H4_PAR)
        assert "NHARM " in aligned.read_text(encoding="utf-8")
        rms = tempo2_rms_ns(aligned, tmp_path, mode)
        assert rms == pytest.approx(control_rms, abs=1.0)


# ---------------------------------------------------------------------------
# P1–P4: gated ELL1-family → DD/DDH conversion
# ---------------------------------------------------------------------------

J2145_PLAIN = """PSR J2145-0750
EPHEM DE440
CLOCK TT(BIPM2021)
UNITS TDB
RAJ 21:45:50.4
DECJ -07:50:18
F0 62.295 1
PEPOCH 55000
DM 0.0
DMEPOCH 55000
BINARY ELL1
PB 6.83890261 1 1e-10
A1 10.1641056 1 1e-8
TASC 55000.0 1 1e-6
EPS1 7.0e-6 1 1e-9
EPS2 -1.8e-5 1 1e-9
A1DOT 1e-14 0 1e-16
"""

J0610_SMALL = """PSR J0610-2100
EPHEM DE440
CLOCK TT(BIPM2021)
UNITS TDB
RAJ 06:10:00
DECJ -21:00:00
F0 200.0 1
PEPOCH 55000
DM 0.0
DMEPOCH 55000
BINARY ELL1
PB 0.3 1
A1 0.074 1
TASC 55000.0 1
EPS1 1.5e-5 1
EPS2 0.0 1
"""

J2145_H3_ONLY = """PSR J2145-0750
EPHEM DE440
CLOCK TT(BIPM2021)
UNITS TDB
RAJ 21:45:50.4
DECJ -07:50:18
F0 62.295 1
PEPOCH 55000
DM 0.0
DMEPOCH 55000
BINARY ELL1H
PB 6.83890261 1
A1 10.1641056 1
TASC 55000.0 1
EPS1 7.0e-6 1
EPS2 -1.8e-5 1
H3 1.8e-7 1
"""


def _mixed_manager(tmp_path, par_text, policy: AlignmentPolicy, name="psr"):
    return ParameterManager(
        file_data={
            "t2": {"timing_package": "tempo2", "par_content": par_text},
            "pint": {"timing_package": "pint", "par_content": par_text},
        },
        output_dir=tmp_path / f"aligned_{name}",
        pulsar_name=name,
        combine_components=["binary"],
        add_dm_derivatives=False,
        alignment_policy=policy,
    )


class TestBinaryFamilyConversionParity:
    """P1–P4: conversion path under mixed engines."""

    def test_p1_plain_ell1_converts_and_parity(self, tmp_path):
        policy = AlignmentPolicy(ne_sw=0.0)
        pm = _mixed_manager(tmp_path, J2145_PLAIN, policy, name="j2145_plain")
        written = pm.make_parfiles_shared()
        report = pm.last_binary_conversion_report
        assert report is not None
        assert report.decision.outcome == "convert"
        assert report.decision.target_family == "DD"
        text_t2 = Path(written["t2"]).read_text(encoding="utf-8")
        text_pint = Path(written["pint"]).read_text(encoding="utf-8")
        assert re.search(r"^BINARY\s+DD\b", text_t2, re.M)
        assert re.search(r"^BINARY\s+DD\b", text_pint, re.M)
        assert not re.search(r"^EPS1\b", text_t2, re.M)
        rms = tempo2_rms_ns(Path(written["t2"]), tmp_path, "full")
        assert rms <= 2.0

    def test_p2_small_scale_no_conversion(self, tmp_path, control_rms):
        policy = AlignmentPolicy(ne_sw=0.0)
        pm = _mixed_manager(tmp_path, J0610_SMALL, policy, name="j0610")
        written = pm.make_parfiles_shared()
        report = pm.last_binary_conversion_report
        assert report is not None
        assert report.decision.outcome == "skip"
        assert report.decision.reason == "below_threshold"
        text = Path(written["t2"]).read_text(encoding="utf-8")
        assert "ELL1" in text
        rms = tempo2_rms_ns(Path(written["t2"]), tmp_path, "absorbed")
        assert rms <= 2.0 or rms == pytest.approx(control_rms, abs=2.0)

    def test_p3_h3_only_default_errors_sample_stigma_builds(self, tmp_path):
        from metapulsar.binary_family_convert import BinaryConversionError

        policy_err = AlignmentPolicy(ne_sw=0.0)
        pm_err = _mixed_manager(tmp_path, J2145_H3_ONLY, policy_err, name="h3err")
        with pytest.raises(
            BinaryConversionError, match="ell1h_h3_only_underdetermined"
        ):
            pm_err.make_parfiles_shared()

        policy_ok = AlignmentPolicy(
            ne_sw=0.0,
            h3_only="sample_stigma",
            stigma_central=0.37,
            stigma_provenance="mass-function closure, m_p=1.4",
        )
        pm_ok = _mixed_manager(tmp_path, J2145_H3_ONLY, policy_ok, name="h3ok")
        written = pm_ok.make_parfiles_shared()
        report = pm_ok.last_binary_conversion_report
        assert report is not None and report.record is not None
        assert report.record.required_sampling == ("STIGMA",)
        text = Path(written["t2"]).read_text(encoding="utf-8")
        assert re.search(r"^BINARY\s+DDH\b", text, re.M)

        # Tempo2 reads the orthometric ratio as STIG only; DDHmodel.C *exits*
        # the process when it is unset. PINT accepts STIG, so both published
        # pars use the portable spelling.
        assert re.search(r"^STIG\s", text, re.M)
        assert not re.search(r"^(STIGMA|VARSIGMA)\s", text, re.M)
        text_pint = Path(written["pint"]).read_text(encoding="utf-8")
        assert re.search(r"^STIG\s", text_pint, re.M)
        assert not re.search(r"^(STIGMA|VARSIGMA)\s", text_pint, re.M)

        # The converted DDH par must actually load and evaluate in tempo2 —
        # a text-only assertion is what let the STIGMA/STIG defect through.
        rms = tempo2_rms_ns(Path(written["t2"]), tmp_path, "full")
        assert rms <= 2.0

        # Exclude binary from combine_components → no error
        pm_excl = ParameterManager(
            file_data={
                "t2": {"timing_package": "tempo2", "par_content": J2145_H3_ONLY},
                "pint": {"timing_package": "pint", "par_content": J2145_H3_ONLY},
            },
            output_dir=tmp_path / "aligned_excl",
            pulsar_name="h3excl",
            combine_components=["astrometry"],
            add_dm_derivatives=False,
            alignment_policy=AlignmentPolicy(ne_sw=0.0),
        )
        pm_excl.make_parfiles_shared()
        assert (
            pm_excl.last_binary_conversion_report.decision.reason == "binary_not_shared"
        )

    def test_p5_h3_stig_fixture_converts_to_ddh_under_default_policy(self, tmp_path):
        """The fixture the other ELL1H tests pin `binary_conversion="off"` for.

        Its two-term gate value is ~7 ns, so under the *default* policy it
        converts — this is the case those tests stop exercising, so it gets its
        own end-to-end tempo2 check here.
        """
        pm = _mixed_manager(
            tmp_path, H3_STIG_PAR, AlignmentPolicy(ne_sw=0.0), name="h3stig_conv"
        )
        written = pm.make_parfiles_shared()
        report = pm.last_binary_conversion_report
        assert report is not None and report.record is not None
        assert report.decision.outcome == "convert"
        assert report.decision.target_family == "DDH"
        assert report.record.gauge == "absorbed"
        assert report.record.required_sampling == ()  # STIGMA was measured
        assert report.decision.scale is not None
        assert report.decision.scale.scale_s > 1e-9

        text = Path(written["t2"]).read_text(encoding="utf-8")
        assert re.search(r"^BINARY\s+DDH\b", text, re.M)
        assert re.search(r"^STIG\s", text, re.M)
        for gone in ("EPS1", "EPS2", "TASC", "NHARM", "NHARMS", "H4"):
            assert not re.search(rf"^{gone}\s", text, re.M)
        rms = tempo2_rms_ns(Path(written["t2"]), tmp_path, "full")
        assert rms <= 2.0

    def test_p4_existing_orthometric_tests_still_green(self, tmp_path, control_rms):
        """P4: absorbed evaluator + NHARM path remain intact with conversion off."""
        aligned, mode = align_mixed(tmp_path, H3_STIG_PAR)
        assert mode == "absorbed"
        rms = tempo2_rms_ns(aligned, tmp_path, mode)
        assert rms == pytest.approx(control_rms, abs=1.0)
