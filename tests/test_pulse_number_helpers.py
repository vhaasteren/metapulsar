"""Tests for pulse-number mode helpers."""

from pathlib import Path
from unittest.mock import patch

import pytest

from metapulsar.pint_helpers import (
    PULSE_NUMBER_MODES,
    classify_tim_pulse_numbers,
    sanitize_tempo2_tim_noise_directives,
    validate_pulse_number_mode,
    _should_derive_pulse_numbers,
)


class TestValidatePulseNumberMode:
    def test_accepts_valid_modes(self):
        for mode in PULSE_NUMBER_MODES:
            assert validate_pulse_number_mode(mode) == mode

    def test_normalizes_case(self):
        assert validate_pulse_number_mode("YES") == "yes"

    def test_rejects_bool(self):
        with pytest.raises(ValueError, match="must be one of"):
            validate_pulse_number_mode(True)

    def test_rejects_unknown(self):
        with pytest.raises(ValueError, match="must be one of"):
            validate_pulse_number_mode("maybe")


class TestSanitizeTempo2TimNoiseDirectives:
    def test_strips_t2e_and_tne_lines(self):
        raw = (
            "FORMAT 1\n"
            "MODE 1\n"
            "T2EFAC -sys GM_GWB_500_100_b1 3.04\n"
            "T2EQUAD -sys foo 1.0\n"
            "TNEQUAD -sys bar 2.0\n"
            " obs1 1400.0 58000.0 1.0 g -pn 0\n"
        )
        out = sanitize_tempo2_tim_noise_directives(raw)
        assert "T2EFAC" not in out
        assert "T2EQUAD" not in out
        assert "TNEQUAD" not in out
        assert "FORMAT 1" in out
        assert "-pn 0" in out

    def test_preserves_comments(self):
        raw = "C comment\n# also\nMODE 1\n"
        assert sanitize_tempo2_tim_noise_directives(raw) == raw


class TestClassifyTimPulseNumbers:
    def test_complete_when_all_toas_have_pn(self, tmp_path):
        tim = tmp_path / "all_pn.tim"
        tim.write_text(
            "FORMAT 1\n"
            "MODE 1\n"
            " obs1 1400.0 58000.0 1.0 g -pn 0\n"
            " obs2 1400.0 58001.0 1.0 g -pn 1\n",
            encoding="utf-8",
        )
        status, n_with, n_without = classify_tim_pulse_numbers(tim)
        assert status == "complete"
        assert n_with == 2
        assert n_without == 0

    def test_none_when_no_pn(self, tmp_path):
        tim = tmp_path / "no_pn.tim"
        tim.write_text(
            "FORMAT 1\n"
            "MODE 1\n"
            " obs1 1400.0 58000.0 1.0 g\n"
            " obs2 1400.0 58001.0 1.0 g\n",
            encoding="utf-8",
        )
        status, n_with, n_without = classify_tim_pulse_numbers(tim)
        assert status == "none"
        assert n_with == 0
        assert n_without == 2

    def test_mixed_when_partial_pn(self, tmp_path):
        tim = tmp_path / "mixed_pn.tim"
        tim.write_text(
            "FORMAT 1\n"
            "MODE 1\n"
            " obs1 1400.0 58000.0 1.0 g -pn 0\n"
            " obs2 1400.0 58001.0 1.0 g\n",
            encoding="utf-8",
        )
        status, n_with, n_without = classify_tim_pulse_numbers(tim)
        assert status == "mixed"
        assert n_with == 1
        assert n_without == 1


class TestShouldDerivePulseNumbers:
    def test_no_never_derives(self):
        assert not _should_derive_pulse_numbers("no", "complete", Path("x.tim"), 2, 0)

    def test_overwrite_always_derives(self):
        assert _should_derive_pulse_numbers(
            "overwrite", "complete", Path("x.tim"), 2, 0
        )

    def test_yes_complete_no_derive(self):
        assert not _should_derive_pulse_numbers("yes", "complete", Path("x.tim"), 2, 0)

    def test_yes_none_derives_without_warning(self):
        with patch("metapulsar.pint_helpers.loguru_logger") as mock_log:
            assert _should_derive_pulse_numbers("yes", "none", Path("x.tim"), 0, 2)
            mock_log.warning.assert_not_called()

    def test_reuse_none_warns_and_derives(self):
        with patch("metapulsar.pint_helpers.loguru_logger") as mock_log:
            assert _should_derive_pulse_numbers("reuse", "none", Path("x.tim"), 0, 2)
            mock_log.warning.assert_called_once()

    def test_yes_mixed_warns_and_derives(self):
        with patch("metapulsar.pint_helpers.loguru_logger") as mock_log:
            assert _should_derive_pulse_numbers("yes", "mixed", Path("x.tim"), 1, 1)
            mock_log.warning.assert_called_once()


class TestResolvedTimReusesCompletePn:
    @staticmethod
    def _complete_pn_tim(tmp_path: Path) -> Path:
        tim = tmp_path / "complete_pn.tim"
        tim.write_text(
            "FORMAT 1\n"
            "MODE 1\n"
            " obs1 1400.0 58000.0 1.0 g -pn 0\n"
            " obs2 1400.0 58001.0 1.0 g -pn 1\n",
            encoding="utf-8",
        )
        return tim

    def test_reuse_complete_pn_skips_pint_derivation(self, tmp_path):
        from metapulsar.pint_helpers import resolved_tim_for_pulse_numbers

        tim_path = self._complete_pn_tim(tmp_path)
        par_text = "PSR J0000+0000\nF0 1.0\n"

        with (
            patch(
                "metapulsar.pint_helpers.temporary_pn_tim_from_par_tim_pint"
            ) as mock_pint,
            patch(
                "metapulsar.pint_helpers.temporary_pn_tim_from_par_tim_tempo2"
            ) as mock_tempo2,
        ):
            with resolved_tim_for_pulse_numbers(
                "reuse",
                par_text,
                tim_path,
                derive_backend="pint",
            ) as tim_out:
                assert tim_out == str(tim_path)
            mock_pint.assert_not_called()
            mock_tempo2.assert_not_called()

    def test_reuse_complete_pn_skips_tempo2_derivation(self, tmp_path):
        from metapulsar.pint_helpers import resolved_tim_for_pulse_numbers

        tim_path = self._complete_pn_tim(tmp_path)
        par_text = "PSR J0000+0000\nF0 1.0\n"

        with (
            patch(
                "metapulsar.pint_helpers.temporary_pn_tim_from_par_tim_pint"
            ) as mock_pint,
            patch(
                "metapulsar.pint_helpers.temporary_pn_tim_from_par_tim_tempo2"
            ) as mock_tempo2,
        ):
            with resolved_tim_for_pulse_numbers(
                "reuse",
                par_text,
                tim_path,
                derive_backend="tempo2",
            ) as tim_out:
                assert tim_out == str(tim_path)
            mock_pint.assert_not_called()
            mock_tempo2.assert_not_called()


class TestParTextWithTrackMinus2:
    def test_injects_track_without_reserializing_tempo2_fit_flags(self):
        from metapulsar.pint_helpers import par_text_with_track_minus_2

        par = "PSR J0000+0000\nDM 120 0 0 Y\nF0 300 0 0 N\n"
        out = par_text_with_track_minus_2(par)
        assert "TRACK -2" in out
        assert "DM 120 0 0 Y" in out
        assert "F0 300 0 0 N" in out

    def test_replaces_existing_track_line(self):
        from metapulsar.pint_helpers import par_text_with_track_minus_2

        par = "PSR J0000+0000\nTRACK 0\nF0 1\n"
        out = par_text_with_track_minus_2(par)
        assert "TRACK -2" in out
        assert "TRACK 0" not in out


class TestEnsurePintTrackMinus2:
    def test_warns_when_track_missing(self):
        from metapulsar.pint_helpers import ensure_pint_track_minus_2

        class ModelWithoutTrack:
            params = {}

        with patch("metapulsar.pint_helpers.loguru_logger") as mock_log:
            ensure_pint_track_minus_2(ModelWithoutTrack())  # type: ignore[arg-type]
            mock_log.warning.assert_called_once()


class TestTemporaryPnTimTempo2Sanitization:
    def test_sanitizes_plugin_output(self, tmp_path):
        from metapulsar.pint_helpers import temporary_pn_tim_from_par_tim_tempo2

        par_text = "PSR J0030+0451\nF0 205.5307\n"
        tim_path = tmp_path / "in.tim"
        tim_path.write_text("FORMAT 1\nMODE 1\n", encoding="utf-8")

        def fake_run(cmd, **kwargs):
            cwd = Path(kwargs["cwd"])
            (cwd / "withpn.tim").write_text(
                "FORMAT 1\nMODE 1\nT2EFAC -sys foo 1.0\n obs1 1400.0 58000.0 1.0 g -pn 0\n",
                encoding="utf-8",
            )

        with patch("metapulsar.pint_helpers.subprocess.run", side_effect=fake_run):
            with temporary_pn_tim_from_par_tim_tempo2(par_text, tim_path) as pn_tim:
                text = Path(pn_tim).read_text(encoding="utf-8")
        assert "T2EFAC" not in text
        assert "-pn 0" in text
