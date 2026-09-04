"""Tests for FileDiscovery."""

import pytest
from pathlib import Path
from unittest.mock import patch
from metapulsar.file_discovery import (
    FileDiscovery,
    PTA_DATA_RELEASES,
    AmbiguousFileError,
    FileSelectionError,
    MissingOverrideError,
    _normalize_precedence,
    select_release_file,
)
from metapulsar import discover_files
from metapulsar.layout_discovery import discover_layout
from tests.helpers import make_tim_metadata


def _with_catalog_aliases(file_entry, catalog_names, path_name=None):
    """Attach identity fields expected by build_alias_map in filter tests."""
    enriched = dict(file_entry)
    enriched["catalog_names"] = list(catalog_names)
    enriched["path_name"] = path_name or catalog_names[0]
    return enriched


def _write_release(root, name, files):
    """Create a release tree; ``files`` maps release-relative path -> text."""
    for relative, text in files.items():
        path = root / name / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root / name


def _sole_selection(path):
    """Provenance dict for a mocked single-candidate file pair."""
    return {
        "chosen": path,
        "candidates": [path],
        "reason": "sole",
        "rule": None,
    }


class TestFileDiscovery:
    """Test FileDiscovery functionality."""

    def test_init_default_configs(self):
        """Test initialization with default configurations."""
        service = FileDiscovery()
        assert service.data_releases == PTA_DATA_RELEASES

    def test_init_custom_configs(self):
        """Test initialization with custom configurations."""
        custom_data_releases = {
            "test_pta": {
                "base_dir": "/test/path",
                "par_pattern": r"test_(\w+)\.par",
                "tim_pattern": r"test_(\w+)\.tim",
                "timing_package": "pint",
            }
        }
        service = FileDiscovery(pta_data_releases=custom_data_releases)
        assert service.data_releases == custom_data_releases

    def test_tim_metadata_failure_propagates(self, tmp_path):
        service = FileDiscovery()
        tim_path = tmp_path / "broken.tim"
        sentinel = RuntimeError("broken TIM tree")
        with patch.object(
            service._tim_analyzer, "get_tim_metadata", side_effect=sentinel
        ):
            with pytest.raises(RuntimeError, match="broken TIM tree") as exc_info:
                service._get_tim_metadata(tim_path)
        assert exc_info.value is sentinel

    def test_discover_patterns_in_data_release_success(self):
        """Test discovering patterns in a single data release."""
        service = FileDiscovery()

        with patch.object(
            service, "_discover_patterns_in_data_release"
        ) as mock_discover:
            mock_discover.return_value = ["J1857+0943", "B1855+09"]

            result = service.discover_patterns_in_data_release("epta_dr2")

            assert result == ["J1857+0943", "B1855+09"]
            mock_discover.assert_called_once()

    def test_discover_patterns_in_data_release_not_found(self):
        """Test discovering patterns with non-existent data release."""
        service = FileDiscovery()

        with pytest.raises(KeyError, match="Data release 'nonexistent' not found"):
            service.discover_patterns_in_data_release("nonexistent")

    def test_discover_patterns_in_data_releases_success(self):
        """Test discovering patterns in multiple data releases."""
        service = FileDiscovery()

        with patch.object(
            service, "discover_patterns_in_data_release"
        ) as mock_discover:
            mock_discover.side_effect = [["J1857+0943"], ["J1857+0943", "B1855+09"]]

            result = service.discover_patterns_in_data_releases(
                ["epta_dr2", "ppta_dr2"]
            )

            assert result == {
                "epta_dr2": ["J1857+0943"],
                "ppta_dr2": ["J1857+0943", "B1855+09"],
            }

    def test_discover_files_success(self):
        """Test discovering files in data releases."""
        service = FileDiscovery()

        with patch.object(
            service, "_discover_all_file_pairs_in_data_release"
        ) as mock_discover:
            mock_discover.return_value = [
                {
                    "par": Path("/test/J1857+0943.par"),
                    "tim": Path("/test/J1857+0943.tim"),
                    "timing_package": "tempo2",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    "par_content": "PSR J1857+0943\nRAJ 18:57:36.4\nDECJ 09:43:17.1\n",
                    "par_selection": _sole_selection(Path("/test/J1857+0943.par")),
                    "tim_selection": _sole_selection(Path("/test/J1857+0943.tim")),
                }
            ]

            result = service.discover_files(["epta_dr2"])

            assert "epta_dr2" in result
            assert len(result["epta_dr2"]) == 1
            assert result["epta_dr2"][0]["par"] == Path("/test/J1857+0943.par")
            assert result["epta_dr2"][0]["tim"] == Path("/test/J1857+0943.tim")
            assert result["epta_dr2"][0]["timing_package"] == "tempo2"

    def test_discover_files_all_data_releases(self):
        """Test discovering files in all data releases when no specific data releases provided."""
        service = FileDiscovery()

        with patch.object(service, "list_data_releases") as mock_list:
            mock_list.return_value = ["epta_dr2", "ppta_dr2"]

            with patch.object(
                service, "_discover_all_file_pairs_in_data_release"
            ) as mock_discover:
                mock_discover.return_value = []

                result = service.discover_files()

                assert "epta_dr2" in result
                assert "ppta_dr2" in result

    def test_discover_files_single_string_input(self):
        """Test discovering files with single string input."""
        service = FileDiscovery()

        with patch.object(
            service, "_discover_all_file_pairs_in_data_release"
        ) as mock_discover:
            mock_discover.return_value = [
                {
                    "par": Path("/test/J1857+0943.par"),
                    "tim": Path("/test/J1857+0943.tim"),
                    "timing_package": "tempo2",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    "par_selection": _sole_selection(Path("/test/J1857+0943.par")),
                    "tim_selection": _sole_selection(Path("/test/J1857+0943.tim")),
                }
            ]

            result = service.discover_files("epta_dr2")

            assert "epta_dr2" in result
            assert len(result["epta_dr2"]) == 1

    def test_discover_files_verbose_output(self, capsys):
        """Test verbose output of discover_files method."""
        service = FileDiscovery()

        with patch.object(
            service, "_discover_all_files_in_data_releases"
        ) as mock_discover:
            mock_discover.return_value = {
                "epta_dr2": [
                    {
                        "par": Path("test1.par"),
                        "tim": Path("test1.tim"),
                        "timing_package": "tempo2",
                        "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                        "par_selection": _sole_selection(Path("test1.par")),
                        "tim_selection": _sole_selection(Path("test1.tim")),
                    }
                ],
                "ppta_dr2": [],
            }

            service.discover_files(["epta_dr2", "ppta_dr2"], verbose=True)

            captured = capsys.readouterr()
            assert "Found:" in captured.out
            assert "- epta_dr2: 1 pulsars" in captured.out
            assert "(No pulsars for: ppta_dr2)" in captured.out

    def test_list_data_releases_alphabetical(self):
        """Test listing data releases sorted alphabetically."""
        service = FileDiscovery()

        result = service.list_data_releases()

        # Should be sorted alphabetically
        assert isinstance(result, list)
        assert len(result) > 0

    def test_add_data_release_success(self):
        """Test adding a new data release configuration."""
        service = FileDiscovery()

        new_config = {
            "base_dir": "/test/path",
            "par_pattern": r"test_(\w+)\.par",
            "tim_pattern": r"test_(\w+)\.tim",
            "timing_package": "pint",
        }

        service.add_data_release("test_data_release", new_config)

        assert "test_data_release" in service.data_releases
        assert service.data_releases["test_data_release"] == new_config

    def test_add_data_release_duplicate(self):
        """Test adding duplicate data release configuration."""
        service = FileDiscovery()

        with pytest.raises(ValueError, match="Data release 'epta_dr2' already exists"):
            service.add_data_release("epta_dr2", {})

    def test_add_data_release_invalid_config(self):
        """Test adding data release with invalid configuration."""
        service = FileDiscovery()

        invalid_config = {
            "base_dir": "/test/path",
            # Missing required keys
        }

        with pytest.raises(ValueError, match="Missing required keys"):
            service.add_data_release("test_data_release", invalid_config)

    def test_flat_dr3_layouts_select_the_native_pair(self, tmp_path):
        """MPTA DR2 and PPTA DR3 are flat releases with look-alike neighbours.

        PPTA DR3 ships a derived ``<PSR>_pint.par`` beside the Tempo2 solution
        and keeps working subdirectories that repeat the pulsar names, and both
        releases carry editor and tooling leftovers (``.par~``, ``.tim.bak``).
        Only the pulsar's own pair directly under the release root may match.
        """
        mpta = tmp_path / "MPTA_DR2"
        mpta.mkdir()
        for name in ("J1909-3744.par", "J1909-3744.tim", "J1909-3744.tim.bak"):
            (mpta / name).write_text("PSR J1909-3744\n", encoding="utf-8")

        ppta = tmp_path / "PPTA_DR3"
        (ppta / "uwl").mkdir(parents=True)
        (ppta / "J1909-3744.par").write_text("PSR J1909-3744\n", encoding="utf-8")
        (ppta / "J1909-3744.tim").write_text("FORMAT 1\n", encoding="utf-8")
        (ppta / "J1909-3744_pint.par").write_text("PSR J1909-3744\n", encoding="utf-8")
        (ppta / "J1909-3744.par~").write_text("PSR J1909-3744\n", encoding="utf-8")
        (ppta / "uwl" / "J1909-3744.par").write_text(
            "PSR J1909-3744\n", encoding="utf-8"
        )
        (ppta / "uwl" / "J1909-3744.tim").write_text("FORMAT 1\n", encoding="utf-8")
        # Globular-cluster suffix (e.g. J1824-2452A); still only the root pair.
        (ppta / "J1824-2452A.par").write_text("PSR J1824-2452A\n", encoding="utf-8")
        (ppta / "J1824-2452A.tim").write_text("FORMAT 1\n", encoding="utf-8")
        (ppta / "J1824-2452A_pint.par").write_text(
            "PSR J1824-2452A\n", encoding="utf-8"
        )
        (ppta / "uwl" / "J1824-2452A.par").write_text(
            "PSR J1824-2452A\n", encoding="utf-8"
        )
        (ppta / "uwl" / "J1824-2452A.tim").write_text("FORMAT 1\n", encoding="utf-8")

        service = FileDiscovery(working_dir=str(tmp_path), verbose=False)
        found = service.discover_files(["mpta_dr2", "ppta_dr3"])

        assert [e["par"] for e in found["mpta_dr2"]] == [mpta / "J1909-3744.par"]
        assert [e["tim"] for e in found["mpta_dr2"]] == [mpta / "J1909-3744.tim"]
        assert [e["par"] for e in found["ppta_dr3"]] == [
            ppta / "J1824-2452A.par",
            ppta / "J1909-3744.par",
        ]
        assert [e["tim"] for e in found["ppta_dr3"]] == [
            ppta / "J1824-2452A.tim",
            ppta / "J1909-3744.tim",
        ]
        assert all(
            entry["timing_package"] == "tempo2"
            for entries in found.values()
            for entry in entries
        )

    def test_inpta_dr2_layout_selects_dmx_par_and_all_tim(self, tmp_path):
        """InPTA DR2 publishes ``<PSR>.DMX.par`` plus ``<PSR>_all.tim``.

        The release also ships a top-level ``DM12.par/`` tree and per-backend
        ``tims/*.tim`` includes; discovery must ignore those neighbours.
        """
        release = tmp_path / "InPTA.DR2"
        psr_dir = release / "J1713+0747"
        (psr_dir / "tims").mkdir(parents=True)
        (psr_dir / "J1713+0747.DMX.par").write_text(
            "PSRJ J1713+0747\n", encoding="utf-8"
        )
        (psr_dir / "J1713+0747_all.tim").write_text(
            "FORMAT 1\nINCLUDE tims/GM_GWB_1460_100.0_b1_pre36.tim\n",
            encoding="utf-8",
        )
        (psr_dir / "tims" / "GM_GWB_1460_100.0_b1_pre36.tim").write_text(
            "FORMAT 1\n", encoding="utf-8"
        )
        dm12 = release / "DM12.par"
        dm12.mkdir()
        (dm12 / "J1713+0747.nonesw.par").write_text(
            "PSRJ J1713+0747\n", encoding="utf-8"
        )

        service = FileDiscovery(working_dir=str(tmp_path), verbose=False)
        found = service.discover_files(["inpta_dr2"])

        assert [e["par"] for e in found["inpta_dr2"]] == [
            psr_dir / "J1713+0747.DMX.par"
        ]
        assert [e["tim"] for e in found["inpta_dr2"]] == [
            psr_dir / "J1713+0747_all.tim"
        ]
        assert found["inpta_dr2"][0]["timing_package"] == "tempo2"

    def test_validate_config_success(self):
        """Test validating valid configuration."""
        service = FileDiscovery()

        valid_config = {
            "base_dir": "/test/path",
            "par_pattern": r"test_(\w+)\.par",
            "tim_pattern": r"test_(\w+)\.tim",
            "timing_package": "pint",
        }

        # Should not raise any exception
        service._validate_data_release(valid_config, "test_release")

    def test_validate_config_missing_keys(self):
        """Test validating configuration with missing keys."""
        service = FileDiscovery()

        invalid_config = {
            "base_dir": "/test/path",
            # Missing par_pattern, tim_pattern, timing_package
        }

        with pytest.raises(ValueError, match="Missing required keys"):
            service._validate_data_release(invalid_config, "test_release")

    def test_validate_config_invalid_timing_package(self):
        """Test validating configuration with invalid timing package."""
        service = FileDiscovery()

        invalid_config = {
            "base_dir": "/test/path",
            "par_pattern": r"test_(\w+)\.par",
            "tim_pattern": r"test_(\w+)\.tim",
            "timing_package": "invalid",
        }

        with pytest.raises(ValueError, match="Invalid timing_package"):
            service._validate_data_release(invalid_config, "test_release")

    def test_validate_config_invalid_regex(self):
        """Test validating configuration with invalid regex patterns."""
        service = FileDiscovery()

        invalid_config = {
            "base_dir": "/test/path",
            "par_pattern": r"invalid[regex",  # Invalid regex
            "tim_pattern": r"test_(\w+)\.tim",
            "timing_package": "pint",
        }

        with pytest.raises(ValueError, match="Invalid regex pattern"):
            service._validate_data_release(invalid_config, "test_release")

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rglob")
    def test_discover_patterns_in_config_success(self, mock_rglob, mock_exists):
        """Test discovering patterns in a configuration."""
        service = FileDiscovery()

        mock_exists.return_value = True
        mock_rglob.return_value = [
            Path("/test/J1857+0943.par"),
            Path("/test/B1855+09.par"),
        ]

        config = {"base_dir": "/test", "par_pattern": r"([BJ]\d{4}[+-]\d{2,4})\.par"}

        result = service._discover_patterns_in_data_release(config)

        assert "J1857+0943" in result
        assert "B1855+09" in result

    @patch("pathlib.Path.exists")
    def test_discover_patterns_in_config_no_base_dir(self, mock_exists):
        """Test discovering patterns when base directory doesn't exist."""
        service = FileDiscovery()

        mock_exists.return_value = False

        config = {
            "base_dir": "/nonexistent",
            "par_pattern": r"([BJ]\d{4}[+-]\d{2,4})\.par",
        }

        result = service._discover_patterns_in_data_release(config)

        assert result == []

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rglob")
    @patch("metapulsar.file_discovery.FileDiscovery._get_tim_metadata")
    @patch("pathlib.Path.read_text")
    def test_discover_all_file_pairs_in_config_success(
        self, mock_read_text, mock_timespan, mock_rglob, mock_exists
    ):
        """Test discovering all file pairs in a configuration."""
        service = FileDiscovery()

        mock_exists.return_value = True
        mock_rglob.return_value = [
            Path("/test/J1857+0943.par"),
            Path("/test/J1857+0943.tim"),
        ]
        mock_read_text.return_value = (
            "PSR J1857+0943\nRAJ 18:57:36.4\nDECJ 09:43:17.1\n"
        )
        mock_timespan.return_value = make_tim_metadata(
            timespan_days=1000.0, toa_count=500
        )

        config = {
            "base_dir": "/test",
            "par_pattern": r"([BJ]\d{4}[+-]\d{2,4})\.par",
            "tim_pattern": r"([BJ]\d{4}[+-]\d{2,4})\.tim",
            "timing_package": "tempo2",
        }

        result = service._discover_all_file_pairs_in_data_release(
            config, "test_release"
        )

        assert len(result) == 1
        assert result[0]["par"] == Path("/test/J1857+0943.par")
        assert result[0]["tim"] == Path("/test/J1857+0943.tim")
        assert result[0]["tim_metadata"].timespan_days == 1000.0
        assert result[0]["tim_metadata"].toa_count == 500

    @patch("pathlib.Path.exists")
    def test_discover_all_file_pairs_in_config_no_base_dir(self, mock_exists):
        """Test discovering file pairs when base directory doesn't exist."""
        service = FileDiscovery()

        mock_exists.return_value = False

        config = {
            "base_dir": "/nonexistent",
            "par_pattern": r"([BJ]\d{4}[+-]\d{2,4})\.par",
            "tim_pattern": r"([BJ]\d{4}[+-]\d{2,4})\.tim",
        }

        result = service._discover_all_file_pairs_in_data_release(
            config, "test_release"
        )

        assert result == []

    def test_sole_candidate_records_provenance(self, tmp_path):
        """One candidate per kind: reason 'sole', rule None, candidates listed."""
        _write_release(
            tmp_path,
            "REL",
            {
                "par/J1713+0747.par": "PSRJ J1713+0747\n",
                "tim/J1713+0747.tim": "FORMAT 1\n",
            },
        )
        spec = {
            "base_dir": "REL/",
            "par_pattern": r"par/([BJ]\d{4}[+-]\d{2,4})\.par$",
            "tim_pattern": r"tim/([BJ]\d{4}[+-]\d{2,4})\.tim$",
            "timing_package": "tempo2",
        }
        service = FileDiscovery(
            working_dir=str(tmp_path), pta_data_releases={"r": spec}
        )
        entry = service.discover_files("r", verbose=False)["r"][0]

        assert entry["par_selection"]["reason"] == "sole"
        assert entry["par_selection"]["rule"] is None
        assert entry["par_selection"]["chosen"] == entry["par"]
        assert entry["par_selection"]["candidates"] == [entry["par"]]
        assert entry["tim_selection"]["reason"] == "sole"

    def test_ambiguous_par_without_precedence_raises(self, tmp_path):
        """Two equally-ranked pars must never be resolved silently."""
        _write_release(
            tmp_path,
            "REL",
            {
                "par/J1713+0747_a.par": "PSRJ J1713+0747\n",
                "par/J1713+0747_b.par": "PSRJ J1713+0747\n",
                "tim/J1713+0747.tim": "FORMAT 1\n",
            },
        )
        spec = {
            "base_dir": "REL/",
            "par_pattern": r"par/([BJ]\d{4}[+-]\d{2,4})_[ab]\.par$",
            "tim_pattern": r"tim/([BJ]\d{4}[+-]\d{2,4})\.tim$",
            "timing_package": "tempo2",
        }
        service = FileDiscovery(
            working_dir=str(tmp_path), pta_data_releases={"r": spec}
        )

        with pytest.raises(AmbiguousFileError) as excinfo:
            service.discover_files("r", verbose=False)
        message = str(excinfo.value)
        assert "par/J1713+0747_a.par" in message
        assert "par/J1713+0747_b.par" in message
        assert "par_precedence" in message and "par_overrides" in message

    def test_par_precedence_ranks_variant_first(self, tmp_path):
        """First matching precedence entry wins; provenance names the rule."""
        _write_release(
            tmp_path,
            "REL",
            {
                "par/J1713+0747.gls.par": "PSRJ J1713+0747\n",
                "par/J1713+0747.t2.gls.par": "PSRJ J1713+0747\n",
                "tim/J1713+0747.tim": "FORMAT 1\n",
            },
        )
        spec = {
            "base_dir": "REL/",
            "par_pattern": r"par/([BJ]\d{4}[+-]\d{2,4})(?:\.t2)?\.gls\.par$",
            "par_precedence": [r"\.t2\.gls\.par$", r"(?<!\.t2)\.gls\.par$"],
            "tim_pattern": r"tim/([BJ]\d{4}[+-]\d{2,4})\.tim$",
            "timing_package": "tempo2",
        }
        service = FileDiscovery(
            working_dir=str(tmp_path), pta_data_releases={"r": spec}
        )
        entry = service.discover_files("r", verbose=False)["r"][0]

        assert entry["par"].name == "J1713+0747.t2.gls.par"
        assert entry["par_selection"]["reason"] == "precedence"
        assert entry["par_selection"]["rule"] == r"\.t2\.gls\.par$"
        assert len(entry["par_selection"]["candidates"]) == 2

    @pytest.mark.parametrize(
        "timing_package,expected",
        [("tempo2", "J1909-3744.par"), ("pint", "J1909-3744_pint.par")],
    )
    def test_precedence_entry_can_key_off_timing_package(
        self, tmp_path, timing_package, expected
    ):
        """A qualified entry applies only for the spec's own timing package."""
        _write_release(
            tmp_path,
            "REL",
            {
                "J1909-3744.par": "PSRJ J1909-3744\n",
                "J1909-3744_pint.par": "PSRJ J1909-3744\n",
                "J1909-3744.tim": "FORMAT 1\n",
            },
        )
        spec = {
            "base_dir": "REL/",
            "par_pattern": r"REL/([BJ]\d{4}[+-]\d{2,4})(?:_pint)?\.par$",
            "par_precedence": [
                {"pattern": r"_pint\.par$", "timing_package": "pint"},
                r"(?<!_pint)\.par$",
            ],
            "tim_pattern": r"REL/([BJ]\d{4}[+-]\d{2,4})\.tim$",
            "timing_package": timing_package,
        }
        service = FileDiscovery(
            working_dir=str(tmp_path), pta_data_releases={"r": spec}
        )
        entry = service.discover_files("r", verbose=False)["r"][0]

        assert entry["par"].name == expected
        assert entry["par_selection"]["reason"] == "precedence"

    def test_override_wins_over_precedence_and_bypasses_the_pattern(self, tmp_path):
        """An override may name a file the pattern never matches."""
        _write_release(
            tmp_path,
            "REL",
            {
                "par/J1713+0747.par": "PSRJ J1713+0747\n",
                "alternate/hand_fit.par": "PSRJ J1713+0747\n",
                "tim/J1713+0747.tim": "FORMAT 1\n",
            },
        )
        spec = {
            "base_dir": "REL/",
            "par_pattern": r"par/([BJ]\d{4}[+-]\d{2,4})\.par$",
            "par_overrides": {"J1713+0747": "alternate/hand_fit.par"},
            "tim_pattern": r"tim/([BJ]\d{4}[+-]\d{2,4})\.tim$",
            "timing_package": "tempo2",
        }
        service = FileDiscovery(
            working_dir=str(tmp_path), pta_data_releases={"r": spec}
        )
        entry = service.discover_files("r", verbose=False)["r"][0]

        assert entry["par"].name == "hand_fit.par"
        assert entry["par_selection"]["reason"] == "override"
        assert entry["par_selection"]["rule"] == "alternate/hand_fit.par"
        # the pattern candidate is still reported, so the audit shows what was overridden
        assert [p.name for p in entry["par_selection"]["candidates"]] == [
            "J1713+0747.par"
        ]

    def test_override_seeds_a_pulsar_with_no_pattern_candidates(self, tmp_path):
        """A pulsar invisible to par_pattern is still discoverable via an override."""
        _write_release(
            tmp_path,
            "REL",
            {
                "alternate/J1713+0747_handfit.par": "PSRJ J1713+0747\n",
                "tim/J1713+0747.tim": "FORMAT 1\n",
            },
        )
        spec = {
            "base_dir": "REL/",
            "par_pattern": r"par/([BJ]\d{4}[+-]\d{2,4})\.par$",  # matches nothing
            "par_overrides": {"J1713+0747": "alternate/J1713+0747_handfit.par"},
            "tim_pattern": r"tim/([BJ]\d{4}[+-]\d{2,4})\.tim$",
            "timing_package": "tempo2",
        }
        service = FileDiscovery(
            working_dir=str(tmp_path), pta_data_releases={"r": spec}
        )
        entries = service.discover_files("r", verbose=False)["r"]

        assert len(entries) == 1
        assert entries[0]["par"].name == "J1713+0747_handfit.par"
        assert entries[0]["par_selection"]["reason"] == "override"
        assert entries[0]["par_selection"]["candidates"] == []

    def test_missing_override_raises(self, tmp_path):
        """A stale override must fail loudly, never fall back to the pattern."""
        _write_release(
            tmp_path,
            "REL",
            {
                "par/J1713+0747.par": "PSRJ J1713+0747\n",
                "tim/J1713+0747.tim": "FORMAT 1\n",
            },
        )
        spec = {
            "base_dir": "REL/",
            "par_pattern": r"par/([BJ]\d{4}[+-]\d{2,4})\.par$",
            "par_overrides": {"J1713+0747": "par/does_not_exist.par"},
            "tim_pattern": r"tim/([BJ]\d{4}[+-]\d{2,4})\.tim$",
            "timing_package": "tempo2",
        }
        service = FileDiscovery(
            working_dir=str(tmp_path), pta_data_releases={"r": spec}
        )

        with pytest.raises(MissingOverrideError, match="does_not_exist"):
            service.discover_files("r", verbose=False)

    def test_tim_selection_is_symmetric(self, tmp_path):
        """Precedence and ambiguity apply identically to the tim side."""
        _write_release(
            tmp_path,
            "REL",
            {
                "par/J1713+0747.par": "PSRJ J1713+0747\n",
                "tim/J1713+0747.tim": "FORMAT 1\n",
                "tim/J1713+0747_all.tim": "FORMAT 1\n",
            },
        )
        spec = {
            "base_dir": "REL/",
            "par_pattern": r"par/([BJ]\d{4}[+-]\d{2,4})\.par$",
            "tim_pattern": r"tim/([BJ]\d{4}[+-]\d{2,4})(?:_all)?\.tim$",
            "timing_package": "tempo2",
        }
        service = FileDiscovery(
            working_dir=str(tmp_path), pta_data_releases={"r": spec}
        )
        with pytest.raises(AmbiguousFileError, match="tim_precedence"):
            service.discover_files("r", verbose=False)

        spec["tim_precedence"] = [r"_all\.tim$", r"(?<!_all)\.tim$"]
        entry = FileDiscovery(
            working_dir=str(tmp_path), pta_data_releases={"r": spec}
        ).discover_files("r", verbose=False)["r"][0]
        assert entry["tim"].name == "J1713+0747_all.tim"
        assert entry["tim_selection"]["rule"] == r"_all\.tim$"

    def test_select_release_file_rejects_an_empty_candidate_list(self, tmp_path):
        """Nothing to choose from is a FileSelectionError, not a bare min() crash."""
        with pytest.raises(FileSelectionError, match="no par candidates"):
            select_release_file(
                [],
                pulsar_name="J1713+0747",
                kind="par",
                release_name="rel",
                base_path=tmp_path,
                rules=(),
                override=None,
                timing_package="pint",
            )

    def test_select_release_file_override_precedes_ranking(self, tmp_path):
        """The override branch returns before any rank is computed."""
        chosen_path = tmp_path / "alt.par"
        chosen_path.write_text("PSRJ J1713+0747\n", encoding="utf-8")
        ranked_path = tmp_path / "ranked.par"

        chosen, provenance = select_release_file(
            [ranked_path],
            pulsar_name="J1713+0747",
            kind="par",
            release_name="rel",
            base_path=tmp_path,
            rules=_normalize_precedence([r"ranked\.par$"], "par", "rel"),
            override="alt.par",
            timing_package="pint",
        )

        assert chosen == chosen_path
        assert provenance == {
            "chosen": chosen_path,
            "candidates": [ranked_path],
            "reason": "override",
            "rule": "alt.par",
        }

    @pytest.mark.parametrize(
        "precedence,match",
        [
            ([{"patern": r"x"}], "unknown keys"),
            ([{"timing_package": "pint"}], "missing required key 'pattern'"),
            ([{"pattern": r"x", "timing_package": "tempo1"}], "invalid timing_package"),
            ([r"("], "invalid regex"),
            ([42], "expected a regex string or a dict"),
        ],
    )
    def test_add_data_release_rejects_bad_precedence(self, precedence, match):
        """add_data_release validates precedence through the discovery helpers."""
        service = FileDiscovery()
        with pytest.raises(ValueError, match=match):
            service.add_data_release(
                "broken",
                {
                    "base_dir": "X/",
                    "par_pattern": r"([BJ]\d{4}[+-]\d{2,4})\.par$",
                    "par_precedence": precedence,
                    "tim_pattern": r"([BJ]\d{4}[+-]\d{2,4})\.tim$",
                    "timing_package": "tempo2",
                },
            )

    @pytest.mark.parametrize(
        "overrides,match",
        [
            (["J1713+0747"], "expected a dict"),
            ({"J1713+0747": 42}, "expected str -> str"),
        ],
    )
    def test_add_data_release_rejects_bad_overrides(self, overrides, match):
        """add_data_release validates overrides through the discovery helpers."""
        service = FileDiscovery()
        with pytest.raises(ValueError, match=match):
            service.add_data_release(
                "broken",
                {
                    "base_dir": "X/",
                    "par_pattern": r"([BJ]\d{4}[+-]\d{2,4})\.par$",
                    "par_overrides": overrides,
                    "tim_pattern": r"([BJ]\d{4}[+-]\d{2,4})\.tim$",
                    "timing_package": "tempo2",
                },
            )

    def test_shipped_precedence_rules_are_disjoint(self):
        """No shipped release may rely on rule order to break an overlap.

        Overlapping rules still give the right answer, but only by accident of
        ordering: reversing them raises AmbiguousFileError instead of choosing
        the other variant. Disjointness is per (path, timing_package), so two
        rules sharing a regex but carrying complementary timing_package
        qualifiers are legal.
        """
        probes = [
            "J1713+0747_NANOGrav_9yv1.gls.par",
            "J1713+0747_NANOGrav_9yv1.t2.gls.par",
            "J1909-3744.par",
            "J1909-3744_pint.par",
        ]
        for release_name, spec in PTA_DATA_RELEASES.items():
            for kind in ("par", "tim"):
                rules = _normalize_precedence(
                    spec.get(f"{kind}_precedence") or (), kind, release_name
                )
                for probe in probes:
                    for package in ("pint", "tempo2"):
                        matched = [
                            r.pattern
                            for r in rules
                            if r.timing_package in (None, package)
                            and r.regex.search(probe)
                        ]
                        assert len(matched) <= 1, (
                            f"{release_name} {kind}_precedence rules overlap on "
                            f"{probe} under {package}: {matched}"
                        )

    @pytest.mark.slow
    @pytest.mark.requires_ipta_data
    @pytest.mark.parametrize("data_root", ["data/ipta-dr2"])
    def test_shipped_releases_resolve_without_ambiguity(self, data_root):
        """The M1 no-op claim, as a test: no shipped release ties on real trees."""
        repo_root = Path(__file__).resolve().parents[1]
        root = repo_root / data_root
        if not root.exists():
            pytest.skip(f"{data_root} not present")

        service = FileDiscovery(working_dir=str(root), verbose=False)
        present = [
            key
            for key, spec in PTA_DATA_RELEASES.items()
            if (root / spec["base_dir"]).exists()
        ]
        if not present:
            pytest.skip(f"no known release layouts under {data_root}")

        # raises AmbiguousFileError / MissingOverrideError if any release is unresolvable
        found = service.discover_files(present)
        assert any(found[key] for key in present)

    def test_auto_discovered_layout_ties_are_ambiguous(self, tmp_path):
        """discover_layout emits no precedence, so a variant release must raise.

        The inferred pattern deliberately matches every ``<PSR>*.par`` and an
        inferred spec has no way to rank them, so the selection is genuinely
        ambiguous and must not silently last-win. Callers of discover_layout on
        a variant-shipping release have to supply par_precedence themselves.
        """
        release = _write_release(
            tmp_path,
            "FLAT",
            {
                "J1909-3744.par": "PSRJ J1909-3744\n",
                "J1909-3744_pint.par": "PSRJ J1909-3744\n",
                "J1909-3744.tim": "FORMAT 1\n",
            },
        )
        layout = discover_layout(str(release), verbose=False, name="flat")

        with pytest.raises(AmbiguousFileError, match=r"_pint\.par"):
            discover_files(layout, working_dir=str(tmp_path), verbose=False)

    def test_nanograv_9y_spec_selects_the_t2_par_for_j1713(self, tmp_path):
        """Regression lock: the shipped NG9 spec must pick the engine-runnable par.

        The tempo1 par carries PAASCNODE, which neither PINT nor tempo2
        implements; the t2 par is the solution both engines can evaluate.
        """
        _write_release(
            tmp_path,
            "NANOGrav_9y",
            {
                "par/J1713+0747_NANOGrav_9yv1.gls.par": "PSRJ J1713+0747\n",
                "par/J1713+0747_NANOGrav_9yv1.t2.gls.par": "PSRJ J1713+0747\n",
                "par/B1855+09_NANOGrav_9yv1.gls.par": "PSRJ B1855+09\n",
                "tim/J1713+0747_NANOGrav_9yv1.tim": "FORMAT 1\n",
                "tim/B1855+09_NANOGrav_9yv1.tim": "FORMAT 1\n",
            },
        )
        service = FileDiscovery(working_dir=str(tmp_path), verbose=False)
        entries = {
            e["par"].name.split("_NANOGrav")[0]: e
            for e in service.discover_files("nanograv_9y")["nanograv_9y"]
        }

        assert entries["J1713+0747"]["par"].name.endswith(".t2.gls.par")
        assert entries["J1713+0747"]["par_selection"]["reason"] == "precedence"
        # every other pulsar is untouched
        assert entries["B1855+09"]["par_selection"]["reason"] == "sole"

    def test_ppta_dr3_spec_follows_the_timing_package(self, tmp_path):
        """The shipped PPTA_DR3 spec selects the par matching its engine."""
        _write_release(
            tmp_path,
            "PPTA_DR3",
            {
                "J1909-3744.par": "PSRJ J1909-3744\n",
                "J1909-3744_pint.par": "PSRJ J1909-3744\n",
                "J1909-3744.tim": "FORMAT 1\n",
            },
        )
        service = FileDiscovery(working_dir=str(tmp_path), verbose=False)
        assert (
            service.discover_files("ppta_dr3")["ppta_dr3"][0]["par"].name
            == "J1909-3744.par"
        )

        spec = dict(PTA_DATA_RELEASES["ppta_dr3"])
        spec["timing_package"] = "pint"
        pint_service = FileDiscovery(
            working_dir=str(tmp_path),
            pta_data_releases={"ppta_dr3": spec},
            verbose=False,
        )
        assert (
            pint_service.discover_files("ppta_dr3")["ppta_dr3"][0]["par"].name
            == "J1909-3744_pint.par"
        )

    def test_nanograv_12y_still_excludes_its_t2_par(self, tmp_path):
        """NG12 is deliberately NOT converted.

        Its default par is already BINARY DDK with the Kopeikin term fitted, and
        the spec is timing_package='pint', so the lookahead is correct. Note the
        real variant suffix is '.gls.t2.par', not the 9y '.t2.gls.par'.
        """
        _write_release(
            tmp_path,
            "NANOGrav_12y",
            {
                "par/J1713+0747_NANOGrav_12yv2.gls.par": "PSRJ J1713+0747\n",
                "par/J1713+0747_NANOGrav_12yv2.gls.t2.par": "PSRJ J1713+0747\n",
                "tim/J1713+0747_NANOGrav_12yv2.tim": "FORMAT 1\n",
            },
        )
        service = FileDiscovery(working_dir=str(tmp_path), verbose=False)
        entry = service.discover_files("nanograv_12y")["nanograv_12y"][0]

        assert entry["par"].name == "J1713+0747_NANOGrav_12yv2.gls.par"
        assert entry["par_selection"]["reason"] == "sole"

    @pytest.mark.slow
    @pytest.mark.requires_ipta_data
    def test_nanograv_9y_real_tree_selects_the_t2_par(self):
        """Real-data lock: on the actual release, NG9 J1713 is the t2 par."""
        repo_root = Path(__file__).resolve().parents[1]
        root = repo_root / "data" / "ipta-dr2"
        if not (root / "NANOGrav_9y" / "par").is_dir():
            pytest.skip("data/ipta-dr2 not present")

        service = FileDiscovery(working_dir=str(root), verbose=False)
        chosen = {
            entry["par"].name
            for entry in service.discover_files("nanograv_9y")["nanograv_9y"]
            if "J1713+0747" in entry["par"].name
        }
        assert chosen == {"J1713+0747_NANOGrav_9yv1.t2.gls.par"}


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_discover_files_convenience_function(self):
        """Test discover_files convenience function."""
        with patch.object(FileDiscovery, "discover_files") as mock_discover:
            mock_discover.return_value = {"epta_dr2": []}

            # Mock data releases
            mock_data_releases = {"epta_dr2": {"base_dir": "/test"}}
            result = discover_files(
                mock_data_releases, working_dir="/test", data_release_names="epta_dr2"
            )

            mock_discover.assert_called_once_with("epta_dr2", True)
            assert result == {"epta_dr2": []}

    def test_discover_files_convenience_function_with_list(self):
        """Test discover_files convenience function with list input."""
        with patch.object(FileDiscovery, "discover_files") as mock_discover:
            mock_discover.return_value = {"epta_dr2": [], "ppta_dr2": []}

            # Mock data releases
            mock_data_releases = {
                "epta_dr2": {"base_dir": "/test"},
                "ppta_dr2": {"base_dir": "/test"},
            }
            result = discover_files(
                mock_data_releases,
                working_dir="/test",
                data_release_names=["epta_dr2", "ppta_dr2"],
            )

            mock_discover.assert_called_once_with(["epta_dr2", "ppta_dr2"], True)
            assert result == {"epta_dr2": [], "ppta_dr2": []}

    def test_discover_files_convenience_function_verbose_false(self):
        """Test discover_files convenience function with verbose=False."""
        with patch.object(FileDiscovery, "discover_files") as mock_discover:
            mock_discover.return_value = {"epta_dr2": []}

            # Mock data releases
            mock_data_releases = {"epta_dr2": {"base_dir": "/test"}}
            result = discover_files(
                mock_data_releases,
                working_dir="/test",
                data_release_names="epta_dr2",
                verbose=False,
            )

            mock_discover.assert_called_once_with("epta_dr2", False)
            assert result == {"epta_dr2": []}


class TestPulsarHelperFunctions:
    """Test pulsar helper functions."""

    def test_get_pulsar_names_from_file_data_success(self):
        """Test getting pulsar names from file data successfully."""
        from metapulsar.file_discovery import get_pulsar_names_from_file_data

        # Mock file data
        file_data = {
            "epta_dr2": [
                {
                    "par": "test/J0613-0200.par",
                    "tim": "test/J0613-0200.tim",
                    "par_content": "PSR J0613-0200\nRAJ 06:13:43.9754\nDECJ -02:00:47.1755\n",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    "timing_package": "tempo2",
                }
            ],
            "ppta_dr2": [
                {
                    "par": "test/J1857+0943.par",
                    "tim": "test/J1857+0943.tim",
                    "par_content": "PSR J1857+0943\nRAJ 18:57:36.3907\nDECJ +09:43:17.2070\n",
                    "tim_metadata": make_tim_metadata(timespan_days=1200.0),
                    "timing_package": "tempo2",
                }
            ],
        }

        with patch("metapulsar.metapulsar_factory.MetaPulsarFactory") as mock_factory:
            mock_instance = mock_factory.return_value
            mock_instance.group_files_by_pulsar.return_value = {
                "J0613-0200": {"epta_dr2": [file_data["epta_dr2"][0]]},
                "J1857+0943": {"ppta_dr2": [file_data["ppta_dr2"][0]]},
            }

            result = get_pulsar_names_from_file_data(file_data)

            assert result == ["J0613-0200", "J1857+0943"]
            mock_instance.group_files_by_pulsar.assert_called_once_with(file_data)

    def test_get_pulsar_names_from_file_data_no_pulsars(self):
        """Test getting pulsar names when no pulsars found."""
        from metapulsar.file_discovery import get_pulsar_names_from_file_data

        file_data = {"epta_dr2": []}

        with patch("metapulsar.metapulsar_factory.MetaPulsarFactory") as mock_factory:
            mock_instance = mock_factory.return_value
            mock_instance.group_files_by_pulsar.return_value = {}

            with pytest.raises(
                ValueError, match="No valid pulsar files found in file_data"
            ):
                get_pulsar_names_from_file_data(file_data)

    def test_filter_file_data_by_pulsars_single_j_name(self):
        """Test filtering file data by single J-name."""
        from metapulsar.file_discovery import filter_file_data_by_pulsars

        file_data = {
            "epta_dr2": [
                {
                    "par": "test/J0613-0200.par",
                    "tim": "test/J0613-0200.tim",
                    "par_content": "PSR J0613-0200\nRAJ 06:13:43.9754\nDECJ -02:00:47.1755\n",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    "timing_package": "tempo2",
                }
            ],
            "ppta_dr2": [
                {
                    "par": "test/J1857+0943.par",
                    "tim": "test/J1857+0943.tim",
                    "par_content": "PSR J1857+0943\nRAJ 18:57:36.3907\nDECJ +09:43:17.2070\n",
                    "tim_metadata": make_tim_metadata(timespan_days=1200.0),
                    "timing_package": "tempo2",
                }
            ],
        }

        epta_file = _with_catalog_aliases(file_data["epta_dr2"][0], ["J0613-0200"])
        ppta_file = _with_catalog_aliases(file_data["ppta_dr2"][0], ["J1857+0943"])

        with patch("metapulsar.metapulsar_factory.MetaPulsarFactory") as mock_factory:
            mock_instance = mock_factory.return_value
            mock_instance.group_files_by_pulsar.return_value = {
                "J0613-0200": {"epta_dr2": [epta_file]},
                "J1857+0943": {"ppta_dr2": [ppta_file]},
            }

            result = filter_file_data_by_pulsars(file_data, "J0613-0200")

            assert result == {"epta_dr2": [epta_file]}

    def test_filter_file_data_by_pulsars_multiple_j_names(self):
        """Test filtering file data by multiple J-names."""
        from metapulsar.file_discovery import filter_file_data_by_pulsars

        file_data = {
            "epta_dr2": [
                {
                    "par": "test/J0613-0200.par",
                    "tim": "test/J0613-0200.tim",
                    "par_content": "PSR J0613-0200\nRAJ 06:13:43.9754\nDECJ -02:00:47.1755\n",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    "timing_package": "tempo2",
                }
            ],
            "ppta_dr2": [
                {
                    "par": "test/J1857+0943.par",
                    "tim": "test/J1857+0943.tim",
                    "par_content": "PSR J1857+0943\nRAJ 18:57:36.3907\nDECJ +09:43:17.2070\n",
                    "tim_metadata": make_tim_metadata(timespan_days=1200.0),
                    "timing_package": "tempo2",
                }
            ],
        }

        epta_file = _with_catalog_aliases(file_data["epta_dr2"][0], ["J0613-0200"])
        ppta_file = _with_catalog_aliases(file_data["ppta_dr2"][0], ["J1857+0943"])

        with patch("metapulsar.metapulsar_factory.MetaPulsarFactory") as mock_factory:
            mock_instance = mock_factory.return_value
            mock_instance.group_files_by_pulsar.return_value = {
                "J0613-0200": {"epta_dr2": [epta_file]},
                "J1857+0943": {"ppta_dr2": [ppta_file]},
            }

            result = filter_file_data_by_pulsars(
                file_data, ["J0613-0200", "J1857+0943"]
            )

            expected = {
                "epta_dr2": [epta_file],
                "ppta_dr2": [ppta_file],
            }
            assert result == expected

    def test_filter_file_data_by_pulsars_b_name(self):
        """Test filtering file data by B-name."""
        from metapulsar.file_discovery import filter_file_data_by_pulsars

        file_data = {
            "epta_dr2": [
                {
                    "par": "test/B0613-02.par",
                    "tim": "test/B0613-02.tim",
                    "par_content": "PSR B0613-02\nRAJ 06:13:43.9754\nDECJ -02:00:47.1755\n",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    "timing_package": "tempo2",
                }
            ]
        }

        epta_file = _with_catalog_aliases(file_data["epta_dr2"][0], ["B0613-02"])

        with patch("metapulsar.metapulsar_factory.MetaPulsarFactory") as mock_factory:
            mock_instance = mock_factory.return_value
            mock_instance.group_files_by_pulsar.return_value = {
                "B0613-02": {"epta_dr2": [epta_file]}
            }

            result = filter_file_data_by_pulsars(file_data, "B0613-02")

            assert result == {"epta_dr2": [epta_file]}

    def test_filter_file_data_by_pulsars_mixed_names(self):
        """Test filtering file data by mixed J and B names."""
        from metapulsar.file_discovery import filter_file_data_by_pulsars

        file_data = {
            "epta_dr2": [
                {
                    "par": "test/J0613-0200.par",
                    "tim": "test/J0613-0200.tim",
                    "par_content": "PSR J0613-0200\nRAJ 06:13:43.9754\nDECJ -02:00:47.1755\n",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    "timing_package": "tempo2",
                }
            ],
            "ppta_dr2": [
                {
                    "par": "test/J1857+0943.par",
                    "tim": "test/J1857+0943.tim",
                    "par_content": "PSR J1857+0943\nRAJ 18:57:36.3907\nDECJ +09:43:17.2070\n",
                    "tim_metadata": make_tim_metadata(timespan_days=1200.0),
                    "timing_package": "tempo2",
                }
            ],
        }

        epta_file = _with_catalog_aliases(file_data["epta_dr2"][0], ["J0613-0200"])
        ppta_file = _with_catalog_aliases(
            file_data["ppta_dr2"][0],
            ["J1857+0943", "B1855+09"],
            path_name="J1857+0943",
        )

        with patch("metapulsar.metapulsar_factory.MetaPulsarFactory") as mock_factory:
            mock_instance = mock_factory.return_value
            mock_instance.group_files_by_pulsar.return_value = {
                "J0613-0200": {"epta_dr2": [epta_file]},
                "B1855+09": {"ppta_dr2": [ppta_file]},
            }

            result = filter_file_data_by_pulsars(file_data, ["J0613-0200", "B1855+09"])

            expected = {
                "epta_dr2": [epta_file],
                "ppta_dr2": [ppta_file],
            }
            assert result == expected

    def test_filter_file_data_by_pulsars_pulsar_not_found(self):
        """Test filtering file data when requested pulsar not found."""
        from metapulsar.file_discovery import filter_file_data_by_pulsars

        file_data = {
            "epta_dr2": [
                {
                    "par": "test/J0613-0200.par",
                    "tim": "test/J0613-0200.tim",
                    "par_content": "PSR J0613-0200\nRAJ 06:13:43.9754\nDECJ -02:00:47.1755\n",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    "timing_package": "tempo2",
                }
            ]
        }

        epta_file = _with_catalog_aliases(file_data["epta_dr2"][0], ["J0613-0200"])

        with patch("metapulsar.metapulsar_factory.MetaPulsarFactory") as mock_factory:
            mock_instance = mock_factory.return_value
            mock_instance.group_files_by_pulsar.return_value = {
                "J0613-0200": {"epta_dr2": [epta_file]}
            }

            with pytest.raises(
                ValueError, match="Pulsar 'J9999\\+9999' not found in file data"
            ):
                filter_file_data_by_pulsars(file_data, "J9999+9999")

    def test_filter_file_data_by_pulsars_no_pulsars_in_data(self):
        """Test filtering file data when no pulsars found in input data."""
        from metapulsar.file_discovery import filter_file_data_by_pulsars

        file_data = {"epta_dr2": []}

        with patch("metapulsar.metapulsar_factory.MetaPulsarFactory") as mock_factory:
            mock_instance = mock_factory.return_value
            mock_instance.group_files_by_pulsar.return_value = {}

            with pytest.raises(
                ValueError, match="No valid pulsar files found in file_data"
            ):
                filter_file_data_by_pulsars(file_data, "J0613-0200")

    def test_filter_file_data_by_pulsars_no_matching_pulsars(self):
        """Test filtering file data when no matching pulsars found."""
        from metapulsar.file_discovery import filter_file_data_by_pulsars

        file_data = {
            "epta_dr2": [
                {
                    "par": "test/J0613-0200.par",
                    "tim": "test/J0613-0200.tim",
                    "par_content": "PSR J0613-0200\nRAJ 06:13:43.9754\nDECJ -02:00:47.1755\n",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    "timing_package": "tempo2",
                }
            ]
        }

        epta_file = _with_catalog_aliases(file_data["epta_dr2"][0], ["J0613-0200"])

        with patch("metapulsar.metapulsar_factory.MetaPulsarFactory") as mock_factory:
            mock_instance = mock_factory.return_value
            mock_instance.group_files_by_pulsar.return_value = {
                "J0613-0200": {"epta_dr2": [epta_file]}
            }

            with pytest.raises(
                ValueError, match="Pulsar 'J9999\\+9999' not found in file data"
            ):
                filter_file_data_by_pulsars(file_data, ["J9999+9999"])
