"""Tests for FileDiscovery."""

import pytest
from pathlib import Path
from unittest.mock import patch
from metapulsar.file_discovery import FileDiscovery, PTA_DATA_RELEASES
from metapulsar import discover_files
from tests.helpers import make_tim_metadata


def _with_catalog_aliases(file_entry, catalog_names, path_name=None):
    """Attach identity fields expected by build_alias_map in filter tests."""
    enriched = dict(file_entry)
    enriched["catalog_names"] = list(catalog_names)
    enriched["path_name"] = path_name or catalog_names[0]
    return enriched


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
        service._validate_data_release(valid_config)

    def test_validate_config_missing_keys(self):
        """Test validating configuration with missing keys."""
        service = FileDiscovery()

        invalid_config = {
            "base_dir": "/test/path",
            # Missing par_pattern, tim_pattern, timing_package
        }

        with pytest.raises(ValueError, match="Missing required keys"):
            service._validate_data_release(invalid_config)

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
            service._validate_data_release(invalid_config)

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
            service._validate_data_release(invalid_config)

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

        result = service._discover_all_file_pairs_in_data_release(config)

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

        result = service._discover_all_file_pairs_in_data_release(config)

        assert result == []


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
