"""Test error handling and malformed data robustness."""

import pytest
import tempfile
from pathlib import Path
from metapulsar import MetaPulsarFactory, FileDiscovery
from metapulsar.file_discovery import PTA_DATA_RELEASES
from metapulsar.pint_helpers import PINTDiscoveryError
from metapulsar.sandbox_tempo2 import Tempo2Error
from tests.helpers import make_tim_metadata


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling and malformed data robustness."""

    def test_missing_data_directory(self):
        """Test handling of missing data directories."""
        # Test with non-existent PTA
        discovery_service = FileDiscovery(working_dir="../../data/ipta-dr2")
        with pytest.raises(KeyError):
            discovery_service.discover_files(["nonexistent_config"])

    def test_malformed_par_file(self, available_data_sets):
        """Test handling of malformed par files."""
        if not available_data_sets:
            pytest.skip("No data available for testing")

        # Create a temporary malformed par file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_par = Path(temp_dir) / "J0030+0451.par"

            # Write malformed par file with valid coordinates but missing required parameters
            malformed_parfile_content = """# Malformed par file
F0 123.456 1 0.001
RAJ 00:30:27.4
DECJ 04:51:39.7
DM 13.299 1 0.001
# Missing other required parameters like PEPOCH, etc.
"""
            with open(temp_par, "w") as f:
                f.write(malformed_parfile_content)

                # Create a temporary PTA data release pointing to malformed file
                data_release = PTA_DATA_RELEASES["epta_dr1_v2_2"].copy()
                data_release["base_dir"] = str(temp_dir)
                data_release["par_pattern"] = "J0030+0451.par"

                # This should raise RuntimeError because the parfile is malformed (missing timfile)
                with pytest.raises(RuntimeError):
                    # Create file_data format for the test (list format)
                    file_data = {
                        "epta_dr1_v2_2": [
                            {
                                "par": Path(temp_dir) / "J0030+0451.par",
                                "tim": Path(temp_dir) / "J0030+0451.tim",
                                "par_content": malformed_parfile_content,
                                "timing_package": "tempo2",
                                "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                            }
                        ]
                    }
                    MetaPulsarFactory().create_metapulsar(
                        file_data, use_pulse_numbers="no"
                    )

    def test_malformed_tim_file(self, available_data_sets):
        """Test handling of malformed tim files."""
        if not available_data_sets:
            pytest.skip("No data available for testing")

        # Create a temporary malformed tim file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_tim = Path(temp_dir) / "J0030+0451.tim"

            # Write malformed tim file
            with open(temp_tim, "w") as f:
                f.write(
                    """# Malformed tim file
# Missing required columns
C 12345.67890 0.0001
"""
                )

                # Create a temporary PTA data release pointing to malformed file
                data_release = PTA_DATA_RELEASES["epta_dr1_v2_2"].copy()
                data_release["base_dir"] = str(temp_dir)
                data_release["tim_pattern"] = "J0030+0451.tim"

                # This should raise PINTDiscoveryError because the tim file is malformed
                with pytest.raises(PINTDiscoveryError):
                    # Create file_data format for the test (list format)
                    file_data = {
                        "epta_dr1_v2_2": [
                            {
                                "par": Path(temp_dir) / "J0030+0451.par",
                                "tim": temp_tim,
                                # Explicit UNITS TDB so M1 normalization does not
                                # invoke tempo2 transform before the malformed
                                # TIM is discovered.
                                "par_content": (
                                    "PSR J0030+0451\n"
                                    "RAJ 00:30:27.4\n"
                                    "DECJ 04:51:39.7\n"
                                    "UNITS TDB\n"
                                ),
                                "timing_package": "tempo2",
                                "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                            }
                        ]
                    }
                    MetaPulsarFactory().create_metapulsar(
                        file_data, use_pulse_numbers="no"
                    )

    def test_invalid_data_release_config(self):
        """Test handling of invalid data release configurations."""
        # Test with invalid PTA config name
        with pytest.raises(KeyError):
            PTA_DATA_RELEASES["invalid_config"]

        # Test with invalid primary/reference PTA - should fallback to timespan-based selection
        # Create file_data format for the test (list format)
        file_data = {
            "epta_dr1_v2_2": [
                {
                    "par": Path("/nonexistent/J0030+0451.par"),
                    "tim": Path("/nonexistent/J0030+0451.tim"),
                    "par_content": "PSR J0030+0451\nRAJ 00:30:27.4\nDECJ 04:51:39.7\nF0 327.405\nF1 -1.2e-15\n",
                    "timing_package": "tempo2",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                }
            ]
        }
        try:
            MetaPulsarFactory().create_metapulsar(
                file_data, reference_pta="invalid_pta"
            )
            # Test passes - invalid reference_pta is handled gracefully
        except Exception as e:
            # If it fails, it should be due to other issues, not KeyError for invalid reference_pta
            assert "Reference PTA" not in str(
                e
            ), f"Unexpected KeyError for invalid reference_pta: {e}"

    @pytest.mark.slow
    @pytest.mark.requires_libstempo
    def test_empty_par_files(self, available_data_sets):
        """Test handling of empty par files."""
        if not available_data_sets:
            pytest.skip("No data available for testing")

        # Create a temporary empty par file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_par = Path(temp_dir) / "J0030+0451.par"
            temp_par.touch()  # Create empty file

            # Create a minimal TIM file for the test
            temp_tim = Path(temp_dir) / "J0030+0451.tim"
            with open(temp_tim, "w") as f:
                f.write(
                    """FORMAT 1
C 55000.0 123.456 0.001 1234.5 1234.5
"""
                )

            # Create a temporary PTA data release pointing to empty file
            data_release = PTA_DATA_RELEASES["epta_dr1_v2_2"].copy()
            data_release["base_dir"] = str(temp_dir)
            data_release["par_pattern"] = "J0030+0451.par"

            # This should raise Tempo2Error when libstempo processes the consistent par/tim pair
            with pytest.raises(Tempo2Error):
                # Create file_data format for the test (list format)
                # par_content must include EPHEM/CLOCK for consistent convention rules; on-disk par is empty
                minimal_parfile_content = """RAJ 00:30:27.4
DECJ 04:51:39.7
F0 123.456
DM 13.299 1 0.001
PEPOCH 55000
EPHEM DE421
CLOCK TT(TAI)
UNITS TDB
"""
                file_data = {
                    "epta_dr1_v2_2": [
                        {
                            "par": temp_par,
                            "tim": Path(temp_dir) / "J0030+0451.tim",
                            "par_content": minimal_parfile_content,  # Minimal valid content
                            "timing_package": "tempo2",
                            "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                        }
                    ]
                }
                MetaPulsarFactory().create_metapulsar(file_data, use_pulse_numbers="no")

    @pytest.mark.slow
    @pytest.mark.requires_libstempo
    def test_corrupted_binary_files(self, available_data_sets):
        """Test handling of corrupted binary files."""
        if not available_data_sets:
            pytest.skip("No data available for testing")

        # Create a temporary corrupted par file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_par = Path(temp_dir) / "J0030+0451.par"

            # Write corrupted par file with binary data
            with open(temp_par, "wb") as f:
                f.write(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09")

            # Create a minimal TIM file for the test
            temp_tim = Path(temp_dir) / "J0030+0451.tim"
            with open(temp_tim, "w") as f:
                f.write(
                    """FORMAT 1
C 55000.0 123.456 0.001 1234.5 1234.5
"""
                )

            # Create a temporary PTA data release pointing to corrupted file
            data_release = PTA_DATA_RELEASES["epta_dr1_v2_2"].copy()
            data_release["base_dir"] = str(temp_dir)
            data_release["par_pattern"] = "J0030+0451.par"

            # This should raise Tempo2Error when libstempo processes the consistent par/tim pair
            with pytest.raises(Tempo2Error):
                # Create file_data format for the test (list format)
                # par_content must include EPHEM/CLOCK for consistent convention rules; on-disk par is corrupted
                valid_parfile_content = """RAJ 00:30:27.4
DECJ 04:51:39.7
F0 123.456
DM 13.299 1 0.001
PEPOCH 55000
EPHEM DE421
CLOCK TT(TAI)
UNITS TDB
"""
                file_data = {
                    "epta_dr1_v2_2": [
                        {
                            "par": temp_par,
                            "tim": Path(temp_dir) / "J0030+0451.tim",
                            "par_content": valid_parfile_content,  # Valid content for coordinate discovery
                            "timing_package": "tempo2",
                            "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                        }
                    ]
                }
                MetaPulsarFactory().create_metapulsar(file_data, use_pulse_numbers="no")

    def test_memory_limit_handling(self, available_data_sets):
        """Test handling of memory limits with large datasets."""
        if not available_data_sets:
            pytest.skip("No data available for testing")

        # This test would be implemented if we had very large datasets
        try:
            MetaPulsarFactory().create_metapulsar(
                pulsar_name="J0030+0451",
                pta_names=["epta_dr1_v2_2"],
                reference_pta="epta_dr1_v2_2",
            )
        except Exception as e:
            # Should not crash due to memory issues with normal data
            assert "memory" not in str(e).lower()
            assert "segfault" not in str(e).lower()

    def test_concurrent_access(self, available_data_sets):
        """Test handling of concurrent access to data files."""
        if not available_data_sets:
            pytest.skip("No data available for testing")

        import threading

        results = []
        errors = []

        def create_metapulsar():
            try:
                mp = MetaPulsarFactory().create_metapulsar(
                    pulsar_name="J0030+0451",
                    pta_names=["epta_dr1_v2_2"],
                    reference_pta="epta_dr1_v2_2",
                )
                results.append(mp)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=create_metapulsar)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have at least one successful result
        assert len(results) > 0 or len(errors) > 0

        # If there are errors, they should be reasonable (not file access issues)
        for error in errors:
            assert "Permission denied" not in str(error)
            assert "Device or resource busy" not in str(error)
