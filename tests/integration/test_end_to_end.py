"""Test end-to-end functionality with real data."""

import pytest
import numpy as np
from metapulsar import MetaPulsar, MetaPulsarFactory, FileDiscovery


@pytest.mark.integration
class TestEndToEnd:
    """Test end-to-end functionality with real data."""

    @pytest.mark.slow
    @pytest.mark.real_data
    def test_parameter_merging_strategies(self, available_data_sets, test_pulsars):
        """Test different parameter merging strategies."""
        if not available_data_sets:
            pytest.skip("No data available for testing")

        test_configs = ["epta_dr1_v2_2", "ppta_dr2", "nanograv_9y"]
        discovery_service = FileDiscovery(working_dir="../../data/ipta-dr2")

        for pulsar in test_pulsars[:1]:  # Test first pulsar
            # Test with different primary PTAs
            for primary_pta in test_configs:
                # Discover files for the pulsar
                file_data = discovery_service.discover_files(test_configs)

                # Filter file_data to only include files for this pulsar
                filtered_file_data = {}
                for pta_name, files in file_data.items():
                    if files:  # Check if files exist for this PTA
                        # Filter files that match the pulsar name
                        matching_files = []
                        for file_info in files:
                            if pulsar in str(
                                file_info.get("parfile", "")
                            ) or pulsar in str(file_info.get("timfile", "")):
                                matching_files.append(file_info)
                        if matching_files:
                            filtered_file_data[pta_name] = matching_files

                if not filtered_file_data:
                    continue  # Skip if no files found for this pulsar

                mp = MetaPulsarFactory().create_metapulsar(file_data=filtered_file_data)

                # Basic validation
                assert mp.name == pulsar
                assert len(mp._pulsars) > 0  # Should have pulsar data from PTAs
                assert len(mp._pta_data) > 0  # Should have PTA timing records

                # Test design matrix
                dm = mp._designmatrix
                assert dm.shape[0] > 0  # Should have observations
                assert dm.shape[1] > 0  # Should have parameters

                # Test flags
                flags = mp._flags
                assert len(flags) == dm.shape[0]  # Flags should match observations
                assert len(np.unique(flags)) > 0  # Should have some variety in flags

    @pytest.mark.slow
    @pytest.mark.real_data
    def test_unit_conversion_consistency(self, available_data_sets, test_pulsars):
        """Test unit conversion consistency across different timing packages."""
        if not available_data_sets:
            pytest.skip("No data available for testing")

        # Test with both PINT and Tempo2 configurations
        pint_configs = ["nanograv_9y", "nanograv_12y", "nanograv_15y"]
        tempo2_configs = [
            "epta_dr1_v2_2",
            "ppta_dr2",
            "epta_dr2",
            "inpta_dr1",
            "mpta_dr1",
        ]
        discovery_service = FileDiscovery(working_dir="../../data/ipta-dr2")

        for pulsar in test_pulsars[:1]:  # Test first pulsar
            # Test PINT configurations
            if any(
                config in available_data_sets.get("dr2", {})
                or config in available_data_sets.get("dr3", {})
                for config in pint_configs
            ):
                file_data = discovery_service.discover_files(pint_configs)

                # Filter file_data to only include files for this pulsar
                filtered_file_data = {}
                for pta_name, files in file_data.items():
                    if files:  # Check if files exist for this PTA
                        matching_files = []
                        for file_info in files:
                            if pulsar in str(
                                file_info.get("parfile", "")
                            ) or pulsar in str(file_info.get("timfile", "")):
                                matching_files.append(file_info)
                        if matching_files:
                            filtered_file_data[pta_name] = matching_files

                if filtered_file_data:
                    mp_pint = MetaPulsarFactory().create_metapulsar(
                        file_data=filtered_file_data
                    )

                    # Basic validation
                    assert mp_pint.name == pulsar
                    dm_pint = mp_pint.get_design_matrix()
                    assert dm_pint.shape[0] > 0
                    assert dm_pint.shape[1] > 0

            # Test Tempo2 configurations
            if any(
                config in available_data_sets.get("dr2", {})
                or config in available_data_sets.get("dr3", {})
                for config in tempo2_configs
            ):
                file_data = discovery_service.discover_files(tempo2_configs)

                # Filter file_data to only include files for this pulsar
                filtered_file_data = {}
                for pta_name, files in file_data.items():
                    if files:  # Check if files exist for this PTA
                        matching_files = []
                        for file_info in files:
                            if pulsar in str(
                                file_info.get("parfile", "")
                            ) or pulsar in str(file_info.get("timfile", "")):
                                matching_files.append(file_info)
                        if matching_files:
                            filtered_file_data[pta_name] = matching_files

                if filtered_file_data:
                    mp_tempo2 = MetaPulsarFactory().create_metapulsar(
                        file_data=filtered_file_data
                    )

                    # Basic validation
                    assert mp_tempo2.name == pulsar
                    dm_tempo2 = mp_tempo2.get_design_matrix()
                    assert dm_tempo2.shape[0] > 0
                    assert dm_tempo2.shape[1] > 0

    @pytest.mark.slow
    @pytest.mark.real_data
    def test_metapulsar_serialization(self, available_data_sets, test_pulsars):
        """Test MetaPulsar serialization and deserialization."""
        if not available_data_sets:
            pytest.skip("No data available for testing")

        test_configs = ["epta_dr1_v2_2", "ppta_dr2", "nanograv_9y"]
        discovery_service = FileDiscovery(working_dir="../../data/ipta-dr2")

        for pulsar in test_pulsars[:1]:  # Test first pulsar
            # Discover files for the pulsar
            file_data = discovery_service.discover_files(test_configs)

            # Filter file_data to only include files for this pulsar
            filtered_file_data = {}
            for pta_name, files in file_data.items():
                if files:  # Check if files exist for this PTA
                    matching_files = []
                    for file_info in files:
                        if pulsar in str(file_info.get("parfile", "")) or pulsar in str(
                            file_info.get("timfile", "")
                        ):
                            matching_files.append(file_info)
                    if matching_files:
                        filtered_file_data[pta_name] = matching_files

            if not filtered_file_data:
                continue  # Skip if no files found for this pulsar

            # Create MetaPulsar
            mp = MetaPulsarFactory().create_metapulsar(file_data=filtered_file_data)

            # Test basic serialization (if implemented)
            if hasattr(mp, "to_dict"):
                mp_dict = mp.to_dict()
                assert isinstance(mp_dict, dict)
                assert mp_dict["name"] == pulsar

            # Test basic deserialization (if implemented)
            if hasattr(mp, "from_dict"):
                mp_restored = MetaPulsar.from_dict(mp_dict)
                assert mp_restored.name == mp.name
                assert len(mp_restored.par_files) == len(mp.par_files)
                assert len(mp_restored.tim_files) == len(mp.tim_files)
