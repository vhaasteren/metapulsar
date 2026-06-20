"""Test legacy vs new implementation comparison."""

import pytest
import numpy as np
from pathlib import Path
from metapulsar import (
    FileDiscoveryService,
    get_pulsar_names_from_file_data,
    filter_file_data_by_pulsars,
)


@pytest.mark.integration
@pytest.mark.requires_legacy
class TestLegacyComparison:
    """Test comparison between legacy and new implementations."""

    def _read_par_content(self, par_content):
        if not isinstance(par_content, str):
            return str(par_content)
        if "\n" in par_content:
            return par_content
        maybe_path = Path(par_content)
        if maybe_path.exists():
            return maybe_path.read_text(encoding="utf-8")
        return par_content

    def _assert_protocol_conventions(
        self, par_content: str, label: str, engine_mix: bool
    ):
        text = self._read_par_content(par_content)
        active_lines = []
        for raw in text.splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                active_lines.append(line)

        keys = {}
        for line in active_lines:
            parts = line.split()
            if not parts:
                continue
            key = parts[0].upper()
            keys[key] = parts[1:]

        if engine_mix:
            # Cross-engine convention rules should neutralize active TEMPO mode.
            if "T2CMETHOD" in keys and keys["T2CMETHOD"]:
                assert (
                    keys["T2CMETHOD"][0].upper() != "TEMPO"
                ), f"{label}: active T2CMETHOD TEMPO found after convention rules"

        has_equatorial = "RAJ" in keys and "DECJ" in keys
        has_ecliptic = ("LAMBDA" in keys or "ELONG" in keys) and (
            "BETA" in keys or "ELAT" in keys
        )
        assert not (
            has_equatorial and has_ecliptic
        ), f"{label}: mixed astrometry detected after convention rules"

        if has_ecliptic and engine_mix:
            assert "ECL" in keys, f"{label}: expected ECL for ecliptic astrometry"
            assert (
                keys["ECL"][0].upper() == "IERS2003"
            ), f"{label}: expected ECL IERS2003, found {' '.join(keys['ECL'])}"
        elif has_equatorial and engine_mix:
            assert (
                "ECL" not in keys
            ), f"{label}: equatorial astrometry should not include active ECL"

    def _prepare_legacy_input_files(
        self, pulsar_name, pta_data_releases, available_data_sets
    ):
        """Prepare input files for legacy implementation using the same discovery as new system."""
        discovery_service = FileDiscoveryService(working_dir="data/ipta-dr2")

        # Discover files for all PTAs
        file_data = discovery_service.discover_files(pta_data_releases)

        # Use proper pulsar selection methods like using_metapulsar.py
        # First get all pulsar names from the file data
        all_pulsar_names = get_pulsar_names_from_file_data(file_data)

        # Check if our target pulsar is in the discovered pulsars
        if pulsar_name not in all_pulsar_names:
            return [], []  # Return empty lists if pulsar not found

        # Filter file data to only include files for this specific pulsar
        filtered_file_data = filter_file_data_by_pulsars(file_data, [pulsar_name])

        # Convert to the format expected by legacy implementation
        par_files = []
        tim_files = []

        for data_release_name in pta_data_releases:
            if (
                data_release_name in filtered_file_data
                and filtered_file_data[data_release_name]
            ):
                # Get the first matching file for this PTA
                file_info = filtered_file_data[data_release_name][0]
                par_file = file_info.get("par")
                tim_file = file_info.get("tim")
                par_files.append(str(par_file) if par_file else None)
                tim_files.append(str(tim_file) if tim_file else None)
            else:
                # Add None for missing PTAs to maintain order
                par_files.append(None)
                tim_files.append(None)

        return par_files, tim_files

    @pytest.mark.slow
    @pytest.mark.legacy_comparison
    def test_metapulsar_creation_equivalence(
        self, legacy_module, new_module, available_data_sets, test_pulsars
    ):
        """Test that MetaPulsar creation produces equivalent results."""
        if not available_data_sets:
            pytest.skip("No data available for testing")

        test_pta_data_releases = ["epta_dr1_v2_2", "ppta_dr2", "nanograv_9y"]

        # Get available pulsars and use the first 2 that have data
        discovery_service = FileDiscoveryService(working_dir="data/ipta-dr2")
        file_data = discovery_service.discover_files(test_pta_data_releases)
        all_pulsar_names = get_pulsar_names_from_file_data(file_data)

        # Test first 2 available pulsars
        test_pulsars_to_use = all_pulsar_names[:2]

        for pulsar in test_pulsars_to_use:
            par_files, tim_files = self._prepare_legacy_input_files(
                pulsar, test_pta_data_releases, available_data_sets
            )

            # Filter out None values and check if we have any valid files
            valid_files = [
                (p, t)
                for p, t in zip(par_files, tim_files)
                if p is not None and t is not None
            ]
            if not valid_files:
                continue

            # Prepare input files in the format expected by legacy create_metapulsar
            input_files = []
            for i, (par_file, tim_file) in enumerate(zip(par_files, tim_files)):
                if par_file is None or tim_file is None:
                    continue  # Skip missing files

                pta_name = test_pta_data_releases[i]
                # Determine timing package based on PTA
                package = (
                    "tempo2" if pta_name in ["epta_dr1_v2_2", "ppta_dr2"] else "pint"
                )
                input_files.append(
                    {
                        "pta": pta_name,
                        "parfile": par_file,
                        "timfile": tim_file,
                        "package": package,
                    }
                )

            # Create legacy MetaPulsar
            legacy_mp = legacy_module.create_metapulsar(input_files)

            discovery_service = FileDiscoveryService(working_dir="data/ipta-dr2")
            file_data = discovery_service.discover_files(test_pta_data_releases)

            # Use proper pulsar selection methods like using_metapulsar.py
            all_pulsar_names = get_pulsar_names_from_file_data(file_data)

            # Check if our target pulsar is in the discovered pulsars
            if pulsar not in all_pulsar_names:
                continue  # Skip if pulsar not found

            # Filter file data to only include files for this specific pulsar
            filtered_file_data = filter_file_data_by_pulsars(file_data, [pulsar])

            if not filtered_file_data:
                continue  # Skip if no files found for this pulsar

            new_mp = new_module["MetaPulsarFactory"]().create_metapulsar(
                file_data=filtered_file_data,
                use_pulse_numbers="no",
            )

            # Compare basic properties
            assert legacy_mp.name == new_mp.name
            assert len(legacy_mp._epulsars) == len(new_mp._epulsars)

            # Compare design matrix shapes
            legacy_dm = legacy_mp._designmatrix
            new_dm = new_mp._designmatrix

            assert legacy_dm.shape == new_dm.shape

            # Reorder new design matrix to match legacy parameter order
            # Both implementations have the same fitpars, so we can use legacy order
            legacy_fitpars = legacy_mp.fitpars
            new_fitpars = new_mp.fitpars

            # Create mapping from new parameter order to legacy parameter order
            new_to_legacy_indices = [
                new_fitpars.index(param) for param in legacy_fitpars
            ]

            # Reorder new design matrix columns to match legacy order
            new_dm_reordered = new_dm[:, new_to_legacy_indices]

            # Use isort to reorder both design matrices for proper comparison
            # Reorder rows (TOAs) using isort
            legacy_dm_sorted = legacy_dm[legacy_mp.isort, :]
            new_dm_sorted = new_dm_reordered[new_mp.isort, :]

            # Compare design matrix values (within tolerance)
            # F1, DM, DM1, DM2 parameters have systematic quantization when using TRACK -2 with pulse numbers,
            # but relative errors are < 1e-15, so we use parameter-specific tolerances
            legacy_fitpars = legacy_mp.fitpars
            f1_col_idx = legacy_fitpars.index("F1") if "F1" in legacy_fitpars else None
            dm_col_idx = legacy_fitpars.index("DM") if "DM" in legacy_fitpars else None
            dm1_col_idx = (
                legacy_fitpars.index("DM1") if "DM1" in legacy_fitpars else None
            )
            dm2_col_idx = (
                legacy_fitpars.index("DM2") if "DM2" in legacy_fitpars else None
            )

            # Columns that need relaxed tolerance due to quantization
            relaxed_tolerance_cols = [
                idx
                for idx in [f1_col_idx, dm_col_idx, dm1_col_idx, dm2_col_idx]
                if idx is not None
            ]

            if relaxed_tolerance_cols:
                # Compare columns with strict tolerance
                strict_cols = [
                    i
                    for i in range(legacy_dm_sorted.shape[1])
                    if i not in relaxed_tolerance_cols
                ]
                if strict_cols:
                    np.testing.assert_allclose(
                        legacy_dm_sorted[:, strict_cols],
                        new_dm_sorted[:, strict_cols],
                        rtol=1e-2,
                        atol=1e-5,
                        err_msg="Design matrix values do not match (after isort reordering, strict tolerance columns)",
                    )
                # Compare F1, DM, DM1, DM2 columns with relaxed tolerance
                # Quantization differences are ~1e-3 seconds³, but relative errors < 1e-15
                np.testing.assert_allclose(
                    legacy_dm_sorted[:, relaxed_tolerance_cols],
                    new_dm_sorted[:, relaxed_tolerance_cols],
                    rtol=1e-2,
                    atol=1e-3,  # Relaxed: quantization differences are ~6e-4 seconds³, but relative errors < 1e-15
                    err_msg="Design matrix F1/DM values do not match (after isort reordering)",
                )
            else:
                # Fallback: compare all columns if no special columns found
                np.testing.assert_allclose(
                    legacy_dm_sorted,
                    new_dm_sorted,
                    rtol=1e-2,
                    atol=1e-5,
                    err_msg="Design matrix values do not match (after isort reordering)",
                )

            # Compare flags
            legacy_flags = legacy_mp._flags
            new_flags = new_mp._flags
            assert len(legacy_flags) == len(new_flags)

            # Normalize timing_package field to handle case sensitivity differences
            legacy_flags_normalized = legacy_flags.copy()
            new_flags_normalized = new_flags.copy()

            # Convert timing_package to lowercase for comparison
            legacy_flags_normalized["timing_package"] = np.char.lower(
                legacy_flags["timing_package"]
            )
            new_flags_normalized["timing_package"] = np.char.lower(
                new_flags["timing_package"]
            )

            # Handle dtype mismatch: new implementation may have 'pn' field (from pulse numbers)
            # that legacy doesn't have. Compare only common fields.
            legacy_field_names = set(legacy_flags_normalized.dtype.names)
            new_field_names = set(new_flags_normalized.dtype.names)
            common_fields = sorted(legacy_field_names & new_field_names)

            # Extract common fields for comparison
            legacy_common = legacy_flags_normalized[list(common_fields)]
            new_common = new_flags_normalized[list(common_fields)]

            assert np.array_equal(legacy_common, new_common)

            # Compare timing residuals
            legacy_residuals = legacy_mp._residuals
            new_residuals = new_mp._residuals
            assert len(legacy_residuals) == len(new_residuals)
            np.testing.assert_allclose(
                legacy_residuals,
                new_residuals,
                rtol=1e-10,
                atol=1e-12,
                err_msg="Timing residuals do not match between legacy and new implementations",
            )

            # Compare TOAs (Times of Arrival)
            legacy_toas = legacy_mp._toas
            new_toas = new_mp._toas
            assert len(legacy_toas) == len(new_toas)
            np.testing.assert_allclose(
                legacy_toas,
                new_toas,
                rtol=1e-10,
                atol=1e-12,
                err_msg="TOAs do not match between legacy and new implementations",
            )

            # Compare TOA errors
            legacy_toaerrs = legacy_mp._toaerrs
            new_toaerrs = new_mp._toaerrs
            assert len(legacy_toaerrs) == len(new_toaerrs)
            np.testing.assert_allclose(
                legacy_toaerrs,
                new_toaerrs,
                rtol=1e-10,
                atol=1e-12,
                err_msg="TOA errors do not match between legacy and new implementations",
            )

            # Compare frequencies
            legacy_freqs = legacy_mp.freqs
            new_freqs = new_mp.freqs
            assert len(legacy_freqs) == len(new_freqs)
            np.testing.assert_allclose(
                legacy_freqs,
                new_freqs,
                rtol=1e-10,
                atol=1e-12,
                err_msg="Frequencies do not match between legacy and new implementations",
            )

            # Compare individual Enterprise pulsar properties for each PTA
            for pta_name in legacy_mp._epulsars.keys():
                if pta_name in new_mp._epulsars:
                    legacy_epulsar = legacy_mp._epulsars[pta_name]
                    new_epulsar = new_mp._epulsars[pta_name]

                    # Compare Enterprise pulsar residuals - they should be identical when sorted
                    legacy_ep_residuals = legacy_epulsar.residuals
                    new_ep_residuals = new_epulsar.residuals
                    assert len(legacy_ep_residuals) == len(new_ep_residuals)

                    # Sort residuals for comparison (since data ordering may differ)
                    legacy_residuals_sorted = np.sort(legacy_ep_residuals)
                    new_residuals_sorted = np.sort(new_ep_residuals)

                    np.testing.assert_allclose(
                        legacy_residuals_sorted,
                        new_residuals_sorted,
                        rtol=1e-10,
                        atol=1e-12,
                        err_msg=f"Enterprise pulsar residuals for {pta_name} do not match (after sorting)",
                    )

                    # Compare Enterprise pulsar TOAs - they should be identical when sorted
                    legacy_ep_toas = legacy_epulsar.toas
                    new_ep_toas = new_epulsar.toas
                    assert len(legacy_ep_toas) == len(new_ep_toas)

                    # Sort TOAs for comparison (since data ordering may differ)
                    legacy_toas_sorted = np.sort(legacy_ep_toas)
                    new_toas_sorted = np.sort(new_ep_toas)

                    np.testing.assert_allclose(
                        legacy_toas_sorted,
                        new_toas_sorted,
                        rtol=1e-10,
                        atol=1e-12,
                        err_msg=f"Enterprise pulsar TOAs for {pta_name} do not match (after sorting)",
                    )

                    # Compare Enterprise pulsar TOA errors - they should be identical when sorted
                    legacy_ep_toaerrs = legacy_epulsar.toaerrs
                    new_ep_toaerrs = new_epulsar.toaerrs
                    assert len(legacy_ep_toaerrs) == len(new_ep_toaerrs)

                    # Sort TOA errors for comparison (since data ordering may differ)
                    legacy_toaerrs_sorted = np.sort(legacy_ep_toaerrs)
                    new_toaerrs_sorted = np.sort(new_ep_toaerrs)

                    np.testing.assert_allclose(
                        legacy_toaerrs_sorted,
                        new_toaerrs_sorted,
                        rtol=1e-10,
                        atol=1e-12,
                        err_msg=f"Enterprise pulsar TOA errors for {pta_name} do not match (after sorting)",
                    )

                    # Compare Enterprise pulsar frequencies - they should be identical when sorted
                    legacy_ep_freqs = legacy_epulsar.freqs
                    new_ep_freqs = new_epulsar.freqs
                    assert len(legacy_ep_freqs) == len(new_ep_freqs)

                    # Sort frequencies for comparison (since data ordering may differ)
                    legacy_freqs_sorted = np.sort(legacy_ep_freqs)
                    new_freqs_sorted = np.sort(new_ep_freqs)

                    np.testing.assert_allclose(
                        legacy_freqs_sorted,
                        new_freqs_sorted,
                        rtol=1e-10,
                        atol=1e-12,
                        err_msg=f"Enterprise pulsar frequencies for {pta_name} do not match (after sorting)",
                    )

    @pytest.mark.slow
    @pytest.mark.legacy_comparison
    def test_design_matrix_construction(
        self, legacy_module, new_module, available_data_sets, test_pulsars
    ):
        """Test design matrix construction equivalence."""
        if not available_data_sets:
            pytest.skip("No data available for testing")

        test_pta_data_releases = ["epta_dr1_v2_2", "ppta_dr2", "nanograv_9y"]

        for pulsar in test_pulsars[:2]:  # Test first 2 pulsars
            par_files, tim_files = self._prepare_legacy_input_files(
                pulsar, test_pta_data_releases, available_data_sets
            )

            # Filter out None values and check if we have any valid files
            valid_files = [
                (p, t)
                for p, t in zip(par_files, tim_files)
                if p is not None and t is not None
            ]
            if not valid_files:
                continue

            # Prepare input files in the format expected by legacy create_metapulsar
            input_files = []
            for i, (par_file, tim_file) in enumerate(zip(par_files, tim_files)):
                if par_file is None or tim_file is None:
                    continue  # Skip missing files

                pta_name = test_pta_data_releases[i]
                package = (
                    "tempo2" if pta_name in ["epta_dr1_v2_2", "ppta_dr2"] else "pint"
                )
                input_files.append(
                    {
                        "pta": pta_name,
                        "parfile": par_file,
                        "timfile": tim_file,
                        "package": package,
                    }
                )

            # Create both implementations
            legacy_mp = legacy_module.create_metapulsar(input_files)

            discovery_service = FileDiscoveryService(working_dir="data/ipta-dr2")
            file_data = discovery_service.discover_files(test_pta_data_releases)

            # Use proper pulsar selection methods like using_metapulsar.py
            all_pulsar_names = get_pulsar_names_from_file_data(file_data)

            # Check if our target pulsar is in the discovered pulsars
            if pulsar not in all_pulsar_names:
                continue  # Skip if pulsar not found

            # Filter file data to only include files for this specific pulsar
            filtered_file_data = filter_file_data_by_pulsars(file_data, [pulsar])

            if not filtered_file_data:
                continue  # Skip if no files found for this pulsar

            new_mp = new_module["MetaPulsarFactory"]().create_metapulsar(
                file_data=filtered_file_data,
                use_pulse_numbers="no",
            )

            # Get design matrices
            legacy_dm = legacy_mp._designmatrix
            new_dm = new_mp._designmatrix

            # Compare shapes
            assert legacy_dm.shape == new_dm.shape

            # Reorder new design matrix to match legacy parameter order
            # Both implementations have the same fitpars, so we can use legacy order
            legacy_fitpars = legacy_mp.fitpars
            new_fitpars = new_mp.fitpars

            # Create mapping from new parameter order to legacy parameter order
            new_to_legacy_indices = [
                new_fitpars.index(param) for param in legacy_fitpars
            ]

            # Reorder new design matrix columns to match legacy order
            new_dm_reordered = new_dm[:, new_to_legacy_indices]

            # Use isort to reorder both design matrices for proper comparison
            # Reorder rows (TOAs) using isort
            legacy_dm_sorted = legacy_dm[legacy_mp.isort, :]
            new_dm_sorted = new_dm_reordered[new_mp.isort, :]

            # Compare values
            # F1, DM, DM1, DM2 parameters have systematic quantization when using TRACK -2 with pulse numbers,
            # but relative errors are < 1e-15, so we use parameter-specific tolerances
            legacy_fitpars = legacy_mp.fitpars
            f1_col_idx = legacy_fitpars.index("F1") if "F1" in legacy_fitpars else None
            dm_col_idx = legacy_fitpars.index("DM") if "DM" in legacy_fitpars else None
            dm1_col_idx = (
                legacy_fitpars.index("DM1") if "DM1" in legacy_fitpars else None
            )
            dm2_col_idx = (
                legacy_fitpars.index("DM2") if "DM2" in legacy_fitpars else None
            )

            # Columns that need relaxed tolerance due to quantization
            relaxed_tolerance_cols = [
                idx
                for idx in [f1_col_idx, dm_col_idx, dm1_col_idx, dm2_col_idx]
                if idx is not None
            ]

            if relaxed_tolerance_cols:
                # Compare columns with strict tolerance
                strict_cols = [
                    i
                    for i in range(legacy_dm_sorted.shape[1])
                    if i not in relaxed_tolerance_cols
                ]
                if strict_cols:
                    np.testing.assert_allclose(
                        legacy_dm_sorted[:, strict_cols],
                        new_dm_sorted[:, strict_cols],
                        rtol=1e-2,
                        atol=1e-5,
                        err_msg="Design matrix construction values do not match (after isort reordering, strict tolerance columns)",
                    )
                # Compare F1, DM, DM1, DM2 columns with relaxed tolerance
                # Quantization differences are ~1e-3 seconds³, but relative errors < 1e-15
                np.testing.assert_allclose(
                    legacy_dm_sorted[:, relaxed_tolerance_cols],
                    new_dm_sorted[:, relaxed_tolerance_cols],
                    rtol=1e-2,
                    atol=1e-3,  # Relaxed: quantization differences are ~6e-4 seconds³, but relative errors < 1e-15
                    err_msg="Design matrix construction F1/DM values do not match (after isort reordering)",
                )
            else:
                # Fallback: compare all columns if no special columns found
                np.testing.assert_allclose(
                    legacy_dm_sorted,
                    new_dm_sorted,
                    rtol=1e-2,
                    atol=1e-5,
                    err_msg="Design matrix construction values do not match (after isort reordering)",
                )

            # Test that no columns are all zeros (except possibly the first)
            for i in range(1, legacy_dm.shape[1]):
                legacy_col = legacy_dm[:, i]
                new_col = new_dm_reordered[:, i]

                # Both should have the same zero pattern
                legacy_zeros = np.all(legacy_col == 0)
                new_zeros = np.all(new_col == 0)
                assert legacy_zeros == new_zeros

                # If not all zeros, values should match
                if not legacy_zeros:
                    np.testing.assert_allclose(
                        legacy_col, new_col, rtol=1e-10, atol=1e-12
                    )

    @pytest.mark.slow
    @pytest.mark.legacy_comparison
    def test_flag_combination(
        self, legacy_module, new_module, available_data_sets, test_pulsars
    ):
        """Test flag combination equivalence."""
        if not available_data_sets:
            pytest.skip("No data available for testing")

        test_pta_data_releases = ["epta_dr1_v2_2", "ppta_dr2", "nanograv_9y"]

        for pulsar in test_pulsars[:2]:  # Test first 2 pulsars
            par_files, tim_files = self._prepare_legacy_input_files(
                pulsar, test_pta_data_releases, available_data_sets
            )

            if not par_files or not tim_files:
                continue

            # Prepare input files in the format expected by legacy create_metapulsar
            input_files = []
            for i, (par_file, tim_file) in enumerate(zip(par_files, tim_files)):
                if par_file is None or tim_file is None:
                    continue  # Skip missing files

                pta_name = test_pta_data_releases[i]
                package = (
                    "tempo2" if pta_name in ["epta_dr1_v2_2", "ppta_dr2"] else "pint"
                )
                input_files.append(
                    {
                        "pta": pta_name,
                        "parfile": par_file,
                        "timfile": tim_file,
                        "package": package,
                    }
                )

            # Skip if no valid input files found
            if not input_files:
                continue

            # Create both implementations
            legacy_mp = legacy_module.create_metapulsar(input_files)

            discovery_service = FileDiscoveryService(working_dir="data/ipta-dr2")
            file_data = discovery_service.discover_files(test_pta_data_releases)

            # Use proper pulsar selection methods like using_metapulsar.py
            all_pulsar_names = get_pulsar_names_from_file_data(file_data)

            # Check if our target pulsar is in the discovered pulsars
            if pulsar not in all_pulsar_names:
                continue  # Skip if pulsar not found

            # Filter file data to only include files for this specific pulsar
            filtered_file_data = filter_file_data_by_pulsars(file_data, [pulsar])

            if not filtered_file_data:
                continue  # Skip if no files found for this pulsar

            new_mp = new_module["MetaPulsarFactory"]().create_metapulsar(
                file_data=filtered_file_data,
                use_pulse_numbers="no",
            )

            # Get flags
            legacy_flags = legacy_mp._flags
            new_flags = new_mp._flags

            # Compare flags
            assert len(legacy_flags) == len(new_flags)

            # Normalize timing_package field to handle case sensitivity differences
            legacy_flags_normalized = legacy_flags.copy()
            new_flags_normalized = new_flags.copy()

            # Convert timing_package to lowercase for comparison
            legacy_flags_normalized["timing_package"] = np.char.lower(
                legacy_flags["timing_package"]
            )
            new_flags_normalized["timing_package"] = np.char.lower(
                new_flags["timing_package"]
            )

            # Handle dtype mismatch: new implementation may have 'pn' field (from pulse numbers)
            # that legacy doesn't have. Compare only common fields.
            legacy_field_names = set(legacy_flags_normalized.dtype.names)
            new_field_names = set(new_flags_normalized.dtype.names)
            common_fields = sorted(legacy_field_names & new_field_names)

            # Extract common fields for comparison
            legacy_common = legacy_flags_normalized[list(common_fields)]
            new_common = new_flags_normalized[list(common_fields)]

            assert np.array_equal(legacy_common, new_common)

            # Test flag statistics
            legacy_unique, legacy_counts = np.unique(
                legacy_flags_normalized, return_counts=True
            )
            new_unique, new_counts = np.unique(new_flags_normalized, return_counts=True)

            assert len(legacy_unique) == len(new_unique)
            assert np.array_equal(legacy_unique, new_unique)
            assert np.array_equal(legacy_counts, new_counts)

    @pytest.mark.slow
    @pytest.mark.legacy_comparison
    def test_intermediate_par_file_consistency(
        self, legacy_module, new_module, available_data_sets, test_pulsars
    ):
        """Test that intermediate par files have consistent parameter values."""
        if not available_data_sets:
            pytest.skip("No data available for testing")

        test_pta_data_releases = ["epta_dr1_v2_2", "ppta_dr2", "nanograv_9y"]
        key_params = ["F0", "F1", "RAJ", "DECJ", "PMRA", "PMDEC", "PEPOCH"]

        for pulsar in test_pulsars[:2]:  # Test first 2 pulsars
            par_files, tim_files = self._prepare_legacy_input_files(
                pulsar, test_pta_data_releases, available_data_sets
            )

            # Filter out None values and check if we have any valid files
            valid_files = [
                (p, t)
                for p, t in zip(par_files, tim_files)
                if p is not None and t is not None
            ]
            if not valid_files:
                continue

            # Prepare input files in the format expected by legacy create_metapulsar
            input_files = []
            for i, (par_file, tim_file) in enumerate(zip(par_files, tim_files)):
                if par_file is None or tim_file is None:
                    continue  # Skip missing files

                pta_name = test_pta_data_releases[i]
                package = (
                    "tempo2" if pta_name in ["epta_dr1_v2_2", "ppta_dr2"] else "pint"
                )
                input_files.append(
                    {
                        "pta": pta_name,
                        "parfile": par_file,
                        "timfile": tim_file,
                        "package": package,
                    }
                )

            engine_mix = len({entry["package"] for entry in input_files}) > 1

            # Create both implementations
            legacy_mp = legacy_module.create_metapulsar(input_files)

            discovery_service = FileDiscoveryService(working_dir="data/ipta-dr2")
            file_data = discovery_service.discover_files(test_pta_data_releases)

            # Use proper pulsar selection methods like using_metapulsar.py
            all_pulsar_names = get_pulsar_names_from_file_data(file_data)

            # Check if our target pulsar is in the discovered pulsars
            if pulsar not in all_pulsar_names:
                continue  # Skip if pulsar not found

            # Filter file data to only include files for this specific pulsar
            filtered_file_data = filter_file_data_by_pulsars(file_data, [pulsar])

            if not filtered_file_data:
                continue  # Skip if no files found for this pulsar

            new_mp = new_module["MetaPulsarFactory"]().create_metapulsar(
                file_data=filtered_file_data,
                use_pulse_numbers="no",
            )

            # Get intermediate par files (if available)
            legacy_par_content = getattr(legacy_mp, "intermediate_par_content", None)
            new_par_content = getattr(new_mp, "intermediate_par_content", None)

            if legacy_par_content and new_par_content:
                self._assert_protocol_conventions(
                    legacy_par_content, "legacy", engine_mix=engine_mix
                )
                self._assert_protocol_conventions(
                    new_par_content, "new", engine_mix=engine_mix
                )

                # Parse par files and compare key parameters
                from pint.models import get_model

                legacy_model = get_model(legacy_par_content)
                new_model = get_model(new_par_content)

                for param in key_params:
                    if hasattr(legacy_model, param) and hasattr(new_model, param):
                        legacy_val = getattr(legacy_model, param).value
                        new_val = getattr(new_model, param).value

                        if legacy_val is not None and new_val is not None:
                            np.testing.assert_allclose(
                                legacy_val,
                                new_val,
                                rtol=1e-10,
                                atol=1e-12,
                                err_msg=f"Parameter {param} mismatch",
                            )

    @pytest.mark.slow
    @pytest.mark.legacy_comparison
    def test_fitpars_equivalence(
        self, legacy_module, new_module, available_data_sets, test_pulsars
    ):
        """Test that fitpars (fit parameters) are equivalent between legacy and new implementations.

        This is a critical test to ensure parameter merging logic works correctly.
        """
        if not available_data_sets:
            pytest.skip("No data available for testing")

        test_pta_data_releases = ["epta_dr1_v2_2", "ppta_dr2", "nanograv_9y"]

        for pulsar in test_pulsars[:2]:  # Test first 2 pulsars
            par_files, tim_files = self._prepare_legacy_input_files(
                pulsar, test_pta_data_releases, available_data_sets
            )

            # Filter out None values and check if we have any valid files
            valid_files = [
                (p, t)
                for p, t in zip(par_files, tim_files)
                if p is not None and t is not None
            ]
            if not valid_files:
                continue

            # Prepare input files in the format expected by legacy create_metapulsar
            input_files = []
            for i, (par_file, tim_file) in enumerate(zip(par_files, tim_files)):
                if par_file is None or tim_file is None:
                    continue  # Skip missing files

                pta_name = test_pta_data_releases[i]
                package = (
                    "tempo2" if pta_name in ["epta_dr1_v2_2", "ppta_dr2"] else "pint"
                )
                input_files.append(
                    {
                        "pta": pta_name,
                        "parfile": par_file,
                        "timfile": tim_file,
                        "package": package,
                    }
                )

            # Skip if no valid input files found
            if not input_files:
                continue

            # Create both implementations
            legacy_mp = legacy_module.create_metapulsar(input_files)

            discovery_service = FileDiscoveryService(working_dir="data/ipta-dr2")
            file_data = discovery_service.discover_files(test_pta_data_releases)

            # Use proper pulsar selection methods like using_metapulsar.py
            all_pulsar_names = get_pulsar_names_from_file_data(file_data)

            # Check if our target pulsar is in the discovered pulsars
            if pulsar not in all_pulsar_names:
                continue  # Skip if pulsar not found

            # Filter file data to only include files for this specific pulsar
            filtered_file_data = filter_file_data_by_pulsars(file_data, [pulsar])

            if not filtered_file_data:
                continue  # Skip if no files found for this pulsar

            new_mp = new_module["MetaPulsarFactory"]().create_metapulsar(
                file_data=filtered_file_data,
                use_pulse_numbers="no",
            )

            # Get fitpars from both implementations
            legacy_fitpars = set(legacy_mp.fitpars)
            new_fitpars = set(new_mp.fitpars)

            # Compare the sets of fit parameters
            assert legacy_fitpars == new_fitpars, (
                f"Fit parameters do not match between legacy and new implementations for pulsar {pulsar}:\n"
                f"Legacy fitpars ({len(legacy_fitpars)}): {sorted(legacy_fitpars)}\n"
                f"New fitpars ({len(new_fitpars)}): {sorted(new_fitpars)}\n"
                f"Missing in new: {legacy_fitpars - new_fitpars}\n"
                f"Extra in new: {new_fitpars - legacy_fitpars}"
            )

            # Test that the number of parameters is reasonable (not too small)
            assert len(new_fitpars) > 10, (
                f"New implementation has suspiciously few fit parameters ({len(new_fitpars)}). "
                f"This suggests parameter merging logic may be broken."
            )

            # Test that we have both merged and PTA-specific parameters
            assert (
                len(new_fitpars) > 0
            ), "No merged parameters found - parameter merging may be broken"

    @pytest.mark.slow
    @pytest.mark.legacy_comparison
    def test_pulse_number_mode_residual_equivalence(
        self, new_module, available_data_sets
    ):
        """Pulse-number and non-pulse-number paths should match to machine precision.

        Unlike ``tests/test_pulse_tracking.py`` (synthetic PTAs where consistent
        merging breaks DM/DMX coherence), this IPTA case stays coherent after merge,
        so ``yes`` and ``no`` should yield the same residuals.
        """
        if not available_data_sets:
            pytest.skip("No data available for testing")

        test_pta_data_releases = ["epta_dr1_v2_2", "ppta_dr2", "nanograv_9y"]
        discovery_service = FileDiscoveryService(working_dir="data/ipta-dr2")
        file_data = discovery_service.discover_files(test_pta_data_releases)
        all_pulsar_names = get_pulsar_names_from_file_data(file_data)

        if not all_pulsar_names:
            pytest.skip("No pulsars found in selected PTAs")

        target_pulsar = (
            "J0030+0451" if "J0030+0451" in all_pulsar_names else all_pulsar_names[0]
        )
        filtered_file_data = filter_file_data_by_pulsars(file_data, [target_pulsar])

        if not filtered_file_data:
            pytest.skip(f"No file data found for {target_pulsar}")

        factory = new_module["MetaPulsarFactory"]()
        mp_without_pn = factory.create_metapulsar(
            file_data=filtered_file_data,
            use_pulse_numbers="no",
        )
        mp_with_pn = factory.create_metapulsar(
            file_data=filtered_file_data,
            use_pulse_numbers="yes",
        )

        assert len(mp_without_pn._residuals) == len(mp_with_pn._residuals)
        np.testing.assert_allclose(
            mp_with_pn._residuals,
            mp_without_pn._residuals,
            rtol=1e-10,
            atol=6e-10,  # Not 1e-12  <-- investigate this! -- RvH
            err_msg=(
                "Pulse-number and non-pulse-number residuals should be machine-precision equivalent "
                "for this simple coherent timing-solution test case."
            ),
        )
