"""Main MetaPulsar class for combining multi-PTA pulsar timing data."""

from __future__ import annotations

from itertools import groupby
from typing import List
import numpy as np
from loguru import logger

# Import Enterprise Pulsar classes
import enterprise.pulsar as ep

# Import PINT classes
from pint.models import TimingModel
from pint.toa import TOAs

# Import libstempo

# Import our supporting infrastructure
from .parameter_manager import ParameterManager
from .position_helpers import bj_name_from_pulsar


class MetaPulsar(ep.BasePulsar):
    """Elegant composite pulsar for multi-PTA data combination.

    This class combines pulsar timing data from multiple PTA collaborations
    into a unified object suitable for gravitational wave detection analysis.
    Inherits from enterprise.pulsar.BasePulsar for full Enterprise compatibility.

    Supports two combination strategies:
    - "consistent": Astrophysical consistency (modifies par files for consistency)
    - "composite": Multi-PTA composition (preserves original parameters)
    """

    def __init__(
        self,
        pulsars,
        *,  # Remove parfile_dicts parameter
        combination_strategy="consistent",
        combine_components: List[str] = [
            "astrometry",
            "spindown",
            "binary",
            "dispersion",
        ],
        add_dm_derivatives: bool = True,
        exclude_from_consistent: List[str] | tuple[str, ...] = ("DM",),
        sort=True,
    ):
        """Create MetaPulsar from multiple PTA pulsars.

        Args:
            pulsars: Dict mapping PTA names to pulsar data:
                - PINT: {pta: (pint_model, pint_toas)}
                - Tempo2: {pta: tempo2_psr}
            combination_strategy: Strategy for combining PTAs:
                - "consistent": Astrophysical consistency (modifies par files for consistency)
                - "composite": Multi-PTA composition (preserves original parameters)
            combine_components: List of components to make consistent (consistent strategy only):
                - "astrometry": Position and proper motion parameters
                - "spindown": Spin frequency and derivatives
                - "binary": Binary orbital parameters
                - "dispersion": Dispersion measure parameters
                Defaults to all components
            add_dm_derivatives: Whether to ensure DM1, DM2 are present (consistent strategy only)
            exclude_from_consistent: Canonical timing-model parameter names to keep
                PTA-specific even when their component is in combine_components.
                Defaults to ("DM",) so each PTA keeps its own reference DM while
                consistent dispersion still shares DM1/DM2. Pass an empty list to
                merge all parameters in selected components.
            sort: Whether to sort data by time
        """
        self._pulsars = pulsars
        # Extract parfile data from objects
        self._parfile_dicts = self._get_parfile_data(pulsars)
        self.combination_strategy = combination_strategy
        self.combine_components = (
            combine_components if combination_strategy == "consistent" else []
        )
        self.add_dm_derivatives = add_dm_derivatives
        self.exclude_from_consistent = exclude_from_consistent
        self._sort = sort  # BasePulsar handles sorting

        # Elegant initialization flow
        self._create_enterprise_pulsars()
        self._setup_parameters()
        self._combine_timing_data()
        self._build_design_matrix()
        self._remove_nonidentifiable_parameters()
        self._setup_position_and_planets()

        # BasePulsar handles sorting automatically
        self.sort_data()

        # Calculate canonical name from pulsar data using B-name preference logic
        self.name = self._get_pulsar_name(pulsars)

    def validate_consistency(self):
        """Validate that all PTAs contain the same pulsar.

        Returns:
            str: Pulsar name if consistent, raises ValueError if not
        """
        if not hasattr(self, "_epulsars") or self._epulsars is None:
            raise ValueError("No Enterprise Pulsars created yet")

        # Extract pulsar names from Enterprise Pulsars
        pulsar_names = []
        for pta, psr in self._epulsars.items():
            if hasattr(psr, "name") and psr.name and psr.name != "None":
                pulsar_names.append(psr.name)
            else:
                logger.warning(f"PTA {pta} pulsar has no valid name attribute")

        if not pulsar_names:
            raise ValueError("No pulsar names found")

        if not self._all_equal(pulsar_names):
            raise ValueError(f"Not all the same pulsar: {pulsar_names}")

        return pulsar_names[0]

    def _create_enterprise_pulsars(self):
        """Create Enterprise Pulsar objects from input data."""
        self._epulsars = {}
        pint_models, pint_toas, lt_pulsars = self._unpack_pulsar_data()

        if pint_models or lt_pulsars:
            self.name = self._validate_pulsar_consistency(pint_models, lt_pulsars)

            # Create Enterprise Pulsars from raw PINT objects
            for pta, (pmodel, ptoas) in zip(
                pint_models.keys(), zip(pint_models.values(), pint_toas.values())
            ):
                try:
                    self._epulsars[pta] = ep.PintPulsar(
                        ptoas, pmodel
                    )  # Use default planets=True
                except Exception as e:
                    logger.error(f"Failed to create PintPulsar for PTA {pta}: {e}")
                    raise

            # Create Enterprise Pulsars from raw Tempo2 objects
            for pta, lt_psr in lt_pulsars.items():
                try:
                    self._epulsars[pta] = ep.Tempo2Pulsar(lt_psr, planets=True)
                except Exception as e:
                    logger.error(f"Failed to create Tempo2Pulsar for PTA {pta}: {e}")
                    raise
        else:
            # All pulsars are already Enterprise Pulsars, get name from first one
            if self._epulsars:
                first_psr = next(iter(self._epulsars.values()))
                self.name = getattr(first_psr, "name", "unknown")
            else:
                self.name = "unknown"

    def _unpack_pulsar_data(self):
        """Unpack pulsars dictionary into PINT and libstempo objects."""
        lt_pulsars = {}
        pint_models = {}
        pint_toas = {}

        for pta, psritem in self._pulsars.items():
            # Check if it's a PINT tuple (model, toas)
            if isinstance(psritem, tuple) and len(psritem) == 2:
                pmodel, ptoas = psritem
                if isinstance(pmodel, TimingModel) and isinstance(ptoas, TOAs):
                    pint_models[pta] = pmodel
                    pint_toas[pta] = ptoas
                else:
                    raise TypeError(
                        f"Invalid PINT objects for {pta}: {type(pmodel)}, {type(ptoas)}"
                    )
            else:
                # Duck typing: anything else is treated as libstempo-like
                lt_pulsars[pta] = psritem

        return pint_models, pint_toas, lt_pulsars

    def _validate_pulsar_consistency(self, pint_models, lt_pulsars):
        """Validate single pulsar across all PTAs using standardized J-names."""
        pulsar_names = []

        # Extract standardized J-names from PINT models
        for m in pint_models.values():
            j_name = bj_name_from_pulsar(m, "J")
            pulsar_names.append(j_name)

        # Extract standardized J-names from libstempo pulsars
        for psr in lt_pulsars.values():
            j_name = bj_name_from_pulsar(psr, "J")
            pulsar_names.append(j_name)

        if not pulsar_names:
            raise ValueError("No valid pulsars found for validation")

        if not self._all_equal(pulsar_names):
            raise ValueError(f"Not all the same pulsar: {pulsar_names}")

        return pulsar_names[0]

    def _all_equal(self, iterable):
        """Check if all items in iterable are equal."""
        g = groupby(iterable)
        return next(g, True) and not next(g, False)

    def _get_libstempo_parfile_content(self, lt_psr):
        """Get parfile content as string from libstempo pulsar object.

        Args:
            lt_psr: libstempo tempopulsar object

        Returns:
            str: Parfile content as string
        """
        import tempfile
        import os

        # Create temporary file for libstempo to write to
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".par", delete=False
        ) as temp_file:
            temp_parfile = temp_file.name

        try:
            # Use libstempo's savepar method to write parfile content
            lt_psr.savepar(temp_parfile)

            # Read the content back
            with open(temp_parfile, "r") as f:
                return f.read()
        finally:
            # Clean up temporary file
            if os.path.exists(temp_parfile):
                os.unlink(temp_parfile)

    def _setup_parameters(self):
        """Setup parameter management using existing infrastructure."""
        # Get both PINT models and libstempo pulsars from the unpacked data
        pint_models, _, lt_pulsars = self._unpack_pulsar_data()

        # Convert individual merge flags to combine_components list
        # Use combine_components from constructor
        combine_components = self.combine_components

        # Create file data for ParameterManager
        file_data = {}

        # Handle PINT models
        for pta_name, model in pint_models.items():
            file_data[pta_name] = {
                "par": None,
                "par_content": model.as_parfile(),
                "timing_package": "pint",
            }

        # Handle libstempo pulsars
        for pta_name, lt_psr in lt_pulsars.items():
            parfile_content = self._get_libstempo_parfile_content(lt_psr)
            file_data[pta_name] = {
                "par": None,
                "par_content": parfile_content,
                "timing_package": "tempo2",
            }

        # Create ParameterManager for parameter mapping
        parameter_manager = ParameterManager(
            file_data=file_data,
            combine_components=combine_components,
            add_dm_derivatives=self.add_dm_derivatives,
            exclude_from_consistent=self.exclude_from_consistent,
        )

        mapping = parameter_manager.build_parameter_mappings()

        self._fitparameters = mapping.fitparameters
        self._setparameters = mapping.setparameters
        self.fitpars = list(self._fitparameters.keys())
        self.setpars = list(self._setparameters.keys())

        # Setup canonical parameter lists for each pulsar for
        # inter-pta consistent parameter lookups
        self._setup_canonical_parameters()

    def _setup_canonical_parameters(self):
        """Setup canonical parameter lists for each pulsar."""
        from .pint_helpers import resolve_parameter_alias

        for pta_name, psr in self._epulsars.items():
            # Create canonical versions of fitpars and setpars
            psr.fitpars_canonical = [resolve_parameter_alias(p) for p in psr.fitpars]
            psr.setpars_canonical = [resolve_parameter_alias(p) for p in psr.setpars]

    def _combine_timing_data(self):
        """Combine timing data from all PTAs."""

        def concat(attribute):
            """Concatenate attribute across all PTAs."""
            values = []
            for pta, psr in self._epulsars.items():
                if hasattr(psr, attribute):
                    values.append(getattr(psr, attribute))
            return np.concatenate(values) if values else np.array([])

        # Combine core timing data
        self._toas = concat("_toas")
        self._stoas = concat("_stoas")
        self._residuals = concat("_residuals")
        self._toaerrs = concat("_toaerrs")
        self._ssbfreqs = concat("_ssbfreqs")
        self._telescope = concat("_telescope")

        # Combine flags
        self._combine_flags()

    def _combine_flags(self):
        """Combine flags from all PTAs."""
        from collections import defaultdict

        pta_slice = self._get_pta_slices()
        flags = defaultdict(lambda: np.zeros(len(self._toas), dtype="U128"))

        for pta, psr in self._epulsars.items():
            flag_pta = False

            # Handle both dictionary and structured array formats for flags
            if isinstance(psr._flags, dict):
                # Dictionary format (legacy Enterprise Pulsars)
                for flag, flagvals in psr._flags.items():
                    flags[flag][pta_slice[pta]] = flagvals

                    # Handle PTA flag specifically
                    if flag == "pta" and not np.any(flagvals == ""):
                        flags[flag][pta_slice[pta]] = [
                            pta_flag.strip() for pta_flag in flagvals
                        ]
                        flag_pta = True
            else:
                if hasattr(psr._flags, "dtype") and psr._flags.dtype.names:
                    # Structured array with fields
                    for field_name in psr._flags.dtype.names:
                        flagvals = psr._flags[field_name]
                        flags[field_name][pta_slice[pta]] = flagvals

                        # Handle PTA flag specifically
                        if field_name == "pta" and not np.any(flagvals == ""):
                            flags[field_name][pta_slice[pta]] = [
                                pta_flag.strip() for pta_flag in flagvals
                            ]
                            flag_pta = True
                else:
                    # Use the flags property for Enterprise Pulsars
                    for flag, flagvals in psr.flags.items():
                        flags[flag][pta_slice[pta]] = flagvals

                        # Handle PTA flag specifically
                        if flag == "pta" and not np.any(flagvals == ""):
                            flags[flag][pta_slice[pta]] = [
                                pta_flag.strip() for pta_flag in flagvals
                            ]
                            flag_pta = True

            timing_package = self._get_timing_package(psr)
            flags["pta_dataset"][pta_slice[pta]] = pta
            flags["timing_package"][pta_slice[pta]] = timing_package

            if not flag_pta:
                flags["pta"][pta_slice[pta]] = pta

        # Store as numpy record array
        self._flags = np.zeros(
            len(self._toas), dtype=[(key, val.dtype) for key, val in flags.items()]
        )
        for key, val in flags.items():
            self._flags[key] = val

    def _get_pta_slices(self):
        """Get slice objects for each PTA in the combined data."""
        slices = {}
        start_idx = 0

        for pta, psr in self._epulsars.items():
            if hasattr(psr, "_toas"):
                end_idx = start_idx + len(psr._toas)
                slices[pta] = slice(start_idx, end_idx)
                start_idx = end_idx

        return slices

    def _get_timing_package(self, psr):
        """Determine timing package used by pulsar."""
        if hasattr(psr, "_pint_model"):
            return "pint"
        elif hasattr(psr, "_lt_pulsar"):
            return "tempo2"
        else:
            # Fallback: check Enterprise Pulsar type
            if hasattr(psr, "__class__"):
                class_name = psr.__class__.__name__
                if "PintPulsar" in class_name:
                    return "pint"
                elif "Tempo2Pulsar" in class_name:
                    return "tempo2"
            return "unknown"

    def _build_design_matrix(self):
        """Build combined design matrix with unit conversion."""
        n_toas = len(self._toas)
        n_params = len(self.fitpars)

        self._designmatrix = np.zeros((n_toas, n_params))

        for i, parname in enumerate(self.fitpars):
            self._designmatrix[:, i] = self._build_design_matrix_column(parname)

    def _remove_nonidentifiable_parameters(self):
        """Remove parameters with zero-information design matrix columns.

        Any parameter whose design matrix column sums to zero in absolute value
        is considered non-identifiable and is removed from:
        - self._designmatrix (column removed)
        - self._fitparameters (entry deleted)
        - self.fitpars (name removed)
        Additionally, if this MetaPulsar instance defines a meta-level
        self.fitpars_canonical list with the same ordering as self.fitpars,
        it will be updated consistently. This method does NOT modify per-PTA
        psr.fitpars_canonical lists to preserve alignment with their own
        underlying design matrices.
        """
        if self._designmatrix.size == 0:
            return

        # Compute per-column absolute sum to detect zero-information columns
        column_abs_sums = np.sum(np.abs(self._designmatrix), axis=0)

        if column_abs_sums.shape[0] != len(self.fitpars):
            # Safety check: inconsistent state; do nothing
            logger.error("Design matrix column count does not match fitpars length")
            raise ValueError("Design matrix column count does not match fitpars length")

        keep_indices = [i for i, s in enumerate(column_abs_sums) if s != 0.0]
        if len(keep_indices) == len(self.fitpars):
            # Nothing to remove
            return

        removed_indices = [i for i, s in enumerate(column_abs_sums) if s == 0.0]
        original_fitpars = list(self.fitpars)
        removed_parameters = [original_fitpars[i] for i in removed_indices]

        # Warn about each removed parameter
        for param_name in removed_parameters:
            logger.warning(
                f"Parameter '{param_name}' is non-identifiable (zero design matrix column); removing from fit"
            )

        # Update mapping dict
        for param_name in removed_parameters:
            del self._fitparameters[param_name]

        # Slice the design matrix to keep only identifiable parameters
        self._designmatrix = self._designmatrix[:, keep_indices]

        # Update fitpars to reflect kept parameters
        self.fitpars = [original_fitpars[i] for i in keep_indices]

    def _build_design_matrix_column(self, full_parname):
        """Build design matrix column for a single parameter."""
        pta_slices = self._get_pta_slices()
        n_toas = len(self._toas)
        column = np.zeros(n_toas)

        for pta, psr in self._epulsars.items():
            if pta not in pta_slices:
                continue

            slice_obj = pta_slices[pta]
            timing_package = self._get_timing_package(psr)

            # Get design matrix from Enterprise Pulsar
            if hasattr(psr, "_designmatrix"):
                dm = psr._designmatrix
                if full_parname in self._fitparameters:
                    for mapped_pta, mapped_param in self._fitparameters[
                        full_parname
                    ].items():
                        if mapped_pta == pta:
                            from .pint_helpers import resolve_parameter_alias

                            par_idx = psr.fitpars_canonical.index(
                                resolve_parameter_alias(mapped_param)
                            )
                            column[slice_obj] = dm[:, par_idx]
                            break

            # Apply unit conversion if needed
            column[slice_obj] = self._convert_design_matrix_units(
                column[slice_obj], full_parname, timing_package
            )

        return column

    def _convert_design_matrix_units(self, column, param_name, timing_package):
        """Convert design matrix units between PINT and libstempo."""
        import astropy.units as u

        # Complete units correction matching legacy system
        # TODO: move this to pint_helpers.py or another suitable location -- RvH
        units_correction = {
            ("elong", "tempo2"): (1.0 * u.second / u.radian).to(u.second / u.deg).value,
            ("elong", "pint"): 1.0,
            ("elat", "tempo2"): (1.0 * u.second / u.radian).to(u.second / u.deg).value,
            ("elat", "pint"): 1.0,
            ("lambda", "tempo2"): (1.0 * u.second / u.radian)
            .to(u.second / u.deg)
            .value,
            ("lambda", "pint"): 1.0,
            ("beta", "tempo2"): (1.0 * u.second / u.radian).to(u.second / u.deg).value,
            ("beta", "pint"): 1.0,
            ("raj", "tempo2"): (1.0 * u.second / u.radian)
            .to(u.second / u.hourangle)
            .value,
            ("raj", "pint"): 1.0,
            ("decj", "tempo2"): (1.0 * u.second / u.radian).to(u.second / u.deg).value,
            ("decj", "pint"): 1.0,
        }

        if param_name.lower() in ["raj", "decj", "elong", "elat", "lambda", "beta"]:
            key = (param_name.lower(), timing_package.lower())
            factor = units_correction.get(key, 1.0)
            return column * factor

        return column

    def _setup_position_and_planets(self):
        """Setup position and planetary data using PositionHelpers."""
        # Check if we have any pulsars
        if not self._epulsars:
            # No pulsars available, set default values
            self._raj = 0.0
            self._decj = 0.0
            self._pos = np.zeros((len(self._toas), 3))
            self._pos_t = np.zeros((len(self._toas), 3))
            self._planetssb = None
            self._sunssb = None
            self._pdist = None
            return

        # Get reference pulsar for position
        ref_psr = next(iter(self._epulsars.values()))

        # Set basic position attributes
        self._raj = ref_psr._raj
        self._decj = ref_psr._decj

        # Generate B/J name using position_helpers

        bj_name = bj_name_from_pulsar(ref_psr)
        logger.debug(f"Generated B/J name: {bj_name}")

        # Set position vector and time array
        pta_slice = self._get_pta_slices()
        self._pos = np.zeros((len(self._toas), 3))
        self._pos_t = np.zeros((len(self._toas), 3))
        self._planetssb = np.zeros((len(self._toas), 9, 6))
        self._sunssb = np.zeros((len(self._toas), 6))
        for pta, psr in self._epulsars.items():
            self._pos[pta_slice[pta], :] = psr._pos
            self._pos_t[pta_slice[pta], :] = psr._pos_t
            self._planetssb[pta_slice[pta], :, :] = psr._planetssb
            self._sunssb[pta_slice[pta], :] = psr._sunssb

        # Set planetary data
        self._pdist = ref_psr._pdist

        # Set pulsar sky position
        self._pos = ref_psr._pos

    def _get_parfile_data(self, pulsars):
        """Extract parfile data from pulsar objects."""
        parfile_dicts = {}
        for pta_name, pulsar in pulsars.items():
            try:
                if isinstance(pulsar, tuple) and len(pulsar) == 2:
                    # PINT tuple (model, toas) - extract from model
                    model, toas = pulsar
                    parfile_dicts[pta_name] = model.get_params_dict()
                else:
                    # Libstempo object - extract from parfile
                    parfile_dicts[pta_name] = pulsar.parfile
            except Exception as e:
                self.logger.error(
                    f"Failed to extract parfile data from {pta_name}: {e}"
                )
                parfile_dicts[pta_name] = {}
        return parfile_dicts

    def _get_pulsar_name(self, pulsars):
        """Get canonical pulsar name with B-name preference logic.

        Returns B-name if any PTA uses B-names internally, otherwise J-name.
        Matching is always done on J-name for coordinate-based identification.
        """
        from .position_helpers import bj_name_from_pulsar

        # Extract all pulsar names to check for B-name usage
        pulsar_names = self._extract_pulsar_names(pulsars)

        # Use first pulsar for coordinate-based name generation
        first_pulsar = next(iter(pulsars.values()))

        # Check if any PTA uses B-names and return appropriate name
        if any(name.startswith("B") and len(name) >= 6 for name in pulsar_names):
            return bj_name_from_pulsar(first_pulsar, "B")
        else:
            return bj_name_from_pulsar(first_pulsar, "J")

    def _extract_pulsar_names(self, pulsars):
        """Extract all pulsar names from PTA objects.

        Args:
            pulsars: Dictionary mapping PTA names to pulsar objects

        Returns:
            List of pulsar names from all PTAs
        """
        pulsar_names = []

        for pta_name, pulsar in pulsars.items():
            try:
                if isinstance(pulsar, tuple) and len(pulsar) == 2:
                    # PINT tuple (model, toas) - access PSR.value
                    model, toas = pulsar
                    pulsar_names.append(model.PSR.value)
                else:
                    # Libstempo object - access name property
                    pulsar_names.append(pulsar.name)
            except Exception as e:
                self.logger.error(f"Failed to extract pulsar name {pta_name}: {e}")
                raise e

        return pulsar_names


# We don't need to implement custom sorting since we inherit from BasePulsar
