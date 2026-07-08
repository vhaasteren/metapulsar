"""Main MetaPulsar class for combining multi-PTA pulsar timing data."""

from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import List, Mapping
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


@dataclass(frozen=True)
class SessionFiles:
    """Durable timing-package input files for one PTA session."""

    par_path: Path
    tim_path: Path
    timing_package: str


class MetaPulsar:
    """Elegant composite pulsar for multi-PTA data combination.

    This class combines pulsar timing data from multiple PTA collaborations
    into a unified object suitable for gravitational wave detection analysis.
    This class implements the EnterprisePulsar-like surface directly.

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
        session_files: dict[str, dict] | None = None,
        clock_dir: str | Path | None = None,
        sort=False,
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
            sort: Whether to sort data by time
        """
        self._pulsars = pulsars
        self.combination_strategy = combination_strategy
        self.combine_components = (
            combine_components if combination_strategy == "consistent" else []
        )
        self.add_dm_derivatives = add_dm_derivatives
        # Retained session par/tim must be available before reference-theta lookup:
        # pulse-number tracking uses temporary TRACK -2 par paths that are deleted
        # after libstempo construction.
        self._session_files = self._normalize_session_files(session_files)
        self._parfile_dicts = self._get_parfile_data(pulsars)
        self._clock_dir = None if clock_dir is None else Path(clock_dir)
        self._sort = sort
        self._timing_backend_cache = {}
        self._pint_model_cache = None

        # Elegant initialization flow
        self._create_enterprise_pulsars()
        self._setup_parameters()
        self._combine_timing_data()
        self._build_design_matrix()
        self._remove_nonidentifiable_parameters()
        self._setup_position_and_planets()

        self.sort_data()

        # Calculate canonical name from pulsar data using B-name preference logic
        self.name = self._get_pulsar_name(pulsars)

    @staticmethod
    def _normalize_session_files(
        session_files: dict[str, dict] | None,
    ) -> dict[str, SessionFiles]:
        if session_files is None:
            return {}
        normalized: dict[str, SessionFiles] = {}
        for pta_name, files in session_files.items():
            normalized[pta_name] = SessionFiles(
                par_path=Path(files["par_path"]),
                tim_path=Path(files["tim_path"]),
                timing_package=str(files["timing_package"]),
            )
        return normalized

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
                        ptoas, pmodel, sort=False
                    )  # Use default planets=True
                except Exception as e:
                    logger.error(f"Failed to create PintPulsar for PTA {pta}: {e}")
                    raise

            # Create Enterprise Pulsars from raw Tempo2 objects
            for pta, lt_psr in lt_pulsars.items():
                try:
                    self._epulsars[pta] = ep.Tempo2Pulsar(
                        lt_psr, sort=False, planets=True
                    )
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
            }

        # Handle libstempo pulsars
        for pta_name, lt_psr in lt_pulsars.items():
            parfile_content = self._get_libstempo_parfile_content(lt_psr)
            file_data[pta_name] = {
                "par": None,
                "par_content": parfile_content,
            }

        # Create ParameterManager for parameter mapping
        parameter_manager = ParameterManager(
            file_data=file_data,
            combine_components=combine_components,
            add_dm_derivatives=self.add_dm_derivatives,
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

    def _parfile_content_for_pta(self, pta_name: str) -> str:
        """Return canonical parfile content for one PTA.

        Priority:
        1. Pulsar-retained session par file (exact runtime input, possibly TRACK-modified)
        2. In-memory PINT model as_parfile()
        3. libstempo savepar() dump
        """
        session = getattr(self, "_session_files", {}).get(pta_name)
        if session is not None and session.par_path.is_file():
            return session.par_path.read_text(encoding="utf-8")

        source = self._pulsars.get(pta_name)
        if isinstance(source, tuple) and len(source) == 2:
            model = source[0]
            if isinstance(model, TimingModel):
                return model.as_parfile()
        if source is not None:
            # Test doubles and legacy callers may surface dict-like par metadata
            # via ``pulsar.parfile``; normalize that into valid parfile text.
            parfile_attr = getattr(source, "parfile", None)
            if isinstance(parfile_attr, dict):
                from .pint_helpers import dict_to_parfile_string

                return dict_to_parfile_string(parfile_attr, format="pint")
            if isinstance(parfile_attr, str):
                if "\n" in parfile_attr or parfile_attr.lstrip().startswith("PSR"):
                    return parfile_attr
                par_path = Path(parfile_attr)
                if par_path.is_file():
                    return par_path.read_text(encoding="utf-8")
            return self._get_libstempo_parfile_content(source)

        raise KeyError(f"No PTA source available for {pta_name!r}")

    def _get_parfile_data(self, pulsars):
        """Extract per-PTA parfile dictionaries for reference-theta lookup.

        This uses ``_parfile_content_for_pta`` as the single source of truth for
        non-PINT sessions, so retained session par files (e.g. TRACK-modified
        pulse-number tracking inputs) are preferred over transient object paths.
        """
        from .pint_helpers import create_pint_model

        if not hasattr(self, "_pulsars") or not self._pulsars:
            self._pulsars = pulsars

        parfile_dicts = {}
        for pta_name, pulsar in pulsars.items():
            if isinstance(pulsar, tuple) and len(pulsar) == 2:
                # PINT tuple (model, toas) - preserve direct metadata extraction.
                model, _ = pulsar
                parfile_dicts[pta_name] = model.get_params_dict()
                continue

            par_attr = getattr(pulsar, "parfile", None)
            if isinstance(par_attr, dict):
                # Test doubles and legacy paths may expose dict-like metadata.
                parfile_dicts[pta_name] = par_attr
                continue

            par_content = self._parfile_content_for_pta(pta_name)
            try:
                parfile_dicts[pta_name] = create_pint_model(
                    par_content
                ).get_params_dict()
            except Exception as exc:
                raise ValueError(
                    "Failed to extract parfile metadata for "
                    f"pta={pta_name!r} from canonical par content: {exc}"
                ) from exc
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

    def _invalidate_timing_caches(self) -> None:
        """Invalidate timing API caches after pulsar-state mutations."""
        self._timing_backend_cache.clear()
        self._pint_model_cache = None

    def _reference_pta_name(self) -> str:
        """Return deterministic reference PTA key for pulsar-level metadata."""
        if not self._epulsars:
            raise ValueError("No PTA sessions are available on this MetaPulsar")
        return next(iter(self._epulsars.keys()))

    @staticmethod
    def _stringify_par_value(value) -> str:
        """Convert parfile values to stable decimal-like strings when possible."""
        if value is None:
            return "0.0"
        if isinstance(value, str):
            return value
        if hasattr(value, "value"):
            return str(getattr(value, "value"))
        if isinstance(value, (list, tuple)) and value:
            return str(value[0])
        return str(value)

    def _session_theta_exact(
        self, pta_name: str, session_fitpars: tuple[str, ...]
    ) -> dict[str, str]:
        """Build reference-theta exact mapping for one session using parfile metadata."""
        from .pint_helpers import create_pint_model, resolve_parameter_alias

        par_source = self._parfile_dicts.get(pta_name, {})
        pint_model = None
        if not isinstance(par_source, dict):
            pint_model = create_pint_model(self._parfile_content_for_pta(pta_name))
            par_source = {}

        exact: dict[str, str] = {}
        for name in session_fitpars:
            mapped_name = self._fitparameters.get(name, {}).get(pta_name, name)
            # Prefer exact key, then canonical alias of mapped name.
            if mapped_name in par_source:
                exact[name] = self._stringify_par_value(par_source[mapped_name])
            else:
                alias = resolve_parameter_alias(mapped_name)
                if alias in par_source:
                    exact[name] = self._stringify_par_value(par_source[alias])
                elif pint_model is not None and hasattr(pint_model, alias):
                    param = getattr(pint_model, alias)
                    exact[name] = self._stringify_par_value(
                        getattr(param, "value", param)
                    )
                elif pint_model is not None and hasattr(pint_model, mapped_name):
                    param = getattr(pint_model, mapped_name)
                    exact[name] = self._stringify_par_value(
                        getattr(param, "value", param)
                    )
                elif mapped_name.lower() in {"offset", "phoff"}:
                    exact[name] = "0.0"
                else:
                    raise ValueError(
                        "Missing reference theta for "
                        f"pta={pta_name!r}, canonical_fitpar={name!r}, "
                        f"mapped_fitpar={mapped_name!r}"
                    )
        return exact

    def pint_model(self):
        """Return canonical reference-PTA PINT model for timing-space discovery."""
        if self._pint_model_cache is not None:
            return self._pint_model_cache

        ref_pta = self._reference_pta_name()
        from .pint_helpers import create_pint_model

        # Always build from retained session par content so the reference model
        # matches runtime inputs (including any TRACK-modified session par file).
        self._pint_model_cache = create_pint_model(
            self._parfile_content_for_pta(ref_pta)
        )
        return self._pint_model_cache

    def can_use_engines(self, engines="jug", *, linearized: bool = False) -> bool:
        """Return whether every pulsar session can honor the engine selection."""
        from .timing.backends import _IMPL_FAMILY, normalize_engines

        engines = normalize_engines(engines)
        for pta_name in self._epulsars:
            native_compat = self._native_compat(pta_name)
            if native_compat not in engines:
                return False
            family = _IMPL_FAMILY[engines[native_compat]]
            if linearized:
                if family == "jug":
                    continue
                if native_compat != family:
                    return False
                continue
            if family == "jug":
                if not self._can_import_jug() or not self._session_files_available(
                    pta_name
                ):
                    return False
            elif family == "pint":
                src = self._pulsars.get(pta_name)
                if not (isinstance(src, tuple) and len(src) == 2):
                    return False
            elif family == "tempo2":
                if pta_name not in self._pulsars:
                    return False
        return True

    @staticmethod
    def _can_import_jug() -> bool:
        try:
            import jug.engine.session  # noqa: F401
        except Exception:
            return False
        return True

    def _session_files_available(self, pta_name: str) -> bool:
        files = self._session_files.get(pta_name)
        if files is None:
            return False
        if not files.par_path.is_file() or not files.tim_path.is_file():
            return False
        if self._clock_dir is not None and not self._clock_dir.exists():
            return False
        return True

    def _ensure_clock_aliases(self) -> None:
        if self._clock_dir is None:
            return
        clock_dir = Path(self._clock_dir)
        anchor = clock_dir / "tai2tt_bipm2017.clk"
        if not anchor.is_file():
            return
        for alias in ("tai2tt_bipm2011.clk", "tai2tt_bipm2024.clk"):
            link = clock_dir / alias
            if link.exists():
                continue
            try:
                link.symlink_to(anchor.name)
            except FileExistsError:
                pass

    def _native_compat(self, pta_name: str) -> str:
        files = self._session_files.get(pta_name)
        if files is not None and files.timing_package in {"pint", "tempo2"}:
            return files.timing_package
        return self._get_timing_package(self._epulsars[pta_name])

    def _build_jug_session(
        self,
        pta_name: str,
        compatibility: str,
        *,
        tempo2_native: object | None = None,
    ):
        if not self._session_files_available(pta_name):
            raise ValueError(
                f"Cannot build JUG timing session for '{pta_name}': missing par/tim inputs"
            )
        from jug.engine.session import TimingSession

        self._ensure_clock_aliases()
        files = self._session_files[pta_name]
        return TimingSession(
            par_file=str(files.par_path),
            tim_file=str(files.tim_path),
            clock_dir=None if self._clock_dir is None else str(self._clock_dir),
            verbose=False,
            compatibility=compatibility,
            tempo2_native=tempo2_native,
        )

    def prime_jug_tempo2_sessions(
        self,
        engines: Mapping[str, str] | None = None,
        *,
        tempo2_native: object | None = None,
        subtract_tzr: bool = False,
        force_recompute: bool = False,
    ) -> list[str]:
        """Refresh tempo2 JUG session caches so native_chain_static can be built."""
        from .timing.backends import _IMPL_FAMILY, normalize_engines

        engines = normalize_engines(engines or {"tempo2": "jug", "pint": "jug"})
        primed: list[str] = []
        for pta_name in self._epulsars:
            native = self._native_compat(pta_name)
            if not str(native).lower().startswith("tempo2"):
                continue
            if _IMPL_FAMILY.get(engines.get(native, "jug")) != "jug":
                continue
            session = self._build_jug_session(
                pta_name, native, tempo2_native=tempo2_native
            )
            session.compute_residuals(
                subtract_tzr=subtract_tzr,
                force_recompute=force_recompute,
            )
            cached = session._cached_result_by_mode.get(subtract_tzr)
            td = (cached or {}).get("term_diagnostics") or {}
            if "tempo2_obs_state" not in td:
                raise RuntimeError(
                    f"PTA {pta_name!r}: tempo2 JUG cache missing "
                    "term_diagnostics['tempo2_obs_state'] after compute_residuals."
                )
            primed.append(pta_name)
        return primed

    def timing_backend(
        self,
        engines="jug",
        *,
        linearized: bool = False,
        design_matrix_method: str = "analytic",
        tempo2_native: object | None = None,
        prime_sessions: bool = True,
        verify_wiring: bool = False,
        subtract_tzr: bool = False,
    ):
        """Return a TimingBackend in canonical pulsar row order."""
        from .timing.backends import _IMPL_FAMILY, normalize_engines

        engines = normalize_engines(engines)
        if not self.can_use_engines(engines, linearized=linearized):
            raise ValueError(
                f"engines {engines} cannot be honored for pulsar '{self.name}'"
            )

        cache_key = (
            tuple(sorted(engines.items())),
            linearized,
            design_matrix_method,
            str(tempo2_native),
            prime_sessions,
            verify_wiring,
            subtract_tzr,
            self.cache_token(),
        )
        if cache_key in self._timing_backend_cache:
            return self._timing_backend_cache[cache_key]

        from .timing.backends import (
            JugEngine,
            LibstempoEngine,
            LinearizedJugEngine,
            LinearizedLibstempoEngine,
            LinearizedPintEngine,
            PintEngine,
            PulsarSession,
            build_backend,
        )
        from .timing.backends.base import LinearModel, validate_backend_against_pulsar

        pta_slices = self._get_pta_slices()
        fitpars = tuple(self.fitpars)
        global_index = {par: i for i, par in enumerate(fitpars)}
        sessions: list[PulsarSession] = []

        if (
            prime_sessions
            and design_matrix_method == "autodiff"
            and any(
                _IMPL_FAMILY[engines[self._native_compat(pta)]] == "jug"
                and str(self._native_compat(pta)).lower().startswith("tempo2")
                for pta in self._epulsars
            )
        ):
            from jug.timing import normalize_tempo2_native

            cfg = normalize_tempo2_native(tempo2_native, compatibility="tempo2")
            force = bool(getattr(cfg, "force_cache_refresh", False))
            self.prime_jug_tempo2_sessions(
                engines,
                tempo2_native=tempo2_native,
                subtract_tzr=subtract_tzr,
                force_recompute=force,
            )

        for pta_name, psr in self._epulsars.items():
            slc = pta_slices[pta_name]
            rows = np.arange(slc.start, slc.stop, dtype=int)
            session_fitpars = tuple(
                par for par in fitpars if pta_name in self._fitparameters.get(par, {})
            )
            session_cols = [global_index[par] for par in session_fitpars]
            session_design = self._designmatrix[rows][:, session_cols]
            theta_exact = self._session_theta_exact(pta_name, session_fitpars)
            linear_model = LinearModel.from_host(
                fitpars=session_fitpars,
                design=session_design,
                theta_exact=theta_exact,
            )

            native_compat = self._native_compat(pta_name)
            family = _IMPL_FAMILY[engines[native_compat]]
            if linearized and family == "pint":
                backend = LinearizedPintEngine.from_linear_model(linear_model)
            elif linearized and family == "tempo2":
                backend = LinearizedLibstempoEngine.from_linear_model(linear_model)
            elif linearized:
                backend = LinearizedJugEngine.from_linear_model(
                    linear_model, compatibility=native_compat
                )
            elif family == "pint":
                source = self._pulsars[pta_name]
                if not (isinstance(source, tuple) and len(source) == 2):
                    raise ValueError(f"PTA '{pta_name}' does not have PINT inputs")
                backend = PintEngine.from_session(
                    source[0],
                    source[1],
                    linear_model=linear_model,
                )
            elif family == "tempo2":
                session_mapping = {
                    name: self._fitparameters.get(name, {}).get(pta_name, name)
                    for name in session_fitpars
                }
                backend = LibstempoEngine.from_session(
                    self._pulsars[pta_name],
                    linear_model=linear_model,
                    param_mapping=session_mapping,
                )
            else:
                jug_session = self._build_jug_session(
                    pta_name,
                    native_compat,
                    tempo2_native=tempo2_native,
                )
                session_mapping = {
                    name: self._fitparameters.get(name, {}).get(pta_name, name)
                    for name in session_fitpars
                }
                backend = JugEngine.from_session(
                    jug_session,
                    linear_model=linear_model,
                    compatibility=native_compat,
                    param_mapping=session_mapping,
                    design_matrix_method=design_matrix_method,
                )

            sessions.append(
                PulsarSession(
                    name=pta_name,
                    row_indices=rows,
                    backend=backend,
                    exact_linear_fitpars=(
                        backend.exact_linear_fitpars()
                        if hasattr(backend, "exact_linear_fitpars")
                        else frozenset()
                    ),
                )
            )

        backend = build_backend(
            fitpars=fitpars,
            nrows=len(self._toas),
            sessions=sessions,
            host_design=self._designmatrix,
        )
        validate_backend_against_pulsar(backend, self, tol=1e-9)
        if verify_wiring:
            from .timing.backends.jug import verify_jug_native_chain_wiring

            verify_jug_native_chain_wiring(
                backend, design_matrix_method=design_matrix_method
            )
        self._timing_backend_cache[cache_key] = backend
        return backend

    def cache_token(self) -> str:
        """Return stable token for pulsar-tied cache invalidation."""
        pta_tags = [
            f"{pta}:{self._get_timing_package(psr)}"
            for pta, psr in self._epulsars.items()
        ]
        dm_checksum = float(np.sum(np.abs(self._designmatrix)))
        return (
            f"{self.name}|ntoa={len(self._toas)}|nfit={len(self.fitpars)}|"
            f"fitpars={','.join(self.fitpars)}|pta={','.join(pta_tags)}|"
            f"dmabs={dm_checksum:.12e}"
        )

    def sort_data(self):
        """Sort data by time when requested; otherwise preserve storage order."""
        if self._sort:
            self._isort = np.argsort(self._toas, kind="mergesort")
            self._iisort = np.zeros(len(self._isort), dtype=int)
            for ii, p in enumerate(self._isort):
                self._iisort[p] = ii
        else:
            self._isort = slice(None, None, None)
            self._iisort = slice(None, None, None)

    def filter_data(self, mask=None, start_time=None, end_time=None):
        """Filter TOAs by mask and/or time range."""
        start_time = (
            start_time * 86400 if start_time is not None else np.min(self._toas)
        )
        end_time = end_time * 86400 if end_time is not None else np.max(self._toas)
        mask_times = np.logical_and(self._toas >= start_time, self._toas <= end_time)
        mask = np.logical_and(mask, mask_times) if mask is not None else mask_times

        self._toas = self._toas[mask]
        self._stoas = self._stoas[mask]
        self._toaerrs = self._toaerrs[mask]
        self._residuals = self._residuals[mask]
        self._ssbfreqs = self._ssbfreqs[mask]
        self._telescope = self._telescope[mask]
        self._designmatrix = self._designmatrix[mask, :]
        self._pos_t = self._pos_t[mask, :]
        self._planetssb = self._planetssb[mask, :, :]
        self._sunssb = self._sunssb[mask, :]

        dmx_mask = np.sum(self._designmatrix, axis=0) != 0.0
        self._designmatrix = self._designmatrix[:, dmx_mask]

        if isinstance(self._flags, np.ndarray):
            self._flags = self._flags[mask]
        else:
            for key in self._flags:
                self._flags[key] = self._flags[key][mask]

        self.sort_data()
        self._invalidate_timing_caches()

    @property
    def isort(self):
        """Return sorting indices."""
        return self._isort

    @property
    def iisort(self):
        """Return inverse sorting indices."""
        return self._iisort

    @property
    def toas(self):
        """Return array of TOAs in seconds."""
        return self._toas[self._isort]

    @property
    def stoas(self):
        """Return array of observatory TOAs in seconds."""
        return self._stoas[self._isort]

    @property
    def residuals(self):
        """Return array of residuals in seconds."""
        return self._residuals[self._isort]

    @property
    def toaerrs(self):
        """Return array of TOA errors in seconds."""
        return self._toaerrs[self._isort]

    @property
    def freqs(self):
        """Return array of radio frequencies in MHz."""
        return self._ssbfreqs[self._isort]

    @property
    def Mmat(self):
        """Return ntoa x npar design matrix."""
        return self._designmatrix[self._isort, :]

    @property
    def pdist(self):
        """Return tuple of pulsar distance and uncertainty in kpc."""
        return self._pdist

    @property
    def dm(self):
        """Return DM parameter from parfile."""
        return self._dm

    @property
    def dmx(self):
        """Return DMX parameter dictionary from parfile."""
        return self._dmx

    @property
    def flags(self):
        """Return a dictionary of tim-file flags."""
        flagnames = (
            self._flags.dtype.names
            if isinstance(self._flags, np.ndarray)
            else self._flags.keys()
        )
        return {flag: self._flags[flag][self._isort] for flag in flagnames}

    def set_flags(self, flagname, values):
        """Set value of existing or new flags."""
        if isinstance(self._flags, np.ndarray):
            raise NotImplementedError("Cannot set flags when stored as numpy.ndarray.")
        self._flags[flagname] = values[self._iisort]

    @property
    def backend_flags(self):
        """Return array of backend flags."""
        flagnames = (
            self._flags.dtype.names
            if isinstance(self._flags, np.ndarray)
            else list(self._flags.keys())
        )
        ret = np.zeros(
            len(self._toas),
            dtype=max([self._flags[name].dtype for name in flagnames]),
        )

        if "fe" in flagnames and "be" in flagnames:
            ret[:] = [
                (a + "_" + b if (a and b) else "")
                for a, b in zip(self._flags["fe"], self._flags["be"])
            ]

        for flag in ["f", "i", "sys", "g", "group"]:
            if flag in flagnames:
                ret[:] = np.where(self._flags[flag] == "", ret, self._flags[flag])

        return ret[self._isort]

    @property
    def theta(self):
        """Return polar angle of pulsar in radians."""
        return np.pi / 2 - self._decj

    @property
    def phi(self):
        """Return azimuthal angle of pulsar in radians."""
        return self._raj

    @property
    def pos(self):
        """Return unit vector from SSB to pulsar at fiducial POSEPOCH."""
        return self._pos

    @property
    def pos_t(self):
        """Return unit vector from SSB to pulsar as function of time."""
        return self._pos_t[self._isort, :]

    @property
    def planetssb(self):
        """Return planetary position vectors at all timestamps."""
        return self._planetssb[self._isort, :, :]

    @property
    def sunssb(self):
        """Return sun position vectors at all timestamps."""
        return self._sunssb[self._isort, :]

    @property
    def telescope(self):
        """Return telescope names at all timestamps."""
        return self._telescope[self._isort]
