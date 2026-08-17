"""Main MetaPulsar class for combining multi-PTA pulsar timing data."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, List, Mapping
import numpy as np
from loguru import logger

# Import PINT classes
from pint.models import TimingModel
from pint.toa import TOAs

# Import our supporting infrastructure
from .parameter_manager import ParameterInconsistencyError, ParameterManager
from .parfile_lines import iter_active_par_lines
from .pint_helpers import (
    NativeParam,
    ParameterSourceMappingError,
    resolve_fit_column_name,
    resolve_native_source_name,
)
from .position_helpers import (
    assert_catalog_suffixes_compatible,
    bj_name_from_pulsar,
    positions_within_tolerance,
    preferred_group_name,
    _skycoord_from_pint_model,
    _skycoord_from_libstempo,
)
from .pta_data import _PtaTimingData, materialize_pint, materialize_tempo2


@dataclass(frozen=True)
class PtaFiles:
    """Durable timing-package input files for one PTA."""

    par_path: Path
    tim_path: Path
    timing_package: str


_COMBINATION_STRATEGY_ALIASES = {"consistent": "shared", "composite": "per_pta"}
_COMBINATION_STRATEGIES = ("shared", "per_pta")


def normalize_combination_strategy(value: str) -> str:
    """Normalize a ``combination_strategy`` value to its canonical spelling.

    Canonical values are ``"shared"`` (merge shared timing-model params across
    PTAs, ex-``"consistent"``) and ``"per_pta"`` (keep per-PTA params,
    ex-``"composite"``). The legacy ``"consistent"``/``"composite"`` spellings
    are still accepted as deprecated aliases and emit a ``DeprecationWarning``.
    """
    if value in _COMBINATION_STRATEGY_ALIASES:
        canonical = _COMBINATION_STRATEGY_ALIASES[value]
        warnings.warn(
            f"combination_strategy={value!r} is deprecated; use {canonical!r}",
            DeprecationWarning,
            stacklevel=2,
        )
        return canonical
    if value not in _COMBINATION_STRATEGIES:
        raise ValueError(
            "combination_strategy must be one of "
            f"{_COMBINATION_STRATEGIES} (deprecated aliases: "
            f"{sorted(_COMBINATION_STRATEGY_ALIASES)}); got {value!r}"
        )
    return value


class MetaPulsar:
    """Elegant composite pulsar for multi-PTA data combination.

    This class combines pulsar timing data from multiple PTA collaborations
    into a unified object suitable for gravitational wave detection analysis.
    This class implements an Enterprise-compatible duck surface directly.

    Supports two combination strategies:
    - "shared": shared timing-model params across PTAs (modifies par files for
      consistency; ex-"consistent")
    - "per_pta": per-PTA timing-model params preserved (ex-"composite")
    """

    def __init__(
        self,
        pulsars,
        *,  # Remove parfile_dicts parameter
        combination_strategy="shared",
        combine_components: List[str] = [
            "astrometry",
            "spindown",
            "binary",
            "dispersion",
        ],
        add_dm_derivatives: bool = True,
        exclude_from_shared: List[str] | tuple[str, ...] = ("DM",),
        pta_files: dict[str, dict] | None = None,
        clock_dir: str | Path | None = None,
        sort=False,
    ):
        """Create MetaPulsar from multiple PTA pulsars.

        Args:
            pulsars: Dict mapping PTA names to pulsar data:
                - PINT: {pta: (pint_model, pint_toas)}
                - Tempo2: {pta: tempo2_psr}
            combination_strategy: Strategy for combining PTAs:
                - "shared": shared timing-model params across PTAs (modifies par
                  files for consistency; ex-"consistent")
                - "per_pta": per-PTA timing-model params preserved (ex-"composite")
                The legacy "consistent"/"composite" spellings are accepted as
                deprecated aliases.
            combine_components: List of components to share ("shared" strategy only):
                - "astrometry": Position and proper motion parameters
                - "spindown": Spin frequency and derivatives
                - "binary": Binary orbital parameters
                - "dispersion": Dispersion measure parameters
                Defaults to all components
            add_dm_derivatives: Whether to ensure DM1, DM2 are present ("shared" strategy only)
            exclude_from_shared: Canonical timing-model parameter names to keep
                PTA-specific even when their component is in combine_components.
                Defaults to ("DM",) so each PTA keeps its own reference DM while
                shared dispersion still shares DM1/DM2. Pass an empty list to
                merge all parameters in selected components.
            pta_files: **Required, one entry per PTA in ``pulsars``:**
                ``{pta: {"par_path": ..., "tim_path": ..., "timing_package": ...}}``.
                MetaPulsar reads every leg's par text from ``par_path`` and never
                re-serializes an engine object to recover it, so a PTA without a
                retained par is rejected here rather than silently answered from
                a different document. ``create_metapulsar()`` retains the engine
                inputs and passes this for you; mock-backed tests can build it
                with :func:`metapulsar.mockpulsar.write_mock_pta_files`. Only an
                empty ``pulsars`` map may omit it.
            sort: Whether to sort data by time

        Raises:
            ValueError: If ``pta_files`` is missing an entry for any PTA, or an
                entry lacks ``par_path`` / ``tim_path`` / ``timing_package``.
        """
        self._pulsars = pulsars
        self.combination_strategy = normalize_combination_strategy(combination_strategy)
        self.combine_components = (
            combine_components if self.combination_strategy == "shared" else []
        )
        self.add_dm_derivatives = add_dm_derivatives
        self.exclude_from_shared = exclude_from_shared
        # Retained per-PTA par/tim must be available before reference-theta lookup:
        # pulse-number tracking uses temporary TRACK -2 par paths that are deleted
        # after libstempo construction.
        self._pta_files = self._normalize_pta_files(pta_files, pulsars)
        self._parfile_dicts = self._get_parfile_data(pulsars)
        self._clock_dir = None if clock_dir is None else Path(clock_dir)
        self._sort = sort
        self._timing_engine_cache = {}
        self._timing_rows_filtered = False
        self._pint_model_cache = None
        self._shared_theta_exact_cache: dict[str, str] = {}
        self._retained_pint_model_cache: dict = {}
        self._dispersion_metadata_ready = False
        self._dm: float | None = None
        self._dmx: dict | None = None
        self.binary_conversion_report = None
        self.combination_write_result = None

        # Elegant initialization flow
        self._materialize_pta_data()
        self._setup_parameters()
        self._assert_engine_chart_consistency()
        self._combine_timing_data()
        self._build_design_matrix()
        self._remove_nonidentifiable_parameters()
        self._assert_gauge_columns()
        self._setup_position_and_planets()

        self.sort_data()

        # Calculate canonical name from pulsar data using B-name preference logic
        self.name = self._get_pulsar_name(pulsars)

    def conversion_metadata(self):
        """Derived from binary_conversion_report; None when no conversion ran.

        TimingPulsar-protocol extension point for nltiming Case-D STIGMA
        required-sampling.
        """
        from .binary_family_convert import metadata_from_report

        return metadata_from_report(self.binary_conversion_report)

    @staticmethod
    def _normalize_pta_files(
        pta_files: dict[str, dict] | None,
        pulsars: Mapping[str, Any],
    ) -> dict[str, PtaFiles]:
        """Normalize ``pta_files``, requiring one entry per PTA.

        MetaPulsar is a view over files it was given: par text is read from
        them, never re-serialized out of an engine object (see
        :meth:`_parfile_content_for_pta`). A PTA with no retained par has no
        authority to answer "what is this PTA's par?", so construction fails
        here rather than silently falling back to a different document.
        """
        entries = dict(pta_files or {})
        missing = [name for name in pulsars if name not in entries]
        if missing:
            raise ValueError(
                "MetaPulsar requires pta_files for every PTA; missing: "
                f"{sorted(missing)}. Build through create_metapulsar(), which "
                "retains the engine inputs, or pass "
                "{'<pta>': {'par_path': ..., 'tim_path': ..., "
                "'timing_package': ...}} explicitly."
            )
        extra = [name for name in entries if name not in pulsars]
        if extra:
            raise ValueError(
                f"pta_files has entries for PTAs absent from pulsars: {sorted(extra)}; "
                "the two must describe the same set of PTAs."
            )
        normalized: dict[str, PtaFiles] = {}
        for pta_name, files in entries.items():
            absent = [
                key
                for key in ("par_path", "tim_path", "timing_package")
                if key not in files
            ]
            if absent:
                raise ValueError(
                    f"pta_files[{pta_name!r}] is missing {sorted(absent)}; each "
                    "entry needs par_path, tim_path and timing_package."
                )
            normalized[pta_name] = PtaFiles(
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
        if not hasattr(self, "_pta_data") or self._pta_data is None:
            raise ValueError("No PTA timing records created yet")

        pulsar_names = []
        for pta, record in self._pta_data.items():
            if record.name and record.name != "None":
                pulsar_names.append(record.name)
            else:
                logger.warning(f"PTA {pta} pulsar has no valid name attribute")

        if not pulsar_names:
            raise ValueError("No pulsar names found")

        if not self._all_equal(pulsar_names):
            raise ValueError(f"Not all the same pulsar: {pulsar_names}")

        return pulsar_names[0]

    def _materialize_pta_data(self) -> None:
        """Materialize MetaPulsar-owned PTA timing records from input data."""
        self._pta_data = {}
        pint_models, pint_toas, lt_pulsars = self._unpack_pulsar_data()
        if not pint_models and not lt_pulsars:
            raise ValueError("MetaPulsar requires at least one PTA input")
        self.name = self._validate_pulsar_consistency(pint_models, lt_pulsars)
        for pta, model in pint_models.items():
            self._pta_data[pta] = materialize_pint(model, pint_toas[pta])
        for pta, pulsar in lt_pulsars.items():
            self._pta_data[pta] = materialize_tempo2(pulsar)

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
        """Validate single pulsar across all PTAs by sky position and catalog suffixes."""
        sky_coords = []

        for m in pint_models.values():
            sky_coords.append(_skycoord_from_pint_model(m))

        for psr in lt_pulsars.values():
            sky_coords.append(_skycoord_from_libstempo(psr))

        if not sky_coords:
            raise ValueError("No valid pulsars found for validation")

        if not positions_within_tolerance(sky_coords, match_tol_arcsec=10.0):
            raise ValueError(
                "Not all PTAs refer to the same sky position within 10″ tolerance"
            )

        catalog_names = []
        for m in pint_models.values():
            catalog_names.append(m.PSR.value)
        for psr in lt_pulsars.values():
            catalog_names.append(psr.name)

        assert_catalog_suffixes_compatible(catalog_names)

        return preferred_group_name(catalog_names)

    def _all_equal(self, iterable):
        """Check if all items in iterable are equal."""
        g = groupby(iterable)
        return next(g, True) and not next(g, False)

    def _setup_parameters(self):
        """Setup parameter management using existing infrastructure."""
        pint_models, _, lt_pulsars = self._unpack_pulsar_data()

        # One source per PTA: the retained session par, for both engines
        # (:meth:`_parfile_content_for_pta`).
        file_data = {}
        for pta_name in list(pint_models) + list(lt_pulsars):
            file_data[pta_name] = {
                "par": None,
                "par_content": self._parfile_content_for_pta(pta_name),
                "timing_package": "pint" if pta_name in pint_models else "tempo2",
            }

        # Create ParameterManager for parameter mapping
        parameter_manager = ParameterManager(
            file_data=file_data,
            combine_components=self.combine_components,
            add_dm_derivatives=self.add_dm_derivatives,
            exclude_from_shared=self.exclude_from_shared,
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
        """Setup canonical parameter lists for each PTA timing record."""
        for pta_name, record in self._pta_data.items():
            record.fitpars_canonical = [
                resolve_fit_column_name(name) for name in record.fitpars
            ]
            record.setpars_canonical = [
                resolve_fit_column_name(name) for name in record.setpars
            ]

    def _assert_engine_chart_consistency(self) -> None:
        """Fail loud when PINT identity and PTA fit columns disagree on chart.

        Design-matrix assembly resolves ``_fitparameters`` names against each
        PTA's ``fitpars_canonical``. A representation mismatch (canonical ``FB0``
        vs a native ``PB`` column) otherwise surfaces as a bare
        ``ValueError: 'FB0' is not in list`` from ``list.index``.
        ``ParameterManager`` aligns the orbital chart of every par it produces
        so reaching this method means the engine object was
        built from a par that never went through it -- typically by calling
        ``MetaPulsar(...)`` directly with pre-built engine objects instead of
        ``create_metapulsar()``.
        """
        for meta_param, owners in self._fitparameters.items():
            for pta_name, native in owners.items():
                record = self._pta_data.get(pta_name)
                if record is None or not record.fitpars_canonical:
                    continue
                if native.identity in record.fitpars_canonical:
                    continue
                hint = ""
                if native.identity.upper().startswith("FB"):
                    hint = (
                        " This is the hybrid PB+FBn orbital chart: the par must "
                        "be aligned to FB0 before the engine is built. Construct "
                        "via create_metapulsar() instead of MetaPulsar(...) "
                        "directly."
                    )
                raise ValueError(
                    f"PTA {pta_name!r} has no PTA fit column for meta "
                    f"parameter {meta_param!r} (pint_name={native.pint_name!r} "
                    f"par_key={native.par_key!r} identity={native.identity!r}); "
                    f"available fitpars={list(record.fitpars)!r}.{hint}"
                )

    def _combine_timing_data(self):
        """Combine timing data from all PTAs."""

        def concat(attribute: str) -> np.ndarray:
            return np.concatenate(
                [
                    np.asarray(getattr(record, attribute))
                    for record in self._pta_data.values()
                ]
            )

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
        """Combine flags from all PTAs.

        ``pta``, ``pta_dataset`` and ``timing_package`` normally arrive from the
        canonical ``.tim`` the factory writes when ``canonicalize_tim=True``
        (see :mod:`metapulsar.tim_canonical`). They are filled here only for legs
        that lack them — MetaPulsar built with ``canonicalize_tim=False``, or
        directly from pulsar objects rather than from files.
        """
        from collections import defaultdict

        pta_slice = self._get_pta_slices()
        flags = defaultdict(lambda: np.zeros(len(self._toas), dtype="U128"))

        fallback_flags = ("pta", "pta_dataset", "timing_package")

        for pta, record in self._pta_data.items():
            pta_rows = pta_slice[pta]
            for flag, flag_values in record._flags.items():
                if flag in fallback_flags:
                    # EPTA tim files carry '\t' around some values.
                    flag_values = np.char.strip(np.asarray(flag_values, dtype="U128"))
                flags[flag][pta_rows] = flag_values

            fallbacks = {
                "pta": pta,
                "pta_dataset": pta,
                "timing_package": self._get_timing_package(record),
            }
            for flag in fallback_flags:
                values = flags[flag][pta_rows]
                missing = values == ""
                if np.any(missing):
                    values[missing] = fallbacks[flag]
                    flags[flag][pta_rows] = values

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

        for pta, record in self._pta_data.items():
            end_idx = start_idx + len(record._toas)
            slices[pta] = slice(start_idx, end_idx)
            start_idx = end_idx

        return slices

    @staticmethod
    def _get_timing_package(record: _PtaTimingData) -> str:
        """Return the typed timing package for a PTA timing record."""
        return record.timing_package

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

    def _assert_gauge_columns(self) -> None:
        """Assert the combined Mmat spans one constant per PTA."""
        from nltiming.nonlinear_timing_model import assert_gauge_column_present

        pta_slices = self._get_pta_slices()

        class _Leaf:
            gauge_applied = False

            def gauge_provenance(self):
                from nltiming.protocols import GaugeProvenance

                return GaugeProvenance(export="none", reference_mode="none")

        class _Contribution:
            def __init__(self, name: str, row_indices):
                self.name = name
                self.row_indices = np.asarray(row_indices, dtype=int)
                self.engine = _Leaf()

        contributions = [
            _Contribution(pta, np.arange(slc.start, slc.stop, dtype=int))
            for pta, slc in pta_slices.items()
        ]

        class _StubEngine:
            def __init__(self, contribs):
                self.contributions = contribs

        assert_gauge_column_present(
            self,
            _StubEngine(contributions),
            np.asarray(self._designmatrix, dtype=float),
        )

    def _build_design_matrix_column(self, full_parname):
        """Build design matrix column for a single parameter."""
        pta_slices = self._get_pta_slices()
        n_toas = len(self._toas)
        column = np.zeros(n_toas)

        for pta, record in self._pta_data.items():
            if pta not in pta_slices:
                continue

            slice_obj = pta_slices[pta]
            timing_package = self._get_timing_package(record)
            dm = record._designmatrix
            if full_parname in self._fitparameters:
                for mapped_pta, native in self._fitparameters[full_parname].items():
                    if mapped_pta == pta:
                        par_idx = record.fitpars_canonical.index(native.identity)
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
        """Setup position and planetary data from PTA timing records."""
        ref_record = next(iter(self._pta_data.values()))

        # Set basic position attributes
        self._raj = ref_record._raj
        self._decj = ref_record._decj

        bj_name = bj_name_from_pulsar(ref_record)
        logger.debug(f"Generated B/J name: {bj_name}")

        pta_slice = self._get_pta_slices()
        self._pos = np.zeros((len(self._toas), 3))
        self._pos_t = np.zeros((len(self._toas), 3))
        self._planetssb = np.zeros((len(self._toas), 9, 6))
        self._sunssb = np.zeros((len(self._toas), 6))
        for pta, record in self._pta_data.items():
            self._pos[pta_slice[pta], :] = record._pos
            self._pos_t[pta_slice[pta], :] = record._pos_t
            self._planetssb[pta_slice[pta], :, :] = record._planetssb
            self._sunssb[pta_slice[pta], :] = record._sunssb

        self._pdist = ref_record._pdist
        self._pos = ref_record._pos

    def _parfile_content_for_pta(self, pta_name: str) -> str:
        """Return the retained par text for one PTA.

        The retained file is the *only* answer: it is the exact input the engine
        was built from, so every derived view (parameter mapping, exact-string
        lookup, reference metadata) describes the same par. Re-serializing an
        engine object instead -- PINT ``as_parfile()`` or libstempo
        ``savepar()`` -- yields a different document (reordered, dialect-shifted,
        and in tempo2's case with a recomputed ``TZRMJD`` that can be ``-nan``),
        which is how one MetaPulsar ended up holding two disagreeing notions of
        its own par.
        """
        pta_file = self._pta_files.get(pta_name)
        if pta_file is None:
            raise KeyError(
                f"No retained par file for PTA {pta_name!r}; MetaPulsar requires "
                "pta_files for every PTA (see MetaPulsar.__init__)"
            )
        if not pta_file.par_path.is_file():
            raise FileNotFoundError(
                f"Retained par file for PTA {pta_name!r} is missing: "
                f"{pta_file.par_path}"
            )
        return pta_file.par_path.read_text(encoding="utf-8")

    def _get_parfile_data(self, pulsars):
        """Per-PTA parfile dictionaries, parsed from the retained pars.

        Same single source as :meth:`_parfile_content_for_pta`, including for
        PINT legs: the engine's in-memory model is a re-serialization of this
        file, not an independent authority.
        """
        from .pint_helpers import create_pint_model

        parfile_dicts = {}
        for pta_name in pulsars:
            par_content = self._parfile_content_for_pta(pta_name)
            try:
                parfile_dicts[pta_name] = create_pint_model(
                    par_content
                ).get_params_dict()
            except Exception as exc:
                raise ValueError(
                    "Failed to extract parfile metadata for "
                    f"pta={pta_name!r} from retained par content: {exc}"
                ) from exc
        return parfile_dicts

    def _get_pulsar_name(self, pulsars):
        """Return B-preferred catalog name from parfile PSR fields across PTAs."""
        pulsar_names = self._extract_pulsar_names(pulsars)
        return preferred_group_name(pulsar_names)

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
        self._timing_engine_cache.clear()
        self._pint_model_cache = None
        self._shared_theta_exact_cache.clear()
        self._retained_pint_model_cache.clear()
        self._dispersion_metadata_ready = False
        self._dm = None
        self._dmx = None

    def _reference_pta_name(self) -> str:
        """Return deterministic reference PTA key for pulsar-level metadata."""
        if not self._pta_data:
            raise ValueError("No PTA sessions are available on this MetaPulsar")
        return next(iter(self._pta_data.keys()))

    def _native_param(self, name: str, pta_name: str) -> NativeParam:
        """Typed native record for one host fit column on one PTA leg.

        Falls back to ``NativeParam(name, name)`` when the host key is absent,
        preserving today's permissive behaviour for hand-built pulsars. The
        fallback deliberately uses the *host key*, not a native stem: a
        suffixed host key cannot fold, so a bare fallback stays inert.
        """
        native = self._fitparameters.get(name, {}).get(pta_name)
        return native if native is not None else NativeParam(name, name)

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

    def _retained_pint_model(self, pta_name: str):
        """PINT model parsed from post-harmonization retained par content."""
        from .pint_helpers import create_pint_model

        cached = self._retained_pint_model_cache.get(pta_name)
        if cached is not None:
            return cached
        model = create_pint_model(self._parfile_content_for_pta(pta_name))
        self._retained_pint_model_cache[pta_name] = model
        return model

    def _lookup_theta_exact_from_sources(
        self,
        *,
        pta_name: str,
        name: str,
        native: NativeParam,
        par_source: dict,
        pint_model,
    ) -> str:
        """Resolve one fitpar exact string from a PINT-keyed dict and/or model.

        Both sources are PINT-keyed -- ``_parfile_dicts`` holds
        ``create_pint_model(retained par).get_params_dict()`` and the retained
        model is that same model -- so ``native.pint_name`` is a direct key and
        no alias resolution happens to *find* a value. Check order puts value
        sources before the ``Offset``/``PHOFF`` zero so a real ``PHOFF`` wins.
        """
        if native.pint_name in par_source:
            return self._stringify_par_value(par_source[native.pint_name])
        if pint_model is not None and hasattr(pint_model, native.pint_name):
            param = getattr(pint_model, native.pint_name)
            return self._stringify_par_value(getattr(param, "value", param))
        if native.identity.lower() in {"offset", "phoff"}:
            return "0.0"
        raise ValueError(
            "Missing reference theta for "
            f"pta={pta_name!r}, fitpar={name!r}, pint_name={native.pint_name!r}, "
            f"par_key={native.par_key!r}, identity={native.identity!r}; "
            f"free params in source: {sorted(par_source)!r}"
        )

    def _local_theta_exact(
        self, pta_name: str, name: str, *, from_retained: bool = False
    ) -> str:
        """Exact reference string for one fitpar from one PTA.

        When ``from_retained`` is True (shared-parameter path), parse the
        post-harmonization retained par content so validation and conversion
        consume the same bytes. PTA-specific parameters keep the construction-
        time ``_parfile_dicts`` / model-metadata path.
        """
        native = self._native_param(name, pta_name)
        if from_retained:
            return self._lookup_theta_exact_from_sources(
                pta_name=pta_name,
                name=name,
                native=native,
                par_source={},
                pint_model=self._retained_pint_model(pta_name),
            )

        return self._lookup_theta_exact_from_sources(
            pta_name=pta_name,
            name=name,
            native=native,
            par_source=self._parfile_dicts.get(pta_name, {}),
            pint_model=None,
        )

    def _shared_theta_source(self, name: str) -> str:
        """Deterministic source PTA for a shared fitpar."""
        owners = self._fitparameters.get(name, {})
        ref = self._reference_pta_name()
        if ref in owners:
            return ref
        return next(pta for pta in self._pta_data if pta in owners)

    def _retained_value_token(self, pta_name: str, name: str) -> str:
        """Raw value token for one fitpar from one PTA's retained par content.

        Parses ``self._parfile_content_for_pta(pta_name)`` — never
        ``_parfile_dicts``, which may predate harmonization. Par first tokens
        are a genuinely foreign spelling, so lines are matched by *identity*
        rather than by string equality with either stored spelling: a retained
        document whose spelling drifted (``ECC``/``E`` after harmonization)
        still resolves.

        Precondition: this serves shared parameters only
        (``_validate_shared_retained_tokens`` is its sole caller), and the
        default ``combine_components`` (astrometry, spindown, binary,
        dispersion) contains no mask/prefix instances, so first-token identity
        is sufficient. A custom ``combine_components`` that shared ``fdjump``
        could put two same-exponent, different-mask lines here; that fails
        loud below rather than returning a wrong token.
        """
        native = self._native_param(name, pta_name)
        content = self._parfile_content_for_pta(pta_name)
        matches: list[str] = []
        first_tokens: list[str] = []
        for _index, line in iter_active_par_lines(content):
            tokens = line.split()
            first_tokens.append(tokens[0])
            # ``.upper()`` keeps today's case-insensitive token match: the
            # FDJUMP regexes are re.I but PINT's alias table is not.
            if resolve_fit_column_name(tokens[0].upper()) != native.identity:
                continue
            if len(tokens) < 2:
                raise ParameterInconsistencyError(
                    f"PTA {pta_name!r} retained par has parameter "
                    f"{tokens[0]!r} with no value token"
                )
            matches.append(tokens[1])
        if not matches:
            raise ParameterSourceMappingError(
                f"PTA {pta_name!r} retained par has no active line for fitpar "
                f"{name!r} (pint_name={native.pint_name!r}, "
                f"par_key={native.par_key!r}, identity={native.identity!r}); "
                f"active first tokens={first_tokens!r}"
            )
        if len(matches) > 1:
            raise ParameterInconsistencyError(
                f"PTA {pta_name!r} retained par has {len(matches)} active "
                f"lines matching identity {native.identity!r} (fitpar "
                f"{name!r}); require exactly one"
            )
        return matches[0]

    def _validate_shared_retained_tokens(self, name: str) -> None:
        """Require byte-identical retained value tokens across all owners."""
        owners = self._fitparameters.get(name, {})
        tokens = {pta: self._retained_value_token(pta, name) for pta in owners}
        if len(set(tokens.values())) > 1:
            detail = ", ".join(
                f"{pta} ({owners[pta].par_key}): {token!r}"
                for pta, token in tokens.items()
            )
            raise ParameterInconsistencyError(
                f"Shared parameter '{name}' has inconsistent retained par "
                f"values across contributions: {detail}"
            )

    def _pta_theta_exact(
        self, pta_name: str, pta_fitpars: tuple[str, ...]
    ) -> dict[str, str]:
        """Build reference-theta exact mapping for one PTA using parfile metadata."""
        exact: dict[str, str] = {}
        for name in pta_fitpars:
            owners = self._fitparameters.get(name, {})
            if len(owners) > 1:
                if name not in self._shared_theta_exact_cache:
                    self._validate_shared_retained_tokens(name)
                    source = self._shared_theta_source(name)
                    self._shared_theta_exact_cache[name] = self._local_theta_exact(
                        source, name, from_retained=True
                    )
                exact[name] = self._shared_theta_exact_cache[name]
            else:
                exact[name] = self._local_theta_exact(pta_name, name)
        return exact

    def pint_model(self):
        """Return canonical reference-PTA PINT model for timing-space discovery."""
        if self._pint_model_cache is not None:
            return self._pint_model_cache

        ref_pta = self._reference_pta_name()
        from .pint_helpers import create_pint_model

        # Always build from retained per-PTA par content so the reference model
        # matches runtime inputs (including any TRACK-modified per-PTA par file).
        self._pint_model_cache = create_pint_model(
            self._parfile_content_for_pta(ref_pta)
        )
        return self._pint_model_cache

    def _ensure_dispersion_metadata(self) -> None:
        """Memoize ``_dm``/``_dmx`` from ``pint_model()`` (Enterprise shape).

        Best effort by design: these are optional feather-snapshot metadata, so
        any failure degrades to ``None`` with a warning instead of propagating
        out of the ``dm``/``dmx`` properties. The feather writer probes them
        with ``hasattr``, which re-raises anything that is not an
        ``AttributeError`` — an unparseable reference par file would otherwise
        lose the whole snapshot. Whatever was extracted before a failure is
        kept.
        """
        if getattr(self, "_dispersion_metadata_ready", False):
            return

        try:
            model = self.pint_model()  # cached; uses _reference_pta_name()

            # Enterprise leaves _dm unset when DM is absent; we store None so the
            # @property always exists and the feather writer emits "dm": null.
            self._dm = float(model.DM.value) if hasattr(model, "DM") else None

            # Enterprise quirk (keep): only unfrozen DMX_* appear, so "fit" is
            # always True for written bins; frozen bins are omitted. The
            # DMXR1/DMXR2 keying via par[:3]+"R1"+par[3:] raises KeyError if a
            # range parameter is missing — enterprise.pulsar.PintPulsar._set_dm
            # propagates that at construction; here it is caught below.
            free = {par for par in model.params if not getattr(model, par).frozen}
            self._dmx = {
                par: {
                    "DMX": float(model[par].value),
                    "DMXerr": (
                        None
                        if model[par].uncertainty_value is None
                        else float(model[par].uncertainty_value)
                    ),
                    "DMXR1": float(model[par[:3] + "R1" + par[3:]].value),
                    "DMXR2": float(model[par[:3] + "R2" + par[3:]].value),
                    "fit": True,  # always True by construction (par in free)
                }
                for par in free
                if "DMX_" in par
            } or None
        except Exception as exc:
            logger.warning(
                f"Dispersion metadata unavailable for {getattr(self, 'name', '?')}: "
                f"{type(exc).__name__}: {exc}"
            )

        self._dispersion_metadata_ready = True

    def to_feather(
        self,
        filename: str | os.PathLike[str],
        noisedict: Mapping[str, Any] | None = None,
    ) -> None:
        """Write an Enterprise/Discovery-compatible feather snapshot of this pulsar."""
        from .feather_io import save_pulsar_feather

        save_pulsar_feather(self, filename, noisedict=noisedict)

    def timing_parameter_mapping(self) -> dict[str, dict[str, str]]:
        """Return canonical fitpars mapped to their per-PTA retained-par spellings.

        The value is ``NativeParam.par_key`` — how *this PTA's retained par*
        spells the parameter. ``nltiming.selection`` matches user selections
        against these values (its ``ECC``/``E`` grouping depends on the par
        spelling), so this rendering is part of the public contract.

        Values are exact par tokens except for repeated-keyword families -- see
        :class:`~metapulsar.pint_helpers.NativeParam`. A par writer must handle
        ``JUMPn`` by container token plus occurrence index: instance 1 reports
        ``JUMP``, later instances report the PINT name (``JUMP2``), which is not
        a token in the par.

        The returned dictionaries are fresh so interactive timing clients can
        inspect provenance without mutating the pulsar's canonical mapping.
        """
        return {
            name: {
                pta: native.par_key
                for pta, native in self._fitparameters.get(name, {}).items()
            }
            for name in self.fitpars
        }

    def timing(self, engines="jug", **engine_kwargs):
        """Open an immutable, engine-independent timing evaluator.

        This is a convenience wrapper over :class:`nltiming.TimingEvaluator`;
        all evaluation, scan, Jacobian, and fit logic remains in ``nltiming``.
        """
        from nltiming import TimingEvaluator

        return TimingEvaluator(
            self,
            engines=engines,
            **engine_kwargs,
        )

    def can_use_engines(self, engines="jug", *, linearized: bool = False) -> bool:
        """Return whether every PTA can honor the engine selection."""
        if getattr(self, "_timing_rows_filtered", False):
            return False
        from .engines import _IMPL_FAMILY, normalize_engines

        engines = normalize_engines(engines)
        for pta_name in self._pta_data:
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
                if not self._can_import_jug() or not self._pta_files_available(
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
            elif family == "vela":
                if native_compat != "pint":
                    return False
                if not self._can_import_vela():
                    return False
                if not self._pta_files_available(pta_name):
                    return False
            else:
                return False
        return True

    @staticmethod
    def _can_import_jug() -> bool:
        try:
            import jug.engine.session  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def _can_import_vela() -> bool:
        try:
            from pyvela import SPNTA  # noqa: F401
        except Exception:
            return False
        return True

    def _pta_files_available(self, pta_name: str) -> bool:
        files = self._pta_files.get(pta_name)
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
        files = self._pta_files.get(pta_name)
        if files is not None and files.timing_package in {"pint", "tempo2"}:
            return files.timing_package
        return self._get_timing_package(self._pta_data[pta_name])

    @staticmethod
    def _enforce_tempo2_native_cache_requirement(
        term_diagnostics: dict | None,
        *,
        tempo2_jug_options: dict | None,
        context: str,
    ) -> None:
        """Raise when MetaPulsar requires tempo2_obs_state in session cache."""
        from jug.timing import resolve_tempo2_jug_options

        options = resolve_tempo2_jug_options(tempo2_jug_options)
        if not options.get("require_native_cache", True):
            return
        td = term_diagnostics or {}
        if "tempo2_obs_state" not in td:
            raise RuntimeError(
                f"{context}: tempo2 JUG cache missing "
                "term_diagnostics['tempo2_obs_state']; call "
                "compute_residuals(force_recompute=True) with tempo2 compatibility."
            )

    def _build_jug_session(
        self,
        pta_name: str,
        compatibility: str,
        *,
        tempo2_native: str | None = None,
        tempo2_jug_options: dict | None = None,
        nonlinear_params: str | None = None,
    ):
        if not self._pta_files_available(pta_name):
            raise ValueError(
                f"Cannot build JUG timing session for '{pta_name}': missing par/tim inputs"
            )
        from jug.engine.session import TimingSession

        self._ensure_clock_aliases()
        files = self._pta_files[pta_name]
        return TimingSession(
            par_file=str(files.par_path),
            tim_file=str(files.tim_path),
            clock_dir=None if self._clock_dir is None else str(self._clock_dir),
            verbose=False,
            compatibility=compatibility,
            tempo2_native=tempo2_native,
            tempo2_jug_options=tempo2_jug_options,
            nonlinear_params=nonlinear_params,
        )

    def prime_jug_tempo2_sessions(
        self,
        engines: Mapping[str, str] | None = None,
        *,
        tempo2_native: str | None = None,
        tempo2_jug_options: dict | None = None,
        nonlinear_params: str | None = None,
        subtract_tzr: bool = False,
        force_recompute: bool = False,
    ) -> list[str]:
        """Refresh tempo2 JUG session caches so native_chain_static can be built."""
        from .engines import _IMPL_FAMILY, normalize_engines

        engines = normalize_engines(engines or {"tempo2": "jug", "pint": "jug"})
        primed: list[str] = []
        for pta_name in self._pta_data:
            native = self._native_compat(pta_name)
            if not str(native).lower().startswith("tempo2"):
                continue
            if _IMPL_FAMILY.get(engines.get(native, "jug")) != "jug":
                continue
            session = self._build_jug_session(
                pta_name,
                native,
                tempo2_native=tempo2_native,
                tempo2_jug_options=tempo2_jug_options,
                nonlinear_params=nonlinear_params,
            )
            session.compute_residuals(
                subtract_tzr=subtract_tzr,
                force_recompute=force_recompute,
            )
            cached = session._cached_result_by_mode.get(subtract_tzr)
            td = (cached or {}).get("term_diagnostics") or {}
            self._enforce_tempo2_native_cache_requirement(
                td,
                tempo2_jug_options=tempo2_jug_options,
                context=f"PTA {pta_name!r}",
            )
            primed.append(pta_name)
        return primed

    def timing_engine(
        self,
        engines="jug",
        *,
        linearized: bool = False,
        derivative_method: str = "analytic",
        tempo2_native: str | None = None,
        tempo2_jug_options: dict | None = None,
        nonlinear_params: str | None = None,
        prime_sessions: bool = True,
        verify_wiring: bool = False,
        subtract_tzr: bool = False,
    ):
        """Return a TimingEngine in canonical pulsar row order."""
        if getattr(self, "_timing_rows_filtered", False):
            raise ValueError(
                "nonlinear timing engines are not available after filter_data(); "
                "the retained engine per-PTA inputs still describe the original TOA rows. "
                "Filter the input tim files before constructing MetaPulsar."
            )
        from .engines import _IMPL_FAMILY, normalize_engines

        engines = normalize_engines(engines)
        if not self.can_use_engines(engines, linearized=linearized):
            raise ValueError(
                f"engines {engines} cannot be honored for pulsar '{self.name}'"
            )

        from jug.timing import resolve_tempo2_jug_options

        resolved_options = resolve_tempo2_jug_options(tempo2_jug_options)
        cache_key = (
            tuple(sorted(engines.items())),
            linearized,
            derivative_method,
            str(tempo2_native),
            tuple(sorted(resolved_options.items())),
            str(nonlinear_params),
            prime_sessions,
            verify_wiring,
            subtract_tzr,
            self.state_id(),
        )
        if cache_key in self._timing_engine_cache:
            return self._timing_engine_cache[cache_key]

        from nltiming.engine_support import LinearModel, validate_engine_against_pulsar

        from .engines import (
            JugEngine,
            LibstempoEngine,
            LinearizedJugEngine,
            LinearizedLibstempoEngine,
            LinearizedPintEngine,
            PintEngine,
            PtaContribution,
            build_engine,
        )

        pta_slices = self._get_pta_slices()
        fitpars = tuple(self.fitpars)
        global_index = {par: i for i, par in enumerate(fitpars)}
        contributions: list[PtaContribution] = []

        # Prime any tempo2-compatible JUG contribution that will build a real
        # JugEngine (not a linearized stand-in). Independent of derivative_method.
        if (
            prime_sessions
            and not linearized
            and any(
                _IMPL_FAMILY[engines[self._native_compat(pta)]] == "jug"
                and str(self._native_compat(pta)).lower().startswith("tempo2")
                for pta in self._pta_data
            )
        ):
            from jug.timing import resolve_tempo2_jug_options

            options = resolve_tempo2_jug_options(tempo2_jug_options)
            force = bool(options.get("force_cache_refresh", False))
            self.prime_jug_tempo2_sessions(
                engines,
                tempo2_native=tempo2_native,
                tempo2_jug_options=options,
                nonlinear_params=nonlinear_params,
                subtract_tzr=subtract_tzr,
                force_recompute=force,
            )

        for pta_name, psr in self._pta_data.items():
            slc = pta_slices[pta_name]
            rows = np.arange(slc.start, slc.stop, dtype=int)
            pta_fitpars = tuple(
                par for par in fitpars if pta_name in self._fitparameters.get(par, {})
            )
            pta_cols = [global_index[par] for par in pta_fitpars]
            pta_design = self._designmatrix[rows][:, pta_cols]
            theta_exact = self._pta_theta_exact(pta_name, pta_fitpars)
            linear_model = LinearModel.from_design(
                fitpars=pta_fitpars,
                design=pta_design,
                theta_exact=theta_exact,
            )

            native_compat = self._native_compat(pta_name)
            family = _IMPL_FAMILY[engines[native_compat]]
            if linearized and family in ("pint", "vela"):
                engine = LinearizedPintEngine.from_linear_model(linear_model)
            elif linearized and family == "tempo2":
                engine = LinearizedLibstempoEngine.from_linear_model(linear_model)
            elif linearized:
                engine = LinearizedJugEngine.from_linear_model(
                    linear_model, compatibility=native_compat
                )
            elif family == "vela":
                if not self._pta_files_available(pta_name):
                    raise ValueError(
                        f"Cannot build Vela timing session for '{pta_name}': "
                        "missing par/tim inputs"
                    )
                from .engines.vela import VelaEngine

                files = self._pta_files[pta_name]
                # SPNTA's param_names are PINT-format, so the PINT spelling is
                # what a Vela session can match; forwarding the par keyword
                # silently dropped aliased columns (XDOT, E, RA) to exact-linear.
                session_mapping = {
                    name: self._native_param(name, pta_name).pint_name
                    for name in pta_fitpars
                }
                # A Vela leg is PINT-native, so the leg's own (model, toas)
                # pair spares pyvela's empty-mask sweep a second TOA read.
                source = self._pulsars.get(pta_name)
                mask_reference = (
                    source if isinstance(source, tuple) and len(source) == 2 else None
                )
                engine = VelaEngine.from_files(
                    files.par_path,
                    files.tim_path,
                    linear_model=linear_model,
                    param_mapping=session_mapping,
                    mask_reference=mask_reference,
                )
            elif family == "pint":
                source = self._pulsars[pta_name]
                if not (isinstance(source, tuple) and len(source) == 2):
                    raise ValueError(f"PTA '{pta_name}' does not have PINT inputs")
                engine = PintEngine.from_contribution(
                    source[0],
                    source[1],
                    linear_model=linear_model,
                )
            elif family == "tempo2":
                # tempo2 remembers the spelling it parsed, which is the
                # retained par keyword.
                session_mapping = {
                    name: self._native_param(name, pta_name).par_key
                    for name in pta_fitpars
                }
                engine = LibstempoEngine.from_contribution(
                    self._pulsars[pta_name],
                    linear_model=linear_model,
                    param_mapping=session_mapping,
                )
            else:
                jug_session = self._build_jug_session(
                    pta_name,
                    native_compat,
                    tempo2_native=tempo2_native,
                    tempo2_jug_options=resolved_options,
                    nonlinear_params=nonlinear_params,
                )
                # Always required for tempo2 JUG: residual_delta goes through the
                # JAX graph under either derivative_method.
                if str(native_compat).lower().startswith("tempo2"):
                    cached = jug_session._cached_result_by_mode.get(subtract_tzr)
                    if cached is None:
                        jug_session.compute_residuals(
                            subtract_tzr=subtract_tzr,
                            force_recompute=False,
                        )
                        cached = jug_session._cached_result_by_mode.get(subtract_tzr)
                    self._enforce_tempo2_native_cache_requirement(
                        (cached or {}).get("term_diagnostics"),
                        tempo2_jug_options=resolved_options,
                        context=f"PTA {pta_name!r}",
                    )
                # Never guess a live engine's spelling: resolve each column
                # against the session's own parameter names by identity (JUG
                # ids are literally the identity form). Unmatched columns are
                # omitted, fail ``validate_fit_param``, and become
                # exact-linear -- the same outcome as Vela.
                session_param_names = tuple(getattr(jug_session, "params", {}).keys())
                session_mapping = {}
                for name in pta_fitpars:
                    resolved = resolve_native_source_name(
                        self._native_param(name, pta_name).pint_name,
                        session_param_names,
                        role=f"JUG session for PTA {pta_name!r}",
                        required=False,
                    )
                    if resolved is not None:
                        session_mapping[name] = resolved
                # Pass through timing_engine(subtract_tzr=...); previously the
                # JugEngine default (True) silently ignored this kwarg.
                engine = JugEngine.from_contribution(
                    jug_session,
                    linear_model=linear_model,
                    param_mapping=session_mapping,
                    subtract_tzr=subtract_tzr,
                    nonlinear_params=nonlinear_params,
                )

            contributions.append(
                PtaContribution(
                    name=pta_name,
                    row_indices=rows,
                    engine=engine,
                    exact_linear_fitpars=(
                        engine.exact_linear_fitpars()
                        if hasattr(engine, "exact_linear_fitpars")
                        else frozenset()
                    ),
                )
            )

        engine = build_engine(
            fitpars=fitpars,
            nrows=len(self._toas),
            contributions=contributions,
            design_matrix=self._designmatrix,
        )
        validate_engine_against_pulsar(engine, self, tol=1e-9)
        if verify_wiring:
            from .engines.jug import verify_jug_native_chain

            verify_jug_native_chain(engine)
        self._timing_engine_cache[cache_key] = engine
        return engine

    def state_id(self) -> str:
        """Return stable token for pulsar-tied cache invalidation."""
        pta_tags = [
            f"{pta}:{self._get_timing_package(psr)}"
            for pta, psr in self._pta_data.items()
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

        self._remove_nonidentifiable_parameters()

        if isinstance(self._flags, np.ndarray):
            self._flags = self._flags[mask]
        else:
            for key in self._flags:
                self._flags[key] = self._flags[key][mask]

        self.sort_data()
        self._timing_rows_filtered = True
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
    def dm(self) -> float | None:
        """Return reference-PTA DM (pc cm^-3), or None if absent."""
        self._ensure_dispersion_metadata()
        return self._dm

    @property
    def dmx(self) -> dict | None:
        """Return reference-PTA DMX metadata dict, or None if none are free."""
        self._ensure_dispersion_metadata()
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
        """Return array of engine flags."""
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
