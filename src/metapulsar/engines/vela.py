"""Per-PTA Vela.jl (pyvela) timing engine.

Vela evaluates residuals through ``SPNTA.time_residuals`` on its internal
(raw) parameter vector; ``scale_factors`` map PINT-unit values to raw units
componentwise. Working in *deltas* around the par-file reference sidesteps
Vela's float64 storage conventions entirely: the F0 big/small split and the
epoch-from-PEPOCH offsets are additive constants that cancel, so a native
PINT-unit delta scales directly into a raw-vector delta.

Enterprise/discovery own EFAC/EQUAD/ECORR/RN. The par handed to ``SPNTA`` is
therefore a delay-only ingest: WN/RN hyperparameter lines are stripped so
PINT never builds ``EcorrNoise`` and pyvela never ``ecorr_sort``s TOAs.

Not JAX-capable. Enterprise/PTMCMC works directly. nltiming may place this
engine behind a value-only Discovery host callback for derivative-free
sampling. NUTS and autodiff still require a JAX timing engine such as JUG.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nltiming.protocols import GaugeProvenance

from nltiming.engine_support import LinearModel, is_exact_linear_param

from ..parfile_lines import is_noise_line
from .delta import _is_zero_delta


class EmptyMaskParameterError(ValueError):
    """A *fitted* mask parameter selects no TOAs, so it has no design column."""


def _normalized_key_value(value) -> float | str:
    """Put a raw par token and a parsed PINT key value on equal footing.

    PINT parses ``MJD``/``FREQ`` key values to numbers (a ``Quantity`` for
    ``FREQ``) and sorts them, so the par text ``56160`` and the parsed
    ``56160.0`` are the same selector written two ways.
    """
    value = getattr(value, "value", value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value).lower()


def _mask_signature(key, key_values) -> tuple[str, tuple]:
    """Selector identity of a mask parameter: its key and its key values."""
    normalized = sorted((_normalized_key_value(v) for v in key_values), key=repr)
    return (str(key).lower(), tuple(normalized))


def _empty_jump_masks(model, toas):
    """Split JUMPs that select no TOA into ``(frozen, fitted)``."""
    component = model.components.get("PhaseJump")
    if component is None:
        return [], []
    frozen, fitted = [], []
    for param in component.get_jump_param_objects():
        if param.key is None:
            continue
        if len(param.select_toa_mask(toas)):
            continue
        (frozen if param.frozen else fitted).append(param)
    return frozen, fitted


def _is_jump_line(line: str) -> bool:
    tokens = line.split()
    return bool(tokens) and tokens[0].upper() == "JUMP"


def _line_matches_signature(tokens, signature) -> bool:
    key, values = signature
    end = 2 + len(values)
    if len(tokens) < end:
        return False
    return _mask_signature(tokens[1], tokens[2:end]) == signature


def _strip_empty_frozen_jumps(lines, model, toas) -> tuple[list[str], bool]:
    """Drop par lines for frozen JUMPs that select no TOA.

    ``pyvela.model.read_mask`` asserts that every mask parameter selects at
    least one TOA. Release pars carry leftover backend flag JUMPs (PPTA-style
    keys surviving into IPTA/AEI combined products, including flags named for
    another pulsar) that match nothing in the combined TOA set; tempo2 accepts
    them and they contribute nothing to residuals. A frozen empty JUMP is
    therefore dropped, while a *fitted* one is a real model error — it would
    be an all-zero design column — and raises.
    """
    frozen, fitted = _empty_jump_masks(model, toas)
    if fitted:
        raise EmptyMaskParameterError(
            "fitted JUMP parameters select no TOAs, so they carry no design "
            f"column: {sorted(p.name for p in fitted)}. Freeze or remove them "
            "in the par file."
        )
    if not frozen:
        return list(lines), False

    wanted = {_mask_signature(p.key, p.key_value) for p in frozen}
    kept: list[str] = []
    matched: set[tuple[str, tuple]] = set()
    for line in lines:
        tokens = line.split()
        if _is_jump_line(line):
            hit = next((s for s in wanted if _line_matches_signature(tokens, s)), None)
            if hit is not None:
                matched.add(hit)
                continue
        kept.append(line)
    if matched != wanted:
        missed = sorted(str(s) for s in wanted - matched)
        raise EmptyMaskParameterError(
            "could not locate par lines for empty frozen JUMP selectors "
            f"{missed}; refusing to hand pyvela a par it will reject"
        )
    return kept, True


def _load_mask_reference(par_file, tim_file):
    """Load the ``(model, toas)`` pair needed to evaluate TOA masks."""
    from pint.models import get_model_and_toas

    return get_model_and_toas(
        str(par_file),
        str(tim_file),
        planets=False,
        allow_T2=True,
        allow_tcb=True,
        add_tzr_to_model=False,
    )


def _prepare_par_for_spnta(par_file, tim_file, *, mask_reference=None) -> Path:
    """Return a par file pyvela's ``SPNTA`` can ingest as a residual engine.

    Residual deltas do not use Vela's likelihood kernel. This shim therefore
    strips WN/RN hyperparameter lines (same classifier as combination_writer)
    so PINT never builds ``EcorrNoise`` and pyvela never ``ecorr_sort``s TOAs.
    The caller's engine/retained par is not rewritten; only a temp ingest
    file is. Wideband-only DM-measurement lines (``DMJUMP``, ``DMEFAC``,
    ``DMEQUAD``) are unused on MetaPulsar's narrowband path and go with them.

    Additional no-op-for-residuals patches, already required by pyvela:

    * a fitted parameter with no frequentist uncertainty gets a placeholder
      uncertainty (cheat priors never enter this engine);
    * frozen mask parameters selecting zero TOAs abort ``read_mask``, so
      empty frozen JUMPs are dropped (see :func:`_strip_empty_frozen_jumps`).

    ``mask_reference`` is an optional pre-loaded ``(TimingModel, TOAs)`` pair
    for the JUMP sweep only. Noise-line removal does not read the tim.
    """
    par_file = Path(par_file)
    lines = par_file.read_text().splitlines()
    changed = False

    stripped: list[str] = []
    for line in lines:
        if is_noise_line(line):
            changed = True
            continue
        stripped.append(line)
    lines = stripped

    if any(_is_jump_line(line) for line in lines):
        if mask_reference is None:
            mask_reference = _load_mask_reference(par_file, tim_file)
        model, toas = mask_reference
        lines, jumped = _strip_empty_frozen_jumps(lines, model, toas)
        changed = changed or jumped

    patched: list[str] = []
    for line in lines:
        tokens = line.split()
        if len(tokens) == 3 and tokens[2] == "1":
            line = f"{line} 1.0"
            changed = True
        patched.append(line)
    if not changed:
        return par_file
    out = Path(tempfile.mkdtemp(prefix="metapulsar_vela_")) / par_file.name
    out.write_text("\n".join(patched) + "\n")
    return out


def _refuse_ecorr_kernel(spnta) -> None:
    """Raise if SPNTA still built an ECORR model that would permute TOAs."""
    model_pint = getattr(spnta, "model_pint", None)
    pint_ecorr = model_pint is not None and "EcorrNoise" in getattr(
        model_pint, "components", {}
    )
    if pint_ecorr or getattr(spnta, "has_ecorr_noise", False):
        raise RuntimeError(
            "Vela residual ingest still has EcorrNoise; noise lines must "
            "be stripped before SPNTA so TOAs stay in host order"
        )


class VelaDeltaEngine:
    """pyvela-backed residual-deviation engine.

    ``Vela.form_residuals`` does not remove a phase mean. The default
    ``phase_mean_mode=None`` therefore exports gauge-free residual deltas.
    Optional ``"weighted"`` / ``"unweighted"`` modes remain available for
    diagnostics that deliberately re-apply a mean.
    """

    def __init__(
        self,
        spnta: Any,
        *,
        isort: np.ndarray | None = None,
        phase_mean_mode: str | None = None,
        weights: np.ndarray | None = None,
    ):
        self._spnta = spnta
        self.param_names = [str(name) for name in spnta.param_names]
        self.fitpars = list(self.param_names)
        self._index = {name: i for i, name in enumerate(self.param_names)}
        self._scale = np.asarray(spnta.scale_factors, dtype=float)
        self._raw_ref = np.asarray(spnta.default_params, dtype=float)
        self._isort = None if isort is None else np.asarray(isort, dtype=int)
        reference = np.asarray(spnta.time_residuals(self._raw_ref), dtype=float)
        if self._isort is not None:
            reference = reference[self._isort]
        self._reference_residuals = reference

        if phase_mean_mode not in (None, "weighted", "unweighted"):
            raise ValueError(
                "phase_mean_mode must be None, 'weighted', or 'unweighted'; "
                f"got {phase_mean_mode!r}"
            )
        if phase_mean_mode is None:
            self._weights = None
        elif phase_mean_mode == "unweighted":
            self._weights = np.ones_like(self._reference_residuals)
        elif weights is not None:
            self._weights = np.asarray(weights, dtype=float).reshape(-1)
        else:
            errors = np.asarray(
                spnta.scaled_toa_unceritainties(self._raw_ref), dtype=float
            )
            self._weights = 1.0 / errors**2

    def delta_residuals(self, delta_params: dict[str, float]) -> np.ndarray:
        if _is_zero_delta(delta_params):
            return np.zeros_like(self._reference_residuals)

        raw = self._raw_ref.copy()
        for name, delta in delta_params.items():
            if name not in self._index:
                raise KeyError(f"Vela model has no free parameter '{name}'")
            idx = self._index[name]
            raw[idx] = raw[idx] + float(delta) * self._scale[idx]
        residuals = np.asarray(self._spnta.time_residuals(raw), dtype=float)
        if self._isort is not None:
            residuals = residuals[self._isort]
        delta_residuals = residuals - self._reference_residuals
        if self._weights is None:
            return delta_residuals
        mean = (self._weights @ delta_residuals) / self._weights.sum()
        return delta_residuals - mean


class VelaEngine:
    """Native Vela.jl residual-deltan engine.

    Nonlinear residual deltas come from ``VelaDeltaEngine``; the design matrix
    and reference theta metadata are served from the pulsar-derived
    ``LinearModel`` so the composite pulsar engine uses the same canonical
    columns as the pulsar design matrix. Fit parameters Vela cannot evaluate
    natively are routed to the exact-linear path.
    """

    engine_name = "vela"

    def __init__(
        self,
        *,
        engine: VelaDeltaEngine,
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
        native_fitpars: tuple[str, ...] | None = None,
        exact_linear_fitpars: frozenset[str] | set[str] | None = None,
    ):
        self._engine = engine
        self._model = linear_model
        self._param_mapping = dict(param_mapping or {})
        self.fitpars = tuple(linear_model.fitpars)
        self.native_units = dict(linear_model.native_units)
        self._native_fitpars = (
            self.fitpars if native_fitpars is None else tuple(native_fitpars)
        )
        self._native_indices = tuple(
            self.fitpars.index(name) for name in self._native_fitpars
        )
        self._exact_linear_fitpars = frozenset(exact_linear_fitpars or frozenset())
        self._exact_linear_indices = tuple(
            self.fitpars.index(name) for name in self._exact_linear_fitpars
        )

    @classmethod
    def from_contribution(
        cls,
        spnta: Any,
        *,
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
        isort: np.ndarray | None = None,
        phase_mean_mode: str | None = None,
        weights: np.ndarray | None = None,
    ) -> "VelaEngine":
        """Build a native Velan engine from an already-created ``SPNTA``."""
        engine = VelaDeltaEngine(
            spnta, isort=isort, phase_mean_mode=phase_mean_mode, weights=weights
        )
        mapping = dict(param_mapping or {})
        settable = set(engine.param_names)

        native_fitpars: list[str] = []
        exact_linear: list[str] = []
        for name in tuple(linear_model.fitpars):
            engine_param = mapping.get(name, name)
            if is_exact_linear_param(engine_param):
                exact_linear.append(name)
                continue
            if engine_param not in settable:
                exact_linear.append(name)
                continue
            native_fitpars.append(name)

        if not native_fitpars:
            raise ValueError(
                "No Vela-evaluable fit parameters remain after filtering; "
                f"exact-linear candidates: {exact_linear}"
            )

        return cls(
            engine=engine,
            linear_model=linear_model,
            param_mapping=mapping,
            native_fitpars=tuple(native_fitpars),
            exact_linear_fitpars=frozenset(exact_linear),
        )

    @classmethod
    def from_files(
        cls,
        par_file,
        tim_file,
        *,
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
        isort: np.ndarray | None = None,
        phase_mean_mode: str | None = None,
        weights: np.ndarray | None = None,
        spnta_kwargs: Mapping[str, Any] | None = None,
        mask_reference: tuple[Any, Any] | None = None,
    ) -> "VelaEngine":
        """Build a native Velan engine directly from par/tim files.

        ``mask_reference`` is an optional ``(TimingModel, TOAs)`` pair already
        loaded from these same files; it only serves the empty-mask sweep in
        :func:`_prepare_par_for_spnta`, which otherwise re-reads the TOAs.
        Residual ingest strips WN/RN lines first so pyvela cannot
        ``ecorr_sort`` TOAs.
        """
        from pyvela import SPNTA

        kwargs: dict[str, Any] = {"center_epochs": False, "check": False}
        kwargs.update(dict(spnta_kwargs or {}))
        par = _prepare_par_for_spnta(par_file, tim_file, mask_reference=mask_reference)
        spnta = SPNTA(str(par), str(tim_file), **kwargs)
        _refuse_ecorr_kernel(spnta)
        return cls.from_contribution(
            spnta,
            linear_model=linear_model,
            param_mapping=param_mapping,
            isort=isort,
            phase_mean_mode=phase_mean_mode,
            weights=weights,
        )

    def exact_linear_fitpars(self) -> frozenset[str]:
        """Pulsar fitpars evaluated exactly via the design matrix."""
        return self._exact_linear_fitpars

    def identically_linear_fitpars(self) -> frozenset[str]:
        """Fitpars whose engine delay is affine in delta."""
        return self._exact_linear_fitpars

    def reference_theta(self) -> np.ndarray:
        return self._model.reference_theta()

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self._model.theta_exact)

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta_theta, dtype=float).reshape(-1)
        if delta.shape != (len(self.fitpars),):
            raise ValueError("delta_theta shape mismatch with fitpars")

        delta_native = delta[np.asarray(self._native_indices, dtype=int)]
        delta_params = {
            self._param_mapping.get(name, name): float(value)
            for name, value in zip(self._native_fitpars, delta_native, strict=True)
        }
        return self._engine.delta_residuals(delta_params) + self._exact_linear_delta(
            delta
        )

    def design_matrix(self, params=None) -> np.ndarray:
        return np.asarray(self._model.design, dtype=float)

    def _exact_linear_delta(self, delta: np.ndarray) -> np.ndarray:
        if not self._exact_linear_indices:
            return np.zeros(self.design_matrix().shape[0], dtype=float)
        columns = np.asarray(
            self._model.design[:, list(self._exact_linear_indices)], dtype=float
        )
        return -(columns @ delta[np.asarray(self._exact_linear_indices, dtype=int)])

    def gauge_provenance(self) -> GaugeProvenance:
        return GaugeProvenance(
            export="none",
            reference_mode="none",
            reporting_mode="mean",
            reporting_weighted=True,
        )

    @property
    def gauge_applied(self) -> bool:
        return self.gauge_provenance().export != "none"
