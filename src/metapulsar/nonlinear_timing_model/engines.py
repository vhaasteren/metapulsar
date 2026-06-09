"""Backend engines for nonlinear timing residual deviations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import astropy.units as u
import numpy as np


@runtime_checkable
class TimingDeltaEngine(Protocol):
    """Engine that computes timing residual deviations for one PTA dataset."""

    param_names: list[str]
    fitpars: list[str]

    def delta_residuals(self, delta_params: dict[str, float]) -> np.ndarray:
        """Return ``r(theta0 + delta_params) - r(theta0)`` in seconds."""
        ...


@runtime_checkable
class JaxTimingDeltaEngine(Protocol):
    """JAX-native timing residual engine for Discovery/NUTS."""

    param_names: list[str]
    fitpars: list[str]
    sampled_params: tuple[str, ...]
    output_shape: tuple[int, ...]
    output_dtype: Any

    def residual_delta_jax(self, z_flat): ...

    def timing_delay_jax(self, z_flat): ...

    def residual_delta_np(self, z_flat: np.ndarray) -> np.ndarray: ...


def _is_zero_delta(delta_params: dict[str, float]) -> bool:
    return not delta_params or all(
        float(value) == 0.0 for value in delta_params.values()
    )


class PintDeltaEngine:
    """PINT-backed residual-deviation engine."""

    def __init__(self, model, toas, *, isort: np.ndarray | None = None):
        self._model = model
        self._toas = toas
        self._isort = None if isort is None else np.asarray(isort, dtype=int)
        self.param_names = list(getattr(model, "params", []))
        self.fitpars = list(self.param_names)
        self._reference_time_residuals = self._time_residuals(model)
        self._reference_residuals = self._reference_time_residuals

    def delta_residuals(self, delta_params: dict[str, float]) -> np.ndarray:
        if _is_zero_delta(delta_params):
            return np.zeros_like(self._reference_time_residuals)

        model = deepcopy(self._model)
        for name, delta in delta_params.items():
            self._set_parameter_delta(model, name, delta)

        return self._time_residuals(model) - self._reference_time_residuals

    def _set_parameter_delta(self, model, name: str, delta: float) -> None:
        if not hasattr(model, name):
            raise KeyError(f"PINT model has no parameter '{name}'")

        param = getattr(model, name)
        try:
            param.quantity = param.quantity + float(delta) * param.units
        except Exception:
            param.value = param.value + float(delta)

    def _time_residuals(self, model) -> np.ndarray:
        phase_resids = self._phase_residuals(model)
        frequency = self._spin_frequency(model)
        return (
            (phase_resids / frequency)
            .to(u.s)
            .value.astype(float)[
                self._isort if self._isort is not None else slice(None)
            ]
        )

    def _spin_frequency(self, model):
        if "Spindown" in model.components:
            return model.F0.quantity.to(u.Hz)
        if "P0" in model.params:
            return (1.0 / model.P0.quantity).to(u.Hz)
        raise AttributeError("PINT model has no F0/P0 spin frequency parameter")

    def _phase_residuals(self, model):
        from pint.phase import Phase

        if "delta_pulse_number" not in self._toas.table.colnames:
            self._toas.table["delta_pulse_number"] = np.zeros(
                len(self._toas.get_mjds())
            )
        delta_pulse_numbers = Phase(self._toas.table["delta_pulse_number"])

        subtract_mean = "PhaseOffset" not in model.components
        if getattr(model, "TRACK").value == "-2":
            track_mode = "use_pulse_numbers"
        elif getattr(model, "TRACK").value == "0":
            track_mode = "nearest"
        elif "pulse_number" in self._toas.table.columns and not np.any(
            np.isnan(self._toas.table["pulse_number"])
        ):
            track_mode = "use_pulse_numbers"
        else:
            track_mode = "nearest"

        if track_mode == "use_pulse_numbers":
            pulse_num = self._toas.get_pulse_numbers()
            if pulse_num is None or np.any(np.isnan(pulse_num)):
                raise ValueError("Pulse numbers are required but missing from TOAs")
            modelphase = model.phase(self._toas, abs_phase=True) + delta_pulse_numbers
            residualphase = modelphase - Phase(
                pulse_num.copy(), np.zeros_like(pulse_num)
            )
            full = residualphase.int + residualphase.frac
        else:
            modelphase = model.phase(self._toas) + delta_pulse_numbers
            if subtract_mean:
                modelphase -= Phase(modelphase.int[0], modelphase.frac[0])
            residualphase = Phase(np.zeros_like(modelphase.frac), modelphase.frac)
            full = residualphase.int + residualphase.frac

        if not subtract_mean:
            return full

        errors = self._toas.get_errors().to(u.s).value
        if np.any(errors == 0):
            raise ValueError(
                "Some TOA errors are zero - cannot calculate residual mean"
            )
        weights = 1.0 / errors**2
        mean = np.average(full.value, weights=weights)
        return (full.value - mean) * full.unit


class Tempo2DeltaEngine:
    """libstempo-backed residual-deviation engine."""

    def __init__(self, lt_psr):
        self._psr = lt_psr
        self._fit_param_names = list(lt_psr.pars())
        offset = ["Offset"] if "Offset" in self._fit_param_names else []
        self.param_names = offset + list(lt_psr.pars(which="set"))
        self.fitpars = list(self._fit_param_names)
        self._reference_values = {}
        for name in self.param_names:
            try:
                par = lt_psr[name]
            except KeyError:
                continue
            if hasattr(par, "val"):
                self._reference_values[name] = par.val
        self._reference_residuals = np.asarray(lt_psr.residuals(), dtype=float)
        self._designmatrix = np.asarray(lt_psr.designmatrix(), dtype=float)

    def delta_residuals(self, delta_params: dict[str, float]) -> np.ndarray:
        if _is_zero_delta(delta_params):
            return np.zeros_like(self._reference_residuals)

        try:
            for name, delta in delta_params.items():
                if name not in self._reference_values:
                    if name == "Offset":
                        continue
                    raise KeyError(f"libstempo pulsar has no parameter '{name}'")
                self._psr[name].val = self._reference_values[name] + float(delta)
            self._psr.formbats()
            residuals = np.asarray(self._psr.residuals(), dtype=float)
            delta_residuals = residuals - self._reference_residuals
            return delta_residuals + self._linearized_unrecomputed_delta(
                delta_params, delta_residuals
            )
        finally:
            for name, value in self._reference_values.items():
                self._psr[name].val = value
            self._psr.formbats()

    def _linearized_unrecomputed_delta(
        self, delta_params: dict[str, float], delta_residuals: np.ndarray
    ) -> np.ndarray:
        linearized = np.zeros_like(self._reference_residuals)
        if "Offset" in delta_params:
            linearized += self._designmatrix[:, 0] * float(delta_params["Offset"])

        if not np.array_equal(delta_residuals, np.zeros_like(delta_residuals)):
            return linearized

        for name, delta in delta_params.items():
            if name == "Offset" or name not in self._fit_param_names:
                continue
            col = self._fit_param_names.index(name) + 1
            linearized += self._designmatrix[:, col] * float(delta)

        return linearized


# Canonical timing names (Enterprise/libstempo fitpars) -> JUG backend aliases.
_CANONICAL_TO_BACKEND_ALIASES: dict[str, tuple[str, ...]] = {
    "RAJ": ("RAJ", "RA"),
    "DECJ": ("DECJ", "DEC"),
    "ELONG": ("ELONG", "LAMBDA", "RAJ", "RA"),
    "ELAT": ("ELAT", "BETA", "DECJ", "DEC"),
    "PMRA": ("PMRA",),
    "PMDEC": ("PMDEC",),
    "PMELONG": ("PMELONG", "PMLAMBDA", "PMRA"),
    "PMELAT": ("PMELAT", "PMBETA", "PMDEC"),
    "ECC": ("ECC", "E"),
    "TASC": ("TASC", "T0"),
}


def infer_jug_param_mapping(
    canonical_names: Sequence[str],
    backend_names: Sequence[str] | set[str],
) -> dict[str, str]:
    """Map canonical fit parameter names to JUG backend keys when they differ."""
    backend = set(backend_names)
    mapping: dict[str, str] = {}
    for canon in canonical_names:
        if canon in backend:
            continue
        for candidate in _CANONICAL_TO_BACKEND_ALIASES.get(canon, (canon,)):
            if candidate in backend:
                mapping[canon] = candidate
                break
    return mapping


class JugDeltaEngine:
    """JUG residual-deviation engine.

    Parameters
    ----------
    residual_source
        Either a JUG session-like object exposing ``compute_residuals`` and
        optionally ``residuals_at_params`` for fast in-memory evaluation,
        or a callable with signature ``callable(overrides: dict[str, float] | None)``.
        The return value may be:
        - a dict containing ``residuals_us`` or ``residuals_sec``, or
        - a 1D residual array in seconds.
    fitpars
        Canonical parameter names used by higher-level partitioning logic.
        If omitted, defaults to ``param_names``.
    param_names
        Backend parameter names accepted by ``residual_source`` overrides.
        If omitted, inferred from ``residual_source.params`` when available.
    param_mapping
        Optional canonical-to-backend mapping for overrides.
    reference_params
        Optional backend reference parameter values. If omitted, inferred from
        ``residual_source.params`` when available.
    subtract_tzr
        Forwarded to JUG session residual evaluation.
    isort
        Optional index array that reorders residuals into Discovery feather /
        Enterprise PINT TOA order (same convention as ``PintDeltaEngine``).
    """

    def __init__(
        self,
        residual_source,
        *,
        fitpars: list[str] | None = None,
        param_names: list[str] | None = None,
        param_mapping: Mapping[str, str] | None = None,
        reference_params: Mapping[str, float] | None = None,
        subtract_tzr: bool = True,
        isort: np.ndarray | None = None,
    ):
        self._residual_source = residual_source
        self._use_fast_jug_path = hasattr(
            residual_source, "residuals_at_params"
        ) and hasattr(residual_source, "compute_residuals")
        self._residual_callable = (
            None if self._use_fast_jug_path else self._coerce_callable(residual_source)
        )
        self._subtract_tzr = bool(subtract_tzr)
        self._isort = None if isort is None else np.asarray(isort, dtype=int)
        self._param_mapping = dict(param_mapping or {})

        inferred_backend_names = self._infer_backend_param_names(
            residual_source, param_names
        )
        self.param_names = list(inferred_backend_names)

        self.fitpars = list(fitpars) if fitpars is not None else list(self.param_names)

        inferred_reference = self._infer_reference_params(
            residual_source, reference_params
        )
        self._reference_params = inferred_reference
        self._reference_time_residuals = self._evaluate_seconds(overrides=None)
        self._reference_residuals = self._reference_time_residuals

    def _coerce_callable(
        self, residual_source
    ) -> Callable[[dict[str, float] | None], object]:
        if hasattr(residual_source, "compute_residuals"):

            def _session_eval(overrides: dict[str, float] | None):
                return residual_source.compute_residuals(
                    params=overrides,
                    subtract_tzr=self._subtract_tzr,
                )

            return _session_eval

        if callable(residual_source):
            return residual_source

        raise TypeError(
            "JugDeltaEngine requires a session-like object with 'compute_residuals' "
            "or a callable residual evaluator."
        )

    @staticmethod
    def _infer_backend_param_names(
        residual_source, param_names: list[str] | None
    ) -> list[str]:
        if param_names is not None:
            return [str(name) for name in param_names]
        source_params = getattr(residual_source, "params", None)
        if isinstance(source_params, Mapping):
            return [str(name) for name in source_params.keys()]
        return []

    @staticmethod
    def _infer_reference_params(
        residual_source, reference_params: Mapping[str, float] | None
    ) -> dict[str, float]:
        if reference_params is not None:
            return {str(name): float(value) for name, value in reference_params.items()}
        source_params = getattr(residual_source, "params", None)
        if isinstance(source_params, Mapping):
            return {
                str(name): float(value)
                for name, value in source_params.items()
                if isinstance(value, (int, float, np.number))
            }
        return {}

    @staticmethod
    def _extract_residuals_seconds(result) -> np.ndarray:
        if isinstance(result, Mapping):
            if "residuals_us" in result:
                return np.asarray(result["residuals_us"], dtype=float) * 1.0e-6
            if "residuals_sec" in result:
                return np.asarray(result["residuals_sec"], dtype=float)
            if "residuals" in result:
                return np.asarray(result["residuals"], dtype=float)
            raise KeyError(
                "Residual result mapping must include 'residuals_us', "
                "'residuals_sec', or 'residuals'."
            )

        return np.asarray(result, dtype=float)

    def _evaluate_session(self, overrides: dict[str, float] | None):
        session = self._residual_source
        if overrides is None or not overrides:
            return session.compute_residuals(
                params=None,
                subtract_tzr=self._subtract_tzr,
            )
        if self._use_fast_jug_path:
            return session.residuals_at_params(
                overrides,
                subtract_tzr=self._subtract_tzr,
            )
        return session.compute_residuals(
            params=overrides,
            subtract_tzr=self._subtract_tzr,
        )

    def _evaluate_seconds(self, overrides: dict[str, float] | None) -> np.ndarray:
        if self._use_fast_jug_path or hasattr(
            self._residual_source, "compute_residuals"
        ):
            result = self._evaluate_session(overrides)
        else:
            result = self._residual_callable(overrides)
        residuals = self._extract_residuals_seconds(result)
        if residuals.ndim != 1:
            raise ValueError("Residual evaluator must return a 1D residual vector.")
        if self._isort is not None:
            return residuals[self._isort]
        return residuals

    def _canonical_to_backend(self, param_name: str) -> str:
        return self._param_mapping.get(param_name, param_name)

    def _build_absolute_overrides(
        self, delta_params: dict[str, float]
    ) -> dict[str, float]:
        overrides: dict[str, float] = {}
        for canonical_name, delta in delta_params.items():
            if float(delta) == 0.0:
                continue
            backend_name = self._canonical_to_backend(canonical_name)
            if self.param_names and backend_name not in self.param_names:
                raise KeyError(f"JUG backend has no parameter '{backend_name}'")
            if backend_name not in self._reference_params:
                raise KeyError(
                    f"Reference value is unavailable for backend parameter "
                    f"'{backend_name}'."
                )
            overrides[backend_name] = float(
                self._reference_params[backend_name]
            ) + float(delta)
        return overrides

    def delta_residuals(self, delta_params: dict[str, float]) -> np.ndarray:
        if _is_zero_delta(delta_params):
            return np.zeros_like(self._reference_residuals)

        unknown = sorted(set(delta_params) - set(self.fitpars))
        if unknown:
            raise KeyError(
                f"Unknown JUG timing-delta parameter(s): {', '.join(unknown)}"
            )

        overrides = self._build_absolute_overrides(delta_params)
        perturbed_residuals = self._evaluate_seconds(overrides)
        if perturbed_residuals.shape != self._reference_residuals.shape:
            raise ValueError(
                "Perturbed residual vector length does not match reference residuals."
            )
        return perturbed_residuals - self._reference_residuals


def build_delta_engine(pta_input) -> TimingDeltaEngine:
    """Build the default residual-deviation engine for a retained PTA input."""
    if isinstance(pta_input, tuple) and len(pta_input) == 2:
        model, toas = pta_input
        return PintDeltaEngine(model, toas)
    if hasattr(pta_input, "compute_residuals"):
        return JugDeltaEngine(pta_input)
    return Tempo2DeltaEngine(pta_input)
