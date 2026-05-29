"""Backend engines for nonlinear timing residual deviations."""

from __future__ import annotations

from copy import deepcopy
from typing import Protocol, runtime_checkable

import astropy.units as u
import numpy as np


@runtime_checkable
class TimingDeltaEngine(Protocol):
    """Engine that computes timing residual deviations for one PTA dataset."""

    param_names: list[str]

    def delta_residuals(self, delta_params: dict[str, float]) -> np.ndarray:
        """Return ``r(theta0 + delta_params) - r(theta0)`` in seconds."""
        ...


def _is_zero_delta(delta_params: dict[str, float]) -> bool:
    return not delta_params or all(
        float(value) == 0.0 for value in delta_params.values()
    )


class PintDeltaEngine:
    """PINT-backed residual-deviation engine."""

    def __init__(self, model, toas):
        self._model = model
        self._toas = toas
        self.param_names = list(getattr(model, "params", []))
        self._reference_time_residuals = self._time_residuals(model)

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
        return (phase_resids / frequency).to(u.s).value.astype(float)

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
        self.param_names = ["Offset", *list(lt_psr.pars(which="set"))]
        self._reference_values = {
            name: lt_psr[name].val
            for name in self.param_names
            if hasattr(lt_psr[name], "val")
        }
        self._reference_residuals = np.asarray(lt_psr.residuals(), dtype=float)
        self._designmatrix = np.asarray(lt_psr.designmatrix(), dtype=float)

    def delta_residuals(self, delta_params: dict[str, float]) -> np.ndarray:
        if _is_zero_delta(delta_params):
            return np.zeros_like(self._reference_residuals)

        try:
            for name, delta in delta_params.items():
                if name not in self._reference_values:
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


class JugDeltaEngine:
    """Placeholder for a future JUG backend adapter."""

    param_names: list[str] = []

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "JUG timing backend adapter is not implemented yet for nonlinear timing deltas."
        )

    def delta_residuals(
        self, delta_params: dict[str, float]
    ) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError(
            "JUG timing backend adapter is not implemented yet for nonlinear timing deltas."
        )


def build_delta_engine(pta_input) -> TimingDeltaEngine:
    """Build the default residual-deviation engine for a retained PTA input."""
    if isinstance(pta_input, tuple) and len(pta_input) == 2:
        model, toas = pta_input
        return PintDeltaEngine(model, toas)
    return Tempo2DeltaEngine(pta_input)
