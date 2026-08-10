"""Apply timing-fit deltas to a PINT model and write a MetaPulsar ``.par``.

Two design-matrix conventions are supported:

* ``native`` — MetaPulsar / nltiming engine ``design_matrix()`` columns are
  ``∂r/∂θ`` with ``θ`` in parameter ``.value`` units. Apply with
  :func:`apply_native_deltas` (same rule as
  :class:`~metapulsar.engines.delta.PintDeltaEngine`).
* ``pint_designmatrix`` — raw ``TimingModel.designmatrix()`` columns carry the
  heterogeneous units PINT returns (e.g. ``1/(Hz mas)``). Apply with
  :func:`apply_pint_designmatrix_deltas`, matching ``pint.fitter.WLSFitter``.

GLS MPE writeback for AEI combination products goes through
:func:`gls_update_and_write_par` so unit handling and validate/revert live in
one place.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import astropy.units as u
import numpy as np

from .parfile_header import ensure_metapulsar_par_header
from .pint_helpers import par_text_with_track_minus_2

DeltaConvention = Literal["native", "pint_designmatrix"]


@dataclass(frozen=True)
class ParUpdateResult:
    """Result of a GLS / delta apply + par write."""

    path: Path
    applied: dict[str, float]
    reverted: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    convention: DeltaConvention = "pint_designmatrix"


def apply_native_delta_to_param(param: Any, delta: float) -> None:
    """Apply one native-``.value`` delta (PintDeltaEngine convention)."""
    try:
        param.quantity = param.quantity + float(delta) * param.units
    except Exception:
        param.value = float(param.value) + float(delta)


def apply_native_deltas(model: Any, deltas: Mapping[str, float]) -> dict[str, float]:
    """Apply native-``.value`` deltas in-place; return new ``.value`` map."""
    applied: dict[str, float] = {}
    for name, delta in deltas.items():
        if str(name) not in getattr(model, "params", []):
            continue
        param = model[name]
        apply_native_delta_to_param(param, float(delta))
        applied[str(name)] = float(param.value)
    return applied


def pint_designmatrix_delta_to_quantity(dpar: float, column_unit: Any) -> u.Quantity:
    """Convert a PINT ``designmatrix`` solve coefficient to a quantity δ."""
    un = (1.0 / column_unit) * u.s
    return float(dpar) * un


def apply_pint_designmatrix_deltas(
    model: Any,
    deltas: Mapping[str, float],
    units_by_name: Mapping[str, Any],
) -> dict[str, float]:
    """Apply deltas from a raw PINT ``model.designmatrix()`` solve in-place."""
    applied: dict[str, float] = {}
    for name, dpar in deltas.items():
        if str(name) not in getattr(model, "params", []):
            continue
        if str(name) not in units_by_name:
            raise KeyError(
                f"missing designmatrix unit for parameter {name!r}; "
                "cannot apply pint_designmatrix delta"
            )
        param = model[name]
        dpv = pint_designmatrix_delta_to_quantity(float(dpar), units_by_name[name])
        try:
            param.quantity = param.quantity + dpv
        except Exception:
            # Fall back to dimensionless value add after converting into param units.
            scale = (dpv / param.units).decompose()
            param.value = float(param.value) + float(scale.value)
        applied[str(name)] = float(param.value)
    return applied


def validate_error(model: Any) -> str | None:
    """Return ``model.validate()`` error text, or ``None`` if valid."""
    try:
        model.validate()
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def revert_until_pint_valid(model: Any, before: Mapping[str, float]) -> list[str]:
    """Selectively revert updates until ``model.validate()`` passes.

    PINT is the source of truth for physical bounds (SINI, STIGMA, M2, ECC, …).

    Strategy:

    1. Error-chasing: revert updated parameters named in the validate message.
    2. If still invalid, revert remaining updates in descending ``|δ|`` order.
    """
    err = validate_error(model)
    if err is None:
        return []

    def _abs_delta(name: str) -> float:
        try:
            return abs(float(model[name].value) - float(before[name]))
        except Exception:  # noqa: BLE001
            return 0.0

    still = {n for n in before if n in getattr(model, "params", [])}
    reverted: list[str] = []

    for _ in range(len(still) + 1):
        err = validate_error(model)
        if err is None:
            return reverted
        msg = err.upper()
        hits = [
            n
            for n in still
            if n.upper() in msg and abs(float(model[n].value) - float(before[n])) > 0.0
        ]
        if not hits:
            break
        hits.sort(key=_abs_delta, reverse=True)
        name = hits[0]
        try:
            model[name].value = before[name]
        except Exception:  # noqa: BLE001
            still.discard(name)
            continue
        reverted.append(name)
        still.discard(name)

    err = validate_error(model)
    if err is None:
        return reverted
    for name in sorted(still, key=_abs_delta, reverse=True):
        try:
            model[name].value = before[name]
        except Exception:  # noqa: BLE001
            continue
        reverted.append(name)
        if validate_error(model) is None:
            return reverted
    return reverted


def _is_model_param(model: Any, name: str) -> bool:
    if str(name) in getattr(model, "params", []):
        return True
    try:
        model[name]
        return True
    except Exception:  # noqa: BLE001
        return False


def _solve_wls(
    residuals: np.ndarray, design: np.ndarray, variance: np.ndarray
) -> np.ndarray:
    """Weighted least squares: minimize ``||(r - M δ) / σ||`` with Tikhonov."""
    w = 1.0 / np.maximum(np.asarray(variance, dtype=float), 1e-30)
    M = np.asarray(design, dtype=float)
    mw = M * w[:, None]
    ata = M.T @ mw
    atr = M.T @ (w * np.asarray(residuals, dtype=float))
    ata = ata + 1e-18 * np.eye(ata.shape[0])
    try:
        return np.linalg.solve(ata, atr)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(ata, atr, rcond=None)[0]


def write_timing_model_par(
    model: Any,
    out_par: Path,
    *,
    source_par: Path | str | None = None,
    header_extra: Mapping[str, Any] | None = None,
    header_notes: Sequence[str] | None = None,
    preserve_track_minus_2: bool = True,
    include_info: bool = False,
) -> Path:
    """Write ``model`` to ``out_par`` with a MetaPulsar comment header."""
    out_par = Path(out_par)
    out_par.parent.mkdir(parents=True, exist_ok=True)
    model.write_parfile(str(out_par), include_info=include_info)
    body = out_par.read_text(encoding="utf-8")
    if preserve_track_minus_2 and source_par is not None:
        src = Path(source_par).read_text(encoding="utf-8")
        if re.search(r"^TRACK\s+-2\b", src, re.M) and not re.search(
            r"^TRACK\s+-2\b", body, re.M
        ):
            body = par_text_with_track_minus_2(body)
    body = ensure_metapulsar_par_header(
        body,
        extra=header_extra,
        notes=header_notes,
    )
    out_par.write_text(body, encoding="utf-8")
    return out_par


def gls_update_and_write_par(
    *,
    par_path: Path,
    tim_path: Path,
    variance: np.ndarray,
    out_par: Path,
    design_matrix: np.ndarray | None = None,
    param_names: Sequence[str] | None = None,
    designmatrix_units: Sequence[Any] | None = None,
    delta_convention: DeltaConvention | None = None,
    header_extra: Mapping[str, Any] | None = None,
    header_notes: Sequence[str] | None = None,
    validate: bool = True,
    preserve_track_minus_2: bool = True,
) -> ParUpdateResult:
    """WLS-update free timing params from ``par``+``tim`` and write ``out_par``.

    If ``design_matrix`` / ``param_names`` are omitted, loads via PINT and uses
    ``model.designmatrix()`` with ``delta_convention='pint_designmatrix'``.

    Pass a MetaPulsar / nltiming engine design matrix with
    ``delta_convention='native'`` to apply engine-native deltas.
    """
    from pint.models import get_model_and_toas
    from pint.residuals import Residuals

    par_path = Path(par_path)
    tim_path = Path(tim_path)
    out_par = Path(out_par)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, toas = get_model_and_toas(
            str(par_path),
            str(tim_path),
            allow_T2=True,
            allow_tcb=True,
            planets=True,
            include_pn=True,
            ell1h_shapiro="absorbed",
        )
        res = np.asarray(Residuals(toas, model).time_resids.to_value("s"), dtype=float)

        units_list: list[Any] | None
        if design_matrix is None:
            design, fit_params, units_list = model.designmatrix(toas, incfrozen=False)
            M = np.asarray(design, dtype=float)
            names = list(fit_params)
            convention: DeltaConvention = (
                delta_convention
                if delta_convention is not None
                else "pint_designmatrix"
            )
        else:
            M = np.asarray(design_matrix, dtype=float)
            if param_names is None:
                raise ValueError(
                    "param_names is required when design_matrix is provided"
                )
            names = list(param_names)
            units_list = (
                list(designmatrix_units) if designmatrix_units is not None else None
            )
            convention = delta_convention if delta_convention is not None else "native"

        if M.size == 0 or M.ndim != 2 or not names:
            text = par_path.read_text(encoding="utf-8")
            out_par.parent.mkdir(parents=True, exist_ok=True)
            out_par.write_text(
                ensure_metapulsar_par_header(
                    text, extra=header_extra, notes=header_notes
                ),
                encoding="utf-8",
            )
            return ParUpdateResult(path=out_par, applied={}, convention=convention)

        if M.shape[0] != res.size:
            raise RuntimeError(
                f"design matrix shape {M.shape} incompatible with residuals {res.size}"
            )
        if M.shape[1] != len(names):
            raise RuntimeError(
                f"design matrix columns {M.shape[1]} != len(param_names) {len(names)}"
            )

        var = np.asarray(variance, dtype=float)
        if var.size != res.size:
            var = np.asarray(toas.get_errors().to_value("s"), dtype=float) ** 2

        delta_vec = _solve_wls(res, M, var)
        deltas: dict[str, float] = {}
        skipped: list[str] = []
        units_by_name: dict[str, Any] = {}
        for name, d, unit in zip(
            names,
            delta_vec,
            units_list if units_list is not None else [None] * len(names),
        ):
            if not _is_model_param(model, str(name)):
                skipped.append(str(name))
                continue
            deltas[str(name)] = float(d)
            if unit is not None:
                units_by_name[str(name)] = unit

        before = {name: float(model[name].value) for name in deltas}
        if convention == "native":
            applied = apply_native_deltas(model, deltas)
        elif convention == "pint_designmatrix":
            if units_list is None:
                raise ValueError(
                    "designmatrix_units required for delta_convention="
                    "'pint_designmatrix' when supplying an external design matrix"
                )
            applied = apply_pint_designmatrix_deltas(model, deltas, units_by_name)
        else:
            raise ValueError(f"unknown delta_convention: {convention!r}")

        reverted: list[str] = []
        if validate:
            reverted = revert_until_pint_valid(model, before)
            for name in reverted:
                applied[name] = float(before[name])
                applied[f"{name}__reverted"] = 1.0
            final_err = validate_error(model)
            if final_err is not None:
                raise ValueError(
                    "GLS timing update still fails PINT validate() after reverting "
                    f"updated parameters: {final_err}; reverted={reverted}"
                )

        extras = {
            "Product": "gls-optimized",
            "gls_delta_convention": convention,
            **(dict(header_extra) if header_extra else {}),
        }
        write_timing_model_par(
            model,
            out_par,
            source_par=par_path if preserve_track_minus_2 else None,
            header_extra=extras,
            header_notes=header_notes,
            preserve_track_minus_2=preserve_track_minus_2,
        )
        return ParUpdateResult(
            path=out_par,
            applied=applied,
            reverted=tuple(reverted),
            skipped=tuple(skipped),
            convention=convention,
        )
