"""Apply timing-fit deltas to a PINT model and write a MetaPulsar ``.par``.

Two design-matrix conventions are supported:

* ``native`` — MetaPulsar / nltiming engine ``design_matrix()`` columns are
  ``∂r/∂θ`` with ``θ`` in parameter ``.value`` units. Apply with
  :func:`apply_native_deltas` (same rule as
  :class:`~metapulsar.engines.delta.PintDeltaEngine`).
* ``pint_designmatrix`` — raw ``TimingModel.designmatrix()`` columns carry the
  heterogeneous units PINT returns (e.g. ``1/(Hz mas)``). Apply with
  :func:`apply_pint_designmatrix_deltas`, matching ``pint.fitter.WLSFitter``.

GLS MPE writeback goes through :func:`gls_update_and_write_par` so unit
handling, validate/revert and the dual-engine par write live in one place.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Literal, Mapping, Sequence

import astropy.units as u
import numpy as np

from .parfile_header import ensure_metapulsar_par_header
from .parfile_lines import iter_active_par_lines, join_par_lines, replace_token
from .pint_helpers import resolve_parameter_alias

DeltaConvention = Literal["native", "pint_designmatrix"]


class ParTransplantError(ValueError):
    """A model parameter could not be matched to exactly one source par line."""


@dataclass(frozen=True)
class ParUpdateResult:
    """Result of a GLS / delta apply + par write."""

    path: Path
    applied: dict[str, float]
    reverted: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    convention: DeltaConvention = "pint_designmatrix"
    # (name, old_token, new_token) for every line the writer actually edited.
    changed_tokens: tuple[tuple[str, str, str], ...] = ()


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


# PINT stores FDJUMPs internally as ``FDpJUMPq`` and only respells them
# ``FDJUMPp`` in ``FDJump.print_par(format="tempo2")`` -- component level, so a
# line-level editor has to reconcile the two spellings itself (pint/models/
# fdjump.py:66,195). Everything else goes through PINT's alias table.
_FDJUMP_PINT_RE: Final[re.Pattern[str]] = re.compile(r"^FD(\d+)JUMP(\d+)?$", re.I)
_FDJUMP_TEMPO2_RE: Final[re.Pattern[str]] = re.compile(r"^FDJUMP(\d+)$", re.I)


def _normalized_par_key(name: str) -> str:
    """Dialect-free key for one par line or PINT parameter name."""
    match = _FDJUMP_TEMPO2_RE.match(name) or _FDJUMP_PINT_RE.match(name)
    if match:
        return f"FD{int(match.group(1))}JUMP"
    return resolve_parameter_alias(name.upper()).upper()


def _render_key_value(key_value: Any) -> str:
    """Spell one mask key-value the way PINT's own writer does.

    Mirrors ``maskParameter.as_parfile_line`` (pint/models/parameter.py:2062):
    ``Time`` via ``time_to_mjd_string``, ``Quantity`` via its bare ``.value``,
    everything else via ``str``. Needed because ``key_identifier``
    (parameter.py:1866) parses ``FREQ`` into a ``Quantity`` (``str(kv)`` would
    be ``"1400.0 MHz"``) and ``MJD`` into a ``float``.
    """
    from astropy.time import Time
    from pint.pulsar_mjd import time_to_mjd_string

    if isinstance(key_value, Time):
        return time_to_mjd_string(key_value)
    if isinstance(key_value, u.Quantity):
        return str(key_value.value)
    return str(key_value)


def _key_values_match(
    tokens: Sequence[str], key_values: Sequence[Any], key: str
) -> bool:
    """Compare a source line's key-value tokens against a parameter's.

    By key type:

    * ``MJD`` / ``FREQ`` -- numeric, compared as a **sorted multiset**, because
      ``maskParameter.__init__`` sorts ``key_value`` (parameter.py:1895): a par
      line reading ``JUMP MJD 56000 55000`` is stored as ``[55000.0, 56000.0]``.
    * ``TEL`` -- PINT normalizes the code through the observatory registry
      (``_get_observatory_name``, parameter.py:77), so ``TEL ao`` is stored as
      ``arecibo``; compare via the same lookup, falling back to text.
    * flags (``-pta``, ``-fe``, ``-sys``, ...) and ``NAME`` -- case-insensitive
      text.
    """
    rendered = [_render_key_value(kv) for kv in key_values]
    if len(tokens) != len(rendered):
        return False

    key_lower = str(key).lower()
    if key_lower in ("mjd", "freq"):
        try:
            return sorted(float(t) for t in tokens) == sorted(
                float(r) for r in rendered
            )
        except (TypeError, ValueError):
            return False

    if key_lower == "tel":
        from pint.observatory import get_observatory

        try:
            return [get_observatory(t).name for t in tokens] == [
                get_observatory(r).name for r in rendered
            ]
        except Exception:  # noqa: BLE001 - unknown code: fall through to text
            pass

    return [t.upper() for t in tokens] == [r.upper() for r in rendered]


@dataclass(frozen=True)
class _ParLineSlot:
    """Where one parameter's value token sits in the source text."""

    line_index: int
    value_index: int


def _find_par_line_slot(par_text: str, param: Any) -> _ParLineSlot:
    """Locate the single active line that carries ``param``'s value.

    Plain parameters match on the alias-resolved key alone. Mask parameters
    (``JUMP``, ``FDJUMPn``, ``EFAC``...) must also match their key and every
    key-value, and their value token sits at ``2 + len(key_value)``.

    Zero or multiple matches raise rather than guess -- the same posture as
    :class:`~metapulsar.file_discovery.AmbiguousFileError`.
    """
    name = str(getattr(param, "origin_name", None) or param.name)
    key = _normalized_par_key(name)
    mask_key = getattr(param, "key", None)
    key_values = list(getattr(param, "key_value", None) or [])

    hits: list[_ParLineSlot] = []
    for index, line in iter_active_par_lines(par_text):
        tokens = line.split()
        if _normalized_par_key(tokens[0]) != key:
            continue
        if not mask_key:
            if len(tokens) < 2:
                continue
            hits.append(_ParLineSlot(index, 1))
            continue
        value_index = 2 + len(key_values)
        if len(tokens) <= value_index:
            continue
        if tokens[1].upper() != str(mask_key).upper():
            continue
        if not _key_values_match(tokens[2:value_index], key_values, mask_key):
            continue
        hits.append(_ParLineSlot(index, value_index))

    if len(hits) != 1:
        detail = f"key={key!r}"
        if mask_key:
            detail += f" mask={mask_key!r} key_value={key_values!r}"
        raise ParTransplantError(
            f"parameter {name!r} matches {len(hits)} active par lines ({detail}); "
            "require exactly one to transplant its value"
        )
    return hits[0]


def _value_token(param: Any) -> str:
    """New value token, formatted exactly as PINT would print it."""
    return str(param.str_quantity(param.quantity)).strip()


def transplant_param_values(
    par_text: str,
    model: Any,
    names: Iterable[str],
) -> tuple[str, dict[str, tuple[str, str]]]:
    """Return ``par_text`` with ``names``' value tokens taken from ``model``.

    Every other byte of the source is preserved: line order, whitespace, fit
    flags, uncertainties, and tempo2-only spelling (``FDJUMPn``, ``MODE 1``,
    ``TZRMJD``/``TZRFRQ``/``TZRSITE``, ``TRACK -2``, ``T2CMETHOD``). That is
    what keeps the product readable by both engines; a PINT re-serialization
    does not.

    ``names`` are PINT parameter names (``ECC``, ``FD1JUMP1``), not source
    spellings (``E``, ``FDJUMP1``); an unknown name raises ``KeyError`` from
    PINT's ``TimingModel.__getitem__``. :class:`ParTransplantError` is reserved
    for line matching -- a name PINT knows but the source par does not carry, or
    carries twice.

    Parameters whose formatted token is unchanged are left untouched. That is a
    backstop only: PINT reprints many unchanged quantities differently
    (``0.0000216340`` -> ``2.1634e-05``), so callers must pass only parameters
    whose *value* moved -- see :func:`gls_update_and_write_par`.

    Returns ``(text, changed)`` where ``changed`` maps parameter name to
    ``(old_token, new_token)``.
    """
    lines = par_text.splitlines()
    changed: dict[str, tuple[str, str]] = {}
    for name in names:
        param = model[name]
        # Resolved against the original text; line indices are stable because a
        # splice never adds or removes lines.
        slot = _find_par_line_slot(par_text, param)
        line = lines[slot.line_index]
        old_token = line.split()[slot.value_index]
        new_token = _value_token(param)
        if new_token == old_token:
            continue
        lines[slot.line_index] = replace_token(line, slot.value_index, new_token)
        changed[str(name)] = (old_token, new_token)
    return join_par_lines(lines, like=par_text), changed


def write_timing_model_par(
    model: Any,
    out_par: Path,
    *,
    source_par: Path | str | None = None,
    updated_params: Sequence[str] | None = None,
    header_extra: Mapping[str, Any] | None = None,
    header_notes: Sequence[str] | None = None,
    include_info: bool = False,
) -> tuple[Path, dict[str, tuple[str, str]]]:
    """Write ``model`` to ``out_par`` with a MetaPulsar comment header.

    With ``source_par`` the source text is *edited*: only ``updated_params``'
    value tokens change (:func:`transplant_param_values`), so the file keeps the
    source's dialect -- including ``TRACK -2`` -- and stays readable by both
    PINT and tempo2. Without a source par the model is serialized by PINT, which
    yields a PINT-dialect file that tempo2 will open but only partly understand.
    ``# writer:`` in the header records which path ran.

    Returns ``(path, changed)``; ``changed`` is empty on the PINT-dump path.
    """
    out_par = Path(out_par)
    out_par.parent.mkdir(parents=True, exist_ok=True)

    if source_par is not None:
        source_text = Path(source_par).read_text(encoding="utf-8")
        body, changed = transplant_param_values(
            source_text, model, list(updated_params or ())
        )
        writer = "transplant"
    else:
        model.write_parfile(str(out_par), include_info=include_info)
        body = out_par.read_text(encoding="utf-8")
        changed = {}
        writer = "pint-dump"

    # ``writer`` is stamped last: it records which path actually ran, so a
    # caller's header_extra must not be able to misreport it.
    extra = {**(dict(header_extra) if header_extra else {}), "writer": writer}
    if source_par is not None:
        extra.setdefault("source_par", str(source_par))
        extra["changed_params"] = len(changed)
    body = ensure_metapulsar_par_header(
        body, format="PINT", extra=extra, notes=header_notes
    )
    out_par.write_text(body, encoding="utf-8")
    return out_par, changed


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
) -> ParUpdateResult:
    """WLS-update free timing params from ``par``+``tim`` and write ``out_par``.

    If ``design_matrix`` / ``param_names`` are omitted, loads via PINT and uses
    ``model.designmatrix()`` with ``delta_convention='pint_designmatrix'``.

    Pass a MetaPulsar / nltiming engine design matrix with
    ``delta_convention='native'`` to apply engine-native deltas.

    The product is written by value transplant onto ``par_path``
    (:func:`write_timing_model_par`), so it keeps that file's dialect and stays
    readable by both engines. Only parameters whose *value* moved are spliced.
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

        # Built once, before the empty-designmatrix return, so both products
        # carry identical provenance.
        extras = {
            "Product": "gls-optimized",
            "gls_delta_convention": convention,
            **(dict(header_extra) if header_extra else {}),
        }

        if M.size == 0 or M.ndim != 2 or not names:
            out_path, _changed = write_timing_model_par(
                model,
                out_par,
                source_par=par_path,
                updated_params=(),
                header_extra=extras,
                header_notes=header_notes,
            )
            return ParUpdateResult(path=out_path, applied={}, convention=convention)

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

        # Only parameters whose value actually moved are spliced: the revert
        # sentinels must never reach the transplant, and a zero-delta free
        # parameter would otherwise be rewritten purely cosmetically (PINT
        # reprints 0.0000216340 as 2.1634e-05).
        updated_names = [
            name
            for name in deltas
            if name not in set(reverted) and applied[name] != before[name]
        ]

        out_path, changed = write_timing_model_par(
            model,
            out_par,
            source_par=par_path,
            updated_params=updated_names,
            header_extra=extras,
            header_notes=header_notes,
        )
        return ParUpdateResult(
            path=out_path,
            applied=applied,
            reverted=tuple(reverted),
            skipped=tuple(skipped),
            convention=convention,
            changed_tokens=tuple(
                (name, old, new) for name, (old, new) in changed.items()
            ),
        )
