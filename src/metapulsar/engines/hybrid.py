"""Caller-declared residual linearization for every engine family.

``nonlinear_params`` is the closed hybrid mode JUG executes inside its
residual graph (``jug.fitting.nonlinear_params``): ``"binary"`` keeps only the
binary axes nonlinear and routes every other axis through the design matrix
(``-M δ``); ``"binary+"`` additionally keeps ``PX`` nonlinear. The libstempo,
Vela and PINT adapters realise the *same* model with their native/exact-linear
split: the native call sees only the nonlinear axes — so astrometry sits at the
par-file reference inside the binary delay, exactly as JUG freezes it — and the
linearized axes add ``-M[:, lin] δ_lin``.

Binary classification is delegated to JUG's parameter registry so the
partition is identical across engine families. Only callers that set a
non-``None`` mode pay the JUG import.
"""

from __future__ import annotations


def validate_nonlinear_params(value: str | None) -> str | None:
    """Return ``None`` or the normalized closed-set mode string."""
    if value is None:
        return None
    try:
        from jug.fitting.nonlinear_params import validate_nonlinear_params as _v
    except ImportError as exc:  # pragma: no cover - exercised without jug
        raise ImportError(
            "nonlinear_params requires jug (it owns the closed mode set and the "
            "binary-parameter registry); install jug to use hybrid residual "
            "linearization"
        ) from exc
    return _v(value)


def is_hybrid_native_param(engine_param: str, mode: str | None) -> bool:
    """Return whether ``engine_param`` stays on the native residual path.

    ``mode=None`` keeps every axis native. Under ``"binary"`` only binary
    parameters are native; under ``"binary+"`` ``PX`` is native as well.
    ``engine_param`` is the engine spelling (unsuffixed), which is what JUG's
    registry classifies; PTA-suffixed host names must be mapped first.
    """
    resolved = validate_nonlinear_params(mode)
    if resolved is None:
        return True
    from jug.fitting.nonlinear_params import NONLINEAR_PARAMS_BINARY_PLUS
    from jug.model.parameter_spec import canonicalize_param_name, is_binary_param

    if is_binary_param(engine_param):
        return True
    if resolved == NONLINEAR_PARAMS_BINARY_PLUS:
        return str(canonicalize_param_name(engine_param)).upper() == "PX"
    return False


def hybrid_linearized_fitpars(
    fitpars, engine_names, mode: str | None
) -> frozenset[str]:
    """Host fitpars the hybrid mode moves onto the design-matrix path."""
    resolved = validate_nonlinear_params(mode)
    if resolved is None:
        return frozenset()
    engine_names = dict(engine_names or {})
    return frozenset(
        name
        for name in fitpars
        if not is_hybrid_native_param(engine_names.get(name, name), resolved)
    )


def resolve_hybrid_partition(
    *,
    fitpars,
    param_mapping,
    mode: str | None,
    native_fitpars,
    exact_linear_fitpars,
):
    """Resolve one adapter's ``(native, exact_linear)`` split for a mode.

    Used by the adapter constructors so a directly built engine can never
    carry a hybrid mode it does not execute: with no explicit
    ``native_fitpars`` the hybrid partition *is* the default, and an explicit
    native list that contains a linearized axis is refused rather than
    silently evaluated natively. ``mode=None`` keeps today's defaults (every
    fitpar native unless the caller said otherwise).
    """
    resolved = validate_nonlinear_params(mode)
    fitpars = tuple(fitpars)
    mapping = dict(param_mapping or {})
    exact = frozenset(exact_linear_fitpars or frozenset())
    if resolved is None:
        native = fitpars if native_fitpars is None else tuple(native_fitpars)
        return native, exact

    linearized = hybrid_linearized_fitpars(fitpars, mapping, resolved)
    if native_fitpars is None:
        native = tuple(name for name in fitpars if name not in linearized)
        return native, exact | linearized

    native = tuple(native_fitpars)
    offenders = sorted(set(native) & linearized)
    if offenders:
        raise ValueError(
            f"nonlinear_params={resolved!r} linearizes {offenders}, but they "
            "were passed as native_fitpars; a stamped mode must match the "
            "partition the engine actually evaluates"
        )
    # An explicit native list may name only the axes the engine can evaluate;
    # the mode still owns every linearized axis, so fold them into the
    # design-matrix set rather than leaving their deltas unevaluated.
    exact = exact | linearized
    dropped = sorted(set(fitpars) - set(native) - exact)
    if dropped:
        raise ValueError(
            f"fitpars {dropped} are neither native nor exact-linear under "
            f"nonlinear_params={resolved!r}; their deltas would be silently "
            "dropped from the residual"
        )
    return native, exact
