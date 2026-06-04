"""Analytical WLS 1σ uncertainties for timing parameters."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def wls_uncertainties(
    pint_model,
    param_names: Sequence[str],
    *,
    pint_toas=None,
) -> dict[str, float]:
    """Return 1σ WLS uncertainties in PINT native ``param.value`` units.

    When ``pint_toas`` is provided, run PINT's ``WLSFitter`` and read the diagonal
    of the parameter covariance matrix. Otherwise use positive ``uncertainty_value``
    from the model, or raise if neither is available.
    """
    names = list(param_names)
    if pint_toas is not None:
        return _wls_from_fitter(pint_model, pint_toas, names)

    sigmas: dict[str, float] = {}
    missing: list[str] = []
    for name in names:
        if name not in pint_model.params:
            missing.append(name)
            continue
        param = pint_model[name]
        err = getattr(param, "uncertainty_value", None)
        if err is not None and float(err) > 0.0:
            sigmas[name] = float(err)
        else:
            missing.append(name)
    if missing:
        raise ValueError(
            "Cannot determine WLS sigma for parameter(s) "
            f"{', '.join(missing)} without pint_toas or positive uncertainty_value."
        )
    return sigmas


def _wls_from_fitter(pint_model, pint_toas, names: list[str]) -> dict[str, float]:
    from pint.fitter import WLSFitter

    fitter = WLSFitter(pint_toas, pint_model)
    fitter.fit_toas()

    if hasattr(fitter, "get_fitparams_uncertainty"):
        unc_by_name = fitter.get_fitparams_uncertainty()
        sigmas: dict[str, float] = {}
        for name in names:
            if name not in unc_by_name:
                param = pint_model[name]
                unc = getattr(param, "uncertainty_value", None)
                if unc is not None and float(unc) > 0.0:
                    sigmas[name] = float(unc)
                    continue
                raise ValueError(
                    f"Parameter {name!r} missing from WLS fit uncertainties."
                )
            sigma = float(unc_by_name[name])
            if sigma <= 0.0 or not np.isfinite(sigma):
                raise ValueError(f"Non-positive WLS sigma for {name!r}: {sigma}")
            sigmas[name] = sigma
        return sigmas

    raise ValueError("PINT WLSFitter did not expose get_fitparams_uncertainty().")


def coerce_standardization_scale(
    name: str,
    *,
    wls_sigma: float,
    standardization: Mapping[str, object] | None,
) -> float:
    """Resolve transform scale: explicit standardization overrides WLS σ."""
    if standardization and name in standardization:
        spec = standardization[name]
        if isinstance(spec, Mapping):
            scale = spec.get("scale")
            if scale is not None:
                scale_f = float(scale)
                if scale_f <= 0.0:
                    raise ValueError(f"Non-positive scale for {name!r}: {scale_f}")
                return scale_f
    sigma = float(wls_sigma)
    if sigma <= 0.0:
        raise ValueError(
            f"Non-positive WLS sigma for {name!r}: {sigma}. "
            "Provide standardization scale explicitly."
        )
    return sigma
