"""PINT-backed priors and standardized timing-parameter registry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy import stats

from .transforms import AffineTransform
from .wls import coerce_standardization_scale, wls_uncertainties

CHEAT_PRIOR_SIGMA_MULTIPLIER = 50.0


class PriorPolicy(str, Enum):
    FALLBACK = "fallback"
    STRICT = "strict"
    EXPLICIT = "explicit"


@dataclass(frozen=True)
class PriorOverrideSpec:
    kind: Literal["uniform"]
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if float(self.lower) >= float(self.upper):
            raise ValueError(
                f"Prior override requires lower < upper, got {self.lower} >= {self.upper}."
            )


def _require_pint():
    try:
        from pint.models import parameter as pint_parameter
        from pint.models.priors import Prior, UniformBoundedRV, UniformUnboundedRV
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PINT is required for nonlinear timing priors.") from exc
    return Prior, UniformBoundedRV, UniformUnboundedRV, pint_parameter


@dataclass(frozen=True)
class PintPriorAdapter:
    """Thin wrapper around a PINT or scipy frozen prior on physical theta."""

    distribution_name: str
    _rv: Any
    is_proper: bool
    is_uniform: bool
    source: str

    @classmethod
    def from_pint_parameter(cls, param, *, source: str) -> PintPriorAdapter:
        Prior, UniformBoundedRV, UniformUnboundedRV, pint_parameter = _require_pint()
        if isinstance(getattr(param, "prior", None), Prior):
            rv = param.prior._rv
            improper = isinstance(rv, UniformUnboundedRV)
            return cls(
                distribution_name=type(rv).__name__,
                _rv=rv,
                is_proper=not improper,
                is_uniform=isinstance(rv, stats.uniform)
                or isinstance(rv, UniformBoundedRV),
                source=source,
            )
        raise TypeError(f"Expected pint.models.priors.Prior on {param.name!r}.")

    @classmethod
    def from_uniform(
        cls, lower: float, upper: float, *, source: str
    ) -> PintPriorAdapter:
        Prior, UniformBoundedRV, _, _ = _require_pint()
        rv = UniformBoundedRV(float(lower), float(upper))
        return cls(
            distribution_name="UniformBoundedRV",
            _rv=rv,
            is_proper=True,
            is_uniform=True,
            source=source,
        )

    @classmethod
    def from_override(
        cls, spec: PriorOverrideSpec | Mapping[str, Any] | Any
    ) -> PintPriorAdapter:
        Prior, UniformBoundedRV, UniformUnboundedRV, _ = _require_pint()
        if isinstance(spec, PriorOverrideSpec):
            return cls.from_uniform(spec.lower, spec.upper, source="user_override")
        if isinstance(spec, Prior):
            if isinstance(spec._rv, UniformUnboundedRV):
                raise ValueError("prior_overrides must use a proper bounded prior.")
            return cls(
                distribution_name=type(spec._rv).__name__,
                _rv=spec._rv,
                is_proper=True,
                is_uniform=isinstance(spec._rv, stats.uniform)
                or isinstance(spec._rv, UniformBoundedRV),
                source="user_override",
            )
        if isinstance(spec, Mapping):
            if spec.get("kind") != "uniform":
                raise ValueError(f"Unsupported override kind: {spec.get('kind')!r}")
            return cls.from_uniform(
                float(spec["lower"]),
                float(spec["upper"]),
                source="user_override",
            )
        raise TypeError(f"Unsupported prior override spec: {type(spec)!r}")

    def logpdf(self, theta: float) -> float:
        return float(self._rv.logpdf(float(theta)))

    def ppf(self, q: float) -> float:
        q = float(np.clip(q, 0.0, 1.0))
        return float(self._rv.ppf(q))

    def support(self) -> tuple[float, float]:
        lo = float(self.ppf(0.0))
        hi = float(self.ppf(1.0))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return (-np.inf, np.inf)
        return (lo, hi)


@dataclass(frozen=True)
class SampledTimingParameter:
    name: str
    theta_ref: float
    transform: AffineTransform
    prior: PintPriorAdapter
    source: str
    units: str
    sigma_wls: float

    def theta_from_z(self, z: float) -> float:
        return float(self.theta_ref) + self.transform.to_physical(float(z))

    def delta_from_z(self, z: float) -> float:
        return self.transform.to_physical(float(z))

    def z_from_theta(self, theta: float) -> float:
        return self.transform.to_standardized(float(theta) - float(self.theta_ref))

    def z_bounds(self) -> tuple[float, float]:
        lo, hi = self.prior.support()
        if not np.isfinite(lo) or not np.isfinite(hi):
            return (-np.inf, np.inf)
        z_lo = self.z_from_theta(lo)
        z_hi = self.z_from_theta(hi)
        return (min(z_lo, z_hi), max(z_lo, z_hi))

    def logpdf_theta(self, theta: float) -> float:
        return self.prior.logpdf(theta)

    def logpdf_z(self, z: float) -> float:
        return self.logpdf_theta(self.theta_from_z(z)) + math.log(
            abs(float(self.transform.scale))
        )

    def prior_transform_z(self, q: float) -> float:
        theta = self.prior.ppf(float(q))
        return self.z_from_theta(theta)


class SampledTimingParameterRegistry:
    """Ordered registry of sampled timing parameters with PINT-backed priors."""

    def __init__(self, parameters: Sequence[SampledTimingParameter]):
        self.sampled_params = [p.name for p in parameters]
        self._params = {p.name: p for p in parameters}

    @property
    def parameters(self) -> tuple[SampledTimingParameter, ...]:
        return tuple(self._params[name] for name in self.sampled_params)

    def theta_from_z_params(self, z_params: Mapping[str, float]) -> dict[str, float]:
        self._ensure_known(z_params)
        return {
            name: self._params[name].theta_from_z(float(z_params[name]))
            for name in self.sampled_params
        }

    def delta_from_z_params(self, z_params: Mapping[str, float]) -> dict[str, float]:
        self._ensure_known(z_params)
        return {
            name: self._params[name].delta_from_z(float(z_params[name]))
            for name in self.sampled_params
        }

    def z_from_theta_params(
        self, theta_params: Mapping[str, float]
    ) -> dict[str, float]:
        self._ensure_known(theta_params)
        return {
            name: self._params[name].z_from_theta(float(theta_params[name]))
            for name in self.sampled_params
        }

    def reference_z_params(self) -> dict[str, float]:
        return {name: 0.0 for name in self.sampled_params}

    def logprior_z(self, z_params: Mapping[str, float]) -> float:
        self._ensure_known(z_params)
        total = 0.0
        for name in self.sampled_params:
            lp = self._params[name].logpdf_z(float(z_params[name]))
            if not np.isfinite(lp):
                return -np.inf
            total += lp
        return float(total)

    def prior_transform_z(self, cube: np.ndarray) -> np.ndarray:
        cube = np.asarray(cube, dtype=float).reshape(-1)
        if cube.shape[0] != len(self.sampled_params):
            raise ValueError(
                f"prior_transform cube length {cube.shape[0]} != "
                f"{len(self.sampled_params)} parameters."
            )
        return np.array(
            [
                self._params[name].prior_transform_z(float(cube[idx]))
                for idx, name in enumerate(self.sampled_params)
            ],
            dtype=float,
        )

    def z_bounds_by_name(self) -> dict[str, tuple[float, float]]:
        return {name: self._params[name].z_bounds() for name in self.sampled_params}

    def metadata(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in self.sampled_params:
            p = self._params[name]
            lo, hi = p.prior.support()
            z_lo, z_hi = p.z_bounds()
            rows.append(
                {
                    "name": name,
                    "theta_ref": p.theta_ref,
                    "center": p.transform.center,
                    "scale": p.transform.scale,
                    "sigma_wls": p.sigma_wls,
                    "prior_source": p.source,
                    "prior_distribution": p.prior.distribution_name,
                    "theta_support_lo": lo,
                    "theta_support_hi": hi,
                    "z_support_lo": z_lo,
                    "z_support_hi": z_hi,
                    "units": p.units,
                }
            )
        return rows

    def to_parameter_space(self):
        from .parameter_space import SampledTimingParameterSpace

        return SampledTimingParameterSpace.from_registry(self)

    def enterprise_parameter_priors(self) -> dict[str, object]:
        from enterprise.signals import parameter

        priors: dict[str, object] = {}
        for name in self.sampled_params:
            p = self._params[name]
            if not p.prior.is_uniform:
                raise NotImplementedError(
                    f"Enterprise Phase 1 supports uniform induced-z priors only; "
                    f"{name} has {p.prior.distribution_name} (source={p.source})."
                )
            z_lo, z_hi = p.z_bounds()
            if not np.isfinite(z_lo) or not np.isfinite(z_hi):
                raise NotImplementedError(
                    f"Enterprise Phase 1 requires finite z bounds for {name!r}."
                )
            priors[name] = parameter.Uniform(float(z_lo), float(z_hi))
        return priors

    def _ensure_known(self, params: Mapping[str, float]) -> None:
        unknown = sorted(set(params.keys()) - set(self.sampled_params))
        if unknown:
            raise KeyError(f"Unknown standardized parameter(s): {', '.join(unknown)}")


def _reject_derived_sampled(pint_model, name: str) -> None:
    _, _, _, pint_parameter = _require_pint()
    param = pint_model[name]
    if isinstance(param, pint_parameter.funcParameter):
        raise ValueError(
            f"Cannot sample derived parameter {name!r}; sample underlying fit parameters."
        )


def _resolve_prior(
    *,
    param,
    name: str,
    theta_ref: float,
    sigma_wls: float,
    prior_policy: PriorPolicy,
    prior_overrides: Mapping[str, Any] | None,
) -> PintPriorAdapter:
    overrides = prior_overrides or {}
    if name in overrides:
        return PintPriorAdapter.from_override(overrides[name])

    Prior, _, UniformUnboundedRV, _ = _require_pint()
    pint_prior = getattr(param, "prior", None)
    improper = isinstance(pint_prior, Prior) and isinstance(
        pint_prior._rv, UniformUnboundedRV
    )
    has_proper_pint = isinstance(pint_prior, Prior) and not improper

    if prior_policy == PriorPolicy.EXPLICIT:
        raise ValueError(
            f"Parameter {name!r} missing from prior_overrides under explicit policy."
        )

    if prior_policy == PriorPolicy.STRICT:
        if has_proper_pint:
            return PintPriorAdapter.from_pint_parameter(param, source="pint")
        raise ValueError(
            f"strict policy: parameter {name!r} lacks a proper PINT prior "
            f"(got {type(getattr(pint_prior, '_rv', None)).__name__})."
        )

    # fallback (default)
    if has_proper_pint:
        return PintPriorAdapter.from_pint_parameter(param, source="pint")

    half = CHEAT_PRIOR_SIGMA_MULTIPLIER * float(sigma_wls)
    return PintPriorAdapter.from_uniform(
        float(theta_ref) - half,
        float(theta_ref) + half,
        source="cheat_wls",
    )


def build_sampled_timing_parameter_registry(
    *,
    pint_model,
    sampled_params: Sequence[str],
    reference_values: Mapping[str, float] | None = None,
    standardization: Mapping[str, object] | None = None,
    prior_policy: str | PriorPolicy = PriorPolicy.FALLBACK,
    prior_overrides: Mapping[str, Any] | None = None,
    wls_sigmas: Mapping[str, float] | None = None,
    pint_toas=None,
) -> SampledTimingParameterRegistry:
    """Build a registry from a PINT timing model and sampled parameter names."""
    policy = (
        PriorPolicy(prior_policy) if isinstance(prior_policy, str) else prior_policy
    )
    names = list(sampled_params)
    overrides = dict(prior_overrides or {})
    unknown = sorted(set(overrides) - set(names))
    if unknown:
        allowed = ", ".join(names)
        raise ValueError(
            f"Unknown prior_overrides keys: {', '.join(unknown)}. "
            f"Allowed sampled parameter names: {allowed}."
        )
    refs = dict(reference_values or {})
    sigmas = dict(
        wls_sigmas or wls_uncertainties(pint_model, names, pint_toas=pint_toas)
    )

    parameters: list[SampledTimingParameter] = []
    for name in names:
        if name not in pint_model.params:
            raise KeyError(f"PINT model has no parameter {name!r}.")
        _reject_derived_sampled(pint_model, name)
        param = pint_model[name]
        theta_ref = float(refs.get(name, param.value))
        sigma_wls = float(sigmas[name])
        center = 0.0
        if standardization and name in standardization:
            spec = standardization[name]
            if isinstance(spec, Mapping) and "center" in spec:
                center = float(spec["center"])
        scale = coerce_standardization_scale(
            name, wls_sigma=sigma_wls, standardization=standardization
        )
        prior = _resolve_prior(
            param=param,
            name=name,
            theta_ref=theta_ref,
            sigma_wls=sigma_wls,
            prior_policy=policy,
            prior_overrides=prior_overrides,
        )
        units = str(getattr(param, "units", "") or "")
        parameters.append(
            SampledTimingParameter(
                name=name,
                theta_ref=theta_ref,
                transform=AffineTransform(center=center, scale=scale),
                prior=prior,
                source=prior.source,
                units=units,
                sigma_wls=sigma_wls,
            )
        )
    return SampledTimingParameterRegistry(parameters)
