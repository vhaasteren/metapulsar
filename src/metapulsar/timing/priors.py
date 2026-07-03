"""Prior extraction and policy resolution for timing parameters."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Literal

from metapulsar.pint_helpers import resolve_parameter_alias

from .bijectors import AxisPrior, PriorBijector
from .precision import ExactNativeRef

PriorPolicy = Literal["fallback", "strict", "explicit"]
ABSOLUTE_FORMING_FAMILIES = frozenset({"log_uniform"})


def validate_prior_policy(policy: str) -> PriorPolicy:
    """Validate the configured prior policy."""
    if policy not in {"fallback", "strict", "explicit"}:
        raise ValueError(
            "prior_policy must be one of {'fallback', 'strict', 'explicit'}"
        )
    return policy  # type: ignore[return-value]


def set_prior(
    overrides: dict[str, AxisPrior], param_name: str, prior: AxisPrior
) -> dict[str, AxisPrior]:
    """Set/override one prior with canonical-name normalization."""
    canonical = resolve_parameter_alias(param_name)
    merged = dict(overrides)
    merged[canonical] = prior
    return merged


def _axis_prior_from_object(prior_obj) -> AxisPrior | None:
    """Best-effort extraction of simple prior families from PINT prior objects."""
    if prior_obj is None:
        return None

    family_name = type(prior_obj).__name__.lower()

    # Uniform-like with explicit bounds.
    lower = getattr(prior_obj, "lower_bound", None)
    upper = getattr(prior_obj, "upper_bound", None)
    if lower is not None and upper is not None:
        if "log" in family_name:
            return AxisPrior(
                family="log_uniform", lower=float(lower), upper=float(upper)
            )
        if "trunc" in family_name:
            mean = getattr(prior_obj, "mu", getattr(prior_obj, "mean", 0.0))
            std = getattr(prior_obj, "sigma", getattr(prior_obj, "std", 1.0))
            return AxisPrior(
                family="truncated_normal",
                lower=float(lower),
                upper=float(upper),
                mean=float(mean),
                std=float(std),
            )
        return AxisPrior(family="uniform", lower=float(lower), upper=float(upper))

    # Normal-like with mean/sigma attrs.
    mean = getattr(prior_obj, "mu", None)
    std = getattr(prior_obj, "sigma", None)
    if mean is None:
        mean = getattr(prior_obj, "mean", None)
    if std is None:
        std = getattr(prior_obj, "std", None)
    if mean is not None and std is not None:
        return AxisPrior(family="normal", mean=float(mean), std=float(std))

    return None


def _decimal_string(value) -> str:
    if isinstance(value, str):
        return value
    return str(float(value))


def _decimal_delta(value, ref_str: str) -> float:
    with localcontext() as ctx:
        ctx.prec = 50
        return float(Decimal(_decimal_string(value)) - Decimal(ref_str))


def _ref_mapping(
    names: tuple[str, ...],
    theta_ref: ExactNativeRef | dict[str, str | float | int] | None,
) -> dict[str, str]:
    if theta_ref is None:
        return {name: "0.0" for name in names}
    if isinstance(theta_ref, ExactNativeRef):
        mapping = theta_ref.as_mapping()
    elif all(isinstance(v, str) for v in theta_ref.values()):
        mapping = ExactNativeRef.from_mapping(theta_ref).as_mapping()  # type: ignore[arg-type]
    else:
        mapping = ExactNativeRef.from_float_mapping(theta_ref).as_mapping()  # type: ignore[arg-type]
    missing = [name for name in names if name not in mapping]
    if missing:
        raise ValueError(f"Missing theta_ref entries for parameters: {missing}")
    return mapping


def _to_delta_prior(prior: AxisPrior, ref_str: str) -> AxisPrior:
    """Convert an absolute/native prior description into delta-valued constants."""
    if prior.family == "normal":
        return AxisPrior(
            family="normal",
            mean=_decimal_delta(prior.mean, ref_str),
            std=prior.std,
        )
    if prior.family == "uniform":
        return AxisPrior(
            family="uniform",
            lower=_decimal_delta(prior.lower, ref_str),
            upper=_decimal_delta(prior.upper, ref_str),
        )
    if prior.family == "truncated_normal":
        return AxisPrior(
            family="truncated_normal",
            lower=_decimal_delta(prior.lower, ref_str),
            upper=_decimal_delta(prior.upper, ref_str),
            mean=_decimal_delta(prior.mean, ref_str),
            std=prior.std,
        )
    if prior.family == "log_uniform":
        with localcontext() as ctx:
            ctx.prec = 50
            offset = float(Decimal(ref_str))
        return AxisPrior(
            family="log_uniform",
            lower=prior.lower,
            upper=prior.upper,
            offset=offset,
        )
    raise ValueError(f"Unsupported prior family: {prior.family}")


@dataclass(frozen=True)
class PriorBlock:
    """Resolved per-parameter priors and their source labels."""

    names: tuple[str, ...]
    priors: tuple[AxisPrior, ...]
    sources: dict[str, str]

    @classmethod
    def from_fitpars(
        cls,
        fitpars: list[str] | tuple[str, ...],
        *,
        policy: PriorPolicy = "fallback",
        overrides: dict[str, AxisPrior] | None = None,
        named_defaults: dict[str, AxisPrior] | None = None,
        theta_ref: ExactNativeRef | dict[str, str | float | int] | None = None,
        pint_model=None,
    ) -> "PriorBlock":
        """Resolve priors for fit parameters from overrides/PINT/fallback policy."""
        policy = validate_prior_policy(policy)
        overrides = overrides or {}
        named_defaults = named_defaults or {}
        canonical_overrides = {
            resolve_parameter_alias(name): prior for name, prior in overrides.items()
        }
        canonical_defaults = {
            resolve_parameter_alias(name): prior
            for name, prior in named_defaults.items()
        }

        names: list[str] = [resolve_parameter_alias(p) for p in fitpars]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate fit parameters after alias normalization")
        refs = _ref_mapping(tuple(names), theta_ref)

        priors: list[AxisPrior] = []
        sources: dict[str, str] = {}

        for name in names:
            if name in canonical_overrides:
                priors.append(_to_delta_prior(canonical_overrides[name], refs[name]))
                sources[name] = "override"
                continue

            discovered = None
            if (
                policy != "explicit"
                and pint_model is not None
                and hasattr(pint_model, name)
            ):
                param = getattr(pint_model, name)
                discovered = _axis_prior_from_object(getattr(param, "prior", None))
            if discovered is not None:
                priors.append(_to_delta_prior(discovered, refs[name]))
                sources[name] = "pint"
                continue

            if name in canonical_defaults:
                priors.append(_to_delta_prior(canonical_defaults[name], refs[name]))
                sources[name] = "named_default"
                continue

            if policy == "explicit":
                raise ValueError(
                    f"Missing explicit prior override or named default for parameter '{name}'"
                )
            if policy == "strict":
                raise ValueError(
                    f"No proper PINT prior, override, or named default for strict policy parameter '{name}'"
                )

            # Pulsar-bound model resolution replaces this sentinel with the final wide
            # uniform cheat-prior box before any bijector is built.
            priors.append(AxisPrior(family="cheat_wls"))
            sources[name] = "cheat_wls"

        return cls(names=tuple(names), priors=tuple(priors), sources=sources)

    def to_bijector(
        self, precision_critical_fitpars: frozenset[str] | set[str] = frozenset()
    ) -> PriorBijector:
        """Convert resolved priors to the per-axis bijector used by ParameterSpace."""
        critical = {
            resolve_parameter_alias(name) for name in precision_critical_fitpars
        }
        for name, prior in zip(self.names, self.priors, strict=True):
            if prior.family == "cheat_wls":
                raise ValueError(
                    "cheat_wls fallback priors require pulsar-bound resolution before "
                    "building a PriorBijector"
                )
            if name in critical and prior.family in ABSOLUTE_FORMING_FAMILIES:
                raise ValueError(
                    f"Prior family '{prior.family}' forms absolute values for precision-critical "
                    f"parameter '{name}'; use a delta-safe override such as uniform or normal"
                )
        return PriorBijector(names=self.names, priors=self.priors)

    def source_labels(self) -> dict[str, str]:
        return dict(self.sources)
