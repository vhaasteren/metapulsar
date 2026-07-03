"""Nonlinear timing model configuration and pulsar binding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .bijectors import AxisPrior, WhiteningLinear
from .partition import PartitionResult, resolve_partition
from .priors import PriorBlock, PriorPolicy, set_prior, validate_prior_policy
from .space import ParameterSpace, default_coord_for_transform
from .units import lookup_pint_param, native_physical_bounds, to_native
from .whitening import diagonal_white, fixed_hyperparameters, schur_delta_wls
from .backends import normalize_engines

_TRANSFORMS = {"none", "standardized", "whitening"}
_DESIGN_MATRIX_METHODS = {"analytic", "autodiff"}


def _normalize_design_matrix_method(method: str) -> str:
    normalized = str(method or "analytic").lower()
    if normalized not in _DESIGN_MATRIX_METHODS:
        raise ValueError(
            "design_matrix_method must be 'analytic' or 'autodiff'; " f"got {method!r}"
        )
    return normalized


def _timing_design_matrix(pulsar, backend, *, method: str) -> np.ndarray:
    if method == "autodiff":
        matrix_fn = getattr(backend, "linearized_design_matrix", None)
        if matrix_fn is None:
            raise ValueError(
                "design_matrix_method='autodiff' requires a backend that exposes "
                "linearized_design_matrix()."
            )
        return np.asarray(matrix_fn(), dtype=float)
    return np.asarray(pulsar.Mmat, dtype=float)


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class BoundTiming:
    """Pulsar-bound nonlinear timing context resolved from model config."""

    backend: object
    partition: PartitionResult
    prior_block: PriorBlock
    space: ParameterSpace
    coord: str
    site_name: str
    delay_keys: tuple[str, ...]
    design_matrix: np.ndarray


class NonLinearTimingModel:
    """Nonlinear timing model configuration and likelihood-frontend glue.

    Binds to a ``PulsarInterface`` at call time. Does not own noise models or samplers;
    the user assembles Enterprise/Discovery likelihood frontends and runs their chosen
    sampler.
    """

    def __init__(
        self,
        *,
        engines: str | Mapping[str, str] = "jug",
        design_matrix_method: str = "analytic",
        transform: str = "whitening",
        analytically_marginalize: str | Sequence[str] | None = "default",
        prior_policy: PriorPolicy = "fallback",
        cheat_prior_scale: float = 50.0,
        whitening_config: Mapping[str, Any] | None = None,
        name: str = "nonlinear_timing_model",
    ):
        if transform not in _TRANSFORMS:
            raise ValueError(f"Unsupported transform: {transform}")
        if not (float(cheat_prior_scale) > 0.0):
            raise ValueError("cheat_prior_scale must be positive")
        self.engines = normalize_engines(engines)
        self.design_matrix_method = _normalize_design_matrix_method(
            design_matrix_method
        )
        self.transform = transform
        self.analytically_marginalize = analytically_marginalize
        self.prior_policy = validate_prior_policy(prior_policy)
        self.cheat_prior_scale = float(cheat_prior_scale)
        self.whitening_config = (
            None if whitening_config is None else dict(whitening_config)
        )
        self.name = name
        self._prior_overrides: dict[str, AxisPrior] = {}
        self._resolved_cache: dict[str, BoundTiming] = {}

    def set_prior(self, name: str, kind: str, **bounds) -> None:
        """Set or override one sampled-parameter prior."""
        if kind == "normal":
            prior = AxisPrior(
                family="normal",
                mean=float(bounds["mean"]),
                std=float(bounds["std"]),
            )
        elif kind == "uniform":
            prior = AxisPrior(
                family="uniform",
                lower=float(bounds["lower"]),
                upper=float(bounds["upper"]),
            )
        elif kind == "log_uniform":
            prior = AxisPrior(
                family="log_uniform",
                lower=float(bounds["lower"]),
                upper=float(bounds["upper"]),
            )
        elif kind == "truncated_normal":
            prior = AxisPrior(
                family="truncated_normal",
                lower=float(bounds["lower"]),
                upper=float(bounds["upper"]),
                mean=float(bounds["mean"]),
                std=float(bounds["std"]),
            )
        else:
            raise ValueError(f"Unsupported prior kind: {kind}")
        self._prior_overrides = set_prior(self._prior_overrides, name, prior)
        self._resolved_cache.clear()

    def with_engines(self, engines) -> "NonLinearTimingModel":
        """Return a new model config with a different engine selection."""
        other = NonLinearTimingModel(
            engines=engines,
            design_matrix_method=self.design_matrix_method,
            transform=self.transform,
            analytically_marginalize=self.analytically_marginalize,
            prior_policy=self.prior_policy,
            cheat_prior_scale=self.cheat_prior_scale,
            whitening_config=self.whitening_config,
            name=self.name,
        )
        other._prior_overrides = dict(self._prior_overrides)
        return other

    def _config_fingerprint(self) -> str:
        payload = {
            "engines": sorted(self.engines.items()),
            "design_matrix_method": self.design_matrix_method,
            "transform": self.transform,
            "analytically_marginalize": self.analytically_marginalize,
            "prior_policy": self.prior_policy,
            "cheat_prior_scale": self.cheat_prior_scale,
            "whitening_config": self.whitening_config,
            "name": self.name,
            "prior_overrides": {
                key: vars(prior) for key, prior in sorted(self._prior_overrides.items())
            },
        }
        return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()

    def _backend_for_pulsar(self, pulsar):
        return pulsar.timing_backend(
            self.engines, design_matrix_method=self.design_matrix_method
        )

    def _partition(self, pulsar) -> PartitionResult:
        return resolve_partition(
            pulsar, analytically_marginalize=self.analytically_marginalize
        )

    def _effective_overrides(self, partition: PartitionResult) -> dict[str, AxisPrior]:
        sampled_set = set(partition.sampled)
        fitpar_set = set(partition.fitpars)
        unknown = sorted(
            name for name in self._prior_overrides if name not in fitpar_set
        )
        if unknown:
            raise ValueError(
                "Prior overrides target unknown fit parameters for this pulsar: "
                f"{unknown}"
            )
        invalid = sorted(
            name
            for name in self._prior_overrides
            if name in fitpar_set and name not in sampled_set
        )
        if invalid:
            raise ValueError(
                "Prior overrides target non-sampled parameters for this pulsar: "
                f"{invalid}"
            )
        return {
            name: prior
            for name, prior in self._prior_overrides.items()
            if name in sampled_set
        }

    def _build_prior_block(
        self,
        *,
        pulsar,
        backend,
        partition: PartitionResult,
        design_matrix: np.ndarray,
    ) -> PriorBlock:
        ref_exact = backend.reference_theta_exact()
        sampled_refs = {
            name: ref_exact[name] for name in partition.sampled if name in ref_exact
        }
        missing = [name for name in partition.sampled if name not in sampled_refs]
        if missing:
            raise ValueError(
                f"reference_theta_exact missing sampled parameters: {missing}"
            )
        block = PriorBlock.from_fitpars(
            partition.sampled,
            policy=self.prior_policy,
            overrides=self._effective_overrides(partition),
            theta_ref=sampled_refs,
            pint_model=pulsar.pint_model(),
        )
        theta_ref_native = {name: float(value) for name, value in sampled_refs.items()}
        return self._fill_wls_cheat_priors(
            pulsar=pulsar,
            partition=partition,
            block=block,
            theta_ref_native=theta_ref_native,
            design_matrix=design_matrix,
        )

    def _parfile_cheat_stds(self, pulsar, names) -> dict[str, float]:
        """Per-parameter par-file frequentist uncertainties in native units."""
        out: dict[str, float] = {}
        pint_model = pulsar.pint_model()
        if pint_model is None:
            return out
        for name in names:
            param = lookup_pint_param(pint_model, name)
            if param is None:
                continue
            unc = getattr(param, "uncertainty_value", None)
            if unc is None:
                continue
            unc = float(unc)
            if not np.isfinite(unc) or unc <= 0.0:
                continue
            # ``uncertainty_value`` is a magnitude in the parameter's display
            # unit; the same linear scaling maps it to native delta units.
            out[name] = float(np.abs(to_native(name, unc, pint_model=pint_model)))
        return out

    def _fill_wls_cheat_priors(
        self,
        *,
        pulsar,
        partition: PartitionResult,
        block: PriorBlock,
        theta_ref_native: dict[str, float] | None = None,
        design_matrix: np.ndarray | None = None,
    ) -> PriorBlock:
        if not partition.sampled or "cheat_wls" not in block.sources.values():
            return block
        theta_ref_native = theta_ref_native or {}
        variance = np.asarray(pulsar.toaerrs, dtype=float) ** 2
        wls = schur_delta_wls(
            pulsar=pulsar,
            partition=partition,
            variance=variance,
            design_matrix=design_matrix,
        )
        wls_stds = np.sqrt(np.diag(wls.covariance))
        parfile_stds = self._parfile_cheat_stds(pulsar, block.names)
        scale = self.cheat_prior_scale
        priors = []
        for idx, (name, prior) in enumerate(
            zip(block.names, block.priors, strict=True)
        ):
            if block.sources.get(name) != "cheat_wls":
                priors.append(prior)
                continue
            # Prefer the par-file frequentist uncertainty; fall back to the
            # recomputed WLS marginal sigma when it is unavailable.
            sigma = parfile_stds.get(name)
            if sigma is None or not np.isfinite(sigma) or sigma <= 0.0:
                sigma = float(wls_stds[idx])
            half = scale * sigma
            # Flat box in delta units centered on the par-file value (delta=0),
            # matching the external reference-stack cheat-prior convention, then clipped
            # to physical bounds.
            lower, upper = -half, half
            ref = theta_ref_native.get(name)
            bound_lo, bound_hi = native_physical_bounds(name)
            if ref is not None:
                if bound_lo is not None:
                    lower = max(lower, bound_lo - ref)
                if bound_hi is not None:
                    upper = min(upper, bound_hi - ref)
            if not (upper > lower):
                lower, upper = -half, half
            priors.append(
                AxisPrior(family="uniform", lower=float(lower), upper=float(upper))
            )
        return PriorBlock(
            names=block.names, priors=tuple(priors), sources=block.sources
        )

    def _linear_transform(
        self,
        *,
        pulsar,
        partition,
        prior_bijector,
        design_matrix: np.ndarray,
    ) -> WhiteningLinear:
        ndim = len(partition.sampled)
        if self.transform == "none":
            return WhiteningLinear.identity(ndim)

        cfg = self.whitening_config
        if cfg is None:
            return diagonal_white(
                pulsar=pulsar,
                partition=partition,
                prior_bijector=prior_bijector,
                mode=self.transform,
                design_matrix=design_matrix,
            )

        cfg = dict(cfg)
        builder = cfg.pop("name", "diagonal_white")
        if builder == "diagonal_white":
            return diagonal_white(
                pulsar=pulsar,
                partition=partition,
                prior_bijector=prior_bijector,
                mode=self.transform,
                design_matrix=design_matrix,
                **cfg,
            )
        if builder == "fixed_hyperparameters":
            return fixed_hyperparameters(
                pulsar=pulsar,
                partition=partition,
                prior_bijector=prior_bijector,
                mode=self.transform,
                design_matrix=design_matrix,
                **cfg,
            )
        raise ValueError(f"Unsupported whitening builder: {builder}")

    def _pulsar_state_fingerprint(self, pulsar, backend) -> str:
        token = None
        if hasattr(pulsar, "cache_token"):
            token = pulsar.cache_token()
        if token is not None:
            return f"token:{token}"
        design = np.asarray(pulsar.Mmat, dtype=float)
        refs = backend.reference_theta_exact()
        payload = {
            "fitpars": tuple(pulsar.fitpars),
            "n_toa": int(len(pulsar.toas)),
            "design_shape": tuple(design.shape),
            "design_checksum": float(np.sum(np.abs(design))),
            "residual_shape": tuple(np.asarray(pulsar.residuals).shape),
            "refs": {name: refs.get(name) for name in pulsar.fitpars},
        }
        return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()

    def _resolved_cache_key(self, pulsar, backend) -> str:
        pulsar_id = f"{id(pulsar)}:{getattr(pulsar, 'name', 'unknown')}"
        return "|".join(
            [
                pulsar_id,
                self._config_fingerprint(),
                self._pulsar_state_fingerprint(pulsar, backend),
            ]
        )

    def _resolve(self, pulsar) -> BoundTiming:
        backend = self._backend_for_pulsar(pulsar)
        key = self._resolved_cache_key(pulsar, backend)
        if key in self._resolved_cache:
            return self._resolved_cache[key]

        partition = self._partition(pulsar)
        design_matrix = _timing_design_matrix(
            pulsar,
            backend,
            method=self.design_matrix_method,
        )
        prior_block = self._build_prior_block(
            pulsar=pulsar,
            backend=backend,
            partition=partition,
            design_matrix=design_matrix,
        )
        prior_bijector = prior_block.to_bijector(
            precision_critical_fitpars=getattr(
                backend, "precision_critical_fitpars", lambda: frozenset()
            )()
        )
        linear = self._linear_transform(
            pulsar=pulsar,
            partition=partition,
            prior_bijector=prior_bijector,
            design_matrix=design_matrix,
        )
        ref_exact = backend.reference_theta_exact()
        sampled_ref_exact = {name: ref_exact[name] for name in partition.sampled}
        space = ParameterSpace.build(
            sampled_ref_exact,
            prior_bijector=prior_bijector,
            transform=self.transform,
            linear_transform=linear,
            pint_model=pulsar.pint_model(),
        )
        coord = default_coord_for_transform(self.transform)
        resolved = BoundTiming(
            backend=backend,
            partition=partition,
            prior_block=prior_block,
            space=space,
            coord=coord,
            site_name=f"{pulsar.name}_{self.name}_{coord}",
            delay_keys=tuple(
                f"{pulsar.name}_{self.name}_{name}" for name in partition.sampled
            ),
            design_matrix=design_matrix,
        )
        self._resolved_cache[key] = resolved
        return resolved

    def sampled(self, pulsar) -> tuple[str, ...]:
        return self._resolve(pulsar).partition.sampled

    def analytically_marginalized(self, pulsar) -> tuple[str, ...]:
        return self._resolve(pulsar).partition.analytically_marginalized

    def priors(self, pulsar) -> PriorBlock:
        return self._resolve(pulsar).prior_block

    def space(self, pulsar) -> ParameterSpace:
        return self._resolve(pulsar).space

    def discovery_signals(self, pulsar) -> list:
        from .frontends.discovery import discovery_signals

        resolved = self._resolve(pulsar)
        return discovery_signals(
            pulsar=pulsar,
            space=resolved.space,
            backend=resolved.backend,
            partition=resolved.partition,
            name=self.name,
            design_matrix=resolved.design_matrix,
        )

    def enterprise_signal(self):
        from .frontends.enterprise import enterprise_signal

        return enterprise_signal(
            space_fn=self.space,
            engines=self.engines,
            design_matrix_method=self.design_matrix_method,
            partition_spec=self._partition,
            name=self.name,
            transform=self.transform,
        )

    def _coord_site_name(self, pulsar, coord: str | None = None) -> str:
        coord = default_coord_for_transform(self.transform) if coord is None else coord
        if coord not in {"delta", "z", "x"}:
            raise ValueError(f"Unsupported coord: {coord}")
        if coord == default_coord_for_transform(self.transform):
            return self._resolve(pulsar).site_name
        return f"{pulsar.name}_{self.name}_{coord}"

    def _delay_keys(self, pulsar) -> tuple[str, ...]:
        return self._resolve(pulsar).delay_keys

    def timing_param_keys(self, pulsar) -> tuple[str, ...]:
        resolved = self._resolve(pulsar)
        if not resolved.partition.sampled:
            return tuple()
        keys = [resolved.site_name, *resolved.delay_keys]
        return tuple(keys)

    def non_timing_params(self, pulsar, params: Sequence[str]) -> tuple[str, ...]:
        owned = set(self.timing_param_keys(pulsar))
        return tuple(name for name in params if name not in owned)

    def contribute_timing(
        self,
        pulsar,
        params: Mapping[str, Any],
        *,
        coord: str | None = None,
    ) -> Mapping[str, Any]:
        resolved = self._resolve(pulsar)
        sampled = resolved.partition.sampled
        if not sampled:
            return params

        coord = resolved.coord if coord is None else coord
        if coord not in {"delta", "z", "x"}:
            raise ValueError(f"Unsupported coord: {coord}")

        import jax.numpy as jnp
        import numpyro
        from numpyro import distributions as dist
        from numpyro.distributions import constraints

        space = resolved.space
        site_name = self._coord_site_name(pulsar, coord=coord)
        q = numpyro.sample(
            site_name,
            dist.ImproperUniform(constraints.real, (), (len(sampled),)),
        )
        numpyro.factor(
            f"{site_name}_logprior", space.logprior_coord(q, jnp, coord=coord)
        )
        delta = space.delta_from_coord(q, jnp, coord=coord)

        out = dict(params)
        for i, name in enumerate(sampled):
            out[f"{pulsar.name}_{self.name}_{name}"] = delta[i]
        return out

    def _delta_from_params(
        self,
        pulsar,
        params: Mapping[str, Any],
        *,
        coord: str | None = None,
        coord_explicit: bool = False,
    ) -> np.ndarray:
        resolved = self._resolve(pulsar)
        sampled = resolved.partition.sampled
        if not sampled:
            return np.zeros((0,), dtype=float)

        if coord is not None and coord not in {"delta", "z", "x"}:
            raise ValueError("coord must be one of {'delta', 'z', 'x'}")
        coord = resolved.coord if coord is None else coord

        delay_keys = resolved.delay_keys
        site_name = self._coord_site_name(pulsar, coord=coord)
        space = resolved.space

        if site_name in params:
            q = np.asarray(params[site_name], dtype=float)
            return np.asarray(space.delta_from_coord(q, np, coord=coord), dtype=float)

        if coord == "delta":
            if all(key in params for key in delay_keys):
                return np.asarray([params[key] for key in delay_keys], dtype=float)
            raise ValueError(
                "record_physical(coord='delta') requires injected delta keys "
                "or a delta site"
            )

        # Enterprise standardized scalars reuse delay-key names for sampler x axes.
        if (
            coord == "x"
            and self.transform == "standardized"
            and all(key in params for key in delay_keys)
        ):
            values = np.asarray([params[key] for key in delay_keys], dtype=float)
            # Implicit default coord (record_physical called without coord) corresponds
            # to contribute_timing outputs where delay keys already carry delta values.
            if not coord_explicit:
                return values
            # Explicit coord="x" keeps Enterprise standardized-scalar semantics.
            return np.asarray(
                space.delta_from_coord(values, np, coord="x"), dtype=float
            )

        raise ValueError(
            f"record_physical(coord={coord!r}) could not find matching timing coordinates"
        )

    def record_physical(
        self,
        pulsar,
        params: Mapping[str, Any],
        *,
        scope: str = "timing",
        coord: str | None = None,
    ) -> None:
        coord_was_explicit = coord is not None
        if coord is not None and coord not in {"delta", "z", "x"}:
            raise ValueError("coord must be one of {'delta', 'z', 'x'}")
        if coord is None:
            coord = default_coord_for_transform(self.transform)

        if scope == "all":
            raise NotImplementedError("scope='all' is deferred")
        if scope != "timing":
            raise ValueError("scope must be one of {'timing', 'all'}")

        resolved = self._resolve(pulsar)
        sampled = resolved.partition.sampled
        if not sampled:
            return

        import numpyro

        delta = self._delta_from_params(
            pulsar,
            params,
            coord=coord,
            coord_explicit=coord_was_explicit,
        )
        theta = resolved.space.theta_from_delta(delta)
        for i, name in enumerate(sampled):
            numpyro.deterministic(
                f"{pulsar.name}_{self.name}_{name}_theta",
                theta[i],
            )
