"""Config-only nonlinear timing component."""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .bijectors import AxisPrior, WhiteningLinear
from .partition import PartitionResult, resolve_partition
from .priors import PriorBlock, PriorPolicy, set_prior, validate_prior_policy
from .space import ParameterSpace
from .whitening import diagonal_white, fixed_hyperparameters, schur_delta_wls

_BACKENDS = {"jug", "pint", "tempo2"}
_TRANSFORMS = {"none", "standardized", "whitening"}


def _default_coord_for_transform(transform: str) -> str:
    if transform == "none":
        return "delta"
    if transform == "standardized":
        return "x"
    if transform == "whitening":
        return "x"
    raise ValueError(f"Unsupported transform: {transform}")


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class NonLinearTimingModel:
    """Stateless timing config that binds to a host at call time."""

    def __init__(
        self,
        *,
        backend: str = "jug",
        jug_compatibility: str = "auto",
        transform: str = "whitening",
        marginalize: str | Sequence[str] | None = "default",
        prior_policy: PriorPolicy = "fallback",
        whitening_config: Mapping[str, Any] | None = None,
        name: str = "nonlinear_timing_model",
    ):
        if backend not in _BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}")
        if transform not in _TRANSFORMS:
            raise ValueError(f"Unsupported transform: {transform}")
        self.backend = backend
        self.jug_compatibility = jug_compatibility
        self.transform = transform
        self.marginalize = marginalize
        self.prior_policy = validate_prior_policy(prior_policy)
        self.whitening_config = (
            None if whitening_config is None else dict(whitening_config)
        )
        self.name = name
        self._prior_overrides: dict[str, AxisPrior] = {}
        self._space_cache: dict[str, ParameterSpace] = {}

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
        self._space_cache.clear()

    def with_backend(self, backend: str) -> "NonLinearTimingModel":
        """Return a new component config with a different backend family."""
        other = NonLinearTimingModel(
            backend=backend,
            jug_compatibility=self.jug_compatibility,
            transform=self.transform,
            marginalize=self.marginalize,
            prior_policy=self.prior_policy,
            whitening_config=self.whitening_config,
            name=self.name,
        )
        other._prior_overrides = dict(self._prior_overrides)
        return other

    def _config_fingerprint(self) -> str:
        payload = {
            "backend": self.backend,
            "jug_compatibility": (
                self.jug_compatibility if self.backend == "jug" else None
            ),
            "transform": self.transform,
            "marginalize": self.marginalize,
            "prior_policy": self.prior_policy,
            "whitening_config": self.whitening_config,
            "name": self.name,
            "prior_overrides": {
                key: vars(prior) for key, prior in sorted(self._prior_overrides.items())
            },
        }
        return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()

    def _backend_for_host(self, host):
        if self.backend != "jug":
            return host.timing_backend(self.backend)
        signature = inspect.signature(host.timing_backend)
        params = signature.parameters
        accepts_jug_compat = "jug_compatibility" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if accepts_jug_compat:
            return host.timing_backend(
                self.backend,
                jug_compatibility=self.jug_compatibility,
            )
        return host.timing_backend(self.backend)

    def _partition(self, host) -> PartitionResult:
        return resolve_partition(host, marginalize=self.marginalize)

    def _effective_overrides(self, partition: PartitionResult) -> dict[str, AxisPrior]:
        sampled_set = set(partition.sampled)
        fitpar_set = set(partition.fitpars)
        unknown = sorted(
            name for name in self._prior_overrides if name not in fitpar_set
        )
        if unknown:
            raise ValueError(
                "Prior overrides target unknown fit parameters for this host: "
                f"{unknown}"
            )
        invalid = sorted(
            name
            for name in self._prior_overrides
            if name in fitpar_set and name not in sampled_set
        )
        if invalid:
            raise ValueError(
                "Prior overrides target non-sampled parameters for this host: "
                f"{invalid}"
            )
        return {
            name: prior
            for name, prior in self._prior_overrides.items()
            if name in sampled_set
        }

    def _prior_block(self, host, backend=None) -> PriorBlock:
        partition = self._partition(host)
        backend = self._backend_for_host(host) if backend is None else backend
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
            pint_model=host.pint_model(),
        )
        return self._fill_wls_cheat_priors(host=host, partition=partition, block=block)

    def _fill_wls_cheat_priors(
        self,
        *,
        host,
        partition: PartitionResult,
        block: PriorBlock,
    ) -> PriorBlock:
        if not partition.sampled or "cheat_wls" not in block.sources.values():
            return block
        variance = np.asarray(host.toaerrs, dtype=float) ** 2
        wls = schur_delta_wls(host=host, partition=partition, variance=variance)
        stds = np.sqrt(np.diag(wls.covariance))
        priors = []
        for idx, (name, prior) in enumerate(
            zip(block.names, block.priors, strict=True)
        ):
            if block.sources.get(name) == "cheat_wls":
                priors.append(
                    AxisPrior(family="normal", mean=0.0, std=float(stds[idx]))
                )
            else:
                priors.append(prior)
        return PriorBlock(
            names=block.names, priors=tuple(priors), sources=block.sources
        )

    def _linear_transform(self, *, host, partition, prior_bijector) -> WhiteningLinear:
        ndim = len(partition.sampled)
        if self.transform == "none":
            return WhiteningLinear.identity(ndim)

        cfg = self.whitening_config
        if cfg is None:
            return diagonal_white(
                host=host,
                partition=partition,
                prior_bijector=prior_bijector,
                mode=self.transform,
            ).to_whitening_linear()

        cfg = dict(cfg)
        builder = cfg.pop("name", "diagonal_white")
        if builder == "diagonal_white":
            return diagonal_white(
                host=host,
                partition=partition,
                prior_bijector=prior_bijector,
                mode=self.transform,
                **cfg,
            ).to_whitening_linear()
        if builder == "fixed_hyperparameters":
            return fixed_hyperparameters(
                host=host,
                partition=partition,
                prior_bijector=prior_bijector,
                mode=self.transform,
                **cfg,
            ).to_whitening_linear()
        raise ValueError(f"Unsupported whitening builder: {builder}")

    def _host_state_fingerprint(self, host, backend) -> str:
        token = None
        if hasattr(host, "cache_token"):
            token = host.cache_token()
        if token is not None:
            return f"token:{token}"
        design = np.asarray(host.Mmat, dtype=float)
        refs = backend.reference_theta_exact()
        payload = {
            "fitpars": tuple(host.fitpars),
            "n_toa": int(len(host.toas)),
            "design_shape": tuple(design.shape),
            "design_checksum": float(np.sum(np.abs(design))),
            "residual_shape": tuple(np.asarray(host.residuals).shape),
            "refs": {name: refs.get(name) for name in host.fitpars},
        }
        return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()

    def _space_cache_key(self, host, backend) -> str:
        host_id = f"{id(host)}:{getattr(host, 'name', 'unknown')}"
        return "|".join(
            [
                host_id,
                self._config_fingerprint(),
                self._host_state_fingerprint(host, backend),
            ]
        )

    def sampled(self, host) -> tuple[str, ...]:
        return self._partition(host).sampled

    def marginalized(self, host) -> tuple[str, ...]:
        return self._partition(host).marginalized

    def priors(self, host) -> PriorBlock:
        return self._prior_block(host)

    def space(self, host) -> ParameterSpace:
        backend = self._backend_for_host(host)
        key = self._space_cache_key(host, backend)
        if key in self._space_cache:
            return self._space_cache[key]

        partition = self._partition(host)
        prior_block = self._prior_block(host, backend=backend)
        prior_bijector = prior_block.to_bijector(
            precision_critical_fitpars=getattr(
                backend, "precision_critical_fitpars", lambda: frozenset()
            )()
        )
        linear = self._linear_transform(
            host=host,
            partition=partition,
            prior_bijector=prior_bijector,
        )
        ref_exact = backend.reference_theta_exact()
        sampled_ref_exact = {name: ref_exact[name] for name in partition.sampled}
        space = ParameterSpace.build(
            sampled_ref_exact,
            prior_bijector=prior_bijector,
            transform=self.transform,
            linear_transform=linear,
        )
        self._space_cache[key] = space
        return space

    def discovery_signals(self, host) -> list:
        from .frontends.discovery import discovery_signals

        partition = self._partition(host)
        backend = self._backend_for_host(host)
        return discovery_signals(
            host=host,
            space=self.space(host),
            backend=backend,
            partition=partition,
            name=self.name,
        )

    def enterprise_signal(self):
        from .frontends.enterprise import enterprise_signal

        backend_kwargs = (
            {"jug_compatibility": self.jug_compatibility}
            if self.backend == "jug"
            else {}
        )
        return enterprise_signal(
            space_fn=self.space,
            backend_name=self.backend,
            backend_kwargs=backend_kwargs,
            partition_spec=self._partition,
            name=self.name,
            transform=self.transform,
        )

    def _coord_site_name(self, host, coord: str | None = None) -> str:
        coord = _default_coord_for_transform(self.transform) if coord is None else coord
        if coord not in {"delta", "z", "x"}:
            raise ValueError(f"Unsupported coord: {coord}")
        return f"{host.name}_{self.name}_{coord}"

    def _delay_keys(self, host) -> tuple[str, ...]:
        return tuple(f"{host.name}_{self.name}_{name}" for name in self.sampled(host))

    def timing_param_keys(self, host) -> tuple[str, ...]:
        if not self.sampled(host):
            return tuple()
        keys = [self._coord_site_name(host), *self._delay_keys(host)]
        return tuple(keys)

    def non_timing_params(self, host, params: Sequence[str]) -> tuple[str, ...]:
        owned = set(self.timing_param_keys(host))
        return tuple(name for name in params if name not in owned)

    def contribute_timing(
        self,
        host,
        params: Mapping[str, Any],
        *,
        coord: str | None = None,
    ) -> Mapping[str, Any]:
        sampled = self.sampled(host)
        if not sampled:
            return params

        coord = _default_coord_for_transform(self.transform) if coord is None else coord
        if coord not in {"delta", "z", "x"}:
            raise ValueError(f"Unsupported coord: {coord}")

        import jax.numpy as jnp
        import numpyro
        from numpyro import distributions as dist
        from numpyro.distributions import constraints

        space = self.space(host)
        site_name = self._coord_site_name(host, coord=coord)
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
            out[f"{host.name}_{self.name}_{name}"] = delta[i]
        return out

    def _delta_from_params(
        self,
        host,
        params: Mapping[str, Any],
        *,
        coord: str | None = None,
    ) -> np.ndarray:
        sampled = self.sampled(host)
        if not sampled:
            return np.zeros((0,), dtype=float)

        if coord is not None and coord not in {"delta", "z", "x"}:
            raise ValueError("coord must be one of {'delta', 'z', 'x'}")

        if coord is None:
            # Backward-compatible inference.
            delay_keys = self._delay_keys(host)
            if all(key in params for key in delay_keys):
                return np.asarray([params[key] for key in delay_keys], dtype=float)

            site_name = self._coord_site_name(host)
            if site_name not in params:
                raise ValueError(
                    "Timing parameters missing from params mapping; call contribute_timing "
                    "before record_physical or include injected delta keys"
                )
            resolved = _default_coord_for_transform(self.transform)
            space = self.space(host)
            q = np.asarray(params[site_name], dtype=float)
            return np.asarray(
                space.delta_from_coord(q, np, coord=resolved), dtype=float
            )

        delay_keys = self._delay_keys(host)
        site_name = self._coord_site_name(host, coord=coord)
        space = self.space(host)

        if coord == "delta":
            if all(key in params for key in delay_keys):
                return np.asarray([params[key] for key in delay_keys], dtype=float)
            if site_name in params:
                q = np.asarray(params[site_name], dtype=float)
                return np.asarray(
                    space.delta_from_coord(q, np, coord="delta"), dtype=float
                )
            raise ValueError(
                "record_physical(coord='delta') requires injected delta keys "
                "or a delta site"
            )

        if site_name in params:
            q = np.asarray(params[site_name], dtype=float)
            return np.asarray(space.delta_from_coord(q, np, coord=coord), dtype=float)

        # Enterprise standardized scalars reuse delay-key names for x axes.
        if (
            coord == "x"
            and self.transform == "standardized"
            and all(key in params for key in delay_keys)
        ):
            q = np.asarray([params[key] for key in delay_keys], dtype=float)
            return np.asarray(space.delta_from_coord(q, np, coord="x"), dtype=float)

        raise ValueError(
            f"record_physical(coord={coord!r}) could not find matching timing coordinates"
        )

    def record_physical(
        self,
        host,
        params: Mapping[str, Any],
        *,
        scope: str = "timing",
        coord: str | None = None,
    ) -> None:
        if coord is not None and coord not in {"delta", "z", "x"}:
            raise ValueError("coord must be one of {'delta', 'z', 'x'}")
        if coord is None:
            coord = _default_coord_for_transform(self.transform)

        if scope == "all":
            raise NotImplementedError("scope='all' is deferred")
        if scope != "timing":
            raise ValueError("scope must be one of {'timing', 'all'}")

        sampled = self.sampled(host)
        if not sampled:
            return

        import numpyro

        space = self.space(host)
        delta = self._delta_from_params(host, params, coord=coord)
        theta = space.theta_from_delta(delta)
        for i, name in enumerate(sampled):
            numpyro.deterministic(
                f"{host.name}_{self.name}_{name}_theta",
                theta[i],
            )
