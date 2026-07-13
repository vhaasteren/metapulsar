"""Nonlinear timing model configuration and pulsar binding."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from . import priors as prior_specs
from .bijectors import AxisPrior, WhiteningLinear
from .partition import (
    PartitionResult,
    fitpar_suffixes,
    match_fitpars,
    resolve_partition,
)
from .priors import (
    PriorBlock,
    PriorBuildContext,
    PriorFrame,
    PriorOverrideSpec,
    PriorPolicy,
    materialize_prior_override,
    store_prior_override,
    validate_prior_policy,
)
from .space import ParameterSpace, default_coord_for_transform
from .units import lookup_pint_param, native_physical_bounds, to_native
from .whitening import diagonal_white, fixed_hyperparameters, schur_delta_wls
from .backends import normalize_engines

_TRANSFORMS = {"none", "standardized", "whitening"}
_DESIGN_MATRIX_METHODS = {"analytic", "autodiff"}
_PRIOR_OVERRIDE_POLICIES = {"warn", "strict"}


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


@dataclass(frozen=True, eq=False)
class TimingBinding:
    """Pulsar-bound nonlinear timing context resolved from model config.

    Produced by :meth:`NonLinearTimingModel.bind`; owns every pulsar-bound
    query (sampled partition, priors, parameter space, frontend signals,
    artifact snapshots). The model itself stays pure configuration.
    """

    model: "NonLinearTimingModel"
    pulsar: Any
    backend: Any
    partition: PartitionResult
    prior_block: PriorBlock
    space: ParameterSpace
    coord: str
    site_name: str
    delay_keys: tuple[str, ...]
    design_matrix: np.ndarray

    @property
    def prefix(self) -> str:
        """Site/deterministic name prefix: ``{pulsar}_{model.name}``."""
        return f"{self.pulsar.name}_{self.model.name}"

    @property
    def sampled(self) -> tuple[str, ...]:
        return self.partition.sampled

    @property
    def marginalized(self) -> tuple[str, ...]:
        return self.partition.analytically_marginalized

    @property
    def priors(self) -> PriorBlock:
        return self.prior_block

    def timing_param_keys(self) -> tuple[str, ...]:
        if not self.partition.sampled:
            return tuple()
        return (self.site_name, *self.delay_keys)

    def non_timing_params(self, params: Sequence[str]) -> tuple[str, ...]:
        owned = set(self.timing_param_keys())
        return tuple(name for name in params if name not in owned)

    def coord_site_name(self, coord: str | None = None) -> str:
        default = default_coord_for_transform(self.model.transform)
        coord = default if coord is None else coord
        if coord not in {"delta", "z", "x"}:
            raise ValueError(f"Unsupported coord: {coord}")
        if coord == default:
            return self.site_name
        return f"{self.prefix}_{coord}"

    def fingerprint(self) -> str:
        """Identity of decoder + frontend config + pulsar/model state."""
        payload = {
            "config": self.model._config_fingerprint(),
            "pulsar_state": self.model._pulsar_state_fingerprint(
                self.pulsar, self.backend
            ),
            "space": self.space.fingerprint(),
            "transform": self.model.transform,
            "coord": default_coord_for_transform(self.model.transform),
            "sampled": list(self.sampled),
            "engines": sorted(self.model.engines.items()),
            "design_matrix_method": self.model.design_matrix_method,
        }
        return (
            "sha256:"
            + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
        )

    def discovery_signals(self) -> list:
        from .frontends.discovery import discovery_signals

        return discovery_signals(
            pulsar=self.pulsar,
            space=self.space,
            backend=self.backend,
            partition=self.partition,
            name=self.model.name,
            design_matrix=self.design_matrix,
        )

    def delta_from_params(
        self,
        params: Mapping[str, Any],
        *,
        coord: str | None = None,
        coord_explicit: bool = False,
    ) -> np.ndarray:
        """Extract sampled delta-theta values from a sampler parameter mapping.

        Accepts either the joint coordinate site (``coord_site_name``), the
        per-parameter delay keys, or Enterprise standardized scalar columns.
        """
        sampled = self.partition.sampled
        if not sampled:
            return np.zeros((0,), dtype=float)

        if coord is not None and coord not in {"delta", "z", "x"}:
            raise ValueError("coord must be one of {'delta', 'z', 'x'}")
        coord = self.coord if coord is None else coord

        site_name = self.coord_site_name(coord)

        if site_name in params:
            q = np.asarray(params[site_name], dtype=float)
            return np.asarray(
                self.space.delta_from_coord(q, np, coord=coord), dtype=float
            )

        if coord == "delta":
            if all(key in params for key in self.delay_keys):
                return np.asarray([params[key] for key in self.delay_keys], dtype=float)
            raise ValueError(
                "delta_from_params(coord='delta') requires injected delta keys "
                "or a delta site"
            )

        # Enterprise standardized scalars reuse delay-key names for sampler x axes.
        if (
            coord == "x"
            and self.model.transform == "standardized"
            and all(key in params for key in self.delay_keys)
        ):
            values = np.asarray([params[key] for key in self.delay_keys], dtype=float)
            # Implicit default coord corresponds to contribute_timing outputs
            # where delay keys already carry delta values.
            if not coord_explicit:
                return values
            # Explicit coord="x" keeps Enterprise standardized-scalar semantics.
            return np.asarray(
                self.space.delta_from_coord(values, np, coord="x"), dtype=float
            )

        raise ValueError(
            f"delta_from_params(coord={coord!r}) could not find matching "
            "timing coordinates"
        )

    def artifact(
        self,
        *,
        frontend: str,
        sampler: str,
        scenario: str | None = None,
        latent: dict[str, Any] | None = None,
        checkpoint: dict[str, Any] | None = None,
        chain_layout: dict[str, Any] | None = None,
        git_commit: str | None = None,
    ):
        """Snapshot this binding as a write-side ``NLTBinding`` artifact."""
        from .artifacts import build_binding

        return build_binding(
            self,
            frontend=frontend,
            sampler=sampler,
            scenario=scenario,
            latent=latent,
            checkpoint=checkpoint,
            chain_layout=chain_layout,
            git_commit=git_commit,
        )

    def write(self, run_dir, **kwargs):
        """Build the artifact and write sidecar + parameter space to ``run_dir``.

        Returns the written ``NLTBinding`` (needed for checkpoint helpers).
        """
        binding = self.artifact(**kwargs)
        binding.write(run_dir)
        return binding


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
        tempo2_native: str | None = None,
        tempo2_jug_options: Mapping[str, Any] | None = None,
        transform: str = "whitening",
        sample: str | Sequence[str] | None = None,
        analytically_marginalize: str | Sequence[str] | None = "default",
        priors: Mapping[str, PriorOverrideSpec] | None = None,
        prior_policy: PriorPolicy = "fallback",
        prior_override_policy: Literal["warn", "strict"] = "warn",
        cheat_prior_scale: float = 50.0,
        whitening_config: Mapping[str, Any] | None = None,
        name: str = "nonlinear_timing_model",
    ):
        if transform not in _TRANSFORMS:
            raise ValueError(f"Unsupported transform: {transform}")
        if not (float(cheat_prior_scale) > 0.0):
            raise ValueError("cheat_prior_scale must be positive")
        override_policy = str(prior_override_policy or "warn").lower()
        if override_policy not in _PRIOR_OVERRIDE_POLICIES:
            raise ValueError(
                "prior_override_policy must be 'warn' or 'strict'; "
                f"got {prior_override_policy!r}"
            )
        if sample is not None and sample != "default":
            if isinstance(sample, str):
                raise ValueError(
                    "sample must be 'default', None, or a sequence of fitpar names"
                )
            if analytically_marginalize != "default":
                raise ValueError(
                    "pass either sample= or analytically_marginalize=, not both"
                )
            sample = tuple(str(name) for name in sample)
        self.engines = normalize_engines(engines)
        self.design_matrix_method = _normalize_design_matrix_method(
            design_matrix_method
        )
        self.tempo2_native = tempo2_native
        self._tempo2_jug_options_raw = (
            None if tempo2_jug_options is None else dict(tempo2_jug_options)
        )
        self._tempo2_jug_options_resolved: dict[str, Any] | None = None
        self.prior_override_policy = override_policy
        self.transform = transform
        self.sample = sample
        self.analytically_marginalize = analytically_marginalize
        self.prior_policy = validate_prior_policy(prior_policy)
        self.cheat_prior_scale = float(cheat_prior_scale)
        self.whitening_config = (
            None if whitening_config is None else dict(whitening_config)
        )
        self.name = name
        self._prior_overrides: dict[str, PriorOverrideSpec] = {}
        self._resolved_cache: dict[str, TimingBinding] = {}
        for prior_name, spec in dict(priors or {}).items():
            if not isinstance(spec, PriorOverrideSpec):
                raise TypeError(
                    f"priors[{prior_name!r}] must be a PriorOverrideSpec (use the "
                    "helpers in metapulsar.timing.priors, e.g. delta_uniform)"
                )
            self._prior_overrides = store_prior_override(
                self._prior_overrides, prior_name, spec
            )

    def _uses_jug(self) -> bool:
        return (
            "jug" in self.engines.values()
            or self.tempo2_native is not None
            or self._tempo2_jug_options_raw is not None
        )

    @property
    def tempo2_jug_options(self) -> dict[str, Any] | None:
        """Resolved JUG tempo2 session options; ``None`` for JUG-free configs.

        Resolution imports ``jug.timing`` lazily so that libstempo/PINT-only
        configurations construct and bind without JUG installed.
        """
        if not self._uses_jug():
            return None
        if self._tempo2_jug_options_resolved is None:
            from jug.timing import resolve_tempo2_jug_options

            self._tempo2_jug_options_resolved = resolve_tempo2_jug_options(
                self._tempo2_jug_options_raw
            )
        return self._tempo2_jug_options_resolved

    def set_prior(
        self,
        name: str,
        kind: str,
        *,
        frame: PriorFrame = "absolute",
        scale: str | None = None,
        **bounds,
    ) -> None:
        """Set or override one sampled-parameter prior.

        frame='absolute': bounds are native/physical values converted to delta at bind time.
        frame='delta': bounds are offsets from the bound parameter's backend reference.
        scale: optional fitpar name; multiplies bounds by that parameter's backend ref
               (delta frame only).
        """
        if kind == "normal":
            spec = prior_specs.normal(
                bounds["mean"], bounds["std"], frame=frame, scale=scale
            )
        elif kind == "uniform":
            spec = prior_specs.uniform(
                bounds["lower"], bounds["upper"], frame=frame, scale=scale
            )
        elif kind == "log_uniform":
            if scale is not None:
                raise ValueError("scale=... is not supported for log_uniform priors")
            spec = prior_specs.log_uniform(bounds["lower"], bounds["upper"])
        elif kind == "truncated_normal":
            spec = prior_specs.truncated_normal(
                bounds["mean"],
                bounds["std"],
                bounds["lower"],
                bounds["upper"],
                frame=frame,
                scale=scale,
            )
        else:
            raise ValueError(f"Unsupported prior kind: {kind}")

        self._prior_overrides = store_prior_override(self._prior_overrides, name, spec)
        self._resolved_cache.clear()

    def set_prior_delta(
        self,
        name: str,
        kind: str,
        *,
        scale: str | None = None,
        **bounds,
    ) -> None:
        """Convenience wrapper for frame='delta' priors."""
        self.set_prior(name, kind, frame="delta", scale=scale, **bounds)

    def with_engines(self, engines) -> "NonLinearTimingModel":
        """Return a new model config with a different engine selection."""
        other = NonLinearTimingModel(
            engines=engines,
            design_matrix_method=self.design_matrix_method,
            tempo2_native=self.tempo2_native,
            tempo2_jug_options=self._tempo2_jug_options_raw,
            transform=self.transform,
            sample=self.sample,
            analytically_marginalize=self.analytically_marginalize,
            prior_policy=self.prior_policy,
            prior_override_policy=self.prior_override_policy,
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
            "tempo2_native": self._tempo2_native_fingerprint(),
            "tempo2_jug_options": self.tempo2_jug_options,
            "transform": self.transform,
            "sample": (
                list(self.sample) if isinstance(self.sample, tuple) else self.sample
            ),
            "analytically_marginalize": self.analytically_marginalize,
            "prior_policy": self.prior_policy,
            "prior_override_policy": self.prior_override_policy,
            "cheat_prior_scale": self.cheat_prior_scale,
            "whitening_config": self.whitening_config,
            "name": self.name,
            "prior_overrides": {
                key: {
                    "frame": spec.frame,
                    "scale": spec.scale,
                    "prior": vars(spec.prior),
                }
                for key, spec in sorted(self._prior_overrides.items())
            },
        }
        return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()

    def _tempo2_native_fingerprint(self) -> str | None:
        if self.tempo2_native is None:
            return None
        return str(self.tempo2_native)

    def _timing_backend_kwargs(self) -> dict[str, Any]:
        return {
            "tempo2_native": self.tempo2_native,
            "tempo2_jug_options": self.tempo2_jug_options,
            "prime_sessions": True,
            "verify_wiring": False,
            "subtract_tzr": False,
        }

    def _backend_for_pulsar(self, pulsar):
        return pulsar.timing_backend(
            self.engines,
            design_matrix_method=self.design_matrix_method,
            **self._timing_backend_kwargs(),
        )

    def _partition(self, pulsar) -> PartitionResult:
        return resolve_partition(
            pulsar,
            analytically_marginalize=self.analytically_marginalize,
            sample=self.sample,
        )

    def _resolve_prior_overrides(
        self,
        *,
        pulsar,
        backend,
        partition: PartitionResult,
    ) -> dict[str, AxisPrior]:
        """Materialize stored override specs into delta-space AxisPrior values.

        Override keys may be base names (``"TASC"``); each expands to every
        matching (possibly PTA-suffixed) sampled fitpar. ``scale=`` references
        resolve suffix-consistently with the target parameter.
        """
        sampled_set = set(partition.sampled)
        expansion = {
            name: match_fitpars(pulsar, name, partition.fitpars)
            for name in self._prior_overrides
        }

        unknown = sorted(name for name, hits in expansion.items() if not hits)
        if unknown:
            if self.prior_override_policy == "strict":
                raise ValueError(
                    "Prior overrides target unknown fit parameters for this pulsar: "
                    f"{unknown}"
                )
            warnings.warn(
                "Skipping prior overrides for fit parameters absent on this pulsar: "
                f"{unknown}",
                UserWarning,
                stacklevel=3,
            )
        invalid = sorted(
            name
            for name, hits in expansion.items()
            if hits and not any(hit in sampled_set for hit in hits)
        )
        if invalid:
            if self.prior_override_policy == "strict":
                raise ValueError(
                    "Prior overrides target non-sampled parameters for this pulsar: "
                    f"{invalid}"
                )
            warnings.warn(
                "Skipping prior overrides for non-sampled fit parameters on this pulsar: "
                f"{invalid}",
                UserWarning,
                stacklevel=3,
            )

        ref_exact = backend.reference_theta_exact()
        ctx = PriorBuildContext(
            refs=ref_exact,
            fitpars=partition.fitpars,
            sampled=partition.sampled,
        )
        resolved: dict[str, AxisPrior] = {}
        for name, spec in self._prior_overrides.items():
            for target in expansion[name]:
                if target not in sampled_set:
                    continue
                target_spec = self._spec_for_target(pulsar, spec, target, partition)
                resolved[target] = materialize_prior_override(target, target_spec, ctx)
        return resolved

    def _spec_for_target(
        self,
        pulsar,
        spec: PriorOverrideSpec,
        target: str,
        partition: PartitionResult,
    ) -> PriorOverrideSpec:
        """Resolve a spec's ``scale=`` reference suffix-consistently with ``target``."""
        if spec.scale is None:
            return spec
        scale_hits = match_fitpars(pulsar, spec.scale, partition.fitpars)
        if not scale_hits:
            # Preserve the standard missing-scale error from materialization.
            return spec
        if len(scale_hits) == 1:
            resolved_scale = scale_hits[0]
        else:
            suffixes = fitpar_suffixes(pulsar, target)
            matched = [
                hit for hit in scale_hits if suffixes & fitpar_suffixes(pulsar, hit)
            ]
            if len(matched) != 1:
                raise ValueError(
                    f"Ambiguous prior scale {spec.scale!r} for parameter "
                    f"{target!r}: candidates {list(scale_hits)}"
                )
            resolved_scale = matched[0]
        if resolved_scale == spec.scale:
            return spec
        return PriorOverrideSpec(
            prior=spec.prior, frame=spec.frame, scale=resolved_scale
        )

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
            overrides=self._resolve_prior_overrides(
                pulsar=pulsar,
                backend=backend,
                partition=partition,
            ),
            overrides_in_delta=True,
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

    def bind(self, pulsar) -> TimingBinding:
        """Resolve this model config against a pulsar (cached per state)."""
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
        resolved = TimingBinding(
            model=self,
            pulsar=pulsar,
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

    def enterprise_signal(self):
        from .frontends.enterprise import enterprise_signal

        return enterprise_signal(
            binding_fn=self.bind,
            name=self.name,
            transform=self.transform,
        )
