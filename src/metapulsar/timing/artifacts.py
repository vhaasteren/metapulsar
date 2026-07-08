"""Self-contained NLT artifact contract: sidecar + serialized ParameterSpace.

A valid NLT read requires ONLY on-disk artifacts (nlt_sidecar.json,
nlt_parameter_space.{json,npz}) plus raw sampler output. It MUST NOT require an
Enterprise PTA, Discovery model, MetaPulsar construction, or a PINT reload.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .space import ParameterSpace, default_coord_for_transform
from .units import units_map

SIDECAR_SCHEMA = "nlt-sidecar-v1"
SIDECAR_FILENAME = "nlt_sidecar.json"
PARAMETER_SPACE_STEM = "nlt_parameter_space"
ENTERPRISE_NPZ_NAME = "enterprise_x.npz"
DISCOVERY_FINAL_NAME = "discovery_x.npz"
DISCOVERY_CHECKPOINT_NAME = "discovery_x_checkpoint.npz"
_NATIVE_SUFFIX = "_theta_native"
_DISPLAY_SUFFIX = "_theta_display"


class NLTArtifactError(RuntimeError):
    """Raised when NLT artifacts are missing, mismatched, or unreadable."""


def _code_block(git_commit: str | None) -> dict[str, Any]:
    from metapulsar import __version__

    return {"package": "metapulsar", "version": __version__, "git_commit": git_commit}


def deterministic_site_name(prefix: str, fitpar: str, units: str) -> str:
    return f"{prefix}_{fitpar}_theta_{units}"


def physical_deterministics(
    space: ParameterSpace,
    x: np.ndarray,
    prefix: str,
    *,
    units: Sequence[str] = ("native", "display"),
) -> dict[str, np.ndarray]:
    """Decode latent draws x into {prefix}_{fitpar}_theta_{units} arrays."""
    x = np.asarray(x, dtype=float)
    out: dict[str, np.ndarray] = {}
    for unit_mode in units:
        phys = space.to_physical(x, units=unit_mode)
        for name, arr in phys.items():
            out[deterministic_site_name(prefix, name, unit_mode)] = np.asarray(
                arr, dtype=float
            )
    return out


@dataclass
class NLTBinding:
    """Write-side snapshot of everything needed to decode a run's samples."""

    space: ParameterSpace
    frontend: str
    sampler: str
    pulsar_name: str
    sample_site: str
    sampled: tuple[str, ...]
    coord: str
    transform: str
    design_matrix_method: str
    engines: dict[str, str]
    native_units: dict[str, str]
    display_units: dict[str, str]
    binding_fingerprint: str
    space_fingerprint: str
    deterministic_prefix: str
    scenario: str | None = None
    latent: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    chain_layout: dict[str, Any] | None = None
    git_commit: str | None = None

    def sidecar(self) -> dict[str, Any]:
        artifacts: dict[str, Any] = {"parameter_space_stem": PARAMETER_SPACE_STEM}
        if self.latent is not None:
            artifacts["latent"] = self.latent
        if self.checkpoint is not None:
            artifacts["checkpoint"] = self.checkpoint
        if self.chain_layout is not None:
            artifacts["chain_layout"] = self.chain_layout
        return {
            "schema": SIDECAR_SCHEMA,
            "frontend": self.frontend,
            "sampler": self.sampler,
            "pulsar": self.pulsar_name,
            "scenario": self.scenario,
            "sample_coord": self.coord,
            "sample_site": self.sample_site,
            "sampled": list(self.sampled),
            "transform": self.transform,
            "design_matrix_method": self.design_matrix_method,
            "engines": dict(self.engines),
            "native_units": self.native_units,
            "display_units": self.display_units,
            "deterministic_prefix": self.deterministic_prefix,
            "deterministic_suffixes": {
                "native": _NATIVE_SUFFIX,
                "display": _DISPLAY_SUFFIX,
            },
            "parameter_space": {
                "json": f"{PARAMETER_SPACE_STEM}.json",
                "arrays": f"{PARAMETER_SPACE_STEM}.npz",
                "fingerprint": self.space_fingerprint,
            },
            "binding_fingerprint": self.binding_fingerprint,
            "artifacts": artifacts,
            "code": _code_block(self.git_commit),
        }

    def write(self, run_dir: str | Path) -> Path:
        """Write sidecar + parameter space. Call before the first sample/checkpoint."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.space.save(run_dir / PARAMETER_SPACE_STEM)
        path = run_dir / SIDECAR_FILENAME
        path.write_text(json.dumps(self.sidecar(), indent=2), encoding="utf-8")
        return path


def build_binding(
    ntm,
    pulsar,
    *,
    frontend: str,
    sampler: str,
    scenario: str | None = None,
    latent: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
    chain_layout: dict[str, Any] | None = None,
    git_commit: str | None = None,
) -> NLTBinding:
    """Materialize an NLTBinding from a bound NonLinearTimingModel + pulsar."""
    space = ntm.space(pulsar)
    sampled = tuple(ntm.sampled(pulsar))
    keys = ntm.timing_param_keys(pulsar)
    if not keys:
        raise NLTArtifactError(
            f"pulsar {pulsar.name!r} has no sampled timing parameters to bind"
        )
    pint_model = pulsar.pint_model()
    prefix = f"{pulsar.name}_{ntm.name}"
    return NLTBinding(
        space=space,
        frontend=frontend,
        sampler=sampler,
        pulsar_name=pulsar.name,
        sample_site=keys[0],
        sampled=sampled,
        coord=default_coord_for_transform(ntm.transform),
        transform=ntm.transform,
        design_matrix_method=ntm.design_matrix_method,
        engines=dict(ntm.engines),
        native_units=units_map(sampled, pint_model, kind="native"),
        display_units=units_map(sampled, pint_model, kind="display"),
        binding_fingerprint=ntm.binding_fingerprint(pulsar),
        space_fingerprint=space.fingerprint(),
        deterministic_prefix=prefix,
        scenario=scenario,
        latent=latent,
        checkpoint=checkpoint,
        chain_layout=chain_layout,
        git_commit=git_commit,
    )


def save_discovery_checkpoint(
    run_dir: str | Path,
    x: np.ndarray,
    binding: NLTBinding,
    *,
    final: bool,
    n_target: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write Discovery latent x + native/display deterministics to an npz."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=float)
    payload: dict[str, Any] = {
        "x": x,
        "n_draws": int(x.shape[0]),
        "sample_site": binding.sample_site,
        "space_fingerprint": binding.space_fingerprint,
        "binding_fingerprint": binding.binding_fingerprint,
    }
    if n_target is not None:
        payload["n_target"] = int(n_target)
    payload.update(
        physical_deterministics(binding.space, x, binding.deterministic_prefix)
    )
    if extra:
        payload.update({k: np.asarray(v) for k, v in extra.items()})
    name = DISCOVERY_FINAL_NAME if final else DISCOVERY_CHECKPOINT_NAME
    out = run_dir / name
    np.savez(out, **payload)
    return out


@dataclass
class NLTChainBundle:
    """Read-side decoder over existing sampler output. No live model needed."""

    run_dir: Path
    sidecar: dict[str, Any]
    space: ParameterSpace

    @classmethod
    def load(
        cls, run_dir: str | Path, *, verify: bool = True, force: bool = False
    ) -> "NLTChainBundle":
        run_dir = Path(run_dir)
        sidecar_path = run_dir / SIDECAR_FILENAME
        if not sidecar_path.is_file():
            raise NLTArtifactError(f"missing {SIDECAR_FILENAME} in {run_dir}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("schema") != SIDECAR_SCHEMA:
            raise NLTArtifactError(
                f"unexpected sidecar schema {sidecar.get('schema')!r}"
            )
        stem = sidecar["artifacts"]["parameter_space_stem"]
        space = ParameterSpace.load(run_dir / stem)
        if verify:
            actual = space.fingerprint()
            expected = sidecar["parameter_space"]["fingerprint"]
            if actual != expected:
                msg = (
                    "parameter space fingerprint mismatch: "
                    f"sidecar={expected} disk={actual} in {run_dir}"
                )
                if force:
                    warnings.warn(msg, stacklevel=2)
                else:
                    raise NLTArtifactError(msg)
        return cls(run_dir=run_dir, sidecar=sidecar, space=space)

    @property
    def coord(self) -> str:
        return self.sidecar["sample_coord"]

    def load_latent(self, *, prefer_npz: bool = False) -> np.ndarray:
        """Load latent sample rows. Returns raw chain (burn-in included for PTMCMC).

        Callers must apply burn/thin slicing themselves (e.g. x[burn:]).
        """
        artifacts = self.sidecar["artifacts"]

        def _load_ptmcmc(spec: dict[str, Any]) -> np.ndarray:
            chain_path = self.run_dir / spec["file"]
            if not chain_path.is_file():
                raise NLTArtifactError(f"missing PTMCMC chain file {chain_path}")
            chain = np.loadtxt(chain_path)
            cols = list(spec["columns"])
            return np.asarray(chain[:, cols], dtype=float)

        def _load_npz(spec: dict[str, Any]) -> np.ndarray:
            data = np.load(self.run_dir / spec["path"])
            return np.asarray(data[spec.get("key_name", "x")], dtype=float)

        chain_spec = artifacts.get("chain_layout")
        latent_spec = artifacts.get("latent")
        checkpoint_spec = artifacts.get("checkpoint")

        if prefer_npz and latent_spec is not None and latent_spec.get("kind") == "npz":
            return _load_npz(latent_spec)

        if chain_spec is not None and chain_spec.get("kind") == "ptmcmc":
            return _load_ptmcmc(chain_spec)

        if latent_spec is not None and latent_spec.get("kind") == "npz":
            return _load_npz(latent_spec)

        if checkpoint_spec is not None and checkpoint_spec.get("kind") == "npz":
            return _load_npz(checkpoint_spec)

        raise NLTArtifactError(
            "sidecar has no readable latent source (chain_layout, latent, or checkpoint)"
        )

    def _load_theta(self, unit_mode: str) -> dict[str, np.ndarray]:
        suffix = _DISPLAY_SUFFIX if unit_mode == "display" else _NATIVE_SUFFIX
        prefix = self.sidecar["deterministic_prefix"]
        artifacts = self.sidecar["artifacts"]
        for key in ("latent", "checkpoint"):
            spec = artifacts.get(key)
            if spec is None or spec.get("kind") != "npz":
                continue
            data = np.load(self.run_dir / spec["path"])
            keys = [k for k in data.files if k.endswith(suffix)]
            if keys:
                return {
                    k[len(prefix) + 1 : -len(suffix)]: np.asarray(data[k], dtype=float)
                    for k in keys
                }
        return self.space.to_physical(self.load_latent(), units=unit_mode)

    def load_display(self) -> dict[str, np.ndarray]:
        return self._load_theta("display")

    def load_native(self) -> dict[str, np.ndarray]:
        return self._load_theta("native")

    def to_physical_table(self) -> dict[str, np.ndarray]:
        """Display-unit physical samples keyed by fitpar name."""
        return self.load_display()

    def export_physical_table(self, path: str | Path) -> Path:
        path = Path(path)
        table = self.to_physical_table()
        if path.suffix == ".parquet":
            try:
                import pandas as pd
            except ImportError as exc:
                raise NLTArtifactError(
                    "parquet export requires pandas; install pandas/pyarrow"
                ) from exc
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(table).to_parquet(path)
            return path
        if path.suffix == ".npz":
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **table)
            return path
        raise NLTArtifactError(f"unsupported physical-table extension {path.suffix!r}")

    def to_arviz(self, *, sample_stats: Mapping[str, np.ndarray] | None = None):
        try:
            import arviz as az
        except ImportError as exc:
            raise NLTArtifactError("to_arviz requires arviz") from exc
        posterior = {k: v[None, ...] for k, v in self.load_display().items()}
        stats = (
            {k: np.asarray(v)[None, ...] for k, v in sample_stats.items()}
            if sample_stats
            else None
        )
        return az.from_dict(
            posterior=posterior,
            sample_stats=stats,
            attrs={
                "nlt_sidecar_schema": self.sidecar["schema"],
                "space_fingerprint": self.sidecar["parameter_space"]["fingerprint"],
                "binding_fingerprint": self.sidecar["binding_fingerprint"],
                "frontend": self.sidecar["frontend"],
                "sample_coord": self.sidecar["sample_coord"],
            },
        )
