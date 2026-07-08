"""Tests for NLT artifact contract (sidecar, binding, chain bundle)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from metapulsar.timing.artifacts import (
    DISCOVERY_CHECKPOINT_NAME,
    DISCOVERY_FINAL_NAME,
    NLTArtifactError,
    NLTChainBundle,
    build_binding,
    deterministic_site_name,
    physical_deterministics,
    save_discovery_checkpoint,
)
from metapulsar.timing.backends.base import LinearModel
from metapulsar.timing.backends.jug import LinearizedJugEngine
from metapulsar.timing.nonlinear_timing_model import NonLinearTimingModel
from metapulsar.timing.space import ParameterSpace


class _Host:
    def __init__(self, *, cache_token: str = "artifact-token"):
        self.name = "J1111+1111"
        self.fitpars = ("F0", "F1")
        self._toas = np.linspace(0.0, 1.0, 5)
        self._residuals = np.zeros(5)
        self._toaerrs = np.full(5, 1.0e-6)
        self._freqs = np.full(5, 1400.0)
        self._flags = {"pta": np.array(["demo"] * 5, dtype="U8")}
        self._backend_flags = np.array(["demo"] * 5, dtype="U8")
        self._cache_token = cache_token
        model = LinearModel.from_host(
            fitpars=self.fitpars,
            design=np.column_stack([np.ones(5), np.linspace(-0.5, 0.5, 5)]),
            theta_exact={"F0": "100.0", "F1": "1.0"},
        )
        self._backend = LinearizedJugEngine.from_linear_model(model)

    @property
    def toas(self):
        return self._toas

    @property
    def residuals(self):
        return self._residuals

    @property
    def toaerrs(self):
        return self._toaerrs

    @property
    def freqs(self):
        return self._freqs

    @property
    def Mmat(self):
        return self._backend.design_matrix()

    @property
    def flags(self):
        return self._flags

    @property
    def backend_flags(self):
        return self._backend_flags

    def cache_token(self):
        return self._cache_token

    def pint_model(self):
        return object()

    def timing_backend(self, engines="jug", **kwargs):
        return self._backend


@pytest.fixture
def host():
    return _Host()


@pytest.fixture
def ntm():
    return NonLinearTimingModel(
        engines="jug",
        transform="whitening",
        analytically_marginalize=["F0"],
        name="timing",
    )


@pytest.fixture
def binding(host, ntm):
    return build_binding(
        ntm,
        host,
        frontend="discovery",
        sampler="numpyro-nuts",
        scenario="demo",
        checkpoint={"kind": "npz", "path": DISCOVERY_CHECKPOINT_NAME, "key_name": "x"},
        latent={"kind": "npz", "path": DISCOVERY_FINAL_NAME, "key_name": "x"},
    )


def test_sidecar_written_before_checkpoint(tmp_path, binding):
    binding.write(tmp_path)
    x = np.array([[0.1], [0.2]], dtype=float)
    save_discovery_checkpoint(tmp_path, x, binding, final=False, n_target=4)
    assert (tmp_path / "nlt_sidecar.json").is_file()
    assert (tmp_path / DISCOVERY_CHECKPOINT_NAME).is_file()


def test_space_fingerprint_stable_across_save_load(tmp_path, binding):
    fp1 = binding.space.fingerprint()
    binding.space.save(tmp_path / "nlt_parameter_space")
    loaded = ParameterSpace.load(tmp_path / "nlt_parameter_space")
    fp2 = loaded.fingerprint()
    assert fp1 == fp2


def test_space_fingerprint_changes_when_C_changes(binding):
    fp_before = binding.space.fingerprint()
    binding.space.linear.C[0, 0] *= 1.1
    assert binding.space.fingerprint() != fp_before


def test_binding_fingerprint_changes_with_pulsar_state(host, ntm):
    fp_a = ntm.binding_fingerprint(host)
    host._cache_token = "artifact-token-updated"
    fp_b = ntm.binding_fingerprint(host)
    assert fp_a != fp_b


def test_bundle_refuses_wrong_space_same_names(tmp_path, binding):
    binding.write(tmp_path)
    arrays = np.load(str(tmp_path / "nlt_parameter_space") + ".npz")
    tampered_C = arrays["C"].copy()
    tampered_C[0, 0] *= 2.0
    np.savez(
        str(tmp_path / "nlt_parameter_space") + ".npz",
        C=tampered_C,
        z0=arrays["z0"],
    )
    with pytest.raises(NLTArtifactError, match="fingerprint mismatch"):
        NLTChainBundle.load(tmp_path, verify=True)


def test_bundle_force_downgrades_mismatch_to_warning(tmp_path, binding):
    binding.write(tmp_path)
    arrays = np.load(str(tmp_path / "nlt_parameter_space") + ".npz")
    tampered_C = arrays["C"].copy()
    tampered_C[0, 0] *= 2.0
    np.savez(
        str(tmp_path / "nlt_parameter_space") + ".npz",
        C=tampered_C,
        z0=arrays["z0"],
    )
    with pytest.warns(UserWarning, match="fingerprint mismatch"):
        NLTChainBundle.load(tmp_path, verify=True, force=True)


def test_decode_requires_no_pulsar_rebuild(tmp_path, binding):
    binding.write(tmp_path)
    x = np.array([[0.05], [0.15]], dtype=float)
    save_discovery_checkpoint(tmp_path, x, binding, final=True, n_target=2)
    bundle = NLTChainBundle.load(tmp_path)
    loaded_x = bundle.load_latent()
    np.testing.assert_allclose(loaded_x, x)
    display = bundle.load_display()
    assert "F1" in display


def test_discovery_checkpoint_contains_display_deterministics(tmp_path, binding):
    binding.write(tmp_path)
    x = np.array([[0.1], [0.2]], dtype=float)
    save_discovery_checkpoint(tmp_path, x, binding, final=False, n_target=4)
    data = np.load(tmp_path / DISCOVERY_CHECKPOINT_NAME)
    prefix = binding.deterministic_prefix
    assert deterministic_site_name(prefix, "F1", "display") in data.files


def test_load_display_prefers_latent_over_checkpoint(tmp_path, binding):
    binding.write(tmp_path)
    x_ckpt = np.array([[0.1]], dtype=float)
    x_final = np.array([[0.9]], dtype=float)
    save_discovery_checkpoint(tmp_path, x_ckpt, binding, final=False, n_target=2)
    save_discovery_checkpoint(tmp_path, x_final, binding, final=True, n_target=2)
    bundle = NLTChainBundle.load(tmp_path)
    prefix = binding.deterministic_prefix
    display_key = deterministic_site_name(prefix, "F1", "display")
    final_data = np.load(tmp_path / DISCOVERY_FINAL_NAME)[display_key]
    loaded = bundle.load_display()["F1"]
    np.testing.assert_allclose(loaded, final_data)


def test_load_latent_prefers_chain_txt_over_npz(tmp_path, host, ntm):
    ndim = len(ntm.sampled(host))
    binding = build_binding(
        ntm,
        host,
        frontend="enterprise",
        sampler="ptmcmc",
        chain_layout={
            "kind": "ptmcmc",
            "file": "chains/chain_1.txt",
            "columns": list(range(ndim)),
            "coord": "x",
        },
        latent={"kind": "npz", "path": "enterprise_x.npz", "key_name": "x"},
    )
    binding.write(tmp_path)
    chains_dir = tmp_path / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)
    chain_x = np.array([[1.0], [2.0], [3.0]], dtype=float)
    meta = np.column_stack(
        [
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
        ]
    )
    np.savetxt(chains_dir / "chain_1.txt", np.hstack([chain_x, meta]))
    npz_x = np.array([[9.0], [8.0]], dtype=float)
    np.savez(tmp_path / "enterprise_x.npz", x=npz_x)
    bundle = NLTChainBundle.load(tmp_path)
    np.testing.assert_allclose(bundle.load_latent(), chain_x)
    np.testing.assert_allclose(bundle.load_latent(prefer_npz=True), npz_x)


def test_load_latent_returns_raw_chain_including_burn(tmp_path, host, ntm):
    ndim = len(ntm.sampled(host))
    binding = build_binding(
        ntm,
        host,
        frontend="enterprise",
        sampler="ptmcmc",
        chain_layout={
            "kind": "ptmcmc",
            "file": "chains/chain_1.txt",
            "columns": list(range(ndim)),
            "coord": "x",
        },
    )
    binding.write(tmp_path)
    chains_dir = tmp_path / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)
    chain_x = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=float)
    meta = np.zeros((4, 4))
    np.savetxt(chains_dir / "chain_1.txt", np.hstack([chain_x, meta]))
    bundle = NLTChainBundle.load(tmp_path)
    loaded = bundle.load_latent()
    assert loaded.shape[0] == 4


def test_enterprise_sidecar_omits_latent_by_default(tmp_path, host, ntm):
    ndim = len(ntm.sampled(host))
    binding = build_binding(
        ntm,
        host,
        frontend="enterprise",
        sampler="ptmcmc",
        chain_layout={
            "kind": "ptmcmc",
            "file": "chains/chain_1.txt",
            "columns": list(range(ndim)),
            "coord": "x",
        },
    )
    binding.write(tmp_path)
    sidecar = json.loads((tmp_path / "nlt_sidecar.json").read_text())
    assert "latent" not in sidecar["artifacts"]


def test_ptmcmc_chain_decodes_with_sidecar(tmp_path, host, ntm):
    ndim = len(ntm.sampled(host))
    binding = build_binding(
        ntm,
        host,
        frontend="enterprise",
        sampler="ptmcmc",
        chain_layout={
            "kind": "ptmcmc",
            "file": "chains/chain_1.txt",
            "columns": list(range(ndim)),
            "coord": "x",
        },
    )
    binding.write(tmp_path)
    chains_dir = tmp_path / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)
    x = np.array([[0.01], [0.02]], dtype=float)
    meta = np.zeros((2, 4))
    np.savetxt(chains_dir / "chain_1.txt", np.hstack([x, meta]))
    bundle = NLTChainBundle.load(tmp_path)
    phys = bundle.load_display()
    assert "F1" in phys
    assert phys["F1"].shape == (2,)


def test_display_units_recorded_without_pint_model(tmp_path, binding):
    binding.write(tmp_path)
    sidecar = json.loads((tmp_path / "nlt_sidecar.json").read_text())
    assert sidecar["display_units"]["F1"] == "native"
    assert sidecar["native_units"]["F1"] == "native"


def test_physical_deterministics_keys(binding):
    x = np.array([[0.1], [0.2]], dtype=float)
    out = physical_deterministics(binding.space, x, binding.deterministic_prefix)
    prefix = binding.deterministic_prefix
    assert deterministic_site_name(prefix, "F1", "native") in out
    assert deterministic_site_name(prefix, "F1", "display") in out


def test_arviz_export_has_fingerprint_attrs(tmp_path, binding):
    pytest.importorskip("arviz")
    binding.write(tmp_path)
    x = np.array([[0.1], [0.2]], dtype=float)
    save_discovery_checkpoint(tmp_path, x, binding, final=True, n_target=2)
    bundle = NLTChainBundle.load(tmp_path)
    idata = bundle.to_arviz()
    assert idata.attrs["space_fingerprint"] == binding.space_fingerprint
    assert idata.attrs["binding_fingerprint"] == binding.binding_fingerprint
