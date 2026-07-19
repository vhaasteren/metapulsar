"""Tests for Discovery compatibility patches in paper validation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = REPO / "paper" / "code" / "validation"


@pytest.fixture(autouse=True)
def validation_path():
    for path in (
        REPO / "src",
        VALIDATION_ROOT,
        REPO / "ref-packages" / "discovery" / "src",
    ):
        path_str = str(path)
        if path.is_dir() and path_str not in sys.path:
            sys.path.insert(0, path_str)
    yield


def test_import_validation_patches_make_uind():
    import discovery.matrix as ds_matrix

    original = ds_matrix.make_uind
    import validation  # noqa: F401

    assert ds_matrix.make_uind is not original

    u = np.array([[1, 0], [0, 1], [1, 1]], dtype=int)
    uind = ds_matrix.make_uind(u)
    assert uind.shape == (2, 3)
    assert uind.dtype == int


def test_make_uind_empty_columns():
    import discovery.matrix as ds_matrix

    import validation  # noqa: F401

    u = np.zeros((5, 0), dtype=int)
    uind = ds_matrix.make_uind(u)
    assert uind.shape == (0, 1)
    assert uind.dtype == int


def test_timing_init_values_use_bound_site():
    from nltiming import sampling

    key = "J0613-0200_timing_x"
    binding = type(
        "Binding",
        (),
        {
            "sampled": tuple(f"P{i}" for i in range(7)),
            "latent_name_for_coord": lambda self: key,
        },
    )()
    init = sampling.numpyro.timing_init_values(binding)
    assert init[key].shape == (7,)


def test_binding_nuts_initializes_multiple_chains():
    import jax.random as jr
    import numpyro
    import numpyro.distributions as dist
    from nltiming import sampling

    key = "J0613-0200_timing_x"
    binding = type(
        "Binding",
        (),
        {
            "sampled": ("F0",),
            "latent_name_for_coord": lambda self: key,
        },
    )()

    def model():
        numpyro.sample(key, dist.Normal(0.0, 1.0).expand((1,)).to_event(1))

    mcmc = sampling.numpyro.nuts(
        model,
        binding,
        num_warmup=2,
        num_samples=2,
        num_chains=2,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(jr.PRNGKey(0))
    assert mcmc.get_samples()[key].shape == (4, 1)
