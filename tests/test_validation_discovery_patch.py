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


def test_timing_init_params_shapes():
    from validation.sampler_discovery import _timing_init_params

    key = "J0613-0200_timing_x"
    one = _timing_init_params(key, 7, num_chains=1)[key]
    assert one.shape == (7,)

    two = _timing_init_params(key, 7, num_chains=2)[key]
    assert two.shape == (2, 7)
