"""Slice-2 tests for prior policy resolution and bijector emission."""

import numpy as np
import pytest

from metapulsar.timing.bijectors import AxisPrior
from metapulsar.timing.priors import PriorBlock, set_prior
from metapulsar.timing.partition import PartitionResult
from metapulsar.timing.space import ParameterSpace
from metapulsar.timing.whitening import (
    diagonal_white,
    fixed_hyperparameters,
    schur_delta_wls,
)


def test_prior_block_fallback_marks_host_bound_cheat_priors():
    block = PriorBlock.from_fitpars(["F0", "F1"], policy="fallback")
    assert block.names == ("F0", "F1")
    assert block.source_labels()["F0"] == "cheat_wls"
    assert [prior.family for prior in block.priors] == ["cheat_wls", "cheat_wls"]
    with pytest.raises(ValueError, match="host-bound resolution"):
        block.to_bijector()


def test_prior_block_explicit_requires_overrides():
    with pytest.raises(ValueError, match="Missing explicit prior override"):
        PriorBlock.from_fitpars(["F0"], policy="explicit", overrides={})


def test_set_prior_normalizes_aliases():
    overrides = {}
    overrides = set_prior(overrides, "ECCDOT", AxisPrior(family="normal", std=2.0))
    assert "EDOT" in overrides
    assert overrides["EDOT"].std == 2.0


def test_prior_block_to_bijector_supports_uniform_and_normal():
    overrides = {
        "F0": AxisPrior(family="normal", mean=1.0, std=2.0),
        "A1": AxisPrior(family="uniform", lower=2.0, upper=6.0),
    }
    block = PriorBlock.from_fitpars(
        ["F0", "A1"],
        policy="fallback",
        overrides=overrides,
        theta_ref={"F0": "1.0", "A1": "3.0"},
    )
    bijector = block.to_bijector()
    z = np.array([0.5, 0.0], dtype=float)
    delta = bijector.delta_from_z(z, np)
    assert delta.shape == (2,)
    np.testing.assert_allclose(delta[0], 1.0, atol=1e-8)  # std=2 scale on z=0.5
    assert -1.0 <= delta[1] <= 3.0


def test_prior_block_offsets_absolute_priors_to_delta_space():
    block = PriorBlock.from_fitpars(
        ["F0"],
        policy="explicit",
        overrides={"F0": AxisPrior(family="uniform", lower=99.0, upper=101.0)},
        theta_ref={"F0": "100.0"},
    )
    prior = block.priors[0]
    assert prior.family == "uniform"
    np.testing.assert_allclose([prior.lower, prior.upper], [-1.0, 1.0])
    np.testing.assert_allclose(
        block.to_bijector().delta_from_z(np.array([0.0]), np), [0.0]
    )


def test_log_uniform_roundtrip_and_precision_guard():
    block = PriorBlock.from_fitpars(
        ["A1"],
        policy="explicit",
        overrides={"A1": AxisPrior(family="log_uniform", lower=1.0, upper=100.0)},
        theta_ref={"A1": "10.0"},
    )
    bijector = block.to_bijector()
    z = np.array([0.25], dtype=float)
    delta = bijector.delta_from_z(z, np)
    np.testing.assert_allclose(bijector.z_from_delta(delta, np), z, atol=1e-8)
    with pytest.raises(ValueError, match="precision-critical"):
        block.to_bijector(precision_critical_fitpars=frozenset({"A1"}))


def test_whitening_builders_change_coord_only_not_physical_prior():
    theta_ref = {"F0": "1.0", "F1": "2.0"}
    block = PriorBlock.from_fitpars(
        ["F0", "F1"],
        policy="explicit",
        overrides={
            "F0": AxisPrior(family="normal", mean=1.0, std=1.0),
            "F1": AxisPrior(family="normal", mean=2.0, std=1.0),
        },
        theta_ref=theta_ref,
    )
    bijector = block.to_bijector()

    default_space = ParameterSpace.build(
        theta_ref_mapping=theta_ref,
        prior_bijector=bijector,
        transform="whitening",
        linear_transform=diagonal_white(2),
    )
    shifted_space = ParameterSpace.build(
        theta_ref_mapping=theta_ref,
        prior_bijector=bijector,
        transform="whitening",
        linear_transform=fixed_hyperparameters(2, {"center": [0.3, -0.2]}),
    )

    delta = np.array([0.2, -0.1], dtype=float)
    np.testing.assert_allclose(
        default_space.logprior_physical(delta, np),
        shifted_space.logprior_physical(delta, np),
    )


class _WhiteningHost:
    fitpars = ("F0", "A1")

    @property
    def Mmat(self):
        return np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
            ],
            dtype=float,
        )

    @property
    def residuals(self):
        return np.array([1.0, 2.0, 2.0, 4.0], dtype=float)

    @property
    def toaerrs(self):
        return np.ones(4, dtype=float)

    @property
    def backend_flags(self):
        return np.array(["a", "a", "b", "b"])


def test_diagonal_white_uses_host_partition_for_nonidentity_transform():
    host = _WhiteningHost()
    partition = PartitionResult(
        fitpars=("F0", "A1"),
        analytically_marginalized=("F0",),
        sampled=("A1",),
        idx_analytically_marginalized=(0,),
        idx_sampled=(1,),
    )
    block = PriorBlock.from_fitpars(
        ["A1"],
        policy="explicit",
        overrides={"A1": AxisPrior(family="normal", mean=0.0, std=1.0)},
        theta_ref={"A1": "0.0"},
    )

    transform = diagonal_white(
        host=host,
        partition=partition,
        prior_bijector=block.to_bijector(),
    )

    assert transform.C.shape == (1, 1)
    assert not np.allclose(transform.C, np.eye(1))
    assert not np.allclose(transform.z0, np.zeros(1))


def test_fixed_hyperparameters_uses_serialized_white_noise_values():
    host = _WhiteningHost()
    partition = PartitionResult(
        fitpars=("F0", "A1"),
        analytically_marginalized=(),
        sampled=("F0", "A1"),
        idx_analytically_marginalized=(),
        idx_sampled=(0, 1),
    )

    default = diagonal_white(host=host, partition=partition)
    fixed = fixed_hyperparameters(
        host=host,
        partition=partition,
        hyperparameters={"efac": {"a": 2.0, "b": 1.0}, "equad": 0.1},
    )

    assert fixed.C.shape == (2, 2)
    assert not np.allclose(fixed.C, default.C)


class _DegenerateWhiteningHost:
    fitpars = ("P1", "P2")

    @property
    def Mmat(self):
        return np.ones((3, 2), dtype=float)

    @property
    def residuals(self):
        return np.zeros(3, dtype=float)

    @property
    def toaerrs(self):
        return np.ones(3, dtype=float)


def test_schur_delta_wls_raises_on_degenerate_fisher():
    host = _DegenerateWhiteningHost()
    partition = PartitionResult(
        fitpars=("P1", "P2"),
        analytically_marginalized=(),
        sampled=("P1", "P2"),
        idx_analytically_marginalized=(),
        idx_sampled=(0, 1),
    )
    with pytest.raises(ValueError, match="positive definite"):
        schur_delta_wls(
            host=host,
            partition=partition,
            variance=np.ones(3, dtype=float),
        )
