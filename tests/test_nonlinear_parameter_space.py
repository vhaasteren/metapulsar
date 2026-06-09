"""Tests for SampledTimingParameterSpace."""

import numpy as np

from metapulsar.nonlinear_timing_model import (
    AffineTransform,
    PintPriorAdapter,
    SampledTimingParameter,
    SampledTimingParameterRegistry,
    SampledTimingParameterSpace,
)


def _uniform_registry():
    params = [
        SampledTimingParameter(
            name="F0",
            theta_ref=316.0,
            transform=AffineTransform(center=0.0, scale=2.0),
            prior=PintPriorAdapter.from_uniform(314.0, 318.0, source="test"),
            source="test",
            units="Hz",
            sigma_wls=1.0,
        ),
        SampledTimingParameter(
            name="DM",
            theta_ref=18.0,
            transform=AffineTransform(center=0.0, scale=0.1),
            prior=PintPriorAdapter.from_uniform(17.0, 19.0, source="test"),
            source="test",
            units="",
            sigma_wls=0.01,
        ),
    ]
    return SampledTimingParameterRegistry(params)


def test_parameter_space_roundtrip_and_prior_parity():
    registry = _uniform_registry()
    space = SampledTimingParameterSpace.from_registry(registry)
    z = np.array([0.25, -0.5], dtype=float)
    theta = space.theta_from_z_np(z)
    recovered = space.z_from_theta_np(theta)
    np.testing.assert_allclose(recovered, z, atol=1e-12, rtol=0.0)

    lp_np = space.logprior_z_np(z)
    lp_jax = float(space.logprior_z_jax(z))
    assert np.isfinite(lp_np)
    np.testing.assert_allclose(lp_np, lp_jax, atol=1e-12, rtol=0.0)


def test_parameter_space_rejects_out_of_support():
    space = SampledTimingParameterSpace.from_registry(_uniform_registry())
    assert space.logprior_z_np(np.array([2.0, 0.0])) == -np.inf


def test_registry_to_parameter_space_metadata():
    registry = _uniform_registry()
    space = registry.to_parameter_space()
    assert space.names == ("F0", "DM")
    assert len(space.metadata()) == 2
