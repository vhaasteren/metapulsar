"""Slice-1 tests for ParameterSpace transform ownership."""

import numpy as np
import pytest

from metapulsar.timing.bijectors import PriorBijector, WhiteningLinear
from metapulsar.timing.precision import ExactNativeRef
from metapulsar.timing.space import DensityParts, ParameterSpace


def _build_standardized_space():
    theta_ref = {"F0": "123.456789", "F1": "-1.23e-15"}
    prior = PriorBijector.from_normal(
        names=("F0", "F1"),
        means=np.zeros(2, dtype=float),
        stds=np.ones(2, dtype=float),
    )
    linear = WhiteningLinear.identity(2)
    return ParameterSpace.build(
        theta_ref_mapping=theta_ref,
        prior_bijector=prior,
        transform="standardized",
        linear_transform=linear,
    )


def test_delta_z_roundtrip_standardized():
    space = _build_standardized_space()
    delta = np.array([0.2, -1.4], dtype=float)
    z = space.z_from_delta(delta, np)
    recovered = space.delta_from_z(z, np)
    np.testing.assert_allclose(recovered, delta)


def test_x_z_delta_roundtrip_whitening():
    theta_ref = {"F0": "1.0", "F1": "2.0"}
    prior = PriorBijector.from_normal(
        names=("F0", "F1"),
        means=np.zeros(2, dtype=float),
        stds=np.ones(2, dtype=float),
    )
    linear = WhiteningLinear(
        C=np.array([[2.0, 0.0], [0.3, 1.2]], dtype=float),
        z0=np.array([0.1, -0.2], dtype=float),
    )
    space = ParameterSpace.build(
        theta_ref_mapping=theta_ref,
        prior_bijector=prior,
        transform="whitening",
        linear_transform=linear,
    )
    x = np.array([0.25, -0.8], dtype=float)
    delta = space.delta_from_coord(x, np, coord="x")
    recovered_x = space.coord_from_delta(delta, np, coord="x")
    np.testing.assert_allclose(recovered_x, x)


def test_logprior_coord_identity():
    space = _build_standardized_space()
    z = np.array([0.5, -0.3], dtype=float)
    lhs = space.logprior_coord(z, np, coord="z")
    rhs = space.logprior_physical(
        space.delta_from_coord(z, np, coord="z"), np
    ) + space.logjacobian(z, np, coord="z")
    np.testing.assert_allclose(lhs, rhs)


def test_density_parts_named_tuple():
    space = _build_standardized_space()
    z = np.array([0.2, 0.1], dtype=float)
    parts = space.density_parts(
        z,
        loglike_fn=lambda delta: -0.5 * float(np.dot(delta, delta)),
        xp=np,
        coord="z",
    )
    assert isinstance(parts, DensityParts)
    assert set(parts._fields) == {
        "logprior_physical",
        "logjacobian",
        "logprior_coord",
        "loglike",
        "logpost",
    }
    np.testing.assert_allclose(parts.logpost, parts.loglike + parts.logprior_coord)


def test_delta_from_cube_uses_physical_prior_not_whitening():
    theta_ref = {"A1": "0.3"}
    prior = PriorBijector.from_uniform(
        names=("A1",),
        lowers=np.array([-2.0], dtype=float),
        uppers=np.array([3.0], dtype=float),
    )
    linear = WhiteningLinear(C=np.array([[3.0]], dtype=float), z0=np.array([5.0]))
    space = ParameterSpace.build(
        theta_ref_mapping=theta_ref,
        prior_bijector=prior,
        transform="whitening",
        linear_transform=linear,
    )
    u = np.array([0.5], dtype=float)
    delta = space.delta_from_cube(u, np)
    np.testing.assert_allclose(delta, np.array([0.5], dtype=float))


def test_save_load_roundtrip(tmp_path):
    space = _build_standardized_space()
    save_path = tmp_path / "space_artifact"
    space.save(save_path)
    loaded = ParameterSpace.load(save_path)
    assert loaded.names == space.names
    assert loaded.theta_ref.values == space.theta_ref.values
    np.testing.assert_allclose(loaded.linear.C, space.linear.C)
    np.testing.assert_allclose(loaded.linear.z0, space.linear.z0)


def test_theta_delta_boundaries_use_exact_ref_strings():
    space = _build_standardized_space()
    delta = np.array([1e-9, -2e-9], dtype=float)
    theta = space.theta_from_delta(delta)
    recovered_delta = space.delta_from_theta(theta)
    np.testing.assert_allclose(recovered_delta, delta, rtol=0.0, atol=1e-12)


def test_exact_native_ref_has_explicit_float_constructor():
    with np.testing.assert_raises(TypeError):
        ExactNativeRef.from_mapping({"F0": 123.0})  # type: ignore[arg-type]
    ref = ExactNativeRef.from_float_mapping({"F0": 123.0, "F1": -1.0e-15})
    assert ref.values == ("123.0", "-1e-15")


def test_fake_pulsar_interface_fixture_is_auto_discovered(fake_pulsar_interface):
    assert fake_pulsar_interface.name == "FAKEPSR"
    assert fake_pulsar_interface.Mmat.shape == (4, 2)


def test_whitening_linear_x_from_z_supports_jax_xp():
    jnp = pytest.importorskip("jax.numpy")
    theta_ref = {"F0": "0.0", "F1": "0.0"}
    prior = PriorBijector.from_normal(
        names=("F0", "F1"),
        means=np.zeros(2, dtype=float),
        stds=np.ones(2, dtype=float),
    )
    linear = WhiteningLinear(
        C=np.array([[2.0, 0.0], [0.5, 1.0]], dtype=float),
        z0=np.array([0.1, -0.2], dtype=float),
    )
    space = ParameterSpace.build(
        theta_ref_mapping=theta_ref,
        prior_bijector=prior,
        transform="whitening",
        linear_transform=linear,
    )
    z = jnp.array([0.5, -1.0], dtype=float)
    x = space.x_from_z(z, jnp)
    z_roundtrip = space.z_from_x(x, jnp)
    np.testing.assert_allclose(np.asarray(z_roundtrip), np.asarray(z), atol=1e-10)
