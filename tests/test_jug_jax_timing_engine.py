"""Tests for JugJaxTimingEngine and MetaPulsar jug_jax_state export."""

from pathlib import Path

import numpy as np
import pytest

from metapulsar.nonlinear_timing_model import (
    JugDeltaEngine,
    JugJaxTimingEngine,
    SampledTimingParameterSpace,
)

DATA_ROOT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "notebooks-dev"
    / "data"
    / "nonlinear-tests"
)


@pytest.fixture(scope="module")
def ng5_paths():
    par = DATA_ROOT / "ng5" / "J1640+2224" / "J1640+2224_NANOGrav_dfg+12.par"
    tim = DATA_ROOT / "ng5" / "J1640+2224" / "J1640+2224_NANOGrav_dfg+12.tim"
    if not par.is_file() or not tim.is_file():
        pytest.skip("ng5 nonlinear test data unavailable")
    return par, tim


def test_jug_jax_engine_zero_delta_matches_host(ng5_paths):
    par, tim = ng5_paths
    pytest.importorskip("jug")
    from jug.engine.session import TimingSession

    session = TimingSession(par, tim, compatibility="tempo2", verbose=False)
    session.compute_residuals(subtract_tzr=True)

    fitpars = ["F0", "F1", "DM", "RAJ", "DECJ"]
    sampled = ["F0", "DM"]
    space = SampledTimingParameterSpace.from_transform_registry(
        names=sampled,
        scale={name: 1.0 for name in sampled},
    )
    host = JugDeltaEngine(session, fitpars=fitpars)
    jax_engine = JugJaxTimingEngine.from_session(
        session,
        sampled_parameter_space=space,
        fitpars=fitpars,
        compatibility="tempo2",
    )

    z0 = np.zeros(len(sampled), dtype=float)
    host_delta = host.delta_residuals({"F0": 0.0, "DM": 0.0})
    np.testing.assert_allclose(jax_engine.residual_delta_np(z0), host_delta, atol=1e-8)
    np.testing.assert_allclose(host_delta, np.zeros_like(host_delta), atol=1e-8)


def test_jug_jax_engine_eager_matches_host_with_isort_nonzero_delta(ng5_paths):
    par, tim = ng5_paths
    pytest.importorskip("jug")

    from jug.engine.session import TimingSession

    session = TimingSession(par, tim, compatibility="tempo2", verbose=False)
    result = session.compute_residuals(subtract_tzr=True)
    residuals = result.get("residuals_sec", result.get("residuals_us"))
    isort = np.arange(len(residuals) - 1, -1, -1)
    fitpars = ["F0", "DM"]
    sampled = ["F0", "DM"]
    space = SampledTimingParameterSpace.from_transform_registry(
        names=sampled,
        scale={"F0": 1.0e-13, "DM": 1.0e-5},
    )
    engine = JugJaxTimingEngine.from_session(
        session,
        sampled_parameter_space=space,
        fitpars=fitpars,
        compatibility="tempo2",
        isort=isort,
    )
    host = JugDeltaEngine(session, fitpars=fitpars, isort=isort)

    z = np.array([0.7, -0.25], dtype=float)
    eager = engine.residual_delta_np(z)
    host_delta = host.delta_residuals(space.delta_dict_from_z_np(z))

    # JUG's full residual paths can differ by a constant phase/mean convention.
    # The nonlinear timing likelihood marginalizes Offset, so the shape after
    # removing the constant mode is the convention that must agree.
    np.testing.assert_allclose(
        eager - np.mean(eager),
        host_delta - np.mean(host_delta),
        atol=1.0e-10,
        rtol=1.0e-8,
    )
