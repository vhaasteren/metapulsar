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


# Per-parameter physical perturbations kept well inside the no-pulse-wrap regime
# (and inside the posterior bulk) where the precision-safe JAX phase-change delta
# must reproduce the host longdouble recompute. Units are each parameter's JUG
# native physical unit (radians for RAJ/DECJ, mas/yr for PMRA/PMDEC, days for
# PB/T0, light-seconds for A1, degrees for OM, dimensionless for ECC, solar masses
# for M2). OM is intentionally small: it is so tightly constrained that larger
# excursions push the host into integer pulse-number wraps (a non-smooth branch
# the autodiff JAX path deliberately does not follow).
_PARITY_STEPS = {
    "RAJ": 1.0e-9,
    "DECJ": 1.0e-9,
    "PMRA": 1.0e-2,
    "PMDEC": 1.0e-2,
    "PB": 1.0e-6,
    "T0": 1.0e-4,
    "A1": 1.0e-6,
    "OM": 1.0e-4,
    "ECC": 1.0e-8,
    "M2": 1.0e-3,
}


def test_jug_jax_engine_astrometry_binary_parity_matches_host(ng5_paths):
    """JAX residual delta must match the host recompute for every fitted
    astrometry/binary parameter (the set that regressed historically).

    The nonlinear timing likelihood marginalizes the phase Offset, so the
    convention that must agree is the residual shape after removing the constant
    mode. Each parameter is perturbed alone inside the validated no-wrap regime.
    """
    par, tim = ng5_paths
    pytest.importorskip("jug")

    from jug.engine.session import TimingSession

    from metapulsar.nonlinear_timing_model import JugJaxTimingEngine

    session = TimingSession(par, tim, compatibility="tempo2", verbose=False)
    session.compute_residuals(subtract_tzr=True)

    fitpars = list(_PARITY_STEPS)
    space = SampledTimingParameterSpace.from_transform_registry(
        names=fitpars,
        scale={name: 1.0 for name in fitpars},
    )
    engine = JugJaxTimingEngine.from_session(
        session,
        sampled_parameter_space=space,
        fitpars=fitpars,
        compatibility="tempo2",
    )

    z0 = np.zeros(len(fitpars), dtype=float)
    np.testing.assert_allclose(
        np.asarray(engine.residual_delta_jax(z0), dtype=float),
        0.0,
        atol=1.0e-10,
    )

    for i, name in enumerate(fitpars):
        z = np.zeros(len(fitpars), dtype=float)
        z[i] = _PARITY_STEPS[name]
        dj = np.asarray(engine.residual_delta_jax(z), dtype=float)
        dh = np.asarray(engine.residual_delta_np(z), dtype=float)
        dj_mr = dj - dj.mean()
        dh_mr = dh - dh.mean()
        host_norm = np.linalg.norm(dh_mr)
        assert host_norm > 0.0, f"{name}: host produced a null response"
        corr = float(dj_mr @ dh_mr / (host_norm * np.linalg.norm(dj_mr)))
        rel = np.linalg.norm(dj_mr - dh_mr) / host_norm
        assert corr > 1.0 - 1.0e-6, f"{name}: corr={corr:.8f}"
        assert rel < 1.0e-4, f"{name}: rel={rel:.3e}"


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


def test_jug_jax_linearized_is_tangent_of_nonlinear(ng5_paths):
    """The linearized timing path must be the tangent of the nonlinear path.

    The linearized residual uses the timing design matrix (partials at z=0) built
    by adaptive finite differences of the host nonlinear residual -- the same
    model that drives the nonlinear curve and Discovery's marginalized lnL. This
    is the design-matrix route (no autodiff) that also serves non-JAX backends.

    The check is run end-to-end through the engine's standardized ``z`` interface
    so it gates the full chain (parameter ordering, units, the standardization
    transform and its Jacobian, and the isort alignment) the same way the
    likelihood does: for each fitted parameter the linearized residual delta must
    equal the directional derivative of the nonlinear residual delta at z=0.
    """
    par, tim = ng5_paths
    pytest.importorskip("jug")

    from jug.engine.session import TimingSession

    session = TimingSession(par, tim, compatibility="tempo2", verbose=False)
    session.compute_residuals(subtract_tzr=True)

    fitpars = ["RAJ", "DECJ", "PMRA", "PMDEC", "PB", "T0", "A1", "OM", "ECC", "M2"]
    space = SampledTimingParameterSpace.from_transform_registry(
        names=fitpars,
        scale={name: 1.0 for name in fitpars},
    )
    nonlinear = JugJaxTimingEngine.from_session(
        session,
        sampled_parameter_space=space,
        fitpars=fitpars,
        compatibility="tempo2",
        evaluation_mode="nonlinear",
    )
    linear = JugJaxTimingEngine(
        jax_state=nonlinear._state,
        parameter_space=nonlinear._parameter_space,
        fitpars=nonlinear.fitpars,
        param_mapping=nonlinear._param_mapping,
        isort=nonlinear._isort,
        evaluation_mode="linearized",
        linearized_context=nonlinear.linearized_context,
    )

    n = len(fitpars)
    for i in range(n):
        z_dir = np.zeros(n, dtype=float)
        z_dir[i] = 1.0
        lin = np.asarray(linear.residual_delta_np(z_dir), dtype=float)

        # Directional derivative of the nonlinear path at z=0, with a step chosen
        # (Richardson) to avoid both round-off and nonlinearity for this
        # parameter's scale/conditioning.
        best = None
        best_score = np.inf
        prev = None
        for eps in 10.0 ** (-np.arange(2, 13, dtype=float)):
            fd = (
                np.asarray(nonlinear.residual_delta_np(z_dir * eps), dtype=float)
                - np.asarray(nonlinear.residual_delta_np(-z_dir * eps), dtype=float)
            ) / (2.0 * eps)
            if prev is not None:
                denom = np.linalg.norm(fd) + np.linalg.norm(prev)
                score = (
                    0.0 if denom == 0.0 else float(np.linalg.norm(fd - prev) / denom)
                )
                if np.isfinite(score) and score < best_score:
                    best_score, best = score, fd
            prev = fd

        norm_lin = np.linalg.norm(lin)
        norm_fd = np.linalg.norm(best)
        corr = float(lin @ best / (norm_lin * norm_fd + 1e-300))
        ratio = norm_fd / (norm_lin + 1e-300)
        assert (
            corr > 0.999
        ), f"{fitpars[i]}: linearized not aligned with tangent (corr={corr:.5f})"
        assert (
            0.99 < ratio < 1.01
        ), f"{fitpars[i]}: tangent magnitude mismatch (ratio={ratio:.5f})"
