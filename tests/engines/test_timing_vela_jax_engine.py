"""Tests for the vela-jax leg engine.

The unit half runs against a linear stand-in for ``vela_jax.Engine``, so the
partition and the JAX path are exercised without pyvela, tempo2 or JAX-heavy
fixtures. The integration half needs the real package and the AEI-DR2 tree.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from nltiming.engine_support import LinearModel

from metapulsar.engines.vela_jax import VelaJaxEngine

AEI = pathlib.Path("data/aei-dr2")
PSR = "J1853+1303"


class _FakeVelaJaxEngine:
    """Linear stand-in with vela-jax's attribute surface."""

    host = "pint"
    binary_conventions = "pint"

    def __init__(self, n=6):
        self.param_names = ("F0", "F1", "PB", "A1", "DM", "PHOFF")
        self.param_units = {name: "1" for name in self.param_names}
        rng = np.random.default_rng(3)
        self.response = rng.normal(size=(n, len(self.param_names)))

    def residual_delta(self, delta):
        return self.response @ np.asarray(delta, dtype=float)

    def residual_delta_jax(self, delta):
        import jax.numpy as jnp

        return jnp.asarray(self.response, dtype=delta.dtype) @ delta

    def design_matrix(self):
        return -self.response

    def identically_linear_params(self):
        return frozenset({"DM", "PHOFF"})

    def precision_critical_params(self):
        return frozenset({"F0", "PB"})

    def binary_chart_facts(self):
        return None


def _linear_model(fitpars, n=6, seed=11):
    rng = np.random.default_rng(seed)
    return LinearModel.from_design(
        fitpars=tuple(fitpars),
        design=rng.normal(size=(n, len(fitpars))),
        theta_exact={name: "1.0" for name in fitpars},
    )


def _engine(fitpars=("F0", "F1", "PB", "A1", "DM", "PHOFF"), **kwargs):
    return VelaJaxEngine.from_engine(
        _FakeVelaJaxEngine(), linear_model=_linear_model(fitpars), **kwargs
    )


def test_native_and_exact_linear_fitpars_are_partitioned():
    """Exact-linear is nltiming's classification (DMX/JUMP/FD/Offset) plus
    anything the engine cannot set -- not the engine's own wider
    "identically linear" set, which still goes down the native path."""
    engine = _engine(("F0", "PB", "DM", "JUMP3", "PHOFF"))
    assert set(engine.exact_linear_fitpars()) == {"JUMP3"}
    assert engine._native_fitpars == ("F0", "PB", "DM", "PHOFF")
    # ...but the engine still reports DM/PHOFF as affine, for the linearity layer.
    assert {"DM", "PHOFF", "JUMP3"} <= engine.identically_linear_fitpars()


def test_zero_delta_is_exactly_zero():
    engine = _engine()
    zero = np.zeros(len(engine.fitpars))
    assert np.array_equal(engine.residual_delta(zero), np.zeros(6))


def test_host_fitpars_are_scattered_onto_the_engines_own_order():
    """The leg's fitpars are a different list, in a different order, and may
    carry names the engine has never heard of."""
    fitpars = ("PB", "F0", "JUMP1")
    engine = _engine(fitpars)
    delta = np.array([2.0, 3.0, 5.0])

    fake = engine._engine
    step = np.zeros(len(fake.param_names))
    step[fake.param_names.index("PB")] = 2.0
    step[fake.param_names.index("F0")] = 3.0
    expected = fake.residual_delta(step) - engine.design_matrix()[:, 2] * 5.0
    np.testing.assert_allclose(engine.residual_delta(delta), expected)


def test_the_jax_path_agrees_with_the_numpy_path():
    import jax
    import jax.numpy as jnp

    if not jax.config.jax_enable_x64:
        pytest.skip("vela-jax requires JAX float64; run with JAX_ENABLE_X64=1")

    engine = _engine(("PB", "F0", "JUMP1"))
    rng = np.random.default_rng(0)
    for _ in range(4):
        delta = rng.normal(size=len(engine.fitpars))
        np.testing.assert_allclose(
            np.asarray(engine.residual_delta_jax(jnp.asarray(delta))),
            engine.residual_delta(delta),
            atol=1e-14,
        )


def test_the_jacobian_is_autodiff_of_that_same_path():
    import jax

    if not jax.config.jax_enable_x64:
        pytest.skip("vela-jax requires JAX float64; run with JAX_ENABLE_X64=1")

    engine = _engine(("PB", "F0", "JUMP1"))
    jacobian = engine.residual_jacobian()
    for column in range(len(engine.fitpars)):
        step = np.zeros(len(engine.fitpars))
        step[column] = 1.0
        np.testing.assert_allclose(
            jacobian[:, column], engine.residual_delta(step), atol=1e-12
        )


def test_a_hybrid_mode_keeps_only_the_binary_axes_native():
    engine = _engine(nonlinear_params="binary")
    assert engine.nonlinear_params == "binary"
    assert set(engine._native_fitpars) == {"PB", "A1"}
    assert {"F0", "F1", "DM", "PHOFF"} <= set(engine.exact_linear_fitpars())


def test_metadata_is_reported_in_host_fitpar_names():
    engine = _engine()
    assert engine.precision_critical_fitpars() == frozenset({"F0", "PB"})
    assert {"DM", "PHOFF"} <= engine.identically_linear_fitpars()
    assert engine.gauge_provenance().export == "none"
    assert engine.gauge_applied is False


# --- integration -----------------------------------------------------------

pytest_integration = pytest.mark.skipif(
    not (AEI / "epta_dr1_v2_2" / "par" / f"{PSR}.par").exists(),
    reason="AEI-DR2 tree not present",
)


@pytest.mark.requires_vela_jax
@pytest.mark.real_data
@pytest_integration
@pytest.mark.parametrize(
    "release,package", [("epta_dr1_v2_2", "tempo2"), ("nanograv_9y", "pint")]
)
def test_a_real_leg_builds_and_is_differentiable(release, package):
    """One impl token, either host: tempo2 reads a tempo2 leg, PINT a PINT leg."""
    pytest.importorskip("vela_jax")
    if package == "tempo2":
        pytest.importorskip("pytempo")

    from metapulsar import create_metapulsar

    base = AEI / release
    pulsar = create_metapulsar(
        {
            release: [
                {
                    "par": base / "par" / f"{PSR}.par",
                    "tim": base / "tim" / f"{PSR}.tim",
                    "timing_package": package,
                }
            ]
        },
        combination_strategy="per_pta",
        use_pulse_numbers="reuse",
    )
    assert pulsar.can_use_engines("vela_jax")
    engine = pulsar.timing_engine("vela_jax")

    zero = np.zeros(len(engine.fitpars))
    assert np.max(np.abs(engine.residual_delta(zero))) == 0.0
    assert hasattr(engine, "residual_delta_jax")

    jacobian = engine.residual_jacobian()
    assert jacobian.shape == (len(pulsar.toas), len(engine.fitpars))
    assert np.all(np.isfinite(jacobian))

    # J ~ -M on the identically-linear columns, on the rows that column touches.
    design = engine.design_matrix()
    for name in sorted(engine.identically_linear_fitpars())[:4]:
        column = engine.fitpars.index(name)
        rows = np.abs(design[:, column]) > 0
        if not rows.any():
            continue
        difference = jacobian[rows, column] + design[rows, column]
        scale = np.max(np.abs(design[rows, column]))
        assert np.max(np.abs(difference - difference.mean())) / scale < 1e-3, name
