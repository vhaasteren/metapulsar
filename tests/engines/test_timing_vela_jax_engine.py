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

from metapulsar.engines.vela_jax import TOARows, VelaJaxEngine
from tests.conftest import vela_jax_tempo2_host

FITPARS = ("F0", "F1", "PB", "A1", "DM", "PHOFF")

AEI = pathlib.Path("data/aei-dr2")
PSR = "J1853+1303"


class _FakeVelaJaxEngine:
    """Linear stand-in with vela-jax's attribute surface.

    vela-jax never reorders TOAs (its SPEC R5.5), so there is no permutation
    to model here -- only the row signature and the reference strings the two
    guards compare against the composite's.
    """

    timing_package = "pint"
    binary_conventions = "pint"

    def __init__(self, n=6, rows=None, theta_exact=None):
        self.param_names = FITPARS
        self.param_units = {name: "1" for name in self.param_names}
        rng = np.random.default_rng(3)
        self.response = rng.normal(size=(n, len(self.param_names)))
        self._rows = rows
        self._theta_exact = dict.fromkeys(self.param_names, "1.0")
        self._theta_exact.update(theta_exact or {})
        self._n = n

    def toa_rows(self):
        return self._rows

    def reference_theta_exact(self):
        return dict(self._theta_exact)

    def gauge_direction(self):
        return 1.0 / np.linspace(100.0, 107.0, self._n)

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


def _rows(n=6, *, stoas=None, freqs=None, toaerrs=None):
    return TOARows(
        stoas=np.arange(n, dtype=float) * 86400.0 if stoas is None else stoas,
        freqs=np.tile([1400.0, 1400.0, 820.0], n)[:n] if freqs is None else freqs,
        toaerrs=np.full(n, 1e-6) if toaerrs is None else toaerrs,
    )


def _leg(engine, fitpars=FITPARS, **kwargs):
    return VelaJaxEngine.from_engine(
        engine, linear_model=_linear_model(fitpars), **kwargs
    )


def test_aligned_rows_are_accepted():
    """The factory path passes ``rows=None``; the from-par/tim path passes three
    columns and they match."""
    rows = _rows()
    assert _leg(_FakeVelaJaxEngine(rows=rows), rows=rows).residual_delta(
        np.zeros(len(FITPARS))
    ).shape == (6,)


def test_a_sub_band_swap_is_refused():
    """Two TOAs at the same site arrival, different frequencies, exchanged.

    This is the case the old single-column guard could not see: the site
    arrival times are *identical*, so comparing them alone says the rows
    match while the residuals line up against the wrong sub-bands.
    """
    stoas = np.array([0.0, 0.0, 86400.0, 86400.0, 172800.0, 172800.0])
    freqs = np.array([1400.0, 820.0, 1400.0, 820.0, 1400.0, 820.0])
    swapped = freqs.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    composite = _rows(stoas=stoas, freqs=freqs)
    engine = _FakeVelaJaxEngine(rows=_rows(stoas=stoas, freqs=swapped))
    with pytest.raises(ValueError, match="not the composite's rows"):
        _leg(engine, rows=composite)


def test_a_microsecond_of_site_arrival_jitter_is_accepted():
    """Two codes rounding an MJD differently is not a different row.

    One ulp of an MJD in seconds is ~1e-6 s, so the tolerance has to sit above
    it; a TOA spacing is minutes, so it has plenty of room below.
    """
    rows = _rows()
    jittered = _rows(stoas=rows.stoas + 1e-6)
    assert _leg(_FakeVelaJaxEngine(rows=jittered), rows=rows) is not None


def test_a_dropped_toa_is_refused():
    rows = _rows()
    engine = _FakeVelaJaxEngine(n=5, rows=_rows(5))
    with pytest.raises(ValueError, match="did not reproduce the leg"):
        _leg(engine, fitpars=FITPARS, rows=rows)


def test_an_infinite_frequency_row_matches_itself():
    """``inf`` is a real value on both sides, not a mismatch."""
    freqs = _rows().freqs.copy()
    freqs[2] = np.inf
    rows = _rows(freqs=freqs)
    assert _leg(_FakeVelaJaxEngine(rows=rows), rows=rows) is not None


@pytest.mark.parametrize(
    "value,refused",
    [("1.0000000001", True), ("1.0000000000000001", False)],
)
def test_a_reference_mismatch_is_refused(value, refused):
    """Aligned rows are not enough: the two sides must expand around one par.

    1e-10 relative is a different par; 1e-16 is two codes printing the same
    number. A different *timescale* sits at 1.5e-8, four orders above the
    bound, so TCB-vs-TDB is caught too.
    """
    rows = _rows()
    engine = _FakeVelaJaxEngine(rows=rows, theta_exact={"PB": value})
    if refused:
        with pytest.raises(ValueError, match="reference theta disagrees"):
            _leg(engine, rows=rows)
    else:
        assert _leg(engine, rows=rows) is not None


@pytest.mark.parametrize("value,refused", [("1e-11", True), ("1e-13", False)])
def test_a_zero_reference_uses_an_absolute_bound(value, refused):
    """``PHOFF`` is 0.0 in the composite, so a relative bound would be vacuous.

    The floor applies *only* when a side is an exact zero. Applying it
    everywhere would put the bound above the 1.5e-8 relative shift a wrong
    timescale causes on a small parameter (``EPS1`` ~ 5e-6), which is the
    thing this guard exists to catch.
    """
    rows = _rows()
    model = _linear_model(FITPARS)
    theta = dict(model.theta_exact)
    theta["PHOFF"] = "0.0"
    model = LinearModel.from_design(
        fitpars=model.fitpars, design=model.design, theta_exact=theta
    )
    engine = _FakeVelaJaxEngine(rows=rows, theta_exact={"PHOFF": value})

    def call():
        return VelaJaxEngine.from_engine(engine, linear_model=model, rows=rows)

    if refused:
        with pytest.raises(ValueError, match="reference theta disagrees"):
            call()
    else:
        assert call() is not None


def test_the_gauge_direction_is_the_engines():
    """nltiming's gauge check asks the leg; the leg asks the engine."""
    rows = _rows()
    engine = _FakeVelaJaxEngine(rows=rows)
    assert np.array_equal(
        _leg(engine, rows=rows).gauge_direction(), engine.gauge_direction()
    )


def test_the_leg_publishes_the_engines_rows_untouched():
    """No permutation on the residual path, in either the numpy or JAX form."""
    import jax.numpy as jnp

    fitpars = ("F0", "F1", "PB", "A1", "DM", "PHOFF")
    engine = _FakeVelaJaxEngine()
    leg = VelaJaxEngine.from_engine(engine, linear_model=_linear_model(fitpars))

    delta = np.zeros(len(fitpars))
    delta[fitpars.index("A1")] = 1e-3
    step = np.zeros(len(engine.param_names))
    step[engine.param_names.index("A1")] = 1e-3

    native = engine.residual_delta(step)
    assert np.allclose(leg.residual_delta(delta), native)
    assert np.allclose(np.asarray(leg.residual_delta_jax(jnp.asarray(delta))), native)


def test_native_and_exact_linear_fitpars_are_partitioned():
    """Exact-linear is nltiming's classification (DMX/JUMP/FD/Offset) plus
    anything the engine cannot set -- not the engine's own wider
    "identically linear" set, which still goes down the native path."""
    engine = _engine(("F0", "PB", "DM", "JUMP3", "PHOFF"))
    assert set(engine.exact_linear_fitpars()) == {"JUMP3"}
    assert engine._engine_fitpars == ("F0", "PB", "DM", "PHOFF")
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
    assert set(engine._engine_fitpars) == {"PB", "A1"}
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
    if package == "tempo2" and not vela_jax_tempo2_host():
        pytest.skip("needs libstempo with the siteVel/clock properties")

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
        engines="vela_jax",
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
