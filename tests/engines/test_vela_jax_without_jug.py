"""A vela-jax leg must not need JUG installed (review finding B2).

JUG is the package vela-jax exists to replace for the sampling use case, and
until `nltiming.hybrid` existed, running a vela-jax leg through
``MetaPulsar.timing_engine`` imported it twice over: once unconditionally for
``resolve_tempo2_jug_options``, and once per fitpar to ask JUG's registry what
a binary parameter is.

The gate is the **import seam**, not a real ingest -- ``write_mock_pta_files``
does not produce a par/tim vela-jax could read -- so JUG is made unimportable
and the vela-jax ``Engine`` is replaced by the linear stand-in. What is under
test is that nothing on the path reaches for JUG.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from metapulsar.engines.vela_jax import TOARows
from metapulsar.mockpulsar import create_mock_libstempo

JUG_MODULES = (
    "jug",
    "jug.timing",
    "jug.fitting",
    "jug.fitting.nonlinear_params",
    "jug.model",
    "jug.model.parameter_spec",
)


@pytest.fixture
def no_jug(pulsar, monkeypatch):
    """Make every JUG module unimportable, including one already imported.

    Takes ``pulsar`` so the fixture is built *first*: poisoning ``sys.modules``
    before construction breaks the sandboxed tempo2 worker the mock pulsar's
    materializer starts, which would fail this test for a reason that has
    nothing to do with the seam under test.
    """
    for name in list(sys.modules):
        if name == "jug" or name.startswith("jug."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name in JUG_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError):
        import jug  # noqa: F401
    return pulsar


@pytest.fixture
def pulsar(mock_metapulsar):
    pulsars = {
        pta: create_mock_libstempo(n_toas=24, name="J1857+0943", telescope=pta, seed=i)
        for i, pta in enumerate(("pta_a",), start=1)
    }
    return mock_metapulsar(pulsars, combination_strategy="per_pta")


def _stand_in(pulsar, monkeypatch):
    """Give the pulsar a vela-jax leg whose engine is a linear stand-in.

    Since PR-4 a vela-jax leg must have been *built* as one -- its record and
    its engine come from one read -- so the seam under test is reached by
    injecting a leg rather than by patching the from-par/tim path. The stand-in
    is still a stand-in: ``write_mock_pta_files`` does not produce a par/tim
    vela-jax could read, and what is being tested is the import graph.
    """

    class _Engine:
        timing_package = "pint"
        binary_conventions = "pint"

        def __init__(self, fitpars, response, theta_exact):
            self.param_names = tuple(fitpars)
            self.param_units = {name: "1" for name in self.param_names}
            # A real leg's record *is* its engine's `-J`; the composite guard
            # checks exactly that, so the stand-in has to be the leg's block.
            self.response = np.asarray(response, dtype=float)
            # A real engine reads the same retained par, so it expands around
            # the same reference values; the guard checks exactly that.
            self._theta_exact = dict(theta_exact)

        def residual_delta(self, delta):
            return self.response @ np.asarray(delta, dtype=float)

        def residual_delta_jax(self, delta):
            import jax.numpy as jnp

            return jnp.asarray(self.response, dtype=delta.dtype) @ delta

        def design_matrix(self):
            return -self.response

        def identically_linear_params(self):
            return frozenset()

        def precision_critical_params(self):
            return frozenset()

        def binary_chart_facts(self):
            return None

        def toa_rows(self):
            return TOARows(
                stoas=np.asarray(pulsar._stoas, dtype=float),
                freqs=np.asarray(pulsar._ssbfreqs, dtype=float),
                toaerrs=np.asarray(pulsar._toaerrs, dtype=float),
            )

        def design_matrix_full(self):
            return -self.response

        def reference_theta_exact(self):
            return dict(self._theta_exact)

        def gauge_direction(self):
            return np.ones(len(pulsar._stoas))

        @property
        def pint_model(self):
            return None

    from types import SimpleNamespace

    slices = pulsar._get_pta_slices()
    index = {name: i for i, name in enumerate(pulsar.fitpars)}
    for pta, record in pulsar._pta_data.items():
        local = [
            name
            for name in pulsar.fitpars
            if pta in pulsar._fitparameters.get(name, {})
        ]
        names = [pulsar._native_param(name, pta).pint_name for name in local]
        slc = slices[pta]
        block = pulsar.Mmat[slc, :][:, [index[name] for name in local]]
        theta = {
            pulsar._native_param(name, pta).pint_name: value
            for name, value in pulsar._pta_theta_exact(
                pta,
                tuple(
                    n for n in pulsar.fitpars if pta in pulsar._fitparameters.get(n, {})
                ),
            ).items()
        }
        engine = _Engine(names, -block, theta)
        record._leg = SimpleNamespace(
            engine=engine,
            engine_name="vela_jax",
            timing_package=record.timing_package,
            record=SimpleNamespace(state_id="stand-in"),
        )


@pytest.mark.parametrize("mode", [None, "binary"])
def test_a_vela_jax_leg_needs_no_jug(pulsar, monkeypatch, no_jug, mode):
    """The whole point: the import seam, with JUG made unimportable."""
    _stand_in(pulsar, monkeypatch)
    engine = pulsar.timing_engine("vela_jax", nonlinear_params=mode)
    assert engine.nonlinear_params == mode
    zero = np.zeros(len(engine.fitpars))
    assert np.max(np.abs(engine.residual_delta(zero))) == 0.0


def test_the_hybrid_partition_is_nltimings_registry(pulsar, monkeypatch, no_jug):
    """Under ``"binary"`` the engine keeps exactly the binary axes -- decided by
    ``nltiming.hybrid``, which is now the only registry in the path."""
    from nltiming.hybrid import is_binary_axis

    _stand_in(pulsar, monkeypatch)
    engine = pulsar.timing_engine("vela_jax", nonlinear_params="binary")
    leg = engine._contributions[0].engine
    expected = {
        name
        for name in leg.fitpars
        if is_binary_axis(leg._param_mapping.get(name, name))
    }
    assert set(leg._engine_fitpars) == expected


def test_a_jug_knob_without_a_jug_leg_is_refused(pulsar, monkeypatch, no_jug):
    """``tempo2_jug_options`` used to be resolved unconditionally, which is how
    the unconditional import got there. Now it says what it is."""
    _stand_in(pulsar, monkeypatch)
    with pytest.raises(ValueError, match="JUG knob"):
        pulsar.timing_engine(
            "vela_jax", tempo2_jug_options={"force_cache_refresh": True}
        )
