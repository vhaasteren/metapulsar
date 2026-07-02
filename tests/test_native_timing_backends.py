"""Tests for native nonlinear timing backend wiring."""

from __future__ import annotations

import numpy as np
import pytest

from metapulsar.metapulsar import MetaPulsar, SessionFiles
from metapulsar.metapulsar_factory import MetaPulsarFactory
from metapulsar.timing.backends.base import LinearModel
from metapulsar.timing.backends.jug import JugTimingBackend
from metapulsar.timing.backends.pint import PintTimingBackend
from metapulsar.timing.backends.tempo2 import Tempo2TimingBackend
from metapulsar.timing.protocols import JaxTimingBackend


class _FakeJaxState:
    def __init__(self, design):
        self.design = np.asarray(design, dtype=float)

    def residual_delta_np(self, delta):
        return self.design @ np.asarray(delta, dtype=float)

    def residual_delta_jax(self, delta):
        import jax.numpy as jnp

        return jnp.asarray(self.design) @ jnp.asarray(delta)


class _FakeDeltaEngine:
    def __init__(self, design):
        self.design = np.asarray(design, dtype=float)
        self.calls = []

    def delta_residuals(self, delta_params):
        self.calls.append(dict(delta_params))
        delta = np.asarray([delta_params["F0"], delta_params["F1"]], dtype=float)
        return self.design @ delta


def _linear_model():
    design = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, -0.5]], dtype=float)
    return LinearModel.from_host(
        fitpars=("F0", "F1"),
        design=design,
        theta_exact={"F0": "10.0", "F1": "-1e-15"},
    )


def test_jug_backend_is_jax_traceable():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    model = _linear_model()
    backend = JugTimingBackend(
        state=_FakeJaxState(model.design),
        linear_model=model,
        precision_critical={"F0"},
    )

    assert isinstance(backend, JaxTimingBackend)
    assert backend.precision_critical_fitpars() == {"F0"}
    np.testing.assert_allclose(backend.design_matrix(), model.design)
    np.testing.assert_allclose(
        backend.residual_delta(np.array([0.2, -0.4])), [0.2, 0.0, 0.4]
    )

    def scalar(delta):
        return jnp.sum(backend.residual_delta_jax(delta) ** 2)

    value = jax.jit(scalar)(jnp.array([0.2, -0.4]))
    grad = jax.grad(scalar)(jnp.array([0.2, -0.4]))
    assert np.isfinite(np.asarray(value))
    assert grad.shape == (2,)


def test_native_pint_and_tempo2_wrappers_use_linear_model_metadata():
    model = _linear_model()
    pint = PintTimingBackend(engine=_FakeDeltaEngine(model.design), linear_model=model)
    tempo2 = Tempo2TimingBackend(
        engine=_FakeDeltaEngine(model.design), linear_model=model
    )

    for backend in (pint, tempo2):
        np.testing.assert_allclose(backend.design_matrix(), model.design)
        assert backend.reference_theta_exact()["F0"] == "10.0"
        np.testing.assert_allclose(
            backend.residual_delta(np.array([0.2, -0.4])),
            np.array([0.2, 0.0, 0.4]),
        )


def test_factory_retains_exact_session_file_bytes(tmp_path):
    par = tmp_path / "loaded.par"
    tim = tmp_path / "loaded.tim"
    par.write_text("F0 123\n", encoding="utf-8")
    tim.write_text("FORMAT 1\n", encoding="utf-8")

    retained = MetaPulsarFactory()._retain_session_files(
        pta_name="ng 5",
        timing_package="pint",
        par_path=par,
        tim_path=tim,
        session_file_dir=tmp_path / "retained",
    )

    assert retained["par_path"].read_text(encoding="utf-8") == "F0 123\n"
    assert retained["tim_path"].read_text(encoding="utf-8") == "FORMAT 1\n"
    assert retained["timing_package"] == "pint"


def test_factory_retains_included_tim_files(tmp_path):
    chunk = tmp_path / "tims" / "chunk.tim"
    chunk.parent.mkdir(parents=True)
    chunk.write_text("FORMAT 1\n obs 1400.0 55000.0 1.0 site\n", encoding="utf-8")
    main = tmp_path / "main.tim"
    main.write_text("FORMAT 1\nINCLUDE tims/chunk.tim\n", encoding="utf-8")
    par = tmp_path / "main.par"
    par.write_text("F0 1\n", encoding="utf-8")

    retained = MetaPulsarFactory()._retain_session_files(
        pta_name="epta",
        timing_package="tempo2",
        par_path=par,
        tim_path=main,
        session_file_dir=tmp_path / "retained",
    )

    included = tmp_path / "retained" / "tims" / "chunk.tim"
    assert included.is_file()
    assert included.read_text(encoding="utf-8") == chunk.read_text(encoding="utf-8")
    assert "INCLUDE tims/chunk.tim" in retained["tim_path"].read_text(encoding="utf-8")


def test_jug_capability_requires_readable_session_files(monkeypatch, tmp_path):
    host = MetaPulsar.__new__(MetaPulsar)
    host._epulsars = {"pta": object()}
    host._clock_dir = None
    monkeypatch.setattr(MetaPulsar, "_can_import_jug", staticmethod(lambda: True))

    host._session_files = {}
    assert not host.has_timing_backend("jug")

    par = tmp_path / "session.par"
    tim = tmp_path / "session.tim"
    par.write_text("F0 1\n", encoding="utf-8")
    tim.write_text("FORMAT 1\n", encoding="utf-8")
    host._session_files = {
        "pta": SessionFiles(par_path=par, tim_path=tim, timing_package="pint")
    }
    assert host.has_timing_backend("jug")
