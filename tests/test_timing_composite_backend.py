"""Slice-3 tests for composite timing backend behavior."""

import numpy as np
import pytest

from metapulsar.timing.backends.base import LinearModel
from metapulsar.timing.backends.composite import (
    PulsarSession,
    PulsarJaxTimingBackend,
    PulsarTimingBackend,
    build_composite_backend,
)
from metapulsar.timing.backends import build_backend
from metapulsar.timing.backends.jug import LinearizedJugEngine
from metapulsar.timing.backends.pint import LinearizedPintEngine


def _session_backends():
    # Host canonical rows are unsorted by time on purpose.
    # session_a rows -> [2, 0], session_b rows -> [3, 1]
    a = LinearizedPintEngine.from_linear_model(
        LinearModel.from_host(
            fitpars=("F0", "A1"),
            design=np.array([[1.0, 10.0], [1.0, 11.0]], dtype=float),
            theta_exact={"F0": "10.0", "A1": "1.0"},
        )
    )
    b = LinearizedJugEngine.from_linear_model(
        LinearModel.from_host(
            fitpars=("F0", "PB"),
            design=np.array([[1.0, 20.0], [1.0, 21.0]], dtype=float),
            theta_exact={"F0": "10.0", "PB": "5.0"},
        ),
        precision_critical=frozenset({"F0"}),
    )
    return [
        PulsarSession(name="pta_a", row_indices=np.array([2, 0]), backend=a),
        PulsarSession(name="pta_b", row_indices=np.array([3, 1]), backend=b),
    ]


def test_composite_residual_and_design_scatter_in_canonical_rows():
    sessions = _session_backends()
    backend = build_composite_backend(
        fitpars=("F0", "A1", "PB"),
        nrows=4,
        sessions=sessions,
    )
    assert isinstance(backend, PulsarTimingBackend)
    delta = np.array([0.5, 2.0, -1.0], dtype=float)

    residual = backend.residual_delta(delta)
    expected = np.zeros(4, dtype=float)
    expected[[2, 0]] = np.array([[1.0, 10.0], [1.0, 11.0]]) @ np.array([0.5, 2.0])
    expected[[3, 1]] = np.array([[1.0, 20.0], [1.0, 21.0]]) @ np.array([0.5, -1.0])
    np.testing.assert_allclose(residual, expected)

    design = backend.design_matrix()
    assert design.shape == (4, 3)
    np.testing.assert_allclose(
        design[[2, 0], :2],
        np.array([[1.0, 10.0], [1.0, 11.0]], dtype=float),
    )
    np.testing.assert_allclose(
        design[[3, 1], ::2],
        np.array([[1.0, 20.0], [1.0, 21.0]], dtype=float),
    )


def test_composite_reference_theta_exact_validates_shared_values():
    sessions = _session_backends()
    backend = build_composite_backend(
        fitpars=("F0", "A1", "PB"),
        nrows=4,
        sessions=sessions,
    )
    ref = backend.reference_theta_exact()
    assert ref["F0"] == "10.0"
    assert ref["A1"] == "1.0"
    assert ref["PB"] == "5.0"

    bad_sessions = _session_backends()
    bad_backend = LinearizedJugEngine.from_linear_model(
        LinearModel.from_host(
            fitpars=("F0",),
            design=np.array([[1.0], [1.0]], dtype=float),
            theta_exact={"F0": "11.0"},
        )
    )
    bad_sessions[1] = PulsarSession(
        name="pta_b",
        row_indices=np.array([3, 1]),
        backend=bad_backend,
    )
    with pytest.raises(ValueError, match="disagrees"):
        build_composite_backend(
            fitpars=("F0", "A1", "PB"),
            nrows=4,
            sessions=bad_sessions,
        )


def test_composite_jax_capability_requires_all_sessions_and_unions_precision():
    sessions = _session_backends()
    backend = build_composite_backend(
        fitpars=("F0", "A1", "PB"),
        nrows=4,
        sessions=sessions,
    )
    assert not isinstance(backend, PulsarJaxTimingBackend)

    only_jax_sessions = [
        PulsarSession(
            name="s1",
            row_indices=np.array([0, 1]),
            backend=LinearizedJugEngine.from_linear_model(
                LinearModel.from_host(
                    fitpars=("F0",),
                    design=np.array([[1.0], [1.0]], dtype=float),
                    theta_exact={"F0": "1.0"},
                ),
                precision_critical=frozenset({"F0"}),
            ),
        ),
        PulsarSession(
            name="s2",
            row_indices=np.array([2, 3]),
            backend=LinearizedJugEngine.from_linear_model(
                LinearModel.from_host(
                    fitpars=("PB",),
                    design=np.array([[2.0], [3.0]], dtype=float),
                    theta_exact={"PB": "2.0"},
                ),
                precision_critical=frozenset({"PB"}),
            ),
        ),
    ]
    jax_backend = build_composite_backend(
        fitpars=("F0", "PB"),
        nrows=4,
        sessions=only_jax_sessions,
    )
    assert isinstance(jax_backend, PulsarJaxTimingBackend)
    assert jax_backend.precision_critical_fitpars() == frozenset({"F0", "PB"})


def test_absent_session_params_contribute_zero():
    sessions = _session_backends()
    delta = np.array([0.0, 3.0, 0.0], dtype=float)  # A1 absent from session_b

    backend = build_composite_backend(
        fitpars=("F0", "A1", "PB"),
        nrows=4,
        sessions=sessions,
    )
    out = backend.residual_delta(delta)
    # session_b does not own A1, so A1 contributes exactly zero on its rows.
    np.testing.assert_allclose(out[[3, 1]], np.array([0.0, 0.0]))


def test_mapped_but_unevaluable_param_uses_exact_linear_design_column():
    session = PulsarSession(
        name="pta_a",
        row_indices=np.array([0, 1]),
        backend=LinearizedPintEngine.from_linear_model(
            LinearModel.from_host(
                fitpars=("F0",),
                design=np.array([[1.0], [1.0]], dtype=float),
                theta_exact={"F0": "1.0"},
            )
        ),
        exact_linear_fitpars=frozenset({"A1"}),
        fallback_reference_exact={"A1": "2.0"},
    )
    host_design = np.array([[1.0, 10.0], [1.0, 11.0]], dtype=float)
    delta = np.array([0.0, 3.0], dtype=float)

    exact_backend = build_composite_backend(
        fitpars=("F0", "A1"),
        nrows=2,
        sessions=[session],
        host_design=host_design,
    )
    np.testing.assert_allclose(exact_backend.residual_delta(delta), [30.0, 33.0])
    np.testing.assert_allclose(exact_backend.design_matrix()[:, 1], [10.0, 11.0])


def test_build_backend_accepts_mixed_engine_families():
    sessions = _session_backends()
    backend = build_backend(fitpars=("F0", "A1", "PB"), nrows=4, sessions=sessions)
    assert isinstance(backend, PulsarTimingBackend)
