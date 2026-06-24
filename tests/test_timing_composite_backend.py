"""Slice-3 tests for composite timing backend behavior."""

import numpy as np
import pytest

from metapulsar.timing.backends.base import LinearModel
from metapulsar.timing.backends.composite import (
    BackendSession,
    CompositeJaxTimingBackend,
    CompositeTimingBackend,
    build_composite_backend,
)
from metapulsar.timing.backends import build_backend
from metapulsar.timing.backends.jug import LinearizedJugTimingBackend
from metapulsar.timing.backends.pint import LinearizedPintTimingBackend


def _session_backends():
    # Host canonical rows are unsorted by time on purpose.
    # session_a rows -> [2, 0], session_b rows -> [3, 1]
    a = LinearizedPintTimingBackend.from_linear_model(
        LinearModel.from_host(
            fitpars=("F0", "A1"),
            design=np.array([[1.0, 10.0], [1.0, 11.0]], dtype=float),
            theta_exact={"F0": "10.0", "A1": "1.0"},
        )
    )
    b = LinearizedJugTimingBackend.from_linear_model(
        LinearModel.from_host(
            fitpars=("F0", "PB"),
            design=np.array([[1.0, 20.0], [1.0, 21.0]], dtype=float),
            theta_exact={"F0": "10.0", "PB": "5.0"},
        ),
        precision_critical=frozenset({"F0"}),
    )
    return [
        BackendSession(name="pta_a", row_indices=np.array([2, 0]), backend=a),
        BackendSession(name="pta_b", row_indices=np.array([3, 1]), backend=b),
    ]


def test_composite_residual_and_design_scatter_in_canonical_rows():
    sessions = _session_backends()
    backend = build_composite_backend(
        fitpars=("F0", "A1", "PB"),
        nrows=4,
        sessions=sessions,
        missing_param_policy="linear_fallback",
    )
    assert isinstance(backend, CompositeTimingBackend)
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
    bad_backend = LinearizedJugTimingBackend.from_linear_model(
        LinearModel.from_host(
            fitpars=("F0",),
            design=np.array([[1.0], [1.0]], dtype=float),
            theta_exact={"F0": "11.0"},
        )
    )
    bad_sessions[1] = BackendSession(
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
    assert not isinstance(backend, CompositeJaxTimingBackend)

    only_jax_sessions = [
        BackendSession(
            name="s1",
            row_indices=np.array([0, 1]),
            backend=LinearizedJugTimingBackend.from_linear_model(
                LinearModel.from_host(
                    fitpars=("F0",),
                    design=np.array([[1.0], [1.0]], dtype=float),
                    theta_exact={"F0": "1.0"},
                ),
                precision_critical=frozenset({"F0"}),
            ),
        ),
        BackendSession(
            name="s2",
            row_indices=np.array([2, 3]),
            backend=LinearizedJugTimingBackend.from_linear_model(
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
    assert isinstance(jax_backend, CompositeJaxTimingBackend)
    assert jax_backend.precision_critical_fitpars() == frozenset({"F0", "PB"})


def test_absent_session_params_contribute_zero_even_in_strict_mode():
    sessions = _session_backends()
    delta = np.array([0.0, 3.0, 0.0], dtype=float)  # A1 absent from session_b

    strict_backend = build_composite_backend(
        fitpars=("F0", "A1", "PB"),
        nrows=4,
        sessions=sessions,
        missing_param_policy="strict",
    )
    out = strict_backend.residual_delta(delta)
    # session_b does not own A1, so A1 contributes exactly zero on its rows.
    np.testing.assert_allclose(out[[3, 1]], np.array([0.0, 0.0]))


def test_mapped_but_unevaluable_param_uses_linear_fallback_or_strict_error():
    session = BackendSession(
        name="pta_a",
        row_indices=np.array([0, 1]),
        backend=LinearizedPintTimingBackend.from_linear_model(
            LinearModel.from_host(
                fitpars=("F0",),
                design=np.array([[1.0], [1.0]], dtype=float),
                theta_exact={"F0": "1.0"},
            )
        ),
        linear_fallback_fitpars=frozenset({"A1"}),
        fallback_reference_exact={"A1": "2.0"},
    )
    host_design = np.array([[1.0, 10.0], [1.0, 11.0]], dtype=float)
    delta = np.array([0.0, 3.0], dtype=float)

    fallback_backend = build_composite_backend(
        fitpars=("F0", "A1"),
        nrows=2,
        sessions=[session],
        missing_param_policy="linear_fallback",
        host_design=host_design,
    )
    np.testing.assert_allclose(fallback_backend.residual_delta(delta), [30.0, 33.0])
    np.testing.assert_allclose(fallback_backend.design_matrix()[:, 1], [10.0, 11.0])

    strict_backend = build_composite_backend(
        fitpars=("F0", "A1"),
        nrows=2,
        sessions=[session],
        missing_param_policy="strict",
        host_design=host_design,
    )
    with pytest.raises(ValueError, match="linear fallback"):
        strict_backend.residual_delta(delta)


def test_build_backend_rejects_sessions_that_cannot_honor_requested_family():
    sessions = _session_backends()
    with pytest.raises(ValueError, match="cannot honor backend 'jug'"):
        build_backend(
            name="jug", fitpars=("F0", "A1", "PB"), nrows=4, sessions=sessions
        )
