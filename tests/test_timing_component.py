"""Slice-4 adapter tests for nonlinear timing frontends."""

import numpy as np
import pytest

from metapulsar.timing.backends.jug import LinearizedJugTimingBackend
from metapulsar.timing.backends.pint import LinearizedPintTimingBackend
from metapulsar.timing.backends.base import LinearModel
from metapulsar.timing.frontends.discovery import discovery_signals
from metapulsar.timing.frontends.enterprise import enterprise_signal
from metapulsar.timing.partition import PartitionResult
from metapulsar.timing.space import ParameterSpace


class _FrontendHost:
    def __init__(self) -> None:
        self.name = "J0000+0000"
        self.fitpars = ("F0", "F1", "DM")
        self._toas = np.linspace(0.0, 1.0, 6)
        self._residuals = np.zeros(6, dtype=float)
        self._toaerrs = np.full(6, 1.0e-6, dtype=float)
        self._freqs = np.full(6, 1400.0, dtype=float)
        self._flags = {"pta": np.array(["demo"] * 6, dtype="U8")}
        self._backend_flags = np.array(["demo"] * 6, dtype="U8")
        self._design = np.array(
            [
                [1.0, 0.0, 0.2],
                [1.0, 0.5, 0.1],
                [1.0, 1.0, -0.1],
                [1.0, -0.2, 0.4],
                [1.0, 0.1, -0.3],
                [1.0, -0.5, 0.2],
            ],
            dtype=float,
        )
        model = LinearModel.from_host(
            fitpars=self.fitpars,
            design=self._design,
            theta_exact={name: "0.0" for name in self.fitpars},
        )
        self._jug_backend = LinearizedJugTimingBackend.from_linear_model(model)
        self._pint_backend = LinearizedPintTimingBackend.from_linear_model(model)
        self.backend_calls = []

    @property
    def toas(self):
        return self._toas

    @property
    def residuals(self):
        return self._residuals

    @property
    def toaerrs(self):
        return self._toaerrs

    @property
    def freqs(self):
        return self._freqs

    @property
    def Mmat(self):
        return self._design

    @property
    def flags(self):
        return self._flags

    @property
    def backend_flags(self):
        return self._backend_flags

    def timing_backend(self, name: str, linearized=None):
        self.backend_calls.append((name, linearized))
        if name == "jug":
            return self._jug_backend
        if name == "pint":
            return self._pint_backend
        raise ValueError(f"Unsupported backend for test host: {name}")


@pytest.fixture
def frontend_host():
    return _FrontendHost()


@pytest.fixture
def partition_with_sampled():
    return PartitionResult(
        fitpars=("F0", "F1", "DM"),
        marginalized=("F0", "DM"),
        sampled=("F1",),
        idx_marginalized=(0, 2),
        idx_sampled=(1,),
    )


@pytest.fixture
def sampled_space_delta():
    return ParameterSpace.build({"F1": "0.0"}, transform="none")


def test_discovery_frontend_emits_gp_and_delay(
    frontend_host, partition_with_sampled, sampled_space_delta
):
    backend = frontend_host.timing_backend("jug", linearized=True)
    signals = discovery_signals(
        host=frontend_host,
        space=sampled_space_delta,
        backend=backend,
        partition=partition_with_sampled,
        name="timing",
    )

    assert len(signals) == 2
    gp, delay = signals
    assert gp.F.shape == (
        len(frontend_host.toas),
        len(partition_with_sampled.marginalized),
    )
    assert delay.params == [f"{frontend_host.name}_timing_F1"]

    probe = {delay.params[0]: 0.25}
    actual = np.asarray(delay(probe), dtype=float)
    expected = -backend.residual_delta(np.array([0.0, 0.25, 0.0]))
    np.testing.assert_allclose(actual, expected)


def test_discovery_frontend_delay_consumes_injected_delta_without_transform(
    frontend_host, partition_with_sampled
):
    backend = frontend_host.timing_backend("jug")
    standardized_space = ParameterSpace.build({"F1": "0.0"}, transform="standardized")
    signals = discovery_signals(
        host=frontend_host,
        space=standardized_space,
        backend=backend,
        partition=partition_with_sampled,
        name="timing",
    )
    delay = signals[-1]

    actual = np.asarray(delay({delay.params[0]: 0.25}), dtype=float)
    expected = -backend.residual_delta(np.array([0.0, 0.25, 0.0]))
    np.testing.assert_allclose(actual, expected)


def test_discovery_frontend_rejects_non_jax_backend_with_sampled_delay(
    frontend_host, partition_with_sampled, sampled_space_delta
):
    backend = frontend_host.timing_backend("pint", linearized=True)
    with pytest.raises(ValueError, match="JAX-capable backend"):
        discovery_signals(
            host=frontend_host,
            space=sampled_space_delta,
            backend=backend,
            partition=partition_with_sampled,
            name="timing",
        )


def test_discovery_frontend_returns_only_gp_when_all_marginalized(frontend_host):
    partition = PartitionResult(
        fitpars=("F0", "F1", "DM"),
        marginalized=("F0", "F1", "DM"),
        sampled=tuple(),
        idx_marginalized=(0, 1, 2),
        idx_sampled=tuple(),
    )
    space = ParameterSpace.build({}, transform="none")
    backend = frontend_host.timing_backend("pint", linearized=True)
    signals = discovery_signals(
        host=frontend_host,
        space=space,
        backend=backend,
        partition=partition,
        name="timing",
    )
    assert len(signals) == 1


def test_enterprise_frontend_is_deferred_and_uses_sampled_exclusion(
    frontend_host, partition_with_sampled, sampled_space_delta
):
    signal = enterprise_signal(
        space_fn=lambda _host: sampled_space_delta,
        backend_name="jug",
        partition_spec=lambda _host: partition_with_sampled,
        name="timing",
        transform="none",
    )
    bound = signal(frontend_host)
    params = {f"{frontend_host.name}_timing_F1": 0.4}

    delay = np.asarray(bound.get_delay(params), dtype=float)
    expected = -frontend_host.timing_backend("jug", linearized=True).residual_delta(
        np.array([0.0, 0.4, 0.0], dtype=float)
    )
    np.testing.assert_allclose(delay, expected)
    assert ("jug", None) in frontend_host.backend_calls

    basis = bound.get_basis(params={})
    assert basis is not None
    assert basis.shape == (len(frontend_host.toas), 2)


def test_enterprise_frontend_whitening_uses_joint_x_parameter(frontend_host):
    partition = PartitionResult(
        fitpars=("F0", "F1", "DM"),
        marginalized=("F0", "DM"),
        sampled=("F1",),
        idx_marginalized=(0, 2),
        idx_sampled=(1,),
    )
    sampled_space_x = ParameterSpace.build({"F1": "0.0"}, transform="whitening")
    signal = enterprise_signal(
        space_fn=lambda _host: sampled_space_x,
        backend_name="jug",
        partition_spec=lambda _host: partition,
        name="timing",
        transform="whitening",
    )
    bound = signal(frontend_host)

    assert f"{frontend_host.name}_timing_x_0" in bound.param_names


def test_enterprise_frontend_all_marginalized_skips_delay_and_backend(frontend_host):
    partition = PartitionResult(
        fitpars=("F0", "F1", "DM"),
        marginalized=("F0", "F1", "DM"),
        sampled=tuple(),
        idx_marginalized=(0, 1, 2),
        idx_sampled=tuple(),
    )
    signal = enterprise_signal(
        space_fn=lambda _host: ParameterSpace.build({}, transform="none"),
        backend_name="jug",
        partition_spec=lambda _host: partition,
        name="timing",
        transform="none",
    )

    bound = signal(frontend_host)

    assert bound.signal_type == "basis"
    assert frontend_host.backend_calls == []


def test_enterprise_scalar_prior_composes_without_overcount(frontend_host):
    partition = PartitionResult(
        fitpars=("F0", "F1", "DM"),
        marginalized=("F0",),
        sampled=("F1", "DM"),
        idx_marginalized=(0,),
        idx_sampled=(1, 2),
    )
    space = ParameterSpace.build({"F1": "0.0", "DM": "0.0"}, transform="standardized")
    signal = enterprise_signal(
        space_fn=lambda _host: space,
        backend_name="jug",
        partition_spec=lambda _host: partition,
        name="timing",
        transform="standardized",
    )
    bound = signal(frontend_host)
    values = {
        f"{frontend_host.name}_timing_F1": 0.2,
        f"{frontend_host.name}_timing_DM": -0.3,
    }

    logprior = sum(param.get_logpdf(params=values) for param in bound.params)
    expected = space.logprior_coord(np.array([0.2, -0.3]), np, coord="z")

    np.testing.assert_allclose(logprior, expected)
