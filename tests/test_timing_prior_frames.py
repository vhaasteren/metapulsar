"""Tests for frame-aware prior override materialization."""

import pytest

from metapulsar.timing.bijectors import AxisPrior
from metapulsar.timing.nonlinear_timing_model import NonLinearTimingModel
from metapulsar.timing.partition import resolve_partition
from metapulsar.timing.priors import (
    PriorBuildContext,
    PriorOverrideSpec,
    apply_prior_scale,
    materialize_prior_override,
)


@pytest.fixture
def composite_binary_ctx() -> PriorBuildContext:
    """Refs mimicking J0613 composite: EPTA vs NG9 TASC differ by ~1776 days."""
    return PriorBuildContext(
        refs={
            "TASC_epta": "53113.7963542019",
            "TASC_ng9": "54889.9918085650",
            "PB_epta": "1.19851257519955",
            "PB": "1.19851257519955",
        },
        fitpars=("TASC_epta", "TASC_ng9", "PB_epta", "PB"),
        sampled=("TASC_epta", "PB_epta"),
    )


def test_apply_prior_scale_uniform():
    prior = AxisPrior(family="uniform", lower=-0.5, upper=0.5)
    scaled = apply_prior_scale(prior, 1.2)
    assert scaled.lower == pytest.approx(-0.6)
    assert scaled.upper == pytest.approx(0.6)


def test_materialize_absolute_subtracts_bound_ref(composite_binary_ctx):
    spec = PriorOverrideSpec(
        prior=AxisPrior(family="uniform", lower=53113.0, upper=53114.0),
        frame="absolute",
    )
    out = materialize_prior_override("TASC_epta", spec, composite_binary_ctx)
    assert out.lower == pytest.approx(53113.0 - 53113.7963542019, rel=0, abs=1e-4)
    assert out.upper == pytest.approx(53114.0 - 53113.7963542019, rel=0, abs=1e-4)


def test_materialize_delta_unchanged(composite_binary_ctx):
    spec = PriorOverrideSpec(
        prior=AxisPrior(family="uniform", lower=-0.1, upper=0.1),
        frame="delta",
    )
    out = materialize_prior_override("TASC_epta", spec, composite_binary_ctx)
    assert out.lower == pytest.approx(-0.1)
    assert out.upper == pytest.approx(0.1)


def test_materialize_delta_with_scale(composite_binary_ctx):
    spec = PriorOverrideSpec(
        prior=AxisPrior(family="uniform", lower=-0.5, upper=0.5),
        frame="delta",
        scale="PB_epta",
    )
    out = materialize_prior_override("TASC_epta", spec, composite_binary_ctx)
    half = 0.5 * 1.19851257519955
    assert out.lower == pytest.approx(-half, rel=1e-12)
    assert out.upper == pytest.approx(+half, rel=1e-12)


def test_old_notebook_mistake_produces_wrong_delta(composite_binary_ctx):
    """Simulate pint_model()['TASC'] absolute bounds applied to TASC_epta ref."""
    wrong_tasc = 54889.9918085650
    pb = 1.19851257519955
    spec = PriorOverrideSpec(
        prior=AxisPrior(
            family="uniform",
            lower=wrong_tasc - 0.5 * pb,
            upper=wrong_tasc + 0.5 * pb,
        ),
        frame="absolute",
    )
    out = materialize_prior_override("TASC_epta", spec, composite_binary_ctx)
    assert out.lower == pytest.approx(1775.596, rel=1e-3)
    assert out.upper == pytest.approx(1776.795, rel=1e-3)


def test_scale_rejected_for_absolute_frame(composite_binary_ctx):
    spec = PriorOverrideSpec(
        prior=AxisPrior(family="uniform", lower=-0.5, upper=0.5),
        frame="absolute",
        scale="PB_epta",
    )
    with pytest.raises(ValueError, match="only supported with frame='delta'"):
        materialize_prior_override("TASC_epta", spec, composite_binary_ctx)


class _FakeBackend:
    def __init__(self, refs: dict[str, str]):
        self._refs = refs

    def reference_theta_exact(self) -> dict[str, str]:
        return dict(self._refs)


class _FakePulsar:
    name = "FAKEPSR"
    fitpars = ["TASC_epta", "PB_epta"]

    def __init__(self, refs: dict[str, str]):
        self._refs = refs

    def pint_model(self):
        return None

    def timing_backend(self, engines, design_matrix_method="analytic"):
        return _FakeBackend(self._refs)

    def can_use_engines(self, engines):
        return True

    def cache_token(self):
        return "fake-v1"


def test_ntm_resolve_prior_overrides(composite_binary_ctx):
    pulsar = _FakePulsar(dict(composite_binary_ctx.refs))
    backend = pulsar.timing_backend({})
    ntm = NonLinearTimingModel(analytically_marginalize=[])
    ntm.set_prior(
        "TASC_epta",
        "uniform",
        frame="delta",
        lower=-0.5,
        upper=0.5,
        scale="PB_epta",
    )
    partition = resolve_partition(pulsar, analytically_marginalize=[])
    resolved = ntm._resolve_prior_overrides(
        pulsar=pulsar, backend=backend, partition=partition
    )
    half = 0.5 * 1.19851257519955
    assert resolved["TASC_epta"].lower == pytest.approx(-half, rel=1e-12)
    assert resolved["TASC_epta"].upper == pytest.approx(+half, rel=1e-12)
