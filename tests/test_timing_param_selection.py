"""Tests for sample=-based partition selection and constructor priors."""

import numpy as np
import pytest

from metapulsar.timing import priors as prior_specs
from metapulsar.timing.backends.base import LinearModel
from metapulsar.timing.backends.jug import LinearizedJugEngine
from metapulsar.timing.nonlinear_timing_model import NonLinearTimingModel
from metapulsar.timing.partition import (
    fitpar_suffixes,
    match_fitpars,
    resolve_partition,
    select_fitpars,
)


class _SuffixHost:
    """Composite-style host with PTA-suffixed fitpars and _fitparameters mapping."""

    def __init__(self):
        self.name = "J2222+2222"
        self.fitpars = ("F1", "PB_a", "TASC_a", "PB_b", "TASC_b")
        self._fitparameters = {
            "PB_a": {"pta_a": "PB"},
            "TASC_a": {"pta_a": "TASC"},
            "PB_b": {"pta_b": "PB"},
            "TASC_b": {"pta_b": "TASC"},
        }
        n = 7
        self._toas = np.linspace(0.0, 1.0, n)
        self._residuals = np.zeros(n)
        self._toaerrs = np.full(n, 1.0e-6)
        self._freqs = np.full(n, 1400.0)
        self._flags = {"pta": np.array(["demo"] * n, dtype="U8")}
        self._backend_flags = np.array(["demo"] * n, dtype="U8")
        rng = np.random.default_rng(42)
        design = np.column_stack(
            [np.linspace(-0.5, 0.5, n)]
            + [rng.normal(size=n) for _ in range(len(self.fitpars) - 1)]
        )
        model = LinearModel.from_host(
            fitpars=self.fitpars,
            design=design,
            theta_exact={
                "F1": "1.0",
                "PB_a": "10.0",
                "TASC_a": "55000.0",
                "PB_b": "20.0",
                "TASC_b": "56000.0",
            },
        )
        self._backend = LinearizedJugEngine.from_linear_model(model)

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
        return self._backend.design_matrix()

    @property
    def flags(self):
        return self._flags

    @property
    def backend_flags(self):
        return self._backend_flags

    def cache_token(self):
        return "suffix-token"

    def pint_model(self):
        return object()

    def timing_backend(self, engines="jug", **kwargs):
        return self._backend


@pytest.fixture
def host():
    return _SuffixHost()


def test_match_fitpars_base_name_matches_all_suffixed(host):
    fitpars = host.fitpars
    assert match_fitpars(host, "PB", fitpars) == ("PB_a", "PB_b")
    assert match_fitpars(host, "PB_a", fitpars) == ("PB_a",)
    assert match_fitpars(host, "F1", fitpars) == ("F1",)
    assert match_fitpars(host, "DMX_0001", fitpars) == ()


def test_fitpar_suffixes(host):
    assert fitpar_suffixes(host, "PB_a") == {"_a"}
    assert fitpar_suffixes(host, "F1") == {""}


def test_select_fitpars_preserves_order_and_raises_on_miss(host):
    assert select_fitpars(host, ["TASC", "PB"]) == ("PB_a", "TASC_a", "PB_b", "TASC_b")
    with pytest.raises(ValueError, match="matches no fit parameter"):
        select_fitpars(host, ["ECC"])


def test_resolve_partition_sample_marginalizes_rest(host):
    partition = resolve_partition(host, sample=["PB", "TASC"])
    assert partition.sampled == ("PB_a", "TASC_a", "PB_b", "TASC_b")
    assert partition.analytically_marginalized == ("F1",)


def test_resolve_partition_sample_and_explicit_marginalize_conflict(host):
    with pytest.raises(ValueError, match="not both"):
        resolve_partition(host, analytically_marginalize=["F1"], sample=["PB"])


def test_model_sample_kwarg_selects_partition(host):
    ntm = NonLinearTimingModel(engines="jug", sample=["PB", "TASC"], name="timing")
    binding = ntm.bind(host)
    assert binding.sampled == ("PB_a", "TASC_a", "PB_b", "TASC_b")
    assert binding.marginalized == ("F1",)


def test_model_sample_string_rejected():
    with pytest.raises(ValueError, match="sequence of fitpar names"):
        NonLinearTimingModel(engines="jug", sample="PB")


def test_model_sample_conflicts_with_marginalize():
    with pytest.raises(ValueError, match="not both"):
        NonLinearTimingModel(
            engines="jug", sample=["PB"], analytically_marginalize=["F1"]
        )


def test_constructor_priors_expand_to_suffixed_targets(host):
    ntm = NonLinearTimingModel(
        engines="jug",
        sample=["PB", "TASC"],
        priors={"TASC": prior_specs.delta_uniform(-0.5, 0.5, scale="PB")},
        name="timing",
    )
    block = ntm.bind(host).priors
    by_name = dict(zip(block.names, block.priors))
    assert block.sources["TASC_a"] == "override"
    assert block.sources["TASC_b"] == "override"
    # scale resolves suffix-consistently: PB_a ref = 10.0, PB_b ref = 20.0
    np.testing.assert_allclose(
        (by_name["TASC_a"].lower, by_name["TASC_a"].upper), (-5.0, 5.0)
    )
    np.testing.assert_allclose(
        (by_name["TASC_b"].lower, by_name["TASC_b"].upper), (-10.0, 10.0)
    )


def test_constructor_priors_reject_non_spec_values():
    with pytest.raises(TypeError, match="PriorOverrideSpec"):
        NonLinearTimingModel(engines="jug", priors={"PB": ("uniform", -1, 1)})


def test_prior_spec_helpers_validate_scale_frame():
    with pytest.raises(ValueError, match="frame='delta'"):
        prior_specs.uniform(-1.0, 1.0, scale="PB")
    spec = prior_specs.delta_normal(0.0, 1.0, scale="PB")
    assert spec.frame == "delta"
    assert spec.scale == "PB"


def test_with_engines_carries_sample_and_priors(host):
    ntm = NonLinearTimingModel(
        engines="jug",
        sample=["PB", "TASC"],
        priors={"TASC": prior_specs.delta_uniform(-0.5, 0.5, scale="PB")},
        name="timing",
    )
    other = ntm.with_engines("jug")
    binding = other.bind(host)
    assert binding.sampled == ("PB_a", "TASC_a", "PB_b", "TASC_b")
    assert binding.priors.sources["TASC_a"] == "override"
