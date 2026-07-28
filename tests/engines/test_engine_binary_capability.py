"""Authoritative §2.4 engine capability on the PINT adapter (review: this lands
without any MetaPulsar change — PintEngine already wraps the PINT model)."""

import pytest

from metapulsar.engines.pint import PintEngine
from nltiming.protocols import BinaryChartCapability


class _LM:
    fitpars = ("ECC", "OM", "T0", "PB")
    native_units: dict[str, str] = {}


class _P:
    def __init__(self, value):
        self.value = value


class _FakeModel:
    def __init__(self, binary, **secular):
        self.BINARY = _P(binary)
        for k, v in secular.items():
            setattr(self, k, _P(v))


def _cap(model):
    eng = PintEngine(engine=None, linear_model=_LM(), pint_model=model)
    return eng.binary_chart_capability("kepler_laplace", "")


def test_pint_capability_returns_none_when_inapplicable():
    m = _FakeModel("DD")
    eng = PintEngine(engine=None, linear_model=_LM(), pint_model=m)
    # wrong family, or no model held -> None (candidacy falls back).
    assert eng.binary_chart_capability("shapiro", "") is None
    assert (
        PintEngine(
            engine=None, linear_model=_LM(), pint_model=None
        ).binary_chart_capability("kepler_laplace", "")
        is None
    )


def test_pint_capability_dd_family_no_secular():
    cap = _cap(_FakeModel("DD"))
    assert isinstance(cap, BinaryChartCapability)
    assert cap.kepler_convention == "dd"
    assert cap.epoch_shift_exact is True
    assert cap.secular_terms == ()
    assert cap.origin_certified is False and cap.supports_domain is True


def test_pint_capability_explicit_secular_not_exact():
    cap = _cap(_FakeModel("DD", OMDOT=1.2e-3, PBDOT=0.0))
    assert cap.epoch_shift_exact is False
    assert "OMDOT" in cap.secular_terms and "PBDOT" not in cap.secular_terms


def test_pint_capability_ddgr_derived_secular():
    # DDGR derives OMDOT/PBDOT internally (no explicit fitpar) -> not exact.
    cap = _cap(_FakeModel("DDGR"))
    assert cap.kepler_convention == "dd"
    assert cap.epoch_shift_exact is False
    assert {"OMDOT", "PBDOT"} <= set(cap.secular_terms)


def test_pint_capability_non_kepler_convention():
    # An ELL1(-family) binary is not the ECC/OM/T0 convention (in practice it is
    # filtered earlier as already_laplace, but the capability is honest).
    cap = _cap(_FakeModel("ELL1"))
    assert cap.kepler_convention == "other"


def test_pint_capability_real_bt_model():
    pytest.importorskip("pint")
    import pint.config as pint_config
    from pint.models import get_model

    model = get_model(pint_config.examplefile("J0613-sim.par"))
    cap = _cap(model)
    assert cap.kepler_convention == "dd"  # BINARY == BT
    assert cap.epoch_shift_exact is True  # no active secular rates
    assert cap.secular_terms == ()


# ---------------------------------------------------------------------------
# Composite forwarding (§2.4.1 step c): nltiming-side, no MetaPulsar change.
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402
from metapulsar.engines.composite import (  # noqa: E402
    PtaContribution,
    PulsarTimingEngine,
)


class _LeafEngine:
    """Minimal leaf engine exposing fitpars + a capability (or none)."""

    def __init__(self, fitpars, capability="_MISSING"):
        self.fitpars = tuple(fitpars)
        self.native_units = {n: "native" for n in self.fitpars}
        self._cap = capability

    def reference_theta_exact(self):
        return {n: "0.0" for n in self.fitpars}

    if True:  # attach the method only when a capability is configured

        def binary_chart_capability(self, family, suffix):
            if self._cap == "_MISSING":
                raise AttributeError
            return self._cap


class _LeafNoCap:
    """Leaf engine WITHOUT binary_chart_capability (e.g. a test double or a
    LinearizedJugEngine) — the composite keeps its group on the fallback."""

    def __init__(self, fitpars):
        self.fitpars = tuple(fitpars)
        self.native_units = {n: "native" for n in self.fitpars}

    def reference_theta_exact(self):
        return {n: "0.0" for n in self.fitpars}


def _cap_obj(**kw):
    base = dict(
        kepler_convention="dd",
        epoch_shift_exact=True,
        secular_terms=(),
        origin_certified=False,
        supports_domain=True,
    )
    base.update(kw)
    return BinaryChartCapability(**base)


def _composite(*contribs):
    fitpars = tuple(dict.fromkeys(n for c in contribs for n in c.engine.fitpars))
    return PulsarTimingEngine(
        fitpars=fitpars,
        nrows=4,
        contributions=list(contribs),
        design_matrix=None,
    )


def _contrib(name, engine):
    return PtaContribution(name=name, row_indices=np.array([0, 1]), engine=engine)


def test_composite_forwards_to_suffixed_owner():
    cap_a = _cap_obj(secular_terms=("OMDOT",), epoch_shift_exact=False)
    eng_a = _LeafEngine(("ECC_a", "OM_a", "T0_a", "PB_a"), capability=cap_a)
    eng_b = _LeafEngine(("ECC_b", "OM_b", "T0_b", "PB_b"), capability=_cap_obj())
    comp = _composite(_contrib("a", eng_a), _contrib("b", eng_b))
    assert comp.binary_chart_capability("kepler_laplace", "_a") is cap_a
    assert comp.binary_chart_capability("kepler_laplace", "_b").epoch_shift_exact


def test_composite_none_when_leaf_lacks_method():
    # A JUG-style leaf without the method keeps the group on the fallback.
    eng = _LeafNoCap(("ECC", "OM", "T0", "PB"))
    comp = _composite(_contrib("j", eng))
    assert comp.binary_chart_capability("kepler_laplace", "") is None


def test_composite_shared_binary_agreement_and_disagreement():
    shared = ("ECC", "OM", "T0", "PB")
    cap = _cap_obj()
    agree = _composite(
        _contrib("a", _LeafEngine(shared, capability=cap)),
        _contrib("b", _LeafEngine(shared, capability=cap)),
    )
    assert agree.binary_chart_capability("kepler_laplace", "") == cap
    # Disagreeing shared-binary owners -> None (never guess).
    disagree = _composite(
        _contrib("a", _LeafEngine(shared, capability=_cap_obj())),
        _contrib(
            "b", _LeafEngine(shared, capability=_cap_obj(epoch_shift_exact=False))
        ),
    )
    assert disagree.binary_chart_capability("kepler_laplace", "") is None


# ---------------------------------------------------------------------------
# JUG facts (source of truth) + JugEngine translator (§2.4.1 steps a/b).
# ---------------------------------------------------------------------------


def test_jug_binary_chart_facts():
    pytest.importorskip("jug")
    from jug.fitting.binary_delay_plan import binary_chart_facts

    base = {"A1": 1.0, "PB": 8.0, "ECC": 8e-4, "OM": 50.7, "T0": 55000.0}
    dd = binary_chart_facts({**base, "BINARY": "DD"}, [])
    assert dd.convention_family == "dd" and dd.epoch_shift_exact is True
    omdot = binary_chart_facts({**base, "BINARY": "DD", "OMDOT": 1.2e-3}, [])
    assert omdot.epoch_shift_exact is False and "OMDOT" in omdot.secular_terms
    # DDGR derives OMDOT/PBDOT internally (invisible to a name search).
    ddgr = binary_chart_facts({**base, "BINARY": "DDGR"}, [])
    assert ddgr.epoch_shift_exact is False
    assert {"OMDOT", "PBDOT"} <= set(ddgr.secular_terms)
    assert binary_chart_facts({"F0": 100.0}, []) is None


def test_jug_engine_translates_facts():
    pytest.importorskip("jug")
    from metapulsar.engines.jug import JugEngine

    class _LM:
        fitpars = ("ECC", "OM", "T0", "PB")
        native_units: dict[str, str] = {}

    class _Facts:
        convention_family = "dd"
        epoch_shift_exact = False
        secular_terms = ("OMDOT", "PBDOT")

    eng = JugEngine(state=None, linear_model=_LM())
    # No facts resolved (direct construction) -> None -> candidacy fallback.
    assert eng.binary_chart_capability("kepler_laplace", "") is None
    # With facts, the translator maps them 1:1 (adding nltiming-owned fields).
    eng._binary_facts = _Facts()
    cap = eng.binary_chart_capability("kepler_laplace", "")
    assert isinstance(cap, BinaryChartCapability)
    assert cap.kepler_convention == "dd" and cap.epoch_shift_exact is False
    assert cap.secular_terms == ("OMDOT", "PBDOT")
    assert cap.origin_certified is False and cap.supports_domain is True
    assert eng.binary_chart_capability("shapiro", "") is None
