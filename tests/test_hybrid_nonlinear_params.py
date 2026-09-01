"""``nonlinear_params`` is executed by every engine family, not only JUG.

The hybrid mode (``"binary"`` | ``"binary+"``) keeps only the binary axes
(plus ``PX`` for ``"binary+"``) on the native residual path and routes every
other fitpar through its design-matrix column. These tests pin that partition
for the libstempo, Vela and PINT adapters with a recording fake native engine,
check the composite/host plumbing, and — when the real engines are installed —
check JUG, libstempo and PINT agree on the hybrid residual for the same delta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from metapulsar.engines import (
    LibstempoEngine,
    PintEngine,
    PtaContribution,
    PulsarTimingEngine,
    VelaEngine,
    hybrid_linearized_fitpars,
    is_hybrid_native_param,
)
from nltiming.engine_support import LinearModel

pytest.importorskip("jug")

FITPARS = ("F0", "RAJ", "PX", "PB", "A1", "JUMP1")
BINARY = {"PB", "A1"}


class _RecordingDeltaEngine:
    """Fake native engine: nonlinear in delta, records which axes it is given."""

    def __init__(self, design, settable):
        self.design = np.asarray(design, dtype=float)
        self.param_names = list(settable)
        self.fitpars = list(settable)
        self._reference_values = {name: 0.0 for name in settable}
        self.calls: list[dict[str, float]] = []

    def delta_residuals(self, delta_params):
        self.calls.append(dict(delta_params))
        delta = np.array(
            [float(delta_params.get(name, 0.0)) for name in self.fitpars],
            dtype=float,
        )
        lin = self.design @ delta
        return -lin + 0.3 * lin**2  # distinguishable from the affine column


def _linear_model():
    rng = np.random.default_rng(3)
    design = rng.normal(size=(7, len(FITPARS)))
    return LinearModel.from_design(
        fitpars=FITPARS,
        design=design,
        theta_exact={name: "0.0" for name in FITPARS},
    )


def _expected(fake, model, delta, native_names):
    """native(δ restricted to native axes) − M[:, lin] δ_lin."""
    native_delta = {n: float(delta[FITPARS.index(n)]) for n in native_names}
    lin = [i for i, n in enumerate(FITPARS) if n not in native_names]
    return fake.delta_residuals(native_delta) - model.design[:, lin] @ delta[lin]


@pytest.mark.parametrize(
    "mode, native",
    [
        (None, {"F0", "RAJ", "PX", "PB", "A1"}),  # JUMP1 is exact-linear anyway
        ("binary", {"PB", "A1"}),
        ("binary+", {"PB", "A1", "PX"}),
    ],
)
@pytest.mark.parametrize("family", ["tempo2", "vela"])
def test_native_adapters_partition_by_hybrid_mode(monkeypatch, family, mode, native):
    model = _linear_model()
    fake = _RecordingDeltaEngine(model.design, FITPARS)
    if family == "tempo2":
        monkeypatch.setattr(
            "metapulsar.engines.tempo2.Tempo2DeltaEngine", lambda lt_psr: fake
        )
        engine = LibstempoEngine.from_contribution(
            object(), linear_model=model, nonlinear_params=mode
        )
    else:
        monkeypatch.setattr(
            "metapulsar.engines.vela.VelaDeltaEngine", lambda spnta, **kw: fake
        )
        engine = VelaEngine.from_contribution(
            object(), linear_model=model, nonlinear_params=mode
        )

    assert engine.nonlinear_params == mode
    assert set(engine._native_fitpars) == native
    assert engine.exact_linear_fitpars() == set(FITPARS) - native
    assert engine.identically_linear_fitpars() == set(FITPARS) - native

    delta = np.array([1e-9, 2e-6, 0.3, 4e-7, 5e-6, 6e-7])
    fake.calls.clear()
    got = engine.residual_delta(delta)
    assert set(fake.calls[0]) == native
    np.testing.assert_allclose(got, _expected(fake, model, delta, native), rtol=1e-12)


@pytest.mark.parametrize(
    "mode, native",
    [
        (None, set(FITPARS)),  # PINT keeps every axis native by default
        ("binary", {"PB", "A1"}),
        ("binary+", {"PB", "A1", "PX"}),
    ],
)
def test_pint_adapter_partitions_by_hybrid_mode(monkeypatch, mode, native):
    model = _linear_model()
    fake = _RecordingDeltaEngine(model.design, FITPARS)
    monkeypatch.setattr(
        "metapulsar.engines.pint.PintDeltaEngine",
        lambda model_, toas, isort=None: fake,
    )
    engine = PintEngine.from_contribution(
        object(), object(), linear_model=model, nonlinear_params=mode
    )
    assert engine.nonlinear_params == mode
    assert set(engine._native_fitpars) == native
    assert engine.identically_linear_fitpars() == set(FITPARS) - native

    delta = np.array([1e-9, 2e-6, 0.3, 4e-7, 5e-6, 6e-7])
    fake.calls.clear()
    got = engine.residual_delta(delta)
    assert set(fake.calls[0]) == native
    np.testing.assert_allclose(got, _expected(fake, model, delta, native), rtol=1e-12)


def test_hybrid_mode_without_binary_axes_degenerates_to_pure_linear(monkeypatch):
    fitpars = ("F0", "F1")
    design = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, -0.5]])
    model = LinearModel.from_design(
        fitpars=fitpars, design=design, theta_exact={"F0": "1.0", "F1": "0.0"}
    )
    fake = _RecordingDeltaEngine(design, fitpars)
    monkeypatch.setattr(
        "metapulsar.engines.vela.VelaDeltaEngine", lambda spnta, **kw: fake
    )
    engine = VelaEngine.from_contribution(
        object(), linear_model=model, nonlinear_params="binary"
    )
    assert engine._native_fitpars == ()
    delta = np.array([0.2, -0.4])
    np.testing.assert_allclose(engine.residual_delta(delta), -(design @ delta))
    # the native (None) request still refuses an engine that can evaluate nothing
    with pytest.raises(ValueError, match="No Vela-evaluable"):
        VelaEngine.from_contribution(
            object(),
            linear_model=model,
            param_mapping={"F0": "nope", "F1": "nope2"},
        )


def test_hybrid_helpers_classify_on_engine_spelling():
    assert is_hybrid_native_param("PB", "binary")
    assert is_hybrid_native_param("E", "binary")  # tempo2 spelling of ECC
    assert is_hybrid_native_param("A1DOT", "binary")  # PINT spelling of XDOT
    assert not is_hybrid_native_param("PX", "binary")
    assert is_hybrid_native_param("PX", "binary+")
    assert not is_hybrid_native_param("F0", "binary+")
    assert is_hybrid_native_param("F0", None)
    # suffixed host names must be mapped to the engine spelling first
    assert hybrid_linearized_fitpars(
        ("PB_epta", "F0_epta", "PX"), {"PB_epta": "PB", "F0_epta": "F0"}, "binary"
    ) == {"F0_epta", "PX"}
    assert hybrid_linearized_fitpars(("PB", "F0"), {}, None) == frozenset()
    with pytest.raises(ValueError, match="nonlinear_params"):
        is_hybrid_native_param("PB", "all")


def test_composite_engine_reports_and_checks_hybrid_mode():
    model = _linear_model()
    fake = _RecordingDeltaEngine(model.design, FITPARS)
    hybrid = LibstempoEngine(engine=fake, linear_model=model, nonlinear_params="binary")
    native = LibstempoEngine(engine=fake, linear_model=model)
    rows = np.arange(model.design.shape[0])

    def contribution(name, engine):
        return PtaContribution(name=name, row_indices=rows, engine=engine)

    same = PulsarTimingEngine(
        fitpars=FITPARS,
        nrows=len(rows),
        contributions=[contribution("a", hybrid)],
        design_matrix=model.design,
    )
    assert same.nonlinear_params == "binary"
    with pytest.raises(ValueError, match="disagree on nonlinear_params"):
        PulsarTimingEngine(
            fitpars=FITPARS,
            nrows=2 * len(rows),
            contributions=[contribution("a", hybrid), contribution("b", native)],
        )


def test_direct_construction_partitions_or_refuses_a_stamped_mode():
    """A stamped mode must imply its partition, never just a label.

    ``from_contribution`` is not the only way an adapter is built; a direct
    construction that only stamped ``nonlinear_params`` would report a hybrid
    mode while evaluating every axis natively.
    """
    model = _linear_model()
    fake = _RecordingDeltaEngine(model.design, FITPARS)
    engine = LibstempoEngine(engine=fake, linear_model=model, nonlinear_params="binary")
    assert set(engine._native_fitpars) == {"PB", "A1"}
    assert engine.identically_linear_fitpars() == set(FITPARS) - {"PB", "A1"}
    delta = np.array([1e-9, 2e-6, 0.3, 4e-7, 5e-6, 6e-7])
    fake.calls.clear()
    got = engine.residual_delta(delta)
    assert set(fake.calls[0]) == {"PB", "A1"}
    np.testing.assert_allclose(
        got, _expected(fake, model, delta, {"PB", "A1"}), rtol=1e-12
    )

    # an explicit native list that contradicts the mode is refused
    with pytest.raises(ValueError, match="were passed as native_fitpars"):
        LibstempoEngine(
            engine=fake,
            linear_model=model,
            native_fitpars=FITPARS,
            nonlinear_params="binary",
        )
    # ... and the same guard holds for the other families
    with pytest.raises(ValueError, match="were passed as native_fitpars"):
        VelaEngine(
            engine=fake,
            linear_model=model,
            native_fitpars=("F0",),
            nonlinear_params="binary+",
        )


class _DeclaringEngine:
    """Leaf that declares an affine set independent of its exact-linear set."""

    def __init__(self, fitpars, design, identically_linear, nonlinear_params=None):
        self.fitpars = tuple(fitpars)
        self._design = np.asarray(design, dtype=float)
        self._il = frozenset(identically_linear)
        self.nonlinear_params = nonlinear_params

    def identically_linear_fitpars(self):
        return self._il

    def reference_theta_exact(self):
        return {name: "0.0" for name in self.fitpars}

    def residual_delta(self, delta):
        return -(self._design @ np.asarray(delta, dtype=float))

    def design_matrix(self, params=None):
        return self._design


def test_composite_reports_leaf_declared_affine_axes_not_routing_set():
    """The composite must read the leaves, not ``exact_linear_fitpars``.

    A JUG leg under a hybrid mode evaluates F0 as ``J @ delta`` — affine — but
    keeps it out of ``exact_linear_fitpars`` (that set is which columns the
    *host* evaluates). Reporting the routing set would under-report the affine
    set for JUG and disagree with a libstempo leg on the same model.
    """
    fitpars = ("F0", "PB", "JUMP1")
    design = np.arange(12, dtype=float).reshape(4, 3)
    rows = np.arange(4)
    jug_like = _DeclaringEngine(
        fitpars, design, {"F0", "JUMP1"}, nonlinear_params="binary"
    )
    composite = PulsarTimingEngine(
        fitpars=fitpars,
        nrows=4,
        contributions=[
            PtaContribution(
                name="a",
                row_indices=rows,
                engine=jug_like,
                exact_linear_fitpars=frozenset({"JUMP1"}),  # routing set only
            )
        ],
        design_matrix=design,
    )
    assert composite.identically_linear_fitpars() == {"F0", "JUMP1"}

    # a name only one leg linearizes is nonlinear for the composite: the
    # composite residual is the sum of the leaf blocks
    nonlinear_leg = _DeclaringEngine(
        fitpars, design, {"JUMP1"}, nonlinear_params="binary"
    )
    mixed = PulsarTimingEngine(
        fitpars=fitpars,
        nrows=8,
        contributions=[
            PtaContribution(name="a", row_indices=rows, engine=jug_like),
            PtaContribution(name="b", row_indices=rows + 4, engine=nonlinear_leg),
        ],
        design_matrix=np.vstack([design, design]),
    )
    assert mixed.identically_linear_fitpars() == {"JUMP1"}


def test_timing_engine_refuses_hybrid_mode_on_linearized_stand_in(mock_metapulsar):
    from metapulsar.mockpulsar import create_mock_libstempo

    pulsar = mock_metapulsar(
        {"pta_a": create_mock_libstempo(n_toas=20, name="J1857+0943", seed=1)},
        combination_strategy="per_pta",
    )
    with pytest.raises(ValueError, match="linearized=True"):
        pulsar.timing_engine(
            {"tempo2": "libstempo", "pint": "jug"},
            linearized=True,
            nonlinear_params="binary",
        )


# ---------------------------------------------------------------------------
# Real engines. The stock J1909 sim fits only binary axes (+ Offset), so under
# "binary" its hybrid partition equals its native one and it cannot separate
# the two formulas. These tests therefore fit F0/F1/ELONG/ELAT/PX as well, so
# the hybrid mode has real linearized axes to move off the native path.
#

SIM_DIR = (
    Path(__file__).resolve().parents[1]
    / "ref-packages"
    / "nltiming"
    / "examples"
    / "data"
    / "J1909-3744-sim"
)


def _enriched_par(tmp_path: Path) -> Path:
    """The sim par with spin, astrometry and parallax turned into fitpars."""
    src = SIM_DIR / "J1909-3744.par"
    if not (src.is_file() and (SIM_DIR / "J1909-3744.tim").is_file()):
        pytest.skip("J1909-3744 sim files missing")
    lines = []
    for line in src.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in {"F0", "F1", "ELONG", "ELAT"}:
            fields = (
                fields + ["1"] if len(fields) < 3 else fields[:2] + ["1"] + fields[3:]
            )
            line = " ".join(fields)
        lines.append(line)
    lines.append("PX 0.5 1")
    par = tmp_path / "J1909-3744.par"
    par.write_text("\n".join(lines) + "\n")
    return par


def _sim_pulsar(par: Path, timing_package: str):
    from metapulsar.metapulsar_factory import create_metapulsar

    return create_metapulsar(
        {
            "sim": [
                {
                    "par": par,
                    "tim": SIM_DIR / "J1909-3744.tim",
                    "timing_package": timing_package,
                }
            ]
        },
        combination_strategy="per_pta",
    )


def _bare(name: str) -> str:
    return name.removesuffix("_sim")


@pytest.mark.slow
@pytest.mark.requires_libstempo
@pytest.mark.requires_jug
@pytest.mark.parametrize("mode", ["binary", "binary+"])
def test_real_engines_report_the_same_linearized_set(tmp_path, mode):
    """JUG and libstempo must agree on the affine set — on the composite too.

    ``PulsarTimingEngine`` reports what the *leaf engines* declare: JUG's
    hybrid-linearized axes are not in its ``exact_linear_fitpars`` (that is the
    residual-routing set), so unioning those would under-report the affine set
    for a JUG leg and disagree with the libstempo leg on the same model.
    """
    pytest.importorskip("libstempo")
    par = _enriched_par(tmp_path)
    t2 = _sim_pulsar(par, "tempo2")
    fitpars = tuple(t2.fitpars)
    expected = {n for n in fitpars if not is_hybrid_native_param(_bare(n), mode)}
    assert {"F0_sim", "F1_sim", "ELONG_sim", "ELAT_sim"} <= expected
    assert ("PX_sim" in expected) is (mode == "binary")

    for impl in ("jug", "libstempo"):
        engine = t2.timing_engine({"tempo2": impl}, nonlinear_params=mode)
        assert engine.nonlinear_params == mode
        assert engine.identically_linear_fitpars() == expected, impl
        leaf = engine.contributions[0].engine
        assert leaf.identically_linear_fitpars() == expected, impl

    # native mode keeps every axis nonlinear but the auto exact-linear ones
    native = t2.timing_engine({"tempo2": "libstempo"})
    assert native.identically_linear_fitpars() == {"Offset_sim"}


@pytest.mark.slow
@pytest.mark.requires_libstempo
def test_hybrid_residual_is_the_linear_column_and_native_is_not(tmp_path):
    """The hybrid formula is exactly ``-M[:, lin] δ_lin`` on linearized axes.

    Uses F0: on this fixture tempo2's incremental ``residuals()`` does not
    rebuild barycentric arrival times for a position change, so its *native*
    astrometry response already degenerates to ``-M δ`` and cannot separate the
    formulas. The spin response is genuinely nonlinear (``r ~ -ΔΦ/F0``).
    """
    pytest.importorskip("libstempo")
    par = _enriched_par(tmp_path)
    t2 = _sim_pulsar(par, "tempo2")
    fitpars = tuple(t2.fitpars)
    native = t2.timing_engine({"tempo2": "libstempo"})
    hybrid = t2.timing_engine({"tempo2": "libstempo"}, nonlinear_params="binary")

    delta = np.zeros(len(fitpars))
    delta[fitpars.index("F0_sim")] = 2e-6
    design_response = -(native.design_matrix() @ delta)

    got = hybrid.residual_delta(delta)
    np.testing.assert_allclose(got, design_response, atol=1e-12)

    # the native residual is a different function: the second-order spin term
    # is ~8 ns here, far above the ns-level cross-engine parity floor
    diff = native.residual_delta(delta) - got
    diff -= np.mean(diff)  # libstempo centres residuals internally
    assert np.max(np.abs(diff)) > 2e-9

    # on the binary axes alone the two formulas coincide (both native there)
    binary_only = np.zeros(len(fitpars))
    binary_only[fitpars.index("A1_sim")] = 2e-7
    binary_only[fitpars.index("EPS1_sim")] = 2e-7
    diff = native.residual_delta(binary_only) - hybrid.residual_delta(binary_only)
    diff -= np.mean(diff)
    assert np.max(np.abs(diff)) < 1e-12


@pytest.mark.slow
@pytest.mark.requires_libstempo
@pytest.mark.requires_jug
@pytest.mark.parametrize("mode", ["binary", "binary+"])
def test_real_engine_families_agree_on_the_hybrid_residual(tmp_path, mode):
    """JUG, libstempo and PINT compute the same hybrid residual.

    Compared on the fitpars all three legs carry (PINT does not expose this
    par's ELONG/ELAT/PX as fitpars), at physically sized steps.
    """
    pytest.importorskip("libstempo")
    pytest.importorskip("pint")
    par = _enriched_par(tmp_path)
    t2 = _sim_pulsar(par, "tempo2")
    pint = _sim_pulsar(par, "pint")
    shared = [n for n in t2.fitpars if n in set(pint.fitpars) and n != "Offset_sim"]
    assert {"F0_sim", "F1_sim", "A1_sim", "EPS1_sim"} <= set(shared)

    steps = {
        "F0": 1e-10,
        "F1": 1e-18,
        "A1": 2e-7,
        "TASC": 3e-7,
        "EPS1": 2e-7,
        "EPS2": -2e-7,
    }

    def delta_for(pulsar):
        out = np.zeros(len(pulsar.fitpars))
        for i, name in enumerate(pulsar.fitpars):
            if name in shared:
                out[i] = steps[_bare(name)]
        return out

    engines = {
        "jug": t2.timing_engine({"tempo2": "jug"}, nonlinear_params=mode),
        "libstempo": t2.timing_engine({"tempo2": "libstempo"}, nonlinear_params=mode),
        "pint": pint.timing_engine({"pint": "pint"}, nonlinear_params=mode),
    }
    residuals = {}
    for name, engine in engines.items():
        pulsar = pint if name == "pint" else t2
        residuals[name] = np.asarray(
            engine.residual_delta(delta_for(pulsar)), dtype=float
        )
    scale = float(np.max(np.abs(residuals["jug"])))
    assert scale > 1e-9, "delta too small to be a meaningful parity probe"
    for name in ("libstempo", "pint"):
        diff = residuals[name] - residuals["jug"]
        diff -= np.mean(diff)  # libstempo centres residuals internally
        assert np.max(np.abs(diff)) < 1e-9 + 1e-6 * scale, (mode, name)
