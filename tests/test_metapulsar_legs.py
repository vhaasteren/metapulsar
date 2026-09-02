"""A vela-jax leg is built as one, and the record is the engine's own -J (PR-4).

Before this, a MetaPulsar leg was read by PINT or libstempo at construction
and an engine was chosen afterwards, so the design matrix the likelihood
marginalizes and the residual the sampler moves came from two different codes.
``engines="vela_jax"`` closes that: the record and the engine come from one
read of one par/tim pair (:class:`metapulsar.leg.TimingLeg`), and what used to
be a tolerance comparison becomes *block equality*.

Every assertion here is exact (``array_equal``) on purpose. There is no
tolerance to choose because there is no second computation to compare against.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import vela_jax_tempo2_host

LOCAL_DATA = Path(__file__).resolve().parents[1] / "data" / "aei-dr2"
PSR = "J1853+1303"

LEGS = {
    "epta_dr1_v2_2": "tempo2",
    "nanograv_9y": "pint",
}

pytestmark = [
    pytest.mark.requires_vela_jax,
    pytest.mark.real_data,
    pytest.mark.skipif(
        not (LOCAL_DATA / "epta_dr1_v2_2" / "par" / f"{PSR}.par").exists(),
        reason="local multi-PTA data tree not present",
    ),
]


def _spec(release):
    base = LOCAL_DATA / release
    return [
        {
            "par": base / "par" / f"{PSR}.par",
            "tim": base / "tim" / f"{PSR}.tim",
            "timing_package": LEGS[release],
        }
    ]


@lru_cache(maxsize=None)
def _build(releases, strategy, engines="vela_jax"):
    """Build once per distinct combination; no test here mutates the pulsar.

    Every build reads real par/tim through PINT or tempo2, which is ~10-20 s.
    Rebuilding per test turned a two-file module into three minutes.
    """
    from metapulsar import create_metapulsar

    kwargs = {"engines": engines} if engines else {}
    return create_metapulsar(
        {release: _spec(release) for release in releases},
        combination_strategy=strategy,
        use_pulse_numbers="reuse",
        **kwargs,
    )


def _blocks_are_the_engines(pulsar):
    """``pulsar.Mmat`` and ``pulsar.residuals`` are the legs' own arrays."""
    slices = pulsar._get_pta_slices()
    index = {name: i for i, name in enumerate(pulsar.fitpars)}
    design = np.asarray(pulsar.Mmat, dtype=float)
    residuals = np.asarray(pulsar.residuals, dtype=float)

    for pta, slc in slices.items():
        leg = pulsar._pta_data[pta]._leg
        assert leg is not None and leg.engine_name == "vela_jax"
        rows = np.arange(slc.start, slc.stop, dtype=int)

        engine_design = np.asarray(leg.engine.design_matrix(), dtype=float)
        engine_names = list(leg.engine.param_names)
        for name in pulsar.fitpars:
            native = pulsar._fitparameters.get(name, {}).get(pta)
            if native is None or native.pint_name not in engine_names:
                continue
            assert np.array_equal(
                design[rows, index[name]],
                engine_design[:, engine_names.index(native.pint_name)],
            ), f"{pta}:{name}"

        assert np.array_equal(
            residuals[rows], np.asarray(leg.engine.residuals(), dtype=float)
        ), pta


@pytest.mark.parametrize("release", sorted(LEGS))
def test_a_single_leg_record_is_its_engines(release):
    """One leg, either host: tempo2 reads a tempo2 leg, PINT a PINT leg."""
    pytest.importorskip("vela_jax")
    if LEGS[release] == "tempo2" and not vela_jax_tempo2_host():
        pytest.skip("needs libstempo with the siteVel/clock properties")
    _blocks_are_the_engines(_build((release,), "per_pta"))


@pytest.mark.parametrize("strategy", ["per_pta", "shared"])
def test_a_two_leg_composite_is_block_equal_under_both_strategies(strategy):
    """D9: neither combination strategy touches a leg's own block.

    ``shared`` merges the timing-model parameters the two releases have in
    common into one column, which is a *column* operation -- the rows each leg
    contributes to that column are still its engine's.
    """
    pytest.importorskip("vela_jax")
    if not vela_jax_tempo2_host():
        pytest.skip("needs libstempo with the siteVel/clock properties")
    _blocks_are_the_engines(_build(tuple(sorted(LEGS)), strategy))


def test_the_composite_engine_passes_block_equality():
    """``timing_engine`` builds over the legs' own engines and validates it."""
    pytest.importorskip("vela_jax")
    if not vela_jax_tempo2_host():
        pytest.skip("needs libstempo with the siteVel/clock properties")
    from metapulsar.engines import validate_composite_against_pulsar

    pulsar = _build(tuple(sorted(LEGS)), "per_pta")
    engine = pulsar.timing_engine("vela_jax")
    validate_composite_against_pulsar(engine, pulsar)

    assert tuple(engine.fitpars) == tuple(pulsar.fitpars)
    zero = np.zeros(len(engine.fitpars))
    assert np.max(np.abs(engine.residual_delta(zero))) == 0.0


def test_shared_axes_expand_around_one_reference():
    """Under ``shared``, both legs' reference strings for a merged axis agree.

    The factory path passes ``rows=None`` (there is no second read to align),
    so ``check_reference_agreement`` does not run there -- this is the
    assertion that replaces it.
    """
    pytest.importorskip("vela_jax")
    if not vela_jax_tempo2_host():
        pytest.skip("needs libstempo with the siteVel/clock properties")

    pulsar = _build(tuple(sorted(LEGS)), "shared")
    shared = [
        name for name in pulsar.fitpars if len(pulsar._fitparameters.get(name, {})) > 1
    ]
    assert shared, "the two releases share no fitted timing parameters"

    for name in shared:
        references = set()
        for pta, native in pulsar._fitparameters[name].items():
            engine = pulsar._pta_data[pta]._leg.engine
            exact = engine.reference_theta_exact()
            if native.pint_name in exact:
                references.add(exact[native.pint_name])
        assert (
            len(references) <= 1
        ), f"{name}: legs disagree on the reference {references}"

    merged = pulsar.timing_engine("vela_jax").reference_theta_exact()
    assert set(shared) <= set(merged)


def test_a_leg_refuses_a_different_engine():
    """D13: no JUG engine over a record that is vela-jax's ``-J``."""
    pytest.importorskip("vela_jax")

    pulsar = _build(("nanograv_9y",), "per_pta")
    assert pulsar.can_use_engines("jug") is False
    with pytest.raises(ValueError, match="cannot be honored"):
        pulsar.timing_engine("jug")


def test_vela_jax_without_a_leg_is_refused_by_name():
    """D8: the choice is made at construction, so a post-hoc ask must say so."""
    pytest.importorskip("vela_jax")
    pulsar = _build(("nanograv_9y",), "per_pta", engines=None)
    assert pulsar.can_use_engines("vela_jax") is False
    with pytest.raises(ValueError, match="cannot be honored"):
        pulsar.timing_engine("vela_jax")


def test_linearized_over_a_leg_is_the_records_own_matrix():
    """D13: ``linearized=True`` is ``-M delta`` over this record's own ``-J``."""
    pytest.importorskip("vela_jax")

    pulsar = _build(("nanograv_9y",), "per_pta")
    engine = pulsar.timing_engine("vela_jax", linearized=True)

    design = np.asarray(engine.design_matrix(), dtype=float)
    assert np.array_equal(design, np.asarray(pulsar.Mmat, dtype=float))

    assert np.max(np.abs(engine.residual_delta(np.zeros(len(engine.fitpars))))) == 0.0

    rng = np.random.default_rng(0)
    delta = rng.normal(size=len(engine.fitpars)) * 1e-9
    expected = -design @ delta
    # Relative, not absolute: the F0 column spans the data, so `-M delta` is
    # ~1e4 s here and float64 rounding on that is ~1e-11 s.
    residual = engine.residual_delta(delta)
    assert np.max(np.abs(residual - expected)) <= 1e-14 * np.max(np.abs(expected))


def test_the_composite_round_trips_through_psrdata(tmp_path):
    """One feather, readable by Discovery, naming both legs."""
    pytest.importorskip("vela_jax")
    if not vela_jax_tempo2_host():
        pytest.skip("needs libstempo with the siteVel/clock properties")
    from psrdata import PulsarData

    pulsar = _build(tuple(sorted(LEGS)), "per_pta")
    path = tmp_path / f"{pulsar.name}.feather"
    pulsar.to_feather(path)

    back = PulsarData.from_feather(path)
    assert back.name == pulsar.name
    assert back.timing_package == "composite"
    assert back.software == "metapulsar"
    assert np.array_equal(back.Mmat, np.asarray(pulsar.Mmat, dtype=float))
    assert np.array_equal(back.residuals, np.asarray(pulsar.residuals, dtype=float))

    legs = back.extra["legs"]
    assert {entry["timing_package"] for entry in legs} == set(LEGS.values())
    assert {entry["engine_name"] for entry in legs} == {"vela_jax"}

    discovery = pytest.importorskip("discovery")
    read = discovery.Pulsar.read_feather(str(path))
    assert np.array_equal(np.asarray(read.residuals), back.residuals)
