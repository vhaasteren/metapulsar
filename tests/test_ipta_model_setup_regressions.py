"""IPTA DR2 Kepler Model D setup regressions (16 pulsars).

Sole location for this acceptance suite (not under ``paper/``). Production
helpers live in ``paper/code/validation``; this module bootstraps that path.
Default ``addopts`` deselects ``slow``/IPTA markers — run with
``--override-ini=addopts=`` (see the bug-fix verification plan).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_VALIDATION_ROOT = Path(__file__).resolve().parents[1] / "paper" / "code" / "validation"
if _VALIDATION_ROOT.is_dir():
    sys.path.insert(0, str(_VALIDATION_ROOT))

pytest.importorskip("validation")

from validation.config import (  # noqa: E402
    SAMPLED_PARAMS_BASE_BT,
    SAMPLED_PARAMS_BASE_ELL1,
    SAMPLED_PARAMS_BASE_ISOLATED,
    SAMPLED_PARAMS_BASE_ISOLATED_ECL,
    RunConfig,
    choose_sampled_base,
)
from validation.run import (  # noqa: E402
    build_pulsar_multi_pta,
    prepare_nonlinear_model,
    resolve_clock_dir,
)

CASES = {
    "J0610-2100": (("epta",), SAMPLED_PARAMS_BASE_ELL1),
    "J0751+1807": (("epta",), SAMPLED_PARAMS_BASE_ELL1),
    "J1022+1001": (("epta", "ppta"), SAMPLED_PARAMS_BASE_BT),
    "J1600-3053": (("epta", "ppta", "ng9"), SAMPLED_PARAMS_BASE_BT),
    "J1713+0747": (("epta", "ppta", "ng9"), SAMPLED_PARAMS_BASE_BT),
    "J1730-2304": (("epta", "ppta"), SAMPLED_PARAMS_BASE_ISOLATED),
    "J1747-4036": (("ng9",), SAMPLED_PARAMS_BASE_ISOLATED_ECL),
    "J1857+0943": (("epta", "ppta"), SAMPLED_PARAMS_BASE_BT),
    "J1903+0327": (("ng9",), SAMPLED_PARAMS_BASE_BT),
    "J1910+1256": (("epta", "ng9"), SAMPLED_PARAMS_BASE_BT),
    "J1918-0642": (("epta", "ng9"), SAMPLED_PARAMS_BASE_BT),
    "J1939+2134": (("epta", "ppta"), SAMPLED_PARAMS_BASE_ISOLATED),
    "J1949+3106": (("ng9",), SAMPLED_PARAMS_BASE_ELL1),
    "J2124-3358": (("epta", "ppta"), SAMPLED_PARAMS_BASE_ISOLATED),
    "J2302+4442": (("ng9",), SAMPLED_PARAMS_BASE_BT),
    "J2317+1439": (("epta", "ng9"), SAMPLED_PARAMS_BASE_ELL1),
}


def _ipta_root() -> Path:
    env = os.environ.get("IPTA_DR2_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "data" / "ipta-dr2"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_ipta_data
@pytest.mark.requires_jug
@pytest.mark.parametrize("name,spec", sorted(CASES.items()))
def test_ipta_model_setup_regression(name, spec):
    ptas, golden_base = spec
    ipta_root = _ipta_root()
    if not ipta_root.is_dir():
        pytest.skip(f"IPTA DR2 root missing: {ipta_root}")

    clock_dir = resolve_clock_dir()
    pulsar, suffix = build_pulsar_multi_pta(ipta_root, name, ptas, clock_dir=clock_dir)
    config = RunConfig.from_preset(
        "discovery_all_jug",
        design_matrix_method="autodiff",
        tempo2_native="fixed_state_stripped",
        prior_override_policy="warn",
        cheat_prior_scale=100.0,
    )
    assert choose_sampled_base(pulsar) == golden_base

    sampled, ntm = prepare_nonlinear_model(pulsar, suffix, config)
    assert sampled
    assert ntm is not None

    ctx = ntm.for_pulsar(pulsar)
    engine = ctx.engine
    matrix = ctx.engine_design_matrix

    for contribution in engine.contributions:
        rows = np.asarray(contribution.row_indices, dtype=int)
        for pname in contribution.engine.fitpars:
            col = pulsar.fitpars.index(pname)
            block = matrix[rows, col]
            assert np.all(np.isfinite(block))
            assert np.linalg.norm(block) > 0.0

    zero = np.zeros(len(ctx.engine.fitpars), dtype=float)
    residual = ctx.engine.residual_delta(zero)
    assert residual.shape == (len(pulsar.toas),)
    assert np.all(np.isfinite(residual))

    merged = dict(engine.reference_theta_exact())
    for pname, owners in pulsar._fitparameters.items():
        if len(owners) > 1 and pname in merged:
            assert isinstance(merged[pname], str)
            assert merged[pname]
