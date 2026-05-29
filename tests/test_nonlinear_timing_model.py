"""Tests for the MetaPulsar nonlinear timing-model package."""

import numpy as np
from enterprise.signals import gp_signals

from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import create_mock_libstempo
from metapulsar.nonlinear_timing_model import (
    AffineTransform,
    TransformRegistry,
    build_nonlinear_timing_signal,
    compute_timing_partition,
)


def _mock_metapulsar():
    return MetaPulsar(
        {
            "pta_a": create_mock_libstempo(
                n_toas=12, name="J1857+0943", telescope="pta_a", seed=10
            ),
            "pta_b": create_mock_libstempo(
                n_toas=9, name="J1857+0943", telescope="pta_b", seed=20
            ),
        },
        combination_strategy="consistent",
    )


def _zero_params(model):
    return {par.name: np.zeros(par.size) if par.size else 0.0 for par in model.params}


def test_transform_registry_roundtrip_and_metadata():
    registry = TransformRegistry(
        sampled_params=["F0", "DM"],
        standardization={
            "F0": {"center": 3.0, "scale": 2.0},
            "DM": AffineTransform(center=-2.0, scale=4.0),
        },
    )

    z_params = {"F0": 1.5, "DM": -0.25}
    physical = registry.to_physical(z_params)
    recovered = registry.to_standardized(physical)

    np.testing.assert_allclose(recovered["F0"], z_params["F0"], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(recovered["DM"], z_params["DM"], atol=1e-12, rtol=0.0)
    metadata = registry.metadata()
    assert metadata["F0"]["center"] == 3.0
    assert metadata["DM"]["scale"] == 4.0
    registry.validate_roundtrip()


def test_transform_registry_rejects_invalid_specs():
    try:
        TransformRegistry(["F0"], {"F0": {"center": 0.0, "scale": 0.0}})
    except ValueError as exc:
        assert "non-zero" in str(exc)
    else:
        raise AssertionError("Expected zero-scale transform validation error.")


def test_partitioning_indices_and_overlap_validation():
    partition = compute_timing_partition(
        fitpars=["F0", "F1", "DM"],
        sampled_params=["F0", "DM"],
        marginalized_params=["F1"],
        idx_from_fitpars={"F0": [0], "F1": [1], "DM": [2]},
    )

    assert partition.idx_sampled == [0, 2]
    assert partition.idx_marginalized == [1]

    try:
        compute_timing_partition(
            fitpars=["F0", "F1"],
            sampled_params=["F0"],
            marginalized_params=["F0"],
        )
    except ValueError as exc:
        assert "disjoint" in str(exc)
    else:
        raise AssertionError("Expected overlap validation error.")


def test_timing_delta_strict_missing_policy_errors():
    mp = _mock_metapulsar()
    test_param = mp.fitpars[0]
    mp._ensure_delta_engines()

    first_pta = next(iter(mp._delta_engines))
    mp._delta_engines[first_pta].param_names = []

    try:
        mp.timing_delta({test_param: 1.0e-9}, missing_param_policy="strict_error")
    except KeyError as exc:
        assert "unavailable in backend" in str(exc)
    else:
        raise AssertionError("Expected strict missing-parameter error.")


def test_nonlinear_signal_nmat_and_basis_modes_compose_and_evaluate():
    mp = _mock_metapulsar()
    sampled = [mp.fitpars[0]]

    signal_nmat = build_nonlinear_timing_signal(
        engine=mp,
        sampled_params=sampled,
        mode="nmat",
        name="nl_tm_nmat",
    )
    model_nmat = signal_nmat(mp)
    params_nmat = _zero_params(model_nmat)
    delay_nmat = model_nmat.get_delay(params_nmat)
    np.testing.assert_allclose(
        delay_nmat, np.zeros_like(delay_nmat), atol=1e-18, rtol=0.0
    )
    nmat_signal = next(
        sig
        for sig in model_nmat.signals
        if sig.signal_name == "marginalizing linear timing model"
    )
    nmat = nmat_signal.get_ndiag(params_nmat)
    assert nmat.Mmat.shape[1] == len(mp.fitpars) - len(sampled)

    signal_basis = build_nonlinear_timing_signal(
        engine=mp,
        sampled_params=sampled,
        mode="basis",
        name="nl_tm_basis",
    )
    model_basis = signal_basis(mp)
    params_basis = _zero_params(model_basis)
    delay_basis = model_basis.get_delay(params_basis)
    np.testing.assert_allclose(
        delay_basis, np.zeros_like(delay_basis), atol=1e-18, rtol=0.0
    )
    basis_signal = next(
        sig for sig in model_basis.signals if sig.signal_name == "linear timing model"
    )
    basis_matrix = basis_signal.get_basis(params_basis)
    assert basis_matrix.shape[1] == len(mp.fitpars) - len(sampled)


def test_empty_sampled_set_matches_linear_only_behavior():
    mp = _mock_metapulsar()

    built = build_nonlinear_timing_signal(
        engine=mp,
        sampled_params=[],
        mode="nmat",
        name="nl_tm_empty_sampled",
    )
    reference = gp_signals.MarginalizingTimingModel(
        name="nl_tm_empty_sampled_linear_nmat",
        idx_exclude=[],
    )

    model_built = built(mp)
    model_ref = reference(mp)
    params_built = _zero_params(model_built)
    params_ref = _zero_params(model_ref)
    np.testing.assert_allclose(
        model_built.get_ndiag(params_built).Mmat,
        model_ref.get_ndiag(params_ref).Mmat,
        atol=1e-10,
        rtol=0.0,
    )


def test_empty_marginalized_set_is_deterministic_only_and_linear_limit():
    mp = _mock_metapulsar()
    sampled_param = mp.fitpars[0]
    signal = build_nonlinear_timing_signal(
        engine=mp,
        sampled_params=[sampled_param],
        marginalized_params=[],
        mode="nmat",
        name="nl_tm_det_only",
    )
    model = signal(mp)

    param_name = next(
        p.name for p in model.params if p.name.endswith(f"_{sampled_param}")
    )
    epsilon = 1.0e-9
    delay = model.get_delay({param_name: epsilon})
    expected = mp._designmatrix[:, mp.fitpars.index(sampled_param)][mp._isort] * epsilon

    np.testing.assert_allclose(delay, expected, atol=1.0e-18, rtol=0.0)
