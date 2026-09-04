"""Fast-TNT epoch factorization (feature_flexfit_fasttnt.md §10)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pylk.flexfit import (
    AssembledModel,
    BasisBlock,
    DiagonalNoise,
    EpochKernelNoise,
    Factorization,
    ShermanMorrisonNoise,
    VarianceGroup,
    analyze_waveforms,
    assemble,
    ecorr_from_kernel,
    expected_squared_residuals,
    factorize,
    fit_white_noise,
    fourier_pair_groups,
    solve_flexible_phi,
)
from pylk.flexfit.fasttnt import quantize


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _epochs_toas(n_ep: int = 20, per: int = 4, dt: float = 0.1):
    """TOAs with clear multi-TOA epochs (within-epoch gap << dt, between > dt)."""
    toas = []
    t = 0.0
    for e in range(n_ep):
        for j in range(per):
            toas.append(t + 1e-3 * j)
        t += 10.0  # >> dt
    return np.asarray(toas, dtype=float)


def _exactly_factorable_model(rng, *, n_ep=15, per=4, k_fam=6, k_sparse=5, k_dense=3):
    """Synthetic 3-tier basis with known family structure."""
    toas = _epochs_toas(n_ep, per)
    n = toas.size
    epochs = quantize(toas, dt=1.0)
    n_ep_q = int(epochs.max()) + 1
    freqs = rng.uniform(800.0, 2000.0, n)

    # Family I: epoch-constant columns
    bbar_i = rng.normal(size=(n_ep_q, k_fam // 2))
    fam_i = bbar_i[epochs]

    # Family dm_2
    h_dm = (1400.0 / freqs) ** 2
    bbar_dm = rng.normal(size=(n_ep_q, k_fam - k_fam // 2))
    fam_dm = h_dm[:, None] * bbar_dm[epochs]

    # Sparse: one-hot-ish epoch indicators (low fill)
    sparse_cols = []
    for j in range(k_sparse):
        col = np.zeros(n)
        ep = j % n_ep_q
        col[epochs == ep] = 1.0
        sparse_cols.append(col)
    sparse = np.column_stack(sparse_cols)

    # Dense: fast binary variation within epoch
    t = toas
    dense = np.column_stack(
        [
            np.sin(2 * np.pi * t / (0.1 * 86400) + phase)
            for phase in rng.uniform(0, 6, k_dense)
        ]
    )

    matrix = np.hstack([fam_i, fam_dm, sparse, dense])
    k = matrix.shape[1]
    names = tuple(f"c{i}" for i in range(k))
    groups = (
        VarianceGroup("all", tuple(range(k)), lower=1e-20, upper=1e-2, initial=1e-8),
    )
    block = BasisBlock(
        name="joint",
        matrix=matrix,
        coefficient_names=names,
        groups=groups,
        kind="custom",
    )
    # Mark sparse columns as ecorr so they land in the sparse tier by kind.
    # Instead: force fill rule — sparse already has low fill.
    model = assemble((block,))
    return model, toas, freqs, DiagonalNoise(rng.uniform(0.5, 2.0, n) * 1e-12)


# --------------------------------------------------------------------------- #
# T1–T4: quantization / classification
# --------------------------------------------------------------------------- #
def test_t1_quantize_basics():
    toas = np.array([0.0, 0.5, 1.5, 1.6, 10.0])
    bins = quantize(toas, dt=1.0)
    assert bins.tolist() == [0, 0, 1, 1, 2]
    with pytest.raises(ValueError, match="at least one"):
        quantize(np.array([]))


def test_t1_quantize_matches_discovery():
    discovery = pytest.importorskip("discovery")
    rng = np.random.default_rng(0)
    toas = np.sort(rng.uniform(0, 1e8, 500))
    ours = quantize(toas, dt=1.0)
    theirs = discovery.signals.quantize(toas, dt=1.0)
    np.testing.assert_array_equal(ours, theirs)


def test_t2_classification_families():
    rng = np.random.default_rng(1)
    toas = _epochs_toas(10, 3)
    n = toas.size
    epochs = quantize(toas)
    freqs = rng.uniform(900, 1800, n)
    # I, dm_2, fd_1 columns
    col_i = np.ones(n) * 2.0
    col_i = col_i + 0.0 * epochs  # epoch-constant
    b = rng.normal(size=int(epochs.max()) + 1)
    col_i = b[epochs]
    h_dm = (1400.0 / freqs) ** 2
    col_dm = h_dm * b[epochs]
    h_fd = np.log(1000.0 / freqs)
    col_fd = h_fd * b[epochs]
    # low-fill sparse window
    col_sp = np.zeros(n)
    col_sp[: n // 10] = 1.0
    matrix = np.column_stack([col_i, col_dm, col_fd, col_sp])
    block = BasisBlock(
        name="b",
        matrix=matrix,
        coefficient_names=("i", "dm", "fd", "sp"),
        groups=(VarianceGroup("g", (0, 1, 2, 3), lower=1e-20, upper=1.0),),
        kind="custom",
    )
    model = assemble((block,))
    fm = factorize(model, toas=toas, freqs_mhz=freqs, tol=1e-6)
    fam_names = {f.name: f for f in fm.families}
    assert 0 in fam_names["I"].columns
    assert 1 in fam_names["dm_2"].columns
    assert 2 in fam_names["fd_1"].columns
    assert 3 in fm.sparse_columns


def test_t3_negative_fd_wrong_convention():
    rng = np.random.default_rng(2)
    toas = _epochs_toas(8, 3)
    n = toas.size
    epochs = quantize(toas)
    freqs = rng.uniform(900, 1800, n)
    b = rng.normal(size=int(epochs.max()) + 1)
    # log10 / 1400 MHz — must NOT classify as fd
    bad = (np.log10(1400.0 / freqs)) * b[epochs]
    # pad with a dense-looking random column so fill > sparse_max_fill
    junk = rng.normal(size=n)
    matrix = np.column_stack([bad, junk])
    block = BasisBlock(
        name="b",
        matrix=matrix,
        coefficient_names=("bad", "junk"),
        groups=(VarianceGroup("g", (0, 1), lower=1e-20, upper=1.0),),
    )
    model = assemble((block,))
    fm = factorize(model, toas=toas, freqs_mhz=freqs, tol=1e-6, sparse_max_fill=0.0)
    assert 0 in fm.dense_columns


def test_t4_fd_zero_crossing_finite():
    toas = _epochs_toas(5, 4)
    n = toas.size
    epochs = quantize(toas)
    freqs = np.full(n, 1000.0)  # exactly 1 GHz → ln(1000/ν)=0
    freqs[0] = 1400.0
    b = np.arange(int(epochs.max()) + 1, dtype=float)
    h = np.log(1000.0 / freqs)
    col = h * b[epochs]
    matrix = col[:, None]
    block = BasisBlock(
        name="b",
        matrix=matrix,
        coefficient_names=("fd",),
        groups=(VarianceGroup("g", (0,), lower=1e-20, upper=1.0),),
    )
    model = assemble((block,))
    fm = factorize(model, toas=toas, freqs_mhz=freqs, tol=1e-6, sparse_max_fill=0.0)
    # Classification must not raise; column may land dense or fd depending on zeros.
    assert fm.n_coef == 1


# --------------------------------------------------------------------------- #
# T5–T9: Gram / expand parity
# --------------------------------------------------------------------------- #
def test_t5_diagonal_gram_parity_exactly_factorable():
    rng = np.random.default_rng(3)
    model, toas, freqs, noise = _exactly_factorable_model(rng)
    y = rng.normal(size=model.n_obs) * 1e-6
    fm = factorize(model, toas=toas, freqs_mhz=freqs, tol=1e-6, sparse_max_fill=0.25)
    assert fm.families  # at least one family
    g_f, b_f = fm.gram_project(noise, y)
    g_d, b_d = model.gram_project(noise, y)
    # Use substituted basis as the dense oracle for family columns
    tsub = fm.substituted_matrix()
    g_sub = tsub.T @ noise.solve(tsub)
    b_sub = tsub.T @ noise.solve(y)
    np.testing.assert_allclose(g_f, g_sub, rtol=0, atol=1e-13 * np.max(np.abs(g_sub)))
    np.testing.assert_allclose(
        b_f, b_sub, rtol=0, atol=1e-13 * (np.max(np.abs(b_sub)) + 1e-30)
    )
    v = rng.normal(size=model.n_coef)
    np.testing.assert_allclose(
        fm.expand(v), tsub @ v, atol=1e-13 * np.max(np.abs(tsub @ v))
    )


def test_t6_sherman_morrison_downdate():
    rng = np.random.default_rng(4)
    model, toas, freqs, _ = _exactly_factorable_model(rng, k_sparse=3, k_dense=2)
    n = model.n_obs
    epochs = quantize(toas)
    n_ep = int(epochs.max()) + 1
    # Disjoint indicator
    u = np.zeros((n, n_ep))
    u[np.arange(n), epochs] = 1.0
    # Drop some epochs (set those TOAs to no-ECORR by removing columns)
    keep = np.arange(0, n_ep, 2)
    u = u[:, keep]
    jitter = rng.uniform(1e-13, 1e-12, u.shape[1])
    diag = rng.uniform(0.5, 2.0, n) * 1e-12
    noise = ShermanMorrisonNoise(diagonal=diag, u=u, jitter=jitter)
    y = rng.normal(size=n) * 1e-6
    fm = factorize(model, toas=toas, freqs_mhz=freqs)
    g_f, b_f = fm.gram_project(noise, y)
    tsub = fm.substituted_matrix()
    g_d = tsub.T @ noise.solve(tsub)
    b_d = tsub.T @ noise.solve(y)
    scale_g = np.max(np.abs(g_d)) + 1e-30
    scale_b = np.max(np.abs(b_d)) + 1e-30
    np.testing.assert_allclose(g_f, g_d, atol=1e-12 * scale_g)
    np.testing.assert_allclose(b_f, b_d, atol=1e-12 * scale_b)

    # Non-disjoint u raises
    u_bad = u.copy()
    u_bad[0, 0] = 1.0
    if u_bad.shape[1] > 1:
        u_bad[0, 1] = 1.0
        noise_bad = ShermanMorrisonNoise(
            diagonal=diag, u=u_bad, jitter=rng.uniform(1e-13, 1e-12, u_bad.shape[1])
        )
        with pytest.raises(ValueError, match="column-disjoint"):
            fm.gram_project(noise_bad, y)


def test_t7_real_shaped_fourier_tolerance():
    rng = np.random.default_rng(5)
    # Multi-year span, 30 Fourier components at true TOA times
    n_ep, per = 40, 3
    toas = _epochs_toas(n_ep, per)
    # stretch to ~5 years
    toas = toas / toas.max() * (5 * 365.25 * 86400)
    n = toas.size
    freqs_mhz = rng.uniform(800, 2000, n)
    tspan = toas.max() - toas.min()
    cols = []
    for k in range(1, 16):
        cols.append(np.sin(2 * np.pi * k * toas / tspan))
        cols.append(np.cos(2 * np.pi * k * toas / tspan))
    matrix = np.column_stack(cols)
    groups = fourier_pair_groups(
        matrix, prefix="red", n_freq=15, sigma_min=1e-10, sigma_max=1e-4
    )
    block = BasisBlock(
        name="red",
        matrix=matrix,
        coefficient_names=tuple(f"c{i}" for i in range(matrix.shape[1])),
        groups=groups,
        kind="red",
    )
    model = assemble((block,))
    noise = DiagonalNoise(np.full(n, 1e-12))
    y = rng.normal(size=n) * 1e-6
    fm = factorize(model, toas=toas, freqs_mhz=freqs_mhz, tol=1e-6)
    assert any(f.name == "I" for f in fm.families)
    g_f, _ = fm.gram_project(noise, y)
    g_d, _ = model.gram_project(noise, y)
    rel = np.max(np.abs(g_f - g_d)) / (np.max(np.abs(g_d)) + 1e-30)
    assert rel <= 10 * 1e-6

    fm_tight = factorize(model, toas=toas, freqs_mhz=freqs_mhz, tol=1e-12)
    # At tol=1e-12 Fourier columns demote to dense (graceful degradation)
    assert fm_tight.dense_columns.size == model.n_coef or fm_tight.max_error <= 1e-12
    g_t, _ = fm_tight.gram_project(noise, y)
    rel_t = np.max(np.abs(g_t - g_d)) / (np.max(np.abs(g_d)) + 1e-30)
    assert rel_t < 1e-12 or fm_tight.dense_columns.size == model.n_coef


def test_t8_fast_binary_lands_dense():
    rng = np.random.default_rng(6)
    toas = _epochs_toas(12, 4)
    n = toas.size
    pb = 0.1 * 86400  # 0.1 day
    col = np.sin(2 * np.pi * toas / pb)
    # pad with epoch-constant so we have a family too
    epochs = quantize(toas)
    pad = np.ones(n)
    pad = (np.arange(int(epochs.max()) + 1))[epochs].astype(float)
    matrix = np.column_stack([pad, col])
    block = BasisBlock(
        name="b",
        matrix=matrix,
        coefficient_names=("smooth", "bin"),
        groups=(VarianceGroup("g", (0, 1), lower=1e-20, upper=1.0),),
    )
    model = assemble((block,))
    fm = factorize(model, toas=toas, freqs_mhz=None, tol=1e-6, sparse_max_fill=0.0)
    assert 1 in fm.dense_columns
    noise = DiagonalNoise(np.full(n, 1e-12))
    y = rng.normal(size=n)
    g_f, _ = fm.gram_project(noise, y)
    g_d, _ = model.gram_project(noise, y)
    np.testing.assert_allclose(g_f, g_d, atol=1e-13 * (np.max(np.abs(g_d)) + 1e-30))


def test_t9_no_matrix_attribute():
    rng = np.random.default_rng(7)
    model, toas, freqs, _ = _exactly_factorable_model(rng)
    fm = factorize(model, toas=toas, freqs_mhz=freqs)
    with pytest.raises(AttributeError):
        _ = fm.matrix
    tsub = fm.substituted_matrix()
    np.testing.assert_allclose(tsub, fm.expand(np.eye(fm.n_coef)), atol=0.0)
    # Family columns within tol; sparse/dense exact
    orig = model.matrix
    fam_cols = set()
    for fa in fm.families:
        fam_cols.update(int(c) for c in fa.columns)
        scale = np.maximum(np.abs(orig[:, fa.columns]).max(axis=0), 1e-300)
        err = np.abs(orig[:, fa.columns] - tsub[:, fa.columns]).max(axis=0)
        assert np.all(err <= fm.tol * scale + 1e-15)
    for j in list(fm.sparse_columns) + list(fm.dense_columns):
        np.testing.assert_allclose(orig[:, j], tsub[:, j], atol=0.0)


# --------------------------------------------------------------------------- #
# T12: EpochKernelNoise
# --------------------------------------------------------------------------- #
def test_t12_epoch_kernel_noise_oracle():
    rng = np.random.default_rng(8)
    n, n_ep = 40, 7
    epoch = rng.integers(-1, n_ep, size=n)
    d = rng.uniform(0.5, 2.0, n)
    lam = rng.uniform(0.1, 1.0, n_ep)
    noise = EpochKernelNoise(diagonal=d, epoch=epoch, jitter=lam)
    E = np.zeros((n, n_ep))
    E[epoch >= 0, epoch[epoch >= 0]] = 1.0
    N = np.diag(d) + E @ np.diag(lam) @ E.T
    v = rng.normal(size=(n, 3))
    np.testing.assert_allclose(noise.solve(v), np.linalg.solve(N, v), atol=1e-12)
    np.testing.assert_allclose(noise.logdet(), np.linalg.slogdet(N)[1], atol=1e-12)
    dv = noise.diagonal_variance()
    expected_dv = d.copy()
    expected_dv[epoch >= 0] += lam[epoch[epoch >= 0]]
    np.testing.assert_allclose(dv, expected_dv)
    ind = noise.indicator
    assert isinstance(ind, sp.csr_matrix)
    assert ind.nnz == int((epoch >= 0).sum())


def test_t12_from_backends_matches_ecorr_blocks_structure():
    n_ep, per = 30, 4
    toas = _epochs_toas(n_ep, per)
    n = toas.size
    backends = np.array(["A", "B"])[np.arange(n) // (n // 2 + 1) % 2]
    # Ensure both backends appear
    backends[: n // 2] = "A"
    backends[n // 2 :] = "B"
    d = np.full(n, 1e-12)
    ecorr = {"A": 2e-6, "B": 1e-6}
    noise = EpochKernelNoise.from_backends(
        diagonal=d, toas=toas, backend_flags=backends, ecorr=ecorr
    )
    # Every multi-TOA epoch within a backend should be covered
    from pylk.flexfit.adapters import discovery as dx

    class _P:
        pass

    psr = _P()
    psr.toas = toas
    blocks = dx.ecorr_blocks(psr, selection_labels=backends)
    n_cols_a = sum(b.n_col for b in blocks)
    n_valid = int((noise.epoch >= 0).sum())
    # Same number of epoch columns / assigned TOAs structure
    assert noise.jitter.size == n_cols_a
    assert n_valid == sum(int(b.matrix.sum()) for b in blocks)


def test_from_backends_selection_mismatch():
    toas = _epochs_toas(5, 3)
    n = toas.size
    labels = np.array(["pta"] * n)
    with pytest.raises(ValueError, match="selection mismatch|absent"):
        EpochKernelNoise.from_backends(
            diagonal=np.ones(n),
            toas=toas,
            backend_flags=labels,
            ecorr={"fine_backend": 1e-6},
        )


# --------------------------------------------------------------------------- #
# F1–F2, F6: solver integration
# --------------------------------------------------------------------------- #
def test_f1_solve_dense_vs_factored():
    rng = np.random.default_rng(10)
    model, toas, freqs, noise = _exactly_factorable_model(rng)
    y = rng.normal(size=model.n_obs) * 1e-6
    fm = factorize(model, toas=toas, freqs_mhz=freqs)
    # Compare on substituted basis for both (fair)
    tsub = fm.substituted_matrix()
    names = tuple(f"c{i}" for i in range(tsub.shape[1]))
    block = BasisBlock(
        name="joint",
        matrix=tsub,
        coefficient_names=names,
        groups=model.groups,
        kind="custom",
    )
    dense_sub = assemble((block,))
    r_d = solve_flexible_phi(y, dense_sub, noise, n_sweeps=3)
    r_f = solve_flexible_phi(y, fm, noise, n_sweeps=3)
    np.testing.assert_allclose(r_d.phi_history, r_f.phi_history, atol=1e-12)


def test_f2_gram_cached_once_per_call():
    rng = np.random.default_rng(11)
    model, toas, freqs, noise = _exactly_factorable_model(rng)
    y = rng.normal(size=model.n_obs) * 1e-6
    fm = factorize(model, toas=toas, freqs_mhz=freqs)
    import pylk.flexfit.flexible_phi as fp

    real = fp._gram_project
    counter = {"n": 0}

    def wrapped(model_, noise_, y_):
        counter["n"] += 1
        return real(model_, noise_, y_)

    fp._gram_project = wrapped
    try:
        solve_flexible_phi(y, fm, noise, n_sweeps=4)
    finally:
        fp._gram_project = real
    assert counter["n"] == 1


def test_f6_block_waveforms_via_expand():
    rng = np.random.default_rng(12)
    model, toas, freqs, noise = _exactly_factorable_model(rng)
    y = rng.normal(size=model.n_obs) * 1e-6
    fm = factorize(model, toas=toas, freqs_mhz=freqs)
    r_f = solve_flexible_phi(y, fm, noise, n_sweeps=2)
    for name, wave in r_f.block_waveforms.items():
        span = r_f.block_spans[name]
        expected = fm.expand(r_f.coefficient_mean, span=span)
        np.testing.assert_allclose(wave, expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# F4, F7: expected_squared_residuals
# --------------------------------------------------------------------------- #
def test_f4_expected_squared_residuals_fast_vs_dense():
    rng = np.random.default_rng(13)
    model, toas, freqs, noise = _exactly_factorable_model(rng)
    y = rng.normal(size=model.n_obs) * 1e-6
    fm = factorize(model, toas=toas, freqs_mhz=freqs)
    solve = solve_flexible_phi(y, fm, noise, n_sweeps=2)
    e_f = expected_squared_residuals(y, fm, solve)
    e_d = expected_squared_residuals(y, fm.substituted_matrix(), solve)
    np.testing.assert_allclose(e_f, e_d, rtol=1e-10)


def test_f7_chunking_agreement(monkeypatch):
    rng = np.random.default_rng(14)
    model, toas, freqs, noise = _exactly_factorable_model(rng, k_fam=8)
    y = rng.normal(size=model.n_obs) * 1e-6
    solve = solve_flexible_phi(y, model, noise, n_sweeps=2)
    import pylk.flexfit.whitenoise as wn

    results = []
    for chunk in (1, 7, model.n_coef):
        monkeypatch.setattr(wn, "CHUNK", chunk)
        results.append(expected_squared_residuals(y, model, solve))
    np.testing.assert_allclose(results[0], results[1], atol=1e-14)
    np.testing.assert_allclose(results[0], results[2], atol=1e-14)


# --------------------------------------------------------------------------- #
# F3, F5, F8: fit_white_noise + Factorization
# --------------------------------------------------------------------------- #
def test_f3_fit_white_noise_factorization_parity():
    rng = np.random.default_rng(15)
    model, toas, freqs, _ = _exactly_factorable_model(
        rng, k_fam=4, k_sparse=2, k_dense=1
    )
    # Use the red-like block alone for WN fit
    n = model.n_obs
    toaerrs = rng.uniform(0.8e-6, 1.2e-6, n)
    backends = np.array(["A", "B"])[np.arange(n) % 2]
    # Build a simple Fourier block from the family part
    t = toas
    tspan = t.max() - t.min() + 1.0
    F = np.column_stack(
        [
            np.sin(2 * np.pi * t / tspan),
            np.cos(2 * np.pi * t / tspan),
            np.sin(4 * np.pi * t / tspan),
            np.cos(4 * np.pi * t / tspan),
        ]
    )
    groups = fourier_pair_groups(
        F, prefix="red", n_freq=2, sigma_min=1e-9, sigma_max=1e-4
    )
    block = BasisBlock(
        name="red",
        matrix=F,
        coefficient_names=tuple(f"c{i}" for i in range(4)),
        groups=groups,
        kind="red",
    )
    y = rng.normal(size=n) * toaerrs * 1.2
    cold = fit_white_noise(
        toaerrs=toaerrs,
        backend_flags=backends,
        blocks=[block],
        residuals=y,
        fit_equad=False,
        max_iterations=8,
    )
    fact = Factorization(toas=toas, freqs_mhz=None, mode="fast")
    hot = fit_white_noise(
        toaerrs=toaerrs,
        backend_flags=backends,
        blocks=[block],
        residuals=y,
        fit_equad=False,
        max_iterations=8,
        factorization=fact,
    )
    for b in ("A", "B"):
        assert cold.efac[b] == pytest.approx(hot.efac[b], rel=1e-6)


def test_f5_auto_mode_falls_back_on_no_multiplicity():
    # One TOA per epoch, no sparse → predicted speedup may be < 1.5
    n = 30
    toas = np.arange(n, dtype=float) * 10.0  # all isolated
    matrix = np.eye(n)[:, :5]  # dense-ish columns
    # make columns dense (full fill)
    matrix = np.ones((n, 5))
    matrix += np.arange(5)[None, :]
    # within-epoch variation forced by unique toas — I family still works
    block = BasisBlock(
        name="b",
        matrix=matrix,
        coefficient_names=tuple(f"c{i}" for i in range(5)),
        groups=(VarianceGroup("g", tuple(range(5)), lower=1e-20, upper=1.0),),
    )
    model = assemble((block,))
    fact = Factorization(toas=toas, freqs_mhz=None, mode="auto", min_speedup=1e9)
    out = fact.apply(model)
    assert isinstance(out, AssembledModel)


def test_f8_warm_start_fewer_iterations():
    rng = np.random.default_rng(16)
    n = 900
    t = np.sort(rng.uniform(0, 1, n))
    cols = []
    for k in range(1, 6):
        cols.append(np.sin(2 * np.pi * k * t))
        cols.append(np.cos(2 * np.pi * k * t))
    F = np.column_stack(cols)
    groups = fourier_pair_groups(
        F, prefix="red", n_freq=5, sigma_min=1e-9, sigma_max=1e-4
    )
    block = BasisBlock(
        name="red",
        matrix=F,
        coefficient_names=tuple(f"c{i}" for i in range(10)),
        groups=groups,
        kind="red",
    )
    toaerrs = rng.uniform(0.8e-6, 1.2e-6, n)
    backends = np.array(["X", "Y", "Z"])[np.arange(n) % 3]
    # Truth far from the (1, 0) cold start
    efac_true = {"X": 2.2, "Y": 0.55, "Z": 1.8}
    y = np.empty(n)
    for b, f in efac_true.items():
        m = backends == b
        y[m] = rng.standard_normal(int(m.sum())) * toaerrs[m] * f
    cold = fit_white_noise(
        toaerrs=toaerrs,
        backend_flags=backends,
        blocks=[block],
        residuals=y,
        fit_equad=False,
        max_iterations=25,
        tolerance=1e-5,
    )
    warm = fit_white_noise(
        toaerrs=toaerrs,
        backend_flags=backends,
        blocks=[block],
        residuals=y,
        fit_equad=False,
        max_iterations=25,
        tolerance=1e-5,
        initial_efac=dict(cold.efac),
        initial_equad=dict(cold.equad),
    )
    assert warm.iterations < cold.iterations
    for b in ("X", "Y", "Z"):
        assert warm.efac[b] == pytest.approx(cold.efac[b], rel=1e-4)


# --------------------------------------------------------------------------- #
# T10 / F10: Topology A vs B parity
# --------------------------------------------------------------------------- #
def _t10_problem(rng):
    n_ep, per = 40, 4
    toas = _epochs_toas(n_ep, per)
    n = toas.size
    backends = np.where(
        (np.arange(n) // per) % 3 == 0,
        "A",
        np.where((np.arange(n) // per) % 3 == 1, "B", "C"),
    )
    toaerrs = rng.uniform(0.8e-6, 1.2e-6, n)
    efac_true = {"A": 1.1, "B": 0.9, "C": 1.3}
    ecorr_true = {"A": 2.0e-6, "B": 1.5e-6}  # C has no ECORR
    epochs = quantize(toas)
    y = rng.standard_normal(n) * toaerrs
    for b, f in efac_true.items():
        y[backends == b] *= f
    # Inject ECORR on A/B multi-TOA epochs
    for b, amp in ecorr_true.items():
        mask = backends == b
        for e in np.unique(epochs[mask]):
            m = mask & (epochs == e)
            if m.sum() > 1:
                y[m] += rng.normal() * amp
    # Shared Fourier block
    tspan = toas.max() - toas.min() + 1.0
    cols = []
    for k in range(1, 4):
        cols.append(np.sin(2 * np.pi * k * toas / tspan))
        cols.append(np.cos(2 * np.pi * k * toas / tspan))
    F = np.column_stack(cols)
    groups = fourier_pair_groups(
        F, prefix="red", n_freq=3, sigma_min=1e-10, sigma_max=1e-4
    )
    red = BasisBlock(
        name="red",
        matrix=F,
        coefficient_names=tuple(f"c{i}" for i in range(6)),
        groups=groups,
        kind="red",
    )
    return toas, backends, toaerrs, y, red, ecorr_true, efac_true


def test_t10_topology_parity_and_bias():
    rng = np.random.default_rng(17)
    toas, backends, toaerrs, y, red, ecorr_true, efac_true = _t10_problem(rng)
    from pylk.flexfit.adapters import discovery as dx

    class _P:
        pass

    psr = _P()
    psr.toas = toas
    ecorr_blocks = dx.ecorr_blocks(psr, selection_labels=backends)
    # Pin ECORR group variances at truth (only backends that carry ECORR)
    pinned = []
    for blk in ecorr_blocks:
        backend = blk.metadata["backend"]
        if backend not in ecorr_true:
            continue
        lam = ecorr_true[backend] ** 2
        g = VarianceGroup(
            blk.groups[0].name,
            blk.groups[0].indices,
            lower=lam,
            upper=lam,
            initial=lam,
            update_from_sweep=10**9,
        )
        pinned.append(
            BasisBlock(
                name=blk.name,
                matrix=blk.matrix,
                coefficient_names=blk.coefficient_names,
                groups=(g,),
                kind="ecorr",
                metadata=dict(blk.metadata),
            )
        )
    # Topology A
    res_a = fit_white_noise(
        toaerrs=toaerrs,
        backend_flags=backends,
        blocks=[red, *pinned],
        residuals=y,
        fit_equad=False,
        max_iterations=12,
        tolerance=1e-4,
    )
    # Topology B
    d0 = toaerrs**2
    kernel = EpochKernelNoise.from_backends(
        diagonal=d0, toas=toas, backend_flags=backends, ecorr=ecorr_true
    )
    res_b = fit_white_noise(
        toaerrs=toaerrs,
        backend_flags=backends,
        blocks=[red],
        residuals=y,
        fit_equad=False,
        max_iterations=12,
        tolerance=1e-4,
        kernel_ecorr=kernel,
    )
    for b in ("A", "B", "C"):
        assert res_a.efac[b] == pytest.approx(res_b.efac[b], rel=1e-6, abs=1e-10)

    # Negative: uncorrected e_i biases EFAC high on ECORR backends
    from dataclasses import replace

    solve_b = res_b.solve
    assert solve_b is not None
    noise_b = replace(kernel, diagonal=res_b.variance)
    e_corr = expected_squared_residuals(y, assemble((red,)), solve_b, noise=noise_b)
    e_raw = expected_squared_residuals(y, assemble((red,)), solve_b, noise=None)
    for b in ("A", "B"):
        m = backends == b
        # Uncorrected mean e/sigma^2 is larger → larger EFAC
        assert np.mean(e_raw[m] / toaerrs[m] ** 2) > np.mean(
            e_corr[m] / toaerrs[m] ** 2
        )


def test_t11_learned_kernel_ecorr_matches_topology_a():
    """§3.7: one EM update of kernel jitter matches Topology-A group update.

    At a shared prior ``λ`` and fixed ``D``, ``mean(E[a_e^2])`` equals the
    flexible-``Φ`` ecorr-group second-moment mean to machine precision — the
    same-fixed-point claim. ``fit_white_noise(..., learn_kernel_ecorr=True)``
    exercises the driver path (mode 2.2).
    """
    from dataclasses import replace

    from pylk.flexfit.adapters import discovery as dx
    from pylk.flexfit.flexible_phi import (
        _gram_project,
        _initial_phi,
        _moments_from_gram,
        bounded_variance_update,
    )
    from pylk.flexfit.whitenoise import _kernel_ecorr_m_step, kernel_ecorr_moments

    rng = np.random.default_rng(21)
    toas, backends, toaerrs, y, red, ecorr_true, efac_true = _t10_problem(rng)

    class _P:
        pass

    psr = _P()
    psr.toas = toas
    ecorr_blocks = [
        blk
        for blk in dx.ecorr_blocks(psr, selection_labels=backends)
        if blk.metadata["backend"] in ecorr_true
    ]
    variance = np.empty_like(toaerrs)
    for b, f in efac_true.items():
        variance[backends == b] = (f * toaerrs[backends == b]) ** 2
    lam0 = {b: 1.0e-6 for b in ecorr_true}

    # Topology A — ecorr groups initialized at λ₀; one E-step then group M-step.
    pinned = []
    for blk in ecorr_blocks:
        backend = blk.metadata["backend"]
        lam = lam0[backend] ** 2
        g = VarianceGroup(
            blk.groups[0].name,
            blk.groups[0].indices,
            lower=1e-18,
            upper=1e-10,
            initial=lam,
        )
        pinned.append(
            BasisBlock(
                name=blk.name,
                matrix=blk.matrix,
                coefficient_names=blk.coefficient_names,
                groups=(g,),
                kind="ecorr",
                metadata=dict(blk.metadata),
            )
        )
    model_a = assemble((red, *pinned))
    phi_a = _initial_phi(model_a)
    gram_a, proj_a = _gram_project(model_a, DiagonalNoise(variance), y)
    _, _, sm_a = _moments_from_gram(gram_a, proj_a, phi_a)
    ecorr_a = {}
    for group in model_a.groups:
        if group.name.startswith("ecorr_"):
            rho, _ = bounded_variance_update(sm_a, group)
            ecorr_a[group.name[len("ecorr_") :]] = float(np.sqrt(rho))

    # Topology B — same λ₀ in the kernel; one E-step then §3.7 M-step.
    kernel = EpochKernelNoise.from_backends(
        diagonal=variance, toas=toas, backend_flags=backends, ecorr=lam0
    )
    model_b = assemble((red,))
    phi_b = _initial_phi(model_b)
    gram_b, proj_b = _gram_project(model_b, kernel, y)
    mean_b, cov_b, _ = _moments_from_gram(gram_b, proj_b, phi_b)

    class _Solve:
        coefficient_mean = mean_b
        coefficient_covariance = cov_b

    moments = kernel_ecorr_moments(y, model_b, _Solve, kernel)
    new_jitter = _kernel_ecorr_m_step(moments, kernel, ecorr_min=1e-9, ecorr_max=1e-5)
    ecorr_b = ecorr_from_kernel(replace(kernel, jitter=new_jitter))
    for b in ecorr_true:
        assert ecorr_b[b] == pytest.approx(ecorr_a[b], rel=1e-10, abs=0.0)
        assert dx.ecorr_from_kernel(replace(kernel, jitter=new_jitter))[
            b
        ] == pytest.approx(ecorr_b[b])

    # Driver path (mode 2.2) learns ECORR and recovers amplitudes near truth.
    res = fit_white_noise(
        toaerrs=toaerrs,
        backend_flags=backends,
        blocks=[red],
        residuals=y,
        fit_equad=False,
        max_iterations=15,
        tolerance=1e-5,
        kernel_ecorr=EpochKernelNoise.from_backends(
            diagonal=toaerrs**2,
            toas=toas,
            backend_flags=backends,
            ecorr=lam0,
        ),
        learn_kernel_ecorr=True,
        initial_efac=efac_true,
    )
    assert res.kernel is not None
    ecorr_fit = ecorr_from_kernel(res.kernel)
    for b in ecorr_true:
        assert ecorr_fit[b] == pytest.approx(ecorr_true[b], rel=0.35)


def test_f10_kernel_waveform_wiring():
    rng = np.random.default_rng(18)
    toas, backends, toaerrs, y, red, ecorr_true, _ = _t10_problem(rng)
    d = np.empty_like(toaerrs)
    for b in np.unique(backends):
        d[backends == b] = (1.0 * toaerrs[backends == b]) ** 2
    kernel = EpochKernelNoise.from_backends(
        diagonal=d, toas=toas, backend_flags=backends, ecorr=ecorr_true
    )
    model = assemble((red,))
    solve = solve_flexible_phi(y, model, kernel, n_sweeps=2)
    mjd = toas / 86400.0
    kinds = dict(model.block_kinds)
    # (a) with noise=
    a_with = analyze_waveforms(
        y, d, solve, toas=toas, toa_mjd=mjd, block_kinds=kinds, noise=kernel
    )
    assert "ecorr_kernel" in a_with.waveforms
    assert a_with.summary()["blocks"]["ecorr_kernel"]["n_coef"] is None
    with pytest.raises(KeyError):
        a_with.predict_gp("ecorr_kernel", toas)
    # (d) variance is D
    np.testing.assert_allclose(a_with.variance, d)
    # (b) without noise= whitened RMS higher
    a_without = analyze_waveforms(
        y, d, solve, toas=toas, toa_mjd=mjd, block_kinds=kinds, noise=None
    )
    assert a_without.stage("whitened").rms > a_with.stage("whitened").rms


def test_f9_waveform_analysis_factored():
    rng = np.random.default_rng(19)
    model, toas, freqs, noise = _exactly_factorable_model(rng)
    y = rng.normal(size=model.n_obs) * 1e-6
    fm = factorize(model, toas=toas, freqs_mhz=freqs)
    tsub = fm.substituted_matrix()
    dense = assemble(
        (
            BasisBlock(
                name="joint",
                matrix=tsub,
                coefficient_names=tuple(f"c{i}" for i in range(tsub.shape[1])),
                groups=model.groups,
                kind="custom",
            ),
        )
    )
    r_d = solve_flexible_phi(y, dense, noise, n_sweeps=2)
    r_f = solve_flexible_phi(y, fm, noise, n_sweeps=2)
    mjd = toas / 86400.0
    a_d = analyze_waveforms(
        y, noise.variance, r_d, toas=toas, toa_mjd=mjd, block_kinds=dense.block_kinds
    )
    a_f = analyze_waveforms(
        y, noise.variance, r_f, toas=toas, toa_mjd=mjd, block_kinds=fm.block_kinds
    )
    for s in a_d.stages:
        assert s.rms == pytest.approx(a_f.stage(s.name).rms, rel=0, abs=1e-12)


def test_report_and_predicted_speedup():
    rng = np.random.default_rng(20)
    model, toas, freqs, _ = _exactly_factorable_model(rng)
    fm = factorize(model, toas=toas, freqs_mhz=freqs)
    text = fm.report()
    assert "FactoredModel:" in text
    assert "predicted speedup" in text
    assert fm.predicted_speedup > 0
    assert fm.predicted_end_to_end_speedup > 0


# --------------------------------------------------------------------------- #
# Review follow-ups: kernel products without rebuilding T, and silent-bias guards
# --------------------------------------------------------------------------- #
def _kernel_products_problem(rng):
    """A 3-tier basis plus a disjoint epoch kernel."""
    n_ep, per = 24, 4
    toas = _epochs_toas(n_ep, per)
    n = toas.size
    freqs = np.tile([700.0, 1400.0, 2100.0, 3000.0], n // 4 + 1)[:n]
    span = toas.max() - toas.min()
    smooth = np.column_stack([np.ones(n), toas / span, np.sin(2 * np.pi * toas / span)])
    chroma = (1400.0 / freqs)[:, None] ** 2 * np.column_stack(
        [np.ones(n), np.cos(2 * np.pi * toas / span)]
    )
    windows = np.zeros((n, 3))
    for j in range(3):
        windows[j * (n // 3) : (j + 1) * (n // 3), j] = 1.0
    wobble = np.column_stack([np.sin(2 * np.pi * toas / 3.0)])  # sub-epoch -> dense
    block = BasisBlock(
        name="mixed",
        matrix=np.hstack([smooth, chroma, windows, wobble]),
        coefficient_names=tuple(f"c{i}" for i in range(9)),
        groups=(VarianceGroup("mixed", tuple(range(9)), lower=1e-30, upper=1e10),),
        kind="custom",
    )
    model = assemble((block,))
    factored = factorize(model, toas=toas, freqs_mhz=freqs)
    epoch = np.repeat(np.arange(n_ep), per)
    noise = EpochKernelNoise(
        diagonal=rng.uniform(0.5, 2.0, n),
        epoch=epoch,
        jitter=rng.uniform(0.1, 1.0, n_ep),
    )
    return model, factored, noise


def test_kernel_weights_and_row_dot_match_the_explicit_oracle():
    """W and the cross-term contraction agree with dense algebra on T (and T̃)."""
    rng = np.random.default_rng(21)
    model, factored, noise = _kernel_products_problem(rng)
    e_ind = noise.indicator
    d = 1.0 / noise.diagonal
    e_dense = e_ind.toarray()

    for container, matrix in (
        (model, model.matrix),
        (factored, factored.substituted_matrix()),
    ):
        w = container.kernel_weights(e_ind, d)
        np.testing.assert_allclose(w, e_dense.T @ (d[:, None] * matrix), atol=1e-12)

        g = rng.normal(size=(noise.jitter.size, model.n_coef))
        got = container.epoch_row_dot(g, noise.epoch)
        want = np.einsum("ij,ij->i", matrix, g[noise.epoch])
        np.testing.assert_allclose(got, want, atol=1e-12)


def test_epoch_row_dot_zeroes_toas_outside_any_epoch():
    rng = np.random.default_rng(22)
    model, factored, noise = _kernel_products_problem(rng)
    epoch = np.array(noise.epoch, dtype=np.int64, copy=True)
    epoch[::5] = -1
    g = rng.normal(size=(noise.jitter.size, model.n_coef))
    for container, matrix in (
        (model, model.matrix),
        (factored, factored.substituted_matrix()),
    ):
        got = container.epoch_row_dot(g, epoch)
        want = np.einsum("ij,ij->i", matrix, g[epoch.clip(0)]) * (epoch >= 0)
        np.testing.assert_allclose(got, want, atol=1e-12)


def test_kernel_moments_never_materialize_the_design_matrix(monkeypatch):
    """The §3.6 statistic must not rebuild T column by column (O(n k^2))."""
    rng = np.random.default_rng(23)
    model, factored, noise = _kernel_products_problem(rng)
    y = rng.normal(size=model.n_obs) * 1e-6
    solve = solve_flexible_phi(y, factored, noise, n_sweeps=2)

    columns: list[int] = []
    original = type(factored).expand

    def counting_expand(self, v, *, span=None):
        arr = np.asarray(v)
        columns.append(1 if arr.ndim == 1 else arr.shape[1])
        return original(self, v, span=span)

    monkeypatch.setattr(type(factored), "expand", counting_expand)
    expected_squared_residuals(y, factored, solve, noise=noise)
    # One pass over chol(Sigma) (k columns) plus the (k,) coefficient mean.
    # Rebuilding T for W and for the cross term would cost 2k columns more.
    assert sum(columns) == model.n_coef + 1


def test_fit_white_noise_rejects_kernel_ecorr_without_a_basis():
    rng = np.random.default_rng(24)
    _, _, noise = _kernel_products_problem(rng)
    n = noise.n_obs
    with pytest.raises(ValueError, match="kernel_ecorr needs a basis"):
        fit_white_noise(
            toaerrs=np.full(n, 1e-6),
            backend_flags=np.array(["A"] * n),
            residuals=rng.normal(size=n) * 1e-6,
            kernel_ecorr=noise,
        )


def test_white_noise_adapter_reports_selection_mismatch():
    from pylk.flexfit.adapters import discovery as dx

    class _P:
        name = "J0000+0000"
        toas = _epochs_toas(6, 3)
        toaerrs = np.full(18, 1e-6)
        backend_flags = np.array(["fine_backend"] * 18)

    noisedict = {"J0000+0000_epta_dr2_efac": 1.0}
    with pytest.raises(ValueError, match="selection mismatch"):
        dx.white_noise(_P(), noisedict)


def test_predicted_speedup_is_calibrated_not_a_raw_flop_ratio():
    from pylk.flexfit.fasttnt import FAST_FLOP_PENALTY

    rng = np.random.default_rng(25)
    _, factored, _ = _kernel_products_problem(rng)
    c_dense, c_fast = factored._cost_terms()
    assert factored.predicted_speedup == pytest.approx(c_dense / c_fast)
    assert FAST_FLOP_PENALTY > 1.0  # fast-path flops are not BLAS flops
