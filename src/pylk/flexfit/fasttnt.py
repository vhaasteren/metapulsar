"""Fast-TNT epoch factorization for flexfit Gram / projection products.

Classifies assembled basis columns into sparse, family-factored, and dense
tiers so ``T^T N^{-1} T`` and ``T^T N^{-1} y`` evaluate in
``O(n k_fam^2 + n_ep k^2 + nnz)`` rather than ``O(n k^2)``. NumPy/SciPy only;
see ``feature_flexfit_fasttnt.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import scipy.sparse as sp

from .basis import AssembledModel
from .noise import DiagonalNoise, EpochKernelNoise, NoiseOperator, ShermanMorrisonNoise


# Fast-path work (bincount / sparse products / gathers) runs well below the
# BLAS efficiency of the dense Gram it replaces. Calibrated on J1713+0747
# (n = 71578, k = 221, n_ep = 5406): raw flop ratio 40x, measured wall-clock
# 4.1x diagonal / 5.9x kernel. Keeps `predicted_speedup` inside the +-2-3x the
# design note claims instead of overselling by an order of magnitude.
FAST_FLOP_PENALTY = 8.0


def quantize(toas: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Epoch ids for TOAs (seconds), matching ``discovery.signals.quantize``."""
    toas = np.asarray(toas, dtype=float)
    if toas.size == 0:
        raise ValueError("quantize needs at least one TOA")
    isort = np.argsort(toas, kind="stable")
    bins = np.zeros(toas.shape, dtype=np.int64)
    b, v = 0, float(toas[isort[0]])
    for j in isort:
        if toas[j] - v > dt:
            v, b = float(toas[j]), b + 1
        bins[j] = b
    return bins


@dataclass(frozen=True)
class PrefactorFamily:
    """A per-TOA multiplicative prefactor ``h``; columns ``t ≈ h ⊙ (U b̄)``."""

    name: str
    values: np.ndarray  # (n,)

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=float)
        arr.setflags(write=False)
        object.__setattr__(self, "values", arr)


@dataclass(frozen=True)
class FamilyGroup:
    """One classified family with its epoch-level basis."""

    name: str
    prefactor: np.ndarray  # (n,)
    columns: np.ndarray  # global column indices
    bbar: np.ndarray  # (n_ep, k_f)
    max_error: float

    def __post_init__(self) -> None:
        for name in ("prefactor", "columns", "bbar"):
            arr = np.asarray(getattr(self, name))
            if name != "columns":
                arr = np.asarray(arr, dtype=float)
            else:
                arr = np.asarray(arr, dtype=np.int64)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


def _build_families(
    freqs_mhz: np.ndarray | None,
    model: AssembledModel,
    extra: Sequence[PrefactorFamily],
) -> list[PrefactorFamily]:
    n = model.n_obs
    families: list[PrefactorFamily] = [
        PrefactorFamily("I", np.ones(n, dtype=float)),
    ]
    if freqs_mhz is not None:
        nu = np.asarray(freqs_mhz, dtype=float)
        if nu.shape != (n,):
            raise ValueError(f"freqs_mhz shape {nu.shape} does not match n_obs={n}")
        for p in range(1, 5):
            families.append(PrefactorFamily(f"dm_{p}", (1400.0 / nu) ** p))
        # tempo2/PINT/JUG FD: natural log with 1 GHz reference
        ldm = np.log(1000.0 / nu)
        for q in range(1, 7):
            families.append(PrefactorFamily(f"fd_{q}", ldm**q))
        # Per-block chromatic hints (alpha, fref) not covered by dm_p defaults.
        seen: set[tuple[float, float]] = set()
        for meta in model.block_metadata.values():
            if "alpha" not in meta or "fref" not in meta:
                continue
            alpha = float(meta["alpha"])
            fref = float(meta["fref"])
            if alpha == int(alpha) and 1 <= int(alpha) <= 4 and fref == 1400.0:
                continue
            key = (alpha, fref)
            if key in seen:
                continue
            seen.add(key)
            name = f"pl_{alpha:g}_{fref:g}"
            families.append(PrefactorFamily(name, (fref / nu) ** alpha))
    families.extend(extra)
    return families


def _fill_pair(
    gram: np.ndarray, rows: np.ndarray, cols: np.ndarray, block: np.ndarray
) -> None:
    """Scatter an off-diagonal tier block and its transpose."""
    gram[np.ix_(rows, cols)] = block
    gram[np.ix_(cols, rows)] = block.T


def _indicator(noise: NoiseOperator) -> sp.csr_matrix:
    """Column-disjoint epoch indicator as CSR, for either kernel operator."""
    if isinstance(noise, EpochKernelNoise):
        return noise.indicator
    if not isinstance(noise, ShermanMorrisonNoise):
        raise TypeError(f"unsupported kernel noise type {type(noise)!r}")
    e = sp.csr_matrix(noise.u)
    if np.diff(e.indptr).max(initial=0) > 1:
        raise ValueError(
            "fast-TNT needs column-disjoint ShermanMorrisonNoise.u (one ECORR "
            "epoch per TOA); use the dense AssembledModel path instead"
        )
    return e


@dataclass(frozen=True)
class FactoredModel:
    """Epoch-factorized view of an AssembledModel (fast ``T^T N^-1 T`` / ``T v``).

    Satisfies the solver-facing :class:`~pylk.flexfit.basis.LinearModel`
    protocol. Immutable and noise-independent: bind a new noise operator by
    calling :meth:`gram_project` again.
    """

    model: AssembledModel
    epochs: np.ndarray
    n_epochs: int
    dt: float
    tol: float
    families: tuple[FamilyGroup, ...]
    sparse: sp.csr_matrix | None
    sparse_columns: np.ndarray
    dense: np.ndarray | None
    dense_columns: np.ndarray
    max_error: float
    _u: sp.csr_matrix = field(repr=False)

    def __post_init__(self) -> None:
        for name in ("epochs", "sparse_columns", "dense_columns"):
            arr = np.asarray(getattr(self, name), dtype=np.int64)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        if self.dense is not None:
            d = np.asarray(self.dense, dtype=float)
            d.setflags(write=False)
            object.__setattr__(self, "dense", d)

    # --- AssembledModel delegation (deliberately WITHOUT `matrix`) --------
    @property
    def groups(self):
        return self.model.groups

    @property
    def block_spans(self):
        return self.model.block_spans

    @property
    def block_kinds(self):
        return self.model.block_kinds

    @property
    def block_metadata(self):
        return self.model.block_metadata

    @property
    def coefficient_names(self):
        return self.model.coefficient_names

    @property
    def column_kinds(self):
        return self.model.column_kinds

    @property
    def n_obs(self) -> int:
        return self.model.n_obs

    @property
    def n_coef(self) -> int:
        return self.model.n_coef

    def gram_project(
        self, noise: NoiseOperator, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(T^T N^-1 T, T^T N^-1 y)`` in the assembled global column order."""
        y = np.asarray(y, dtype=float)
        if isinstance(noise, DiagonalNoise):
            return self._gram_project_diagonal(1.0 / noise.variance, y)
        if isinstance(noise, (ShermanMorrisonNoise, EpochKernelNoise)):
            d = 1.0 / noise.diagonal
            gram, proj = self._gram_project_diagonal(d, y)
            self._sherman_morrison_downdate(noise, d, y, gram, proj)
            return gram, proj
        tsub = self.substituted_matrix()
        return tsub.T @ noise.solve(tsub), tsub.T @ noise.solve(y)

    def _gram_project_diagonal(
        self, d: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        eps, n_ep, u = self.epochs, self.n_epochs, self._u
        gram = np.zeros((self.n_coef, self.n_coef))
        proj = np.zeros(self.n_coef)
        dy = d * y
        for i, fa in enumerate(self.families):
            proj[fa.columns] = fa.bbar.T @ np.bincount(
                eps, weights=fa.prefactor * dy, minlength=n_ep
            )
            for fb in self.families[i:]:
                c = np.bincount(
                    eps, weights=fa.prefactor * fb.prefactor * d, minlength=n_ep
                )
                block = fa.bbar.T @ (c[:, None] * fb.bbar)
                if fb is fa:
                    gram[np.ix_(fa.columns, fa.columns)] = block
                else:
                    _fill_pair(gram, fa.columns, fb.columns, block)
        s, r = self.sparse, self.dense
        if s is not None:
            js = self.sparse_columns
            proj[js] = s.T @ dy
            gram[np.ix_(js, js)] = (s.T @ s.multiply(d[:, None])).toarray()
            for fa in self.families:
                z = (u.T @ s.multiply((fa.prefactor * d)[:, None])).toarray()
                _fill_pair(gram, fa.columns, js, fa.bbar.T @ z)
        if r is not None:
            jr = self.dense_columns
            proj[jr] = r.T @ dy
            rd = d[:, None] * r
            gram[np.ix_(jr, jr)] = r.T @ rd
            for fa in self.families:
                w = np.asarray(u.T @ (fa.prefactor[:, None] * rd))
                _fill_pair(gram, fa.columns, jr, fa.bbar.T @ w)
            if s is not None:
                _fill_pair(gram, self.sparse_columns, jr, np.asarray(s.T @ rd))
        return gram, proj

    def _sherman_morrison_downdate(
        self,
        noise: ShermanMorrisonNoise | EpochKernelNoise,
        d: np.ndarray,
        y: np.ndarray,
        gram: np.ndarray,
        proj: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the epoch-space capacitance downdate in place; return ``(W, s)``."""
        e = _indicator(noise)
        s = getattr(noise, "capacitance_scale", None)
        if s is None:
            t = np.asarray(e.multiply(e).T @ d).ravel()
            s = 1.0 / (1.0 / noise.jitter + t)
        w = self.kernel_weights(e, d)
        q = np.asarray(e.T @ (d * y)).ravel()
        gram -= w.T @ (s[:, None] * w)
        proj -= w.T @ (s * q)
        return w, np.asarray(s, dtype=float)

    def kernel_weights(self, indicator, d: np.ndarray) -> np.ndarray:
        """``W = E^T diag(d) T̃``, ``(n_ep, k)``, assembled per tier.

        Tier-wise so no column of ``T̃`` is ever materialized; this is the same
        product the white-noise M step needs (§3.6).
        """
        d = np.asarray(d, dtype=float)
        w = np.zeros((indicator.shape[1], self.n_coef))
        for fa in self.families:
            m = (indicator.T @ self._u.multiply((fa.prefactor * d)[:, None])).tocsr()
            w[:, fa.columns] = m @ fa.bbar
        if self.sparse is not None:
            w[:, self.sparse_columns] = (
                indicator.T @ self.sparse.multiply(d[:, None])
            ).toarray()
        if self.dense is not None:
            w[:, self.dense_columns] = np.asarray(
                indicator.T @ (d[:, None] * self.dense)
            )
        return w

    def epoch_row_dot(self, g: np.ndarray, epoch: np.ndarray) -> np.ndarray:
        """``out_i = T̃_i . g[epoch_i]`` (0 where ``epoch_i < 0``).

        Family columns depend on ``i`` only through the pair
        ``(bin(i), epoch(i))``, so one dot per distinct pair suffices (§3.6);
        the exact tiers contract directly. Never reconstructs ``T̃``.
        """
        g = np.asarray(g, dtype=float)
        epoch = np.asarray(epoch, dtype=np.int64)
        n_e = g.shape[0]
        out = np.zeros(self.n_obs, dtype=float)
        valid = epoch >= 0
        if not valid.any():
            return out

        if self.families:
            # One dot per distinct (quantize bin, ECORR epoch) pair.
            key = self.epochs[valid].astype(np.int64) * n_e + epoch[valid]
            uniq, inverse = np.unique(key, return_inverse=True)
            bins_u, eps_u = np.divmod(uniq, n_e)
            for fa in self.families:
                vals = np.einsum(
                    "ij,ij->i", fa.bbar[bins_u], g[np.ix_(eps_u, fa.columns)]
                )
                out[valid] += fa.prefactor[valid] * vals[inverse]
        if self.sparse is not None:
            coo = self.sparse.tocoo()
            keep = valid[coo.row]
            rows = coo.row[keep]
            out += np.bincount(
                rows,
                weights=coo.data[keep]
                * g[epoch[rows], self.sparse_columns[coo.col[keep]]],
                minlength=self.n_obs,
            )
        if self.dense is not None:
            gd = g[np.ix_(epoch[valid], self.dense_columns)]
            out[valid] += np.einsum("ij,ij->i", self.dense[valid], gd)
        return out

    def expand(self, v: np.ndarray, *, span: slice | None = None) -> np.ndarray:
        """``T̃ @ v`` for ``v`` of shape ``(k,)`` or ``(k, m)``."""
        v = np.asarray(v, dtype=float)
        if span is not None:
            masked = np.zeros_like(v)
            masked[span] = v[span]
            v = masked
        squeeze = v.ndim == 1
        vv = v[:, None] if squeeze else v
        out = np.zeros((self.n_obs, vv.shape[1]))
        for fam in self.families:
            out += fam.prefactor[:, None] * (fam.bbar @ vv[fam.columns])[self.epochs]
        if self.sparse is not None:
            out += self.sparse @ vv[self.sparse_columns]
        if self.dense is not None:
            out += self.dense @ vv[self.dense_columns]
        return out[:, 0] if squeeze else out

    def substituted_matrix(self) -> np.ndarray:
        """Densely reconstitute ``T̃`` — debugging, oracles, generic-noise fallback."""
        return self.expand(np.eye(self.n_coef))

    def _cost_terms(self) -> tuple[float, float]:
        """Return ``(C_dense, C_fast)`` cost heuristics (§7).

        ``C_fast`` carries the :data:`FAST_FLOP_PENALTY` calibration: the fast
        path's work is bincounts, sparse products and gathers, which run far
        below the BLAS efficiency of the dense Gram's GEMM, so a raw flop ratio
        overstates the achievable speedup by about an order of magnitude.
        """
        n, k = self.n_obs, self.n_coef
        n_ep = self.n_epochs
        c_dense = float(n * k * (k + 1))
        fam_sizes = [int(fa.columns.size) for fa in self.families]
        c_fast = 0.0
        for i, ka in enumerate(fam_sizes):
            for kb in fam_sizes[i:]:
                c_fast += 2.0 * n + n_ep * ka * kb
        k_fam = int(sum(fam_sizes))
        k_s = int(self.sparse_columns.size)
        k_r = int(self.dense_columns.size)
        nnz = int(self.sparse.nnz) if self.sparse is not None else 0
        n_fam = len(self.families)
        c_fast += (3 + n_fam) * nnz + (n_ep * k_fam if k_s > 0 else 0.0)
        c_fast += n * k_r * (k_r + k_s + k_fam) + 2.0 * n * k_r
        return c_dense, FAST_FLOP_PENALTY * c_fast

    @property
    def predicted_speedup(self) -> float:
        c_dense, c_fast = self._cost_terms()
        if c_fast <= 0.0:
            return float("inf")
        return c_dense / c_fast

    @property
    def predicted_end_to_end_speedup(self) -> float:
        c_dense, c_fast = self._cost_terms()
        k3 = (4.0 / 3.0) * self.n_coef**3
        return (c_dense + k3) / (c_fast + k3)

    def report(self) -> str:
        """Human-readable factorization census and speedup heuristics."""
        lines = [
            f"FactoredModel: {self.n_obs} toas -> {self.n_epochs} epochs "
            f"(dt={self.dt} s), {self.n_coef} columns"
        ]

        def _cols(count: int) -> str:
            return f"{count:>4} col " if count == 1 else f"{count:>4} cols"

        for fa in self.families:
            lines.append(
                f"  family {fa.name:<6} {_cols(int(fa.columns.size))}"
                f"   max err {fa.max_error:.1e}"
            )
        k_s = int(self.sparse_columns.size)
        k_r = int(self.dense_columns.size)
        lines.append(f"  sparse       {_cols(k_s)}   exact")
        lines.append(f"  dense        {_cols(k_r)}   exact")
        lines.append(
            f"  predicted speedup vs dense gram: {self.predicted_speedup:.1f}x"
        )
        _, c_fast = self._cost_terms()
        k3 = (4.0 / 3.0) * self.n_coef**3
        share = 100.0 * k3 / (c_fast + k3) if (c_fast + k3) > 0 else 0.0
        lines.append(
            f"  predicted speedup, whole E-step:  "
            f"{self.predicted_end_to_end_speedup:.1f}x  "
            f"(k^3 term {share:.0f}% of fast E-step)"
        )
        n_ecorr = sum(1 for kd in self.column_kinds if kd == "ecorr")
        if n_ecorr > 0.5 * self.n_coef:
            lines.append(
                f"  k={self.n_coef}: {n_ecorr} ecorr columns dominate; "
                "see feature_flexfit_fasttnt.md §1.7"
            )
        return "\n".join(lines)


def factorize(
    model: AssembledModel,
    *,
    toas: np.ndarray,
    freqs_mhz: np.ndarray | None = None,
    dt: float = 1.0,
    tol: float = 1.0e-6,
    sparse_max_fill: float = 0.25,
    extra_families: Sequence[PrefactorFamily] = (),
) -> FactoredModel:
    """Classify assembled columns into sparse / family / dense tiers."""
    matrix = model.matrix
    n, k = matrix.shape
    if np.asarray(toas).shape != (n,):
        raise ValueError(f"toas shape must be ({n},)")
    epochs = quantize(toas, dt)
    n_ep = int(epochs.max()) + 1
    u = sp.csr_matrix((np.ones(n), (np.arange(n), epochs)), shape=(n, n_ep))

    fill = np.count_nonzero(matrix, axis=0) / n
    is_ecorr = np.array([kd == "ecorr" for kd in model.column_kinds])
    to_sparse = is_ecorr | (fill <= sparse_max_fill)

    families = _build_families(freqs_mhz, model, tuple(extra_families))
    remaining = np.flatnonzero(~to_sparse)
    groups: list[FamilyGroup] = []
    for fam in families:
        if remaining.size == 0:
            break
        h = fam.values
        cols = matrix[:, remaining]
        num = np.asarray(u.T @ (h[:, None] * cols))
        den = np.bincount(epochs, weights=h * h, minlength=n_ep).astype(float)
        bbar = np.divide(
            num, den[:, None], out=np.zeros_like(num), where=den[:, None] > 0
        )
        err = np.abs(cols - h[:, None] * bbar[epochs, :]).max(axis=0)
        scale = np.maximum(np.abs(cols).max(axis=0), np.finfo(float).tiny)
        hit = err <= tol * scale
        if hit.any():
            picked = remaining[hit]
            groups.append(
                FamilyGroup(
                    name=fam.name,
                    prefactor=h,
                    columns=picked,
                    bbar=np.ascontiguousarray(bbar[:, hit]),
                    max_error=float((err[hit] / scale[hit]).max()),
                )
            )
            remaining = remaining[~hit]

    sparse_columns = np.flatnonzero(to_sparse)
    sparse = sp.csr_matrix(matrix[:, sparse_columns]) if sparse_columns.size else None
    dense = matrix[:, remaining].copy() if remaining.size else None
    return FactoredModel(
        model=model,
        epochs=epochs,
        n_epochs=n_ep,
        dt=dt,
        tol=tol,
        families=tuple(groups),
        sparse=sparse,
        sparse_columns=sparse_columns,
        dense=dense,
        dense_columns=remaining.astype(np.int64),
        max_error=max((g.max_error for g in groups), default=0.0),
        _u=u,
    )


@dataclass(frozen=True)
class Factorization:
    """Opt-in fast-TNT spec for drivers that assemble internally."""

    toas: np.ndarray
    freqs_mhz: np.ndarray | None
    dt: float = 1.0
    tol: float = 1.0e-6
    sparse_max_fill: float = 0.25
    extra_families: tuple[PrefactorFamily, ...] = ()
    mode: Literal["fast", "auto"] = "auto"
    min_speedup: float = 1.5

    def apply(self, model: AssembledModel) -> AssembledModel | FactoredModel:
        factored = factorize(
            model,
            toas=self.toas,
            freqs_mhz=self.freqs_mhz,
            dt=self.dt,
            tol=self.tol,
            sparse_max_fill=self.sparse_max_fill,
            extra_families=self.extra_families,
        )
        if self.mode == "auto" and factored.predicted_speedup < self.min_speedup:
            return model
        return factored
