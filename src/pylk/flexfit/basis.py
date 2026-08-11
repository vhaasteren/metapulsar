"""Pure data containers for the joint timing + GP basis.

A :class:`BasisBlock` is one contiguous set of columns in the Enterprise-style
``T = [J_z, F_red, F_dm, ...]`` matrix, together with the :class:`VarianceGroup`
structure of its diagonal ``Phi`` prior. These are deliberately dumb containers:
they hold matrices, names, groups, units, and normalization metadata, and know
nothing about spectra (power law, free spectrum) or how the columns were built.
Spectral semantics live in the adapters, which defer to Discovery/Enterprise.

Variance-group bounds are stored as raw coefficient variances ``rho`` (the
diagonal of ``Phi``). Because raw ``rho`` is unintuitive and basis-scale
dependent, :func:`rho_bounds_from_rms` converts a physical induced-RMS range
into ``rho`` bounds using the basis normalization ``q = tr(T_g T_g^T)/n_toa``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

BlockKind = Literal["timing", "red", "dm", "chromatic", "ecorr", "custom"]

# Enterprise-style broad finite prior used to effectively marginalize the timing
# directions on the first sweep.
INITIAL_TIMING_VARIANCE = 1.0e40


def _readonly(matrix: np.ndarray, *, dtype=float) -> np.ndarray:
    out = np.array(matrix, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class VarianceGroup:
    """One tied variance in the diagonal ``Phi`` prior.

    ``indices`` are column positions *within the owning block*; the assembler
    offsets them to global ``T`` columns. All columns in a group share a single
    variance ``rho`` bounded to ``[lower, upper]``. ``alpha``/``beta`` are the
    inverse-gamma hyperprior shape/scale (0 gives the plain EM/ML update).
    ``update_from_sweep`` gates when the group becomes eligible for the bounded
    hyperparameter update; timing groups default to sweep 2.
    """

    name: str
    indices: tuple[int, ...]
    lower: float
    upper: float
    alpha: float = 0.0
    beta: float = 0.0
    update_from_sweep: int = 1
    initial: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", tuple(int(i) for i in self.indices))
        if not self.indices:
            raise ValueError(f"variance group {self.name!r} has no indices")
        if not (self.upper >= self.lower > 0.0):
            raise ValueError(
                f"variance group {self.name!r} needs 0 < lower <= upper; "
                f"got lower={self.lower}, upper={self.upper}"
            )
        if self.update_from_sweep < 1:
            raise ValueError("update_from_sweep must be >= 1")
        if self.initial is not None and not (self.lower <= self.initial <= self.upper):
            # Allow the timing 1e40 sentinel to sit above finite bounds.
            if not (self.initial >= self.upper):
                raise ValueError(
                    f"variance group {self.name!r} initial={self.initial} "
                    f"outside [{self.lower}, {self.upper}]"
                )

    @property
    def size(self) -> int:
        return len(self.indices)

    def initial_rho(self) -> float:
        """Starting variance for the first E-step (geometric mean by default)."""
        if self.initial is not None:
            return float(self.initial)
        return float(np.sqrt(self.lower * self.upper))

    def shifted(self, offset: int) -> "VarianceGroup":
        return VarianceGroup(
            name=self.name,
            indices=tuple(i + offset for i in self.indices),
            lower=self.lower,
            upper=self.upper,
            alpha=self.alpha,
            beta=self.beta,
            update_from_sweep=self.update_from_sweep,
            initial=self.initial,
        )


@dataclass(frozen=True)
class BasisBlock:
    """One column block of ``T`` with its variance-group structure."""

    name: str
    matrix: np.ndarray
    coefficient_names: tuple[str, ...]
    groups: tuple[VarianceGroup, ...]
    kind: BlockKind = "custom"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = _readonly(self.matrix)
        if matrix.ndim != 2:
            raise ValueError(f"block {self.name!r} matrix must be 2-D")
        n_col = matrix.shape[1]
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "coefficient_names", tuple(self.coefficient_names))
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if len(self.coefficient_names) != n_col:
            raise ValueError(
                f"block {self.name!r}: {len(self.coefficient_names)} names for "
                f"{n_col} columns"
            )
        covered: list[int] = []
        for group in self.groups:
            for idx in group.indices:
                if not 0 <= idx < n_col:
                    raise ValueError(
                        f"block {self.name!r} group {group.name!r} index {idx} "
                        f"out of range [0, {n_col})"
                    )
            covered.extend(group.indices)
        if sorted(covered) != list(range(n_col)):
            raise ValueError(
                f"block {self.name!r} groups must partition all {n_col} columns "
                "exactly once"
            )

    @property
    def n_obs(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_col(self) -> int:
        return int(self.matrix.shape[1])


@dataclass(frozen=True)
class AssembledModel:
    """The joint ``T`` matrix with globally-indexed variance groups."""

    matrix: np.ndarray
    groups: tuple[VarianceGroup, ...]
    coefficient_names: tuple[str, ...]
    column_kinds: tuple[BlockKind, ...]
    block_spans: Mapping[str, slice]
    block_kinds: Mapping[str, BlockKind]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", _readonly(self.matrix))
        object.__setattr__(
            self, "block_spans", MappingProxyType(dict(self.block_spans))
        )
        object.__setattr__(
            self, "block_kinds", MappingProxyType(dict(self.block_kinds))
        )

    @property
    def n_obs(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_coef(self) -> int:
        return int(self.matrix.shape[1])

    def block_matrix(self, name: str) -> np.ndarray:
        return self.matrix[:, self.block_spans[name]]

    def block_slice(self, name: str) -> slice:
        return self.block_spans[name]


def assemble(blocks: tuple[BasisBlock, ...] | list[BasisBlock]) -> AssembledModel:
    """Horizontally concatenate blocks and offset their variance groups.

    Blocks are laid out in the given order; the timing block (if any) should be
    listed first so its columns own the leading ``z`` coordinates, mirroring the
    ``T = [J_z, F_red, ...]`` convention.
    """
    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("assemble requires at least one block")
    n_obs = blocks[0].n_obs
    names_seen: set[str] = set()
    matrices: list[np.ndarray] = []
    groups: list[VarianceGroup] = []
    coefficient_names: list[str] = []
    column_kinds: list[BlockKind] = []
    block_spans: dict[str, slice] = {}
    block_kinds: dict[str, BlockKind] = {}
    offset = 0
    for block in blocks:
        if block.n_obs != n_obs:
            raise ValueError(
                f"block {block.name!r} has {block.n_obs} rows, expected {n_obs}"
            )
        if block.name in names_seen:
            raise ValueError(f"duplicate block name {block.name!r}")
        names_seen.add(block.name)
        matrices.append(np.asarray(block.matrix, dtype=float))
        groups.extend(group.shifted(offset) for group in block.groups)
        coefficient_names.extend(
            f"{block.name}:{name}" for name in block.coefficient_names
        )
        column_kinds.extend([block.kind] * block.n_col)
        block_spans[block.name] = slice(offset, offset + block.n_col)
        block_kinds[block.name] = block.kind
        offset += block.n_col
    return AssembledModel(
        matrix=np.hstack(matrices),
        groups=tuple(groups),
        coefficient_names=tuple(coefficient_names),
        column_kinds=tuple(column_kinds),
        block_spans=block_spans,
        block_kinds=block_kinds,
    )


def column_rms_scale(matrix: np.ndarray, indices: tuple[int, ...]) -> float:
    """Return ``q = tr(T_g T_g^T)/n_toa`` for the group's columns.

    A unit coefficient variance ``rho`` on these columns induces a waveform with
    mean-square ``rho * q`` per TOA, so ``sigma^2 = rho * q``.
    """
    cols = np.asarray(matrix, dtype=float)[:, list(indices)]
    n_toa = cols.shape[0]
    return float(np.sum(cols**2) / n_toa)


def rho_bounds_from_rms(
    matrix: np.ndarray,
    indices: tuple[int, ...],
    *,
    sigma_min: float,
    sigma_max: float,
) -> tuple[float, float]:
    """Convert an induced residual-RMS range (seconds) to ``rho`` bounds."""
    if not (sigma_max >= sigma_min > 0.0):
        raise ValueError("require 0 < sigma_min <= sigma_max")
    q = column_rms_scale(matrix, indices)
    if q <= 0.0:
        raise ValueError("degenerate basis columns (zero norm) for RMS bounds")
    return sigma_min**2 / q, sigma_max**2 / q


def fourier_pair_groups(
    matrix: np.ndarray,
    *,
    prefix: str,
    n_freq: int,
    sigma_min: float,
    sigma_max: float,
    update_from_sweep: int = 1,
    alpha: float = 0.0,
    beta: float = 0.0,
    interleaved: bool = True,
) -> tuple[VarianceGroup, ...]:
    """One tied variance per Fourier frequency (sin/cos share ``rho``).

    Tying each sine/cosine pair keeps the fit invariant to the arbitrary Fourier
    phase origin. ``interleaved=True`` matches Discovery's ``[sin_0, cos_0,
    sin_1, cos_1, ...]`` column order.
    """
    groups: list[VarianceGroup] = []
    for k in range(n_freq):
        if interleaved:
            indices = (2 * k, 2 * k + 1)
        else:
            indices = (k, k + n_freq)
        lower, upper = rho_bounds_from_rms(
            matrix, indices, sigma_min=sigma_min, sigma_max=sigma_max
        )
        groups.append(
            VarianceGroup(
                name=f"{prefix}_f{k}",
                indices=indices,
                lower=lower,
                upper=upper,
                alpha=alpha,
                beta=beta,
                update_from_sweep=update_from_sweep,
            )
        )
    return tuple(groups)


def per_column_groups(
    matrix: np.ndarray,
    *,
    names: tuple[str, ...],
    lower: float,
    upper: float,
    update_from_sweep: int = 1,
    initial: float | None = None,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> tuple[VarianceGroup, ...]:
    """One independent variance per column (used for timing coordinates)."""
    n_col = np.asarray(matrix).shape[1]
    if len(names) != n_col:
        raise ValueError("names length must match number of columns")
    return tuple(
        VarianceGroup(
            name=name,
            indices=(i,),
            lower=lower,
            upper=upper,
            alpha=alpha,
            beta=beta,
            update_from_sweep=update_from_sweep,
            initial=initial,
        )
        for i, name in enumerate(names)
    )
