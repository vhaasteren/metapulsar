"""Parameter partitioning helpers for nonlinear timing modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class TimingPartition:
    """Partitioned timing-parameter sets and corresponding matrix indices."""

    sampled_params: list[str]
    marginalized_params: list[str]
    idx_sampled: list[int]
    idx_marginalized: list[int]


def _index_lookup(
    idx_from_fitpars: Mapping[str, int | Sequence[int]] | None, fitpars: Sequence[str]
) -> dict[str, list[int]]:
    if idx_from_fitpars is None:
        return {param: [idx] for idx, param in enumerate(fitpars)}

    index_map: dict[str, list[int]] = {}
    for param in fitpars:
        if param not in idx_from_fitpars:
            raise KeyError(f"Parameter '{param}' is missing from idx_from_fitpars.")
        raw_value = idx_from_fitpars[param]
        if isinstance(raw_value, int):
            index_map[param] = [raw_value]
        else:
            index_map[param] = [int(idx) for idx in raw_value]
        if not index_map[param]:
            raise ValueError(
                f"Parameter '{param}' must map to at least one design-matrix index."
            )
    return index_map


def compute_timing_partition(
    fitpars: Sequence[str],
    sampled_params: Sequence[str],
    marginalized_params: Sequence[str] | None = None,
    idx_from_fitpars: Mapping[str, int | Sequence[int]] | None = None,
) -> TimingPartition:
    """Compute sampled/marginalized partitions and corresponding column indices."""

    fitpar_list = list(fitpars)
    sampled = list(sampled_params)
    if len(set(sampled)) != len(sampled):
        raise ValueError("sampled_params contains duplicates.")
    unknown_sampled = sorted(set(sampled) - set(fitpar_list))
    if unknown_sampled:
        raise KeyError(f"Unknown sampled parameter(s): {', '.join(unknown_sampled)}")

    if marginalized_params is None:
        marginalized = [param for param in fitpar_list if param not in sampled]
    else:
        marginalized = list(marginalized_params)
        if len(set(marginalized)) != len(marginalized):
            raise ValueError("marginalized_params contains duplicates.")
        unknown_marginalized = sorted(set(marginalized) - set(fitpar_list))
        if unknown_marginalized:
            raise KeyError(
                f"Unknown marginalized parameter(s): {', '.join(unknown_marginalized)}"
            )

    overlap = sorted(set(sampled) & set(marginalized))
    if overlap:
        raise ValueError(
            "Sampled and marginalized parameter sets must be disjoint. "
            f"Overlap: {', '.join(overlap)}"
        )

    index_map = _index_lookup(idx_from_fitpars, fitpar_list)
    idx_sampled = [idx for param in sampled for idx in index_map[param]]
    idx_marginalized = [idx for param in marginalized for idx in index_map[param]]

    return TimingPartition(
        sampled_params=sampled,
        marginalized_params=marginalized,
        idx_sampled=idx_sampled,
        idx_marginalized=idx_marginalized,
    )
