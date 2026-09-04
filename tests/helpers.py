"""Shared helpers for MetaPulsar tests."""

from typing import Literal, Optional

from metapulsar.tim_file_analyzer import TimMetadata

TimPulseNumberStatus = Literal["complete", "mixed", "none"]


def make_tim_metadata(
    *,
    timespan_days: float = 0.0,
    toa_count: int = 0,
    pn_status: TimPulseNumberStatus = "none",
    pn_with_count: Optional[int] = None,
    pn_without_count: Optional[int] = None,
    mjd_min: Optional[float] = None,
    mjd_max: Optional[float] = None,
) -> TimMetadata:
    """Build TimMetadata for tests without parsing a .tim file."""
    if pn_with_count is None:
        if pn_status == "complete":
            pn_with_count = toa_count
        elif pn_status == "mixed":
            pn_with_count = max(0, toa_count // 2)
        else:
            pn_with_count = 0
    if pn_without_count is None:
        pn_without_count = toa_count - pn_with_count
    if (
        mjd_min is not None
        and mjd_max is not None
        and timespan_days == 0.0
        and mjd_max >= mjd_min
    ):
        timespan_days = float(mjd_max - mjd_min)
    return TimMetadata(
        toa_count=toa_count,
        mjd_min=mjd_min,
        mjd_max=mjd_max,
        timespan_days=timespan_days,
        pn_with_count=pn_with_count,
        pn_without_count=pn_without_count,
        pn_status=pn_status,
    )
