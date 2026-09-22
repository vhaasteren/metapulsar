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


# Self-contained T2/DDK par body shared by the Kopeikin-convention tests.
# ECLIPTIC_BODY stays in test_parameter_manager.py.
T2_DDK_BODY = (
    "PSR J1600-3053\n"
    "PEPOCH 55000\n"
    "F0 277.937 1\n"
    "F1 -7.3e-16 1\n"
    "LAMBDA 244.347677 1\n"
    "BETA -10.071873 1\n"
    "PMLAMBDA -0.35 1\n"
    "PMBETA -7.0 1\n"
    "POSEPOCH 55000\n"
    "DM 52.3 1\n"
    "DMEPOCH 55000\n"
    "BINARY T2\n"
    "PB 14.348466\n"
    "A1 8.801653\n"
    "T0 55000.0\n"
    "ECC 1.737e-04\n"
    "OM 181.85\n"
    "M2 0.33\n"
    "SINI KIN\n"
    "KIN 68.0\n"
    "KOM 76.0\n"
    "PX 0.5\n"
)
