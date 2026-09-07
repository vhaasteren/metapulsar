"""Residual-centering facts for MetaPulsar timing engines.

Descriptive only (psrdata R-4.2): nothing in the engines branches on these
values. They travel with the record and the run manifest.
"""

from __future__ import annotations

from typing import Any

from psrdata import ResidualCentering

#: PINT-family host: residuals stored uncentered; the package's usual report
#: is a weighted mean subtraction.
PINT_FAMILY = ResidualCentering(
    stored_residuals="none",
    standard_output="mean_removed",
    standard_weighted=True,
)

#: libstempo centres internally with no switch, so the stored convention is
#: unknown; tempo2's usual report is an unweighted mean subtraction.
TEMPO2_FAMILY = ResidualCentering(
    stored_residuals="unknown",
    standard_output="mean_removed",
    standard_weighted=False,
)

#: A linearized stand-in over a bare design matrix: stored residuals are
#: unknown unless the caller says otherwise.
LINEARIZED_UNKNOWN = ResidualCentering(stored_residuals="unknown")


def as_mapping(
    centering: ResidualCentering, key: str = "single"
) -> dict[str, ResidualCentering]:
    return {key: centering}


def for_compatibility(compatibility: str) -> ResidualCentering:
    """What a linearized stand-in reports for a host of that par/tim family."""
    weighted = not str(compatibility).lower().startswith("tempo2")
    return ResidualCentering(
        stored_residuals="unknown",
        standard_output="mean_removed",
        standard_weighted=weighted,
    )


def from_jug_state(state: Any) -> ResidualCentering:
    """JUG export is always uncentered; the standard report follows the host."""
    compatibility = str(getattr(state, "compatibility", "pint"))
    weighted = not compatibility.lower().startswith("tempo2")
    return ResidualCentering(
        stored_residuals="none",
        standard_output="mean_removed",
        standard_weighted=weighted,
    )
