"""Tests for catalog + position-based pulsar identity."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from metapulsar.file_discovery import filter_file_data_by_pulsars
from metapulsar.position_helpers import (
    build_alias_map,
    discover_pulsars_by_position,
    letter_suffix,
    preferred_group_name,
)


def _file_entry(
    par_path: str,
    par_content: str,
    pta: str = "PTA",
) -> dict:
    return {
        "par": par_path,
        "tim": par_path.replace(".par", ".tim"),
        "par_content": par_content,
        "timing_package": "tempo2",
    }


def test_preferred_group_name_b_over_j():
    assert preferred_group_name(["J1857+0943", "B1855+09"]) == "B1855+09"


def test_letter_suffix_cluster_msp():
    assert letter_suffix("J1824-2452A") == "A"
    assert letter_suffix("J1857+0943") is None
    assert letter_suffix("B1855+09") is None


def test_merge_b_and_j_catalog_at_same_position():
    coords = "RAJ 18:57:36.3906121\nDECJ +09:43:17.20714\nF0 1 0\n"
    file_data = {
        "ng": [
            _file_entry(
                "data/J1857+0943.par",
                f"PSR J1857+0943\n{coords}",
            )
        ],
        "epta": [
            _file_entry(
                "data/B1855+09.par",
                f"PSR B1855+09\n{coords}",
            )
        ],
    }
    groups = discover_pulsars_by_position(file_data)
    assert len(groups) == 1
    assert "B1855+09" in groups
    aliases = build_alias_map(groups)
    assert aliases["J1857+0943"] == "B1855+09"
    assert aliases["B1855+09"] == "B1855+09"


def test_suffix_a_and_b_within_tolerance_separate_groups():
    coords = "RAJ 18:24:53.5967\nDECJ -24:52:08.87\nF0 1 0\n"
    file_data = {
        "pta": [
            _file_entry(
                "data/J1824-2452A.par",
                f"PSRJ J1824-2452A\n{coords}",
            ),
            _file_entry(
                "data/J1824-2452B.par",
                f"PSRJ J1824-2452B\n{coords}",
            ),
        ]
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        groups = discover_pulsars_by_position(file_data)
    assert len(groups) == 2
    assert any("different letter suffixes" in str(w.message) for w in caught)


def test_suffix_a_and_bare_within_tolerance_raises():
    coords = "RAJ 18:24:53.5967\nDECJ -24:52:08.87\nF0 1 0\n"
    file_data = {
        "pta": [
            _file_entry(
                "data/J1824-2452A.par",
                f"PSRJ J1824-2452A\n{coords}",
            ),
            _file_entry(
                "data/J1824-2452.par",
                f"PSR J1824-2452\n{coords}",
            ),
        ]
    }
    with pytest.raises(ValueError, match="Ambiguous pulsar identity"):
        discover_pulsars_by_position(file_data)


def test_same_catalog_name_far_apart_raises():
    file_data = {
        "pta": [
            _file_entry(
                "data/J0613-0200.par",
                "PSR J0613-0200\nRAJ 06:13:43.9754\nDECJ -02:00:47.1755\nF0 1 0\n",
            ),
            _file_entry(
                "data/J0613-0200b.par",
                "PSR J0613-0200\nRAJ 12:00:00.0\nDECJ +00:00:00.0\nF0 1 0\n",
            ),
        ]
    }
    with pytest.raises(ValueError, match="distinct sky positions"):
        discover_pulsars_by_position(file_data)


def test_j1022_like_scatter_merges_within_10arcsec():
    """PTA position scatter of a few arcsec with the same catalog name should merge."""
    # ~5″ apart on sky (same order as data-check J1022+1001 clumps)
    file_data = {
        "pta_a": [
            _file_entry(
                "data/J1022+1001.par",
                "PSR J1022+1001\nRAJ 10:22:58.0\nDECJ +10:01:52.0\nF0 1 0\n",
            )
        ],
        "pta_b": [
            _file_entry(
                "data/J1022+1001b.par",
                "PSR J1022+1001\nRAJ 10:22:58.3\nDECJ +10:01:56.0\nF0 1 0\n",
            )
        ],
    }
    groups = discover_pulsars_by_position(file_data)
    assert len(groups) == 1
    assert "J1022+1001" in groups
    assert set(groups["J1022+1001"]) == {"pta_a", "pta_b"}


def test_truncated_coord_name_not_in_alias_map():
    coords = "RAJ 05:57:44.1\nDECJ +15:50:06.0\nF0 1 0\n"
    file_data = {
        "ng": [
            _file_entry(
                "data/J0557+1551.par",
                f"PSR J0557+1551\n{coords}",
            )
        ]
    }
    groups = discover_pulsars_by_position(file_data)
    assert "J0557+1551" in groups
    aliases = build_alias_map(groups)
    assert "J0557+1551" in aliases
    assert "J0557+1550" not in aliases


@pytest.mark.slow
@pytest.mark.integration
def test_data_check_j0557_and_j1824_filter_regression():
    data_check = Path("/workspaces/metapulsar/data-check")
    if not (data_check / "NANOGrav_15y").is_dir():
        pytest.skip("data-check not available")

    # Canned layouts, not discover_layout: the inferred PPTA_DR3 / NG15 patterns
    # match <PSR>.par, <PSR>_pint.par and the ao/gbt siblings alike, which is an
    # ambiguous selection by design. This test is about pulsar-name filtering.
    from metapulsar.file_discovery import PTA_DATA_RELEASES, discover_files

    files = discover_files(
        {k: PTA_DATA_RELEASES[k] for k in ("nanograv_15y", "ppta_dr3")},
        working_dir=str(data_check),
        verbose=False,
    )

    out = filter_file_data_by_pulsars(files, "J0557+1551")
    assert any("J0557+1551" in str(f["par"]) for fl in out.values() for f in fl)

    with pytest.raises(ValueError, match="not found"):
        filter_file_data_by_pulsars(files, "J0557+1550")

    out_a = filter_file_data_by_pulsars(files, "J1824-2452A")
    assert any("J1824-2452A" in str(f["par"]) for fl in out_a.values() for f in fl)
