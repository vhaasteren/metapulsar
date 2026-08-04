"""End-to-end contract for the canonical .tim artifacts on real release data.

MetaPulsar always hands its engines a rewritten standalone FORMAT 1 .tim that
carries the PTA identity as real flags. These tests pin that the rewrite is
data-preserving and that the exported artifact is the file the engine consumed.
"""

from pathlib import Path

import numpy as np
import pytest

from metapulsar.file_discovery import FileDiscovery
from metapulsar.metapulsar_factory import MetaPulsarFactory

DATA = Path(__file__).parent.parent / "data" / "ipta-dr2"
PULSAR = "J1857+0943"
# EPTA ships -pn and no -pta; PPTA ships -pta, so one leg exercises each path.
RELEASES = ["epta_dr1_v2_2", "ppta_dr2"]


def _pulsar_file_data():
    if not DATA.is_dir():
        pytest.skip(f"IPTA DR2 data not present at {DATA}")
    discovered = FileDiscovery(working_dir=str(DATA)).discover_files(RELEASES)
    file_data = {
        pta: [entry for entry in entries if PULSAR in str(entry["par"])]
        for pta, entries in discovered.items()
    }
    file_data = {pta: entries for pta, entries in file_data.items() if entries}
    if len(file_data) < 2:
        pytest.skip(f"{PULSAR} not present in both of {RELEASES}")
    return file_data


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    export_dir = tmp_path_factory.mktemp("canonical_tims")
    mp = MetaPulsarFactory().create_metapulsar(
        file_data=_pulsar_file_data(),
        combination_strategy="per_pta",
        timfile_output_dir=export_dir,
        use_pulse_numbers="yes",
    )
    return mp, export_dir


@pytest.mark.slow
@pytest.mark.requires_libstempo
@pytest.mark.requires_ipta_data
class TestCanonicalTimArtifacts:
    def test_metadata_flags_come_from_the_tim_file(self, built):
        """The PTA identity is read back out of the file, not synthesized."""
        mp, export_dir = built

        for pta in mp._pta_data:
            exported = export_dir / f"{mp.name}_{pta}.tim"
            text = exported.read_text(encoding="utf-8")
            toa_lines = [
                line
                for line in text.splitlines()
                if line.split() and line.split()[0] not in ("FORMAT", "MODE", "C", "#")
            ]
            assert toa_lines
            for line in toa_lines:
                assert f"-pta {pta}" in line
                assert f"-pta_dataset {pta}" in line
                assert "-timing_package " in line

        assert set(np.unique(mp.flags["pta"])) == set(mp._pta_data)
        assert set(np.unique(mp.flags["pta_dataset"])) == set(mp._pta_data)
        assert set(np.unique(mp.flags["timing_package"])) <= {"pint", "tempo2"}

    def test_existing_pta_flag_is_preserved_as_pta_orig(self, built):
        """PPTA DR1/DR2 ships its own -pta; the release value stays auditable."""
        mp, export_dir = built

        exported = (export_dir / f"{mp.name}_ppta_dr2.tim").read_text(encoding="utf-8")
        assert "-pta_orig" in exported
        # Every PPTA TOA carried a -pta, so every line keeps a renamed copy.
        assert exported.count("-pta_orig") == exported.count("-pta_dataset ppta_dr2")
        assert "pta_orig" in mp.flags
        assert set(np.unique(mp.flags["pta_orig"])) == {"", "PPTA"}

    def test_exported_file_is_the_engine_consumed_file(self, built):
        """Export copies the same bytes the timing package loaded."""
        mp, export_dir = built

        for pta, files in mp._pta_files.items():
            exported = export_dir / f"{mp.name}_{pta}.tim"
            assert exported.read_text(encoding="utf-8") == files.tim_path.read_text(
                encoding="utf-8"
            )

    def test_pulse_numbers_are_present_when_requested(self, built):
        mp, export_dir = built

        for pta in mp._pta_data:
            text = (export_dir / f"{mp.name}_{pta}.tim").read_text(encoding="utf-8")
            toa_lines = [line for line in text.splitlines() if " -pta " in line]
            assert all(" -pn " in line for line in toa_lines)


@pytest.mark.slow
@pytest.mark.requires_ipta_data
def test_pint_leg_reads_metadata_flags_back_from_the_tim(tmp_path):
    """PINT parses the stamped flags too, and NANOGrav's own -pta is preserved.

    NANOGrav 9y knows this pulsar as B1855+09 and PPTA as J1857+0943; both ship
    a release -pta flag, so this covers the rename on a PINT leg.
    """
    if not DATA.is_dir():
        pytest.skip(f"IPTA DR2 data not present at {DATA}")
    discovered = FileDiscovery(working_dir=str(DATA)).discover_files(
        ["nanograv_9y", "ppta_dr2"]
    )
    file_data = {}
    for pta, entries in discovered.items():
        matching = [
            entry
            for entry in entries
            if "B1855+09" in str(entry["par"]) or PULSAR in str(entry["par"])
        ]
        if matching:
            file_data[pta] = matching
    if "nanograv_9y" not in file_data:
        pytest.skip("NANOGrav 9y B1855+09 not present")
    assert file_data["nanograv_9y"][0]["timing_package"] == "pint"

    export_dir = tmp_path / "mixed"
    export_dir.mkdir()
    mp = MetaPulsarFactory().create_metapulsar(
        file_data=file_data,
        combination_strategy="per_pta",
        timfile_output_dir=export_dir,
        use_pulse_numbers="yes",
    )

    pint_rows = mp.flags["pta"] == "nanograv_9y"
    assert pint_rows.any()
    assert set(np.unique(mp.flags["timing_package"][pint_rows])) == {"pint"}
    assert set(np.unique(mp.flags["pta_dataset"][pint_rows])) == {"nanograv_9y"}
    assert set(np.unique(mp.flags["pta_orig"][pint_rows])) == {"NANOGrav"}


@pytest.mark.slow
@pytest.mark.requires_libstempo
@pytest.mark.requires_ipta_data
def test_rewrite_alone_preserves_toas_exactly(tmp_path):
    """Rewriting is text surgery, so tempo2 reads back bit-identical data.

    Run with ``use_pulse_numbers="no"`` so nothing but flattening and stamping
    touches the file -- no backend reserialization to hide behind. This is the
    guard on INCLUDE flattening: a mis-scoped TIME offset would move the site
    arrival times here.
    """
    from metapulsar.sandbox_tempo2 import tempopulsar

    export_dir = tmp_path / "no_pn"
    export_dir.mkdir()
    file_data = _pulsar_file_data()
    mp = MetaPulsarFactory().create_metapulsar(
        file_data=file_data,
        combination_strategy="per_pta",
        timfile_output_dir=export_dir,
        use_pulse_numbers="no",
    )

    assert len(sorted(export_dir.glob("*.tim"))) == len(mp._pta_data)
    for pta, entries in file_data.items():
        par = str(entries[0]["par"])
        original = tempopulsar(parfile=par, timfile=str(entries[0]["tim"]), dofit=False)
        canonical = tempopulsar(
            parfile=par,
            timfile=str(export_dir / f"{mp.name}_{pta}.tim"),
            dofit=False,
        )

        np.testing.assert_array_equal(
            np.asarray(canonical.stoas), np.asarray(original.stoas)
        )
        np.testing.assert_array_equal(
            np.asarray(canonical.toaerrs), np.asarray(original.toaerrs)
        )
        assert canonical.flagvals("pta")[0] == pta
