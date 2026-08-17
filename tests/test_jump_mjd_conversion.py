"""Factory-level JUMP MJD → -mjd_jump_pta conversion tests."""

from pathlib import Path

import pytest

from metapulsar.metapulsar_factory import MetaPulsarFactory, create_all_metapulsars
from metapulsar.tim_canonical import _pint_legacy_heuristic_hit

FIXTURE_PAR = Path(__file__).parent / "fixtures" / "sample_parfiles" / "simple.par"


def _write_jump_dataset(tmp_path: Path, *, pta: str = "PPTA") -> dict:
    """Minimal single-PTA dataset with one JUMP MJD and in/out-of-window TOAs."""
    par = FIXTURE_PAR.read_text(encoding="utf-8") + "JUMP MJD 54515 54535 -1e-7 1\n"
    par_path = tmp_path / f"{pta}.par"
    tim_path = tmp_path / f"{pta}.tim"
    par_path.write_text(par, encoding="utf-8")
    tim_path.write_text(
        "FORMAT 1\n"
        "test1 1400.0 54510.0 1.5 g -sys TEST\n"
        "test2 1400.0 54520.0 1.5 g -sys TEST\n"
        "test3 1400.0 54535.0 1.5 g -sys TEST\n",
        encoding="utf-8",
    )
    return {
        pta: [
            {
                "par": par_path,
                "tim": tim_path,
                "par_content": par,
                "timing_package": "pint",
            }
        ]
    }


@pytest.mark.unit
def test_default_stamps_tim_but_keeps_jump_mjd_in_engine_par(tmp_path):
    file_data = _write_jump_dataset(tmp_path)
    mp = MetaPulsarFactory().create_metapulsar(
        file_data=file_data,
        combination_strategy="per_pta",
        use_pulse_numbers="no",
        convert_jump_mjd=False,
        canonicalize_tim=True,
    )

    tim_text = mp._pta_files["PPTA"].tim_path.read_text(encoding="utf-8")
    par_text = mp._pta_files["PPTA"].par_path.read_text(encoding="utf-8")
    lines = [line for line in tim_text.splitlines() if " -pta " in line]
    assert "-mjd_jump_pta PPTA_1" not in lines[0]
    assert "-mjd_jump_pta PPTA_1" in lines[1]
    # PINT closed interval includes the upper bound.
    assert "-mjd_jump_pta PPTA_1" in lines[2]
    assert "JUMP MJD 54515 54535" in par_text
    assert "JUMP -mjd_jump_pta" not in par_text


@pytest.mark.unit
def test_convert_jump_mjd_rewrites_engine_par(tmp_path):
    file_data = _write_jump_dataset(tmp_path)
    release_par = file_data["PPTA"][0]["par"]
    mp = MetaPulsarFactory().create_metapulsar(
        file_data=file_data,
        combination_strategy="per_pta",
        use_pulse_numbers="no",
        convert_jump_mjd=True,
        canonicalize_tim=True,
    )

    tim_text = mp._pta_files["PPTA"].tim_path.read_text(encoding="utf-8")
    par_text = mp._pta_files["PPTA"].par_path.read_text(encoding="utf-8")
    assert "-mjd_jump_pta PPTA_1" in tim_text
    assert "JUMP -mjd_jump_pta PPTA_1 -1e-7 1" in par_text
    assert "JUMP MJD" not in par_text
    # Release file must remain untouched.
    assert "JUMP MJD 54515 54535" in Path(release_par).read_text(encoding="utf-8")


@pytest.mark.unit
def test_no_jump_mjd_is_noop(tmp_path):
    par = FIXTURE_PAR.read_text(encoding="utf-8")
    par_path = tmp_path / "EPTA.par"
    tim_path = tmp_path / "EPTA.tim"
    par_path.write_text(par, encoding="utf-8")
    tim_path.write_text(
        "FORMAT 1\n" "test1 1400.0 54510.0 1.5 g -sys TEST\n",
        encoding="utf-8",
    )
    file_data = {
        "EPTA": [
            {
                "par": par_path,
                "tim": tim_path,
                "par_content": par,
                "timing_package": "pint",
            }
        ]
    }
    mp = MetaPulsarFactory().create_metapulsar(
        file_data=file_data,
        combination_strategy="per_pta",
        use_pulse_numbers="no",
        convert_jump_mjd=True,
        canonicalize_tim=True,
    )
    tim_text = mp._pta_files["EPTA"].tim_path.read_text(encoding="utf-8")
    par_text = mp._pta_files["EPTA"].par_path.read_text(encoding="utf-8")
    assert "-mjd_jump_pta" not in tim_text
    assert "JUMP -mjd_jump_pta" not in par_text


@pytest.mark.unit
def test_create_all_metapulsars_forwards_convert_flag(tmp_path, monkeypatch):
    file_data = _write_jump_dataset(tmp_path, pta="PPTA")
    seen = {}
    original = MetaPulsarFactory.create_metapulsar

    def _capture(self, **kwargs):
        seen["convert_jump_mjd"] = kwargs.get("convert_jump_mjd")
        return original(self, **{**kwargs, "convert_jump_mjd": False})

    monkeypatch.setattr(MetaPulsarFactory, "create_metapulsar", _capture)
    create_all_metapulsars(
        file_data=file_data,
        convert_jump_mjd=True,
        canonicalize_tim=True,
    )
    assert seen["convert_jump_mjd"] is True


def _single_pta_file_data(
    tmp_path: Path,
    *,
    pta: str = "EPTA",
    par_extra: str = "",
    tim_text: str,
    timing_package: str = "pint",
) -> dict:
    par = FIXTURE_PAR.read_text(encoding="utf-8") + par_extra
    par_path = tmp_path / f"{pta}.par"
    tim_path = tmp_path / f"{pta}.tim"
    par_path.write_text(par, encoding="utf-8")
    tim_path.write_text(tim_text, encoding="utf-8")
    return {
        pta: [
            {
                "par": par_path,
                "tim": tim_path,
                "par_content": par,
                "timing_package": timing_package,
            }
        ]
    }


@pytest.mark.unit
def test_pn_yes_transfers_release_mode_onto_engine_par(tmp_path):
    """PN rewrite drops MODE from the tim; engine/retained par keep it."""
    file_data = _single_pta_file_data(
        tmp_path,
        tim_text=(
            "FORMAT 1\nMODE 1\n"
            "test1 2627.949 55758.3650868593914 10.917 g -sys TEST\n"
            "test2 2627.949 55768.3650868593914 10.917 g -sys TEST\n"
        ),
    )
    mp = MetaPulsarFactory().create_metapulsar(
        file_data=file_data,
        combination_strategy="per_pta",
        use_pulse_numbers="yes",
        canonicalize_tim=True,
    )
    retained = mp._pta_files["EPTA"]
    tim_text = retained.tim_path.read_text(encoding="utf-8")
    par_text = retained.par_path.read_text(encoding="utf-8")
    assert not any(
        line.split() and line.split()[0].upper() == "MODE"
        for line in tim_text.splitlines()
    )
    assert "MODE 1" in par_text
    toa_lines = [line for line in tim_text.splitlines() if line.startswith(" ")]
    assert all(line.split().count("-pn") == 1 for line in toa_lines)
    assert all(not _pint_legacy_heuristic_hit(line) for line in toa_lines)
    mode_par = retained.par_path.with_name("EPTA.mode.par")
    assert mode_par.is_file()
    assert "MODE 1" in mode_par.read_text(encoding="utf-8")


@pytest.mark.unit
def test_absent_tim_mode_preserves_par_mode(tmp_path):
    """None means no override — do not write MODE 0."""
    file_data = _single_pta_file_data(
        tmp_path,
        par_extra="MODE 1\n",
        tim_text="FORMAT 1\ntest1 1400.0 54510.0 1.5 g -sys TEST\n",
    )
    factory = MetaPulsarFactory()
    # Exercise the transfer helper directly to assert changed-set semantics.
    engine_pars = {"EPTA": file_data["EPTA"][0]["par"]}
    single = {pta: entries[0] for pta, entries in file_data.items()}
    updated, changed = factory._apply_tim_mode_transfer(
        engine_pars=engine_pars,
        file_data=single,
        pta_file_dir=tmp_path / "session",
    )
    assert changed == set()
    assert updated["EPTA"] == engine_pars["EPTA"]
    assert not (tmp_path / "session" / "EPTA.mode.par").exists()

    mp = factory.create_metapulsar(
        file_data=file_data,
        combination_strategy="per_pta",
        use_pulse_numbers="no",
        canonicalize_tim=True,
    )
    assert "MODE 1" in mp._pta_files["EPTA"].par_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_legacy_untagged_release_mode_survives_factory_conversion(tmp_path):
    """Untagged Princeton TOAs convert via PINT; MODE transfers."""
    file_data = _single_pta_file_data(
        tmp_path,
        timing_package="pint",
        tim_text=(
            "MODE 1\n"
            "1               1400.000 54510.2858714192189    1.50\n"
            "1               1400.000 54520.2767051885166    1.50\n"
        ),
    )
    mp = MetaPulsarFactory().create_metapulsar(
        file_data=file_data,
        combination_strategy="per_pta",
        use_pulse_numbers="no",
        canonicalize_tim=True,
    )
    assert "MODE 1" in mp._pta_files["EPTA"].par_path.read_text(encoding="utf-8")
    assert mp._pta_files["EPTA"].tim_path.is_file()


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.requires_libstempo
def test_legacy_format0_release_mode_survives_factory_conversion(tmp_path):
    """Explicit FORMAT 0 (reference-data shape) via tempo2."""
    file_data = _single_pta_file_data(
        tmp_path,
        timing_package="tempo2",
        tim_text=(
            "FORMAT 0\nMODE 1\n"
            " test1 1400.0 54510.0 1.5 g\n"
            " test2 1400.0 54520.0 1.5 g\n"
        ),
    )
    mp = MetaPulsarFactory().create_metapulsar(
        file_data=file_data,
        combination_strategy="per_pta",
        use_pulse_numbers="no",
        canonicalize_tim=True,
    )
    assert "MODE 1" in mp._pta_files["EPTA"].par_path.read_text(encoding="utf-8")
    assert mp._pta_files["EPTA"].tim_path.is_file()


@pytest.mark.unit
def test_shared_export_skips_self_copy_without_mode(tmp_path):
    """Unchanged engine par is already the shared export destination."""
    file_data = _single_pta_file_data(
        tmp_path,
        tim_text="FORMAT 1\ntest1 1400.0 54510.0 1.5 g -sys TEST\n",
    )
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    mp = MetaPulsarFactory().create_metapulsar(
        file_data=file_data,
        combination_strategy="shared",
        parfile_output_dir=export_dir,
        use_pulse_numbers="no",
        canonicalize_tim=True,
    )
    assert mp is not None
    exports = list(export_dir.glob("*_shared_EPTA.par"))
    assert exports


@pytest.mark.unit
def test_shared_export_skips_self_copy_without_jump_mjd(tmp_path):
    """convert_jump_mjd with no JUMP MJD must not SameFileError."""
    file_data = _single_pta_file_data(
        tmp_path,
        tim_text="FORMAT 1\ntest1 1400.0 54510.0 1.5 g -sys TEST\n",
    )
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    mp = MetaPulsarFactory().create_metapulsar(
        file_data=file_data,
        combination_strategy="shared",
        parfile_output_dir=export_dir,
        use_pulse_numbers="no",
        convert_jump_mjd=True,
        canonicalize_tim=True,
    )
    assert mp is not None


@pytest.mark.unit
def test_shared_export_contains_transferred_mode(tmp_path):
    """Changed PTA's shared export carries MODE 1."""
    file_data = _single_pta_file_data(
        tmp_path,
        tim_text=("FORMAT 1\nMODE 1\ntest1 1400.0 54510.0 1.5 g -sys TEST\n"),
    )
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    mp = MetaPulsarFactory().create_metapulsar(
        file_data=file_data,
        combination_strategy="shared",
        parfile_output_dir=export_dir,
        use_pulse_numbers="no",
        canonicalize_tim=True,
    )
    exported = export_dir / f"{mp.name}_shared_EPTA.par"
    assert "MODE 1" in exported.read_text(encoding="utf-8")
