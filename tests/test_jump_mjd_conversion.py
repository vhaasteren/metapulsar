"""Factory-level JUMP MJD → -mjd_jump_pta conversion tests."""

from pathlib import Path

import pytest

from metapulsar.metapulsar_factory import MetaPulsarFactory, create_all_metapulsars

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
    create_all_metapulsars(file_data=file_data, convert_jump_mjd=True)
    assert seen["convert_jump_mjd"] is True
