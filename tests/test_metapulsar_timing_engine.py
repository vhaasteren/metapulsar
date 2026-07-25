"""Multi-PTA timing-engine exact-string merge tests (Fix M2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from metapulsar.metapulsar import MetaPulsar
from metapulsar.parameter_manager import ParameterInconsistencyError


def _par_text(*, px: str = "0.5123456789012345", dm: str = "13.299") -> str:
    return (
        "PSR J1857+0943\n"
        "PEPOCH 55000\n"
        "F0 186.49408138134548363 1\n"
        "F1 -6.20415e-16 1\n"
        "F2 0.0 1\n"
        "RAJ 18:57:36.3937051 1\n"
        "DECJ +09:43:17.29123 1\n"
        "PMRA -2.5644 1\n"
        "PMDEC -5.0575 1\n"
        f"PX {px} 1\n"
        f"DM {dm}\n"
        "POSEPOCH 55000\n"
        "DMEPOCH 55000\n"
        "UNITS TDB\n"
    )


def _minimal_shared_metapulsar(
    tmp_path: Path, *, epta_dm: str = "13.299", ppta_dm: str = "13.299"
) -> MetaPulsar:
    """Build a two-PTA MetaPulsar with identical retained shared tokens."""
    from metapulsar.mockpulsar import create_mock_libstempo

    (tmp_path / "epta.par").write_text(_par_text(dm=epta_dm), encoding="utf-8")
    (tmp_path / "ppta.par").write_text(_par_text(dm=ppta_dm), encoding="utf-8")
    for pta in ("epta", "ppta"):
        (tmp_path / f"{pta}.tim").write_text("FORMAT 1\n", encoding="utf-8")

    mock_a = create_mock_libstempo(
        n_toas=20, name="J1857+0943", telescope="epta", seed=1
    )
    mock_b = create_mock_libstempo(
        n_toas=20, name="J1857+0943", telescope="ppta", seed=2
    )

    pulsar = MetaPulsar(
        {"epta": mock_a, "ppta": mock_b},
        combination_strategy="shared",
        combine_components=["astrometry", "spindown"],
        pta_files={
            "epta": {
                "par_path": tmp_path / "epta.par",
                "tim_path": tmp_path / "epta.tim",
                "timing_package": "tempo2",
            },
            "ppta": {
                "par_path": tmp_path / "ppta.par",
                "tim_path": tmp_path / "ppta.tim",
                "timing_package": "tempo2",
            },
        },
    )
    # Mock construction leaves PTA-specific DM out of `_parfile_dicts`; seed
    # it from the retained pars so timing_engine() can resolve local exact
    # strings the same way a real tempo2 load would.
    for pta_name, dm in (("epta", epta_dm), ("ppta", ppta_dm)):
        pulsar._parfile_dicts[pta_name]["DM"] = dm
    return pulsar


def _build_linearized_engine(pulsar: MetaPulsar):
    """Public boundary used by nltiming: MetaPulsar.timing_engine()."""
    return pulsar.timing_engine(
        {"tempo2": "libstempo"},
        linearized=True,
    )


@pytest.mark.slow
def test_shared_exact_strings_agree_across_contributions(tmp_path):
    pulsar = _minimal_shared_metapulsar(tmp_path)
    for name in ("RAJ", "PX", "F0"):
        assert len(pulsar._fitparameters[name]) > 1

    engine = _build_linearized_engine(pulsar)
    merged = dict(engine.reference_theta_exact())
    for name in ("RAJ", "PX", "F0"):
        assert name in merged
        # Same exact string is projected into every owning contribution.
        for contribution in engine.contributions:
            if name in contribution.engine.fitpars:
                local = contribution.engine.reference_theta_exact()[name]
                assert local == merged[name]


@pytest.mark.slow
def test_pta_specific_parameters_keep_local_exact_strings(tmp_path):
    """Distinct per-PTA DM tokens must survive as PTA-specific exact strings."""
    pulsar = _minimal_shared_metapulsar(
        tmp_path, epta_dm="11.111111", ppta_dm="22.222222"
    )
    dm_names = [
        n
        for n, owners in pulsar._fitparameters.items()
        if len(owners) == 1 and n.split("_", 1)[0] == "DM"
    ]
    assert len(dm_names) >= 2

    engine = _build_linearized_engine(pulsar)
    exact_by_name = {}
    for contribution in engine.contributions:
        local = contribution.engine.reference_theta_exact()
        for name in dm_names:
            if name in local:
                exact_by_name[name] = local[name]

    assert len(exact_by_name) >= 2
    values = set(exact_by_name.values())
    assert len(values) >= 2
    assert "11.111111" in values
    assert "22.222222" in values


@pytest.mark.slow
def test_post_harmonization_corruption_raises_before_engine(tmp_path):
    pulsar = _minimal_shared_metapulsar(tmp_path)
    shared = "PX"
    owners = list(pulsar._fitparameters[shared])
    victim = owners[-1]
    mapped = pulsar._fitparameters[shared][victim]
    par_path = pulsar._pta_files[victim].par_path
    text = par_path.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        tokens = line.split()
        if tokens and tokens[0].upper() == mapped.upper():
            tokens[1] = "9.999999999"
            lines.append(" ".join(tokens))
        else:
            lines.append(line)
    par_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pulsar._invalidate_timing_caches()

    with pytest.raises(ParameterInconsistencyError, match=shared) as excinfo:
        _build_linearized_engine(pulsar)
    message = str(excinfo.value)
    assert owners[0] in message and victim in message
    # Engine cache must remain empty — failure happened before construction.
    assert pulsar._timing_engine_cache == {}
