"""Unit tests for legacy MetaParfiles parity protocol harmonization."""

from io import StringIO
from unittest.mock import patch

import pytest

from metapulsar.legacy.metapulsar import MetaParfiles


def _build_input(pta: str, par_content: str, package: str = "tempo2") -> dict:
    return {
        "pta": pta,
        "parfile": StringIO(par_content),
        "package": package,
    }


def test_legacy_parity_protocol_cross_engine_ecliptic_harmonization():
    par1 = (
        "PSR J1600-3053\n"
        "LAMBDA 244.347\n"
        "BETA -10.07\n"
        "ECL IERS2010\n"
        "T2CMETHOD TEMPO\n"
        "EPHEM DE436\n"
        "CLOCK TT(BIPM2015)\n"
        "UNITS TDB\n"
    )
    par2 = (
        "PSR J1600-3053\n"
        "LAMBDA 244.347\n"
        "BETA -10.07\n"
        "ECL IERS2010\n"
        "T2CMETHOD TEMPO\n"
        "EPHEM DE440\n"
        "CLK TT(BIPM2021)\n"
        "UNITS TDB\n"
    )
    mpf = MetaParfiles(
        parfiles=[
            _build_input("epta", par1, package="tempo2"),
            _build_input("ppta", par2, package="pint"),
        ],
        merge_astrometry=False,
        merge_spin=False,
        merge_binary=False,
        merge_dm=False,
        convert=True,
    )

    for pfd in mpf._parfiles:
        pd = pfd["pardict_conv"]
        assert pd["UNITS"] == ["TDB"]
        assert pd["ECL"] == ["IERS2003"]
        assert "T2CMETHOD" not in pd
        assert pd["EPHEM"] == ["DE436"]
        assert pd["CLOCK"] == ["TT(BIPM2015)"]
        assert "CLK" not in pd


def test_legacy_parity_protocol_equatorial_warning_and_no_ecl():
    par1 = (
        "PSR J1857+0943\n"
        "RAJ 18:57:36.3937\n"
        "DECJ +09:43:17.291\n"
        "ECL IERS2010\n"
        "T2CMETHOD TEMPO\n"
        "EPHEM DE440\n"
        "CLOCK TT(BIPM2021)\n"
        "UNITS TDB\n"
    )
    par2 = (
        "PSR J1857+0943\n"
        "RAJ 18:57:36.3937\n"
        "DECJ +09:43:17.291\n"
        "T2CMETHOD TEMPO\n"
        "EPHEM DE436\n"
        "CLK TT(BIPM2015)\n"
        "UNITS TDB\n"
    )

    with patch("metapulsar.legacy.metapulsar.logger.warning") as mock_warning:
        mpf = MetaParfiles(
            parfiles=[
                _build_input("epta", par1, package="tempo2"),
                _build_input("ppta", par2, package="pint"),
            ],
            merge_astrometry=False,
            merge_spin=False,
            merge_binary=False,
            merge_dm=False,
            convert=True,
        )

    for pfd in mpf._parfiles:
        pd = pfd["pardict_conv"]
        assert pd["UNITS"] == ["TDB"]
        assert "ECL" not in pd
        assert "T2CMETHOD" not in pd
    assert mock_warning.call_count == 2


def test_legacy_parity_protocol_pint_only_aligns_ecl_to_reference_or_default():
    par1 = (
        "PSR J1600-3053\n"
        "LAMBDA 244.347\n"
        "BETA -10.07\n"
        "ECL IERS2010\n"
        "EPHEM DE440\n"
        "CLOCK TT(BIPM2021)\n"
        "UNITS TDB\n"
    )
    par2 = (
        "PSR J1600-3053\n"
        "LAMBDA 244.347\n"
        "BETA -10.07\n"
        "EPHEM DE436\n"
        "CLK TT(BIPM2015)\n"
        "UNITS TDB\n"
    )

    mpf = MetaParfiles(
        parfiles=[
            _build_input("epta", par1, package="pint"),
            _build_input("ppta", par2, package="pint"),
        ],
        merge_astrometry=False,
        merge_spin=False,
        merge_binary=False,
        merge_dm=False,
        convert=True,
    )

    for pfd in mpf._parfiles:
        pd = pfd["pardict_conv"]
        assert pd["ECL"] == ["IERS2010"]
        assert "T2CMETHOD" not in pd


def test_legacy_parity_protocol_tempo2_only_preserves_shared_t2cmethod():
    par1 = (
        "PSR J1600-3053\n"
        "LAMBDA 244.347\n"
        "BETA -10.07\n"
        "ECL IERS2010\n"
        "T2CMETHOD TEMPO\n"
        "EPHEM DE436\n"
        "CLOCK TT(BIPM2015)\n"
        "UNITS TDB\n"
    )
    par2 = (
        "PSR J1600-3053\n"
        "LAMBDA 244.347\n"
        "BETA -10.07\n"
        "ECL IERS2010\n"
        "T2CMETHOD TEMPO\n"
        "EPHEM DE440\n"
        "CLK TT(BIPM2021)\n"
        "UNITS TDB\n"
    )
    mpf = MetaParfiles(
        parfiles=[
            _build_input("epta", par1, package="tempo2"),
            _build_input("ppta", par2, package="tempo2"),
        ],
        merge_astrometry=False,
        merge_spin=False,
        merge_binary=False,
        merge_dm=False,
        convert=True,
    )

    for pfd in mpf._parfiles:
        pd = pfd["pardict_conv"]
        assert pd["ECL"] == ["IERS2010"]
        assert pd["T2CMETHOD"] == ["TEMPO"]


def test_legacy_parity_protocol_tempo2_only_aligns_heterogeneous_conventions():
    par1 = (
        "PSR J1600-3053\n"
        "LAMBDA 244.347\n"
        "BETA -10.07\n"
        "ECL IERS2010\n"
        "T2CMETHOD TEMPO\n"
        "EPHEM DE436\n"
        "CLOCK TT(BIPM2015)\n"
        "UNITS TDB\n"
    )
    par2 = (
        "PSR J1600-3053\n"
        "LAMBDA 244.347\n"
        "BETA -10.07\n"
        "ECL IERS2003\n"
        "T2CMETHOD IAU2000B\n"
        "EPHEM DE440\n"
        "CLK TT(BIPM2021)\n"
        "UNITS TDB\n"
    )
    mpf = MetaParfiles(
        parfiles=[
            _build_input("epta", par1, package="tempo2"),
            _build_input("ppta", par2, package="tempo2"),
        ],
        merge_astrometry=False,
        merge_spin=False,
        merge_binary=False,
        merge_dm=False,
        convert=True,
    )

    for pfd in mpf._parfiles:
        pd = pfd["pardict_conv"]
        assert pd["ECL"] == ["IERS2003"]
        assert pd["T2CMETHOD"] == ["TEMPO"]


def test_legacy_parity_protocol_single_pta_tempo2_keeps_t2cmethod():
    par = (
        "PSR J1600-3053\n"
        "LAMBDA 244.347\n"
        "BETA -10.07\n"
        "ECL IERS2010\n"
        "T2CMETHOD TEMPO\n"
        "EPHEM DE436\n"
        "CLOCK TT(BIPM2015)\n"
        "UNITS TDB\n"
    )
    mpf = MetaParfiles(
        parfiles=[_build_input("epta", par, package="tempo2")],
        merge_astrometry=False,
        merge_spin=False,
        merge_binary=False,
        merge_dm=False,
        convert=True,
    )
    pd = mpf._parfiles[0]["pardict_conv"]
    assert pd["T2CMETHOD"] == ["TEMPO"]
    assert pd["ECL"] == ["IERS2010"]


def test_legacy_parity_protocol_mixed_astrometry_raises():
    par = (
        "PSR J1857+0943\n"
        "RAJ 18:57:36.3937\n"
        "DECJ +09:43:17.291\n"
        "LAMBDA 244.347\n"
        "BETA -10.07\n"
        "EPHEM DE440\n"
        "CLOCK TT(BIPM2021)\n"
        "UNITS TDB\n"
    )
    with pytest.raises(ValueError, match="Mixed astrometry detected"):
        MetaParfiles(
            parfiles=[_build_input("epta", par, package="tempo2")],
            merge_astrometry=False,
            merge_spin=False,
            merge_binary=False,
            merge_dm=False,
            convert=True,
        )


def test_legacy_parity_protocol_missing_reference_clock_raises():
    par = (
        "PSR J1857+0943\n"
        "RAJ 18:57:36.3937\n"
        "DECJ +09:43:17.291\n"
        "EPHEM DE440\n"
        "UNITS TDB\n"
    )
    with pytest.raises(ValueError, match="CLOCK/CLK"):
        MetaParfiles(
            parfiles=[_build_input("epta", par, package="tempo2")],
            merge_astrometry=False,
            merge_spin=False,
            merge_binary=False,
            merge_dm=False,
            convert=True,
        )
