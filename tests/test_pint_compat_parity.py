"""The two ``pint_compat`` twins must stay behaviourally identical.

MetaPulsar vendors its own copy of the pure PINT name helpers so that
``import metapulsar`` never pulls in nltiming (see ``test_import_isolation``).
That isolation test keeps the two modules *separate*; nothing else keeps them
*equal*, and a fold that exists on one side only is exactly how a host key and
a mapping value stop denoting the same parameter.

The nltiming import here is test-only, so package-level isolation is untouched.
It is also deliberately a plain import, not ``importorskip``: nltiming is a
mandatory dependency, so its absence must fail this contract rather than skip
it into a green run.
"""

import pytest

from nltiming import pint_compat as nlt_compat

from metapulsar import pint_compat as mp_compat

NAMES = [
    "F0",
    "F1",
    "PX",
    "ECC",
    "E",
    "ECCDOT",
    "EDOT",
    "STIG",
    "STIGMA",
    "A1DOT",
    "XDOT",
    "RAJ",
    "RA",
    "ELONG",
    "LAMBDA",
    "DMX_0001",
    "JUMP",
    "JUMP1",
    "JUMP12",
    "FDJUMP1",
    "FD1JUMP1",
    "FDJUMP1_2",
    "FD1JUMP2",
    "FD2JUMP1",
    "FD1JUMP",
    "FDJUMPDM",
    "FDJUMPDM1",
    "FDJUMPDM_2",
    "FDJUMPLOG",
    "FDJUMP_SCALE",
    "Offset",
    "NOT_A_PARAM",
    "",
]

FUNCTIONS = [
    "resolve_parameter_alias",
    "resolve_fit_column_name",
    "canonicalize_fdjump_name",
    "fdjump_aliases",
    "pint_parameter_name",
    "get_aliases_for_parameter",
]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("fname", FUNCTIONS)
def test_twins_agree(fname, name):
    mine = getattr(mp_compat, fname)(name)
    theirs = getattr(nlt_compat, fname)(name)
    assert mine == theirs, f"{fname}({name!r}): {mine!r} != {theirs!r}"


@pytest.mark.parametrize("fname", FUNCTIONS)
def test_twins_reject_non_strings_alike(fname):
    for module in (mp_compat, nlt_compat):
        with pytest.raises(TypeError, match="parameter-name string"):
            getattr(module, fname)(None)


@pytest.mark.parametrize("fname", FUNCTIONS)
def test_twins_expose_the_same_name_resolution_surface(fname):
    assert callable(getattr(mp_compat, fname, None))
    assert callable(getattr(nlt_compat, fname, None))
