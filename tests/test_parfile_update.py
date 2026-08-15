"""Tests for GLS / native delta apply and par write helpers."""

from __future__ import annotations

import warnings
from pathlib import Path

import astropy.units as u
import pytest

from metapulsar.parfile_update import (
    ParTransplantError,
    apply_native_deltas,
    apply_pint_designmatrix_deltas,
    pint_designmatrix_delta_to_quantity,
    revert_until_pint_valid,
    transplant_param_values,
)

# --- Fixtures -------------------------------------------------------------- #

#: Source par carrying the dual-engine surface a PINT re-serialization destroys:
#: tempo2 ``FDJUMPn`` spelling, ``MODE 1``, ``TRACK -2``, the TZR trio, and the
#: ``E`` alias for ``ECC``.
SOURCE_PAR = """\
PSR              B1855+09
EPHEM            DE421
CLK              TT(BIPM2015)
UNITS            TDB
T2CMETHOD        IAU2000B
F0               186.4940812707752116 1 0.0000000000328468
F1               -6.205147513395E-16 1 1.379566413719E-19
PEPOCH           54978.000000
POSEPOCH         54978.000000
DMEPOCH          54978.000000
DM               13.299393 1
RAJ              18:57:36.3932884 1 0.00002
DECJ             +09:43:17.29196 1 0.0005
PX               0.2929 1 0.2186
TZRMJD           53358.767912764015642
TZRFRQ           424.000000
TZRSITE          ao
BINARY           DD
A1               9.230780480 1 0.000000203
E                0.0000216340 1 0.0000000236
T0               54975.5128660817 1 0.0019286695
PB               12.32717119132762 1 0.00000000019722
OM               276.536118059963 1 0.056323656112
MODE 1
JUMP             -fe L-wide -0.000009449 1 0.000009439
FDJUMPLOG Y
FDJUMP_SCALE LOG
FDJUMP1 -pta nanograv_9y 1.61666384E-04 1 3.38650356E-05
TRACK -2
"""

#: End-to-end GLS fixture, paired with ``tests/fixtures/sample_parfiles/simple.tim``
#: (5 TOAs, all flagged ``-sys TEST -pta TEST``). No ``TRACK -2``: that tim has no
#: ``-pn`` flags and PINT's TRACK -2 residual path requires them.
GLS_PAR = """\
PSR              J1909-3744
RAJ              19:09:47.4280
DECJ             -37:44:14.326
F0               339.315686 1 0.000001
F1               -1.61e-15 1 1.0e-19
PEPOCH           55000
POSEPOCH         55000
DMEPOCH          55000
DM               10.39
EPHEM            DE440
CLK              UTC(NIST)
UNITS            TDB
T2CMETHOD        IAU2000B
MODE 1
TZRMJD           54500.123456789
TZRFRQ           1400.000000
TZRSITE          g
JUMP             -sys TEST 0.0 0
FDJUMPLOG Y
FDJUMP_SCALE LOG
FDJUMP1 -pta TEST 1.5E-06 1 3.0E-07
"""

SIMPLE_TIM = Path(__file__).parent / "fixtures" / "sample_parfiles" / "simple.tim"


def _load(par_text: str, tmp_path: Path):
    """Load ``par_text`` with PINT; returns the model."""
    from pint.models import get_model

    par = tmp_path / "src.par"
    par.write_text(par_text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return get_model(str(par), allow_T2=True, allow_tcb=True)


def _normalized(text: str) -> set[str]:
    """Whitespace-normalized lines; the fixtures pad their value columns."""
    return {" ".join(line.split()) for line in text.splitlines()}


def _heads(text: str) -> set[str]:
    """First three tokens per line -- key plus mask flag and flag value."""
    return {" ".join(line.split()[:3]) for line in text.splitlines() if line.split()}


def _run_gls(tmp_path: Path, par_text: str = GLS_PAR):
    """Run the GLS writer over the fixture pair; returns (result, output text)."""
    import numpy as np

    from metapulsar.parfile_update import gls_update_and_write_par

    par = tmp_path / "gls.par"
    par.write_text(par_text)
    out = tmp_path / "gls_optimized.par"
    result = gls_update_and_write_par(
        par_path=par,
        tim_path=SIMPLE_TIM,
        variance=np.zeros(0),  # falls back to the TOA errors
        out_par=out,
    )
    return result, out.read_text()


class _FakeParam:
    def __init__(self, value, units):
        self.units = units
        self._quantity = float(value) * units
        self.value = float(value)

    @property
    def quantity(self):
        return self._quantity

    @quantity.setter
    def quantity(self, q):
        self._quantity = q
        self.value = float(q.to(self.units).value)


class _FakeModel:
    def __init__(self, params: dict[str, _FakeParam]):
        self.params = list(params)
        self._params = params

    def __getitem__(self, name):
        return self._params[name]


def test_pint_designmatrix_delta_to_quantity_px_scale():
    # PINT returns column unit 1/(Hz mas); conversion must yield mas.
    dqty = pint_designmatrix_delta_to_quantity(2.5, 1.0 / (u.Hz * u.mas))
    assert dqty.unit.is_equivalent(u.mas)
    assert dqty.to_value(u.mas) == pytest.approx(2.5)


def test_apply_pint_designmatrix_deltas_updates_quantity():
    model = _FakeModel({"PX": _FakeParam(1.0, u.mas)})
    applied = apply_pint_designmatrix_deltas(
        model, {"PX": 3.0}, {"PX": 1.0 / (u.Hz * u.mas)}
    )
    assert applied["PX"] == pytest.approx(4.0)
    assert model["PX"].quantity.to_value(u.mas) == pytest.approx(4.0)


def test_apply_native_deltas_matches_pint_delta_engine_rule():
    model = _FakeModel({"PX": _FakeParam(0.9, u.mas), "F0": _FakeParam(100.0, u.Hz)})
    applied = apply_native_deltas(model, {"PX": 0.1, "F0": 1e-12})
    assert applied["PX"] == pytest.approx(1.0)
    assert applied["F0"] == pytest.approx(100.0 + 1e-12)


def test_revert_until_pint_valid_error_chases_named_param():
    model = _FakeModel({"SINI": _FakeParam(0.5, u.dimensionless_unscaled)})
    before = {"SINI": 0.5}
    model["SINI"].value = 1.5
    model._fail_names = {"SINI"}

    # Make validate fail while SINI differs from before.
    def _validate():
        if abs(model["SINI"].value - before["SINI"]) > 0:
            raise ValueError("SINI must be <= 1")

    model.validate = _validate  # type: ignore[method-assign]
    reverted = revert_until_pint_valid(model, before)
    assert reverted == ["SINI"]
    assert model["SINI"].value == pytest.approx(0.5)


class TestTransplantParValues:
    """An optimized par is the source par with new values. Nothing else."""

    def test_changes_only_updated_value_tokens(self, tmp_path):
        model = _load(SOURCE_PAR, tmp_path)
        model.F0.quantity = model.F0.quantity * (1 + 1e-12)

        text, changed = transplant_param_values(SOURCE_PAR, model, ["F0", "PX"])

        # PX round-trips to its own token, so its line is never touched.
        assert set(changed) == {"F0"}
        src_lines = SOURCE_PAR.splitlines()
        out_lines = text.splitlines()
        assert len(src_lines) == len(out_lines)
        differing = [i for i, (a, b) in enumerate(zip(src_lines, out_lines)) if a != b]
        assert len(differing) == 1
        assert src_lines[differing[0]].split()[0] == "F0"

    def test_preserves_dual_engine_dialect(self, tmp_path):
        model = _load(SOURCE_PAR, tmp_path)
        model.F0.quantity = model.F0.quantity * (1 + 1e-12)

        text, _changed = transplant_param_values(SOURCE_PAR, model, ["F0"])

        kept = _normalized(text)
        for line in (
            "MODE 1",
            "TRACK -2",
            "TZRSITE ao",
            "TZRMJD 53358.767912764015642",
            "TZRFRQ 424.000000",
            "T2CMETHOD IAU2000B",
            "FDJUMP_SCALE LOG",
        ):
            assert line in kept
        heads = _heads(text)
        assert "FDJUMP1 -pta nanograv_9y" in heads
        assert "JUMP -fe L-wide" in heads
        # PINT-only spellings and invented keys must not appear.
        assert "FD1JUMP" not in text
        assert "DMDATA" not in text
        assert "NE_SW1" not in text
        # The source's ECC alias survives as written.
        assert not any(line.split()[0] == "ECC" for line in text.splitlines())

    def test_mask_parameters_keep_flag_and_flag_value(self, tmp_path):
        model = _load(SOURCE_PAR, tmp_path)
        model.FD1JUMP1.value = model.FD1JUMP1.value * 1.01
        model.JUMP1.value = model.JUMP1.value * 1.01

        text, changed = transplant_param_values(
            SOURCE_PAR, model, ["FD1JUMP1", "JUMP1"]
        )

        assert set(changed) == {"FD1JUMP1", "JUMP1"}
        by_key = {line.split()[0]: line.split() for line in text.splitlines()}
        # Only the value token (index 3) moved; flag and flag value are intact.
        assert by_key["FDJUMP1"][1:3] == ["-pta", "nanograv_9y"]
        assert float(by_key["FDJUMP1"][3]) == pytest.approx(model.FD1JUMP1.value)
        assert by_key["FDJUMP1"][4:] == ["1", "3.38650356E-05"]
        assert by_key["JUMP"][1:3] == ["-fe", "L-wide"]
        assert float(by_key["JUMP"][3]) == pytest.approx(model.JUMP1.value)
        assert by_key["JUMP"][4:] == ["1", "0.000009439"]

    def test_every_mask_key_type_is_matched(self, tmp_path):
        """MJD / FREQ / TEL / flag jumps all splice their own value token.

        Each key type stores ``key_value`` differently: ``MJD`` as floats
        (``56000.0`` vs the file's ``56000``), ``FREQ`` as ``Quantity``
        (``str`` would give ``"1200.0 MHz"``), ``TEL`` normalized through the
        observatory registry (``ao`` -> ``arecibo``).
        """
        mask_par = """\
PSR              J1909-3744
RAJ              19:09:47.4280
DECJ             -37:44:14.326
F0               339.315686 1 0.000001
PEPOCH           55000
DM               10.39
EPHEM            DE440
UNITS            TDB
JUMP MJD 56000 55000 1.5e-06 1
JUMP FREQ 1200 1600 2.5e-06 1
JUMP TEL ao 3.5e-06 1
JUMP -fe L-wide 4.5e-06 1
"""
        model = _load(mask_par, tmp_path)
        names = [
            p
            for p in model.params
            if p.startswith("JUMP") and model[p].quantity is not None
        ]
        assert len(names) == 4
        for name in names:
            model[name].value = model[name].value * 2

        text, changed = transplant_param_values(mask_par, model, names)

        assert set(changed) == set(names)
        for source, out in zip(mask_par.splitlines(), text.splitlines()):
            if not source.startswith("JUMP"):
                assert source == out
                continue
            # Key, key-values and fit flag intact; only the value doubled.
            src_tokens, out_tokens = source.split(), out.split()
            value_index = len(src_tokens) - 2
            assert out_tokens[:value_index] == src_tokens[:value_index]
            assert out_tokens[value_index + 1 :] == src_tokens[value_index + 1 :]
            assert float(out_tokens[value_index]) == pytest.approx(
                2 * float(src_tokens[value_index])
            )

    def test_alias_spelled_source_line_is_matched(self, tmp_path):
        """``ECC`` is the PINT name; the source spells it ``E``."""
        model = _load(SOURCE_PAR, tmp_path)
        model.ECC.value = model.ECC.value * 1.1

        text, changed = transplant_param_values(SOURCE_PAR, model, ["ECC"])

        assert set(changed) == {"ECC"}
        by_key = {line.split()[0]: line.split() for line in text.splitlines()}
        assert "ECC" not in by_key
        assert float(by_key["E"][1]) == pytest.approx(model.ECC.value)

    def test_duplicate_source_line_raises(self, tmp_path):
        model = _load(SOURCE_PAR, tmp_path)
        model.PX.value = 0.5
        doubled = SOURCE_PAR.replace(
            "PX               0.2929 1 0.2186",
            "PX               0.2929 1 0.2186\nPX               0.2929 1 0.2186",
        )
        with pytest.raises(ParTransplantError, match="matches 2 active par lines"):
            transplant_param_values(doubled, model, ["PX"])

    def test_parameter_absent_from_source_raises(self, tmp_path):
        model = _load(SOURCE_PAR, tmp_path)
        model.PX.value = 0.5
        without_px = "\n".join(
            line for line in SOURCE_PAR.splitlines() if not line.startswith("PX")
        )
        with pytest.raises(ParTransplantError, match="matches 0 active par lines"):
            transplant_param_values(without_px, model, ["PX"])

    def test_unknown_parameter_raises_keyerror(self, tmp_path):
        model = _load(SOURCE_PAR, tmp_path)
        with pytest.raises(KeyError):
            transplant_param_values(SOURCE_PAR, model, ["NOT_A_PARAM"])

    def test_output_round_trips_through_pint(self, tmp_path):
        from pint.models import get_model

        model = _load(SOURCE_PAR, tmp_path)
        model.PX.value = 0.4102
        text, _changed = transplant_param_values(SOURCE_PAR, model, ["PX"])

        out = tmp_path / "out.par"
        out.write_text(text)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reloaded = get_model(str(out), allow_T2=True, allow_tcb=True)
        assert reloaded.PX.value == pytest.approx(0.4102)
        assert reloaded.FD1JUMP1.value == pytest.approx(1.61666384e-04)


class TestGlsUpdateAndWritePar:
    def test_writes_dual_engine_par_with_provenance(self, tmp_path):
        result, body = _run_gls(tmp_path)

        assert result.skipped == ("Offset",)
        changed = dict((name, (old, new)) for name, old, new in result.changed_tokens)
        assert "FD1JUMP1" in changed

        heads = _heads(body)
        assert "FDJUMP1 -pta TEST" in heads
        assert "FD1JUMP" not in body
        kept = _normalized(body)
        assert "MODE 1" in kept
        assert "TZRSITE g" in kept
        assert "FDJUMP_SCALE LOG" in kept
        # The frozen JUMP line covers every TOA and must survive byte-identical.
        assert "JUMP -sys TEST 0.0 0" in kept

        assert "# writer: transplant" in body
        assert "# Product: gls-optimized" in body
        assert "# gls_delta_convention: pint_designmatrix" in body
        assert "# changed_params: " in body

    def test_source_product_header_is_replaced(self, tmp_path):
        from metapulsar.parfile_header import ensure_metapulsar_par_header

        stamped = ensure_metapulsar_par_header(
            GLS_PAR,
            extra={"Product": "combination", "reference_pta": "nanograv_9y"},
        )
        _result, body = _run_gls(tmp_path, stamped)

        assert "# Product: gls-optimized" in body
        assert "# Product: combination" not in body
        assert "# reference_pta: nanograv_9y" not in body

    def test_reverted_parameters_are_left_byte_identical(self, tmp_path):
        """A parameter reverted by PINT validate() must not reach the transplant."""
        import numpy as np

        from metapulsar.parfile_update import gls_update_and_write_par

        # SINI free and already at the edge: any positive delta pushes it past 1,
        # so revert_until_pint_valid() puts it back. TRACK -2 is dropped because
        # simple.tim carries no -pn and PINT's tracked residuals require them.
        par_text = SOURCE_PAR.replace(
            "OM               276.536118059963 1 0.056323656112",
            "OM               276.536118059963 1 0.056323656112\n"
            "SINI             0.999999 1 0.000178\n"
            "M2               0.233837 1 0.011278",
        ).replace("TRACK -2\n", "")
        par = tmp_path / "revert.par"
        par.write_text(par_text)
        out = tmp_path / "revert_out.par"

        result = gls_update_and_write_par(
            par_path=par,
            tim_path=SIMPLE_TIM,
            variance=np.zeros(0),
            out_par=out,
            design_matrix=np.tile(np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]), 2),
            param_names=["SINI", "F0"],
            delta_convention="native",
        )

        # No skip guard: the push is deterministic here, so a PINT that starts
        # accepting SINI > 1 must fail this test rather than silently drop it.
        assert result.reverted == ("SINI",)
        changed = {name for name, _old, _new in result.changed_tokens}
        for name in result.reverted:
            assert name not in changed
        source_lines = {
            line.split()[0]: line for line in par_text.splitlines() if line.split()
        }
        out_lines = {
            line.split()[0]: line
            for line in out.read_text().splitlines()
            if line.split() and not line.startswith("#")
        }
        for name in result.reverted:
            assert out_lines[name] == source_lines[name]

    def test_unmoved_free_parameters_are_not_rewritten(self, tmp_path):
        """A zero-delta free parameter must not pick up a cosmetic reformat."""
        import numpy as np

        from metapulsar.parfile_update import gls_update_and_write_par

        par = tmp_path / "gls.par"
        par.write_text(GLS_PAR)
        out = tmp_path / "out.par"
        # A zero design matrix solves to zero deltas: nothing moves at all.
        result = gls_update_and_write_par(
            par_path=par,
            tim_path=SIMPLE_TIM,
            variance=np.zeros(0),
            design_matrix=np.zeros((5, 2)),
            param_names=["F0", "FD1JUMP1"],
            delta_convention="native",
            out_par=out,
        )
        assert result.changed_tokens == ()
        body = out.read_text()
        source_lines = [ln for ln in GLS_PAR.splitlines() if ln.strip()]
        body_lines = [
            ln for ln in body.splitlines() if ln.strip() and not ln.startswith("#")
        ]
        assert body_lines == source_lines


@pytest.mark.requires_libstempo
def test_optimized_par_is_readable_by_tempo2(tmp_path):
    """The reported regression: tempo2 must see the GLS FDJUMP value.

    Runs through ``sandbox_tempo2`` (out-of-process) like every other tempo2
    test here -- in-process libstempo segfaults at interpreter teardown.
    """
    pytest.importorskip("libstempo")
    from metapulsar.sandbox_tempo2 import tempopulsar

    result, _body = _run_gls(tmp_path)
    changed = {name: new for name, _old, new in result.changed_tokens}
    assert "FD1JUMP1" in changed

    psr = tempopulsar(
        parfile=str(tmp_path / "gls_optimized.par"),
        timfile=str(SIMPLE_TIM),
        dofit=False,
    )
    names = [p for p in psr.pars(which="all") if p.upper().startswith("FDJUMP")]
    assert "FDJUMP1" in names, f"tempo2 did not parse FDJUMP1 (saw {names})"
    assert psr["FDJUMP1"].val == pytest.approx(float(changed["FD1JUMP1"]))


def test_write_combination_par_includes_metapulsar_header(tmp_path):
    from metapulsar.combination_writer import write_combination_par

    write_combination_par(
        reference_pta="pta_a",
        pta_par_texts={
            "pta_a": "PSR J1\nF0 100\nDM 10\n",
            "pta_b": "PSR J1\nF0 100\nDM 10.5\n",
        },
        out_path=tmp_path / "c.par",
        combination_options={
            "Product": "combination",
            "combination_strategy": "shared",
            "use_pulse_numbers": "yes",
            "alignment_policy.unsupported": "strip",
        },
    )
    body = (tmp_path / "c.par").read_text()
    assert body.startswith("# Created:")
    assert "# By:      MetaPulsar" in body
    assert "# Product: combination" in body
    assert "# combination_strategy: shared" in body
    assert "# use_pulse_numbers: yes" in body
    assert "# alignment_policy.unsupported: strip" in body
    assert "PSR J1" in body
