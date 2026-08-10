"""Tests for GLS / native delta apply and par write helpers."""

from __future__ import annotations

import astropy.units as u
import pytest

from metapulsar.parfile_update import (
    apply_native_deltas,
    apply_pint_designmatrix_deltas,
    pint_designmatrix_delta_to_quantity,
    revert_until_pint_valid,
)


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
