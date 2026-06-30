"""Slice-2 tests for timing fit-parameter partition policy."""

from dataclasses import dataclass

import pytest

from metapulsar.timing.partition import (
    default_marginalized_fitpars,
    resolve_partition,
)


@dataclass
class _FakeComponent:
    category: str
    params: tuple[str, ...]


class _FakeModel:
    def __init__(self):
        self.components = {
            "astro": _FakeComponent("astrometry", ("RAJ", "DECJ")),
            "spin": _FakeComponent("spindown", ("F0", "F1", "EDOT")),
            "disp": _FakeComponent("dispersion_constant", ("DM", "DM1")),
            "dmx": _FakeComponent("dispersion_dmx", ("DMX_0001",)),
            "binary": _FakeComponent("pulsar_system", ("A1", "PB")),
        }


class _FakeHost:
    def __init__(self):
        self.fitpars = (
            "RAJ",
            "DECJ",
            "F0",
            "F1",
            "DM",
            "DM1",
            "DMX_0001",
            "A1",
            "PB",
            "JUMP1",
        )
        self._model = _FakeModel()

    def pint_model(self):
        return self._model


def test_default_partition_uses_pint_components():
    host = _FakeHost()
    marginalized = default_marginalized_fitpars(host)
    assert marginalized == ("RAJ", "DECJ", "F0", "F1", "DM", "DM1", "DMX_0001")


def test_resolve_partition_default_and_indices():
    host = _FakeHost()
    part = resolve_partition(host, marginalize="default")
    assert part.marginalized == ("RAJ", "DECJ", "F0", "F1", "DM", "DM1", "DMX_0001")
    assert part.sampled == ("A1", "PB", "JUMP1")
    assert part.idx_marginalized == (0, 1, 2, 3, 4, 5, 6)
    assert part.idx_sampled == (7, 8, 9)


def test_resolve_partition_explicit_list():
    host = _FakeHost()
    part = resolve_partition(host, marginalize=["F0", "PB"])
    assert part.marginalized == ("F0", "PB")
    assert part.sampled == ("RAJ", "DECJ", "F1", "DM", "DM1", "DMX_0001", "A1", "JUMP1")


def test_resolve_partition_none_marginalized():
    host = _FakeHost()
    part = resolve_partition(host, marginalize=None)
    assert part.marginalized == ()
    assert part.sampled == host.fitpars


def test_resolve_partition_unknown_and_duplicate_errors():
    host = _FakeHost()
    with pytest.raises(ValueError, match="Unknown fit parameters"):
        resolve_partition(host, marginalize=["DOES_NOT_EXIST"])
    with pytest.raises(ValueError, match="Duplicate entries"):
        resolve_partition(host, marginalize=["F0", "F0"])
    with pytest.raises(ValueError, match="marginalize must be"):
        resolve_partition(host, marginalize="F0")


class _FakeCompositeModel:
    """Model whose components carry canonical (unsuffixed) PINT param names."""

    def __init__(self):
        self.components = {
            "astro": _FakeComponent("astrometry", ("RAJ", "DECJ", "PMRA", "PMDEC")),
            "spin": _FakeComponent("spindown", ("F0", "F1")),
            "dmx": _FakeComponent("dispersion_dmx", ("DMX_0001", "DMX_0002")),
            "jumps": _FakeComponent("phase_jump", ("JUMP1", "JUMP2", "JUMP3")),
            "binary": _FakeComponent("pulsar_system", ("PB", "A1", "ECC", "T0")),
        }


class _FakeCompositeHost:
    """Composite host with PTA-suffixed fitpars and a base-name mapping."""

    def __init__(self, *, with_mapping: bool = True):
        bases = (
            "RAJ",
            "DECJ",
            "PMRA",
            "PMDEC",
            "F0",
            "F1",
            "DMX_0001",
            "DMX_0002",
            "JUMP1",
            "JUMP2",
            "JUMP3",
            "PB",
            "A1",
            "ECC",
            "T0",
            "Offset",
        )
        self.fitpars = tuple(f"{b}_ng5" for b in bases)
        self._fitparameters = (
            {f"{b}_ng5": {"ng5": b} for b in bases} if with_mapping else {}
        )
        self._model = _FakeCompositeModel()

    def pint_model(self):
        return self._model


def test_default_marginalizes_linear_block_on_suffixed_composite_host():
    """Regression: PTA-suffixed names must still map to canonical categories.

    The binary params plus the explicit Offset stay sampled; every discovered
    linear nuisance family (astrometry/spindown/DMX/JUMP) is marginalized.
    """
    host = _FakeCompositeHost()
    part = resolve_partition(host, marginalize="default")
    assert part.sampled == ("PB_ng5", "A1_ng5", "ECC_ng5", "T0_ng5", "Offset_ng5")
    assert "JUMP1_ng5" in part.marginalized
    assert "DMX_0001_ng5" in part.marginalized
    assert "RAJ_ng5" in part.marginalized
    assert "Offset_ng5" not in part.marginalized
    assert len(part.marginalized) == 11


def test_default_guard_raises_when_suffix_mapping_unavailable():
    """If name matching silently fails, refuse to sample every timing param."""
    host = _FakeCompositeHost(with_mapping=False)
    with pytest.raises(ValueError, match="empty marginalized set"):
        resolve_partition(host, marginalize="default")
