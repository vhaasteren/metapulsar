"""Typed native names for fit columns (``NativeParam``) end to end.

Covers the BUG 004 shape: a combined PINT leg whose par spells an FDJUMP
``FDJUMP1`` while the PINT model attribute is ``FD1JUMP1`` and the chart id is
``FDJUMP1_1``. Storing one string per (fit column, PTA) made those three names
interchangeable at the call site; storing both native spellings makes the
exact-reference miss impossible to express.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from metapulsar.metapulsar import MetaPulsar, PtaFiles
from metapulsar.parameter_manager import (
    ParameterInconsistencyError,
    ParameterManager,
)
from metapulsar.pint_helpers import (
    NativeParam,
    ParameterSourceMappingError,
    create_pint_model,
)

# A combined-style leg: tempo2-flavoured FDJUMP keywords (what the AEI
# combination writer emits) that PINT reads as FD1JUMP1 / FD2JUMP1.
COMBINED_PAR = (
    "PSR J0613-0200\n"
    "PEPOCH 55000\n"
    "F0 326.6005620 1\n"
    "F1 -1.023e-15 1\n"
    "RAJ 06:13:43.975 1\n"
    "DECJ -02:00:47.22 1\n"
    "DM 38.778 1\n"
    "FDJUMPLOG Y\n"
    "FDJUMP1 -pta nanograv_9y -6.83086470E-06 1\n"
    "FDJUMP2 -pta nanograv_9y 2.42187398E-05 1\n"
    "UNITS TDB\n"
)


def _combined_fitparameters() -> dict:
    """Real producer output for the combined-style par."""
    return ParameterManager(
        file_data={"combined": {"timing_package": "pint", "par_content": COMBINED_PAR}},
        combine_components=["astrometry", "spindown"],
    ).build_parameter_mappings()


def _bare_pulsar(fitparameters: dict, **attrs) -> MetaPulsar:
    """MetaPulsar wired with just enough state for the lookup paths."""
    pulsar = MetaPulsar.__new__(MetaPulsar)
    pulsar.name = "J0613-0200"
    pulsar._fitparameters = fitparameters
    pulsar.fitpars = list(fitparameters)
    pulsar._parfile_dicts = {}
    pulsar._pta_files = {}
    pulsar._pta_data = {}
    pulsar._shared_theta_exact_cache = {}
    pulsar._retained_pint_model_cache = {}
    for key, value in attrs.items():
        setattr(pulsar, key, value)
    return pulsar


class TestCombinedFdjumpProducer:
    """The construction site has both spellings; it must store both."""

    def test_combined_fdjump_columns_carry_both_spellings(self):
        result = _combined_fitparameters()

        first = result.fitparameters["FD1JUMP1_combined"]["combined"]
        assert first == NativeParam(pint_name="FD1JUMP1", par_key="FDJUMP1")
        assert first.identity == "FDJUMP1_1"

        # The identity test: any matching that "finds some FDJUMP" is wrong.
        second = result.fitparameters["FD2JUMP1_combined"]["combined"]
        assert second == NativeParam(pint_name="FD2JUMP1", par_key="FDJUMP2")
        assert second.identity == "FDJUMP2_1"

    def test_offset_synthesis_stores_a_record(self):
        result = _combined_fitparameters()
        assert result.fitparameters["Offset_combined"]["combined"] == NativeParam(
            "Offset", "Offset"
        )


class TestExactReferenceLookup:
    """BUG 004: the source is PINT-keyed, so ``pint_name`` is a direct key."""

    def test_combined_fdjump_resolves_without_missing_theta(self):
        result = _combined_fitparameters()
        pulsar = _bare_pulsar(
            result.fitparameters,
            _parfile_dicts={
                "combined": create_pint_model(COMBINED_PAR).get_params_dict()
            },
        )

        exact = pulsar._pta_theta_exact(
            "combined", ("FD1JUMP1_combined", "FD2JUMP1_combined", "F0")
        )

        # Distinct coefficients, each from its own instance -- not "some FDJUMP".
        assert float(exact["FD1JUMP1_combined"]) == pytest.approx(-6.83086470e-06)
        assert float(exact["FD2JUMP1_combined"]) == pytest.approx(2.42187398e-05)
        assert float(exact["F0"]) == pytest.approx(326.6005620)

    def test_offset_zero_still_wins_only_after_value_sources(self):
        result = _combined_fitparameters()
        pulsar = _bare_pulsar(
            result.fitparameters,
            _parfile_dicts={
                "combined": create_pint_model(COMBINED_PAR).get_params_dict()
            },
        )
        assert pulsar._pta_theta_exact("combined", ("Offset_combined",)) == {
            "Offset_combined": "0.0"
        }

    def test_genuine_absence_names_both_spellings_and_identity(self):
        result = _combined_fitparameters()
        pulsar = _bare_pulsar(result.fitparameters, _parfile_dicts={"combined": {}})

        with pytest.raises(ValueError) as excinfo:
            pulsar._pta_theta_exact("combined", ("FD1JUMP1_combined",))

        message = str(excinfo.value)
        assert "pint_name='FD1JUMP1'" in message
        assert "par_key='FDJUMP1'" in message
        assert "identity='FDJUMP1_1'" in message

    def test_absent_host_key_falls_back_to_the_host_key_itself(self):
        """Hand-built pulsars keep today's permissive behaviour."""
        pulsar = _bare_pulsar({}, _parfile_dicts={"epta": {"F0": "1.0"}})
        assert pulsar._native_param("F0", "epta") == NativeParam("F0", "F0")
        assert pulsar._local_theta_exact("epta", "F0") == "1.0"


class TestRetainedValueToken:
    """Retained par first tokens are foreign spellings: match by identity."""

    @staticmethod
    def _pulsar_with_par(tmp_path: Path, par_text: str, native: NativeParam):
        par_path = tmp_path / "retained.par"
        par_path.write_text(par_text, encoding="utf-8")
        return _bare_pulsar(
            {"FD1JUMP1_combined": {"combined": native}},
            _pta_files={
                "combined": PtaFiles(
                    par_path=par_path,
                    tim_path=tmp_path / "retained.tim",
                    timing_package="pint",
                )
            },
        )

    def test_tempo2_and_pint_spellings_both_satisfy_one_record(self, tmp_path):
        # Maskless lines on purpose: this path serves shared parameters, and
        # the default combine_components carries no mask/prefix instances, so
        # first-token identity is the whole matching rule here.
        native = NativeParam(pint_name="FD1JUMP1", par_key="FDJUMP1")
        for line in (
            "FDJUMP1 -6.83086470E-06 1",
            "FD1JUMP1 -6.83086470E-06 1",
        ):
            pulsar = self._pulsar_with_par(
                tmp_path, f"PSR J0613-0200\n{line}\n", native
            )
            token = pulsar._retained_value_token("combined", "FD1JUMP1_combined")
            assert token == "-6.83086470E-06"

    def test_wrong_instance_does_not_match(self, tmp_path):
        native = NativeParam(pint_name="FD1JUMP1", par_key="FDJUMP1")
        pulsar = self._pulsar_with_par(
            tmp_path,
            "PSR J0613-0200\nFDJUMP2 -pta nanograv_9y 2.42187398E-05 1\n",
            native,
        )
        with pytest.raises(ParameterSourceMappingError) as excinfo:
            pulsar._retained_value_token("combined", "FD1JUMP1_combined")
        message = str(excinfo.value)
        assert "identity='FDJUMP1_1'" in message
        assert "FDJUMP2" in message

    def test_zero_matches_raise_source_mapping_error(self, tmp_path):
        pulsar = self._pulsar_with_par(
            tmp_path, "PSR J0613-0200\nF0 326.6\n", NativeParam("F0", "F0")
        )
        pulsar._fitparameters = {"PX": {"combined": NativeParam("PX", "PX")}}
        with pytest.raises(ParameterSourceMappingError, match="no active line"):
            pulsar._retained_value_token("combined", "PX")

    def test_duplicate_identity_lines_keep_inconsistency_error(self, tmp_path):
        """Both spellings of one instance: still a defective document."""
        native = NativeParam(pint_name="FD1JUMP1", par_key="FDJUMP1")
        pulsar = self._pulsar_with_par(
            tmp_path,
            "PSR J0613-0200\nFDJUMP1 1.0 1\nFD1JUMP1 1.0 1\n",
            native,
        )
        with pytest.raises(ParameterInconsistencyError, match="2 active"):
            pulsar._retained_value_token("combined", "FD1JUMP1_combined")

    def test_missing_value_token_keeps_inconsistency_error(self, tmp_path):
        pulsar = self._pulsar_with_par(
            tmp_path, "PSR J0613-0200\nFDJUMP1\n", NativeParam("FD1JUMP1", "FDJUMP1")
        )
        with pytest.raises(ParameterInconsistencyError, match="no value token"):
            pulsar._retained_value_token("combined", "FD1JUMP1_combined")

    def test_drifted_spelling_still_validates_shared_tokens(self, tmp_path):
        """``ECC``/``E`` after harmonization: exact string equality would miss."""
        epta = tmp_path / "epta.par"
        ng9 = tmp_path / "ng9.par"
        epta.write_text("PSR J0000+0000\nECC 0.00079729755 1\n", encoding="utf-8")
        ng9.write_text("PSR J0000+0000\nE 0.00079729755 1\n", encoding="utf-8")
        pulsar = _bare_pulsar(
            {
                "ECC": {
                    "epta": NativeParam("ECC", "ECC"),
                    "ng9": NativeParam("ECC", "E"),
                }
            },
            _pta_files={
                "epta": PtaFiles(epta, tmp_path / "a.tim", "tempo2"),
                "ng9": PtaFiles(ng9, tmp_path / "b.tim", "tempo2"),
            },
        )
        pulsar._validate_shared_retained_tokens("ECC")  # does not raise

        ng9.write_text("PSR J0000+0000\nE 9.99999999 1\n", encoding="utf-8")
        with pytest.raises(ParameterInconsistencyError, match="ECC"):
            pulsar._validate_shared_retained_tokens("ECC")


class TestPublicMappingRendersParKey:
    def test_timing_parameter_mapping_is_the_par_spelling(self):
        result = _combined_fitparameters()
        pulsar = _bare_pulsar(result.fitparameters)
        mapping = pulsar.timing_parameter_mapping()

        assert mapping["FD1JUMP1_combined"] == {"combined": "FDJUMP1"}
        assert mapping["FD2JUMP1_combined"] == {"combined": "FDJUMP2"}
        assert mapping["F0"] == {"combined": "F0"}
        assert all(
            isinstance(value, str)
            for owners in mapping.values()
            for value in owners.values()
        )


class TestNltimingAcceptsTheRenderedMapping:
    """The consumer side of the par-spelling contract.

    ``nltiming.selection`` compares a host fitpar against the mapping's native
    value. The combined host keys FDJUMP columns by the PINT attribute while
    the mapping renders the par spelling, so only the shared FDJUMP fold sees
    them as one coefficient -- PINT cannot alias ``FDpJUMPq`` <-> ``FDJUMPp_q``.
    """

    def test_fitpar_suffix_accepts_combined_fdjump_columns(self):
        from nltiming.selection import validated_parameter_mapping_view

        result = _combined_fitparameters()
        pulsar = _bare_pulsar(result.fitparameters)

        # Total validation over every fitpar: this is the gate for_pulsar hits.
        view = validated_parameter_mapping_view(pulsar, tuple(pulsar.fitpars))
        assert view.mapping["FD1JUMP1_combined"] == {"combined": "FDJUMP1"}

    def test_selectors_reach_the_combined_fdjump_columns(self):
        from nltiming.selection import match_fitpars

        result = _combined_fitparameters()
        pulsar = _bare_pulsar(result.fitparameters)
        fitpars = tuple(pulsar.fitpars)

        # Instance stays specific across the spelling fold.
        assert match_fitpars(pulsar, "FDJUMP1", fitpars) == ("FD1JUMP1_combined",)
        assert match_fitpars(pulsar, "FD2JUMP1", fitpars) == ("FD2JUMP1_combined",)


# ===== D5: per-family engine override spelling =====


class _Captured(Exception):
    """Spy sentinel: the engine branch ran and handed over its mapping."""

    def __init__(self, mapping):
        super().__init__("captured")
        self.mapping = dict(mapping or {})


class _SessionPulsar:
    def __init__(self, n_toa: int, *, timing_package: str):
        self._toas = np.arange(n_toa, dtype=float)
        self.timing_package = timing_package


def _engine_mapping_pulsar(tmp_path: Path, timing_package: str) -> MetaPulsar:
    """One-PTA pulsar carrying an aliased column (A1DOT/XDOT) plus F0."""
    par = tmp_path / "session.par"
    tim = tmp_path / "session.tim"
    par.write_text("F0 1\n", encoding="utf-8")
    tim.write_text("FORMAT 1\n", encoding="utf-8")

    pulsar = MetaPulsar.__new__(MetaPulsar)
    pulsar.name = "J0000+0000"
    pulsar._pta_data = {"epta": _SessionPulsar(2, timing_package=timing_package)}
    pulsar._pta_files = {
        "epta": PtaFiles(par_path=par, tim_path=tim, timing_package=timing_package)
    }
    pulsar._pulsars = {"epta": object()}
    pulsar._fitparameters = {
        "F0": {"epta": NativeParam("F0", "F0")},
        "A1DOT_epta": {"epta": NativeParam(pint_name="A1DOT", par_key="XDOT")},
    }
    pulsar.fitpars = ["F0", "A1DOT_epta"]
    pulsar._parfile_dicts = {"epta": {"F0": "1.0", "A1DOT": "8.1e-15"}}
    pulsar._designmatrix = np.ones((2, 2), dtype=float)
    pulsar._toas = np.arange(2, dtype=float)
    pulsar._residuals = np.zeros(2, dtype=float)
    pulsar._toaerrs = np.ones(2, dtype=float)
    pulsar._ssbfreqs = np.full(2, 1400.0, dtype=float)
    pulsar._backend_flags = np.array(["a", "a"])
    pulsar._flags = {"f": pulsar._backend_flags}
    pulsar._isort = slice(None, None, None)
    pulsar._clock_dir = None
    pulsar._timing_engine_cache = {}
    pulsar._pint_model_cache = None
    pulsar._shared_theta_exact_cache = {}
    pulsar._retained_pint_model_cache = {}
    return pulsar


def _capture(monkeypatch, target, attr):
    def _spy(*args, param_mapping=None, **kwargs):
        raise _Captured(param_mapping)

    monkeypatch.setattr(target, attr, _spy)


class TestEngineOverrideSpelling:
    """Each engine family gets the spelling its own name list uses."""

    def test_vela_receives_the_pint_spelling(self, tmp_path, monkeypatch):
        from metapulsar.engines.vela import VelaEngine

        pulsar = _engine_mapping_pulsar(tmp_path, "pint")
        # Spy never constructs SPNTA; do not gate this D5 pin on pyvela.
        monkeypatch.setattr(MetaPulsar, "_can_import_vela", lambda self: True)
        _capture(monkeypatch, VelaEngine, "from_files")

        with pytest.raises(_Captured) as excinfo:
            pulsar.timing_engine({"pint": "vela", "tempo2": "libstempo"})

        # SPNTA's param_names are PINT-format; forwarding "XDOT" would have
        # silently dropped this column to exact-linear.
        assert excinfo.value.mapping == {"F0": "F0", "A1DOT_epta": "A1DOT"}

    def test_tempo2_receives_the_par_spelling(self, tmp_path, monkeypatch):
        from metapulsar.engines import LibstempoEngine

        pulsar = _engine_mapping_pulsar(tmp_path, "tempo2")
        _capture(monkeypatch, LibstempoEngine, "from_contribution")

        with pytest.raises(_Captured) as excinfo:
            pulsar.timing_engine({"tempo2": "libstempo", "pint": "pint"})

        assert excinfo.value.mapping == {"F0": "F0", "A1DOT_epta": "XDOT"}

    def test_jug_resolves_against_the_live_session_names(self, tmp_path, monkeypatch):
        from metapulsar.engines import JugEngine

        pulsar = _engine_mapping_pulsar(tmp_path, "pint")

        class _Session:
            # JUG spells A1DOT natively as XDOT; F0 matches by exact hit.
            params = {"F0": None, "XDOT": None}

        monkeypatch.setattr(
            MetaPulsar, "_build_jug_session", lambda self, *a, **k: _Session()
        )
        monkeypatch.setattr(MetaPulsar, "_can_import_jug", lambda self: True)
        _capture(monkeypatch, JugEngine, "from_contribution")

        with pytest.raises(_Captured) as excinfo:
            pulsar.timing_engine({"pint": "jug", "tempo2": "libstempo"})

        assert excinfo.value.mapping == {"F0": "F0", "A1DOT_epta": "XDOT"}

    def test_jug_omits_columns_the_session_does_not_know(self, tmp_path, monkeypatch):
        """Omitted entries fail validate_fit_param and become exact-linear."""
        from metapulsar.engines import JugEngine

        pulsar = _engine_mapping_pulsar(tmp_path, "pint")

        class _Session:
            params = {"F0": None}

        monkeypatch.setattr(
            MetaPulsar, "_build_jug_session", lambda self, *a, **k: _Session()
        )
        monkeypatch.setattr(MetaPulsar, "_can_import_jug", lambda self: True)
        _capture(monkeypatch, JugEngine, "from_contribution")

        with pytest.raises(_Captured) as excinfo:
            pulsar.timing_engine({"pint": "jug", "tempo2": "libstempo"})

        assert excinfo.value.mapping == {"F0": "F0"}
