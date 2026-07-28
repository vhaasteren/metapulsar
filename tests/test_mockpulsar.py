"""Unit tests for MockLibstempo and convenience factories."""

import numpy as np
import pytest

from metapulsar.mockpulsar import (
    MockParameter,
    create_mock_flags,
    create_mock_libstempo,
    create_mock_timing_data,
    validate_mock_data,
)


class TestMockParameter:
    def test_mutable_val(self):
        param = MockParameter(1.5, 0.1)
        assert param.val == 1.5
        assert param.err == 0.1
        param.val = 2.0
        assert param.val == 2.0

    def test_default_error(self):
        param = MockParameter(2.0)
        assert param.val == 2.0
        assert param.err == 0.0


class TestMockLibstempoInterface:
    @pytest.fixture
    def mock_lt(self):
        return create_mock_libstempo(
            n_toas=30,
            name="J1857+0943",
            telescope="test_tel",
            include_astrometry=True,
            include_spin=True,
            seed=42,
        )

    def test_toas_returns_mjd(self, mock_lt):
        toas = mock_lt.toas()
        assert toas.dtype == np.float64
        assert np.all((toas >= 50000) & (toas <= 60000))

    def test_stoas_property(self, mock_lt):
        assert np.array_equal(mock_lt.stoas, mock_lt.toas())

    def test_residuals_seconds(self, mock_lt):
        residuals = mock_lt.residuals()
        assert residuals.dtype == np.float64
        assert len(residuals) == 30

    def test_toaerrs_microseconds(self, mock_lt):
        errs = mock_lt.toaerrs
        assert errs.dtype == np.float64
        assert np.all(errs > 0)

    def test_designmatrix_shape(self, mock_lt):
        designmatrix = mock_lt.designmatrix()
        assert designmatrix.shape == (30, 1 + len(mock_lt.pars()))
        assert np.allclose(designmatrix[:, 0], 1.0)

    def test_ssbfreqs_hz(self, mock_lt):
        freqs_hz = mock_lt.ssbfreqs()
        assert np.all(freqs_hz > 1e8)

    def test_telescope_bytes(self, mock_lt):
        telescope = mock_lt.telescope()
        assert telescope.dtype.kind == "S"

    def test_pars_fit_excludes_offset(self, mock_lt):
        fit_pars = mock_lt.pars()
        assert "Offset" not in fit_pars
        assert "F0" in fit_pars

    def test_pars_set(self, mock_lt):
        set_pars = mock_lt.pars(which="set")
        assert "DM" in set_pars

    def test_getitem_returns_mock_parameter(self, mock_lt):
        raj_param = mock_lt["RAJ"]
        assert isinstance(raj_param, MockParameter)
        assert raj_param.val != 0.0

    def test_getitem_mutable(self, mock_lt):
        mock_lt["DMASSPLANET1"].val = 0.0
        assert mock_lt["DMASSPLANET1"].val == 0.0

    def test_getitem_unknown_returns_zero(self, mock_lt):
        unknown = mock_lt["NONEXISTENT"]
        assert unknown.val == 0.0

    def test_flags_and_flagvals(self, mock_lt):
        assert "telescope" in mock_lt.flags()
        vals = mock_lt.flagvals("telescope")
        assert len(vals) == 30
        assert np.all(vals == "test_tel")

    def test_psrPos_shape(self, mock_lt):
        pos = mock_lt.psrPos
        assert pos.shape == (30, 3)
        norms = np.linalg.norm(pos, axis=1)
        assert np.allclose(norms, 1.0)

    def test_formbats_noop(self, mock_lt):
        mock_lt.formbats()

    def test_planetary_ssb_shapes(self, mock_lt):
        planets = [
            "mercury_ssb",
            "venus_ssb",
            "earth_ssb",
            "mars_ssb",
            "jupiter_ssb",
            "saturn_ssb",
            "uranus_ssb",
            "neptune_ssb",
            "pluto_ssb",
            "sun_ssb",
        ]
        for planet in planets:
            data = getattr(mock_lt, planet)
            assert data.shape == (30, 6)

    def test_savepar_produces_pint_parseable_content(self, mock_lt, tmp_path):
        parfile = tmp_path / "test.par"
        mock_lt.savepar(str(parfile))
        content = parfile.read_text()
        assert "PSR" in content
        assert "RAJ" in content
        assert "F0" in content
        assert "UNITS" in content
        assert "TDB" in content

        from pint.models import get_model

        model = get_model(str(parfile))
        assert model.PSR.value == "J1857+0943"

    def test_name_attribute(self, mock_lt):
        assert mock_lt.name == "J1857+0943"


class TestDuckTypingCompatibility:
    def test_has_tempo2_interface(self):
        mock_lt = create_mock_libstempo(n_toas=10, seed=0)
        assert callable(getattr(mock_lt, "toas", None))
        assert hasattr(mock_lt, "stoas")
        assert callable(getattr(mock_lt, "residuals", None))
        assert hasattr(mock_lt, "toaerrs")
        assert callable(getattr(mock_lt, "designmatrix", None))
        assert callable(getattr(mock_lt, "ssbfreqs", None))
        assert callable(getattr(mock_lt, "telescope", None))
        assert callable(getattr(mock_lt, "flags", None))
        assert callable(getattr(mock_lt, "flagvals", None))
        assert callable(getattr(mock_lt, "pars", None))
        assert hasattr(mock_lt, "psrPos")
        assert hasattr(mock_lt, "__getitem__")
        assert hasattr(mock_lt, "name")


class TestConvenienceFactories:
    def test_create_mock_timing_data(self):
        toas, residuals, errs, freqs = create_mock_timing_data(100, seed=0)
        assert len(toas) == 100
        assert np.all(np.diff(toas) >= 0)
        assert np.all(errs > 0)
        assert np.all(freqs > 0)

    def test_create_mock_flags(self):
        flags = create_mock_flags(50, telescope="GBT", backend="GUPPI", band="L")
        assert len(flags["telescope"]) == 50
        assert np.all(flags["telescope"] == "GBT")
        assert np.all(flags["backend"] == "GUPPI")
        assert np.all(flags["band"] == "L")

    def test_create_mock_libstempo(self):
        mock_lt = create_mock_libstempo(n_toas=25, name="J1909-3744", seed=1)
        assert mock_lt.name == "J1909-3744"
        assert len(mock_lt.toas()) == 25

    def test_validate_mock_data(self):
        toas, residuals, errs, freqs = create_mock_timing_data(30, seed=0)
        assert validate_mock_data(toas, residuals, errs, freqs)
        assert not validate_mock_data(toas, residuals[:-1], errs, freqs)
        errs_bad = errs.copy()
        errs_bad[0] = -1.0
        assert not validate_mock_data(toas, residuals, errs_bad, freqs)
