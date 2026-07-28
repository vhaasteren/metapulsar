"""Mock libstempo.tempopulsar for testing without a tempo2 install."""

import numpy as np
from loguru import logger


class MockParameter:
    """Mutable parameter object with .val and .err attributes."""

    def __init__(self, value=0.0, error=0.0):
        self.val = value
        self.err = error


class MockLibstempo:
    """Mock libstempo.tempopulsar with a libstempo-compatible duck surface."""

    def __init__(
        self,
        toas_mjd,
        residuals_s,
        toaerrs_us,
        freqs_hz,
        flags,
        telescope,
        name="J0000+0000",
        raj_rad=0.7853981633974483,
        decj_rad=0.16962634524619668,
        f0=186.49408156698,
        f1=-6.205e-16,
        f2=0.0,
        dm=13.306,
        pepoch=55000.0,
        posepoch=55000.0,
        pmra=0.0,
        pmdec=0.0,
        px=0.0,
        include_astrometry=True,
        include_spin=True,
    ):
        n = len(toas_mjd)
        self.name = name
        self._toas_mjd = np.asarray(toas_mjd, dtype=np.float64)
        self._residuals_s = np.asarray(residuals_s, dtype=np.float64)
        self._toaerrs_us = np.asarray(toaerrs_us, dtype=np.float64)
        self._freqs_hz = np.asarray(freqs_hz, dtype=np.float64)

        if isinstance(flags, dict):
            self._flag_dict = {k: np.asarray(v) for k, v in flags.items()}
        else:
            self._flag_dict = {}

        if isinstance(telescope, str):
            self._telescope = np.array([telescope] * n)
        else:
            self._telescope = np.asarray(telescope)

        self._raj_rad = raj_rad
        self._decj_rad = decj_rad
        self._f0 = f0
        self._f1 = f1
        self._f2 = f2
        self._dm = dm
        self._pepoch = pepoch
        self._posepoch = posepoch
        self._pmra = pmra
        self._pmdec = pmdec
        self._px = px
        self._include_astrometry = include_astrometry
        self._include_spin = include_spin

        self._params = {}
        self._setup_parameters()
        self._fitpars, self._setpars = self._build_par_lists()
        self._designmatrix = self._build_design_matrix()
        self._psrPos = self._build_position_array()

    def toas(self):
        return self._toas_mjd.copy()

    @property
    def stoas(self):
        return self._toas_mjd.copy()

    def residuals(self):
        return self._residuals_s.copy()

    @property
    def toaerrs(self):
        return self._toaerrs_us.copy()

    def designmatrix(self):
        return self._designmatrix.copy()

    def ssbfreqs(self):
        return self._freqs_hz.copy()

    def telescope(self):
        return self._telescope.astype("S")

    def pars(self, which="fit"):
        if which == "fit":
            return tuple(self._fitpars)
        return tuple(self._setpars)

    def __getitem__(self, param_name):
        if param_name not in self._params:
            self._params[param_name] = MockParameter(0.0, 0.0)
        return self._params[param_name]

    def __setitem__(self, param_name, value):
        if isinstance(value, MockParameter):
            self._params[param_name] = value
        else:
            self._params[param_name] = MockParameter(value, 0.0)

    def flags(self):
        return list(self._flag_dict.keys())

    def flagvals(self, key):
        return self._flag_dict[key]

    @property
    def psrPos(self):
        return self._psrPos.copy()

    def formbats(self):
        pass

    @property
    def mercury_ssb(self):
        return np.zeros((len(self._toas_mjd), 6))

    @property
    def venus_ssb(self):
        return np.zeros((len(self._toas_mjd), 6))

    @property
    def earth_ssb(self):
        return np.zeros((len(self._toas_mjd), 6))

    @property
    def mars_ssb(self):
        return np.zeros((len(self._toas_mjd), 6))

    @property
    def jupiter_ssb(self):
        return np.zeros((len(self._toas_mjd), 6))

    @property
    def saturn_ssb(self):
        return np.zeros((len(self._toas_mjd), 6))

    @property
    def uranus_ssb(self):
        return np.zeros((len(self._toas_mjd), 6))

    @property
    def neptune_ssb(self):
        return np.zeros((len(self._toas_mjd), 6))

    @property
    def pluto_ssb(self):
        return np.zeros((len(self._toas_mjd), 6))

    @property
    def sun_ssb(self):
        return np.zeros((len(self._toas_mjd), 6))

    def savepar(self, parfile_path):
        content = self._generate_parfile_content()
        with open(parfile_path, "w", encoding="utf-8") as f:
            f.write(content)

    @property
    def parfile(self):
        return self._generate_parfile_content()

    def _setup_parameters(self):
        self._params["RAJ"] = MockParameter(self._raj_rad)
        self._params["DECJ"] = MockParameter(self._decj_rad)
        self._params["F0"] = MockParameter(self._f0)
        self._params["F1"] = MockParameter(self._f1)
        self._params["F2"] = MockParameter(self._f2)
        self._params["DM"] = MockParameter(self._dm)
        self._params["PMRA"] = MockParameter(self._pmra)
        self._params["PMDEC"] = MockParameter(self._pmdec)
        self._params["PX"] = MockParameter(self._px)
        for ii in range(1, 10):
            self._params[f"DMASSPLANET{ii}"] = MockParameter(0.0)

    def _build_par_lists(self):
        fitpars = []
        setpars = []
        if self._include_spin:
            fitpars.extend(["F0", "F1", "F2"])
            setpars.extend(["F0", "F1", "F2"])
        if self._include_astrometry:
            fitpars.extend(["RAJ", "DECJ", "PMRA", "PMDEC", "PX"])
            setpars.extend(["RAJ", "DECJ", "PMRA", "PMDEC", "PX"])
        fitpars.extend(["DM"])
        setpars.extend(["DM"])
        return fitpars, setpars

    def _build_design_matrix(self):
        n = len(self._toas_mjd)
        dm = np.zeros((n, 1 + len(self._fitpars)))
        dm[:, 0] = 1.0
        t_days = self._toas_mjd - self._pepoch
        for i, par in enumerate(self._fitpars):
            col = i + 1
            if par == "F0":
                dm[:, col] = 1.0
            elif par == "F1":
                dm[:, col] = t_days
            elif par == "F2":
                dm[:, col] = t_days**2
            elif par in ("RAJ", "DECJ"):
                dm[:, col] = np.sin(2 * np.pi * (self._freqs_hz / 1e9))
            elif par in ("PMRA", "PMDEC"):
                dm[:, col] = t_days / 365.25
            elif par == "PX":
                dm[:, col] = np.sin(2 * np.pi * t_days / 365.25)
            elif par == "DM":
                dm[:, col] = 1.0 / np.maximum(self._freqs_hz / 1e9, 1e-12) ** 2
        return dm

    def _build_position_array(self):
        pos = np.array(
            [
                np.cos(self._raj_rad) * np.cos(self._decj_rad),
                np.sin(self._raj_rad) * np.cos(self._decj_rad),
                np.sin(self._decj_rad),
            ]
        )
        return np.tile(pos, (len(self._toas_mjd), 1))

    def _raj_to_hms(self, rad):
        hours = rad * 12.0 / np.pi
        h = int(hours)
        m = int((hours - h) * 60)
        s = (hours - h - m / 60.0) * 3600
        return f"{h:02d}:{m:02d}:{s:010.7f}"

    def _decj_to_dms(self, rad):
        deg = np.degrees(rad)
        sign = "+" if deg >= 0 else "-"
        deg = abs(deg)
        d = int(deg)
        m = int((deg - d) * 60)
        s = (deg - d - m / 60.0) * 3600
        return f"{sign}{d:02d}:{m:02d}:{s:09.6f}"

    def _generate_parfile_content(self):
        lines = [f"PSR              {self.name}"]
        if self._include_astrometry:
            lines.append(f"RAJ              {self._raj_to_hms(self._raj_rad)} 1")
            lines.append(f"DECJ             {self._decj_to_dms(self._decj_rad)} 1")
            lines.append(f"PMRA             {self._pmra} 1")
            lines.append(f"PMDEC            {self._pmdec} 1")
            lines.append(f"PX               {self._px} 1")
            lines.append(f"POSEPOCH         {self._posepoch}")
        else:
            lines.append(f"RAJ              {self._raj_to_hms(self._raj_rad)} 0")
            lines.append(f"DECJ             {self._decj_to_dms(self._decj_rad)} 0")
        if self._include_spin:
            lines.append(f"F0               {self._f0} 1")
            lines.append(f"F1               {self._f1} 1")
            lines.append(f"F2               {self._f2} 1")
        else:
            lines.append(f"F0               {self._f0} 0")
        lines.append(f"PEPOCH           {self._pepoch}")
        lines.append(f"DM               {self._dm} 1")
        lines.append(f"DMEPOCH          {self._pepoch}")
        lines.append("EPHEM            DE440")
        lines.append("CLK              TT(BIPM2021)")
        lines.append("UNITS            TDB")
        lines.append(f"NTOA             {len(self._toas_mjd)}")
        return "\n".join(lines) + "\n"


def create_mock_timing_data(
    n_toas,
    toa_range=(50000.0, 60000.0),
    residual_std=1e-6,
    error_mean=1e-7,
    freq_range=(100.0, 2000.0),
    seed=None,
):
    rng = np.random.default_rng(seed)
    toas_mjd = np.sort(rng.uniform(toa_range[0], toa_range[1], n_toas))
    residuals_s = rng.normal(0, residual_std, n_toas)
    toaerrs_us = np.abs(rng.normal(error_mean * 1e6, error_mean * 0.1e6, n_toas))
    freqs_hz = rng.uniform(freq_range[0] * 1e6, freq_range[1] * 1e6, n_toas)
    return toas_mjd, residuals_s, toaerrs_us, freqs_hz


def create_mock_flags(n_toas, telescope="mock", backend="mock", **kwargs):
    flags = {
        "telescope": np.array([telescope] * n_toas),
        "backend": np.array([backend] * n_toas),
    }
    for key, value in kwargs.items():
        if isinstance(value, (list, np.ndarray)):
            if len(value) == n_toas:
                flags[key] = np.asarray(value)
            else:
                logger.warning(
                    f"Flag '{key}' length {len(value)} != n_toas {n_toas}, skipping"
                )
        else:
            flags[key] = np.array([value] * n_toas)
    return flags


def create_mock_libstempo(
    n_toas=50,
    name="J1857+0943",
    telescope="mock",
    backend="mock",
    include_astrometry=True,
    include_spin=True,
    toa_range=(50000.0, 60000.0),
    freq_range=(100.0, 2000.0),
    residual_std=1e-6,
    error_mean=1e-7,
    seed=None,
    **flag_kwargs,
):
    toas_mjd, residuals_s, toaerrs_us, freqs_hz = create_mock_timing_data(
        n_toas,
        toa_range=toa_range,
        freq_range=freq_range,
        residual_std=residual_std,
        error_mean=error_mean,
        seed=seed,
    )
    flags = create_mock_flags(
        n_toas, telescope=telescope, backend=backend, **flag_kwargs
    )
    return MockLibstempo(
        toas_mjd=toas_mjd,
        residuals_s=residuals_s,
        toaerrs_us=toaerrs_us,
        freqs_hz=freqs_hz,
        flags=flags,
        telescope=telescope,
        name=name,
        include_astrometry=include_astrometry,
        include_spin=include_spin,
    )


def validate_mock_data(toas, residuals, errors, freqs, flags=None):
    n_toas = len(toas)
    if len(residuals) != n_toas:
        logger.error(f"Residuals length {len(residuals)} != TOAs length {n_toas}")
        return False
    if len(errors) != n_toas:
        logger.error(f"Errors length {len(errors)} != TOAs length {n_toas}")
        return False
    if len(freqs) != n_toas:
        logger.error(f"Frequencies length {len(freqs)} != TOAs length {n_toas}")
        return False
    if flags:
        for key, value in flags.items():
            if len(value) != n_toas:
                logger.error(
                    f"Flag '{key}' length {len(value)} != TOAs length {n_toas}"
                )
                return False
    if np.any(np.isnan(toas)) or np.any(np.isinf(toas)):
        logger.error("Invalid TOAs (NaN or Inf)")
        return False
    if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
        logger.error("Invalid residuals (NaN or Inf)")
        return False
    if np.any(errors <= 0):
        logger.error("Non-positive errors found")
        return False
    if np.any(freqs <= 0):
        logger.error("Non-positive frequencies found")
        return False
    return True
