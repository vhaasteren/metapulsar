"""MetaPulsar-owned PTA timing-record materialization.

Replaces ``enterprise.pulsar.PintPulsar`` / ``Tempo2Pulsar`` construction with
a validated internal record that preserves Enterprise 3.x numerical
conventions (units, ephemerides, ecliptic conversion via PyEphem).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Literal

import astropy.constants
import astropy.units as u
import numpy as np
from astropy.time import Time
from ephem import Ecliptic, Equatorial
from loguru import logger

import pint.residuals


@dataclass
class _PtaTimingData:
    name: str
    timing_package: Literal["pint", "tempo2"]
    _toas: np.ndarray
    _stoas: np.ndarray
    _residuals: np.ndarray
    _toaerrs: np.ndarray
    _ssbfreqs: np.ndarray
    _telescope: np.ndarray
    _designmatrix: np.ndarray
    _flags: dict[str, np.ndarray]
    fitpars: list[str]
    setpars: list[str]
    _raj: float
    _decj: float
    _pos: np.ndarray
    _pos_t: np.ndarray
    _planetssb: np.ndarray
    _sunssb: np.ndarray
    _pdist: tuple[float, float]
    _pint_model: object | None = None
    _pint_toas: object | None = None
    _lt_pulsar: object | None = None
    fitpars_canonical: list[str] = field(default_factory=list, init=False)
    setpars_canonical: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        n = len(self._toas)
        one_dimensional = (
            "_toas",
            "_stoas",
            "_residuals",
            "_toaerrs",
            "_ssbfreqs",
            "_telescope",
        )
        for field_name in one_dimensional:
            value = np.asarray(getattr(self, field_name))
            if value.ndim != 1 or len(value) != n:
                raise ValueError(
                    f"{field_name} must be a 1D array with {n} rows; "
                    f"got shape {value.shape}"
                )
        if np.asarray(self._designmatrix).shape != (n, len(self.fitpars)):
            raise ValueError(
                "_designmatrix shape must be "
                f"({n}, {len(self.fitpars)}); got "
                f"{np.asarray(self._designmatrix).shape}"
            )
        if np.asarray(self._pos).shape != (3,):
            raise ValueError("_pos must have shape (3,)")
        if np.asarray(self._pos_t).shape != (n, 3):
            raise ValueError(f"_pos_t must have shape ({n}, 3)")
        if np.asarray(self._planetssb).shape != (n, 9, 6):
            raise ValueError(f"_planetssb must have shape ({n}, 9, 6)")
        if np.asarray(self._sunssb).shape != (n, 6):
            raise ValueError(f"_sunssb must have shape ({n}, 6)")
        for flag, values in self._flags.items():
            values = np.asarray(values)
            if values.ndim != 1 or len(values) != n:
                raise ValueError(f"flag {flag!r} must be a 1D array with {n} rows")


@lru_cache(maxsize=1)
def _load_pulsar_distances() -> dict[str, list[float]]:
    path = resources.files("metapulsar") / "resources" / "pulsar_distances.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pulsar_distance(name: str) -> tuple[float, float]:
    """Look up an approximate pulsar distance (Enterprise catalog policy)."""
    pdict = _load_pulsar_distances()
    if not name or name[0] not in ("J", "B"):
        if "J" + name in pdict:
            return tuple(pdict["J" + name])  # type: ignore[return-value]
        return tuple(pdict.get("B" + name, (1.0, 0.2)))  # type: ignore[return-value]
    return tuple(pdict.get(name, (1.0, 0.2)))  # type: ignore[return-value]


def ecliptic_to_icrs_vectors(values: np.ndarray) -> np.ndarray:
    """Rotate vectors from ecliptic to ICRS using Enterprise's M_ecl."""
    obliquity = np.deg2rad(23.43704)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(obliquity), -np.sin(obliquity)],
            [0.0, np.sin(obliquity), np.cos(obliquity)],
        ]
    )
    return np.einsum("jk,...k->...j", rotation, values)


def _get_radec_from_ecliptic(
    elong: float, elat: float, *, name: str
) -> tuple[float, float]:
    try:
        eq = Equatorial(Ecliptic(elong, elat), epoch="2000")
        return float(eq.ra), float(eq.dec)
    except TypeError:
        logger.warning(
            "WARNING: Cannot find sky location coordinates for PSR {}. "
            "Setting values to 0.0",
            name,
        )
        return 0.0, 0.0


def _unit_vector_from_radec(raj: float, decj: float) -> np.ndarray:
    return np.array(
        [
            np.cos(raj) * np.cos(decj),
            np.sin(raj) * np.cos(decj),
            np.sin(decj),
        ]
    )


def _pint_flags(toas) -> dict[str, np.ndarray]:
    flags: dict[str, list] = {}
    for ii, obsflags in enumerate(toas.get_flags()):
        for flag in obsflags:
            if flag not in flags:
                flags[flag] = [""] * toas.ntoas
            flags[flag][ii] = obsflags[flag]
    out: dict[str, np.ndarray] = {}
    for key, val in flags.items():
        if val and isinstance(val[0], u.Quantity):
            out[key] = np.array([v.value for v in val])
        else:
            out[key] = np.array(val)
    return out


def _pint_ssb_light_seconds(toas, column: str) -> np.ndarray:
    if column not in toas.table.colnames:
        raise ValueError(
            f"{column} is not in toas.table.colnames. Load TOAs with "
            "planets=True so required ephemeris columns are present."
        )
    vector = toas.table[column] + toas.table["ssb_obs_pos"]
    return (vector / astropy.constants.c).to_value(u.s)


def _pint_planetssb(toas, n: int) -> np.ndarray:
    planetssb = np.full((n, 9, 6), np.nan)
    planetssb[:, 2, :3] = _pint_ssb_light_seconds(toas, "obs_earth_pos")
    planetssb[:, 4, :3] = _pint_ssb_light_seconds(toas, "obs_jupiter_pos")
    planetssb[:, 5, :3] = _pint_ssb_light_seconds(toas, "obs_saturn_pos")
    planetssb[:, 6, :3] = _pint_ssb_light_seconds(toas, "obs_uranus_pos")
    planetssb[:, 7, :3] = _pint_ssb_light_seconds(toas, "obs_neptune_pos")
    return planetssb


def _pint_radec(model) -> tuple[float, float]:
    if hasattr(model, "RAJ") and hasattr(model, "DECJ"):
        raj = model.RAJ.quantity.to(u.rad).value
        decj = model.DECJ.quantity.to(u.rad).value
        return float(raj), float(decj)
    elong = model.ELONG.quantity.to_value(u.rad)
    elat = model.ELAT.quantity.to_value(u.rad)
    return _get_radec_from_ecliptic(elong, elat, name=str(model.PSR.value))


def materialize_pint(model, toas) -> _PtaTimingData:
    """Materialize a PINT model/TOAs pair into a PTA timing record."""
    toas_s = (
        np.asarray(model.get_barycentric_toas(toas).value, dtype=np.float64) * 86400.0
    )
    stoas_s = np.asarray(toas.get_mjds().value, dtype=np.float64) * 86400.0
    residuals_s = np.asarray(
        pint.residuals.Residuals(toas, model).time_resids.to_value(u.s),
        dtype=np.float64,
    )
    toaerrs_s = np.asarray(toas.get_errors().to_value(u.s), dtype=np.float64)
    design, fitpars, _design_units = model.designmatrix(toas)
    ssbfreqs_mhz = np.asarray(model.barycentric_radio_freq(toas), dtype=np.float64)
    telescope = np.asarray(toas.get_obss())
    setpars = [name for name in model.params if name not in fitpars]
    flags = _pint_flags(toas)

    raj, decj = _pint_radec(model)
    pos = _unit_vector_from_radec(raj, decj)

    which_astrometry = (
        "AstrometryEquatorial"
        if "AstrometryEquatorial" in model.components
        else "AstrometryEcliptic"
    )
    pos_t = np.asarray(
        model.components[which_astrometry]
        .ssb_to_psb_xyz_ICRS(Time(model.get_barycentric_toas(toas), format="mjd"))
        .value,
        dtype=np.float64,
    )
    n = len(toas_s)
    # PINT may return a single ICRS unit vector when all rows share the same
    # direction; normalize to the required (n, 3) record contract.
    if pos_t.shape == (3,):
        pos_t = np.broadcast_to(pos_t, (n, 3)).copy()
    elif pos_t.shape == (3, n):
        pos_t = pos_t.T

    planetssb = _pint_planetssb(toas, n)
    sunssb = np.zeros((n, 6))
    sunssb[:, :3] = _pint_ssb_light_seconds(toas, "obs_sun_pos")

    return _PtaTimingData(
        name=str(model.PSR.value),
        timing_package="pint",
        _toas=toas_s,
        _stoas=stoas_s,
        _residuals=residuals_s,
        _toaerrs=toaerrs_s,
        _ssbfreqs=ssbfreqs_mhz,
        _telescope=telescope,
        _designmatrix=np.asarray(design, dtype=np.float64),
        _flags=flags,
        fitpars=list(fitpars),
        setpars=setpars,
        _raj=float(raj),
        _decj=float(decj),
        _pos=pos,
        _pos_t=pos_t,
        _planetssb=planetssb,
        _sunssb=sunssb,
        _pdist=pulsar_distance(str(model.PSR.value)),
        _pint_model=model,
        _pint_toas=toas,
    )


def _tempo2_is_ecliptic(lt_pulsar) -> bool:
    allpars = {
        *map(str, lt_pulsar.pars(which="fit")),
        *map(str, lt_pulsar.pars(which="set")),
    }
    return "ELONG" in allpars and "ELAT" in allpars


def _tempo2_radec(lt_pulsar) -> tuple[float, float]:
    allpars = {
        *map(str, lt_pulsar.pars(which="fit")),
        *map(str, lt_pulsar.pars(which="set")),
    }
    if "RAJ" in allpars:
        return float(lt_pulsar["RAJ"].val), float(lt_pulsar["DECJ"].val)
    elong = float(lt_pulsar["ELONG"].val)
    elat = float(lt_pulsar["ELAT"].val)
    return _get_radec_from_ecliptic(elong, elat, name=str(lt_pulsar.name))


def _tempo2_planet_arrays(lt_pulsar, n: int) -> tuple[np.ndarray, np.ndarray]:
    for ii in range(1, 10):
        lt_pulsar[f"DMASSPLANET{ii}"].val = 0.0
    lt_pulsar.formbats()
    planetssb = np.zeros((n, 9, 6))
    planetssb[:, 0, :] = lt_pulsar.mercury_ssb
    planetssb[:, 1, :] = lt_pulsar.venus_ssb
    planetssb[:, 2, :] = lt_pulsar.earth_ssb
    planetssb[:, 3, :] = lt_pulsar.mars_ssb
    planetssb[:, 4, :] = lt_pulsar.jupiter_ssb
    planetssb[:, 5, :] = lt_pulsar.saturn_ssb
    planetssb[:, 6, :] = lt_pulsar.uranus_ssb
    planetssb[:, 7, :] = lt_pulsar.neptune_ssb
    planetssb[:, 8, :] = lt_pulsar.pluto_ssb
    sunssb = np.zeros((n, 6))
    sunssb[:, :] = lt_pulsar.sun_ssb
    return planetssb, sunssb


def materialize_tempo2(lt_pulsar) -> _PtaTimingData:
    """Materialize a libstempo-like pulsar into a PTA timing record."""
    toas_s = np.asarray(lt_pulsar.toas(), dtype=np.float64) * 86400.0
    stoas_s = np.asarray(lt_pulsar.stoas, dtype=np.float64) * 86400.0
    residuals_s = np.asarray(lt_pulsar.residuals(), dtype=np.float64)
    toaerrs_s = np.asarray(lt_pulsar.toaerrs, dtype=np.float64) * 1.0e-6
    design = np.asarray(lt_pulsar.designmatrix(), dtype=np.float64)
    ssbfreqs_mhz = np.asarray(lt_pulsar.ssbfreqs(), dtype=np.float64) / 1.0e6
    telescope = np.char.decode(np.asarray(lt_pulsar.telescope()), encoding="ascii")
    fitpars = ["Offset", *map(str, lt_pulsar.pars())]
    set_names = map(str, lt_pulsar.pars(which="set"))
    setpars = [name for name in set_names if name not in fitpars]
    flags = {str(key): np.asarray(lt_pulsar.flagvals(key)) for key in lt_pulsar.flags()}

    n = len(toas_s)
    planetssb, sunssb = _tempo2_planet_arrays(lt_pulsar, n)
    pos_t = np.asarray(lt_pulsar.psrPos.copy(), dtype=np.float64)

    if _tempo2_is_ecliptic(lt_pulsar):
        pos_t = ecliptic_to_icrs_vectors(pos_t)
        for ii in range(9):
            planetssb[:, ii, :3] = ecliptic_to_icrs_vectors(planetssb[:, ii, :3])
            planetssb[:, ii, 3:] = ecliptic_to_icrs_vectors(planetssb[:, ii, 3:])
        sunssb[:, :3] = ecliptic_to_icrs_vectors(sunssb[:, :3])
        sunssb[:, 3:] = ecliptic_to_icrs_vectors(sunssb[:, 3:])

    raj, decj = _tempo2_radec(lt_pulsar)
    pos = _unit_vector_from_radec(raj, decj)

    return _PtaTimingData(
        name=str(lt_pulsar.name),
        timing_package="tempo2",
        _toas=toas_s,
        _stoas=stoas_s,
        _residuals=residuals_s,
        _toaerrs=toaerrs_s,
        _ssbfreqs=ssbfreqs_mhz,
        _telescope=telescope,
        _designmatrix=design,
        _flags=flags,
        fitpars=fitpars,
        setpars=setpars,
        _raj=float(raj),
        _decj=float(decj),
        _pos=pos,
        _pos_t=np.asarray(pos_t, dtype=np.float64),
        _planetssb=planetssb,
        _sunssb=sunssb,
        _pdist=pulsar_distance(str(lt_pulsar.name)),
        _lt_pulsar=lt_pulsar,
    )
