"""
Position helpers for pulsar coordinate conversion and B/J-name generation.

This module provides robust coordinate conversion between different pulsar object
types (PINT TimingModel, libstempo tempopulsar, Enterprise Pulsar) and generates
canonical B-names (BHHMM±DD) or J-names (JHHMM±DDMM) from actual coordinate data.

Functions:
    bj_name_from_pulsar: Generate B-name or J-name from any supported pulsar object
    _skycoord_from_pint_model: Extract coordinates from PINT TimingModel
    _skycoord_from_libstempo: Extract coordinates from libstempo tempopulsar
    _skycoord_from_enterprise: Extract coordinates from Enterprise Pulsar
    _format_j_name_from_icrs: Format ICRS coordinates into J-name string
    _format_b_name_from_icrs: Format ICRS coordinates into B-name string
"""

from typing import Any, Dict, Optional, Tuple, List
import numpy as np
from astropy.coordinates import (
    SkyCoord,
    ICRS,
    FK4,
    Angle,
    BarycentricMeanEcliptic,
)
from astropy.time import Time
import astropy.units as u
from loguru import logger
from io import StringIO

# Import PINT utilities for robust parfile parsing
from pint.models.model_builder import parse_parfile

# Import alias resolution for parameter access
from .pint_helpers import get_aliases_for_parameter

# Constants for J2000 normalization
J2000_TIME = Time("J2000")
MAS_TO_DEG = 1.0 / 3.6e6  # 1 mas = 1/3.6e6 deg


def _format_j_name_from_icrs(c: SkyCoord) -> str:
    """Format ICRS coordinates into a JHHMM±DDMM label using TRUNCATION."""
    # RA
    ra_h = c.ra.to(u.hourangle).value
    hh = int(np.floor(ra_h)) % 24
    mm = int((ra_h - hh) * 60.0)  # truncate minutes

    # Dec
    dec_deg = c.dec.to(u.deg).value
    sign = "-" if dec_deg < 0 else "+"
    a = abs(dec_deg)
    DD = int(np.floor(a))
    MM = int((a - DD) * 60.0)  # truncate arcminutes

    return f"J{hh:02d}{mm:02d}{sign}{DD:02d}{MM:02d}"


def _format_b_name_from_icrs(c: SkyCoord) -> str:
    """Format ICRS coordinates into a B1234±56 label using TRUNCATION."""
    # RA
    ra_h = c.ra.to(u.hourangle).value
    hh = int(np.floor(ra_h)) % 24
    mm = int((ra_h - hh) * 60.0)  # truncate minutes

    # Dec
    dec_deg = c.dec.to(u.deg).value
    sign = "-" if dec_deg < 0 else "+"
    a = abs(dec_deg)
    DD = int(np.floor(a))

    return f"B{hh:02d}{mm:02d}{sign}{DD:02d}"


def _skycoord_from_pint_model(model: Any) -> SkyCoord:
    """Build a SkyCoord from a PINT TimingModel, normalized to J2000 when possible.

    Preference order:
    1. Equatorial (RAJ/DECJ) with PM + POSEPOCH propagation to J2000.
    2. Ecliptic (prefer LAMBDA/BETA, else ELONG/ELAT) with PMELONG/PMELAT + POSEPOCH propagation, then to ICRS.
    3. FK4 (RA/DEC B1950) as legacy fallback, then to ICRS.
    """
    # POSEPOCH falls back to PEPOCH if not explicitly set, mirroring the
    # tempo2/NANOGrav convention used for parfile-dict extraction.
    posepoch_q = _get_model_quantity(model, "POSEPOCH") or _get_model_quantity(
        model, "PEPOCH"
    )
    posepoch_mjd = float(posepoch_q.value) if posepoch_q is not None else None

    # Equatorial path (canonical only; PINT maps aliases to canonical attributes)
    ra_q = _get_model_quantity(model, "RAJ")
    dec_q = _get_model_quantity(model, "DECJ")
    if ra_q is not None and dec_q is not None:
        ra_hours = Angle(ra_q).to(u.hourangle).value
        dec_deg = Angle(dec_q).to(u.deg).value

        pmra_q = _get_model_quantity(model, "PMRA")
        pmdec_q = _get_model_quantity(model, "PMDEC")
        pmra = pmra_q.to(u.mas / u.yr).value if pmra_q is not None else None
        pmdec = pmdec_q.to(u.mas / u.yr).value if pmdec_q is not None else None

        # Propagate to J2000 if possible
        ra_hours_j2000, dec_deg_j2000 = _propagate_equatorial_to_j2000(
            ra_hours, dec_deg, pmra, pmdec, posepoch_mjd
        )
        return SkyCoord(
            ra=ra_hours_j2000 * u.hourangle, dec=dec_deg_j2000 * u.deg, frame=ICRS()
        )

    # Ecliptic path (canonical; ELONG/ELAT exist regardless of LAMBDA/BETA usage)
    lon_q = _get_model_quantity(model, "ELONG")
    lat_q = _get_model_quantity(model, "ELAT")
    if lon_q is not None and lat_q is not None:
        lam_deg = Angle(lon_q).to(u.deg).value
        bet_deg = Angle(lat_q).to(u.deg).value

        pmelong_q = _get_model_quantity(model, "PMELONG")
        pmelat_q = _get_model_quantity(model, "PMELAT")
        pmelong = pmelong_q.to(u.mas / u.yr).value if pmelong_q is not None else None
        pmelat = pmelat_q.to(u.mas / u.yr).value if pmelat_q is not None else None

        lam_deg_j2000, bet_deg_j2000 = _propagate_ecliptic_to_j2000(
            lam_deg, bet_deg, pmelong, pmelat, posepoch_mjd
        )
        c_ecl = SkyCoord(
            lon=lam_deg_j2000 * u.deg,
            lat=bet_deg_j2000 * u.deg,
            distance=1 * u.pc,
            frame=BarycentricMeanEcliptic(equinox=J2000_TIME),
        )
        return c_ecl.transform_to(ICRS())

    # Legacy FK4/B1950 fallback
    ra_q = _get_model_quantity(model, "RA")
    dec_q = _get_model_quantity(model, "DEC")
    if ra_q is not None and dec_q is not None:
        ra = Angle(ra_q).to(u.hourangle)
        dec = Angle(dec_q).to(u.deg)
        c_fk4 = SkyCoord(ra=ra, dec=dec, frame=FK4(equinox=Time("B1950")))
        return c_fk4.transform_to(ICRS())

    raise ValueError("Could not derive coordinates from PINT TimingModel.")


def _skycoord_from_libstempo(psr: Any) -> SkyCoord:
    """Build a SkyCoord from a libstempo tempopulsar, normalized to J2000 when possible.

    Uses RAJ/DECJ (radians) with optional PMRA/PMDEC (mas/yr) and POSEPOCH (MJD).
    Assumes PMELONG/PMELAT and PMRA/PMDEC are in mas/yr (tempo2/PINT conventions).
    Falls back to ecliptic or FK4 as needed.
    """

    def _val_aliases(psr, canonical: str):
        for key in get_aliases_for_parameter(canonical):
            try:
                return psr[key].val
            except Exception:
                pass
        return None

    raj = _val_aliases(psr, "RAJ")  # covers RA
    decj = _val_aliases(psr, "DECJ")  # covers DEC
    if raj is not None and decj is not None:
        ra_hours = (raj * u.rad).to(u.hourangle).value
        dec_deg = (decj * u.rad).to(u.deg).value

        # Attempt PM propagation
        pmra = _val_aliases(psr, "PMRA")  # mas/yr
        pmdec = _val_aliases(psr, "PMDEC")  # mas/yr
        posepoch_mjd = _val_aliases(psr, "POSEPOCH")  # MJD
        if posepoch_mjd is None:
            # Tempo2/NANOGrav convention: fall back to PEPOCH
            posepoch_mjd = _val_aliases(psr, "PEPOCH")

        ra_hours_j2000, dec_deg_j2000 = _propagate_equatorial_to_j2000(
            ra_hours, dec_deg, pmra, pmdec, posepoch_mjd
        )
        return SkyCoord(
            ra=ra_hours_j2000 * u.hourangle, dec=dec_deg_j2000 * u.deg, frame=ICRS()
        )

    # Ecliptic variants (in radians)
    lam = _val_aliases(psr, "ELONG")  # covers LAMBDA
    bet = _val_aliases(psr, "ELAT")  # covers BETA
    if lam is not None and bet is not None:
        lam_deg = (lam * u.rad).to(u.deg).value
        bet_deg = (bet * u.rad).to(u.deg).value

        pmelong = _val_aliases(psr, "PMELONG")  # covers PMLAMBDA
        pmelat = _val_aliases(psr, "PMELAT")  # covers PMBETA
        posepoch_mjd = _val_aliases(psr, "POSEPOCH")
        if posepoch_mjd is None:
            # Tempo2/NANOGrav convention: fall back to PEPOCH
            posepoch_mjd = _val_aliases(psr, "PEPOCH")

        lam_deg_j2000, bet_deg_j2000 = _propagate_ecliptic_to_j2000(
            lam_deg, bet_deg, pmelong, pmelat, posepoch_mjd
        )
        c = SkyCoord(
            lon=lam_deg_j2000 * u.deg,
            lat=bet_deg_j2000 * u.deg,
            distance=1 * u.pc,
            frame=BarycentricMeanEcliptic(equinox=J2000_TIME),
        )
        return c.transform_to(ICRS())

    # FK4 B1950 fallback (rare)
    ra_b = _val_aliases(psr, "RA")
    dec_b = _val_aliases(psr, "DEC")
    if ra_b is not None and dec_b is not None:
        c_fk4 = SkyCoord(
            ra=ra_b * u.rad, dec=dec_b * u.rad, frame=FK4(equinox=Time("B1950"))
        )
        return c_fk4.transform_to(ICRS())

    raise ValueError("Could not derive coordinates from libstempo tempopulsar.")


def _skycoord_from_enterprise(psr: Any) -> SkyCoord:
    """
    Build a SkyCoord from an Enterprise Pulsar (PintPulsar or Tempo2Pulsar).

    Uses internal _raj/_decj attributes stored in radians (ICRS-equivalent).

    Args:
        psr: Enterprise Pulsar object with _raj/_decj attributes

    Returns:
        SkyCoord object in ICRS frame

    Raises:
        ValueError: If _raj/_decj attributes not found
    """
    if hasattr(psr, "_raj") and hasattr(psr, "_decj"):
        return SkyCoord(ra=psr._raj * u.rad, dec=psr._decj * u.rad, frame=ICRS())
    raise ValueError("Enterprise pulsar lacks _raj/_decj.")


def bj_name_from_pulsar(psr_obj: Any, name_type: str = "J") -> str:
    """Generate canonical B-name or J-name from pulsar object coordinates.

    Coordinates are normalized to J2000 using POSEPOCH + proper motion when available
    before formatting the name, ensuring epoch-stable canonical naming.

    Supports multiple pulsar object types:
    - PINT TimingModel
    - PINT tuple (model, toas) - uses the model
    - libstempo tempopulsar
    - Enterprise Pulsar (PintPulsar or Tempo2Pulsar)

    Args:
        psr_obj: Pulsar object with coordinate information
        name_type: "J" for J-name (JHHMM±DDMM) or "B" for B-name (BHHMM±DD)

    Returns:
        Canonical name string (e.g., "J1857+0943" or "B1857+09")

    Raises:
        ValueError: If coordinates cannot be extracted from object or invalid name_type
    """
    # Validate name_type
    if name_type.upper() not in ["J", "B"]:
        raise ValueError(f"Invalid name_type '{name_type}'. Must be 'J' or 'B'")

    # Handle PINT tuple (model, toas) - extract the model
    if isinstance(psr_obj, tuple) and len(psr_obj) == 2:
        psr_obj = psr_obj[0]  # Use the model from the tuple

    # Try enterprise first (common in your MetaPulsar flow)
    try:
        c = _skycoord_from_enterprise(psr_obj)
    except Exception:
        # Try PINT TimingModel
        try:
            c = _skycoord_from_pint_model(psr_obj)
        except Exception:
            # Try libstempo tempopulsar
            c = _skycoord_from_libstempo(psr_obj)

    # Ensure we're in ICRS (if any upstream gave a different frame)
    c_icrs = c.transform_to(ICRS())

    if name_type.upper() == "B":
        # B-names should be based on FK4 B1950 coordinates, not ICRS
        c_fk4 = c_icrs.transform_to(FK4(equinox=Time("B1950")))
        return _format_b_name_from_icrs(c_fk4)
    else:
        return _format_j_name_from_icrs(c_icrs)


# ============================================================================
# ALIAS-DRIVEN PARAMETER ACCESS
# ============================================================================


def _get_first_par_value_by_aliases(
    parfile_dict: Dict[str, str], canonical_param: str
) -> Optional[str]:
    """Return first non-empty value among all aliases for a canonical parameter.

    This leverages PINT's alias map so that we accept ELONG/LAMBDA, ELAT/BETA,
    PMELONG/PMLAMBDA, PMELAT/PMBETA, RAJ/RA, DECJ/DEC, etc., without hard-coding names.
    """
    for key in get_aliases_for_parameter(canonical_param):
        val = parfile_dict.get(key)
        if val:
            return val
    return None


def _get_model_quantity(model, canonical_name: str):
    """Get a model parameter's quantity by its canonical name; None if missing/empty."""
    if (
        hasattr(model, canonical_name)
        and getattr(model, canonical_name).value is not None
    ):
        return getattr(model, canonical_name).quantity
    return None


# ============================================================================
# OPTIMIZED COORDINATE EXTRACTION FUNCTIONS
# ============================================================================


def _parse_parfile_optimized(parfile_content: str) -> Dict[str, str]:
    """Parse parfile content using PINT's robust parser."""
    parfile_dict = parse_parfile(StringIO(parfile_content))
    # Convert defaultdict(list) to dict with first values for compatibility
    # Also split on whitespace to get only the first value (before uncertainty columns)
    result = {}
    for k, v in parfile_dict.items():
        if v:
            # Take first value and split to get only the parameter value (not uncertainty)
            first_value = v[0].split()[0] if v[0].split() else ""
            result[k] = first_value
        else:
            result[k] = ""
    return result


def _parse_ra_string_optimized(ra_str: str) -> Optional[float]:
    """Parse RA string using Astropy's Angle parsing."""
    try:
        angle = Angle(ra_str, unit=u.hourangle)
        return angle.to(u.hourangle).value
    except Exception:
        return None


def _parse_dec_string_optimized(dec_str: str) -> Optional[float]:
    """Parse DEC string using Astropy's Angle parsing."""
    try:
        angle = Angle(dec_str, unit=u.deg)
        return angle.to(u.deg).value
    except Exception:
        return None


def _parse_angle_string_optimized(angle_str: str) -> Optional[float]:
    """Parse angle string using Astropy's Angle parsing."""
    try:
        angle = Angle(angle_str, unit=u.deg)
        return angle.to(u.deg).value
    except Exception:
        return None


def _parse_float_optimized(value: Optional[str]) -> Optional[float]:
    """Parse a float from a simple string; return None on failure.

    The optimized parfile parser already strips uncertainty/fit columns,
    so we can safely attempt a plain float conversion.
    """
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _get_pm_equatorial_masyr_optimized(
    parfile_dict: Dict[str, str],
) -> Tuple[Optional[float], Optional[float]]:
    """Return (PMRA, PMDEC) in mas/yr if available via aliases; otherwise (None, None).

    Conventions: PMRA is μ_α cosδ (mas/yr), PMDEC is μ_δ (mas/yr).
    """
    pmra_val = _get_first_par_value_by_aliases(parfile_dict, "PMRA")
    pmdec_val = _get_first_par_value_by_aliases(parfile_dict, "PMDEC")
    return _parse_float_optimized(pmra_val), _parse_float_optimized(pmdec_val)


def _get_pm_ecliptic_masyr_optimized(
    parfile_dict: Dict[str, str],
) -> Tuple[Optional[float], Optional[float]]:
    """Return (PMELONG, PMELAT) in mas/yr if available via aliases; otherwise (None, None).

    Conventions: PMELONG is μ_λ cosβ (mas/yr), PMELAT is μ_β (mas/yr).
    """
    pm_lon_val = _get_first_par_value_by_aliases(
        parfile_dict, "PMELONG"
    )  # covers PMLAMBDA
    pm_lat_val = _get_first_par_value_by_aliases(
        parfile_dict, "PMELAT"
    )  # covers PMBETA
    return _parse_float_optimized(pm_lon_val), _parse_float_optimized(pm_lat_val)


def _get_pulsar_name_from_parfile_dict(parfile_dict: Dict[str, str]) -> str:
    """Best-effort pulsar name from a parsed parfile, used purely for diagnostics.

    Tries PSRJ first, then PSRB, then PSR, and finally returns ``"<unknown>"``
    if none of those fields are present. Never raises.
    """
    for key in ("PSRJ", "PSRB", "PSR"):
        val = parfile_dict.get(key)
        if val:
            return val.strip()
    return "<unknown>"


def _get_posepoch_mjd_optimized(parfile_dict: Dict[str, str]) -> Optional[float]:
    """Return POSEPOCH (MJD), falling back to PEPOCH when POSEPOCH is absent.

    Tempo2/NANOGrav convention: when POSEPOCH is omitted from a parfile, the
    position epoch is taken to equal PEPOCH (the spin reference epoch). PINT's
    alias map intentionally does not link these two parameters because they
    are physically distinct, so we emulate that convention here for the
    coordinate-extraction / canonical-naming path only.
    """
    posepoch = _parse_float_optimized(parfile_dict.get("POSEPOCH"))
    if posepoch is not None:
        return posepoch
    return _parse_float_optimized(parfile_dict.get("PEPOCH"))


def _propagate_equatorial_to_j2000(
    ra_hours: float,
    dec_deg: float,
    pm_ra_cosdec_masyr: Optional[float],
    pm_dec_masyr: Optional[float],
    posepoch_mjd: Optional[float],
) -> Tuple[float, float]:
    """Propagate an equatorial position from POSEPOCH to J2000 using small-angle spherical propagation.

    Args:
        ra_hours: Right ascension in hours at POSEPOCH.
        dec_deg: Declination in degrees at POSEPOCH.
        pm_ra_cosdec_masyr: μ_α cosδ in mas/yr (None to skip).
        pm_dec_masyr: μ_δ in mas/yr (None to skip).
        posepoch_mjd: POSEPOCH in MJD (None to skip).

    Returns:
        (ra_hours_at_J2000, dec_deg_at_J2000)
    """
    if pm_ra_cosdec_masyr is None or pm_dec_masyr is None or posepoch_mjd is None:
        return ra_hours, dec_deg

    dt_yr = (J2000_TIME - Time(posepoch_mjd, format="mjd")).to_value("yr")
    if dt_yr == 0:
        return ra_hours, dec_deg

    dec_rad = np.deg2rad(dec_deg)
    dra_deg = (pm_ra_cosdec_masyr / np.cos(dec_rad)) * dt_yr * MAS_TO_DEG
    ddec_deg = pm_dec_masyr * dt_yr * MAS_TO_DEG

    ra_deg_new = (ra_hours * 15.0 + dra_deg) % 360.0
    dec_deg_new = dec_deg + ddec_deg
    return ra_deg_new / 15.0, dec_deg_new


def _propagate_ecliptic_to_j2000(
    lon_deg: float,
    lat_deg: float,
    pm_lon_coslat_masyr: Optional[float],
    pm_lat_masyr: Optional[float],
    posepoch_mjd: Optional[float],
) -> Tuple[float, float]:
    """Propagate an ecliptic position from POSEPOCH to J2000 using small-angle spherical propagation.

    Args:
        lon_deg: Ecliptic longitude (deg) at POSEPOCH.
        lat_deg: Ecliptic latitude (deg) at POSEPOCH.
        pm_lon_coslat_masyr: μ_λ cosβ in mas/yr (None to skip).
        pm_lat_masyr: μ_β in mas/yr (None to skip).
        posepoch_mjd: POSEPOCH in MJD (None to skip).

    Returns:
        (lon_deg_at_J2000, lat_deg_at_J2000)
    """
    if pm_lon_coslat_masyr is None or pm_lat_masyr is None or posepoch_mjd is None:
        return lon_deg, lat_deg

    dt_yr = (J2000_TIME - Time(posepoch_mjd, format="mjd")).to_value("yr")
    if dt_yr == 0:
        return lon_deg, lat_deg

    lat_rad = np.deg2rad(lat_deg)
    dlon_deg = (pm_lon_coslat_masyr / np.cos(lat_rad)) * dt_yr * MAS_TO_DEG
    dlat_deg = pm_lat_masyr * dt_yr * MAS_TO_DEG

    lon_deg_new = (lon_deg + dlon_deg) % 360.0
    lat_deg_new = lat_deg + dlat_deg
    return lon_deg_new, lat_deg_new


def _extract_equatorial_coordinates_optimized(
    parfile_dict: Dict[str, str],
) -> Tuple[Optional[float], Optional[float]]:
    """Extract RAJ/DECJ (via aliases) and propagate from POSEPOCH to J2000 if PM/POSEPOCH exist.

    POSEPOCH falls back to PEPOCH when not explicitly set, matching the
    tempo2/NANOGrav convention used by many publicly released parfiles
    (e.g. NANOGrav 9-yr/12.5-yr ``.gls.par`` files).

    Returns RA (hours) and DEC (degrees) at J2000 when propagation is possible,
    otherwise returns the catalogued values. Output coordinates are suitable for
    canonical naming and cross-PTA matching.
    """
    try:
        # Use alias map to accept RAJ/RA and DECJ/DEC
        raj = _get_first_par_value_by_aliases(parfile_dict, "RAJ")
        decj = _get_first_par_value_by_aliases(parfile_dict, "DECJ")

        if not raj or not decj:
            return None, None

        ra_hours = _parse_ra_string_optimized(raj)
        dec_deg = _parse_dec_string_optimized(decj)
        if ra_hours is None or dec_deg is None:
            return None, None

        # Equatorial PM + POSEPOCH extraction (POSEPOCH falls back to PEPOCH)
        pmra, pmdec = _get_pm_equatorial_masyr_optimized(parfile_dict)
        posepoch_mjd = _get_posepoch_mjd_optimized(parfile_dict)

        # Issue warning if PM/POSEPOCH missing (epoch-stable naming requires them)
        if pmra is None or pmdec is None or posepoch_mjd is None:
            psr_name = _get_pulsar_name_from_parfile_dict(parfile_dict)
            logger.warning(
                f"[{psr_name}] Missing PMRA/PMDEC or POSEPOCH/PEPOCH in parfile. "
                "Using catalogued position without proper motion propagation. "
                "Canonical naming may be unstable across epochs."
            )

        # Propagate to J2000 if possible
        ra_hours_j2000, dec_deg_j2000 = _propagate_equatorial_to_j2000(
            ra_hours, dec_deg, pmra, pmdec, posepoch_mjd
        )
        return ra_hours_j2000, dec_deg_j2000

    except Exception:
        return None, None


def _extract_ecliptic_coordinates_optimized(
    parfile_dict: Dict[str, str],
) -> Tuple[Optional[float], Optional[float]]:
    """Extract ecliptic coords (via aliases), propagate to J2000 using PM if available, then convert to ICRS.

    POSEPOCH falls back to PEPOCH when not explicitly set, matching the
    tempo2/NANOGrav convention used by many publicly released parfiles
    (e.g. NANOGrav 9-yr/12.5-yr ``.gls.par`` files).

    Returns RA (hours) and DEC (degrees) at J2000 when possible, otherwise None.
    """
    try:
        # Use alias map to accept ELONG/LAMBDA and ELAT/BETA
        lam = _get_first_par_value_by_aliases(parfile_dict, "ELONG")
        bet = _get_first_par_value_by_aliases(parfile_dict, "ELAT")
        if not lam or not bet:
            return None, None

        lam_deg = _parse_angle_string_optimized(lam)
        bet_deg = _parse_angle_string_optimized(bet)
        if lam_deg is None or bet_deg is None:
            return None, None

        # Ecliptic PM + POSEPOCH extraction (POSEPOCH falls back to PEPOCH)
        pmelong, pmelat = _get_pm_ecliptic_masyr_optimized(parfile_dict)
        posepoch_mjd = _get_posepoch_mjd_optimized(parfile_dict)

        # Issue warning if PM/POSEPOCH missing (epoch-stable naming requires them)
        if pmelong is None or pmelat is None or posepoch_mjd is None:
            psr_name = _get_pulsar_name_from_parfile_dict(parfile_dict)
            logger.warning(
                f"[{psr_name}] Missing PMELONG/PMELAT or POSEPOCH/PEPOCH in parfile. "
                "Using catalogued position without proper motion propagation. "
                "Canonical naming may be unstable across epochs."
            )

        # Propagate ecliptic coords to J2000 if possible
        lam_deg_j2000, bet_deg_j2000 = _propagate_ecliptic_to_j2000(
            lam_deg, bet_deg, pmelong, pmelat, posepoch_mjd
        )

        # Convert ecliptic (J2000) -> ICRS (J2000)
        c_ecl_j2000 = SkyCoord(
            lon=lam_deg_j2000 * u.deg,
            lat=bet_deg_j2000 * u.deg,
            distance=1 * u.pc,
            frame=BarycentricMeanEcliptic(equinox=J2000_TIME),
        )
        c_icrs = c_ecl_j2000.transform_to(ICRS())

        return c_icrs.ra.to(u.hourangle).value, c_icrs.dec.to(u.deg).value

    except Exception:
        return None, None


def _extract_fk4_coordinates_optimized(
    parfile_dict: Dict[str, str],
) -> Tuple[Optional[float], Optional[float]]:
    """Extract FK4/B1950 coordinates and convert to equatorial (optimized version)."""
    try:
        ra = parfile_dict.get("RA")
        dec = parfile_dict.get("DEC")

        if not ra or not dec:
            return None, None

        # Parse coordinates
        ra_hours = _parse_ra_string_optimized(ra)
        dec_deg = _parse_dec_string_optimized(dec)

        if ra_hours is None or dec_deg is None:
            return None, None

        # Convert FK4 to ICRS
        c_fk4 = SkyCoord(
            ra=ra_hours * u.hourangle,
            dec=dec_deg * u.deg,
            frame=FK4(equinox=Time("B1950")),
        )
        c_icrs = c_fk4.transform_to(ICRS())

        return c_icrs.ra.to(u.hourangle).value, c_icrs.dec.to(u.deg).value

    except Exception:
        return None, None


def extract_coordinates_from_parfile_optimized(
    parfile_content: str,
) -> Optional[Tuple[float, float]]:
    """
    Extract RA/DEC coordinates directly from parfile content (optimized version).

    This function bypasses PINT model creation and extracts coordinates using
    lightweight parsing for significant performance improvements.

    Args:
        parfile_content: Raw parfile content as string

    Returns:
        Tuple of (RA_hours, DEC_degrees) or None if extraction fails
    """
    try:
        # Parse parfile into simple dictionary
        parfile_dict = _parse_parfile_optimized(parfile_content)

        # Try direct equatorial coordinates first (most common)
        ra_hours, dec_deg = _extract_equatorial_coordinates_optimized(parfile_dict)
        if ra_hours is not None and dec_deg is not None:
            return ra_hours, dec_deg

        # Try ecliptic coordinates as fallback
        ra_hours, dec_deg = _extract_ecliptic_coordinates_optimized(parfile_dict)
        if ra_hours is not None and dec_deg is not None:
            return ra_hours, dec_deg

        # Try FK4/B1950 coordinates as last resort
        ra_hours, dec_deg = _extract_fk4_coordinates_optimized(parfile_dict)
        if ra_hours is not None and dec_deg is not None:
            return ra_hours, dec_deg

        return None

    except Exception as e:
        logger.debug(f"Failed to extract coordinates: {e}")
        return None


def bj_name_from_coordinates_optimized(
    ra_hours: float, dec_deg: float, name_type: str = "J"
) -> str:
    """
    Generate B-name or J-name from coordinates without PINT model creation (optimized version).

    Args:
        ra_hours: Right ascension in hours
        dec_deg: Declination in degrees
        name_type: "J" for J-name (JHHMM±DDMM) or "B" for B-name (BHHMM±DD)

    Returns:
        Canonical name string (e.g., "J1857+0943" or "B1857+09")
    """
    # Create SkyCoord for coordinate transformations
    c_icrs = SkyCoord(ra=ra_hours * u.hourangle, dec=dec_deg * u.deg, frame=ICRS())

    if name_type.upper() == "B":
        # B-names should be based on FK4 B1950 coordinates
        c_fk4 = c_icrs.transform_to(FK4(equinox=Time("B1950")))
        return _format_b_name_from_coordinates_optimized(
            c_fk4.ra.to(u.hourangle).value, c_fk4.dec.to(u.deg).value
        )
    else:
        return _format_j_name_from_coordinates_optimized(ra_hours, dec_deg)


def _format_j_name_from_coordinates_optimized(ra_hours: float, dec_deg: float) -> str:
    """Format ICRS coordinates into a JHHMM±DDMM label using TRUNCATION (optimized version)."""
    # RA
    hh = int(np.floor(ra_hours)) % 24
    mm = int((ra_hours - hh) * 60.0)  # truncate minutes

    # Dec
    sign = "-" if dec_deg < 0 else "+"
    a = abs(dec_deg)
    DD = int(np.floor(a))
    MM = int((a - DD) * 60.0)  # truncate arcminutes

    return f"J{hh:02d}{mm:02d}{sign}{DD:02d}{MM:02d}"


def _format_b_name_from_coordinates_optimized(ra_hours: float, dec_deg: float) -> str:
    """Format FK4 coordinates into a B1234±56 label using TRUNCATION (optimized version)."""
    # RA
    hh = int(np.floor(ra_hours)) % 24
    mm = int((ra_hours - hh) * 60.0)  # truncate minutes

    # Dec
    sign = "-" if dec_deg < 0 else "+"
    a = abs(dec_deg)
    DD = int(np.floor(a))

    return f"B{hh:02d}{mm:02d}{sign}{DD:02d}"


def discover_pulsars_by_coordinates_optimized(
    file_data: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Discover pulsars by extracting coordinates directly from parfiles (optimized version).

    This optimized version bypasses PINT model creation and extracts
    coordinates using lightweight parsing for significant performance improvements.

    Args:
        file_data: Dictionary mapping PTA names to file lists

    Returns:
        Dictionary mapping J-names to PTA file data
    """
    coordinate_map = {}

    for pta_name, file_list in file_data.items():
        logger.debug(f"Processing {len(file_list)} files for PTA {pta_name}")

        for file_dict in file_list:
            try:
                # Extract coordinates directly from parfile content
                coords = extract_coordinates_from_parfile_optimized(
                    file_dict["par_content"]
                )

                if coords is None:
                    logger.warning(
                        f"Could not extract coordinates from {file_dict.get('par', 'unknown')}"
                    )
                    continue

                ra_hours, dec_deg = coords

                # Generate J-name directly from coordinates
                j_name = bj_name_from_coordinates_optimized(ra_hours, dec_deg, "J")

                # Add to coordinate map
                if j_name not in coordinate_map:
                    coordinate_map[j_name] = {}
                if pta_name not in coordinate_map[j_name]:
                    coordinate_map[j_name][pta_name] = []

                coordinate_map[j_name][pta_name].append(file_dict)

                logger.debug(
                    f"Found pulsar {j_name} at RA={ra_hours:.4f}h, DEC={dec_deg:.4f}°"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to process {file_dict.get('par', 'unknown')}: {e}"
                )
                continue

    logger.info(f"Discovered {len(coordinate_map)} unique pulsars across all PTAs")
    return coordinate_map
