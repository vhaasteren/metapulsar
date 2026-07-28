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

from typing import Any, Dict, Optional, Tuple, List, Iterable, Set
import re
import warnings
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


_CATALOG_NAME_RE = re.compile(r"^([BJ]\d{4}[+-]\d{2,4})([A-Z])?$")


def parse_catalog_names(parfile_content: str) -> Dict[str, Optional[str]]:
    """Parse PSRJ / PSRB / PSR fields from parfile content."""
    parfile_dict = _parse_parfile_optimized(parfile_content)
    result: Dict[str, Optional[str]] = {"psrj": None, "psrb": None, "psr": None}
    for key, out_key in (("PSRJ", "psrj"), ("PSRB", "psrb"), ("PSR", "psr")):
        val = parfile_dict.get(key)
        if val:
            result[out_key] = val.strip()
    return result


def catalog_name_candidates(parfile_dict: Dict[str, str]) -> List[str]:
    """Ordered unique catalog strings present on a parfile."""
    seen: Set[str] = set()
    out: List[str] = []
    for key in ("PSRJ", "PSRB", "PSR"):
        val = parfile_dict.get(key)
        if val:
            name = val.strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


def letter_suffix(name: str) -> Optional[str]:
    """Trailing cluster letter (A–Z) after the numeric designator, if any."""
    m = _CATALOG_NAME_RE.match(name.strip())
    if not m:
        return None
    return m.group(2)


def preferred_file_label(parfile_dict: Dict[str, str], path_name: Optional[str]) -> str:
    """B-preferred catalog label for one file; path_name is last resort."""
    psrj = (parfile_dict.get("PSRJ") or "").strip() or None
    psrb = (parfile_dict.get("PSRB") or "").strip() or None
    psr = (parfile_dict.get("PSR") or "").strip() or None

    if psrb:
        return psrb
    if psr and psr.startswith("B") and len(psr) >= 6:
        return psr
    if psrj:
        return psrj
    if psr:
        return psr
    if path_name:
        logger.warning(
            f"No PSRJ/PSRB/PSR in parfile; using path name {path_name!r} as label"
        )
        return path_name
    return "<unknown>"


def preferred_group_name(catalog_strings: Iterable[str]) -> str:
    """B-preferred display name for a pulsar group from all member catalog strings."""
    names = [n.strip() for n in catalog_strings if n and n.strip()]
    if not names:
        raise ValueError("Cannot determine group name: no catalog strings")

    b_names = sorted({n for n in names if n.startswith("B") and len(n) >= 6})
    if b_names:
        return b_names[0]

    j_names = sorted({n for n in names if n.startswith("J")})
    if j_names:
        suffixed = sorted(n for n in j_names if letter_suffix(n))
        if suffixed:
            return suffixed[0]
        return j_names[0]

    return sorted(set(names))[0]


def _suffix_sets_compatible(
    suffixes_a: Set[Optional[str]], suffixes_b: Set[Optional[str]]
) -> Tuple[bool, Optional[str], bool]:
    """Compare suffix sets for two clusters. Returns (may_merge, error, warn_separate)."""
    combined = suffixes_a | suffixes_b
    non_null = {s for s in combined if s is not None}
    has_null = None in combined

    if non_null and has_null and len(non_null) > 0:
        return (
            False,
            f"Ambiguous pulsar identity: mixed letter suffix and bare catalog names "
            f"(suffixes={non_null}) within match tolerance",
            False,
        )
    if len(non_null) > 1:
        return False, None, True
    return True, None, False


def _collect_suffixes_from_catalogs(catalogs: Iterable[str]) -> Set[Optional[str]]:
    out: Set[Optional[str]] = set()
    for name in catalogs:
        out.add(letter_suffix(name))
    if not out:
        out.add(None)
    return out


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _build_position_records(
    file_data: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """One record per file with sky position and catalog metadata."""
    from pathlib import Path

    from .file_discovery import extract_pulsar_name_from_path

    records: List[Dict[str, Any]] = []
    for pta_name, file_list in file_data.items():
        for file_dict in file_list:
            par_content = file_dict.get("par_content")
            if not par_content:
                logger.warning(
                    f"Skipping file without par_content: {file_dict.get('par', 'unknown')}"
                )
                continue

            coords = extract_coordinates_from_parfile_optimized(par_content)
            if coords is None:
                logger.warning(
                    f"Could not extract coordinates from {file_dict.get('par', 'unknown')}"
                )
                continue

            ra_hours, dec_deg = coords
            parfile_dict = _parse_parfile_optimized(par_content)
            path_name: Optional[str] = None
            par_path = file_dict.get("par")
            if par_path is not None:
                try:
                    path_name = extract_pulsar_name_from_path(Path(par_path))
                except ValueError:
                    path_name = None

            catalogs = catalog_name_candidates(parfile_dict)
            if path_name and path_name not in catalogs:
                catalogs = catalogs + [path_name]

            file_label = preferred_file_label(parfile_dict, path_name)
            skycoord = SkyCoord(
                ra=ra_hours * u.hourangle, dec=dec_deg * u.deg, frame=ICRS()
            )

            records.append(
                {
                    "pta_name": pta_name,
                    "file_dict": file_dict,
                    "skycoord": skycoord,
                    "catalogs": catalogs,
                    "file_label": file_label,
                    "suffixes": _collect_suffixes_from_catalogs(catalogs),
                }
            )

            file_dict["catalog_names"] = list(catalog_name_candidates(parfile_dict))
            file_dict["path_name"] = path_name
            file_dict["file_label"] = file_label

    return records


def _cluster_records_by_position(
    records: List[Dict[str, Any]], match_tol_arcsec: float
) -> List[List[int]]:
    """Cluster record indices by on-sky separation and suffix rules."""
    n = len(records)
    if n == 0:
        return []

    tol = match_tol_arcsec * u.arcsec
    uf = _UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            sep = records[i]["skycoord"].separation(records[j]["skycoord"])
            if sep > tol:
                continue

            ok, err, warn_sep = _suffix_sets_compatible(
                records[i]["suffixes"], records[j]["suffixes"]
            )
            if err:
                raise ValueError(err)
            if warn_sep:
                warnings.warn(
                    f"Pulsars within {match_tol_arcsec}″ have different letter suffixes "
                    f"({records[i]['catalogs']!r} vs {records[j]['catalogs']!r}); "
                    "keeping separate groups",
                    stacklevel=2,
                )
                continue
            if ok:
                uf.union(i, j)

    clusters: Dict[int, List[int]] = {}
    for idx in range(n):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)

    return list(clusters.values())


def _check_catalog_alias_position_consistency(
    records: List[Dict[str, Any]], match_tol_arcsec: float
) -> None:
    """Same catalog/path alias on files separated by > tolerance → ValueError."""
    tol = match_tol_arcsec * u.arcsec
    alias_to_idx: Dict[str, int] = {}
    for idx, rec in enumerate(records):
        for alias in rec["catalogs"]:
            if alias in alias_to_idx:
                other = alias_to_idx[alias]
                sep = records[other]["skycoord"].separation(rec["skycoord"])
                if sep > tol:
                    raise ValueError(
                        f"Catalog name {alias!r} appears on distinct sky positions "
                        f"(separation {sep.to(u.arcsec).value:.3f}″ > {match_tol_arcsec}″)"
                    )
            else:
                alias_to_idx[alias] = idx


def discover_pulsars_by_position(
    file_data: Dict[str, List[Dict[str, Any]]],
    match_tol_arcsec: float = 10.0,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Group PTA file data by J2000 position (≤ match_tol) and catalog identity."""
    records = _build_position_records(file_data)
    if not records:
        logger.info("Discovered 0 unique pulsars across all PTAs")
        return {}

    _check_catalog_alias_position_consistency(records, match_tol_arcsec)
    cluster_lists = _cluster_records_by_position(records, match_tol_arcsec)

    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for member_indices in cluster_lists:
        all_catalogs: List[str] = []
        for idx in member_indices:
            all_catalogs.extend(records[idx]["catalogs"])

        group_name = preferred_group_name(all_catalogs)
        if group_name in groups:
            raise ValueError(
                f"Duplicate pulsar group name {group_name!r} for distinct clusters"
            )

        pta_map: Dict[str, List[Dict[str, Any]]] = {}
        for idx in member_indices:
            rec = records[idx]
            pta = rec["pta_name"]
            pta_map.setdefault(pta, []).append(rec["file_dict"])

        groups[group_name] = pta_map
        logger.debug(
            f"Group {group_name!r}: {len(member_indices)} file(s), "
            f"aliases={sorted(set(all_catalogs))}"
        )

    logger.info(f"Discovered {len(groups)} unique pulsars across all PTAs")
    return groups


def build_alias_map(
    pulsar_groups: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> Dict[str, str]:
    """Map catalog and path aliases to group display names (no truncated coord names)."""
    alias_map: Dict[str, str] = {}
    for group_name, pta_data in pulsar_groups.items():
        alias_map[group_name] = group_name
        for files in pta_data.values():
            for file_dict in files:
                for key in ("catalog_names",):
                    for alias in file_dict.get(key) or []:
                        if alias in alias_map and alias_map[alias] != group_name:
                            raise ValueError(
                                f"Alias {alias!r} maps to both "
                                f"{alias_map[alias]!r} and {group_name!r}"
                            )
                        alias_map[alias] = group_name
                path_name = file_dict.get("path_name")
                if path_name:
                    if path_name in alias_map and alias_map[path_name] != group_name:
                        raise ValueError(
                            f"Path alias {path_name!r} maps to both "
                            f"{alias_map[path_name]!r} and {group_name!r}"
                        )
                    alias_map[path_name] = group_name
    return alias_map


def positions_within_tolerance(
    coords: Iterable[SkyCoord], match_tol_arcsec: float = 10.0
) -> bool:
    """True if all sky positions are pairwise within match_tol_arcsec."""
    coord_list = list(coords)
    if len(coord_list) <= 1:
        return True
    tol = match_tol_arcsec * u.arcsec
    for i in range(len(coord_list)):
        for j in range(i + 1, len(coord_list)):
            if coord_list[i].separation(coord_list[j]) > tol:
                return False
    return True


def assert_catalog_suffixes_compatible(catalog_names: Iterable[str]) -> None:
    """Raise if letter-suffix rules forbid a single MetaPulsar."""
    names = [n for n in catalog_names if n]
    suffixes = {letter_suffix(n) for n in names}
    non_null = {s for s in suffixes if s is not None}
    if None in suffixes and non_null:
        raise ValueError(
            f"Inconsistent pulsar letter suffixes across PTAs: {sorted(names)}"
        )
    if len(non_null) > 1:
        raise ValueError(f"Distinct letter suffixes in one MetaPulsar: {sorted(names)}")


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
    """Deprecated alias for :func:`discover_pulsars_by_position`."""
    import warnings

    warnings.warn(
        "discover_pulsars_by_coordinates_optimized is deprecated; "
        "use discover_pulsars_by_position",
        DeprecationWarning,
        stacklevel=2,
    )
    return discover_pulsars_by_position(file_data)
