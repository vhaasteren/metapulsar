#!/usr/bin/env python3
"""Generate Enterprise-produced golden fixtures for PTA materializer parity.

This is not a pytest module. It requires enterprise-pulsar and writes NPZ/JSON
under tests/fixtures/enterprise_surface/. Ordinary tests only read those files.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIXTURE_DIR.parents[2]
SAMPLE_DIR = REPO_ROOT / "tests" / "fixtures" / "sample_parfiles"
PULSE_DIR = REPO_ROOT / "tests" / "fixtures" / "pulse_tracking"

RECORD_ARRAYS = (
    "_toas",
    "_stoas",
    "_residuals",
    "_toaerrs",
    "_ssbfreqs",
    "_telescope",
    "_designmatrix",
    "_pos",
    "_pos_t",
    "_planetssb",
    "_sunssb",
)

PUBLIC_ARRAYS = (
    "toas",
    "stoas",
    "residuals",
    "toaerrs",
    "freqs",
    "telescope",
    "Mmat",
    "backend_flags",
    "pos",
    "pos_t",
    "planetssb",
    "sunssb",
)


def _require_enterprise():
    try:
        import enterprise
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "enterprise-pulsar is required to regenerate golden fixtures"
        ) from exc
    return enterprise


def _flags_to_dict(flags) -> dict[str, np.ndarray]:
    if isinstance(flags, dict):
        return {str(k): np.asarray(v) for k, v in flags.items()}
    if hasattr(flags, "dtype") and flags.dtype.names:
        return {str(name): np.asarray(flags[name]) for name in flags.dtype.names}
    raise TypeError(f"unsupported flags type: {type(flags)!r}")


def _save_record(path: Path, psr, *, timing_package: str) -> None:
    payload: dict[str, object] = {
        "name": np.asarray(str(psr.name)),
        "timing_package": np.asarray(timing_package),
        "fitpars": np.asarray(list(psr.fitpars), dtype="U64"),
        "setpars": np.asarray(list(psr.setpars), dtype="U64"),
        "_raj": np.asarray(float(psr._raj)),
        "_decj": np.asarray(float(psr._decj)),
        "_pdist": np.asarray(tuple(psr._pdist), dtype=np.float64),
    }
    for key in RECORD_ARRAYS:
        payload[key] = np.asarray(getattr(psr, key))
    for flag, values in _flags_to_dict(psr._flags).items():
        payload[f"flag__{flag}"] = np.asarray(values)
    np.savez(path, **payload)


def _save_public_surface(path: Path, mp) -> None:
    payload: dict[str, object] = {
        "name": np.asarray(str(mp.name)),
        "fitpars": np.asarray(list(mp.fitpars)),
        "pdist": np.asarray(tuple(mp.pdist), dtype=np.float64),
    }
    for key in PUBLIC_ARRAYS:
        payload[key] = np.asarray(getattr(mp, key))
    flags = mp.flags
    if hasattr(flags, "dtype") and flags.dtype.names:
        for name in flags.dtype.names:
            payload[f"flag__{name}"] = np.asarray(flags[name])
    else:
        for name, values in flags.items():
            payload[f"flag__{name}"] = np.asarray(values)
    np.savez(path, **payload)


def _build_pint(par: Path, tim: Path):
    from enterprise.pulsar import PintPulsar
    from pint.models import get_model_and_toas

    model, toas = get_model_and_toas(str(par), str(tim), planets=True)
    return PintPulsar(toas, model, sort=False, planets=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing fixture files",
    )
    args = parser.parse_args(argv)

    enterprise = _require_enterprise()
    from enterprise.pulsar import Tempo2Pulsar
    from metapulsar.metapulsar import MetaPulsar
    from metapulsar.mockpulsar import create_mock_libstempo, write_mock_pta_files

    targets = {
        "pint_equatorial.npz": FIXTURE_DIR / "pint_equatorial.npz",
        "pint_ecliptic.npz": FIXTURE_DIR / "pint_ecliptic.npz",
        "tempo2_mock_equatorial.npz": FIXTURE_DIR / "tempo2_mock_equatorial.npz",
        "metapulsar_tempo2_pair.npz": FIXTURE_DIR / "metapulsar_tempo2_pair.npz",
        "manifest.json": FIXTURE_DIR / "manifest.json",
    }
    existing = [p for p in targets.values() if p.exists()]
    if existing and not args.overwrite:
        names = ", ".join(p.name for p in existing)
        raise SystemExit(
            f"refusing to overwrite existing fixtures ({names}); "
            "pass --overwrite to regenerate"
        )

    import astropy
    import numpy
    import pint

    equatorial = _build_pint(SAMPLE_DIR / "simple.par", SAMPLE_DIR / "simple.tim")
    ecliptic = _build_pint(
        PULSE_DIR / "nanograv_like.par", PULSE_DIR / "nanograv_like.tim"
    )
    mock = create_mock_libstempo(
        n_toas=30, name="J1857+0943", telescope="pta_a", seed=10
    )
    tempo2 = Tempo2Pulsar(mock, sort=False, planets=True)

    _save_record(targets["pint_equatorial.npz"], equatorial, timing_package="pint")
    _save_record(targets["pint_ecliptic.npz"], ecliptic, timing_package="pint")
    _save_record(targets["tempo2_mock_equatorial.npz"], tempo2, timing_package="tempo2")

    pulsars = {
        "pta_a": create_mock_libstempo(
            n_toas=30, name="J1857+0943", telescope="pta_a", seed=10
        ),
        "pta_b": create_mock_libstempo(
            n_toas=30, name="J1857+0943", telescope="pta_b", seed=20
        ),
    }
    # The retained pars must outlive every read: MetaPulsar loads par text
    # lazily, so keep the directory alive for the whole surface dump.
    with tempfile.TemporaryDirectory(prefix="metapulsar_surface_") as pta_file_dir:
        mp = MetaPulsar(
            pulsars,
            combination_strategy="per_pta",
            pta_files=write_mock_pta_files(pulsars, pta_file_dir),
        )
        _save_public_surface(targets["metapulsar_tempo2_pair.npz"], mp)

    manifest = {
        "enterprise_version": enterprise.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": numpy.__version__,
        "astropy_version": astropy.__version__,
        "pint_version": pint.__version__,
        "files": sorted(targets),
    }
    targets["manifest.json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote fixtures under {FIXTURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
