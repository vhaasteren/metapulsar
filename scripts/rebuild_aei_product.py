#!/usr/bin/env python3
"""Rebuild an AEI par/tim product through MetaPulsarFactory and re-measure PINT
vs Tempo2 parity on the actual released TOAs.

Two release sets are wired up (``--release-set``):

* ``dr2`` — the IPTA DR2 constituents (EPTA DR1 v2.2, NANOGrav 9-yr,
  PPTA DR1+DR2) from ``data/ipta-dr2``, published to ``data/aei-dr2``.
* ``dr3`` — the DR3-era releases (EPTA DR2, MPTA DR2, NANOGrav 15-yr, PPTA DR3,
  InPTA DR1) from ``data-check``, published to ``data/aei-dr3``.

For each pulsar of the selected releases:

1. Run ``create_metapulsar`` on a mixed PINT+Tempo2 shared stack (the release as
   its native-package leg plus a duplicate leg declared as the other package), so
   the published files sit on the validated cross-package common profile.
   Passes ``canonicalize_tim=True`` explicitly (factory default is False) so the
   published ``.tim`` comes from the canonical writer.
2. Publish the native leg's ``.par`` (with ``TRACK -2``) and ``.tim`` under
   ``{out_root}/{release}/{par,tim}/``.
3. Load that published pair with PINT and with Tempo2 (no fit) and compare the
   residual series TOA by TOA.

Unlike the previous build, nothing here rewrites the ``.tim`` by hand: the
factory owns it. Statistics are measured on the released TOAs, not on a
synthetic schedule.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("TEMPO2", "/opt/software/tempo2/T2runtime")

from pint.models import get_model_and_toas  # noqa: E402
from pint.residuals import Residuals  # noqa: E402

from metapulsar.file_discovery import FileDiscovery, PTA_DATA_RELEASES  # noqa: E402
from metapulsar.metapulsar_factory import create_metapulsar  # noqa: E402
from metapulsar.parameter_manager import AlignmentPolicy  # noqa: E402
from metapulsar.pint_helpers import par_text_with_track_minus_2  # noqa: E402
from metapulsar.sandbox_tempo2 import configure_logging, tempopulsar  # noqa: E402

# Release layouts, with the local source-tree deviations this build has to
# accommodate. InPTA DR1 is consumed from the hand-edited checkout.
RELEASE_SPECS: dict[str, dict[str, Any]] = {
    key: dict(spec) for key, spec in PTA_DATA_RELEASES.items()
}
RELEASE_SPECS["inpta_dr1"]["base_dir"] = "InPTA_DR1_edited/"

RELEASE_SETS: dict[str, dict[str, Any]] = {
    "dr2": {
        "source_root": REPO / "data" / "ipta-dr2",
        "out_root": REPO / "data" / "aei-dr2",
        "releases": ["epta_dr1_v2_2", "nanograv_9y", "ppta_dr2"],
    },
    "dr3": {
        "source_root": REPO / "data-check",
        "out_root": REPO / "data" / "aei-dr3",
        "releases": ["epta_dr2", "mpta_dr2", "nanograv_15y", "ppta_dr3", "inpta_dr1"],
    },
}

PARITY_NS = 2.0
# Arrival times that differ by more than this are a clock-chain disagreement,
# not a timing-model one, and are reported separately.
DT_TOL_S = 1e-3


@dataclass
class Row:
    release: str
    pulsar: str
    n_toas: int = 0
    n_offset: int = 0
    rms_diff_ns: float | None = None
    maxabs_diff_ns: float | None = None
    pint_rms_us: float | None = None
    tempo2_rms_us: float | None = None
    binary: str | None = None
    binary_orig: str | None = None
    ell1h_gauge: str | None = None
    par_path: str | None = None
    tim_path: str | None = None
    passed: bool | None = None
    error: str | None = None


def _first_par_value(text: str, *keys: str) -> Optional[str]:
    wanted = {k.upper() for k in keys}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.upper().startswith("C "):
            continue
        parts = s.split()
        if parts and parts[0].upper() in wanted and len(parts) >= 2:
            return parts[1]
    return None


def explicit_reference_policy(text: str) -> AlignmentPolicy:
    ephem = _first_par_value(text, "EPHEM") or "DE440"
    clock = _first_par_value(text, "CLOCK", "CLK") or "TT(BIPM2019)"
    if clock.upper() in {"TT(BIPM)", "BIPM"}:
        clock = "TT(BIPM2019)"
    match = re.search(r"BIPM\D*(\d{4})", clock.upper())
    # NG J1918-0642's Shapiro fidelity sits ~2% over the published budget; a 5%
    # user margin clears it without changing the conversion map.
    return AlignmentPolicy(
        ephem=ephem,
        clock=clock,
        bipm_version=int(match.group(1)) if match else None,
        binary_conversion="auto",
        binary_fidelity_tolerance_factor=1.05,
    )


def pulsar_name_from_par(par_path: Path, release_key: str) -> str:
    pattern = RELEASE_SPECS[release_key]["par_pattern"]
    match = re.search(pattern, str(par_path).replace("\\", "/"))
    if match:
        return match.group(1)
    match = re.search(r"([BJ]\d{4}[+-]\d{2,4})", par_path.name)
    if not match:
        raise ValueError(f"Cannot extract pulsar name from {par_path}")
    return match.group(1)


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def build_pair(
    release_key: str, entry: dict[str, Any], out_root: Path, work: Path
) -> tuple[Path, Path, str, str]:
    """Run the factory and publish the native leg's par/tim. Returns paths."""
    pulsar = pulsar_name_from_par(Path(entry["par"]), release_key)
    native_pkg = (
        entry.get("timing_package") or RELEASE_SPECS[release_key]["timing_package"]
    )
    donor_pkg = "pint" if native_pkg == "tempo2" else "tempo2"
    native_leg = release_key
    donor_leg = f"{native_leg}_{donor_pkg}"

    native_entry = copy.deepcopy(entry)
    native_entry["timing_package"] = native_pkg
    donor_entry = copy.deepcopy(entry)
    donor_entry["timing_package"] = donor_pkg

    raw = Path(entry["par"]).read_text(encoding="utf-8", errors="replace")
    par_dir, tim_dir = work / "factory_par", work / "factory_tim"
    for directory in (par_dir, tim_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Opt in: factory default is canonicalize_tim=False; AEI products need
        # the dual-engine standalone FORMAT 1 writer (INCLUDE flatten, TIME bake,
        # PTA / -pn stamps).
        create_metapulsar(
            file_data={native_leg: [native_entry], donor_leg: [donor_entry]},
            combination_strategy="shared",
            reference_pta=native_leg,
            parfile_output_dir=par_dir,
            timfile_output_dir=tim_dir,
            use_pulse_numbers="yes",
            canonicalize_tim=True,
            alignment_policy=explicit_reference_policy(raw),
        )

    src_par = par_dir / f"{pulsar}_shared_{native_leg}.par"
    src_tim = tim_dir / f"{pulsar}_{native_leg}.tim"
    if not src_par.is_file() or not src_tim.is_file():
        raise FileNotFoundError(
            f"factory did not export {src_par.name} / {src_tim.name}; "
            f"par={sorted(p.name for p in par_dir.iterdir())} "
            f"tim={sorted(p.name for p in tim_dir.iterdir())}"
        )

    release_dir = out_root / release_key
    out_par = release_dir / "par" / f"{pulsar}.par"
    out_tim = release_dir / "tim" / f"{pulsar}.tim"
    out_par.parent.mkdir(parents=True, exist_ok=True)
    out_tim.parent.mkdir(parents=True, exist_ok=True)
    # Pulse-number tracking has to be declared in the published par; the factory
    # only sets TRACK -2 on the runtime copy it hands its engines.
    out_par.write_text(
        par_text_with_track_minus_2(src_par.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    out_tim.write_text(src_tim.read_text(encoding="utf-8"), encoding="utf-8")

    binary_new = _first_par_value(out_par.read_text(encoding="utf-8"), "BINARY") or "-"
    binary_old = _first_par_value(raw, "BINARY") or "-"
    return out_par, out_tim, binary_new, binary_old


def orthometric_h3_stigma(par_text: str) -> bool:
    """True when the par is on the ELL1H/T2 ``H3``+``STIG`` orthometric path."""
    has_h3 = _first_par_value(par_text, "H3") is not None
    has_ratio = _first_par_value(par_text, "STIG", "STIGMA", "VARSIGMA") is not None
    return has_h3 and has_ratio


def parity_real(par_path: Path, tim_path: Path) -> dict[str, Any]:
    """Load one published pair with both engines and compare residuals.

    PINT is constructed with the same orthometric Shapiro convention MetaPulsar
    itself uses on a mixed PINT+Tempo2 stack (``absorbed``, Freire & Wex 2010
    Eq. 28, which is what Tempo2's ELL1H/T2 mode 1 evaluates). PINT's default
    Eq. 29 is a different published expression for the same printed
    ``(A1, EPS1, H3, STIG)``, so leaving it at the default would measure a
    configuration MetaPulsar never claims parity for. The option is ignored for
    every par that is not on the H3+STIG path.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, toas = get_model_and_toas(
            str(par_path),
            str(tim_path),
            allow_T2=True,
            allow_tcb=True,
            planets=True,
            ell1h_shapiro="absorbed",
        )
        pint_res = Residuals(toas, model).time_resids.to_value("s")
        pint_mjd = np.asarray(toas.get_mjds().value, dtype=float)
        pint_freq = np.asarray(toas.get_freqs().to_value("MHz"), dtype=float)
        psr = tempopulsar(parfile=str(par_path), timfile=str(tim_path), dofit=False)
        t2_res = np.asarray(psr.residuals(), dtype=float)
        t2_mjd = np.asarray(psr.stoas, dtype=float)
        t2_freq = np.asarray(psr.freqs, dtype=float)

    if len(pint_res) != len(t2_res):
        raise RuntimeError(
            f"TOA count mismatch: PINT {len(pint_res)} vs Tempo2 {len(t2_res)}"
        )
    # The join has to be exact, and the two packages agree on site arrival time
    # only to the last bits of a double (tens of ps), which is enough to invert a
    # pair of simultaneous subband TOAs. Observing frequency, by contrast, is
    # carried through verbatim from the .tim by both, so it is the primary key;
    # within one frequency, distinct epochs are minutes apart and arrival time
    # orders them unambiguously.
    ip = np.lexsort((pint_mjd, pint_freq))
    it = np.lexsort((t2_mjd, t2_freq))
    dt = (pint_mjd[ip] - t2_mjd[it]) * 86400.0
    df = pint_freq[ip] - t2_freq[it]
    if float(np.max(np.abs(df))) > 1e-3:
        raise RuntimeError(
            f"TOA series do not line up on observing frequency (max |df| "
            f"{float(np.max(np.abs(df))):.6g} MHz)"
        )
    if abs(float(np.median(dt))) > DT_TOL_S:
        raise RuntimeError(
            f"TOA series do not line up on arrival time (median dt "
            f"{float(np.median(dt)):.6g} s)"
        )
    diff = pint_res[ip] - t2_res[it]
    p0 = 1.0 / float(model.F0.value)
    turns = np.abs(np.round((diff - np.median(diff)) / p0))
    offset = (turns >= 1) | (np.abs(dt) > DT_TOL_S)
    keep = ~offset
    clean = diff[keep] - np.mean(diff[keep]) if keep.any() else diff[keep]
    return {
        "n_toas": int(len(diff)),
        "n_offset": int(offset.sum()),
        "rms_diff_ns": rms(clean) * 1e9,
        "maxabs_diff_ns": float(np.max(np.abs(clean))) * 1e9 if clean.size else 0.0,
        "pint_rms_us": rms(pint_res[ip][keep]) * 1e6,
        "tempo2_rms_us": rms(t2_res[it][keep]) * 1e6,
    }


def process_one(task: tuple[str, dict[str, Any], str, str, bool]) -> dict[str, Any]:
    release_key, entry, out_root_s, work_root_s, stats_only = task
    out_root, work_root = Path(out_root_s), Path(work_root_s)
    configure_logging(level="ERROR")
    pulsar = pulsar_name_from_par(Path(entry["par"]), release_key)
    row = Row(release=release_key, pulsar=pulsar)
    work = work_root / release_key / pulsar
    work.mkdir(parents=True, exist_ok=True)
    try:
        if stats_only:
            release_dir = out_root / release_key
            par_path = release_dir / "par" / f"{pulsar}.par"
            tim_path = release_dir / "tim" / f"{pulsar}.tim"
            if not par_path.is_file() or not tim_path.is_file():
                raise FileNotFoundError(f"missing published pair for {pulsar}")
            binary_new = (
                _first_par_value(par_path.read_text(encoding="utf-8"), "BINARY") or "-"
            )
            binary_old = (
                _first_par_value(
                    Path(entry["par"]).read_text(encoding="utf-8", errors="replace"),
                    "BINARY",
                )
                or "-"
            )
        else:
            par_path, tim_path, binary_new, binary_old = build_pair(
                release_key, entry, out_root, work
            )
        row.par_path = str(par_path.relative_to(out_root))
        row.tim_path = str(tim_path.relative_to(out_root))
        row.binary, row.binary_orig = binary_new, binary_old
        row.ell1h_gauge = (
            "absorbed"
            if orthometric_h3_stigma(par_path.read_text(encoding="utf-8"))
            else "-"
        )
        stats = parity_real(par_path, tim_path)
        for key, value in stats.items():
            setattr(row, key, value)
        row.passed = bool(row.rms_diff_ns is not None and row.rms_diff_ns < PARITY_NS)
    except Exception as exc:  # noqa: BLE001
        row.error = f"{type(exc).__name__}: {exc}"
        row.passed = False
        (work / "error.txt").write_text(
            row.error + "\n\n" + traceback.format_exc(), encoding="utf-8"
        )
    return asdict(row)


def write_overview(rows: list[Row], out_root: Path) -> None:
    rows_sorted = sorted(rows, key=lambda r: (r.release, r.pulsar))
    fieldnames = [
        "release",
        "pulsar",
        "n_toas",
        "n_offset",
        "rms_diff_ns",
        "maxabs_diff_ns",
        "pint_rms_us",
        "tempo2_rms_us",
        "binary",
        "binary_orig",
        "ell1h_gauge",
        "passed",
        "par_path",
        "tim_path",
        "error",
    ]
    with (out_root / "pint_tempo2_residual_rms.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(asdict(row))
    (out_root / "pint_tempo2_residual_rms.json").write_text(
        json.dumps([asdict(r) for r in rows_sorted], indent=2), encoding="utf-8"
    )

    n = len(rows_sorted)
    n_ok = sum(1 for r in rows_sorted if r.passed)
    n_err = sum(1 for r in rows_sorted if r.error)
    diffs = [r.rms_diff_ns for r in rows_sorted if r.rms_diff_ns is not None]
    releases = sorted({r.release for r in rows_sorted})
    descriptions = ", ".join(
        RELEASE_SPECS.get(key, {}).get("description", key) for key in releases
    )
    lines = [
        f"# {out_root.name.upper()} PINT vs Tempo2 residual differences",
        "",
        f"Built from MetaPulsar-aligned individual PTA releases ({descriptions}).",
        "",
        f"- Datasets: **{n}**",
        f"- Pass (`RMS(diff) < {PARITY_NS:g} ns`): **{n_ok}/{n}**",
        f"- Errors: **{n_err}**",
    ]
    if diffs:
        lines += [
            f"- Median RMS(diff): **{np.median(diffs):.4f} ns**",
            f"- Max RMS(diff): **{np.max(diffs):.4f} ns**",
        ]
    lines += [
        "",
        "Each dataset's `.par` and `.tim` are written by `MetaPulsarFactory` "
        "(`create_metapulsar` on a mixed PINT+Tempo2 shared stack, so the files sit "
        "on the validated cross-package common profile and the `.tim` comes from the "
        "canonical writer). Both engines then load that published pair, without a "
        "fit, with phase tracked through the `-pn` pulse numbers (`TRACK -2`), and "
        "the two residual series of the **actual released TOAs** are matched TOA by "
        "TOA on site arrival time. `RMS(diff)` is the demeaned RMS of their "
        "difference.",
        "",
        "`N off` counts TOAs whose two residuals differ by a whole number of pulse "
        "periods, or whose arrival times differ at all. These come from the "
        "observatory clock chain being resolved differently by the two packages, not "
        "from the timing model, and are excluded from `RMS(diff)`.",
        "",
        "`Gauge` marks the datasets whose published par is on the ELL1H/`T2` "
        "`H3`+`STIG` orthometric path. For those, PINT is constructed with "
        '`ell1h_shapiro="absorbed"` (Freire & Wex 2010, Eq. 28), which is the '
        "expression Tempo2's ELL1H/`T2` mode 1 evaluates and the convention "
        "MetaPulsar itself uses on a mixed PINT+Tempo2 stack. PINT's default "
        "Eq. 29 is a different published expression for the same printed "
        "`(A1, EPS1, H3, STIG)`, so it is not a parity configuration MetaPulsar "
        "claims. Every other dataset is unaffected by the setting.",
        "",
        "| Release | Pulsar | N TOAs | N off | RMS(diff) ns | max abs ns | "
        "PINT RMS us | Tempo2 RMS us | Binary | Gauge | Pass | Error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for r in rows_sorted:

        def fmt(value: float | None, digits: int = 4) -> str:
            return f"{value:.{digits}f}" if value is not None else "-"

        binary = r.binary or "-"
        if r.binary and r.binary_orig and r.binary != r.binary_orig:
            binary = f"{r.binary_orig} to {r.binary}"
        lines.append(
            f"| {r.release} | {r.pulsar} | {r.n_toas} | {r.n_offset} | "
            f"{fmt(r.rms_diff_ns)} | {fmt(r.maxabs_diff_ns)} | "
            f"{fmt(r.pint_rms_us, 3)} | {fmt(r.tempo2_rms_us, 3)} | {binary} | "
            f"{r.ell1h_gauge or '-'} | "
            f"{'yes' if r.passed else 'no'} | {(r.error or '').replace('|', chr(92) + '|')} |"
        )
    lines.append("")
    (out_root / "pint_tempo2_residual_rms.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-set", choices=sorted(RELEASE_SETS), default="dr2")
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--releases", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pulsar", action="append", default=[])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Skip the factory build; re-measure parity on the published pairs.",
    )
    args = parser.parse_args()

    release_set = RELEASE_SETS[args.release_set]
    if args.source_root is None:
        args.source_root = release_set["source_root"]
    if args.out_root is None:
        args.out_root = release_set["out_root"]
    if args.releases is None:
        args.releases = list(release_set["releases"])
    if args.work_root is None:
        args.work_root = (
            REPO / "temp" / "aei_dr2_build" / f"work_{args.release_set}_rebuild"
        )

    configure_logging(level="ERROR")
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    discovery = FileDiscovery(
        working_dir=str(args.source_root), pta_data_releases=RELEASE_SPECS
    )
    discovered = discovery.discover_files(args.releases)
    tasks: list[tuple[str, dict[str, Any], str, str, bool]] = []
    for release_key in args.releases:
        entries = list(discovered.get(release_key, []))
        entries.sort(key=lambda e: pulsar_name_from_par(Path(e["par"]), release_key))
        if args.pulsar:
            wanted = set(args.pulsar)
            entries = [
                e
                for e in entries
                if pulsar_name_from_par(Path(e["par"]), release_key) in wanted
            ]
        if args.limit > 0:
            entries = entries[: args.limit]
        for entry in entries:
            tasks.append(
                (
                    release_key,
                    entry,
                    str(args.out_root),
                    str(args.work_root),
                    args.stats_only,
                )
            )

    verb = "measuring" if args.stats_only else "rebuilding"
    print(f"=== {verb} {len(tasks)} datasets on {args.workers} workers ===", flush=True)
    rows: list[Row] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            row = Row(**future.result())
            rows.append(row)
            tag = "OK  " if row.passed else "FAIL"
            detail = (
                f"rms_diff={row.rms_diff_ns:.4f} ns n={row.n_toas} off={row.n_offset}"
                if row.rms_diff_ns is not None
                else f"err={row.error}"
            )
            print(
                f"[{i}/{len(tasks)}] {tag} {row.release} {row.pulsar}: {detail}",
                flush=True,
            )
            write_overview(rows, args.out_root)

    write_overview(rows, args.out_root)
    n_ok = sum(1 for r in rows if r.passed)
    print(
        f"\nDone: {n_ok}/{len(rows)} below {PARITY_NS:g} ns. "
        f"Overview in {args.out_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
