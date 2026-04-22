# MetaPulsar / FrankenStat tutorial -- NANOGrav Spring 2026

This directory contains the materials for the **Friday 12:00-13:00 EDT, Room 6**
hands-on session at the NANOGrav Spring 2026 meeting. The room is shared by:

- **Rutger van Haasteren** -- *MetaPulsar* (the `consistent` strategy).
- **David Wright** -- *FrankenStat* (the `composite` strategy in `metapulsar`).

One-shot git clone to get started:
```bash
git clone -b nanograv2026-tutorial --recurse-submodules git@github.com:vhaasteren/metapulsar.git
```

The two halves of the session use **different data sources**:

- *FrankenStat* (notebook 01) runs on a small **simulated** PTA generated from
  scratch by Dave's vendored pipeline under [`./frankenstat/`](frankenstat/).
  This notebook does not require the IPTA-DR2 submodule.
- *MetaPulsar* (notebooks 00, 02, 03) runs on the **real IPTA-DR2** release
  that ships as a git submodule under `data/ipta-dr2/`.

## Notebooks

Notebook `00` sets `DATA_ROOT` and a small `PULSAR_SUBSET = ["J1853+1303",
"B1953+29"]` that the later real-data notebooks reuse via `%store`. The two
pulsars were chosen to be 2-PTA (NANOGrav PINT + EPTA Tempo2) with low TOA
counts so the live build stays under a minute on a laptop.
Notebook 01 is use a vendored snapshot of
David Wright's [`frankenstat-paper-1`](https://github.com/davecwright3/frankenstat-paper-1)
repo (see [`frankenstat/README.md`](frankenstat/README.md)). Its outputs land
in `frankenstat/frankenstat_demo_outputs/` (gitignored).

## Installation

The package and its dependencies are pinned in the top-level
[`pyproject.toml`](../../pyproject.toml). For the tutorial, the simplest path is

```bash
pip install metapulsar
# or, from a checkout of this repo:
pip install -e ".[dev,libstempo]"
```

Notebook 3 additionally needs `getdist` and (optionally) `tensiometer`:

```bash
pip install getdist tensiometer
```

If your environment hits the `pkg_resources` import error from
`enterprise-pulsar`, pin setuptools as documented in the top-level
[`README.md`](../../README.md#ci-import-error-with-pkg_resources):

```bash
pip install "setuptools<81"
```

## Data

The notebooks expect the IPTA-DR2 release at `../../data/ipta-dr2/`
(EPTA v2.2, PPTA dr1dr2, NANOGrav 9y). This directory is a **git submodule**
pointing at <https://gitlab.com/IPTA/DR2.git>. On a fresh checkout the
directory is empty until you initialise it:

```bash
git clone --recurse-submodules <metapulsar-url>
# or, if you already cloned without --recurse-submodules:
git submodule update --init data/ipta-dr2
```

The submodule is several hundred MB of `.par` / `.tim` files