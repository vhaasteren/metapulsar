# FrankenStat tutorial subdirectory

This directory contains a **vendored snapshot** of David Wright's FrankenStat
pipeline used by `01_frankenstat_composite.ipynb`. Upstream:

- Repo: <https://github.com/davecwright3/frankenstat-paper-1>
- Vendored at commit `a5c785d` (master, 2026-04)

## Files

| File                   | Source                | Notes                                   |
|------------------------|-----------------------|-----------------------------------------|
| `simulation.py`        | upstream, **patched** | `import schwimmbad`/`cyclopts` wrapped in try/except so the module imports cleanly without those deps. |
| `analysis.py`          | upstream, **patched** | Same `cyclopts` try/except.             |
| `frankenstat.py`       | upstream, verbatim    | FrankenPulsar combination.              |
| `discovery_utils.py`   | upstream, verbatim    | `read_pulsar_feathers`, model helpers.  |
| `add_to_model.py`      | upstream, verbatim    | Noise -> timing-model injection.        |
| `optimization.py`      | upstream, verbatim    | SVI helpers used by `analysis.py`.      |
| `_shims.py`            | tutorial-only         | ~50-line drop-in replacements for `schwimmbad.MultiPool`/`MPIPool` and `cyclopts.App` so this folder is importable in environments that lack those packages. |
| `frankenstat_demo_outputs/` | generated      | Per-realization simulation outputs (gitignored). |

## Layout note

This is a **flat directory, not a Python package** -- it deliberately has no
`__init__.py`. Dave's modules use bare sibling imports (e.g.
`from discovery_utils import read_pulsar_feathers`), and we want to stay
byte-compatible with his upstream layout so re-vendoring is trivial.

`01_frankenstat_composite.ipynb` adds this directory to `sys.path` before
importing.

## Updating the vendored snapshot

```bash
cd examples/tutorial/frankenstat
for f in simulation.py frankenstat.py discovery_utils.py add_to_model.py \
         analysis.py optimization.py; do
    curl -sSLO "https://raw.githubusercontent.com/davecwright3/frankenstat-paper-1/master/$f"
done
# Then re-apply the small try/except shims around `cyclopts` and `schwimmbad`
# in simulation.py and analysis.py (see git history for the exact diff).
```

Dave is free to replace any of these files with new versions; the only
constraint is that `simulation.py`/`analysis.py` keep their fallback-to-shim
import lines, or that `cyclopts`/`schwimmbad` get added to the tutorial env.
