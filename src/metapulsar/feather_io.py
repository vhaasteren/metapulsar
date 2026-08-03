"""Enterprise/Discovery-compatible pulsar feather I/O.

Wire format matches enterprise.pulsar.FeatherPulsar and discovery.pulsar.Pulsar.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

# Enterprise superset (Discovery column list is this minus "telescope").
COLUMNS: tuple[str, ...] = (
    "toas",
    "stoas",
    "toaerrs",
    "residuals",
    "freqs",
    "backend_flags",
    "telescope",
)
VECTOR_COLUMNS: tuple[str, ...] = ("Mmat", "sunssb", "pos_t")
TENSOR_COLUMNS: tuple[str, ...] = ("planetssb",)
METADATA: tuple[str, ...] = (
    "name",
    "dm",
    "dmx",
    "pdist",
    "pos",
    "phi",
    "theta",
    "fitpars",
    "setpars",
    "_pdist",
)


def _to_list(value: Any) -> Any:
    return value.tolist() if isinstance(value, np.ndarray) else value


def save_pulsar_feather(
    psr: Any,
    filename: str | os.PathLike[str],
    noisedict: Mapping[str, Any] | None = None,
) -> None:
    """Write a pulsar duck to feather (Enterprise/Discovery layout)."""
    filename = os.fspath(filename)

    # Match Enterprise: force float64 TOAs on the public sorted view.
    toas = np.asarray(psr.toas, dtype=float)

    pydict: dict[str, Any] = {}
    for name in COLUMNS:
        if name == "toas":
            pydict[name] = toas
        else:
            pydict[name] = np.asarray(getattr(psr, name))

    for name in VECTOR_COLUMNS:
        arr = np.asarray(getattr(psr, name))
        for i in range(arr.shape[1]):
            pydict[f"{name}_{i}"] = arr[:, i]

    for name in TENSOR_COLUMNS:
        arr = np.asarray(getattr(psr, name))
        for i in range(arr.shape[1]):
            for j in range(arr.shape[2]):
                pydict[f"{name}_{i}_{j}"] = arr[:, i, j]

    flags = getattr(psr, "flags", {}) or {}
    for flag, values in flags.items():
        pydict[f"flags_{flag}"] = np.asarray(values)

    meta: dict[str, Any] = {}
    for attr in METADATA:
        if hasattr(psr, attr):
            meta[attr] = _to_list(getattr(psr, attr))

    resolved = getattr(psr, "noisedict", None) if noisedict is None else noisedict
    if resolved:
        pname = str(getattr(psr, "name"))
        meta["noisedict"] = {
            par: val for par, val in resolved.items() if str(par).startswith(pname)
        }

    table = pa.Table.from_pydict(pydict, metadata={"json": json.dumps(meta)})
    feather.write_feather(table, filename)


def read_pulsar_feather(filename: str | os.PathLike[str]) -> SimpleNamespace:
    """Read a pulsar feather into a SimpleNamespace duck (not a MetaPulsar)."""
    filename = os.fspath(filename)
    table = feather.read_table(filename)
    out = SimpleNamespace()

    for name in COLUMNS:
        if name in table.column_names:
            setattr(out, name, table[name].to_numpy())

    for name in VECTOR_COLUMNS:
        cols = [c for c in table.column_names if c.startswith(f"{name}_")]
        if not cols:
            continue
        # Stable numeric order: name_0, name_1, ... (Mmat_10 after Mmat_2).
        cols = sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))
        setattr(
            out,
            name,
            np.array([table[c].to_numpy() for c in cols]).swapaxes(0, 1).copy(),
        )

    for name in TENSOR_COLUMNS:
        # Enterprise/Discovery: group {name}_{i}_{j} by row prefix, then reshape.
        # Empirically: (9,6,n) -> swapaxes(0,2) -> swapaxes(1,2) == (n,9,6).
        prefixed = [c for c in table.column_names if c.startswith(f"{name}_")]
        if not prefixed:
            continue
        rows = sorted({"_".join(c.split("_")[:-1]) for c in prefixed})
        cols = [
            sorted(
                [c for c in prefixed if c.startswith(row + "_")],
                key=lambda c: int(c.rsplit("_", 1)[1]),
            )
            for row in rows
        ]
        setattr(
            out,
            name,
            np.array([[table[c].to_numpy() for c in row] for row in cols])
            .swapaxes(0, 2)
            .swapaxes(1, 2)
            .copy(),
        )

    out.flags = {}
    for col in table.column_names:
        if col.startswith("flags_"):
            flag = col[len("flags_") :]
            out.flags[flag] = table[col].to_numpy().astype("U")

    meta = json.loads(table.schema.metadata[b"json"])
    for attr in METADATA:
        if attr in meta:
            setattr(out, attr, meta[attr])
    if "noisedict" in meta:
        out.noisedict = meta["noisedict"]

    return out
