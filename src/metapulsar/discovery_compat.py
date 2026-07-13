"""Discovery compatibility shims for MetaPulsar examples.

Host-side monkeypatches only — nothing here is part of the timing package or
its future ``nltiming`` split. Each shim should eventually become an upstream
Discovery fix and then be deleted.
"""

from __future__ import annotations


def apply_discovery_compat_patches() -> None:
    """Apply small Discovery compatibility shims required by MetaPulsar examples.

    Currently replaces ``discovery.matrix.make_uind`` with a version that
    handles empty ECORR bases (zero columns) and variable per-epoch TOA counts.
    """
    import numpy as np

    import discovery.matrix as ds_matrix

    def make_uind(U):
        u = np.asarray(U)
        if u.shape[1] == 0:
            return np.zeros((0, 1), dtype=int)
        max_count = int(np.max(np.sum(u, axis=0)))
        uind = np.zeros((u.shape[1], max_count + 1), dtype=int)
        for i in range(u.shape[1]):
            ind = np.where(u[:, i])[0]
            uind[i, : len(ind)] = ind + 1
        return uind

    ds_matrix.make_uind = make_uind
