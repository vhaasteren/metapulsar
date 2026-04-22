#!/usr/bin/env python
"""Frankenization utilities for creating composite pulsars from multiple PTAs."""

from copy import deepcopy
from functools import reduce
from pathlib import Path

import discovery as ds
import numpy as np
from scipy.linalg import block_diag


def find_exp(number):
    """Find the order of magnitude for value truncation.

    Parameters
    ----------
    number : float or np.ndarray
        Value(s) for which to find the order of magnitude.

    Returns
    -------
    int or np.ndarray
        Order of magnitude (base 10 exponent minus 1).

    """
    base10 = np.log10(np.abs(number))
    return (np.abs(np.floor(base10)) - 1).astype(int)


def trunc(values, decs=0):
    """Truncate values to a specified number of decimal places.

    Parameters
    ----------
    values : float or np.ndarray
        Value(s) to truncate.
    decs : int, optional
        Number of decimal places to keep, by default 0.

    Returns
    -------
    float or np.ndarray
        Truncated value(s).

    """
    return np.trunc(values * 10.0**decs) / (10.0**decs)


def truncate_vals(vals):
    """Recursively truncate values at first decimal where they disagree.

    Compares values pairwise and truncates at the first decimal place where
    they differ. Useful for merging pulsar positions from multiple PTAs.

    Parameters
    ----------
    vals : list
        List of values (floats or arrays) to compare and truncate.

    Returns
    -------
    float or np.ndarray
        Truncated value representing the consensus across all inputs.

    Notes
    -----
    This function recursively processes the list, removing elements as it
    progresses. The input list is modified in-place.

    """
    diff = np.array(vals[0]) - np.array(vals[1])
    if np.all(diff == 0):
        val_trunc = vals[1]
    else:
        val_trunc = trunc(
            np.array(vals[0]),
            find_exp(diff),
        )
    if len(vals) == 2 or not isinstance(vals, list):
        return val_trunc
    vals.pop(0)
    vals[0] = val_trunc
    return truncate_vals(vals)


def join_noise(dict1, dict2):
    """Merge two noise parameter dictionaries, removing PTA suffixes.

    Combines noise dictionaries from different PTAs by removing PTA-specific
    suffixes (_epta, _ppta, _ng) from parameter names and merging.

    Parameters
    ----------
    dict1 : dict
        First noise parameter dictionary.
    dict2 : dict
        Second noise parameter dictionary.

    Returns
    -------
    dict
        Merged noise dictionary with PTA suffixes removed.

    """
    dict1 = {
        key.replace("_epta", "").replace("_ppta", "").replace("_ng", ""): val
        for key, val in dict1.items()
    }
    dict2 = {
        key.replace("_epta", "").replace("_ppta", "").replace("_ng", ""): val
        for key, val in dict2.items()
    }
    return dict1 | dict2


def frankenize_duplicate_pulsar(
    psrs: list[ds.Pulsar], outdir=Path.cwd(), prefix="franken", noisedict=dict(),
) -> None:
    """Create a composite "FrankenPulsar" by combining observations from multiple PTAs.

    Merges timing data for the same pulsar observed by different pulsar timing
    arrays into a single synthetic pulsar. Combines TOAs, design matrices,
    and metadata while resolving differences in pulsar positions through truncation.

    Parameters
    ----------
    psrs : list[ds.Pulsar]
        List of Pulsar objects representing the same pulsar from different PTAs.
    outdir : Path, optional
        Output directory for the FrankenPulsar feather file, by default current directory.
    prefix : str, optional
        Prefix for the output filename, by default "franken".
    noisedict : dict, optional
        Noise parameter dictionary to use. If empty, merges noise dicts from
        all input pulsars, by default empty dict.

    Notes
    -----
    The function performs the following operations:
    - Concatenates all TOA-related data (residuals, uncertainties, etc.)
    - Creates block-diagonal design matrix to maintain PTA-specific fitting
    - Merges pulsar positions by truncating at first disagreement
    - Combines noise dictionaries from all PTAs
    - Clears pulsar flags (no elegant merge strategy implemented)

    """
    # Stack all 1d things that are toas-sized
    franken_columns = np.vstack(
        [
            np.stack([getattr(psr, param) for param in psr.columns], axis=-1)
            for psr in psrs
        ],
    )

    # Didn't iterate over vector columns because of Mmat
    franken_planetssb = np.vstack([psr.planetssb for psr in psrs])
    franken_sunssb = np.vstack([psr.sunssb for psr in psrs])

    # Block diag so each set of toas only "sees"
    # the appropriate Mmat
    franken_mmat = block_diag(*[psr.Mmat for psr in psrs])

    # Merge position(s) using recursive function
    # Truncates values at first decimal place where they
    # all disagree
    franken_pos = np.array(truncate_vals([psr.pos for psr in psrs]))
    franken_post = np.repeat(franken_pos[None, :], franken_columns.shape[0], axis=0)

    # Phi, theta, and pdist also need to be
    # truncated
    franken_phi = truncate_vals([psr.phi for psr in psrs])
    franken_theta = truncate_vals([psr.theta for psr in psrs])
    franken_pdist = truncate_vals([psr.pdist for psr in psrs])

    # Get all fit_pars, set_pars, dm, and dmx
    franken_fitpars = [psr.fitpars for psr in psrs]
    franken_setpars = [psr.setpars for psr in psrs]
    franken_dm = [getattr(psr, "dm", None) for psr in psrs]
    # franken_dm = [psr.dm for psr in psrs]

    # Might not have dmx
    franken_dmx = [getattr(psr, "dmx", None) for psr in psrs]

    # At this point I make a copy so I can
    # safely edit attrs
    franken_psr = deepcopy(psrs[0])

    # Set franken columns
    for par, val in zip(franken_psr.columns, franken_columns.T, strict=False):
        setattr(franken_psr, par, val)

    # Set franken vectors
    franken_vectors = [franken_mmat, franken_sunssb, franken_post]
    for par, val in zip(franken_psr.vector_columns, franken_vectors, strict=False):
        setattr(franken_psr, par, val)

    # Set franken tensors
    franken_tensors = [
        franken_planetssb,
    ]
    for par, val in zip(franken_psr.tensor_columns, franken_tensors, strict=False):
        setattr(franken_psr, par, val)

    # Set franken metadata
    franken_metadata = [
        franken_psr.name.split("_")[0],  # get rid of the PTA name
        franken_dm,
        franken_dmx,
        franken_pdist,
        franken_pos,
        franken_phi,
        franken_theta,
        franken_fitpars,
        franken_setpars,
    ]
    for par, val in zip(psrs[0].metadata, franken_metadata, strict=False):
        if not hasattr(psrs[0], par):
            continue
        setattr(franken_psr, par, val)

    # Get merged noisedict
    joined_noise = reduce(join_noise, [psr.noisedict for psr in psrs])

    # Clear flags
    # I couldn't find an elegant way to merge these, so I'm just removing
    # them for now.
    franken_psr.flags = []
    # Save to feather file
    franken_psr.save_feather(
        outdir / f"{prefix}-{franken_psr.name}.feather",
        noisedict=noisedict if noisedict else joined_noise,
    )
