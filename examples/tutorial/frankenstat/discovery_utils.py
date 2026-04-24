#!/usr/bin/env python3

from pathlib import Path

import discovery as ds


def read_pulsar_feathers(data_dir: Path, prefix: str = "") -> list[ds.Pulsar]:
    """Read Pulsar data from feather files in the specified directory and with a given prefix.

    Parameters
    ----------
    data_dir : Path
        Directory containing subdirectory with feather files.
    prefix : str, optional
        Prefix of subdirectory within `data_dir` to search for feather files (default is no prefix).

    Returns
    -------
    list[ds.Pulsar]
        A list of Pulsar objects read from the feather files.

    """
    search_string = f"{prefix}*-[JB]*.feather" if prefix != "" else "[JB]*.feather"
    return [
        ds.Pulsar.read_feather(psrfile)
        for psrfile in sorted(data_dir.glob(search_string))
    ]
