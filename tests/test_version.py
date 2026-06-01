"""Ensure package __version__ stays aligned with installed distribution metadata."""

from importlib.metadata import version

import metapulsar


def test_version_matches_metadata():
    assert metapulsar.__version__ == version("metapulsar")
