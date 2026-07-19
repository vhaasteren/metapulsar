"""Deprecation-alias behavior for combination_strategy values.

Canonical values are "shared"/"per_pta"; the legacy "consistent"/"composite"
spellings are still accepted but emit a DeprecationWarning and normalize to the
canonical spelling.
"""

import pytest

from metapulsar.metapulsar import normalize_combination_strategy


def test_canonical_values_pass_through_without_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        assert normalize_combination_strategy("shared") == "shared"
        assert normalize_combination_strategy("per_pta") == "per_pta"


def test_consistent_alias_warns_and_maps_to_shared():
    with pytest.warns(DeprecationWarning, match="consistent.*deprecated"):
        assert normalize_combination_strategy("consistent") == "shared"


def test_composite_alias_warns_and_maps_to_per_pta():
    with pytest.warns(DeprecationWarning, match="composite.*deprecated"):
        assert normalize_combination_strategy("composite") == "per_pta"


def test_unknown_value_raises():
    with pytest.raises(ValueError, match="combination_strategy must be one of"):
        normalize_combination_strategy("nonsense")
