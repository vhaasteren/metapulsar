"""Selection utilities for MetaPulsar.

This module provides a modern, well-documented API for creating Enterprise-compatible
selection functions. It replaces the legacy `create_selection_stag` function with
improved functionality, better documentation, and enhanced type safety.

Key Features:
    - Hierarchical flag selection with automatic fallback
    - Single flag selection for simple use cases
    - Frequency band filtering
    - Full Enterprise compatibility
    - Complete type hints for better IDE support
    - Robust error handling

Basic Usage:
    >>> from metapulsar.selection_utils import create_staggered_selection
    >>> from enterprise.signals.selections import Selection
    >>>
    >>> # Simple group-based selection
    >>> group_sel = create_staggered_selection("efac", {"group": None})
    >>> selection = Selection(group_sel)
    >>>
    >>> # Staggered selection with fallback
    >>> staggered_sel = create_staggered_selection("ecorr", {("group", "f"): None})
    >>> selection = Selection(staggered_sel)
    >>>
    >>> # Frequency band selection
    >>> band_sel = create_staggered_selection("band", {"group": None}, freq_range=(400, 1000))
    >>> selection = Selection(band_sel)

Advanced Usage:
    >>> # Multiple criteria
    >>> multi_sel = create_staggered_selection("efac", {
    ...     "pta": "EPTA",  # PTA-specific
    ...     ("group", "f"): None  # All groups with fallback
    ... })
    >>>
    >>> # Complex staggered selection
    >>> complex_sel = create_staggered_selection("efac", {
    ...     ("group", "f", "B"): None,  # Triple fallback
    ...     "pta": "EPTA"  # PTA-specific
    ... })

Migration from Legacy:
    The new API is designed to be a drop-in replacement for the legacy function:

    >>> # Legacy code
    >>> from legacy.metapulsar import create_selection_stag
    >>> legacy_sel = create_selection_stag("efac", {"group": None}, lowfreq=400, highfreq=1000)
    >>>
    >>>
    >>> from metapulsar.selection_utils import create_staggered_selection
    >>> new_sel = create_staggered_selection("efac", {"group": None}, freq_range=(400, 1000))

See Also:
    - `enterprise.signals.selections.Selection`: Enterprise Selection class
    - `examples/staggered_selection_usage.ipynb`: Comprehensive usage examples
    - `docs/selection_utils_api.md`: Complete API documentation
    - `docs/enterprise_integration_guide.md`: Enterprise integration guide

Examples:
    See the examples directory for comprehensive usage examples including:
    - Basic flag selection
    - Staggered selection with fallback
    - Frequency band filtering
    - Enterprise integration
    - Real-world scenarios
    - Performance optimization
"""

from typing import Dict, Union, Tuple, Optional, Callable
import numpy as np


def create_staggered_selection(
    name: str,
    flag_criteria: Dict[Union[str, Tuple[str, ...]], Optional[str]] = None,
    freq_range: Optional[Tuple[float, float]] = None,
) -> Callable:
    """Create a staggered selection function for Enterprise.

    This function creates a selection function that supports hierarchical
    flag selection with fallback mechanisms. It can handle both single flags
    and tuple flags (for staggered selection), and optionally filter by
    frequency range.

    Args:
        name: Base name for the selection (e.g., "efac", "ecorr")
        flag_criteria: Dictionary mapping flag(s) to target values.
            - Key can be a string (single flag) or tuple of strings (staggered flags)
            - Value can be None (all values) or specific string value to match
            - Example: {"group": None} - use all values of "group" flag
            - Example: {("group", "f"): "ASP_430"} - use "group" if available and matches "ASP_430", fallback to "f"
        freq_range: Optional (low_freq, high_freq) tuple for frequency filtering.
            If provided, only frequencies within this range will be selected.

    Returns:
        Selection function compatible with enterprise.signals.selections.Selection.
        The function takes (flags, freqs) as arguments and returns {name: boolean_mask}.

    Examples:
        >>> # Simple group-based selection (all values)
        >>> group_sel = create_staggered_selection("efac", {"group": None})

        >>> # Staggered selection (group fallback to f, all values)
        >>> staggered_sel = create_staggered_selection("ecorr", {("group", "f"): None})

        >>> # Specific flag value selection
        >>> specific_sel = create_staggered_selection("efac", {"group": "ASP_430"})

        >>> # Frequency band selection
        >>> band_sel = create_staggered_selection("band", {"pta": "PPTA"}, freq_range=(0, 960))
    """
    if flag_criteria is None:
        flag_criteria = {}

    def selection_function(flags, freqs) -> Dict[str, np.ndarray]:
        """Selection function that works with Enterprise.

        This function takes (flags, freqs) as arguments. Enterprise's
        selection_func decorator automatically extracts these from the pulsar
        object.

        Args:
            flags: Dictionary of flag arrays (e.g., {"group": ["ASP_430", "ASP_800"], "B": ["1", "2"]})
            freqs: Array of frequencies

        Returns:
            Dictionary mapping selection names to boolean masks
        """
        result = {}

        # Apply frequency filtering if specified
        freq_mask = np.ones(len(freqs), dtype=bool)
        if freq_range is not None:
            low_freq, high_freq = freq_range
            freq_mask = (freqs >= low_freq) & (freqs < high_freq)

        # Process each flag criterion
        for flag_spec, target_value in flag_criteria.items():
            if isinstance(flag_spec, str):
                # Single flag
                flag_name = flag_spec
                if flag_name in flags:
                    flag_values = flags[flag_name]
                    selections = _create_selections_for_flag(
                        flag_values, target_value, name, freq_mask
                    )
                    result.update(selections)

            elif isinstance(flag_spec, tuple):
                selections = _create_staggered_selections_for_flags(
                    flags, flag_spec, target_value, name, freq_mask
                )
                result.update(selections)

        return result

    return selection_function


def _create_staggered_selections_for_flags(
    flags: Dict[str, np.ndarray],
    flag_names: Tuple[str, ...],
    target_value: Optional[str],
    base_name: str,
    freq_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Create staggered selections with per-TOA fallback across flag columns."""
    n_toas = len(freq_mask)
    for flag_name in flag_names:
        if flag_name in flags:
            n_toas = len(flags[flag_name])
            break
    else:
        if not flags:
            return {}
        n_toas = len(next(iter(flags.values())))

    msk_todo = np.ones(n_toas, dtype=bool)
    selections: Dict[str, np.ndarray] = {}

    for flag_name in flag_names:
        if flag_name not in flags:
            continue

        flag_values = flags[flag_name]
        if target_value is None:
            values = set(flag_values) - {""}
        else:
            values = {target_value}

        for value in values:
            msk_flag = (flag_values == value) & freq_mask
            if np.any(msk_todo & msk_flag):
                selections[f"{base_name}_{value}"] = msk_flag
                msk_todo &= ~msk_flag

    return selections


def _create_selections_for_flag(
    flag_values: np.ndarray,
    target_value: Optional[str],
    base_name: str,
    freq_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Create selection masks for a specific flag.

    Args:
        flag_values: Array of flag values
        target_value: Target value to match (None means all values)
        base_name: Base name for the selection
        freq_mask: Frequency mask to apply

    Returns:
        Dictionary mapping selection names to boolean masks
    """
    selections = {}

    if target_value is None:
        # Select all unique values, excluding empty strings
        unique_values = np.unique(flag_values)
        for value in unique_values:
            if value != "":  # Skip empty strings
                selection_name = f"{base_name}_{value}"
                # Create mask for this specific value - only True where flag_values equals this value
                mask = (flag_values == value) & freq_mask
                selections[selection_name] = mask
    else:
        # Select specific value
        selection_name = f"{base_name}_{target_value}"
        # Create mask for this specific value
        mask = (flag_values == target_value) & freq_mask
        selections[selection_name] = mask

    return selections
