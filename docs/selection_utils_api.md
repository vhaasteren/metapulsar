# Selection Utilities API Documentation

This document provides comprehensive API documentation for the staggered selection utilities in MetaPulsar.

## Table of Contents

- [Overview](#overview)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Enterprise Integration](#enterprise-integration)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Overview

The `selection_utils` module provides a well-documented API for creating Enterprise-compatible selection functions with hierarchical flag fallback, frequency filtering, and full type hints.

### Key Features

- **Hierarchical Flag Selection**: Support for staggered selection with automatic fallback
- **Single Flag Selection**: Simple flag-based selection for basic use cases
- **Frequency Filtering**: Optional frequency band filtering
- **Enterprise Compatibility**: Full compatibility with `enterprise.signals.selections.Selection`
- **Type Safety**: Complete type hints for better IDE support
- **Error Handling**: Graceful handling of edge cases

## API Reference

### `create_staggered_selection`

```python
def create_staggered_selection(
    name: str,
    flag_criteria: Dict[Union[str, Tuple[str, ...]], Optional[str]] = None,
    freq_range: Optional[Tuple[float, float]] = None,
) -> Callable
```

Creates a staggered selection function for Enterprise.

#### Parameters

**`name`** : `str`
- Base name for the selection (e.g., "efac", "ecorr", "band")
- Used as prefix for all generated selection names
- Example: `"efac"` → `"efac_ASP_430"`, `"efac_ASP_800"`

**`flag_criteria`** : `Dict[Union[str, Tuple[str, ...]], Optional[str]]`, optional
- Dictionary mapping flag specifications to target values
- **Key types**:
  - `str`: Single flag name (e.g., `"group"`)
  - `Tuple[str, ...]`: Staggered flags for fallback (e.g., `("group", "f")`)
- **Value types**:
  - `None`: Select all unique values (excluding empty strings)
  - `str`: Select specific value only
- **Examples**:
  - `{"group": None}` - Use all values of "group" flag
  - `{"group": "ASP_430"}` - Use only "ASP_430" value of "group" flag
  - `{("group", "f"): None}` - Use "group" if available, fallback to "f"
  - `{("group", "f"): "ASP_430"}` - Use "group" if available and matches "ASP_430", fallback to "f"

**`freq_range`** : `Tuple[float, float]`, optional
- Optional frequency range for filtering (low_freq, high_freq)
- Only frequencies within this range will be selected
- Range is inclusive of low_freq, exclusive of high_freq: `[low_freq, high_freq)`
- Example: `(400, 1000)` selects frequencies 400-999.999 MHz

#### Returns

**`Callable`**
- Selection function compatible with `enterprise.signals.selections.Selection`
- Function signature: `(flags: Dict[str, np.ndarray], freqs: np.ndarray) -> Dict[str, np.ndarray]`
- Returns dictionary mapping selection names to boolean masks

#### Function Signature

The returned selection function has the signature:

```python
def selection_function(flags: Dict[str, np.ndarray], freqs: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Selection function that works with Enterprise.
    
    Args:
        flags: Dictionary of flag arrays (e.g., {"group": ["ASP_430", "ASP_800"], "B": ["1", "2"]})
        freqs: Array of frequencies
        
    Returns:
        Dictionary mapping selection names to boolean masks
    """
```

### `_create_selections_for_flag`

```python
def _create_selections_for_flag(
    flag_values: np.ndarray,
    target_value: Optional[str],
    base_name: str,
    freq_mask: np.ndarray,
) -> Dict[str, np.ndarray]
```

Internal helper function for creating selection masks for a specific flag.

#### Parameters

**`flag_values`** : `np.ndarray`
- Array of flag values for the current flag

**`target_value`** : `Optional[str]`
- Target value to match (None means all values)

**`base_name`** : `str`
- Base name for the selection

**`freq_mask`** : `np.ndarray`
- Frequency mask to apply

#### Returns

**`Dict[str, np.ndarray]`**
- Dictionary mapping selection names to boolean masks

## Usage Examples

### Basic Flag Selection

```python
from metapulsar.selection_utils import create_staggered_selection

# Simple group-based selection (all values)
group_sel = create_staggered_selection("efac", {"group": None})

# Specific value selection
specific_sel = create_staggered_selection("efac", {"group": "ASP_430"})

# Test with mock data
flags = {"group": np.array(["ASP_430", "ASP_800", "ASP_430"])}
freqs = np.array([100.0, 200.0, 300.0])

result = group_sel(flags, freqs)
# Returns: {"efac_ASP_430": [True, False, True], "efac_ASP_800": [False, True, False]}
```

### Staggered Selection

```python
# Staggered selection with fallback
staggered_sel = create_staggered_selection("ecorr", {("group", "f"): None})

# Test with both flags available (uses 'group')
flags_both = {
    "group": np.array(["ASP_430", "ASP_800"]),
    "f": np.array(["GASP_430", "GASP_800"])
}
result_both = staggered_sel(flags_both, freqs)
# Returns: {"ecorr_ASP_430": [True, False], "ecorr_ASP_800": [False, True]}

# Test with only fallback flag (uses 'f')
flags_fallback = {"f": np.array(["GASP_430", "GASP_800"])}
result_fallback = staggered_sel(flags_fallback, freqs)
# Returns: {"ecorr_GASP_430": [True, False], "ecorr_GASP_800": [False, True]}
```

### Frequency Filtering

```python
# Frequency band selection
band_sel = create_staggered_selection(
    "band", 
    {"group": None}, 
    freq_range=(400, 1000)  # 400-1000 MHz band
)

flags = {"group": np.array(["ASP_430", "ASP_800", "ASP_430"])}
freqs = np.array([100.0, 500.0, 1500.0])  # Only 500.0 is in range

result = band_sel(flags, freqs)
# Returns: {"band_ASP_430": [False, True, False], "band_ASP_800": [False, False, False]}
```

### Multiple Flag Criteria

```python
# Multiple criteria
multi_sel = create_staggered_selection("efac", {
    "pta": "EPTA",  # EPTA-specific
    ("group", "f"): None  # All groups with fallback
})

flags = {
    "pta": np.array(["EPTA", "PPTA", "EPTA"]),
    "group": np.array(["ASP_430", "ASP_800", "ASP_430"]),
    "f": np.array(["GASP_430", "GASP_800", "GASP_430"])
}
freqs = np.array([100.0, 200.0, 300.0])

result = multi_sel(flags, freqs)
# Returns: {
#     "efac_ASP_430": [True, False, True],  # From group flag
#     "efac_ASP_800": [False, False, False]  # From group flag
# }
```

## Enterprise Integration

### Basic Integration

```python
from enterprise.signals.selections import Selection
from metapulsar.selection_utils import create_staggered_selection

# Create selection function
efac_sel = create_staggered_selection("efac", {"group": None})

# Wrap with Enterprise Selection
selection = Selection(efac_sel)

# Create selection instance with pulsar
selection_instance = selection(pulsar)
masks = selection_instance.masks

# Use in Enterprise model
from enterprise.signals import white_signals
white_signal = white_signals.MeasurementNoise(efac=selection)
```

### Parameter Generation

```python
# Generate parameters for Enterprise
params, param_masks = selection_instance("efac", lambda x: f"param_{x}")

# params: {"efac_ASP_430_efac": "param_pulsar_name_efac_ASP_430_efac", ...}
# param_masks: {"efac_ASP_430_efac": [True, False, True], ...}
```

### Advanced Integration

```python
# Complex selection with multiple criteria
complex_sel = create_staggered_selection("efac", {
    ("group", "f"): None,  # Staggered selection
    "pta": "EPTA"  # PTA-specific
})

# Use in Enterprise model
white_signal = white_signals.MeasurementNoise(
    efac=Selection(complex_sel),
    log10_efac=Uniform(-10, 10)
)
```

## Performance Considerations

### Memory Usage

- Selection functions use numpy arrays for efficient memory usage
- Boolean masks are created on-demand to minimize memory footprint
- Large datasets (>10,000 TOAs) may benefit from chunked processing

### Computational Complexity

- **Time complexity**: O(n) where n is the number of TOAs
- **Space complexity**: O(n × m) where m is the number of unique flag values
- Frequency filtering adds minimal overhead

### Optimization Tips

1. **Pre-filter data**: Apply frequency filtering early if possible
2. **Use specific values**: Specify target values instead of `None` when possible
3. **Minimize flag criteria**: Use only necessary flag criteria
4. **Batch processing**: Process multiple pulsars in batches for large datasets

## Troubleshooting

### Common Issues

#### 1. Empty Selection Results

**Problem**: Selection returns empty dictionary
```python
result = sel_func(flags, freqs)  # Returns: {}
```

**Causes**:
- No flags match the criteria
- All flag values are empty strings
- Frequency range excludes all TOAs

**Solutions**:
- Check flag names and values
- Verify frequency range parameters
- Use `None` for target_value to select all values

#### 2. Missing Flag Values

**Problem**: Expected flag values not found
```python
# Expected "ASP_430" but got different values
result = sel_func({"group": np.array(["ASP_800", "ASP_1400"])}, freqs)
```

**Solutions**:
- Check actual flag values in your data
- Use `None` for target_value to select all available values
- Verify flag names are correct

#### 3. Enterprise Integration Issues

**Problem**: Selection doesn't work with Enterprise
```python
# TypeError: selection_function() missing 1 required positional argument
```

**Solutions**:
- Ensure function signature is `(flags, freqs)`
- Check that pulsar has required attributes
- Verify Enterprise version compatibility

### Debugging Tips

1. **Test selection function directly**:
   ```python
   result = sel_func(flags, freqs)
   print(f"Selection keys: {list(result.keys())}")
   print(f"Mask shapes: {[mask.shape for mask in result.values()]}")
   ```

2. **Check flag values**:
   ```python
   print(f"Available flags: {list(flags.keys())}")
   for flag_name, values in flags.items():
       print(f"{flag_name}: {np.unique(values)}")
   ```

3. **Verify frequency range**:
   ```python
   print(f"Frequency range: {freqs.min():.1f} - {freqs.max():.1f}")
   print(f"TOAs in range: {((freqs >= 400) & (freqs < 1000)).sum()}")
   ```

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `TypeError: create_staggered_selection() missing 1 required positional argument` | Missing required arguments | Provide `name` parameter |
| `KeyError: 'flag_name'` | Flag not found in flags dictionary | Check flag names in your data |
| `ValueError: operands could not be broadcast together` | Array shape mismatch | Verify flag arrays have same length as freqs |
| `IndexError: boolean index did not match indexed array` | Mask length mismatch | Check that all arrays have same length |

## Contributing

When contributing to the selection utilities:

1. **Add tests** for new functionality
2. **Update documentation** for API changes
3. **Follow type hints** for better IDE support
4. **Test with Enterprise** to ensure compatibility

## License

This module is part of MetaPulsar and follows the same license terms.
