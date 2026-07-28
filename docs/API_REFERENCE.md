# MetaPulsar API Reference

Complete API documentation for the MetaPulsar package.

## Table of Contents

- [Core Classes](#core-classes)
- [Convenience Functions](#convenience-functions)
- [File Discovery](#file-discovery)
- [Layout Discovery](#layout-discovery)
- [Parameter Management](#parameter-management)
- [Selection Utilities](#selection-utilities)
- [Utilities](#utilities)
- [Exceptions](#exceptions)

## Core Classes

### MetaPulsar

The main class for combining pulsar timing data from multiple PTA collaborations.

```python
class MetaPulsar:
    """Elegant composite pulsar for multi-PTA data combination.
    
    This class combines pulsar timing data from multiple PTA collaborations
    into a unified object suitable for gravitational wave detection analysis.
    Implements the Enterprise/Discovery pulsar surface by duck typing.
    
    Supports two combination strategies:
    - "consistent": Astrophysical consistency (modifies par files for consistency)
    - "composite": Multi-PTA composition (preserves original parameters)
    """
    
    def __init__(
        self,
        pulsars,
        *,
        combination_strategy="consistent",
        combine_components: List[str] = [
            "astrometry",
            "spindown", 
            "binary",
            "dispersion",
        ],
        add_dm_derivatives: bool = True,
        sort=False,
    ):
        """Initialize MetaPulsar.
        
        Args:
            pulsars: Dict mapping PTA names to pulsar data:
                - PINT: {pta: (pint_model, pint_toas)}
                - Tempo2: {pta: tempo2_psr}
            combination_strategy: Strategy for combining PTAs:
                - "consistent": Astrophysical consistency (modifies par files for consistency)
                - "composite": Multi-PTA composition (preserves original parameters)
            combine_components: List of components to make consistent (consistent strategy only):
                - "astrometry": Position and proper motion parameters
                - "spindown": Spin frequency and derivatives
                - "binary": Binary orbital parameters
                - "dispersion": Dispersion measure parameters
                Defaults to all components
            add_dm_derivatives: Whether to ensure DM1, DM2 are present (consistent strategy only)
            sort: Whether to sort data by time
        """
```

### Interactive timing evaluation

`MetaPulsar.timing()` opens nltiming's immutable, engine-independent evaluator.

Locked linear vocabulary (see [`design_matrix_terminology.md`](design_matrix_terminology.md)):
`design_matrix` / \(M\) = delay tangent (fitter sign); `residual_jacobian` /
\(J=-M\) from the gauge-free residual. The old `waveform_jacobian` noun is
deleted. The evaluator’s `jacobian(...)` is a **residual Jacobian**, not a
design matrix.

```python
timing = metapulsar.timing(
    engines={"pint": "jug", "tempo2": "jug"},
    derivative_method="autodiff",
)

timing.parameters["F0"]       # reference, units, uncertainty, PTA aliases
evaluation = timing.evaluate({"F0": 1e-10}, frame="delta")
evaluation.residual_delta     # r(theta) - r(theta_ref), seconds
evaluation.residuals          # absolute residuals, seconds
evaluation.delay              # -residual_delta, likelihood convention

scan = timing.scan("TASC", [-0.5, 0.0, 0.5], scale="PB")
jacobian = timing.jacobian(method="autodiff")  # residual Jacobian J
fit = timing.fit(["F0", "F1"])
```

The evaluator and its result objects live in the standalone `nltiming`
package. MetaPulsar owns only host construction, multi-PTA parameter/session
mapping, and the convenience constructor.

### MetaPulsarFactory

Factory class for creating MetaPulsar objects.

```python
class MetaPulsarFactory:
    """Factory for creating MetaPulsar objects with various combination strategies."""
    
    def create_metapulsar(
        self,
        file_data: Dict[str, List[Dict[str, Any]]],
        combination_strategy: str = "consistent",
        reference_pta: str = None,
        combine_components: List[str] = DEFAULT_COMBINE_COMPONENTS,
        add_dm_derivatives: bool = True,
        parfile_output_dir: Path = None,
    ) -> MetaPulsar:
        """Create MetaPulsar using specified combination strategy."""
    
    def create_all_metapulsars(
        self,
        file_data: Dict[str, List[Dict[str, Any]]],
        combination_strategy: str = "consistent",
        reference_pta: str = None,
        combine_components: List[str] = DEFAULT_COMBINE_COMPONENTS,
        add_dm_derivatives: bool = True,
        parfile_output_dir: Path = None,
    ) -> Dict[str, MetaPulsar]:
        """Create MetaPulsars for all pulsars in file_data."""
```

## Convenience Functions

### create_metapulsar

Create a single MetaPulsar object.

```python
def create_metapulsar(
    file_data: Dict[str, List[Dict[str, Any]]],
    combination_strategy: str = "consistent",
    reference_pta: str = None,
    combine_components: List[str] = DEFAULT_COMBINE_COMPONENTS,
    add_dm_derivatives: bool = True,
    parfile_output_dir: Path = None,
) -> MetaPulsar:
    """Create MetaPulsar using specified combination strategy.

    Args:
        file_data: File data from FileDiscoveryService (should contain data for single pulsar only)
        combination_strategy: Strategy for combining PTAs:
            - "consistent": Astrophysical consistency (modifies par files for consistency, the default)
            - "composite": Multi-PTA composition (preserves original parameters, Borg/FrankenStat methods)
        reference_pta: PTA to use as reference (for consistent strategy). If None, uses first PTA in file_data.
        combine_components: List of components to make consistent (for consistent strategy).
            Defaults to all components: ["astrometry", "spindown", "binary", "dispersion"]
        add_dm_derivatives: Whether to ensure DM1, DM2 are present in all par files (for consistent strategy)
        parfile_output_dir: Directory to save consistent par files (for consistent strategy only).
            If None, par files are not saved to disk.

    Returns:
        MetaPulsar object

    Raises:
        ValueError: If no files found, multiple pulsars detected, or invalid parameters
        RuntimeError: If Enterprise Pulsar creation fails
    """
```

### create_all_metapulsars

Create MetaPulsar objects for multiple pulsars.

```python
def create_all_metapulsars(
    file_data: Dict[str, List[Dict[str, Any]]],
    combination_strategy: str = "consistent",
    reference_pta: str = None,
    combine_components: List[str] = DEFAULT_COMBINE_COMPONENTS,
    add_dm_derivatives: bool = True,
    parfile_output_dir: Path = None,
) -> Dict[str, MetaPulsar]:
    """Create MetaPulsars for all pulsars in file_data.

    Args:
        file_data: File data from FileDiscoveryService (per data release)
        combination_strategy: Strategy for combining PTAs
        reference_pta: PTA to use as reference for all pulsars. If None, auto-selects by timespan.
        combine_components: List of components to make consistent
        add_dm_derivatives: Whether to ensure DM1, DM2 are present
        parfile_output_dir: Directory to save consistent par files (for consistent strategy only).
            If None, par files are not saved to disk. Creates subdirectories for each pulsar.

    Returns:
        Dictionary mapping pulsar names to MetaPulsar objects
    """
```

### pta_summary

Print summary of discovered PTA data.

```python
def pta_summary(file_data: Dict[str, List[Dict[str, Any]]]) -> None:
    """Print summary of discovered PTA data.
    
    Args:
        file_data: File data from FileDiscoveryService
    """
```

### reorder_ptas_for_pulsar

Reorder PTAs for a specific pulsar to put the reference PTA first.

```python
def reorder_ptas_for_pulsar(
    pulsar_file_data: Dict[str, List[Dict[str, Any]]],
    reference_pta: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Reorder PTAs for a specific pulsar to put specified PTA first as reference.
    
    Args:
        pulsar_file_data: PTA data for a specific pulsar
        reference_pta: PTA name to use as reference (will be first in dict)

    Returns:
        Reordered pulsar data with reference_pta first
    """
```

## File Discovery

### discover_files

Discover PTA data files using layout patterns.

```python
def discover_files(
    pta_data_releases: Dict,
    working_dir: str = None,
    data_release_names: Union[str, List[str], None] = None,
    verbose: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Convenience function for file discovery.

    Args:
        pta_data_releases: Dictionary of data releases (typically from layout discovery).
        working_dir: Working directory for resolving relative paths. If None, uses current working directory.
        data_release_names: Single data release name, list of data release names, or None to search all.
        verbose: If True, prints summary of found files to console.

    Returns:
        Dictionary mapping data release names to lists of file dictionaries
    """
```

### get_pulsar_names_from_file_data

Extract B-preferred catalog pulsar names from file data (10″ J2000 position grouping).

```python
def get_pulsar_names_from_file_data(
    file_data: Dict[str, List[Dict[str, Any]]]
) -> List[str]:
    """Extract pulsar names from file data using position-based catalog identity.
    
    Args:
        file_data: File data from FileDiscoveryService
        
    Returns:
        List of catalog names (B-preferred when parfiles use B-names, e.g. 'B1855+09')
    """
```

### filter_file_data_by_pulsars

Filter file data to specific pulsars by catalog or path alias.

```python
def filter_file_data_by_pulsars(
    file_data: Dict[str, List[Dict[str, Any]]],
    pulsar_names: Union[str, List[str]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Filter file data to specific pulsars.
    
    Args:
        file_data: File data from FileDiscoveryService
        pulsar_names: Catalog names (PSRJ/PSR/PSRB) or path-derived aliases
        
    Returns:
        Filtered file data containing only specified pulsars
    """
```

### discover_pulsars_by_position

Group discovered file pairs by on-sky position (default 10″ at J2000) and catalog identity.

```python
def discover_pulsars_by_position(
    file_data: Dict[str, List[Dict[str, Any]]],
    match_tol_arcsec: float = 10.0,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Returns dict keyed by B-preferred catalog name → PTA → file records."""
```

## Layout Discovery

### discover_layout

Discover PTA data release directory structure.

```python
def discover_layout(
    working_dir: str = None,
    verbose: bool = True,
    excluded_dirs: Sequence[str] = DEFAULT_EXCLUDED_DIRS,
    name: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Convenience function for layout discovery.

    Args:
        working_dir: Directory to analyze. If None, uses current directory.
        verbose: If True, prints discovered layout to console.
        excluded_dirs: List of directory names to exclude from analysis.
        name: Optional name to use for the returned layout key.

    Returns:
        Dictionary of data release configurations
    """
```

### combine_layouts

Combine multiple layout discoveries.

```python
def combine_layouts(
    *layouts: Dict[str, Dict[str, Any]], 
    include_defaults: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Combine multiple layout discoveries.

    Args:
        *layouts: Variable number of layout dictionaries from discover_layout()
        include_defaults: If True, includes default PTA_DATA_RELEASES in the combination

    Returns:
        Combined dictionary with all data releases

    Example:
        layout1 = discover_layout("../../data/ipta-dr2/EPTA_v2.2")
        layout2 = discover_layout("../../data/ipta-dr2/NANOGrav_9y")
        layout3 = discover_layout("../../data/ipta-dr2/PPTA_dr1dr2")
        combined = combine_layouts(layout1, layout2, layout3, include_defaults=True)
    """
```

## Parameter Management

### ParameterManager

Manages parameter consistency across PTAs.

```python
class ParameterManager:
    """Manages parameter consistency and mapping across PTAs."""
    
    def __init__(
        self,
        file_data: Dict[str, Dict[str, Any]],
        combine_components: List[str] = ["astrometry", "spindown", "binary", "dispersion"],
        add_dm_derivatives: bool = True,
        output_dir: Path = None,
        pulsar_name: str = None,
    ):
        """Initialize parameter manager with file data and configuration."""
```

### ParameterMapping

Maps parameters between different PTA formats.

```python
class ParameterMapping:
    """Maps parameters between different PTA formats."""
```

## Selection Utilities

### create_staggered_selection

Create Enterprise-compatible selection functions.

```python
def create_staggered_selection(
    name: str,
    flag_criteria: Dict[Union[str, Tuple[str, ...]], Optional[str]] = None,
    freq_range: Optional[Tuple[float, float]] = None,
) -> Callable:
    """Create Enterprise-compatible selection function with hierarchical fallback.

    Args:
        name: Base name for the selection (e.g., 'efac', 'ecorr')
        flag_criteria: Mapping from flag(s) to target value or None (for all values)
        freq_range: Optional frequency range tuple (low, high) in MHz

    Returns:
        Selection function compatible with Enterprise Selection class

    Example:
        # Simple group-based selection
        group_sel = create_staggered_selection("efac", {"group": None})
        selection = Selection(group_sel)
        
        # Staggered selection with fallback
        staggered_sel = create_staggered_selection("ecorr", {("group", "f"): None})
        selection = Selection(staggered_sel)
    """
```

## Exceptions

### ParameterInconsistencyError

Raised when parameter inconsistencies are detected.

```python
class ParameterInconsistencyError(Exception):
    """Raised when parameter inconsistencies are detected across PTAs."""
```

### PINTDiscoveryError

Raised when PINT model discovery fails.

```python
class PINTDiscoveryError(Exception):
    """Raised when PINT model discovery fails."""
```

## Constants

### PTA_DATA_RELEASES

Predefined PTA data release patterns.

```python
PTA_DATA_RELEASES: Dict[str, Dict[str, Any]]
```

Contains regex patterns and directory structures for:
- EPTA DR1 v2.2
- EPTA DR2
- InPTA DR1
- MPTA DR1
- NANOGrav 9-year
- NANOGrav 12-year
- NANOGrav 15-year
- PPTA DR1+DR2

## Data Structures

### File Data Format

The standard file data format used throughout MetaPulsar:

```python
file_data = {
    "pta_name": [
        {
            "par": "path/to/file.par",
            "tim": "path/to/file.tim", 
            "timing_package": "tempo2",  # or "pint"
            "parfile_content": "par file content as string",  # optional
        }
    ]
}
```

### Layout Format

The layout format returned by `discover_layout`:

```python
layout = {
    "layout_name": {
        "base_path": "/path/to/data",
        "par_pattern": "regex pattern for .par files",
        "tim_pattern": "regex pattern for .tim files",
        "discovery_confidence": 0.95,
        # ... other discovery metadata
    }
}
```

## Utilities

### TimFileAnalyzer

Fast analyzer for TIM files to compute timespan and TOA counts without constructing full TOA objects.

```python
class TimFileAnalyzer:
    def calculate_timespan(self, tim_file_path: Path) -> float:
        """Calculate timespan in days from a TIM file."""

    def count_toas(self, tim_file_path: Path) -> int:
        """Count number of TOAs in a TIM file."""

    def get_timespan_and_count(self, tim_file_path: Path) -> Tuple[float, int]:
        """Return (timespan_in_days, toa_count) efficiently."""
```

## Usage Examples

### Basic Workflow

```python
from metapulsar import (
    discover_layout, combine_layouts, discover_files,
    get_pulsar_names_from_file_data, filter_file_data_by_pulsars,
    create_metapulsar, create_all_metapulsars
)

# Discover layouts
epta_layout = discover_layout('data/ipta-dr2/EPTA_v2.2')
nanograv_layout = discover_layout('data/ipta-dr2/NANOGrav_9y')

# Combine layouts
combined_layout = combine_layouts(epta_layout, nanograv_layout)

# Discover files
file_data = discover_files(combined_layout)

# Filter to specific pulsars
pulsar_names = get_pulsar_names_from_file_data(file_data)
filtered_data = filter_file_data_by_pulsars(file_data, ['J0613-0200'])

# Create MetaPulsar
metapulsar = create_metapulsar(filtered_data)
```

### Batch Processing

```python
# Create MetaPulsars for all discovered pulsars
metapulsars = create_all_metapulsars(file_data, reference_pta=None)
```

### Selection Functions

```python
from metapulsar import create_staggered_selection
from enterprise.signals.selections import Selection

# Create selection function
efac_sel = create_staggered_selection("efac", {"group": None})
selection = Selection(efac_sel)
```
