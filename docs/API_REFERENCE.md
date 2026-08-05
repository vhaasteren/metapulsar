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
    - "shared": Share selected timing-model parameters across PTAs
    - "per_pta": Preserve per-PTA timing-model parameters
    """
    
    def __init__(
        self,
        pulsars,
        *,
        combination_strategy="shared",
        combine_components: List[str] = [
            "astrometry",
            "spindown", 
            "binary",
            "dispersion",
        ],
        add_dm_derivatives: bool = True,
        exclude_from_shared: List[str] | tuple[str, ...] = ("DM",),
        pta_files: dict[str, dict] | None = None,
        clock_dir: str | Path | None = None,
        sort=False,
    ):
        """Initialize MetaPulsar.
        
        Args:
            pulsars: Dict mapping PTA names to pulsar data:
                - PINT: {pta: (pint_model, pint_toas)}
                - Tempo2: {pta: tempo2_psr}
            combination_strategy: Strategy for combining PTAs:
                - "shared": Share selected timing-model parameters across PTAs
                - "per_pta": Preserve per-PTA timing-model parameters
            combine_components: List of components to share (shared strategy only):
                - "astrometry": Position and proper motion parameters
                - "spindown": Spin frequency and derivatives
                - "binary": Binary orbital parameters
                - "dispersion": Dispersion measure parameters
                Defaults to all components
            add_dm_derivatives: Whether to ensure DM1, DM2 are present (shared strategy only)
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
        combination_strategy: str = "shared",
        reference_pta: str = None,
        combine_components: List[str] = DEFAULT_COMBINE_COMPONENTS,
        add_dm_derivatives: bool = True,
        exclude_from_shared: List[str] | tuple[str, ...] = ("DM",),
        parfile_output_dir: Path = None,
        timfile_output_dir: Path = None,
        use_pulse_numbers: str = "yes",
        clock_dir: Path | str | None = None,
        alignment_policy: AlignmentPolicy | None = None,
        convert_jump_mjd: bool = False,
    ) -> MetaPulsar:
        """Create MetaPulsar using specified combination strategy."""

    def create_all_metapulsars(
        self,
        file_data: Dict[str, List[Dict[str, Any]]],
        combination_strategy: str = "shared",
        reference_pta: str = None,
        combine_components: List[str] = DEFAULT_COMBINE_COMPONENTS,
        add_dm_derivatives: bool = True,
        exclude_from_shared: List[str] | tuple[str, ...] = ("DM",),
        parfile_output_dir: Path = None,
        timfile_output_dir: Path = None,
        use_pulse_numbers: str = "yes",
        clock_dir: Path | str | None = None,
        alignment_policy: AlignmentPolicy | None = None,
        convert_jump_mjd: bool = False,
    ) -> Dict[str, MetaPulsar]:
        """Create MetaPulsars for all pulsars in file_data."""
```

## Convenience Functions

### create_metapulsar

Create a single MetaPulsar object.

```python
def create_metapulsar(
    file_data: Dict[str, List[Dict[str, Any]]],
    combination_strategy: str = "shared",
    reference_pta: str = None,
    combine_components: List[str] = DEFAULT_COMBINE_COMPONENTS,
    add_dm_derivatives: bool = True,
    exclude_from_shared: List[str] | tuple[str, ...] = ("DM",),
    parfile_output_dir: Path = None,
    timfile_output_dir: Path = None,
    use_pulse_numbers: str = "yes",
    clock_dir: Path | str | None = None,
    alignment_policy: AlignmentPolicy | None = None,
    convert_jump_mjd: bool = False,
) -> MetaPulsar:
    """Create MetaPulsar using specified combination strategy.

    Args:
        file_data: File data from FileDiscovery (should contain data for single pulsar only)
        combination_strategy: Strategy for combining PTAs:
            - "shared": Share selected timing-model parameters across PTAs (default)
            - "per_pta": Preserve per-PTA timing-model parameters
        reference_pta: PTA to use as reference (for shared strategy). If None, uses first PTA in file_data.
        combine_components: List of components to share (for shared strategy).
            Defaults to all components: ["astrometry", "spindown", "binary", "dispersion"]
        add_dm_derivatives: Whether to ensure DM1, DM2 are present in all par files (for shared strategy)
        exclude_from_shared: Canonical parameter names kept PTA-specific even
            when their component is merged. Defaults to ("DM",).
        parfile_output_dir: Directory to save shared par files (for shared strategy only).
            If None, par files are not saved to disk.
        timfile_output_dir: Directory to save the canonical .tim files the engines
            consumed, as {pulsar}_{pta}.tim. These are standalone Tempo2 FORMAT 1
            files (INCLUDEs flattened) carrying -pta, -pta_dataset,
            -timing_package, and (when the release par has JUMP MJD windows)
            -mjd_jump_pta flags, so they can be reused directly. If None, they
            are not saved to disk.
        use_pulse_numbers: Pulse-number mode: "no", "yes" (default), "reuse", "overwrite".
            The .tim is always rewritten; this only controls pulse numbers.
        clock_dir: Optional directory containing local clock-correction files.
        alignment_policy: AlignmentPolicy for the multi-PTA common profile.
            None means AlignmentPolicy(). Passing a policy together with
            combination_strategy="per_pta" raises ValueError.
        convert_jump_mjd: If True, rewrite each engine-par JUMP MJD line to
            JUMP -mjd_jump_pta {pta}_{k} ... using the same values stamped on
            the canonical tim. Default False (tim flags are still stamped).

    Returns:
        MetaPulsar object

    Raises:
        ValueError: If no files found, multiple pulsars detected, or invalid parameters
        RuntimeError: If PTA timing-record materialization fails
    """
```

### create_all_metapulsars

Create MetaPulsar objects for multiple pulsars.

```python
def create_all_metapulsars(
    file_data: Dict[str, List[Dict[str, Any]]],
    combination_strategy: str = "shared",
    reference_pta: str = None,
    combine_components: List[str] = DEFAULT_COMBINE_COMPONENTS,
    add_dm_derivatives: bool = True,
    exclude_from_shared: List[str] | tuple[str, ...] = ("DM",),
    parfile_output_dir: Path = None,
    timfile_output_dir: Path = None,
    use_pulse_numbers: str = "yes",
    clock_dir: Path | str | None = None,
    alignment_policy: AlignmentPolicy | None = None,
    convert_jump_mjd: bool = False,
) -> Dict[str, MetaPulsar]:
    """Create MetaPulsars for all pulsars in file_data.

    Args:
        file_data: File data from FileDiscovery (per data release)
        combination_strategy: Strategy for combining PTAs
        reference_pta: PTA to use as reference for all pulsars. If None, auto-selects by timespan.
        combine_components: List of components to share
        add_dm_derivatives: Whether to ensure DM1, DM2 are present
        exclude_from_shared: Canonical parameter names kept PTA-specific.
        parfile_output_dir: Directory to save shared par files (for shared strategy only).
            If None, par files are not saved to disk. Files are named per pulsar.
        timfile_output_dir: Directory to save the canonical .tim files the engines
            consumed, as {pulsar}_{pta}.tim. If None, they are not saved to disk.
        use_pulse_numbers: Pulse-number tracking mode.
        clock_dir: Optional directory containing local clock-correction files.
        alignment_policy: Alignment policy for the shared strategy.
        convert_jump_mjd: If True, rewrite each engine-par JUMP MJD line to
            JUMP -mjd_jump_pta {pta}_{k} ... Default False (tim flags still stamped).

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
        file_data: File data from FileDiscovery
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
        file_data: File data from FileDiscovery
        
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
        file_data: File data from FileDiscovery
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

### AlignmentPolicy

Policy for the multi-PTA `shared` combination strategy. This is the only
user-facing knob on the cross-engine alignment; everything else in the profile
is fixed, because only that combination is validated for residual parity.

```python
from metapulsar import AlignmentPolicy


@dataclass(frozen=True)
class AlignmentPolicy:
    unsupported: Literal["strip", "error"] = "strip"
    ephem: str | None = None
    clock: str | None = None
    bipm_version: int | None = None
    ne_sw: float | None = None

    # Gated ELL1-family -> DD/DDH conversion (mixed PINT+Tempo2 only)
    binary_conversion: Literal["auto", "off", "always"] = "auto"
    binary_conversion_threshold_s: float = 1e-9
    unsupported_binary: Literal["error", "keep"] = "error"
    binary_fidelity_floor_s: float = 1e-10
    h3_only: Literal["error", "sample_stigma"] = "error"
    stigma_central: float | None = None
    stigma_provenance: str | None = None
```

| Field | Meaning |
|-------|---------|
| `unsupported` | `"strip"` (default) removes deterministic families outside the common PINT/Tempo2 surface, logging a warning that names the PTA and every removed key. `"error"` raises `ValueError` listing every offender instead. |
| `ephem` | Override the reference PTA's `EPHEM`. Required when no PTA declares one. |
| `clock` | Override the reference PTA's `CLOCK`/`CLK`. Required when no PTA declares one. |
| `bipm_version` | Year used to resolve a bare `TT(BIPM)`, which is otherwise ambiguous across environments and raises. A dated clock that disagrees with this year also raises. |
| `ne_sw` | Override the resolved constant solar-wind density in cm⁻³. Without it: the reference PTA's explicit `NE_SW`/`NE1AU`/`SOLARN0`, else `4` when Tempo2 is in the stack, else no line. |
| `binary_conversion` | `"auto"` (default) rewrites a shared ELL1-family binary to `DD`/`DDH` only when the scale gate fires; `"off"` never classifies or converts; `"always"` bypasses the threshold but **never** widens the supported family set. |
| `binary_conversion_threshold_s` | Scale gate on `a1_max·e_max² + ½·n_b·a1_max²·e_max`, in seconds. Must be finite and > 0. This is a *scale gate*, not a predicted residual. |
| `unsupported_binary` | What to do when the gate fires on a family outside the supported sets (ELL1k, `FB` series, ELL1H domain violation, H4 tail above threshold, H3-only under the default `h3_only`, unknown span, unsupported fit pattern). `"error"` (default) raises `BinaryConversionError` with the reason and a remediation list; `"keep"` warns, records the decision, and proceeds unconverted. |
| `binary_fidelity_floor_s` | Absolute floor of the mandatory delay-fidelity tolerance. Must be finite and > 0. |
| `h3_only` | ELL1H sources carrying `H3` but neither `STIGMA` nor `H4`. `"error"` (default) refuses: no fixed ς is determined by such a par. `"sample_stigma"` converts at `stigma_central` and marks `required_sampling=("STIGMA",)` — the emitted ς is a **prior center, never a measurement**, and the analysis must sample it (or use a proper z-prior). |
| `stigma_central` | Prior-central ς in (0, 1] for `h3_only="sample_stigma"`. Required by, and only valid with, that mode. |
| `stigma_provenance` | Free-text provenance for `stigma_central` (e.g. `"mass-function closure, m_p=1.4"`), copied into the conversion record and the emitted par comment. Required by, and only valid with, `"sample_stigma"`. |

Constructor validation: `unsupported` must be `"strip"` or `"error"`; `ne_sw`
must be non-negative; `binary_conversion_threshold_s` and
`binary_fidelity_floor_s` must be finite and > 0; and
`stigma_central`/`stigma_provenance` may be set only when
`h3_only="sample_stigma"` (which in turn requires both).

Conversion applies only to `shared` stacks that carry **both** PINT and Tempo2
and share `"binary"`; single-engine, non-shared-binary, and `per_pta` stacks are
never converted. The result is reachable as
`MetaPulsar.binary_conversion_report` (and `MetaPulsar.conversion_metadata()`
for the nltiming STIGMA contract), reset on every materialization.

The policy applies only to `combination_strategy="shared"`. Passing it with
`"per_pta"` raises `ValueError` rather than being silently ignored; the
per-PTA strategy exists precisely to preserve engine-native models.

```python
# Default: strip unsupported deterministic terms with a warning.
mp = create_metapulsar(files, combination_strategy="shared")

# Fail instead of stripping.
mp = create_metapulsar(
    files,
    combination_strategy="shared",
    alignment_policy=AlignmentPolicy(unsupported="error"),
)

# Pin the reference conventions explicitly.
mp = create_metapulsar(
    files,
    combination_strategy="shared",
    alignment_policy=AlignmentPolicy(
        ephem="DE440",
        clock="TT(BIPM2023)",
        ne_sw=4.0,
    ),
)
```

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
        exclude_from_shared: List[str] | tuple[str, ...] = ("DM",),
        alignment_policy: AlignmentPolicy | None = None,
    ):
        """Initialize parameter manager with file data and configuration."""
```

`ParameterManager.ell1h_shapiro` reports which orthometric Shapiro expression
PINT should evaluate for this stack: `"absorbed"` (Freire & Wex 2010, Eq. 28)
for mixed PINT+Tempo2 stacks, `"full"` (PINT's default, Eq. 29) otherwise. The
factory passes it to `get_model_and_toas` so materialization matches the
temporary models built during alignment.

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
