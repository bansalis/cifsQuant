# Tumor Boundary Implementation

## Problem Addressed

**Original Issue**: DBSCAN clustering only identifies dense TOM+ cells, creating hard boundaries that exclude infiltrating immune cells and tumor-associated macrophages that are spatially inside the tumor mass.

**Result**: "Within tumor" infiltration metrics were showing 0 or very low values because immune cells physically inside the tumor weren't being counted.

## Solution: Dual Boundary System

### 1. Core Tumor Structures (DBSCAN)
- **Purpose**: Identify discrete tumor structures
- **Method**: DBSCAN clustering on TOM+ cells
- **Parameters**: eps=800μm, min_samples=250
- **Used for**: Distance-based infiltration zones (50-200μm from tumor edge)

### 2. Expanded Tumor Regions (Buffer)
- **Purpose**: Capture all cells spatially inside/near tumor mass
- **Method**: Include any cell within buffer distance of core tumor
- **Default**: 100μm buffer expansion
- **Used for**: "Within tumor" infiltration calculations

## How It Works

```python
# Step 1: DBSCAN identifies core tumor structures
core_tumor_cells = DBSCAN(TOM+ cells)

# Step 2: Expand boundaries using KDTree
for each core_tumor_structure:
    expanded_region = all_cells within buffer_distance of core
    # This includes TOM- cells: immune, stromal, etc.

# Step 3: Calculate infiltration
within_tumor_immune = immune_cells in expanded_region  # NEW
zone_50um_immune = immune_cells 0-50μm from core edge
zone_100um_immune = immune_cells 50-100μm from core edge
```

## Configuration

Edit `spatial_quantification/config/spatial_config.yaml`:

```yaml
tumor_definition:
  structure_detection:
    # Core structure detection
    method: 'DBSCAN'
    eps: 800
    min_samples: 250
    min_cluster_size: 250

    # Boundary expansion (NEW)
    boundary_buffer: 100  # μm, expand tumor region by this distance
    use_expanded_boundary: true  # Use expanded boundary for "within tumor"
```

### Configuration Options

- **boundary_buffer**: Distance (μm) to expand tumor boundaries
  - Default: 100μm
  - Smaller (50μm): More conservative, closer to core tumor
  - Larger (150-200μm): More inclusive, captures broader microenvironment

- **use_expanded_boundary**: Enable/disable boundary expansion
  - `true`: Use expanded regions for "within tumor" (recommended)
  - `false`: Use distance threshold instead (old behavior)

## Impact on Results

### Before (DBSCAN only)
```
within_tumor infiltration = 0-5 cells (missed infiltrating lymphocytes)
```

### After (Expanded boundaries)
```
within_tumor infiltration = 50-200 cells (captures true tumor microenvironment)
```

## Output Changes

The infiltration results CSV now includes:
- **structure_size**: Number of core TOM+ tumor cells
- **region_size**: Number of cells in expanded tumor region (includes TOM- cells)

Example:
```
structure_size = 1000  (TOM+ cells)
region_size = 1450     (TOM+ cells + infiltrating immune + associated stroma)
```

## Biological Interpretation

**Core Tumor (DBSCAN)**: Dense tumor cell mass
**Expanded Region (Buffer)**: Tumor microenvironment including:
- Infiltrating CD8+ T cells
- Tumor-associated macrophages (CD45+)
- Activated fibroblasts
- Other stromal cells physically inside the tumor

**Distance Zones (from core edge)**: Tumor periphery and surrounding stroma

## Technical Details

### File: `infiltration_analysis_optimized.py`

**New method**: `_expand_tumor_boundaries()`
- Uses KDTree for efficient spatial queries
- Finds all cells within buffer distance of core tumor
- Handles overlapping structures (assigns to nearest)

**Modified method**: `_calculate_infiltration()`
- Distinguishes core structures vs expanded regions
- Uses expanded regions for "within_tumor" zone
- Uses core structures for distance-based zones

### Algorithm Complexity
- Core detection: O(n log n) - DBSCAN with KDTree
- Boundary expansion: O(n log m) - KDTree query all cells against core
- Infiltration calculation: O(k log m) - KDTree query immune cells against core

Where:
- n = total cells in sample
- m = tumor cells in structure
- k = immune cells

## Backward Compatibility

To use old behavior (distance threshold only):
```yaml
structure_detection:
  use_expanded_boundary: false
```

## Future Enhancements

Possible improvements:
1. **Alpha shapes**: More precise boundary following tumor contours
2. **Convex hull**: Simple enclosed boundary (may include too much space)
3. **Gaussian mixture**: Probabilistic tumor region definition
4. **Marker-specific expansion**: Different buffers for different markers

## Testing Recommendations

After changing `boundary_buffer`, check:
1. Spatial plots show reasonable tumor regions
2. "Within tumor" infiltration is non-zero
3. Region_size is larger than structure_size
4. Results make biological sense for your data

Typical buffer ranges:
- **50μm**: Conservative, immediate microenvironment
- **100μm**: Balanced (default), captures most infiltrating cells
- **150-200μm**: Inclusive, broader tumor region
- **>200μm**: May include too much non-tumor tissue

## Summary

This implementation solves the "within tumor = 0" bug by:
1. Maintaining DBSCAN for core tumor structure identification
2. Adding configurable buffer expansion for softer boundaries
3. Capturing infiltrating immune cells in expanded tumor regions
4. Providing clear distinction between core tumor and tumor microenvironment
5. Remaining backward compatible and fully configurable
