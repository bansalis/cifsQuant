# Marker-Based Regional Analysis with SpatialCells

## Overview

The **MarkerRegionAnalysisSpatialCells** module provides comprehensive analysis of heterogeneous marker-defined spatial regions (e.g., pERK+/- zones, Ki67+/- zones) and characterizes immune cell enrichment within these regions.

This goes beyond simple single-cell marker positivity by:
1. Detecting **spatial communities** of marker+ and marker- cells
2. Creating **geometric boundaries** around these regions using alpha shapes
3. Comparing **immune cell enrichment** in marker+ vs marker- spatial zones
4. Analyzing **regional composition** and **holes/gaps** within marker regions

## Key Features

### 1. Marker Region Detection
- Detects pERK+, pERK-, Ki67+, Ki67- (and other marker) spatial communities
- Uses DBSCAN + alpha shapes for robust region boundaries
- Identifies discrete spatial components within each marker type

### 2. Regional Composition Analysis
- Quantifies cell type composition within each marker region
- Analyzes immune cell populations in marker+ vs marker- zones
- Calculates density and percent composition metrics

### 3. Marker Region Comparison
- Direct comparison of marker+ vs marker- regions
- Fold-change and difference calculations
- Statistical comparisons across samples

### 4. Immune Enrichment Analysis
- Tests if immune cells are enriched in marker+ vs marker- regions
- Calculates enrichment fold-change and statistical significance
- Identifies marker regions with high/low immune infiltration

### 5. Hole Detection (Optional)
- Detects holes/gaps within marker regions
- Analyzes cells in inner, boundary, and outer zones of holes
- Quantifies spatial heterogeneity within marker regions

## Installation

Requires SpatialCells library (see main migration guide for installation).

## Configuration

Add to your `spatial_config.yaml`:

```yaml
marker_region_analysis:
  enabled: true

  # Markers to analyze
  markers:
    - name: 'pERK'
      positive_col: 'is_pERK_positive_tumor'
      negative_col: 'is_pERK_negative_tumor'
    - name: 'Ki67'
      positive_col: 'is_Ki67_positive_tumor'
      negative_col: 'is_Ki67_negative_tumor'
    - name: 'NINJA'
      positive_col: 'is_AGFP_positive_tumor'
      negative_col: 'is_AGFP_negative_tumor'

  # Region detection parameters
  region_detection:
    eps: 55                    # DBSCAN epsilon (smaller = more outliers)
    min_samples: 5             # Minimum samples for DBSCAN
    alpha: 27                  # Alpha shape parameter (smaller = more detailed)
    core_only: true            # Use only core samples (more stringent)
    min_area: 0                # Minimum area for regions (μm²)
    min_edges: 20              # Minimum edges for boundary polygons
    holes_min_area: 10000      # Minimum area for holes (μm²)
    holes_min_edges: 10        # Minimum edges for holes

  # Phenotypes to analyze within regions
  phenotype_columns:
    - 'is_CD8_T_cells'
    - 'is_CD3_positive'
    - 'is_CD45_positive'
    - 'is_Tumor'

  # Immune populations for enrichment analysis
  immune_populations:
    - 'is_CD8_T_cells'
    - 'is_CD3_positive'
    - 'is_CD45_positive'

  # Optional hole analysis
  analyze_holes: true
  hole_buffer_distance: 30   # Distance for inner/outer hole zones (μm)
```

## Usage

### Basic Usage

```python
from spatial_quantification.analyses import MarkerRegionAnalysisSpatialCells

# Initialize analysis
marker_analysis = MarkerRegionAnalysisSpatialCells(adata, config, output_dir)

# Run analysis
results = marker_analysis.run()

# Access results
detected_regions = results['detected_marker_regions']
composition = results['regional_composition']
comparison = results['marker_region_comparison']
enrichment = results['immune_enrichment']
holes = results['region_holes']  # If enabled
```

### Integration with Main Pipeline

Add to `run_spatial_quantification.py`:

```python
# After other analyses
if config.get('marker_region_analysis', {}).get('enabled', False):
    print("\n  Running marker region analysis...")
    from spatial_quantification.analyses import MarkerRegionAnalysisSpatialCells

    marker_region = MarkerRegionAnalysisSpatialCells(adata, config, output_dir)
    all_results['marker_region_analysis'] = marker_region.run()

    # Generate plots if configured
    if config.get('marker_region_analysis', {}).get('generate_plots', True):
        try:
            from spatial_quantification.visualization.marker_region_plotter import MarkerRegionPlotter
            plotter = MarkerRegionPlotter(output_dir / 'marker_region_analysis', config)
            plotter.generate_all_plots(all_results['marker_region_analysis'])
        except Exception as e:
            print(f"  ⚠ Could not generate marker region plots: {e}")
```

### Advanced: Direct Boundary Access

```python
# Get boundary for specific marker region
boundary = marker_analysis.get_boundary(
    sample='sample_001',
    marker='pERK',
    polarity='positive',
    region_id=0
)

# Use with SpatialCells functions
import spatialcells as spc

# Calculate area
area = spc.msmt.getRegionArea(boundary)

# Get centroid
centroid = spc.msmt.getRegionCentroid(boundary)

# Analyze composition
spc.spatial.assignPointsToRegions(
    adata,
    [boundary],
    ['pERK_positive_region'],
    assigncolumn='marker_region'
)
composition = spc.msmt.getRegionComposition(
    adata,
    'phenotype_col',
    regions=['pERK_positive_region']
)
```

## Output Files

### 1. detected_marker_regions.csv
Basic information about each detected marker region.

**Columns:**
- `sample_id`: Sample identifier
- `marker`: Marker name (e.g., 'pERK', 'Ki67')
- `polarity`: 'positive' or 'negative'
- `region_id`: Unique region identifier
- `n_communities`: Number of DBSCAN communities detected
- `n_components`: Number of boundary components after pruning
- `area_um2`: Region area in μm²
- `centroid_x`, `centroid_y`: Region centroid coordinates
- `n_marker_cells`: Number of marker-positive cells
- `timepoint`, `group`, `main_group`: Metadata

**Example:**
```
sample_id,marker,polarity,region_id,n_communities,n_components,area_um2,centroid_x,centroid_y
sample_001,pERK,positive,0,3,2,45678.2,1234.5,5678.9
sample_001,pERK,negative,0,2,1,32145.7,2345.6,6789.0
sample_001,Ki67,positive,0,4,3,56789.1,3456.7,7890.1
```

### 2. regional_composition.csv
Cell type composition within each marker region.

**Columns:**
- `sample_id`, `marker`, `polarity`, `region_id`: Region identifiers
- `phenotype`: Cell type/phenotype analyzed
- `n_cells`: Number of cells of this phenotype
- `n_total_cells`: Total cells in region
- `composition`: Fraction (0-1)
- `percent`: Percentage (0-100)
- `timepoint`, `group`, `main_group`: Metadata

**Example:**
```
sample_id,marker,polarity,region_id,phenotype,n_cells,n_total_cells,composition,percent
sample_001,pERK,positive,0,is_CD8_T_cells,523,4567,0.1145,11.45
sample_001,pERK,positive,0,is_CD3_positive,892,4567,0.1953,19.53
sample_001,pERK,negative,0,is_CD8_T_cells,234,3456,0.0677,6.77
sample_001,pERK,negative,0,is_CD3_positive,445,3456,0.1288,12.88
```

### 3. marker_region_comparison.csv
Comparison of marker+ vs marker- regions.

**Columns:**
- `sample_id`, `marker`, `phenotype`: Identifiers
- `positive_mean_composition`: Mean composition in marker+ regions
- `positive_std_composition`: Std dev in marker+ regions
- `positive_n_regions`: Number of marker+ regions
- `positive_total_cells`: Total cells in marker+ regions
- `negative_mean_composition`: Mean composition in marker- regions
- `negative_std_composition`: Std dev in marker- regions
- `negative_n_regions`: Number of marker- regions
- `negative_total_cells`: Total cells in marker- regions
- `fold_change_pos_vs_neg`: Positive / Negative ratio
- `difference_pos_minus_neg`: Positive - Negative difference
- `timepoint`, `group`, `main_group`: Metadata

**Example:**
```
sample_id,marker,phenotype,positive_mean_composition,negative_mean_composition,fold_change_pos_vs_neg,difference_pos_minus_neg
sample_001,pERK,is_CD8_T_cells,0.1145,0.0677,1.691,0.0468
sample_001,pERK,is_CD3_positive,0.1953,0.1288,1.516,0.0665
```

**Key Insights:**
- `fold_change > 1`: Enriched in marker+ regions
- `fold_change < 1`: Depleted in marker+ regions
- `difference > 0`: More abundant in marker+ regions

### 4. immune_enrichment.csv
Immune cell enrichment in marker+ vs marker- regions.

**Columns:**
- `sample_id`, `marker`, `immune_population`: Identifiers
- `n_positive_regions`: Number of marker+ regions
- `n_negative_regions`: Number of marker- regions
- `n_cells_positive_regions`: Total cells in marker+ regions
- `n_cells_negative_regions`: Total cells in marker- regions
- `n_immune_in_positive`: Immune cells in marker+ regions
- `n_immune_in_negative`: Immune cells in marker- regions
- `immune_density_positive`: Immune density in marker+ regions
- `immune_density_negative`: Immune density in marker- regions
- `enrichment_fold_change`: Density ratio (positive/negative)
- `enrichment_difference`: Density difference (positive - negative)
- `percent_immune_positive`: % immune in marker+ regions
- `percent_immune_negative`: % immune in marker- regions
- `timepoint`, `group`, `main_group`: Metadata

**Example:**
```
sample_id,marker,immune_population,n_immune_in_positive,n_immune_in_negative,enrichment_fold_change,percent_immune_positive,percent_immune_negative
sample_001,pERK,is_CD8_T_cells,523,234,1.691,11.45,6.77
sample_001,Ki67,is_CD8_T_cells,678,345,1.487,13.56,9.12
```

**Key Insights:**
- `enrichment_fold_change > 1`: Immune cells enriched in marker+ regions
- `enrichment_fold_change < 1`: Immune cells depleted in marker+ regions
- High fold-change suggests marker expression correlates with immune presence

### 5. region_holes.csv (Optional)
Analysis of holes/gaps within marker regions.

**Columns:**
- `sample_id`, `marker`, `polarity`, `region_id`, `hole_id`: Identifiers
- `hole_area_um2`: Hole area in μm²
- `hole_centroid_x`, `hole_centroid_y`: Hole centroid coordinates
- `n_cells_inner_zone`: Cells in inner zone (hole - buffer)
- `n_cells_boundary_zone`: Cells in hole boundary
- `n_cells_outer_zone`: Cells in outer zone (hole + buffer)
- `timepoint`, `group`, `main_group`: Metadata

**Example:**
```
sample_id,marker,polarity,region_id,hole_id,hole_area_um2,n_cells_inner_zone,n_cells_boundary_zone,n_cells_outer_zone
sample_001,pERK,positive,0,0,12345.6,45,123,234
sample_001,pERK,positive,0,1,8765.4,23,89,156
```

## Example Analysis Workflow

### 1. Detect pERK+/- Regions and Compare Immune Infiltration

```python
# Run analysis
marker_analysis = MarkerRegionAnalysisSpatialCells(adata, config, output_dir)
results = marker_analysis.run()

# Get immune enrichment results
enrichment_df = results['immune_enrichment']

# Filter for pERK and CD8 T cells
perk_cd8 = enrichment_df[
    (enrichment_df['marker'] == 'pERK') &
    (enrichment_df['immune_population'] == 'is_CD8_T_cells')
]

# Analyze enrichment
for _, row in perk_cd8.iterrows():
    print(f"Sample: {row['sample_id']}")
    print(f"  CD8+ in pERK+ regions: {row['percent_immune_positive']:.2f}%")
    print(f"  CD8+ in pERK- regions: {row['percent_immune_negative']:.2f}%")
    print(f"  Enrichment: {row['enrichment_fold_change']:.2f}x")
    print()
```

### 2. Compare Multiple Markers

```python
comparison_df = results['marker_region_comparison']

# Filter for CD8 T cells across all markers
cd8_comparison = comparison_df[
    comparison_df['phenotype'] == 'is_CD8_T_cells'
]

# Rank markers by immune enrichment
cd8_comparison_sorted = cd8_comparison.sort_values(
    'fold_change_pos_vs_neg',
    ascending=False
)

print("Markers ranked by CD8 T cell enrichment (+ vs -):")
for _, row in cd8_comparison_sorted.iterrows():
    print(f"  {row['marker']}: {row['fold_change_pos_vs_neg']:.2f}x")
```

### 3. Visualize Regional Boundaries

```python
import matplotlib.pyplot as plt
import spatialcells as spc

# Get boundaries for pERK+ regions
boundaries = marker_analysis.get_marker_boundaries()
sample = 'sample_001'
perk_pos_boundaries = boundaries[sample]['pERK_positive']

# Plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plot all cells
coords = adata[adata.obs['sample_id'] == sample].obsm['spatial']
ax.scatter(coords[:, 0], coords[:, 1], s=1, c='lightgray', alpha=0.3)

# Plot pERK+ region boundaries
for region_id, boundary in perk_pos_boundaries.items():
    spc.plt.plotBoundary(boundary, ax=ax, color='red', linewidth=2,
                         label=f'pERK+ region {region_id}')

ax.set_title(f'{sample}: pERK+ Spatial Regions')
ax.legend()
ax.invert_yaxis()
plt.show()
```

### 4. Statistical Analysis

```python
import scipy.stats as stats

# Compare immune enrichment across timepoints
enrichment_df = results['immune_enrichment']

# Filter for pERK and CD8
perk_cd8 = enrichment_df[
    (enrichment_df['marker'] == 'pERK') &
    (enrichment_df['immune_population'] == 'is_CD8_T_cells')
]

# Group by timepoint
timepoint_groups = perk_cd8.groupby('timepoint')

# Compare enrichment fold-change across timepoints
for tp, group in timepoint_groups:
    print(f"Timepoint {tp}:")
    print(f"  Mean enrichment: {group['enrichment_fold_change'].mean():.2f}x")
    print(f"  Median enrichment: {group['enrichment_fold_change'].median():.2f}x")
    print(f"  n={len(group)}")
    print()

# Statistical test
tp_values = [group['enrichment_fold_change'].values
             for tp, group in timepoint_groups]
statistic, pvalue = stats.kruskal(*tp_values)
print(f"Kruskal-Wallis test p-value: {pvalue:.4f}")
```

## Biological Interpretation

### Immune Enrichment Patterns

**High enrichment in marker+ regions** (fold-change > 1.5):
- Suggests marker expression correlates with immune infiltration
- May indicate immune-reactive tumor regions
- Could reflect immunogenic marker expression

**High enrichment in marker- regions** (fold-change < 0.67):
- Suggests immune exclusion from marker+ regions
- May indicate immune evasion mechanisms
- Could reflect immunosuppressive marker expression

### Regional Heterogeneity

**Multiple discrete components** per marker:
- Indicates spatial heterogeneity of marker expression
- Suggests distinct microenvironmental niches
- May reflect clonal evolution or microenvironmental gradients

**Large holes within regions**:
- May represent necrotic cores
- Could indicate vascular structures
- Might reflect stromal infiltration

## Parameters Guide

### Region Detection

**eps** (default: 55):
- Larger values: More cells grouped together, fewer outliers
- Smaller values: More stringent clustering, more cells excluded
- Tune based on cell density and expected region size

**alpha** (default: 27):
- Smaller values: More detailed, jagged boundaries
- Larger values: Smoother, more generalized boundaries
- Tune based on desired boundary precision

**core_only** (default: true):
- `true`: Only core samples used (more stringent)
- `false`: All clustered cells included (more inclusive)
- Use `true` for well-defined regions, `false` for sparse data

**min_area** (default: 0):
- Filter out very small regions
- Set to exclude artifacts or noise
- Typical values: 0-10000 μm²

### Hole Analysis

**hole_buffer_distance** (default: 30):
- Distance for inner/outer hole zones
- Larger values: Wider transition zones
- Smaller values: Narrower boundary analysis
- Typical values: 20-50 μm

## Troubleshooting

### No Regions Detected

**Problem**: `detected_marker_regions.csv` is empty

**Solutions**:
1. Check marker column exists: `'is_pERK_positive_tumor' in adata.obs.columns`
2. Reduce `eps` parameter (try 30-40)
3. Reduce `min_samples` (try 3-5)
4. Disable `core_only` (set to `false`)
5. Check marker-positive cell count: `adata.obs['is_pERK_positive_tumor'].sum()`

### Too Many Small Regions

**Problem**: Many tiny regions detected

**Solutions**:
1. Increase `min_area` (try 5000-10000)
2. Increase `min_edges` (try 30-40)
3. Increase `eps` (try 70-100)
4. Enable `core_only` (set to `true`)

### Boundaries Too Detailed/Jagged

**Problem**: Boundaries have too much detail

**Solutions**:
1. Increase `alpha` parameter (try 40-60)
2. Increase `min_edges` (try 30-40)
3. Increase `holes_min_edges` (try 20-30)

### Memory Issues

**Problem**: Analysis runs out of memory

**Solutions**:
1. Process samples individually
2. Increase `min_area` to filter small regions early
3. Disable hole analysis if not needed
4. Reduce number of markers analyzed

## References

- **SpatialCells Paper**: [Briefings in Bioinformatics (2024)](https://academic.oup.com/bib/article/25/3/bbae189/7663435)
- **Main Migration Guide**: `SPATIALCELLS_MIGRATION.md`
- **GitHub**: https://github.com/SemenovLab/SpatialCells
