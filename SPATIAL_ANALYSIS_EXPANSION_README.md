# Spatial Analysis Expansion - Comprehensive Documentation

## Overview

This expansion significantly enhances the spatial tumor analysis pipeline with extensive immune infiltration analyses, heterogeneity metrics, and dual-level statistics.

## New Features

### 1. T Cell-Tumor Distance Analysis (Dual-Level)

**File**: `spatial_analysis_expansions.py` - `ImmuneInfiltrationAnalysis` class

**Key Features**:
- **Dual-level analysis**: Both per-structure (n = tumors) AND per-sample (n = samples)
- **Tumor subtype-specific distances**:
  - T cells to all tumor cells
  - T cells to pERK+ tumor cells
  - T cells to pERK- tumor cells
  - T cells to NINJA+ (aGFP+) tumor cells
  - T cells to NINJA- tumor cells

**Outputs**:
- `data/distances/tcell_tumor_distances_per_structure.csv` - Structure-level metrics
- `data/distances/tcell_tumor_distances_per_sample.csv` - Sample-level metrics
- `statistics/distances/tcell_tumor_distance_statistics.csv` - Statistical tests

**Metrics Computed**:
- Mean/median/min/max distance
- Distance distributions (25th, 75th percentiles)
- Percentage of T cells within specified distance
- Correlations with tumor size

**Statistical Tests**:
- Temporal trends (Spearman correlation, linear regression)
- Group comparisons (Mann-Whitney U, t-tests)
- Per-timepoint comparisons
- Effect sizes (Cohen's d)
- FDR correction for multiple testing

### 2. Tumor Heterogeneity Analysis

**File**: `spatial_analysis_expansions.py` - `ImmuneInfiltrationAnalysis.analyze_tumor_heterogeneity()`

**Purpose**: Answers the question "Are NINJA+ and pERK+ regions spatially isolated or randomly distributed?"

**Key Metrics**:
- **Heterogeneity Index**: Standard deviation of local marker percentages
  - Measures how much marker expression varies across the tumor
  - Higher = more heterogeneous (patchy distribution)
  - Lower = more homogeneous (uniform distribution)

- **Clustering Index**: Ratio of expected to observed nearest-neighbor distances
  - \> 1 = Marker-positive cells are clustered together (spatial isolation)
  - = 1 = Random distribution
  - < 1 = Marker-positive cells are dispersed

**Outputs**:
- `data/heterogeneity/tumor_heterogeneity.csv` - Heterogeneity metrics per structure
- `statistics/heterogeneity/heterogeneity_statistics.csv` - Statistical comparisons
- `figures/heterogeneity/[marker]_heterogeneity.png` - Visualization plots

**Questions Answered**:
- Are pERK+ regions clustered or dispersed in KPT vs KPNT?
- Are NINJA+ regions clustered or dispersed?
- Does heterogeneity change over time?
- Does heterogeneity differ between genotypes?

### 3. NINJA+/- Tumor Comparison

**File**: `spatial_analysis_expansions.py` - `ImmuneInfiltrationAnalysis.compare_ninja_positive_negative_tumors()`

**Purpose**:
- Identify NINJA-negative tumors in KPNT
- Compare NINJA-negative tumors between KPT and KPNT
- Test if NINJA- KPNT tumors are similar to KPT tumors

**Classification**:
- NINJA+ tumor: > 10% aGFP+ cells
- NINJA- tumor: ≤ 10% aGFP+ cells

**Outputs**:
- `data/ninja_tumor_classification.csv` - Tumor classifications
- `statistics/ninja_comparison_statistics.csv` - Statistical comparisons

**Questions Answered**:
- Do NINJA-negative tumors exist in KPNT?
- How common are they?
- Are NINJA-negative KPNT tumors similar to KPT tumors in size and composition?

### 4. Neighborhood Composition Temporal Tracking

**File**: `spatial_analysis_expansions.py` - `NeighborhoodTemporalAnalysis` class

**Purpose**: Track how cellular neighborhood composition changes over time

**Key Features**:
- Composition of each neighborhood type over time
- Per-group tracking (KPT vs KPNT)
- Temporal divergence analysis

**Outputs**:
- `data/neighborhood_composition_temporal.csv` - Composition by timepoint
- `figures/neighborhoods/temporal/neighborhood_composition_temporal.png` - Stacked area plot
- `figures/neighborhoods/temporal/neighborhood_heatmap_[group].png` - Heatmap per group

**Questions Answered**:
- How does neighborhood composition change over time?
- When do KPT and KPNT diverge?
- Which neighborhood types expand/contract over time?

### 5. Spatial Maps with Analysis Ranges

**File**: `spatial_analysis_expansions.py` - `create_spatial_maps_with_analysis_ranges()`

**Purpose**: Visualize exactly what is being measured

**Features**:
- Shows tumor boundaries
- Shows margin, peri-tumor, and distal zones
- Color-coded by tumor structure
- Overlays analysis distances (30μm, 100μm, 200μm circles)

**Outputs**:
- `figures/spatial_maps/analysis_ranges/[sample]_analysis_ranges.png`

**Benefits**:
- Visual confirmation of analysis parameters
- Quality control for structure detection
- Publication-ready figures showing analysis methodology

### 6. Comprehensive Visualizations with Statistics

**File**: `comprehensive_visualizations_stats.py` - `ComprehensiveVisualizationsStats` class

**All plots include**:
- Statistical test results annotated on plots
- P-values with significance stars (*, **, ***)
- Sample sizes (n)
- Error bars (SEM or SD)
- FDR-corrected p-values

**Plot Types**:
1. **Distance Analysis Plots**:
   - Boxplots by timepoint and group
   - Line plots with error bars
   - Distribution comparisons (histograms/density)
   - Scatterplots (distance vs tumor size)
   - Violin plots

2. **Heterogeneity Plots**:
   - Heterogeneity boxplots with p-values
   - Clustering index comparisons
   - Temporal trends
   - Heterogeneity vs clustering scatterplots

3. **Neighborhood Plots**:
   - Stacked area plots (composition over time)
   - Heatmaps (all neighborhood types × timepoints)
   - Per-group comparisons

### 7. Dual-Level Statistics

**Both levels computed throughout**:

**Level 1: Per-Structure (n = tumors)**
- Each tumor is an independent observation
- More statistical power
- Captures tumor-to-tumor variability

**Level 2: Per-Sample (n = samples)**
- Each sample is an independent observation
- Accounts for sample-level batch effects
- More conservative

**All statistical tests run at both levels**:
- Temporal trends
- Group comparisons
- Correlations

## Usage

### Basic Usage

```python
python run_expanded_comprehensive_analysis.py --config configs/comprehensive_config.yaml --metadata sample_metadata.csv
```

### Configuration

Ensure your `configs/comprehensive_config.yaml` includes:

```yaml
input_data: "path/to/adata.h5ad"
output_directory: "expanded_comprehensive_analysis"

tumor_markers:
  - Pan-CK
  - aGFP
  - pERK

immune_markers:
  - CD3
  - CD8
  - CD4
  - CD45

populations:
  Tumor:
    markers:
      Pan-CK: true
    color: "#FF6B6B"

  CD3:
    markers:
      CD3: true
    color: "#4ECDC4"

  CD8:
    markers:
      CD8: true
    color: "#45B7D1"

tumor_structure_detection:
  min_cluster_size: 50
  eps: 30
  min_samples: 10

immune_infiltration:
  populations:
    - CD3
    - CD8
    - CD4

infiltration_boundaries:
  boundary_widths: [30, 100, 200]

buffer_distance: 500

cellular_neighborhoods:
  enabled: true
  window_size: 100
  n_clusters: 10

statistical_analysis:
  enabled: true

visualizations:
  enabled: true
```

## Output Structure

```
expanded_comprehensive_analysis/
├── data/
│   ├── distances/
│   │   ├── tcell_tumor_distances_per_structure.csv
│   │   └── tcell_tumor_distances_per_sample.csv
│   ├── heterogeneity/
│   │   └── tumor_heterogeneity.csv
│   ├── ninja_tumor_classification.csv
│   ├── neighborhood_composition_temporal.csv
│   ├── marker_expression_temporal.csv
│   └── tumor_size_by_sample.csv
│
├── statistics/
│   ├── distances/
│   │   └── tcell_tumor_distance_statistics.csv
│   ├── heterogeneity/
│   │   └── heterogeneity_statistics.csv
│   ├── ninja_comparison_statistics.csv
│   └── [other statistical tests]
│
├── figures/
│   ├── spatial_maps/
│   │   ├── by_sample/
│   │   ├── by_timepoint/
│   │   ├── by_genotype/
│   │   └── analysis_ranges/
│   ├── distances/
│   │   ├── [tcell]_to_[tumor_subtype]_per_structure.png
│   │   └── [tcell]_to_[tumor_subtype]_per_sample.png
│   ├── heterogeneity/
│   │   ├── aGFP_heterogeneity.png
│   │   └── pERK_heterogeneity.png
│   └── neighborhoods/
│       └── temporal/
│           ├── neighborhood_composition_temporal.png
│           └── neighborhood_heatmap_[group].png
│
└── ANALYSIS_SUMMARY.txt
```

## Key Questions Answered

### 1. T Cell-Tumor Interactions

**Q: What is the distance from T cells to tumor cells?**
- See `data/distances/tcell_tumor_distances_per_structure.csv`
- Plots in `figures/distances/`

**Q: Are T cells closer to pERK+ tumor cells or pERK- tumor cells?**
- Compare mean_distance columns for pERK_positive vs pERK_negative in distance files
- Statistical tests in `statistics/distances/tcell_tumor_distance_statistics.csv`

**Q: How does this change over time?**
- Temporal trend tests in statistics files (Spearman rho, p-values)
- Line plots in `figures/distances/` show temporal trajectories

**Q: Does pERK expression change in KPT over time?**
- See `data/marker_expression_temporal.csv` filtered for marker=pERK and main_group=KPT
- Temporal trends tested statistically

### 2. T Cell-NINJA+ Tumor Interactions

**Q: Are T cells close to NINJA+ (aGFP+) tumor cells?**
- See distance metrics for tumor_subtype=NINJA_positive
- Compare to NINJA_negative distances

**Q: What type of T cells are close to NINJA+ cells?**
- Separate analyses for CD3, CD8, CD4 populations
- All in distance CSV files

**Q: What is the difference in NINJA+/- tumors?**
- See `data/ninja_tumor_classification.csv`
- Statistical comparisons in `statistics/ninja_comparison_statistics.csv`

### 3. Tumor Heterogeneity

**Q: Are NINJA+ and pERK+ regions spatially isolated or randomly distributed?**
- **Clustering Index > 1** = Spatially isolated (clustered together)
- **Clustering Index ≈ 1** = Randomly distributed
- **Clustering Index < 1** = Dispersed (avoid each other)
- See `data/heterogeneity/tumor_heterogeneity.csv`
- Visualizations in `figures/heterogeneity/`

**Q: How heterogeneous are tumors?**
- Higher heterogeneity index = more patchy/heterogeneous
- Lower heterogeneity index = more uniform
- Compare KPT vs KPNT in heterogeneity plots

### 4. Cellular Neighborhoods (RCNs)

**Q: How do neighborhoods differ between KPT and KPNT?**
- See `data/neighborhood_composition_temporal.csv`
- Stacked area plots show composition differences

**Q: If we ignore NINJA/aGFP, how similar are KPT and KPNT?**
- You can re-run the neighborhood analysis excluding aGFP from the population list
- Current analysis includes all populations

**Q: When do they diverge?**
- Look at temporal plots in `figures/neighborhoods/temporal/`
- Identify timepoints where composition starts to differ

**Q: Do NINJA-negative tumors exist in KPNT?**
- See `data/ninja_tumor_classification.csv`
- Group by main_group and ninja_status to count

**Q: Are NINJA-negative KPNT tumors similar to KPT?**
- See `statistics/ninja_comparison_statistics.csv`
- Test: "KPT_vs_KPNT_NINJA_negative_only"

## Statistical Interpretation Guide

### P-value Annotations
- `***` : p < 0.001 (highly significant)
- `**`  : p < 0.01 (very significant)
- `*`   : p < 0.05 (significant)
- `ns`  : p ≥ 0.05 (not significant)

### Effect Sizes
- **Cohen's d**:
  - Small: 0.2
  - Medium: 0.5
  - Large: 0.8

### Temporal Trends
- **Spearman rho** (ρ):
  - -1 to 1 range
  - Positive: increases over time
  - Negative: decreases over time
  - Close to 0: no temporal trend

- **R²** (linear regression):
  - 0 to 1 range
  - How much variance is explained by time
  - Higher = stronger temporal trend

### Multiple Testing Correction
- All p-values are FDR-corrected (Benjamini-Hochberg)
- Columns: `p_adjusted`, `significant`
- Use adjusted p-values for final conclusions

## Troubleshooting

### Issue: Memory errors during distance analysis

**Solution**: Reduce `max_distance` parameter or process fewer structures at a time

### Issue: Too few NINJA-negative tumors

**Solution**: Adjust threshold in `compare_ninja_positive_negative_tumors()` (default is 10%)

### Issue: Neighborhood plots show too many types

**Solution**: Reduce `n_clusters` in cellular_neighborhoods config

### Issue: Want to focus on specific T cell populations

**Solution**: Modify `tcell_populations` list in `run_expanded_comprehensive_analysis.py`

## Customization

### Add More Tumor Subtypes

In `run_expanded_comprehensive_analysis.py`, modify the `tumor_subtypes` dictionary:

```python
tumor_subtypes = {
    'Tumor_all': {'is_Tumor': True},
    'pERK_positive': {'pERK': True, 'is_Tumor': True},
    'pERK_negative': {'pERK': False, 'is_Tumor': True},
    'NINJA_positive': {'aGFP': True, 'is_Tumor': True},
    'NINJA_negative': {'aGFP': False, 'is_Tumor': True},
    # Add your own:
    'double_positive': {'pERK': True, 'aGFP': True, 'is_Tumor': True},
}
```

### Add More T Cell Populations

```python
tcell_populations = ['CD3', 'CD8', 'CD4', 'YourMarker']
```

### Modify Distance Thresholds

In the function call:
```python
structure_distances, sample_distances = immune_analyzer.analyze_tcell_tumor_distances_comprehensive(
    tcell_populations=tcell_populations,
    tumor_subtypes=tumor_subtypes,
    max_distance=300  # Change this value
)
```

## Citation

If you use this expanded analysis framework, please cite the original SpatialCells package and note the custom expansions for immune-tumor spatial interaction analysis.

## Contact

For questions or issues with the expanded analysis, please refer to the main project repository or contact the analysis team.

---

**Last Updated**: 2025-10-25
**Version**: 1.0 (Expanded Comprehensive Analysis)
