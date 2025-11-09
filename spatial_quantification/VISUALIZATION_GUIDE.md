# Spatial Quantification Visualization Guide

## Overview

The spatial quantification pipeline now includes comprehensive visualization capabilities across all analyses. This guide describes all available plots and how to interpret them.

---

## 1. Pseudotime Differentiation Plots

**Location**: `pseudotime_analysis/plots/`

### Differentiation Trajectories (`differentiation_trajectories.png`)

**What it shows**: How marker expression changes along inferred differentiation trajectory

**Interpretation**:
- X-axis: Pseudotime (0 = early state, 1 = late state)
- Y-axis: % of cells expressing each marker
- Lines show KPT vs KPNT progression patterns
- Identifies marker dynamics during tumor cell differentiation

**Biological insights**:
- Early markers (high at pseudotime=0): Progenitor/stem-like state
- Late markers (high at pseudotime=1): Differentiated state
- Diverging trajectories between groups suggest different differentiation programs

### PC Space (`pseudotime_pc_space.png`)

**What it shows**: Cells in principal component space colored by markers

**Panels**:
1. **Pseudotime**: Cells colored by pseudotime value (progression along trajectory)
2. **pERK**: Red = pERK+, Gray = pERK-
3. **NINJA**: Red = NINJA+, Gray = NINJA-
4. **Ki67**: Red = Ki67+, Gray = Ki67+

**Interpretation**:
- Smooth gradients suggest continuous differentiation
- Discrete clusters suggest distinct cell states
- Marker overlap reveals coexpression patterns

---

## 2. UMAP Visualizations

**Location**: `umap_visualization/plots/`

### UMAP Clusters (`umap_clusters.png`)

**What it shows**: Unsupervised clustering of all cells in UMAP space

**Interpretation**:
- Each color = distinct cell population
- Spatial separation = transcriptional differences
- Overlapping clusters = similar cell types
- Use to identify major cell populations and rare subtypes

**Example clusters**:
- Cluster 0: Tumor cells (TOM+)
- Cluster 1-2: CD8+ T cells
- Cluster 3: CD45+ non-T immune
- Clusters 4-5: pERK+ tumor subpopulations
- Clusters 6-7: NINJA+ tumor subpopulations

### UMAP by Markers (`umap_markers.png`)

**What it shows**: Distribution of marker expression across cell populations

**Panels**: One per marker (pERK, NINJA, Ki67, CD45, CD3, CD8)

**Interpretation**:
- Red = marker positive cells
- Gray = marker negative cells
- Localized red = marker defines specific population
- Scattered red = marker expressed across populations

**Key patterns to look for**:
1. **Exclusive expression**: pERK+ and NINJA+ in distinct regions → mutually exclusive subpopulations
2. **Overlapping expression**: pERK+ overlaps Ki67+ → proliferative pERK+ cells
3. **Gradient expression**: Smooth transition → differentiation continuum
4. **Bimodal expression**: Two separate red regions → two distinct marker+ populations

### UMAP by Phenotypes (`umap_phenotypes.png`)

**What it shows**: Gated phenotypes mapped onto UMAP space

**Panels**: Tumor, CD45+, CD3+, CD8+, CD4+, pERK+ tumor, NINJA+ tumor, Ki67+ tumor

**Interpretation**:
- Validates that UMAP separates known phenotypes
- Shows phenotype diversity within major cell types
- Reveals rare phenotype combinations

---

## 3. Infiltration Analysis Plots

**Location**: `infiltration_analysis/plots/`

### Infiltration Trends Over Time (`infiltration_trends_over_time.png`)

**What it shows**: How immune infiltration changes over time across zones

**Panels**: One per immune population (CD8, CD3, CD45, etc.)

**Lines**:
- Different colors for zones (within tumor, 0-50μm, 50-100μm, etc.)
- Solid vs dashed for KPT vs KPNT

**Interpretation**:
- Increasing trends = progressive infiltration
- Diverging groups = differential immune response
- Zone-specific patterns = spatial organization of infiltration

### Heterogeneity Metrics Over Time (`heterogeneity_metrics_over_time.png`)

**What it shows**: Spatial clustering of markers (pERK, NINJA, Ki67) over time

**Metrics**:
1. **Mean Gi* Z-score**: Average local clustering (higher = more clustered)
2. **# Significant Hotspots**: Number of significant marker+ clusters
3. **Ripley's L (30μm)**: Short-range clustering
4. **DBSCAN Clustering Score**: Fraction of cells in dense clusters

**Interpretation**:
- High heterogeneity = distinct marker+ and marker- regions
- Low heterogeneity = stochastic/random marker expression
- Temporal changes = evolving tumor spatial organization

### Zone-Specific Infiltration (`zone_specific_infiltration.png`)

**What it shows**: Immune infiltration in marker+ vs marker- regions

**Panels**: One per immune population per marker

**Lines**:
- Solid: Infiltration in marker+ regions
- Dashed: Infiltration in marker- regions

**Interpretation**:
- Higher in marker+ = preferential infiltration to marker+ tumors
- Higher in marker- = immune exclusion from marker+ regions
- Converging over time = equalizing infiltration patterns

### Summary Heatmap (`infiltration_summary_heatmap.png`)

**What it shows**: Overview of infiltration density across all populations and timepoints

**Interpretation**:
- Color intensity = infiltration density
- Patterns across rows = population-specific dynamics
- Patterns across columns = temporal evolution

---

## 4. Enhanced Neighborhood Plots

**Location**: `enhanced_neighborhoods/plots/`

### Regional Infiltration (`{marker}_regional_infiltration.png`)

**What it shows**: Two-step infiltration analysis for marker+ vs marker- regions

**Panel 1 (left)**: % of immune cells in region
**Panel 2 (right)**: Mean distance to region

**Lines**:
- Solid: marker+ regions
- Dashed: marker- regions

**Interpretation**:
- High % in marker+ = immune cells preferentially located in marker+ regions
- Lower distance to marker+ = immune cells closer to marker+ regions
- Diverging patterns = spatial selectivity of immune infiltration

### Regional Neighborhoods (`{marker}_regional_neighborhoods.png`)

**What it shows**: Cell type composition within marker+ vs marker- regions

**Upper panels**: marker+ region composition
**Lower panels**: marker- region composition

**Interpretation**:
- Compare upper vs lower to see compositional differences
- High CD8 in marker+ = CD8 enrichment in marker+ tumors
- Temporal changes = evolving neighborhood structure

### Per-Cell Neighborhoods (`{marker}_per_cell_neighborhoods.png`)

**What it shows**: Immediate neighborhood of individual marker+ vs marker- cells

**Lines**:
- Solid: marker+ cells
- Dashed: marker- cells

**Interpretation**:
- Reveals cell-cell interaction patterns
- Higher immune in marker+ neighborhoods = marker+ cells more immunogenic
- Similar profiles = marker doesn't affect local microenvironment

---

## 5. Per-Tumor Analysis Plots

**Location**: `per_tumor_analysis/plots/`

### Tumor Size (`n_tumor_cells_per_tumor.png`)

**Panel 1**: Boxplots per timepoint and group
**Panel 2**: Temporal trends (mean ± SEM)

**Interpretation**:
- Individual dots = individual tumor structures
- Tracks tumor growth dynamics
- Group differences in tumor size

### Marker Percentages (`all_marker_percentages_per_tumor.png`)

**Panels**: pERK+, NINJA+, Ki67+ per tumor

**Interpretation**:
- Heterogeneity between tumors at same timepoint
- Temporal dynamics of marker expression
- Group-specific marker frequencies

### Growth-Normalized pERK (`growth_normalized_pERK.png`)

**Panels**:
1. pERK / Ki67 ratio
2. pERK - Ki67 (%)
3. pERK residual (regression-based)

**Interpretation**:
- Accounts for tumor growth rate (Ki67)
- Positive residual = pERK expression exceeds growth-predicted level
- Group differences = differential pERK activation independent of proliferation

---

## 6. Coexpression Analysis Plots

**Location**: `coexpression_analysis/plots/`

### Temporal Trends (`coexpression_temporal_trends.png`)

**What it shows**: Frequencies of single and coexpressed markers over time

**Interpretation**:
- High coexpression = markers coordinately regulated
- Low coexpression = independent marker expression
- Temporal changes = evolving coexpression patterns

### Coexpression Heatmap (`coexpression_heatmap.png`)

**What it shows**: Jaccard index matrix for marker pairs

**Interpretation**:
- 1.0 = perfect coexpression
- 0.0 = mutually exclusive
- Reveals marker relationships

### Triple Coexpression (`triple_coexpression_stacked.png`)

**What it shows**: All 8 combinations of 3 markers (pERK+/-, NINJA+/-, Ki67+/-)

**Interpretation**:
- Identifies rare vs common marker combinations
- Shows tumor subpopulation frequencies
- Temporal dynamics of subpopulations

---

## How to Use These Plots

### For Initial Data Exploration

1. **Start with UMAP** to understand cell populations
2. **Check infiltration trends** for immune dynamics
3. **Look at per-tumor metrics** for growth and marker frequencies

### For Hypothesis Testing

1. **Infiltration analysis** for immune-tumor interactions
2. **Heterogeneity metrics** for spatial organization
3. **Coexpression analysis** for marker relationships

### For Mechanistic Insights

1. **Enhanced neighborhoods** for cell-cell interactions
2. **Pseudotime** for differentiation programs
3. **Growth-normalized metrics** for activity vs proliferation

### For Publication

All plots are generated at 300 DPI and are publication-ready. Key figures to include:

- **Figure 1**: UMAP clusters + markers (population overview)
- **Figure 2**: Infiltration trends (immune response)
- **Figure 3**: Heterogeneity metrics (spatial organization)
- **Figure 4**: Enhanced neighborhoods (cell interactions)
- **Figure 5**: Pseudotime trajectories (differentiation)

---

## Configuration

All visualizations are controlled via `spatial_config.yaml`:

```yaml
# Enable/disable specific visualizations
pseudotime_analysis:
  enabled: true
  generate_plots: true

umap_visualization:
  enabled: true
  n_clusters: 10
  subsample: 100000

# Marker and phenotype lists are configurable
```

---

## Troubleshooting

### No plots generated
- Check config: `enabled: true` and `generate_plots: true`
- Check console for error messages
- Verify input data has required phenotypes

### UMAP requires umap-learn
```bash
pip install umap-learn
```

### Empty plots
- Insufficient cells: UMAP requires >100 cells
- Missing phenotypes: Check manual gating output
- No marker+ cells: Verify gating thresholds

### Plots cut off
- All plots use `bbox_inches='tight'`
- If still cut off, increase figure size in plotter code

---

## Summary

This visualization suite provides:
- **7 analysis modules** with comprehensive plots
- **20+ plot types** covering all spatial metrics
- **Fully configurable** via YAML
- **Publication-ready** 300 DPI output
- **Automated generation** in pipeline

Use these visualizations to understand:
- Cell population structure (UMAP)
- Immune infiltration dynamics (infiltration)
- Spatial organization (heterogeneity)
- Cell-cell interactions (neighborhoods)
- Differentiation programs (pseudotime)
- Marker coexpression (coexpression)
- Tumor heterogeneity (per-tumor)
