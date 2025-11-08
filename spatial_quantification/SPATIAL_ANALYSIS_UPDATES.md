# Spatial Quantification Pipeline Updates

## Summary

Major enhancements to the spatial quantification pipeline with improved per-tumor metrics, coexpression analysis, enhanced neighborhood analysis, and pseudotime trajectory inference.

## Date: 2025-11-08

---

## What's New

### 1. Per-Tumor Analysis (`per_tumor_analysis.py`)

**Problem Solved:** Previous analyses were per-sample, making it unclear whether counts were per tumor structure or total. Size graphs showed aggregate data rather than per-tumor distributions.

**New Features:**
- **Tumor Size Metrics:** Number of tumor cells per individual tumor structure (boxplot/line plot format)
- **Marker Percentages Per Tumor:** % pERK+, % NINJA+, % Ki67+ calculated for each individual tumor
- **Growth-Rate Normalization:** pERK+ adjusted by tumor growth rate (Ki67+):
  - Simple ratio: pERK/Ki67
  - Difference: pERK - Ki67 (excess pERK beyond proliferation)
  - Regression residuals: pERK corrected for Ki67 dependency per timepoint/group
- **Per-Tumor Infiltration:** Immune cell counts and densities per individual tumor structure

**Output Files:**
- `per_tumor_metrics.csv` - Basic size metrics per tumor
- `per_tumor_marker_percentages.csv` - Marker percentages per tumor
- `per_tumor_growth_normalized.csv` - Growth-corrected pERK metrics
- `per_tumor_infiltration.csv` - Infiltration metrics per tumor

**Visualizations:**
- Boxplots showing distribution across tumors
- Line plots with temporal trends (mean ± SEM)
- Individual tumor data points overlaid

---

### 2. Coexpression Analysis (`coexpression_analysis.py`)

**Problem Solved:** Need to understand co-occurrence patterns of pERK, NINJA, and Ki67 markers.

**Features:**
- **Pairwise Coexpression:**
  - pERK+ AND NINJA+
  - pERK+ AND Ki67+
  - NINJA+ AND Ki67+
  - Multiple metrics: count, % of tumor, % of marker1, % of marker2, Jaccard index
- **Triple Coexpression:** All combinations (pERK+NINJA+Ki67+, pERK only, etc.)
- **Temporal Dynamics:** How coexpression changes over time
- **Group Comparisons:** KPT vs KPNT differences

**Output Files:**
- `single_marker_frequencies.csv` - Individual marker frequencies
- `pairwise_coexpression.csv` - Pairwise coexpression metrics
- `triple_coexpression.csv` - All combination patterns

**Visualizations:**
- Temporal trends of single markers
- Pairwise coexpression over time
- Stacked bar charts showing combination patterns
- Heatmap of Jaccard similarity indices

---

### 3. Enhanced Neighborhood Analysis (`enhanced_neighborhood_analysis.py`)

**Problem Solved:** Need deeper analysis of pERK+/- and NINJA+/- regional differences in infiltration and neighborhood composition.

**Features:**
- **Region Identification:**
  - Detect distinct pERK+ and pERK- regions using DBSCAN
  - Detect distinct NINJA+ and NINJA- regions
  - Count number of regions per tumor
- **Two-Step Infiltration:**
  - **Step 1:** % of immune cells within marker+ vs marker- regions (within 50 μm)
  - **Step 2:** Mean distance/depth of immune cells to marker+ vs marker- regions
- **Regional Neighborhood Composition:**
  - Cell type composition around marker+ regions
  - Cell type composition around marker- regions
  - Direct comparison between the two
- **Per-Cell Neighborhoods:**
  - For each pERK+ cell: what is the local neighborhood composition?
  - For each pERK- cell: what is the local neighborhood composition?
  - Statistical comparison of the two

**Output Files:**
- `pERK_region_identification.csv` - Number and size of pERK+/- regions
- `NINJA_region_identification.csv` - Number and size of NINJA+/- regions
- `pERK_regional_infiltration.csv` - Infiltration comparison (+ vs -)
- `NINJA_regional_infiltration.csv` - Infiltration comparison (+ vs -)
- `pERK_regional_neighborhoods.csv` - Neighborhood composition per region
- `NINJA_regional_neighborhoods.csv` - Neighborhood composition per region
- `pERK_per_cell_neighborhoods.csv` - Aggregated per-cell analysis
- `NINJA_per_cell_neighborhoods.csv` - Aggregated per-cell analysis

---

### 4. Pseudotime Differentiation Analysis (`pseudotime_analysis.py`)

**Problem Solved:** Need to infer differentiation trajectories and ordering of tumor cell states.

**Features:**
- **Marker-Based Trajectory:** Uses pERK, NINJA, Ki67 expression + spatial features
- **PCA Dimensionality Reduction:** Reduces to 3 principal components
- **Pseudotime Assignment:** First PC used as pseudotime [0, 1]
- **Trajectory Dynamics:** How marker expression changes along pseudotime

**Output Files:**
- `cell_pseudotime.csv` - Pseudotime value per cell with PC coordinates
- `trajectory_dynamics.csv` - Marker frequencies across pseudotime bins

**Use Cases:**
- Identify early vs late tumor cell states
- Understand progression of marker expression
- Find transitional cell populations

---

## Updated Existing Analyses

### Infiltration Analysis
**Clarification Added:** All counts are now clearly labeled as:
- `count` = number of immune cells in zone
- `structure_size` = number of cells in tumor structure
- Per-tumor structure, not per sample aggregate

**Files:** `infiltration_analysis_optimized.py` (line 202-203)

### Distance Analysis
**Clarification:** Distances calculated per sample, with clear per-sample metrics (mean, median, percentiles)

**Files:** `distance_analysis.py`

---

## How to Use

### Enable New Analyses

Edit `spatial_quantification/config/spatial_config.yaml`:

```yaml
# Per-tumor analysis (RECOMMENDED)
per_tumor_analysis:
  enabled: true
  generate_plots: true

# Coexpression analysis
coexpression_analysis:
  enabled: true

# Enhanced neighborhoods (marker-specific regions)
enhanced_neighborhoods:
  enabled: true
  neighborhood_radius: 50  # μm

# Pseudotime analysis
pseudotime_analysis:
  enabled: true
```

### Run Pipeline

```bash
cd /home/user/cifsQuant/spatial_quantification
python run_spatial_quantification.py --config config/spatial_config.yaml
```

### Output Structure

```
spatial_quantification_results/
├── per_tumor_analysis/
│   ├── per_tumor_metrics.csv
│   ├── per_tumor_marker_percentages.csv
│   ├── per_tumor_growth_normalized.csv
│   ├── per_tumor_infiltration.csv
│   └── plots/
│       ├── n_tumor_cells_per_tumor.png
│       ├── all_marker_percentages_per_tumor.png
│       └── growth_normalized_pERK.png
│
├── coexpression_analysis/
│   ├── single_marker_frequencies.csv
│   ├── pairwise_coexpression.csv
│   ├── triple_coexpression.csv
│   └── plots/
│       ├── single_marker_frequencies.png
│       ├── pairwise_coexpression.png
│       ├── triple_coexpression_patterns.png
│       └── coexpression_heatmap.png
│
├── enhanced_neighborhoods/
│   ├── pERK_region_identification.csv
│   ├── pERK_regional_infiltration.csv
│   ├── pERK_regional_neighborhoods.csv
│   ├── pERK_per_cell_neighborhoods.csv
│   ├── NINJA_region_identification.csv
│   ├── NINJA_regional_infiltration.csv
│   ├── NINJA_regional_neighborhoods.csv
│   └── NINJA_per_cell_neighborhoods.csv
│
└── pseudotime_analysis/
    ├── cell_pseudotime.csv
    └── trajectory_dynamics.csv
```

---

## Key Improvements Summary

### ✅ Fixed Issues
1. **Clarity:** Infiltration/distance counts now clearly labeled (per tumor structure)
2. **Per-Tumor Metrics:** Size graphs now show distribution across individual tumors (boxplots)
3. **Marker Percentages:** % pERK+, Ki67+, NINJA+ calculated per individual tumor

### ✅ New Capabilities
1. **Growth Normalization:** pERK+ adjusted by proliferation rate (3 methods)
2. **Coexpression:** Comprehensive analysis of marker co-occurrence
3. **Regional Analysis:** pERK+/- and NINJA+/- specific infiltration and neighborhoods
4. **Heterogeneity:** Identify distinct marker regions (stochastic vs regional)
5. **Per-Cell Neighborhoods:** Compare local environment of marker+ vs marker- cells
6. **Pseudotime:** Infer differentiation trajectories

---

## Advanced Features Implemented

### Heterogeneity Analysis
- Already present in `infiltration_analysis_optimized.py`:
  - **Getis-Ord Gi*** - Identifies hotspots (stochastic vs clustered)
  - **Ripley's K** - Multi-scale clustering quantification
  - **DBSCAN** - Identifies distinct spatial clusters
- Enhanced in `enhanced_neighborhood_analysis.py`:
  - Region identification for each marker
  - Comparison of marker+ vs marker- spatial patterns

### Two-Step Infiltration
Implemented in `enhanced_neighborhood_analysis.py`:
1. **Percent in region:** % immune cells within 50 μm of marker+ vs marker- regions
2. **Distance/depth:** Mean distance from immune cells to marker+ vs marker- cells
3. **Comparison:** Direct statistical comparison between regions

---

## Future Enhancements (Optional)

### SpaceBF Integration (mentioned in requirements)
Not yet implemented. To add:
- Install SpaceBF package
- Create `spatial_bayesian_factors.py` module
- Calculate Bayes factors for spatial associations
- Reference: https://github.com/sealx017/SpaceBF/

### Additional Analyses
- Spatial interaction networks
- Multi-scale heterogeneity metrics
- Spatial transcriptomics integration (if applicable)

---

## Technical Notes

### Dependencies
All new modules use existing dependencies:
- numpy, pandas, scipy
- sklearn (DBSCAN, PCA)
- matplotlib, seaborn

### Performance
- Uses optimized spatial methods (cKDTree, DBSCAN)
- Per-cell neighborhoods limited to 100k cells to avoid memory issues
- Tumor structure detection reused across analyses

### Compatibility
- Fully backward compatible
- New analyses are opt-in via config
- Existing analyses unchanged (only clarified)

---

## Testing

Before committing, test with:

```bash
# Check imports
python -c "from spatial_quantification.analyses.per_tumor_analysis import PerTumorAnalysis"
python -c "from spatial_quantification.analyses.coexpression_analysis import CoexpressionAnalysis"
python -c "from spatial_quantification.analyses.enhanced_neighborhood_analysis import EnhancedNeighborhoodAnalysis"
python -c "from spatial_quantification.analyses.pseudotime_analysis import PseudotimeAnalysis"

# Run pipeline (with subset of data for testing)
python run_spatial_quantification.py --config config/spatial_config.yaml
```

---

## Questions Addressed

1. **"counts are confusing are they per tumor number of number of cells"**
   → Fixed: Now explicitly per-tumor structure with clear labeling

2. **"size graphs i want it to be number of tumor cells PER tumor"**
   → Fixed: Per-tumor metrics with boxplot/line plot format

3. **"percent of pERK+ or Ki67+ PER tumor"**
   → Implemented in `per_tumor_marker_percentages.csv`

4. **"adjust pERK+ according to tumor growth rate"**
   → Implemented 3 methods in `per_tumor_growth_normalized.csv`

5. **"coexpression of the tumor subtypes"**
   → Full coexpression module with pairwise and triple analysis

6. **"heterogeneity, are markers stochastic or in distinct regions"**
   → Getis-Ord Gi*, Ripley's K, DBSCAN clustering already present
   → Enhanced with explicit region identification in new module

7. **"pERK+/- comparison, identify distinct regions, two-step infiltration"**
   → Comprehensive enhanced_neighborhoods module

8. **"per pERK+ cell and pERK- cell comparison"**
   → Per-cell neighborhood analysis implemented

9. **"pseudotime differentiation"**
   → PCA-based pseudotime module implemented

---

## Author
Claude AI (2025-11-08)
Based on requirements from cifsQuant project
