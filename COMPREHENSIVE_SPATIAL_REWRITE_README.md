# Comprehensive Spatial Analysis - Complete Rewrite

## Overview

This is a **ground-up rewrite** of the spatial analysis pipeline addressing ALL previous issues and implementing ALL requirements.

**Date:** 2025-10-31
**Status:** Production-ready

---

## Key Features

### ✅ Dual-Level Analysis
Every analysis is performed at **BOTH** levels:
1. **Per-tumor-structure level**: Individual tumor structures analyzed separately
2. **Per-sample level**: Aggregated across all structures in a sample

### ✅ Comprehensive Distance Analysis
- Calculates distances from **EVERY tumor cell type** to **EVERY immune population type**
- Tracks over time
- Compares KPT vs KPNT
- Available at both structure and sample levels

### ✅ Per-Marker Region Analysis
- Analyzes **EVERY tumor marker** (pERK+, Ki67+, NINJA+, etc.)
- Quantifies marker+ vs marker- fractions
- Tracks temporal dynamics
- Compares KPT vs KPNT
- Available at both structure and sample levels

### ✅ Comprehensive Infiltration Analysis
- Quantifies infiltration for **EVERY immune population**
- Multiple boundary widths (30μm, 100μm, 200μm)
- Temporal dynamics
- KPT vs KPNT comparison
- Available at both structure and sample levels

### ✅ Proper Comparisons
- **PRIMARY:** KPT vs KPNT (main biological comparison)
- **SECONDARY:** 4-way comparison (KPT-cis, KPT-trans, KPNT-cis, KPNT-trans)
- **NEVER:** cis vs trans alone (biologically invalid)

### ✅ Complete Plot Generation
For EVERY analysis:
1. **Individual plots**:
   - Temporal plot (over time, KPT vs KPNT)
   - Boxplot (KPT vs KPNT comparison)
   - Scatter plot (raw data points)
2. **Combo plots**:
   - 3×3 grid with all perspectives
   - Sample-level + structure-level
   - KPT vs KPNT + 4-way genotype
3. **Summary plots**:
   - Heatmaps across all combinations
   - Bar plots with summary statistics

### ✅ Proper Organization
- Only creates directories that will have content
- Clear hierarchical structure
- Individual + combo plots separated
- CSV data files for all analyses

---

## Installation & Requirements

```bash
# Required packages
pip install scanpy pandas numpy scipy scikit-learn matplotlib seaborn shapely
```

---

## Usage

### Quick Start

```bash
python run_comprehensive_rewrite.py \
    --input manual_gating_output/gated_data.h5ad \
    --metadata sample_metadata.csv \
    --output comprehensive_spatial_output_new
```

### Full Options

```bash
python run_comprehensive_rewrite.py \
    --input manual_gating_output/gated_data.h5ad \
    --metadata sample_metadata.csv \
    --output comprehensive_spatial_output_new \
    --tumor-eps 30 \
    --tumor-min-samples 10 \
    --tumor-min-size 50
```

### Parameters

- `--input`: Path to h5ad file with gated data (required)
- `--metadata`: Path to sample metadata CSV (required)
- `--output`: Output directory (default: `comprehensive_spatial_output_new`)
- `--tumor-eps`: DBSCAN epsilon for tumor clustering (default: 30μm)
- `--tumor-min-samples`: DBSCAN min_samples (default: 10)
- `--tumor-min-size`: Minimum tumor structure size (default: 50 cells)

---

## Analysis Phases

### Phase 1: Tumor Structure Detection
- Uses DBSCAN to identify individual tumor structures
- Per-sample clustering
- Stores structure index for dual-level analysis

**Output:**
- Structure index with spatial properties
- Printed summary statistics

### Phase 2: Comprehensive Distance Analysis
- Calculates distances from EVERY tumor type to EVERY immune type
- Two levels:
  - **Per-structure**: Distance within each tumor structure
  - **Per-sample**: Distance aggregated across sample
- Metrics: mean, median, min, max, std

**Output:**
- `distance_analysis/comprehensive_distances.csv`
- Columns: tumor_type, immune_type, structure_id, sample_id, main_group, genotype_full, timepoint, level, mean_distance, median_distance, min_distance, max_distance, std_distance, n_tumor_cells, n_immune_cells

**Plots:**
- Individual plots for each tumor-immune pair:
  - `*_temporal.png`: Over time (KPT vs KPNT)
  - `*_boxplot.png`: KPT vs KPNT comparison
  - `*_scatter.png`: Raw data points
- Combo plots:
  - `*_combo.png`: 3×3 grid (sample-level, structure-level, 4-way)
- Summary:
  - `distance_summary_heatmaps.png`: Heatmaps of all pairs

### Phase 3: Per-Marker Region Temporal Analysis
- Analyzes EVERY tumor marker population
- Quantifies marker+ and marker- fractions
- Tracks over time
- Two levels: per-structure and per-sample

**Output:**
- `marker_regions/marker_regions_temporal.csv`
- Columns: marker_name, marker_type (positive/negative), level, structure_id, sample_id, main_group, genotype_full, timepoint, n_cells, fraction, density, total_cells

**Plots:**
- Individual plots for each marker:
  - `*_temporal.png`: Marker+ fraction over time
  - `*_boxplot.png`: KPT vs KPNT comparison
  - `*_scatter.png`: Raw data points
- Combo plots:
  - `*_combo.png`: 3×3 grid (sample-level, structure-level, 4-way)

### Phase 4: Comprehensive Infiltration Analysis
- Quantifies infiltration for EVERY immune population
- Multiple boundary widths: 30μm, 100μm, 200μm
- Tracks over time
- Two levels: per-structure and per-sample

**Output:**
- `infiltration_analysis/comprehensive_infiltration.csv`
- Columns: immune_type, boundary_width, level, structure_id, sample_id, main_group, genotype_full, timepoint, n_immune_in_boundary, n_immune_total, infiltration_density, infiltration_fraction, boundary_area, tumor_size

**Plots:**
- Individual plots for each immune population:
  - `*_w{width}_temporal.png`: Infiltration over time
  - `*_w{width}_boxplot.png`: KPT vs KPNT comparison
  - `*_w{width}_scatter.png`: Raw data points
- Combo plots:
  - `*_combo.png`: 3×3 grid (sample-level, multi-width, 4-way)

---

## Output Structure

```
comprehensive_spatial_output_new/
├── distance_analysis/
│   └── comprehensive_distances.csv
├── marker_regions/
│   └── marker_regions_temporal.csv
├── infiltration_analysis/
│   └── comprehensive_infiltration.csv
└── figures/
    ├── distance_analysis/
    │   ├── individual_plots/
    │   │   ├── {tumor}_to_{immune}_temporal.png
    │   │   ├── {tumor}_to_{immune}_boxplot.png
    │   │   └── {tumor}_to_{immune}_scatter.png
    │   ├── combo_plots/
    │   │   └── {tumor}_to_{immune}_combo.png
    │   └── distance_summary_heatmaps.png
    ├── marker_regions/
    │   ├── individual_plots/
    │   │   ├── {marker}_temporal.png
    │   │   ├── {marker}_boxplot.png
    │   │   └── {marker}_scatter.png
    │   └── combo_plots/
    │       └── {marker}_combo.png
    └── infiltration_analysis/
        ├── individual_plots/
        │   ├── {immune}_w{width}_temporal.png
        │   ├── {immune}_w{width}_boxplot.png
        │   └── {immune}_w{width}_scatter.png
        └── combo_plots/
            └── {immune}_combo.png
```

---

## Key Differences from Previous Pipeline

| Issue | Previous Pipeline | New Pipeline |
|-------|------------------|--------------|
| **Empty directories** | Many empty folders created | Only directories with content created |
| **Missing individual plots** | Only combo plots | Individual + combo for everything |
| **Wrong comparisons** | cis vs trans comparisons | ONLY KPT vs KPNT (+ 4-way) |
| **Missing distance analysis** | Limited distance calculations | ALL tumor types to ALL immune types |
| **Missing marker analysis** | Not comprehensive | EVERY marker over time |
| **Single-level analysis** | Only sample-level or structure-level | BOTH levels for everything |
| **Plot variety** | Only one plot type | Scatter + bar + temporal + heatmap |
| **Organization** | Messy structure | Clear hierarchical organization |

---

## Data Requirements

### Input h5ad File
Must contain:
- **Spatial coordinates**: `adata.obsm['spatial']` or `adata.obs['X_centroid']` + `adata.obs['Y_centroid']`
- **Population columns**: `is_*` columns (e.g., `is_Tumor`, `is_CD8_T_cells`, `is_Tumor_PERK_positive`)
- **Sample IDs**: `adata.obs['sample_id']`

### Metadata CSV
Must contain:
- **sample_id**: Sample identifier (will be uppercased)
- **group**: Group name (must contain 'KPT' or 'KPNT', and 'cis' or 'trans')
- **timepoint**: Numeric timepoint
- **treatment** (optional): Treatment condition

Example:
```csv
sample_id,group,timepoint,treatment
GUEST29,KPNT cis,8,control
GUEST30,KPNT cis,10,control
GUEST45,KPT Het cis,12,treatment
GUEST46,KPT Het trans,14,treatment
```

---

## Comparison Logic

### PRIMARY Comparison: KPT vs KPNT
This is the **main biological comparison** across all analyses:
- Used in all temporal plots
- Used in all boxplots
- Statistical tests always KPT vs KPNT

### SECONDARY Comparison: 4-Way Genotype
For detailed stratification:
- KPT-cis
- KPT-trans
- KPNT-cis
- KPNT-trans

**CRITICAL:** We NEVER compare cis vs trans alone, as KPT-cis and KPNT-cis are fundamentally different populations.

---

## Extending the Pipeline

### Adding New Analyses

To add a new analysis module:

1. Add analysis method to `ComprehensiveSpatialAnalysisRewrite` class:
```python
def analyze_my_new_analysis(self) -> pd.DataFrame:
    """
    My new analysis.

    Returns DataFrame with columns including:
    - level ('structure' or 'sample')
    - sample_id, structure_id
    - main_group, genotype_full, timepoint
    - [your metrics]
    """
    # Implement dual-level analysis
    results = []

    # Per-structure level
    if self.tumor_structures is not None:
        for _, structure in self.tumor_structures.iterrows():
            # Calculate metrics
            results.append({...})

    # Per-sample level
    for sample_id in self.adata.obs['sample_id'].unique():
        # Calculate metrics
        results.append({...})

    df = pd.DataFrame(results)
    self.results['my_analysis'] = df
    return df
```

2. Add plotting method to `ComprehensivePlotGenerator` class:
```python
def plot_my_new_analysis(self):
    """Generate ALL plots for my analysis."""
    # Individual plots
    self._plot_my_individual(...)

    # Combo plots
    self._plot_my_combo(...)
```

3. Call in `run_all_analyses()` and `generate_all_plots()`

---

## Troubleshooting

### No populations detected
- Check that your h5ad file has `is_*` columns
- Verify columns are boolean type
- Check that populations have cells (count > 0)

### Missing metadata columns
- Ensure metadata CSV has: sample_id, group, timepoint
- Check that group contains 'KPT' or 'KPNT'
- Verify sample_id matches between h5ad and metadata

### Empty plots
- Check that there's data for both KPT and KPNT groups
- Verify timepoints have sufficient data
- Check console output for warnings

### Shapely import errors
- Install shapely: `pip install shapely`
- For conda: `conda install shapely`

---

## Performance Notes

- Distance calculations scale with cell count (use spatial indexing for speed)
- Structure-level analysis is more computationally intensive than sample-level
- Plotting can take time with many populations (one plot per tumor-immune pair)
- Large datasets: Consider subsampling for initial exploratory runs

---

## Citation

If you use this pipeline, please cite:

```
Comprehensive Spatial Analysis Pipeline v2.0
Complete rewrite addressing dual-level analysis requirements
Date: 2025-10-31
```

---

## Support

For issues or questions:
1. Check this README
2. Review console output for error messages
3. Verify input data format
4. Check that required packages are installed

---

## Version History

### v2.0 (2025-10-31) - Complete Rewrite
- ✅ Dual-level analysis (per-structure + per-sample)
- ✅ Comprehensive distance analysis (all tumor-immune pairs)
- ✅ Per-marker region analysis (all markers)
- ✅ Comprehensive infiltration analysis (all immune pops)
- ✅ ONLY KPT vs KPNT comparisons
- ✅ Individual + combo plots for everything
- ✅ Scatter + bar plots
- ✅ Proper directory organization
- ✅ Complete documentation

### v1.x (Previous)
- ❌ Single-level analysis only
- ❌ Limited distance calculations
- ❌ Missing per-marker analysis
- ❌ Inappropriate cis vs trans comparisons
- ❌ Missing individual plots
- ❌ Empty directories
- ❌ Poor organization

---

## Files

- `comprehensive_spatial_rewrite.py`: Main analysis module
- `run_comprehensive_rewrite.py`: Runner script
- `COMPREHENSIVE_SPATIAL_REWRITE_README.md`: This file
