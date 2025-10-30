# VISUALIZATION INTEGRATION INSTRUCTIONS

## CRITICAL FIXES APPLIED

### 1. CORRECT BIOLOGICAL COMPARISONS
**NEVER** compare cis vs trans alone!

**CORRECT comparisons:**
- `main_group`: KPT vs KPNT (primary comparison)
- `genotype_full`: KPT-cis, KPT-trans, KPNT-cis, KPNT-trans (4-way detailed)

### 2. NEW COMPREHENSIVE PLOTTING MODULE
**File:** `comprehensive_plots_fixed.py`

Contains properly corrected plot functions:
- `plot_tumor_size_comprehensive()` - 3×3 grid
- `plot_infiltration_comprehensive()` - 3×3 grid per population
- `plot_marker_expression_comprehensive()` - 3×3 grid per marker
- `plot_neighborhoods_comprehensive()` - 3×3 grid with heatmaps

### 3. HOW TO INTEGRATE

Replace lines in `tumor_spatial_analysis_comprehensive.py`:

```python
# OLD (lines 1367-1415):
def _plot_tumor_size_comprehensive(self, size_df, stats_results):
    # ... shallow 2x2 plots with WRONG genotype comparisons ...

# NEW:
def _plot_tumor_size_comprehensive(self, size_df, stats_results):
    from comprehensive_plots_fixed import plot_tumor_size_comprehensive
    plot_tumor_size_comprehensive(size_df, self.output_dir)
```

Similarly for:
- `_plot_marker_expression_comprehensive()` → call `plot_marker_expression_comprehensive()`
- `_plot_infiltration_comprehensive()` → call `plot_infiltration_comprehensive()`
- `_plot_neighborhoods_comprehensive()` → call `plot_neighborhoods_comprehensive()`

### 4. EXPECTED OUTPUT

After integration, you'll get:

```
figures/
├── tumor_size_COMPREHENSIVE_FIXED.png              # 3×3 grid
├── infiltration_CD3_COMPREHENSIVE.png              # 3×3 per population
├── infiltration_CD8_COMPREHENSIVE.png
├── infiltration_CD45_positive_COMPREHENSIVE.png
├── infiltration_T_cells_COMPREHENSIVE.png
├── marker_CD3_COMPREHENSIVE.png                    # 3×3 per marker
├── marker_CD8_COMPREHENSIVE.png
├── marker_CD45_COMPREHENSIVE.png
├── marker_PERK_COMPREHENSIVE.png
├── marker_Ki67_COMPREHENSIVE.png
├── marker_AGFP_COMPREHENSIVE.png
├── marker_TOM_COMPREHENSIVE.png
└── neighborhoods_COMPREHENSIVE.png                  # 3×3 with heatmaps
```

### 5. EACH 3×3 FIGURE CONTAINS:

**Row 1 (KPT vs KPNT):**
- Temporal trends
- Boxplot comparisons
- Distribution or intensity

**Row 2 (4-way):**
- Temporal trends for all 4 groups
- Boxplot all 4 groups
- Heatmaps or scatter plots

**Row 3 (Detailed):**
- Per-timepoint violins
- Density distributions
- Regional or sub-analyses

### 6. TO RUN

```python
# In run_comprehensive_with_advanced.py, the plots are automatically generated
# when you call analysis.generate_all_visualizations()

# To manually test:
from comprehensive_plots_fixed import *
plot_tumor_size_comprehensive(size_df, output_dir)
plot_infiltration_comprehensive(metrics_df, output_dir)
plot_marker_expression_comprehensive(marker_df, output_dir)
```

### 7. ADVANCED PHASE PLOTS

**TODO:** Create comprehensive advanced plots similarly:
- pERK analysis: 3×3 grids
- NINJA analysis: 3×3 grids  
- Heterogeneity: 3×3 grids with entropy/CV metrics
- All with KPT vs KPNT and 4-way comparisons

---

## KEY POINT

**KPT-cis and KPNT-cis are VERY DIFFERENT biological contexts!**
Never lump them together as "cis". Always compare:
1. KPT vs KPNT (main effect of NK cells)
2. Within KPT: cis vs trans
3. Within KPNT: cis vs trans
4. Or 4-way: KPT-cis vs KPT-trans vs KPNT-cis vs KPNT-trans
