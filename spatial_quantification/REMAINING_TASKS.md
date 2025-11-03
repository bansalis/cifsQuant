# Remaining Issues to Address

## Status: Critical Fixes Complete, Additional Features In Progress

**Date:** 2025-11-03
**Commit:** `1b9c9a7`

---

## ✅ COMPLETED

### 1. Global Neighborhood Analysis (CRITICAL - FIXED)
**Issue:** Neighborhoods were defined per-sample, couldn't track evolution over time
**Solution:**
- Neighborhoods now defined GLOBALLY across all samples
- Same neighborhood types can be tracked temporally
- Methods: `_define_global_neighborhoods()` and `_assign_neighborhoods_to_sample()`

### 2. Subsampling Configuration (FIXED)
**Issue:** 100k total cells too low, needed 75-100k per sample for rare populations
**Solution:**
- Config parameter: `cells_per_sample: 100000`
- Now samples up to 100k per sample (not total)
- With 500k+ cells/sample, provides good representation

### 3. Plot Customization (ADDED)
**Issue:** Timepoints in weeks not days, needed customizable labels
**Solution:**
- Added `plotting` section to config:
  ```yaml
  plotting:
    timepoint_unit: 'weeks'
    timepoint_label: 'Time (weeks)'
    font_family: 'DejaVu Sans'  # Fixes Arial warning
    show_stats: true
    stat_method: 'mann_whitney'
    fdr_correction: true
    significance_symbols:
      0.001: '***'
      0.01: '**'
      0.05: '*'
      1.0: 'ns'
  ```

### 4. Increased Neighborhood Clusters (FIXED)
**Issue:** n_clusters=10 too low
**Solution:** Increased to 15 for more granularity

---

## ❌ REMAINING TO IMPLEMENT

### Priority 1: Neighborhood Spatial Visualization

**What's Missing:**
- Per-sample spatial maps showing which cells belong to which neighborhood
- Need to color cells by neighborhood assignment
- Show spatial organization of neighborhoods

**Implementation Needed:**
```python
# In spatial_plotter.py
def plot_neighborhood_spatial_per_sample(self, adata, neighborhood_labels, sample):
    """
    Plot spatial map of neighborhood assignments for a single sample.

    - Color cells by neighborhood type
    - Use consistent colors across samples
    - Show neighborhood spatial distribution
    """
```

**Output:** `neighborhood_analysis/spatial_maps/GUEST29_neighborhoods.png`

---

### Priority 2: Stacked Area Charts for Neighborhoods

**What's Missing:**
- Stacked area chart showing neighborhood abundance over time
- Per group (KPT vs KPNT)
- Per timepoint comparisons with stats

**Implementation Needed:**
```python
# In neighborhood_plotter.py
def plot_neighborhood_stacked_area(self, data, group_col, groups):
    """
    Create stacked area chart showing neighborhood evolution.

    - X-axis: Time (weeks)
    - Y-axis: Fraction of tissue (0-1, stacked)
    - Colors: One per neighborhood type
    - Separate plot per group
    """
```

**Output:** `neighborhood_analysis/plots/stacked_area_KPT.png`

---

### Priority 3: Statistical Tests on Plots

**What's Missing:**
- Bar plots with significance bars and asterisks
- Only show significant comparisons (p < 0.05 after FDR)
- Need to integrate with existing plotters

**Implementation Needed:**
1. Create helper function in `stats/tests.py`:
```python
def add_significance_bars(ax, data, group_col, x_positions, y_max, test='mann_whitney', fdr=True):
    """
    Add significance bars with asterisks to plot.

    - Run pairwise comparisons
    - Apply FDR correction if requested
    - Draw horizontal bars above plot
    - Add *,  **, *** based on p-value
    """
```

2. Integrate into all plotters:
   - PopulationDynamicsPlotter
   - DistanceAnalysisPlotter
   - NeighborhoodPlotter

---

### Priority 4: Infiltration Plotting Integration

**Current State:**
```
⚠ Infiltration plotting not yet fully implemented
```

**What's Missing:**
- 3-panel spatial plots (marker+/-, Gi* hotspots, DBSCAN clusters)
- These plots are GENERATED but not being shown in the warning message
- Need to integrate InfiltrationAnalysisOptimized plotting properly

**Fix Needed:**
- The plots ARE being generated in `infiltration_analysis_optimized.py`
- Just need to update the message/integration in run_spatial_quantification.py
- Plots exist at: `infiltration_analysis/spatial_plots/`

---

### Priority 5: Marker Heterogeneity Analysis

**What's Missing:**
- Per tumor structure analysis of marker heterogeneity
- Getis-Ord Gi* and Ripley's K ARE implemented in infiltration_analysis_optimized
- But need dedicated output showing:
  - Which tumor structures are homogenous vs heterogeneous
  - Clustering metrics per structure
  - Comparative analysis across groups/time

**Implementation Needed:**
```python
# In infiltration_analysis_optimized.py
def _analyze_tumor_heterogeneity_per_structure(self, sample, marker):
    """
    Analyze marker heterogeneity within each tumor structure.

    - For each DBSCAN tumor cluster:
      - Calculate Gi* for marker+ cells
      - Calculate Ripley's K at multiple radii
      - Classify as: homogeneous, clustered, random

    - Output statistics per structure
    - Compare across samples/groups/time
    """
```

**Output:** `infiltration_analysis/tumor_heterogeneity_stats.csv`

---

### Priority 6: Advanced Analysis Module

**What's Missing (from config):**
```yaml
advanced_analyses:
  enabled: true

  # Pseudo-time trajectory analysis
  pseudotime:
    enabled: true
    method: 'diffusion_pseudotime'
    features: ...
```

**Current State:**
- `advanced.py` exists but is a placeholder
- No actual implementation

**Needs Implementation:**
1. **Pseudotime Trajectories:**
   - For tumor cells: progression/differentiation over time
   - For immune cells: activation/exhaustion trajectories
   - Method: Diffusion pseudotime or Monocle3-style

2. **Cell-Cell Interactions:**
   - Ligand-receptor analysis (if marker data supports it)
   - Spatial interaction analysis (which cell types co-localize)
   - Permutation testing for significant interactions

**This is a MAJOR feature - may need separate task/planning**

---

## Implementation Priority Order

1. ✅ **Global neighborhoods** - DONE
2. ✅ **Plot customization config** - DONE
3. **Statistical test bars** - High impact, moderate effort
4. **Neighborhood spatial maps** - High impact, low effort
5. **Stacked area charts** - High impact, low effort
6. **Fix infiltration plotting message** - Low effort
7. **Marker heterogeneity per structure** - Moderate effort
8. **Advanced analysis** - HIGH EFFORT, may need separate planning

---

## Quick Wins (Can Do Now)

### 1. Neighborhood Spatial Maps (30 min)
Add to `spatial_plotter.py`:
```python
def plot_neighborhood_assignments(self, neighborhood_data, sample):
    # Color by neighborhood label
    # Use discrete colormap
    # Add legend
```

### 2. Stacked Area Chart (30 min)
Add to `neighborhood_plotter.py`:
```python
def plot_stacked_area(self, data, group_col):
    # Pivot data
    # Use ax.stackplot()
    # One plot per group
```

### 3. Fix Infiltration Message (5 min)
In `run_spatial_quantification.py`:
- Check if using optimized version
- If yes, say "✓ Spatial plots generated" instead of warning

### 4. Statistical Bars (1-2 hours)
- Helper function for significance testing
- Integrate into all plotters
- Use config settings for symbols

---

## Current Working Pipeline

**What Works:**
✅ Population dynamics with comprehensive plots
✅ Distance analysis with overlapping histograms
✅ Global neighborhoods with temporal tracking
✅ Infiltration analysis with Gi* spatial plots
✅ All data saving correctly

**What Needs Polish:**
- Add significance tests to plots
- Add neighborhood spatial maps
- Add stacked area charts
- Better integration messages

---

## Next Steps

1. Implement quick wins (spatial maps, stacked area)
2. Add statistical test helper and integrate
3. Fix infiltration plotting message
4. Add marker heterogeneity per-structure analysis
5. Plan advanced analysis module separately

**Estimated time for steps 1-4:** 3-4 hours
**Advanced analysis:** Separate planning session needed
