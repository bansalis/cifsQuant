# cifsQuant Analysis Guide

Reference for all spatial analysis modules — what each measures, when to enable it, key parameters, and output files.

---

## Run-order dependencies

`per_tumor_analysis` **must run before** analyses that use detected structure objects:
- `cluster_composition_analysis`
- `immune_infiltration` (SpatialCells mode)
- `enhanced_neighborhoods`
- `tumor_microenvironment`

These receive `tumor_structures` and `region_detector` from the per-tumor step. If `per_tumor_analysis` is disabled they silently receive `None` and will error or skip spatial computations.

---

## per_tumor_analysis

**Module:** `analyses/per_tumor_analysis_spatialcells.py`

**What it does:** Detects spatial structures (tumors, B cell clusters, follicles) using DBSCAN + alpha-shape boundary fitting (via SpatialCells). Measures composition, area, and immune content for each detected structure.

**When to enable:** Always, if your study has a primary spatial structure. This is the anchor for most downstream analyses.

**Key config:**
```yaml
per_tumor_analysis:
  enabled: true
  use_spatialcells: true
  markers:
    - {name: GL7, phenotype: GL7_positive_B_cells}   # markers to quantify per-structure
  spatial_heterogeneity:
    enabled: true
    window_size: 300      # sliding window size in pixels
    phenotype_col: is_GC_B_cells
```

**Outputs:** `per_structure_analysis/per_tumor_metrics.csv`, per-structure composition plots

---

## population_dynamics

**Module:** `analyses/population_dynamics.py`

**What it does:** Compares cell population frequencies (count, density per mm², fraction of parent) across experimental groups. Supports group comparisons, timepoint series, treatment arms, and Spearman correlation with ordinal variables.

**When to enable:** Almost always — this is the primary cell-type frequency readout.

**Key config:**
```yaml
population_dynamics:
  enabled: true
  populations: [T_cells, CD8_T_cells, ...]   # phenotype keys to quantify
  fractional_populations:
    CD8_T_cells: T_cells    # CD8 fraction relative to T cells
  comparisons:
    - {name: KPT_vs_KPNT, groups: [KPT, KPNT]}
  statistics:
    test_per_timepoint: false   # true for longitudinal studies
    test_method: mannwhitneyu
```

**Outputs:** `population_dynamics/` — frequency tables, violin/box plots per population

---

## distance_analysis

**Module:** `analyses/distance_analysis.py`

**What it does:** Measures nearest-neighbor distances between cell populations. Reports mean/median distance from each source cell to its closest target cell. Compares distributions across groups.

**When to enable:** When you want to quantify spatial proximity between immune cells and tumor/structure cells.

**Key config:**
```yaml
distance_analysis:
  enabled: true
  pairings:
    - source: CD8_T_cells
      targets: [Tumor, pERK_positive_tumor]
  metrics: [mean_nearest_neighbor, median_nearest_neighbor, distance_distribution]
  levels: [per_sample, per_structure]
  max_distance_plot: 500   # truncate distance histograms at this µm value
```

**Outputs:** `distance_analysis/` — distance tables, violin plots, distribution histograms

---

## immune_infiltration

**Module:** `analyses/infiltration_analysis_spatialcells.py` (primary) / `infiltration_analysis_optimized.py` (fallback)

**What it does:** Counts immune cells at defined distances from structure boundaries (0, 50, 100, 150 µm). Uses SpatialCells `getDistanceFromObject` to compute per-cell distance, then bins into infiltration zones.

**When to enable:** When you need quantitative infiltration metrics per structure. Requires `per_tumor_analysis` to run first.

**Key config:**
```yaml
immune_infiltration:
  enabled: true
  use_spatialcells: true
  immune_populations: [T_cells, CD8_T_cells, CD4_T_cells]
  boundaries: [0, 50, 100, 150]   # distance bins in pixels
  per_structure: true
```

**Outputs:** `infiltration_analysis/` — infiltration tables per structure and per sample

---

## spatial_permutation

**Module:** `analyses/spatial_permutation_testing.py`

**What it does:** Tests whether the observed spatial colocalization between two populations exceeds chance. Shuffles `is_{phenotype}` labels 500× and computes a z-score for the observed metric against the null distribution.

**When to enable:** When you want to determine if spatial patterns (e.g. CD8 enrichment near pERK+ tumor) are statistically significant beyond random cell placement.

**Key config:**
```yaml
spatial_permutation:
  enabled: true
  n_permutations: 500
  pairs:
    - {source: CD8_T_cells, target: pERK_positive_tumor, name: CD8_pERK}
```

**Outputs:** `spatial_permutation/` — z-scores, p-values, null distribution plots

---

## distance_permutation_testing

**Module:** `analyses/distance_permutation_testing.py`

**What it does:** Permutation test for distance differences. Tests (a) whether source cells are closer to marker+ vs marker- target cells (differential test), and (b) whether source cells are specifically close to a target beyond chance (proximity test).

**When to enable:** When you want statistically rigorous confirmation of distance analysis results.

**Key config:**
```yaml
distance_permutation_testing:
  enabled: true
  differential_tests:
    - {name: CD8_pERK, source: CD8_T_cells, target_base: Tumor, target_marker: is_pERK_positive_tumor}
  proximity_tests:
    - {name: CD8_to_tumor, source: CD8_T_cells, target: Tumor, shuffle_pool: [CD8_T_cells, CD4_T_cells]}
  parameters:
    n_permutations: 999
    alpha: 0.05
```

**Outputs:** `distance_permutation/` — differential test results, proximity test results

---

## neighborhood_permutation_testing

**Module:** `analyses/neighborhood_permutation_testing.py`

**What it does:** Permutation test for neighborhood enrichment between cell types. Builds a k-NN graph, counts pairwise cell-type interactions, and shuffles type labels to build a null. Tests whether specific immune populations are enriched in the neighborhood of specific tumor states.

**Key config:**
```yaml
neighborhood_permutation_testing:
  enabled: true
  cell_types: [Tumor, CD8_T_cells, Macrophages, ...]
  parameters:
    n_permutations: 1000
    k_neighbors: 30
    min_cells_per_type: 10
```

**Outputs:** `neighborhood_permutation/` — enrichment z-scores, heatmaps

---

## cellular_neighborhoods

**Module:** `analyses/neighborhoods_optimized.py`

**What it does:** Classifies cells by their local neighborhood composition using windowed k-NN. Runs K-means on the neighborhood composition vectors to define recurrent cellular neighborhoods (RCNs). Analyzes how RCN frequencies vary across groups.

**When to enable:** For unsupervised discovery of recurring spatial patterns in tissue. Use `lda_neighborhood_analysis` instead to reproduce the Nirmal et al. 2022 RCN methodology.

**Key config:**
```yaml
cellular_neighborhoods:
  enabled: true
  populations: [T_cells, B_cells, Macrophages, ...]
  window_size: 100      # neighborhood radius in pixels
  k_neighbors: 30
  n_clusters: 8         # number of RCN classes
  clustering_method: kmeans
  define_globally: true  # one global clustering across all samples
```

**Outputs:** `neighborhood_analysis/` — RCN assignments, composition heatmaps, spatial maps

---

## tumor_microenvironment

**Module:** `analyses/tumor_microenvironment_analysis.py`

**What it does:** Defines three spatial zones relative to structure boundary (contact, close, distal) and measures immune composition in each zone. Requires `per_tumor_analysis`.

**When to enable:** For zone-level comparison of immune infiltration at the tumor margin.

**Key config:**
```yaml
tumor_microenvironment:
  enabled: true
  zones:
    contact: [0, 50]      # pixels from boundary
    close: [50, 150]
    distal: [150, 500]
  immune_populations: [CD8_T_cells, Macrophages, ...]
```

**Outputs:** `tumor_microenvironment/` — zone composition tables, heatmaps

---

## enhanced_neighborhoods

**Module:** `analyses/enhanced_neighborhood_analysis.py`

**What it does:** Measures immune cell neighborhood composition inside vs outside marker-defined regions (e.g. GL7+ GC zones, pERK+ tumor regions). Requires `per_tumor_analysis` for boundary-based stratification.

**Key config:**
```yaml
enhanced_neighborhoods:
  enabled: true
  neighborhood_radius: 50
  markers:
    - {name: GL7, pos_col: is_GL7_positive_B_cells, neg_col: is_Naive_B_cells}
  cell_types: [T_cells, B_cells, FDCs, ...]
  immune_cells: [T_cells, Tfh_cells, CD8_T_cells]
```

**Outputs:** `enhanced_neighborhoods/` — inside/outside composition tables, bar plots

---

## marker_region_analysis

**Module:** `analyses/marker_region_analysis_spatialcells.py`

**What it does:** Detects alpha-shape spatial communities of marker+ cells (e.g. pERK+ tumor regions, GL7+ GC zones) and measures immune enrichment inside vs outside those regions.

**When to enable:** When you want to define tumor-state-specific spatial zones and test immune exclusion/engagement hypotheses.

**Key config:**
```yaml
marker_region_analysis:
  enabled: true
  markers:
    - {name: pERK, positive_col: is_pERK_positive_tumor, negative_col: is_pERK_negative_tumor}
  immune_populations: [CD8_T_cells, Macrophages]
  region_detection:
    eps: 80
    min_samples: 5
    alpha: 40
    min_area: 0
    min_edges: 15
```

**Outputs:** `marker_region_analysis/` — region tables, immune enrichment plots

---

## cluster_composition_analysis

**Module:** `analyses/cluster_composition_analysis.py`

**What it does:** Stacked bar charts showing immune composition per detected structure (ordered by structure size or composition metric). Requires `per_tumor_analysis`.

**Key config:**
```yaml
cluster_composition_analysis:
  enabled: true
  populations: [CD8_T_cells, Macrophages, Tregs, ...]
  visualization:
    normalize: true
    group_by_timepoint: true
```

**Outputs:** `cluster_composition_analysis/` — per-structure composition plots

---

## temporal_analysis

**Module:** `stats/temporal.py`

**What it does:** Measures how structure-level metrics change across timepoints. Reports cluster count, total B cells, area, density, and user-defined marker zone metrics over time.

**When to enable:** For longitudinal studies with `timepoint` column in metadata.

**Key config:**
```yaml
temporal_analysis:
  enabled: true
  cluster_metrics: [cluster_count, cluster_size, cluster_area, cluster_density]
```

**Outputs:** `temporal_analysis/` — longitudinal metric tables, line plots

---

## lda_neighborhood_analysis

**Module:** `analyses/lda_neighborhood_analysis.py`

**What it does:** Reproduces the Recurrent Cellular Neighborhood (RCN) methodology from Nirmal et al. 2022 (Cancer Discovery). Builds neighborhood composition vectors for each cell (cells within `proximity_radius` µm), fits Latent Dirichlet Allocation (LDA), then K-means clusters LDA weights into RCNs. Reports how RCN frequencies vary with disease state.

**When to enable:** For publication-grade RCN analysis or when validating against the Nirmal et al. melanoma dataset.

**Key config:**
```yaml
lda_neighborhood_analysis:
  enabled: true
  proximity_radius: 20      # µm neighborhood radius (paper: 20µm)
  n_lda_topics: 10          # LDA components
  n_lda_clusters: 30        # K-means on LDA weights (paper: 30)
  n_rcns: 10                # Final RCN meta-clusters (paper: 10)
  cell_types: [Tumor_cells, T_cells, Macrophages, ...]
  stage_column: stage_group
  correlation_column: stage_numeric   # ordinal for Spearman
  max_cells_per_sample: null          # subsample limit (null = no limit)
  learning_method: online             # 'online' (fast) or 'batch' (exact)
```

**Outputs:** `lda_neighborhood_analysis/` — RCN assignments, composition heatmaps, frequency plots

---

## spatial_lag_analysis

**Module:** `analyses/spatial_lag_analysis.py`

**What it does:** Reproduces the Tumor Cell Community (TCC) methodology from Nirmal et al. 2022. For each tumor cell, computes a spatial lag vector (mean marker expression of neighbors within `lag_radius` µm). PCA reduces the lag vectors, then K-means defines TCCs. Reports how TCC frequencies change with disease progression.

**When to enable:** For TCC analysis on tumor cell populations or when validating against the Nirmal et al. melanoma dataset.

**Key config:**
```yaml
spatial_lag_analysis:
  enabled: true
  lag_radius: 20           # µm spatial lag radius
  n_tccs: 10               # tumor cell communities (paper: 10)
  n_pcs: 20                # PCA components
  tumor_phenotypes: [Melanocytes, Tumor_cells]
  markers: [S100a, MART1, SOX10, CD163, CD3d, ...]   # markers for lag vectors
  stage_column: stage_group
```

**Outputs:** `spatial_lag_analysis/` — TCC assignments, PCA plots, frequency tables

---

## shift_plot_analysis

**Module:** `analyses/shift_plot_analysis.py`

**What it does:** Compares nearest-neighbor distance distributions between two stage groups using the Harrell-Davis quantile estimator and bootstrap confidence intervals. Produces "shift plots" showing which quantiles of the distance distribution differ between groups.

**When to enable:** For robust non-parametric comparison of distance distributions across disease stages. Reproduces Nirmal et al. 2022 Fig 3F.

**Key config:**
```yaml
shift_plot_analysis:
  enabled: true
  proximity_radius: 2000    # max search distance in µm
  percentiles: [10, 20, 30, 40, 50, 60, 70, 80, 90]
  n_bootstrap: 500          # bootstrap samples for CI
  comparisons:
    - {group1: Early, group2: Advanced, label: Early_vs_Advanced}
  pairs:
    - {source: Melanocytes, target: Cytotoxic_T_cells, name: Melanocyte_to_CTL}
```

**Outputs:** `shift_plot_analysis/` — shift plot figures, quantile tables with bootstrap CIs

---

## perk_mfi_analysis

**Module:** `analyses/perk_mfi_analysis.py`

**What it does:** Quantifies pERK mean fluorescence intensity (MFI) per phenotype and per detected structure. Tests whether pERK MFI differs between genotypes (e.g. KPT vs KPNT).

**When to enable:** For studies measuring MAPK/ERK signaling intensity in tumor cells.

**Key config:**
```yaml
perk_mfi_analysis:
  enabled: true
  comparisons:
    - {name: KPT_vs_KPNT, groups: [KPT, KPNT]}
```

**Outputs:** `perk_mfi_analysis/` — MFI tables, violin plots

---

## coexpression_analysis

**Module:** `analyses/coexpression_analysis_comprehensive.py`

**What it does:** Pairwise co-occurrence analysis between hypothesis-relevant phenotypes. Tests which cell type pairs are spatially co-enriched within structures.

**Note:** This analysis defaults to `enabled: false` — it must be explicitly turned on.

**Outputs:** `coexpression_analysis/` — co-occurrence matrices, heatmaps

---

## spatial_overlap_analysis

**Module:** `analyses/spatial_overlap_analysis.py`

**What it does:** Measures spatial overlap (Dice coefficient and spatial correlation) between pairs of cell population density maps.

**Note:** This analysis defaults to `enabled: false` — it must be explicitly turned on.

**Outputs:** `spatial_overlap_analysis/` — overlap scores per sample, heatmaps
