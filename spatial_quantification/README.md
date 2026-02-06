# Spatial Quantification Pipeline

**Clean, comprehensive spatial analysis workflow for cyclic immunofluorescence data**

---

## Overview

This is a ground-up refactor of spatial analysis for cifsQuant, designed for:
- **High customization**: Config-driven phenotypes, flexible metadata, adaptable comparisons
- **Comprehensive outputs**: Individual plots (raw + publication) + statistics for every analysis
- **Scalability**: Handles large datasets with high plot numbers
- **Modularity**: Clean separation of concerns (data → analysis → visualization → statistics)

### Key Features

✅ **Flexible tumor definition** - Currently TOM+, easily configurable for future changes
✅ **Custom phenotypes** - Define ANY combination from manual_gating gates
✅ **Robust metadata** - Easy to add custom grouping columns
✅ **Dual outputs** - Raw data plots + publication-quality versions
✅ **Complete statistics** - All tests saved to CSV alongside plots
✅ **Multi-panel figures** - Automatic composite figure generation
✅ **Tumor subsets** - Automatic fractional calculation (e.g., pERK+ fraction of tumor)

---

## Installation

### Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn pyyaml anndata
```

Or use the existing cifsQuant environment.

---

## Quick Start

### 1. Prepare Your Data

Ensure you have:
- **Gated h5ad file** from `manual_gating.py`
- **Sample metadata CSV** with `sample_id`, `group`, `timepoint` columns

### 2. Configure Your Analysis

Edit `spatial_quantification/config/spatial_config.yaml`:

```yaml
# Define custom phenotypes
phenotypes:
  pERK_positive_tumor:
    base: 'Tumor'  # Requires TOM+
    positive: ['PERK']
    negative: []

  CD8_T_cells:
    positive: ['CD3', 'CD8B']
    negative: []

# Specify analyses
population_dynamics:
  enabled: true
  populations:
    - Tumor
    - pERK_positive_tumor
    - CD8_T_cells

  # Automatic fractional calculation
  fractional_populations:
    pERK_positive_tumor: Tumor  # Will compute pERK+ fraction of Tumor

  comparisons:
    - name: 'KPT_vs_KPNT'
      groups: ['KPT', 'KPNT']
      timepoints: [8, 10, 12, 14, 16, 18, 20]
```

### 3. Run the Pipeline

```bash
cd spatial_quantification
python run_spatial_quantification.py --config config/spatial_config.yaml
```

### 4. Find Your Results

```
spatial_quantification_results/
├── population_dynamics/
│   ├── Tumor_count_over_time_raw.pdf
│   ├── Tumor_count_over_time_publication.pdf
│   ├── Tumor_count_over_time_stats.csv
│   ├── pERK_positive_tumor_fraction_over_time_raw.pdf
│   └── ...
├── distance_analysis/
│   ├── CD8_T_cells_to_Tumor_distance_raw.pdf
│   ├── CD8_T_cells_to_Tumor_distance_publication.pdf
│   ├── CD8_T_cells_to_Tumor_distance_stats.csv
│   └── ...
├── infiltration_analysis/
│   └── ...
├── spatial_permutation/
│   ├── per_tumor_results.csv
│   ├── sample_summary.csv
│   ├── group_comparisons.csv
│   ├── plots/
│   │   └── *.png
│   └── null_distributions/
│       └── *.png
└── neighborhoods/
    └── ...
```

---

## Architecture

### Directory Structure

```
spatial_quantification/
├── config/
│   └── spatial_config.yaml          # Main configuration
│
├── core/
│   ├── data_loader.py               # Load h5ad + metadata
│   ├── phenotype_builder.py         # Build custom phenotypes
│   └── metadata_manager.py          # Process metadata
│
├── analyses/
│   ├── population_dynamics.py       # Population over time
│   ├── distance_analysis.py         # Cell-to-cell distances
│   ├── infiltration_analysis.py     # Immune infiltration + zones
│   ├── spatial_permutation_testing.py # Per-tumor permutation tests
│   ├── neighborhoods.py             # Cellular neighborhoods
│   └── advanced.py                  # Pseudo-time, etc.
│
├── visualization/
│   ├── plot_manager.py              # Orchestrate plotting
│   ├── individual_plots.py          # Single plots
│   ├── composite_plots.py           # Multi-panel figures
│   ├── permutation_plotter.py       # Permutation testing plots
│   └── styles.py                    # Publication settings
│
├── statistics/
│   ├── tests.py                     # Statistical tests
│   ├── comparisons.py               # Group comparisons
│   └── temporal.py                  # Temporal analysis
│
└── run_spatial_quantification.py   # Main entry point
```

### Workflow

```
1. Load Data
   ↓
2. Process Metadata (extract groupings)
   ↓
3. Build Phenotypes (from config)
   ↓
4. Run Analyses
   ├── Population Dynamics
   ├── Distance Analysis
   ├── Infiltration Analysis
   ├── Spatial Permutation Testing
   ├── Neighborhoods
   └── Advanced
   ↓
5. Generate Visualizations
   ├── Individual plots (raw + publication)
   ├── Composite figures
   └── Statistical summaries
```

---

## Analyses

### 1. Population Dynamics

**What it does:**
- Tracks cell population counts/fractions over time
- Compares groups at each timepoint
- Automatic fractional calculation for tumor subsets

**Example use case:**
```yaml
population_dynamics:
  populations:
    - pERK_positive_tumor

  fractional_populations:
    pERK_positive_tumor: Tumor
    # ^ Will generate:
    #   - pERK+ tumor COUNT over time
    #   - pERK+ FRACTION of tumor over time (controls for tumor growth)
```

**Outputs:**
- `{population}_{metric}_over_time_raw.pdf` - With raw data points
- `{population}_{metric}_over_time_publication.pdf` - Clean version
- `{population}_{metric}_stats.csv` - Statistical tests per timepoint

---

### 2. Distance Analysis

**What it does:**
- Calculates nearest-neighbor distances between cell populations
- Per-sample and per-structure analysis
- Temporal and group comparisons

**Example use case:**
```yaml
distance_analysis:
  pairings:
    - source: 'CD8_T_cells'
      targets: ['Tumor', 'pERK_positive_tumor']
```

**Outputs:**
- `{source}_to_{target}_distance_raw.pdf` - Time series with stats
- `{source}_to_{target}_distribution.pdf` - Histogram
- `{source}_to_{target}_stats.csv` - Statistical tests

---

### 3. Infiltration Analysis

**What it does:**
- Detects tumor structures (DBSCAN)
- Quantifies immune infiltration at multiple boundaries
- **Marker zone analysis**: Assesses spatial heterogeneity of marker+ vs marker- regions
- **Zone-specific infiltration**: Compares infiltration in marker+ vs marker- zones

**Example use case:**
```yaml
immune_infiltration:
  immune_populations:
    - CD8_T_cells
    - CD3_positive

  boundaries: [0, 50, 100, 200]  # μm from tumor edge

  marker_zone_analysis:
    enabled: true
    markers:
      - marker: 'PERK'
        positive_phenotype: 'pERK_positive_tumor'
        negative_phenotype: 'pERK_negative_tumor'

    # Heterogeneity metrics
    heterogeneity_metrics:
      - 'morans_i'          # Spatial autocorrelation
      - 'cluster_analysis'  # DBSCAN clustering

    # Compare infiltration in pERK+ vs pERK- zones
    zone_infiltration:
      enabled: true
```

**Outputs:**
- `infiltration.csv` - Infiltration counts per structure/zone
- `PERK_zone_heterogeneity.csv` - Heterogeneity metrics
- `PERK_zone_infiltration.csv` - Zone-specific infiltration

---

### 4. Cellular Neighborhoods

**What it does:**
- Defines neighborhoods across all samples (or per-group)
- Analyzes composition, infiltration, temporal evolution

**Example use case:**
```yaml
cellular_neighborhoods:
  populations:
    - Tumor
    - pERK_positive_tumor
    - CD8_T_cells

  window_size: 100  # μm radius
  n_clusters: 8     # Number of neighborhood types
```

**Outputs:**
- `compositions.csv` - Per-cell neighborhood composition
- `neighborhood_composition.csv` - Mean composition per type
- `temporal_evolution.csv` - Neighborhood abundance over time

---

### 5. Spatial Permutation Testing

**What it does:**
- Determines whether spatial patterns of marker expression within individual tumor structures are biologically meaningful or artifacts of random chance
- Uses **per-tumor Monte Carlo permutation testing**: cell coordinates stay fixed while marker labels are randomly shuffled
- Three test types with distinct biological questions, all operating **within** each tumor structure

**Statistical approach:**
1. Compute an observed spatial statistic for each tumor structure
2. Randomly permute marker labels `n_permutations` times (default 500) while keeping cell positions fixed
3. Build a null distribution of the statistic under the "no spatial pattern" hypothesis
4. Derive a z-score `(observed - null_mean) / null_std` and empirical p-value
5. Apply Benjamini-Hochberg FDR correction per sample
6. Aggregate per-tumor results to sample-level summaries
7. Run Mann-Whitney U group comparisons on the sample-level summaries

#### Test Types

| Test | Biological Question | Statistic | Permutation Strategy |
|------|---------------------|-----------|---------------------|
| **Clustering** | Are marker+ cells spatially clustered within this tumor? | Hopkins statistic (H > 0.5 = clustered, H = 0.5 = random, H < 0.5 = uniform) | Shuffle single marker labels among cells |
| **Co-localization** | Do two markers overlap spatially more than expected by chance? | Cross-K function (mean count of marker2+ cells within radius of each marker1+ cell) | Independently shuffle both marker labels |
| **Enrichment** | Are immune cells enriched near marker+ tumor cells? | Mean immune cell count within radius of marker+ tumor cells | Shuffle tumor marker labels |

#### Interpreting Results

**Per-tumor results** (`per_tumor_results.csv`) — one row per tumor structure per test:

| Column | Meaning |
|--------|---------|
| `sample_id` | Sample the tumor belongs to |
| `structure_id` | Unique tumor structure identifier |
| `test_type` | `clustering`, `colocalization`, or `enrichment` |
| `test_name` | Name from config (e.g., `pERK_clustering`) |
| `marker` | Marker(s) tested |
| `n_cells` | Total cells in the structure |
| `n_positive` | Number of marker+ cells (clustering/enrichment) |
| `prevalence` | Fraction of positive cells |
| `observed` | Observed spatial statistic value |
| `null_mean` | Mean of the permutation null distribution |
| `null_std` | Standard deviation of the null distribution |
| `z_score` | Effect size: `(observed - null_mean) / null_std`. Positive = more clustered/colocalized/enriched than chance |
| `p_value` | Empirical p-value from the null distribution |
| `p_adjusted` | Benjamini-Hochberg FDR-corrected p-value (per sample) |
| `significant` | Boolean: `p_adjusted < alpha` |

**How to read z-scores:**
- **z > 0**: The observed spatial pattern is stronger than expected by chance (marker+ cells are more clustered, co-localized, or associated with immune cells than random)
- **z ~ 0**: No spatial pattern; marker expression is spatially random
- **z < 0**: Anti-pattern (marker+ cells are more dispersed or separated than expected by chance)
- **|z| > 2**: Strong effect; **|z| > 3**: Very strong effect

**Sample summary** (`sample_summary.csv`) — one row per sample per test:

| Column | Meaning |
|--------|---------|
| `n_tumors` | Number of structures tested in this sample |
| `n_significant` | Number of structures with FDR-significant spatial pattern |
| `pct_significant` | Percentage of structures with significant spatial pattern |
| `mean_z_score` | Mean effect size across all structures in the sample |
| `median_p_value` | Median p-value across all structures |

**How to interpret `pct_significant`:**
- **0-10%**: Spatial pattern is rare; most tumors show random marker distribution
- **10-30%**: A subset of tumors shows the pattern; may represent biological heterogeneity
- **30-60%**: The pattern is common; likely a real biological phenomenon in this sample
- **>60%**: The pattern is pervasive across tumors; strong biological signal

**Group comparisons** (`group_comparisons.csv`) — pairwise comparison between groups:

| Column | Meaning |
|--------|---------|
| `group1`, `group2` | Groups being compared |
| `mean_z_g1`, `mean_z_g2` | Mean z-score (effect size) per group |
| `p_value` | Mann-Whitney U p-value for the comparison |

A significant group comparison means one group has systematically stronger (or weaker) spatial patterns than the other across samples.

#### Generated Plots and What They Mean

All plots are generated **per analysis** (i.e., per `test_name` such as `pERK_clustering` or `NINJA_clustering`), not merely per test type. This means each configured test gets its own dedicated plots, so results for different markers or marker pairs are never conflated.

Plots are saved under `spatial_permutation/plots/` and `spatial_permutation/null_distributions/`.

**1. Null Distribution Histograms** (`null_distributions/null_dist_{test_name}_page{N}.png`)

One subplot per tumor structure. Shows a gray histogram of the reconstructed null distribution (from `null_mean` and `null_std`) with a vertical line at the observed statistic value.

*How to read:* If the observed line (colored) falls far into the tail of the null distribution, that tumor has a significant spatial pattern. The line is **red** if FDR-significant, **blue** if not. The annotation shows the p-value and z-score.

*Use case:* Visually confirm that the permutation test is behaving correctly. If the observed value consistently falls outside the null distribution across tumors, the spatial pattern is robust.

**2. Effect Size Distribution** (`plots/effect_size_distribution_{test_name}.png`)

Violin + box plots of z-scores (effect sizes) per sample, colored by group.

*How to read:* Each violin shows the distribution of z-scores across all tumor structures within one sample. A violin shifted above zero means that sample's tumors generally exhibit the tested spatial pattern. Compare violins between groups to see if one group has consistently higher effect sizes.

*Use case:* Assess sample-to-sample variability and spot outlier samples. Samples with violins centered near zero have random spatial organization; those shifted upward have consistent spatial clustering/enrichment.

**3. Prevalence vs Effect Size** (`plots/prevalence_effect_{test_name}.png`)

Scatter plot of marker prevalence (%) vs z-score for each tumor structure, with significant structures in red and non-significant in blue. Includes a linear trend line with 95% CI.

*How to read:* Reveals whether the spatial pattern depends on how common the marker is. A flat trend means the spatial effect is independent of prevalence. A positive slope means higher-prevalence tumors show stronger clustering. The correlation (r) and p-value are annotated.

*Use case:* Rule out prevalence-driven artifacts. If significance is only seen at very low or very high prevalence, the result may be unreliable. Ideally, significant effects should be observed across a range of prevalences.

**4. Group Comparison** (`plots/group_comparison_{test_name}.png`)

Two-panel figure per test. **Left panel:** Violin + box + jitter plots of `mean_z_score` per group with Mann-Whitney U p-value. **Right panel:** Bar chart of mean `pct_significant` per group.

*How to read:* The left panel shows whether one group has systematically larger spatial effects than the other (higher mean z-scores). The right panel shows what fraction of tumors are significant in each group. If KPT shows 45% significant vs KPNT at 15%, spatial clustering is much more prevalent in KPT.

*Use case:* The primary comparison plot. Determines whether the spatial pattern differs between experimental conditions (e.g., treatment vs control, genotypes).

**5. Significance Matrix** (`plots/significance_matrix.png`)

Heatmap with samples on the y-axis and test configurations on the x-axis. Cell color intensity represents `pct_significant` (0-100%).

*How to read:* Hot (red/orange) cells indicate that a large fraction of tumors in that sample are significant for that test. Cool (yellow/white) cells indicate few or no significant tumors. Look for row patterns (samples that are significant across multiple tests) and column patterns (tests that are significant across most samples).

*Use case:* Overview of which samples and tests show the strongest spatial signals. Identifies samples that are outliers across all analyses.

**6. P-value QQ Plot** (`plots/pvalue_qq_plot.png`)

Quantile-quantile plot of observed p-values against a theoretical uniform distribution, with **one panel per analysis** (e.g., `pERK_clustering` and `NINJA_clustering` get separate panels). Includes per-panel sample count and Kolmogorov-Smirnov test statistic.

*How to read:* Under the null hypothesis (no spatial pattern anywhere), p-values should fall along the diagonal red reference line. Points below the line at small p-values indicate an excess of significant results (a true global signal). Points above the line suggest the test may be conservative.

*Use case:* Diagnostic for global signal. If the QQ plot shows systematic deviation from the diagonal, there is a real spatial pattern across the dataset. A KS p-value < 0.05 confirms the p-value distribution deviates from uniform. Because each analysis gets its own panel, you can compare whether pERK clustering shows a different global signal than NINJA clustering.

**7. Temporal Trends** (`plots/temporal_trend_{test_name}.png`)

Two-panel figure per test. **Left panel:** Mean z-score (effect size) over timepoints per group, with individual points and error bars (SEM). **Right panel:** `pct_significant` over timepoints per group.

*How to read:* Shows how the spatial pattern evolves over time. A rising mean z-score indicates that spatial clustering/enrichment is becoming stronger over time. A rising `pct_significant` means more tumors are developing the pattern. Compare lines between groups to see if temporal dynamics differ.

*Use case:* Track spatial reorganization over the course of treatment or disease progression. For example, if pERK clustering increases over time in KPT but not KPNT, this suggests treatment-induced spatial reorganization.

**8. Aggregate Null vs Observed KDE** (`plots/aggregate_null_vs_observed_{test_name}_*.png`)

Overlapped kernel density estimate (KDE) plots showing the aggregate reconstructed null distribution (gray) versus the distribution of observed statistics (red) across all tumor structures. Three variants are generated per analysis:

- **`_all.png`** — Two panels: clean KDE overlay (left) and stats-annotated version (right) pooling all samples. The stats panel shows dashed median lines for null and observed, plus a 2-sample KS test, median shift, and Cohen's d.
- **`_by_group.png`** — One subplot per group, each with its own null and observed KDE plus statistics. Directly shows whether the separation between null and observed differs between groups.
- **`_by_timepoint.png`** — One subplot per timepoint, each with its own null and observed KDE plus statistics. Shows how the null-vs-observed separation evolves over time.

*How to read:* If the red (observed) KDE is shifted to the right of the gray (null) KDE, the observed spatial statistic is systematically larger than expected by chance — meaning the spatial pattern is real. The further apart the two distributions, the stronger the biological signal. Key statistics:
- **KS statistic**: Measures maximum separation between the two distributions (0 = identical, 1 = completely separated)
- **Median shift**: How far the observed median is from the null median (positive = observed > null)
- **Cohen's d**: Standardized effect size (|d| > 0.2 small, |d| > 0.5 medium, |d| > 0.8 large)

*Use case:* The "big picture" plot for each analysis. While the per-tumor null distribution histograms (plot 1) show individual structures, this plot answers: "Across ALL tumors, is there a systematic shift between what we observe and what we'd expect by chance?" The by-group and by-timepoint variants reveal whether this shift is driven by specific experimental conditions or emerges at particular timepoints.

#### Example Configuration

```yaml
spatial_permutation:
  enabled: true
  generate_plots: true
  structure_column: tumor_structure_id

  tests:
    # Are pERK+ cells clustered within tumors?
    - type: clustering
      name: pERK_clustering
      marker: is_pERK_positive_tumor

    # Do pERK+ and NINJA+ cells spatially overlap?
    - type: colocalization
      name: pERK_NINJA_overlap
      marker1: is_pERK_positive_tumor
      marker2: is_NINJA_positive_tumor
      radius: 30  # um

    # Are CD8 T cells enriched near NINJA+ tumor cells?
    - type: enrichment
      name: CD8_near_NINJA
      tumor_marker: is_NINJA_positive_tumor
      immune_phenotype: is_CD8_T_cells
      radius: 50  # um

  parameters:
    n_permutations: 500
    min_tumor_cells: 20
    alpha: 0.05
    min_prevalence: 0.05
    max_prevalence: 0.95
    max_structures: 500
```

**Outputs:**
```
spatial_permutation/
├── per_tumor_results.csv       # One row per tumor per test
├── sample_summary.csv          # Aggregated to sample level
├── group_comparisons.csv       # Between-group statistics
├── exclusion_log.csv           # Structures excluded (too few cells, etc.)
├── config_used.json            # Configuration snapshot
├── plots/
│   ├── effect_size_distribution_{test_name}.png
│   ├── prevalence_effect_{test_name}.png
│   ├── group_comparison_{test_name}.png
│   ├── significance_matrix.png
│   ├── pvalue_qq_plot.png                              # One panel per analysis
│   ├── temporal_trend_{test_name}.png
│   ├── aggregate_null_vs_observed_{test_name}_all.png
│   ├── aggregate_null_vs_observed_{test_name}_by_group.png
│   └── aggregate_null_vs_observed_{test_name}_by_timepoint.png
└── null_distributions/
    └── null_dist_{test_name}_page{N}.png
```

---

### 6. Advanced Analyses

**Placeholder for:**
- Pseudo-time differentiation trajectories
- Spatial transcriptomics integration
- Spatial interaction networks

---

## Configuration Reference

### Tumor Definition

```yaml
tumor_definition:
  base_phenotype: 'Tumor'
  required_positive: ['TOM']    # Currently: TOM+ defines tumor
  required_negative: []         # Future: Could exclude CD45+, etc.

  structure_detection:
    method: 'DBSCAN'
    eps: 100                    # μm distance threshold
    min_samples: 10
    min_cluster_size: 50
```

**To change tumor definition in future:**
```yaml
# Example: Tumor = TOM+ AND DAPI+
required_positive: ['TOM', 'DAPI']

# Example: Tumor = TOM+ but NOT CD45+
required_positive: ['TOM']
required_negative: ['CD45']
```

### Metadata

```yaml
metadata:
  sample_column: 'sample_id'
  group_column: 'group'
  timepoint_column: 'timepoint'

  # Auto-extracted from 'group' column
  additional_groupings:
    - 'genotype'      # cis/trans
    - 'main_group'    # KPT/KPNT
    - 'treatment'

  # Add custom groupings
  # custom_groupings:
  #   responder_status:
  #     sample_mapping:
  #       Guest36: 'responder'
  #       Guest37: 'non_responder'
```

### Phenotypes

```yaml
phenotypes:
  # Basic phenotype
  CD8_T_cells:
    positive: ['CD3', 'CD8B']
    negative: []

  # Tumor subset
  pERK_positive_tumor:
    base: 'Tumor'             # Requires TOM+
    positive: ['PERK']
    negative: []

  # Complex phenotype
  CD8_Ki67_positive:
    positive: ['CD3', 'CD8B', 'KI67']
    negative: []

  # Negative selection
  CD4_T_cells:
    positive: ['CD3']
    negative: ['CD8B']        # CD3+ CD8-
```

### Comparisons

```yaml
population_dynamics:
  comparisons:
    - name: 'KPT_vs_KPNT'
      groups: ['KPT', 'KPNT']
      timepoints: [8, 10, 12, 14, 16, 18, 20]

    # Add more comparisons
    # - name: 'early_vs_late'
    #   groups: ['early_responder', 'late_responder']
    #   timepoints: [8, 10, 12]
```

---

## Outputs

### Plot Types

Every analysis generates **two versions** of each plot:

1. **Raw data** (`_raw.pdf`):
   - Individual data points overlaid
   - Statistical annotations
   - For exploration and validation

2. **Publication** (`_publication.pdf`):
   - Clean, minimalist design
   - High-resolution (300 DPI)
   - Ready for manuscripts

### Statistical Files

All statistical tests are saved to CSV:

```csv
group1,group2,timepoint,n1,n2,mean1,mean2,median1,median2,p_value,significant,effect_size
KPT,KPNT,8,3,5,1245.3,856.2,1180,820,0.042,True,0.65
KPT,KPNT,10,4,6,1580.1,920.4,1520,900,0.018,True,0.78
...
```

---

## Extending the Pipeline

### Add a New Phenotype

1. Edit `config/spatial_config.yaml`:
```yaml
phenotypes:
  my_new_phenotype:
    positive: ['MARKER1', 'MARKER2']
    negative: ['MARKER3']
```

2. Add to analysis:
```yaml
population_dynamics:
  populations:
    - my_new_phenotype
```

3. Run pipeline - plots and stats generated automatically!

### Add a Custom Metadata Grouping

1. Edit `config/spatial_config.yaml`:
```yaml
metadata:
  custom_groupings:
    cohort:
      sample_mapping:
        Guest36: 'cohort_A'
        Guest37: 'cohort_A'
        Guest38: 'cohort_B'
```

2. Use in comparisons:
```yaml
population_dynamics:
  comparisons:
    - name: 'cohort_comparison'
      groups: ['cohort_A', 'cohort_B']
      timepoints: [8, 10, 12, 14, 16, 18, 20]
```

### Add a New Analysis

Create `spatial_quantification/analyses/my_analysis.py`:

```python
class MyAnalysis:
    def __init__(self, adata, config, output_dir):
        self.adata = adata
        self.config = config['my_analysis']
        self.output_dir = output_dir / 'my_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        # Your analysis code
        results = {}
        return results
```

Add to `run_spatial_quantification.py`:

```python
from spatial_quantification.analyses import MyAnalysis

# In main():
if config.get('my_analysis', {}).get('enabled', False):
    my_analysis = MyAnalysis(adata, config, output_dir)
    all_results['my_analysis'] = my_analysis.run()
```

---

## FAQ

**Q: How do I change what defines a tumor cell?**
A: Edit `tumor_definition` in `spatial_config.yaml`. Currently `TOM+`, but you can add/change markers.

**Q: Can I analyze specific tumor subpopulations (e.g., pERK+ tumor)?**
A: Yes! Define the phenotype, then:
1. It will track raw counts over time
2. It will **automatically** calculate fraction relative to parent (Tumor)

**Q: How do I add a new grouping column?**
A: Add to `metadata.custom_groupings` with a sample-to-group mapping. The pipeline is robust to new columns.

**Q: Where are the statistical test results?**
A: Every plot has a corresponding `*_stats.csv` file with p-values, effect sizes, and summary statistics.

**Q: Can I compare more than 2 groups?**
A: The current implementation focuses on pairwise comparisons (e.g., KPT vs KPNT). For multi-way comparisons, the Kruskal-Wallis test is available in the statistics module.

**Q: How do I adjust plot styles?**
A: Edit `visualization.styles.py` or change settings in `spatial_config.yaml`:
```yaml
visualization:
  style: 'publication'  # or 'exploratory'
  font_size: 12
  dpi: 300
```

---

## Troubleshooting

### "Population not found"
- Check phenotype name matches config
- Ensure manual_gating.py has run successfully
- Verify gates exist in h5ad file (`is_MARKER` columns)

### "No spatial coordinates"
- Ensure h5ad has `obsm['spatial']` or `X_centroid`/`Y_centroid` columns

### "Missing metadata"
- Verify sample IDs match between h5ad and metadata CSV
- Check for typos (sample IDs are case-sensitive, converted to uppercase)

### Memory issues
- Reduce `cellular_neighborhoods.window_size`
- Set `performance.chunk_size` lower
- Process subsets of samples separately

---

## Comparison to Old System

| Feature | Old System | New System |
|---------|------------|------------|
| **Files** | 44+ scattered files | 1 clean module |
| **Config** | Hardcoded | YAML-driven |
| **Tumor definition** | Hardcoded TOM+ | Configurable |
| **Phenotypes** | Fixed | Unlimited custom |
| **Metadata** | Rigid | Flexible groupings |
| **Plots** | Inconsistent | Raw + publication |
| **Statistics** | Sometimes missing | Always saved |
| **Fractional pops** | Manual | Automatic |
| **Maintainability** | 😰 | 😊 |

---

## Citation

If you use this pipeline, please cite:

```
cifsQuant Spatial Quantification Pipeline
https://github.com/bansalis/cifsQuant
```

---

## Support

Issues? Questions? Check:
1. This README
2. `spatial_config.yaml` comments
3. GitHub issues: https://github.com/bansalis/cifsQuant/issues

---

**Happy analyzing! 🔬**
