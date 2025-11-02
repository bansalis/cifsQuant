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
│   ├── neighborhoods.py             # Cellular neighborhoods
│   └── advanced.py                  # Pseudo-time, etc.
│
├── visualization/
│   ├── plot_manager.py              # Orchestrate plotting
│   ├── individual_plots.py          # Single plots
│   ├── composite_plots.py           # Multi-panel figures
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

### 5. Advanced Analyses

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
