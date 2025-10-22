# Comprehensive Tumor Spatial Analysis Framework

A complete, publication-ready spatial analysis pipeline for cyclic immunofluorescence (cycIF) data, designed for in-depth tumor-immune interaction studies.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Analysis Modules](#analysis-modules)
- [Outputs](#outputs)
- [Advanced Usage](#advanced-usage)
- [Examples](#examples)

---

## Overview

This framework provides a comprehensive spatial analysis pipeline specifically designed for tumor immunology studies using multiplex immunofluorescence imaging. It integrates tumor structure detection, immune infiltration quantification, temporal tracking, and spatial heterogeneity analysis into a single, customizable workflow.

### Key Capabilities

1. **Cell Population Definition** - Flexible marker-based population hierarchy
2. **Tumor Structure Detection** - Spatial clustering to identify tumor aggregates
3. **Infiltration Boundary Analysis** - Multi-scale concentric zones around tumors
4. **Immune Infiltration Quantification** - Population-specific metrics by region
5. **Temporal Analysis** - Track changes in size, expression, and infiltration over time
6. **Co-enrichment Analysis** - Statistical testing for spatial proximity
7. **Spatial Heterogeneity** - Identify distinct tumor regions by molecular profile
8. **Comparative Analysis** - Compare infiltration across regions and conditions
9. **Publication-Quality Visualizations** - Automated figure generation

---

## Features

### 1. Tumor Structure Detection
- **DBSCAN clustering** to identify contiguous tumor regions
- Calculates size, area, perimeter, and compactness for each structure
- Filters noise and small artifacts
- Assigns structure IDs to cells

### 2. Infiltration Boundary Analysis
- Creates **concentric zones** around tumor structures:
  - **Tumor Core**: Cells within tumor structures
  - **Tumor Margin**: 0-30 μm from tumor
  - **Peri-Tumor**: 30-100 μm from tumor
  - **Distal**: 100-200 μm from tumor
  - **Far**: >200 μm from tumor
- Fully customizable boundary distances

### 3. Immune Infiltration Quantification
- Calculates:
  - Cell counts per region
  - Percentage of total cells
  - Density (cells per mm²)
- Supports sample-wise and timepoint-wise analysis
- Exports data for statistical analysis

### 4. Temporal Analysis
- **Tumor growth tracking**: Size and structure count over time
- **Marker expression trends**: Percentage positive and mean intensity
- **Infiltration dynamics**: Immune population changes over time
- Automated statistical comparisons

### 5. Co-enrichment Analysis
- Permutation-based statistical testing
- Identifies population pairs that co-localize more than expected by chance
- Calculates z-scores and p-values
- Customizable search radius

### 6. Spatial Heterogeneity Detection
- **K-means clustering** on spatial + molecular features
- Identifies distinct tumor regions with different marker profiles
- Calculates region-specific marker expression
- Compares immune infiltration across regions

### 7. Publication-Ready Visualizations
- Spatial overview (4-panel figure)
- Temporal trend plots
- Infiltration heatmaps
- Co-enrichment plots
- Heterogeneity maps
- All figures at 300 DPI, ready for publication

---

## Quick Start

### Prerequisites

```bash
# Required Python packages
pip install numpy pandas matplotlib seaborn scipy scikit-learn scanpy pyyaml
```

### Basic Usage

```bash
# 1. Run the complete analysis with default configuration
python run_tumor_spatial_analysis.py

# 2. Use a custom configuration file
python run_tumor_spatial_analysis.py --config my_config.yaml

# 3. Specify custom output directory
python run_tumor_spatial_analysis.py --output my_results/
```

### Python API Usage

```python
import scanpy as sc
from tumor_spatial_analysis import TumorSpatialAnalysis

# Load your gated data
adata = sc.read_h5ad('manual_gating_output/gated_data.h5ad')

# Initialize the analysis
tsa = TumorSpatialAnalysis(
    adata,
    tumor_markers=['TOM', 'AGFP'],
    immune_markers=['CD45', 'CD3', 'CD8B'],
    output_dir='tumor_spatial_analysis'
)

# Define populations
population_config = {
    'Tumor': {'markers': {'TOM': True}, 'color': '#E41A1C'},
    'CD8_T_cells': {'markers': {'CD3': True, 'CD8B': True}, 'color': '#4DAF4A'}
}
tsa.define_cell_populations(population_config)

# Run analyses
tsa.detect_tumor_structures()
tsa.define_infiltration_boundaries()
tsa.quantify_immune_infiltration(['CD8_T_cells'])

# Generate visualizations
tsa.plot_spatial_overview()
tsa.generate_comprehensive_report()
```

---

## Configuration

The analysis is controlled by a YAML configuration file. Here's a minimal example:

### Minimal Configuration

```yaml
# Input/Output
input_data: 'manual_gating_output/gated_data.h5ad'
output_directory: 'tumor_spatial_analysis'

# Markers
tumor_markers: [TOM, AGFP]
immune_markers: [CD45, CD3, CD8B]

# Populations
populations:
  Tumor:
    markers:
      TOM: true
    color: '#E41A1C'

  CD8_T_cells:
    markers:
      CD3: true
      CD8B: true
    color: '#4DAF4A'

# Analysis settings
tumor_structure_detection:
  tumor_population: 'Tumor'
  min_cluster_size: 50
  eps: 30

immune_infiltration:
  populations: [CD8_T_cells]
```

### Full Configuration Template

See `configs/tumor_spatial_config.yaml` for a complete configuration template with all available options and detailed comments.

### Key Configuration Sections

#### 1. Population Definitions

```yaml
populations:
  # Parent population
  Tumor:
    markers:
      TOM: true
    color: '#E41A1C'

  # Child population (inherits from parent)
  Tumor_AGFP_positive:
    markers:
      TOM: true
      AGFP: true
    parent: Tumor
    color: '#377EB8'

  # Negative marker requirement
  Tumor_AGFP_negative:
    markers:
      TOM: true
      AGFP: false
    parent: Tumor
    color: '#FDB462'
```

#### 2. Temporal Analysis

```yaml
temporal_analysis:
  enabled: true
  timepoint_column: 'timepoint'  # Column name in adata.obs
  populations_to_track:
    - Tumor
    - CD8_T_cells
  marker_trends:
    - AGFP
    - KI67
```

#### 3. Co-enrichment Pairs

```yaml
coenrichment_analysis:
  population_pairs:
    - [CD8_T_cells, Tumor_AGFP_positive]
    - [CD8_T_cells, Tumor_AGFP_negative]
  radius: 50
  n_permutations: 100
```

---

## Analysis Modules

### Module 1: Cell Population Definition

**Purpose**: Define tumor and immune cell populations based on marker combinations.

**Method**:
```python
population_config = {
    'CD8_Ki67_positive': {
        'markers': {'CD3': True, 'CD8B': True, 'KI67': True},
        'parent': 'CD8_T_cells',
        'color': '#999999'
    }
}
tsa.define_cell_populations(population_config)
```

**Output**:
- Boolean columns in `adata.obs`: `is_CD8_Ki67_positive`
- Population statistics (count, percentage)

---

### Module 2: Tumor Structure Detection

**Purpose**: Identify spatial aggregates of tumor cells.

**Algorithm**: DBSCAN clustering on spatial coordinates

**Parameters**:
- `min_cluster_size`: Minimum cells to form a structure (default: 50)
- `eps`: Maximum distance between cells in same cluster (default: 30 μm)
- `min_samples`: DBSCAN core point threshold (default: 10)

**Method**:
```python
tsa.detect_tumor_structures(
    min_cluster_size=50,
    eps=30,
    min_samples=10
)
```

**Output**:
- `adata.obs['tumor_structure_id']`: Structure ID for each cell (-1 = no structure)
- Structure metrics: size, area, perimeter, compactness
- Saved to: `data/tumor_structures.csv`

---

### Module 3: Infiltration Boundary Definition

**Purpose**: Create concentric zones around tumor structures for infiltration analysis.

**Method**:
```python
tsa.define_infiltration_boundaries(
    boundary_widths=[30, 100, 200]
)
```

**Output**:
- `adata.obs['boundary_region']`: Region assignment per cell
- `adata.obs['distance_to_tumor']`: Distance to nearest tumor cell (μm)
- Saved to: `data/infiltration_boundaries.csv`

**Regions Created**:
1. **Tumor_Core**: Inside tumor structures
2. **Tumor_Margin**: 0-30 μm from tumor
3. **Peri_Tumor**: 30-100 μm from tumor
4. **Distal**: 100-200 μm from tumor
5. **Far**: >200 μm from tumor

---

### Module 4: Immune Infiltration Quantification

**Purpose**: Quantify immune populations in each boundary region.

**Method**:
```python
infiltration_df = tsa.quantify_immune_infiltration(
    immune_populations=['CD8_T_cells', 'CD8_Ki67_positive'],
    by_sample=True,
    by_timepoint=True
)
```

**Metrics Calculated**:
- Cell count per region
- Percentage of total cells in region
- Density (cells per mm²)

**Output**: CSV file with columns:
- `region`, `population`, `n_cells`, `total_cells`, `percentage`, `density_per_mm2`
- Optional: `sample_id`, `timepoint`

**Saved to**: `data/immune_infiltration_metrics.csv`

---

### Module 5: Temporal Analysis

**Purpose**: Track changes in tumor size, marker expression, and infiltration over time.

**Method**:
```python
tsa.analyze_temporal_changes(
    timepoint_col='timepoint',
    populations=['Tumor', 'CD8_T_cells'],
    marker_trends=['AGFP', 'KI67']
)
```

**Outputs**:
1. **Tumor Size Metrics**: `data/temporal_tumor_size.csv`
   - Columns: `timepoint`, `n_tumor_cells`, `n_structures`, `avg_structure_size`, `total_tumor_area`

2. **Marker Expression**: `data/temporal_marker_expression.csv`
   - Columns: `timepoint`, `marker`, `pct_positive`, `mean_expression`, `n_positive`

3. **Infiltration Changes**: `data/temporal_infiltration.csv`
   - Columns: `timepoint`, `population`, `region`, `n_cells`, `percentage`

---

### Module 6: Co-enrichment Analysis

**Purpose**: Test if population pairs are spatially closer than expected by chance.

**Algorithm**: Permutation test with spatial randomization

**Method**:
```python
enrichment_df = tsa.analyze_coenrichment(
    population_pairs=[
        ('CD8_T_cells', 'Tumor_AGFP_positive'),
        ('CD8_T_cells', 'Tumor_AGFP_negative')
    ],
    radius=50,
    n_permutations=100
)
```

**Output**: `data/coenrichment_analysis.csv`
- Columns: `population_a`, `population_b`, `enrichment_score`, `z_score`, `p_value`, `significant`
- Interpretation: p < 0.05 indicates significant co-localization

---

### Module 7: Spatial Heterogeneity Detection

**Purpose**: Identify distinct tumor regions based on molecular profiles.

**Algorithm**: K-means clustering on spatial + marker expression features

**Method**:
```python
heterogeneity_df = tsa.detect_spatial_heterogeneity(
    tumor_population='Tumor',
    heterogeneity_markers=['AGFP', 'PERK', 'KI67'],
    n_regions=3,
    min_region_size=100
)
```

**Output**:
- `adata.obs['heterogeneity_region']`: Region ID for each tumor cell
- `data/heterogeneity_regions.csv`: Region profiles with marker expression
- Columns: `region_id`, `n_cells`, `pct_of_tumor`, `AGFP_mean`, `AGFP_pct_pos`, ...

---

### Module 8: Region Infiltration Comparison

**Purpose**: Compare immune infiltration across different tumor heterogeneity regions.

**Method**:
```python
region_comparison = tsa.compare_region_infiltration(
    immune_populations=['CD8_T_cells', 'CD8_Ki67_positive'],
    region_col='heterogeneity_region'
)
```

**Output**: `data/region_infiltration_comparison.csv`
- Compares infiltration within 50 μm of each tumor region
- Columns: `region_id`, `population`, `n_cells`, `total_near`, `pct_infiltration`

---

## Outputs

All outputs are organized in the specified output directory:

```
tumor_spatial_analysis/
├── data/
│   ├── immune_infiltration_metrics.csv
│   ├── temporal_tumor_size.csv
│   ├── temporal_marker_expression.csv
│   ├── temporal_infiltration.csv
│   ├── coenrichment_analysis.csv
│   ├── heterogeneity_regions.csv
│   └── region_infiltration_comparison.csv
├── figures/
│   ├── spatial_overview.png
│   ├── temporal_trends.png
│   ├── infiltration_heatmap.png
│   └── heterogeneity_maps.png
└── analysis_report.md
```

### Data Files

All CSV files can be loaded into R, Python, Excel, or Prism for further statistical analysis and custom visualizations.

### Figures

- **spatial_overview.png**: 4-panel overview (populations, structures, density, heterogeneity)
- **temporal_trends.png**: Multi-panel time series plots
- **infiltration_heatmap.png**: Heatmap of immune infiltration by region
- All figures are 300 DPI, publication-ready

### Report

`analysis_report.md` contains a comprehensive summary in Markdown format, including:
- Population counts and percentages
- Tumor structure statistics
- Boundary region distribution
- Key findings summary

---

## Advanced Usage

### Custom Population Hierarchies

Define complex population hierarchies with multiple levels:

```python
population_config = {
    # Level 1: Pan-immune
    'CD45_positive': {
        'markers': {'CD45': True},
        'color': '#984EA3'
    },

    # Level 2: T cells
    'T_cells': {
        'markers': {'CD3': True},
        'parent': 'CD45_positive',
        'color': '#A65628'
    },

    # Level 3: CD8 T cells
    'CD8_T_cells': {
        'markers': {'CD3': True, 'CD8B': True},
        'parent': 'T_cells',
        'color': '#F781BF'
    },

    # Level 4: Activated CD8 T cells
    'CD8_Ki67_positive': {
        'markers': {'CD3': True, 'CD8B': True, 'KI67': True},
        'parent': 'CD8_T_cells',
        'color': '#999999'
    }
}
```

### Programmatic Analysis

For batch processing or custom workflows:

```python
import scanpy as sc
from tumor_spatial_analysis import TumorSpatialAnalysis

# Process multiple samples
samples = ['sample_A', 'sample_B', 'sample_C']

for sample in samples:
    adata = sc.read_h5ad(f'data/{sample}_gated.h5ad')

    tsa = TumorSpatialAnalysis(
        adata,
        tumor_markers=['TOM'],
        immune_markers=['CD45', 'CD3', 'CD8B'],
        output_dir=f'results/{sample}'
    )

    # Run standard workflow
    tsa.define_cell_populations(population_config)
    tsa.detect_tumor_structures()
    tsa.define_infiltration_boundaries()
    tsa.quantify_immune_infiltration(['CD8_T_cells'])
    tsa.plot_spatial_overview()
```

### Integration with Existing Code

The framework integrates seamlessly with the existing pipeline:

```bash
# 1. Run MCMICRO segmentation
nextflow run mcmicro-tiled.nf

# 2. Run gating pipeline
python manual_gating.py

# 3. Run phenotyping
python phenotyping.py

# 4. Run comprehensive spatial analysis (NEW!)
python run_tumor_spatial_analysis.py
```

---

## Examples

### Example 1: Basic Tumor-Immune Analysis

**Goal**: Quantify CD8 T cell infiltration around tumors

```yaml
# config.yaml
input_data: 'manual_gating_output/gated_data.h5ad'
output_directory: 'basic_analysis'

tumor_markers: [TOM]
immune_markers: [CD45, CD3, CD8B]

populations:
  Tumor:
    markers: {TOM: true}
    color: '#E41A1C'
  CD8_T_cells:
    markers: {CD3: true, CD8B: true}
    color: '#4DAF4A'

tumor_structure_detection:
  tumor_population: 'Tumor'
  min_cluster_size: 50

infiltration_boundaries:
  boundary_widths: [30, 100]

immune_infiltration:
  populations: [CD8_T_cells]
```

```bash
python run_tumor_spatial_analysis.py --config config.yaml
```

### Example 2: Temporal Tracking

**Goal**: Track tumor growth and immune infiltration over time

```yaml
temporal_analysis:
  enabled: true
  timepoint_column: 'timepoint'
  populations_to_track: [Tumor, CD8_T_cells]
  marker_trends: [KI67]
```

### Example 3: Heterogeneity Analysis

**Goal**: Identify AGFP+ and AGFP- tumor regions and compare infiltration

```yaml
populations:
  Tumor_AGFP_positive:
    markers: {TOM: true, AGFP: true}
    color: '#377EB8'
  Tumor_AGFP_negative:
    markers: {TOM: true, AGFP: false}
    color: '#FDB462'

spatial_heterogeneity:
  heterogeneity_markers: [AGFP, PERK, KI67]
  n_regions: 2

region_infiltration_comparison:
  immune_populations: [CD8_T_cells]
```

---

## Troubleshooting

### Common Issues

**1. "No spatial coordinates found"**
- Ensure your AnnData object has `adata.obsm['spatial']` or `X_centroid/Y_centroid` columns

**2. "Marker not found in dataset"**
- Check marker names match exactly (case-sensitive)
- Verify markers are present in `adata.var_names`

**3. "No tumor structures detected"**
- Try reducing `min_cluster_size` or increasing `eps`
- Check that tumor population is properly defined

**4. "Memory error"**
- Process in batches for very large datasets (>1M cells)
- Reduce `n_permutations` in co-enrichment analysis

### Getting Help

- Check the configuration file for typos
- Use `--verbose` flag for detailed debugging
- Examine intermediate data files in `output_dir/data/`

---

## Citation

If you use this framework in your research, please cite:

```
[Your publication details here]
```

## License

[Your license here]

## Contact

[Your contact information]

---

**Last Updated**: 2025-10-22
