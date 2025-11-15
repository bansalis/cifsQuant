# Comprehensive Coexpression and Spatial Overlap Analysis Usage Guide

## Overview

Two new analysis modules have been created for comprehensive phenotype analysis:

1. **CoexpressionAnalysisComprehensive** - Analyzes cellular coexpression patterns for ALL phenotypes
2. **SpatialOverlapAnalysis** - Analyzes spatial region overlap between marker-defined areas

Both modules automatically extract phenotypes from the config file and work with the existing pipeline structure.

---

## Module 1: CoexpressionAnalysisComprehensive

**File:** `/home/user/cifsQuant/spatial_quantification/analyses/coexpression_analysis_comprehensive.py`

### Features

- **Dynamic Phenotype Detection**: Automatically reads ALL phenotypes from `config['phenotypes']`
- **Comprehensive Analysis**: Calculates single, pairwise, and multi-marker (3-way, 4-way, 5-way) coexpression
- **Coexpression Matrices**: Creates symmetric matrices showing all pairwise relationships
- **Visualization**: Generates heatmaps, bar plots, and ranking visualizations
- **CSV Export**: Saves all results as structured CSV files

### Usage

```python
from pathlib import Path
from spatial_quantification.analyses import CoexpressionAnalysisComprehensive

# Initialize
analysis = CoexpressionAnalysisComprehensive(
    adata=adata,           # AnnData object with phenotype annotations
    config=config,         # Config dictionary with 'phenotypes' section
    output_dir=Path('results')
)

# Run analysis
results = analysis.run()
```

### Output Files

The analysis creates the following directory structure:

```
results/coexpression_analysis_comprehensive/
├── single_phenotype_frequencies.csv       # Individual phenotype counts/percentages
├── pairwise_coexpression.csv              # All pairwise combinations with metrics
├── triple_coexpression.csv                # All 3-way combinations
├── quadruple_coexpression.csv             # All 4-way combinations
├── quintuple_coexpression.csv             # All 5-way combinations
├── coexpression_matrix_percent_of_total_*.csv  # Matrix files by group
├── coexpression_matrix_jaccard_*.csv      # Jaccard index matrices by group
└── plots/
    ├── coexpression_heatmap_*.png         # Heatmaps by group
    ├── top_coexpressing_pairs.png         # Top 20 pairs ranked by Jaccard
    ├── triple_coexpression_top_combinations.png
    ├── quadruple_coexpression_top_combinations.png
    └── quintuple_coexpression_top_combinations.png
```

### Metrics Calculated

For each phenotype pair (A, B):

- **count**: Number of cells positive for both A and B
- **percent_of_total**: % of total cells that are A+ AND B+
- **percent_of_A**: % of A+ cells that are also B+
- **percent_of_B**: % of B+ cells that are also A+
- **jaccard**: Jaccard similarity index (intersection/union)

### Configuration

The module automatically reads from `config['phenotypes']`. No additional configuration needed.

Example config structure:
```yaml
phenotypes:
  Tumor:
    positive: ['TOM']
  pERK_positive_tumor:
    base: 'Tumor'
    positive: ['PERK']
  CD8_T_cells:
    positive: ['CD3E', 'CD8A']
  # ... more phenotypes
```

---

## Module 2: SpatialOverlapAnalysis

**File:** `/home/user/cifsQuant/spatial_quantification/analyses/spatial_overlap_analysis.py`

### Features

- **Spatial Region Detection**: Uses SpatialCells to detect marker+ spatial regions
- **Region Overlap**: Calculates overlap even when cells aren't double-positive
- **Multiple Metrics**: Jaccard index, overlap coefficient, Dice coefficient, area overlaps
- **Overlap vs Coexpression**: Compares spatial overlap with cellular coexpression
- **Multi-region Analysis**: Analyzes 3+ region overlaps

### Usage

```python
from pathlib import Path
from spatial_quantification.analyses import SpatialOverlapAnalysis

# Initialize
analysis = SpatialOverlapAnalysis(
    adata=adata,           # AnnData object with spatial coordinates
    config=config,         # Config dictionary
    output_dir=Path('results')
)

# Run analysis
results = analysis.run()

# Access detected boundaries
boundaries = analysis.get_region_boundaries()

# Get specific boundary
boundary = analysis.get_boundary(
    sample='sample_1',
    phenotype='pERK_positive_tumor',
    region_id=0
)
```

### Output Files

```
results/spatial_overlap_analysis/
├── detected_regions.csv                   # All detected spatial regions
├── pairwise_overlap.csv                   # Detailed overlap for all region pairs
├── pairwise_overlap_summary.csv           # Summary statistics per phenotype pair
├── multi_region_overlap.csv               # 3-way region overlaps
├── overlap_vs_coexpression.csv            # Spatial vs cellular comparison
└── plots/
    ├── spatial_overlap_heatmap_*.png      # Heatmaps by group
    ├── overlap_vs_coexpression_comparison.png  # Scatter plots
    └── top_overlapping_pairs.png          # Top 20 overlapping pairs
```

### Metrics Calculated

For each region pair:

- **intersection_area_um2**: Overlapping area in square microns
- **union_area_um2**: Combined area (union)
- **jaccard_index**: Intersection / Union
- **overlap_coefficient**: Intersection / min(area1, area2)
- **dice_coefficient**: 2 * Intersection / (area1 + area2)
- **percent_of_region1**: % of region1 that overlaps with region2
- **percent_of_region2**: % of region2 that overlaps with region1

### Configuration

Add to your config file:

```yaml
spatial_overlap_analysis:
  region_detection:
    eps: 50              # DBSCAN epsilon for community detection
    min_samples: 10      # Minimum cells to form a community
    alpha: 50            # Alpha shape parameter for boundary
    core_only: true      # Use only core samples
    min_area: 5000       # Minimum area for regions (μm²)
    min_edges: 15        # Minimum edges for boundary polygons
```

If not specified, sensible defaults will be used.

### Key Insights

This analysis can reveal:

1. **Spatial segregation**: Phenotypes with low cellular coexpression but high spatial overlap (adjacent but not mixed)
2. **True mixing**: High both cellular and spatial overlap
3. **Spatial exclusion**: Low overlap despite both phenotypes being present
4. **Regional heterogeneity**: Different overlap patterns in different tumor regions

---

## Integration with Pipeline

### Adding to existing pipeline

```python
# In your main analysis script:
from spatial_quantification.analyses import (
    CoexpressionAnalysisComprehensive,
    SpatialOverlapAnalysis
)

# After loading adata and config:

# 1. Run comprehensive coexpression analysis
print("Running comprehensive coexpression analysis...")
coexp_analysis = CoexpressionAnalysisComprehensive(adata, config, output_dir)
coexp_results = coexp_analysis.run()

# 2. Run spatial overlap analysis
print("Running spatial overlap analysis...")
overlap_analysis = SpatialOverlapAnalysis(adata, config, output_dir)
overlap_results = overlap_analysis.run()

# Access results
print(f"Detected {len(overlap_analysis.phenotypes)} phenotypes")
print(f"Generated {len(coexp_results)} coexpression result datasets")
print(f"Generated {len(overlap_results)} spatial overlap result datasets")
```

### Standalone usage

```python
#!/usr/bin/env python3
"""Run coexpression and overlap analyses standalone."""

import scanpy as sc
import yaml
from pathlib import Path
from spatial_quantification.analyses import (
    CoexpressionAnalysisComprehensive,
    SpatialOverlapAnalysis
)

# Load data
adata = sc.read_h5ad('gated_data.h5ad')

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set output directory
output_dir = Path('analysis_results')
output_dir.mkdir(exist_ok=True)

# Run analyses
coexp = CoexpressionAnalysisComprehensive(adata, config, output_dir)
coexp_results = coexp.run()

overlap = SpatialOverlapAnalysis(adata, config, output_dir)
overlap_results = overlap.run()

print("✓ Analysis complete!")
```

---

## Understanding the Results

### Coexpression Results

**Single Phenotype Frequencies**
- Shows baseline expression of each phenotype
- Use to understand prevalence before looking at combinations

**Pairwise Coexpression**
- Key column: `{pheno1}_AND_{pheno2}_jaccard`
  - 0.0 = Completely exclusive (no overlap)
  - 0.5 = Moderate overlap
  - 1.0 = Perfect overlap (all cells positive for both)

- Key column: `{pheno1}_AND_{pheno2}_percent_of_{pheno1}`
  - Answers: "Of cells expressing pheno1, what % also express pheno2?"

**Multi-marker Coexpression**
- Identifies rare multi-positive populations
- Useful for finding complex phenotypes (e.g., CD8+ CD103+ GZMB+ triple-positive)

### Spatial Overlap Results

**Detected Regions**
- Lists all spatial regions found for each phenotype
- Includes area, centroid, and cell counts

**Pairwise Overlap Summary**
- `percent_overlapping`: What % of region pairs show any overlap
- `mean_jaccard_index`: Average overlap strength
- Lower Jaccard but high percent_overlapping = many small overlaps

**Overlap vs Coexpression**
- `spatial_exceeds_cellular = True`: Regions overlap more than cells coexpress
  - Interpretation: Phenotypes are spatially adjacent but cells are exclusive
- `spatial_exceeds_cellular = False`: Cells coexpress more than regions overlap
  - Interpretation: Coexpressing cells are scattered, not regionally organized

---

## Requirements

- **Python packages**: pandas, numpy, matplotlib, seaborn
- **For SpatialOverlapAnalysis**: spatialcells (`pip install spatialcells`)
- **AnnData format**: Phenotypes stored as `is_{phenotype}` boolean columns
- **Spatial coordinates**: Stored in `adata.obsm['spatial']` (for SpatialOverlapAnalysis)
- **Metadata**: Sample IDs, groups, timepoints in `adata.obs`

---

## Tips and Best Practices

### Coexpression Analysis

1. **Interpreting Jaccard Index**:
   - < 0.2: Largely exclusive
   - 0.2-0.5: Partial overlap
   - > 0.5: Strong overlap

2. **Focus on relevant pairs**: Not all combinations are biologically meaningful

3. **Check denominators**: `percent_of_A` vs `percent_of_B` can be very different

### Spatial Overlap Analysis

1. **Parameter tuning**: Adjust `eps` and `alpha` based on your marker patterns:
   - Smaller `eps` = stricter communities
   - Smaller `alpha` = more detailed boundaries

2. **Computational cost**: Analysis is O(n²) for region pairs. With many phenotypes and samples, this can be slow.

3. **Interpretation**: High spatial overlap with low coexpression suggests:
   - Phenotypes occupy same anatomical location
   - But individual cells are mutually exclusive
   - Example: pERK+ and pERK- cells can be in same tumor region

4. **Region size matters**: Very small regions may show artificial overlap

---

## Example Output Interpretation

### Example 1: Perfect Coexpression

```
pERK_AND_NINJA_jaccard: 0.95
pERK_AND_NINJA_percent_of_pERK: 98%
pERK_AND_NINJA_percent_of_NINJA: 97%
```
**Interpretation**: pERK and NINJA are nearly always coexpressed

### Example 2: Asymmetric Coexpression

```
CD8_T_cells_AND_CD8_cytotoxic_percent_of_CD8_T_cells: 30%
CD8_T_cells_AND_CD8_cytotoxic_percent_of_CD8_cytotoxic: 100%
```
**Interpretation**: All cytotoxic cells are CD8+ T cells (expected), but only 30% of CD8+ T cells are cytotoxic

### Example 3: Spatial Segregation

```
Spatial Jaccard: 0.15
Cellular Jaccard: 0.02
spatial_exceeds_cellular: True
```
**Interpretation**: Phenotypes occupy adjacent spatial regions but cells don't coexpress - spatial heterogeneity without cellular mixing

---

## Troubleshooting

**Issue**: "No phenotypes found in config"
- **Solution**: Ensure config has `phenotypes:` section with defined phenotypes

**Issue**: "SpatialCells required"
- **Solution**: `pip install spatialcells`

**Issue**: "No regions detected"
- **Solution**: Lower `min_samples` or `eps` parameters, check that phenotype columns exist

**Issue**: Too many combinations (memory error)
- **Solution**: Multi-marker analysis is limited to first 10 phenotypes to prevent combinatorial explosion

**Issue**: Plots not generating
- **Solution**: Ensure matplotlib backend is set correctly, check write permissions

---

## Contact & Support

For issues or questions about these modules:
- Check the code documentation in the module files
- Review example usage in this guide
- Ensure config file is properly formatted
- Verify input data has required columns and structure
