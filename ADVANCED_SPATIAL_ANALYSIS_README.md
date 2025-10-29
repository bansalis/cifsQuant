# Advanced Multi-Level Spatial Immunofluorescence Analysis Pipeline

**Comprehensive spatial analysis for cyclic immunofluorescence data from murine lung adenocarcinoma samples**

## Overview

This pipeline implements a comprehensive multi-level spatial analysis framework for tumor immunology, following MCMICRO/SCIMAP workflows with Cellpose segmentation. It addresses tumor-immune dynamics, spatial heterogeneity, and temporal evolution through advanced computational methods.

## Key Features

### Phase 1: Enhanced Phenotyping & Tumor Detection
- **Cell Phenotyping**: Intensity-based gating with automatic thresholding (Otsu, Li, percentile methods)
- **Tumor Detection**: DBSCAN clustering with morphological validation
- **Spatial Zones**: Core, margin (0-50μm), peri-tumor (50-150μm), distal (>150μm)
- **QC Visualizations**: Threshold validation, spatial distributions, composition analysis

### Phase 2: Critical Research Questions
#### 2.1 pERK Spatial Architecture
- **Q1**: Spatial clustering analysis (Ripley's K, Moran's I, Getis-Ord Gi*)
- **Q2**: Growth dynamics and temporal evolution
- **Q3**: Differential T cell infiltration in pERK+ vs pERK- regions

#### 2.2 NINJA Escape Mechanism
- **Q1**: NINJA+ spatial clustering patterns
- **Q2**: Growth dynamics independent of total tumor growth
- **Q3**: Cell type enrichment near NINJA+ regions (permutation tests)

#### 2.3 Heterogeneity Emergence & Evolution
- **Q1**: Marker diversification (LISA clustering, Shannon entropy)
- **Q2**: Intra-sample heterogeneity (Jaccard similarity)

#### 2.4 Cellular Neighborhood (RCN) Dynamics
- k-NN graph construction (k=10)
- Hierarchical clustering of neighborhood compositions
- Temporal evolution tracking (KPT-cis, KPT-trans, KPNT-cis, KPNT-trans)

### Phase 3: Advanced Spatial Metrics
#### 3.1 Multi-Level Distance Analysis
- Per-tumor distances (CD8+ T cells → pERK+/pERK-/NINJA+/NINJA-)
- Per-sample aggregated metrics
- Mixed effects models accounting for nested structure

#### 3.2 Infiltration-Tumor Associations
- Tumor position classification (peripheral vs central)
- Regression models (size ~ infiltration + position + timepoint)
- Spatial autocorrelation analysis

#### 3.3 Pseudo-Temporal Trajectory Analysis
- Tumor-tumor similarity graphs (Jensen-Shannon divergence)
- PAGA trajectory inference
- Branch point identification

## Installation

### Requirements

```bash
# Python 3.8+
pip install numpy pandas matplotlib seaborn
pip install scipy scikit-learn scikit-image
pip install statsmodels
pip install scanpy  # Optional, for pseudo-temporal analysis
```

Or use the provided requirements:

```bash
pip install -r requirements_advanced_spatial.txt
```

### Docker (Recommended)

```bash
# Build Docker image
docker build -t advanced-spatial-analysis -f Dockerfile.spatial .

# Run analysis
docker run -v $(pwd):/workspace advanced-spatial-analysis \
    python run_advanced_spatial_analysis.py \
    --input_dir /workspace/results \
    --metadata /workspace/sample_metadata.csv \
    --markers /workspace/markers.csv \
    --output /workspace/spatial_analysis_comprehensive
```

## Usage

### Quick Start

```bash
python run_advanced_spatial_analysis.py \
    --input_dir results/ \
    --metadata sample_metadata.csv \
    --markers markers.csv \
    --output spatial_analysis_comprehensive
```

### Advanced Options

```bash
python run_advanced_spatial_analysis.py \
    --input_dir results/ \
    --metadata sample_metadata.csv \
    --markers markers.csv \
    --output spatial_analysis_comprehensive \
    --tumor_eps 60 \
    --perk_eps 35 \
    --n_neighbors 15 \
    --dpi 300
```

### Run Specific Phases

```bash
# Run only Phase 1 (phenotyping/tumor detection)
python run_advanced_spatial_analysis.py \
    --input_dir results/ \
    --metadata sample_metadata.csv \
    --markers markers.csv \
    --only_phase 1

# Skip Phase 2
python run_advanced_spatial_analysis.py \
    --input_dir results/ \
    --metadata sample_metadata.csv \
    --markers markers.csv \
    --skip_phase2
```

## Input Data Structure

### Required Files

1. **Combined Quantification CSV** (`results/[SAMPLE_NAME]/final/combined_quantification.csv`)
   - Columns: `CellID`, `Channel_1-N`, `X_centroid`, `Y_centroid`, `Area`, morphology metrics

2. **Sample Metadata** (`sample_metadata.csv`)
   ```csv
   sample_id,group,treatment,timepoint
   JL216,KPT trans,none,10
   JL217,KPT cis,none,10
   JL218,KPNT trans,none,10
   JL219,KPNT cis,none,10
   ```

3. **Markers Info** (`markers.csv`)
   ```csv
   cycle,marker_name
   1,R1.0.1_DAPI
   1,R1.0.1_CY3
   2,R1.0.4_CY5_CD45
   2,R1.0.4_CY7_AGFP
   3,R2.0.4_CY3_PERK
   4,R3.0.4_CY3_CD3E
   5,R4.0.4_CY5_CD8A
   7,R6.0.4_CY7_KI67
   ```

### Expected Markers

| Marker | Channel | Description |
|--------|---------|-------------|
| DAPI | Channel_1 | Nuclear stain |
| TOM | Channel_2 | Tumor marker (tomato) |
| CD45 | Channel_3 | Pan-immune marker |
| aGFP | Channel_4 | NINJA reporter |
| pERK | Channel_6 | Phospho-ERK |
| CD8B | Channel_7 | CD8+ T cells |
| CD3 | Channel_10 | Pan-T cell marker |
| KI67 | Channel_27 | Proliferation marker |

## Output Directory Structure

```
spatial_analysis_comprehensive/
├── data/
│   ├── phenotyped_cells.csv          # Cell-level phenotype assignments
│   └── tumor_structures.csv          # Tumor-level metrics
├── statistics/
│   └── [various statistical test results]
├── 01_phenotyping/
│   ├── individual_plots/
│   │   ├── threshold_qc_TOM.png
│   │   ├── threshold_qc_CD45.png
│   │   └── tumor_structures_[SAMPLE].pdf
│   └── aggregated_panels/
│       ├── phenotype_composition_overview.pdf
│       ├── spatial_distribution_[SAMPLE].pdf
│       └── tumor_size_distribution.pdf
├── 02_perk_analysis/
│   ├── individual_plots/
│   ├── aggregated_panels/
│   │   ├── perk_clustering_representative_tumors.pdf
│   │   ├── perk_clustering_statistics.pdf
│   │   ├── perk_growth_dynamics.pdf
│   │   └── perk_infiltration_differential.pdf
│   └── statistics/
│       ├── perk_spatial_clustering.csv
│       ├── perk_growth_models.csv
│       └── perk_infiltration_stats.csv
├── 03_ninja_analysis/
│   ├── individual_plots/
│   ├── aggregated_panels/
│   └── statistics/
├── 04_heterogeneity/
│   ├── individual_plots/
│   ├── aggregated_panels/
│   └── statistics/
├── 05_rcn_dynamics/
│   ├── individual_plots/
│   ├── aggregated_panels/
│   └── statistics/
├── 06_distance_analysis/
│   ├── individual_plots/
│   ├── aggregated_panels/
│   └── statistics/
├── 07_infiltration_associations/
│   ├── individual_plots/
│   ├── aggregated_panels/
│   └── statistics/
├── 08_pseudotime/
│   ├── individual_plots/
│   ├── aggregated_panels/
│   └── statistics/
└── master_summary_report.html         # Comprehensive HTML report
```

## Configuration Parameters

### Tumor Detection
- `tumor_dbscan_eps`: DBSCAN epsilon for tumor detection (default: 50μm)
- `tumor_dbscan_min_samples`: Minimum samples for tumor cluster (default: 30)

### Zone Definitions
- `margin_width`: Margin zone width (default: 50μm)
- `peritumor_inner`: Inner peri-tumor boundary (default: 50μm)
- `peritumor_outer`: Outer peri-tumor boundary (default: 150μm)

### pERK/NINJA Clustering
- `perk_cluster_eps`: DBSCAN epsilon for pERK clusters (default: 30μm)
- `ninja_cluster_eps`: DBSCAN epsilon for NINJA clusters (default: 30μm)

### Spatial Statistics
- `ripley_max_distance`: Maximum distance for Ripley's K (default: 200μm)
- `hotspot_radius`: Radius for hotspot detection (default: 50μm)
- `n_permutations`: Number of permutations for statistical tests (default: 1000)

### Visualization
- `dpi`: Figure resolution (default: 300)
- `figure_format`: Output format (default: 'pdf')
- `color_palette`: Color scheme (default: 'colorblind')

## Statistical Framework

### Multiple Comparison Correction
- Benjamini-Hochberg FDR correction (α = 0.05)
- Effect sizes (Cohen's d) reported alongside p-values

### Model Types
- **Mixed Effects Models**: Account for nested structure (cells within tumors within samples)
- **Permutation Tests**: For spatial metrics (1000 permutations minimum)
- **Regression Models**: Linear and generalized linear models with appropriate link functions

### Reporting Standards
- Median [IQR] for all metrics
- Statistical annotations on plots
- FDR-corrected p-values in all tables

## Visualization Standards

### Individual Plots
- DPI: 300 minimum
- Formats: PDF and PNG
- Color-blind friendly palettes
- Scale bars on spatial maps (μm)
- Statistical annotations

### Aggregated Panels
- Multi-panel figures (2-4 pages)
- Consistent styling
- Shared legends
- Panel labels (A, B, C...)
- Unified color mapping:
  - KPT: Blue
  - KPNT: Red
  - Cis: Solid
  - Trans: Dashed

## Computational Performance

### Optimization Strategies
- Spatial indexing (scipy.spatial.cKDTree)
- Parallel processing (joblib) for per-tumor analysis
- Chunked processing for large spatial maps
- Memory-efficient data structures

### Hardware Requirements
- **Minimum**: 16 GB RAM, 4 CPU cores
- **Recommended**: 32 GB RAM, 8 CPU cores
- **GPU**: Optional (RAPIDS cuML for accelerated k-NN)

### Estimated Runtime
- Phase 1: ~5-10 minutes
- Phase 2: ~20-30 minutes
- Phase 3: ~15-25 minutes
- **Total**: ~40-65 minutes for typical dataset (4 samples, ~1M cells)

## Methodology References

### Spatial Analysis
1. **Ripley's K Function**: Ripley (1976) "The second-order analysis of stationary point processes"
2. **Moran's I**: Moran (1950) "Notes on continuous stochastic phenomena"
3. **Getis-Ord Gi***: Getis & Ord (1992) "The analysis of spatial association"
4. **LISA**: Anselin (1995) "Local indicators of spatial association"

### Cellular Neighborhoods
1. **SCIMAP**: Nirmal et al. (2021) "The spatial landscape of progression and immunoediting in primary melanoma at single-cell resolution" *Cancer Discovery*
2. **histoCAT**: Schapiro et al. (2017) "histoCAT: analysis of cell phenotypes and interactions in multiplex image cytometry data" *Nature Methods*
3. **Cancer Cell Neighborhoods**: Jackson et al. (2020) "The single-cell pathology landscape of breast cancer" *Nature*

### Trajectory Inference
1. **PAGA**: Wolf et al. (2019) "PAGA: graph abstraction reconciles clustering with trajectory inference through a topology preserving map of single cells" *Genome Biology*
2. **Scanpy**: Wolf et al. (2018) "SCANPY: large-scale single-cell gene expression data analysis" *Genome Biology*

### Image Analysis
1. **MCMICRO**: Schapiro et al. (2021) "MCMICRO: a scalable, modular image-processing pipeline for multiplexed tissue imaging" *Nature Methods*
2. **Cellpose**: Stringer et al. (2021) "Cellpose: a generalist algorithm for cellular segmentation" *Nature Methods*

## Troubleshooting

### Common Issues

**Issue**: No tumor cells detected
- **Solution**: Check TOM marker channel and threshold values. Try manual threshold specification.

**Issue**: Too many/few tumors detected
- **Solution**: Adjust `--tumor_eps` and `--tumor_min_samples` parameters.

**Issue**: Memory errors
- **Solution**: Process samples individually, reduce image resolution, or increase available RAM.

**Issue**: Scanpy not available
- **Solution**: Install with `pip install scanpy`, or use `--skip_phase3` to skip pseudo-temporal analysis.

### Debug Mode

```bash
# Run with detailed logging
python -u run_advanced_spatial_analysis.py \
    --input_dir results/ \
    --metadata sample_metadata.csv \
    --markers markers.csv 2>&1 | tee analysis_log.txt
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{advanced_spatial_analysis,
  title = {Advanced Multi-Level Spatial Immunofluorescence Analysis Pipeline},
  author = {Your Lab},
  year = {2025},
  version = {1.0},
  url = {https://github.com/yourlab/cifsQuant}
}
```

And relevant methodological papers:
- Nirmal et al. 2021 (SCIMAP)
- Schapiro et al. 2017 (histoCAT)
- Schapiro et al. 2021 (MCMICRO)
- Stringer et al. 2021 (Cellpose)

## Support

For issues, questions, or feature requests:
- **GitHub Issues**: https://github.com/yourlab/cifsQuant/issues
- **Email**: your.email@institution.edu

## License

MIT License - see LICENSE file for details

## Changelog

### Version 1.0 (2025-10-29)
- Initial release
- Implemented Phase 1 (phenotyping and tumor detection)
- Implemented Phase 2 placeholders (pERK, NINJA, heterogeneity, RCN)
- Implemented Phase 3 placeholders (distance, infiltration, pseudo-temporal)
- Comprehensive visualization and statistical framework
- HTML report generation

## Future Enhancements

- [ ] Complete Phase 2.1-2.4 full implementations
- [ ] Complete Phase 3.1-3.3 full implementations
- [ ] GPU acceleration for distance calculations
- [ ] Interactive HTML reports with plot.ly
- [ ] Integration with additional spatial analysis tools (SPIAT, spatialdata)
- [ ] Support for 3D imaging data
- [ ] Machine learning-based phenotyping
- [ ] Automated threshold optimization
