# Advanced Spatial Analysis - Integration Guide

## Overview

This integration extends the existing **ComprehensiveTumorSpatialAnalysis** pipeline (Phases 1-10) with advanced multi-level spatial analysis methods (Phases 11-18), following MCMICRO/SCIMAP workflows.

## Architecture

### Existing Pipeline (Phases 1-10)
Located in `tumor_spatial_analysis_comprehensive.py`:
1. Tumor Structure Detection
2. Infiltration Quantification
3. Marker Expression Analysis
4. Tumor Size Analysis
5. Cellular Neighborhood Analysis
6. Spatial Distance Analysis
7. Co-localization Analysis
8. Comprehensive Statistical Analysis
9. All Visualizations
10. Summary Report

### Advanced Extensions (Phases 11-18)
Located in `advanced_spatial_extensions.py`:
11. Enhanced Phenotyping (auto-thresholding)
12. pERK Spatial Architecture (clustering, growth, infiltration)
13. NINJA Escape Mechanism Analysis
14. Heterogeneity Emergence & Evolution
15. Enhanced RCN Temporal Dynamics
16. Multi-Level Distance Analysis
17. Infiltration-Tumor Associations
18. Pseudo-Temporal Trajectory Analysis

## Usage

### Option 1: Comprehensive Analysis Only (Phases 1-10)

```bash
python run_comprehensive_analysis.py \
    --config configs/comprehensive_config.yaml
```

### Option 2: Comprehensive + Advanced Analysis (Phases 1-18)

```bash
python run_comprehensive_with_advanced.py \
    --config configs/advanced_spatial_config.yaml \
    --metadata sample_metadata.csv \
    --run-advanced
```

### Option 3: Advanced Analysis Only (Phases 11-18)

```bash
python run_comprehensive_with_advanced.py \
    --config configs/advanced_spatial_config.yaml \
    --metadata sample_metadata.csv \
    --skip-comprehensive \
    --run-advanced
```

## Configuration

### Using Existing Config
Use `configs/comprehensive_config.yaml` for standard analysis (Phases 1-10).

### Using Advanced Config
Use `configs/advanced_spatial_config.yaml` for extended analysis. This config includes:

```yaml
# Enable/disable specific advanced phases
perk_analysis:
  enabled: true
  cluster_eps: 30
  hotspot_radius: 50

ninja_analysis:
  enabled: true
  enrichment_radii: [30, 50, 100]

heterogeneity_analysis:
  enabled: true
  markers_for_diversity: [PERK, AGFP, KI67]

# ... and more
```

## Programmatic Usage

```python
from tumor_spatial_analysis_comprehensive import ComprehensiveTumorSpatialAnalysis
from advanced_spatial_extensions import add_advanced_methods
import scanpy as sc
import yaml

# Load data and config
adata = sc.read_h5ad('manual_gating_output/gated_data.h5ad')
config = yaml.safe_load(open('configs/advanced_spatial_config.yaml'))

# Initialize analysis
analysis = ComprehensiveTumorSpatialAnalysis(
    adata=adata,
    sample_metadata=metadata,
    tumor_markers=config['tumor_markers'],
    immune_markers=config['immune_markers'],
    output_dir=config['output_directory']
)

# Run comprehensive analysis (Phases 1-10)
analysis.run_complete_analysis(
    population_config=config['populations'],
    immune_populations=config['immune_infiltration']['populations']
)

# Add and run advanced analysis (Phases 11-18)
add_advanced_methods(analysis)
analysis.run_advanced_analysis(config)
```

## Output Structure

```
comprehensive_spatial_analysis_advanced/
├── data/                          # From Phases 1-10
│   ├── tumor_structures.csv
│   ├── infiltration_metrics.csv
│   └── ...
├── statistics/                    # From Phases 1-10
├── figures/                       # From Phases 1-10
│   ├── spatial_maps/
│   ├── temporal/
│   └── ...
├── advanced_perk_analysis/        # From Phase 12
│   ├── perk_clustering_analysis.csv
│   ├── perk_growth_dynamics.csv
│   └── perk_infiltration_differential.csv
├── advanced_ninja_analysis/       # From Phase 13
├── advanced_heterogeneity/        # From Phase 14
├── advanced_rcn/                  # From Phase 15
├── advanced_distances/            # From Phase 16
├── advanced_infiltration/         # From Phase 17
└── advanced_pseudotime/           # From Phase 18
```

## Key Features

### Seamless Integration
- Advanced methods are added dynamically using `types.MethodType`
- No modification of existing codebase required
- Can run phases independently or together

### Config-Driven
- All parameters controlled via YAML
- Enable/disable individual analysis modules
- Compatible with existing config structure

### Backward Compatible
- Existing scripts continue to work unchanged
- New functionality is opt-in via `--run-advanced` flag

## Implementation Status

### ✅ Fully Implemented (Phase 12)
- pERK spatial clustering (Moran's I)
- pERK growth dynamics tracking
- pERK infiltration differential analysis

### 📋 Placeholder (Phases 11, 13-18)
- Framework in place for full implementation
- Integration points defined
- Can be expanded with complete algorithms

## Files Modified/Added

### New Files
- `configs/advanced_spatial_config.yaml` - Extended configuration
- `advanced_spatial_extensions.py` - Extension methods (Phases 11-18)
- `run_comprehensive_with_advanced.py` - Integrated runner script
- `INTEGRATION_README.md` - This file

### Unchanged Files
- `tumor_spatial_analysis_comprehensive.py` - No modifications
- `run_comprehensive_analysis.py` - No modifications
- `configs/comprehensive_config.yaml` - No modifications

## Development

### Adding New Analysis Methods

1. Define method in `advanced_spatial_extensions.py`:
```python
def phase19_custom_analysis(self, config: dict):
    """Phase 19: Custom analysis."""
    # Implementation here
    pass
```

2. Register in `add_advanced_methods()`:
```python
analysis_instance.phase19_custom_analysis = types.MethodType(
    phase19_custom_analysis, analysis_instance
)
```

3. Call in `run_advanced_analysis()`:
```python
self.phase19_custom_analysis(config)
```

## Requirements

Same as existing pipeline:
- scanpy
- numpy, pandas
- scipy, scikit-learn
- matplotlib, seaborn
- statsmodels

## Citation

Extends the work in:
- Nirmal et al. 2021 (SCIMAP)
- Schapiro et al. 2017 (histoCAT)
- Schapiro et al. 2021 (MCMICRO)

## Support

For issues with:
- **Comprehensive analysis (Phases 1-10)**: Existing support channels
- **Advanced extensions (Phases 11-18)**: GitHub issues with `[advanced]` tag
