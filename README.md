# cifsQuant: Spatial Immunofluorescence Analysis Pipeline

**Multi-level spatial analysis pipeline for cyclic immunofluorescence data using MCMICRO/Cellpose segmentation with comprehensive spatial quantification.**

## Overview

cifsQuant provides an end-to-end workflow for analyzing cyclic immunofluorescence (CycIF) imaging data, from raw OME-TIFF images to publication-ready spatial analysis results.

**Pipeline Capabilities:**
- Automated image tiling and GPU-accelerated Cellpose segmentation
- Marker-based cell phenotyping with percentile normalization
- Tumor structure detection and boundary analysis
- Comprehensive spatial quantification (infiltration, neighborhoods, distances)
- Publication-quality visualizations with statistical testing

---

## Project Structure

```
cifsQuant/
├── rawdata/                              # Raw input data
│   ├── [SAMPLE_NAME]/                    # Per-channel TIFFs (recommended)
│   │   └── *.ome.tif                     # Individual channel files
│   └── [SAMPLE]_aligned_stack.ome.tif    # OR stacked multi-channel TIFFs
│
├── results/                              # MCMICRO segmentation output
│   └── [SAMPLE_NAME]/
│       ├── tiles/                        # Tiled images
│       ├── nuclei_masks/                 # Cellpose nuclei segmentation
│       ├── cell_masks/                   # Cellpose cell segmentation
│       └── final/
│           └── combined_quantification.csv  # Single-cell measurements
│
├── manual_gating_output/                 # Cell phenotyping output
│   ├── gated_data.h5ad                   # Phenotyped cells (AnnData format)
│   ├── gate_definitions.json             # Gate thresholds
│   └── gating_plots/                     # QC visualizations
│
├── spatial_quantification_results/       # Spatial analysis output
│   ├── population_dynamics/
│   ├── distance_analysis/
│   ├── infiltration_analysis/
│   └── neighborhoods/
│
├── scripts/                              # Utility scripts
├── spatial_quantification/               # Spatial analysis modules
├── run_pipeline.sh                       # Main pipeline launcher
├── manual_gating.py                      # Cell phenotyping script
├── nextflow.config                       # Nextflow pipeline config
├── markers.csv                           # Channel definitions
└── sample_metadata.csv                   # Sample grouping metadata
```

---

## Prerequisites

### Required Software
- **Docker** with GPU support (for Cellpose acceleration)
- **Nextflow** (workflow orchestration)
- **Python 3.8+** with packages:
  ```bash
  pip install numpy pandas scipy scikit-learn matplotlib seaborn anndata pyyaml
  ```

### Input Data Requirements
- Raw cyclic immunofluorescence OME-TIFF files (either per-channel or stacked)
- Sample metadata CSV with experimental groupings
- Marker/channel definitions CSV

---

## Complete Workflow

### Stage 0 (Optional): Sample Partitioning

**When to use:** Experiments with multiple samples/conditions on the same slide that need to be separated before processing.

```bash
# Install partitioning dependencies
pip install -r requirements_partition.txt

# Run interactive partitioning for one or more samples
python scripts/partition_samples.py --sample_names SAMPLE1 SAMPLE2 SAMPLE3
```

This will:
1. Generate coordinate grids for each sample
2. Allow interactive selection of partition boundaries
3. Create separate directories (e.g., `SAMPLE1A/`, `SAMPLE1B/`) in `rawdata/`
4. Rename channel files to match partitions

**See [SAMPLE_PARTITIONING_README.md](SAMPLE_PARTITIONING_README.md) for detailed instructions.**

---

### Stage 1: Image Segmentation Pipeline

**Purpose:** Tile large images, segment cells using Cellpose, and quantify marker intensities.

#### Configuration Files (edit per dataset):

1. **`run_pipeline.sh`** - Edit parameters around lines 430-480:
   ```bash
   --tile_size 8192          # Tile dimensions (4096, 8192, etc.)
   --overlap 1024            # Overlap to prevent boundary artifacts
   --dapi_channel 3          # DAPI channel index
   --nuc_diameter 15         # Nuclear diameter (pixels)
   --cyto_diameter 28        # Cell diameter (pixels)
   ```

2. **`nextflow.config`** - Edit Cellpose and processing parameters:
   ```groovy
   params {
       tile_size = 8192
       overlap = 1024
       dapi_channel = 3
       nuc_diameter = 15
       cyto_diameter = 28
       nuclei_batch_size = 6     # Adjust based on GPU memory
       cyto_batch_size_tiles = 4
   }
   ```

3. **`markers.csv`** - Define your antibody panel:
   ```csv
   cycle,marker_name
   1,R1.0.1_DAPI
   1,R1.0.1_CY3
   1,R1.0.2_CY5_CD45
   2,R2.0.1_DAPI
   2,R2.0.2_CY3_PERK
   ...
   ```

4. **`sample_metadata.csv`** - Define sample groupings:
   ```csv
   sample_id,group,timepoint,treatment
   SAMPLE1,control,day0,vehicle
   SAMPLE2,control,day7,vehicle
   SAMPLE3,treated,day0,drug_a
   SAMPLE4,treated,day7,drug_a
   ```

#### Run the Pipeline:

```bash
# Process all samples
bash run_pipeline.sh

# Or process specific samples
bash run_pipeline.sh SAMPLE1 SAMPLE2
```

**Interactive Options:**
- Resume previous tiling (keep existing tiles)
- Resume Nextflow checkpoint (continue from last step)

**Output:**
- `results/[SAMPLE]/final/combined_quantification.csv` - Single-cell intensity measurements
- Segmentation masks, tiles, and QC reports

**Monitor Progress:**
```bash
watch -n 10 '
WORK=$(find work -type f -name ".command.log" -mmin -10 | head -1 | xargs dirname 2>/dev/null);
echo "=== Progress ===";
tail -3 $WORK/.command.log 2>/dev/null;
echo "=== Tiles Created ===";
ls $WORK/tile_*.tif 2>/dev/null | wc -l;
'
```

---

### Stage 2: Cell Phenotyping (Manual Gating)

**Purpose:** Assign cell phenotypes based on marker intensity thresholds using a novel three-tier normalization approach that enables consistent gating across samples while correcting microscope artifacts.

#### Novel Methodology: Hierarchical Multi-Level Normalization

cifsQuant implements a sophisticated normalization pipeline that addresses multiple sources of technical variation:

**1. Tile Boundary Correction (Pre-normalization)**
- **Problem**: MCMICRO tiling creates artificial intensity discontinuities at tile boundaries
- **Solution**: Gradient-based detection and correction of boundary artifacts
- **Applied**: BEFORE UniFORM normalization to ensure clean input data
- **Impact**: Eliminates false-positive marker calls at tile edges

**2. Hierarchical UniFORM Normalization (Within-sample)**
- **Problem**: Microscope tiles (distinct from MCMICRO tiles) show systematic illumination variations
- **Detection**: Spatial intensity pattern analysis identifies physical microscope tile grid
- **Classification**: Tiles categorized as normal, dim, or bright using MAD outlier detection
- **Correction**: UniFORM (Uniform Manifold Approximation) quantile normalization
  - Aligns intensity distributions between dim/bright tiles and normal tiles
  - Preserves biological variation while removing technical artifacts
  - Separate correction strengths for dim tiles (full) vs bright tiles (conservative)
- **Radial Correction**: Optional within-tile vignetting correction (center vs edge)
- **Impact**: Enables shared gates across entire sample despite illumination variations

**3. Cross-Sample Percentile Normalization (Between-samples)**
- **Problem**: Different samples have varying overall intensities
- **Solution**: 99th percentile normalization aligns marker ranges across samples
- **Result**: Single shared gate per marker works across all samples
- **Advantage**: Removes batch effects while preserving biological differences

**4. Hierarchical Marker Relationships**
- **Problem**: Child markers (e.g., FOXP3) can be falsely positive without parent markers (e.g., CD4)
- **Solution**: Enforce biological parent-child relationships
- **Example**: FOXP3+ cells MUST also be CD4+, CD8A+ cells MUST be CD3E+
- **Impact**: Ensures logical consistency and prevents biologically impossible phenotypes

**5. Liberal Gating for Rare Markers**
- **Problem**: Rare functional markers (GZMB, FOXP3, etc.) can be under-called with conservative thresholds
- **Solution**: Configurable gating stringency per marker
- **Implementation**: Reduced peak multiplier, increased valley tolerance for specified markers
- **Impact**: Better capture of rare but important cell populations

#### Configuration (edit per dataset):

Edit the **CONFIGURATION** section at the top of `manual_gating.py` (lines ~30-160):

```python
# Define marker name mappings
MARKERS = {
    'R1.0.1_CY3': 'TOM',
    'R1.0.4_CY5_CD45': 'CD45',
    'R2.0.4_CY3_PERK': 'PERK',
    'R3.0.4_CY3_CD3E': 'CD3E',
    'R4.0.4_CY5_CD8A': 'CD8A',
    # Add your markers here
}

# Tile artifact correction configuration
TILE_CORRECTION_CONFIG = {
    'enabled': True,
    'markers': ['GZMB', 'FOXP3', 'KLRG1', 'PD1'],  # Most affected markers
    'bin_size': 400,                    # Spatial binning resolution
    'outlier_threshold': 2.0,           # MAD units for tile classification
    'correction_strength': 1.0,         # Full correction for dim tiles
    'bright_correction_strength': 1.0,  # Conservative for bright tiles
    'radial_correction': True,          # Within-tile vignetting correction
}

# Hierarchical marker relationships (child: parent)
MARKER_HIERARCHY = {
    'FOXP3': 'CD4',      # Tregs require CD4
    'GZMB': 'CD8A',      # Cytotoxic markers require CD8
    'CD8A': 'CD3E',      # CD8 T cells require CD3
    'CD4': 'CD3E',       # CD4 T cells require CD3
    'PD1': 'CD3E',       # PD1 typically on T cells
    'KLRG1': 'CD8A',     # KLRG1 on effector CD8 T cells
}

# Liberal gating for rare markers
LIBERAL_GATING_CONFIG = {
    'enabled': True,
    'liberal_markers': ['GZMB', 'FOXP3', 'KLRG1', 'PD1'],  # Rare markers
    'liberal_peak_multiplier': 1.9,      # Less stringent than 2.0
    'liberal_valley_max_height': 0.22,   # Tolerates shallower valleys
    'liberal_min_percentile': 82,        # Lower threshold percentile
}
```

#### Run Manual Gating:

```bash
# Full pipeline with normalization (recommended for multi-sample datasets)
python manual_gating.py --results_dir results --n_jobs 16

# Skip normalization (if already normalized or single sample)
python manual_gating.py --results_dir results --skip_normalization

# Force re-normalization
python manual_gating.py --results_dir results --force_normalization --n_jobs 15
```

**Output:**
- `manual_gating_output/gated_data.h5ad` - AnnData file with phenotype assignments
- `manual_gating_output/gate_definitions.json` - Applied gate thresholds
- `manual_gating_output/gating_plots/` - QC visualizations per marker

---

### Stage 3: Spatial Quantification

**Purpose:** Comprehensive spatial analysis including population dynamics, immune infiltration, cellular neighborhoods, and distance-based interactions.

#### Configuration (edit per dataset):

Edit `spatial_quantification/config/spatial_config.yaml`:

```yaml
# Input/output paths
input:
  gated_data: 'manual_gating_output/gated_data.h5ad'
  metadata: 'sample_metadata.csv'

output:
  base_directory: 'spatial_quantification_results'

# Define custom phenotypes
phenotypes:
  pERK_positive_tumor:
    base: 'Tumor'              # Requires TOM+
    positive: ['PERK']
    negative: []

  CD8_T_cells:
    positive: ['CD3E', 'CD8A']
    negative: []

# Configure analyses
population_dynamics:
  enabled: true
  populations:
    - Tumor
    - pERK_positive_tumor
    - CD8_T_cells

  # Automatic fractional calculation
  fractional_populations:
    pERK_positive_tumor: Tumor  # pERK+ fraction of total tumor

  comparisons:
    - name: 'treated_vs_control'
      groups: ['treated', 'control']
      timepoints: [0, 7, 14, 21]

distance_analysis:
  enabled: true
  pairings:
    - source: 'CD8_T_cells'
      targets: ['Tumor', 'pERK_positive_tumor']

immune_infiltration:
  enabled: true
  immune_populations:
    - CD8_T_cells
  boundaries: [0, 50, 100, 200]  # μm from tumor edge

cellular_neighborhoods:
  enabled: true
  populations:
    - Tumor
    - CD8_T_cells
  window_size: 100  # μm
  n_clusters: 8
```

#### Run Spatial Quantification:

```bash
# From project root
python spatial_quantification/run_spatial_quantification.py

# Or from spatial_quantification directory
cd spatial_quantification
python run_spatial_quantification.py

# With custom config
python spatial_quantification/run_spatial_quantification.py --config my_config.yaml
```

**Output:**
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
│   └── CD8_T_cells_to_Tumor_distance_stats.csv
├── infiltration_analysis/
│   ├── infiltration_heatmaps.pdf
│   └── zone_analysis.csv
└── neighborhoods/
    ├── neighborhood_composition.csv
    └── temporal_evolution.pdf
```

**See [spatial_quantification/README.md](spatial_quantification/README.md) for detailed documentation.**

---

## Key Configuration Files Summary

| File | Purpose | When to Edit |
|------|---------|--------------|
| `run_pipeline.sh` | Pipeline execution parameters | Per dataset - tile size, channel indices |
| `nextflow.config` | Cellpose and processing settings | Per dataset - segmentation parameters |
| `markers.csv` | Channel/marker definitions | Per dataset - must match your antibody panel |
| `sample_metadata.csv` | Sample groupings | Per dataset - experimental design |
| `manual_gating.py` (config section) | Marker mappings and gate thresholds | Per dataset - marker names and cutoffs |
| `spatial_quantification/config/spatial_config.yaml` | Spatial analysis settings | Per dataset - phenotypes and comparisons |

---

## Performance Optimization

### Hardware Recommendations

| Image Size | Tile Size | GPU Memory | RAM | Processing Time |
|------------|-----------|------------|-----|-----------------|
| <10GB | 4096 | 8GB | 16GB | 1-2 hours |
| 10-50GB | 8192 | 16GB | 32GB | 3-6 hours |
| 50GB+ | 8192 | 24GB+ | 64GB | 6-12 hours |

### Tips for Large Datasets

1. **Use per-channel TIFFs** instead of stacked TIFFs (5-10x faster tiling)
2. **Adjust batch sizes** in `nextflow.config` based on GPU memory:
   - `nuclei_batch_size`: 4-8 for 16GB GPU
   - `cyto_batch_size_tiles`: 2-6 for 16GB GPU
3. **Enable Nextflow resume** to continue from last checkpoint after failures
4. **Process samples in parallel** on multi-GPU systems

---

## Troubleshooting

### Common Issues

**1. Out of memory during segmentation**
- Reduce `tile_size` to 4096
- Decrease `nuclei_batch_size` and `cyto_batch_size_tiles`
- Increase `pyramid_level` to 1 or 2 (lower resolution)

**2. Docker GPU access fails**
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# If fails, install NVIDIA Docker runtime
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

**3. Missing markers in manual_gating.py**
- Check `results/[SAMPLE]/final/combined_quantification.csv` for actual column names
- Update `MARKERS` dictionary in `manual_gating.py` to match

**4. Spatial quantification file not found**
- Verify paths in `spatial_config.yaml` are relative to project root
- Check that `manual_gating.py` completed successfully
- Ensure `gated_data.h5ad` exists in `manual_gating_output/`

**5. Tiling artifacts at tile boundaries**
- Increase `overlap` parameter (recommended: 1024 for 8192 tiles)
- See [TILE_ARTIFACT_GUIDE.md](TILE_ARTIFACT_GUIDE.md) for details

---

## Scientific Applications

This pipeline enables quantification of:

1. **Cell Population Dynamics** - Track immune and tumor populations over time/treatment
2. **Spatial Infiltration** - Quantify immune cell penetration into tumor structures
3. **Cellular Neighborhoods** - Identify recurrent cellular microenvironments
4. **Distance-Based Interactions** - Measure proximity between cell populations
5. **Marker Coexpression** - Analyze spatial heterogeneity of marker expression
6. **Treatment Response** - Compare spatial organization across conditions

---

## Output Data Formats

### AnnData (`.h5ad`)
- Standard single-cell analysis format
- Compatible with Scanpy, SCIMAP, Squidpy
- Contains: expression matrix, spatial coordinates, metadata, phenotypes

### CSV Files
- Single-cell quantification tables
- Statistical test results
- Population summaries

### Visualizations
- **Raw plots**: Individual data points, full annotations
- **Publication plots**: Clean, high-resolution (300 DPI), ready for manuscripts

---

## Integration with Other Tools

Results are compatible with:
- **SCIMAP** - Advanced spatial analysis workflows
- **Scanpy** - Single-cell analysis and visualization
- **Squidpy** - Spatial single-cell analysis
- **QuPath** - Pathological annotation and visualization
- **napari** - Multi-dimensional image viewer
- **R/Seurat** - Cross-platform spatial analysis

---

## Citation

If you use cifsQuant in your research, please cite:

```
cifsQuant: Spatial Immunofluorescence Analysis Pipeline
https://github.com/bansalis/cifsQuant
```

**Key Dependencies:**
- MCMICRO: Schapiro et al. (2022) *Nature Methods*
- Cellpose: Stringer et al. (2021) *Nature Methods*
- SCIMAP: Nirmal et al. (2021) *Cell Systems*

---

## Support and Documentation

- **General questions**: See documentation in `spatial_quantification/README.md`
- **Sample partitioning**: See `SAMPLE_PARTITIONING_README.md`
- **Tile artifacts**: See `TILE_ARTIFACT_GUIDE.md`
- **Latest updates**: See `LATEST_UPDATES.md` and `FIXES_APPLIED.md`
- **Issues**: https://github.com/bansalis/cifsQuant/issues

---

## Quick Reference Card

```bash
# Full pipeline from scratch
bash run_pipeline.sh                                    # Stage 1: Segmentation
python manual_gating.py --results_dir results --n_jobs 16  # Stage 2: Phenotyping
python spatial_quantification/run_spatial_quantification.py  # Stage 3: Spatial analysis

# Configuration files to edit per dataset
# 1. run_pipeline.sh (lines 430-480)
# 2. nextflow.config (params section)
# 3. markers.csv
# 4. sample_metadata.csv
# 5. manual_gating.py (CONFIGURATION section, lines 30-60)
# 6. spatial_quantification/config/spatial_config.yaml
```

---

**Ready to analyze your spatial immunofluorescence data!**
