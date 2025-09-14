# MCMICRO Tiled Processor

**An integrated workflow for processing large immunofluorescence images through tiling, segmentation, and spatial analysis**

## Overview

This package provides two integrated solutions for handling large (60GB+) multichannel immunofluorescence images that are too big for standard MCMICRO processing:

1. **Nextflow Workflow** (Recommended): Fully automated pipeline with built-in parallelization
2. **Python Script**: Standalone processor with full control over parameters

Both approaches:
- ✅ Tile your large image into manageable pieces
- ✅ Process each tile through MCMICRO-compatible segmentation  
- ✅ Automatically stitch results back together
- ✅ Maintain single-cell data with spatial coordinates
- ✅ Enable downstream spatial analysis with SCIMAP
- ✅ Support spatial questions about cell phenotypes and RCNs (Recurrent Cellular Neighborhoods)

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# For Nextflow approach (recommended):
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/
```

### Option 1: Nextflow Workflow (Recommended)

```bash
# Basic usage
nextflow run mcmicro-tiled.nf \
    --input_image your_60GB_image.tiff \
    --outdir results \
    --sample_name my_sample

# With all parameters
nextflow run mcmicro-tiled.nf \
    --input_image large_image.tiff \
    --markers_csv markers.csv \
    --outdir results \
    --sample_name large_IF_sample \
    --tile_size 2048 \
    --overlap 256 \
    --pyramid_level 0 \
    -profile docker
```

### Option 2: Python Script

```bash
# Basic usage  
python mcmicro_tiled_processor.py \
    --input your_60GB_image.tiff \
    --output results \
    --sample-name my_sample

# With all parameters
python mcmicro_tiled_processor.py \
    --input large_image.tiff \
    --output results \
    --markers markers.csv \
    --tile-size 2048 \
    --overlap 256 \
    --workers 6
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tile_size` | 2048 | Size of square tiles (pixels) |
| `overlap` | 256 | Overlap between tiles (pixels) |
| `pyramid_level` | 0 | Which resolution level (0 = highest) |
| `workers` | 4 | Number of parallel processes |

## Output Structure

```
results/
├── tiles/                          # Individual image tiles
│   ├── tile_y000000_x000000.tiff
│   ├── tile_y000000_x002048.tiff
│   └── tile_info.json              # Tile metadata
├── processed_tiles/                # Segmented tiles  
│   ├── tile_y000000_x000000_mask.tiff
│   ├── tile_y000000_x000000.csv    # Single-cell data per tile
│   └── ...
├── final/                          # Stitched results
│   ├── full_segmentation_mask.tiff # Complete segmentation
│   ├── combined_quantification.csv # All cells with global coordinates
│   └── stitching_report.txt        # Processing summary
└── spatial/                       # Spatial analysis (if enabled)
    ├── spatial_analysis_results.h5ad
    ├── spatial_summary.csv
    └── spatial_analysis_plots.png
```

## Downstream Analysis with SCIMAP

After processing, continue with spatial analysis:

```python
import scimap as sm

# Load results
adata = sm.pp.mcmicro_to_scimap('results/final/combined_quantification.csv')

# Standard SCIMAP workflow  
adata = sm.pp.rescale(adata, gate=99)

# Define cell phenotypes
phenotype = {
    'T cells': ['CD3+'],
    'CD8 T cells': ['CD3+', 'CD8+'],  
    'Tumor cells': ['PanCK+'],
    'Macrophages': ['CD68+']
}
adata = sm.tl.phenotype_cells(adata, phenotype=phenotype)

# Spatial analysis for RCNs and distances
adata = sm.tl.spatial_distance(adata, method='radius', radius=30)
adata = sm.tl.spatial_interaction(adata, method='radius', radius=30)
adata = sm.tl.spatial_cluster(adata, method='kmeans', k=10)

# Distance analysis between specific phenotypes
distances = sm.tl.spatial_distance(adata, 
                                   phenotype='phenotype',
                                   subset=['CD8 T cells', 'Tumor cells'])

# Visualizations
sm.pl.spatial_scatterplot(adata, color=['phenotype'])
sm.pl.spatial_interaction_network(adata)
```

## Advanced Usage

### Cluster Execution (Nextflow)

```bash
# SLURM cluster
nextflow run mcmicro-tiled.nf \
    --input_image large_image.tiff \
    --outdir results \
    -profile slurm \
    --max_memory 64.GB \
    --max_cpus 32

# Custom resource allocation
nextflow run mcmicro-tiled.nf \
    --input_image large_image.tiff \
    --outdir results \
    -c custom.config
```

### Memory Optimization

For very large images or limited memory:

```bash
# Reduce tile size and use lower resolution
nextflow run mcmicro-tiled.nf \
    --input_image large_image.tiff \
    --tile_size 1024 \
    --pyramid_level 1 \
    --outdir results

# Python version with conservative settings
python mcmicro_tiled_processor.py \
    --input large_image.tiff \
    --tile-size 1024 \
    --overlap 128 \
    --pyramid-level 1 \
    --workers 2
```

## Markers CSV Format

Create a `markers.csv` file describing your channels:

```csv
marker_name,channel_number,cycle_number
DAPI,1,1
CD3,2,1
CD8,3,1
PanCK,4,1
CD68,5,1
Ki67,6,1
```

## Troubleshooting

### Common Issues

**Memory errors during tiling:**
- Reduce `tile_size` (try 1024)
- Use higher `pyramid_level` (1 or 2)
- Close other applications

**Segmentation fails:**
- Ensure nuclei channel (DAPI/Hoechst) is first channel
- Check image contrast and quality
- Try different overlap settings

**Stitching artifacts:**
- Increase overlap (try 512)
- Check `tile_info.json` for coverage gaps
- Verify tile processing succeeded

**Slow processing:**
- Increase number of workers
- Use cluster execution for Nextflow
- Consider smaller tiles for better parallelization

### Performance Optimization

| Image Size | Recommended Tile Size | Workers | Memory |
|------------|----------------------|---------|---------|
| < 10 GB | 2048 | 4 | 16 GB |
| 10-50 GB | 2048 | 6-8 | 32 GB |
| 50+ GB | 4096 | 8-12 | 64 GB |

## Scientific Background

This workflow addresses the challenge of processing large multiplexed immunofluorescence images for spatial biology analysis. It's designed to work with the established MCMICRO→SCIMAP pipeline while handling images that exceed memory limits of standard segmentation algorithms.

The approach maintains spatial context by:
1. Preserving global coordinates across tiles
2. Handling cell boundaries at tile edges through overlap
3. Maintaining cell-level feature extraction fidelity
4. Enabling neighborhood analysis and RCN detection

## Supported Formats

- **Input**: Pyramidal TIFF, OME-TIFF, standard TIFF
- **Output**: TIFF masks, CSV quantification, H5AD for spatial analysis
- **Platforms**: CODEX, CyCIF, mIHC, IMC (with proper conversion)

## Citation

If you use this workflow, please cite:

- **MCMICRO**: Schapiro et al. (2022) Nature Methods
- **SCIMAP**: Nirmal et al. (2021) Nature Methods  
- **Your analysis**: [Add your publication]

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review output logs in `results/` directory
3. Open an issue on GitHub
4. Contact the MCMICRO community on Slack

## License

This project is licensed under the MIT License - see LICENSE file for details.