# Spatial Immunofluorescence Analysis Pipeline

Multi-level spatial analysis pipeline for cyclic immunofluorescence data using MCMICRO/Cellpose segmentation with SCIMAP-compatible workflows.

## Project Structure

```
.
├── rawdata/                          # Raw pyramidal OME-TIFF files
├── results/                          # MCMICRO tiling/segmentation output
│   └── [SAMPLE_NAME]/
│       ├── cell_masks/
│       ├── final/
│       │   └── combined_quantification.csv  # Main single-cell data
│       ├── nuclei_masks/
│       ├── spatial/
│       └── tiles/
├── phenotype_analysis/               # Phase 1: Cell phenotyping output
├── tumor_structures/                 # Phase 2: Tumor structure analysis
├── spatial_interactions/             # Phase 3: Immune-tumor interactions
├── neighborhood_analysis/            # Phase 4: Higher-order neighborhoods
├── scripts/
│   ├── phenotype_analysis.py
│   ├── tumor_structure_detection.py
│   └── [additional analysis scripts]
├── mcmicro-tiled.nf                 # Nextflow segmentation pipeline
├── nextflow.config
├── Dockerfile.phenotype             # Docker image for spatial analysis
├── markers.csv                      # Channel definitions
└── sample_metadata.csv              # Sample group assignments
```

## Prerequisites

- Docker with GPU support
- Nextflow
- Raw cyclic immunofluorescence OME-TIFF files
- Sample metadata CSV

## Pipeline Workflow

### Stage 1: Image Segmentation (MCMICRO/Cellpose)

Run the tiling and segmentation pipeline on raw images:

```bash
# Single sample
bash run_pipeline.sh
```

**Output**: `results/[SAMPLE]/final/combined_quantification.csv`

check status

watch -n 10 '
WORK=$(find work -type f -name ".command.log" -mmin -10 | head -1 | xargs dirname 2>/dev/null);
echo "Work: $WORK";
echo "";
echo "=== Progress (from log) ===";
tail -3 $WORK/.command.log 2>/dev/null;
echo "";
echo "=== Tiles Created ===";
ls $WORK/tile_*.tif 2>/dev/null | wc -l;
echo "";
echo "=== CPU Usage ===";
docker stats --no-stream | grep nxf
'

### Stage 2: Cell Phenotyping and Statistical Analysis

Build Docker image:
```bash
docker build -f Dockerfile.phenotype -t phenotype-analysis .
```

Run phenotyping analysis:
```bash
docker run --rm --gpus all -v $(pwd):/app phenotype-analysis \
  --input results/ \
  --metadata sample_metadata.csv \
  --markers markers.csv \
  --phenotypes phenotype_definitions.csv \
  --output phenotype_analysis/

```

**Output**: 
- `phenotype_analysis/[SAMPLE]/phenotyped_cells.csv`
- `phenotype_analysis/[SAMPLE]/phenotype_statistics.csv`
- `phenotype_analysis/plots/`

### Stage 3: Tumor Structure Detection

Detect tumor structures using phenotyped cells:

```bash
docker run --rm --gpus all \
    -v $(pwd):/app \
    -v $(pwd)/results:/app/results \
    --entrypoint python \
    phenotype-analysis \
    tumor_structure_detection.py \
    --input results/ \
    --output results/

done
```

**Output**:
- `tumor_structures/[SAMPLE]/cells_with_tumor_regions.csv`
- `tumor_structures/[SAMPLE]/tumor_statistics.csv`
- Visualization plots

## Sample Metadata Format

Create `sample_metadata.csv`:
```csv
sample_name,group,treatment,timepoint
SAMPLE1,control,vehicle,day0
SAMPLE2,control,vehicle,day7
SAMPLE3,treated,drug_a,day0
SAMPLE4,treated,drug_a,day7
```

# Then run Minerva setup
python setup_minerva_viewer.py \
  --results-dir results/GUEST29 \
  --sample-name GUEST29 \
  --channels channel_names.txt \
  --phenotypes phenotype_definitions.csv

# Start Minerva
cd results/GUEST29/minerva_setup
./start_minerva.sh

# Access at: http://localhost:3000

## Markers Configuration

Update `markers.csv` to match your antibody panel:
```csv
cycle,marker_name
1,DAPI
1,PanCK
2,CD45
2,CD3
3,CD4
3,CD8
4,FOXP3
4,Ki67
```

## Key Parameters

### Segmentation Parameters
- `tile_size`: 2048 (recommended for 60GB+ images)
- `overlap`: 256 (prevents boundary artifacts)
- `pyramid_level`: 1 (balance speed vs resolution)

### Phenotyping Parameters
- `intensity_threshold`: 0.75 (marker positivity cutoff)
- `area_filter`: 20-2000 (cell size limits)

### Tumor Detection Parameters
- `min_tumor_size`: 50 (minimum cells per tumor)
- `dilation_radius`: 30 (morphological boundary extension)
- `dbscan_eps`: 50 (spatial clustering distance)

## Analysis Outputs

### Phase 1: Cell Phenotyping
- **phenotyped_cells.csv**: Single-cell data with phenotype assignments
- **phenotype_statistics.csv**: Cell counts and percentages by group
- **phenotype_distribution_plots.png**: Spatial and statistical visualizations

### Phase 2: Tumor Structure Detection
- **cells_with_tumor_regions.csv**: Cells assigned to tumor structures
- **tumor_statistics.csv**: Tumor metrics (size, infiltration, shape)
- **tumor_structure_analysis.png**: Spatial tumor boundaries and metrics

### Phase 3: Immune-Tumor Interactions (Coming Soon)
- Spatial proximity analysis
- Contact frequency measurements
- Immune infiltration patterns

### Phase 4: Neighborhood Analysis (Coming Soon)
- Cellular neighborhoods (RCNs)
- Spatial clustering patterns
- Treatment response signatures

## Troubleshooting

### Common Issues

1. **Memory errors during segmentation**:
   - Reduce `tile_size` to 1024
   - Increase `pyramid_level` to 2

2. **Docker GPU access**:
   ```bash
   # Test GPU access
   docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
   ```

3. **Missing phenotypes**:
   - Check marker channel assignments in `markers.csv`
   - Verify intensity thresholds with sample images

4. **Empty tumor detection**:
   - Verify tumor marker (PanCK) is correctly identified
   - Adjust `min_tumor_size` parameter

### Performance Optimization

| Image Size | Tile Size | Workers | Memory |
|------------|-----------|---------|---------|
| <10GB | 2048 | 4 | 16GB |
| 10-50GB | 2048 | 6 | 32GB |
| 50GB+ | 4096 | 8 | 64GB |

## Scientific Applications

This pipeline enables analysis of:

1. **Cell Phenotype Distributions**: Statistical comparison of immune populations
2. **Tumor Architecture**: Size, shape, and heterogeneity metrics  
3. **Immune Infiltration**: Spatial relationships and contact patterns
4. **Treatment Responses**: Longitudinal changes in spatial organization
5. **Cellular Neighborhoods**: Higher-order tissue organization patterns

## Integration with Downstream Analysis

Results are compatible with:
- **SCIMAP**: Load phenotyped data for advanced spatial analysis
- **Scanpy**: Single-cell analysis workflows
- **R/Seurat**: Cross-platform spatial analysis
- **QuPath**: Pathological visualization and annotation

## References

- MCMICRO: Schapiro et al. (2022) Nature Methods
- SCIMAP: Nirmal et al. (2021) Nature Methods  
- Cellpose: Stringer et al. (2021) Nature Methods

## Support

For issues:
1. Check log files in respective output directories
2. Verify Docker image compatibility
3. Validate input file formats and metadata
4. Review parameter settings for your tissue type