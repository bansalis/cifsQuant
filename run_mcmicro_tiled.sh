#!/bin/bash

# MCMICRO Tiled Processing - Usage Examples
# =========================================

echo "MCMICRO Tiled Processing Setup and Usage"
echo "========================================="

# Check if we're in the right directory
if [[ ! -f "mcmicro_tiled_processor.py" ]]; then
    echo "Error: mcmicro_tiled_processor.py not found in current directory"
    exit 1
fi

# Setup section
echo ""
echo "1. SETTING UP ENVIRONMENT"
echo "-------------------------"

# Option A: Nextflow approach
if command -v nextflow &> /dev/null; then
    echo "✓ Nextflow found"
else
    echo "Installing Nextflow..."
    curl -s https://get.nextflow.io | bash
    sudo mv nextflow /usr/local/bin/
    echo "✓ Nextflow installed"
fi

# Option B: Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt 2>/dev/null || pip install numpy pandas tifffile scikit-image scipy matplotlib seaborn scimap anndata tqdm

echo ""
echo "2. USAGE EXAMPLES"
echo "=================="

# Example 1: Nextflow approach
echo ""
echo "Option A: Using Nextflow workflow (Recommended)"
echo "----------------------------------------------"

cat << 'EOF'
# Basic usage with Nextflow
nextflow run mcmicro-tiled.nf \
    --input_image your_large_image.tiff \
    --outdir results \
    --sample_name my_sample \
    --tile_size 2048 \
    --overlap 256

# With custom parameters
nextflow run mcmicro-tiled.nf \
    --input_image /path/to/60GB_image.tiff \
    --markers_csv markers.csv \
    --outdir results \
    --sample_name large_IF_sample \
    --tile_size 4096 \
    --overlap 512 \
    --pyramid_level 0 \
    -profile docker

# For cluster/SLURM execution
nextflow run mcmicro-tiled.nf \
    --input_image your_image.tiff \
    --outdir results \
    -profile slurm \
    --max_memory 64.GB \
    --max_cpus 32
EOF

echo ""
echo "Option B: Using Python script directly"
echo "--------------------------------------"

cat << 'EOF'
# Basic usage
python mcmicro_tiled_processor.py \
    --input your_large_image.tiff \
    --output results \
    --sample-name my_sample

# With all options
python mcmicro_tiled_processor.py \
    --input /path/to/60GB_image.tiff \
    --output results \
    --markers markers.csv \
    --tile-size 2048 \
    --overlap 256 \
    --pyramid-level 0 \
    --sample-name large_IF_sample \
    --workers 8

# For smaller test run
python mcmicro_tiled_processor.py \
    --input test_image.tiff \
    --output test_results \
    --tile-size 1024 \
    --overlap 128 \
    --workers 4
EOF

echo ""
echo "3. EXPECTED OUTPUT STRUCTURE"
echo "============================"

cat << 'EOF'
results/
├── tiles/                          # Individual tiles
│   ├── tile_y000000_x000000.tiff
│   ├── tile_y000000_x002048.tiff
│   ├── ...
│   └── tile_info.json              # Tile metadata
├── processed_tiles/                # Processed tile results
│   ├── tile_y000000_x000000_mask.tiff
│   ├── tile_y000000_x000000.csv
│   └── ...
├── final/                          # Stitched results
│   ├── full_segmentation_mask.tiff # Complete segmentation
│   ├── combined_quantification.csv # All cell data
│   └── stitching_report.txt       # Processing summary
└── spatial/                       # Spatial analysis
    ├── spatial_analysis_results.h5ad
    ├── spatial_summary.csv
    └── spatial_analysis_plots.png
EOF

echo ""
echo "4. QUICK START FOR YOUR 60GB IMAGE"
echo "=================================="

echo "For your specific use case, I recommend:"
echo ""
echo "# Create a markers.csv file first:"
cat << 'EOF' > example_markers.csv
marker_name,channel_number,cycle_number
DAPI,1,1
CD3,2,1
CD8,3,1
PanCK,4,1
CD68,5,1
# Add all your channels...
EOF

echo "Created example_markers.csv ✓"
echo ""
echo "# Then run the pipeline:"
echo ""

cat << 'EOF'
# Option 1: Nextflow (handles everything automatically)
nextflow run mcmicro-tiled.nf \
    --input_image your_60GB_image.tiff \
    --markers_csv example_markers.csv \
    --outdir mcmicro_results \
    --sample_name large_IF_analysis \
    --tile_size 2048 \
    --overlap 256 \
    -profile docker

# Option 2: Python script
python mcmicro_tiled_processor.py \
    --input your_60GB_image.tiff \
    --output mcmicro_results \
    --markers example_markers.csv \
    --sample-name large_IF_analysis \
    --tile-size 2048 \
    --overlap 256 \
    --workers 6
EOF

echo ""
echo "5. DOWNSTREAM ANALYSIS WITH SCIMAP"
echo "=================================="

echo "After processing, you can continue with SCIMAP analysis:"
echo ""

cat << 'EOF'
# Python script for continued analysis
import scimap as sm
import pandas as pd

# Load the combined results
adata = sm.pp.mcmicro_to_scimap('mcmicro_results/final/combined_quantification.csv')

# Standard SCIMAP workflow
adata = sm.pp.rescale(adata, gate=99)

# Define phenotypes based on your markers
phenotype = {
    'T cells': ['CD3+'],
    'CD8 T cells': ['CD3+', 'CD8+'],  
    'Tumor cells': ['PanCK+'],
    'Macrophages': ['CD68+'],
    # Add your phenotypes based on your markers
}

adata = sm.tl.phenotype_cells(adata, phenotype=phenotype)

# Spatial analysis - this is where RCNs come in
adata = sm.tl.spatial_distance(adata, method='radius', radius=30)
adata = sm.tl.spatial_interaction(adata, method='radius', radius=30) 
adata = sm.tl.spatial_cluster(adata, method='kmeans', k=10)

# Neighborhood analysis for RCNs
adata = sm.tl.spatial_pscore(adata, label='spatial_cluster')

# Distance analysis between phenotypes
distances = sm.tl.spatial_distance(adata, 
                                   x_coordinate='X_centroid', 
                                   y_coordinate='Y_centroid',
                                   phenotype='phenotype',
                                   subset=['CD8 T cells', 'Tumor cells'])

# Visualizations
sm.pl.spatial_scatterplot(adata, color=['phenotype'])
sm.pl.spatial_interaction_network(adata)
sm.pl.spatial_pscore(adata)
EOF

echo ""
echo "6. TROUBLESHOOTING"
echo "=================="

cat << 'EOF'
Common issues and solutions:

1. Memory errors during tiling:
   - Reduce tile_size (try 1024 instead of 2048)
   - Use higher pyramid_level (1 or 2 for lower resolution)

2. Segmentation fails on some tiles:
   - Check if DAPI/nuclei channel is first channel
   - Adjust segmentation parameters in the script

3. Stitching artifacts:
   - Increase overlap (try 512 instead of 256)
   - Check tile_info.json for proper tile coverage

4. Slow processing:
   - Increase number of workers
   - Use smaller tiles for parallel processing
   - Consider using Nextflow with cluster execution

5. SCIMAP analysis issues:
   - Ensure combined_quantification.csv has proper column names
   - Check that X_centroid and Y_centroid columns exist
   - Verify phenotype definitions match your markers
EOF

echo ""
echo "Setup complete! You can now run the pipeline with your 60GB image."
echo "Start with the Nextflow approach for the most automated experience."

# Make the Python script executable
chmod +x mcmicro_tiled_processor.py

echo ""
echo "Files ready:"
echo "✓ mcmicro-tiled.nf (Nextflow workflow)"
echo "✓ nextflow.config (Nextflow configuration)"
echo "✓ mcmicro_tiled_processor.py (Python script)"
echo "✓ example_markers.csv (Template markers file)"