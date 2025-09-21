#!/bin/bash

# Batch Loading Script for Spatial Analysis
# Creates directory structure and runs batch loading

set -euo pipefail

echo "🔬 Setting up Spatial Analysis Pipeline"
echo "======================================"

# Create directory structure
echo "Creating directory structure..."
mkdir -p spatial_analysis/{configs,scripts,outputs}
mkdir -p spatial_analysis/outputs/{integrated_data,phenotyping,spatial_analysis,plots}

# Copy scripts and configs
echo "Setting up files..."
cp scripts/batch_load_slides.py spatial_analysis/scripts/
cp configs/experiment_config.yaml spatial_analysis/configs/

# Build Docker image
echo "Building Docker image..."
docker build -f Dockerfile.spatial -t spatial-analysis:latest .

# Check if data exists
if [ ! -d "results" ]; then
    echo "❌ Error: results directory not found"
    echo "Make sure you've run the mcmicro-tiled.nf pipeline first"
    exit 1
fi

# Count available slides
slide_count=$(find results -name "combined_quantification.csv" | wc -l)
echo "Found $slide_count slides to process"

if [ $slide_count -eq 0 ]; then
    echo "❌ Error: No combined_quantification.csv files found"
    echo "Make sure the mcmicro-tiled.nf pipeline completed successfully"
    exit 1
fi

# Run batch loading
echo ""
echo "🚀 Running batch loading..."
echo "============================"

docker run --rm \
    -v "$(pwd)/results:/app/spatial_pipeline/results:ro" \
    -v "$(pwd)/spatial_analysis:/app/spatial_analysis" \
    spatial-analysis:latest \
    /app/spatial_analysis/scripts/batch_load_slides.py \
    --config /app/spatial_analysis/configs/experiment_config.yaml \
    --output /app/spatial_analysis/outputs

# Check results
if [ -f "spatial_analysis/outputs/integrated_data/"*"_integrated.h5ad" ]; then
    echo ""
    echo "✅ Batch loading successful!"
    echo "=========================="
    echo "Results saved to: spatial_analysis/outputs/integrated_data/"
    echo ""
    echo "Next steps:"
    echo "1. Review slide summary and QC report"
    echo "2. Update experiment_config.yaml with your actual slide metadata"
    echo "3. Run phenotyping pipeline"
    echo ""
    echo "Files created:"
    ls -la spatial_analysis/outputs/integrated_data/
else
    echo "❌ Batch loading failed - check logs above"
    exit 1
fi