#!/bin/bash

# Run Phenotyping Pipeline

set -euo pipefail

echo "🧬 Running Cell Phenotyping Pipeline"
echo "==================================="

# Copy phenotyping scripts and configs
echo "Setting up phenotyping files..."
cp scripts/phenotype_cells.py spatial_analysis/scripts/
cp configs/phenotyping_config.yaml spatial_analysis/configs/

# Check if integrated data exists
INTEGRATED_FILE=$(find spatial_analysis/outputs/integrated_data -name "*_integrated.h5ad" | head -1)

if [ -z "$INTEGRATED_FILE" ]; then
    echo "❌ Error: No integrated h5ad file found"
    echo "Run batch loading first: ./run_batch_loading.sh"
    exit 1
fi

echo "Found integrated data: $INTEGRATED_FILE"

# Run phenotyping
echo ""
echo "🚀 Running phenotyping..."
echo "========================"

docker run --rm \
    -v "$(pwd)/spatial_analysis:/app/spatial_analysis" \
    spatial-analysis:latest \
    /app/spatial_analysis/scripts/phenotype_cells.py \
    --config /app/spatial_analysis/configs/experiment_config.yaml \
    --phenotype-config /app/spatial_analysis/configs/phenotyping_config.yaml \
    --input "/app/spatial_analysis/outputs/integrated_data/$(basename "$INTEGRATED_FILE")" \
    --output /app/spatial_analysis/outputs

# Check results
if [ -f "spatial_analysis/outputs/phenotyping/"*"_phenotyped.h5ad" ]; then
    echo ""
    echo "✅ Phenotyping successful!"
    echo "========================="
    echo "Results saved to: spatial_analysis/outputs/phenotyping/"
    echo ""
    echo "Files created:"
    ls -la spatial_analysis/outputs/phenotyping/
    echo ""
    echo "Next step: Run spatial neighborhood analysis"
else
    echo "❌ Phenotyping failed - check logs above"
    exit 1
fi