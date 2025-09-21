#!/bin/bash

# Run Spatial Neighborhood Analysis

set -euo pipefail

echo "🔍 Running Spatial Neighborhood Analysis"
echo "======================================="

# Copy spatial analysis script
echo "Setting up spatial analysis script..."
cp scripts/spatial_neighborhood_analysis.py spatial_analysis/scripts/

# Find phenotyped data
PHENOTYPED_FILE=$(find spatial_analysis/outputs/phenotyping -name "*_phenotyped.h5ad" | head -1)

if [ -z "$PHENOTYPED_FILE" ]; then
    echo "❌ Error: No phenotyped h5ad file found"
    echo "Run phenotyping first: ./run_phenotyping.sh"
    exit 1
fi

echo "Found phenotyped data: $(basename "$PHENOTYPED_FILE")"

# Run spatial analysis
echo ""
echo "🚀 Running spatial analysis..."
echo "============================="

docker run --rm \
    -v "$(pwd)/spatial_analysis:/app/spatial_analysis" \
    spatial-analysis:latest \
    /app/spatial_analysis/scripts/spatial_neighborhood_analysis.py \
    --config /app/spatial_analysis/configs/experiment_config.yaml \
    --phenotype-config /app/spatial_analysis/configs/phenotyping_config.yaml \
    --input "/app/spatial_analysis/outputs/phenotyping/$(basename "$PHENOTYPED_FILE")" \
    --output /app/spatial_analysis/outputs

# Check results
if [ -d "spatial_analysis/outputs/spatial_analysis" ]; then
    echo ""
    echo "✅ Spatial analysis complete!"
    echo "============================"
    echo "Results saved to: spatial_analysis/outputs/spatial_analysis/"
    echo ""
    echo "Generated outputs:"
    find spatial_analysis/outputs/spatial_analysis -name "*.png" -o -name "*.csv" | head -10
    echo ""
    echo "📊 Key research questions analyzed:"
    echo "1. pERK+ vs pERK- tumor cell neighborhoods"
    echo "2. NINJA+ vs NINJA- tumor cell neighborhoods" 
    echo "3. Distance relationships with immune cells"
    echo ""
    echo "Next step: Run comparative analysis across treatment groups"
else
    echo "❌ Spatial analysis failed - check logs above"
    exit 1
fi