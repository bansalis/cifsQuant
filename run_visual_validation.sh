#!/bin/bash

# Run Visual Threshold Validation

set -euo pipefail

echo "👁️ Running Visual Threshold Validation"
echo "======================================"

# Copy validation script
echo "Setting up validation script..."
cp scripts/visual_threshold_validation.py spatial_analysis/scripts/

# Find phenotyped data and thresholds
PHENOTYPED_FILE=$(find spatial_analysis/outputs/phenotyping -name "*_phenotyped.h5ad" | head -1)
THRESHOLDS_FILE=$(find spatial_analysis/outputs/phenotyping -name "*_thresholds.csv" | head -1)

if [ -z "$PHENOTYPED_FILE" ] || [ -z "$THRESHOLDS_FILE" ]; then
    echo "❌ Error: Phenotyped data or thresholds not found"
    echo "Run phenotyping first: ./run_phenotyping.sh"
    exit 1
fi

echo "Found phenotyped data: $(basename "$PHENOTYPED_FILE")"
echo "Found thresholds: $(basename "$THRESHOLDS_FILE")"

# Run visual validation
echo ""
echo "🚀 Creating validation plots..."
echo "============================="

docker run --rm \
    -v "$(pwd)/spatial_analysis:/app/spatial_analysis" \
    spatial-analysis:latest \
    /app/spatial_analysis/scripts/visual_threshold_validation.py \
    --config /app/spatial_analysis/configs/experiment_config.yaml \
    --phenotype-config /app/spatial_analysis/configs/phenotyping_config.yaml \
    --phenotyped-data "/app/spatial_analysis/outputs/phenotyping/$(basename "$PHENOTYPED_FILE")" \
    --thresholds "/app/spatial_analysis/outputs/phenotyping/$(basename "$THRESHOLDS_FILE")" \
    --output /app/spatial_analysis/outputs

# Check results
if [ -d "spatial_analysis/outputs/phenotype_validation" ]; then
    echo ""
    echo "✅ Visual validation complete!"
    echo "============================="
    echo "Plots saved to: spatial_analysis/outputs/phenotype_validation/"
    echo ""
    echo "Files created:"
    ls -la spatial_analysis/outputs/phenotype_validation/
    echo ""
    echo "📊 Review the validation plots to verify thresholds look correct"
    echo "📋 Check validation_summary.csv for phenotype statistics"
else
    echo "❌ Visual validation failed - check logs above"
    exit 1
fi