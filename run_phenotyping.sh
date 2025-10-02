#!/bin/bash

# Configurable Phenotyping Workflow Runner - Auto-processes all samples
set -euo pipefail

CONFIG_FILE=${1:-"phenotyping_config.yaml"}

echo "🔬 CONFIGURABLE PHENOTYPING WORKFLOW"
echo "====================================="
echo "Config: ${CONFIG_FILE}"
echo ""

# Check if config exists
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "❌ Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Find all samples in results directory
echo "Scanning for completed samples in results/..."
samples=()
for sample_dir in results/*/; do
    if [[ -d "$sample_dir" ]]; then
        sample_name=$(basename "$sample_dir")
        csv_file="${sample_dir}final/combined_quantification.csv"
        if [[ -f "$csv_file" ]]; then
            samples+=("$sample_name")
        fi
    fi
done

if [[ ${#samples[@]} -eq 0 ]]; then
    echo "❌ No completed samples found in results/"
    echo "Expected: results/SAMPLE_NAME/final/combined_quantification.csv"
    exit 1
fi

echo "Found ${#samples[@]} sample(s): ${samples[*]}"
echo ""

# Process each sample
for SAMPLE_NAME in "${samples[@]}"; do
    echo "Processing sample: ${SAMPLE_NAME}"
    
    # Step 1: Setup workflow
    echo "Setting up workflow..."
    python3 phenotyping_workflow.py --sample "${SAMPLE_NAME}" --config "${CONFIG_FILE}"

    
    OUTPUT_DIR="phenotyping_analysis/${SAMPLE_NAME}"
    
    echo ""
    echo "Running SCIMAP phenotyping for ${SAMPLE_NAME}..."
    
    # Run SCIMAP using the generated script
    docker run --rm \
        -v "$(pwd)/${OUTPUT_DIR}:/data" \
        -v "$(pwd)/${CONFIG_FILE}:/config.yaml" \
        labsyspharm/scimap:latest \
        python /data/scimap_results/run_phenotyping.py
    
    echo "✅ ${SAMPLE_NAME} complete!"
    echo ""
done

echo "🎉 ALL SAMPLES PROCESSED!"
echo ""
echo "Results structure:"
echo "phenotyping_analysis/"
for SAMPLE_NAME in "${samples[@]}"; do
    echo "├── ${SAMPLE_NAME}/"
    echo "│   └── scimap_results/"
    echo "│       ├── phenotyped_cells.csv"
    echo "│       ├── phenotype_counts.csv"
    echo "│       └── phenotyped_data.h5ad"
done
echo ""
echo "To modify phenotypes, edit ${CONFIG_FILE} and re-run"