#!/bin/bash

# Run MCMICRO Tiled Processor with Cellpose Segmentation
# =====================================================

echo "MCMICRO TILED PROCESSOR"
echo "=========================="
echo ""
echo "Process large immunofluorescence images through tiling → CellPose → spatial analysis"
echo ""

# Check environment
if [[ -f "mcmicro_env/bin/activate" ]]; then
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        echo "Activating Python virtual environment..."
        source mcmicro_env/bin/activate
    fi
    echo "✓ Python virtual environment active"
else
    echo "⚠ Virtual environment not found - some features may not work"
fi

echo "✓ Current directory: $(pwd)"
echo ""

# Check Docker setup
echo "Checking Docker setup..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found!"
    echo ""
    echo "Please install Docker:"
    echo "1. Install Docker Desktop from: https://docker.com/products/docker-desktop"
    echo "2. Enable WSL2 integration in Docker Desktop settings"
    echo "3. Restart your WSL2 terminal"
    echo ""
    exit 1
fi

# Test Docker
if ! docker ps &> /dev/null; then
    echo "❌ Docker is not running or not accessible!"
    echo ""
    echo "Please make sure Docker Desktop is running and WSL2 integration is enabled."
    echo ""
    echo "To enable WSL2 integration:"
    echo "1. Open Docker Desktop"
    echo "2. Go to Settings > Resources > WSL Integration"
    echo "3. Enable integration with your WSL2 distribution"
    echo "4. Apply & Restart"
    echo ""
    exit 1
fi

echo "✓ Docker is working"

# Test GPU access
echo "Testing GPU access..."
if docker run --rm --gpus all nvidia/cuda:11.2-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "✓ GPU access confirmed"
else
    echo "⚠ GPU access test failed - Cellpose will use CPU (much slower)"
    echo "  Make sure NVIDIA Docker runtime is installed for GPU acceleration"
fi

# Check if Docker can pull images
echo "Testing Docker image access..."
if docker pull hello-world &> /dev/null; then
    echo "✓ Docker can pull images"
    docker rmi hello-world &> /dev/null
else
    echo "⚠ Docker pull test failed - continuing anyway"
fi

echo ""

# Pre-pull containers to avoid delays
echo "Pre-pulling containers (this may take a few minutes)..."
echo "This ensures the pipeline runs smoothly without delays"

containers=(
    "mcmicro-tiles:latest"
    "biocontainers/cellpose:3.0.1_cv1"
    "labsyspharm/quantification:latest"
    "labsyspharm/scimap:0.17.7"
    "labsyspharm/basic-illumination:1.0.0"
    "vanvalenlab/deepcell-applications:0.4.0"
)

for container in "${containers[@]}"; do
    echo "  Pulling $container..."
    if docker pull "$container" &> /dev/null; then
        echo "    ✓ $container ready"
    else
        echo "    ⚠ Failed to pull $container (will try during pipeline)"
    fi
done

echo ""
echo "✓ Container preparation complete"
echo ""

# Pre-download Cellpose models
# Pre-download Cellpose models
echo "Pre-downloading Cellpose models..."
echo "This is a one-time download (~100MB)"

CELLPOSE_MODELS_DIR="$(pwd)/cellpose_models"
mkdir -p "$CELLPOSE_MODELS_DIR"

# Check if models already exist
if [[ -f "$CELLPOSE_MODELS_DIR/nucleitorch_0" ]] && [[ -f "$CELLPOSE_MODELS_DIR/cyto2torch_0" ]]; then
    echo "✓ Cellpose models already downloaded"
else
    echo "  Downloading nuclei and cyto2 models..."
    docker run --rm \
        -v "$CELLPOSE_MODELS_DIR:/tmp/cellpose_models" \
        biocontainers/cellpose:3.0.1_cv1 \
        bash -c "
export CELLPOSE_LOCAL_MODELS_PATH=/tmp/cellpose_models
python3 << 'EOF'
import os
os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = '/tmp/cellpose_models'
from cellpose import models
print('Downloading nuclei model...')
models.Cellpose(model_type='nuclei', gpu=False)
print('Downloading cyto2 model...')
models.Cellpose(model_type='cyto2', gpu=False)
print('Models downloaded successfully')
EOF
"
    
    if [[ $? -eq 0 ]] && [[ -f "$CELLPOSE_MODELS_DIR/nucleitorch_0" ]]; then
        echo "    ✓ Models downloaded to $CELLPOSE_MODELS_DIR"
    else
        echo "    ⚠ Model download might have failed - pipeline will try to download at runtime"
    fi
fi

# List what we have
echo "  Models in directory:"
ls -lh "$CELLPOSE_MODELS_DIR/" 2>/dev/null | grep -E "torch|npy" || echo "    (none found)"

echo ""

set -euo pipefail

# Save the original working directory
SCRIPT_DIR=$(pwd)
echo "✓ Working directory: $SCRIPT_DIR"

# Detect input mode: per-channel directories or stacked TIFFs
echo "Detecting input mode..."
echo ""

shopt -s nullglob
sample_dirs=(rawdata/*/)
stacked_tiffs=(rawdata/*.ome.tif)
shopt -u nullglob

use_per_channel=false
samples_to_process=()

# Check for per-channel directories
if [ ${#sample_dirs[@]} -gt 0 ]; then
    for dir in "${sample_dirs[@]}"; do
        if ls "$dir"/*.ome.tif &>/dev/null; then
            sample_name=$(basename "$dir")
            samples_to_process+=("$sample_name:$dir")
            echo "  Found per-channel directory: $sample_name/"
            use_per_channel=true
        fi
    done
fi

if [ "$use_per_channel" = true ]; then
    echo ""
    echo "✓ FAST MODE: Using per-channel TIFF directories (5-10x faster!)"
    echo "  Found ${#samples_to_process[@]} sample(s)"
    echo ""
elif [ ${#stacked_tiffs[@]} -gt 0 ]; then
    echo ""
    echo "✓ LEGACY MODE: Using stacked multi-channel TIFFs"
    echo "  Found ${#stacked_tiffs[@]} file(s)"
    echo ""

    # Build samples list for legacy mode
    for img in "${stacked_tiffs[@]}"; do
        sample_name=$(basename "$img" _aligned_stack.ome.tif)
        samples_to_process+=("$sample_name:$img")
    done
else
    echo "❌ No input data found in rawdata/"
    echo ""
    echo "Expected (FAST MODE - recommended):"
    echo "  rawdata/"
    echo "    ├── SampleA/"
    echo "    │   └── *.ome.tif (33 per-channel files)"
    echo "    └── SampleB/"
    echo "        └── *.ome.tif (33 per-channel files)"
    echo ""
    echo "Or (LEGACY MODE):"
    echo "  rawdata/"
    echo "    ├── SampleA_aligned_stack.ome.tif"
    echo "    └── SampleB_aligned_stack.ome.tif"
    echo ""
    exit 1
fi

# Show samples found
echo "Available samples:"
all_samples=("${samples_to_process[@]}")  # Store all detected samples
for sample_entry in "${all_samples[@]}"; do
    sample_name="${sample_entry%%:*}"
    final_csv="results/${sample_name}/final/combined_quantification.csv"

    if [[ -f "$final_csv" ]] && [[ -s "$final_csv" ]]; then
        echo "  $sample_name [COMPLETED ✓]"
    else
        echo "  $sample_name"
    fi
done

echo ""

# SAMPLE SELECTION: Allow user to choose which samples to process
if [ $# -gt 0 ]; then
    # Command-line arguments provided: filter to specified samples
    echo "Processing specified samples: $@"
    selected_samples=()
    for arg in "$@"; do
        for sample_entry in "${all_samples[@]}"; do
            sample_name="${sample_entry%%:*}"
            if [ "$sample_name" == "$arg" ]; then
                selected_samples+=("$sample_entry")
                break
            fi
        done
    done

    if [ ${#selected_samples[@]} -eq 0 ]; then
        echo "❌ None of the specified samples were found!"
        echo "Available samples: ${all_samples[@]%%:*}"
        exit 1
    fi

    samples_to_process=("${selected_samples[@]}")
else
    # No arguments: interactive selection
    echo "Sample Selection:"
    echo "  - Press ENTER to process ALL samples"
    echo "  - Or enter sample names separated by spaces (e.g., 'JL216_Final GUEST29')"
    echo ""
    read -p "Samples to process [ALL]: " user_input

    if [ -n "$user_input" ]; then
        # User specified samples
        selected_samples=()
        for sample_arg in $user_input; do
            for sample_entry in "${all_samples[@]}"; do
                sample_name="${sample_entry%%:*}"
                if [ "$sample_name" == "$sample_arg" ]; then
                    selected_samples+=("$sample_entry")
                    break
                fi
            done
        done

        if [ ${#selected_samples[@]} -eq 0 ]; then
            echo "❌ None of the specified samples were found!"
            exit 1
        fi

        samples_to_process=("${selected_samples[@]}")
        echo "✓ Will process ${#samples_to_process[@]} sample(s)"
    else
        # Process all samples
        echo "✓ Will process ALL ${#samples_to_process[@]} sample(s)"
    fi
fi
echo

# Ensure markers.csv exists
if [[ ! -f "markers.csv" ]]; then
    if [ "$use_per_channel" = true ]; then
        echo "⚠ WARNING: markers.csv is REQUIRED for per-channel mode!"
        echo ""
        echo "Please create markers.csv with your channel order."
        echo "Example format:"
        echo "  cycle,marker_name"
        echo "  1,R1.0.1_DAPI"
        echo "  1,R1.0.1_CY3"
        echo "  ..."
        echo ""
        exit 1
    else
        echo "Creating default markers.csv..."
        cat > markers.csv <<EOF
marker_name
DAPI
Tom
CD45
Channel_4
Channel_5
Channel_6
Channel_7
Channel_8
Channel_9
CD3
EOF
    fi
fi

# Ask about resume
read -p "Resume previous runs? (y/n): " resume_choice
if [[ $resume_choice =~ ^[Nn]$ ]]; then
    echo "Cleaning up previous runs..."
    rm -rf work .nextflow* results
fi

# Process each sample
for sample_entry in "${samples_to_process[@]}"; do
    cd "$SCRIPT_DIR"

    sample_name="${sample_entry%%:*}"
    sample_path="${sample_entry#*:}"
    outdir="results/${sample_name}"
    final_csv="$outdir/final/combined_quantification.csv"

    # Check if already completed
    if [[ -f "$final_csv" ]] && [[ -s "$final_csv" ]]; then
        echo ""
        echo "=== Skipping $sample_name (already completed) ==="
        echo "Final output exists: $final_csv"
        echo "================================================="
        continue
    fi

    mkdir -p "$outdir"

    echo ""
    echo "=== Processing $sample_name ==="

    # STEP 1: Create tiles (different for per-channel vs stacked)
    if [ "$use_per_channel" = true ]; then
        echo "Step 1: Creating tiles from per-channel TIFFs..."
        echo "Sample directory: $sample_path"

        tile_dir="$outdir/tiles"
        mkdir -p "$tile_dir"

        python3 scripts/tile_from_channels.py \
            --sample_dir "$sample_path" \
            --markers_csv markers.csv \
            --output_dir "$tile_dir" \
            --tile_size 8192 \
            --overlap 1024 \
            --dapi_channel 0 \
            --max_workers 3

        if [ $? -ne 0 ]; then
            echo "✗ FAILED: Tiling failed for $sample_name"
            continue
        fi

        echo "✓ Tiles created: $tile_dir"

        # Create a pseudo stacked TIFF path for Nextflow (won't be used, just for compatibility)
        pseudo_tiff="$tile_dir/pseudo_stack.ome.tif"
        touch "$pseudo_tiff"  # Create empty file
        input_image="$pseudo_tiff"
    else
        # Legacy mode - use stacked TIFF directly
        input_image="$sample_path"
        echo "Input: $input_image (stacked TIFF)"
    fi

    # STEP 2: Run Nextflow pipeline (segmentation + quantification)
    echo ""
    echo "Step 2: Running Cellpose segmentation and quantification..."

    nf_args=(
        run mcmicro-tiled.nf
        --input_image "$input_image"
        --markers_csv markers.csv
        --outdir "$outdir"
        --sample_name "$sample_name"
        --tile_size 8192
        --overlap 1024
        --pyramid_level 0
        --cellpose true
        --mcquant true
        --scimap true
        --dapi_channel 0
        --nuc_diameter 15
        --cyto_diameter 28
        --nuclei_batch_size 4
        --cyto_batch_size_tiles 3
        -with-report "$outdir/nextflow_report.html"
        -with-timeline "$outdir/nextflow_timeline.html"
        -with-dag "$outdir/nextflow_dag.html"
    )

    # Add skip_tiling flag if using per-channel mode (tiles already created)
    if [ "$use_per_channel" = true ]; then
        nf_args+=(--skip_tiling true)
        echo "  → Using pre-generated tiles (skip_tiling=true)"
    fi

    if [[ $resume_choice =~ ^[Yy]$ ]]; then
        nf_args+=(-resume)
    fi

    nextflow "${nf_args[@]}"

    pipeline_exit_code=$?

    echo ""
    echo "================================================="
    echo "Pipeline Complete for $sample_name!"
    echo "================================================="

    if [[ $pipeline_exit_code -eq 0 ]]; then
        echo "✓ SUCCESS!"
    else
        echo "✗ FAILED (exit code: $pipeline_exit_code)"
        echo "Check $outdir logs for details"
    fi

    echo ""
done

# Return to original directory
cd "$SCRIPT_DIR"

echo ""
echo "All selected runs complete."
echo ""
echo "Processing Complete!"
echo ""