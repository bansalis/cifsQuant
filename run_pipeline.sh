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

# Detect input mode
echo "Detecting input mode..."
echo ""

# Check for per-channel directories (FAST MODE)
shopt -s nullglob
sample_dirs=(rawdata/*/)
shopt -u nullglob

# Check for stacked TIFFs (LEGACY MODE)
shopt -s nullglob
stacked_tiffs=(rawdata/*.ome.tif)
shopt -u nullglob

use_fast_mode=false
has_sample_dirs=false

# Check if sample directories contain .ome.tif files
if [ ${#sample_dirs[@]} -gt 0 ]; then
    for dir in "${sample_dirs[@]}"; do
        if ls "$dir"/*.ome.tif &>/dev/null; then
            has_sample_dirs=true
            break
        fi
    done
fi

# Determine mode
if [ "$has_sample_dirs" = true ]; then
    echo "✓ FAST MODE: Found sample directories with per-channel TIFFs"
    echo "  This mode is 5-10x faster on network/WSL mounts!"
    echo ""

    # Count valid sample directories
    valid_samples=()
    for dir in "${sample_dirs[@]}"; do
        if ls "$dir"/*.ome.tif &>/dev/null 2>&1; then
            sample_name=$(basename "$dir")
            final_csv="results/${sample_name}/final/combined_quantification.csv"

            if [[ -f "$final_csv" ]] && [[ -s "$final_csv" ]]; then
                echo "  ${sample_name}/ [COMPLETED ✓]"
            else
                echo "  ${sample_name}/"
            fi
            valid_samples+=("$dir")
        fi
    done

    if [ ${#valid_samples[@]} -eq 0 ]; then
        echo "❌ No valid sample directories found in rawdata/"
        exit 1
    fi

    echo ""
    echo "Found ${#valid_samples[@]} sample(s) with per-channel TIFFs"
    echo ""

    use_fast_mode=true

elif [ ${#stacked_tiffs[@]} -gt 0 ]; then
    echo "✓ LEGACY MODE: Found stacked multi-channel TIFFs"
    echo "  Consider using per-channel TIFFs for 5-10x faster processing!"
    echo ""

    for i in "${!stacked_tiffs[@]}"; do
        sample_name=$(basename "${stacked_tiffs[$i]}" _aligned_stack.ome.tif)
        final_csv="results/${sample_name}/final/combined_quantification.csv"

        if [[ -f "$final_csv" ]] && [[ -s "$final_csv" ]]; then
            echo "  [$i] ${stacked_tiffs[$i]} [COMPLETED ✓]"
        else
            echo "  [$i] ${stacked_tiffs[$i]}"
        fi
    done
    echo ""

    read -p "Run on all files? (y/n): " all_choice
    if [[ $all_choice =~ ^[Yy]$ ]]; then
        selected=("${stacked_tiffs[@]}")
    else
        read -p "Enter space-separated indices of files to run: " -a idxs
        selected=()
        for idx in "${idxs[@]}"; do
            selected+=("${stacked_tiffs[$idx]}")
        done
    fi
else
    echo "❌ No input data found in rawdata/"
    echo ""
    echo "Expected structure for FAST MODE (recommended):"
    echo "  rawdata/"
    echo "    ├── SampleA/"
    echo "    │   ├── Sample_1.0.1_R000_DAPI.ome.tif"
    echo "    │   ├── Sample_1.0.1_R000_Cy3.ome.tif"
    echo "    │   └── ... (all channel files)"
    echo "    └── SampleB/"
    echo "        └── ... (all channel files)"
    echo ""
    echo "Or for LEGACY MODE:"
    echo "  rawdata/"
    echo "    ├── SampleA_aligned_stack.ome.tif"
    echo "    └── SampleB_aligned_stack.ome.tif"
    echo ""
    exit 1
fi

# Ensure markers.csv exists
if [[ ! -f "markers.csv" ]]; then
    echo "⚠ WARNING: markers.csv not found!"
    echo ""
    echo "For FAST MODE with per-channel TIFFs, markers.csv is REQUIRED to:"
    echo "  - Define the order of channels"
    echo "  - Match marker names to channel files"
    echo ""
    echo "Example markers.csv format:"
    echo "  cycle,marker_name"
    echo "  1,R1.0.1_DAPI"
    echo "  1,R1.0.1_CY3"
    echo "  2,R1.0.4_CY5_CD45"
    echo "  ..."
    echo ""
    read -p "Create a default markers.csv? (y/n): " create_markers
    if [[ $create_markers =~ ^[Yy]$ ]]; then
        echo "Creating default markers.csv..."
        cat > markers.csv <<EOF
cycle,marker_name
1,R1.0.1_DAPI
1,R1.0.1_CY3
2,R1.0.4_CY5_CD45
2,R1.0.4_CY7_AGFP
2,R1.0.4_DAPI
EOF
        echo "✓ Created default markers.csv - please edit it to match your actual channels!"
        echo ""
        read -p "Press Enter to continue after editing markers.csv..."
    else
        echo "Exiting. Please create markers.csv and try again."
        exit 1
    fi
fi

# Ask about resume
read -p "Resume previous runs? (y/n): " resume_choice
if [[ $resume_choice =~ ^[Nn]$ ]]; then
    echo "Cleaning up previous runs..."
    rm -rf work .nextflow* results
fi

# Run pipeline based on mode
if [ "$use_fast_mode" = true ]; then
    # FAST MODE: Process all samples in one pipeline run
    echo ""
    echo "=== Running FAST MODE Pipeline ==="
    echo "Processing ${#valid_samples[@]} sample(s) from rawdata/"
    echo "Markers: markers.csv"
    echo "Output: results/"
    echo "===================================="
    echo ""

    nf_args=(
        run mcmicro-tiled.nf
        --rawdata_dir rawdata
        --markers_csv markers.csv
        --outdir results
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
        -with-report results/nextflow_report.html
        -with-timeline results/nextflow_timeline.html
        -with-dag results/nextflow_dag.html
    )

    if [[ $resume_choice =~ ^[Yy]$ ]]; then
        nf_args+=(-resume)
    fi

    nextflow "${nf_args[@]}"

    pipeline_exit_code=$?

    echo ""
    echo "Pipeline Execution Complete!"
    echo "============================"

    if [[ $pipeline_exit_code -eq 0 ]]; then
        echo "✓ SUCCESS: Pipeline completed successfully for all samples!"
    else
        echo "✗ FAILED: Pipeline encountered errors (exit code: $pipeline_exit_code)"
        echo "Check results/ logs for details"
    fi

else
    # LEGACY MODE: Process each stacked TIFF individually
    for img in "${selected[@]}"; do
        # Reset to original directory at start of each iteration
        cd "$SCRIPT_DIR"

        sample_name=$(basename "$img" _aligned_stack.ome.tif)
        outdir="results/${sample_name}"
        final_csv="$outdir/final/combined_quantification.csv"

        # Check if sample already completed
        if [[ -f "$final_csv" ]] && [[ -s "$final_csv" ]]; then
            echo ""
            echo "=== Skipping $sample_name (already completed) ==="
            echo "Final output exists: $final_csv"
            echo "To rerun, delete: rm -rf $outdir"
            echo "================================================="
            continue
        fi

        mkdir -p "$outdir"

        echo ""
        echo "=== Running LEGACY MODE pipeline for $sample_name ==="
        echo "Input: $img"
        echo "Output: $outdir"
        echo "======================================================"

        nf_args=(
            run mcmicro-tiled.nf
            --input_image "$img"
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

        if [[ $resume_choice =~ ^[Yy]$ ]]; then
            nf_args+=(-resume)
        fi

        nextflow "${nf_args[@]}"

        pipeline_exit_code=$?

        echo ""
        echo "Pipeline Execution Complete for $sample_name!"
        echo "=============================================="

        if [[ $pipeline_exit_code -eq 0 ]]; then
            echo "✓ SUCCESS: Cellpose pipeline completed successfully for $sample_name!"
        else
            echo "✗ FAILED: Cellpose pipeline encountered errors for $sample_name (exit code: $pipeline_exit_code)"
            echo "Check $outdir logs for details"
        fi

        echo ""
    done
fi

# Return to original directory
cd "$SCRIPT_DIR"

echo ""
echo "All selected runs complete."
echo ""
echo "Processing Complete!"
echo ""