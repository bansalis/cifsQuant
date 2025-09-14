#!/bin/bash

# Run MCMICRO Tiled Processor with Cellpose Segmentation
# =====================================================

echo "🔬 MCMICRO TILED PROCESSOR"
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

set -euo pipefail
shopt -s nullglob
images=(rawdata/*.ome.tif)
shopt -u nullglob

if [ ${#images[@]} -eq 0 ]; then
    echo "❌ No .ome.tif files found in rawdata/"
    exit 1
fi

echo "Found ${#images[@]} image(s) in rawdata/:"
for i in "${!images[@]}"; do
    echo "  [$i] ${images[$i]}"
done
echo

read -p "Run on all files? (y/n): " all_choice
if [[ $all_choice =~ ^[Yy]$ ]]; then
    selected=("${images[@]}")
else
    read -p "Enter space-separated indices of files to run: " -a idxs
    selected=()
    for idx in "${idxs[@]}"; do
        selected+=("${images[$idx]}")
    done
fi

# Ensure markers.csv exists
if [[ ! -f "markers.csv" ]]; then
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

# Ask about resume
read -p "Resume previous runs? (y/n): " resume_choice
if [[ $resume_choice =~ ^[Nn]$ ]]; then
    echo "Cleaning up previous runs..."
    rm -rf work .nextflow* results
fi

for img in "${selected[@]}"; do
    sample_name=$(basename "$img" _aligned_stack.ome.tif)
    outdir="results/${sample_name}"
    mkdir -p "$outdir"

    echo ""
    echo "=== Running pipeline for $sample_name ==="
    echo "Input: $img"
    echo "Output: $outdir"
    echo "========================================="

    nf_args=(
        run mcmicro-tiled.nf
        --input_image "$img"
        --markers_csv markers.csv
        --outdir "$outdir"
        --sample_name "$sample_name"
        --tile_size 2048
        --overlap 256
        --pyramid_level 1
        --cellpose true
        --mcquant true
        --scimap true
        --dapi_channel 8
        --nuc_diameter 15
        --cyto_diameter 28
        --nuclei_batch_size 6
        --cyto_batch_size_tiles 4
        -with-report "$outdir/nextflow_report.html"
        -with-timeline "$outdir/nextflow_timeline.html"
        -with-dag "$outdir/nextflow_dag.html"
    )

    if [[ $resume_choice =~ ^[Yy]$ ]]; then
        nf_args+=(-resume)
    fi

    nextflow "${nf_args[@]}"
done

echo ""
echo "All selected runs complete."

        pipeline_exit_code=$?

        echo ""
        echo "Pipeline Execution Complete!"
        echo "============================"

        if [[ $pipeline_exit_code -eq 0 ]]; then
            echo "✅ SUCCESS: Cellpose pipeline completed successfully!"
            echo ""
            echo "Results are available in:"
            echo "- results/final/full_segmentation_mask.tif"
            echo "- results/final/combined_quantification.csv"
            echo "- results/spatial/ (spatial analysis)"
            echo ""
            echo "Intermediate outputs:"
            echo "- results/nuclei_masks/ (nuclei segmentation)"
            echo "- results/cell_masks/ (cytoplasm segmentation)"
            echo "- results/quantification/ (per-tile measurements)"
            echo ""
            echo "Pipeline reports:"
            echo "- results/nextflow_report.html"
            echo "- results/nextflow_timeline.html"
            echo "- results/nextflow_dag.html"
            
            # Show summary if results exist
            if [[ -f "results/final/combined_quantification.csv" ]]; then
                echo ""
                echo "Quick Results Summary:"
                echo "====================="
                python3 -c "
import pandas as pd
try:
    df = pd.read_csv('results/final/combined_quantification.csv')
    print(f'Total cells detected: {len(df):,}')
    
    if 'X_centroid' in df.columns and 'Y_centroid' in df.columns:
        extent_x = df['X_centroid'].max() - df['X_centroid'].min()
        extent_y = df['Y_centroid'].max() - df['Y_centroid'].min()
        print(f'Spatial extent: {extent_x:.0f} x {extent_y:.0f} pixels')
    
    # Check for area or size columns
    area_cols = [col for col in df.columns if 'area' in col.lower()]
    if area_cols:
        area_col = area_cols[0]
        print(f'Cell area range: {df[area_col].min():.0f} - {df[area_col].max():.0f} pixels²')
    
    print('')
    print('Available columns:')
    for i, col in enumerate(df.columns):
        if i < 10:  # Show first 10 columns
            print(f'  - {col}')
        elif i == 10:
            print(f'  ... and {len(df.columns)-10} more columns')
            break
            
    print('')
    print('✅ Cellpose segmentation data is ready for analysis!')
    
except Exception as e:
    print(f'Could not read results: {e}')
" 2>/dev/null || echo "Could not generate summary (missing pandas)"
            fi
            
            echo ""
            echo "Next steps:"
            echo "==========="
            echo "1. Examine results in results/ directory"
            echo "2. Use combined_quantification.csv for spatial analysis"
            echo "3. Check the HTML reports for pipeline details"
            echo "4. Compare nuclei vs. cytoplasm masks for quality assessment"
            echo ""
            echo "For spatial analysis with SCIMAP:"
            echo "import scimap as sm"
            echo "adata = sm.pp.mcmicro_to_scimap('results/final/combined_quantification.csv')"
            
        else
            echo "❌ FAILED: Cellpose pipeline encountered errors (exit code: $pipeline_exit_code)"
            echo ""
            echo "Troubleshooting suggestions:"
            echo "1. Check that Docker is running properly"
            echo "2. Ensure GPU drivers are installed for acceleration"
            echo "3. Check .nextflow.log for detailed error messages"
            echo "4. Try reducing tile_size if memory issues occur"
            echo "5. Verify DAPI channel number (--dapi_channel 8)"
            echo ""
            echo "Common fixes:"
            echo "- GPU memory issues: Reduce --cyto_batch_size to 4"
            echo "- Wrong channel: Check your image channels and update --dapi_channel"
            echo "- Size issues: Try --nuc_diameter 10 --cyto_diameter 20 for smaller cells"
            
        fi

        echo ""
        echo "🎉 MCMICRO Cellpose Processing Complete!"
        ;;
    [Yy])
        echo "Resuming previous run..."
        # Run the Cellpose Nextflow pipeline with resume
        nextflow run mcmicro-tiled.nf \
            --input_image rawdata/GUEST29_aligned_stack.ome.tif \
            --markers_csv markers.csv \
            --outdir results \
            --sample_name GUEST29 \
            --tile_size 2048 \
            --overlap 256 \
            --pyramid_level 1 \
            --cellpose true \
            --mcquant true \
            --scimap true \
            --dapi_channel 8 \
            --nuc_diameter 15 \
            --cyto_diameter 28 \
            --nuclei_batch_size 6 \
            --cyto_batch_size_tiles 4 \
            -resume \
            -with-report results/nextflow_report.html \
            -with-timeline results/nextflow_timeline.html \
            -with-dag results/nextflow_dag.html

        pipeline_exit_code=$?

        echo ""
        echo "Pipeline Execution Complete!"
        echo "============================"

        if [[ $pipeline_exit_code -eq 0 ]]; then
            echo "✅ SUCCESS: Cellpose pipeline completed successfully!"
            echo ""
            echo "Results are available in:"
            echo "- results/final/full_segmentation_mask.tif"
            echo "- results/final/combined_quantification.csv"
            echo "- results/spatial/ (spatial analysis)"
            echo ""
            echo "Intermediate outputs:"
            echo "- results/nuclei_masks/ (nuclei segmentation)"
            echo "- results/cell_masks/ (cytoplasm segmentation)"
            echo "- results/quantification/ (per-tile measurements)"
            echo ""
            echo "Pipeline reports:"
            echo "- results/nextflow_report.html"
            echo "- results/nextflow_timeline.html"
            echo "- results/nextflow_dag.html"
            
            # Show summary if results exist
            if [[ -f "results/final/combined_quantification.csv" ]]; then
                echo ""
                echo "Quick Results Summary:"
                echo "====================="
                python3 -c "
import pandas as pd
try:
    df = pd.read_csv('results/final/combined_quantification.csv')
    print(f'Total cells detected: {len(df):,}')
    
    if 'X_centroid' in df.columns and 'Y_centroid' in df.columns:
        extent_x = df['X_centroid'].max() - df['X_centroid'].min()
        extent_y = df['Y_centroid'].max() - df['Y_centroid'].min()
        print(f'Spatial extent: {extent_x:.0f} x {extent_y:.0f} pixels')
    
    # Check for area or size columns
    area_cols = [col for col in df.columns if 'area' in col.lower()]
    if area_cols:
        area_col = area_cols[0]
        print(f'Cell area range: {df[area_col].min():.0f} - {df[area_col].max():.0f} pixels²')
    
    print('')
    print('Available columns:')
    for i, col in enumerate(df.columns):
        if i < 10:  # Show first 10 columns
            print(f'  - {col}')
        elif i == 10:
            print(f'  ... and {len(df.columns)-10} more columns')
            break
            
    print('')
    print('✅ Cellpose segmentation data is ready for analysis!')
    
except Exception as e:
    print(f'Could not read results: {e}')
" 2>/dev/null || echo "Could not generate summary (missing pandas)"
            fi
            
            echo ""
            echo "Next steps:"
            echo "==========="
            echo "1. Examine results in results/ directory"
            echo "2. Use combined_quantification.csv for spatial analysis"
            echo "3. Check the HTML reports for pipeline details"
            echo "4. Compare nuclei vs. cytoplasm masks for quality assessment"
            echo ""
            echo "For spatial analysis with SCIMAP:"
            echo "import scimap as sm"
            echo "adata = sm.pp.mcmicro_to_scimap('results/final/combined_quantification.csv')"
            
        else
            echo "❌ FAILED: Cellpose pipeline encountered errors (exit code: $pipeline_exit_code)"
            echo ""
            echo "Troubleshooting suggestions:"
            echo "1. Check that Docker is running properly"
            echo "2. Ensure GPU drivers are installed for acceleration"
            echo "3. Check .nextflow.log for detailed error messages"
            echo "4. Try reducing tile_size if memory issues occur"
            echo "5. Verify DAPI channel number (--dapi_channel 8)"
            echo ""
            echo "Common fixes:"
            echo "- GPU memory issues: Reduce --cyto_batch_size to 4"
            echo "- Wrong channel: Check your image channels and update --dapi_channel"
            echo "- Size issues: Try --nuc_diameter 10 --cyto_diameter 20 for smaller cells"
            
        fi

        echo ""
        echo "🎉 MCMICRO Cellpose Processing Complete!"
        ;;
esac

echo ""
echo "🎯 Processing Complete!"
echo ""