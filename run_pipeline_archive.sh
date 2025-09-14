#!/bin/bash

# MCMICRO Tiled Processor - Complete Solution
# ===========================================

echo "🔬 MCMICRO TILED PROCESSOR"
echo "=========================="
echo ""
echo "Process large immunofluorescence images through tiling → MCMICRO → spatial analysis"
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

# Check input file
if [[ -f "rawdata/GUEST29_aligned_stack.ome.tif" ]]; then
    file_size=$(du -h "rawdata/GUEST29_aligned_stack.ome.tif" | cut -f1)
    echo "✓ Input file: rawdata/GUEST29_aligned_stack.ome.tif (${file_size})"
else
    echo "❌ Input file not found: rawdata/GUEST29_aligned_stack.ome.tif"
    echo "   Please place your large image file at this location"
    echo ""
    exit 1
fi

# Check markers file
if [[ ! -f "markers.csv" ]]; then
    echo "Creating basic markers.csv..."
    cat > markers.csv << EOF
marker_name,channel_number,cycle_number
DAPI,1,1
CD3,2,1
CD8,3,1
PanCK,4,1
CD68,5,1
Ki67,6,1
EOF
    echo "✓ Created markers.csv (edit to match your channels)"
else
    echo "✓ markers.csv found"
fi

echo ""

# Check Docker availability
docker_available=false
if command -v docker &> /dev/null && docker ps &> /dev/null 2>&1; then
    docker_available=true
    echo "✅ Docker is available and running"
else
    echo "⚠ Docker not available or not running"
fi

echo ""
echo "Available Processing Options:"
echo "============================="
echo ""

echo "🐍 Option 1: Python Script (RECOMMENDED)"
echo "   ✅ Most reliable - works without Docker"
echo "   ✅ Uses your virtual environment with all packages"
echo "   ✅ Handles large images well"
echo "   ✅ Full spatial analysis support"
echo ""

if [[ "$docker_available" == "true" ]]; then
    echo "🐳 Option 2: Nextflow with Official MCMICRO Containers"
    echo "   ✅ Uses official MCMICRO containers (UnMICST, S3Segmenter, MCQuant)"
    echo "   ✅ Same results as official MCMICRO"
    echo "   ✅ Fully containerized and reproducible"
    echo "   ⚠ Requires Docker setup"
    echo ""
else
    echo "🐳 Option 2: Nextflow with Official MCMICRO Containers"
    echo "   ❌ Not available (Docker not running)"
    echo "   💡 To enable: Install Docker Desktop and enable WSL2 integration"
    echo ""
fi

echo "What would you like to do?"
echo ""
echo "1) 🐍 Run Python script (recommended)"
echo "2) 🐳 Run Nextflow with MCMICRO containers (if Docker available)"
echo "3) 🧪 Run small test first"
echo "4) 📋 Show all commands"
echo "5) ❌ Exit"
echo ""

read -p "Choose option (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🐍 Running Python Script"
        echo "========================"
        echo ""
        
        if [[ ! -f "mcmicro_tiled_processor.py" ]]; then
            echo "❌ mcmicro_tiled_processor.py not found"
            echo "Please make sure you have all the required files"
            exit 1
        fi
        
        echo "Processing your 60GB image with tiling approach..."
        echo "Command: python mcmicro_tiled_processor.py --input rawdata/GUEST29_aligned_stack.ome.tif --output results --sample-name GUEST29 --tile-size 2048 --overlap 256 --pyramid-level 1 --workers 4"
        echo ""
        
        python mcmicro_tiled_processor.py \
            --input rawdata/GUEST29_aligned_stack.ome.tif \
            --output results \
            --sample-name GUEST29 \
            --tile-size 2048 \
            --overlap 256 \
            --pyramid-level 1 \
            --workers 4
        ;;
        
    2)
        if [[ "$docker_available" != "true" ]]; then
            echo ""
            echo "❌ Docker not available"
            echo ""
            echo "To use this option, please:"
            echo "1. Install Docker Desktop from https://docker.com"
            echo "2. Enable WSL2 integration in Docker settings"
            echo "3. Restart your terminal"
            echo ""
            exit 1
        fi
        
        echo ""
        echo "🐳 Running Nextflow with Official MCMICRO Containers"
        echo "===================================================="
        echo ""
        
        if [[ ! -f "mcmicro-tiled.nf" ]]; then
            echo "❌ mcmicro-tiled.nf not found"
            echo "Please make sure you have all the required files"
            exit 1
        fi
        
        chmod +x run_mcmicro_containers.sh
        bash run_mcmicro_containers.sh
        ;;
        
    3)
        echo ""
        echo "🧪 Running Small Test"
        echo "===================="
        echo ""
        echo "Testing with smaller tiles and lower resolution..."
        
        python mcmicro_tiled_processor.py \
            --input rawdata/GUEST29_aligned_stack.ome.tif \
            --output results_test \
            --sample-name GUEST29_test \
            --tile-size 1024 \
            --overlap 128 \
            --pyramid-level 1 \
            --workers 2
        ;;
        
    4)
        echo ""
        echo "📋 All Available Commands"
        echo "========================="
        echo ""
        echo "Python Script Commands:"
        echo "----------------------"
        echo ""
        echo "# Full processing:"
        echo "python mcmicro_tiled_processor.py \\"
        echo "    --input rawdata/GUEST29_aligned_stack.ome.tif \\"
        echo "    --output results \\"
        echo "    --sample-name GUEST29 \\"
        echo "    --tile-size 2048 \\"
        echo "    --overlap 256 \\"
        echo "    --pyramid-level 1 \\"
        echo "    --workers 4"
        echo ""
        echo "# Small test:"
        echo "python mcmicro_tiled_processor.py \\"
        echo "    --input rawdata/GUEST29_aligned_stack.ome.tif \\"
        echo "    --output results_test \\"
        echo "    --sample-name GUEST29_test \\"
        echo "    --tile-size 1024 \\"
        echo "    --overlap 128 \\"
        echo "    --pyramid-level 1 \\"
        echo "    --workers 2"
        echo ""
        
        if [[ "$docker_available" == "true" ]]; then
            echo "Nextflow Commands (with Docker):"
            echo "--------------------------------"
            echo ""
            echo "# Using official MCMICRO containers:"
            echo "nextflow run mcmicro-tiled.nf \\"
            echo "    --input_image rawdata/GUEST29_aligned_stack.ome.tif \\"
            echo "    --markers_csv markers.csv \\"
            echo "    --outdir results \\"
            echo "    --sample_name GUEST29 \\"
            echo "    --tile_size 2048 \\"
            echo "    --overlap 256 \\"
            echo "    --pyramid_level 1 \\"
            echo "    -profile docker"
            echo ""
        fi
        
        echo "Expected Output Structure:"
        echo "-------------------------"
        cat << 'EOF'
results/
├── tiles/                          # Individual image tiles
├── processed_tiles/                # Segmented tiles
├── final/                          # Main results
│   ├── full_segmentation_mask.tiff # Complete segmentation
│   └── combined_quantification.csv # Single-cell data (for SCIMAP)
└── spatial/                       # Spatial analysis
    ├── spatial_analysis_results.h5ad
    └── spatial_analysis_plots.png
EOF
        echo ""
        echo "Downstream Analysis:"
        echo "-------------------"
        echo "import scimap as sm"
        echo "adata = sm.pp.mcmicro_to_scimap('results/final/combined_quantification.csv')"
        echo "# Continue with SCIMAP workflow for RCN analysis..."
        ;;
        
    5)
        echo "Exiting..."
        exit 0
        ;;
        
    *)
        echo "Invalid option. Please choose 1-5."
        exit 1
        ;;
esac

echo ""
echo "🎯 Processing Complete!"
echo ""

# Show results if they exist
if [[ -d "results/final" ]] || [[ -d "results_test/final" ]]; then
    result_dir="results"
    [[ -d "results_test/final" ]] && result_dir="results_test"
    
    echo "Results Summary:"
    echo "==============="
    
    if [[ -f "${result_dir}/final/combined_quantification.csv" ]]; then
        echo "✅ Single-cell data: ${result_dir}/final/combined_quantification.csv"
        
        # Quick stats if Python available
        python -c "
import pandas as pd
try:
    df = pd.read_csv('${result_dir}/final/combined_quantification.csv')
    print(f'📊 Total cells detected: {len(df):,}')
    if 'X_centroid' in df.columns:
        extent_x = df['X_centroid'].max() - df['X_centroid'].min()  
        extent_y = df['Y_centroid'].max() - df['Y_centroid'].min()
        print(f'📐 Spatial extent: {extent_x:.0f} x {extent_y:.0f} pixels')
except:
    pass
" 2>/dev/null
    fi
    
    if [[ -f "${result_dir}/final/full_segmentation_mask.tiff" ]]; then
        echo "✅ Segmentation mask: ${result_dir}/final/full_segmentation_mask.tiff"
    fi
    
    if [[ -d "${result_dir}/spatial" ]]; then
        echo "✅ Spatial analysis: ${result_dir}/spatial/"
    fi
    
    echo ""
    echo "🚀 Ready for downstream analysis!"
    echo ""
    echo "Next steps:"
    echo "1. Load data in SCIMAP: sm.pp.mcmicro_to_scimap('${result_dir}/final/combined_quantification.csv')"
    echo "2. Define cell phenotypes based on your markers"
    echo "3. Run spatial analysis for RCNs and distance calculations"
fi

echo ""
echo "🎉 MCMICRO Tiled Processing Complete!"
echo ""
echo "Need help? Check the generated report files or reach out for support."