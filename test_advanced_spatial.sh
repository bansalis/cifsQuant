#!/bin/bash
# Test script for Advanced Spatial Analysis Pipeline

echo "=================================================="
echo "Advanced Spatial Analysis Pipeline - Validation"
echo "=================================================="
echo ""

# Check if required files exist
echo "Checking required files..."

if [ -f "advanced_spatial_analysis.py" ]; then
    echo "✓ advanced_spatial_analysis.py found"
else
    echo "✗ advanced_spatial_analysis.py missing"
    exit 1
fi

if [ -f "run_advanced_spatial_analysis.py" ]; then
    echo "✓ run_advanced_spatial_analysis.py found"
else
    echo "✗ run_advanced_spatial_analysis.py missing"
    exit 1
fi

if [ -f "ADVANCED_SPATIAL_ANALYSIS_README.md" ]; then
    echo "✓ ADVANCED_SPATIAL_ANALYSIS_README.md found"
else
    echo "✗ ADVANCED_SPATIAL_ANALYSIS_README.md missing"
    exit 1
fi

if [ -f "requirements_advanced_spatial.txt" ]; then
    echo "✓ requirements_advanced_spatial.txt found"
else
    echo "✗ requirements_advanced_spatial.txt missing"
    exit 1
fi

# Check if scripts are executable
if [ -x "run_advanced_spatial_analysis.py" ]; then
    echo "✓ run_advanced_spatial_analysis.py is executable"
else
    echo "✓ run_advanced_spatial_analysis.py exists but not executable (can be run with python)"
fi

# Check Python syntax
echo ""
echo "Checking Python syntax..."
python3 -m py_compile advanced_spatial_analysis.py && echo "✓ advanced_spatial_analysis.py: syntax OK"
python3 -m py_compile run_advanced_spatial_analysis.py && echo "✓ run_advanced_spatial_analysis.py: syntax OK"

echo ""
echo "=================================================="
echo "Validation Complete!"
echo "=================================================="
echo ""
echo "To run the analysis, first install dependencies:"
echo "  pip install -r requirements_advanced_spatial.txt"
echo ""
echo "Then run the pipeline:"
echo "  python run_advanced_spatial_analysis.py --input_dir results/ \\"
echo "                                           --metadata sample_metadata.csv \\"
echo "                                           --markers markers.csv"
echo ""
