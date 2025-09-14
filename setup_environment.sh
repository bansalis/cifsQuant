#!/bin/bash

# MCMICRO Tiled Processor - Environment Setup
# ===========================================

echo "Setting up Python environment for MCMICRO Tiled Processor..."
echo ""

# Check if we're in the right directory
if [[ ! -f "requirements.txt" ]]; then
    echo "Error: requirements.txt not found in current directory"
    echo "Make sure you're in the directory with the MCMICRO tiled processor files"
    exit 1
fi

# Install system dependencies first
echo "1. Installing system dependencies..."
echo "   (You may be prompted for sudo password)"

# Update package list
sudo apt update

# Install Python 3 and venv if not already installed
sudo apt install -y python3-full python3-pip python3-venv

# Install system-level packages that are often needed
sudo apt install -y python3-dev build-essential libhdf5-dev pkg-config

echo "✓ System dependencies installed"
echo ""

# Create virtual environment
echo "2. Creating Python virtual environment..."
python3 -m venv mcmicro_env

echo "✓ Virtual environment created: mcmicro_env/"
echo ""

# Activate virtual environment and install packages
echo "3. Activating environment and installing Python packages..."
source mcmicro_env/bin/activate

# Upgrade pip in the virtual environment
pip install --upgrade pip setuptools wheel

# Install core scientific packages first (these often have conflicts)
echo "   Installing core scientific packages..."
pip install numpy pandas

# Install image processing packages
echo "   Installing image processing packages..."
pip install tifffile scikit-image scipy matplotlib seaborn tqdm

# Install machine learning packages
echo "   Installing machine learning packages..."
pip install scikit-learn joblib

# Try to install spatial analysis packages (these might fail on some systems)
echo "   Installing spatial analysis packages..."
pip install anndata || echo "   Warning: anndata installation failed - you may need to install manually"

# Try to install SCIMAP (this often has dependency issues)
echo "   Attempting SCIMAP installation..."
pip install scimap || echo "   Warning: scimap installation failed - see alternative installation below"

# Install scanpy as backup for spatial analysis
pip install scanpy || echo "   Warning: scanpy installation failed"

echo "✓ Python packages installed"
echo ""

# Test the installation
echo "4. Testing installation..."
python3 -c "
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from skimage import segmentation
import scipy
print('✓ Core packages working')

try:
    import anndata
    print('✓ anndata working')
except ImportError:
    print('⚠ anndata not available')

try:
    import scimap
    print('✓ scimap working')
except ImportError:
    print('⚠ scimap not available - see installation notes below')

try:
    import scanpy
    print('✓ scanpy working')  
except ImportError:
    print('⚠ scanpy not available')
"

echo ""
echo "5. Environment setup complete!"
echo ""
echo "To use this environment:"
echo "========================"
echo "# Activate the environment:"
echo "source mcmicro_env/bin/activate"
echo ""
echo "# Run your pipeline:"
echo "python mcmicro_tiled_processor.py --input your_image.tiff --output results"
echo ""
echo "# Deactivate when done:"
echo "deactivate"
echo ""

# Create activation script for convenience
cat > activate_mcmicro.sh << 'EOF'
#!/bin/bash
echo "Activating MCMICRO environment..."
source mcmicro_env/bin/activate
echo "Environment activated. You can now run:"
echo "  python mcmicro_tiled_processor.py --help"
echo ""
echo "To deactivate, simply run: deactivate"
EOF

chmod +x activate_mcmicro.sh

echo "Convenience script created: ./activate_mcmicro.sh"
echo ""

# Check if SCIMAP installation failed and provide alternatives
if ! python3 -c "import scimap" 2>/dev/null; then
    echo "SCIMAP Installation Troubleshooting:"
    echo "===================================="
    echo "If SCIMAP failed to install, try these alternatives:"
    echo ""
    echo "Option 1: Install from conda-forge (if you have conda):"
    echo "  conda install -c conda-forge scimap"
    echo ""
    echo "Option 2: Install development version:"
    echo "  pip install git+https://github.com/labsyspharm/scimap.git"
    echo ""
    echo "Option 3: Install without SCIMAP (basic spatial analysis only):"
    echo "  The pipeline will still work for segmentation and quantification"
    echo "  You can do spatial analysis separately with other tools"
    echo ""
fi

# Check if we're still in the virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Environment is currently active. Type 'deactivate' to exit."
else
    echo "Remember to activate the environment before running the pipeline:"
    echo "  source mcmicro_env/bin/activate"
fi

echo ""
echo "Setup complete! 🎉"