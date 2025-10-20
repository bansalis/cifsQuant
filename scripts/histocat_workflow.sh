#!/bin/bash

# Complete histoCAT Workflow Setup
set -euo pipefail

echo "🔬 HISTOCAT MANUAL GATING WORKFLOW"
echo "=================================="

# Step 1: Prepare data
echo "Step 1: Preparing data for histoCAT..."
python3 setup_histocat_complete.py

echo ""
echo "Step 2: Installing histoCAT (if needed)..."

# Check if running on Windows/WSL
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ -f /proc/version ]] && grep -q Microsoft /proc/version; then
    echo "🖥️  WINDOWS/WSL DETECTED"
    echo ""
    echo "📥 DOWNLOAD HISTOCAT:"
    echo "1. Go to: https://github.com/BodenmillerGroup/histoCAT/releases/latest"
    echo "2. Download: histoCAT_windows.zip"
    echo "3. Extract to: C:\\histoCAT\\"
    echo ""
    echo "📦 INSTALL MATLAB RUNTIME (REQUIRED):"
    echo "1. Download MATLAB Runtime R2019b from:"
    echo "   https://www.mathworks.com/products/compiler/matlab-runtime.html"
    echo "2. Install the runtime (free, no MATLAB license needed)"
    echo "3. Add runtime to PATH if prompted"
    echo ""
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 MACOS DETECTED" 
    echo ""
    echo "📥 DOWNLOAD HISTOCAT:"
    echo "1. Go to: https://github.com/BodenmillerGroup/histoCAT/releases/latest"
    echo "2. Download: histoCAT_mac.zip"
    echo "3. Extract and move to Applications"
    echo ""
    echo "📦 INSTALL MATLAB RUNTIME:"
    echo "1. Download MATLAB Runtime R2019b for macOS"
    echo "2. Install using the installer package"
    
else
    echo "🐧 LINUX DETECTED"
    echo ""
    echo "📥 DOWNLOAD HISTOCAT:"
    echo "1. Go to: https://github.com/BodenmillerGroup/histoCAT/releases/latest"
    echo "2. Download: histoCAT_linux.zip"
    echo "3. Extract to ~/histoCAT/"
    echo ""
    echo "📦 INSTALL MATLAB RUNTIME:"
    echo "wget https://ssd.mathworks.com/supportfiles/downloads/R2019b/Release/6/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2019b_Update_6_glnxa64.zip"
    echo "unzip MATLAB_Runtime_R2019b_Update_6_glnxa64.zip"
    echo "sudo ./install -mode silent -agreeToLicense yes"
fi

echo ""
echo "Step 3: Launching histoCAT workflow..."

# Create desktop shortcut script
cat > launch_histocat_workflow.sh << 'EOF'
#!/bin/bash

echo "🔬 HISTOCAT GATING WORKFLOW"
echo "=========================="
echo ""
echo "📁 Your data is ready in: $(pwd)/histocat_analysis/"
echo ""
echo "🚀 WORKFLOW STEPS:"
echo ""
echo "1. LAUNCH HISTOCAT:"
echo "   - Windows: Run histoCAT.exe from installation folder"
echo "   - macOS: Open histoCAT from Applications"  
echo "   - Linux: ./run_histoCAT.sh from histoCAT folder"
echo ""
echo "2. LOAD YOUR FIRST SAMPLE:"
echo "   - Click 'Load Data' → 'Load CSV'"
echo "   - Navigate to: $(pwd)/histocat_analysis/GUEST29/"
echo "   - Select: GUEST29_histocat.csv"
echo "   - Verify channels loaded (should see DAPI, TOM, CD45, etc.)"
echo ""
echo "3. START GATING:"
echo "   - Click 'Gating' tab"
echo "   - Create 2D scatter plot (e.g., DAPI vs CD45)"
echo "   - Draw polygon gate around populations"
echo "   - Name gates descriptively"
echo ""
echo "4. SUGGESTED GATING ORDER:"
echo "   Gate 1: DAPI+ (nucleated cells)"
echo "   Gate 2: CD45+ (from DAPI+, immune cells)"
echo "   Gate 3: CD3+ (from CD45+, T cells)"
echo "   Gate 4: CD8B+ (from CD3+, CD8 T cells)"
echo "   Gate 5: CD8B- (from CD3+, CD4 T cells)"
echo "   Gate 6: KI67+ (from any population, proliferating)"
echo "   Gate 7: TOM+ (tumor cells)"
echo ""
echo "5. EXPORT RESULTS:"
echo "   - Select all gates"
echo "   - Click 'Export' → 'Export Gated Populations'"
echo "   - Save to: $(pwd)/histocat_analysis/gated_results/"
echo ""
echo "📖 Full instructions: $(pwd)/histocat_analysis/HISTOCAT_INSTRUCTIONS.txt"
echo ""
read -p "Press Enter when you've completed gating and exported results..."

# Check if results were exported
if [[ -d "histocat_analysis/gated_results" ]] && [[ -n "$(ls -A histocat_analysis/gated_results/)" ]]; then
    echo "✅ Gated results found!"
    echo "📊 Processing gated populations..."
    
    # Process exported gates
    python3 - << 'PYTHON_EOF'
import pandas as pd
import glob
from pathlib import Path
import yaml

# Load config for phenotype mapping
with open('phenotyping_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

results_dir = Path("histocat_analysis/gated_results")
final_dir = Path("phenotyping_analysis/histocat_results")
final_dir.mkdir(parents=True, exist_ok=True)

print("Processing exported gates...")

# Combine all gated populations
all_gated_cells = []

for gate_file in results_dir.glob("*.csv"):
    gate_name = gate_file.stem
    df = pd.read_csv(gate_file)
    df['histocat_gate'] = gate_name
    all_gated_cells.append(df)
    print(f"  {gate_name}: {len(df)} cells")

if all_gated_cells:
    # Combine all gated populations
    combined_df = pd.concat(all_gated_cells, ignore_index=True)
    
    # Map to phenotypes using config
    phenotype_mapping = {
        'DAPI_positive': 'Nucleated',
        'CD45_positive': 'Immune', 
        'CD3_positive': 'T_cells',
        'CD8B_positive': 'CD8_T_cells',
        'TOM_positive': 'Tumor',
        'KI67_positive': 'Proliferating'
    }
    
    # Apply phenotype mapping
    combined_df['phenotype'] = combined_df['histocat_gate'].map(phenotype_mapping).fillna('Other')
    
    # Save final results
    combined_df.to_csv(final_dir / "histocat_phenotyped_cells.csv", index=False)
    
    # Create summary
    summary = combined_df.groupby(['phenotype', 'histocat_gate']).size().reset_index(name='cell_count')
    summary.to_csv(final_dir / "histocat_phenotype_summary.csv", index=False)
    
    print(f"✅ Final results saved to: {final_dir}")
    print(f"Total gated cells: {len(combined_df)}")
    
else:
    print("⚠️  No gated results found")

PYTHON_EOF

    echo ""
    echo "🎉 HISTOCAT WORKFLOW COMPLETE!"
    echo "📊 Results available in: phenotyping_analysis/histocat_results/"
    
else
    echo "⚠️  No gated results found in histocat_analysis/gated_results/"
    echo "Please complete gating in histoCAT and export results"
fi
EOF

chmod +x launch_histocat_workflow.sh

echo ""
echo "🎯 SETUP COMPLETE!"
echo "=================="
echo ""
echo "NEXT STEPS:"
echo "1. Install histoCAT and MATLAB Runtime (see instructions above)"
echo "2. Run: ./launch_histocat_workflow.sh"
echo "3. Follow the interactive gating workflow"
echo ""
echo "📁 Your data is ready in: $(pwd)/histocat_analysis/"
echo "📖 Instructions: $(pwd)/histocat_analysis/HISTOCAT_INSTRUCTIONS.txt"
echo "🚀 Launcher: $(pwd)/launch_histocat_workflow.sh"