#!/usr/bin/env python3
"""
Complete histoCAT Setup and Data Preparation
Converts nextflow output to histoCAT format with proper structure
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import glob
import shutil
import os

def setup_histocat_data():
    """Convert all samples to histoCAT format"""
    
    # Load config
    with open('phenotyping_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create histoCAT directory structure
    histocat_dir = Path("histocat_analysis")
    histocat_dir.mkdir(exist_ok=True)
    
    print("🔬 SETTING UP HISTOCAT DATA")
    print("===========================")
    
    # Process all samples
    samples_processed = []
    
    for csv_file in glob.glob("results/*/final/combined_quantification.csv"):
        sample_name = Path(csv_file).parent.parent.name
        print(f"\nProcessing {sample_name}...")
        
        # Read data
        df = pd.read_csv(csv_file)
        print(f"  Original data: {len(df)} cells, {len(df.columns)} columns")
        
        # Apply channel mapping from config
        channel_mapping = config['channel_mapping']
        df = df.rename(columns=channel_mapping)
        
        # histoCAT required columns
        df['ImageId'] = 1  # Single image per sample
        df['ObjectNumber'] = df['CellID']
        df['Location_Center_X'] = df['X_centroid']
        df['Location_Center_Y'] = df['Y_centroid']
        
        # Keep only intensity columns and required metadata
        intensity_cols = [col for col in df.columns if col in channel_mapping.values()]
        required_cols = ['ImageId', 'ObjectNumber', 'Location_Center_X', 'Location_Center_Y']
        morphology_cols = ['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'Solidity']
        
        # Select columns for histoCAT
        available_morphology = [col for col in morphology_cols if col in df.columns]
        final_cols = required_cols + intensity_cols + available_morphology
        
        df_histocat = df[final_cols].copy()
        
        # Create sample directory
        sample_dir = histocat_dir / sample_name
        sample_dir.mkdir(exist_ok=True)
        
        # Save histoCAT CSV
        histocat_csv = sample_dir / f"{sample_name}_histocat.csv"
        df_histocat.to_csv(histocat_csv, index=False)
        
        # Create channel information file
        channels_info = {
            'intensity_channels': intensity_cols,
            'morphology_channels': available_morphology,
            'total_cells': len(df_histocat),
            'channel_mapping': channel_mapping
        }
        
        with open(sample_dir / "channel_info.yaml", 'w') as f:
            yaml.dump(channels_info, f)
        
        # Create summary statistics
        summary_stats = []
        for col in intensity_cols + available_morphology:
            stats = {
                'Channel': col,
                'Mean': df_histocat[col].mean(),
                'Median': df_histocat[col].median(),
                'Min': df_histocat[col].min(),
                'Max': df_histocat[col].max(),
                'Std': df_histocat[col].std()
            }
            summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(sample_dir / "channel_summary.csv", index=False)
        
        print(f"  ✅ histoCAT file: {histocat_csv}")
        print(f"     {len(intensity_cols)} intensity channels, {len(available_morphology)} morphology features")
        
        samples_processed.append({
            'sample': sample_name,
            'cells': len(df_histocat),
            'channels': len(intensity_cols),
            'file': str(histocat_csv)
        })
    
    # Create master summary
    master_summary = pd.DataFrame(samples_processed)
    master_summary.to_csv(histocat_dir / "samples_summary.csv", index=False)
    
    # Create histoCAT instructions
    current_dir = os.getcwd()
    instructions = f"""
HISTOCAT WORKFLOW INSTRUCTIONS
==============================

📁 DATA LOCATION: {current_dir}/histocat_analysis

🔧 SETUP STEPS:
1. Download histoCAT from: https://github.com/BodenmillerGroup/histoCAT/releases
2. Install MATLAB Runtime (required for histoCAT)
3. Launch histoCAT application

📊 DATA LOADING:
For each sample in {histocat_dir}/:
1. Open histoCAT
2. Click "Load Data" → "Load CSV"
3. Navigate to sample folder (e.g., {histocat_dir}/GUEST29/)
4. Load: SAMPLE_NAME_histocat.csv
5. Verify channels loaded correctly

🎯 MANUAL GATING WORKFLOW:
1. EXPLORE DATA:
   - Use "Data" tab to view cell measurements
   - Check "Channel Summary" for intensity ranges
   
2. CREATE GATES:
   - Click "Gating" tab
   - Select two channels for 2D plot
   - Draw polygonal gates around cell populations
   - Name each gate (e.g., "CD3_positive", "CD8_high")
   
3. HIERARCHICAL GATING:
   - Create parent gates first (e.g., "Live_cells")
   - Create child gates within parent populations
   - Build gating tree: All cells → CD45+ → CD3+ → CD8+ T cells
   
4. SUGGESTED GATING STRATEGY:
   Based on your channels: {list(config['channel_mapping'].values())}
   
   Gate 1: DAPI+ (nucleated cells)
   Gate 2: CD45+ (immune cells)  
   Gate 3: CD3+ (T cells)
   Gate 4: CD8B+ (CD8 T cells)
   Gate 5: KI67+ (proliferating)
   
   Combinations:
   - CD3+ CD8B+ = CD8 T cells
   - CD3+ CD8B- = CD4 T cells  
   - CD3+ KI67+ = Proliferating T cells
   - TOM+ = Tumor cells
   - CD45+ CD3- = Non-T immune cells

5. EXPORT RESULTS:
   - Select all gates
   - Click "Export" → "Export Gated Populations"
   - Save as CSV files
   - Each gate becomes a separate file

📈 QUALITY CONTROL:
- Check gate statistics in histoCAT
- Verify cell counts make biological sense
- Review 2D plots for gate boundaries
- Export representative plots

🔄 ITERATIVE REFINEMENT:
- Adjust gates based on biological knowledge
- Re-export updated populations
- Compare across samples for consistency

SAMPLES READY FOR HISTOCAT:
{chr(10).join([f"- {s['sample']}: {s['cells']} cells, {s['channels']} channels" for s in samples_processed])}

📧 Need help? Check histoCAT documentation or GitHub issues.
"""
    
    with open(histocat_dir / "HISTOCAT_INSTRUCTIONS.txt", 'w') as f:
        f.write(instructions)
    
    print(f"\n🎉 HISTOCAT SETUP COMPLETE!")
    print(f"📁 Data location: {current_dir}/histocat_analysis")
    print(f"📋 Samples processed: {len(samples_processed)}")
    print(f"📖 Instructions: histocat_analysis/HISTOCAT_INSTRUCTIONS.txt")
    
    return histocat_dir

if __name__ == "__main__":
    setup_histocat_data()