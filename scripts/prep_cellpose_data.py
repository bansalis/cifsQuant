#!/usr/bin/env python3
"""
Prepare data for manual import into CellProfiler
Simple CSV + images ready for direct import
"""

import pandas as pd
import numpy as np
import yaml
import tifffile
from pathlib import Path
import glob
import os

def prepare_manual_import():
    """Prepare simple files for manual CellProfiler import"""
    
    print("📊 PREPARING MANUAL CELLPROFILER IMPORT")
    print("=======================================")
    
    # Load config
    with open('phenotyping_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    manual_dir = Path("cellprofiler_manual_import")
    manual_dir.mkdir(exist_ok=True)
    
    for csv_file in glob.glob("results/*/final/combined_quantification.csv"):
        sample_name = Path(csv_file).parent.parent.name
        raw_image_path = f"rawdata/{sample_name}_aligned_stack.ome.tif"
        
        print(f"\nPreparing {sample_name}...")
        
        # Create sample directory
        sample_dir = manual_dir / sample_name
        sample_dir.mkdir(exist_ok=True)
        
        # Extract level 1 image
        if os.path.exists(raw_image_path):
            try:
                with tifffile.TiffFile(raw_image_path) as tif:
                    if len(tif.series) > 1:
                        image = tif.series[1].asarray()
                    else:
                        image = tif.series[0].asarray()
                        if len(image.shape) == 3:
                            image = image[:, ::4, ::4]
                
                level1_path = sample_dir / f"{sample_name}_level1.tif"
                tifffile.imwrite(str(level1_path), image, photometric='minisblack')
                print(f"  ✅ Image: {level1_path.name}")
                
            except Exception as e:
                print(f"  ⚠️  Image extraction failed: {e}")
                level1_path = None
        else:
            level1_path = None
            print(f"  ⚠️  Raw image not found: {raw_image_path}")
        
        # Prepare CSV data
        df = pd.read_csv(csv_file)
        channel_mapping = config['channel_mapping']
        df = df.rename(columns=channel_mapping)
        
        # Create clean CSV for CellProfiler (SUBSET for performance)
        intensity_cols = [col for col in df.columns if col in channel_mapping.values()]
        
        # Take larger subset for CellProfiler (25000 cells to capture rare populations)
        if len(df) > 25000:
            df_subset = df.sample(n=25000, random_state=42)
            print(f"  📉 Using subset: {len(df_subset)} cells (original: {len(df)})")
        else:
            df_subset = df
        
        cp_df = pd.DataFrame()
        cp_df['CellID'] = df_subset['CellID']
        cp_df['X_centroid'] = df_subset['X_centroid']
        cp_df['Y_centroid'] = df_subset['Y_centroid']
        
        # Add all intensity channels
        for col in intensity_cols:
            cp_df[col] = df_subset[col]
        
        # Add morphology if available
        morph_cols = ['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity']
        for col in morph_cols:
            if col in df_subset.columns:
                cp_df[col] = df_subset[col]
        
        csv_path = sample_dir / f"{sample_name}_data.csv"
        cp_df.to_csv(csv_path, index=False)
        
        print(f"  ✅ Data: {csv_path.name} ({len(cp_df)} cells, {len(intensity_cols)} channels)")
        
        # Create channel info
        channel_info = {
            'sample': sample_name,
            'channels': intensity_cols,
            'cells': len(df),
            'image_file': level1_path.name if level1_path else None,
            'csv_file': csv_path.name
        }
        
        with open(sample_dir / "channel_info.yaml", 'w') as f:
            yaml.dump(channel_info, f)
    
    # Create instructions
    instructions = """MANUAL CELLPROFILER IMPORT INSTRUCTIONS
=======================================

🎯 SIMPLE IMPORT WORKFLOW:

STEP 1: LAUNCH CELLPROFILER
1. Open CellProfiler
2. Start with blank project

STEP 2: IMPORT DATA
1. Modules → Data Tools → LoadData
2. Browse to: cellprofiler_manual_import/SAMPLE_NAME/
3. Select: SAMPLE_NAME_data.csv
4. Configure columns:
   - CellID → Object ID
   - X_centroid → Location X
   - Y_centroid → Location Y
   - Channel columns → Measurements

STEP 3: VISUALIZATION & GATING
1. Add module: DisplayScatterPlot
2. Select X-axis: Any channel (e.g., CD3)
3. Select Y-axis: Another channel (e.g., CD8B)
4. Run to see scatter plot

STEP 4: CLASSIFICATION OPTIONS:

Option A - ClassifyObjects:
1. Add ClassifyObjects module
2. Select measurement to classify by
3. Set thresholds interactively
4. Export classifications

Option B - Manual Gating:
1. Use DisplayScatterPlot
2. Note coordinates of population boundaries
3. Use FilterObjects with custom criteria
4. Export filtered populations

Option C - CellProfiler Analyst:
1. Export data as database
2. Launch CellProfiler Analyst
3. Use interactive classification tools

🎯 FLOW CYTOMETRY-STYLE ANALYSIS:

1. Create 2D scatter plots:
   - CD45 vs CD3 (immune vs T cells)
   - CD3 vs CD8B (T cell subsets)
   - Any channel vs KI67 (proliferation)

2. Gate populations:
   - Draw regions on scatter plots
   - Apply sequential gating
   - Export each population

3. Export results:
   - File → Export → Export to spreadsheet
   - Save gated populations as separate files

📊 YOUR CHANNELS:
- DAPI (nuclei)
- TOM (tumor marker)
- CD45 (pan-immune)
- CD3 (T cells)
- CD8B (CD8 T cells)
- KI67 (proliferation)
- PERK (ER stress)
- And others...

💡 TIPS:
- Start with broad populations (CD45+ immune)
- Refine with specific markers (CD3+ T cells)
- Use biological knowledge for gating
- Validate with known controls

READY FOR MANUAL IMPORT!
"""
    
    with open(manual_dir / "IMPORT_INSTRUCTIONS.txt", 'w') as f:
        f.write(instructions)
    
    print(f"\n🎉 MANUAL IMPORT READY!")
    print(f"📁 Files: cellprofiler_manual_import/")
    print(f"📖 Instructions: cellprofiler_manual_import/IMPORT_INSTRUCTIONS.txt")
    print(f"🚀 Open CellProfiler and import CSV files manually")

if __name__ == "__main__":
    prepare_manual_import()