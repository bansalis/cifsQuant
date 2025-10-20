#!/usr/bin/env python3
"""
CellProfiler Analyst Setup for ML-based Phenotyping
Interactive classification with example-based learning
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import glob
import os
import sqlite3

def create_cpa_database(df, sample_name, output_dir):
    """Create CellProfiler Analyst database"""
    
    db_path = output_dir / f"{sample_name}_cpa.db"
    
    # Create SQLite database
    conn = sqlite3.connect(str(db_path))
    
    # Create Per_Image table
    per_image = pd.DataFrame({
        'ImageNumber': [1],
        'Image_FileName_OrigBlue': [f'{sample_name}.tif'],
        'Image_PathName_OrigBlue': [str(output_dir)],
        'Image_Count_Cells': [len(df)]
    })
    per_image.to_sql('Per_Image', conn, if_exists='replace', index=False)
    
    # Create Per_Object table with proper CPA format
    df_cpa = df.copy()
    df_cpa['ImageNumber'] = 1
    df_cpa['ObjectNumber'] = df_cpa['CellID']
    
    # Rename columns to CPA convention
    column_mapping = {}
    for col in df_cpa.columns:
        if col in ['ImageNumber', 'ObjectNumber', 'X_centroid', 'Y_centroid']:
            continue
        elif 'Channel_' in col or col in ['DAPI', 'TOM', 'CD45', 'CD3', 'CD8B', 'KI67', 'PERK']:
            column_mapping[col] = f'Intensity_MeanIntensity_{col}'
        elif col in ['Area', 'MajorAxisLength', 'MinorAxisLength']:
            column_mapping[col] = f'AreaShape_{col}'
    
    df_cpa = df_cpa.rename(columns=column_mapping)
    
    # Save to database
    df_cpa.to_sql('Per_Object', conn, if_exists='replace', index=False)
    conn.close()
    
    return str(db_path)

def setup_cpa_properties(db_path, sample_name, channels, output_dir):
    """Create CellProfiler Analyst properties file"""
    
    properties_content = f"""# CellProfiler Analyst Properties File
# Generated for {sample_name}

# Database
db_type = sqlite
db_sqlite_file = {os.path.basename(db_path)}

# Tables
image_table = Per_Image
object_table = Per_Object

# Table structure
image_id = ImageNumber
object_id = ObjectNumber

# Object coordinates for visualization
cell_x_loc = X_centroid
cell_y_loc = Y_centroid

# Image information
image_names = Image_FileName_OrigBlue
image_path_cols = Image_PathName_OrigBlue

# Classification columns (intensities)
"""
    
    # Add intensity channels
    for channel in channels:
        if channel in ['DAPI', 'TOM', 'CD45', 'CD3', 'CD8B', 'KI67', 'PERK']:
            properties_content += f"# {channel} intensity\n"
    
    properties_content += f"""
# Morphology features
# Area, shape features available

# Training set storage
training_set = {sample_name}_training_set.txt

# Groups (for organizing data)
group_SQL_OrigBlue = SELECT ImageNumber, Image_FileName_OrigBlue FROM Per_Image

# Default fetch size
image_buffer_size = 1
"""
    
    properties_path = output_dir / f"{sample_name}.properties"
    with open(properties_path, 'w') as f:
        f.write(properties_content)
    
    return str(properties_path)

def setup_cpa_data():
    """Convert all samples to CellProfiler Analyst format"""
    
    # Load config
    with open('phenotyping_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("🤖 CELLPROFILER ANALYST SETUP")
    print("=============================")
    
    cpa_dir = Path("cellprofiler_analyst")
    cpa_dir.mkdir(exist_ok=True)
    
    samples_info = []
    
    for csv_file in glob.glob("results/*/final/combined_quantification.csv"):
        sample_name = Path(csv_file).parent.parent.name
        print(f"\nProcessing {sample_name}...")
        
        # Read data
        df = pd.read_csv(csv_file)
        
        # Apply channel mapping
        channel_mapping = config['channel_mapping']
        df = df.rename(columns=channel_mapping)
        
        # Create sample directory
        sample_dir = cpa_dir / sample_name
        sample_dir.mkdir(exist_ok=True)
        
        # Get channels
        intensity_channels = [col for col in df.columns if col in channel_mapping.values()]
        
        # Create database
        db_path = create_cpa_database(df, sample_name, sample_dir)
        
        # Create properties file
        props_path = setup_cpa_properties(db_path, sample_name, intensity_channels, sample_dir)
        
        # Create sample CSV for backup
        df_export = df[['CellID', 'X_centroid', 'Y_centroid'] + intensity_channels].copy()
        df_export.to_csv(sample_dir / f"{sample_name}_cells.csv", index=False)
        
        print(f"  ✅ Database: {os.path.basename(db_path)}")
        print(f"  ✅ Properties: {os.path.basename(props_path)}")
        print(f"  📊 {len(df)} cells, {len(intensity_channels)} channels")
        
        samples_info.append({
            'sample': sample_name,
            'database': db_path,
            'properties': props_path,
            'cells': len(df),
            'channels': intensity_channels
        })
    
    # Create workflow instructions
    instructions = f"""CELLPROFILER ANALYST WORKFLOW
================================

🎯 MACHINE LEARNING CLASSIFICATION WORKFLOW

📁 Data Location: {os.getcwd()}/cellprofiler_analyst/

🚀 SETUP:
1. Install CellProfiler Analyst:
   pip install cellprofiler-analyst
   
   OR download standalone version:
   https://cellprofiler.org/releases/

2. Launch CellProfiler Analyst:
   cellprofiler-analyst

📊 CLASSIFICATION WORKFLOW:

For each sample:
{chr(10).join([f"- {info['sample']}: {info['cells']} cells" for info in samples_info])}

STEP 1: LOAD DATA
1. Open CellProfiler Analyst
2. File → Load Properties → Select {samples_info[0]['sample']}/{samples_info[0]['sample']}.properties
3. Verify data loads (should see cell measurements)

STEP 2: CREATE CLASSIFIER
1. Tools → Classifier
2. Select "Create New Classifier"
3. Choose classification method:
   - Random Forest (recommended)
   - SVM 
   - Fast Gentle Boosting

STEP 3: TRAIN WITH EXAMPLES
1. Click "Fetch Random Objects"
2. Review cell images and measurements
3. Drag cells to classification bins:
   - CD3+ T cells
   - CD8+ T cells  
   - Tumor cells (TOM+)
   - Immune cells (CD45+)
   - Proliferating (KI67+)
   - Other/Negative

4. Add 50-100 examples per class
5. Click "Train Classifier"

STEP 4: VALIDATE CLASSIFIER
1. Click "Test Classifier"
2. Review accuracy metrics
3. Add more training examples if accuracy < 85%
4. Retrain until satisfied

STEP 5: CLASSIFY ALL CELLS
1. Click "Classify All Objects"
2. Wait for classification to complete
3. Review classification results

STEP 6: EXPORT RESULTS
1. File → Export → Classifications
2. Save as CSV file
3. Results include predicted phenotypes for all cells

🔧 ADVANCED FEATURES:
- Feature selection to identify most important channels
- Cross-validation for robust model evaluation  
- Batch processing multiple samples
- Interactive scatter plots for validation

💡 TIPS:
- Start with clear, obvious examples
- Use biological knowledge to guide classification
- Validate results with known positive controls
- Iterate training with difficult/ambiguous cells

📈 EXPECTED OUTCOMES:
- Automated phenotype classification
- Confidence scores for each prediction
- Feature importance rankings
- Exportable classification models

🔄 QUALITY CONTROL:
- Check classification accuracy > 85%
- Validate on held-out test cells
- Compare with manual gating results
- Review feature importance makes biological sense

READY TO START CLASSIFICATION!
"""
    
    with open(cpa_dir / "CPA_WORKFLOW_INSTRUCTIONS.txt", 'w') as f:
        f.write(instructions)
    
    print(f"\n🎉 CELLPROFILER ANALYST SETUP COMPLETE!")
    print(f"📁 Data: cellprofiler_analyst/")
    print(f"📖 Instructions: cellprofiler_analyst/CPA_WORKFLOW_INSTRUCTIONS.txt")
    print(f"🚀 Next: pip install cellprofiler-analyst && cellprofiler-analyst")

if __name__ == "__main__":
    setup_cpa_data()