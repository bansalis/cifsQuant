#!/usr/bin/env python3
"""
Create CellProfiler project (.cppipe) and extract level 1 images
"""

import pandas as pd
import numpy as np
import yaml
import tifffile
from pathlib import Path
import glob
import os
import json

def extract_level1_image(raw_image_path, output_dir, sample_name):
    """Extract level 1 pyramid image for CellProfiler"""
    
    print(f"Extracting level 1 image for {sample_name}...")
    
    if not os.path.exists(raw_image_path):
        print(f"⚠️  Raw image not found: {raw_image_path}")
        return None
    
    try:
        with tifffile.TiffFile(raw_image_path) as tif:
            # Get pyramid level 1 (reduced resolution)
            if len(tif.series) > 1:
                image = tif.series[1].asarray()
                print(f"  Using pyramid level 1: {image.shape}")
            else:
                # If no pyramid, downsample by factor of 4
                image = tif.series[0].asarray()
                print(f"  No pyramid found, downsampling from: {image.shape}")
                if len(image.shape) == 3:
                    image = image[:, ::4, ::4]
                else:
                    image = image[::4, ::4]
                print(f"  Downsampled to: {image.shape}")
        
        # Save level 1 image
        level1_path = output_dir / f"{sample_name}_level1.ome.tif"
        tifffile.imwrite(str(level1_path), image, photometric='minisblack')
        
        print(f"  ✅ Level 1 image saved: {level1_path}")
        return str(level1_path)
        
    except Exception as e:
        print(f"  ❌ Error extracting image: {e}")
        return None

def create_cellprofiler_pipeline(sample_name, image_path, csv_path, output_dir, channel_names):
    """Create CellProfiler pipeline (.cppipe) file"""
    
    # Use forward slashes for paths
    image_path = image_path.replace('\\', '/')
    csv_path = csv_path.replace('\\', '/')
    csv_basename = os.path.basename(csv_path)
    
    # Use actual channel names from config
    primary_channel = channel_names[0] if channel_names else "DAPI"
    
    pipeline_content = f'''CellProfiler Pipeline: http://www.cellprofiler.org
Version:5
DateRevision:406
GitHash:
ModuleCount:6
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:'Unknown'|variable_revision_number:2|show_window:False|notes:['Load the {sample_name} image']|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    :
    Filter images?:No filtering
    Select the rule criteria:and (extension does isimage) (directory doesnot containregexp "[\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\]\\\\\\\\\\\\\\\\.")

Metadata:[module_num:2|svn_version:'Unknown'|variable_revision_number:6|show_window:False|notes:['Extract metadata']|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Extract metadata?:No
    Metadata data type:Text
    Metadata types:{{}}
    Extraction method count:1
    Metadata extraction method:Extract from file/folder names
    Metadata source:File name
    Regular expression to extract from file name:^(?P<Plate>.*)_(?P<Well>[A-P][0-9]{{2}})_s(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])
    Regular expression to extract from folder name:(?P<Date>[0-9]{{4}}_[0-9]{{2}}_[0-9]{{2}})$
    Extract metadata from:All images
    Select the filtering criteria:and (file does contain "")
    Metadata file location:Elsewhere...|
    Match file and image metadata:[]
    Use case insensitive matching?:No
    Metadata file name:
    Does cached metadata exist?:No

NamesAndTypes:[module_num:3|svn_version:'Unknown'|variable_revision_number:8|show_window:False|notes:['Assign channel names']|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Assign a name to:All images
    Select the image type:Grayscale image
    Name to assign these images:{sample_name}
    Match metadata:[]
    Image set matching method:Order
    Set intensity range from:Image metadata
    Assignments count:1
    Single images count:0
    Max intensity:255.0
    Process as 3D?:No
    Relative pixel spacing in X:1.0
    Relative pixel spacing in Y:1.0
    Relative pixel spacing in Z:1.0
    Select the rule criteria:and (file does startwith "{sample_name}_level1")
    Name to assign these images:{sample_name}
    Name to assign these objects:Cell
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Maximum intensity:255.0

Groups:[module_num:4|svn_version:'Unknown'|variable_revision_number:2|show_window:False|notes:['Group images']|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Do you want to group your images?:No
    grouping metadata count:1
    Metadata category:None

LoadData:[module_num:5|svn_version:'Unknown'|variable_revision_number:6|show_window:False|notes:['Load cell measurements']|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:True]
    Input data file location:Default Input Folder|{csv_basename}
    Name of the file:
    Load images?:No
    Base image location:Default Input Folder|
    Process just a range of rows?:No
    Rows to process:1,100000
    Group images by metadata?:No
    Select metadata tags for grouping:
    Rescale intensities?:Yes

ClassifyObjects:[module_num:6|svn_version:'Unknown'|variable_revision_number:2|show_window:True|notes:['Interactive classifier for phenotyping']|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:True]
    Make each classification decision on a object-by-object basis?:Yes
    Select the object name:Cells
    Select the measurement to classify by:{primary_channel}
    Select bin spacing:Custom-defined bins
    Number of bins:3
    Lower threshold:0.0
    Use a bin for objects below the threshold?:Yes
    Upper threshold:1.0
    Use a bin for objects above the threshold?:Yes
    Enter the custom thresholds separating the values between bins:0.5
    Give each bin a label?:Yes
    Enter the bin labels separated by commas:Negative,Positive,High
    Retain an image of the classified objects?:No
    Name the output image:ClassifiedNuclei
    Select a measurement to use for object highlighting:None
    Select the color to use for object highlighting:Red
    Select the label to use for object highlighting:Positive
'''
    
    pipeline_path = output_dir / f"{sample_name}.cppipe"
    with open(pipeline_path, 'w') as f:
        f.write(pipeline_content)
    
    return str(pipeline_path)

def setup_cellprofiler_projects():
    """Setup CellProfiler projects for all samples"""
    
    print("🧬 CELLPROFILER PROJECT SETUP")
    print("==============================")
    
    # Load config
    with open('phenotyping_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    cp_dir = Path("cellprofiler_projects")
    cp_dir.mkdir(exist_ok=True)
    
    projects_info = []
    
    for csv_file in glob.glob("results/*/final/combined_quantification.csv"):
        sample_name = Path(csv_file).parent.parent.name
        raw_image_path = f"rawdata/{sample_name}_aligned_stack.ome.tif"
        
        print(f"\nProcessing {sample_name}...")
        
        # Create sample directory
        sample_dir = cp_dir / sample_name
        sample_dir.mkdir(exist_ok=True)
        
        # Extract level 1 image
        level1_image = extract_level1_image(raw_image_path, sample_dir, sample_name)
        
        if not level1_image:
            print(f"  ⚠️  Skipping {sample_name} - no image")
            continue
        
        # Read and prepare CSV data
        df = pd.read_csv(csv_file)
        
        # Apply channel mapping
        channel_mapping = config['channel_mapping']
        df = df.rename(columns=channel_mapping)
        
        # Create CellProfiler-compatible CSV
        cp_csv = df[['CellID', 'X_centroid', 'Y_centroid'] + 
                   [col for col in df.columns if col in channel_mapping.values()]].copy()
        
        # Add required columns for CellProfiler
        cp_csv['ImageNumber'] = 1
        cp_csv['ObjectNumber'] = cp_csv['CellID']
        
        csv_path = sample_dir / f"{sample_name}_measurements.csv"
        cp_csv.to_csv(csv_path, index=False)
        
        # Get intensity channels from config
        intensity_channels = [col for col in df.columns if col in channel_mapping.values()]
        
        # Create CellProfiler pipeline
        pipeline_path = create_cellprofiler_pipeline(sample_name, level1_image, str(csv_path), sample_dir, intensity_channels)
        
        print(f"  ✅ Pipeline: {os.path.basename(pipeline_path)}")
        print(f"  ✅ Image: {os.path.basename(level1_image)}")
        print(f"  ✅ Data: {os.path.basename(csv_path)}")
        
        projects_info.append({
            'sample': sample_name,
            'pipeline': pipeline_path,
            'image': level1_image,
            'csv': str(csv_path),
            'cells': len(df)
        })
    
    # Create instructions
    if not projects_info:
        print("⚠️  No projects created - check raw images and imagecodecs installation")
        return
        
    instructions = f"""CELLPROFILER PHENOTYPING WORKFLOW
==================================

📁 Projects Location: {os.getcwd()}/cellprofiler_projects/

🚀 WORKFLOW FOR EACH SAMPLE:
{chr(10).join([f"- {info['sample']}: {info['cells']} cells" for info in projects_info])}

STEP 1: OPEN PROJECT
1. Launch CellProfiler
2. File → Open Project → Select {projects_info[0]['sample']}/{projects_info[0]['sample']}.cppipe
3. Verify image and data load correctly

STEP 2: INTERACTIVE CLASSIFICATION  
1. Click "Analyze Images" 
2. When ClassifyObjects module runs, it will show interactive interface
3. Review cell examples and measurements
4. Classify cells into bins:
   - Negative (low expression)
   - Positive (medium expression) 
   - High (high expression)

STEP 3: TRAIN CLASSIFIER
1. Classify 50-100 example cells per category
2. Use biological knowledge to guide decisions
3. Focus on clear positive/negative examples first
4. Add ambiguous cases later for refinement

STEP 4: APPLY TO ALL CELLS
1. After training, apply classifier to all cells
2. Review classification results
3. Export results as CSV

STEP 5: EXPORT RESULTS
1. File → Export → Export to Database
2. OR use Data Tools → ExportToSpreadsheet
3. Save phenotype classifications

🎯 CLASSIFICATION STRATEGY:
Based on your channels: {list(config['channel_mapping'].values())}

Suggested bins per channel:
- DAPI: Nucleated vs Non-nucleated
- CD45: Immune vs Non-immune  
- CD3: T cells vs Non-T cells
- CD8B: CD8+ vs CD8- T cells
- TOM: Tumor vs Non-tumor
- KI67: Proliferating vs Non-proliferating

💡 TIPS:
- Start with obvious examples
- Use scatter plots to visualize distributions
- Validate on known positive controls
- Save intermediate results frequently

READY TO START CLASSIFICATION!
"""
    
    with open(cp_dir / "CELLPROFILER_INSTRUCTIONS.txt", 'w') as f:
        f.write(instructions)
    
    print(f"\n🎉 CELLPROFILER PROJECTS READY!")
    print(f"📁 Projects: cellprofiler_projects/")
    print(f"📖 Instructions: cellprofiler_projects/CELLPROFILER_INSTRUCTIONS.txt")
    print(f"🚀 Launch CellProfiler and open .cppipe files")

if __name__ == "__main__":
    setup_cellprofiler_projects()