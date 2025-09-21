#!/usr/bin/env python3
"""
Automated Minerva Story Setup for Spatial IF Analysis
Converts MCMICRO results to Minerva-compatible format with phenotype overlays
"""

import json
import pandas as pd
import numpy as np
import tifffile
from pathlib import Path
import argparse
import yaml

def create_minerva_story_file(results_dir, sample_name, channel_names, phenotype_file=None):
    """Create Minerva Author story file that can be opened directly"""
    
    # Load quantification data for waypoint positioning
    phenotype_path = Path(results_dir) / "phenotype_analysis" / "phenotyped_cells.csv"
    if phenotype_path.exists():
        df = pd.read_csv(phenotype_path)
    else:
        quant_path = Path(results_dir) / "final" / "combined_quantification.csv"
        df = pd.read_csv(quant_path)
    
    # Create channel names CSV file for Minerva
    minerva_dir = Path(results_dir) / "minerva_setup"
    minerva_dir.mkdir(exist_ok=True)
    
    channel_csv_path = minerva_dir / "channel_names.csv"
    with open(channel_csv_path, 'w') as f:
        f.write("marker_name\n")
        for name in channel_names:
            f.write(f"{name}\n")
    
    # Raw image path (absolute path for Minerva Author)
    raw_image = Path("rawdata") / f"{sample_name}_aligned_stack.ome.tif"
    
    # Tumor mask path
    tumor_mask = Path("tumor_structures/data/tumor_sections_mask.tif")
    
    # Create correct Minerva Author story format
    story = {
        "in_file": str(raw_image.resolve()),
        "csv_file": str(channel_csv_path.resolve()),
        "sample_info": {
            "name": sample_name,
            "description": f"Spatial analysis of {sample_name}"
        },
        "mask_paths": [str(tumor_mask.resolve())] if tumor_mask.exists() else [],
        "groups": [],
        "waypoints": []
    }
    
    # Add channel groups - correct format with channel objects
    all_channels = []
    for i, name in enumerate(channel_names):
        all_channels.append({
            "id": i,
            "label": name,
            "color": [1.0, 1.0, 1.0],  # Default white
            "min": 0.0,
            "max": 65535.0,
            "visible": True
        })
    
    story["groups"].append({
        "label": "All Channels",
        "channels": all_channels,
        "colors": [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]]
    })
    
    # Add DAPI group
    dapi_channels = []
    for i, name in enumerate(channel_names):
        if 'DAPI' in name.upper():
            dapi_channels.append({
                "id": i,
                "label": name,
                "color": [0.0, 0.0, 1.0],  # Blue
                "min": 0.0,
                "max": 65535.0,
                "visible": True
            })
    
    if dapi_channels:
        story["groups"].append({
            "label": "Nuclei (DAPI)",
            "channels": dapi_channels,
            "colors": [[0,0,1]]
        })
    
    # Add tumor marker group  
    tumor_channels = []
    for i, name in enumerate(channel_names):
        if 'TOM' in name.upper():
            tumor_channels.append({
                "id": i,
                "label": name,
                "color": [1.0, 0.0, 0.0],  # Red
                "min": 0.0,
                "max": 65535.0,
                "visible": True
            })
    
    if tumor_channels:
        story["groups"].append({
            "label": "Tumor Markers",
            "channels": tumor_channels,
            "colors": [[1,0,0]]
        })
    
    # Add immune marker group
    immune_channels = []
    for i, name in enumerate(channel_names):
        if any(marker in name.upper() for marker in ['CD3', 'CD8', 'CD45', 'CD68']):
            immune_channels.append({
                "id": i,
                "label": name,
                "color": [0.0, 1.0, 0.0],  # Green
                "min": 0.0,
                "max": 65535.0,
                "visible": True
            })
    
    if immune_channels:
        story["groups"].append({
            "label": "Immune Markers", 
            "channels": immune_channels,
            "colors": [[0,1,0]]
        })
    
    # Create waypoints based on data
    if len(df) > 0 and 'X_centroid' in df.columns:
        image_width = int(df['X_centroid'].max())
        image_height = int(df['Y_centroid'].max())
        
        story["waypoints"] = [
            {
                "name": "Overview",
                "description": f"Complete tissue overview of {sample_name}",
                "group": "All Channels",
                "zoom": 0.1,
                "pan": [image_width // 2, image_height // 2]
            }
        ]
        
        # Add tumor waypoint if tumor cells exist
        phenotype_col = 'consensus_phenotype' if 'consensus_phenotype' in df.columns else 'hierarchical_phenotype'
        if phenotype_col in df.columns:
            tumor_cells = df[df[phenotype_col].str.contains('Tumor', na=False, case=False)]
            if len(tumor_cells) > 0:
                story["waypoints"].append({
                    "name": "Tumor Regions",
                    "description": "Focus on tumor cell clusters with mask overlay",
                    "group": "Tumor Markers" if tumor_channels else "All Channels",
                    "zoom": 2.0,
                    "pan": [int(tumor_cells['X_centroid'].mean()), int(tumor_cells['Y_centroid'].mean())]
                })
    else:
        # Default waypoints if no cell data
        story["waypoints"] = [
            {
                "name": "Overview",
                "description": f"Tissue overview of {sample_name}",
                "group": "All Channels",
                "zoom": 0.1,
                "pan": [25000, 25000]
            }
        ]
    
    return story

def load_phenotype_definitions(phenotype_file=None):
    """Load phenotype definitions from CSV file"""
    if phenotype_file and Path(phenotype_file).exists():
        df = pd.read_csv(phenotype_file)
        phenotype_dict = {}
        for _, row in df.iterrows():
            phenotype_dict[row['phenotype_name']] = {
                'markers': row['positive_markers'].split(';') if pd.notna(row['positive_markers']) else [],
                'negative_markers': row['negative_markers'].split(';') if pd.notna(row.get('negative_markers', '')) else [],
                'color': row.get('color', '#FF0000') if 'color' in df.columns else '#FF0000'
            }
        return phenotype_dict
    
    # No fallback - require phenotype file
    print("ERROR: No phenotype definitions file provided. Please create phenotype_definitions.csv")
    return {}

def generate_phenotype_overlays(results_dir, phenotype_file=None):
    """Generate colored overlay masks for each defined phenotype"""
    
    quant_path = Path(results_dir) / "final" / "combined_quantification.csv"
    mask_path = Path(results_dir) / "final" / "full_segmentation_mask.tif"
    
    df = pd.read_csv(quant_path)
    mask = tifffile.imread(mask_path)
    
    overlay_dir = Path(results_dir) / "phenotype_masks"
    overlay_dir.mkdir(exist_ok=True)
    
    if 'phenotype' not in df.columns:
        print("No phenotype column found - run phenotyping first")
        return
    
    # Load phenotype definitions
    phenotype_defs = load_phenotype_definitions(phenotype_file)
    
    phenotypes = df['phenotype'].unique()
    colors = {name: int(defs['color'].replace('#', ''), 16) >> 16 
             for name, defs in phenotype_defs.items()}
    
    for phenotype in phenotypes:
        if phenotype == 'Unknown':
            continue
            
        # Create phenotype-specific overlay
        phenotype_overlay = np.zeros_like(mask, dtype=np.uint8)
        
        # Use detected phenotype column
        phenotype_cells = df[df[phenotype_col] == phenotype]['CellID'].values if 'CellID' in df.columns else df[df[phenotype_col] == phenotype].index.values
        
        for cell_id in phenotype_cells:
            phenotype_overlay[mask == cell_id] = colors.get(phenotype, 128)
        
        # Save overlay
        overlay_path = overlay_dir / f"{phenotype}_overlay.tif"
        tifffile.imwrite(overlay_path, phenotype_overlay, compression='lzw')
        print(f"Created overlay: {overlay_path}")

def setup_minerva_docker(results_dir, sample_name, port=3000):
    """Setup Minerva using the correct minerva-author container"""
    
    minerva_dir = Path(results_dir) / "minerva_setup"
    minerva_dir.mkdir(exist_ok=True)
    
    # Create proper Minerva Docker setup
    compose_config = f"""
version: '3.8'
services:
  minerva:
    image: labsyspharm/minerva-author:latest
    ports:
      - "{port}:2020"
    volumes:
      - ./:/app/data
      - ../../../rawdata:/app/images
      - ../final:/app/final
      - ../phenotype_masks:/app/masks
    working_dir: /app
    command: >
      sh -c "pip install minerva-author && 
             minerva-author serve data/story.json --port 2020 --host 0.0.0.0"
"""
    
    with open(minerva_dir / "docker-compose.yml", 'w') as f:
        f.write(compose_config)
    
    # Create startup script
    startup_script = f"""#!/bin/bash
cd {minerva_dir}
echo "Starting Minerva Author for {sample_name}..."
echo "This may take a few minutes on first run (installing minerva-author)..."
echo ""
docker-compose up
echo ""
echo "Access Minerva at: http://localhost:{port}"
echo "Use Ctrl+C to stop, then 'docker-compose down' to cleanup"
"""
    
    script_path = minerva_dir / "start_minerva.sh"
    with open(script_path, 'w') as f:
        f.write(startup_script)
    script_path.chmod(0o755)
    
    # Alternative: Create pip install script for local Minerva
    local_script = f"""#!/bin/bash
echo "Installing Minerva Author locally..."
pip install minerva-author

echo "Starting local Minerva server..."
cd {minerva_dir}
minerva-author serve story.json --port {port} --host 0.0.0.0 &

echo "Minerva running at: http://localhost:{port}"
echo "Use 'pkill -f minerva-author' to stop"
"""
    
    local_path = minerva_dir / "start_local_minerva.sh"
    with open(local_path, 'w') as f:
        f.write(local_script)
    local_path.chmod(0o755)
    
    return script_path

def main():
    parser = argparse.ArgumentParser(description="Setup Minerva viewer for spatial IF data")
    parser.add_argument("--results-dir", required=True, help="Results directory path")
    parser.add_argument("--sample-name", required=True, help="Sample name")
    parser.add_argument("--channels", required=True, help="Channel names file")
    parser.add_argument("--phenotypes", required=True, help="Phenotype definitions CSV file")
    parser.add_argument("--port", type=int, default=3000, help="Minerva port")
    
    args = parser.parse_args()
    
    # Load channel names
    with open(args.channels, 'r') as f:
        channel_names = [line.strip() for line in f.readlines()]
    
    results_path = Path(args.results_dir)
    
    # Generate phenotype overlays
    print("Generating phenotype overlays...")
    generate_phenotype_overlays(results_path, args.phenotypes)
    
    # Create Minerva story file
    print("Creating Minerva Author story file...")
    story = create_minerva_story_file(results_path, args.sample_name, channel_names, args.phenotypes)
    
    # Save story file
    minerva_dir = results_path / "minerva_setup"
    minerva_dir.mkdir(exist_ok=True)
    
    story_path = minerva_dir / f"{args.sample_name}_story.json"
    with open(story_path, 'w') as f:
        json.dump(story, f, indent=2)
    
    print(f"\nMinerva Author story file created: {story_path}")
    print(f"\nTo use:")
    print(f"1. Download Minerva Author from: https://www.minerva.im/download.html")
    print(f"2. Run minerva_author.exe")
    print(f"3. In Minerva Author, click 'Load Story' and select: {story_path}")
    print(f"4. Adjust channels/overlays as needed")
    print(f"5. Click 'Publish' to create interactive viewer")

if __name__ == "__main__":
    main()