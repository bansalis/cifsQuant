#!/usr/bin/env python3
"""
Configurable Manual Phenotyping Workflow
Reads all settings from phenotyping_config.yaml
"""

import pandas as pd
import numpy as np
import yaml
import tifffile
from pathlib import Path
import argparse
import os

def load_config(config_path="phenotyping_config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_reduced_image(raw_image_path, output_dir, sample_name, pyramid_level=1):
    """Create reduced resolution image for CyLinter if needed"""
    
    print(f"Creating reduced image from pyramid level {pyramid_level}...")
    
    with tifffile.TiffFile(raw_image_path) as tif:
        if len(tif.series) > pyramid_level:
            image = tif.series[pyramid_level].asarray()
        else:
            # If pyramid level doesn't exist, downsample level 0
            image = tif.series[0].asarray()
            # Simple downsampling by factor of 2
            if len(image.shape) == 3:
                image = image[:, ::2, ::2]
    
    # Save reduced image
    reduced_path = Path(output_dir) / f"{sample_name}_level{pyramid_level}.ome.tif"
    tifffile.imwrite(str(reduced_path), image, photometric='minisblack')
    
    print(f"Reduced image saved: {reduced_path}")
    return str(reduced_path)

def convert_to_cylinter_format(input_csv, config, output_dir, sample_name):
    """Convert nextflow output using config settings"""
    
    df = pd.read_csv(input_csv)
    
    # Apply channel mapping from config
    channel_mapping = config['channel_mapping']
    df = df.rename(columns=channel_mapping)
    
    # Add required CyLinter columns
    df['Sample'] = sample_name
    df['ROI'] = 1
    
    # Create CyLinter directory
    cylinter_dir = Path(output_dir) / "cylinter_data"
    cylinter_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in CyLinter format
    output_path = cylinter_dir / f"{sample_name}_quantification.csv"
    df.to_csv(output_path, index=False)
    
    print(f"CyLinter data saved: {output_path}")
    return str(output_path)

def create_cylinter_config(config, output_dir, sample_name, image_path=None):
    """Create CyLinter config from YAML settings"""
    
    config_dir = Path(output_dir) / "cylinter_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract channel names
    channels = list(config['channel_mapping'].values())
    
    cylinter_config = f"""
data_dir: {Path(output_dir) / "cylinter_data"}
sample_name: {sample_name}
channels: {channels}
image_path: {image_path if image_path else "null"}
min_cells: {config['cylinter_settings']['min_cells']}
max_cells: {config['cylinter_settings']['max_cells']}
sample_frac: {config['cylinter_settings']['sample_frac']}
random_state: {config['cylinter_settings']['random_state']}
"""
    
    config_path = config_dir / "config.yml"
    with open(config_path, 'w') as f:
        f.write(cylinter_config)
    
    return str(config_path)

def create_scimap_script(config, output_dir):
    """Generate SCIMAP phenotyping script from config"""
    
    # Build gate dictionary
    gate_dict_str = "{\n"
    for channel, threshold in config['initial_gates'].items():
        gate_dict_str += f"    '{channel}': {threshold},\n"
    gate_dict_str += "}"
    
    # Build phenotype dictionary
    phenotype_dict_str = "{\n"
    for phenotype, markers in config['phenotype_definitions'].items():
        markers_str = "[" + ", ".join([f"'{m}'" for m in markers]) + "]"
        phenotype_dict_str += f"    '{phenotype}': {markers_str},\n"
    phenotype_dict_str += "}"
    
    scimap_script = f'''#!/usr/bin/env python3

import scimap as sm
import pandas as pd
import numpy as np
from pathlib import Path

print("Loading data...")
adata = sm.pp.mcmicro_to_scimap("/data/cylinter_data/*.csv")

print("Applying gates...")
gate_dict = {gate_dict_str}

# Apply rescaling and gating
adata = sm.pp.rescale(adata)
adata = sm.tl.gate(adata, gate=gate_dict)

print("Applying phenotypes...")
phenotype_dict = {phenotype_dict_str}

# Apply phenotyping
adata = sm.tl.phenotype_cells(adata, phenotype=phenotype_dict, label='phenotype')

output_dir = Path("/data/scimap_results")
output_dir.mkdir(parents=True, exist_ok=True)

print("Saving results...")
# Save AnnData object
adata.write(str(output_dir / "phenotyped_data.h5ad"))

# Export phenotyped data as CSV with all columns
df_out = adata.obs.copy()
# Add back the raw intensity values
for col in adata.var_names:
    df_out[col + '_intensity'] = adata.X[:, adata.var_names.get_loc(col)]

df_out.to_csv(str(output_dir / "phenotyped_cells.csv"))

# Create phenotype summary
phenotype_counts = df_out['phenotype'].value_counts()
phenotype_counts.to_csv(str(output_dir / "phenotype_counts.csv"))

print("Phenotyping complete!")
print(f"Results saved to: {{str(output_dir)}}")
print(f"Total cells: {{len(df_out)}}")
print(f"Phenotypes found: {{len(phenotype_counts)}}")
'''
    
    # Save SCIMAP script
    scimap_dir = Path(output_dir) / "scimap_results"
    scimap_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = scimap_dir / "run_phenotyping.py"
    with open(script_path, 'w') as f:
        f.write(scimap_script)
    
    os.chmod(script_path, 0o755)
    return str(script_path)

def main():
    parser = argparse.ArgumentParser(description="Configurable Manual Phenotyping")
    parser.add_argument("--sample", required=True, help="Sample name")
    parser.add_argument("--config", default="phenotyping_config.yaml", help="Config file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set paths
    sample_name = args.sample
    input_csv = f"results/{sample_name}/final/combined_quantification.csv"
    raw_image_path = f"rawdata/{sample_name}_aligned_stack.ome.tif"
    output_dir = f"phenotyping_analysis/{sample_name}"
    
    # Check inputs
    if not os.path.exists(input_csv):
        print(f"❌ Input CSV not found: {input_csv}")
        return
    
    print(f"🔬 CONFIGURABLE PHENOTYPING WORKFLOW")
    print(f"Sample: {sample_name}")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Create reduced image if needed
    reduced_image_path = None
    if config['sample_settings']['use_raw_image'] and os.path.exists(raw_image_path):
        reduced_image_path = create_reduced_image(
            raw_image_path, 
            output_dir, 
            sample_name, 
            config['sample_settings']['pyramid_level']
        )
    
    # Step 2: Convert to CyLinter format
    cylinter_path = convert_to_cylinter_format(input_csv, config, output_dir, sample_name)
    
    # Step 3: Create CyLinter config
    config_path = create_cylinter_config(config, output_dir, sample_name, reduced_image_path)
    
    # Step 4: Create SCIMAP script
    scimap_script = create_scimap_script(config, output_dir)
    
    print("\n" + "="*60)
    print("WORKFLOW SETUP COMPLETE")
    print("="*60)
    print("Next steps:")
    print("1. Review/edit phenotyping_config.yaml if needed")
    print("2. Run CyLinter for manual gating")
    print("3. Update gates in config based on CyLinter results") 
    print(f"4. Run: python {scimap_script}")
    print("="*60)

if __name__ == "__main__":
    main()