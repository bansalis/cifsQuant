#!/usr/bin/env python3
"""
Batch Data Loader for Multi-slide Spatial Analysis
Loads all slides into integrated SCIMAP/AnnData object with batch tracking
"""

import os
import sys
import yaml
import pandas as pd
import anndata as ad
import scimap as sm
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse

def load_config(config_path):
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_slide_files(data_path):
    """Find all combined_quantification.csv files"""
    data_dir = Path(data_path)
    slide_files = list(data_dir.glob("*/final/combined_quantification.csv"))
    
    print(f"Found {len(slide_files)} slide files:")
    for f in slide_files:
        slide_id = f.parent.parent.name
        print(f"  - {slide_id}: {f}")
    
    return slide_files

def load_single_slide(csv_path, slide_id, config):
    """Load single slide and add metadata"""
    print(f"Loading slide: {slide_id}")
    
    # Load with SCIMAP
    try:
        adata = sm.pp.mcmicro_to_scimap(str(csv_path))
        
        # Add slide ID
        adata.obs['slide_id'] = slide_id
        
        # Add metadata from config if available
        slide_info = next((s for s in config['slides'] if s['slide_id'] == slide_id), None)
        if slide_info:
            for key, value in slide_info.items():
                if key != 'slide_id':
                    adata.obs[key] = value
        else:
            print(f"  Warning: No metadata found for {slide_id}")
            adata.obs['treatment'] = 'unknown'
            adata.obs['timepoint'] = -1
            adata.obs['immunogenic'] = False
        
        # Add cell counts for QC
        n_cells = adata.n_obs
        print(f"  Loaded {n_cells} cells")
        
        return adata, n_cells
        
    except Exception as e:
        print(f"  Error loading {slide_id}: {e}")
        return None, 0

def integrate_slides(adata_list, config):
    """Concatenate all slides with batch correction setup"""
    print("\nIntegrating slides...")
    
    # Concatenate with batch key
    adata_integrated = ad.concat(
        adata_list, 
        label='slide_id',
        keys=[adata.obs['slide_id'].iloc[0] for adata in adata_list],
        index_unique='-'
    )
    
    # Copy metadata to integrated object
    slide_metadata = []
    for adata in adata_list:
        slide_meta = adata.obs.iloc[0].to_dict()
        slide_metadata.append(slide_meta)
    
    # Create slide-level summary
    slide_summary = pd.DataFrame(slide_metadata)
    adata_integrated.uns['slide_summary'] = slide_summary
    
    print(f"Integrated dataset: {adata_integrated.n_obs} cells across {len(adata_list)} slides")
    print(f"Channels: {list(adata_integrated.var_names)}")
    
    return adata_integrated

def quality_control(adata, config):
    """Basic quality control and filtering"""
    print("\nRunning quality control...")
    
    initial_cells = adata.n_obs
    min_cells = config['analysis']['min_cells_per_slide']
    
    # Check cells per slide
    slide_counts = adata.obs['slide_id'].value_counts()
    print("Cells per slide:")
    for slide, count in slide_counts.items():
        status = "PASS" if count >= min_cells else "FAIL"
        print(f"  {slide}: {count} cells [{status}]")
    
    # Filter slides with too few cells
    valid_slides = slide_counts[slide_counts >= min_cells].index
    if len(valid_slides) < len(slide_counts):
        print(f"Filtering out {len(slide_counts) - len(valid_slides)} slides with <{min_cells} cells")
        adata = adata[adata.obs['slide_id'].isin(valid_slides)].copy()
    
    # Basic stats
    print(f"\nFinal dataset: {adata.n_obs} cells ({initial_cells - adata.n_obs} filtered)")
    print(f"Variables: {adata.n_vars}")
    
    # Add QC metrics to uns
    adata.uns['qc_metrics'] = {
        'initial_cells': initial_cells,
        'final_cells': adata.n_obs,
        'filtered_cells': initial_cells - adata.n_obs,
        'min_cells_threshold': min_cells,
        'slides_passed': len(valid_slides),
        'slides_total': len(slide_counts)
    }
    
    return adata

def save_integrated_data(adata, output_path, experiment_name):
    """Save integrated dataset"""
    output_dir = Path(output_path) / "integrated_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save AnnData object
    h5ad_path = output_dir / f"{experiment_name}_integrated.h5ad"
    adata.write(h5ad_path)
    print(f"Saved integrated data: {h5ad_path}")
    
    # Save slide summary
    summary_path = output_dir / f"{experiment_name}_slide_summary.csv"
    adata.uns['slide_summary'].to_csv(summary_path, index=False)
    print(f"Saved slide summary: {summary_path}")
    
    # Save QC report
    qc_path = output_dir / f"{experiment_name}_qc_report.txt"
    with open(qc_path, 'w') as f:
        f.write(f"Quality Control Report - {experiment_name}\n")
        f.write("=" * 50 + "\n\n")
        
        qc = adata.uns['qc_metrics']
        f.write(f"Initial cells: {qc['initial_cells']}\n")
        f.write(f"Final cells: {qc['final_cells']}\n") 
        f.write(f"Filtered cells: {qc['filtered_cells']}\n")
        f.write(f"Slides passed QC: {qc['slides_passed']}/{qc['slides_total']}\n\n")
        
        f.write("Slide Summary:\n")
        f.write("-" * 20 + "\n")
        slide_counts = adata.obs['slide_id'].value_counts()
        for slide, count in slide_counts.items():
            f.write(f"{slide}: {count} cells\n")
    
    print(f"Saved QC report: {qc_path}")
    
    return h5ad_path

def main():
    parser = argparse.ArgumentParser(description='Batch load slides for spatial analysis')
    parser.add_argument('--config', default='/app/configs/experiment_config.yaml',
                       help='Path to experiment config file')
    parser.add_argument('--output', default='/app/outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config for experiment: {config['experiment_name']}")
    
    # Find slide files
    slide_files = find_slide_files(config['data_path'])
    if not slide_files:
        print("No slide files found!")
        return 1
    
    # Load individual slides
    adata_list = []
    total_cells = 0
    
    for csv_path in tqdm(slide_files, desc="Loading slides"):
        slide_id = csv_path.parent.parent.name
        adata, n_cells = load_single_slide(csv_path, slide_id, config)
        
        if adata is not None:
            adata_list.append(adata)
            total_cells += n_cells
        else:
            print(f"Skipping {slide_id} due to loading error")
    
    if not adata_list:
        print("No slides loaded successfully!")
        return 1
    
    print(f"\nSuccessfully loaded {len(adata_list)} slides with {total_cells} total cells")
    
    # Integrate slides
    adata_integrated = integrate_slides(adata_list, config)
    
    # Quality control
    adata_integrated = quality_control(adata_integrated, config)
    
    # Save results
    output_path = save_integrated_data(
        adata_integrated, 
        args.output, 
        config['experiment_name']
    )
    
    print(f"\n✅ Batch loading complete!")
    print(f"Next step: Run phenotyping on {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())