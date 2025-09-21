#!/usr/bin/env python3
"""
Cell Phenotyping with Multi-method Validation
Supports positive/negative marker combinations with threshold validation
"""

import os
import sys
import yaml
import pandas as pd
import anndata as ad
import scimap as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_otsu
import argparse
from tqdm import tqdm

def load_configs(config_path, phenotype_config_path):
    """Load both experiment and phenotyping configs"""
    with open(config_path, 'r') as f:
        exp_config = yaml.safe_load(f)
    with open(phenotype_config_path, 'r') as f:
        pheno_config = yaml.safe_load(f)
    return exp_config, pheno_config

def load_integrated_data(h5ad_path):
    """Load the integrated dataset"""
    print(f"Loading integrated data: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} channels")
    return adata

def determine_thresholds_otsu(adata, channel):
    """Determine threshold using Otsu method"""
    values = adata[:, channel].X.flatten()
    values = values[values > 0]  # Remove zeros
    if len(values) < 100:
        return None
    try:
        threshold = threshold_otsu(values)
        return float(threshold)
    except:
        return None

def determine_thresholds_gmm(adata, channel, n_components=2):
    """Determine threshold using Gaussian Mixture Model"""
    values = adata[:, channel].X.flatten().reshape(-1, 1)
    values = values[values > 0]  # Remove zeros
    if len(values) < 100:
        return None
    
    try:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(values)
        
        # Find intersection of gaussians as threshold
        means = gmm.means_.flatten()
        if len(means) >= 2:
            threshold = np.mean(sorted(means)[:2])  # Average of two lowest means
            return float(threshold)
    except:
        pass
    return None

def determine_global_thresholds(adata, channels, pheno_config):
    """Determine global thresholds across all slides"""
    print("Determining global thresholds...")
    
    thresholds = {}
    validation_results = {}
    
    for channel in tqdm(channels, desc="Processing channels"):
        if channel not in adata.var_names:
            print(f"Warning: Channel {channel} not found in data")
            continue
            
        methods = {}
        
        # Otsu method
        otsu_thresh = determine_thresholds_otsu(adata, channel)
        if otsu_thresh is not None:
            methods['otsu'] = otsu_thresh
            
        # GMM method  
        gmm_thresh = determine_thresholds_gmm(adata, channel)
        if gmm_thresh is not None:
            methods['gmm'] = gmm_thresh
        
        # Choose consensus threshold
        if len(methods) >= 1:
            values = list(methods.values())
            # Use median if multiple methods agree reasonably well
            cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 1.0
            
            if cv < pheno_config['validation']['max_threshold_cv']:
                threshold = np.median(values)
                status = "CONSENSUS"
            else:
                # Use the more conservative (higher) threshold
                threshold = np.max(values)
                status = "CONSERVATIVE"
                
            thresholds[channel] = threshold
            validation_results[channel] = {
                'methods': methods,
                'final_threshold': threshold,
                'cv': cv,
                'status': status
            }
        else:
            print(f"Warning: Could not determine threshold for {channel}")
            thresholds[channel] = np.percentile(adata[:, channel].X.flatten(), 95)
            validation_results[channel] = {
                'methods': {},
                'final_threshold': thresholds[channel],
                'cv': 1.0,
                'status': "FALLBACK_95TH"
            }
    
    return thresholds, validation_results

def apply_phenotype_rule(adata, rule, thresholds, channel_mapping):
    """Apply a single phenotype rule to data"""
    pos_markers = rule['positive']
    neg_markers = rule['negative']
    
    # Start with all cells as True
    mask = np.ones(adata.n_obs, dtype=bool)
    
    # Apply positive markers (must be above threshold)
    for marker in pos_markers:
        if marker in channel_mapping:
            channel = channel_mapping[marker]
            if channel in adata.var_names and channel in thresholds:
                threshold = thresholds[channel]
                marker_mask = adata[:, channel].X.flatten() > threshold
                mask = mask & marker_mask
    
    # Apply negative markers (must be below threshold)
    for marker in neg_markers:
        if marker in channel_mapping:
            channel = channel_mapping[marker]
            if channel in adata.var_names and channel in thresholds:
                threshold = thresholds[channel]
                marker_mask = adata[:, channel].X.flatten() <= threshold
                mask = mask & marker_mask
    
    return mask

def create_phenotypes(adata, exp_config, pheno_config, thresholds):
    """Create all phenotype classifications"""
    print("Creating phenotype classifications...")
    
    # Map marker names to channels
    channel_mapping = exp_config['channels']
    
    phenotype_results = {}
    phenotype_counts = {}
    
    for pheno_name, rule in tqdm(pheno_config['phenotype_rules'].items(), desc="Applying phenotypes"):
        mask = apply_phenotype_rule(adata, rule, thresholds, channel_mapping)
        
        # Add to adata
        adata.obs[pheno_name] = mask
        
        # Store results
        n_positive = np.sum(mask)
        phenotype_results[pheno_name] = {
            'n_cells': n_positive,
            'percentage': (n_positive / adata.n_obs) * 100,
            'rule': rule
        }
        
        print(f"  {pheno_name}: {n_positive} cells ({phenotype_results[pheno_name]['percentage']:.2f}%)")
    
    return adata, phenotype_results

def plot_threshold_validation(adata, thresholds, validation_results, output_dir):
    """Create validation plots for thresholds"""
    print("Creating threshold validation plots...")
    
    plot_dir = Path(output_dir) / "threshold_validation"
    plot_dir.mkdir(exist_ok=True)
    
    n_channels = len(thresholds)
    n_cols = min(3, n_channels)
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_channels == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (channel, threshold) in enumerate(thresholds.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Plot histogram
        values = adata[:, channel].X.flatten()
        values = values[values > 0]  # Remove zeros
        
        ax.hist(values, bins=100, alpha=0.7, density=True, color='lightblue')
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                  label=f'Threshold: {threshold:.1f}')
        
        # Add method thresholds if available
        if channel in validation_results:
            methods = validation_results[channel]['methods']
            colors = {'otsu': 'green', 'gmm': 'orange'}
            for method, thresh in methods.items():
                if method in colors:
                    ax.axvline(thresh, color=colors[method], linestyle=':', alpha=0.7,
                             label=f'{method.upper()}: {thresh:.1f}')
        
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Density')
        ax.set_title(f'{channel}')
        ax.legend()
        ax.set_yscale('log')
    
    # Remove empty subplots
    for idx in range(n_channels, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.remove()
    
    plt.tight_layout()
    plt.savefig(plot_dir / "threshold_validation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation plots saved to: {plot_dir}")

def save_results(adata, exp_config, thresholds, validation_results, phenotype_results, output_dir):
    """Save all phenotyping results"""
    pheno_dir = Path(output_dir) / "phenotyping"
    pheno_dir.mkdir(exist_ok=True)
    
    experiment_name = exp_config['experiment_name']
    
    # Save updated AnnData
    adata_path = pheno_dir / f"{experiment_name}_phenotyped.h5ad"
    adata.write(adata_path)
    print(f"Saved phenotyped data: {adata_path}")
    
    # Save thresholds
    threshold_df = pd.DataFrame([
        {'channel': ch, 'threshold': th, 'status': validation_results.get(ch, {}).get('status', 'unknown')}
        for ch, th in thresholds.items()
    ])
    threshold_path = pheno_dir / f"{experiment_name}_thresholds.csv"
    threshold_df.to_csv(threshold_path, index=False)
    print(f"Saved thresholds: {threshold_path}")
    
    # Save phenotype summary
    pheno_summary = pd.DataFrame([
        {'phenotype': name, 'n_cells': results['n_cells'], 'percentage': results['percentage']}
        for name, results in phenotype_results.items()
    ])
    summary_path = pheno_dir / f"{experiment_name}_phenotype_summary.csv"
    pheno_summary.to_csv(summary_path, index=False)
    print(f"Saved phenotype summary: {summary_path}")
    
    return adata_path

def main():
    parser = argparse.ArgumentParser(description='Phenotype cells with multi-method validation')
    parser.add_argument('--config', default='/app/spatial_analysis/configs/experiment_config.yaml')
    parser.add_argument('--phenotype-config', default='/app/spatial_analysis/configs/phenotyping_config.yaml')
    parser.add_argument('--input', required=True, help='Path to integrated h5ad file')
    parser.add_argument('--output', default='/app/spatial_analysis/outputs')
    
    args = parser.parse_args()
    
    # Load configs
    exp_config, pheno_config = load_configs(args.config, args.phenotype_config)
    print(f"Processing experiment: {exp_config['experiment_name']}")
    
    # Load data
    adata = load_integrated_data(args.input)
    
    # Get channel list
    channels = list(exp_config['channels'].values())
    
    # Determine thresholds
    thresholds, validation_results = determine_global_thresholds(adata, channels, pheno_config)
    
    # Create phenotypes
    adata, phenotype_results = create_phenotypes(adata, exp_config, pheno_config, thresholds)
    
    # Create validation plots
    if pheno_config['validation']['plot_distributions']:
        plot_threshold_validation(adata, thresholds, validation_results, args.output)
    
    # Save results
    output_path = save_results(adata, exp_config, thresholds, validation_results, 
                              phenotype_results, args.output)
    
    print(f"\n✅ Phenotyping complete!")
    print(f"Phenotyped data saved to: {output_path}")
    print("\nPhenotype summary:")
    for name, results in phenotype_results.items():
        print(f"  {name}: {results['n_cells']} cells ({results['percentage']:.2f}%)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())