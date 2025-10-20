#!/usr/bin/env python3
"""
Streamlined Gating with Rosin Method + Comprehensive Visualization
"""

import pandas as pd
import numpy as np
import scimap as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

CONFIG = {
    'markers': {
        'Channel_2': 'TOM', 
        'Channel_3': 'CD45',
        'Channel_4': 'AGFP',
        'Channel_6': 'PERK',
        'Channel_7': 'CD8B',
        'Channel_8': 'KI67',
        'Channel_10': 'CD3'
    },
    'phenotypes': {
        'T_cells': {'CD3': 'pos'},
        'CD8_T_cells': {'CD3': 'pos', 'CD8B': 'pos'},
        'Immune': {'CD45': 'pos'},
        'Proliferating': {'KI67': 'pos'},
        'Proliferating_T': {'CD3': 'pos', 'KI67': 'pos'},
        'Stressed_T': {'CD3': 'pos', 'PERK': 'pos'},
        'CD8_nonImmune': {'CD8B': 'pos', 'CD45': 'neg'}  # Example with negative
    },
    'sample_metadata': {
        'GUEST29': {'age_weeks': 8, 'genotype': 'cis', 'treatment': 'Control'},
        'GUEST30': {'age_weeks': 8, 'genotype': 'trans', 'treatment': 'Control'}
    },
    'gate_multipliers': {
        'AGFP': 1.8,      # More conservative (fewer positives)
        'PERK': 0.7,      # More liberal (more positives)
    },
    'gate_stringency': 1.5,  # Global multiplier for all gates (higher = more conservative)
}

def load_data(results_dir):
    all_data = []
    for sample_dir in Path(results_dir).iterdir():
        if not sample_dir.is_dir():
            continue
            
        sample_name = sample_dir.name
        quant_file = sample_dir / "final" / "combined_quantification.csv"
        
        if not quant_file.exists():
            continue
            
        df = pd.read_csv(quant_file)
        df = df.rename(columns=CONFIG['markers'])
        df['sample_id'] = sample_name
        
        if sample_name in CONFIG['sample_metadata']:
            for key, value in CONFIG['sample_metadata'][sample_name].items():
                df[key] = value
                
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def rosin_gate(data, stringency=1.0):
    """Rosin unimodal thresholding with stringency adjustment"""
    positive_vals = data[data > 0]
    if len(positive_vals) < 100:
        return np.percentile(data, 95) * stringency
    
    counts, bins = np.histogram(positive_vals, bins=256)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    peak_idx = np.argmax(counts)
    last_idx = len(counts) - 1
    while last_idx > peak_idx and counts[last_idx] < counts[peak_idx] * 0.01:
        last_idx -= 1
    
    if last_idx <= peak_idx:
        return np.percentile(positive_vals, 90) * stringency
    
    x1, y1 = peak_idx, counts[peak_idx]
    x2, y2 = last_idx, counts[last_idx]
    
    distances = []
    for i in range(peak_idx, last_idx):
        d = abs((y2-y1)*i - (x2-x1)*counts[i] + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
        distances.append(d)
    
    if distances:
        threshold_idx = peak_idx + np.argmax(distances)
        return bin_centers[threshold_idx] * stringency
    
    return np.percentile(positive_vals, 90) * stringency

def create_gating_visualizations(adata_raw, gates, output_dir):
    """Create comprehensive gating plots"""
    
    plots_dir = Path(output_dir) / "gating_analysis"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Raw distributions with gates
    for sample in adata_raw.obs['sample_id'].unique():
        sample_mask = adata_raw.obs['sample_id'] == sample
        
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        axes = axes.flatten()
        fig.suptitle(f'{sample} - Raw Intensity Distributions with Rosin Gates', fontsize=16)
        
        for i, marker in enumerate(adata_raw.var.index):
            marker_idx = adata_raw.var.index.get_loc(marker)
            raw_vals = adata_raw.X[sample_mask, marker_idx]
            gate = gates[sample][marker]
            
            axes[i].hist(raw_vals[raw_vals > 0], bins=100, alpha=0.7, color='steelblue')
            axes[i].axvline(gate, color='red', linestyle='--', linewidth=2, label=f'Gate: {gate:.0f}')
            
            pos_pct = (raw_vals > gate).mean() * 100
            axes[i].set_title(f'{marker}\n{pos_pct:.1f}% positive')
            axes[i].set_xlabel('Raw Intensity')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{sample}_raw_distributions.png', dpi=200)
        plt.close()
    
    # 2. FACS-style 2D plots
    facs_pairs = [
        ('CD3', 'CD8B'),
        ('CD45', 'CD8B'),
        ('TOM', 'AGFP'),
        ('TOM', 'PERK')
    ]
    
    for sample in adata_raw.obs['sample_id'].unique():
        sample_mask = adata_raw.obs['sample_id'] == sample
        sample_data = adata_raw[sample_mask]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        axes = axes.flatten()
        fig.suptitle(f'{sample} - FACS-style Gating', fontsize=16)
        
        for i, (marker_x, marker_y) in enumerate(facs_pairs):
            x_idx = adata_raw.var.index.get_loc(marker_x)
            y_idx = adata_raw.var.index.get_loc(marker_y)
            
            x_vals = sample_data.X[:, x_idx]
            y_vals = sample_data.X[:, y_idx]
            
            gate_x = gates[sample][marker_x]
            gate_y = gates[sample][marker_y]
            
            # Subsample for plotting
            if len(x_vals) > 10000:
                idx = np.random.choice(len(x_vals), 10000, replace=False)
                x_vals = x_vals[idx]
                y_vals = y_vals[idx]
            
            axes[i].scatter(x_vals, y_vals, s=1, alpha=0.3, c='gray')
            axes[i].axvline(gate_x, color='red', linestyle='--', linewidth=1)
            axes[i].axhline(gate_y, color='red', linestyle='--', linewidth=1)
            
            # Quadrant percentages
            q1 = ((x_vals > gate_x) & (y_vals > gate_y)).mean() * 100
            q2 = ((x_vals <= gate_x) & (y_vals > gate_y)).mean() * 100
            q3 = ((x_vals <= gate_x) & (y_vals <= gate_y)).mean() * 100
            q4 = ((x_vals > gate_x) & (y_vals <= gate_y)).mean() * 100
            
            axes[i].text(0.95, 0.95, f'{q1:.1f}%', transform=axes[i].transAxes, 
                        ha='right', va='top', fontsize=10, color='red')
            axes[i].text(0.05, 0.95, f'{q2:.1f}%', transform=axes[i].transAxes, 
                        ha='left', va='top', fontsize=10, color='red')
            axes[i].text(0.05, 0.05, f'{q3:.1f}%', transform=axes[i].transAxes, 
                        ha='left', va='bottom', fontsize=10, color='red')
            axes[i].text(0.95, 0.05, f'{q4:.1f}%', transform=axes[i].transAxes, 
                        ha='right', va='bottom', fontsize=10, color='red')
            
            axes[i].set_xlabel(f'{marker_x} (raw intensity)')
            axes[i].set_ylabel(f'{marker_y} (raw intensity)')
            axes[i].set_title(f'{marker_x} vs {marker_y}')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{sample}_facs_plots.png', dpi=200)
        plt.close()
    
    # 3. Spatial distributions per marker
    for marker in adata_raw.var.index:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f'{marker} Spatial Distribution', fontsize=16)
        
        for i, sample in enumerate(adata_raw.obs['sample_id'].unique()):
            sample_mask = adata_raw.obs['sample_id'] == sample
            sample_data = adata_raw[sample_mask]
            
            marker_idx = adata_raw.var.index.get_loc(marker)
            vals = sample_data.X[:, marker_idx]
            gate = gates[sample][marker]
            
            pos_mask = vals > gate
            
            # Plot all cells gray, positive cells red
            axes[i].scatter(sample_data.obs['X_centroid'], sample_data.obs['Y_centroid'], 
                           s=0.5, alpha=0.2, c='lightgray')
            axes[i].scatter(sample_data.obs.loc[pos_mask, 'X_centroid'],
                           sample_data.obs.loc[pos_mask, 'Y_centroid'],
                           s=0.5, alpha=0.8, c='red')
            
            axes[i].set_title(f'{sample}\n{pos_mask.sum():,} positive ({pos_mask.mean()*100:.1f}%)')
            axes[i].set_xlabel('X (pixels)')
            axes[i].set_ylabel('Y (pixels)')
            axes[i].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{marker}_spatial.png', dpi=200)
        plt.close()
    
    # 4. Zoomed TOM regions
    for sample in adata_raw.obs['sample_id'].unique():
        sample_mask = adata_raw.obs['sample_id'] == sample
        sample_data = adata_raw[sample_mask].copy()
        
        tom_idx = adata_raw.var.index.get_loc('TOM')
        tom_vals = sample_data.X[:, tom_idx]
        tom_gate = gates[sample]['TOM']
        tom_pos = tom_vals > tom_gate
        
        if tom_pos.sum() == 0:
            continue
        
        # Find TOM+ regions
        tom_x = sample_data.obs.loc[tom_pos, 'X_centroid']
        tom_y = sample_data.obs.loc[tom_pos, 'Y_centroid']
        x_min, x_max = tom_x.min() - 500, tom_x.max() + 500
        y_min, y_max = tom_y.min() - 500, tom_y.max() + 500
        
        # Filter cells in region using integer indexing
        in_region = ((sample_data.obs['X_centroid'] >= x_min) & 
                    (sample_data.obs['X_centroid'] <= x_max) &
                    (sample_data.obs['Y_centroid'] >= y_min) & 
                    (sample_data.obs['Y_centroid'] <= y_max)).values
        
        region_data = sample_data[in_region, :]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{sample} - TOM+ Region Details', fontsize=16)
        
        markers_to_show = ['TOM', 'AGFP', 'CD3', 'PERK', 'KI67', 'CD45']
        
        for i, marker in enumerate(markers_to_show):
            marker_idx = adata_raw.var.index.get_loc(marker)
            vals = region_data.X[:, marker_idx]
            gate = gates[sample][marker]
            pos = vals > gate
            
            axes.flatten()[i].scatter(region_data.obs['X_centroid'], 
                                    region_data.obs['Y_centroid'],
                                    s=2, alpha=0.3, c='lightgray')
            axes.flatten()[i].scatter(region_data.obs.loc[pos, 'X_centroid'],
                                    region_data.obs.loc[pos, 'Y_centroid'],
                                    s=2, alpha=0.8, c='red')
            
            axes.flatten()[i].set_title(f'{marker} ({pos.sum():,} pos, {pos.mean()*100:.1f}%)')
            axes.flatten()[i].set_xlim(x_min, x_max)
            axes.flatten()[i].set_ylim(y_min, y_max)
            axes.flatten()[i].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{sample}_TOM_region_zoom.png', dpi=200)
        plt.close()

def export_to_fcs(adata_raw, output_dir):
    """
    Export raw intensity data to FCS format for FlowJo analysis.
    Uses fcswrite library.
    """
    try:
        import fcswrite
    except ImportError:
        print("Installing fcswrite...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'fcswrite'])
        import fcswrite
    
    fcs_dir = Path(output_dir) / "fcs_files"
    fcs_dir.mkdir(exist_ok=True, parents=True)
    
    for sample in adata_raw.obs['sample_id'].unique():
        sample_mask = adata_raw.obs['sample_id'] == sample
        sample_data = adata_raw[sample_mask]
        
        # Extract marker data
        marker_data = sample_data.X
        marker_names = list(adata_raw.var.index)
        
        # Add spatial coordinates as pseudo-channels (FlowJo can use these)
        x_coords = sample_data.obs['X_centroid'].values.reshape(-1, 1)
        y_coords = sample_data.obs['Y_centroid'].values.reshape(-1, 1)
        
        # Combine: markers + X + Y
        fcs_data = np.hstack([marker_data, x_coords, y_coords])
        channel_names = marker_names + ['X_centroid', 'Y_centroid']
        
        # Create FCS file
        fcs_path = fcs_dir / f"{sample}.fcs"
        
        fcswrite.write_fcs(
            filename=str(fcs_path),
            chn_names=channel_names,
            data=fcs_data,
            compat_chn_names=False,  # Keep original names
            compat_copy=True,
            compat_negative=True,
            compat_percent=True,
            compat_max_int16=10000  # Scale for display
        )
        
        print(f"  {sample}: {fcs_path.name} ({len(sample_data):,} events)")
    
    print(f"\nFCS files saved to: {fcs_dir}")
    print("Import into FlowJo for manual gating and analysis")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--output_dir', default='scimap_analysis')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load
    df = load_data(args.results_dir)
    marker_cols = list(CONFIG['markers'].values())
    
    import anndata
    adata_raw = anndata.AnnData(X=df[marker_cols].values)
    adata_raw.var.index = marker_cols
    adata_raw.obs = df.drop(columns=marker_cols).reset_index(drop=True)
    
    # Gate with Rosin
    print("\nRosin Gating:")
    gates = {}
    adata_gated = adata_raw.copy()
    
    for sample in adata_raw.obs['sample_id'].unique():
        gates[sample] = {}
        sample_mask = adata_raw.obs['sample_id'] == sample
        
        for marker in adata_raw.var.index:
            marker_idx = adata_raw.var.index.get_loc(marker)
            raw_vals = adata_raw.X[sample_mask, marker_idx].copy()
            
            gate = rosin_gate(raw_vals, stringency=CONFIG.get('gate_stringency', 1.0))
            
            # Apply manual adjustment if specified
            if marker in CONFIG.get('gate_multipliers', {}):
                gate *= CONFIG['gate_multipliers'][marker]
            
            gates[sample][marker] = gate
            
            adata_gated.X[sample_mask, marker_idx] = (raw_vals > gate).astype(float)
            
            pos_pct = (raw_vals > gate).mean() * 100
            print(f"{sample} {marker}: {gate:.0f} → {pos_pct:.1f}%")
    
    # Export to FCS for FlowJo
    print("\nExporting to FCS format for FlowJo:")
    export_to_fcs(adata_raw, output_dir)

    # Visualize
    create_gating_visualizations(adata_raw, gates, output_dir)
    
    # Phenotype - convert CONFIG to SCIMAP format
    phenotype_rows = []
    all_markers = list(CONFIG['markers'].values())

    for pheno_name, marker_gates in CONFIG['phenotypes'].items():
        row = {'phenotype': pheno_name}
        
        for marker in all_markers:
            row[marker] = marker_gates.get(marker, '')
        
        phenotype_rows.append(row)

    phenotype_df = pd.DataFrame(phenotype_rows)

    adata_gated = sm.tl.phenotype_cells(adata_gated, 
                                        phenotype=phenotype_df,
                                        gate=0.5)

    # Save
    adata_gated.write(output_dir / 'rosin_gated.h5ad')
    adata_gated.obs.to_csv(output_dir / 'rosin_phenotyped.csv', index=False)
    
    print(f"\n✅ Complete! Plots in: {output_dir}/gating_analysis/")

if __name__ == '__main__':
    main()