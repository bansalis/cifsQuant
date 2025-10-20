#!/usr/bin/env python3
"""
Phenotyping Script - Independent phenotype columns + h5ad for spatial tools
python phenotyping.py --gating_dir manual_gating_output --output_dir phenotyping_output
"""

import pandas as pd
import numpy as np
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PHENOTYPE CONFIGURATION
# ============================================================================

PHENOTYPES = {
    # Basic immune populations
    'T_cells': {'CD3': 'pos'},
    'CD8_T_cells': {'CD3': 'pos', 'CD8B': 'pos'},
    'Immune': {'CD45': 'pos'},
    'Non_immune': {'CD45': 'neg'},
    'Tumor': {'TOM': 'pos', 'CD45': 'neg'},
    'Tumor_AGFP': {'TOM': 'pos', 'AGFP': 'pos', 'CD45': 'neg'},
    'Proliferating_T': {'CD3': 'pos', 'KI67': 'pos'},
    'Proliferating_Tumor': {'TOM': 'pos', 'KI67': 'pos', 'CD45': 'neg'},
    'pERK+_Tumor': {'TOM': 'pos', 'PERK': 'pos', 'CD45': 'neg'},
    'CD8_Proliferating': {'CD3': 'pos', 'CD8B': 'pos', 'KI67': 'pos'},
}

# For visualization only: pick primary phenotype (most specific first)
PRIMARY_PHENOTYPE_PRIORITY = [
    'CD8_Proliferating',
    'Tumor_AGFP', 'pERK+_Tumor', 'Proliferating_Tumor', 
    'CD8_T_cells', 'Proliferating_T',
    'Tumor', 'T_cells',
    'Immune', 'Non_immune',
]

# ============================================================================

def load_gated_data(gating_dir):
    """Load gated AnnData"""
    adata = ad.read_h5ad(Path(gating_dir) / 'gated_data.h5ad')
    print(f"Loaded {len(adata):,} cells from {adata.obs['sample_id'].nunique()} samples")
    print(f"Markers: {', '.join(adata.var_names)}")
    return adata

def apply_phenotypes(adata):
    """Apply phenotypes as independent binary columns"""
    
    print("\n" + "="*70)
    print("APPLYING PHENOTYPES (Independent Columns)")
    print("="*70)
    
    # Create binary column for each phenotype
    for phenotype, rules in PHENOTYPES.items():
        mask = np.ones(len(adata), dtype=bool)
        
        for marker, state in rules.items():
            if marker not in adata.var_names:
                print(f"⚠ Warning: {marker} not in dataset, skipping {phenotype}")
                mask = np.zeros(len(adata), dtype=bool)
                break
            
            marker_idx = adata.var_names.get_loc(marker)
            marker_values = adata.layers['gated'][:, marker_idx]
            
            if state == 'pos':
                mask &= (marker_values > 0)
            elif state == 'neg':
                mask &= (marker_values == 0)
        
        # Store as binary column
        adata.obs[f'is_{phenotype}'] = mask.astype(int)
        
        # Per-sample stats
        for sample in adata.obs['sample_id'].unique():
            sample_mask = (adata.obs['sample_id'] == sample) & mask
            count = sample_mask.sum()
            total = (adata.obs['sample_id'] == sample).sum()
            pct = count / total * 100 if total > 0 else 0
            print(f"  {phenotype:25s} | {sample:12s} | {count:8,} ({pct:5.1f}%)")
    
    # Create primary phenotype column for visualization
    adata.obs['primary_phenotype'] = 'Other'
    
    for phenotype in reversed(PRIMARY_PHENOTYPE_PRIORITY):
        if f'is_{phenotype}' in adata.obs.columns:
            mask = adata.obs[f'is_{phenotype}'] == 1
            adata.obs.loc[mask, 'primary_phenotype'] = phenotype
    
    # Summary
    print("\n" + "="*70)
    print("PHENOTYPE SUMMARY")
    print("="*70)
    for phenotype in PHENOTYPES.keys():
        col = f'is_{phenotype}'
        if col in adata.obs.columns:
            count = (adata.obs[col] == 1).sum()
            pct = count / len(adata) * 100
            print(f"{phenotype:25s} | {count:8,} ({pct:5.1f}%)")
    
    print("\nPrimary Phenotype Distribution (for visualization):")
    print(adata.obs['primary_phenotype'].value_counts())
    
    return adata

def create_phenotype_plots(adata, output_dir):
    """Create comprehensive phenotype visualization"""
    plots_dir = Path(output_dir) / "phenotype_plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Primary phenotype distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    phenotype_counts = []
    for sample in adata.obs['sample_id'].unique():
        sample_data = adata.obs[adata.obs['sample_id'] == sample]
        counts = sample_data['primary_phenotype'].value_counts()
        counts['sample'] = sample
        phenotype_counts.append(counts)
    
    df_counts = pd.DataFrame(phenotype_counts).fillna(0)
    df_counts = df_counts.set_index('sample')
    
    df_counts.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Cell Count')
    ax.set_title('Primary Phenotype Distribution per Sample')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plots_dir / 'primary_phenotype_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Spatial - each phenotype independently
    for phenotype in PHENOTYPES.keys():
        col = f'is_{phenotype}'
        if col not in adata.obs.columns:
            continue
        
        mask = adata.obs[col] == 1
        if mask.sum() == 0:
            continue
        
        n_samples = adata.obs['sample_id'].nunique()
        fig, axes = plt.subplots(1, n_samples, figsize=(10*n_samples, 10))
        if n_samples == 1:
            axes = [axes]
        fig.suptitle(f'{phenotype} - Spatial Distribution', fontsize=16)
        
        for i, sample in enumerate(adata.obs['sample_id'].unique()):
            sample_mask = adata.obs['sample_id'] == sample
            sample_data = adata[sample_mask]
            pheno_mask = sample_data.obs[col] == 1
            
            axes[i].scatter(sample_data.obsm['spatial'][:, 0],
                           sample_data.obsm['spatial'][:, 1],
                           s=0.5, alpha=0.3, c='lightgray', rasterized=True)
            
            if pheno_mask.sum() > 0:
                axes[i].scatter(sample_data.obsm['spatial'][pheno_mask, 0],
                               sample_data.obsm['spatial'][pheno_mask, 1],
                               s=1, alpha=0.8, c='red', rasterized=True)
            
            axes[i].set_title(f'{sample}\n{pheno_mask.sum():,} cells ({pheno_mask.mean()*100:.1f}%)')
            axes[i].set_xlabel('X (μm)')
            axes[i].set_ylabel('Y (μm)')
            axes[i].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{phenotype}_spatial.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    # 3. Combined primary phenotype map
    n_samples = adata.obs['sample_id'].nunique()
    fig, axes = plt.subplots(1, n_samples, figsize=(12*n_samples, 12))
    if n_samples == 1:
        axes = [axes]
    fig.suptitle('Primary Phenotypes - Spatial Map', fontsize=16)
    
    unique_phenotypes = adata.obs['primary_phenotype'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_phenotypes)))
    phenotype_colors = dict(zip(unique_phenotypes, colors))
    
    for i, sample in enumerate(adata.obs['sample_id'].unique()):
        sample_mask = adata.obs['sample_id'] == sample
        sample_data = adata[sample_mask]
        
        for phenotype in unique_phenotypes:
            pheno_mask = sample_data.obs['primary_phenotype'] == phenotype
            if pheno_mask.sum() > 0:
                axes[i].scatter(sample_data.obsm['spatial'][pheno_mask, 0],
                               sample_data.obsm['spatial'][pheno_mask, 1],
                               s=1, alpha=0.6, c=[phenotype_colors[phenotype]], 
                               label=phenotype, rasterized=True)
        
        axes[i].set_title(f'{sample}')
        axes[i].set_xlabel('X (μm)')
        axes[i].set_ylabel('Y (μm)')
        axes[i].set_aspect('equal')
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'primary_phenotypes_spatial.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 4. Phenotype co-occurrence
    phenotype_cols = [col for col in adata.obs.columns if col.startswith('is_')]
    cooccurrence = adata.obs[phenotype_cols].corr()
    cooccurrence.index = [col.replace('is_', '') for col in cooccurrence.index]
    cooccurrence.columns = [col.replace('is_', '') for col in cooccurrence.columns]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cooccurrence, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={'label': 'Correlation'})
    ax.set_title('Phenotype Co-occurrence Matrix')
    plt.tight_layout()
    plt.savefig(plots_dir / 'phenotype_cooccurrence.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Phenotype plots saved to: {plots_dir}")

def prepare_for_spatial_tools(adata):
    """Ensure h5ad is properly formatted for SpatialCells/Squidpy/SCIMAP"""
    
    # Add required fields
    adata.uns['spatial'] = {
        'sample_ids': list(adata.obs['sample_id'].unique())
    }
    
    # Ensure marker data in .X
    if 'gated' in adata.layers:
        adata.X = adata.layers['gated'].copy()
    
    # Add raw intensities to layers for later use
    if 'raw' in adata.layers:
        adata.layers['raw_intensity'] = adata.layers['raw'].copy()
    
    print("\n✓ AnnData formatted for spatial analysis tools:")
    print(f"  - .X: gated binary data")
    print(f"  - .obs: phenotype columns (is_*, primary_phenotype)")
    print(f"  - .obsm['spatial']: X,Y coordinates")
    print(f"  - .layers: raw, normalized, gated")
    
    return adata

def export_to_fcs(adata, output_dir):
    """Export to FCS format for FlowJo analysis"""
    try:
        import fcswrite
    except ImportError:
        print("Installing fcswrite...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'fcswrite'])
        import fcswrite
    
    fcs_dir = Path(output_dir) / "fcs_files"
    fcs_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print("EXPORTING TO FCS FORMAT")
    print("="*70)
    
    for sample in adata.obs['sample_id'].unique():
        sample_mask = adata.obs['sample_id'] == sample
        sample_data = adata[sample_mask]
        
        # Build FCS data array
        data_arrays = []
        channel_names = []
        
        # 1. Raw marker intensities
        if 'raw' in sample_data.layers:
            raw_data = sample_data.layers['raw']
            for i, marker in enumerate(sample_data.var_names):
                data_arrays.append(raw_data[:, i])
                channel_names.append(f'{marker}_raw')
        
        # 2. Normalized intensities
        if 'normalized' in sample_data.layers:
            norm_data = sample_data.layers['normalized']
            for i, marker in enumerate(sample_data.var_names):
                data_arrays.append(norm_data[:, i])
                channel_names.append(f'{marker}_norm')
        
        # 3. Gated binary
        if 'gated' in sample_data.layers:
            gated_data = sample_data.layers['gated']
            for i, marker in enumerate(sample_data.var_names):
                data_arrays.append(gated_data[:, i] * 1000)  # Scale for visibility
                channel_names.append(f'{marker}_gate')
        
        # 4. Spatial coordinates
        data_arrays.append(sample_data.obsm['spatial'][:, 0])
        channel_names.append('X_centroid')
        data_arrays.append(sample_data.obsm['spatial'][:, 1])
        channel_names.append('Y_centroid')
        
        # 5. Morphology features
        morph_features = ['Area', 'MajorAxisLength', 'MinorAxisLength', 
                         'Eccentricity', 'Solidity', 'Extent', 'Orientation']
        for feat in morph_features:
            if feat in sample_data.obs.columns:
                data_arrays.append(sample_data.obs[feat].values)
                channel_names.append(feat)
        
        # 6. Phenotype binary columns (scaled for visibility)
        phenotype_cols = [col for col in sample_data.obs.columns if col.startswith('is_')]
        for col in phenotype_cols:
            data_arrays.append(sample_data.obs[col].values * 10000)  # Scale up
            channel_names.append(col.replace('is_', 'pheno_'))
        
        # 7. Primary phenotype as categorical (encode as integer)
        if 'primary_phenotype' in sample_data.obs.columns:
            phenotypes = sample_data.obs['primary_phenotype'].unique()
            phenotype_map = {p: i for i, p in enumerate(phenotypes)}
            encoded = sample_data.obs['primary_phenotype'].map(phenotype_map).values
            data_arrays.append(encoded * 1000)  # Scale for visibility
            channel_names.append('primary_phenotype_id')
        
        # Combine all data
        fcs_data = np.column_stack(data_arrays)
        
        # Write FCS file
        fcs_path = fcs_dir / f"{sample}.fcs"
        
        fcswrite.write_fcs(
            filename=str(fcs_path),
            chn_names=channel_names,
            data=fcs_data,
            compat_chn_names=False,
            compat_copy=True,
            compat_negative=True,
            compat_percent=True,
            compat_max_int16=10000
        )
        
        print(f"  {sample}: {fcs_path.name}")
        print(f"    - {len(sample_data):,} cells")
        print(f"    - {len(channel_names)} channels")
        print(f"    - Raw markers: {', '.join(sample_data.var_names)}")
        print(f"    - Phenotypes: {len(phenotype_cols)} binary columns")
        print(f"    - Morphology: {', '.join([f for f in morph_features if f in sample_data.obs.columns])}")
    
    # Create phenotype mapping file for reference
    if 'primary_phenotype' in adata.obs.columns:
        phenotypes = adata.obs['primary_phenotype'].unique()
        phenotype_map = {p: i for i, p in enumerate(phenotypes)}
        
        with open(fcs_dir / 'phenotype_mapping.txt', 'w') as f:
            f.write("Primary Phenotype ID Mapping (for FlowJo):\n")
            f.write("="*50 + "\n\n")
            for phenotype, idx in sorted(phenotype_map.items(), key=lambda x: x[1]):
                f.write(f"{idx*1000:6.0f} = {phenotype}\n")
    
    print(f"\n✓ FCS files saved to: {fcs_dir}")
    print(f"✓ Import into FlowJo to:")
    print(f"  - Manually adjust gates on raw/normalized data")
    print(f"  - Visualize spatial distributions (X_centroid vs Y_centroid)")
    print(f"  - Compare phenotypes using binary channels")
    print(f"  - Analyze morphology vs expression")
    print(f"\nTip: Use 'primary_phenotype_id' channel to color by phenotype")
    print(f"     (see phenotype_mapping.txt for ID → phenotype mapping)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Phenotype cells with independent columns')
    parser.add_argument('--gating_dir', required=True, help='Output from manual_gating.py')
    parser.add_argument('--output_dir', default='phenotyping_output', help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("PHENOTYPING PIPELINE - Independent Columns")
    print("="*70)
    
    # Load
    adata = load_gated_data(args.gating_dir)
    
    # Apply phenotypes
    adata = apply_phenotypes(adata)
    
    # Format for spatial tools
    adata = prepare_for_spatial_tools(adata)
    
    # Export to FCS for FlowJo
    export_to_fcs(adata, output_dir)

    # Plots
    create_phenotype_plots(adata, output_dir)
    
    # Save
    adata.write(output_dir / 'phenotyped_data.h5ad')
    
    # Export definitions
    with open(output_dir / 'phenotype_definitions.json', 'w') as f:
        json.dump(PHENOTYPES, f, indent=2)
    
    # Export cell table
    adata.obs.to_csv(output_dir / 'phenotyped_cells.csv', index=False)
    
    print(f"\n✅ Complete! Output: {output_dir}")
    print(f"   - phenotyped_data.h5ad: Ready for SpatialCells/Squidpy/SCIMAP")
    print(f"   - Independent binary columns: is_<phenotype>")
    print(f"   - Primary phenotype for visualization: primary_phenotype")
    print(f"\nNext: SpatialCells spatial analysis!")

if __name__ == '__main__':
    main()