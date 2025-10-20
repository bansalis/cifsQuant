#!/usr/bin/env python3
"""
Spatial Analysis Pipeline - Tumor Regions & Immune Infiltration
Follows SpatialCells workflow + SCIMAP RCN generation

python spatial_analysis.py --phenotyped phenotyping_output/phenotyped_data.h5ad --metadata sample_metadata.csv --output_dir spatial_analysis_output
"""

import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import squidpy as sq
import scimap as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial import distance_matrix
from scipy.ndimage import binary_dilation
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SPATIAL CONFIGURATION
# ============================================================================

TUMOR_MARKERS = ['TOM']  # Define tumor regions
TUMOR_HETEROGENEITY_MARKERS = ['AGFP', 'PERK', 'KI67']  # Heterogeneity analysis
IMMUNE_MARKERS = ['CD3', 'CD8B']  # Infiltration analysis

RADIUS_UM = 50  # Neighborhood radius in microns
KNN = 15  # K-nearest neighbors for graph construction

# ============================================================================

def load_data(phenotyped_path, metadata_path):
    """Load phenotyped h5ad and merge metadata"""
    adata = ad.read_h5ad(phenotyped_path)
    
    # Load metadata
    meta = pd.read_csv(metadata_path)
    
    # Merge metadata into adata.obs
    adata.obs = adata.obs.merge(
        meta[['sample_id', 'group', 'treatment', 'timepoint']], 
        on='sample_id', how='left'
    )
    
    print(f"Loaded {len(adata):,} cells from {adata.obs['sample_id'].nunique()} samples")
    print(f"Samples by group: {adata.obs.groupby('group')['sample_id'].nunique().to_dict()}")
    print(f"Timepoints: {adata.obs['timepoint'].unique()}")
    
    return adata

def define_tumor_regions(adata):
    """Define tumor regions based on TOM+ cell density"""
    print("\n" + "="*70)
    print("DEFINING TUMOR REGIONS (TOM+ DENSITY)")
    print("="*70)
    
    for sample in adata.obs['sample_id'].unique():
        mask = adata.obs['sample_id'] == sample
        sample_adata = adata[mask].copy()
        
        # Get TOM+ cells
        tumor_mask = sample_adata.obs['is_Tumor'] == 1
        tumor_coords = sample_adata.obsm['spatial'][tumor_mask]
        all_coords = sample_adata.obsm['spatial']
        
        if len(tumor_coords) == 0:
            print(f"  {sample}: No tumor cells found")
            adata.obs.loc[mask, 'tumor_region'] = 'None'
            adata.obs.loc[mask, 'distance_to_tumor'] = np.inf
            continue
        
        # Compute distance to nearest tumor cell
        dist_matrix = distance_matrix(all_coords, tumor_coords)
        min_dist = dist_matrix.min(axis=1)
        
        # Define regions by distance
        adata.obs.loc[mask, 'distance_to_tumor'] = min_dist
        adata.obs.loc[mask, 'tumor_region'] = pd.cut(
            min_dist,
            bins=[-1, 0, 30, 100, np.inf],
            labels=['Tumor_Core', 'Tumor_Margin', 'Stroma_Near', 'Stroma_Far']
        )
        
        counts = adata.obs.loc[mask, 'tumor_region'].value_counts()
        print(f"  {sample}:")
        for region, count in counts.items():
            pct = count / mask.sum() * 100
            print(f"    {region:20s}: {count:6,} cells ({pct:5.1f}%)")
    
    return adata

def compute_spatial_graph(adata):
    """Build spatial neighborhood graph (SpatialCells style)"""
    print("\n" + "="*70)
    print(f"BUILDING SPATIAL GRAPH (radius={RADIUS_UM}μm, k={KNN})")
    print("="*70)
    
    for sample in adata.obs['sample_id'].unique():
        mask = adata.obs['sample_id'] == sample
        sample_adata = adata[mask].copy()
        
        # Spatial graph
        sq.gr.spatial_neighbors(
            sample_adata,
            coord_type='generic',
            spatial_key='spatial',
            radius=RADIUS_UM,
            n_neighs=KNN
        )
        
        # Store in main adata
        adata.uns[f'{sample}_spatial_connectivities'] = sample_adata.obsp['spatial_connectivities']
        adata.uns[f'{sample}_spatial_distances'] = sample_adata.obsp['spatial_distances']
        
        n_neighbors = (sample_adata.obsp['spatial_connectivities'] > 0).sum(axis=1)
        print(f"  {sample}: Neighbors per cell = {np.mean(n_neighbors):.1f} ± {np.std(n_neighbors):.1f}")
    
    return adata

def analyze_immune_infiltration(adata):
    """Quantify CD3/CD8 infiltration into tumor regions"""
    print("\n" + "="*70)
    print("IMMUNE INFILTRATION ANALYSIS")
    print("="*70)
    
    results = []
    
    for sample in adata.obs['sample_id'].unique():
        mask = adata.obs['sample_id'] == sample
        sample_data = adata.obs[mask]
        
        for region in ['Tumor_Core', 'Tumor_Margin', 'Stroma_Near', 'Stroma_Far']:
            region_mask = sample_data['tumor_region'] == region
            
            if region_mask.sum() == 0:
                continue
            
            region_data = sample_data[region_mask]
            
            # Compute infiltration metrics
            cd3_pct = (region_data['is_T_cells'] == 1).mean() * 100
            cd8_pct = (region_data['is_CD8_T_cells'] == 1).mean() * 100
            cd8_cd3_ratio = cd8_pct / cd3_pct if cd3_pct > 0 else 0
            
            results.append({
                'sample_id': sample,
                'group': sample_data['group'].iloc[0],
                'timepoint': sample_data['timepoint'].iloc[0],
                'region': region,
                'n_cells': region_mask.sum(),
                'CD3_pct': cd3_pct,
                'CD8_pct': cd8_pct,
                'CD8_CD3_ratio': cd8_cd3_ratio
            })
    
    df = pd.DataFrame(results)
    
    # Print summary
    for region in df['region'].unique():
        region_df = df[df['region'] == region]
        print(f"\n  {region}:")
        print(f"    CD3+ T cells: {region_df['CD3_pct'].mean():.1f}% ± {region_df['CD3_pct'].std():.1f}%")
        print(f"    CD8+ T cells: {region_df['CD8_pct'].mean():.1f}% ± {region_df['CD8_pct'].std():.1f}%")
        print(f"    CD8/CD3 ratio: {region_df['CD8_CD3_ratio'].mean():.2f} ± {region_df['CD8_CD3_ratio'].std():.2f}")
    
    return df

def analyze_tumor_heterogeneity(adata):
    """Analyze aGFP, pERK, Ki67 heterogeneity in tumor regions"""
    print("\n" + "="*70)
    print("TUMOR MARKER HETEROGENEITY")
    print("="*70)
    
    results = []
    
    for sample in adata.obs['sample_id'].unique():
        mask = (adata.obs['sample_id'] == sample) & (adata.obs['is_Tumor'] == 1)
        
        if mask.sum() == 0:
            continue
        
        tumor_data = adata.obs[mask]
        tumor_idx = np.where(mask)[0]  
        
        for marker in TUMOR_HETEROGENEITY_MARKERS:
            if marker not in adata.var_names:
                continue
            
            marker_idx = adata.var_names.get_loc(marker)
            raw_vals = adata.layers['raw'][tumor_idx, marker_idx]
            
            # Heterogeneity metrics
            cv = np.std(raw_vals) / np.mean(raw_vals) if np.mean(raw_vals) > 0 else 0
            
            results.append({
                'sample_id': sample,
                'group': tumor_data['group'].iloc[0],
                'timepoint': tumor_data['timepoint'].iloc[0],
                'marker': marker,
                'mean_intensity': np.mean(raw_vals),
                'cv': cv,
                'positive_fraction': (raw_vals > np.percentile(raw_vals, 75)).mean()
            })
    
    df = pd.DataFrame(results)

    # Print summary by group
    for marker in TUMOR_HETEROGENEITY_MARKERS:
        marker_df = df[df['marker'] == marker]
        print(f"\n  {marker}:")
        for group in marker_df['group'].unique():
            group_df = marker_df[marker_df['group'] == group]
            print(f"    {group}: CV = {group_df['cv'].mean():.2f} ± {group_df['cv'].std():.2f}")
            print(f"    {group}: Pos% = {group_df['positive_fraction'].mean()*100:.1f}% ± {group_df['positive_fraction'].std()*100:.1f}%")
    return df

def generate_rcns(adata):
    """Generate SCIMAP RCNs (Recurrent Cellular Neighborhoods)"""
    print("\n" + "="*70)
    print("GENERATING SCIMAP RCNs")
    print("="*70)
    
    try:
        adata.obs['imageid'] = adata.obs['sample_id']
        
        # Compute spatial counts
        sm.tl.spatial_count(
            adata,
            x_coordinate='X_centroid',
            y_coordinate='Y_centroid',
            phenotype='primary_phenotype',
            method='radius',
            radius=50,
            imageid='imageid'
        )
        
        # Cluster
        sm.tl.spatial_cluster(
            adata,
            method='kmeans',
            k=10
        )
        
        # SCIMAP puts results in different places - find it
        cluster_col = None
        for col in ['spatial_cluster', 'spatial_kmeans', 'kmeans']:
            if col in adata.obs.columns:
                cluster_col = col
                break
        
        if cluster_col is None:
            raise ValueError("No cluster column found after spatial_cluster()")
        
        adata.obs['rcn'] = adata.obs[cluster_col].astype(int)
        
        # RCN composition
        rcn_comp = []
        for rcn in adata.obs['rcn'].unique():
            rcn_mask = adata.obs['rcn'] == rcn
            rcn_data = adata.obs[rcn_mask]
            
            comp = {'rcn': rcn, 'n_cells': rcn_mask.sum()}
            
            for pheno in ['T_cells', 'CD8_T_cells', 'Tumor', 'Tumor_AGFP', 'pERK+_Tumor', 'Proliferating_Tumor']:
                col = f'is_{pheno}'
                if col in adata.obs.columns:
                    comp[pheno] = (rcn_data[col] == 1).mean() * 100
            
            rcn_comp.append(comp)
        
        df_rcn = pd.DataFrame(rcn_comp)
        
        print(f"\n✓ SCIMAP RCNs generated using column: {cluster_col}")
        print(f"  Identified {len(df_rcn)} RCNs:")
        for _, row in df_rcn.iterrows():
            print(f"    RCN {row['rcn']}: {row['n_cells']:,} cells")
            if 'Tumor' in row:
                print(f"      Tumor: {row['Tumor']:.1f}%, CD8: {row.get('CD8_T_cells', 0):.1f}%")
        
        return df_rcn
        
    except Exception as e:
        print(f"  SCIMAP failed: {e}")
        print(f"  Using manual kmeans fallback...")
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=10, random_state=42)
        adata.obs['rcn'] = kmeans.fit_predict(adata.obsm['spatial'])
        return None

def create_spatial_plots(adata, infiltration_df, heterogeneity_df, output_dir):
    """Comprehensive spatial visualization"""
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Tumor regions map
    fig, axes = plt.subplots(1, adata.obs['sample_id'].nunique(), 
                            figsize=(10*adata.obs['sample_id'].nunique(), 10))
    if adata.obs['sample_id'].nunique() == 1:
        axes = [axes]
    fig.suptitle('Tumor Region Definition', fontsize=16)
    
    region_colors = {'Tumor_Core': 'red', 'Tumor_Margin': 'orange', 
                    'Stroma_Near': 'yellow', 'Stroma_Far': 'lightblue'}
    
    for i, sample in enumerate(adata.obs['sample_id'].unique()):
        mask = adata.obs['sample_id'] == sample
        sample_data = adata[mask]
        
        for region, color in region_colors.items():
            region_mask = sample_data.obs['tumor_region'] == region
            if region_mask.sum() > 0:
                axes[i].scatter(sample_data.obsm['spatial'][region_mask, 0],
                               sample_data.obsm['spatial'][region_mask, 1],
                               s=1, c=color, alpha=0.6, label=region, rasterized=True)
        
        axes[i].set_title(f'{sample}')
        axes[i].legend()
        axes[i].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'tumor_regions_spatial.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. CD8 infiltration by region
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=infiltration_df, x='region', y='CD8_pct', hue='group', ax=ax)
    ax.set_xlabel('Tumor Region')
    ax.set_ylabel('CD8+ T cells (%)')
    ax.set_title('CD8+ T cell Infiltration by Region and Genotype')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / 'cd8_infiltration_by_region.png', dpi=200)
    plt.close()
    
    # 3. Tumor heterogeneity comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, marker in enumerate(TUMOR_HETEROGENEITY_MARKERS):
        marker_df = heterogeneity_df[heterogeneity_df['marker'] == marker]
        sns.boxplot(data=marker_df, x='group', y='cv', ax=axes[i])
        axes[i].set_title(f'{marker} Heterogeneity (CV)')
        axes[i].set_ylabel('Coefficient of Variation')
    plt.tight_layout()
    plt.savefig(plots_dir / 'tumor_heterogeneity.png', dpi=200)
    plt.close()
    
    # 4. RCN spatial map
    if 'rcn' in adata.obs.columns:
        fig, axes = plt.subplots(1, adata.obs['sample_id'].nunique(),
                                figsize=(12*adata.obs['sample_id'].nunique(), 12))
        if adata.obs['sample_id'].nunique() == 1:
            axes = [axes]
        fig.suptitle('Recurrent Cellular Neighborhoods (RCNs)', fontsize=16)
        
        for i, sample in enumerate(adata.obs['sample_id'].unique()):
            mask = adata.obs['sample_id'] == sample
            sample_data = adata[mask]
            
            scatter = axes[i].scatter(sample_data.obsm['spatial'][:, 0],
                                     sample_data.obsm['spatial'][:, 1],
                                     c=sample_data.obs['rcn'], s=1, 
                                     cmap='tab20', alpha=0.6, rasterized=True)
            axes[i].set_title(f'{sample}')
            axes[i].set_aspect('equal')
            plt.colorbar(scatter, ax=axes[i], label='RCN')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'rcns_spatial.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"\n✓ Plots saved to: {plots_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phenotyped', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output_dir', default='spatial_analysis_output')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("SPATIAL ANALYSIS PIPELINE")
    print("="*70)
    
    # Load
    adata = load_data(args.phenotyped, args.metadata)
    
    # Tumor regions
    adata = define_tumor_regions(adata)
    
    # Spatial graph
    adata = compute_spatial_graph(adata)
    
    # Analysis
    infiltration_df = analyze_immune_infiltration(adata)
    heterogeneity_df = analyze_tumor_heterogeneity(adata)
    rcn_df = generate_rcns(adata)
    
    # Plots
    create_spatial_plots(adata, infiltration_df, heterogeneity_df, output_dir)
    
    # Before adata.write(), add this cleanup:
    # Convert categorical columns to string to avoid h5ad save issues
    for col in adata.obs.columns:
        if adata.obs[col].dtype.name == 'category':
            adata.obs[col] = adata.obs[col].astype(str)

    # Drop SCIMAP temp columns that cause save issues
    cols_to_drop = [c for c in adata.obs.columns if 'spatial_kmeans' in c or 'spatial_count' in c]
    if cols_to_drop:
        adata.obs = adata.obs.drop(columns=cols_to_drop)

    adata.write(output_dir / 'spatial_analysis.h5ad')

    # Save
    adata.write(output_dir / 'spatial_analysis.h5ad')
    infiltration_df.to_csv(output_dir / 'immune_infiltration.csv', index=False)
    heterogeneity_df.to_csv(output_dir / 'tumor_heterogeneity.csv', index=False)
    if rcn_df is not None:
        rcn_df.to_csv(output_dir / 'rcn_composition.csv', index=False)
    
    print(f"\n✅ Complete! Output: {output_dir}")

if __name__ == '__main__':
    main()