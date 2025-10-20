#!/usr/bin/env python3
"""
Tumor Subtype Spatial Analysis - Heterogeneity & Immune Infiltration
Analyzes: 1) Spatial cohesion of tumor subtypes
         2) Immune infiltration by tumor subtype
         3) Immune population enrichment patterns

python tumor_subtype_analysis.py \
    --spatial_h5ad spatial_analysis_output/spatial_analysis.h5ad \
    --output_dir subtype_analysis_output
"""

import pandas as pd
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial import distance_matrix
from scipy.stats import ranksums, mannwhitneyu
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

def analyze_tumor_subtype_spatial_cohesion(adata, output_dir):
    """Measure if tumor subtypes (pERK+, AGFP+, Ki67+) are spatially clustered"""
    print("\n" + "="*70)
    print("TUMOR SUBTYPE SPATIAL COHESION")
    print("="*70)
    
    subtypes = ['Tumor_AGFP', 'pERK+_Tumor', 'Proliferating_Tumor']
    results = []
    
    for sample in adata.obs['sample_id'].unique():
        mask = (adata.obs['sample_id'] == sample) & (adata.obs['is_Tumor'] == 1)
        tumor_cells = adata[mask]
        
        if len(tumor_cells) < 50:
            continue
        
        for subtype in subtypes:
            col = f'is_{subtype}'
            if col not in tumor_cells.obs.columns:
                continue
            
            positive = tumor_cells.obs[col] == 1
            n_pos = positive.sum()
            
            if n_pos < 10:
                continue
            
            # Nearest neighbor analysis
            all_coords = tumor_cells.obsm['spatial']
            pos_coords = all_coords[positive]
            
            # Distance to nearest same-type cell
            nbrs = NearestNeighbors(n_neighbors=2).fit(pos_coords)
            distances, _ = nbrs.kneighbors(pos_coords)
            mean_nn_dist = distances[:, 1].mean()  # Skip self
            
            # Expected distance under random distribution
            tumor_area = (all_coords[:, 0].max() - all_coords[:, 0].min()) * \
                        (all_coords[:, 1].max() - all_coords[:, 1].min())
            density = n_pos / tumor_area
            expected_dist = 0.5 / np.sqrt(density)
            
            clustering_index = expected_dist / mean_nn_dist  # >1 = clustered
            
            results.append({
                'sample_id': sample,
                'group': tumor_cells.obs['group'].iloc[0],
                'subtype': subtype,
                'n_positive': n_pos,
                'pct_positive': n_pos / len(tumor_cells) * 100,
                'mean_nn_distance': mean_nn_dist,
                'expected_distance': expected_dist,
                'clustering_index': clustering_index,
                'interpretation': 'Clustered' if clustering_index > 1.2 else 'Random'
            })
    
    df = pd.DataFrame(results)
    
    # Print summary
    print("\nClustering Index > 1.2 indicates spatial clustering")
    for subtype in subtypes:
        subtype_df = df[df['subtype'] == subtype]
        if len(subtype_df) > 0:
            print(f"\n  {subtype}:")
            for _, row in subtype_df.iterrows():
                print(f"    {row['sample_id']}: {row['clustering_index']:.2f} ({row['interpretation']}) - {row['pct_positive']:.1f}% positive")
    
    df.to_csv(Path(output_dir) / 'tumor_subtype_spatial_cohesion.csv', index=False)
    return df

def define_tumor_subregions(adata):
    """Define tumor subregions based on marker expression"""
    print("\n" + "="*70)
    print("DEFINING TUMOR SUBREGIONS BY MARKER EXPRESSION")
    print("="*70)
    
    subtypes = ['Tumor_AGFP', 'pERK+_Tumor', 'Proliferating_Tumor']
    
    for sample in adata.obs['sample_id'].unique():
        mask = adata.obs['sample_id'] == sample
        
        for subtype in subtypes:
            col = f'is_{subtype}'
            if col not in adata.obs.columns:
                continue
            
            # Assign subregion labels
            adata.obs.loc[mask, f'{subtype}_region'] = 'Other'
            
            tumor_mask = mask & (adata.obs['is_Tumor'] == 1)
            positive_mask = tumor_mask & (adata.obs[col] == 1)
            negative_mask = tumor_mask & (adata.obs[col] == 0)
            
            adata.obs.loc[positive_mask, f'{subtype}_region'] = f'{subtype}_High'
            adata.obs.loc[negative_mask, f'{subtype}_region'] = f'{subtype}_Low'
            
            n_high = positive_mask.sum()
            n_low = negative_mask.sum()
            print(f"  {sample} - {subtype}: High={n_high:,}, Low={n_low:,}")
    
    return adata

def analyze_immune_infiltration_by_subtype(adata, output_dir):
    """Compare immune infiltration into different tumor subtypes"""
    print("\n" + "="*70)
    print("IMMUNE INFILTRATION BY TUMOR SUBTYPE")
    print("="*70)
    
    subtypes = ['Tumor_AGFP', 'pERK+_Tumor', 'Proliferating_Tumor']
    immune_types = ['T_cells', 'CD8_T_cells']
    
    results = []
    
    for sample in adata.obs['sample_id'].unique():
        mask = adata.obs['sample_id'] == sample
        sample_data = adata.obs[mask]
        
        for subtype in subtypes:
            region_col = f'{subtype}_region'
            if region_col not in sample_data.columns:
                continue
            
            for region_type in [f'{subtype}_High', f'{subtype}_Low']:
                region_mask = sample_data[region_col] == region_type
                
                if region_mask.sum() < 10:
                    continue
                
                region_data = sample_data[region_mask]
                
                # Quantify each immune type
                for immune_type in immune_types:
                    immune_col = f'is_{immune_type}'
                    if immune_col not in region_data.columns:
                        continue
                    
                    immune_pct = (region_data[immune_col] == 1).mean() * 100
                    
                    results.append({
                        'sample_id': sample,
                        'group': region_data['group'].iloc[0],
                        'subtype': subtype,
                        'region': region_type,
                        'immune_type': immune_type,
                        'n_cells': region_mask.sum(),
                        'immune_pct': immune_pct
                    })
    
    df = pd.DataFrame(results)
    
    # Statistical comparison High vs Low
    print("\nStatistical Comparison (High vs Low):")
    for subtype in subtypes:
        for immune_type in immune_types:
            subtype_df = df[(df['subtype'] == subtype) & (df['immune_type'] == immune_type)]
            
            if len(subtype_df) == 0:
                continue
            
            high_vals = subtype_df[subtype_df['region'].str.contains('High')]['immune_pct'].values
            low_vals = subtype_df[subtype_df['region'].str.contains('Low')]['immune_pct'].values
            
            if len(high_vals) > 0 and len(low_vals) > 0:
                stat, pval = mannwhitneyu(high_vals, low_vals, alternative='two-sided')
                
                print(f"\n  {subtype} - {immune_type}:")
                print(f"    High: {np.mean(high_vals):.2f}% ± {np.std(high_vals):.2f}%")
                print(f"    Low:  {np.mean(low_vals):.2f}% ± {np.std(low_vals):.2f}%")
                print(f"    p-value: {pval:.4f} {'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'}")
    
    df.to_csv(Path(output_dir) / 'immune_infiltration_by_subtype.csv', index=False)
    return df

def compare_immune_enrichment(adata, output_dir):
    """Compare CD3 vs CD8 enrichment in tumor subtypes (flexible for future CD4)"""
    print("\n" + "="*70)
    print("IMMUNE POPULATION ENRICHMENT IN TUMOR SUBTYPES")
    print("="*70)
    
    subtypes = ['Tumor_AGFP', 'pERK+_Tumor', 'Proliferating_Tumor']
    
    # Flexible immune types - will auto-detect available
    available_immune = [col.replace('is_', '') for col in adata.obs.columns 
                       if col.startswith('is_') and 'T_cells' in col or 'CD' in col]
    
    print(f"Detected immune populations: {', '.join(available_immune)}")
    
    results = []
    
    for sample in adata.obs['sample_id'].unique():
        mask = adata.obs['sample_id'] == sample
        sample_data = adata.obs[mask]
        
        for subtype in subtypes:
            region_col = f'{subtype}_region'
            if region_col not in sample_data.columns:
                continue
            
            for region_type in [f'{subtype}_High', f'{subtype}_Low']:
                region_mask = sample_data[region_col] == region_type
                
                if region_mask.sum() < 10:
                    continue
                
                region_data = sample_data[region_mask]
                
                # Get all immune counts
                immune_counts = {}
                for immune_type in available_immune:
                    col = f'is_{immune_type}'
                    if col in region_data.columns:
                        immune_counts[immune_type] = (region_data[col] == 1).sum()
                
                # Calculate ratios (e.g., CD8/CD3)
                if 'CD8_T_cells' in immune_counts and 'T_cells' in immune_counts:
                    cd8_cd3_ratio = immune_counts['CD8_T_cells'] / immune_counts['T_cells'] if immune_counts['T_cells'] > 0 else 0
                else:
                    cd8_cd3_ratio = np.nan
                
                results.append({
                    'sample_id': sample,
                    'group': region_data['group'].iloc[0],
                    'subtype': subtype,
                    'region': region_type,
                    **{f'{k}_count': v for k, v in immune_counts.items()},
                    'CD8_CD3_ratio': cd8_cd3_ratio
                })
    
    df = pd.DataFrame(results)
    
    # Print enrichment patterns
    print("\nCD8/CD3 Ratio by Subtype Region:")
    for subtype in subtypes:
        subtype_df = df[df['subtype'] == subtype]
        
        if len(subtype_df) == 0:
            continue
        
        print(f"\n  {subtype}:")
        for region in subtype_df['region'].unique():
            region_df = subtype_df[subtype_df['region'] == region]
            ratio_mean = region_df['CD8_CD3_ratio'].mean()
            ratio_std = region_df['CD8_CD3_ratio'].std()
            print(f"    {region}: {ratio_mean:.3f} ± {ratio_std:.3f}")
    
    df.to_csv(Path(output_dir) / 'immune_enrichment_by_subtype.csv', index=False)
    return df

def create_subtype_analysis_plots(cohesion_df, infiltration_df, enrichment_df, output_dir):
    """Visualization of tumor subtype heterogeneity and immune infiltration"""
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Spatial cohesion
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=cohesion_df, x='subtype', y='clustering_index', hue='sample_id', ax=ax)
    ax.axhline(y=1.2, color='red', linestyle='--', label='Clustering threshold')
    ax.set_ylabel('Clustering Index')
    ax.set_title('Tumor Subtype Spatial Cohesion\n(>1.2 = Clustered)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'subtype_spatial_cohesion.png', dpi=200)
    plt.close()
    
    # 2. Immune infiltration by subtype
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, subtype in enumerate(['Tumor_AGFP', 'pERK+_Tumor', 'Proliferating_Tumor']):
        subtype_df = infiltration_df[infiltration_df['subtype'] == subtype]
        
        if len(subtype_df) > 0:
            sns.boxplot(data=subtype_df, x='region', y='immune_pct', 
                       hue='immune_type', ax=axes[i])
            axes[i].set_title(f'{subtype}')
            axes[i].set_ylabel('Immune Infiltration (%)')
            axes[i].set_xlabel('')
            plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'immune_infiltration_by_subtype.png', dpi=200)
    plt.close()
    
    # 3. CD8/CD3 ratio comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=enrichment_df, x='subtype', y='CD8_CD3_ratio', hue='region', ax=ax)
    ax.set_ylabel('CD8/CD3 Ratio')
    ax.set_title('CD8 T cell Enrichment in Tumor Subtypes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / 'cd8_enrichment_by_subtype.png', dpi=200)
    plt.close()
    
    print(f"\n✓ Subtype analysis plots saved to: {plots_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--spatial_h5ad', required=True, help='Output from spatial_analysis.py')
    parser.add_argument('--output_dir', default='subtype_analysis_output')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("TUMOR SUBTYPE HETEROGENEITY & IMMUNE ANALYSIS")
    print("="*70)
    
    # Load
    adata = ad.read_h5ad(args.spatial_h5ad)
    
    # Analyses
    cohesion_df = analyze_tumor_subtype_spatial_cohesion(adata, output_dir)
    adata = define_tumor_subregions(adata)
    infiltration_df = analyze_immune_infiltration_by_subtype(adata, output_dir)
    enrichment_df = compare_immune_enrichment(adata, output_dir)
    
    # Plots
    create_subtype_analysis_plots(cohesion_df, infiltration_df, enrichment_df, output_dir)
    
    # Save updated h5ad
    adata.write(output_dir / 'subtype_analysis.h5ad')
    
    print(f"\n✅ Complete! Output: {output_dir}")
    print("\nGenerated files:")
    print("  - tumor_subtype_spatial_cohesion.csv: Clustering metrics")
    print("  - immune_infiltration_by_subtype.csv: Infiltration comparisons")
    print("  - immune_enrichment_by_subtype.csv: CD8/CD3 ratios")
    print("\nFlexible for future: Add CD4 phenotype → auto-detected")

if __name__ == '__main__':
    main()