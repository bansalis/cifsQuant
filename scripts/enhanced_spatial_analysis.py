#!/usr/bin/env python3
"""
Enhanced Spatial Analysis with RCN Detection - Fully Adaptive Version
Automatically detects available phenotypes and adapts analysis accordingly
"""

import os
import sys
import pandas as pd
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Publication-quality plotting defaults
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 1.5,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.spines.top': False,
    'axes.spines.right': False
})

def load_and_analyze_phenotypes(h5ad_path):
    """Load data and automatically detect available phenotypes"""
    print(f"Loading phenotyped data: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} channels")
    
    # Get all phenotype columns
    phenotype_cols = [col for col in adata.obs.columns if col.startswith(('tumor_', 't_cells_', 'immune_'))]
    print(f"Phenotype columns found: {phenotype_cols}")
    
    # Analyze each phenotype column
    phenotype_info = {}
    for col in phenotype_cols:
        unique_vals = adata.obs[col].unique()
        n_positive = adata.obs[col].sum() if adata.obs[col].dtype == bool else (adata.obs[col] == 1).sum()
        phenotype_info[col] = {
            'unique_values': unique_vals,
            'n_positive': n_positive,
            'total_cells': len(adata.obs),
            'percentage': n_positive / len(adata.obs) * 100,
            'is_boolean': adata.obs[col].dtype == bool
        }
        print(f"  {col}: {n_positive:,} positive cells ({phenotype_info[col]['percentage']:.1f}%)")
    
    return adata, phenotype_info

def get_spatial_coordinates(adata):
    """Get spatial coordinates from various possible locations"""
    if 'spatial' in adata.obsm.keys():
        return adata.obsm['spatial']
    elif 'X_centroid' in adata.obs.columns and 'Y_centroid' in adata.obs.columns:
        coords = adata.obs[['X_centroid', 'Y_centroid']].values
        adata.obsm['spatial'] = coords  # Store for future use
        return coords
    else:
        raise ValueError("No spatial coordinates found")

def get_positive_cells(adata, phenotype, phenotype_info):
    """Get mask for positive cells, handling both boolean and numeric phenotypes"""
    if phenotype_info[phenotype]['is_boolean']:
        return adata.obs[phenotype] == True
    else:
        return adata.obs[phenotype] == 1

def detect_rcns(adata, phenotype_info, n_neighborhoods=12):
    """Detect RCNs using available phenotypes"""
    print(f"\nDetecting {n_neighborhoods} RCNs...")
    
    coords = get_spatial_coordinates(adata)
    
    # Use phenotypes with sufficient positive cells (>1% of total)
    viable_phenotypes = [col for col, info in phenotype_info.items() 
                        if info['percentage'] > 1.0 and info['n_positive'] > 100]
    
    if len(viable_phenotypes) < 3:
        print("Not enough viable phenotypes for RCN analysis")
        return None, None
    
    print(f"Using {len(viable_phenotypes)} phenotypes for RCN detection: {viable_phenotypes}")
    
    # Build neighborhoods
    nbrs = NearestNeighbors(n_neighbors=15)
    nbrs.fit(coords)
    
    # Sample cells for computational efficiency
    n_sample = min(50000, adata.n_obs)
    cell_indices = np.random.choice(adata.n_obs, size=n_sample, replace=False)
    
    neighborhood_data = []
    for idx in tqdm(cell_indices, desc="Building neighborhoods"):
        distances, neighbor_indices = nbrs.kneighbors([coords[idx]])
        neighbor_indices = neighbor_indices[0]
        
        # Count phenotypes in neighborhood
        neighborhood_composition = []
        for phenotype in viable_phenotypes:
            mask = get_positive_cells(adata, phenotype, phenotype_info)
            count = mask.iloc[neighbor_indices].sum()
            neighborhood_composition.append(count)
        
        neighborhood_data.append(neighborhood_composition)
    
    # Apply LDA
    neighborhood_matrix = np.array(neighborhood_data)
    lda = LatentDirichletAllocation(n_components=n_neighborhoods, random_state=42, max_iter=100)
    lda_weights = lda.fit_transform(neighborhood_matrix)
    
    # Assign RCNs to all cells
    neighborhood_assignments = np.argmax(lda_weights, axis=1)
    rcn_assignments = np.full(adata.n_obs, -1)
    rcn_assignments[cell_indices] = neighborhood_assignments
    
    # Interpolate for non-sampled cells
    sampled_coords = coords[cell_indices]
    nbrs_interp = NearestNeighbors(n_neighbors=1)
    nbrs_interp.fit(sampled_coords)
    
    unassigned_mask = rcn_assignments == -1
    if np.any(unassigned_mask):
        unassigned_coords = coords[unassigned_mask]
        _, nearest_indices = nbrs_interp.kneighbors(unassigned_coords)
        rcn_assignments[unassigned_mask] = neighborhood_assignments[nearest_indices.flatten()]
    
    adata.obs['RCN'] = rcn_assignments.astype(str)
    
    print(f"Identified {n_neighborhoods} RCNs")
    return lda.components_, viable_phenotypes

def create_rcn_plots(adata, rcn_profiles, viable_phenotypes, output_dir):
    """Create RCN composition plots"""
    if rcn_profiles is None:
        return
    
    plot_dir = Path(output_dir) / "spatial_analysis" / "rcn_analysis"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # RCN composition heatmap
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Clean names for plotting
    clean_names = [name.replace('_', ' ').title() for name in viable_phenotypes]
    rcn_df = pd.DataFrame(rcn_profiles.T, 
                         index=clean_names,
                         columns=[f'RCN {i+1}' for i in range(len(rcn_profiles))])
    
    # Composition heatmap
    sns.heatmap(rcn_df, annot=True, fmt='.2f', cmap='Reds', ax=axes[0],
                cbar_kws={'label': 'Cell Type Probability'}, 
                linewidths=0.5, square=True)
    axes[0].set_title('RCN Composition Profiles', fontweight='bold', pad=20)
    axes[0].set_xlabel('Recurrent Cellular Neighborhoods')
    
    # RCN distribution
    rcn_dist = adata.obs['RCN'].value_counts().sort_index()
    bars = axes[1].bar(range(len(rcn_dist)), rcn_dist.values, color='steelblue', alpha=0.7)
    axes[1].set_title('RCN Distribution', fontweight='bold', pad=20)
    axes[1].set_xlabel('RCN')
    axes[1].set_ylabel('Cell Count')
    axes[1].set_xticks(range(len(rcn_dist)))
    axes[1].set_xticklabels([f'RCN {int(i)+1}' for i in rcn_dist.index])
    
    # Add value labels on bars
    for bar, count in zip(bars, rcn_dist.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rcn_dist.values)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'rcn_composition_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save composition table
    rcn_df.to_csv(plot_dir / 'rcn_composition_table.csv')
    print(f"RCN analysis saved to: {plot_dir}")

def analyze_distances_adaptively(adata, phenotype_info, output_dir):
    """Perform distance analysis using available phenotypes"""
    print("\nPerforming adaptive distance analysis...")
    
    coords = get_spatial_coordinates(adata)
    
    # Find target phenotypes (specific tumor subtypes)
    target_phenotypes = [col for col, info in phenotype_info.items() 
                        if col.startswith('tumor_') and '_pos' in col and info['n_positive'] > 50]
    
    # Find reference phenotypes (immune cells)
    reference_phenotypes = [col for col, info in phenotype_info.items() 
                           if (col.startswith(('t_cells_', 'immune_')) and 
                               not col.endswith('_all') and info['n_positive'] > 100)]
    
    if not target_phenotypes:
        print("No suitable target phenotypes found")
        return
    
    if not reference_phenotypes:
        print("No suitable reference phenotypes found")
        return
    
    print(f"Target phenotypes: {target_phenotypes}")
    print(f"Reference phenotypes: {reference_phenotypes}")
    
    plot_dir = Path(output_dir) / "spatial_analysis" / "distance_analysis"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    for target in target_phenotypes:
        print(f"Analyzing {target}...")
        
        target_mask = get_positive_cells(adata, target, phenotype_info)
        target_coords = coords[target_mask]
        
        if len(target_coords) == 0:
            continue
        
        # Create plots for this target
        n_refs = len(reference_phenotypes)
        fig, axes = plt.subplots(2, n_refs, figsize=(6*n_refs, 10))
        if n_refs == 1:
            axes = axes.reshape(2, 1)
        
        for i, reference in enumerate(reference_phenotypes):
            ref_mask = get_positive_cells(adata, reference, phenotype_info)
            ref_coords = coords[ref_mask]
            
            if len(ref_coords) == 0:
                continue
            
            # Calculate distances
            distances = cdist(target_coords, ref_coords)
            min_distances = np.min(distances, axis=1)
            
            # Plot 1: Distance distribution
            ax1 = axes[0, i]
            parts = ax1.violinplot([min_distances], positions=[1], widths=0.6, showmeans=True)
            parts['bodies'][0].set_facecolor('#2E86AB')
            parts['bodies'][0].set_alpha(0.7)
            
            ax1.set_ylabel('Distance (μm)')
            ax1.set_title(f'{target.replace("_", " ").title()}\nvs {reference.replace("_", " ").title()}')
            ax1.set_xticks([])
            ax1.grid(True, alpha=0.3)
            
            # Add median line
            median_dist = np.median(min_distances)
            ax1.axhline(median_dist, color='red', linestyle='--', linewidth=2)
            ax1.text(1.1, median_dist, f'Median: {median_dist:.1f}μm', 
                    verticalalignment='center', fontweight='bold')
            
            # Plot 2: Cumulative distribution
            ax2 = axes[1, i]
            sorted_distances = np.sort(min_distances)
            cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
            
            ax2.plot(sorted_distances, cumulative, linewidth=3, color='#A23B72')
            ax2.axvline(50, color='red', linestyle='--', alpha=0.7, label='50μm')
            ax2.axvline(100, color='orange', linestyle='--', alpha=0.7, label='100μm')
            ax2.set_xlabel('Distance (μm)')
            ax2.set_ylabel('Cumulative Probability')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add proximity stats
            within_50 = np.mean(min_distances <= 50) * 100
            within_100 = np.mean(min_distances <= 100) * 100
            ax2.text(0.05, 0.95, f'Within 50μm: {within_50:.1f}%\nWithin 100μm: {within_100:.1f}%', 
                    transform=ax2.transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(plot_dir / f'distance_analysis_{target}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distance analysis for {target} saved")

def analyze_by_conditions(adata, phenotype_info, output_dir):
    """Analyze patterns by experimental conditions"""
    print("\nAnalyzing by experimental conditions...")
    
    # Find condition columns
    condition_cols = ['treatment', 'timepoint', 'immunogenic']
    available_conditions = [col for col in condition_cols if col in adata.obs.columns]
    
    if not available_conditions:
        print("No condition columns found")
        return
    
    print(f"Available conditions: {available_conditions}")
    
    plot_dir = Path(output_dir) / "spatial_analysis" / "comparative_analysis"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    for condition in available_conditions:
        unique_values = adata.obs[condition].unique()
        unique_values = [v for v in unique_values if pd.notna(v)]
        
        if len(unique_values) < 2:
            continue
        
        print(f"Analyzing condition: {condition} with values: {unique_values}")
        
        # Create condition summary plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Get viable phenotypes for plotting
        viable_phenotypes = [col for col, info in phenotype_info.items() 
                           if info['percentage'] > 1.0]
        
        condition_summary = []
        for value in unique_values:
            subset = adata.obs[adata.obs[condition] == value]
            for phenotype in viable_phenotypes:
                if phenotype_info[phenotype]['is_boolean']:
                    n_positive = subset[phenotype].sum()
                else:
                    n_positive = (subset[phenotype] == 1).sum()
                
                percentage = n_positive / len(subset) * 100
                condition_summary.append({
                    'condition': value,
                    'phenotype': phenotype,
                    'percentage': percentage
                })
        
        if condition_summary:
            summary_df = pd.DataFrame(condition_summary)
            summary_pivot = summary_df.pivot(index='phenotype', columns='condition', values='percentage')
            
            # Create heatmap
            sns.heatmap(summary_pivot, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                       cbar_kws={'label': 'Percentage of Cells'})
            ax.set_title(f'Phenotype Distribution by {condition.title()}', 
                        fontweight='bold', pad=20)
            ax.set_xlabel(condition.title())
            ax.set_ylabel('Phenotype')
            
            plt.tight_layout()
            plt.savefig(plot_dir / f'phenotype_by_{condition}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save summary data
            summary_df.to_csv(plot_dir / f'summary_{condition}.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='Adaptive enhanced spatial analysis')
    parser.add_argument('--input', required=True, help='Path to phenotyped h5ad file')
    parser.add_argument('--output', default='/app/spatial_analysis/outputs')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ADAPTIVE ENHANCED SPATIAL ANALYSIS")
    print("="*80)
    
    # Load and analyze data
    adata, phenotype_info = load_and_analyze_phenotypes(args.input)
    
    # Phase 1: RCN Detection
    print("\nPhase 1: RCN Detection")
    print("-" * 40)
    rcn_profiles, viable_phenotypes = detect_rcns(adata, phenotype_info)
    if rcn_profiles is not None:
        create_rcn_plots(adata, rcn_profiles, viable_phenotypes, args.output)
    
    # Phase 2: Distance Analysis
    print("\nPhase 2: Distance Analysis")
    print("-" * 40)
    analyze_distances_adaptively(adata, phenotype_info, args.output)
    
    # Phase 3: Condition Analysis
    print("\nPhase 3: Condition Analysis")
    print("-" * 40)
    analyze_by_conditions(adata, phenotype_info, args.output)
    
    print(f"\n{'='*80}")
    print("ADAPTIVE ANALYSIS COMPLETE!")
    print(f"Results saved to: {args.output}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()