#!/usr/bin/env python3
"""
Spatial Neighborhood Analysis
Analyzes spatial relationships around pERK+ and NINJA+ tumor cells
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
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu, chi2_contingency
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_configs(config_path, phenotype_config_path):
    """Load both experiment and phenotyping configs"""
    with open(config_path, 'r') as f:
        exp_config = yaml.safe_load(f)
    with open(phenotype_config_path, 'r') as f:
        pheno_config = yaml.safe_load(f)
    return exp_config, pheno_config

def load_phenotyped_data(h5ad_path):
    """Load the phenotyped dataset"""
    print(f"Loading phenotyped data: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} channels")
    
    # Show available phenotypes
    phenotype_cols = [col for col in adata.obs.columns if col.startswith(('tumor_', 't_cells_', 'immune_'))]
    print(f"Available phenotypes: {phenotype_cols}")
    
    return adata

def calculate_distances_to_targets(adata, target_phenotype, reference_phenotypes, max_distance=200):
    """
    Calculate distances from target cells to reference cell types
    
    Parameters:
    - target_phenotype: e.g., 'tumor_perk_pos'
    - reference_phenotypes: list, e.g., ['t_cells_cd8', 't_cells_cd4']
    - max_distance: maximum distance to consider (pixels)
    """
    results = {}
    
    # Get target cells
    target_mask = adata.obs[target_phenotype] == True
    target_coords = adata.obs.loc[target_mask, ['X_centroid', 'Y_centroid']].values
    
    if len(target_coords) == 0:
        print(f"No {target_phenotype} cells found")
        return {}
    
    print(f"Analyzing {len(target_coords)} {target_phenotype} cells")
    
    for ref_phenotype in reference_phenotypes:
        ref_mask = adata.obs[ref_phenotype] == True
        ref_coords = adata.obs.loc[ref_mask, ['X_centroid', 'Y_centroid']].values
        
        if len(ref_coords) == 0:
            print(f"  No {ref_phenotype} cells found")
            continue
        
        print(f"  Computing distances to {len(ref_coords)} {ref_phenotype} cells")
        
        # Calculate pairwise distances
        distances = cdist(target_coords, ref_coords)
        
        # Get minimum distance for each target cell
        min_distances = np.min(distances, axis=1)
        
        # Filter by max distance
        valid_distances = min_distances[min_distances <= max_distance]
        
        results[ref_phenotype] = {
            'distances': min_distances,
            'valid_distances': valid_distances,
            'n_target_cells': len(target_coords),
            'n_reference_cells': len(ref_coords),
            'n_within_range': len(valid_distances),
            'median_distance': np.median(min_distances),
            'mean_distance': np.mean(min_distances)
        }
        
        print(f"    Median distance: {np.median(min_distances):.1f} pixels")
        print(f"    Cells within {max_distance}px: {len(valid_distances)}/{len(min_distances)}")
    
    return results

def analyze_neighborhood_composition(adata, target_phenotype, radius=50, reference_phenotypes=None):
    """
    Analyze the cellular composition within a radius of target cells
    """
    if reference_phenotypes is None:
        reference_phenotypes = ['t_cells_cd8', 't_cells_cd4', 'immune_non_t', 'tumor_all']
    
    # Get target cells
    target_mask = adata.obs[target_phenotype] == True
    target_coords = adata.obs.loc[target_mask, ['X_centroid', 'Y_centroid']].values
    target_indices = adata.obs.index[target_mask]
    
    if len(target_coords) == 0:
        return {}
    
    print(f"Analyzing neighborhoods around {len(target_coords)} {target_phenotype} cells (radius={radius}px)")
    
    # Get all cell coordinates
    all_coords = adata.obs[['X_centroid', 'Y_centroid']].values
    
    neighborhood_data = []
    
    for i, (target_idx, target_coord) in enumerate(zip(target_indices, target_coords)):
        # Calculate distances to all cells
        distances = cdist([target_coord], all_coords)[0]
        
        # Find cells within radius (excluding self)
        neighbor_mask = (distances <= radius) & (distances > 0)
        neighbor_indices = adata.obs.index[neighbor_mask]
        
        if len(neighbor_indices) == 0:
            continue
        
        # Count each phenotype in neighborhood
        neighbor_data = {
            'target_cell_id': target_idx,
            'slide_id': adata.obs.loc[target_idx, 'slide_id'],
            'total_neighbors': len(neighbor_indices)
        }
        
        for phenotype in reference_phenotypes:
            if phenotype in adata.obs.columns:
                phenotype_count = adata.obs.loc[neighbor_indices, phenotype].sum()
                neighbor_data[f'{phenotype}_count'] = phenotype_count
                neighbor_data[f'{phenotype}_fraction'] = phenotype_count / len(neighbor_indices) if len(neighbor_indices) > 0 else 0
        
        neighborhood_data.append(neighbor_data)
    
    return pd.DataFrame(neighborhood_data)

def compare_neighborhoods(adata, positive_phenotype, negative_phenotype, radius=50):
    """
    Compare neighborhood composition between positive and negative populations
    """
    print(f"\nComparing neighborhoods: {positive_phenotype} vs {negative_phenotype}")
    
    # Analyze both phenotypes
    pos_neighborhoods = analyze_neighborhood_composition(adata, positive_phenotype, radius)
    neg_neighborhoods = analyze_neighborhood_composition(adata, negative_phenotype, radius)
    
    if pos_neighborhoods.empty or neg_neighborhoods.empty:
        print("Insufficient data for comparison")
        return None
    
    # Add labels
    pos_neighborhoods['phenotype'] = positive_phenotype
    neg_neighborhoods['phenotype'] = negative_phenotype
    
    # Combine
    combined = pd.concat([pos_neighborhoods, neg_neighborhoods], ignore_index=True)
    
    # Statistical comparisons
    comparison_results = {}
    
    fraction_cols = [col for col in combined.columns if col.endswith('_fraction')]
    
    for col in fraction_cols:
        pos_values = pos_neighborhoods[col].values
        neg_values = neg_neighborhoods[col].values
        
        # Mann-Whitney U test
        try:
            statistic, p_value = mannwhitneyu(pos_values, neg_values, alternative='two-sided')
            comparison_results[col] = {
                'positive_median': np.median(pos_values),
                'negative_median': np.median(neg_values),
                'positive_mean': np.mean(pos_values),
                'negative_mean': np.mean(neg_values),
                'p_value': p_value,
                'statistic': statistic,
                'effect_size': (np.mean(pos_values) - np.mean(neg_values)) / np.std(np.concatenate([pos_values, neg_values]))
            }
        except Exception as e:
            print(f"Error in statistical test for {col}: {e}")
    
    return combined, comparison_results

def create_distance_plots(distance_results, target_phenotype, output_dir):
    """Create plots showing distance distributions"""
    
    if not distance_results:
        return
    
    plot_dir = Path(output_dir) / "spatial_analysis" / "distance_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create distance distribution plots
    n_refs = len(distance_results)
    n_cols = min(3, n_refs)
    n_rows = (n_refs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_refs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, (ref_phenotype, results) in enumerate(distance_results.items()):
        ax = axes[idx]
        
        distances = results['distances']
        
        # Plot histogram
        ax.hist(distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(results['median_distance'], color='red', linestyle='--', 
                  label=f'Median: {results["median_distance"]:.1f}px')
        
        ax.set_xlabel('Distance (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{target_phenotype} → {ref_phenotype}\n'
                    f'n={results["n_target_cells"]} target cells')
        ax.legend()
        
        # Add statistics text
        stats_text = f'Within 50px: {np.sum(distances <= 50)}/{len(distances)}\n' \
                    f'Within 100px: {np.sum(distances <= 100)}/{len(distances)}'
        ax.text(0.7, 0.8, stats_text, transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    
    # Remove empty subplots
    for idx in range(n_refs, len(axes)):
        axes[idx].remove()
    
    plt.suptitle(f'Distance Analysis: {target_phenotype}', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plot_path = plot_dir / f'{target_phenotype}_distances.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distance plots saved: {plot_path}")

def create_neighborhood_comparison_plots(combined_data, comparison_results, 
                                       pos_phenotype, neg_phenotype, output_dir):
    """Create plots comparing neighborhood compositions"""
    
    if combined_data is None:
        return
    
    plot_dir = Path(output_dir) / "spatial_analysis" / "neighborhood_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Get fraction columns
    fraction_cols = [col for col in combined_data.columns if col.endswith('_fraction')]
    
    if not fraction_cols:
        return
    
    # Create comparison plots
    n_cols = min(3, len(fraction_cols))
    n_rows = (len(fraction_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if len(fraction_cols) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(fraction_cols):
        ax = axes[idx]
        
        # Box plot
        data_to_plot = [
            combined_data[combined_data['phenotype'] == pos_phenotype][col].values,
            combined_data[combined_data['phenotype'] == neg_phenotype][col].values
        ]
        
        box_plot = ax.boxplot(data_to_plot, labels=[pos_phenotype.replace('_', '\n'), neg_phenotype.replace('_', '\n')],
                             patch_artist=True)
        
        # Color boxes
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        
        cell_type = col.replace('_fraction', '').replace('_', ' ').title()
        ax.set_ylabel('Fraction in Neighborhood')
        ax.set_title(f'{cell_type} Enrichment')
        
        # Add p-value if available
        if col in comparison_results:
            p_val = comparison_results[col]['p_value']
            if p_val < 0.001:
                p_text = 'p < 0.001'
            elif p_val < 0.01:
                p_text = f'p < 0.01'
            elif p_val < 0.05:
                p_text = f'p < 0.05'
            else:
                p_text = f'p = {p_val:.3f}'
            
            ax.text(0.5, 0.95, p_text, transform=ax.transAxes, ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor="yellow" if p_val < 0.05 else "lightgray"))
    
    # Remove empty subplots
    for idx in range(len(fraction_cols), len(axes)):
        axes[idx].remove()
    
    plt.suptitle(f'Neighborhood Comparison: {pos_phenotype} vs {neg_phenotype}', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    plot_path = plot_dir / f'{pos_phenotype}_vs_{neg_phenotype}_neighborhoods.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Neighborhood comparison plots saved: {plot_path}")

def save_analysis_results(distance_results, neighborhood_comparisons, output_dir, experiment_name):
    """Save all analysis results"""
    
    results_dir = Path(output_dir) / "spatial_analysis" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save distance analysis
    distance_summary = []
    for target_phenotype, results_dict in distance_results.items():
        for ref_phenotype, metrics in results_dict.items():
            distance_summary.append({
                'target_phenotype': target_phenotype,
                'reference_phenotype': ref_phenotype,
                'n_target_cells': metrics['n_target_cells'],
                'n_reference_cells': metrics['n_reference_cells'],
                'median_distance': metrics['median_distance'],
                'mean_distance': metrics['mean_distance'],
                'fraction_within_50px': np.sum(metrics['distances'] <= 50) / len(metrics['distances']),
                'fraction_within_100px': np.sum(metrics['distances'] <= 100) / len(metrics['distances'])
            })
    
    if distance_summary:
        distance_df = pd.DataFrame(distance_summary)
        distance_path = results_dir / f"{experiment_name}_distance_analysis.csv"
        distance_df.to_csv(distance_path, index=False)
        print(f"Distance analysis saved: {distance_path}")
    
    # Save neighborhood comparisons
    if neighborhood_comparisons:
        comparison_summary = []
        for comparison_name, (combined_data, comparison_results) in neighborhood_comparisons.items():
            for col, stats in comparison_results.items():
                comparison_summary.append({
                    'comparison': comparison_name,
                    'cell_type': col.replace('_fraction', ''),
                    'positive_mean': stats['positive_mean'],
                    'negative_mean': stats['negative_mean'],
                    'positive_median': stats['positive_median'],
                    'negative_median': stats['negative_median'],
                    'p_value': stats['p_value'],
                    'effect_size': stats['effect_size']
                })
        
        if comparison_summary:
            comparison_df = pd.DataFrame(comparison_summary)
            comparison_path = results_dir / f"{experiment_name}_neighborhood_comparisons.csv"
            comparison_df.to_csv(comparison_path, index=False)
            print(f"Neighborhood comparisons saved: {comparison_path}")

def main():
    parser = argparse.ArgumentParser(description='Spatial neighborhood analysis')
    parser.add_argument('--config', default='/app/spatial_analysis/configs/experiment_config.yaml')
    parser.add_argument('--phenotype-config', default='/app/spatial_analysis/configs/phenotyping_config.yaml')
    parser.add_argument('--input', required=True, help='Path to phenotyped h5ad file')
    parser.add_argument('--output', default='/app/spatial_analysis/outputs')
    
    args = parser.parse_args()
    
    # Load configs
    exp_config, pheno_config = load_configs(args.config, args.phenotype_config)
    print(f"Running spatial analysis for: {exp_config['experiment_name']}")
    
    # Load data
    adata = load_phenotyped_data(args.input)
    
    # Define analysis parameters
    radii = exp_config['analysis']['neighborhood_radii']
    reference_phenotypes = ['t_cells_cd8', 't_cells_cd4', 'immune_non_t']
    
    print(f"\nAnalysis parameters:")
    print(f"  Neighborhood radii: {radii}")
    print(f"  Reference cell types: {reference_phenotypes}")
    
    # Main analyses
    distance_results = {}
    neighborhood_comparisons = {}
    
    print(f"\n{'='*60}")
    print("RESEARCH QUESTION 1: pERK+ vs pERK- tumor cells")
    print(f"{'='*60}")
    
    # Distance analysis for pERK+ tumor cells
    print("\n1.1 Distance analysis: pERK+ tumor cells to immune cells")
    perk_pos_distances = calculate_distances_to_targets(
        adata, 'tumor_perk_pos', reference_phenotypes, max_distance=200
    )
    distance_results['tumor_perk_pos'] = perk_pos_distances
    
    # Create distance plots
    create_distance_plots(perk_pos_distances, 'tumor_perk_pos', args.output)
    
    # Neighborhood comparison: pERK+ vs pERK- 
    print("\n1.2 Neighborhood comparison: pERK+ vs pERK- tumor cells")
    perk_comparison = compare_neighborhoods(
        adata, 'tumor_perk_pos', 'tumor_perk_neg', radius=radii[1]  # Use middle radius
    )
    if perk_comparison is not None:
        neighborhood_comparisons['perk_pos_vs_neg'] = perk_comparison
        create_neighborhood_comparison_plots(
            perk_comparison[0], perk_comparison[1], 
            'tumor_perk_pos', 'tumor_perk_neg', args.output
        )
    
    print(f"\n{'='*60}")
    print("RESEARCH QUESTION 2: NINJA+ vs NINJA- tumor cells")
    print(f"{'='*60}")
    
    # Distance analysis for NINJA+ tumor cells
    print("\n2.1 Distance analysis: NINJA+ tumor cells to immune cells")
    ninja_pos_distances = calculate_distances_to_targets(
        adata, 'tumor_ninja_pos', reference_phenotypes, max_distance=200
    )
    distance_results['tumor_ninja_pos'] = ninja_pos_distances
    
    # Create distance plots
    create_distance_plots(ninja_pos_distances, 'tumor_ninja_pos', args.output)
    
    # Neighborhood comparison: NINJA+ vs NINJA-
    print("\n2.2 Neighborhood comparison: NINJA+ vs NINJA- tumor cells")
    ninja_comparison = compare_neighborhoods(
        adata, 'tumor_ninja_pos', 'tumor_ninja_neg', radius=radii[1]
    )
    if ninja_comparison is not None:
        neighborhood_comparisons['ninja_pos_vs_neg'] = ninja_comparison
        create_neighborhood_comparison_plots(
            ninja_comparison[0], ninja_comparison[1],
            'tumor_ninja_pos', 'tumor_ninja_neg', args.output
        )
    
    # Save all results
    save_analysis_results(distance_results, neighborhood_comparisons, 
                         args.output, exp_config['experiment_name'])
    
    print(f"\n✅ Spatial neighborhood analysis complete!")
    print(f"\nKey findings:")
    
    # Summary of key findings
    if 'tumor_perk_pos' in distance_results:
        perk_cd8_dist = distance_results['tumor_perk_pos'].get('t_cells_cd8', {})
        if perk_cd8_dist:
            print(f"  pERK+ tumor → CD8 T cells: median distance = {perk_cd8_dist['median_distance']:.1f}px")
    
    if 'tumor_ninja_pos' in distance_results:
        ninja_cd8_dist = distance_results['tumor_ninja_pos'].get('t_cells_cd8', {})
        if ninja_cd8_dist:
            print(f"  NINJA+ tumor → CD8 T cells: median distance = {ninja_cd8_dist['median_distance']:.1f}px")
    
    print(f"\nNext step: Run comparative analysis across treatment groups")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())