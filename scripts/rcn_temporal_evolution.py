#!/usr/bin/env python3
"""
RCN Temporal Evolution Analysis
Following Schurch et al. and Nirmal et al. methodologies for longitudinal RCN tracking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import anndata as ad
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import adjusted_rand_score
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Publication settings
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'font.family': 'Arial',
    'axes.linewidth': 1.5,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False
})

def load_temporal_data(input_dir):
    """Load all temporal datasets with RCN assignments"""
    data_files = list(Path(input_dir).glob("*/*_phenotyped.h5ad"))
    
    temporal_data = {}
    for file_path in data_files:
        try:
            adata = ad.read_h5ad(file_path)
            sample_id = adata.obs['sample_id'].iloc[0] if 'sample_id' in adata.obs.columns else file_path.stem
            
            # Extract timepoint and other metadata
            if 'timepoint' in adata.obs.columns:
                timepoint = adata.obs['timepoint'].iloc[0]
            else:
                # Try to extract from filename
                timepoint = extract_timepoint_from_name(file_path.stem)
            
            temporal_data[f"{sample_id}_{timepoint}"] = {
                'adata': adata,
                'sample_id': sample_id,
                'timepoint': timepoint,
                'file_path': file_path
            }
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    print(f"Loaded {len(temporal_data)} temporal datasets")
    return temporal_data

def extract_timepoint_from_name(filename):
    """Extract timepoint from filename patterns"""
    import re
    
    # Common patterns: D7, day7, 7d, 7day, week1, etc.
    patterns = [
        r'[Dd](\d+)',
        r'(\d+)[Dd]',
        r'day(\d+)',
        r'(\d+)day',
        r'week(\d+)',
        r'(\d+)w',
        r't(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    
    return 0  # Default if no timepoint found

def analyze_rcn_temporal_evolution(temporal_data, output_dir):
    """Analyze how RCN composition evolves over time"""
    
    plot_dir = Path(output_dir) / "rcn_temporal_evolution"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by sample and sort by timepoint
    sample_groups = {}
    for key, data in temporal_data.items():
        sample_id = data['sample_id']
        if sample_id not in sample_groups:
            sample_groups[sample_id] = []
        sample_groups[sample_id].append(data)
    
    # Sort each group by timepoint
    for sample_id in sample_groups:
        sample_groups[sample_id].sort(key=lambda x: x['timepoint'])
    
    # Create temporal evolution plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: RCN prevalence over time
    plot_rcn_prevalence_over_time(sample_groups, axes[0,0])
    
    # Plot 2: RCN diversity over time
    plot_rcn_diversity_over_time(sample_groups, axes[0,1])
    
    # Plot 3: Individual RCN trajectories
    plot_individual_rcn_trajectories(sample_groups, axes[1,0])
    
    # Plot 4: RCN stability analysis
    plot_rcn_stability(sample_groups, axes[1,1])
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'rcn_temporal_evolution_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate detailed temporal analysis
    generate_detailed_temporal_analysis(sample_groups, plot_dir)
    
    return sample_groups

def plot_rcn_prevalence_over_time(sample_groups, ax):
    """Plot RCN prevalence changes over time"""
    
    all_timepoints = []
    all_prevalences = []
    all_samples = []
    all_rcns = []
    
    for sample_id, time_series in sample_groups.items():
        for data in time_series:
            if 'RCN' not in data['adata'].obs.columns:
                continue
                
            timepoint = data['timepoint']
            rcn_counts = data['adata'].obs['RCN'].value_counts(normalize=True)
            
            for rcn, prevalence in rcn_counts.items():
                all_timepoints.append(timepoint)
                all_prevalences.append(prevalence * 100)
                all_samples.append(sample_id)
                all_rcns.append(f'RCN {rcn}')
    
    if all_timepoints:
        df = pd.DataFrame({
            'timepoint': all_timepoints,
            'prevalence': all_prevalences,
            'sample': all_samples,
            'RCN': all_rcns
        })
        
        # Plot mean trajectories
        for rcn in df['RCN'].unique():
            rcn_data = df[df['RCN'] == rcn]
            mean_prev = rcn_data.groupby('timepoint')['prevalence'].agg(['mean', 'std']).reset_index()
            
            ax.plot(mean_prev['timepoint'], mean_prev['mean'], 
                   marker='o', linewidth=2, label=rcn, markersize=6)
            ax.fill_between(mean_prev['timepoint'], 
                           mean_prev['mean'] - mean_prev['std'],
                           mean_prev['mean'] + mean_prev['std'], alpha=0.2)
    
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('RCN Prevalence (%)')
    ax.set_title('RCN Prevalence Evolution Over Time', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

def plot_rcn_diversity_over_time(sample_groups, ax):
    """Plot RCN diversity (Shannon entropy) over time"""
    
    diversity_data = []
    
    for sample_id, time_series in sample_groups.items():
        for data in time_series:
            if 'RCN' not in data['adata'].obs.columns:
                continue
                
            rcn_counts = data['adata'].obs['RCN'].value_counts(normalize=True)
            shannon_entropy = -np.sum(rcn_counts * np.log(rcn_counts + 1e-10))
            
            diversity_data.append({
                'sample': sample_id,
                'timepoint': data['timepoint'],
                'shannon_entropy': shannon_entropy
            })
    
    if diversity_data:
        df = pd.DataFrame(diversity_data)
        
        # Plot individual trajectories
        for sample in df['sample'].unique():
            sample_data = df[df['sample'] == sample].sort_values('timepoint')
            ax.plot(sample_data['timepoint'], sample_data['shannon_entropy'], 
                   'o-', alpha=0.6, linewidth=1, markersize=4)
        
        # Plot mean trajectory
        mean_diversity = df.groupby('timepoint')['shannon_entropy'].agg(['mean', 'std']).reset_index()
        ax.plot(mean_diversity['timepoint'], mean_diversity['mean'], 
               'ko-', linewidth=3, markersize=8, label='Mean')
        ax.fill_between(mean_diversity['timepoint'],
                       mean_diversity['mean'] - mean_diversity['std'],
                       mean_diversity['mean'] + mean_diversity['std'], 
                       alpha=0.3, color='black')
    
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Shannon Entropy')
    ax.set_title('RCN Diversity Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_individual_rcn_trajectories(sample_groups, ax):
    """Plot individual RCN trajectories showing composition changes"""
    
    # Get the most prevalent RCNs
    all_rcns = set()
    for sample_id, time_series in sample_groups.items():
        for data in time_series:
            if 'RCN' in data['adata'].obs.columns:
                all_rcns.update(data['adata'].obs['RCN'].unique())
    
    top_rcns = sorted(list(all_rcns))[:6]  # Show top 6 RCNs
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_rcns)))
    
    for i, rcn in enumerate(top_rcns):
        rcn_trajectories = []
        timepoints = []
        
        for sample_id, time_series in sample_groups.items():
            sample_timepoints = []
            sample_prevalences = []
            
            for data in time_series:
                if 'RCN' not in data['adata'].obs.columns:
                    continue
                    
                rcn_prev = (data['adata'].obs['RCN'] == rcn).mean() * 100
                sample_timepoints.append(data['timepoint'])
                sample_prevalences.append(rcn_prev)
            
            if sample_timepoints:
                ax.plot(sample_timepoints, sample_prevalences, 
                       color=colors[i], alpha=0.3, linewidth=1)
                rcn_trajectories.extend(sample_prevalences)
                timepoints.extend(sample_timepoints)
        
        # Plot mean trajectory
        if timepoints:
            df_temp = pd.DataFrame({'timepoint': timepoints, 'prevalence': rcn_trajectories})
            mean_traj = df_temp.groupby('timepoint')['prevalence'].mean().reset_index()
            ax.plot(mean_traj['timepoint'], mean_traj['prevalence'], 
                   color=colors[i], linewidth=3, label=f'RCN {rcn}', marker='o')
    
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('RCN Prevalence (%)')
    ax.set_title('Individual RCN Trajectories', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_rcn_stability(sample_groups, ax):
    """Analyze RCN stability between consecutive timepoints"""
    
    stability_scores = []
    timepoint_pairs = []
    
    for sample_id, time_series in sample_groups.items():
        if len(time_series) < 2:
            continue
            
        for i in range(len(time_series) - 1):
            data1 = time_series[i]
            data2 = time_series[i + 1]
            
            if 'RCN' not in data1['adata'].obs.columns or 'RCN' not in data2['adata'].obs.columns:
                continue
            
            # Calculate adjusted rand index between consecutive timepoints
            # Use spatial overlap for cells in similar locations
            coords1 = data1['adata'].obsm['spatial']
            coords2 = data2['adata'].obsm['spatial']
            rcn1 = data1['adata'].obs['RCN'].values
            rcn2 = data2['adata'].obs['RCN'].values
            
            # Find overlapping regions (simplified approach)
            from scipy.spatial.distance import cdist
            distances = cdist(coords1, coords2)
            overlap_threshold = 50  # 50 μm overlap threshold
            
            overlap_pairs = np.where(distances < overlap_threshold)
            if len(overlap_pairs[0]) > 10:  # Need minimum overlap
                overlapping_rcn1 = rcn1[overlap_pairs[0]]
                overlapping_rcn2 = rcn2[overlap_pairs[1]]
                
                stability = adjusted_rand_score(overlapping_rcn1, overlapping_rcn2)
                stability_scores.append(stability)
                timepoint_pairs.append(f"{data1['timepoint']}-{data2['timepoint']}")
    
    if stability_scores:
        bars = ax.bar(range(len(stability_scores)), stability_scores, 
                     color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(timepoint_pairs)))
        ax.set_xticklabels(timepoint_pairs, rotation=45)
        ax.set_ylabel('RCN Stability (Adjusted Rand Index)')
        ax.set_title('RCN Stability Between Timepoints', fontweight='bold')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, stability_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.grid(True, alpha=0.3)

def generate_detailed_temporal_analysis(sample_groups, output_dir):
    """Generate detailed temporal analysis tables and plots"""
    
    # RCN transition matrix
    generate_rcn_transition_matrix(sample_groups, output_dir)
    
    # Temporal statistics table
    generate_temporal_statistics_table(sample_groups, output_dir)
    
    # RCN composition evolution heatmaps
    generate_rcn_composition_evolution(sample_groups, output_dir)

def generate_rcn_transition_matrix(sample_groups, output_dir):
    """Generate RCN transition probability matrices"""
    
    transition_data = []
    
    for sample_id, time_series in sample_groups.items():
        for i in range(len(time_series) - 1):
            data1 = time_series[i]
            data2 = time_series[i + 1]
            
            if 'RCN' not in data1['adata'].obs.columns or 'RCN' not in data2['adata'].obs.columns:
                continue
            
            # Track transitions (simplified - based on spatial proximity)
            coords1 = data1['adata'].obsm['spatial']
            coords2 = data2['adata'].obsm['spatial']
            rcn1 = data1['adata'].obs['RCN'].values
            rcn2 = data2['adata'].obs['RCN'].values
            
            from scipy.spatial.distance import cdist
            distances = cdist(coords1, coords2)
            
            # Find nearest neighbors between timepoints
            nearest_indices = np.argmin(distances, axis=1)
            for j, nearest_idx in enumerate(nearest_indices):
                if distances[j, nearest_idx] < 50:  # Within 50μm
                    transition_data.append({
                        'from_rcn': rcn1[j],
                        'to_rcn': rcn2[nearest_idx],
                        'sample': sample_id,
                        'transition': f"{data1['timepoint']}-{data2['timepoint']}"
                    })
    
    if transition_data:
        df = pd.DataFrame(transition_data)
        
        # Create transition matrix
        transition_matrix = pd.crosstab(df['from_rcn'], df['to_rcn'], normalize='index')
        
        # Plot transition matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='Blues',
                   cbar_kws={'label': 'Transition Probability'})
        plt.title('RCN Transition Probability Matrix', fontweight='bold', pad=20)
        plt.xlabel('To RCN')
        plt.ylabel('From RCN')
        plt.tight_layout()
        plt.savefig(output_dir / 'rcn_transition_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save transition data
        transition_matrix.to_csv(output_dir / 'rcn_transition_matrix.csv')

def generate_temporal_statistics_table(sample_groups, output_dir):
    """Generate comprehensive temporal statistics table"""
    
    stats_data = []
    
    for sample_id, time_series in sample_groups.items():
        for data in time_series:
            if 'RCN' not in data['adata'].obs.columns:
                continue
            
            rcn_counts = data['adata'].obs['RCN'].value_counts()
            rcn_proportions = rcn_counts / rcn_counts.sum()
            shannon_entropy = -np.sum(rcn_proportions * np.log(rcn_proportions + 1e-10))
            
            # Calculate additional metrics
            n_rcns = len(rcn_counts)
            dominant_rcn = rcn_counts.index[0]
            dominant_proportion = rcn_proportions.iloc[0]
            
            stats_data.append({
                'sample_id': sample_id,
                'timepoint': data['timepoint'],
                'total_cells': len(data['adata']),
                'n_unique_rcns': n_rcns,
                'shannon_entropy': shannon_entropy,
                'dominant_rcn': dominant_rcn,
                'dominant_proportion': dominant_proportion,
                'evenness': shannon_entropy / np.log(n_rcns) if n_rcns > 1 else 1.0
            })
    
    if stats_data:
        df = pd.DataFrame(stats_data)
        df.to_csv(output_dir / 'temporal_statistics_table.csv', index=False)
        
        print(f"Temporal statistics saved: {output_dir / 'temporal_statistics_table.csv'}")

def generate_rcn_composition_evolution(sample_groups, output_dir):
    """Generate RCN composition evolution heatmaps"""
    
    # For each sample, create a heatmap showing RCN prevalence over time
    for sample_id, time_series in sample_groups.items():
        if len(time_series) < 2:
            continue
        
        timepoints = []
        rcn_data = []
        
        for data in time_series:
            if 'RCN' not in data['adata'].obs.columns:
                continue
            
            timepoints.append(data['timepoint'])
            rcn_counts = data['adata'].obs['RCN'].value_counts(normalize=True)
            
            # Ensure all RCNs are represented
            all_rcns = sorted([str(i) for i in range(12)])  # Assuming 12 RCNs
            rcn_vector = [rcn_counts.get(rcn, 0) for rcn in all_rcns]
            rcn_data.append(rcn_vector)
        
        if len(rcn_data) >= 2:
            # Create heatmap
            rcn_matrix = np.array(rcn_data).T
            
            plt.figure(figsize=(max(8, len(timepoints)*1.5), 10))
            sns.heatmap(rcn_matrix, 
                       xticklabels=[f'T{tp}' for tp in timepoints],
                       yticklabels=[f'RCN {i+1}' for i in range(len(all_rcns))],
                       annot=True, fmt='.2f', cmap='Reds',
                       cbar_kws={'label': 'RCN Prevalence'})
            plt.title(f'RCN Composition Evolution - {sample_id}', 
                     fontweight='bold', pad=20)
            plt.xlabel('Timepoint')
            plt.ylabel('RCN')
            plt.tight_layout()
            plt.savefig(output_dir / f'rcn_evolution_{sample_id}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='RCN temporal evolution analysis')
    parser.add_argument('--input', required=True, 
                       help='Directory containing temporal datasets with RCN assignments')
    parser.add_argument('--output', default='/app/spatial_analysis/temporal_outputs')
    
    args = parser.parse_args()
    
    print("RCN Temporal Evolution Analysis")
    print("="*50)
    
    # Load temporal data
    temporal_data = load_temporal_data(args.input)
    
    if not temporal_data:
        print("No temporal data found with RCN assignments")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run temporal evolution analysis
    sample_groups = analyze_rcn_temporal_evolution(temporal_data, output_dir)
    
    print(f"\nTemporal evolution analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("\nGenerated outputs:")
    print("- rcn_temporal_evolution_summary.png: Overview of RCN changes")
    print("- rcn_transition_matrix.png: RCN transition probabilities")
    print("- rcn_evolution_[sample].png: Individual sample trajectories")
    print("- temporal_statistics_table.csv: Quantitative temporal metrics")

if __name__ == "__main__":
    main()