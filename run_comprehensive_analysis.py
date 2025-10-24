#!/usr/bin/env python3
"""
COMPREHENSIVE Spatial Analysis Runner

This script runs the complete spatial analysis with ALL visualizations and statistics
that were missing from the basic implementation.

Specifically addresses:
1. Proper genotype parsing (cis, trans, KPT Het variants)
2. Spatial maps by sample/timepoint/genotype
3. Tumor size across time and genotype WITH statistics
4. Marker expression (fractional percentages) across time
5. Neighborhood dynamics across time and genotype
6. Multiple visualization formats
7. Comprehensive statistics

Usage:
    python run_comprehensive_analysis.py --config configs/comprehensive_config.yaml

Author: Complete rewrite for comprehensive analysis
Date: 2025-10-24
"""

import argparse
import yaml
import sys
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np

# Set matplotlib to use non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Import the efficient framework
from tumor_spatial_analysis_efficient import EfficientTumorSpatialAnalysis


def parse_metadata_properly(metadata_path: str) -> pd.DataFrame:
    """
    Parse metadata to extract main group (KPT vs KPNT), cis/trans, and full group.

    Handles: "KPNT cis", "KPNT trans", "KPT Het cis", "KPT Het trans"

    Creates:
    - main_group: KPT or KPNT (for 2-group comparisons)
    - genotype: cis or trans (cis/trans status)
    - genotype_full: Full group name (for 4-group comparisons)
    """
    metadata = pd.read_csv(metadata_path)

    # Standardize sample_id casing
    metadata['sample_id'] = metadata['sample_id'].str.upper()

    # Extract main group: KPT or KPNT
    metadata['main_group'] = metadata['group'].apply(
        lambda x: 'KPT' if 'KPT' in str(x) else 'KPNT'
    )

    # Extract cis/trans status
    metadata['genotype'] = metadata['group'].apply(
        lambda x: 'cis' if 'cis' in str(x).lower() else
                 ('trans' if 'trans' in str(x).lower() else 'Unknown')
    )

    # Full group name for detailed 4-group analysis
    metadata['genotype_full'] = metadata['group']

    # Convert timepoint to numeric
    metadata['timepoint'] = pd.to_numeric(metadata['timepoint'])

    print("="*80)
    print("PARSED SAMPLE METADATA")
    print("="*80)
    print(f"Samples: {len(metadata)}")
    print(f"Main groups: {sorted(metadata['main_group'].unique())}")
    print(f"Cis/Trans: {sorted(metadata['genotype'].unique())}")
    print(f"Full groups (4 subgroups): {sorted(metadata['genotype_full'].unique())}")
    print(f"Timepoints: {sorted(metadata['timepoint'].unique())}")
    print("\nSample breakdown:")
    print(metadata.groupby(['main_group', 'genotype', 'timepoint']).size().to_string())
    print("="*80 + "\n")

    return metadata


def merge_metadata_into_adata(adata, metadata: pd.DataFrame):
    """Merge parsed metadata into adata.obs."""
    # Standardize adata sample_id
    adata.obs['sample_id'] = adata.obs['sample_id'].str.upper()

    # Merge all metadata columns including main_group
    for col in ['main_group', 'genotype', 'genotype_full', 'timepoint', 'treatment']:
        if col in metadata.columns:
            mapping = dict(zip(metadata['sample_id'], metadata[col]))
            adata.obs[col] = adata.obs['sample_id'].map(mapping)

    print(f"Merged metadata into adata.obs")
    print(f"  Cells with main_group: {adata.obs['main_group'].notna().sum():,}")
    print(f"  Cells with genotype: {adata.obs['genotype'].notna().sum():,}")
    print(f"  Cells with timepoint: {adata.obs['timepoint'].notna().sum():,}\n")


def create_spatial_maps(adata, output_dir: str, populations: list,
                       max_samples_per_plot: int = 4):
    """
    Create spatial maps showing cell type distributions.

    Creates maps by:
    - Individual samples
    - Timepoints (combined)
    - Genotypes (combined)
    """
    print("\n" + "="*80)
    print("CREATING SPATIAL MAPS")
    print("="*80)

    # Create all required directories
    import os
    os.makedirs(f"{output_dir}/figures/spatial_maps/by_sample", exist_ok=True)
    os.makedirs(f"{output_dir}/figures/spatial_maps/by_timepoint", exist_ok=True)
    os.makedirs(f"{output_dir}/figures/spatial_maps/by_genotype", exist_ok=True)

    coords = adata.obsm['spatial']

    # Color map for populations
    colors = plt.cm.tab20(np.linspace(0, 1, len(populations)))
    pop_colors = {pop: colors[i] for i, pop in enumerate(populations)}

    # 1. Per-sample maps
    print("\n1. Creating per-sample spatial maps...")
    samples = adata.obs['sample_id'].unique()

    for sample in samples:
        sample_mask = adata.obs['sample_id'] == sample
        sample_coords = coords[sample_mask]

        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

        # Plot each population
        for pop in populations:
            if f'is_{pop}' in adata.obs:
                pop_mask = sample_mask & adata.obs[f'is_{pop}']
                if pop_mask.sum() > 0:
                    pop_coords = coords[pop_mask]
                    ax.scatter(pop_coords[:, 0], pop_coords[:, 1],
                             c=[pop_colors[pop]], s=1, alpha=0.6, label=pop)

        ax.set_title(f'{sample} - Spatial Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (μm)', fontsize=12)
        ax.set_ylabel('Y (μm)', fontsize=12)
        ax.legend(markerscale=5, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/spatial_maps/by_sample/{sample}_spatial.png",
                   dpi=150, bbox_inches='tight')
        plt.close()

    print(f"   Created {len(samples)} per-sample spatial maps")

    # 2. Per-timepoint maps (combine samples)
    print("\n2. Creating per-timepoint spatial maps...")
    timepoints = sorted(adata.obs['timepoint'].dropna().unique())

    for tp in timepoints:
        tp_mask = adata.obs['timepoint'] == tp

        fig, ax = plt.subplots(figsize=(14, 12), dpi=150)

        for pop in populations:
            if f'is_{pop}' in adata.obs:
                pop_mask = tp_mask & adata.obs[f'is_{pop}']
                if pop_mask.sum() > 0:
                    pop_coords = coords[pop_mask]
                    # Subsample if too many points
                    if len(pop_coords) > 50000:
                        idx = np.random.choice(len(pop_coords), 50000, replace=False)
                        pop_coords = pop_coords[idx]
                    ax.scatter(pop_coords[:, 0], pop_coords[:, 1],
                             c=[pop_colors[pop]], s=0.5, alpha=0.4, label=pop)

        ax.set_title(f'Timepoint {tp} - Spatial Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (μm)', fontsize=12)
        ax.set_ylabel('Y (μm)', fontsize=12)
        ax.legend(markerscale=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/spatial_maps/by_timepoint/timepoint_{tp}_spatial.png",
                   dpi=150, bbox_inches='tight')
        plt.close()

    print(f"   Created {len(timepoints)} per-timepoint spatial maps")

    # 3. Per-genotype maps
    print("\n3. Creating per-genotype spatial maps...")
    genotypes = adata.obs['genotype'].dropna().unique()

    for geno in genotypes:
        geno_mask = adata.obs['genotype'] == geno

        fig, ax = plt.subplots(figsize=(14, 12), dpi=150)

        for pop in populations:
            if f'is_{pop}' in adata.obs:
                pop_mask = geno_mask & adata.obs[f'is_{pop}']
                if pop_mask.sum() > 0:
                    pop_coords = coords[pop_mask]
                    if len(pop_coords) > 50000:
                        idx = np.random.choice(len(pop_coords), 50000, replace=False)
                        pop_coords = pop_coords[idx]
                    ax.scatter(pop_coords[:, 0], pop_coords[:, 1],
                             c=[pop_colors[pop]], s=0.5, alpha=0.4, label=pop)

        ax.set_title(f'{geno} - Spatial Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (μm)', fontsize=12)
        ax.set_ylabel('Y (μm)', fontsize=12)
        ax.legend(markerscale=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/spatial_maps/by_genotype/{geno}_spatial.png",
                   dpi=150, bbox_inches='tight')
        plt.close()

    print(f"   Created {len(genotypes)} per-genotype spatial maps")
    print("\n✓ Spatial maps complete\n")


def analyze_and_plot_tumor_size(structure_index: pd.DataFrame, output_dir: str):
    """
    Comprehensive tumor size analysis across time and genotype.

    Creates multiple plot formats WITH statistical annotations.
    """
    print("\n" + "="*80)
    print("TUMOR SIZE ANALYSIS")
    print("="*80)

    # Create all required directories
    import os
    os.makedirs(f"{output_dir}/data", exist_ok=True)
    os.makedirs(f"{output_dir}/statistics", exist_ok=True)
    os.makedirs(f"{output_dir}/figures/temporal/tumor_size", exist_ok=True)

    # Aggregate by sample
    sample_agg = structure_index.groupby(['sample_id', 'timepoint', 'main_group',
                                         'genotype', 'genotype_full']).agg({
        'n_cells': ['sum', 'mean', 'count'],
        'area_um2': ['sum', 'mean']
    }).reset_index()

    sample_agg.columns = ['_'.join(col).strip('_') for col in sample_agg.columns.values]
    sample_agg.rename(columns={
        'sample_id_': 'sample_id',
        'timepoint_': 'timepoint',
        'main_group_': 'main_group',
        'genotype_': 'genotype',
        'genotype_full_': 'genotype_full',
        'n_cells_sum': 'total_tumor_cells',
        'n_cells_mean': 'mean_structure_size',
        'n_cells_count': 'n_structures',
        'area_um2_sum': 'total_area',
        'area_um2_mean': 'mean_area'
    }, inplace=True)

    sample_agg.to_csv(f"{output_dir}/data/tumor_size_by_sample.csv", index=False)

    # Statistical tests
    print("\n1. Statistical testing...")

    # Test temporal trends per main_group and genotype_full
    temporal_results = []

    # Main group temporal trends (KPT vs KPNT)
    for group in sample_agg['main_group'].unique():
        group_data = sample_agg[sample_agg['main_group'] == group]

        if len(group_data) >= 3:
            rho_total, p_total = spearmanr(group_data['timepoint'], group_data['total_tumor_cells'])
            rho_mean, p_mean = spearmanr(group_data['timepoint'], group_data['mean_structure_size'])

            temporal_results.append({
                'test_type': 'main_group',
                'group': group,
                'metric': 'total_tumor_cells',
                'spearman_rho': rho_total,
                'p_value': p_total,
                'n_samples': len(group_data)
            })

            temporal_results.append({
                'test_type': 'main_group',
                'group': group,
                'metric': 'mean_structure_size',
                'spearman_rho': rho_mean,
                'p_value': p_mean,
                'n_samples': len(group_data)
            })

    # Full group temporal trends (4 subgroups)
    for geno_full in sample_agg['genotype_full'].unique():
        geno_data = sample_agg[sample_agg['genotype_full'] == geno_full]

        if len(geno_data) >= 3:
            rho_total, p_total = spearmanr(geno_data['timepoint'], geno_data['total_tumor_cells'])
            rho_mean, p_mean = spearmanr(geno_data['timepoint'], geno_data['mean_structure_size'])

            temporal_results.append({
                'test_type': 'subgroup',
                'group': geno_full,
                'metric': 'total_tumor_cells',
                'spearman_rho': rho_total,
                'p_value': p_total,
                'n_samples': len(geno_data)
            })

            temporal_results.append({
                'test_type': 'subgroup',
                'group': geno_full,
                'metric': 'mean_structure_size',
                'spearman_rho': rho_mean,
                'p_value': p_mean,
                'n_samples': len(geno_data)
            })

    temporal_df = pd.DataFrame(temporal_results)
    if len(temporal_df) > 0:
        _, p_adj, _, _ = multipletests(temporal_df['p_value'], method='fdr_bh')
        temporal_df['p_adjusted'] = p_adj
        temporal_df['significant'] = temporal_df['p_adjusted'] < 0.05
        temporal_df.to_csv(f"{output_dir}/statistics/tumor_size_temporal.csv", index=False)

    # Test group differences per timepoint
    genotype_results = []

    # Main group comparisons (KPT vs KPNT) per timepoint
    for tp in sample_agg['timepoint'].unique():
        tp_data = sample_agg[sample_agg['timepoint'] == tp]

        main_groups = tp_data['main_group'].unique()
        if len(main_groups) == 2:
            g1, g2 = main_groups[0], main_groups[1]
            d1 = tp_data[tp_data['main_group'] == g1]['total_tumor_cells'].values
            d2 = tp_data[tp_data['main_group'] == g2]['total_tumor_cells'].values

            if len(d1) >= 1 and len(d2) >= 1:
                if len(d1) >= 2 and len(d2) >= 2:
                    stat, p = mannwhitneyu(d1, d2, alternative='two-sided')
                else:
                    stat, p = np.nan, np.nan

                genotype_results.append({
                    'test_type': 'main_group',
                    'timepoint': tp,
                    'group_1': g1,
                    'group_2': g2,
                    'metric': 'total_tumor_cells',
                    'mean_1': d1.mean(),
                    'mean_2': d2.mean(),
                    'fold_change': d2.mean() / (d1.mean() + 1),
                    'statistic': stat,
                    'p_value': p
                })

    # 4-subgroup pairwise comparisons per timepoint
    for tp in sample_agg['timepoint'].unique():
        tp_data = sample_agg[sample_agg['timepoint'] == tp]
        full_groups = tp_data['genotype_full'].unique()

        for i, g1 in enumerate(full_groups):
            for g2 in full_groups[i+1:]:
                d1 = tp_data[tp_data['genotype_full'] == g1]['total_tumor_cells'].values
                d2 = tp_data[tp_data['genotype_full'] == g2]['total_tumor_cells'].values

                if len(d1) >= 1 and len(d2) >= 1:
                    if len(d1) >= 2 and len(d2) >= 2:
                        stat, p = mannwhitneyu(d1, d2, alternative='two-sided')
                    else:
                        stat, p = np.nan, np.nan

                    genotype_results.append({
                        'test_type': 'subgroup',
                        'timepoint': tp,
                        'group_1': g1,
                        'group_2': g2,
                        'metric': 'total_tumor_cells',
                        'mean_1': d1.mean(),
                        'mean_2': d2.mean(),
                        'fold_change': d2.mean() / (d1.mean() + 1),
                        'statistic': stat,
                        'p_value': p
                    })

    genotype_df = pd.DataFrame(genotype_results)
    if len(genotype_df) > 0:
        _, p_adj, _, _ = multipletests(genotype_df['p_value'], method='fdr_bh')
        genotype_df['p_adjusted'] = p_adj
        genotype_df['significant'] = genotype_df['p_adjusted'] < 0.05
        genotype_df.to_csv(f"{output_dir}/statistics/tumor_size_genotype.csv", index=False)

    print(f"   Temporal tests: {len(temporal_df)} tests")
    print(f"   Genotype tests: {len(genotype_df)} tests")

    # 2. Visualizations
    print("\n2. Creating visualizations...")

    # Plot 1: Total tumor cells over time - Main groups (KPT vs KPNT)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Plot 1a: Main group line plot (KPT vs KPNT)
    ax = axes[0]
    for group in sample_agg['main_group'].unique():
        group_data = sample_agg[sample_agg['main_group'] == group]
        group_mean = group_data.groupby('timepoint')['total_tumor_cells'].agg(['mean', 'sem'])

        ax.errorbar(group_mean.index, group_mean['mean'], yerr=group_mean['sem'],
                   marker='o', linewidth=2, markersize=8, capsize=5, label=group)

    ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Tumor Cells', fontsize=12, fontweight='bold')
    ax.set_title('Tumor Growth: KPT vs KPNT', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 1b: 4 subgroups
    ax = axes[1]
    for geno_full in sample_agg['genotype_full'].unique():
        geno_data = sample_agg[sample_agg['genotype_full'] == geno_full]
        geno_mean = geno_data.groupby('timepoint')['total_tumor_cells'].agg(['mean', 'sem'])

        ax.errorbar(geno_mean.index, geno_mean['mean'], yerr=geno_mean['sem'],
                   marker='s', linewidth=2, markersize=8, capsize=5, label=geno_full)

    ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Tumor Cells', fontsize=12, fontweight='bold')
    ax.set_title('Tumor Growth: All 4 Subgroups', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/temporal/tumor_size/tumor_size_temporal_line.png",
               dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Boxplots - Main groups (KPT vs KPNT)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=300)

    ax = axes[0]
    sns.boxplot(data=sample_agg, x='timepoint', y='total_tumor_cells', hue='main_group',
               ax=ax, palette='Set2')
    ax.set_title('Total Tumor Cells: KPT vs KPNT', fontsize=14, fontweight='bold')
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Total Tumor Cells', fontsize=12)

    ax = axes[1]
    sns.boxplot(data=sample_agg, x='timepoint', y='mean_structure_size', hue='main_group',
               ax=ax, palette='Set2')
    ax.set_title('Mean Structure Size: KPT vs KPNT', fontsize=14, fontweight='bold')
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Mean Structure Size (cells)', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/temporal/tumor_size/tumor_size_boxplots_maingroup.png",
               dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Boxplots - All 4 subgroups
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=300)

    ax = axes[0]
    sns.boxplot(data=sample_agg, x='timepoint', y='total_tumor_cells', hue='genotype_full',
               ax=ax, palette='Set3')
    ax.set_title('Total Tumor Cells: All 4 Subgroups', fontsize=14, fontweight='bold')
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Total Tumor Cells', fontsize=12)

    ax = axes[1]
    sns.boxplot(data=sample_agg, x='timepoint', y='mean_structure_size', hue='genotype_full',
               ax=ax, palette='Set3')
    ax.set_title('Mean Structure Size: All 4 Subgroups', fontsize=14, fontweight='bold')
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Mean Structure Size (cells)', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/temporal/tumor_size/tumor_size_boxplots_subgroups.png",
               dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: Violin plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300)

    ax = axes[0]
    sns.violinplot(data=sample_agg, x='timepoint', y='total_tumor_cells', hue='main_group',
                  ax=ax, palette='Set2', split=False)
    ax.set_title('KPT vs KPNT Distribution',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Total Tumor Cells', fontsize=12)

    ax = axes[1]
    sns.violinplot(data=sample_agg, x='timepoint', y='total_tumor_cells', hue='genotype_full',
                  ax=ax, palette='Set3', split=False)
    ax.set_title('All 4 Subgroups Distribution',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Total Tumor Cells', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/temporal/tumor_size/tumor_size_violin.png",
               dpi=300, bbox_inches='tight')
    plt.close()

    print("   Created tumor size temporal plots")
    print("\n✓ Tumor size analysis complete\n")

    return sample_agg


def analyze_and_plot_marker_expression(adata, output_dir: str, markers: list):
    """
    Analyze marker expression (fractional percentages) across time and genotype.

    Creates comprehensive temporal plots for ALL markers.
    """
    print("\n" + "="*80)
    print("MARKER EXPRESSION ANALYSIS")
    print("="*80)

    # Create all required directories
    import os
    os.makedirs(f"{output_dir}/data", exist_ok=True)
    os.makedirs(f"{output_dir}/figures/temporal/marker_expression", exist_ok=True)

    results = []

    for marker in markers:
        if marker not in adata.var_names:
            continue

        marker_idx = adata.var_names.get_loc(marker)

        # Get positivity
        if 'gated' in adata.layers:
            values = adata.layers['gated'][:, marker_idx]
            positivity = values > 0
        else:
            values = adata.X[:, marker_idx]
            threshold = np.percentile(values[values > 0], 90) if np.any(values > 0) else 0
            positivity = values > threshold

        # Aggregate by sample
        for sample in adata.obs['sample_id'].unique():
            sample_mask = adata.obs['sample_id'] == sample
            sample_pos = positivity[sample_mask]

            # Get metadata
            sample_meta = adata.obs[sample_mask].iloc[0]

            results.append({
                'marker': marker,
                'sample_id': sample,
                'timepoint': sample_meta.get('timepoint'),
                'main_group': sample_meta.get('main_group'),
                'genotype': sample_meta.get('genotype'),
                'genotype_full': sample_meta.get('genotype_full'),
                'n_cells': len(sample_pos),
                'n_positive': sample_pos.sum(),
                'pct_positive': 100 * sample_pos.sum() / len(sample_pos)
            })

    marker_df = pd.DataFrame(results)
    marker_df.to_csv(f"{output_dir}/data/marker_expression_temporal.csv", index=False)

    print(f"Analyzed {len(markers)} markers across {len(marker_df['sample_id'].unique())} samples")

    # Create plots for each marker
    print("\nCreating marker expression plots...")

    for marker in marker_df['marker'].unique():
        marker_data = marker_df[marker_df['marker'] == marker]

        # Create 2 plots: main_group and genotype_full
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

        # Row 1: Main group (KPT vs KPNT)
        ax = axes[0, 0]
        for group in marker_data['main_group'].dropna().unique():
            group_data = marker_data[marker_data['main_group'] == group]
            group_mean = group_data.groupby('timepoint')['pct_positive'].agg(['mean', 'sem'])

            ax.errorbar(group_mean.index, group_mean['mean'], yerr=group_mean['sem'],
                       marker='o', linewidth=2, markersize=8, capsize=5, label=group)

        ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'% {marker}+ Cells', fontsize=12, fontweight='bold')
        ax.set_title(f'{marker}: KPT vs KPNT', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        sns.boxplot(data=marker_data, x='timepoint', y='pct_positive', hue='main_group',
                   ax=ax, palette='Set2')
        ax.set_xlabel('Timepoint', fontsize=12)
        ax.set_ylabel(f'% {marker}+ Cells', fontsize=12)
        ax.set_title(f'{marker}: KPT vs KPNT Distribution', fontsize=14, fontweight='bold')

        # Row 2: All 4 subgroups
        ax = axes[1, 0]
        for geno_full in marker_data['genotype_full'].dropna().unique():
            geno_data = marker_data[marker_data['genotype_full'] == geno_full]
            geno_mean = geno_data.groupby('timepoint')['pct_positive'].agg(['mean', 'sem'])

            ax.errorbar(geno_mean.index, geno_mean['mean'], yerr=geno_mean['sem'],
                       marker='s', linewidth=2, markersize=8, capsize=5, label=geno_full)

        ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'% {marker}+ Cells', fontsize=12, fontweight='bold')
        ax.set_title(f'{marker}: All 4 Subgroups', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        sns.boxplot(data=marker_data, x='timepoint', y='pct_positive', hue='genotype_full',
                   ax=ax, palette='Set3')
        ax.set_xlabel('Timepoint', fontsize=12)
        ax.set_ylabel(f'% {marker}+ Cells', fontsize=12)
        ax.set_title(f'{marker}: All 4 Subgroups Distribution', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/temporal/marker_expression/{marker}_temporal.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

    print(f"   Created plots for {len(marker_df['marker'].unique())} markers")
    print("\n✓ Marker expression analysis complete\n")

    return marker_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive tumor spatial analysis with ALL visualizations'
    )

    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Configuration file')
    parser.add_argument('--metadata', '-m', type=str, default='sample_metadata.csv',
                       help='Sample metadata CSV')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load and parse metadata properly
    metadata = parse_metadata_properly(args.metadata)

    # Load data
    print("Loading data...")
    adata = sc.read_h5ad(config['input_data'])
    print(f"Loaded {len(adata):,} cells\n")

    # Merge metadata
    merge_metadata_into_adata(adata, metadata)

    # Run efficient framework for structure detection and infiltration
    output_dir = config.get('output_directory', 'comprehensive_spatial_analysis')

    # Create base directories
    import os
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/structures", exist_ok=True)

    etsa = EfficientTumorSpatialAnalysis(
        adata,
        sample_metadata=metadata,
        tumor_markers=config['tumor_markers'],
        immune_markers=config['immune_markers'],
        output_dir=output_dir
    )

    # Run analysis
    population_config = {}
    for pop_name, pop_def in config['populations'].items():
        population_config[pop_name] = {
            'markers': pop_def['markers'],
            'color': pop_def.get('color', '#999999')
        }

    # Phase 1: Structures
    struct_config = config.get('tumor_structure_detection', {})
    etsa.detect_all_tumor_structures(
        population_config=population_config,
        **struct_config
    )

    # Phase 2: Infiltration
    infil_config = config.get('immune_infiltration', {})
    boundary_config = config.get('infiltration_boundaries', {})

    metrics_df = etsa.analyze_structures_individually(
        immune_populations=infil_config.get('populations', []),
        boundary_widths=boundary_config.get('boundary_widths', [30, 100, 200]),
        buffer_distance=config.get('buffer_distance', 500)
    )

    # Phase 3: ALL ADDITIONAL ANALYSES
    # 3a. Spatial maps
    create_spatial_maps(adata, output_dir, list(population_config.keys()))

    # 3b. Tumor size analysis
    size_df = analyze_and_plot_tumor_size(etsa.structure_index, output_dir)

    # 3c. Marker expression
    all_markers = config['tumor_markers'] + config['immune_markers']
    marker_df = analyze_and_plot_marker_expression(adata, output_dir, all_markers)

    # Phase 4: Neighborhoods
    neighborhood_config = config.get('cellular_neighborhoods', {})
    if neighborhood_config.get('enabled', True):
        etsa.detect_cellular_neighborhoods(
            populations=list(population_config.keys()),
            **{k: v for k, v in neighborhood_config.items() if k not in ['enabled', 'populations']}
        )

    # Phase 5: Statistics
    if config.get('statistical_analysis', {}).get('enabled', True):
        etsa.statistical_analysis(metrics_df)

    # Phase 6: Remaining visualizations
    if config.get('visualizations', {}).get('enabled', True):
        etsa.create_publication_figures(metrics_df)

    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs in: {output_dir}/")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
