#!/usr/bin/env python3
"""
COMPREHENSIVE Spatial Analysis Runner - FULLY INTEGRATED

This script runs the complete spatial analysis with ALL visualizations and statistics.

Features implemented:
1. ✅ Proper 2×2 factorial design parsing (KPT/KPNT × cis/trans)
2. ✅ Multi-level analysis (KPT vs KPNT AND all 4 groups)
3. ✅ Spatial maps by sample/timepoint/model/4-way groups
4. ✅ Tumor size temporal analysis with comprehensive statistics
5. ✅ Marker expression (fractional %) for ALL markers across time
6. ✅ Immune infiltration temporal dynamics (multi-level)
7. ✅ Neighborhood temporal dynamics and heatmaps
8. ✅ Comprehensive heatmaps (infiltration, markers, tumor size)
9. ✅ Multi-panel summary dashboard
10. ✅ Multiple visualization formats (line, box, violin, heatmap)
11. ✅ FDR-corrected statistics throughout

Output structure:
    comprehensive_spatial_analysis/
    ├── data/
    │   ├── tumor_size_by_sample.csv
    │   ├── marker_expression_temporal.csv
    │   ├── neighborhood_temporal.csv
    │   └── infiltration_metrics.csv
    ├── statistics/
    │   ├── tumor_size_temporal_KPT_vs_KPNT.csv
    │   ├── tumor_size_KPT_vs_KPNT_by_timepoint.csv
    │   ├── tumor_size_temporal_4way.csv
    │   └── tumor_size_4way_by_timepoint.csv
    └── figures/
        ├── spatial_maps/ (by sample, timepoint, model, 4-way groups)
        ├── temporal/ (tumor size, marker expression, infiltration)
        ├── neighborhoods/ (temporal dynamics, heatmaps)
        ├── heatmaps/ (comprehensive summaries)
        └── combined/ (summary dashboard)

Usage:
    python run_comprehensive_analysis.py \\
        --config configs/comprehensive_config.yaml \\
        --metadata sample_metadata.csv

Author: Comprehensive analysis framework
Date: 2025-10-24
"""

import argparse
import yaml
import sys
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Import the efficient framework
from tumor_spatial_analysis_efficient import EfficientTumorSpatialAnalysis

# Import comprehensive additions
from comprehensive_analysis_additions import (
    analyze_and_plot_infiltration_comprehensive,
    analyze_neighborhoods_temporal,
    create_comprehensive_heatmaps,
    create_summary_dashboard
)


def parse_metadata_properly(metadata_path: str) -> pd.DataFrame:
    """
    Parse metadata with proper experimental design extraction.

    Experimental design:
    - Two main groups: KPT, KPNT
    - Each with genotypes: cis, trans
    - Total: 4 groups (KPT Het cis, KPT Het trans, KPNT cis, KPNT trans)
    """
    metadata = pd.read_csv(metadata_path)

    # Standardize sample_id casing
    metadata['sample_id'] = metadata['sample_id'].str.upper()

    # Parse model (KPT vs KPNT)
    metadata['model'] = metadata['group'].apply(
        lambda x: 'KPT' if 'KPT' in str(x) else 'KPNT'
    )

    # Parse genotype (cis vs trans)
    metadata['genotype'] = metadata['group'].apply(
        lambda x: 'cis' if 'cis' in str(x).lower() else
                 ('trans' if 'trans' in str(x).lower() else 'Unknown')
    )

    # Combined group for 4-way analysis
    metadata['group_full'] = metadata['group']

    # Simplified combined for 2-way model comparison
    metadata['model_genotype'] = metadata['model'] + '_' + metadata['genotype']

    # Convert timepoint to numeric
    metadata['timepoint'] = pd.to_numeric(metadata['timepoint'])

    print("="*80)
    print("PARSED SAMPLE METADATA - EXPERIMENTAL DESIGN")
    print("="*80)
    print(f"Total samples: {len(metadata)}")
    print(f"\nMain groups (model):")
    print(f"  {metadata['model'].value_counts().to_dict()}")
    print(f"\nGenotypes:")
    print(f"  {metadata['genotype'].value_counts().to_dict()}")
    print(f"\n4-way groups:")
    print(f"  {metadata['model_genotype'].value_counts().to_dict()}")
    print(f"\nTimepoints: {sorted(metadata['timepoint'].unique())}")

    print("\nSample breakdown by model, genotype, and timepoint:")
    breakdown = metadata.groupby(['model', 'genotype', 'timepoint']).size().unstack(fill_value=0)
    print(breakdown)

    print("\nFull group names:")
    for grp in sorted(metadata['group_full'].unique()):
        n = (metadata['group_full'] == grp).sum()
        print(f"  {grp}: {n} samples")

    print("="*80 + "\n")

    return metadata


def merge_metadata_into_adata(adata, metadata: pd.DataFrame):
    """Merge parsed metadata into adata.obs."""
    # Standardize adata sample_id
    adata.obs['sample_id'] = adata.obs['sample_id'].str.upper()

    # Merge all metadata columns
    for col in ['model', 'genotype', 'group_full', 'model_genotype', 'timepoint', 'treatment']:
        if col in metadata.columns:
            mapping = dict(zip(metadata['sample_id'], metadata[col]))
            adata.obs[col] = adata.obs['sample_id'].map(mapping)

    print(f"Merged metadata into adata.obs")
    print(f"  Cells with model: {adata.obs['model'].notna().sum():,}")
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

    # 3. Per-model maps (KPT vs KPNT)
    print("\n3. Creating per-model spatial maps (KPT vs KPNT)...")
    models = adata.obs['model'].dropna().unique()

    for model in models:
        model_mask = adata.obs['model'] == model

        fig, ax = plt.subplots(figsize=(14, 12), dpi=150)

        for pop in populations:
            if f'is_{pop}' in adata.obs:
                pop_mask = model_mask & adata.obs[f'is_{pop}']
                if pop_mask.sum() > 0:
                    pop_coords = coords[pop_mask]
                    if len(pop_coords) > 50000:
                        idx = np.random.choice(len(pop_coords), 50000, replace=False)
                        pop_coords = pop_coords[idx]
                    ax.scatter(pop_coords[:, 0], pop_coords[:, 1],
                             c=[pop_colors[pop]], s=0.5, alpha=0.4, label=pop)

        ax.set_title(f'{model} - Spatial Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (μm)', fontsize=12)
        ax.set_ylabel('Y (μm)', fontsize=12)
        ax.legend(markerscale=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/spatial_maps/by_genotype/{model}_spatial.png",
                   dpi=150, bbox_inches='tight')
        plt.close()

    print(f"   Created {len(models)} per-model spatial maps")

    # 4. Per 4-way group maps
    print("\n4. Creating per 4-way group spatial maps...")
    groups_4way = adata.obs['model_genotype'].dropna().unique()

    for grp in groups_4way:
        grp_mask = adata.obs['model_genotype'] == grp

        fig, ax = plt.subplots(figsize=(14, 12), dpi=150)

        for pop in populations:
            if f'is_{pop}' in adata.obs:
                pop_mask = grp_mask & adata.obs[f'is_{pop}']
                if pop_mask.sum() > 0:
                    pop_coords = coords[pop_mask]
                    if len(pop_coords) > 50000:
                        idx = np.random.choice(len(pop_coords), 50000, replace=False)
                        pop_coords = pop_coords[idx]
                    ax.scatter(pop_coords[:, 0], pop_coords[:, 1],
                             c=[pop_colors[pop]], s=0.5, alpha=0.4, label=pop)

        ax.set_title(f'{grp} - Spatial Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (μm)', fontsize=12)
        ax.set_ylabel('Y (μm)', fontsize=12)
        ax.legend(markerscale=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/spatial_maps/by_genotype/{grp.replace(' ', '_')}_spatial.png",
                   dpi=150, bbox_inches='tight')
        plt.close()

    print(f"   Created {len(groups_4way)} per 4-way group spatial maps")
    print("\n✓ Spatial maps complete\n")


def analyze_and_plot_tumor_size(structure_index: pd.DataFrame, output_dir: str):
    """
    Comprehensive tumor size analysis at BOTH levels:
    1. KPT vs KPNT (2-way comparison)
    2. All 4 groups (KPT cis, KPT trans, KPNT cis, KPNT trans)

    Creates multiple plot formats WITH statistical annotations.
    """
    print("\n" + "="*80)
    print("TUMOR SIZE ANALYSIS - MULTI-LEVEL")
    print("="*80)

    # Aggregate by sample
    sample_agg = structure_index.groupby(['sample_id', 'timepoint', 'model',
                                         'genotype', 'model_genotype', 'group_full']).agg({
        'n_cells': ['sum', 'mean', 'count'],
        'area_um2': ['sum', 'mean']
    }).reset_index()

    sample_agg.columns = ['_'.join(col).strip('_') for col in sample_agg.columns.values]
    sample_agg.rename(columns={
        'sample_id_': 'sample_id',
        'timepoint_': 'timepoint',
        'model_': 'model',
        'genotype_': 'genotype',
        'model_genotype_': 'model_genotype',
        'group_full_': 'group_full',
        'n_cells_sum': 'total_tumor_cells',
        'n_cells_mean': 'mean_structure_size',
        'n_cells_count': 'n_structures',
        'area_um2_sum': 'total_area',
        'area_um2_mean': 'mean_area'
    }, inplace=True)

    sample_agg.to_csv(f"{output_dir}/data/tumor_size_by_sample.csv", index=False)

    # Statistical tests at BOTH levels
    print("\n1. Statistical testing...")

    # === LEVEL 1: KPT vs KPNT ===
    print("  Level 1: KPT vs KPNT comparison...")

    # Temporal trends per model
    temporal_model_results = []
    for model in sample_agg['model'].unique():
        model_data = sample_agg[sample_agg['model'] == model]

        if len(model_data) >= 3:
            rho_total, p_total = spearmanr(model_data['timepoint'], model_data['total_tumor_cells'])
            rho_mean, p_mean = spearmanr(model_data['timepoint'], model_data['mean_structure_size'])

            temporal_model_results.append({
                'model': model,
                'metric': 'total_tumor_cells',
                'spearman_rho': rho_total,
                'p_value': p_total,
                'n_samples': len(model_data)
            })

            temporal_model_results.append({
                'model': model,
                'metric': 'mean_structure_size',
                'spearman_rho': rho_mean,
                'p_value': p_mean,
                'n_samples': len(model_data)
            })

    temporal_model_df = pd.DataFrame(temporal_model_results)
    if len(temporal_model_df) > 0:
        _, p_adj, _, _ = multipletests(temporal_model_df['p_value'], method='fdr_bh')
        temporal_model_df['p_adjusted'] = p_adj
        temporal_model_df['significant'] = temporal_model_df['p_adjusted'] < 0.05
        temporal_model_df.to_csv(f"{output_dir}/statistics/tumor_size_temporal_KPT_vs_KPNT.csv", index=False)

    # Model differences per timepoint (KPT vs KPNT)
    model_comparison_results = []
    for tp in sample_agg['timepoint'].unique():
        tp_data = sample_agg[sample_agg['timepoint'] == tp]

        kpt_data = tp_data[tp_data['model'] == 'KPT']['total_tumor_cells'].values
        kpnt_data = tp_data[tp_data['model'] == 'KPNT']['total_tumor_cells'].values

        if len(kpt_data) >= 1 and len(kpnt_data) >= 1:
            if len(kpt_data) >= 2 and len(kpnt_data) >= 2:
                stat, p = mannwhitneyu(kpt_data, kpnt_data, alternative='two-sided')
            else:
                stat, p = np.nan, np.nan

            model_comparison_results.append({
                'timepoint': tp,
                'comparison': 'KPT_vs_KPNT',
                'KPT_mean': kpt_data.mean(),
                'KPNT_mean': kpnt_data.mean(),
                'fold_change': kpnt_data.mean() / (kpt_data.mean() + 1),
                'n_KPT': len(kpt_data),
                'n_KPNT': len(kpnt_data),
                'statistic': stat,
                'p_value': p
            })

    model_comparison_df = pd.DataFrame(model_comparison_results)
    if len(model_comparison_df) > 0:
        valid_p = model_comparison_df['p_value'].notna()
        if valid_p.sum() > 0:
            _, p_adj, _, _ = multipletests(model_comparison_df.loc[valid_p, 'p_value'], method='fdr_bh')
            model_comparison_df.loc[valid_p, 'p_adjusted'] = p_adj
            model_comparison_df['significant'] = model_comparison_df['p_adjusted'] < 0.05
        model_comparison_df.to_csv(f"{output_dir}/statistics/tumor_size_KPT_vs_KPNT_by_timepoint.csv", index=False)

    # === LEVEL 2: All 4 groups ===
    print("  Level 2: 4-way group comparison...")

    # Temporal trends per 4-way group
    temporal_4way_results = []
    for grp in sample_agg['model_genotype'].unique():
        grp_data = sample_agg[sample_agg['model_genotype'] == grp]

        if len(grp_data) >= 3:
            rho_total, p_total = spearmanr(grp_data['timepoint'], grp_data['total_tumor_cells'])
            rho_mean, p_mean = spearmanr(grp_data['timepoint'], grp_data['mean_structure_size'])

            temporal_4way_results.append({
                'group': grp,
                'metric': 'total_tumor_cells',
                'spearman_rho': rho_total,
                'p_value': p_total,
                'n_samples': len(grp_data)
            })

            temporal_4way_results.append({
                'group': grp,
                'metric': 'mean_structure_size',
                'spearman_rho': rho_mean,
                'p_value': p_mean,
                'n_samples': len(grp_data)
            })

    temporal_4way_df = pd.DataFrame(temporal_4way_results)
    if len(temporal_4way_df) > 0:
        _, p_adj, _, _ = multipletests(temporal_4way_df['p_value'], method='fdr_bh')
        temporal_4way_df['p_adjusted'] = p_adj
        temporal_4way_df['significant'] = temporal_4way_df['p_adjusted'] < 0.05
        temporal_4way_df.to_csv(f"{output_dir}/statistics/tumor_size_temporal_4way.csv", index=False)

    # 4-way group comparisons per timepoint
    fourway_comparison_results = []
    groups_4way = sample_agg['model_genotype'].unique()

    for tp in sample_agg['timepoint'].unique():
        tp_data = sample_agg[sample_agg['timepoint'] == tp]

        # Pairwise comparisons
        for i, g1 in enumerate(groups_4way):
            for g2 in groups_4way[i+1:]:
                d1 = tp_data[tp_data['model_genotype'] == g1]['total_tumor_cells'].values
                d2 = tp_data[tp_data['model_genotype'] == g2]['total_tumor_cells'].values

                if len(d1) >= 1 and len(d2) >= 1:
                    if len(d1) >= 2 and len(d2) >= 2:
                        stat, p = mannwhitneyu(d1, d2, alternative='two-sided')
                    else:
                        stat, p = np.nan, np.nan

                    fourway_comparison_results.append({
                        'timepoint': tp,
                        'group_1': g1,
                        'group_2': g2,
                        'mean_1': d1.mean(),
                        'mean_2': d2.mean(),
                        'fold_change': d2.mean() / (d1.mean() + 1),
                        'n_1': len(d1),
                        'n_2': len(d2),
                        'statistic': stat,
                        'p_value': p
                    })

    fourway_comparison_df = pd.DataFrame(fourway_comparison_results)
    if len(fourway_comparison_df) > 0:
        valid_p = fourway_comparison_df['p_value'].notna()
        if valid_p.sum() > 0:
            _, p_adj, _, _ = multipletests(fourway_comparison_df.loc[valid_p, 'p_value'], method='fdr_bh')
            fourway_comparison_df.loc[valid_p, 'p_adjusted'] = p_adj
            fourway_comparison_df['significant'] = fourway_comparison_df['p_adjusted'] < 0.05
        fourway_comparison_df.to_csv(f"{output_dir}/statistics/tumor_size_4way_by_timepoint.csv", index=False)

    print(f"   KPT vs KPNT temporal: {len(temporal_model_df)} tests")
    print(f"   KPT vs KPNT by timepoint: {len(model_comparison_df)} tests")
    print(f"   4-way temporal: {len(temporal_4way_df)} tests")
    print(f"   4-way pairwise: {len(fourway_comparison_df)} tests")

    # 2. Visualizations at BOTH levels
    print("\n2. Creating visualizations...")

    # === LEVEL 1 PLOTS: KPT vs KPNT ===
    print("  Level 1: KPT vs KPNT plots...")

    # KPT vs KPNT line plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    ax = axes[0]
    for model in sample_agg['model'].unique():
        model_data = sample_agg[sample_agg['model'] == model]
        model_mean = model_data.groupby('timepoint')['total_tumor_cells'].agg(['mean', 'sem'])
        ax.errorbar(model_mean.index, model_mean['mean'], yerr=model_mean['sem'],
                   marker='o', linewidth=3, markersize=10, capsize=5, label=model)

    ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Tumor Cells', fontsize=12, fontweight='bold')
    ax.set_title('KPT vs KPNT: Tumor Growth', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sns.boxplot(data=sample_agg, x='timepoint', y='total_tumor_cells', hue='model',
               ax=ax, palette='Set1')
    ax.set_title('KPT vs KPNT: Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Total Tumor Cells', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/temporal/tumor_size/KPT_vs_KPNT_temporal.png",
               dpi=300, bbox_inches='tight')
    plt.close()

    # === LEVEL 2 PLOTS: All 4 groups ===
    print("  Level 2: 4-way group plots...")

    # 4-way line plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=300)

    ax = axes[0]
    colors_4way = {'KPT_cis': '#E41A1C', 'KPT_trans': '#377EB8',
                   'KPNT_cis': '#4DAF4A', 'KPNT_trans': '#984EA3'}

    for grp in sample_agg['model_genotype'].unique():
        grp_data = sample_agg[sample_agg['model_genotype'] == grp]
        grp_mean = grp_data.groupby('timepoint')['total_tumor_cells'].agg(['mean', 'sem'])
        color = colors_4way.get(grp, '#999999')
        ax.errorbar(grp_mean.index, grp_mean['mean'], yerr=grp_mean['sem'],
                   marker='o', linewidth=2, markersize=8, capsize=5, label=grp, color=color)

    ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Tumor Cells', fontsize=12, fontweight='bold')
    ax.set_title('All 4 Groups: Tumor Growth', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sns.boxplot(data=sample_agg, x='timepoint', y='total_tumor_cells', hue='model_genotype',
               ax=ax, palette=colors_4way)
    ax.set_title('All 4 Groups: Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Total Tumor Cells', fontsize=12)
    ax.legend(title='Group', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/temporal/tumor_size/4way_groups_temporal.png",
               dpi=300, bbox_inches='tight')
    plt.close()

    # Violin plots for 4-way comparison
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    sns.violinplot(data=sample_agg, x='timepoint', y='total_tumor_cells', hue='model_genotype',
                  ax=ax, palette=colors_4way, split=False)
    ax.set_title('All 4 Groups: Tumor Cell Distribution',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Total Tumor Cells', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/temporal/tumor_size/4way_violin.png",
               dpi=300, bbox_inches='tight')
    plt.close()

    print("   Created KPT vs KPNT plots")
    print("   Created 4-way group plots")
    print("\n✓ Tumor size analysis complete\n")

    return sample_agg


def analyze_and_plot_marker_expression(adata, output_dir: str, markers: list):
    """
    Analyze marker expression (fractional percentages) at BOTH levels:
    1. KPT vs KPNT
    2. All 4 groups

    Creates comprehensive temporal plots for ALL markers.
    """
    print("\n" + "="*80)
    print("MARKER EXPRESSION ANALYSIS - MULTI-LEVEL")
    print("="*80)

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
                'model': sample_meta.get('model'),
                'genotype': sample_meta.get('genotype'),
                'model_genotype': sample_meta.get('model_genotype'),
                'group_full': sample_meta.get('group_full'),
                'n_cells': len(sample_pos),
                'n_positive': sample_pos.sum(),
                'pct_positive': 100 * sample_pos.sum() / len(sample_pos)
            })

    marker_df = pd.DataFrame(results)
    marker_df.to_csv(f"{output_dir}/data/marker_expression_temporal.csv", index=False)

    print(f"Analyzed {len(markers)} markers across {len(marker_df['sample_id'].unique())} samples")

    # Create plots for each marker at BOTH levels
    print("\nCreating marker expression plots...")

    colors_4way = {'KPT_cis': '#E41A1C', 'KPT_trans': '#377EB8',
                   'KPNT_cis': '#4DAF4A', 'KPNT_trans': '#984EA3'}

    for marker in marker_df['marker'].unique():
        marker_data = marker_df[marker_df['marker'] == marker]

        # LEVEL 1: KPT vs KPNT
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

        ax = axes[0]
        for model in marker_data['model'].dropna().unique():
            model_data = marker_data[marker_data['model'] == model]
            model_mean = model_data.groupby('timepoint')['pct_positive'].agg(['mean', 'sem'])
            ax.errorbar(model_mean.index, model_mean['mean'], yerr=model_mean['sem'],
                       marker='o', linewidth=2, markersize=8, capsize=5, label=model)

        ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'% {marker}+ Cells', fontsize=12, fontweight='bold')
        ax.set_title(f'{marker}: KPT vs KPNT', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        sns.boxplot(data=marker_data, x='timepoint', y='pct_positive', hue='model',
                   ax=ax, palette='Set1')
        ax.set_xlabel('Timepoint', fontsize=12)
        ax.set_ylabel(f'% {marker}+ Cells', fontsize=12)
        ax.set_title(f'{marker}: Distribution by Model', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/temporal/marker_expression/{marker}_KPT_vs_KPNT.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

        # LEVEL 2: All 4 groups
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=300)

        ax = axes[0]
        for grp in marker_data['model_genotype'].dropna().unique():
            grp_data = marker_data[marker_data['model_genotype'] == grp]
            grp_mean = grp_data.groupby('timepoint')['pct_positive'].agg(['mean', 'sem'])
            color = colors_4way.get(grp, '#999999')
            ax.errorbar(grp_mean.index, grp_mean['mean'], yerr=grp_mean['sem'],
                       marker='o', linewidth=2, markersize=8, capsize=5, label=grp, color=color)

        ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'% {marker}+ Cells', fontsize=12, fontweight='bold')
        ax.set_title(f'{marker}: All 4 Groups', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        sns.boxplot(data=marker_data, x='timepoint', y='pct_positive', hue='model_genotype',
                   ax=ax, palette=colors_4way)
        ax.set_xlabel('Timepoint', fontsize=12)
        ax.set_ylabel(f'% {marker}+ Cells', fontsize=12)
        ax.set_title(f'{marker}: 4-way Distribution', fontsize=14, fontweight='bold')
        ax.legend(title='Group', fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/temporal/marker_expression/{marker}_4way.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

    print(f"   Created 2 plot types for {len(marker_df['marker'].unique())} markers")
    print(f"   Total plots: {2 * len(marker_df['marker'].unique())}")
    print("\n✓ Marker expression analysis complete\n")

    return marker_df


def setup_output_directories(output_dir: str):
    """Create all necessary output directories."""
    dirs = [
        f"{output_dir}/data",
        f"{output_dir}/statistics",
        f"{output_dir}/figures/spatial_maps/by_sample",
        f"{output_dir}/figures/spatial_maps/by_timepoint",
        f"{output_dir}/figures/spatial_maps/by_genotype",
        f"{output_dir}/figures/temporal/tumor_size",
        f"{output_dir}/figures/temporal/marker_expression",
        f"{output_dir}/figures/temporal/infiltration",
        f"{output_dir}/figures/neighborhoods/temporal",
        f"{output_dir}/figures/heatmaps",
        f"{output_dir}/figures/combined",
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


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

    # Setup all output directories
    setup_output_directories(output_dir)

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

    # 3d. Infiltration comprehensive temporal analysis
    analyze_and_plot_infiltration_comprehensive(metrics_df, output_dir)

    # Phase 4: Neighborhoods
    neighborhood_config = config.get('cellular_neighborhoods', {})
    if neighborhood_config.get('enabled', True):
        etsa.detect_cellular_neighborhoods(
            populations=list(population_config.keys()),
            **{k: v for k, v in neighborhood_config.items() if k != 'enabled'}
        )

        # 4a. Neighborhood temporal dynamics
        analyze_neighborhoods_temporal(adata, output_dir)

    # Phase 5: Statistics
    if config.get('statistical_analysis', {}).get('enabled', True):
        etsa.statistical_analysis(metrics_df)

    # Phase 6: Remaining visualizations
    if config.get('visualizations', {}).get('enabled', True):
        etsa.create_publication_figures(metrics_df)

    # Phase 7: Comprehensive heatmaps and summary dashboard
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE SUMMARY VISUALIZATIONS")
    print("="*80)

    create_comprehensive_heatmaps(metrics_df, marker_df, size_df, output_dir)
    create_summary_dashboard(metrics_df, marker_df, size_df, output_dir)

    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs in: {output_dir}/")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
