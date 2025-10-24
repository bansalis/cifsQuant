#!/usr/bin/env python3
"""
Additional comprehensive analysis functions.

These add the missing features:
- Infiltration temporal/genotype analysis
- Neighborhood temporal dynamics
- Comprehensive heatmaps
- Summary multi-panel figures

Import and use these in run_comprehensive_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests


def analyze_and_plot_infiltration_comprehensive(metrics_df: pd.DataFrame, output_dir: str):
    """
    Comprehensive infiltration analysis at BOTH levels.

    Creates plots for immune infiltration across time and groups.
    """
    print("\n" + "="*80)
    print("INFILTRATION ANALYSIS - MULTI-LEVEL")
    print("="*80)

    populations = metrics_df['population'].unique()
    regions = metrics_df['region'].unique()

    colors_4way = {'KPT_cis': '#E41A1C', 'KPT_trans': '#377EB8',
                   'KPNT_cis': '#4DAF4A', 'KPNT_trans': '#984EA3'}

    # For each population
    for pop in populations:
        pop_data = metrics_df[metrics_df['population'] == pop]

        # For key regions
        for region in ['Margin', 'Peri_Tumor']:
            region_data = pop_data[pop_data['region'] == region]

            if len(region_data) < 3:
                continue

            # LEVEL 1: KPT vs KPNT
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

            ax = axes[0]
            for model in region_data['model'].dropna().unique():
                model_subset = region_data[region_data['model'] == model]
                model_mean = model_subset.groupby('timepoint')['percentage'].agg(['mean', 'sem'])
                ax.errorbar(model_mean.index, model_mean['mean'], yerr=model_mean['sem'],
                           marker='o', linewidth=2, markersize=8, capsize=5, label=model)

            ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'% {pop}', fontsize=12, fontweight='bold')
            ax.set_title(f'{pop} Infiltration in {region}: KPT vs KPNT',
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            sns.boxplot(data=region_data, x='timepoint', y='percentage', hue='model',
                       ax=ax, palette='Set1')
            ax.set_xlabel('Timepoint', fontsize=12)
            ax.set_ylabel(f'% {pop}', fontsize=12)
            ax.set_title(f'{pop} in {region}: Distribution', fontsize=14, fontweight='bold')

            plt.tight_layout()
            plt.savefig(f"{output_dir}/figures/temporal/infiltration/{pop}_{region}_KPT_vs_KPNT.png",
                       dpi=300, bbox_inches='tight')
            plt.close()

            # LEVEL 2: All 4 groups
            fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=300)

            ax = axes[0]
            for grp in region_data['model_genotype'].dropna().unique():
                grp_subset = region_data[region_data['model_genotype'] == grp]
                grp_mean = grp_subset.groupby('timepoint')['percentage'].agg(['mean', 'sem'])
                color = colors_4way.get(grp, '#999999')
                ax.errorbar(grp_mean.index, grp_mean['mean'], yerr=grp_mean['sem'],
                           marker='o', linewidth=2, markersize=8, capsize=5,
                           label=grp, color=color)

            ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'% {pop}', fontsize=12, fontweight='bold')
            ax.set_title(f'{pop} Infiltration in {region}: All 4 Groups',
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            sns.boxplot(data=region_data, x='timepoint', y='percentage', hue='model_genotype',
                       ax=ax, palette=colors_4way)
            ax.set_xlabel('Timepoint', fontsize=12)
            ax.set_ylabel(f'% {pop}', fontsize=12)
            ax.set_title(f'{pop} in {region}: 4-way Distribution', fontsize=14, fontweight='bold')
            ax.legend(title='Group', fontsize=8)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/figures/temporal/infiltration/{pop}_{region}_4way.png",
                       dpi=300, bbox_inches='tight')
            plt.close()

    print(f"✓ Created infiltration plots for {len(populations)} populations")
    print()


def analyze_neighborhoods_temporal(adata, output_dir: str):
    """
    Analyze neighborhood composition changes over time and by genotype.
    """
    print("\n" + "="*80)
    print("NEIGHBORHOOD TEMPORAL DYNAMICS")
    print("="*80)

    if 'neighborhood_type' not in adata.obs:
        print("  No neighborhood data found, skipping...")
        return

    # Aggregate neighborhood counts by sample
    neighborhood_data = []

    for sample in adata.obs['sample_id'].unique():
        sample_mask = adata.obs['sample_id'] == sample
        sample_meta = adata.obs[sample_mask].iloc[0]

        # Count cells in each neighborhood type
        neighborhood_counts = adata.obs[sample_mask]['neighborhood_type'].value_counts()
        total_cells = len(adata.obs[sample_mask])

        for neighborhood_id, count in neighborhood_counts.items():
            if neighborhood_id >= 0:  # Valid neighborhood
                neighborhood_data.append({
                    'sample_id': sample,
                    'timepoint': sample_meta.get('timepoint'),
                    'model': sample_meta.get('model'),
                    'genotype': sample_meta.get('genotype'),
                    'model_genotype': sample_meta.get('model_genotype'),
                    'neighborhood_id': neighborhood_id,
                    'n_cells': count,
                    'percentage': 100 * count / total_cells
                })

    neighborhood_df = pd.DataFrame(neighborhood_data)
    neighborhood_df.to_csv(f"{output_dir}/data/neighborhood_temporal.csv", index=False)

    # Plot neighborhood composition over time
    n_neighborhoods = len(neighborhood_df['neighborhood_id'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, n_neighborhoods))

    # LEVEL 1: KPT vs KPNT
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=300)

    for model in ['KPT', 'KPNT']:
        model_data = neighborhood_df[neighborhood_df['model'] == model]

        ax = axes[0 if model == 'KPT' else 1]

        for nid in sorted(neighborhood_df['neighborhood_id'].unique()):
            nid_data = model_data[model_data['neighborhood_id'] == nid]
            nid_mean = nid_data.groupby('timepoint')['percentage'].agg(['mean', 'sem'])

            ax.plot(nid_mean.index, nid_mean['mean'], marker='o', linewidth=2,
                   markersize=6, label=f'CN{nid}', color=colors[nid])

        ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
        ax.set_ylabel('% of Cells', fontsize=12, fontweight='bold')
        ax.set_title(f'{model}: Neighborhood Dynamics', fontsize=14, fontweight='bold')
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/neighborhoods/temporal/neighborhood_dynamics_KPT_vs_KPNT.png",
               dpi=300, bbox_inches='tight')
    plt.close()

    # Heatmap: Neighborhood composition by group and timepoint
    pivot_data = neighborhood_df.pivot_table(
        values='percentage',
        index='neighborhood_id',
        columns=['model_genotype', 'timepoint'],
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(20, 8), dpi=300)
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
               cbar_kws={'label': '% of Cells'}, ax=ax)
    ax.set_title('Neighborhood Composition: By Group and Timepoint',
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Group - Timepoint', fontsize=12)
    ax.set_ylabel('Neighborhood ID', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/neighborhoods/temporal/neighborhood_heatmap.png",
               dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Analyzed neighborhood temporal dynamics")
    print(f"  Saved: {output_dir}/data/neighborhood_temporal.csv")
    print()

    return neighborhood_df


def create_comprehensive_heatmaps(metrics_df: pd.DataFrame, marker_df: pd.DataFrame,
                                 size_df: pd.DataFrame, output_dir: str):
    """
    Create comprehensive heatmaps summarizing all data.
    """
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE HEATMAPS")
    print("="*80)

    # 1. Infiltration heatmap by group and region
    print("  1. Infiltration by group and region...")
    pivot_infiltration = metrics_df.pivot_table(
        values='percentage',
        index='population',
        columns=['model_genotype', 'region'],
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
    sns.heatmap(pivot_infiltration, annot=True, fmt='.1f', cmap='RdYlBu_r',
               cbar_kws={'label': '% Infiltration'}, linewidths=0.5, ax=ax)
    ax.set_title('Immune Infiltration: By Group and Region', fontsize=16, fontweight='bold')
    ax.set_xlabel('Group - Region', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/heatmaps/infiltration_by_group_region.png",
               dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Marker expression heatmap by group and timepoint
    print("  2. Marker expression by group and timepoint...")
    pivot_markers = marker_df.pivot_table(
        values='pct_positive',
        index='marker',
        columns=['model_genotype', 'timepoint'],
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(20, 8), dpi=300)
    sns.heatmap(pivot_markers, annot=True, fmt='.1f', cmap='viridis',
               cbar_kws={'label': '% Positive'}, linewidths=0.5, ax=ax)
    ax.set_title('Marker Expression: By Group and Timepoint', fontsize=16, fontweight='bold')
    ax.set_xlabel('Group - Timepoint', fontsize=12)
    ax.set_ylabel('Marker', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/heatmaps/marker_expression_by_group_timepoint.png",
               dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Tumor size heatmap by group and timepoint
    print("  3. Tumor size by group and timepoint...")
    pivot_size = size_df.pivot_table(
        values='total_tumor_cells',
        index='model_genotype',
        columns='timepoint',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    sns.heatmap(pivot_size, annot=True, fmt='.0f', cmap='Reds',
               cbar_kws={'label': 'Total Tumor Cells'}, linewidths=0.5, ax=ax)
    ax.set_title('Tumor Size: By Group and Timepoint', fontsize=16, fontweight='bold')
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Group', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/heatmaps/tumor_size_by_group_timepoint.png",
               dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created comprehensive heatmaps")
    print()


def create_summary_dashboard(metrics_df: pd.DataFrame, marker_df: pd.DataFrame,
                             size_df: pd.DataFrame, output_dir: str):
    """
    Create multi-panel summary dashboard figure.
    """
    print("\n" + "="*80)
    print("CREATING SUMMARY DASHBOARD")
    print("="*80)

    fig = plt.figure(figsize=(20, 12), dpi=300)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    colors_4way = {'KPT_cis': '#E41A1C', 'KPT_trans': '#377EB8',
                   'KPNT_cis': '#4DAF4A', 'KPNT_trans': '#984EA3'}

    # Panel 1: Tumor size over time
    ax1 = fig.add_subplot(gs[0, :2])
    for grp in size_df['model_genotype'].unique():
        grp_data = size_df[size_df['model_genotype'] == grp]
        grp_mean = grp_data.groupby('timepoint')['total_tumor_cells'].mean()
        ax1.plot(grp_mean.index, grp_mean.values, marker='o', linewidth=2,
                markersize=8, label=grp, color=colors_4way.get(grp, '#999'))
    ax1.set_xlabel('Timepoint', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Tumor Cells', fontsize=11, fontweight='bold')
    ax1.set_title('A. Tumor Growth Dynamics', fontsize=12, fontweight='bold', loc='left')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Key marker expression
    ax2 = fig.add_subplot(gs[0, 2])
    key_marker = 'AGFP' if 'AGFP' in marker_df['marker'].values else marker_df['marker'].iloc[0]
    marker_subset = marker_df[marker_df['marker'] == key_marker]
    sns.boxplot(data=marker_subset, x='model', y='pct_positive', ax=ax2, palette='Set1')
    ax2.set_title(f'B. {key_marker} Expression', fontsize=12, fontweight='bold', loc='left')
    ax2.set_ylabel('% Positive', fontsize=10)

    # Panel 3: Infiltration by region
    ax3 = fig.add_subplot(gs[1, :])
    key_pop = 'CD8_T_cells' if 'CD8_T_cells' in metrics_df['population'].values else metrics_df['population'].iloc[0]
    pop_subset = metrics_df[metrics_df['population'] == key_pop]
    sns.boxplot(data=pop_subset, x='region', y='percentage', hue='model_genotype',
               ax=ax3, palette=colors_4way)
    ax3.set_title(f'C. {key_pop} Infiltration by Region', fontsize=12, fontweight='bold', loc='left')
    ax3.set_ylabel('% Infiltration', fontsize=10)
    ax3.legend(title='Group', fontsize=8)

    # Panel 4: Group comparison
    ax4 = fig.add_subplot(gs[2, :])
    size_latest = size_df[size_df['timepoint'] == size_df['timepoint'].max()]
    sns.barplot(data=size_latest, x='model_genotype', y='total_tumor_cells',
               ax=ax4, palette=colors_4way, errorbar='se')
    ax4.set_title(f'D. Final Tumor Burden (Timepoint {size_df["timepoint"].max()})',
                 fontsize=12, fontweight='bold', loc='left')
    ax4.set_ylabel('Total Tumor Cells', fontsize=10)
    ax4.set_xlabel('Group', fontsize=10)

    plt.suptitle('Comprehensive Analysis Summary Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(f"{output_dir}/figures/combined/summary_dashboard.png",
               dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created summary dashboard")
    print()


if __name__ == '__main__':
    print("Import these functions into run_comprehensive_analysis.py")
