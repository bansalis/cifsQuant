#!/usr/bin/env python3
"""
FIXED Comprehensive Visualization Module

CRITICAL CORRECTIONS:
1. NEVER compare cis vs trans alone
2. Always use main_group (KPT vs KPNT) for main comparison
3. Always use genotype_full for 4-way comparison (KPT-cis, KPT-trans, KPNT-cis, KPNT-trans)
4. Create MANY comprehensive multi-panel plots
5. Add statistical annotations

Author: Fixed comprehensive plots
Date: 2025-10-30
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr, linregress
from statsmodels.stats.multitest import multipletests
from typing import Dict, Optional
import os


def add_pvalue_annotation(ax, df, value_col, group_col='main_group'):
    """Add p-value annotation comparing KPT vs KPNT."""
    groups = df[group_col].dropna().unique()

    if len(groups) == 2:
        g1, g2 = groups[0], groups[1]
        data1 = df[df[group_col] == g1][value_col].dropna().values
        data2 = df[df[group_col] == g2][value_col].dropna().values

        if len(data1) >= 2 and len(data2) >= 2:
            stat, p = mannwhitneyu(data1, data2, alternative='two-sided')

            # Format p-value
            if p < 0.001:
                p_text = 'p < 0.001***'
            elif p < 0.01:
                p_text = f'p = {p:.3f}**'
            elif p < 0.05:
                p_text = f'p = {p:.3f}*'
            else:
                p_text = f'p = {p:.3f} ns'

            # Add to plot
            y_max = ax.get_ylim()[1]
            ax.text(0.5, 0.95, p_text, transform=ax.transAxes,
                   ha='center', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_tumor_size_comprehensive(size_df: pd.DataFrame, output_dir: str):
    """
    Generate comprehensive tumor size plots with correct comparisons.

    Creates 3x3 grid with:
    - KPT vs KPNT comparisons
    - 4-way genotype_full comparisons
    - Temporal trends
    - Statistical annotations
    """
    print("\nGenerating comprehensive tumor size plots...")

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: KPT vs KPNT Main Comparison
    # Plot 1: Mean structure size over time (KPT vs KPNT)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'main_group' in size_df.columns:
        for group in size_df['main_group'].unique():
            group_data = size_df[size_df['main_group'] == group]
            if 'timepoint' in group_data.columns:
                group_mean = group_data.groupby('timepoint')['mean_structure_size'].mean()
                ax1.plot(group_mean.index, group_mean.values,
                        marker='o', label=group, linewidth=2, markersize=8)
        ax1.set_xlabel('Timepoint', fontweight='bold')
        ax1.set_ylabel('Mean Structure Size (cells)', fontweight='bold')
        ax1.set_title('Tumor Growth: KPT vs KPNT', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot 2: Boxplot KPT vs KPNT
    ax2 = fig.add_subplot(gs[0, 1])
    if 'main_group' in size_df.columns:
        sns.boxplot(data=size_df, x='main_group', y='mean_structure_size',
                   ax=ax2, palette='Set2')
        ax2.set_ylabel('Mean Structure Size (cells)', fontweight='bold')
        ax2.set_xlabel('Main Group', fontweight='bold')
        ax2.set_title('KPT vs KPNT Comparison', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        add_pvalue_annotation(ax2, size_df, 'mean_structure_size', 'main_group')

    # Plot 3: Total tumor area KPT vs KPNT
    ax3 = fig.add_subplot(gs[0, 2])
    if 'main_group' in size_df.columns and 'total_tumor_area' in size_df.columns:
        for group in size_df['main_group'].unique():
            group_data = size_df[size_df['main_group'] == group]
            if 'timepoint' in group_data.columns:
                group_mean = group_data.groupby('timepoint')['total_tumor_area'].mean()
                ax3.plot(group_mean.index, group_mean.values,
                        marker='s', label=group, linewidth=2, markersize=8)
        ax3.set_xlabel('Timepoint', fontweight='bold')
        ax3.set_ylabel('Total Tumor Area (μm²)', fontweight='bold')
        ax3.set_title('Total Area: KPT vs KPNT', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Row 2: 4-way Genotype Comparison (KPT-cis, KPT-trans, KPNT-cis, KPNT-trans)
    # Plot 4: Mean size over time (4-way)
    ax4 = fig.add_subplot(gs[1, 0])
    if 'genotype_full' in size_df.columns:
        for genotype_full in size_df['genotype_full'].unique():
            gf_data = size_df[size_df['genotype_full'] == genotype_full]
            if 'timepoint' in gf_data.columns:
                gf_mean = gf_data.groupby('timepoint')['mean_structure_size'].mean()
                ax4.plot(gf_mean.index, gf_mean.values,
                        marker='o', label=genotype_full, linewidth=2, markersize=6)
        ax4.set_xlabel('Timepoint', fontweight='bold')
        ax4.set_ylabel('Mean Structure Size (cells)', fontweight='bold')
        ax4.set_title('4-Way Comparison: KPT-cis/trans vs KPNT-cis/trans', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Plot 5: Boxplot 4-way
    ax5 = fig.add_subplot(gs[1, 1])
    if 'genotype_full' in size_df.columns:
        sns.boxplot(data=size_df, x='genotype_full', y='mean_structure_size',
                   ax=ax5, palette='Set3')
        ax5.set_ylabel('Mean Structure Size (cells)', fontweight='bold')
        ax5.set_xlabel('', fontweight='bold')
        ax5.set_title('4-Way Genotype Comparison', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 6: Number of structures over time (4-way)
    ax6 = fig.add_subplot(gs[1, 2])
    if 'genotype_full' in size_df.columns and 'n_structures' in size_df.columns:
        for genotype_full in size_df['genotype_full'].unique():
            gf_data = size_df[size_df['genotype_full'] == genotype_full]
            if 'timepoint' in gf_data.columns:
                gf_mean = gf_data.groupby('timepoint')['n_structures'].mean()
                ax6.plot(gf_mean.index, gf_mean.values,
                        marker='^', label=genotype_full, linewidth=2, markersize=6)
        ax6.set_xlabel('Timepoint', fontweight='bold')
        ax6.set_ylabel('Number of Structures', fontweight='bold')
        ax6.set_title('Structure Count: 4-Way', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    # Row 3: Per-timepoint comparisons
    # Plot 7: Violin plot by timepoint (KPT vs KPNT)
    ax7 = fig.add_subplot(gs[2, 0])
    if 'timepoint' in size_df.columns and 'main_group' in size_df.columns:
        sns.violinplot(data=size_df, x='timepoint', y='mean_structure_size',
                      hue='main_group', ax=ax7, palette='Set2', split=False)
        ax7.set_xlabel('Timepoint', fontweight='bold')
        ax7.set_ylabel('Mean Structure Size (cells)', fontweight='bold')
        ax7.set_title('Per-Timepoint: KPT vs KPNT', fontweight='bold')

    # Plot 8: Violin plot by timepoint (4-way)
    ax8 = fig.add_subplot(gs[2, 1])
    if 'timepoint' in size_df.columns and 'genotype_full' in size_df.columns:
        sns.boxplot(data=size_df, x='timepoint', y='mean_structure_size',
                   hue='genotype_full', ax=ax8, palette='Set3')
        ax8.set_xlabel('Timepoint', fontweight='bold')
        ax8.set_ylabel('Mean Structure Size (cells)', fontweight='bold')
        ax8.set_title('Per-Timepoint: 4-Way', fontweight='bold')
        ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Plot 9: Distribution comparison
    ax9 = fig.add_subplot(gs[2, 2])
    if 'main_group' in size_df.columns:
        for group in size_df['main_group'].unique():
            group_data = size_df[size_df['main_group'] == group]['mean_structure_size'].dropna()
            ax9.hist(group_data, bins=30, alpha=0.5, label=group, density=True)
        ax9.set_xlabel('Mean Structure Size (cells)', fontweight='bold')
        ax9.set_ylabel('Density', fontweight='bold')
        ax9.set_title('Size Distribution: KPT vs KPNT', fontweight='bold')
        ax9.legend()

    plt.suptitle(f'Comprehensive Tumor Size Analysis (n = {len(size_df)} samples)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(f"{output_dir}/figures/tumor_size_COMPREHENSIVE_FIXED.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Tumor size comprehensive plot saved")


def plot_infiltration_comprehensive(metrics_df: pd.DataFrame, output_dir: str):
    """
    Generate comprehensive infiltration plots for each population.

    Creates separate 3x3 figures for top populations with:
    - KPT vs KPNT by region
    - 4-way comparisons
    - Temporal trends
    """
    print("\nGenerating comprehensive infiltration plots...")

    # Get top populations
    pop_counts = metrics_df.groupby('population').size()
    top_pops = pop_counts.nlargest(4).index.tolist()

    for pop in top_pops:
        pop_data = metrics_df[metrics_df['population'] == pop]

        if len(pop_data) < 10:
            continue

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Row 1: KPT vs KPNT by region
        regions = ['Tumor_Core', 'Margin', 'Peri_Tumor', 'Distal']

        # Plot 1: Temporal trends by region (KPT vs KPNT)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'main_group' in pop_data.columns and 'region' in pop_data.columns:
            for region in regions:
                region_data = pop_data[pop_data['region'] == region]
                for group in region_data['main_group'].unique():
                    group_region = region_data[region_data['main_group'] == group]
                    if 'timepoint' in group_region.columns:
                        temporal_mean = group_region.groupby('timepoint')['percentage'].mean()
                        ax1.plot(temporal_mean.index, temporal_mean.values,
                               marker='o', label=f'{group}-{region}', linewidth=1.5)
            ax1.set_xlabel('Timepoint', fontweight='bold')
            ax1.set_ylabel('% Infiltration', fontweight='bold')
            ax1.set_title(f'{pop}: Temporal Trends (KPT vs KPNT)', fontweight='bold')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
            ax1.grid(True, alpha=0.3)

        # Plot 2: Boxplot by region (KPT vs KPNT)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'main_group' in pop_data.columns and 'region' in pop_data.columns:
            sns.boxplot(data=pop_data, x='region', y='percentage',
                       hue='main_group', ax=ax2, palette='Set2')
            ax2.set_ylabel('% Infiltration', fontweight='bold')
            ax2.set_xlabel('Region', fontweight='bold')
            ax2.set_title(f'{pop}: KPT vs KPNT by Region', fontweight='bold')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 3: Tumor Core only (KPT vs KPNT)
        ax3 = fig.add_subplot(gs[0, 2])
        core_data = pop_data[pop_data['region'] == 'Tumor_Core']
        if len(core_data) > 0 and 'main_group' in core_data.columns:
            sns.boxplot(data=core_data, x='main_group', y='percentage',
                       ax=ax3, palette='Set2')
            ax3.set_ylabel('% in Tumor Core', fontweight='bold')
            ax3.set_xlabel('Main Group', fontweight='bold')
            ax3.set_title(f'{pop}: Tumor Core (KPT vs KPNT)', fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            add_pvalue_annotation(ax3, core_data, 'percentage', 'main_group')

        # Row 2: 4-way comparisons
        # Plot 4: Temporal 4-way by region
        ax4 = fig.add_subplot(gs[1, 0])
        if 'genotype_full' in pop_data.columns and 'region' in pop_data.columns:
            for genotype_full in pop_data['genotype_full'].unique():
                gf_data = pop_data[pop_data['genotype_full'] == genotype_full]
                for region in ['Tumor_Core', 'Margin']:  # Limit to 2 regions for clarity
                    region_gf = gf_data[gf_data['region'] == region]
                    if 'timepoint' in region_gf.columns:
                        temporal_mean = region_gf.groupby('timepoint')['percentage'].mean()
                        ax4.plot(temporal_mean.index, temporal_mean.values,
                               marker='o', label=f'{genotype_full}-{region}', linewidth=1.5)
            ax4.set_xlabel('Timepoint', fontweight='bold')
            ax4.set_ylabel('% Infiltration', fontweight='bold')
            ax4.set_title(f'{pop}: 4-Way by Region', fontweight='bold')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
            ax4.grid(True, alpha=0.3)

        # Plot 5: Boxplot 4-way
        ax5 = fig.add_subplot(gs[1, 1])
        if 'genotype_full' in pop_data.columns:
            core_4way = pop_data[pop_data['region'] == 'Tumor_Core']
            if len(core_4way) > 0:
                sns.boxplot(data=core_4way, x='genotype_full', y='percentage',
                           ax=ax5, palette='Set3')
                ax5.set_ylabel('% in Tumor Core', fontweight='bold')
                ax5.set_xlabel('', fontweight='bold')
                ax5.set_title(f'{pop}: 4-Way in Tumor Core', fontweight='bold')
                ax5.grid(True, alpha=0.3, axis='y')
                plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 6: Heatmap of infiltration by region and 4-way genotype
        ax6 = fig.add_subplot(gs[1, 2])
        if 'genotype_full' in pop_data.columns and 'region' in pop_data.columns:
            pivot_data = pop_data.pivot_table(
                values='percentage',
                index='genotype_full',
                columns='region',
                aggfunc='mean'
            )
            sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
                       ax=ax6, cbar_kws={'label': '% Infiltration'})
            ax6.set_title(f'{pop}: Heatmap (4-Way × Region)', fontweight='bold')
            ax6.set_ylabel('Genotype', fontweight='bold')
            ax6.set_xlabel('Region', fontweight='bold')

        # Row 3: Per-timepoint and density analyses
        # Plot 7: Violin by timepoint (KPT vs KPNT)
        ax7 = fig.add_subplot(gs[2, 0])
        core_temporal = pop_data[pop_data['region'] == 'Tumor_Core']
        if len(core_temporal) > 0 and 'timepoint' in core_temporal.columns:
            sns.violinplot(data=core_temporal, x='timepoint', y='percentage',
                          hue='main_group', ax=ax7, palette='Set2', split=False)
            ax7.set_xlabel('Timepoint', fontweight='bold')
            ax7.set_ylabel('% in Tumor Core', fontweight='bold')
            ax7.set_title(f'{pop}: Per-Timepoint (KPT vs KPNT)', fontweight='bold')

        # Plot 8: Density comparison
        ax8 = fig.add_subplot(gs[2, 1])
        if 'main_group' in pop_data.columns and 'density' in pop_data.columns:
            core_density = pop_data[pop_data['region'] == 'Tumor_Core']
            if len(core_density) > 0:
                sns.boxplot(data=core_density, x='main_group', y='density',
                           ax=ax8, palette='Set2')
                ax8.set_ylabel('Density (cells/mm²)', fontweight='bold')
                ax8.set_xlabel('Main Group', fontweight='bold')
                ax8.set_title(f'{pop}: Density in Core', fontweight='bold')
                ax8.grid(True, alpha=0.3, axis='y')
                add_pvalue_annotation(ax8, core_density, 'density', 'main_group')

        # Plot 9: All regions comparison (KPT vs KPNT)
        ax9 = fig.add_subplot(gs[2, 2])
        if 'main_group' in pop_data.columns:
            region_means = pop_data.groupby(['region', 'main_group'])['percentage'].mean().unstack()
            region_means.plot(kind='bar', ax=ax9, width=0.8, color=['#66c2a5', '#fc8d62'])
            ax9.set_ylabel('% Infiltration', fontweight='bold')
            ax9.set_xlabel('Region', fontweight='bold')
            ax9.set_title(f'{pop}: Regional Summary', fontweight='bold')
            ax9.legend(title='Group')
            ax9.grid(True, alpha=0.3, axis='y')
            plt.setp(ax9.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.suptitle(f'{pop} Infiltration: Comprehensive Analysis (n = {len(pop_data)})',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(f"{output_dir}/figures/infiltration_{pop}_COMPREHENSIVE.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  ✓ Generated comprehensive plots for {len(top_pops)} populations")


if __name__ == '__main__':
    print("Import this module and call the plot functions")


def plot_marker_expression_comprehensive(marker_df: pd.DataFrame, output_dir: str):
    """Generate comprehensive marker expression plots for ALL markers."""
    print("\nGenerating comprehensive marker expression plots...")

    all_markers = marker_df['marker'].unique()

    for marker in all_markers:
        marker_data = marker_df[marker_df['marker'] == marker]

        if len(marker_data) < 5:
            continue

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Row 1: KPT vs KPNT
        # Plot 1: Temporal trend (KPT vs KPNT)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'main_group' in marker_data.columns and 'timepoint' in marker_data.columns:
            for group in marker_data['main_group'].unique():
                group_data = marker_data[marker_data['main_group'] == group]
                temporal_mean = group_data.groupby('timepoint')['pct_positive'].mean()
                ax1.plot(temporal_mean.index, temporal_mean.values,
                        marker='o', label=group, linewidth=2, markersize=8)
            ax1.set_xlabel('Timepoint', fontweight='bold')
            ax1.set_ylabel('% Positive Cells', fontweight='bold')
            ax1.set_title(f'{marker}: KPT vs KPNT Over Time', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Boxplot (KPT vs KPNT)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'main_group' in marker_data.columns:
            sns.boxplot(data=marker_data, x='main_group', y='pct_positive',
                       ax=ax2, palette='Set2')
            ax2.set_ylabel('% Positive Cells', fontweight='bold')
            ax2.set_xlabel('Main Group', fontweight='bold')
            ax2.set_title(f'{marker}: KPT vs KPNT', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            add_pvalue_annotation(ax2, marker_data, 'pct_positive', 'main_group')

        # Plot 3: Mean expression (KPT vs KPNT)
        ax3 = fig.add_subplot(gs[0, 2])
        if 'main_group' in marker_data.columns and 'mean_expression' in marker_data.columns:
            sns.violinplot(data=marker_data, x='main_group', y='mean_expression',
                          ax=ax3, palette='Set2')
            ax3.set_ylabel('Mean Expression', fontweight='bold')
            ax3.set_xlabel('Main Group', fontweight='bold')
            ax3.set_title(f'{marker}: Expression Level', fontweight='bold')

        # Row 2: 4-way comparison
        # Plot 4: Temporal 4-way
        ax4 = fig.add_subplot(gs[1, 0])
        if 'genotype_full' in marker_data.columns and 'timepoint' in marker_data.columns:
            for genotype_full in marker_data['genotype_full'].unique():
                gf_data = marker_data[marker_data['genotype_full'] == genotype_full]
                temporal_mean = gf_data.groupby('timepoint')['pct_positive'].mean()
                ax4.plot(temporal_mean.index, temporal_mean.values,
                        marker='o', label=genotype_full, linewidth=2, markersize=6)
            ax4.set_xlabel('Timepoint', fontweight='bold')
            ax4.set_ylabel('% Positive Cells', fontweight='bold')
            ax4.set_title(f'{marker}: 4-Way Over Time', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # Plot 5: Boxplot 4-way
        ax5 = fig.add_subplot(gs[1, 1])
        if 'genotype_full' in marker_data.columns:
            sns.boxplot(data=marker_data, x='genotype_full', y='pct_positive',
                       ax=ax5, palette='Set3')
            ax5.set_ylabel('% Positive Cells', fontweight='bold')
            ax5.set_xlabel('', fontweight='bold')
            ax5.set_title(f'{marker}: 4-Way Comparison', fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 6: Mean positive expression (4-way)
        ax6 = fig.add_subplot(gs[1, 2])
        if 'genotype_full' in marker_data.columns and 'mean_positive' in marker_data.columns:
            sns.boxplot(data=marker_data, x='genotype_full', y='mean_positive',
                       ax=ax6, palette='Set3')
            ax6.set_ylabel('Mean Expression (in positive cells)', fontweight='bold')
            ax6.set_xlabel('', fontweight='bold')
            ax6.set_title(f'{marker}: Expression in Positive Cells', fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')
            plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Row 3: Per-timepoint and distributions
        # Plot 7: Violin by timepoint (KPT vs KPNT)
        ax7 = fig.add_subplot(gs[2, 0])
        if 'timepoint' in marker_data.columns and 'main_group' in marker_data.columns:
            sns.violinplot(data=marker_data, x='timepoint', y='pct_positive',
                          hue='main_group', ax=ax7, palette='Set2', split=False)
            ax7.set_xlabel('Timepoint', fontweight='bold')
            ax7.set_ylabel('% Positive Cells', fontweight='bold')
            ax7.set_title(f'{marker}: Per-Timepoint Distribution', fontweight='bold')

        # Plot 8: Scatter - positivity vs expression
        ax8 = fig.add_subplot(gs[2, 1])
        if 'main_group' in marker_data.columns and 'pct_positive' in marker_data.columns and 'mean_positive' in marker_data.columns:
            for group in marker_data['main_group'].unique():
                group_data = marker_data[marker_data['main_group'] == group]
                ax8.scatter(group_data['pct_positive'], group_data['mean_positive'],
                           alpha=0.6, s=50, label=group)
            ax8.set_xlabel('% Positive Cells', fontweight='bold')
            ax8.set_ylabel('Mean Expression (in positive)', fontweight='bold')
            ax8.set_title(f'{marker}: Prevalence vs Intensity', fontweight='bold')
            ax8.legend()
            ax8.grid(True, alpha=0.3)

        # Plot 9: Distribution comparison
        ax9 = fig.add_subplot(gs[2, 2])
        if 'main_group' in marker_data.columns:
            for group in marker_data['main_group'].unique():
                group_data = marker_data[marker_data['main_group'] == group]['pct_positive'].dropna()
                ax9.hist(group_data, bins=20, alpha=0.5, label=group, density=True)
            ax9.set_xlabel('% Positive Cells', fontweight='bold')
            ax9.set_ylabel('Density', fontweight='bold')
            ax9.set_title(f'{marker}: Distribution', fontweight='bold')
            ax9.legend()

        plt.suptitle(f'{marker} Expression: Comprehensive Analysis (n = {len(marker_data)} samples)',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(f"{output_dir}/figures/marker_{marker}_COMPREHENSIVE.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  ✓ Generated comprehensive plots for {len(all_markers)} markers")


def plot_neighborhoods_comprehensive(neighborhood_df: pd.DataFrame, cell_neighborhoods_df: pd.DataFrame, output_dir: str):
    """Generate comprehensive neighborhood analysis plots."""
    print("\nGenerating comprehensive neighborhood plots...")

    # Get unique neighborhood types
    if 'neighborhood_type' in cell_neighborhoods_df.columns:
        nh_types = cell_neighborhoods_df['neighborhood_type'].unique()
        n_types = len(nh_types)

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Row 1: Overall distribution
        # Plot 1: Stacked area KPT vs KPNT
        ax1 = fig.add_subplot(gs[0, :2])
        if 'main_group' in cell_neighborhoods_df.columns and 'timepoint' in cell_neighborhoods_df.columns:
            for group in cell_neighborhoods_df['main_group'].unique():
                group_data = cell_neighborhoods_df[cell_neighborhoods_df['main_group'] == group]
                nh_temporal = group_data.groupby(['timepoint', 'neighborhood_type']).size().unstack(fill_value=0)
                nh_temporal_pct = nh_temporal.div(nh_temporal.sum(axis=1), axis=0) * 100

                nh_temporal_pct.plot(kind='area', stacked=True, ax=ax1, alpha=0.5, legend=False)

            ax1.set_xlabel('Timepoint', fontweight='bold')
            ax1.set_ylabel('% Cells', fontweight='bold')
            ax1.set_title('Neighborhood Composition Over Time', fontweight='bold')
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3)

        # Plot 2: Pie chart overall
        ax2 = fig.add_subplot(gs[0, 2])
        if 'neighborhood_type' in neighborhood_df.columns and 'n_cells' in neighborhood_df.columns:
            ax2.pie(neighborhood_df['n_cells'], labels=neighborhood_df['neighborhood_type'],
                   autopct='%1.1f%%', startangle=90)
            ax2.set_title('Overall Distribution', fontweight='bold')

        # Row 2: KPT vs KPNT comparison
        # Plot 3: Heatmap by timepoint (KPT)
        ax3 = fig.add_subplot(gs[1, 0])
        if 'main_group' in cell_neighborhoods_df.columns:
            kpt_data = cell_neighborhoods_df[cell_neighborhoods_df['main_group'] == 'KPT']
            if len(kpt_data) > 0 and 'timepoint' in kpt_data.columns:
                nh_pivot = kpt_data.groupby(['timepoint', 'neighborhood_type']).size().unstack(fill_value=0)
                nh_pivot_pct = nh_pivot.div(nh_pivot.sum(axis=1), axis=0) * 100
                sns.heatmap(nh_pivot_pct.T, annot=True, fmt='.1f', cmap='YlOrRd',
                           ax=ax3, cbar_kws={'label': '% Cells'})
                ax3.set_title('KPT: Neighborhood × Time', fontweight='bold')
                ax3.set_xlabel('Timepoint', fontweight='bold')
                ax3.set_ylabel('Neighborhood Type', fontweight='bold')

        # Plot 4: Heatmap by timepoint (KPNT)
        ax4 = fig.add_subplot(gs[1, 1])
        if 'main_group' in cell_neighborhoods_df.columns:
            kpnt_data = cell_neighborhoods_df[cell_neighborhoods_df['main_group'] == 'KPNT']
            if len(kpnt_data) > 0 and 'timepoint' in kpnt_data.columns:
                nh_pivot = kpnt_data.groupby(['timepoint', 'neighborhood_type']).size().unstack(fill_value=0)
                nh_pivot_pct = nh_pivot.div(nh_pivot.sum(axis=1), axis=0) * 100
                sns.heatmap(nh_pivot_pct.T, annot=True, fmt='.1f', cmap='YlGnBu',
                           ax=ax4, cbar_kws={'label': '% Cells'})
                ax4.set_title('KPNT: Neighborhood × Time', fontweight='bold')
                ax4.set_xlabel('Timepoint', fontweight='bold')
                ax4.set_ylabel('Neighborhood Type', fontweight='bold')

        # Plot 5: Difference heatmap (KPT - KPNT)
        ax5 = fig.add_subplot(gs[1, 2])
        if 'main_group' in cell_neighborhoods_df.columns and 'timepoint' in cell_neighborhoods_df.columns:
            kpt_pivot = cell_neighborhoods_df[cell_neighborhoods_df['main_group'] == 'KPT'].groupby(
                ['timepoint', 'neighborhood_type']).size().unstack(fill_value=0)
            kpt_pivot_pct = kpt_pivot.div(kpt_pivot.sum(axis=1), axis=0) * 100

            kpnt_pivot = cell_neighborhoods_df[cell_neighborhoods_df['main_group'] == 'KPNT'].groupby(
                ['timepoint', 'neighborhood_type']).size().unstack(fill_value=0)
            kpnt_pivot_pct = kpnt_pivot.div(kpnt_pivot.sum(axis=1), axis=0) * 100

            diff = kpt_pivot_pct - kpnt_pivot_pct
            sns.heatmap(diff.T, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
                       ax=ax5, cbar_kws={'label': '% Difference (KPT - KPNT)'})
            ax5.set_title('Difference: KPT - KPNT', fontweight='bold')
            ax5.set_xlabel('Timepoint', fontweight='bold')
            ax5.set_ylabel('Neighborhood Type', fontweight='bold')

        # Row 3: 4-way comparison
        # Plot 6-9: Bar charts for each neighborhood type
        for idx, nh_type in enumerate(list(nh_types)[:4]):  # Top 4 neighborhoods
            ax = fig.add_subplot(gs[2, idx % 3])
            nh_data = cell_neighborhoods_df[cell_neighborhoods_df['neighborhood_type'] == nh_type]

            if len(nh_data) > 0 and 'genotype_full' in nh_data.columns and 'timepoint' in nh_data.columns:
                nh_temporal = nh_data.groupby(['timepoint', 'genotype_full']).size().unstack(fill_value=0)
                nh_temporal.plot(kind='line', ax=ax, marker='o', linewidth=2)
                ax.set_xlabel('Timepoint', fontweight='bold')
                ax.set_ylabel('Number of Cells', fontweight='bold')
                ax.set_title(f'Neighborhood {nh_type}: 4-Way', fontweight='bold')
                ax.legend(title='Genotype', fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.suptitle(f'Comprehensive Neighborhood Analysis (n = {len(cell_neighborhoods_df)} cells, {n_types} types)',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(f"{output_dir}/figures/neighborhoods_COMPREHENSIVE.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    print("  ✓ Neighborhood comprehensive plot saved")
