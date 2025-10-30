#!/usr/bin/env python3
"""
Advanced Spatial Analysis Visualizations - COMPREHENSIVE EDITION

Comprehensive visualization functions for advanced analysis phases 11-18.
Generates publication-quality 3×3 grid plots for:
- Phase 12: pERK spatial architecture
- Phase 13: NINJA escape mechanisms
- Phase 14: Tumor heterogeneity
- Phase 15: Enhanced RCN dynamics
- Phase 16: Multi-level distance analysis
- Phase 17: Infiltration-tumor associations
- Phase 18: Pseudo-temporal trajectories

CRITICAL: All plots use CORRECT biological comparisons:
- main_group: KPT vs KPNT (primary comparison)
- genotype_full: KPT-cis, KPT-trans, KPNT-cis, KPNT-trans (4-way)
- NEVER compare cis vs trans alone (KPT cis and KPNT cis are VERY DIFFERENT)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import os
from scipy.stats import mannwhitneyu


def add_pvalue_annotation(ax, df, value_col, group_col='main_group', x1=0, x2=1):
    """Add p-value annotation comparing two groups."""
    try:
        groups = df[group_col].dropna().unique()
        if len(groups) == 2:
            data1 = df[df[group_col] == groups[0]][value_col].dropna()
            data2 = df[df[group_col] == groups[1]][value_col].dropna()

            if len(data1) > 0 and len(data2) > 0:
                stat, p = mannwhitneyu(data1, data2, alternative='two-sided')

                if p < 0.001:
                    p_text = 'p < 0.001***'
                elif p < 0.01:
                    p_text = f'p = {p:.3f}**'
                elif p < 0.05:
                    p_text = f'p = {p:.3f}*'
                else:
                    p_text = f'p = {p:.3f} ns'

                # Get y position for annotation
                y_max = df[value_col].max()
                y_pos = y_max * 1.1

                # Draw annotation
                ax.plot([x1, x1, x2, x2], [y_pos, y_pos*1.02, y_pos*1.02, y_pos],
                       lw=1.5, c='black')
                ax.text((x1+x2)/2, y_pos*1.03, p_text, ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
    except Exception as e:
        print(f"    Warning: Could not add p-value annotation: {e}")


def plot_perk_analysis_comprehensive(output_dir: str):
    """Generate comprehensive 3×3 pERK analysis plots with CORRECT comparisons."""
    print("\nGenerating comprehensive pERK analysis plots...")

    perk_dir = f"{output_dir}/advanced_perk_analysis"
    if not os.path.exists(perk_dir):
        print("  No pERK analysis data found, skipping...")
        return

    os.makedirs(f"{perk_dir}/figures", exist_ok=True)

    # ========== CLUSTERING ANALYSIS ==========
    clustering_file = f"{perk_dir}/perk_clustering_analysis.csv"
    if os.path.exists(clustering_file):
        try:
            df = pd.read_csv(clustering_file)

            # Create 3×3 comprehensive figure
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # ===== ROW 1: KPT vs KPNT Main Comparisons =====

            # Plot 1: Number of clusters - KPT vs KPNT over time
            ax1 = fig.add_subplot(gs[0, 0])
            if 'main_group' in df.columns and 'timepoint' in df.columns and 'n_ninja_clusters' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['n_ninja_clusters'].mean()
                    temporal_sem = group_data.groupby('timepoint')['n_ninja_clusters'].sem()
                    ax1.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                               marker='o', label=group, linewidth=2.5, markersize=8, capsize=5)
                ax1.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax1.set_ylabel('Number of pERK+ Clusters', fontweight='bold', fontsize=11)
                ax1.set_title('pERK+ Cluster Count Over Time\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax1.legend(frameon=True, fontsize=10)
                ax1.grid(True, alpha=0.3)

            # Plot 2: % Clustered - KPT vs KPNT over time
            ax2 = fig.add_subplot(gs[0, 1])
            if 'main_group' in df.columns and 'pct_clustered' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['pct_clustered'].mean()
                    temporal_sem = group_data.groupby('timepoint')['pct_clustered'].sem()
                    ax2.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                               marker='o', label=group, linewidth=2.5, markersize=8, capsize=5)
                ax2.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax2.set_ylabel('% pERK+ Cells Clustered', fontweight='bold', fontsize=11)
                ax2.set_title('pERK+ Clustering Efficiency\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax2.legend(frameon=True, fontsize=10)
                ax2.grid(True, alpha=0.3)

            # Plot 3: Boxplot comparison - KPT vs KPNT
            ax3 = fig.add_subplot(gs[0, 2])
            if 'main_group' in df.columns and 'n_ninja_clusters' in df.columns:
                sns.boxplot(data=df, x='main_group', y='n_ninja_clusters',
                           ax=ax3, palette='Set2', linewidth=2)
                add_pvalue_annotation(ax3, df, 'n_ninja_clusters', 'main_group')
                ax3.set_ylabel('Number of pERK+ Clusters', fontweight='bold', fontsize=11)
                ax3.set_xlabel('Group', fontweight='bold', fontsize=11)
                ax3.set_title('pERK+ Cluster Count\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax3.grid(True, alpha=0.3, axis='y')

            # ===== ROW 2: 4-Way Genotype Comparisons =====

            # Plot 4: Cluster count - 4-way over time
            ax4 = fig.add_subplot(gs[1, 0])
            if 'genotype_full' in df.columns and 'n_ninja_clusters' in df.columns:
                for genotype in df['genotype_full'].unique():
                    gf_data = df[df['genotype_full'] == genotype]
                    temporal_mean = gf_data.groupby('timepoint')['n_ninja_clusters'].mean()
                    ax4.plot(temporal_mean.index, temporal_mean.values,
                           marker='o', label=genotype, linewidth=2, markersize=7)
                ax4.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax4.set_ylabel('Number of pERK+ Clusters', fontweight='bold', fontsize=11)
                ax4.set_title('pERK+ Clusters: 4-Way Comparison', fontweight='bold', fontsize=12)
                ax4.legend(frameon=True, fontsize=9)
                ax4.grid(True, alpha=0.3)

            # Plot 5: % Clustered - 4-way over time
            ax5 = fig.add_subplot(gs[1, 1])
            if 'genotype_full' in df.columns and 'pct_clustered' in df.columns:
                for genotype in df['genotype_full'].unique():
                    gf_data = df[df['genotype_full'] == genotype]
                    temporal_mean = gf_data.groupby('timepoint')['pct_clustered'].mean()
                    ax5.plot(temporal_mean.index, temporal_mean.values,
                           marker='s', label=genotype, linewidth=2, markersize=7)
                ax5.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax5.set_ylabel('% pERK+ Cells Clustered', fontweight='bold', fontsize=11)
                ax5.set_title('Clustering Efficiency: 4-Way', fontweight='bold', fontsize=12)
                ax5.legend(frameon=True, fontsize=9)
                ax5.grid(True, alpha=0.3)

            # Plot 6: Violin plot - 4-way cluster count
            ax6 = fig.add_subplot(gs[1, 2])
            if 'genotype_full' in df.columns and 'n_ninja_clusters' in df.columns:
                sns.violinplot(data=df, x='genotype_full', y='n_ninja_clusters',
                              ax=ax6, palette='tab10')
                ax6.set_ylabel('Number of pERK+ Clusters', fontweight='bold', fontsize=11)
                ax6.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                ax6.set_title('Cluster Distribution: 4-Way', fontweight='bold', fontsize=12)
                ax6.tick_params(axis='x', rotation=45)
                ax6.grid(True, alpha=0.3, axis='y')

            # ===== ROW 3: Detailed Analyses =====

            # Plot 7: Mean cluster size - KPT vs KPNT
            ax7 = fig.add_subplot(gs[2, 0])
            if 'main_group' in df.columns and 'mean_cluster_size' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['mean_cluster_size'].mean()
                    ax7.plot(temporal_mean.index, temporal_mean.values,
                           marker='o', label=group, linewidth=2.5, markersize=8)
                ax7.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax7.set_ylabel('Mean Cluster Size (cells)', fontweight='bold', fontsize=11)
                ax7.set_title('pERK+ Cluster Size\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax7.legend(frameon=True, fontsize=10)
                ax7.grid(True, alpha=0.3)

            # Plot 8: Scatter - cluster count vs % clustered
            ax8 = fig.add_subplot(gs[2, 1])
            if 'main_group' in df.columns and 'n_ninja_clusters' in df.columns and 'pct_clustered' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    ax8.scatter(group_data['n_ninja_clusters'], group_data['pct_clustered'],
                              alpha=0.6, s=60, label=group)
                ax8.set_xlabel('Number of Clusters', fontweight='bold', fontsize=11)
                ax8.set_ylabel('% Cells Clustered', fontweight='bold', fontsize=11)
                ax8.set_title('Cluster Organization\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax8.legend(frameon=True, fontsize=10)
                ax8.grid(True, alpha=0.3)

            # Plot 9: Heatmap of mean values
            ax9 = fig.add_subplot(gs[2, 2])
            if 'genotype_full' in df.columns:
                # Create summary table
                metrics = []
                if 'n_ninja_clusters' in df.columns:
                    metrics.append('n_ninja_clusters')
                if 'pct_clustered' in df.columns:
                    metrics.append('pct_clustered')
                if 'mean_cluster_size' in df.columns:
                    metrics.append('mean_cluster_size')

                if metrics:
                    summary = df.groupby('genotype_full')[metrics].mean()
                    # Normalize for heatmap
                    summary_norm = (summary - summary.min()) / (summary.max() - summary.min())
                    sns.heatmap(summary_norm.T, annot=summary.T, fmt='.1f',
                               cmap='YlOrRd', ax=ax9, cbar_kws={'label': 'Normalized Value'})
                    ax9.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                    ax9.set_ylabel('Metric', fontweight='bold', fontsize=11)
                    ax9.set_title('pERK+ Clustering Summary', fontweight='bold', fontsize=12)

            plt.suptitle('Comprehensive pERK+ Clustering Analysis', fontsize=16, fontweight='bold', y=0.995)
            plt.savefig(f"{perk_dir}/figures/perk_clustering_comprehensive.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Comprehensive pERK clustering plots saved (3×3 grid)")
        except Exception as e:
            print(f"  WARNING: pERK clustering plots failed: {e}")

    # ========== GROWTH DYNAMICS ANALYSIS ==========
    growth_file = f"{perk_dir}/perk_growth_dynamics.csv"
    if os.path.exists(growth_file):
        try:
            df = pd.read_csv(growth_file)

            # Create 3×3 comprehensive figure
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # ===== ROW 1: KPT vs KPNT Growth Patterns =====

            # Plot 1: pERK+ fraction over time - KPT vs KPNT
            ax1 = fig.add_subplot(gs[0, 0])
            if 'main_group' in df.columns and 'pct_ninja' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['pct_ninja'].mean()
                    temporal_sem = group_data.groupby('timepoint')['pct_ninja'].sem()
                    ax1.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                               marker='o', label=group, linewidth=2.5, markersize=8, capsize=5)
                ax1.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax1.set_ylabel('% pERK+ of Total Tumor', fontweight='bold', fontsize=11)
                ax1.set_title('pERK+ Fraction Growth\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax1.legend(frameon=True, fontsize=10)
                ax1.grid(True, alpha=0.3)

            # Plot 2: pERK+ absolute count - KPT vs KPNT
            ax2 = fig.add_subplot(gs[0, 1])
            if 'main_group' in df.columns and 'n_ninja' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['n_ninja'].mean()
                    temporal_sem = group_data.groupby('timepoint')['n_ninja'].sem()
                    ax2.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                               marker='o', label=group, linewidth=2.5, markersize=8, capsize=5)
                ax2.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax2.set_ylabel('pERK+ Cell Count', fontweight='bold', fontsize=11)
                ax2.set_title('pERK+ Absolute Growth\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax2.legend(frameon=True, fontsize=10)
                ax2.grid(True, alpha=0.3)

            # Plot 3: Boxplot - pERK+ fraction by group
            ax3 = fig.add_subplot(gs[0, 2])
            if 'main_group' in df.columns and 'pct_ninja' in df.columns:
                sns.boxplot(data=df, x='main_group', y='pct_ninja',
                           ax=ax3, palette='Set2', linewidth=2)
                add_pvalue_annotation(ax3, df, 'pct_ninja', 'main_group')
                ax3.set_ylabel('% pERK+ of Total Tumor', fontweight='bold', fontsize=11)
                ax3.set_xlabel('Group', fontweight='bold', fontsize=11)
                ax3.set_title('pERK+ Fraction\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax3.grid(True, alpha=0.3, axis='y')

            # ===== ROW 2: 4-Way Comparisons =====

            # Plot 4: pERK+ fraction - 4-way
            ax4 = fig.add_subplot(gs[1, 0])
            if 'genotype_full' in df.columns and 'pct_ninja' in df.columns:
                for genotype in df['genotype_full'].unique():
                    gf_data = df[df['genotype_full'] == genotype]
                    temporal_mean = gf_data.groupby('timepoint')['pct_ninja'].mean()
                    ax4.plot(temporal_mean.index, temporal_mean.values,
                           marker='o', label=genotype, linewidth=2, markersize=7)
                ax4.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax4.set_ylabel('% pERK+ of Total Tumor', fontweight='bold', fontsize=11)
                ax4.set_title('pERK+ Fraction: 4-Way', fontweight='bold', fontsize=12)
                ax4.legend(frameon=True, fontsize=9)
                ax4.grid(True, alpha=0.3)

            # Plot 5: pERK+ count - 4-way
            ax5 = fig.add_subplot(gs[1, 1])
            if 'genotype_full' in df.columns and 'n_ninja' in df.columns:
                for genotype in df['genotype_full'].unique():
                    gf_data = df[df['genotype_full'] == genotype]
                    temporal_mean = gf_data.groupby('timepoint')['n_ninja'].mean()
                    ax5.plot(temporal_mean.index, temporal_mean.values,
                           marker='s', label=genotype, linewidth=2, markersize=7)
                ax5.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax5.set_ylabel('pERK+ Cell Count', fontweight='bold', fontsize=11)
                ax5.set_title('pERK+ Count: 4-Way', fontweight='bold', fontsize=12)
                ax5.legend(frameon=True, fontsize=9)
                ax5.grid(True, alpha=0.3)

            # Plot 6: Violin plot - 4-way fraction
            ax6 = fig.add_subplot(gs[1, 2])
            if 'genotype_full' in df.columns and 'pct_ninja' in df.columns:
                sns.violinplot(data=df, x='genotype_full', y='pct_ninja',
                              ax=ax6, palette='tab10')
                ax6.set_ylabel('% pERK+ of Total Tumor', fontweight='bold', fontsize=11)
                ax6.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                ax6.set_title('pERK+ Fraction: 4-Way', fontweight='bold', fontsize=12)
                ax6.tick_params(axis='x', rotation=45)
                ax6.grid(True, alpha=0.3, axis='y')

            # ===== ROW 3: Growth Rate and Dynamics =====

            # Plot 7: Growth rate (delta pERK+ between timepoints)
            ax7 = fig.add_subplot(gs[2, 0])
            if 'main_group' in df.columns and 'timepoint' in df.columns and 'pct_ninja' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group].sort_values('timepoint')
                    temporal_mean = group_data.groupby('timepoint')['pct_ninja'].mean()
                    # Calculate growth rate
                    growth_rate = temporal_mean.diff()
                    ax7.plot(growth_rate.index[1:], growth_rate.values[1:],
                           marker='o', label=group, linewidth=2.5, markersize=8)
                ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax7.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax7.set_ylabel('Δ pERK+ Fraction', fontweight='bold', fontsize=11)
                ax7.set_title('pERK+ Growth Rate\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax7.legend(frameon=True, fontsize=10)
                ax7.grid(True, alpha=0.3)

            # Plot 8: Scatter - total tumor vs pERK+ fraction
            ax8 = fig.add_subplot(gs[2, 1])
            if 'main_group' in df.columns and 'n_total' in df.columns and 'pct_ninja' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    ax8.scatter(group_data['n_total'], group_data['pct_ninja'],
                              alpha=0.6, s=60, label=group)
                ax8.set_xlabel('Total Tumor Cells', fontweight='bold', fontsize=11)
                ax8.set_ylabel('% pERK+', fontweight='bold', fontsize=11)
                ax8.set_title('pERK+ vs Tumor Size\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax8.legend(frameon=True, fontsize=10)
                ax8.grid(True, alpha=0.3)

            # Plot 9: Summary heatmap
            ax9 = fig.add_subplot(gs[2, 2])
            if 'genotype_full' in df.columns:
                metrics = []
                if 'pct_ninja' in df.columns:
                    metrics.append('pct_ninja')
                if 'n_ninja' in df.columns:
                    metrics.append('n_ninja')

                if metrics:
                    summary = df.groupby('genotype_full')[metrics].mean()
                    summary_norm = (summary - summary.min()) / (summary.max() - summary.min())
                    sns.heatmap(summary_norm.T, annot=summary.T, fmt='.1f',
                               cmap='RdPu', ax=ax9, cbar_kws={'label': 'Normalized Value'})
                    ax9.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                    ax9.set_ylabel('Metric', fontweight='bold', fontsize=11)
                    ax9.set_title('pERK+ Growth Summary', fontweight='bold', fontsize=12)

            plt.suptitle('Comprehensive pERK+ Growth Dynamics', fontsize=16, fontweight='bold', y=0.995)
            plt.savefig(f"{perk_dir}/figures/perk_growth_comprehensive.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Comprehensive pERK growth plots saved (3×3 grid)")
        except Exception as e:
            print(f"  WARNING: pERK growth plots failed: {e}")


def plot_ninja_analysis_comprehensive(output_dir: str):
    """Generate comprehensive 3×3 NINJA analysis plots with CORRECT comparisons."""
    print("\nGenerating comprehensive NINJA analysis plots...")

    ninja_dir = f"{output_dir}/advanced_ninja_analysis"
    if not os.path.exists(ninja_dir):
        print("  No NINJA analysis data found, skipping...")
        return

    os.makedirs(f"{ninja_dir}/figures", exist_ok=True)

    # ========== CLUSTERING ANALYSIS ==========
    clustering_file = f"{ninja_dir}/ninja_clustering_analysis.csv"
    if os.path.exists(clustering_file):
        try:
            df = pd.read_csv(clustering_file)

            # Create 3×3 comprehensive figure
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # ===== ROW 1: KPT vs KPNT Main Comparisons =====

            # Plot 1: Number of clusters - KPT vs KPNT over time
            ax1 = fig.add_subplot(gs[0, 0])
            if 'main_group' in df.columns and 'timepoint' in df.columns and 'n_ninja_clusters' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['n_ninja_clusters'].mean()
                    temporal_sem = group_data.groupby('timepoint')['n_ninja_clusters'].sem()
                    ax1.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                               marker='s', label=group, linewidth=2.5, markersize=8, capsize=5)
                ax1.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax1.set_ylabel('Number of NINJA+ Clusters', fontweight='bold', fontsize=11)
                ax1.set_title('NINJA+ Cluster Count Over Time\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax1.legend(frameon=True, fontsize=10)
                ax1.grid(True, alpha=0.3)

            # Plot 2: % Clustered - KPT vs KPNT over time
            ax2 = fig.add_subplot(gs[0, 1])
            if 'main_group' in df.columns and 'pct_clustered' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['pct_clustered'].mean()
                    temporal_sem = group_data.groupby('timepoint')['pct_clustered'].sem()
                    ax2.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                               marker='s', label=group, linewidth=2.5, markersize=8, capsize=5)
                ax2.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax2.set_ylabel('% NINJA+ Cells Clustered', fontweight='bold', fontsize=11)
                ax2.set_title('NINJA+ Clustering Efficiency\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax2.legend(frameon=True, fontsize=10)
                ax2.grid(True, alpha=0.3)

            # Plot 3: Boxplot comparison - KPT vs KPNT
            ax3 = fig.add_subplot(gs[0, 2])
            if 'main_group' in df.columns and 'n_ninja_clusters' in df.columns:
                sns.boxplot(data=df, x='main_group', y='n_ninja_clusters',
                           ax=ax3, palette='Set3', linewidth=2)
                add_pvalue_annotation(ax3, df, 'n_ninja_clusters', 'main_group')
                ax3.set_ylabel('Number of NINJA+ Clusters', fontweight='bold', fontsize=11)
                ax3.set_xlabel('Group', fontweight='bold', fontsize=11)
                ax3.set_title('NINJA+ Cluster Count\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax3.grid(True, alpha=0.3, axis='y')

            # ===== ROW 2: 4-Way Genotype Comparisons =====

            # Plot 4: Cluster count - 4-way over time
            ax4 = fig.add_subplot(gs[1, 0])
            if 'genotype_full' in df.columns and 'n_ninja_clusters' in df.columns:
                for genotype in df['genotype_full'].unique():
                    gf_data = df[df['genotype_full'] == genotype]
                    temporal_mean = gf_data.groupby('timepoint')['n_ninja_clusters'].mean()
                    ax4.plot(temporal_mean.index, temporal_mean.values,
                           marker='s', label=genotype, linewidth=2, markersize=7)
                ax4.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax4.set_ylabel('Number of NINJA+ Clusters', fontweight='bold', fontsize=11)
                ax4.set_title('NINJA+ Clusters: 4-Way Comparison', fontweight='bold', fontsize=12)
                ax4.legend(frameon=True, fontsize=9)
                ax4.grid(True, alpha=0.3)

            # Plot 5: % Clustered - 4-way over time
            ax5 = fig.add_subplot(gs[1, 1])
            if 'genotype_full' in df.columns and 'pct_clustered' in df.columns:
                for genotype in df['genotype_full'].unique():
                    gf_data = df[df['genotype_full'] == genotype]
                    temporal_mean = gf_data.groupby('timepoint')['pct_clustered'].mean()
                    ax5.plot(temporal_mean.index, temporal_mean.values,
                           marker='D', label=genotype, linewidth=2, markersize=7)
                ax5.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax5.set_ylabel('% NINJA+ Cells Clustered', fontweight='bold', fontsize=11)
                ax5.set_title('Clustering Efficiency: 4-Way', fontweight='bold', fontsize=12)
                ax5.legend(frameon=True, fontsize=9)
                ax5.grid(True, alpha=0.3)

            # Plot 6: Violin plot - 4-way cluster count
            ax6 = fig.add_subplot(gs[1, 2])
            if 'genotype_full' in df.columns and 'n_ninja_clusters' in df.columns:
                sns.violinplot(data=df, x='genotype_full', y='n_ninja_clusters',
                              ax=ax6, palette='pastel')
                ax6.set_ylabel('Number of NINJA+ Clusters', fontweight='bold', fontsize=11)
                ax6.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                ax6.set_title('Cluster Distribution: 4-Way', fontweight='bold', fontsize=12)
                ax6.tick_params(axis='x', rotation=45)
                ax6.grid(True, alpha=0.3, axis='y')

            # ===== ROW 3: Detailed Analyses =====

            # Plot 7: Mean cluster size - KPT vs KPNT
            ax7 = fig.add_subplot(gs[2, 0])
            if 'main_group' in df.columns and 'mean_cluster_size' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['mean_cluster_size'].mean()
                    ax7.plot(temporal_mean.index, temporal_mean.values,
                           marker='s', label=group, linewidth=2.5, markersize=8)
                ax7.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax7.set_ylabel('Mean Cluster Size (cells)', fontweight='bold', fontsize=11)
                ax7.set_title('NINJA+ Cluster Size\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax7.legend(frameon=True, fontsize=10)
                ax7.grid(True, alpha=0.3)

            # Plot 8: Scatter - cluster count vs % clustered
            ax8 = fig.add_subplot(gs[2, 1])
            if 'main_group' in df.columns and 'n_ninja_clusters' in df.columns and 'pct_clustered' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    ax8.scatter(group_data['n_ninja_clusters'], group_data['pct_clustered'],
                              alpha=0.6, s=60, label=group, marker='s')
                ax8.set_xlabel('Number of Clusters', fontweight='bold', fontsize=11)
                ax8.set_ylabel('% Cells Clustered', fontweight='bold', fontsize=11)
                ax8.set_title('NINJA+ Cluster Organization\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax8.legend(frameon=True, fontsize=10)
                ax8.grid(True, alpha=0.3)

            # Plot 9: Heatmap of mean values
            ax9 = fig.add_subplot(gs[2, 2])
            if 'genotype_full' in df.columns:
                metrics = []
                if 'n_ninja_clusters' in df.columns:
                    metrics.append('n_ninja_clusters')
                if 'pct_clustered' in df.columns:
                    metrics.append('pct_clustered')
                if 'mean_cluster_size' in df.columns:
                    metrics.append('mean_cluster_size')

                if metrics:
                    summary = df.groupby('genotype_full')[metrics].mean()
                    summary_norm = (summary - summary.min()) / (summary.max() - summary.min())
                    sns.heatmap(summary_norm.T, annot=summary.T, fmt='.1f',
                               cmap='Purples', ax=ax9, cbar_kws={'label': 'Normalized Value'})
                    ax9.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                    ax9.set_ylabel('Metric', fontweight='bold', fontsize=11)
                    ax9.set_title('NINJA+ Clustering Summary', fontweight='bold', fontsize=12)

            plt.suptitle('Comprehensive NINJA+ Clustering Analysis', fontsize=16, fontweight='bold', y=0.995)
            plt.savefig(f"{ninja_dir}/figures/ninja_clustering_comprehensive.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Comprehensive NINJA clustering plots saved (3×3 grid)")
        except Exception as e:
            print(f"  WARNING: NINJA clustering plots failed: {e}")


def plot_heterogeneity_analysis_comprehensive(output_dir: str):
    """Generate comprehensive 3×3 heterogeneity analysis plots with CORRECT comparisons."""
    print("\nGenerating comprehensive heterogeneity analysis plots...")

    het_dir = f"{output_dir}/advanced_heterogeneity"
    if not os.path.exists(het_dir):
        print("  No heterogeneity analysis data found, skipping...")
        return

    os.makedirs(f"{het_dir}/figures", exist_ok=True)

    # ========== ENTROPY ANALYSIS ==========
    entropy_file = f"{het_dir}/marker_entropy_analysis.csv"
    if os.path.exists(entropy_file):
        try:
            df = pd.read_csv(entropy_file)

            # Create 3×3 comprehensive figure
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # ===== ROW 1: KPT vs KPNT Main Comparisons =====

            # Plot 1: Mean entropy over time - KPT vs KPNT
            ax1 = fig.add_subplot(gs[0, 0])
            if 'main_group' in df.columns and 'mean_marker_entropy' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['mean_marker_entropy'].mean()
                    temporal_sem = group_data.groupby('timepoint')['mean_marker_entropy'].sem()
                    ax1.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                               marker='o', label=group, linewidth=2.5, markersize=8, capsize=5)
                ax1.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax1.set_ylabel('Mean Marker Entropy', fontweight='bold', fontsize=11)
                ax1.set_title('Marker Diversification Over Time\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax1.legend(frameon=True, fontsize=10)
                ax1.grid(True, alpha=0.3)

            # Plot 2: Max entropy over time - KPT vs KPNT
            ax2 = fig.add_subplot(gs[0, 1])
            if 'main_group' in df.columns and 'max_marker_entropy' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['max_marker_entropy'].mean()
                    temporal_sem = group_data.groupby('timepoint')['max_marker_entropy'].sem()
                    ax2.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                               marker='o', label=group, linewidth=2.5, markersize=8, capsize=5)
                ax2.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax2.set_ylabel('Max Marker Entropy', fontweight='bold', fontsize=11)
                ax2.set_title('Peak Marker Diversity\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax2.legend(frameon=True, fontsize=10)
                ax2.grid(True, alpha=0.3)

            # Plot 3: Boxplot - mean entropy by group
            ax3 = fig.add_subplot(gs[0, 2])
            if 'main_group' in df.columns and 'mean_marker_entropy' in df.columns:
                sns.boxplot(data=df, x='main_group', y='mean_marker_entropy',
                           ax=ax3, palette='viridis', linewidth=2)
                add_pvalue_annotation(ax3, df, 'mean_marker_entropy', 'main_group')
                ax3.set_ylabel('Mean Marker Entropy', fontweight='bold', fontsize=11)
                ax3.set_xlabel('Group', fontweight='bold', fontsize=11)
                ax3.set_title('Marker Entropy\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax3.grid(True, alpha=0.3, axis='y')

            # ===== ROW 2: 4-Way Genotype Comparisons =====

            # Plot 4: Mean entropy - 4-way over time
            ax4 = fig.add_subplot(gs[1, 0])
            if 'genotype_full' in df.columns and 'mean_marker_entropy' in df.columns:
                for genotype in df['genotype_full'].unique():
                    gf_data = df[df['genotype_full'] == genotype]
                    temporal_mean = gf_data.groupby('timepoint')['mean_marker_entropy'].mean()
                    ax4.plot(temporal_mean.index, temporal_mean.values,
                           marker='o', label=genotype, linewidth=2, markersize=7)
                ax4.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax4.set_ylabel('Mean Marker Entropy', fontweight='bold', fontsize=11)
                ax4.set_title('Mean Entropy: 4-Way', fontweight='bold', fontsize=12)
                ax4.legend(frameon=True, fontsize=9)
                ax4.grid(True, alpha=0.3)

            # Plot 5: Max entropy - 4-way over time
            ax5 = fig.add_subplot(gs[1, 1])
            if 'genotype_full' in df.columns and 'max_marker_entropy' in df.columns:
                for genotype in df['genotype_full'].unique():
                    gf_data = df[df['genotype_full'] == genotype]
                    temporal_mean = gf_data.groupby('timepoint')['max_marker_entropy'].mean()
                    ax5.plot(temporal_mean.index, temporal_mean.values,
                           marker='s', label=genotype, linewidth=2, markersize=7)
                ax5.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax5.set_ylabel('Max Marker Entropy', fontweight='bold', fontsize=11)
                ax5.set_title('Max Entropy: 4-Way', fontweight='bold', fontsize=12)
                ax5.legend(frameon=True, fontsize=9)
                ax5.grid(True, alpha=0.3)

            # Plot 6: Violin plot - 4-way mean entropy
            ax6 = fig.add_subplot(gs[1, 2])
            if 'genotype_full' in df.columns and 'mean_marker_entropy' in df.columns:
                sns.violinplot(data=df, x='genotype_full', y='mean_marker_entropy',
                              ax=ax6, palette='viridis')
                ax6.set_ylabel('Mean Marker Entropy', fontweight='bold', fontsize=11)
                ax6.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                ax6.set_title('Entropy Distribution: 4-Way', fontweight='bold', fontsize=12)
                ax6.tick_params(axis='x', rotation=45)
                ax6.grid(True, alpha=0.3, axis='y')

            # ===== ROW 3: Detailed Analyses =====

            # Plot 7: Entropy range (max - min) - KPT vs KPNT
            ax7 = fig.add_subplot(gs[2, 0])
            if 'main_group' in df.columns and 'max_marker_entropy' in df.columns and 'min_marker_entropy' in df.columns:
                df['entropy_range'] = df['max_marker_entropy'] - df['min_marker_entropy']
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['entropy_range'].mean()
                    ax7.plot(temporal_mean.index, temporal_mean.values,
                           marker='o', label=group, linewidth=2.5, markersize=8)
                ax7.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax7.set_ylabel('Entropy Range (max - min)', fontweight='bold', fontsize=11)
                ax7.set_title('Marker Diversity Spread\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax7.legend(frameon=True, fontsize=10)
                ax7.grid(True, alpha=0.3)

            # Plot 8: Scatter - mean vs max entropy
            ax8 = fig.add_subplot(gs[2, 1])
            if 'main_group' in df.columns and 'mean_marker_entropy' in df.columns and 'max_marker_entropy' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    ax8.scatter(group_data['mean_marker_entropy'], group_data['max_marker_entropy'],
                              alpha=0.6, s=60, label=group)
                ax8.set_xlabel('Mean Marker Entropy', fontweight='bold', fontsize=11)
                ax8.set_ylabel('Max Marker Entropy', fontweight='bold', fontsize=11)
                ax8.set_title('Entropy Distribution\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax8.legend(frameon=True, fontsize=10)
                ax8.grid(True, alpha=0.3)

            # Plot 9: Heatmap of summary
            ax9 = fig.add_subplot(gs[2, 2])
            if 'genotype_full' in df.columns:
                metrics = []
                if 'mean_marker_entropy' in df.columns:
                    metrics.append('mean_marker_entropy')
                if 'max_marker_entropy' in df.columns:
                    metrics.append('max_marker_entropy')
                if 'min_marker_entropy' in df.columns:
                    metrics.append('min_marker_entropy')

                if metrics:
                    summary = df.groupby('genotype_full')[metrics].mean()
                    summary_norm = (summary - summary.min()) / (summary.max() - summary.min())
                    sns.heatmap(summary_norm.T, annot=summary.T, fmt='.2f',
                               cmap='plasma', ax=ax9, cbar_kws={'label': 'Normalized Value'})
                    ax9.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                    ax9.set_ylabel('Metric', fontweight='bold', fontsize=11)
                    ax9.set_title('Entropy Summary', fontweight='bold', fontsize=12)

            plt.suptitle('Comprehensive Marker Entropy Analysis', fontsize=16, fontweight='bold', y=0.995)
            plt.savefig(f"{het_dir}/figures/entropy_comprehensive.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Comprehensive entropy plots saved (3×3 grid)")
        except Exception as e:
            print(f"  WARNING: Entropy plots failed: {e}")

    # ========== HETEROGENEITY METRICS ==========
    het_file = f"{het_dir}/heterogeneity_metrics.csv"
    if os.path.exists(het_file):
        try:
            df = pd.read_csv(het_file)

            # Create 3×3 comprehensive figure
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # ===== ROW 1: KPT vs KPNT Main Comparisons =====

            # Plot 1: Heterogeneity score over time
            ax1 = fig.add_subplot(gs[0, 0])
            if 'main_group' in df.columns and 'heterogeneity_score' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['heterogeneity_score'].mean()
                    temporal_sem = group_data.groupby('timepoint')['heterogeneity_score'].sem()
                    ax1.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                               marker='o', label=group, linewidth=2.5, markersize=8, capsize=5)
                ax1.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax1.set_ylabel('Heterogeneity Score (CV)', fontweight='bold', fontsize=11)
                ax1.set_title('Intra-Tumor Heterogeneity\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax1.legend(frameon=True, fontsize=10)
                ax1.grid(True, alpha=0.3)

            # Plot 2: Marker variance over time
            ax2 = fig.add_subplot(gs[0, 1])
            if 'main_group' in df.columns and 'marker_variance' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['marker_variance'].mean()
                    ax2.plot(temporal_mean.index, temporal_mean.values,
                           marker='o', label=group, linewidth=2.5, markersize=8)
                ax2.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax2.set_ylabel('Marker Variance', fontweight='bold', fontsize=11)
                ax2.set_title('Marker Expression Variance\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax2.legend(frameon=True, fontsize=10)
                ax2.grid(True, alpha=0.3)

            # Plot 3: Boxplot - heterogeneity score
            ax3 = fig.add_subplot(gs[0, 2])
            if 'main_group' in df.columns and 'heterogeneity_score' in df.columns:
                sns.boxplot(data=df, x='main_group', y='heterogeneity_score',
                           ax=ax3, palette='coolwarm', linewidth=2)
                add_pvalue_annotation(ax3, df, 'heterogeneity_score', 'main_group')
                ax3.set_ylabel('Heterogeneity Score (CV)', fontweight='bold', fontsize=11)
                ax3.set_xlabel('Group', fontweight='bold', fontsize=11)
                ax3.set_title('Heterogeneity Score\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax3.grid(True, alpha=0.3, axis='y')

            # ===== ROW 2: 4-Way Comparisons =====

            # Plot 4: Heterogeneity score - 4-way
            ax4 = fig.add_subplot(gs[1, 0])
            if 'genotype_full' in df.columns and 'heterogeneity_score' in df.columns:
                for genotype in df['genotype_full'].unique():
                    gf_data = df[df['genotype_full'] == genotype]
                    temporal_mean = gf_data.groupby('timepoint')['heterogeneity_score'].mean()
                    ax4.plot(temporal_mean.index, temporal_mean.values,
                           marker='o', label=genotype, linewidth=2, markersize=7)
                ax4.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax4.set_ylabel('Heterogeneity Score', fontweight='bold', fontsize=11)
                ax4.set_title('Heterogeneity: 4-Way', fontweight='bold', fontsize=12)
                ax4.legend(frameon=True, fontsize=9)
                ax4.grid(True, alpha=0.3)

            # Plot 5: Marker variance - 4-way
            ax5 = fig.add_subplot(gs[1, 1])
            if 'genotype_full' in df.columns and 'marker_variance' in df.columns:
                for genotype in df['genotype_full'].unique():
                    gf_data = df[df['genotype_full'] == genotype]
                    temporal_mean = gf_data.groupby('timepoint')['marker_variance'].mean()
                    ax5.plot(temporal_mean.index, temporal_mean.values,
                           marker='s', label=genotype, linewidth=2, markersize=7)
                ax5.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax5.set_ylabel('Marker Variance', fontweight='bold', fontsize=11)
                ax5.set_title('Variance: 4-Way', fontweight='bold', fontsize=12)
                ax5.legend(frameon=True, fontsize=9)
                ax5.grid(True, alpha=0.3)

            # Plot 6: Violin plot - 4-way heterogeneity
            ax6 = fig.add_subplot(gs[1, 2])
            if 'genotype_full' in df.columns and 'heterogeneity_score' in df.columns:
                sns.violinplot(data=df, x='genotype_full', y='heterogeneity_score',
                              ax=ax6, palette='coolwarm')
                ax6.set_ylabel('Heterogeneity Score', fontweight='bold', fontsize=11)
                ax6.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                ax6.set_title('Heterogeneity: 4-Way', fontweight='bold', fontsize=12)
                ax6.tick_params(axis='x', rotation=45)
                ax6.grid(True, alpha=0.3, axis='y')

            # ===== ROW 3: Advanced Metrics =====

            # Plot 7: Coefficient of variation trends
            ax7 = fig.add_subplot(gs[2, 0])
            if 'main_group' in df.columns and 'heterogeneity_score' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group].sort_values('timepoint')
                    temporal_mean = group_data.groupby('timepoint')['heterogeneity_score'].mean()
                    # Calculate change rate
                    cv_change = temporal_mean.diff()
                    ax7.plot(cv_change.index[1:], cv_change.values[1:],
                           marker='o', label=group, linewidth=2.5, markersize=8)
                ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax7.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax7.set_ylabel('Δ Heterogeneity Score', fontweight='bold', fontsize=11)
                ax7.set_title('Heterogeneity Rate of Change\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax7.legend(frameon=True, fontsize=10)
                ax7.grid(True, alpha=0.3)

            # Plot 8: Scatter - heterogeneity vs variance
            ax8 = fig.add_subplot(gs[2, 1])
            if 'main_group' in df.columns and 'heterogeneity_score' in df.columns and 'marker_variance' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    ax8.scatter(group_data['marker_variance'], group_data['heterogeneity_score'],
                              alpha=0.6, s=60, label=group)
                ax8.set_xlabel('Marker Variance', fontweight='bold', fontsize=11)
                ax8.set_ylabel('Heterogeneity Score', fontweight='bold', fontsize=11)
                ax8.set_title('Variance vs Heterogeneity\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax8.legend(frameon=True, fontsize=10)
                ax8.grid(True, alpha=0.3)

            # Plot 9: Summary heatmap
            ax9 = fig.add_subplot(gs[2, 2])
            if 'genotype_full' in df.columns:
                metrics = []
                if 'heterogeneity_score' in df.columns:
                    metrics.append('heterogeneity_score')
                if 'marker_variance' in df.columns:
                    metrics.append('marker_variance')

                if metrics:
                    summary = df.groupby('genotype_full')[metrics].mean()
                    summary_norm = (summary - summary.min()) / (summary.max() - summary.min())
                    sns.heatmap(summary_norm.T, annot=summary.T, fmt='.2f',
                               cmap='coolwarm', ax=ax9, cbar_kws={'label': 'Normalized Value'})
                    ax9.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                    ax9.set_ylabel('Metric', fontweight='bold', fontsize=11)
                    ax9.set_title('Heterogeneity Summary', fontweight='bold', fontsize=12)

            plt.suptitle('Comprehensive Heterogeneity Analysis', fontsize=16, fontweight='bold', y=0.995)
            plt.savefig(f"{het_dir}/figures/heterogeneity_comprehensive.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Comprehensive heterogeneity plots saved (3×3 grid)")
        except Exception as e:
            print(f"  WARNING: Heterogeneity plots failed: {e}")


def plot_enhanced_rcn_comprehensive(output_dir: str):
    """Generate comprehensive 3×3 enhanced RCN dynamics plots."""
    print("\nGenerating comprehensive enhanced RCN dynamics plots...")

    rcn_dir = f"{output_dir}/advanced_rcn_dynamics"
    if not os.path.exists(rcn_dir):
        print("  No enhanced RCN data found, skipping...")
        return

    os.makedirs(f"{rcn_dir}/figures", exist_ok=True)

    # Check for RCN metrics file
    rcn_file = f"{rcn_dir}/enhanced_rcn_metrics.csv"
    if os.path.exists(rcn_file):
        try:
            df = pd.read_csv(rcn_file)

            # Create 3×3 comprehensive figure
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # ===== ROW 1: KPT vs KPNT RCN Metrics =====

            # Plot 1: RCN index over time
            ax1 = fig.add_subplot(gs[0, 0])
            if 'main_group' in df.columns and 'rcn_index' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['rcn_index'].mean()
                    temporal_sem = group_data.groupby('timepoint')['rcn_index'].sem()
                    ax1.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                               marker='o', label=group, linewidth=2.5, markersize=8, capsize=5)
                ax1.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax1.set_ylabel('RCN Index', fontweight='bold', fontsize=11)
                ax1.set_title('Recirculating Niche Index\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax1.legend(frameon=True, fontsize=10)
                ax1.grid(True, alpha=0.3)

            # Plot 2: RCN cell fraction
            ax2 = fig.add_subplot(gs[0, 1])
            if 'main_group' in df.columns and 'pct_rcn_cells' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['pct_rcn_cells'].mean()
                    ax2.plot(temporal_mean.index, temporal_mean.values,
                           marker='o', label=group, linewidth=2.5, markersize=8)
                ax2.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax2.set_ylabel('% RCN Cells', fontweight='bold', fontsize=11)
                ax2.set_title('RCN Cell Fraction\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax2.legend(frameon=True, fontsize=10)
                ax2.grid(True, alpha=0.3)

            # Plot 3: Boxplot - RCN index
            ax3 = fig.add_subplot(gs[0, 2])
            if 'main_group' in df.columns and 'rcn_index' in df.columns:
                sns.boxplot(data=df, x='main_group', y='rcn_index',
                           ax=ax3, palette='Set1', linewidth=2)
                add_pvalue_annotation(ax3, df, 'rcn_index', 'main_group')
                ax3.set_ylabel('RCN Index', fontweight='bold', fontsize=11)
                ax3.set_xlabel('Group', fontweight='bold', fontsize=11)
                ax3.set_title('RCN Index\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax3.grid(True, alpha=0.3, axis='y')

            # ===== ROW 2: 4-Way Comparisons =====

            # Plot 4: RCN index - 4-way
            ax4 = fig.add_subplot(gs[1, 0])
            if 'genotype_full' in df.columns and 'rcn_index' in df.columns:
                for genotype in df['genotype_full'].unique():
                    gf_data = df[df['genotype_full'] == genotype]
                    temporal_mean = gf_data.groupby('timepoint')['rcn_index'].mean()
                    ax4.plot(temporal_mean.index, temporal_mean.values,
                           marker='o', label=genotype, linewidth=2, markersize=7)
                ax4.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax4.set_ylabel('RCN Index', fontweight='bold', fontsize=11)
                ax4.set_title('RCN Index: 4-Way', fontweight='bold', fontsize=12)
                ax4.legend(frameon=True, fontsize=9)
                ax4.grid(True, alpha=0.3)

            # Plot 5: % RCN cells - 4-way
            ax5 = fig.add_subplot(gs[1, 1])
            if 'genotype_full' in df.columns and 'pct_rcn_cells' in df.columns:
                for genotype in df['genotype_full'].unique():
                    gf_data = df[df['genotype_full'] == genotype]
                    temporal_mean = gf_data.groupby('timepoint')['pct_rcn_cells'].mean()
                    ax5.plot(temporal_mean.index, temporal_mean.values,
                           marker='s', label=genotype, linewidth=2, markersize=7)
                ax5.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax5.set_ylabel('% RCN Cells', fontweight='bold', fontsize=11)
                ax5.set_title('RCN Fraction: 4-Way', fontweight='bold', fontsize=12)
                ax5.legend(frameon=True, fontsize=9)
                ax5.grid(True, alpha=0.3)

            # Plot 6: Violin plot - RCN index
            ax6 = fig.add_subplot(gs[1, 2])
            if 'genotype_full' in df.columns and 'rcn_index' in df.columns:
                sns.violinplot(data=df, x='genotype_full', y='rcn_index',
                              ax=ax6, palette='Set1')
                ax6.set_ylabel('RCN Index', fontweight='bold', fontsize=11)
                ax6.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                ax6.set_title('RCN Distribution: 4-Way', fontweight='bold', fontsize=12)
                ax6.tick_params(axis='x', rotation=45)
                ax6.grid(True, alpha=0.3, axis='y')

            # ===== ROW 3: Detailed Analyses =====

            # Plot 7: RCN dynamics (rate of change)
            ax7 = fig.add_subplot(gs[2, 0])
            if 'main_group' in df.columns and 'rcn_index' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group].sort_values('timepoint')
                    temporal_mean = group_data.groupby('timepoint')['rcn_index'].mean()
                    rcn_change = temporal_mean.diff()
                    ax7.plot(rcn_change.index[1:], rcn_change.values[1:],
                           marker='o', label=group, linewidth=2.5, markersize=8)
                ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax7.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax7.set_ylabel('Δ RCN Index', fontweight='bold', fontsize=11)
                ax7.set_title('RCN Dynamics\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax7.legend(frameon=True, fontsize=10)
                ax7.grid(True, alpha=0.3)

            # Plot 8: Scatter - RCN index vs cell fraction
            ax8 = fig.add_subplot(gs[2, 1])
            if 'main_group' in df.columns and 'rcn_index' in df.columns and 'pct_rcn_cells' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    ax8.scatter(group_data['pct_rcn_cells'], group_data['rcn_index'],
                              alpha=0.6, s=60, label=group)
                ax8.set_xlabel('% RCN Cells', fontweight='bold', fontsize=11)
                ax8.set_ylabel('RCN Index', fontweight='bold', fontsize=11)
                ax8.set_title('RCN Organization\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax8.legend(frameon=True, fontsize=10)
                ax8.grid(True, alpha=0.3)

            # Plot 9: Summary heatmap
            ax9 = fig.add_subplot(gs[2, 2])
            if 'genotype_full' in df.columns:
                metrics = []
                if 'rcn_index' in df.columns:
                    metrics.append('rcn_index')
                if 'pct_rcn_cells' in df.columns:
                    metrics.append('pct_rcn_cells')

                if metrics:
                    summary = df.groupby('genotype_full')[metrics].mean()
                    summary_norm = (summary - summary.min()) / (summary.max() - summary.min())
                    sns.heatmap(summary_norm.T, annot=summary.T, fmt='.2f',
                               cmap='Blues', ax=ax9, cbar_kws={'label': 'Normalized Value'})
                    ax9.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                    ax9.set_ylabel('Metric', fontweight='bold', fontsize=11)
                    ax9.set_title('RCN Summary', fontweight='bold', fontsize=12)

            plt.suptitle('Comprehensive Enhanced RCN Dynamics', fontsize=16, fontweight='bold', y=0.995)
            plt.savefig(f"{rcn_dir}/figures/rcn_comprehensive.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Comprehensive RCN plots saved (3×3 grid)")
        except Exception as e:
            print(f"  WARNING: RCN plots failed: {e}")


def plot_multilevel_distances_comprehensive(output_dir: str):
    """Generate comprehensive 3×3 multi-level distance analysis plots."""
    print("\nGenerating comprehensive multi-level distance plots...")

    dist_dir = f"{output_dir}/advanced_distances"
    if not os.path.exists(dist_dir):
        print("  No multi-level distance data found, skipping...")
        return

    os.makedirs(f"{dist_dir}/figures", exist_ok=True)

    # Get distance files by population
    distance_files = [f for f in os.listdir(dist_dir) if f.endswith('_distances.csv')]

    for dist_file in distance_files[:4]:  # Top 4 populations
        try:
            pop_name = dist_file.replace('_distances.csv', '')
            df = pd.read_csv(f"{dist_dir}/{dist_file}")

            # Create 3×3 comprehensive figure for this population
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # ===== ROW 1: Mean Distances - KPT vs KPNT =====

            # Plot 1: Mean distance over time
            ax1 = fig.add_subplot(gs[0, 0])
            if 'main_group' in df.columns and 'mean_distance' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['mean_distance'].mean()
                    temporal_sem = group_data.groupby('timepoint')['mean_distance'].sem()
                    ax1.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                               marker='o', label=group, linewidth=2.5, markersize=8, capsize=5)
                ax1.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax1.set_ylabel('Mean Distance (µm)', fontweight='bold', fontsize=11)
                ax1.set_title(f'{pop_name}: Mean Distance to Tumor\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax1.legend(frameon=True, fontsize=10)
                ax1.grid(True, alpha=0.3)

            # Plot 2: Median distance
            ax2 = fig.add_subplot(gs[0, 1])
            if 'main_group' in df.columns and 'median_distance' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['median_distance'].mean()
                    ax2.plot(temporal_mean.index, temporal_mean.values,
                           marker='o', label=group, linewidth=2.5, markersize=8)
                ax2.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax2.set_ylabel('Median Distance (µm)', fontweight='bold', fontsize=11)
                ax2.set_title(f'{pop_name}: Median Distance\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax2.legend(frameon=True, fontsize=10)
                ax2.grid(True, alpha=0.3)

            # Plot 3: Boxplot - mean distance
            ax3 = fig.add_subplot(gs[0, 2])
            if 'main_group' in df.columns and 'mean_distance' in df.columns:
                sns.boxplot(data=df, x='main_group', y='mean_distance',
                           ax=ax3, palette='rocket', linewidth=2)
                add_pvalue_annotation(ax3, df, 'mean_distance', 'main_group')
                ax3.set_ylabel('Mean Distance (µm)', fontweight='bold', fontsize=11)
                ax3.set_xlabel('Group', fontweight='bold', fontsize=11)
                ax3.set_title(f'{pop_name}: Distance\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax3.grid(True, alpha=0.3, axis='y')

            # ===== ROW 2: 4-Way Comparisons =====

            # Plot 4: Mean distance - 4-way
            ax4 = fig.add_subplot(gs[1, 0])
            if 'genotype_full' in df.columns and 'mean_distance' in df.columns:
                for genotype in df['genotype_full'].unique():
                    gf_data = df[df['genotype_full'] == genotype]
                    temporal_mean = gf_data.groupby('timepoint')['mean_distance'].mean()
                    ax4.plot(temporal_mean.index, temporal_mean.values,
                           marker='o', label=genotype, linewidth=2, markersize=7)
                ax4.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax4.set_ylabel('Mean Distance (µm)', fontweight='bold', fontsize=11)
                ax4.set_title(f'{pop_name}: 4-Way Comparison', fontweight='bold', fontsize=12)
                ax4.legend(frameon=True, fontsize=9)
                ax4.grid(True, alpha=0.3)

            # Plot 5: % within 30µm - 4-way
            ax5 = fig.add_subplot(gs[1, 1])
            if 'genotype_full' in df.columns and 'pct_within_30um' in df.columns:
                for genotype in df['genotype_full'].unique():
                    gf_data = df[df['genotype_full'] == genotype]
                    temporal_mean = gf_data.groupby('timepoint')['pct_within_30um'].mean()
                    ax5.plot(temporal_mean.index, temporal_mean.values,
                           marker='s', label=genotype, linewidth=2, markersize=7)
                ax5.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax5.set_ylabel('% Within 30µm', fontweight='bold', fontsize=11)
                ax5.set_title(f'{pop_name}: Proximity (4-Way)', fontweight='bold', fontsize=12)
                ax5.legend(frameon=True, fontsize=9)
                ax5.grid(True, alpha=0.3)

            # Plot 6: Violin plot - 4-way distance
            ax6 = fig.add_subplot(gs[1, 2])
            if 'genotype_full' in df.columns and 'mean_distance' in df.columns:
                sns.violinplot(data=df, x='genotype_full', y='mean_distance',
                              ax=ax6, palette='rocket')
                ax6.set_ylabel('Mean Distance (µm)', fontweight='bold', fontsize=11)
                ax6.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                ax6.set_title(f'{pop_name}: Distribution (4-Way)', fontweight='bold', fontsize=12)
                ax6.tick_params(axis='x', rotation=45)
                ax6.grid(True, alpha=0.3, axis='y')

            # ===== ROW 3: Detailed Proximity Metrics =====

            # Plot 7: % within 30µm - KPT vs KPNT
            ax7 = fig.add_subplot(gs[2, 0])
            if 'main_group' in df.columns and 'pct_within_30um' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['pct_within_30um'].mean()
                    ax7.plot(temporal_mean.index, temporal_mean.values,
                           marker='o', label=group, linewidth=2.5, markersize=8)
                ax7.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax7.set_ylabel('% Within 30µm', fontweight='bold', fontsize=11)
                ax7.set_title(f'{pop_name}: Proximity\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax7.legend(frameon=True, fontsize=10)
                ax7.grid(True, alpha=0.3)

            # Plot 8: Scatter - mean vs median distance
            ax8 = fig.add_subplot(gs[2, 1])
            if 'main_group' in df.columns and 'mean_distance' in df.columns and 'median_distance' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    ax8.scatter(group_data['median_distance'], group_data['mean_distance'],
                              alpha=0.6, s=60, label=group)
                ax8.set_xlabel('Median Distance (µm)', fontweight='bold', fontsize=11)
                ax8.set_ylabel('Mean Distance (µm)', fontweight='bold', fontsize=11)
                ax8.set_title(f'{pop_name}: Distance Distribution\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax8.legend(frameon=True, fontsize=10)
                ax8.grid(True, alpha=0.3)

            # Plot 9: Summary heatmap
            ax9 = fig.add_subplot(gs[2, 2])
            if 'genotype_full' in df.columns:
                metrics = []
                if 'mean_distance' in df.columns:
                    metrics.append('mean_distance')
                if 'median_distance' in df.columns:
                    metrics.append('median_distance')
                if 'pct_within_30um' in df.columns:
                    metrics.append('pct_within_30um')

                if metrics:
                    summary = df.groupby('genotype_full')[metrics].mean()
                    summary_norm = (summary - summary.min()) / (summary.max() - summary.min())
                    sns.heatmap(summary_norm.T, annot=summary.T, fmt='.1f',
                               cmap='rocket_r', ax=ax9, cbar_kws={'label': 'Normalized Value'})
                    ax9.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                    ax9.set_ylabel('Metric', fontweight='bold', fontsize=11)
                    ax9.set_title(f'{pop_name} Summary', fontweight='bold', fontsize=12)

            plt.suptitle(f'Comprehensive Multi-Level Distance Analysis: {pop_name}',
                        fontsize=16, fontweight='bold', y=0.995)
            plt.savefig(f"{dist_dir}/figures/{pop_name}_distances_comprehensive.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Comprehensive {pop_name} distance plots saved (3×3 grid)")
        except Exception as e:
            print(f"  WARNING: {pop_name} distance plots failed: {e}")


def plot_infiltration_associations_comprehensive(output_dir: str):
    """Generate comprehensive 3×3 infiltration-tumor association plots."""
    print("\nGenerating comprehensive infiltration association plots...")

    infil_dir = f"{output_dir}/advanced_infiltration"
    if not os.path.exists(infil_dir):
        print("  No infiltration association data found, skipping...")
        return

    os.makedirs(f"{infil_dir}/figures", exist_ok=True)

    # Load association data
    assoc_file = f"{infil_dir}/tumor_infiltration_associations.csv"
    if os.path.exists(assoc_file):
        try:
            df = pd.read_csv(assoc_file)

            # Get top populations
            pop_counts = df.groupby('population').size()
            top_pops = pop_counts.nlargest(4).index.tolist()

            for pop in top_pops:
                pop_data = df[df['population'] == pop]

                if len(pop_data) == 0:
                    continue

                # Create 3×3 comprehensive figure for this population
                fig = plt.figure(figsize=(20, 16))
                gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

                # ===== ROW 1: Tumor Size vs Infiltration =====

                # Plot 1: Tumor size vs mean infiltration - KPT vs KPNT
                ax1 = fig.add_subplot(gs[0, 0])
                if 'main_group' in pop_data.columns and 'tumor_size' in pop_data.columns and 'mean_infiltration' in pop_data.columns:
                    for group in pop_data['main_group'].unique():
                        group_data = pop_data[pop_data['main_group'] == group]
                        ax1.scatter(group_data['tumor_size'], group_data['mean_infiltration'],
                                  alpha=0.6, s=60, label=group)
                    ax1.set_xlabel('Tumor Size (cells)', fontweight='bold', fontsize=11)
                    ax1.set_ylabel('Mean Infiltration (%)', fontweight='bold', fontsize=11)
                    ax1.set_title(f'{pop}: Infiltration vs Tumor Size\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                    ax1.legend(frameon=True, fontsize=10)
                    ax1.grid(True, alpha=0.3)

                # Plot 2: Infiltration over time - KPT vs KPNT
                ax2 = fig.add_subplot(gs[0, 1])
                if 'main_group' in pop_data.columns and 'timepoint' in pop_data.columns and 'mean_infiltration' in pop_data.columns:
                    for group in pop_data['main_group'].unique():
                        group_data = pop_data[pop_data['main_group'] == group]
                        temporal_mean = group_data.groupby('timepoint')['mean_infiltration'].mean()
                        temporal_sem = group_data.groupby('timepoint')['mean_infiltration'].sem()
                        ax2.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                                   marker='o', label=group, linewidth=2.5, markersize=8, capsize=5)
                    ax2.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                    ax2.set_ylabel('Mean Infiltration (%)', fontweight='bold', fontsize=11)
                    ax2.set_title(f'{pop}: Infiltration Over Time\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                    ax2.legend(frameon=True, fontsize=10)
                    ax2.grid(True, alpha=0.3)

                # Plot 3: Boxplot - infiltration by group
                ax3 = fig.add_subplot(gs[0, 2])
                if 'main_group' in pop_data.columns and 'mean_infiltration' in pop_data.columns:
                    sns.boxplot(data=pop_data, x='main_group', y='mean_infiltration',
                               ax=ax3, palette='mako', linewidth=2)
                    add_pvalue_annotation(ax3, pop_data, 'mean_infiltration', 'main_group')
                    ax3.set_ylabel('Mean Infiltration (%)', fontweight='bold', fontsize=11)
                    ax3.set_xlabel('Group', fontweight='bold', fontsize=11)
                    ax3.set_title(f'{pop}: Infiltration\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                    ax3.grid(True, alpha=0.3, axis='y')

                # ===== ROW 2: 4-Way Comparisons =====

                # Plot 4: Tumor size vs infiltration - 4-way
                ax4 = fig.add_subplot(gs[1, 0])
                if 'genotype_full' in pop_data.columns and 'tumor_size' in pop_data.columns and 'mean_infiltration' in pop_data.columns:
                    for genotype in pop_data['genotype_full'].unique():
                        gf_data = pop_data[pop_data['genotype_full'] == genotype]
                        ax4.scatter(gf_data['tumor_size'], gf_data['mean_infiltration'],
                                  alpha=0.5, s=40, label=genotype)
                    ax4.set_xlabel('Tumor Size (cells)', fontweight='bold', fontsize=11)
                    ax4.set_ylabel('Mean Infiltration (%)', fontweight='bold', fontsize=11)
                    ax4.set_title(f'{pop}: 4-Way Comparison', fontweight='bold', fontsize=12)
                    ax4.legend(frameon=True, fontsize=9)
                    ax4.grid(True, alpha=0.3)

                # Plot 5: Infiltration over time - 4-way
                ax5 = fig.add_subplot(gs[1, 1])
                if 'genotype_full' in pop_data.columns and 'mean_infiltration' in pop_data.columns:
                    for genotype in pop_data['genotype_full'].unique():
                        gf_data = pop_data[pop_data['genotype_full'] == genotype]
                        temporal_mean = gf_data.groupby('timepoint')['mean_infiltration'].mean()
                        ax5.plot(temporal_mean.index, temporal_mean.values,
                               marker='o', label=genotype, linewidth=2, markersize=7)
                    ax5.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                    ax5.set_ylabel('Mean Infiltration (%)', fontweight='bold', fontsize=11)
                    ax5.set_title(f'{pop}: Temporal (4-Way)', fontweight='bold', fontsize=12)
                    ax5.legend(frameon=True, fontsize=9)
                    ax5.grid(True, alpha=0.3)

                # Plot 6: Violin plot - 4-way infiltration
                ax6 = fig.add_subplot(gs[1, 2])
                if 'genotype_full' in pop_data.columns and 'mean_infiltration' in pop_data.columns:
                    sns.violinplot(data=pop_data, x='genotype_full', y='mean_infiltration',
                                  ax=ax6, palette='mako')
                    ax6.set_ylabel('Mean Infiltration (%)', fontweight='bold', fontsize=11)
                    ax6.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                    ax6.set_title(f'{pop}: Distribution (4-Way)', fontweight='bold', fontsize=12)
                    ax6.tick_params(axis='x', rotation=45)
                    ax6.grid(True, alpha=0.3, axis='y')

                # ===== ROW 3: Regional and Advanced Analyses =====

                # Plot 7: Infiltration by region - KPT vs KPNT
                ax7 = fig.add_subplot(gs[2, 0])
                if 'main_group' in pop_data.columns and 'region' in pop_data.columns and 'mean_infiltration' in pop_data.columns:
                    regions = pop_data['region'].unique()
                    x_pos = np.arange(len(regions))
                    width = 0.35
                    groups = pop_data['main_group'].unique()

                    for idx, group in enumerate(groups):
                        group_data = pop_data[pop_data['main_group'] == group]
                        regional_means = [group_data[group_data['region'] == r]['mean_infiltration'].mean()
                                        for r in regions]
                        ax7.bar(x_pos + idx*width, regional_means, width, label=group, alpha=0.8)

                    ax7.set_xlabel('Region', fontweight='bold', fontsize=11)
                    ax7.set_ylabel('Mean Infiltration (%)', fontweight='bold', fontsize=11)
                    ax7.set_title(f'{pop}: Regional Infiltration\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                    ax7.set_xticks(x_pos + width/2)
                    ax7.set_xticklabels(regions, rotation=45)
                    ax7.legend(frameon=True, fontsize=10)
                    ax7.grid(True, alpha=0.3, axis='y')

                # Plot 8: Max infiltration over time
                ax8 = fig.add_subplot(gs[2, 1])
                if 'main_group' in pop_data.columns and 'max_infiltration' in pop_data.columns:
                    for group in pop_data['main_group'].unique():
                        group_data = pop_data[pop_data['main_group'] == group]
                        temporal_mean = group_data.groupby('timepoint')['max_infiltration'].mean()
                        ax8.plot(temporal_mean.index, temporal_mean.values,
                               marker='o', label=group, linewidth=2.5, markersize=8)
                    ax8.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                    ax8.set_ylabel('Max Infiltration (%)', fontweight='bold', fontsize=11)
                    ax8.set_title(f'{pop}: Peak Infiltration\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                    ax8.legend(frameon=True, fontsize=10)
                    ax8.grid(True, alpha=0.3)

                # Plot 9: Summary heatmap
                ax9 = fig.add_subplot(gs[2, 2])
                if 'genotype_full' in pop_data.columns:
                    metrics = []
                    if 'mean_infiltration' in pop_data.columns:
                        metrics.append('mean_infiltration')
                    if 'max_infiltration' in pop_data.columns:
                        metrics.append('max_infiltration')

                    if metrics:
                        summary = pop_data.groupby('genotype_full')[metrics].mean()
                        summary_norm = (summary - summary.min()) / (summary.max() - summary.min())
                        sns.heatmap(summary_norm.T, annot=summary.T, fmt='.1f',
                                   cmap='mako', ax=ax9, cbar_kws={'label': 'Normalized Value'})
                        ax9.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                        ax9.set_ylabel('Metric', fontweight='bold', fontsize=11)
                        ax9.set_title(f'{pop} Summary', fontweight='bold', fontsize=12)

                plt.suptitle(f'Comprehensive Infiltration-Tumor Associations: {pop}',
                            fontsize=16, fontweight='bold', y=0.995)
                plt.savefig(f"{infil_dir}/figures/{pop}_associations_comprehensive.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Comprehensive {pop} infiltration plots saved (3×3 grid)")
        except Exception as e:
            print(f"  WARNING: Infiltration association plots failed: {e}")


def plot_pseudotime_comprehensive(output_dir: str):
    """Generate comprehensive 3×3 pseudotime trajectory plots."""
    print("\nGenerating comprehensive pseudotime trajectory plots...")

    pseudo_dir = f"{output_dir}/advanced_pseudotime"
    if not os.path.exists(pseudo_dir):
        print("  No pseudotime data found, skipping...")
        return

    os.makedirs(f"{pseudo_dir}/figures", exist_ok=True)

    # Load pseudotime metrics
    pseudo_file = f"{pseudo_dir}/pseudotime_metrics.csv"
    if os.path.exists(pseudo_file):
        try:
            df = pd.read_csv(pseudo_file)

            # Create 3×3 comprehensive figure
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # ===== ROW 1: Pseudotime Metrics - KPT vs KPNT =====

            # Plot 1: Mean pseudotime over real time
            ax1 = fig.add_subplot(gs[0, 0])
            if 'main_group' in df.columns and 'mean_pseudotime' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['mean_pseudotime'].mean()
                    temporal_sem = group_data.groupby('timepoint')['mean_pseudotime'].sem()
                    ax1.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                               marker='o', label=group, linewidth=2.5, markersize=8, capsize=5)
                ax1.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax1.set_ylabel('Mean Pseudotime', fontweight='bold', fontsize=11)
                ax1.set_title('Pseudotemporal Progression\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax1.legend(frameon=True, fontsize=10)
                ax1.grid(True, alpha=0.3)

            # Plot 2: Pseudotime variance
            ax2 = fig.add_subplot(gs[0, 1])
            if 'main_group' in df.columns and 'pseudotime_variance' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    temporal_mean = group_data.groupby('timepoint')['pseudotime_variance'].mean()
                    ax2.plot(temporal_mean.index, temporal_mean.values,
                           marker='o', label=group, linewidth=2.5, markersize=8)
                ax2.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax2.set_ylabel('Pseudotime Variance', fontweight='bold', fontsize=11)
                ax2.set_title('Trajectory Heterogeneity\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax2.legend(frameon=True, fontsize=10)
                ax2.grid(True, alpha=0.3)

            # Plot 3: Boxplot - mean pseudotime
            ax3 = fig.add_subplot(gs[0, 2])
            if 'main_group' in df.columns and 'mean_pseudotime' in df.columns:
                sns.boxplot(data=df, x='main_group', y='mean_pseudotime',
                           ax=ax3, palette='twilight', linewidth=2)
                add_pvalue_annotation(ax3, df, 'mean_pseudotime', 'main_group')
                ax3.set_ylabel('Mean Pseudotime', fontweight='bold', fontsize=11)
                ax3.set_xlabel('Group', fontweight='bold', fontsize=11)
                ax3.set_title('Pseudotime\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax3.grid(True, alpha=0.3, axis='y')

            # ===== ROW 2: 4-Way Comparisons =====

            # Plot 4: Mean pseudotime - 4-way
            ax4 = fig.add_subplot(gs[1, 0])
            if 'genotype_full' in df.columns and 'mean_pseudotime' in df.columns:
                for genotype in df['genotype_full'].unique():
                    gf_data = df[df['genotype_full'] == genotype]
                    temporal_mean = gf_data.groupby('timepoint')['mean_pseudotime'].mean()
                    ax4.plot(temporal_mean.index, temporal_mean.values,
                           marker='o', label=genotype, linewidth=2, markersize=7)
                ax4.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax4.set_ylabel('Mean Pseudotime', fontweight='bold', fontsize=11)
                ax4.set_title('Pseudotime: 4-Way', fontweight='bold', fontsize=12)
                ax4.legend(frameon=True, fontsize=9)
                ax4.grid(True, alpha=0.3)

            # Plot 5: Variance - 4-way
            ax5 = fig.add_subplot(gs[1, 1])
            if 'genotype_full' in df.columns and 'pseudotime_variance' in df.columns:
                for genotype in df['genotype_full'].unique():
                    gf_data = df[df['genotype_full'] == genotype]
                    temporal_mean = gf_data.groupby('timepoint')['pseudotime_variance'].mean()
                    ax5.plot(temporal_mean.index, temporal_mean.values,
                           marker='s', label=genotype, linewidth=2, markersize=7)
                ax5.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax5.set_ylabel('Pseudotime Variance', fontweight='bold', fontsize=11)
                ax5.set_title('Variance: 4-Way', fontweight='bold', fontsize=12)
                ax5.legend(frameon=True, fontsize=9)
                ax5.grid(True, alpha=0.3)

            # Plot 6: Violin plot - 4-way pseudotime
            ax6 = fig.add_subplot(gs[1, 2])
            if 'genotype_full' in df.columns and 'mean_pseudotime' in df.columns:
                sns.violinplot(data=df, x='genotype_full', y='mean_pseudotime',
                              ax=ax6, palette='twilight')
                ax6.set_ylabel('Mean Pseudotime', fontweight='bold', fontsize=11)
                ax6.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                ax6.set_title('Distribution: 4-Way', fontweight='bold', fontsize=12)
                ax6.tick_params(axis='x', rotation=45)
                ax6.grid(True, alpha=0.3, axis='y')

            # ===== ROW 3: Advanced Trajectory Metrics =====

            # Plot 7: Pseudotime progression rate
            ax7 = fig.add_subplot(gs[2, 0])
            if 'main_group' in df.columns and 'mean_pseudotime' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group].sort_values('timepoint')
                    temporal_mean = group_data.groupby('timepoint')['mean_pseudotime'].mean()
                    progression_rate = temporal_mean.diff()
                    ax7.plot(progression_rate.index[1:], progression_rate.values[1:],
                           marker='o', label=group, linewidth=2.5, markersize=8)
                ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax7.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
                ax7.set_ylabel('Δ Pseudotime', fontweight='bold', fontsize=11)
                ax7.set_title('Trajectory Velocity\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax7.legend(frameon=True, fontsize=10)
                ax7.grid(True, alpha=0.3)

            # Plot 8: Scatter - pseudotime vs variance
            ax8 = fig.add_subplot(gs[2, 1])
            if 'main_group' in df.columns and 'mean_pseudotime' in df.columns and 'pseudotime_variance' in df.columns:
                for group in df['main_group'].unique():
                    group_data = df[df['main_group'] == group]
                    ax8.scatter(group_data['mean_pseudotime'], group_data['pseudotime_variance'],
                              alpha=0.6, s=60, label=group)
                ax8.set_xlabel('Mean Pseudotime', fontweight='bold', fontsize=11)
                ax8.set_ylabel('Pseudotime Variance', fontweight='bold', fontsize=11)
                ax8.set_title('Trajectory Organization\n(KPT vs KPNT)', fontweight='bold', fontsize=12)
                ax8.legend(frameon=True, fontsize=10)
                ax8.grid(True, alpha=0.3)

            # Plot 9: Summary heatmap
            ax9 = fig.add_subplot(gs[2, 2])
            if 'genotype_full' in df.columns:
                metrics = []
                if 'mean_pseudotime' in df.columns:
                    metrics.append('mean_pseudotime')
                if 'pseudotime_variance' in df.columns:
                    metrics.append('pseudotime_variance')

                if metrics:
                    summary = df.groupby('genotype_full')[metrics].mean()
                    summary_norm = (summary - summary.min()) / (summary.max() - summary.min())
                    sns.heatmap(summary_norm.T, annot=summary.T, fmt='.2f',
                               cmap='twilight', ax=ax9, cbar_kws={'label': 'Normalized Value'})
                    ax9.set_xlabel('Genotype', fontweight='bold', fontsize=11)
                    ax9.set_ylabel('Metric', fontweight='bold', fontsize=11)
                    ax9.set_title('Pseudotime Summary', fontweight='bold', fontsize=12)

            plt.suptitle('Comprehensive Pseudotemporal Trajectory Analysis', fontsize=16, fontweight='bold', y=0.995)
            plt.savefig(f"{pseudo_dir}/figures/pseudotime_comprehensive.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Comprehensive pseudotime plots saved (3×3 grid)")
        except Exception as e:
            print(f"  WARNING: Pseudotime plots failed: {e}")


def plot_all_advanced_visualizations_comprehensive(output_dir: str):
    """
    Generate ALL comprehensive advanced analysis visualizations (Phases 12-18).

    This is the main entry point for generating COMPREHENSIVE 3×3 grid plots
    for all advanced phases with CORRECT biological comparisons.
    """
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE ADVANCED VISUALIZATIONS (PHASES 12-18)")
    print("="*80)
    print("\nUsing CORRECT comparisons:")
    print("  - KPT vs KPNT (main_group)")
    print("  - KPT-cis, KPT-trans, KPNT-cis, KPNT-trans (genotype_full)")
    print("  - NEVER comparing cis vs trans alone")
    print("="*80)

    # Phase 12: pERK analysis (2 comprehensive figures)
    plot_perk_analysis_comprehensive(output_dir)

    # Phase 13: NINJA analysis (1 comprehensive figure)
    plot_ninja_analysis_comprehensive(output_dir)

    # Phase 14: Heterogeneity (2 comprehensive figures)
    plot_heterogeneity_analysis_comprehensive(output_dir)

    # Phase 15: Enhanced RCN dynamics
    plot_enhanced_rcn_comprehensive(output_dir)

    # Phase 16: Multi-level distance analysis (multiple figures, one per population)
    plot_multilevel_distances_comprehensive(output_dir)

    # Phase 17: Infiltration associations (multiple figures, one per population)
    plot_infiltration_associations_comprehensive(output_dir)

    # Phase 18: Pseudotemporal trajectories
    plot_pseudotime_comprehensive(output_dir)

    print("\n" + "="*80)
    print("✓ ALL COMPREHENSIVE ADVANCED VISUALIZATIONS COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    print("Use this module with your advanced analysis pipeline")
    print("Call: plot_all_advanced_visualizations_comprehensive(output_dir)")
