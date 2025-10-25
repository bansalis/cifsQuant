#!/usr/bin/env python3
"""
Comprehensive Visualizations and Statistics Module

Generates all required plots with statistical annotations:
1. T cell-tumor distance plots (boxplots, distributions, scatters)
2. Marker expression temporal plots with stats
3. Neighborhood composition temporal plots
4. Heterogeneity plots
5. Dual-level statistics (by sample and by tumor)

All plots include:
- Statistical test results
- P-values with FDR correction
- Effect sizes
- Sample sizes

Author: Comprehensive visualization module
Date: 2025-10-25
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import (mannwhitneyu, kruskal, ks_2samp, ttest_ind,
                         f_oneway, spearmanr, pearsonr, linregress,
                         shapiro, levene)
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf
from typing import Dict, List, Tuple, Optional
import warnings
import os
warnings.filterwarnings('ignore')


class ComprehensiveVisualizationsStats:
    """
    Comprehensive visualization and statistics generator.
    """

    def __init__(self, output_dir: str):
        """
        Initialize visualizations and stats.

        Parameters
        ----------
        output_dir : str
            Output directory
        """
        self.output_dir = output_dir

        # Setup plotting style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.3)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300


    def plot_tcell_tumor_distances_comprehensive(
        self,
        structure_df: pd.DataFrame,
        sample_df: pd.DataFrame
    ):
        """
        Create comprehensive T cell-tumor distance plots with statistics.

        Parameters
        ----------
        structure_df : pd.DataFrame
            Per-structure distance data (n = tumors)
        sample_df : pd.DataFrame
            Per-sample distance data (n = samples)
        """
        print("\n" + "="*80)
        print("GENERATING T CELL-TUMOR DISTANCE PLOTS")
        print("="*80)

        # 1. Per-structure analysis (n = tumors)
        print("\n1. Per-structure plots...")
        self._plot_distances_per_structure(structure_df)

        # 2. Per-sample analysis (n = samples)
        print("\n2. Per-sample plots...")
        self._plot_distances_per_sample(sample_df)

        # 3. Statistical tests at both levels
        print("\n3. Statistical testing...")
        self._test_distance_statistics(structure_df, sample_df)

        print("\n✓ T cell-tumor distance plots complete")


    def _plot_distances_per_structure(self, df: pd.DataFrame):
        """Plot distance analyses at structure level (n = tumors)."""

        # For each T cell population and tumor subtype
        for tcell_pop in df['tcell_population'].unique():
            for tumor_subtype in df['tumor_subtype'].unique():
                subset = df[
                    (df['tcell_population'] == tcell_pop) &
                    (df['tumor_subtype'] == tumor_subtype)
                ]

                if len(subset) < 5:
                    continue

                # Create comprehensive figure
                fig = plt.figure(figsize=(20, 12))
                gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

                # Row 1: Temporal trends
                ax1 = fig.add_subplot(gs[0, 0])
                self._plot_temporal_boxplot(
                    subset, 'mean_distance', 'main_group', ax1,
                    title=f'{tcell_pop} to {tumor_subtype}\nDistance by Time (KPT vs KPNT)',
                    ylabel='Mean Distance (μm)'
                )

                ax2 = fig.add_subplot(gs[0, 1])
                self._plot_temporal_line(
                    subset, 'mean_distance', 'main_group', ax2,
                    title='Temporal Trend', ylabel='Mean Distance (μm)'
                )

                ax3 = fig.add_subplot(gs[0, 2])
                self._plot_distribution_comparison(
                    subset, 'mean_distance', 'main_group', ax3,
                    title='Distribution Comparison'
                )

                # Row 2: All 4 subgroups
                ax4 = fig.add_subplot(gs[1, 0])
                self._plot_temporal_boxplot(
                    subset, 'mean_distance', 'genotype_full', ax4,
                    title='Distance by Time (All 4 Subgroups)',
                    ylabel='Mean Distance (μm)'
                )

                ax5 = fig.add_subplot(gs[1, 1])
                self._plot_temporal_line(
                    subset, 'mean_distance', 'genotype_full', ax5,
                    title='Temporal Trend (4 Subgroups)', ylabel='Mean Distance (μm)'
                )

                ax6 = fig.add_subplot(gs[1, 2])
                # Scatter: tumor size vs distance
                for group in subset['main_group'].dropna().unique():
                    group_data = subset[subset['main_group'] == group]
                    ax6.scatter(group_data['tumor_size'], group_data['mean_distance'],
                               alpha=0.5, s=30, label=group)
                ax6.set_xlabel('Tumor Size (cells)', fontweight='bold')
                ax6.set_ylabel('Mean Distance (μm)', fontweight='bold')
                ax6.set_title('Distance vs Tumor Size', fontweight='bold')
                ax6.legend()
                ax6.grid(True, alpha=0.3)

                # Row 3: Per-timepoint comparisons
                ax7 = fig.add_subplot(gs[2, :])
                # Violin plot by timepoint and group
                if 'timepoint' in subset.columns:
                    subset_sorted = subset.sort_values('timepoint')
                    sns.violinplot(data=subset_sorted, x='timepoint', y='mean_distance',
                                  hue='main_group', ax=ax7, palette='Set2', split=False)
                    ax7.set_xlabel('Timepoint', fontweight='bold')
                    ax7.set_ylabel('Mean Distance (μm)', fontweight='bold')
                    ax7.set_title('Distance Distribution per Timepoint', fontweight='bold')

                plt.suptitle(
                    f'{tcell_pop} to {tumor_subtype} Distance Analysis (n = {len(subset)} tumors)',
                    fontsize=16, fontweight='bold', y=0.995
                )

                plt.savefig(
                    f"{self.output_dir}/figures/distances/{tcell_pop}_to_{tumor_subtype}_per_structure.png",
                    dpi=300, bbox_inches='tight'
                )
                plt.close()


    def _plot_distances_per_sample(self, df: pd.DataFrame):
        """Plot distance analyses at sample level (n = samples)."""

        for tcell_pop in df['tcell_population'].unique():
            for tumor_subtype in df['tumor_subtype'].unique():
                subset = df[
                    (df['tcell_population'] == tcell_pop) &
                    (df['tumor_subtype'] == tumor_subtype)
                ]

                if len(subset) < 3:
                    continue

                fig, axes = plt.subplots(2, 3, figsize=(20, 12))

                # Plot 1: Boxplot over time (main groups)
                ax = axes[0, 0]
                if 'timepoint' in subset.columns:
                    sns.boxplot(data=subset, x='timepoint', y='mean_distance',
                               hue='main_group', ax=ax, palette='Set2')
                    ax.set_title('KPT vs KPNT', fontweight='bold')
                    ax.set_ylabel('Mean Distance (μm)', fontweight='bold')

                # Plot 2: Line plot over time
                ax = axes[0, 1]
                self._plot_temporal_line(subset, 'mean_distance', 'main_group', ax,
                                        ylabel='Mean Distance (μm)')

                # Plot 3: Boxplot by group
                ax = axes[0, 2]
                sns.boxplot(data=subset, x='main_group', y='mean_distance',
                           ax=ax, palette='Set2')
                ax.set_title('Overall Comparison', fontweight='bold')
                ax.set_ylabel('Mean Distance (μm)', fontweight='bold')
                self._add_pvalue_annotation(ax, subset, 'mean_distance', 'main_group')

                # Plot 4: All 4 subgroups boxplot
                ax = axes[1, 0]
                if 'timepoint' in subset.columns:
                    sns.boxplot(data=subset, x='timepoint', y='mean_distance',
                               hue='genotype_full', ax=ax, palette='Set3')
                    ax.set_title('All 4 Subgroups', fontweight='bold')
                    ax.set_ylabel('Mean Distance (μm)', fontweight='bold')

                # Plot 5: Violin plot
                ax = axes[1, 1]
                sns.violinplot(data=subset, x='main_group', y='mean_distance',
                              ax=ax, palette='Set2')
                ax.set_title('Distribution', fontweight='bold')
                ax.set_ylabel('Mean Distance (μm)', fontweight='bold')

                # Plot 6: Strip + box plot
                ax = axes[1, 2]
                sns.boxplot(data=subset, x='main_group', y='mean_distance',
                           ax=ax, palette='Set2', alpha=0.5)
                sns.stripplot(data=subset, x='main_group', y='mean_distance',
                             ax=ax, color='black', alpha=0.6, size=6)
                ax.set_title('Individual Samples', fontweight='bold')
                ax.set_ylabel('Mean Distance (μm)', fontweight='bold')

                plt.suptitle(
                    f'{tcell_pop} to {tumor_subtype} Distance Analysis (n = {len(subset)} samples)',
                    fontsize=16, fontweight='bold'
                )
                plt.tight_layout()

                plt.savefig(
                    f"{self.output_dir}/figures/distances/{tcell_pop}_to_{tumor_subtype}_per_sample.png",
                    dpi=300, bbox_inches='tight'
                )
                plt.close()


    def _test_distance_statistics(self, structure_df: pd.DataFrame, sample_df: pd.DataFrame):
        """
        Comprehensive statistical testing at both levels.

        Tests:
        1. Temporal trends (Spearman correlation)
        2. Group comparisons (Mann-Whitney, t-test)
        3. Effect of tumor size (linear regression)
        4. Interactions (2-way ANOVA)
        """
        all_results = []

        # Test at structure level
        print("  Testing at structure level (n = tumors)...")
        struct_results = self._test_distances_single_level(
            structure_df, level='structure'
        )
        all_results.append(struct_results)

        # Test at sample level
        print("  Testing at sample level (n = samples)...")
        sample_results = self._test_distances_single_level(
            sample_df, level='sample'
        )
        all_results.append(sample_results)

        # Combine and save
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(
            f"{self.output_dir}/statistics/distances/tcell_tumor_distance_statistics.csv",
            index=False
        )

        print(f"  Saved {len(combined_results)} statistical tests")


    def _test_distances_single_level(self, df: pd.DataFrame, level: str) -> pd.DataFrame:
        """Test distances at a single aggregation level."""

        results = []

        for tcell_pop in df['tcell_population'].unique():
            for tumor_subtype in df['tumor_subtype'].unique():
                subset = df[
                    (df['tcell_population'] == tcell_pop) &
                    (df['tumor_subtype'] == tumor_subtype)
                ]

                if len(subset) < 3:
                    continue

                # 1. Temporal trend per group
                for group in subset['main_group'].dropna().unique():
                    group_data = subset[subset['main_group'] == group]

                    if len(group_data) >= 3 and 'timepoint' in group_data.columns:
                        rho, p_spearman = spearmanr(
                            group_data['timepoint'], group_data['mean_distance']
                        )

                        slope, intercept, r_value, p_linear, std_err = linregress(
                            group_data['timepoint'], group_data['mean_distance']
                        )

                        results.append({
                            'level': level,
                            'test_type': 'temporal_trend',
                            'tcell_population': tcell_pop,
                            'tumor_subtype': tumor_subtype,
                            'group': group,
                            'n': len(group_data),
                            'spearman_rho': rho,
                            'p_spearman': p_spearman,
                            'slope': slope,
                            'r_squared': r_value**2,
                            'p_linear': p_linear
                        })

                # 2. Group comparisons (KPT vs KPNT)
                main_groups = subset['main_group'].dropna().unique()
                if len(main_groups) == 2:
                    g1, g2 = main_groups[0], main_groups[1]
                    data1 = subset[subset['main_group'] == g1]['mean_distance'].values
                    data2 = subset[subset['main_group'] == g2]['mean_distance'].values

                    if len(data1) >= 2 and len(data2) >= 2:
                        # Mann-Whitney U
                        stat_mw, p_mw = mannwhitneyu(data1, data2, alternative='two-sided')

                        # t-test
                        stat_t, p_t = ttest_ind(data1, data2)

                        # Effect size (Cohen's d)
                        cohens_d = (data1.mean() - data2.mean()) / np.sqrt(
                            ((len(data1)-1)*data1.std()**2 + (len(data2)-1)*data2.std()**2) /
                            (len(data1) + len(data2) - 2)
                        )

                        results.append({
                            'level': level,
                            'test_type': 'group_comparison',
                            'tcell_population': tcell_pop,
                            'tumor_subtype': tumor_subtype,
                            'group_1': g1,
                            'group_2': g2,
                            'n_1': len(data1),
                            'n_2': len(data2),
                            'mean_1': data1.mean(),
                            'mean_2': data2.mean(),
                            'diff': data2.mean() - data1.mean(),
                            'fold_change': data2.mean() / (data1.mean() + 1e-6),
                            'cohens_d': cohens_d,
                            'stat_mannwhitney': stat_mw,
                            'p_mannwhitney': p_mw,
                            'stat_ttest': stat_t,
                            'p_ttest': p_t
                        })

                # 3. Per-timepoint comparisons
                if 'timepoint' in subset.columns:
                    for tp in subset['timepoint'].dropna().unique():
                        tp_data = subset[subset['timepoint'] == tp]
                        tp_groups = tp_data['main_group'].dropna().unique()

                        if len(tp_groups) == 2:
                            g1, g2 = tp_groups[0], tp_groups[1]
                            data1 = tp_data[tp_data['main_group'] == g1]['mean_distance'].values
                            data2 = tp_data[tp_data['main_group'] == g2]['mean_distance'].values

                            if len(data1) >= 2 and len(data2) >= 2:
                                stat_mw, p_mw = mannwhitneyu(data1, data2, alternative='two-sided')

                                results.append({
                                    'level': level,
                                    'test_type': 'per_timepoint_comparison',
                                    'tcell_population': tcell_pop,
                                    'tumor_subtype': tumor_subtype,
                                    'timepoint': tp,
                                    'group_1': g1,
                                    'group_2': g2,
                                    'n_1': len(data1),
                                    'n_2': len(data2),
                                    'mean_1': data1.mean(),
                                    'mean_2': data2.mean(),
                                    'diff': data2.mean() - data1.mean(),
                                    'stat_mannwhitney': stat_mw,
                                    'p_mannwhitney': p_mw
                                })

        if len(results) == 0:
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        # FDR correction
        if 'p_mannwhitney' in results_df.columns:
            _, p_adj_mw, _, _ = multipletests(
                results_df['p_mannwhitney'].fillna(1), method='fdr_bh'
            )
            results_df['p_adj_mannwhitney'] = p_adj_mw
            results_df['significant_mannwhitney'] = p_adj_mw < 0.05

        if 'p_spearman' in results_df.columns:
            spearman_mask = results_df['p_spearman'].notna()
            if spearman_mask.sum() > 0:
                _, p_adj_sp, _, _ = multipletests(
                    results_df.loc[spearman_mask, 'p_spearman'], method='fdr_bh'
                )
                results_df.loc[spearman_mask, 'p_adj_spearman'] = p_adj_sp
                results_df.loc[spearman_mask, 'significant_spearman'] = p_adj_sp < 0.05

        return results_df


    def plot_tumor_heterogeneity(self, heterogeneity_df: pd.DataFrame, markers: List[str]):
        """
        Plot tumor heterogeneity analyses.

        Parameters
        ----------
        heterogeneity_df : pd.DataFrame
            Heterogeneity metrics
        markers : list
            Markers analyzed
        """
        print("\n" + "="*80)
        print("GENERATING TUMOR HETEROGENEITY PLOTS")
        print("="*80)

        for marker in markers:
            het_col = f'{marker}_heterogeneity'
            clust_col = f'{marker}_clustering_index'

            if het_col not in heterogeneity_df.columns:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(20, 12))

            # Row 1: Heterogeneity
            ax = axes[0, 0]
            sns.boxplot(data=heterogeneity_df, x='main_group', y=het_col,
                       ax=ax, palette='Set2')
            ax.set_title(f'{marker} Spatial Heterogeneity\nKPT vs KPNT',
                        fontweight='bold')
            ax.set_ylabel('Heterogeneity (SD of local %)', fontweight='bold')
            self._add_pvalue_annotation(ax, heterogeneity_df, het_col, 'main_group')

            ax = axes[0, 1]
            if 'timepoint' in heterogeneity_df.columns:
                sns.boxplot(data=heterogeneity_df, x='timepoint', y=het_col,
                           hue='main_group', ax=ax, palette='Set2')
                ax.set_title('Heterogeneity Over Time', fontweight='bold')
                ax.set_ylabel('Heterogeneity', fontweight='bold')

            ax = axes[0, 2]
            for group in heterogeneity_df['main_group'].dropna().unique():
                group_data = heterogeneity_df[heterogeneity_df['main_group'] == group]
                ax.hist(group_data[het_col].dropna(), bins=20, alpha=0.5, label=group)
            ax.set_xlabel('Heterogeneity', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title('Distribution', fontweight='bold')
            ax.legend()

            # Row 2: Clustering index
            ax = axes[1, 0]
            sns.boxplot(data=heterogeneity_df, x='main_group', y=clust_col,
                       ax=ax, palette='Set2')
            ax.set_title(f'{marker} Spatial Clustering\nKPT vs KPNT',
                        fontweight='bold')
            ax.set_ylabel('Clustering Index (>1 = clustered)', fontweight='bold')
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
            self._add_pvalue_annotation(ax, heterogeneity_df, clust_col, 'main_group')

            ax = axes[1, 1]
            if 'timepoint' in heterogeneity_df.columns:
                sns.boxplot(data=heterogeneity_df, x='timepoint', y=clust_col,
                           hue='main_group', ax=ax, palette='Set2')
                ax.set_title('Clustering Over Time', fontweight='bold')
                ax.set_ylabel('Clustering Index', fontweight='bold')
                ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)

            ax = axes[1, 2]
            # Scatter: heterogeneity vs clustering
            for group in heterogeneity_df['main_group'].dropna().unique():
                group_data = heterogeneity_df[heterogeneity_df['main_group'] == group]
                ax.scatter(group_data[het_col], group_data[clust_col],
                          alpha=0.5, s=30, label=group)
            ax.set_xlabel('Heterogeneity', fontweight='bold')
            ax.set_ylabel('Clustering Index', fontweight='bold')
            ax.set_title('Heterogeneity vs Clustering', fontweight='bold')
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.3)
            ax.legend()

            plt.suptitle(
                f'{marker} Spatial Distribution Analysis (n = {len(heterogeneity_df)} tumors)',
                fontsize=16, fontweight='bold'
            )
            plt.tight_layout()

            plt.savefig(
                f"{self.output_dir}/figures/heterogeneity/{marker}_heterogeneity.png",
                dpi=300, bbox_inches='tight'
            )
            plt.close()

        print("✓ Heterogeneity plots complete")


    def plot_neighborhood_composition_temporal(self, composition_df: pd.DataFrame):
        """
        Plot neighborhood composition over time.

        Parameters
        ----------
        composition_df : pd.DataFrame
            Neighborhood composition by timepoint and group
        """
        print("\n" + "="*80)
        print("GENERATING NEIGHBORHOOD COMPOSITION PLOTS")
        print("="*80)

        # 1. Stacked area plot per group
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        for ax_idx, group in enumerate(composition_df['genotype'].dropna().unique()[:2]):
            ax = axes[ax_idx]

            group_data = composition_df[composition_df['genotype'] == group]

            # Pivot for stacked plot
            pivot_data = group_data.pivot_table(
                index='timepoint',
                columns='neighborhood_type',
                values='pct_cells',
                aggfunc='mean'
            ).fillna(0)

            pivot_data.plot(kind='area', stacked=True, ax=ax, alpha=0.7)
            ax.set_xlabel('Timepoint', fontweight='bold', fontsize=12)
            ax.set_ylabel('% Cells', fontweight='bold', fontsize=12)
            ax.set_title(f'Neighborhood Composition: {group}',
                        fontweight='bold', fontsize=14)
            ax.legend(title='Neighborhood Type', bbox_to_anchor=(1.05, 1),
                     loc='upper left')
            ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/figures/neighborhoods/temporal/neighborhood_composition_temporal.png",
            dpi=300, bbox_inches='tight'
        )
        plt.close()

        # 2. Heatmap per group
        for group in composition_df['genotype'].dropna().unique():
            group_data = composition_df[composition_df['genotype'] == group]

            pivot_data = group_data.pivot_table(
                index='neighborhood_type',
                columns='timepoint',
                values='pct_cells',
                aggfunc='mean'
            ).fillna(0)

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
                       ax=ax, cbar_kws={'label': '% Cells'})
            ax.set_title(f'Neighborhood Composition Over Time: {group}',
                        fontweight='bold', fontsize=14)
            ax.set_xlabel('Timepoint', fontweight='bold', fontsize=12)
            ax.set_ylabel('Neighborhood Type', fontweight='bold', fontsize=12)

            plt.tight_layout()
            plt.savefig(
                f"{self.output_dir}/figures/neighborhoods/temporal/neighborhood_heatmap_{group}.png",
                dpi=300, bbox_inches='tight'
            )
            plt.close()

        print("✓ Neighborhood composition plots complete")


    # Helper plotting functions
    def _plot_temporal_boxplot(self, df, value_col, group_col, ax, title, ylabel):
        """Create temporal boxplot."""
        if 'timepoint' in df.columns:
            sns.boxplot(data=df, x='timepoint', y=value_col, hue=group_col,
                       ax=ax, palette='Set2' if group_col == 'main_group' else 'Set3')
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Timepoint', fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')


    def _plot_temporal_line(self, df, value_col, group_col, ax, title='', ylabel=''):
        """Create temporal line plot."""
        if 'timepoint' in df.columns:
            for group in df[group_col].dropna().unique():
                group_data = df[df[group_col] == group]
                group_mean = group_data.groupby('timepoint')[value_col].agg(['mean', 'sem'])

                ax.errorbar(group_mean.index, group_mean['mean'],
                           yerr=group_mean['sem'], marker='o', linewidth=2,
                           markersize=8, capsize=5, label=group)

            ax.set_xlabel('Timepoint', fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            if title:
                ax.set_title(title, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)


    def _plot_distribution_comparison(self, df, value_col, group_col, ax, title=''):
        """Create distribution comparison plot."""
        for group in df[group_col].dropna().unique():
            group_data = df[df[group_col] == group][value_col].dropna()
            ax.hist(group_data, bins=30, alpha=0.5, label=group, density=True)

        ax.set_xlabel(value_col, fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        if title:
            ax.set_title(title, fontweight='bold')
        ax.legend()


    def _add_pvalue_annotation(self, ax, df, value_col, group_col):
        """Add p-value annotation to plot."""
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


if __name__ == '__main__':
    print("Use this module with your main analysis script")
