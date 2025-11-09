"""
Infiltration Analysis Visualization
Comprehensive plots for infiltration, heterogeneity, and zone analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings


class InfiltrationPlotter:
    """
    Comprehensive visualization for infiltration analysis.

    Generates:
    - Infiltration trends over time (per immune population)
    - Heterogeneity metrics over time (Gi*, Ripley's K)
    - Zone-specific infiltration comparisons
    - Group comparisons with statistics
    """

    def __init__(self, output_dir: Path, config: Dict):
        """
        Initialize plotter.

        Parameters
        ----------
        output_dir : Path
            Output directory for plots
        config : dict
            Configuration dictionary
        """
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Get plotting settings
        plotting_config = config.get('plotting', {})
        self.group_colors = plotting_config.get('group_colors', {
            'KPT': '#E41A1C', 'KPNT': '#377EB8'
        })
        self.timepoint_label = plotting_config.get('timepoint_label', 'Time (weeks)')

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300

    def plot_infiltration_over_time(self, infiltration_df: pd.DataFrame):
        """
        Plot infiltration trends over time for all immune populations.

        Parameters
        ----------
        infiltration_df : pd.DataFrame
            Infiltration data with columns: sample_id, structure_id, immune_population,
            zone, count, timepoint, main_group
        """
        if infiltration_df is None or len(infiltration_df) == 0:
            return

        # Get unique immune populations
        immune_pops = infiltration_df['immune_population'].unique()
        zones = infiltration_df['zone'].unique()

        # Filter to key zones for summary (within tumor and close zones)
        key_zones = ['within_tumor', '0_50um', '50_100um']
        available_key_zones = [z for z in key_zones if z in zones]

        if not available_key_zones:
            available_key_zones = zones[:3]  # Take first 3 zones

        # Create figure: one row per immune population
        n_pops = len(immune_pops)
        n_zones = len(available_key_zones)

        fig, axes = plt.subplots(n_pops, n_zones, figsize=(6*n_zones, 5*n_pops))

        if n_pops == 1:
            axes = axes.reshape(1, -1)
        if n_zones == 1:
            axes = axes.reshape(-1, 1)

        groups = sorted(infiltration_df['main_group'].unique())

        for pop_idx, immune_pop in enumerate(immune_pops):
            pop_data = infiltration_df[infiltration_df['immune_population'] == immune_pop]

            for zone_idx, zone in enumerate(available_key_zones):
                ax = axes[pop_idx, zone_idx]

                zone_data = pop_data[pop_data['zone'] == zone]

                if len(zone_data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(f'{immune_pop} - {zone}')
                    continue

                # Aggregate by sample (mean across structures)
                agg_data = zone_data.groupby(['sample_id', 'timepoint', 'main_group']).agg({
                    'count': 'sum',  # Total infiltration
                    'structure_size': 'sum'  # Total tumor cells
                }).reset_index()

                # Calculate density (immune cells per tumor cell)
                agg_data['density'] = agg_data['count'] / agg_data['structure_size']

                for group in groups:
                    group_data = agg_data[agg_data['main_group'] == group]

                    if len(group_data) == 0:
                        continue

                    # Calculate mean and SEM per timepoint
                    summary = group_data.groupby('timepoint')['density'].agg(['mean', 'sem', 'count'])
                    timepoints = summary.index.values
                    means = summary['mean'].values
                    sems = summary['sem'].values

                    color = self.group_colors.get(group, '#000000')

                    # Plot individual points
                    ax.scatter(group_data['timepoint'], group_data['density'],
                              alpha=0.2, s=20, color=color, edgecolors='none')

                    # Plot line
                    ax.plot(timepoints, means, '-o', color=color, linewidth=2.5,
                           markersize=8, label=group, zorder=10)

                    # Confidence band
                    ax.fill_between(timepoints, means - sems, means + sems,
                                   alpha=0.2, color=color)

                ax.set_xlabel(self.timepoint_label, fontsize=10, fontweight='bold')
                ax.set_ylabel('Immune/Tumor Density', fontsize=10, fontweight='bold')
                ax.set_title(f'{immune_pop} - {zone}', fontsize=11, fontweight='bold')

                if pop_idx == 0 and zone_idx == 0:
                    ax.legend(frameon=True, loc='best', fontsize=9)

                ax.grid(True, alpha=0.3)

        plt.suptitle('Immune Infiltration Over Time', fontsize=16, fontweight='bold')
        plt.tight_layout()

        plot_path = self.plots_dir / 'infiltration_trends_over_time.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved infiltration trends plot: {plot_path.name}")

    def plot_heterogeneity_over_time(self, heterogeneity_df: pd.DataFrame):
        """
        Plot heterogeneity metrics over time.

        Parameters
        ----------
        heterogeneity_df : pd.DataFrame
            Heterogeneity data with Gi*, Ripley's K metrics
        """
        if heterogeneity_df is None or len(heterogeneity_df) == 0:
            return

        markers = heterogeneity_df['marker'].unique()

        # Metrics to plot
        metrics = [
            ('gi_star_mean', 'Mean Gi* (hotspot score)', 'Getis-Ord Gi*'),
            ('gi_star_hotspots', 'Number of Hotspots', 'Significant Hotspots (p<0.05)'),
            ('ripleys_l_50um', "Ripley's L (50μm)", "Ripley's L Function"),
            ('clustering_score', 'Clustering Score', 'DBSCAN Clustering')
        ]

        # Filter to available metrics
        available_metrics = [(col, lab, tit) for col, lab, tit in metrics
                            if col in heterogeneity_df.columns]

        if not available_metrics:
            return

        n_metrics = len(available_metrics)
        n_markers = len(markers)

        fig, axes = plt.subplots(n_markers, n_metrics, figsize=(6*n_metrics, 5*n_markers))

        if n_markers == 1:
            axes = axes.reshape(1, -1)
        if n_metrics == 1:
            axes = axes.reshape(-1, 1)

        groups = sorted(heterogeneity_df['main_group'].unique())

        for marker_idx, marker in enumerate(markers):
            marker_data = heterogeneity_df[heterogeneity_df['marker'] == marker]

            for metric_idx, (metric_col, ylabel, title) in enumerate(available_metrics):
                ax = axes[marker_idx, metric_idx]

                plot_data = marker_data[~marker_data[metric_col].isna()]

                if len(plot_data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(f'{marker} - {title}')
                    continue

                # Aggregate by sample (mean across structures)
                agg_data = plot_data.groupby(['sample_id', 'timepoint', 'main_group']).agg({
                    metric_col: 'mean'
                }).reset_index()

                for group in groups:
                    group_data = agg_data[agg_data['main_group'] == group]

                    if len(group_data) == 0:
                        continue

                    summary = group_data.groupby('timepoint')[metric_col].agg(['mean', 'sem'])
                    timepoints = summary.index.values
                    means = summary['mean'].values
                    sems = summary['sem'].values

                    color = self.group_colors.get(group, '#000000')

                    # Plot individual points
                    ax.scatter(group_data['timepoint'], group_data[metric_col],
                              alpha=0.2, s=20, color=color, edgecolors='none')

                    # Plot line
                    ax.plot(timepoints, means, '-o', color=color, linewidth=2.5,
                           markersize=8, label=group, zorder=10)

                    # Confidence band
                    ax.fill_between(timepoints, means - sems, means + sems,
                                   alpha=0.2, color=color)

                ax.set_xlabel(self.timepoint_label, fontsize=10, fontweight='bold')
                ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
                ax.set_title(f'{marker} - {title}', fontsize=11, fontweight='bold')

                if marker_idx == 0 and metric_idx == 0:
                    ax.legend(frameon=True, loc='best', fontsize=9)

                ax.grid(True, alpha=0.3)

        plt.suptitle('Spatial Heterogeneity Metrics Over Time', fontsize=16, fontweight='bold')
        plt.tight_layout()

        plot_path = self.plots_dir / 'heterogeneity_metrics_over_time.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved heterogeneity metrics plot: {plot_path.name}")

    def plot_zone_infiltration_comparison(self, zone_infiltration_df: pd.DataFrame):
        """
        Plot zone-specific infiltration (marker+ vs marker- regions).

        Parameters
        ----------
        zone_infiltration_df : pd.DataFrame
            Zone infiltration data
        """
        if zone_infiltration_df is None or len(zone_infiltration_df) == 0:
            return

        markers = zone_infiltration_df['marker'].unique()
        immune_pops = zone_infiltration_df['immune_population'].unique()

        # Create figure: one column per marker
        n_markers = len(markers)
        n_pops = len(immune_pops)

        fig, axes = plt.subplots(n_pops, n_markers, figsize=(7*n_markers, 5*n_pops))

        if n_pops == 1:
            axes = axes.reshape(1, -1)
        if n_markers == 1:
            axes = axes.reshape(-1, 1)

        groups = sorted(zone_infiltration_df['main_group'].unique())

        for pop_idx, immune_pop in enumerate(immune_pops):
            pop_data = zone_infiltration_df[zone_infiltration_df['immune_population'] == immune_pop]

            for marker_idx, marker in enumerate(markers):
                ax = axes[pop_idx, marker_idx]

                marker_data = pop_data[pop_data['marker'] == marker]

                if len(marker_data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(f'{marker} - {immune_pop}')
                    continue

                # Calculate enrichment ratio (pos region infiltration / neg region infiltration)
                marker_data = marker_data.copy()
                marker_data['pos_density'] = marker_data['zone_positive_infiltration'] / marker_data['n_positive_cells']
                marker_data['neg_density'] = marker_data['zone_negative_infiltration'] / marker_data['n_negative_cells']

                # Aggregate by sample
                agg_data = marker_data.groupby(['sample_id', 'timepoint', 'main_group']).agg({
                    'pos_density': 'mean',
                    'neg_density': 'mean'
                }).reset_index()

                # Plot both pos and neg with different line styles
                for group in groups:
                    group_data = agg_data[agg_data['main_group'] == group]

                    if len(group_data) == 0:
                        continue

                    color = self.group_colors.get(group, '#000000')

                    # Plot marker+ region infiltration
                    summary_pos = group_data.groupby('timepoint')['pos_density'].agg(['mean', 'sem'])
                    tp = summary_pos.index.values
                    means_pos = summary_pos['mean'].values
                    sems_pos = summary_pos['sem'].values

                    ax.plot(tp, means_pos, '-o', color=color, linewidth=2.5,
                           markersize=8, label=f'{group} ({marker}+)', zorder=10)
                    ax.fill_between(tp, means_pos - sems_pos, means_pos + sems_pos,
                                   alpha=0.2, color=color)

                    # Plot marker- region infiltration
                    summary_neg = group_data.groupby('timepoint')['neg_density'].agg(['mean', 'sem'])
                    means_neg = summary_neg['mean'].values
                    sems_neg = summary_neg['sem'].values

                    ax.plot(tp, means_neg, '--s', color=color, linewidth=2,
                           markersize=6, label=f'{group} ({marker}-)', zorder=9, alpha=0.7)
                    ax.fill_between(tp, means_neg - sems_neg, means_neg + sems_neg,
                                   alpha=0.1, color=color)

                ax.set_xlabel(self.timepoint_label, fontsize=10, fontweight='bold')
                ax.set_ylabel('Infiltration Density', fontsize=10, fontweight='bold')
                ax.set_title(f'{marker}: {immune_pop}\n(solid={marker}+, dashed={marker}-)',
                            fontsize=11, fontweight='bold')

                if pop_idx == 0 and marker_idx == 0:
                    ax.legend(frameon=True, loc='best', fontsize=8)

                ax.grid(True, alpha=0.3)

        plt.suptitle('Zone-Specific Infiltration (Marker+ vs Marker- Regions)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        plot_path = self.plots_dir / 'zone_specific_infiltration.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved zone-specific infiltration plot: {plot_path.name}")

    def plot_infiltration_summary_heatmap(self, infiltration_df: pd.DataFrame):
        """
        Create heatmap summarizing infiltration across all conditions.

        Parameters
        ----------
        infiltration_df : pd.DataFrame
            Infiltration data
        """
        if infiltration_df is None or len(infiltration_df) == 0:
            return

        # Focus on key zone
        key_zone = '0_50um'
        if key_zone not in infiltration_df['zone'].unique():
            key_zone = infiltration_df['zone'].unique()[0]

        zone_data = infiltration_df[infiltration_df['zone'] == key_zone]

        # Aggregate by group and timepoint
        agg_data = zone_data.groupby(['main_group', 'timepoint', 'immune_population']).agg({
            'count': 'sum',
            'structure_size': 'sum'
        }).reset_index()

        agg_data['density'] = agg_data['count'] / agg_data['structure_size']

        # Create pivot table for each group
        groups = sorted(agg_data['main_group'].unique())
        immune_pops = sorted(agg_data['immune_population'].unique())

        fig, axes = plt.subplots(1, len(groups), figsize=(8*len(groups), 6))
        if len(groups) == 1:
            axes = [axes]

        for idx, group in enumerate(groups):
            ax = axes[idx]
            group_data = agg_data[agg_data['main_group'] == group]

            # Pivot: rows = immune populations, columns = timepoints
            pivot = group_data.pivot_table(
                values='density',
                index='immune_population',
                columns='timepoint',
                aggfunc='mean'
            )

            if len(pivot) == 0:
                continue

            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                       cbar_kws={'label': 'Infiltration Density'},
                       ax=ax, vmin=0)

            ax.set_title(f'{group} - Infiltration Density\n(Zone: {key_zone})',
                        fontsize=13, fontweight='bold')
            ax.set_xlabel(self.timepoint_label, fontsize=11, fontweight='bold')
            ax.set_ylabel('Immune Population', fontsize=11, fontweight='bold')

        plt.tight_layout()

        plot_path = self.plots_dir / 'infiltration_summary_heatmap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved infiltration summary heatmap: {plot_path.name}")

    def generate_all_plots(self, results: Dict):
        """
        Generate all infiltration visualization plots.

        Parameters
        ----------
        results : dict
            Results dictionary from InfiltrationAnalysisOptimized
        """
        print("  Generating infiltration analysis plots...")

        # Plot 1: Infiltration trends over time
        if 'infiltration' in results:
            self.plot_infiltration_over_time(results['infiltration'])

        # Plot 2: Heterogeneity metrics - combine all markers to avoid overwriting
        heterogeneity_dfs = []
        for key in results.keys():
            if 'zone_heterogeneity' in key:
                heterogeneity_dfs.append(results[key])

        if heterogeneity_dfs:
            combined_heterogeneity = pd.concat(heterogeneity_dfs, ignore_index=True)
            self.plot_heterogeneity_over_time(combined_heterogeneity)

        # Plot 3: Zone-specific infiltration - combine all markers to avoid overwriting
        zone_infiltration_dfs = []
        for key in results.keys():
            if 'zone_infiltration' in key:
                zone_infiltration_dfs.append(results[key])

        if zone_infiltration_dfs:
            combined_zone_infiltration = pd.concat(zone_infiltration_dfs, ignore_index=True)
            self.plot_zone_infiltration_comparison(combined_zone_infiltration)

        # Plot 4: Summary heatmap
        if 'infiltration' in results:
            self.plot_infiltration_summary_heatmap(results['infiltration'])

        print(f"  ✓ All infiltration plots saved to {self.plots_dir}/")
