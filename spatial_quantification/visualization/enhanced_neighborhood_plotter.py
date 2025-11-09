"""
Enhanced Neighborhood Analysis Visualization
Comprehensive plots for regional infiltration and neighborhood composition
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings


class EnhancedNeighborhoodPlotter:
    """
    Comprehensive visualization for enhanced neighborhood analysis.

    Generates:
    - Regional infiltration comparison (marker+ vs marker-)
    - Distance to marker regions over time
    - Neighborhood composition within regions
    - Per-cell neighborhood comparisons
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

    def plot_regional_infiltration(self, regional_infiltration_df: pd.DataFrame, marker: str):
        """
        Plot regional infiltration (marker+ vs marker-).

        Parameters
        ----------
        regional_infiltration_df : pd.DataFrame
            Regional infiltration data
        marker : str
            Marker name (pERK or NINJA)
        """
        if regional_infiltration_df is None or len(regional_infiltration_df) == 0:
            return

        immune_pops = regional_infiltration_df['immune_population'].unique()
        n_pops = len(immune_pops)

        fig, axes = plt.subplots(n_pops, 2, figsize=(14, 5*n_pops))

        if n_pops == 1:
            axes = axes.reshape(1, -1)

        groups = sorted(regional_infiltration_df['main_group'].unique())

        for pop_idx, immune_pop in enumerate(immune_pops):
            pop_data = regional_infiltration_df[
                regional_infiltration_df['immune_population'] == immune_pop
            ]

            # Panel 1: Percent in region
            ax = axes[pop_idx, 0]

            # Aggregate by sample
            agg_data = pop_data.groupby(['sample_id', 'timepoint', 'main_group']).agg({
                f'{immune_pop}_in_pos_region_percent': 'mean',
                f'{immune_pop}_in_neg_region_percent': 'mean'
            }).reset_index()

            for group in groups:
                group_data = agg_data[agg_data['main_group'] == group]

                if len(group_data) == 0:
                    continue

                color = self.group_colors.get(group, '#000000')

                # Plot marker+ region
                col_pos = f'{immune_pop}_in_pos_region_percent'
                summary_pos = group_data.groupby('timepoint')[col_pos].agg(['mean', 'sem'])
                tp = summary_pos.index.values
                means_pos = summary_pos['mean'].values
                sems_pos = summary_pos['sem'].values

                ax.plot(tp, means_pos, '-o', color=color, linewidth=2.5,
                       markersize=8, label=f'{group} ({marker}+)', zorder=10)
                ax.fill_between(tp, means_pos - sems_pos, means_pos + sems_pos,
                               alpha=0.2, color=color)

                # Plot marker- region
                col_neg = f'{immune_pop}_in_neg_region_percent'
                summary_neg = group_data.groupby('timepoint')[col_neg].agg(['mean', 'sem'])
                means_neg = summary_neg['mean'].values
                sems_neg = summary_neg['sem'].values

                ax.plot(tp, means_neg, '--s', color=color, linewidth=2,
                       markersize=6, label=f'{group} ({marker}-)', zorder=9, alpha=0.7)
                ax.fill_between(tp, means_neg - sems_neg, means_neg + sems_neg,
                               alpha=0.1, color=color)

            ax.set_xlabel(self.timepoint_label, fontsize=11, fontweight='bold')
            ax.set_ylabel('% Immune in Region', fontsize=11, fontweight='bold')
            ax.set_title(f'{immune_pop}: % in {marker}+ vs {marker}- Regions',
                        fontsize=12, fontweight='bold')
            ax.legend(frameon=True, loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

            # Panel 2: Mean distance
            ax = axes[pop_idx, 1]

            agg_data = pop_data.groupby(['sample_id', 'timepoint', 'main_group']).agg({
                f'{immune_pop}_mean_dist_to_pos': 'mean',
                f'{immune_pop}_mean_dist_to_neg': 'mean'
            }).reset_index()

            for group in groups:
                group_data = agg_data[agg_data['main_group'] == group]

                if len(group_data) == 0:
                    continue

                color = self.group_colors.get(group, '#000000')

                # Plot distance to marker+ region
                col_pos = f'{immune_pop}_mean_dist_to_pos'
                plot_data_pos = group_data[~group_data[col_pos].isna()]
                if len(plot_data_pos) > 0:
                    summary_pos = plot_data_pos.groupby('timepoint')[col_pos].agg(['mean', 'sem'])
                    tp = summary_pos.index.values
                    means_pos = summary_pos['mean'].values
                    sems_pos = summary_pos['sem'].values

                    ax.plot(tp, means_pos, '-o', color=color, linewidth=2.5,
                           markersize=8, label=f'{group} (to {marker}+)', zorder=10)
                    ax.fill_between(tp, means_pos - sems_pos, means_pos + sems_pos,
                                   alpha=0.2, color=color)

                # Plot distance to marker- region
                col_neg = f'{immune_pop}_mean_dist_to_neg'
                plot_data_neg = group_data[~group_data[col_neg].isna()]
                if len(plot_data_neg) > 0:
                    summary_neg = plot_data_neg.groupby('timepoint')[col_neg].agg(['mean', 'sem'])
                    tp = summary_neg.index.values
                    means_neg = summary_neg['mean'].values
                    sems_neg = summary_neg['sem'].values

                    ax.plot(tp, means_neg, '--s', color=color, linewidth=2,
                           markersize=6, label=f'{group} (to {marker}-)', zorder=9, alpha=0.7)
                    ax.fill_between(tp, means_neg - sems_neg, means_neg + sems_neg,
                                   alpha=0.1, color=color)

            ax.set_xlabel(self.timepoint_label, fontsize=11, fontweight='bold')
            ax.set_ylabel('Mean Distance (μm)', fontsize=11, fontweight='bold')
            ax.set_title(f'{immune_pop}: Distance to {marker}+ vs {marker}- Regions',
                        fontsize=12, fontweight='bold')
            ax.legend(frameon=True, loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{marker} Regional Infiltration Analysis',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        plot_path = self.plots_dir / f'{marker}_regional_infiltration.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved {marker} regional infiltration plot: {plot_path.name}")

    def plot_regional_neighborhoods(self, regional_neighborhoods_df: pd.DataFrame, marker: str):
        """
        Plot neighborhood composition in marker+ vs marker- regions.

        Parameters
        ----------
        regional_neighborhoods_df : pd.DataFrame
            Regional neighborhood composition data
        marker : str
            Marker name
        """
        if regional_neighborhoods_df is None or len(regional_neighborhoods_df) == 0:
            return

        # Get cell type columns
        cell_type_cols = [col for col in regional_neighborhoods_df.columns
                         if col.startswith('pos_region_') and col.endswith('_fraction')]

        if not cell_type_cols:
            return

        # Extract cell type names
        cell_types = [col.replace('pos_region_', '').replace('_fraction', '')
                     for col in cell_type_cols]

        # Select key cell types for visualization
        key_types = ['CD8_T_cells', 'CD3_positive', 'CD45_positive', 'Tumor']
        available_key_types = [ct for ct in key_types if ct in cell_types]

        if not available_key_types:
            available_key_types = cell_types[:4]  # Take first 4

        n_types = len(available_key_types)
        groups = sorted(regional_neighborhoods_df['main_group'].unique())

        fig, axes = plt.subplots(2, n_types, figsize=(6*n_types, 10))

        if n_types == 1:
            axes = axes.reshape(-1, 1)

        for type_idx, cell_type in enumerate(available_key_types):
            col_pos = f'pos_region_{cell_type}_fraction'
            col_neg = f'neg_region_{cell_type}_fraction'

            # Panel 1: Marker+ region
            ax = axes[0, type_idx]

            for group in groups:
                group_data = regional_neighborhoods_df[
                    regional_neighborhoods_df['main_group'] == group
                ]

                if len(group_data) == 0 or col_pos not in group_data.columns:
                    continue

                summary = group_data.groupby('timepoint')[col_pos].agg(['mean', 'sem'])
                tp = summary.index.values
                means = summary['mean'].values
                sems = summary['sem'].values

                color = self.group_colors.get(group, '#000000')

                ax.plot(tp, means * 100, '-o', color=color, linewidth=2.5,
                       markersize=8, label=group, zorder=10)
                ax.fill_between(tp, (means - sems) * 100, (means + sems) * 100,
                               alpha=0.2, color=color)

            ax.set_xlabel(self.timepoint_label, fontsize=10, fontweight='bold')
            ax.set_ylabel('% of Neighborhood', fontsize=10, fontweight='bold')
            ax.set_title(f'{cell_type}\nin {marker}+ Regions',
                        fontsize=11, fontweight='bold')
            if type_idx == 0:
                ax.legend(frameon=True, loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

            # Panel 2: Marker- region
            ax = axes[1, type_idx]

            for group in groups:
                group_data = regional_neighborhoods_df[
                    regional_neighborhoods_df['main_group'] == group
                ]

                if len(group_data) == 0 or col_neg not in group_data.columns:
                    continue

                summary = group_data.groupby('timepoint')[col_neg].agg(['mean', 'sem'])
                tp = summary.index.values
                means = summary['mean'].values
                sems = summary['sem'].values

                color = self.group_colors.get(group, '#000000')

                ax.plot(tp, means * 100, '-o', color=color, linewidth=2.5,
                       markersize=8, label=group, zorder=10)
                ax.fill_between(tp, (means - sems) * 100, (means + sems) * 100,
                               alpha=0.2, color=color)

            ax.set_xlabel(self.timepoint_label, fontsize=10, fontweight='bold')
            ax.set_ylabel('% of Neighborhood', fontsize=10, fontweight='bold')
            ax.set_title(f'{cell_type}\nin {marker}- Regions',
                        fontsize=11, fontweight='bold')
            if type_idx == 0:
                ax.legend(frameon=True, loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{marker}: Neighborhood Composition Comparison',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        plot_path = self.plots_dir / f'{marker}_regional_neighborhoods.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved {marker} regional neighborhoods plot: {plot_path.name}")

    def plot_per_cell_neighborhoods(self, per_cell_df: pd.DataFrame, marker: str):
        """
        Plot per-cell neighborhood comparison (marker+ vs marker-).

        Parameters
        ----------
        per_cell_df : pd.DataFrame
            Per-cell neighborhood data
        marker : str
            Marker name
        """
        if per_cell_df is None or len(per_cell_df) == 0:
            return

        # Get cell type columns
        neighbor_cols = [col for col in per_cell_df.columns
                        if col.startswith('neighbor_') and col.endswith('_fraction')]

        if not neighbor_cols:
            return

        cell_types = [col.replace('neighbor_', '').replace('_fraction', '')
                     for col in neighbor_cols]

        # Select key cell types
        key_types = ['CD8_T_cells', 'CD3_positive', 'CD45_positive', 'Tumor']
        available_key_types = [ct for ct in key_types if ct in cell_types]

        if not available_key_types:
            available_key_types = cell_types[:4]

        n_types = len(available_key_types)
        groups = sorted(per_cell_df['main_group'].unique())

        fig, axes = plt.subplots(1, n_types, figsize=(6*n_types, 5))

        if n_types == 1:
            axes = [axes]

        for type_idx, cell_type in enumerate(available_key_types):
            ax = axes[type_idx]
            col = f'neighbor_{cell_type}_fraction'

            for group in groups:
                group_data = per_cell_df[per_cell_df['main_group'] == group]

                if len(group_data) == 0 or col not in group_data.columns:
                    continue

                color = self.group_colors.get(group, '#000000')

                # Plot marker+ cells
                pos_data = group_data[group_data['cell_marker_status'] == 'positive']
                if len(pos_data) > 0:
                    summary_pos = pos_data.groupby('timepoint')[col].agg(['mean', 'sem'])
                    tp = summary_pos.index.values
                    means = summary_pos['mean'].values
                    sems = summary_pos['sem'].values

                    ax.plot(tp, means * 100, '-o', color=color, linewidth=2.5,
                           markersize=8, label=f'{group} ({marker}+)', zorder=10)
                    ax.fill_between(tp, (means - sems) * 100, (means + sems) * 100,
                                   alpha=0.2, color=color)

                # Plot marker- cells
                neg_data = group_data[group_data['cell_marker_status'] == 'negative']
                if len(neg_data) > 0:
                    summary_neg = neg_data.groupby('timepoint')[col].agg(['mean', 'sem'])
                    tp = summary_neg.index.values
                    means = summary_neg['mean'].values
                    sems = summary_neg['sem'].values

                    ax.plot(tp, means * 100, '--s', color=color, linewidth=2,
                           markersize=6, label=f'{group} ({marker}-)', zorder=9, alpha=0.7)
                    ax.fill_between(tp, (means - sems) * 100, (means + sems) * 100,
                                   alpha=0.1, color=color)

            ax.set_xlabel(self.timepoint_label, fontsize=10, fontweight='bold')
            ax.set_ylabel('% of Neighborhood', fontsize=10, fontweight='bold')
            ax.set_title(f'{cell_type} in\nLocal Neighborhood',
                        fontsize=11, fontweight='bold')
            if type_idx == 0:
                ax.legend(frameon=True, loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{marker}: Per-Cell Neighborhood Comparison',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        plot_path = self.plots_dir / f'{marker}_per_cell_neighborhoods.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved {marker} per-cell neighborhoods plot: {plot_path.name}")

    def generate_all_plots(self, results: Dict):
        """
        Generate all enhanced neighborhood visualization plots.

        Parameters
        ----------
        results : dict
            Results dictionary from EnhancedNeighborhoodAnalysis
        """
        print("  Generating enhanced neighborhood analysis plots...")

        # Get markers from config
        enh_config = self.config.get('enhanced_neighborhoods', {})
        marker_configs = enh_config.get('markers', [])

        # Extract marker names from config
        if marker_configs:
            markers = [m['name'] for m in marker_configs]
        else:
            # Default markers (fallback)
            markers = ['pERK', 'NINJA']

        print(f"    Processing {len(markers)} markers: {', '.join(markers)}")

        # Process each marker
        for marker in markers:
            # Regional infiltration
            key = f'{marker}_regional_infiltration'
            if key in results:
                self.plot_regional_infiltration(results[key], marker)
            else:
                print(f"    ⚠ {key} not found in results")

            # Regional neighborhoods
            key = f'{marker}_regional_neighborhoods'
            if key in results:
                self.plot_regional_neighborhoods(results[key], marker)
            else:
                print(f"    ⚠ {key} not found in results")

            # Per-cell neighborhoods - use summary data for plotting
            key_summary = f'{marker}_per_cell_neighborhoods_summary'
            key_individual = f'{marker}_per_cell_neighborhoods_individual'

            if key_summary in results:
                self.plot_per_cell_neighborhoods(results[key_summary], marker)
            elif key_individual in results:
                # If only individual data exists, aggregate it for plotting
                print(f"    Using individual cell data for {marker} (aggregating for plot)")
                self.plot_per_cell_neighborhoods(results[key_individual], marker)
            else:
                print(f"    ⚠ No per-cell neighborhood data found for {marker}")

        print(f"  ✓ All enhanced neighborhood plots saved to {self.plots_dir}/")
