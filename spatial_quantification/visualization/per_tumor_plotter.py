"""
Per-Tumor Visualization
Generate plots for per-tumor metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
from ..stats.plot_stats import add_significance_bars, perform_pairwise_tests, get_significance_symbol

try:
    from .plot_utils import detect_plot_type, create_dual_plots, calculate_statistics
    HAS_PLOT_UTILS = True
except ImportError:
    HAS_PLOT_UTILS = False
    warnings.warn("Plot utilities not available")


class PerTumorPlotter:
    """
    Comprehensive plotting for per-tumor/per-structure metrics.

    Generates:
    - Boxplots showing distribution of metrics across structures
    - Line plots showing temporal trends
    - Scatter plots with individual structure data points
    """

    # Default colors for dynamic assignment
    DEFAULT_COLORS = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#984EA3',
                      '#A65628', '#F781BF', '#999999', '#66C2A5', '#FC8D62']

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

        # Get grouping column from metadata config
        meta_config = config.get('metadata', {})
        self.group_col = meta_config.get('primary_grouping') or meta_config.get('group_column', 'group')

        # Get plotting settings
        plotting_config = config.get('plotting', {})
        self.group_colors = plotting_config.get('group_colors', {})
        self.show_stats = plotting_config.get('show_stats', True)
        self.timepoint_label = plotting_config.get('timepoint_label', 'Time (weeks)')

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300

    def _get_color(self, group: str, groups: List[str] = None) -> str:
        """Get color for a group, dynamically assigning if not predefined."""
        if group in self.group_colors:
            return self.group_colors[group]
        if groups is not None and group in groups:
            idx = groups.index(group) % len(self.DEFAULT_COLORS)
        else:
            idx = hash(group) % len(self.DEFAULT_COLORS)
        return self.DEFAULT_COLORS[idx]

    def plot_per_tumor_metric(self, data: pd.DataFrame, metric_col: str,
                              ylabel: str, title: str,
                              group_col: str = None):
        """
        Create boxplot + line plot for a per-tumor metric.

        Parameters
        ----------
        data : pd.DataFrame
            Per-tumor data
        metric_col : str
            Column to plot
        ylabel : str
            Y-axis label
        title : str
            Plot title
        group_col : str
            Grouping column
        """
        if metric_col not in data.columns:
            return

        # Use config group_col if not specified
        if group_col is None:
            group_col = self.group_col

        # Filter out invalid values
        plot_data = data[~data[metric_col].isna()].copy()

        if len(plot_data) == 0:
            return

        # Check if group_col exists in data
        if group_col not in plot_data.columns:
            # Try 'group' as fallback
            if 'group' in plot_data.columns:
                group_col = 'group'
            else:
                print(f"  ⚠ Group column '{group_col}' not found in data")
                return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        groups = sorted(plot_data[group_col].dropna().unique())
        timepoints = sorted(plot_data['timepoint'].unique())

        # Panel 1: Boxplots per timepoint
        ax = axes[0]

        # Prepare data for boxplot
        box_data = []
        box_positions = []
        box_colors = []
        labels = []

        width = 0.35
        for i, tp in enumerate(timepoints):
            for j, group in enumerate(groups):
                group_tp_data = plot_data[(plot_data['timepoint'] == tp) &
                                          (plot_data[group_col] == group)]
                if len(group_tp_data) > 0:
                    box_data.append(group_tp_data[metric_col].values)
                    box_positions.append(i + j * width)
                    box_colors.append(self._get_color(group, groups))
                    labels.append(f'{tp}\n{group}')

        if box_data:
            bp = ax.boxplot(box_data, positions=box_positions, widths=width*0.8,
                           patch_artist=True, showfliers=False)

            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_xticks([i + width/2 for i in range(len(timepoints))])
            ax.set_xticklabels([str(tp) for tp in timepoints])
            ax.set_xlabel(self.timepoint_label, fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title('Distribution Across Structures', fontsize=13, fontweight='bold')

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=self._get_color(g, groups),
                                    alpha=0.6, label=g) for g in groups]
            ax.legend(handles=legend_elements, loc='best', frameon=True)
            ax.grid(True, alpha=0.3, axis='y')

        # Panel 2: Line plot with mean ± SEM
        ax = axes[1]

        for group in groups:
            group_data = plot_data[plot_data[group_col] == group]

            if len(group_data) == 0:
                continue

            # Calculate mean and SEM per timepoint
            summary = group_data.groupby('timepoint')[metric_col].agg(['mean', 'sem', 'count'])
            tp_values = summary.index.values
            means = summary['mean'].values
            sems = summary['sem'].values

            color = self._get_color(group, groups)

            # Plot individual points (semi-transparent)
            ax.scatter(group_data['timepoint'], group_data[metric_col],
                      alpha=0.2, s=20, color=color, edgecolors='none')

            # Plot line
            ax.plot(tp_values, means, '-o', color=color, linewidth=2.5,
                   markersize=8, label=group, zorder=10)

            # Confidence band
            ax.fill_between(tp_values, means - sems, means + sems,
                           alpha=0.2, color=color)

        ax.set_xlabel(self.timepoint_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title('Temporal Trend (mean ± SEM)', fontsize=13, fontweight='bold')
        ax.legend(frameon=True, loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save
        safe_name = metric_col.replace(' ', '_').replace('/', '_per_')
        plot_path = self.plots_dir / f'{safe_name}_per_tumor.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_all_marker_percentages(self, data: pd.DataFrame,
                                    group_col: str = None):
        """
        Create overview plot for ALL marker percentages (dynamically from data columns).
        Uses box plots for single timepoint data.

        Parameters
        ----------
        data : pd.DataFrame
            Per-tumor marker percentage data
        group_col : str
            Grouping column
        """
        # Use config group_col if not specified
        if group_col is None:
            group_col = self.group_col

        # Check if group_col exists
        if group_col not in data.columns:
            if 'group' in data.columns:
                group_col = 'group'
            else:
                print(f"    ⚠ Group column '{group_col}' not found")
                return

        # Dynamically find ALL percent_*_positive columns
        percent_cols = [col for col in data.columns if col.startswith('percent_') and col.endswith('_positive')]

        if not percent_cols:
            print("    ⚠ No marker percentage columns found")
            return

        # Extract marker names
        markers = [col.replace('percent_', '').replace('_positive', '') for col in percent_cols]

        # Check if single timepoint (use box plots) or multiple (use line plots)
        timepoints = sorted(data['timepoint'].unique()) if 'timepoint' in data.columns else [0]
        single_timepoint = len(timepoints) == 1

        groups = sorted(data[group_col].dropna().unique())

        # Create multi-panel figure (max 6 per row)
        n_markers = len(markers)
        n_cols = min(6, n_markers)
        n_rows = int(np.ceil(n_markers / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_markers == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes

        for idx, (marker, col) in enumerate(zip(markers, percent_cols)):
            ax = axes[idx]
            plot_data = data[~data[col].isna()].copy()

            if len(plot_data) == 0:
                ax.axis('off')
                continue

            if single_timepoint:
                # BOX PLOT for single timepoint
                box_data = []
                positions = []
                colors_list = []
                labels = []

                for i, group in enumerate(groups):
                    group_data = plot_data[plot_data[group_col] == group]
                    if len(group_data) > 0:
                        box_data.append(group_data[col].values)
                        positions.append(i)
                        colors_list.append(self._get_color(group, groups))
                        labels.append(group)

                if box_data:
                    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                                   patch_artist=True, showfliers=True,
                                   flierprops=dict(marker='o', markersize=4, alpha=0.5))

                    for patch, color in zip(bp['boxes'], colors_list):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels, fontsize=10)
                    ax.set_ylabel(f'% {marker}+', fontsize=10, fontweight='bold')
            else:
                # LINE PLOT for multiple timepoints
                for group in groups:
                    group_data = plot_data[plot_data[group_col] == group]

                    if len(group_data) == 0:
                        continue

                    summary = group_data.groupby('timepoint')[col].agg(['mean', 'sem'])
                    tp_values = summary.index.values
                    means = summary['mean'].values
                    sems = summary['sem'].values

                    color = self._get_color(group, groups)

                    ax.scatter(group_data['timepoint'], group_data[col],
                              alpha=0.15, s=10, color=color, edgecolors='none')
                    ax.plot(tp_values, means, '-o', color=color, linewidth=2,
                           markersize=6, label=group, zorder=10)
                    ax.fill_between(tp_values, means - sems, means + sems,
                                   alpha=0.2, color=color)

                ax.set_xlabel(self.timepoint_label, fontsize=10)
                ax.set_ylabel(f'% {marker}+', fontsize=10, fontweight='bold')
                if idx == 0:
                    ax.legend(frameon=True, loc='best', fontsize=8)

            ax.set_title(f'{marker}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        # Hide unused subplots
        for idx in range(n_markers, len(axes)):
            axes[idx].axis('off')

        plot_type = "Box Plots" if single_timepoint else "Temporal Trends"
        plt.suptitle(f'All Marker Percentages Per Tumor ({plot_type})\nn={n_markers} markers',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        plot_path = self.plots_dir / 'all_marker_percentages_comprehensive.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Plotted {n_markers} marker percentages ({plot_type})")

    def plot_growth_normalized_pERK(self, data: pd.DataFrame,
                                   group_col: str = None):
        """
        Plot growth-normalized pERK metrics.

        Parameters
        ----------
        data : pd.DataFrame
            Growth-normalized data
        group_col : str
            Grouping column
        """
        # Metrics to plot
        metrics = [
            ('pERK_per_Ki67_ratio', 'pERK / Ki67 Ratio', 'pERK per Ki67 Ratio'),
            ('pERK_minus_Ki67', 'pERK - Ki67 (%)', 'pERK Minus Ki67'),
            ('pERK_residual_from_Ki67', 'pERK Residual', 'pERK Residual (growth-corrected)')
        ]

        available_metrics = [(col, lab, tit) for col, lab, tit in metrics if col in data.columns]

        if not available_metrics:
            return

        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        groups = sorted(data[group_col].unique())

        for idx, (metric_col, ylabel, title) in enumerate(available_metrics):
            ax = axes[idx]

            plot_data = data[~data[metric_col].isna()]

            for group in groups:
                group_data = plot_data[plot_data[group_col] == group]

                if len(group_data) == 0:
                    continue

                summary = group_data.groupby('timepoint')[metric_col].agg(['mean', 'sem'])
                tp_values = summary.index.values
                means = summary['mean'].values
                sems = summary['sem'].values

                color = self._get_color(group, groups)

                ax.scatter(group_data['timepoint'], group_data[metric_col],
                          alpha=0.15, s=15, color=color, edgecolors='none')

                ax.plot(tp_values, means, '-o', color=color, linewidth=2.5,
                       markersize=8, label=group, zorder=10)

                ax.fill_between(tp_values, means - sems, means + sems,
                               alpha=0.2, color=color)

            ax.set_xlabel(self.timepoint_label, fontsize=11, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            if idx == 0:
                ax.legend(frameon=True, loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            # Add horizontal line at 0 for residual plot
            if 'residual' in metric_col or 'minus' in metric_col:
                ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        plt.suptitle('Growth-Normalized pERK Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()

        plot_path = self.plots_dir / 'growth_normalized_pERK.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def generate_all_plots(self, results: Dict):
        """
        Generate all per-structure plots.

        Parameters
        ----------
        results : dict
            Results dictionary from PerTumorAnalysis/PerStructureAnalysis
        """
        print("  Generating per-structure plots...")

        # Plot basic metrics
        if 'per_tumor_metrics' in results:
            data = results['per_tumor_metrics']

            # Structure size (cells per structure)
            self.plot_per_tumor_metric(data, 'n_tumor_cells',
                                      'Number of Cells',
                                      'Structure Size (cells per structure)')

            # Area if available
            if 'area_um2' in data.columns:
                self.plot_per_tumor_metric(data, 'area_um2',
                                          'Area (μm²)',
                                          'Structure Area Over Time')

            # Density if available
            if 'density_cells_per_um2' in data.columns:
                self.plot_per_tumor_metric(data, 'density_cells_per_um2',
                                          'Density (cells/μm²)',
                                          'Structure Density Over Time')

            # Number of structures per sample over time
            self.plot_structure_counts_over_time(data)

        # Plot marker percentages
        if 'per_tumor_marker_percentages' in results:
            data = results['per_tumor_marker_percentages']
            self.plot_all_marker_percentages(data)

        # Plot growth-normalized metrics (if pERK/Ki67 present)
        if 'per_tumor_growth_normalized' in results:
            data = results['per_tumor_growth_normalized']
            self.plot_growth_normalized_pERK(data)

        # Plot infiltration metrics - dynamically find columns
        if 'per_tumor_infiltration' in results:
            data = results['per_tumor_infiltration']
            # Find all density columns dynamically
            density_cols = [col for col in data.columns if col.endswith('_density_per_tumor_cell')]
            for col in density_cols:
                immune_pop = col.replace('_density_per_tumor_cell', '')
                self.plot_per_tumor_metric(data, col,
                                          f'{immune_pop} per Structure Cell',
                                          f'{immune_pop} Infiltration Density')

        print(f"  ✓ Generated per-structure plots in {self.plots_dir}/")

    def plot_structure_counts_over_time(self, data: pd.DataFrame):
        """
        Plot number of structures per sample over time.

        Parameters
        ----------
        data : pd.DataFrame
            Per-structure data with sample_id and timepoint columns
        """
        if 'sample_id' not in data.columns or 'timepoint' not in data.columns:
            return

        # Count structures per sample
        counts = data.groupby(['sample_id', 'timepoint']).size().reset_index(name='n_structures')

        # Add group info
        group_col = self.group_col
        if group_col in data.columns:
            sample_to_group = data.drop_duplicates('sample_id').set_index('sample_id')[group_col].to_dict()
            counts['group'] = counts['sample_id'].map(sample_to_group)
        elif 'group' in data.columns:
            group_col = 'group'
            sample_to_group = data.drop_duplicates('sample_id').set_index('sample_id')[group_col].to_dict()
            counts['group'] = counts['sample_id'].map(sample_to_group)
        else:
            counts['group'] = 'All'

        groups = sorted(counts['group'].dropna().unique())
        timepoints = sorted(counts['timepoint'].unique())

        fig, ax = plt.subplots(figsize=(10, 6))

        for group in groups:
            group_data = counts[counts['group'] == group]

            if len(group_data) == 0:
                continue

            summary = group_data.groupby('timepoint')['n_structures'].agg(['mean', 'sem', 'sum'])
            tp_values = summary.index.values
            means = summary['mean'].values
            sems = summary['sem'].values

            color = self._get_color(group, groups)

            # Individual points (smaller)
            ax.scatter(group_data['timepoint'], group_data['n_structures'],
                      alpha=0.4, s=25, color=color, edgecolors='none')

            # Line
            ax.plot(tp_values, means, '-o', color=color, linewidth=2.5,
                   markersize=10, label=group, zorder=10)

            # Confidence band
            ax.fill_between(tp_values, means - sems, means + sems,
                           alpha=0.2, color=color)

        ax.set_xlabel(self.timepoint_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Structures per Sample', fontsize=12, fontweight='bold')
        ax.set_title('Structure Count Over Time', fontsize=14, fontweight='bold')
        ax.legend(frameon=True, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = self.plots_dir / 'structure_count_over_time.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Plotted structure count over time")
