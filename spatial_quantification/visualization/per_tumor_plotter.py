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
from ..stats.plot_stats import add_significance_bars, perform_pairwise_tests, get_significance_symbol


class PerTumorPlotter:
    """
    Comprehensive plotting for per-tumor metrics.

    Generates:
    - Boxplots showing distribution of metrics across tumors
    - Line plots showing temporal trends
    - Scatter plots with individual tumor data points
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
        self.show_stats = plotting_config.get('show_stats', True)
        self.timepoint_label = plotting_config.get('timepoint_label', 'Time (weeks)')

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300

    def plot_per_tumor_metric(self, data: pd.DataFrame, metric_col: str,
                              ylabel: str, title: str,
                              group_col: str = 'main_group'):
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

        # Filter out invalid values
        plot_data = data[~data[metric_col].isna()].copy()

        if len(plot_data) == 0:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        groups = sorted(plot_data[group_col].unique())
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
                    box_colors.append(self.group_colors.get(group, '#000000'))
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
            ax.set_title('Distribution Across Tumors', fontsize=13, fontweight='bold')

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=self.group_colors.get(g, '#000000'),
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

            color = self.group_colors.get(group, '#000000')

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
                                    group_col: str = 'main_group'):
        """
        Create overview plot for all marker percentages.

        Parameters
        ----------
        data : pd.DataFrame
            Per-tumor marker percentage data
        group_col : str
            Grouping column
        """
        markers = ['pERK', 'NINJA', 'Ki67']
        marker_cols = [f'percent_{m}_positive' for m in markers]

        # Filter to available markers
        available_markers = [m for m, col in zip(markers, marker_cols) if col in data.columns]

        if not available_markers:
            return

        n_markers = len(available_markers)
        fig, axes = plt.subplots(1, n_markers, figsize=(6*n_markers, 5))

        if n_markers == 1:
            axes = [axes]

        groups = sorted(data[group_col].unique())
        timepoints = sorted(data['timepoint'].unique())

        for idx, marker in enumerate(available_markers):
            ax = axes[idx]
            col = f'percent_{marker}_positive'

            plot_data = data[~data[col].isna()]

            for group in groups:
                group_data = plot_data[plot_data[group_col] == group]

                if len(group_data) == 0:
                    continue

                # Calculate mean and SEM per timepoint
                summary = group_data.groupby('timepoint')[col].agg(['mean', 'sem'])
                tp_values = summary.index.values
                means = summary['mean'].values
                sems = summary['sem'].values

                color = self.group_colors.get(group, '#000000')

                # Plot individual points
                ax.scatter(group_data['timepoint'], group_data[col],
                          alpha=0.15, s=15, color=color, edgecolors='none')

                # Plot line
                ax.plot(tp_values, means, '-o', color=color, linewidth=2.5,
                       markersize=8, label=group, zorder=10)

                # Confidence band
                ax.fill_between(tp_values, means - sems, means + sems,
                               alpha=0.2, color=color)

            ax.set_xlabel(self.timepoint_label, fontsize=11, fontweight='bold')
            ax.set_ylabel(f'% {marker}+ per tumor', fontsize=11, fontweight='bold')
            ax.set_title(f'{marker}+ Tumor Cells', fontsize=12, fontweight='bold')
            if idx == 0:
                ax.legend(frameon=True, loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Marker Percentages Per Tumor', fontsize=16, fontweight='bold')
        plt.tight_layout()

        plot_path = self.plots_dir / 'all_marker_percentages_per_tumor.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_growth_normalized_pERK(self, data: pd.DataFrame,
                                   group_col: str = 'main_group'):
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

                color = self.group_colors.get(group, '#000000')

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
        Generate all per-tumor plots.

        Parameters
        ----------
        results : dict
            Results dictionary from PerTumorAnalysis
        """
        print("  Generating per-tumor plots...")

        # Plot basic metrics
        if 'per_tumor_metrics' in results:
            data = results['per_tumor_metrics']
            self.plot_per_tumor_metric(data, 'n_tumor_cells',
                                      'Number of Tumor Cells',
                                      'Tumor Size (cells per tumor)')

        # Plot marker percentages
        if 'per_tumor_marker_percentages' in results:
            data = results['per_tumor_marker_percentages']
            self.plot_all_marker_percentages(data)

        # Plot growth-normalized metrics
        if 'per_tumor_growth_normalized' in results:
            data = results['per_tumor_growth_normalized']
            self.plot_growth_normalized_pERK(data)

        # Plot infiltration metrics
        if 'per_tumor_infiltration' in results:
            data = results['per_tumor_infiltration']
            for immune_pop in ['CD8_T_cells', 'CD3_positive', 'CD45_positive']:
                col = f'{immune_pop}_density_per_tumor_cell'
                if col in data.columns:
                    self.plot_per_tumor_metric(data, col,
                                              f'{immune_pop} per Tumor Cell',
                                              f'{immune_pop} Infiltration Density')

        print(f"  ✓ Generated per-tumor plots in {self.plots_dir}/")
