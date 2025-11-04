"""
Enhanced plotting for Population Dynamics
Multiple plot types with statistical tests
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Optional
import warnings
from ..stats.plot_stats import add_significance_bars, perform_pairwise_tests


class PopulationDynamicsPlotter:
    """
    Comprehensive plotting for population dynamics.

    Generates:
    - Scatter plots (raw data)
    - Line plots (trends with CI)
    - Box plots (distributions)
    - Violin plots (full distributions)
    - Overlaid combinations
    - Statistical annotations
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

        # Get plotting settings from config
        plotting_config = config.get('plotting', {})
        self.timepoint_label = plotting_config.get('timepoint_label', 'Time (days)')
        self.show_stats = plotting_config.get('show_stats', True)
        self.stat_method = plotting_config.get('stat_method', 'mann_whitney')
        self.fdr_correction = plotting_config.get('fdr_correction', True)
        self.sig_symbols = plotting_config.get('significance_symbols', {
            0.001: '***', 0.01: '**', 0.05: '*', 1.0: 'ns'
        })
        self.group_colors = plotting_config.get('group_colors', {
            'KPT': '#E41A1C', 'KPNT': '#377EB8', 'cis': '#4DAF4A', 'trans': '#FF7F00'
        })

        # Set font
        font_family = plotting_config.get('font_family', 'DejaVu Sans')
        font_size = plotting_config.get('font_size', 11)

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.family'] = font_family
        plt.rcParams['font.size'] = font_size

    def plot_population_over_time(self, data: pd.DataFrame,
                                  population: str,
                                  value_col: str = 'count',
                                  group_col: str = 'main_group',
                                  groups: List[str] = None):
        """
        Create comprehensive plots for a population over time.

        Generates 4 plot types:
        1. Scatter + line with confidence bands
        2. Box plots per timepoint
        3. Violin plots per timepoint
        4. Combined overlay
        """
        if groups is None:
            groups = sorted(data[group_col].unique())

        # Filter to groups
        plot_data = data[data[group_col].isin(groups)].copy()

        if len(plot_data) == 0:
            return

        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{population} - {value_col.replace("_", " ").title()}',
                    fontsize=16, fontweight='bold', y=0.995)

        # Plot 1: Scatter + Line with CI
        self._plot_scatter_line(plot_data, axes[0, 0], value_col, group_col, groups)

        # Plot 2: Box plots
        self._plot_boxes(plot_data, axes[0, 1], value_col, group_col, groups)

        # Plot 3: Violin plots
        self._plot_violins(plot_data, axes[1, 0], value_col, group_col, groups)

        # Plot 4: Combined overlay
        self._plot_combined(plot_data, axes[1, 1], value_col, group_col, groups)

        plt.tight_layout()

        # Save
        plot_path = self.plots_dir / f'{population}_{value_col}_comprehensive.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Also create publication version (just scatter+line, clean)
        self._plot_publication_version(plot_data, population, value_col, group_col, groups)

    def _plot_scatter_line(self, data: pd.DataFrame, ax, value_col: str,
                          group_col: str, groups: List[str]):
        """Scatter plot with line and confidence band."""
        for group in groups:
            group_data = data[data[group_col] == group]

            if len(group_data) == 0:
                continue

            # Calculate mean and SEM per timepoint
            summary = group_data.groupby('timepoint')[value_col].agg(['mean', 'sem', 'count'])
            timepoints = summary.index.values
            means = summary['mean'].values
            sems = summary['sem'].values

            color = self.group_colors.get(group, '#000000')

            # Plot individual points (semi-transparent)
            ax.scatter(group_data['timepoint'], group_data[value_col],
                      alpha=0.3, s=40, color=color, edgecolors='none')

            # Plot line
            ax.plot(timepoints, means, '-o', color=color, linewidth=2.5,
                   markersize=8, label=group, zorder=10)

            # Confidence band (±SEM)
            ax.fill_between(timepoints, means - sems, means + sems,
                           alpha=0.2, color=color)

        ax.set_xlabel(self.timepoint_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title('Scatter + Line with SEM', fontsize=13, fontweight='bold')
        ax.legend(frameon=True, loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

    def _plot_boxes(self, data: pd.DataFrame, ax, value_col: str,
                   group_col: str, groups: List[str]):
        """Box plots per timepoint with statistical tests."""
        timepoints = sorted(data['timepoint'].unique())

        # Prepare data for grouped box plot
        plot_data = []
        for tp in timepoints:
            tp_data = data[data['timepoint'] == tp]
            for group in groups:
                group_data = tp_data[tp_data[group_col] == group]
                for val in group_data[value_col]:
                    plot_data.append({'timepoint': tp, 'group': group, 'value': val})

        plot_df = pd.DataFrame(plot_data)

        if len(plot_df) > 0:
            # Create box plot
            box_positions = []
            box_data = []
            box_colors = []
            labels = []

            width = 0.35
            for i, tp in enumerate(timepoints):
                for j, group in enumerate(groups):
                    group_tp_data = plot_df[(plot_df['timepoint'] == tp) &
                                            (plot_df['group'] == group)]
                    if len(group_tp_data) > 0:
                        box_data.append(group_tp_data['value'].values)
                        box_positions.append(i + j * width)
                        box_colors.append(self.group_colors.get(group, '#000000'))
                        labels.append(f'{tp}\n{group}')

            bp = ax.boxplot(box_data, positions=box_positions, widths=width*0.8,
                           patch_artist=True, showfliers=False)

            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_xticks([i + width/2 for i in range(len(timepoints))])
            ax.set_xticklabels([str(tp) for tp in timepoints])
            ax.set_xlabel(self.timepoint_label, fontsize=12, fontweight='bold')
            ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_title('Box Plots', fontsize=13, fontweight='bold')

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=self.group_colors.get(g, '#000000'),
                                    alpha=0.6, label=g) for g in groups]
            ax.legend(handles=legend_elements, loc='best', frameon=True)
            ax.grid(True, alpha=0.3, axis='y')

            # Add statistical tests if enabled
            if self.show_stats and len(groups) == 2:
                # Add significance bars for each timepoint
                for i, tp in enumerate(timepoints):
                    tp_data = plot_df[plot_df['timepoint'] == tp]
                    if len(tp_data) > 10:  # Need enough data
                        try:
                            # Perform test
                            test_results = perform_pairwise_tests(
                                tp_data, 'value', 'group', groups,
                                test=self.stat_method, fdr_correction=self.fdr_correction
                            )

                            if len(test_results) > 0 and test_results.iloc[0]['significant']:
                                from ..stats.plot_stats import get_significance_symbol
                                pval = test_results.iloc[0]['pvalue_adj']
                                symbol = get_significance_symbol(pval, self.sig_symbols)

                                if symbol != 'ns':
                                    # Get max y for this timepoint
                                    max_y = tp_data['value'].max()
                                    y_min, y_max = ax.get_ylim()
                                    y_range = y_max - y_min

                                    # Draw significance bar
                                    x1 = i
                                    x2 = i + width
                                    bar_y = max_y + y_range * 0.03
                                    ax.plot([x1, x2], [bar_y, bar_y], 'k-', linewidth=1.0, zorder=100)
                                    ax.text((x1 + x2) / 2, bar_y + y_range * 0.01, symbol,
                                           ha='center', va='bottom', fontsize=9, fontweight='bold',
                                           zorder=101)
                        except:
                            pass  # Skip if test fails

    def _plot_violins(self, data: pd.DataFrame, ax, value_col: str,
                     group_col: str, groups: List[str]):
        """Violin plots per timepoint."""
        timepoints = sorted(data['timepoint'].unique())

        # Prepare data
        plot_data = []
        for tp in timepoints:
            tp_data = data[data['timepoint'] == tp]
            for group in groups:
                group_data = tp_data[tp_data[group_col] == group]
                for val in group_data[value_col]:
                    plot_data.append({'timepoint': tp, 'group': group, 'value': val})

        plot_df = pd.DataFrame(plot_data)

        if len(plot_df) > 0:
            # Create violin plot
            positions = []
            violin_data = []
            violin_colors = []

            width = 0.35
            for i, tp in enumerate(timepoints):
                for j, group in enumerate(groups):
                    group_tp_data = plot_df[(plot_df['timepoint'] == tp) &
                                            (plot_df['group'] == group)]
                    if len(group_tp_data) > 0:
                        violin_data.append(group_tp_data['value'].values)
                        positions.append(i + j * width)
                        violin_colors.append(self.group_colors.get(group, '#000000'))

            parts = ax.violinplot(violin_data, positions=positions, widths=width*0.8,
                                 showmeans=True, showmedians=True)

            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(violin_colors[i])
                pc.set_alpha(0.6)

            ax.set_xticks([i + width/2 for i in range(len(timepoints))])
            ax.set_xticklabels([str(tp) for tp in timepoints])
            ax.set_xlabel(self.timepoint_label, fontsize=12, fontweight='bold')
            ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_title('Violin Plots', fontsize=13, fontweight='bold')

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=self.group_colors.get(g, '#000000'),
                                    alpha=0.6, label=g) for g in groups]
            ax.legend(handles=legend_elements, loc='best', frameon=True)
            ax.grid(True, alpha=0.3, axis='y')

    def _plot_combined(self, data: pd.DataFrame, ax, value_col: str,
                      group_col: str, groups: List[str]):
        """Combined overlay plot."""
        for group in groups:
            group_data = data[data[group_col] == group]

            if len(group_data) == 0:
                continue

            # Calculate stats
            summary = group_data.groupby('timepoint')[value_col].agg(['mean', 'sem'])
            timepoints = summary.index.values
            means = summary['mean'].values
            sems = summary['sem'].values

            color = self.group_colors.get(group, '#000000')

            # Box plot style overlay
            for tp in timepoints:
                tp_vals = group_data[group_data['timepoint'] == tp][value_col].values
                if len(tp_vals) > 0:
                    # Box
                    q1, median, q3 = np.percentile(tp_vals, [25, 50, 75])
                    ax.plot([tp-0.1, tp+0.1], [median, median], color=color, linewidth=3)
                    ax.plot([tp, tp], [q1, q3], color=color, linewidth=2, alpha=0.6)

            # Line with CI
            ax.plot(timepoints, means, '-o', color=color, linewidth=2,
                   markersize=6, label=group, alpha=0.8)
            ax.fill_between(timepoints, means - sems, means + sems,
                           alpha=0.15, color=color)

        ax.set_xlabel(self.timepoint_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title('Combined Overlay', fontsize=13, fontweight='bold')
        ax.legend(frameon=True, loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

    def _plot_publication_version(self, data: pd.DataFrame, population: str,
                                  value_col: str, group_col: str, groups: List[str]):
        """Clean publication-quality plot."""
        fig, ax = plt.subplots(figsize=(8, 6))

        for group in groups:
            group_data = data[data[group_col] == group]

            if len(group_data) == 0:
                continue

            summary = group_data.groupby('timepoint')[value_col].agg(['mean', 'sem'])
            timepoints = summary.index.values
            means = summary['mean'].values
            sems = summary['sem'].values

            color = self.group_colors.get(group, '#000000')

            ax.plot(timepoints, means, '-o', color=color, linewidth=3,
                   markersize=10, label=group, markeredgecolor='white',
                   markeredgewidth=1.5)
            ax.fill_between(timepoints, means - sems, means + sems,
                           alpha=0.2, color=color)

        ax.set_xlabel(self.timepoint_label, fontsize=14, fontweight='bold')
        ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=14, fontweight='bold')
        ax.legend(frameon=False, loc='best', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

        plt.tight_layout()

        plot_path = self.plots_dir / f'{population}_{value_col}_publication.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_all_populations_overview(self, all_results: Dict, value_col: str = 'count',
                                     group_col: str = 'main_group'):
        """
        Create overview plot with all populations.

        Parameters
        ----------
        all_results : dict
            Dict of {population_name: dataframe}
        value_col : str
            Column to plot
        group_col : str
            Group column for comparison
        """
        populations = list(all_results.keys())

        # Filter to non-fraction populations for cleaner overview
        populations = [p for p in populations if 'fraction' not in p]

        if len(populations) == 0:
            return

        n_pops = len(populations)
        ncols = 3
        nrows = (n_pops + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5*nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        colors = {'KPT': '#E41A1C', 'KPNT': '#377EB8'}

        for idx, pop in enumerate(populations):
            ax = axes[idx]
            data = all_results[pop]

            if value_col not in data.columns or group_col not in data.columns:
                continue

            groups = sorted(data[group_col].unique())

            for group in groups:
                group_data = data[data[group_col] == group]
                summary = group_data.groupby('timepoint')[value_col].agg(['mean', 'sem'])

                timepoints = summary.index.values
                means = summary['mean'].values
                sems = summary['sem'].values

                color = colors.get(group, '#000000')

                ax.plot(timepoints, means, '-o', color=color, linewidth=2,
                       label=group, markersize=6)
                ax.fill_between(timepoints, means - sems, means + sems,
                               alpha=0.2, color=color)

            ax.set_title(pop.replace('_', ' '), fontsize=11, fontweight='bold')
            ax.set_xlabel('Time (days)', fontsize=10)
            ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=10)
            if idx == 0:
                ax.legend(frameon=False, fontsize=9)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_pops, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'All Populations - {value_col.replace("_", " ").title()}',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        plot_path = self.plots_dir / f'all_populations_{value_col}_overview.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
