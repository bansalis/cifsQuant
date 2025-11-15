"""
Distance Analysis Plotter
Comprehensive plotting with histograms, box plots, and time series
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Optional
import warnings
from ..stats.plot_stats import add_significance_bars, perform_pairwise_tests, get_significance_symbol

try:
    from .plot_utils import detect_plot_type, create_dual_plots, calculate_statistics
    HAS_PLOT_UTILS = True
except ImportError:
    HAS_PLOT_UTILS = False
    warnings.warn("Plot utilities not available")


class DistanceAnalysisPlotter:
    """
    Comprehensive plotting for distance analysis.

    Generates:
    - Overlapping histograms (distribution comparison)
    - Box plots per timepoint
    - Time series with confidence bands
    - Violin plots
    - Statistical annotations
    """

    def __init__(self, output_dir: Path, config: Dict):
        """Initialize plotter."""
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Get plotting config
        plotting_config = config.get('plotting', {})
        self.timepoint_label = plotting_config.get('timepoint_label', 'Time (weeks)')
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

    def plot_distance_comprehensive(self, data: pd.DataFrame,
                                    source: str, target: str,
                                    group_col: str = 'main_group',
                                    groups: List[str] = None):
        """
        Create comprehensive distance plots.

        4 panels:
        1. Time series (line + CI)
        2. Overlapping histograms
        3. Box plots per timepoint
        4. Violin plots per timepoint
        """
        if groups is None:
            groups = sorted(data[group_col].unique())

        plot_data = data[data[group_col].isin(groups)].copy()

        if len(plot_data) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Distance: {source} → {target}',
                    fontsize=16, fontweight='bold', y=0.995)

        # Plot 1: Time series
        self._plot_time_series(plot_data, axes[0, 0], group_col, groups)

        # Plot 2: Overlapping histograms
        self._plot_overlapping_histograms(plot_data, axes[0, 1], group_col, groups)

        # Plot 3: Box plots
        self._plot_boxes_per_timepoint(plot_data, axes[1, 0], group_col, groups)

        # Plot 4: Violin plots
        self._plot_violins_per_timepoint(plot_data, axes[1, 1], group_col, groups)

        plt.tight_layout()

        plot_path = self.plots_dir / f'{source}_to_{target}_comprehensive.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Publication version
        self._plot_publication_version(plot_data, source, target, group_col, groups)

    def _plot_time_series(self, data: pd.DataFrame, ax, group_col: str, groups: List[str]):
        """Time series with confidence bands."""
        colors = {'KPT': '#E41A1C', 'KPNT': '#377EB8', 'cis': '#4DAF4A', 'trans': '#FF7F00'}

        for group in groups:
            group_data = data[data[group_col] == group]

            if len(group_data) == 0:
                continue

            summary = group_data.groupby('timepoint')['mean_distance'].agg(['mean', 'sem'])
            timepoints = summary.index.values
            means = summary['mean'].values
            sems = summary['sem'].values

            color = colors.get(group, '#000000')

            # Scatter individual points
            ax.scatter(group_data['timepoint'], group_data['mean_distance'],
                      alpha=0.3, s=40, color=color, edgecolors='none')

            # Line with CI
            ax.plot(timepoints, means, '-o', color=color, linewidth=2.5,
                   markersize=8, label=group, zorder=10)
            ax.fill_between(timepoints, means - sems, means + sems,
                           alpha=0.2, color=color)

        ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Distance (μm)', fontsize=12, fontweight='bold')
        ax.set_title('Distance Over Time', fontsize=13, fontweight='bold')
        ax.legend(frameon=True, loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_overlapping_histograms(self, data: pd.DataFrame, ax,
                                     group_col: str, groups: List[str]):
        """Overlapping histograms showing full distribution."""
        colors = {'KPT': '#E41A1C', 'KPNT': '#377EB8', 'cis': '#4DAF4A', 'trans': '#FF7F00'}

        # Get all distances
        all_distances = data['mean_distance'].values
        bins = np.linspace(0, np.percentile(all_distances, 99), 50)

        for group in groups:
            group_data = data[data[group_col] == group]

            if len(group_data) == 0:
                continue

            distances = group_data['mean_distance'].values
            color = colors.get(group, '#000000')

            # Histogram
            ax.hist(distances, bins=bins, alpha=0.5, color=color,
                   label=group, density=True, edgecolor='white', linewidth=0.5)

            # KDE overlay
            if len(distances) > 10:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(distances)
                x_range = np.linspace(distances.min(), distances.max(), 200)
                ax.plot(x_range, kde(x_range), color=color, linewidth=2.5, alpha=0.8)

        ax.set_xlabel('Distance (μm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title('Distance Distribution', fontsize=13, fontweight='bold')
        ax.legend(frameon=True, loc='best')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_boxes_per_timepoint(self, data: pd.DataFrame, ax,
                                  group_col: str, groups: List[str]):
        """Box plots per timepoint."""
        colors = {'KPT': '#E41A1C', 'KPNT': '#377EB8', 'cis': '#4DAF4A', 'trans': '#FF7F00'}

        timepoints = sorted(data['timepoint'].unique())

        box_data = []
        box_positions = []
        box_colors = []
        width = 0.35

        for i, tp in enumerate(timepoints):
            tp_data = data[data['timepoint'] == tp]
            for j, group in enumerate(groups):
                group_data = tp_data[tp_data[group_col] == group]['mean_distance'].values
                if len(group_data) > 0:
                    box_data.append(group_data)
                    box_positions.append(i + j * width)
                    box_colors.append(colors.get(group, '#000000'))

        if box_data:
            bp = ax.boxplot(box_data, positions=box_positions, widths=width*0.8,
                           patch_artist=True, showfliers=False)

            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_xticks([i + width/2 for i in range(len(timepoints))])
            ax.set_xticklabels([str(tp) for tp in timepoints])

        ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance (μm)', fontsize=12, fontweight='bold')
        ax.set_title('Box Plots by Timepoint', fontsize=13, fontweight='bold')

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors.get(g, '#000000'),
                                alpha=0.6, label=g) for g in groups]
        ax.legend(handles=legend_elements, loc='best', frameon=True)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_violins_per_timepoint(self, data: pd.DataFrame, ax,
                                    group_col: str, groups: List[str]):
        """Violin plots per timepoint."""
        colors = {'KPT': '#E41A1C', 'KPNT': '#377EB8', 'cis': '#4DAF4A', 'trans': '#FF7F00'}

        timepoints = sorted(data['timepoint'].unique())

        violin_data = []
        positions = []
        violin_colors = []
        width = 0.35

        for i, tp in enumerate(timepoints):
            tp_data = data[data['timepoint'] == tp]
            for j, group in enumerate(groups):
                group_data = tp_data[tp_data[group_col] == group]['mean_distance'].values
                if len(group_data) > 0:
                    violin_data.append(group_data)
                    positions.append(i + j * width)
                    violin_colors.append(colors.get(group, '#000000'))

        if violin_data:
            parts = ax.violinplot(violin_data, positions=positions, widths=width*0.8,
                                 showmeans=True, showmedians=True)

            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(violin_colors[i])
                pc.set_alpha(0.6)

            ax.set_xticks([i + width/2 for i in range(len(timepoints))])
            ax.set_xticklabels([str(tp) for tp in timepoints])

        ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance (μm)', fontsize=12, fontweight='bold')
        ax.set_title('Violin Plots by Timepoint', fontsize=13, fontweight='bold')

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors.get(g, '#000000'),
                                alpha=0.6, label=g) for g in groups]
        ax.legend(handles=legend_elements, loc='best', frameon=True)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_publication_version(self, data: pd.DataFrame, source: str, target: str,
                                  group_col: str, groups: List[str]):
        """Clean publication version."""
        fig, ax = plt.subplots(figsize=(8, 6))

        colors = {'KPT': '#E41A1C', 'KPNT': '#377EB8', 'cis': '#4DAF4A', 'trans': '#FF7F00'}

        for group in groups:
            group_data = data[data[group_col] == group]

            if len(group_data) == 0:
                continue

            summary = group_data.groupby('timepoint')['mean_distance'].agg(['mean', 'sem'])
            timepoints = summary.index.values
            means = summary['mean'].values
            sems = summary['sem'].values

            color = colors.get(group, '#000000')

            ax.plot(timepoints, means, '-o', color=color, linewidth=3,
                   markersize=10, label=group, markeredgecolor='white',
                   markeredgewidth=1.5)
            ax.fill_between(timepoints, means - sems, means + sems,
                           alpha=0.2, color=color)

        ax.set_xlabel('Time (weeks)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Distance (μm)', fontsize=14, fontweight='bold')
        ax.legend(frameon=False, loc='best', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

        plt.tight_layout()

        plot_path = self.plots_dir / f'{source}_to_{target}_publication.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_all_distances_heatmap(self, all_results: Dict, group_col: str = 'main_group'):
        """Create heatmap of all distance pairings."""
        # Extract mean distances per pairing
        pairing_means = {}

        for pairing_name, data in all_results.items():
            if group_col not in data.columns or 'mean_distance' not in data.columns:
                continue

            groups = sorted(data[group_col].unique())

            for group in groups:
                group_data = data[data[group_col] == group]
                mean_dist = group_data['mean_distance'].mean()

                # Parse pairing name
                parts = pairing_name.split('_to_')
                if len(parts) == 2:
                    source, target = parts
                    key = (source, target, group)
                    pairing_means[key] = mean_dist

        if not pairing_means:
            return

        # Create heatmap data
        sources = sorted(set(k[0] for k in pairing_means.keys()))
        targets = sorted(set(k[1] for k in pairing_means.keys()))
        groups = sorted(set(k[2] for k in pairing_means.keys()))

        for group in groups:
            heatmap_data = np.zeros((len(sources), len(targets)))

            for i, source in enumerate(sources):
                for j, target in enumerate(targets):
                    key = (source, target, group)
                    if key in pairing_means:
                        heatmap_data[i, j] = pairing_means[key]
                    else:
                        heatmap_data[i, j] = np.nan

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')

            ax.set_xticks(range(len(targets)))
            ax.set_yticks(range(len(sources)))
            ax.set_xticklabels(targets, rotation=45, ha='right')
            ax.set_yticklabels(sources)

            plt.colorbar(im, ax=ax, label='Mean Distance (μm)')

            ax.set_title(f'Distance Heatmap - {group}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Target Population', fontsize=12)
            ax.set_ylabel('Source Population', fontsize=12)

            plt.tight_layout()

            plot_path = self.plots_dir / f'distance_heatmap_{group}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

    def plot_distance_histograms_binned(self, data: pd.DataFrame,
                                        source: str, target: str,
                                        group_col: str = 'main_group',
                                        groups: List[str] = None,
                                        distance_bins: List[int] = None):
        """
        Create binned distance histograms showing peak shifts between groups/timepoints.

        Shows histograms with distance bands on x-axis (e.g., 0-50, 50-100, 100-200, etc.)
        to visualize shifts in distance distributions.

        Parameters
        ----------
        data : pd.DataFrame
            Distance data with 'mean_distance' column
        source : str
            Source cell population
        target : str
            Target cell population
        group_col : str
            Column defining groups
        groups : List[str], optional
            Groups to plot
        distance_bins : List[int], optional
            Distance bin edges (e.g., [0, 50, 100, 200, 500])
        """
        if groups is None:
            groups = sorted(data[group_col].unique())

        if distance_bins is None:
            distance_bins = [0, 50, 100, 200, 300, 500]

        plot_data = data[data[group_col].isin(groups)].copy()

        if len(plot_data) == 0:
            return

        # Create bin labels
        bin_labels = [f'{distance_bins[i]}-{distance_bins[i+1]}' for i in range(len(distance_bins)-1)]
        bin_labels.append(f'>{distance_bins[-1]}')

        # Bin the distances
        plot_data['distance_bin'] = pd.cut(plot_data['mean_distance'],
                                           bins=distance_bins + [np.inf],
                                           labels=bin_labels,
                                           include_lowest=True)

        # Check if we have timepoints
        has_timepoints = 'timepoint' in plot_data.columns and plot_data['timepoint'].nunique() > 1

        if has_timepoints:
            # Plot per timepoint
            timepoints = sorted(plot_data['timepoint'].unique())
            n_timepoints = len(timepoints)

            fig, axes = plt.subplots(1, n_timepoints, figsize=(6*n_timepoints, 5))
            if n_timepoints == 1:
                axes = [axes]

            fig.suptitle(f'Distance Histograms: {source} → {target} (Binned by Distance)',
                        fontsize=14, fontweight='bold')

            for tp_idx, tp in enumerate(timepoints):
                ax = axes[tp_idx]
                tp_data = plot_data[plot_data['timepoint'] == tp]

                # Count cells in each bin for each group
                for group in groups:
                    group_data = tp_data[tp_data[group_col] == group]

                    if len(group_data) == 0:
                        continue

                    bin_counts = group_data['distance_bin'].value_counts()
                    bin_counts = bin_counts.reindex(bin_labels, fill_value=0)

                    # Normalize to percentages
                    bin_percentages = (bin_counts / bin_counts.sum()) * 100

                    color = self.group_colors.get(group, '#000000')

                    x_positions = np.arange(len(bin_labels))
                    width = 0.35
                    offset = (groups.index(group) - len(groups)/2 + 0.5) * width

                    ax.bar(x_positions + offset, bin_percentages.values,
                          width=width, alpha=0.7, label=group, color=color,
                          edgecolor='white', linewidth=1)

                ax.set_xlabel('Distance Range (μm)', fontsize=11, fontweight='bold')
                ax.set_ylabel('Percentage of Cells (%)', fontsize=11, fontweight='bold')
                ax.set_title(f'Timepoint {tp}', fontsize=12, fontweight='bold')
                ax.set_xticks(np.arange(len(bin_labels)))
                ax.set_xticklabels(bin_labels, rotation=45, ha='right')
                ax.legend(frameon=True, loc='best', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()

            plot_path = self.plots_dir / f'{source}_to_{target}_distance_histograms_binned.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        else:
            # Single plot for all groups (no timepoints)
            fig, ax = plt.subplots(figsize=(10, 6))

            for group in groups:
                group_data = plot_data[plot_data[group_col] == group]

                if len(group_data) == 0:
                    continue

                bin_counts = group_data['distance_bin'].value_counts()
                bin_counts = bin_counts.reindex(bin_labels, fill_value=0)

                # Normalize to percentages
                bin_percentages = (bin_counts / bin_counts.sum()) * 100

                color = self.group_colors.get(group, '#000000')

                x_positions = np.arange(len(bin_labels))
                width = 0.35
                offset = (groups.index(group) - len(groups)/2 + 0.5) * width

                ax.bar(x_positions + offset, bin_percentages.values,
                      width=width, alpha=0.7, label=group, color=color,
                      edgecolor='white', linewidth=1)

            ax.set_xlabel('Distance Range (μm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage of Cells (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'Distance Distribution: {source} → {target}',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(np.arange(len(bin_labels)))
            ax.set_xticklabels(bin_labels, rotation=45, ha='right')
            ax.legend(frameon=True, loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()

            plot_path = self.plots_dir / f'{source}_to_{target}_distance_histogram_binned.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
