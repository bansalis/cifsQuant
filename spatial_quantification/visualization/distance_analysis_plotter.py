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
        # Default colors - will be extended dynamically for unknown groups
        self.default_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#984EA3',
                               '#A65628', '#F781BF', '#999999', '#66C2A5', '#FC8D62']
        self.group_colors = plotting_config.get('group_colors', {})

        # Set font
        font_family = plotting_config.get('font_family', 'DejaVu Sans')
        font_size = plotting_config.get('font_size', 11)

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.family'] = font_family
        plt.rcParams['font.size'] = font_size

    def _get_color(self, group: str, groups: List[str] = None) -> str:
        """Get color for a group, dynamically assigning if not predefined."""
        if group in self.group_colors:
            return self.group_colors[group]
        # Assign color based on position in groups list
        if groups is not None and group in groups:
            idx = groups.index(group) % len(self.default_colors)
        else:
            # Hash-based assignment for consistency
            idx = hash(group) % len(self.default_colors)
        return self.default_colors[idx]

    def plot_distance_comprehensive(self, data: pd.DataFrame,
                                    source: str, target: str,
                                    group_col: str = 'group',
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
        for group in groups:
            group_data = data[data[group_col] == group]

            if len(group_data) == 0:
                continue

            summary = group_data.groupby('timepoint')['mean_distance'].agg(['mean', 'sem'])
            timepoints = summary.index.values
            means = summary['mean'].values
            sems = summary['sem'].values

            color = self._get_color(group, groups)

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
        # Get all distances
        all_distances = data['mean_distance'].values
        bins = np.linspace(0, np.percentile(all_distances, 99), 50)

        for group in groups:
            group_data = data[data[group_col] == group]

            if len(group_data) == 0:
                continue

            distances = group_data['mean_distance'].values
            color = self._get_color(group, groups)

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
                    box_colors.append(self._get_color(group, groups))

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
        legend_elements = [Patch(facecolor=self._get_color(g, groups),
                                alpha=0.6, label=g) for g in groups]
        ax.legend(handles=legend_elements, loc='best', frameon=True)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_violins_per_timepoint(self, data: pd.DataFrame, ax,
                                    group_col: str, groups: List[str]):
        """Violin plots per timepoint."""

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
                    violin_colors.append(self._get_color(group, groups))

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
        legend_elements = [Patch(facecolor=self._get_color(g, groups),
                                alpha=0.6, label=g) for g in groups]
        ax.legend(handles=legend_elements, loc='best', frameon=True)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_publication_version(self, data: pd.DataFrame, source: str, target: str,
                                  group_col: str, groups: List[str]):
        """Clean publication version."""
        fig, ax = plt.subplots(figsize=(8, 6))

        for group in groups:
            group_data = data[data[group_col] == group]

            if len(group_data) == 0:
                continue

            summary = group_data.groupby('timepoint')['mean_distance'].agg(['mean', 'sem'])
            timepoints = summary.index.values
            means = summary['mean'].values
            sems = summary['sem'].values

            color = self._get_color(group, groups)

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

    def plot_all_distances_heatmap(self, all_results: Dict, group_col: str = 'group'):
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

    def plot_differential_distances(self, diff_data: pd.DataFrame,
                                    group_col: str = 'group',
                                    groups: List[str] = None):
        """
        Plot differential distance (mean_dist_to_pos - mean_dist_to_neg) for each
        source/marker pair.

        Two panels per marker:
        1. Paired dot plot: mean_dist_to_pos and mean_dist_to_neg per sample, connected
           by a line. Slope direction shows whether source cells are closer to pos or neg.
        2. Violin/box of differential_distance by group. Negative = closer to positive zone.

        Label note: 'within-tumor immune cells only' if within_tumor_only is set.
        """
        if diff_data is None or len(diff_data) == 0:
            return

        print("    Generating differential distance plots...")

        if groups is None and group_col in diff_data.columns:
            groups = sorted(diff_data[group_col].dropna().unique())

        within_tumor = diff_data['within_tumor_only'].any() if 'within_tumor_only' in diff_data.columns else False
        scope_label = ' (within-tumor immune cells only)' if within_tumor else ''

        for source in diff_data['source_population'].unique():
            source_data = diff_data[diff_data['source_population'] == source]

            for marker in source_data['marker'].unique():
                marker_data = source_data[source_data['marker'] == marker].copy()

                if len(marker_data) == 0:
                    continue

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                fig.suptitle(
                    f'Differential Distance: {source} → {marker}+/- tumor{scope_label}',
                    fontsize=13, fontweight='bold'
                )

                # Panel 1: Paired dot plot
                ax1 = axes[0]
                if groups is None or group_col not in marker_data.columns:
                    plot_groups = ['all']
                    marker_data['_grp'] = 'all'
                    grp_col = '_grp'
                else:
                    plot_groups = [g for g in groups if g in marker_data[group_col].values]
                    grp_col = group_col

                x_pos = 0
                x_ticks = []
                x_labels = []

                for grp in plot_groups:
                    grp_data = marker_data[marker_data[grp_col] == grp] if grp != 'all' else marker_data
                    color = self._get_color(grp, plot_groups)

                    for _, row in grp_data.iterrows():
                        if pd.isna(row.get('mean_dist_to_pos')) or pd.isna(row.get('mean_dist_to_neg')):
                            continue
                        ax1.plot([x_pos - 0.1, x_pos + 0.1],
                                 [row['mean_dist_to_pos'], row['mean_dist_to_neg']],
                                 '-', color=color, alpha=0.5, linewidth=1)
                        ax1.scatter(x_pos - 0.1, row['mean_dist_to_pos'],
                                    color=color, s=40, zorder=5, marker='o')
                        ax1.scatter(x_pos + 0.1, row['mean_dist_to_neg'],
                                    color=color, s=40, zorder=5, marker='s')

                    x_ticks.append(x_pos)
                    x_labels.append(grp)
                    x_pos += 1

                ax1.set_xticks(x_ticks)
                ax1.set_xticklabels(x_labels, rotation=30, ha='right')
                ax1.set_ylabel('Mean Distance (μm)', fontsize=11, fontweight='bold')
                ax1.set_title('Paired distances: circle=pos, square=neg', fontsize=10)
                ax1.grid(True, alpha=0.3, axis='y')

                from matplotlib.lines import Line2D
                legend_handles = [
                    Line2D([0], [0], marker='o', color='gray', linestyle='none', label=f'{marker}+'),
                    Line2D([0], [0], marker='s', color='gray', linestyle='none', label=f'{marker}-'),
                ]
                ax1.legend(handles=legend_handles, loc='best', frameon=True)

                # Panel 2: Violin/box of differential_distance by group
                ax2 = axes[1]
                if group_col in marker_data.columns and len(plot_groups) > 1:
                    plot_data_list = []
                    color_list = []
                    label_list = []
                    for grp in plot_groups:
                        grp_vals = marker_data[marker_data[group_col] == grp]['differential_distance'].dropna().values
                        if len(grp_vals) > 0:
                            plot_data_list.append(grp_vals)
                            color_list.append(self._get_color(grp, plot_groups))
                            label_list.append(grp)

                    if plot_data_list:
                        positions = list(range(len(plot_data_list)))
                        vparts = ax2.violinplot(plot_data_list, positions=positions,
                                               showmeans=True, showmedians=False)
                        for i, pc in enumerate(vparts['bodies']):
                            pc.set_facecolor(color_list[i])
                            pc.set_alpha(0.6)

                        bp = ax2.boxplot(plot_data_list, positions=positions, widths=0.15,
                                        patch_artist=True, showfliers=False)
                        for patch in bp['boxes']:
                            patch.set_facecolor('white')
                            patch.set_alpha(0.8)

                        ax2.set_xticks(positions)
                        ax2.set_xticklabels(label_list)
                else:
                    diffs = marker_data['differential_distance'].dropna().values
                    if len(diffs) > 0:
                        ax2.violinplot([diffs], positions=[0], showmeans=True)
                        ax2.set_xticks([0])
                        ax2.set_xticklabels(['all'])

                ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
                ax2.set_ylabel('Differential Distance (μm)\n(pos - neg; negative = closer to positive zone)',
                               fontsize=10, fontweight='bold')
                ax2.set_title('Differential Distance by Group', fontsize=11, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')

                plt.tight_layout()

                safe_source = source.replace('+', 'pos').replace('-', 'neg')
                plot_path = self.plots_dir / f'differential_distance_{safe_source}_{marker}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

        print(f"    ✓ Saved differential distance plots to {self.plots_dir.name}/")

    def plot_distance_histograms_binned(self, data: pd.DataFrame,
                                        source: str, target: str,
                                        group_col: str = 'group',
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

                    color = self._get_color(group, groups)

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

                color = self._get_color(group, groups)

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
