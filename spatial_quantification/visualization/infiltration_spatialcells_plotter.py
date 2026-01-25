"""
Comprehensive Visualization for SpatialCells Infiltration Analysis
Extensive plots for immune infiltration data with distance-based zones
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
from scipy import stats


class InfiltrationSpatialCellsPlotter:
    """
    Comprehensive visualization for SpatialCells-based infiltration analysis.

    Generates extensive plots:
    - Infiltration density by zone (box plots, violin plots, bar plots)
    - Per-immune-population analysis across all zones
    - Group/condition comparisons with statistics
    - Heatmaps of infiltration patterns
    - Per-tumor infiltration profiles
    - Time-based comparisons (if multiple timepoints)
    - Zone gradient analysis (infiltration decay with distance)
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
        self.group_colors = plotting_config.get('group_colors', {})
        self.default_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#984EA3', '#A65628']
        self.timepoint_label = plotting_config.get('timepoint_label', 'Time (weeks)')

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10

    def _perform_statistical_test(self, group1_data: np.ndarray, group2_data: np.ndarray,
                                  test_type: str = 'auto') -> Tuple[float, float, str]:
        """
        Perform statistical test between two groups.

        Parameters
        ----------
        group1_data : np.ndarray
            Data for group 1
        group2_data : np.ndarray
            Data for group 2
        test_type : str
            Type of test ('ttest', 'mannwhitneyu', 'auto')

        Returns
        -------
        tuple
            (statistic, p_value, test_name)
        """
        # Remove NaNs
        group1_clean = group1_data[~np.isnan(group1_data)]
        group2_clean = group2_data[~np.isnan(group2_data)]

        if len(group1_clean) < 3 or len(group2_clean) < 3:
            return np.nan, np.nan, 'insufficient_data'

        if test_type == 'auto':
            # Use Shapiro-Wilk to check normality
            try:
                _, p1 = stats.shapiro(group1_clean)
                _, p2 = stats.shapiro(group2_clean)

                # If both p > 0.05, assume normal and use t-test
                if p1 > 0.05 and p2 > 0.05:
                    test_type = 'ttest'
                else:
                    test_type = 'mannwhitneyu'
            except:
                test_type = 'mannwhitneyu'

        if test_type == 'ttest':
            try:
                stat, pval = stats.ttest_ind(group1_clean, group2_clean)
                return stat, pval, 't-test'
            except:
                return np.nan, np.nan, 'test_failed'
        elif test_type == 'mannwhitneyu':
            try:
                stat, pval = stats.mannwhitneyu(group1_clean, group2_clean, alternative='two-sided')
                return stat, pval, 'Mann-Whitney U'
            except:
                return np.nan, np.nan, 'test_failed'
        else:
            return np.nan, np.nan, 'unknown_test'

    def _format_pvalue(self, pval: float) -> str:
        """Format p-value with significance stars."""
        if np.isnan(pval):
            return ''
        elif pval < 0.001:
            return '***'
        elif pval < 0.01:
            return '**'
        elif pval < 0.05:
            return '*'
        else:
            return 'ns'

    def _add_significance_stars(self, ax, x_pos1: float, x_pos2: float, y_pos: float,
                                pval: float, height: float = 0.02):
        """Add significance stars/bars to plot."""
        sig_text = self._format_pvalue(pval)
        if sig_text and sig_text != 'ns':
            # Draw horizontal line
            ax.plot([x_pos1, x_pos2], [y_pos, y_pos], 'k-', linewidth=1.5)
            # Add text
            ax.text((x_pos1 + x_pos2) / 2, y_pos + height, sig_text,
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    def plot_infiltration_by_zone_boxplots(self, infiltration_df: pd.DataFrame):
        """
        Box plots showing infiltration density by zone for each immune population.

        Parameters
        ----------
        infiltration_df : pd.DataFrame
            Infiltration data with columns: sample_id, structure_id, immune_population,
            zone, count, infiltration_density, timepoint, main_group
        """
        if infiltration_df is None or len(infiltration_df) == 0:
            return

        immune_pops = sorted(infiltration_df['immune_population'].unique())
        zones = sorted(infiltration_df['zone'].unique())
        groups = sorted(infiltration_df['group'].unique())

        # Create subplots: one row per immune population
        n_pops = len(immune_pops)
        fig, axes = plt.subplots(n_pops, 1, figsize=(14, 5*n_pops))

        if n_pops == 1:
            axes = [axes]

        for idx, immune_pop in enumerate(immune_pops):
            ax = axes[idx]
            pop_data = infiltration_df[infiltration_df['immune_population'] == immune_pop]

            # Prepare data for box plot
            plot_data = []
            labels = []
            colors = []

            for zone in zones:
                for group in groups:
                    subset = pop_data[(pop_data['zone'] == zone) &
                                     (pop_data['group'] == group)]
                    if len(subset) > 0:
                        plot_data.append(subset['infiltration_density'].values)
                        labels.append(f"{zone}\n{group}")
                        colors.append(self.group_colors.get(group, '#888888'))

            if not plot_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{immune_pop}', fontsize=14, fontweight='bold')
                continue

            # Create box plot
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True,
                           showfliers=True, widths=0.6)

            # Color boxes by group
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            # Style
            for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color='black', linewidth=1.5)

            # Add statistical comparisons between groups within each zone
            if len(groups) == 2:
                y_max = max([d.max() for d in plot_data if len(d) > 0])
                y_range = y_max - min([d.min() for d in plot_data if len(d) > 0])

                for zone_idx, zone in enumerate(zones):
                    # Get data for both groups in this zone
                    group1_data = pop_data[(pop_data['zone'] == zone) &
                                          (pop_data['group'] == groups[0])]['infiltration_density'].values
                    group2_data = pop_data[(pop_data['zone'] == zone) &
                                          (pop_data['group'] == groups[1])]['infiltration_density'].values

                    if len(group1_data) >= 3 and len(group2_data) >= 3:
                        # Perform statistical test
                        _, pval, _ = self._perform_statistical_test(group1_data, group2_data)

                        # Add significance annotation
                        if not np.isnan(pval) and pval < 0.05:
                            x_pos1 = zone_idx * len(groups) + 1
                            x_pos2 = zone_idx * len(groups) + 2
                            y_pos = y_max + y_range * 0.05 * (zone_idx + 1)

                            self._add_significance_stars(ax, x_pos1, x_pos2, y_pos, pval,
                                                        height=y_range * 0.02)

            ax.set_ylabel('Infiltration Density\n(immune cells / tumor cells)',
                         fontsize=12, fontweight='bold')
            ax.set_title(f'{immune_pop} - Infiltration by Zone and Group\n(*, **, *** = p<0.05, 0.01, 0.001)',
                        fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = self.plots_dir / 'infiltration_by_zone_boxplots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {plot_path.name}")

    def plot_infiltration_heatmaps(self, infiltration_df: pd.DataFrame):
        """
        Heatmaps showing infiltration density: immune populations x zones.
        One heatmap per group/condition.

        Parameters
        ----------
        infiltration_df : pd.DataFrame
            Infiltration data
        """
        if infiltration_df is None or len(infiltration_df) == 0:
            return

        groups = sorted(infiltration_df['group'].unique())

        # Create subplots: one per group
        fig, axes = plt.subplots(1, len(groups), figsize=(10*len(groups), 8))

        if len(groups) == 1:
            axes = [axes]

        for idx, group in enumerate(groups):
            ax = axes[idx]
            group_data = infiltration_df[infiltration_df['group'] == group]

            # Aggregate across samples and structures
            agg = group_data.groupby(['immune_population', 'zone']).agg({
                'infiltration_density': 'mean'
            }).reset_index()

            # Pivot for heatmap
            pivot = agg.pivot(index='immune_population', columns='zone',
                             values='infiltration_density')

            if len(pivot) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{group}', fontsize=14, fontweight='bold')
                continue

            # Create heatmap
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd',
                       cbar_kws={'label': 'Mean Infiltration Density'},
                       ax=ax, vmin=0, linewidths=0.5, linecolor='gray')

            ax.set_title(f'{group} - Mean Infiltration Density',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Zone', fontsize=12, fontweight='bold')
            ax.set_ylabel('Immune Population', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plot_path = self.plots_dir / 'infiltration_heatmaps_by_group.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {plot_path.name}")

    def plot_zone_gradient(self, infiltration_df: pd.DataFrame):
        """
        Plot infiltration density gradient across distance zones.
        Shows how infiltration decays with distance from tumor boundary.

        Parameters
        ----------
        infiltration_df : pd.DataFrame
            Infiltration data
        """
        if infiltration_df is None or len(infiltration_df) == 0:
            return

        immune_pops = sorted(infiltration_df['immune_population'].unique())
        groups = sorted(infiltration_df['group'].unique())

        # Extract zone midpoints for plotting
        infiltration_df = infiltration_df.copy()
        infiltration_df['zone_midpoint'] = infiltration_df.apply(
            lambda row: (row['boundary_lower'] + row['boundary_upper']) / 2
            if row['zone'] != 'within_tumor' else -2.5,
            axis=1
        )

        # Create subplots: one row per immune population
        n_pops = len(immune_pops)
        fig, axes = plt.subplots(n_pops, 1, figsize=(12, 5*n_pops))

        if n_pops == 1:
            axes = [axes]

        for idx, immune_pop in enumerate(immune_pops):
            ax = axes[idx]
            pop_data = infiltration_df[infiltration_df['immune_population'] == immune_pop]

            for group in groups:
                group_data = pop_data[pop_data['group'] == group]

                if len(group_data) == 0:
                    continue

                # Aggregate by zone
                zone_summary = group_data.groupby('zone_midpoint').agg({
                    'infiltration_density': ['mean', 'sem', 'count']
                }).reset_index()

                zone_summary.columns = ['zone_midpoint', 'mean', 'sem', 'count']
                zone_summary = zone_summary.sort_values('zone_midpoint')

                color = self.group_colors.get(group, '#000000')

                # Plot line with error bars
                ax.errorbar(zone_summary['zone_midpoint'], zone_summary['mean'],
                           yerr=zone_summary['sem'], color=color, linewidth=2.5,
                           marker='o', markersize=10, capsize=5, capthick=2,
                           label=group, alpha=0.8)

            ax.set_xlabel('Distance from Tumor Boundary (μm)',
                         fontsize=12, fontweight='bold')
            ax.set_ylabel('Mean Infiltration Density', fontsize=12, fontweight='bold')
            ax.set_title(f'{immune_pop} - Infiltration Gradient',
                        fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5,
                      alpha=0.5, label='Tumor Boundary')
            ax.legend(frameon=True, loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.plots_dir / 'infiltration_zone_gradient.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {plot_path.name}")

    def plot_per_tumor_infiltration_profiles(self, infiltration_df: pd.DataFrame,
                                             top_n: int = 20):
        """
        Plot infiltration profiles for individual tumors (top N by total infiltration).

        Parameters
        ----------
        infiltration_df : pd.DataFrame
            Infiltration data
        top_n : int
            Number of top tumors to plot
        """
        if infiltration_df is None or len(infiltration_df) == 0:
            return

        # Calculate total infiltration per tumor
        tumor_totals = infiltration_df.groupby(['sample_id', 'structure_id']).agg({
            'count': 'sum',
            'group': 'first'
        }).reset_index()
        tumor_totals = tumor_totals.sort_values('count', ascending=False).head(top_n)

        immune_pops = sorted(infiltration_df['immune_population'].unique())
        zones = sorted(infiltration_df['zone'].unique())

        # Create figure: grid of tumor profiles
        n_cols = 4
        n_rows = int(np.ceil(top_n / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = axes.flatten()

        for idx, (_, tumor_row) in enumerate(tumor_totals.iterrows()):
            if idx >= len(axes):
                break

            ax = axes[idx]
            sample = tumor_row['sample_id']
            structure = tumor_row['structure_id']
            group = tumor_row['group']

            tumor_data = infiltration_df[
                (infiltration_df['sample_id'] == sample) &
                (infiltration_df['structure_id'] == structure)
            ]

            # Pivot: immune populations x zones
            pivot = tumor_data.pivot_table(
                values='infiltration_density',
                index='immune_population',
                columns='zone',
                aggfunc='mean'
            )

            if len(pivot) > 0:
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                           cbar_kws={'label': 'Density'}, ax=ax, vmin=0,
                           linewidths=0.5, linecolor='gray')

                ax.set_title(f'{sample} - Tumor {structure}\n({group})',
                            fontsize=11, fontweight='bold')
                ax.set_xlabel('Zone', fontsize=9)
                ax.set_ylabel('Immune Population', fontsize=9)
                ax.tick_params(labelsize=8)

        # Hide unused subplots
        for idx in range(len(tumor_totals), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Top {top_n} Tumors by Total Infiltration',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        plot_path = self.plots_dir / 'per_tumor_infiltration_profiles.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {plot_path.name}")

    def plot_violin_plots_by_immune_population(self, infiltration_df: pd.DataFrame):
        """
        Violin plots showing distribution of infiltration density by immune population.

        Parameters
        ----------
        infiltration_df : pd.DataFrame
            Infiltration data
        """
        if infiltration_df is None or len(infiltration_df) == 0:
            return

        zones = sorted(infiltration_df['zone'].unique())

        # Create subplots: one per zone
        n_zones = len(zones)
        fig, axes = plt.subplots(1, n_zones, figsize=(8*n_zones, 8))

        if n_zones == 1:
            axes = [axes]

        for idx, zone in enumerate(zones):
            ax = axes[idx]
            zone_data = infiltration_df[infiltration_df['zone'] == zone]

            if len(zone_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{zone}', fontsize=14, fontweight='bold')
                continue

            # Violin plot
            sns.violinplot(data=zone_data, x='immune_population', y='infiltration_density',
                          hue='group', ax=ax, palette=self.group_colors,
                          split=False, inner='quartile', cut=0)

            ax.set_xlabel('Immune Population', fontsize=12, fontweight='bold')
            ax.set_ylabel('Infiltration Density', fontsize=12, fontweight='bold')
            ax.set_title(f'{zone} - Infiltration Distribution',
                        fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.legend(title='Group', fontsize=10, title_fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = self.plots_dir / 'infiltration_violin_plots_by_zone.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {plot_path.name}")

    def plot_bar_chart_total_infiltration(self, infiltration_df: pd.DataFrame):
        """
        Bar chart showing total infiltration counts by immune population and group.

        Parameters
        ----------
        infiltration_df : pd.DataFrame
            Infiltration data
        """
        if infiltration_df is None or len(infiltration_df) == 0:
            return

        # Aggregate total counts
        totals = infiltration_df.groupby(['immune_population', 'group', 'zone']).agg({
            'count': 'sum'
        }).reset_index()

        immune_pops = sorted(totals['immune_population'].unique())
        zones = sorted(totals['zone'].unique())
        groups = sorted(totals['group'].unique())

        # Create subplots: one per zone
        n_zones = len(zones)
        fig, axes = plt.subplots(n_zones, 1, figsize=(14, 6*n_zones))

        if n_zones == 1:
            axes = [axes]

        for idx, zone in enumerate(zones):
            ax = axes[idx]
            zone_data = totals[totals['zone'] == zone]

            # Pivot for grouped bar chart
            pivot = zone_data.pivot(index='immune_population', columns='group',
                                   values='count').fillna(0)

            if len(pivot) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{zone}', fontsize=14, fontweight='bold')
                continue

            # Create grouped bar chart
            x = np.arange(len(pivot.index))
            width = 0.8 / len(groups)

            for i, group in enumerate(groups):
                if group in pivot.columns:
                    values = pivot[group].values
                    color = self.group_colors.get(group, '#888888')
                    ax.bar(x + i * width, values, width, label=group,
                          color=color, alpha=0.8)

            ax.set_xlabel('Immune Population', fontsize=12, fontweight='bold')
            ax.set_ylabel('Total Immune Cell Count', fontsize=12, fontweight='bold')
            ax.set_title(f'{zone} - Total Infiltration Counts',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x + width * (len(groups) - 1) / 2)
            ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=10)
            ax.legend(title='Group', fontsize=11, title_fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = self.plots_dir / 'infiltration_total_counts_bar_chart.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {plot_path.name}")

    def plot_timepoint_comparison(self, infiltration_df: pd.DataFrame):
        """
        Compare infiltration across timepoints (if multiple timepoints exist).

        Parameters
        ----------
        infiltration_df : pd.DataFrame
            Infiltration data
        """
        if infiltration_df is None or len(infiltration_df) == 0:
            return

        timepoints = sorted(infiltration_df['timepoint'].unique())

        if len(timepoints) <= 1:
            print("    ℹ Single timepoint detected - skipping timepoint comparison plots")
            return

        immune_pops = sorted(infiltration_df['immune_population'].unique())
        groups = sorted(infiltration_df['group'].unique())

        # Focus on key zones
        key_zones = ['within_tumor', '0_50um']
        available_zones = [z for z in key_zones if z in infiltration_df['zone'].unique()]

        if not available_zones:
            available_zones = [infiltration_df['zone'].unique()[0]]

        # Create subplots: one row per immune population, one column per zone
        n_pops = len(immune_pops)
        n_zones = len(available_zones)

        fig, axes = plt.subplots(n_pops, n_zones, figsize=(8*n_zones, 5*n_pops))

        if n_pops == 1 and n_zones == 1:
            axes = np.array([[axes]])
        elif n_pops == 1:
            axes = axes.reshape(1, -1)
        elif n_zones == 1:
            axes = axes.reshape(-1, 1)

        for pop_idx, immune_pop in enumerate(immune_pops):
            pop_data = infiltration_df[infiltration_df['immune_population'] == immune_pop]

            for zone_idx, zone in enumerate(available_zones):
                ax = axes[pop_idx, zone_idx]
                zone_data = pop_data[pop_data['zone'] == zone]

                if len(zone_data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'{immune_pop} - {zone}', fontsize=12, fontweight='bold')
                    continue

                for group in groups:
                    group_data = zone_data[zone_data['group'] == group]

                    if len(group_data) == 0:
                        continue

                    # Aggregate by timepoint
                    summary = group_data.groupby('timepoint').agg({
                        'infiltration_density': ['mean', 'sem']
                    }).reset_index()
                    summary.columns = ['timepoint', 'mean', 'sem']

                    color = self.group_colors.get(group, '#000000')

                    # Plot line
                    ax.plot(summary['timepoint'], summary['mean'], '-o',
                           color=color, linewidth=2.5, markersize=10,
                           label=group, alpha=0.8)

                    # Error bars
                    ax.fill_between(summary['timepoint'],
                                   summary['mean'] - summary['sem'],
                                   summary['mean'] + summary['sem'],
                                   color=color, alpha=0.2)

                ax.set_xlabel(self.timepoint_label, fontsize=11, fontweight='bold')
                ax.set_ylabel('Infiltration Density', fontsize=11, fontweight='bold')
                ax.set_title(f'{immune_pop} - {zone}', fontsize=12, fontweight='bold')

                if pop_idx == 0 and zone_idx == 0:
                    ax.legend(frameon=True, loc='best', fontsize=10)

                ax.grid(True, alpha=0.3)

        plt.suptitle('Infiltration Over Time', fontsize=16, fontweight='bold')
        plt.tight_layout()

        plot_path = self.plots_dir / 'infiltration_timepoint_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {plot_path.name}")

    def plot_sample_level_summary(self, infiltration_df: pd.DataFrame):
        """
        Per-sample summary showing infiltration across all immune populations.

        Parameters
        ----------
        infiltration_df : pd.DataFrame
            Infiltration data
        """
        if infiltration_df is None or len(infiltration_df) == 0:
            return

        # Aggregate per sample
        sample_summary = infiltration_df.groupby(['sample_id', 'immune_population',
                                                  'group']).agg({
            'infiltration_density': 'mean',
            'count': 'sum'
        }).reset_index()

        samples = sorted(sample_summary['sample_id'].unique())

        # Limit to top samples by total infiltration
        sample_totals = sample_summary.groupby('sample_id')['count'].sum().sort_values(ascending=False)
        top_samples = sample_totals.head(16).index.tolist()

        immune_pops = sorted(sample_summary['immune_population'].unique())

        # Create grid: 4x4 samples
        n_cols = 4
        n_rows = int(np.ceil(len(top_samples) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = axes.flatten()

        for idx, sample in enumerate(top_samples):
            if idx >= len(axes):
                break

            ax = axes[idx]
            sample_data = sample_summary[sample_summary['sample_id'] == sample]

            # Pivot: immune populations x group
            pivot = sample_data.pivot_table(
                values='infiltration_density',
                index='immune_population',
                columns='group',
                aggfunc='mean'
            ).fillna(0)

            if len(pivot) > 0:
                # Bar chart
                pivot.plot(kind='bar', ax=ax, color=[self.group_colors.get(g, '#888888')
                                                     for g in pivot.columns],
                          alpha=0.8, width=0.8)

                group = sample_data['group'].iloc[0]
                ax.set_title(f'{sample}\n({group})', fontsize=11, fontweight='bold')
                ax.set_xlabel('', fontsize=9)
                ax.set_ylabel('Mean Density', fontsize=9)
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.legend(fontsize=8, loc='upper right')
                ax.grid(True, alpha=0.3, axis='y')

        # Hide unused subplots
        for idx in range(len(top_samples), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Per-Sample Infiltration Summary (Top 16 Samples)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        plot_path = self.plots_dir / 'per_sample_infiltration_summary.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {plot_path.name}")

    def generate_all_plots(self, results: Dict):
        """
        Generate all comprehensive infiltration visualization plots.

        Parameters
        ----------
        results : dict
            Results dictionary from InfiltrationAnalysisSpatialCells
        """
        print("\n  Generating comprehensive infiltration visualizations...")

        if 'infiltration' not in results or results['infiltration'] is None:
            print("    ⚠ No infiltration data found")
            return

        infiltration_df = results['infiltration']

        if len(infiltration_df) == 0:
            print("    ⚠ Infiltration data is empty")
            return

        # Generate all plot types
        self.plot_infiltration_by_zone_boxplots(infiltration_df)
        self.plot_infiltration_heatmaps(infiltration_df)
        self.plot_zone_gradient(infiltration_df)
        self.plot_per_tumor_infiltration_profiles(infiltration_df, top_n=20)
        self.plot_violin_plots_by_immune_population(infiltration_df)
        self.plot_bar_chart_total_infiltration(infiltration_df)
        self.plot_timepoint_comparison(infiltration_df)
        self.plot_sample_level_summary(infiltration_df)

        print(f"\n  ✓ All infiltration plots saved to {self.plots_dir}/")
