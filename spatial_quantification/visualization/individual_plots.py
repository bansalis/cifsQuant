"""
Individual Plots for Spatial Quantification
Generate single plots for each metric/comparison
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from .styles import (setup_publication_style, setup_exploratory_style,
                    get_group_colors, add_significance_stars, format_p_value)


class IndividualPlots:
    """Generate individual plots for each analysis."""

    # Default colors for dynamic assignment
    DEFAULT_COLORS = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#984EA3',
                      '#A65628', '#F781BF', '#999999', '#66C2A5', '#FC8D62']

    def __init__(self, config: Dict, output_dir: Path):
        """
        Initialize individual plots.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        output_dir : Path
            Base output directory
        """
        self.config = config['visualization']
        self.full_config = config
        self.output_dir = Path(output_dir)
        self.formats = config['output'].get('formats', ['pdf', 'png'])
        self.dpi = config['output'].get('dpi', 300)

        # Set up styles
        if self.config.get('style') == 'publication':
            setup_publication_style()
        else:
            setup_exploratory_style()

        self.group_colors = get_group_colors()

    def _get_color(self, group: str, groups: List[str] = None) -> str:
        """Get color for a group, dynamically assigning if not predefined."""
        if group in self.group_colors:
            return self.group_colors[group]
        # Assign color based on position in groups list
        if groups is not None and group in groups:
            idx = groups.index(group) % len(self.DEFAULT_COLORS)
        else:
            idx = hash(group) % len(self.DEFAULT_COLORS)
        return self.DEFAULT_COLORS[idx]

    def plot_population_over_time(self, data: pd.DataFrame,
                                  population: str,
                                  value_col: str = 'count',
                                  group_col: str = 'group',
                                  groups: List[str] = None,
                                  stats: Optional[pd.DataFrame] = None,
                                  output_name: str = None):
        """
        Plot population over time for group comparison.

        Parameters
        ----------
        data : pd.DataFrame
            Data with timepoint, value, and group columns
        population : str
            Population name
        value_col : str
            Column containing values (count, fraction, density)
        group_col : str
            Column containing group labels
        groups : list of str
            Groups to plot
        stats : pd.DataFrame, optional
            Statistical test results with timepoint and p_value columns
        output_name : str, optional
            Custom output filename
        """
        # Auto-detect groups if not provided
        if groups is None:
            if group_col in data.columns:
                groups = sorted(data[group_col].dropna().unique().tolist())
            else:
                groups = []

        if not groups:
            print(f"  ⚠ No groups found for {population}, skipping plot")
            return

        # Create figure with two versions
        for style in ['exploratory', 'publication']:
            if style == 'exploratory':
                setup_exploratory_style()
                suffix = '_raw'
            else:
                setup_publication_style()
                suffix = '_publication'

            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot data for each group
            for group in groups:
                group_data = data[data[group_col] == group]

                if len(group_data) == 0:
                    continue

                # Calculate summary statistics per timepoint
                summary = group_data.groupby('timepoint')[value_col].agg(['mean', 'sem'])

                # Plot
                timepoints = summary.index.values
                means = summary['mean'].values
                sems = summary['sem'].values

                color = self._get_color(group, groups)

                # Line plot with error bars
                ax.errorbar(timepoints, means, yerr=sems,
                          label=group, color=color,
                          marker='o', markersize=8, linewidth=2.5,
                          capsize=5, capthick=2)

                # Add raw data points if exploratory
                if style == 'exploratory':
                    ax.scatter(group_data['timepoint'], group_data[value_col],
                             alpha=0.3, s=30, color=color)

            # Labels
            ylabel = value_col.replace('_', ' ').title()
            ax.set_xlabel('Timepoint (days)', fontsize=14)
            ax.set_ylabel(f'{population}\n{ylabel}', fontsize=14)
            ax.set_title(f'{population} Over Time', fontsize=16)

            # Legend
            ax.legend(frameon=False, loc='best')

            # Add statistics if provided
            if stats is not None and style == 'exploratory':
                self._add_stats_to_plot(ax, stats, groups)

            plt.tight_layout()

            # Save
            if output_name is None:
                output_name = f'{population}_{value_col}_over_time'

            self._save_figure(fig, f'{output_name}{suffix}')
            plt.close(fig)

    def plot_group_comparison_boxplot(self, data: pd.DataFrame,
                                     value_col: str,
                                     group_col: str = 'group',
                                     groups: List[str] = None,
                                     title: str = '',
                                     ylabel: str = '',
                                     stats: Optional[Dict] = None,
                                     output_name: str = None):
        """
        Create boxplot comparing two groups.

        Parameters
        ----------
        data : pd.DataFrame
            Data to plot
        value_col : str
            Column containing values
        group_col : str
            Column containing group labels
        groups : list of str
            Groups to compare
        title : str
            Plot title
        ylabel : str
            Y-axis label
        stats : dict, optional
            Statistical test results
        output_name : str
            Output filename
        """
        # Auto-detect groups if not provided
        if groups is None:
            if group_col in data.columns:
                groups = sorted(data[group_col].dropna().unique().tolist())
            else:
                groups = []

        if not groups:
            return

        for style in ['exploratory', 'publication']:
            if style == 'exploratory':
                setup_exploratory_style()
                suffix = '_raw'
            else:
                setup_publication_style()
                suffix = '_publication'

            fig, ax = plt.subplots(figsize=(6, 6))

            # Filter to groups
            plot_data = data[data[group_col].isin(groups)]

            if len(plot_data) == 0:
                plt.close(fig)
                continue

            # Create boxplot
            positions = range(len(groups))
            colors = [self._get_color(g, groups) for g in groups]

            box_data = [plot_data[plot_data[group_col] == g][value_col].dropna().values
                       for g in groups]

            bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                          patch_artist=True, showfliers=style=='exploratory')

            # Color boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add raw data points if exploratory
            if style == 'exploratory':
                for i, group in enumerate(groups):
                    group_data = plot_data[plot_data[group_col] == group][value_col].dropna()
                    x = np.random.normal(i, 0.04, size=len(group_data))
                    ax.scatter(x, group_data, alpha=0.4, s=30, color=colors[i])

            # Labels
            ax.set_xticks(positions)
            ax.set_xticklabels(groups)
            ax.set_ylabel(ylabel)
            ax.set_title(title)

            # Add statistics
            if stats is not None:
                p_value = stats.get('p_value', np.nan)
                if not np.isnan(p_value):
                    y_max = ax.get_ylim()[1]
                    add_significance_stars(ax, 0, 1, y_max * 0.95, p_value)

                    # Add p-value text if exploratory
                    if style == 'exploratory':
                        ax.text(0.5, 0.95, format_p_value(p_value),
                               transform=ax.transAxes,
                               ha='center', va='top', fontsize=10)

            plt.tight_layout()

            # Save
            self._save_figure(fig, f'{output_name}{suffix}')
            plt.close(fig)

    def plot_distance_distribution(self, data: pd.DataFrame,
                                   source: str,
                                   target: str,
                                   group_col: str = 'group',
                                   groups: List[str] = None,
                                   output_name: str = None):
        """Plot distance distribution between cell populations."""
        # Auto-detect groups if not provided
        if groups is None:
            if group_col in data.columns:
                groups = sorted(data[group_col].dropna().unique().tolist())
            else:
                groups = []

        if not groups:
            return

        for style in ['exploratory', 'publication']:
            if style == 'exploratory':
                setup_exploratory_style()
                suffix = '_raw'
            else:
                setup_publication_style()
                suffix = '_publication'

            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot histogram for each group
            for group in groups:
                group_data = data[data[group_col] == group]

                if len(group_data) == 0:
                    continue

                color = self._get_color(group, groups)

                # Plot histogram
                ax.hist(group_data['mean_distance'], bins=30, alpha=0.5,
                       label=group, color=color, density=True)

                # Add median line
                median = group_data['mean_distance'].median()
                ax.axvline(median, color=color, linestyle='--', linewidth=2,
                          label=f'{group} median')

            ax.set_xlabel('Distance (μm)', fontsize=14)
            ax.set_ylabel('Density', fontsize=14)
            ax.set_title(f'{source} to {target} Distance', fontsize=16)
            ax.legend(frameon=False)

            plt.tight_layout()

            if output_name is None:
                output_name = f'{source}_to_{target}_distance_dist'

            self._save_figure(fig, f'{output_name}{suffix}')
            plt.close(fig)

    def _add_stats_to_plot(self, ax, stats: pd.DataFrame, groups: List[str]):
        """Add statistical annotations to time series plot."""
        # Add significance stars at each timepoint
        if 'p_value' not in stats.columns:
            return

        y_max = ax.get_ylim()[1]

        for _, row in stats.iterrows():
            tp = row['timepoint']
            p_val = row['p_value']

            if pd.isna(p_val):
                continue

            # Determine significance
            if p_val < 0.05:
                if p_val < 0.001:
                    marker = '***'
                elif p_val < 0.01:
                    marker = '**'
                else:
                    marker = '*'

                # Add marker above plot
                ax.text(tp, y_max * 0.95, marker,
                       ha='center', va='bottom', fontsize=12)

    def _save_figure(self, fig, name: str):
        """Save figure in specified formats."""
        for fmt in self.formats:
            output_path = self.output_dir / f'{name}.{fmt}'
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
