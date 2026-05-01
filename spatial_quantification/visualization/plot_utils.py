"""
Plotting Utilities
Shared utilities for adaptive plotting and statistical testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings


def detect_plot_type(data: pd.DataFrame, timepoint_col: str = 'timepoint') -> str:
    """
    Detect appropriate plot type based on number of timepoints.

    Parameters
    ----------
    data : pd.DataFrame
        Data to analyze
    timepoint_col : str
        Column name for timepoint

    Returns
    -------
    str
        'line' for multiple timepoints, 'box' for single timepoint
    """
    if timepoint_col not in data.columns:
        return 'box'

    n_timepoints = data[timepoint_col].nunique()

    if n_timepoints <= 1:
        return 'box'
    else:
        return 'line'


def calculate_statistics(data: pd.DataFrame,
                         value_col: str,
                         group_col: str = 'group',
                         timepoint_col: str = 'timepoint',
                         test_method: str = 'mannwhitneyu',
                         per_timepoint: bool = True) -> pd.DataFrame:
    """
    Calculate statistical comparisons between groups.

    Parameters
    ----------
    data : pd.DataFrame
        Data to analyze
    value_col : str
        Column with values to compare
    group_col : str
        Column defining groups to compare
    timepoint_col : str
        Column for timepoint (used if per_timepoint=True)
    test_method : str
        Statistical test to use ('mannwhitneyu', 'mann_whitney', 't-test', 'kruskal')
    per_timepoint : bool
        If True, perform tests per timepoint; otherwise across all data

    Returns
    -------
    pd.DataFrame
        Results with columns: group1, group2, timepoint (if per_timepoint),
        statistic, pvalue, significant
    """
    results = []
    groups = sorted(data[group_col].unique())

    if len(groups) < 2:
        return pd.DataFrame()

    # For now, focus on pairwise comparisons
    group1, group2 = groups[0], groups[1]

    if per_timepoint and timepoint_col in data.columns:
        # Test per timepoint
        for tp in sorted(data[timepoint_col].unique()):
            tp_data = data[data[timepoint_col] == tp]

            g1_values = tp_data[tp_data[group_col] == group1][value_col].dropna()
            g2_values = tp_data[tp_data[group_col] == group2][value_col].dropna()

            if len(g1_values) < 2 or len(g2_values) < 2:
                continue

            # Perform test (support both naming conventions)
            if test_method in ('mannwhitneyu', 'mann_whitney'):
                stat, pval = stats.mannwhitneyu(g1_values, g2_values, alternative='two-sided')
            elif test_method == 't-test':
                stat, pval = stats.ttest_ind(g1_values, g2_values)
            elif test_method == 'kruskal':
                stat, pval = stats.kruskal(g1_values, g2_values)
            else:
                raise ValueError(f"Unknown test method: {test_method}")

            results.append({
                'group1': group1,
                'group2': group2,
                'timepoint': tp,
                'statistic': stat,
                'pvalue': pval,
                'n1': len(g1_values),
                'n2': len(g2_values)
            })
    else:
        # Test across all data
        g1_values = data[data[group_col] == group1][value_col].dropna()
        g2_values = data[data[group_col] == group2][value_col].dropna()

        if len(g1_values) >= 2 and len(g2_values) >= 2:
            # Perform test (support both naming conventions)
            if test_method in ('mannwhitneyu', 'mann_whitney'):
                stat, pval = stats.mannwhitneyu(g1_values, g2_values, alternative='two-sided')
            elif test_method == 't-test':
                stat, pval = stats.ttest_ind(g1_values, g2_values)
            elif test_method == 'kruskal':
                stat, pval = stats.kruskal(g1_values, g2_values)
            else:
                raise ValueError(f"Unknown test method: {test_method}")

            results.append({
                'group1': group1,
                'group2': group2,
                'statistic': stat,
                'pvalue': pval,
                'n1': len(g1_values),
                'n2': len(g2_values)
            })

    if results:
        df = pd.DataFrame(results)
        # Add significance markers
        df['significant'] = df['pvalue'] < 0.05
        df['sig_symbol'] = df['pvalue'].apply(format_pvalue)
        return df
    else:
        return pd.DataFrame()


def format_pvalue(pval: float) -> str:
    """
    Format p-value as significance symbol.

    Parameters
    ----------
    pval : float
        P-value

    Returns
    -------
    str
        Significance symbol (*** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05)
    """
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return 'ns'


def plot_with_stats(data: pd.DataFrame,
                    value_col: str,
                    group_col: str = 'group',
                    timepoint_col: str = 'timepoint',
                    ax: plt.Axes = None,
                    group_colors: Dict = None,
                    title: str = '',
                    ylabel: str = '',
                    xlabel: str = '',
                    show_stats: bool = True,
                    test_method: str = 'mannwhitneyu') -> plt.Axes:
    """
    Create adaptive plot with optional statistical annotations.

    Automatically detects whether to use line plot (multiple timepoints) or
    box/column plot (single timepoint).

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    value_col : str
        Column with values to plot
    group_col : str
        Column defining groups
    timepoint_col : str
        Column for timepoint
    ax : plt.Axes, optional
        Axes to plot on (creates new if None)
    group_colors : dict, optional
        Colors for each group
    title : str
        Plot title
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    show_stats : bool
        Whether to show statistical annotations
    test_method : str
        Statistical test method

    Returns
    -------
    plt.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if group_colors is None:
        # Dynamic color assignment based on groups in data
        default_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#984EA3', '#A65628']
        groups = sorted(data[group_col].unique())
        group_colors = {g: default_colors[i % len(default_colors)] for i, g in enumerate(groups)}

    plot_type = detect_plot_type(data, timepoint_col)
    groups = sorted(data[group_col].unique())

    if plot_type == 'line':
        # Multiple timepoints - use line plot
        for group in groups:
            group_data = data[data[group_col] == group]

            if len(group_data) == 0:
                continue

            # Aggregate by timepoint
            summary = group_data.groupby(timepoint_col)[value_col].agg(['mean', 'sem']).reset_index()

            color = group_colors.get(group, '#000000')

            ax.plot(summary[timepoint_col], summary['mean'],
                   '-o', color=color, linewidth=3.0, markersize=10,
                   label=group, zorder=10)
            ax.fill_between(summary[timepoint_col],
                           summary['mean'] - summary['sem'],
                           summary['mean'] + summary['sem'],
                           alpha=0.2, color=color)

        # Add statistical annotations for line plots
        if show_stats and len(groups) == 2:
            stats_df = calculate_statistics(data, value_col, group_col, timepoint_col,
                                           test_method, per_timepoint=True)

            if len(stats_df) > 0:
                # Add significance markers at top of plot
                ymax = ax.get_ylim()[1]
                for _, row in stats_df.iterrows():
                    if row['pvalue'] < 0.05:
                        tp = row['timepoint']
                        ax.text(tp, ymax * 0.95, row['sig_symbol'],
                               ha='center', va='top', fontsize=12, fontweight='bold')

        if not xlabel:
            xlabel = 'Timepoint'
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')

    else:
        # Single timepoint - use box/column plot
        # Create side-by-side box plots
        positions = []
        box_data = []
        labels = []
        colors = []

        for idx, group in enumerate(groups):
            group_data = data[data[group_col] == group][value_col].dropna()
            if len(group_data) > 0:
                box_data.append(group_data)
                positions.append(idx)
                labels.append(group)
                colors.append(group_colors.get(group, '#000000'))

        # Create box plots with narrower boxes and bolder lines
        bp = ax.boxplot(box_data, positions=positions, widths=0.5,
                       patch_artist=True, showfliers=True,
                       boxprops=dict(linewidth=2.5),
                       whiskerprops=dict(linewidth=2.5),
                       capprops=dict(linewidth=2.5),
                       medianprops=dict(linewidth=3, color='black'),
                       flierprops=dict(marker='o', markersize=6, alpha=0.5))

        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')

        # Add individual points
        for idx, (group, color) in enumerate(zip(labels, colors)):
            group_data = data[data[group_col] == group][value_col].dropna()
            # Add jitter
            x = np.random.normal(idx, 0.04, size=len(group_data))
            ax.scatter(x, group_data, alpha=0.4, s=30, color=color, zorder=5)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=13)

        # Add statistical annotations for box plots
        if show_stats and len(groups) == 2:
            stats_df = calculate_statistics(data, value_col, group_col, timepoint_col,
                                           test_method, per_timepoint=False)

            if len(stats_df) > 0 and len(stats_df) == 1:
                row = stats_df.iloc[0]
                # Add significance bracket
                y_max = max([d.max() for d in box_data])
                y_range = y_max - min([d.min() for d in box_data])
                y_bracket = y_max + y_range * 0.05
                h = y_range * 0.02

                ax.plot([0, 0, 1, 1], [y_bracket, y_bracket+h, y_bracket+h, y_bracket],
                       lw=2.5, c='black')
                ax.text(0.5, y_bracket+h, row['sig_symbol'],
                       ha='center', va='bottom', fontsize=14, fontweight='bold')

        if not xlabel:
            xlabel = ''  # No label for single timepoint boxplots
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')

    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(frameon=True, loc='best', fontsize=12)
    ax.grid(False)  # No grid for clean publication look

    # Make tick labels bigger
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Increase spine thickness
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    return ax


def plot_with_stats_clean(data: pd.DataFrame,
                         value_col: str,
                         group_col: str = 'group',
                         timepoint_col: str = 'timepoint',
                         ax: plt.Axes = None,
                         group_colors: Dict = None,
                         title: str = '',
                         ylabel: str = '',
                         xlabel: str = '',
                         show_stats: bool = True,
                         test_method: str = 'mannwhitneyu') -> plt.Axes:
    """
    Create clean plot with stats - boxplot/line but NO raw data overlay.
    This is the publication-ready "clean stats" version.

    Parameters are same as plot_with_stats but this version excludes raw data points.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if group_colors is None:
        # Dynamic color assignment based on groups in data
        default_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#984EA3', '#A65628']
        groups = sorted(data[group_col].unique())
        group_colors = {g: default_colors[i % len(default_colors)] for i, g in enumerate(groups)}

    plot_type = detect_plot_type(data, timepoint_col)
    groups = sorted(data[group_col].unique())

    if plot_type == 'line':
        # Same as regular line plot (no raw data to remove)
        for group in groups:
            group_data = data[data[group_col] == group]

            if len(group_data) == 0:
                continue

            summary = group_data.groupby(timepoint_col)[value_col].agg(['mean', 'sem']).reset_index()
            color = group_colors.get(group, '#000000')

            ax.plot(summary[timepoint_col], summary['mean'],
                   '-o', color=color, linewidth=3.0, markersize=10,
                   label=group, zorder=10)
            ax.fill_between(summary[timepoint_col],
                           summary['mean'] - summary['sem'],
                           summary['mean'] + summary['sem'],
                           alpha=0.2, color=color)

        if show_stats and len(groups) == 2:
            stats_df = calculate_statistics(data, value_col, group_col, timepoint_col,
                                           test_method, per_timepoint=True)

            if len(stats_df) > 0:
                ymax = ax.get_ylim()[1]
                for _, row in stats_df.iterrows():
                    if row['pvalue'] < 0.05:
                        tp = row['timepoint']
                        ax.text(tp, ymax * 0.95, row['sig_symbol'],
                               ha='center', va='top', fontsize=12, fontweight='bold')

        if not xlabel:
            xlabel = 'Timepoint'
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')

    else:
        # Boxplot without scatter points
        positions = []
        box_data = []
        labels = []
        colors = []

        for idx, group in enumerate(groups):
            group_data = data[data[group_col] == group][value_col].dropna()
            if len(group_data) > 0:
                box_data.append(group_data)
                positions.append(idx)
                labels.append(group)
                colors.append(group_colors.get(group, '#000000'))

        # Create box plots - NO scatter points added
        bp = ax.boxplot(box_data, positions=positions, widths=0.5,
                       patch_artist=True, showfliers=True,
                       boxprops=dict(linewidth=2.5),
                       whiskerprops=dict(linewidth=2.5),
                       capprops=dict(linewidth=2.5),
                       medianprops=dict(linewidth=3, color='black'),
                       flierprops=dict(marker='o', markersize=6, alpha=0.5))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=13)

        if show_stats and len(groups) == 2:
            stats_df = calculate_statistics(data, value_col, group_col, timepoint_col,
                                           test_method, per_timepoint=False)

            if len(stats_df) > 0 and len(stats_df) == 1:
                row = stats_df.iloc[0]
                y_max = max([d.max() for d in box_data])
                y_range = y_max - min([d.min() for d in box_data])
                y_bracket = y_max + y_range * 0.05
                h = y_range * 0.02

                ax.plot([0, 0, 1, 1], [y_bracket, y_bracket+h, y_bracket+h, y_bracket],
                       lw=2.5, c='black')
                ax.text(0.5, y_bracket+h, row['sig_symbol'],
                       ha='center', va='bottom', fontsize=14, fontweight='bold')

        if not xlabel:
            xlabel = ''
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')

    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(frameon=True, loc='best', fontsize=12)
    ax.grid(False)

    ax.tick_params(axis='both', which='major', labelsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    return ax


def create_dual_plots(data: pd.DataFrame,
                     value_col: str,
                     group_col: str = 'group',
                     timepoint_col: str = 'timepoint',
                     group_colors: Dict = None,
                     title_base: str = '',
                     ylabel: str = '',
                     xlabel: str = '',
                     test_method: str = 'mannwhitneyu',
                     output_path_base: str = None) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    Create THREE versions of the same plot:
    1. With statistics and raw data overlay
    2. Without statistics
    3. Clean stats version (stats but NO raw data overlay) - publication ready

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    value_col : str
        Column with values to plot
    group_col : str
        Column defining groups
    timepoint_col : str
        Column for timepoint
    group_colors : dict, optional
        Colors for each group
    title_base : str
        Base title for plots
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    test_method : str
        Statistical test method
    output_path_base : str, optional
        Base path for saving (will append _with_stats.png, _no_stats.png, _clean_stats.png)

    Returns
    -------
    tuple
        (fig_with_stats, fig_no_stats, fig_clean_stats)
    """
    # Plot with stats
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot_with_stats(data, value_col, group_col, timepoint_col, ax1,
                   group_colors, title_base, ylabel, xlabel,
                   show_stats=True, test_method=test_method)
    plt.tight_layout()

    if output_path_base:
        fig1.savefig(f"{output_path_base}_with_stats.png", dpi=300, bbox_inches='tight')

    # Plot without stats
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_with_stats(data, value_col, group_col, timepoint_col, ax2,
                   group_colors, title_base, ylabel, xlabel,
                   show_stats=False, test_method=test_method)
    plt.tight_layout()

    if output_path_base:
        fig2.savefig(f"{output_path_base}_no_stats.png", dpi=300, bbox_inches='tight')

    # Plot with stats but NO raw data overlay (clean stats version)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    plot_with_stats_clean(data, value_col, group_col, timepoint_col, ax3,
                         group_colors, title_base, ylabel, xlabel,
                         show_stats=True, test_method=test_method)
    plt.tight_layout()

    if output_path_base:
        fig3.savefig(f"{output_path_base}_clean_stats.png", dpi=300, bbox_inches='tight')

    if output_path_base:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

    return fig1, fig2, fig3
