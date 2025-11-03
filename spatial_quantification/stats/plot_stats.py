"""
Statistical Testing for Plots
Helper functions to add significance bars and annotations to plots
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def perform_pairwise_tests(data: pd.DataFrame,
                           value_col: str,
                           group_col: str,
                           groups: List[str],
                           test: str = 'mann_whitney',
                           fdr_correction: bool = True) -> pd.DataFrame:
    """
    Perform pairwise statistical tests between groups.

    Parameters
    ----------
    data : pd.DataFrame
        Data with values and group labels
    value_col : str
        Column name for values to compare
    group_col : str
        Column name for group labels
    groups : List[str]
        List of groups to compare
    test : str
        Statistical test: 'mann_whitney', 't_test', or 'kruskal'
    fdr_correction : bool
        Apply FDR correction to p-values

    Returns
    -------
    pd.DataFrame
        Results with columns: group1, group2, pvalue, pvalue_adj, significant
    """
    results = []

    # Pairwise comparisons
    for i, group1 in enumerate(groups):
        for group2 in groups[i+1:]:
            data1 = data[data[group_col] == group1][value_col].dropna()
            data2 = data[data[group_col] == group2][value_col].dropna()

            if len(data1) < 3 or len(data2) < 3:
                continue

            # Perform test
            if test == 'mann_whitney':
                statistic, pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            elif test == 't_test':
                statistic, pvalue = stats.ttest_ind(data1, data2)
            elif test == 'kruskal':
                statistic, pvalue = stats.kruskal(data1, data2)
            else:
                raise ValueError(f"Unknown test: {test}")

            results.append({
                'group1': group1,
                'group2': group2,
                'statistic': statistic,
                'pvalue': pvalue
            })

    if len(results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # FDR correction
    if fdr_correction and len(results_df) > 0:
        _, pvals_adj, _, _ = multipletests(results_df['pvalue'], method='fdr_bh')
        results_df['pvalue_adj'] = pvals_adj
    else:
        results_df['pvalue_adj'] = results_df['pvalue']

    # Mark significant
    results_df['significant'] = results_df['pvalue_adj'] < 0.05

    return results_df


def get_significance_symbol(pvalue: float,
                            symbols: Dict[float, str] = None) -> str:
    """
    Get significance symbol for p-value.

    Parameters
    ----------
    pvalue : float
        P-value
    symbols : Dict[float, str], optional
        Custom symbol mapping (threshold: symbol)

    Returns
    -------
    str
        Significance symbol
    """
    if symbols is None:
        symbols = {
            0.001: '***',
            0.01: '**',
            0.05: '*',
            1.0: 'ns'
        }

    for threshold, symbol in sorted(symbols.items()):
        if pvalue < threshold:
            return symbol

    return 'ns'


def add_significance_bars(ax: plt.Axes,
                         data: pd.DataFrame,
                         value_col: str,
                         group_col: str,
                         groups: List[str],
                         x_positions: np.ndarray,
                         test: str = 'mann_whitney',
                         fdr_correction: bool = True,
                         symbols: Dict[float, str] = None,
                         bar_height: float = 0.05,
                         text_offset: float = 0.02,
                         only_significant: bool = True) -> None:
    """
    Add significance bars with symbols to plot.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    data : pd.DataFrame
        Data for statistical testing
    value_col : str
        Column with values
    group_col : str
        Column with group labels
    groups : List[str]
        Groups to compare
    x_positions : np.ndarray
        X-axis positions for groups
    test : str
        Statistical test to use
    fdr_correction : bool
        Apply FDR correction
    symbols : Dict[float, str], optional
        Significance symbol mapping
    bar_height : float
        Height of significance bar as fraction of y-range
    text_offset : float
        Text offset above bar as fraction of y-range
    only_significant : bool
        Only show significant comparisons
    """
    # Perform tests
    test_results = perform_pairwise_tests(
        data, value_col, group_col, groups, test, fdr_correction
    )

    if len(test_results) == 0:
        return

    # Filter to significant only if requested
    if only_significant:
        test_results = test_results[test_results['significant']]

    if len(test_results) == 0:
        return

    # Get y-axis limits
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Get max y value in data
    max_y = data[value_col].max()

    # Calculate bar heights
    bar_increment = y_range * bar_height
    current_height = max_y + y_range * 0.05

    # Group position mapping
    group_to_x = {group: x_positions[i] for i, group in enumerate(groups)}

    # Draw bars
    for idx, row in test_results.iterrows():
        group1 = row['group1']
        group2 = row['group2']
        pvalue = row['pvalue_adj']

        # Get x positions
        x1 = group_to_x.get(group1)
        x2 = group_to_x.get(group2)

        if x1 is None or x2 is None:
            continue

        # Get significance symbol
        symbol = get_significance_symbol(pvalue, symbols)

        if symbol == 'ns' and only_significant:
            continue

        # Draw horizontal bar
        ax.plot([x1, x1, x2, x2],
               [current_height, current_height + bar_increment,
                current_height + bar_increment, current_height],
               'k-', linewidth=1.0, zorder=100)

        # Add text
        mid_x = (x1 + x2) / 2
        text_y = current_height + bar_increment + y_range * text_offset
        ax.text(mid_x, text_y, symbol, ha='center', va='bottom',
               fontsize=10, fontweight='bold', zorder=101)

        # Increment height for next bar
        current_height += bar_increment + y_range * 0.08

    # Adjust y-axis to fit bars
    new_y_max = current_height + y_range * 0.05
    ax.set_ylim(y_min, new_y_max)


def add_significance_to_boxplot(ax: plt.Axes,
                                data: pd.DataFrame,
                                value_col: str,
                                group_col: str,
                                x_col: str,
                                groups: List[str],
                                x_values: List,
                                box_width: float = 0.8,
                                test: str = 'mann_whitney',
                                fdr_correction: bool = True,
                                symbols: Dict[float, str] = None) -> None:
    """
    Add significance bars to box plots with multiple groups per x position.

    Useful for plots with groups on x-axis and multiple categories within each.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    data : pd.DataFrame
        Data for testing
    value_col : str
        Column with values
    group_col : str
        Column with group labels (e.g., 'KPT', 'KPNT')
    x_col : str
        Column with x-axis categories (e.g., 'timepoint')
    groups : List[str]
        Groups to compare within each x category
    x_values : List
        X-axis categories
    box_width : float
        Width of boxes (for positioning bars)
    test : str
        Statistical test
    fdr_correction : bool
        Apply FDR correction
    symbols : Dict[float, str], optional
        Significance symbols
    """
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    for x_idx, x_val in enumerate(x_values):
        # Get data for this x position
        x_data = data[data[x_col] == x_val]

        if len(x_data) < 2:
            continue

        # Perform tests
        test_results = perform_pairwise_tests(
            x_data, value_col, group_col, groups, test, fdr_correction
        )

        if len(test_results) == 0:
            continue

        # Only significant
        test_results = test_results[test_results['significant']]

        if len(test_results) == 0:
            continue

        # Get max y for this x position
        max_y = x_data[value_col].max()

        # Calculate positions (assuming groups are side-by-side)
        n_groups = len(groups)
        spacing = box_width / n_groups
        base_x = x_idx

        for idx, row in test_results.iterrows():
            group1 = row['group1']
            group2 = row['group2']
            pvalue = row['pvalue_adj']

            # Calculate x positions for groups
            idx1 = groups.index(group1) if group1 in groups else None
            idx2 = groups.index(group2) if group2 in groups else None

            if idx1 is None or idx2 is None:
                continue

            x1 = base_x + (idx1 - n_groups/2 + 0.5) * spacing
            x2 = base_x + (idx2 - n_groups/2 + 0.5) * spacing

            # Get symbol
            symbol = get_significance_symbol(pvalue, symbols)

            if symbol == 'ns':
                continue

            # Draw bar
            bar_y = max_y + y_range * 0.05
            ax.plot([x1, x2], [bar_y, bar_y], 'k-', linewidth=1.0, zorder=100)

            # Add text
            mid_x = (x1 + x2) / 2
            ax.text(mid_x, bar_y + y_range * 0.02, symbol,
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   zorder=101)


def add_compact_significance(ax: plt.Axes,
                             data: pd.DataFrame,
                             value_col: str,
                             group_col: str,
                             groups: List[str],
                             x_position: float,
                             test: str = 'mann_whitney',
                             fdr_correction: bool = True,
                             symbols: Dict[float, str] = None) -> None:
    """
    Add compact significance annotation (single symbol at position).

    Useful when space is limited - just shows symbol above data.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    data : pd.DataFrame
        Data for testing
    value_col : str
        Value column
    group_col : str
        Group column
    groups : List[str]
        Groups to compare (typically 2)
    x_position : float
        X position for annotation
    test : str
        Statistical test
    fdr_correction : bool
        Apply FDR correction
    symbols : Dict[float, str], optional
        Significance symbols
    """
    # Perform test (only works for 2 groups)
    if len(groups) != 2:
        return

    test_results = perform_pairwise_tests(
        data, value_col, group_col, groups, test, fdr_correction
    )

    if len(test_results) == 0 or not test_results.iloc[0]['significant']:
        return

    pvalue = test_results.iloc[0]['pvalue_adj']
    symbol = get_significance_symbol(pvalue, symbols)

    if symbol == 'ns':
        return

    # Get max y
    max_y = data[value_col].max()
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Add symbol
    ax.text(x_position, max_y + y_range * 0.05, symbol,
           ha='center', va='bottom', fontsize=11, fontweight='bold',
           zorder=101)
