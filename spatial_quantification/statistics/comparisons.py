"""
Group Comparisons for Spatial Quantification
High-level wrapper for comparing groups
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from .tests import StatisticalTests


class GroupComparison:
    """Compare groups (KPT vs KPNT, etc.) across metrics."""

    def __init__(self, config: Dict):
        """
        Initialize group comparison.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.tests = StatisticalTests(config)

    def compare_two_groups(self, data: pd.DataFrame,
                          value_col: str,
                          group_col: str,
                          groups: List[str],
                          stratify_by: str = None) -> Dict:
        """
        Compare two groups.

        Parameters
        ----------
        data : pd.DataFrame
            Data to compare
        value_col : str
            Column containing values
        group_col : str
            Column containing group labels
        groups : list of str
            Two groups to compare
        stratify_by : str, optional
            Column to stratify by (e.g., timepoint)

        Returns
        -------
        dict
            Comparison results
        """
        if stratify_by is None:
            # Simple comparison
            group1_data = data[data[group_col] == groups[0]][value_col].values
            group2_data = data[data[group_col] == groups[1]][value_col].values

            return self.tests.mann_whitney_u(
                group1_data, group2_data,
                group1_name=groups[0],
                group2_name=groups[1]
            )
        else:
            # Stratified comparison
            results = []
            for stratum in data[stratify_by].dropna().unique():
                data_stratum = data[data[stratify_by] == stratum]

                group1_data = data_stratum[data_stratum[group_col] == groups[0]][value_col].values
                group2_data = data_stratum[data_stratum[group_col] == groups[1]][value_col].values

                result = self.tests.mann_whitney_u(
                    group1_data, group2_data,
                    group1_name=groups[0],
                    group2_name=groups[1]
                )
                result[stratify_by] = stratum
                results.append(result)

            return results

    def compare_at_each_timepoint(self, data: pd.DataFrame,
                                 value_col: str,
                                 group_col: str,
                                 groups: List[str],
                                 timepoints: List[float]) -> pd.DataFrame:
        """
        Compare groups at each timepoint.

        Parameters
        ----------
        data : pd.DataFrame
            Data with timepoint column
        value_col : str
            Column containing values
        group_col : str
            Column containing group labels
        groups : list of str
            Two groups to compare
        timepoints : list of float
            Timepoints to compare

        Returns
        -------
        pd.DataFrame
            Results for each timepoint
        """
        results = []

        for tp in timepoints:
            result = self.tests.compare_groups_at_timepoint(
                data, value_col, group_col, groups, tp
            )
            results.append(result)

        df_results = pd.DataFrame(results)

        # Apply multiple testing correction
        if len(df_results) > 1:
            p_values = df_results['p_value'].values
            reject, p_adj = self.tests.multiple_testing_correction(p_values)

            df_results['p_adjusted'] = p_adj
            df_results['significant_fdr'] = reject

        return df_results
