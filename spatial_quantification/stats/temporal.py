"""
Temporal Analysis for Spatial Quantification
Analyze trends over time
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from .tests import StatisticalTests


class TemporalAnalysis:
    """Analyze temporal trends and patterns."""

    def __init__(self, config: Dict):
        """
        Initialize temporal analysis.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.tests = StatisticalTests(config)

    def test_temporal_trend(self, data: pd.DataFrame,
                           value_col: str,
                           group_col: str = None,
                           groups: List[str] = None) -> Dict:
        """
        Test for temporal trend.

        Parameters
        ----------
        data : pd.DataFrame
            Data with timepoint and value columns
        value_col : str
            Column containing values
        group_col : str, optional
            Column containing groups
        groups : list of str, optional
            Groups to analyze separately

        Returns
        -------
        dict
            Trend test results
        """
        if group_col is None or groups is None:
            # Overall trend
            return self.tests.temporal_trend_test(data, value_col)
        else:
            # Per-group trends
            results = {}
            for group in groups:
                data_group = data[data[group_col] == group]
                results[group] = self.tests.temporal_trend_test(data_group, value_col)

            return results

    def compare_temporal_trends(self, data: pd.DataFrame,
                               value_col: str,
                               group_col: str,
                               groups: List[str]) -> Dict:
        """
        Compare temporal trends between groups.

        Parameters
        ----------
        data : pd.DataFrame
            Data with timepoint, value, and group columns
        value_col : str
            Column containing values
        group_col : str
            Column containing groups
        groups : list of str
            Two groups to compare

        Returns
        -------
        dict
            Comparison of trends
        """
        # Get trends for each group
        results = {}

        for group in groups:
            data_group = data[data[group_col] == group]
            trend_result = self.tests.temporal_trend_test(data_group, value_col)
            results[f'{group}_trend'] = trend_result

        # Compare slopes (correlation coefficients)
        rho1 = results[f'{groups[0]}_trend']['rho']
        rho2 = results[f'{groups[1]}_trend']['rho']

        results['slope_difference'] = rho1 - rho2
        results['interpretation'] = self._interpret_slope_difference(rho1, rho2)

        return results

    def _interpret_slope_difference(self, rho1: float, rho2: float) -> str:
        """Interpret difference in slopes."""
        diff = abs(rho1 - rho2)

        if np.isnan(diff):
            return "Cannot compare (insufficient data)"

        if diff < 0.1:
            return "Similar trends"
        elif diff < 0.3:
            return "Moderately different trends"
        else:
            return "Substantially different trends"

    def calculate_rate_of_change(self, data: pd.DataFrame,
                                 value_col: str,
                                 sample_col: str = 'sample_id',
                                 timepoint_col: str = 'timepoint') -> pd.DataFrame:
        """
        Calculate rate of change over time for each sample.

        Parameters
        ----------
        data : pd.DataFrame
            Data with sample, timepoint, and value columns
        value_col : str
            Column containing values
        sample_col : str
            Column identifying samples
        timepoint_col : str
            Column containing timepoints

        Returns
        -------
        pd.DataFrame
            Rate of change per sample
        """
        rates = []

        for sample in data[sample_col].unique():
            data_sample = data[data[sample_col] == sample].sort_values(timepoint_col)

            if len(data_sample) < 2:
                continue

            timepoints = data_sample[timepoint_col].values
            values = data_sample[value_col].values

            # Calculate rate (linear regression slope)
            if len(timepoints) > 1:
                slope = np.polyfit(timepoints, values, 1)[0]
            else:
                slope = np.nan

            rates.append({
                sample_col: sample,
                'rate_of_change': slope,
                'n_timepoints': len(timepoints)
            })

        return pd.DataFrame(rates)
