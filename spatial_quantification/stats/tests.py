"""
Statistical Tests for Spatial Quantification
Comprehensive testing framework with multiple comparison correction
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr, kruskal, chi2_contingency
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class StatisticalTests:
    """Comprehensive statistical testing framework."""

    def __init__(self, config: Dict):
        """
        Initialize statistical tests.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config.get('statistics', {})
        self.alpha = self.config.get('alpha', 0.05)
        self.fdr_method = self.config.get('fdr_correction', 'benjamini_hochberg')

    def mann_whitney_u(self, group1: np.ndarray, group2: np.ndarray,
                       group1_name: str = 'Group1',
                       group2_name: str = 'Group2') -> Dict:
        """
        Perform Mann-Whitney U test (non-parametric).

        Parameters
        ----------
        group1, group2 : array-like
            Data for two groups
        group1_name, group2_name : str
            Names of groups for reporting

        Returns
        -------
        dict
            Test results with p-value, statistics, effect size
        """
        # Remove NaN values
        group1 = np.array(group1)[~np.isnan(group1)]
        group2 = np.array(group2)[~np.isnan(group2)]

        if len(group1) < 3 or len(group2) < 3:
            return {
                'test': 'Mann-Whitney U',
                'group1': group1_name,
                'group2': group2_name,
                'n1': len(group1),
                'n2': len(group2),
                'mean1': np.nan,
                'mean2': np.nan,
                'median1': np.nan,
                'median2': np.nan,
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'effect_size': np.nan,
                'error': 'Insufficient data'
            }

        # Perform test
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

        # Calculate effect size (rank biserial correlation)
        effect_size = self._rank_biserial(group1, group2)

        # Calculate descriptive statistics
        results = {
            'test': 'Mann-Whitney U',
            'group1': group1_name,
            'group2': group2_name,
            'n1': len(group1),
            'n2': len(group2),
            'mean1': np.mean(group1),
            'mean2': np.mean(group2),
            'median1': np.median(group1),
            'median2': np.median(group2),
            'std1': np.std(group1),
            'std2': np.std(group2),
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size': effect_size,
            'effect_size_type': 'rank_biserial'
        }

        return results

    def kruskal_wallis(self, *groups, group_names: List[str] = None) -> Dict:
        """
        Perform Kruskal-Wallis H-test (non-parametric ANOVA).

        Parameters
        ----------
        *groups : array-like
            Data for multiple groups
        group_names : list of str
            Names of groups

        Returns
        -------
        dict
            Test results
        """
        # Remove NaN values
        groups = [np.array(g)[~np.isnan(g)] for g in groups]

        if group_names is None:
            group_names = [f'Group{i+1}' for i in range(len(groups))]

        # Check sample sizes
        if any(len(g) < 3 for g in groups):
            return {
                'test': 'Kruskal-Wallis',
                'n_groups': len(groups),
                'group_names': group_names,
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'error': 'Insufficient data in one or more groups'
            }

        # Perform test
        statistic, p_value = kruskal(*groups)

        results = {
            'test': 'Kruskal-Wallis',
            'n_groups': len(groups),
            'group_names': group_names,
            'group_sizes': [len(g) for g in groups],
            'group_medians': [np.median(g) for g in groups],
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }

        return results

    def spearman_correlation(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Perform Spearman correlation test.

        Parameters
        ----------
        x, y : array-like
            Data for correlation

        Returns
        -------
        dict
            Correlation results
        """
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        if len(x) < 3:
            return {
                'test': 'Spearman correlation',
                'n': len(x),
                'rho': np.nan,
                'p_value': np.nan,
                'significant': False,
                'error': 'Insufficient data'
            }

        # Perform test
        rho, p_value = spearmanr(x, y)

        results = {
            'test': 'Spearman correlation',
            'n': len(x),
            'rho': rho,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }

        return results

    def multiple_testing_correction(self, p_values: List[float],
                                    method: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multiple testing correction.

        Parameters
        ----------
        p_values : list of float
            P-values to correct
        method : str, optional
            Correction method (default from config)

        Returns
        -------
        tuple
            (reject, p_adjusted) - boolean array and adjusted p-values
        """
        if method is None:
            method = self.fdr_method

        # Convert method names
        method_map = {
            'benjamini_hochberg': 'fdr_bh',
            'bonferroni': 'bonferroni',
            'holm': 'holm'
        }

        method_key = method_map.get(method, 'fdr_bh')

        # Apply correction
        reject, p_adjusted, _, _ = multipletests(
            p_values,
            alpha=self.alpha,
            method=method_key
        )

        return reject, p_adjusted

    def cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.

        Parameters
        ----------
        group1, group2 : array-like
            Data for two groups

        Returns
        -------
        float
            Cohen's d
        """
        # Remove NaN
        group1 = np.array(group1)[~np.isnan(group1)]
        group2 = np.array(group2)[~np.isnan(group2)]

        if len(group1) < 2 or len(group2) < 2:
            return np.nan

        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_std

        return d

    def _rank_biserial(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate rank biserial correlation (effect size for Mann-Whitney U).

        Parameters
        ----------
        group1, group2 : array-like
            Data for two groups

        Returns
        -------
        float
            Rank biserial correlation
        """
        n1, n2 = len(group1), len(group2)
        U, _ = mannwhitneyu(group1, group2)

        # Rank biserial correlation
        r = 1 - (2 * U) / (n1 * n2)

        return r

    def bootstrap_confidence_interval(self, data: np.ndarray,
                                     statistic_func=np.mean,
                                     n_bootstrap: int = 10000,
                                     ci_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.

        Parameters
        ----------
        data : array-like
            Data to bootstrap
        statistic_func : callable
            Function to calculate statistic (default: mean)
        n_bootstrap : int
            Number of bootstrap samples
        ci_level : float
            Confidence interval level (0-1)

        Returns
        -------
        tuple
            (lower_bound, upper_bound)
        """
        data = np.array(data)[~np.isnan(data)]

        if len(data) < 3:
            return np.nan, np.nan

        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(sample))

        # Calculate percentile confidence interval
        alpha = 1 - ci_level
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        return lower, upper

    def compare_groups_at_timepoint(self, data: pd.DataFrame,
                                   value_col: str,
                                   group_col: str,
                                   groups: List[str],
                                   timepoint: float) -> Dict:
        """
        Compare groups at a specific timepoint.

        Parameters
        ----------
        data : pd.DataFrame
            Data with value_col, group_col, timepoint columns
        value_col : str
            Column containing values to compare
        group_col : str
            Column containing group labels
        groups : list of str
            Two groups to compare
        timepoint : float
            Timepoint to analyze

        Returns
        -------
        dict
            Test results
        """
        # Filter to timepoint
        data_tp = data[data['timepoint'] == timepoint]

        # Get groups
        group1_data = data_tp[data_tp[group_col] == groups[0]][value_col].values
        group2_data = data_tp[data_tp[group_col] == groups[1]][value_col].values

        # Perform test
        results = self.mann_whitney_u(
            group1_data, group2_data,
            group1_name=groups[0], group2_name=groups[1]
        )

        results['timepoint'] = timepoint

        return results

    def temporal_trend_test(self, data: pd.DataFrame,
                           value_col: str,
                           timepoint_col: str = 'timepoint') -> Dict:
        """
        Test for temporal trend using Spearman correlation.

        Parameters
        ----------
        data : pd.DataFrame
            Data with value and timepoint columns
        value_col : str
            Column containing values
        timepoint_col : str
            Column containing timepoints

        Returns
        -------
        dict
            Trend test results
        """
        x = data[timepoint_col].values
        y = data[value_col].values

        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        if len(x) < 3:
            return {
                'test': 'Temporal trend (Spearman)',
                'n': len(x),
                'rho': np.nan,
                'p_value': np.nan,
                'significant': False,
                'error': 'Insufficient data'
            }

        rho, p_value = spearmanr(x, y)

        results = {
            'test': 'Temporal trend (Spearman)',
            'n': len(x),
            'rho': rho,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'direction': 'increasing' if rho > 0 else 'decreasing'
        }

        return results

    def save_test_results(self, results: Dict, output_path: str):
        """
        Save test results to CSV.

        Parameters
        ----------
        results : dict
            Test results
        output_path : str
            Path to save CSV
        """
        df = pd.DataFrame([results])
        df.to_csv(output_path, index=False)

    def save_multiple_results(self, results: List[Dict], output_path: str):
        """
        Save multiple test results to CSV.

        Parameters
        ----------
        results : list of dict
            List of test results
        output_path : str
            Path to save CSV
        """
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
