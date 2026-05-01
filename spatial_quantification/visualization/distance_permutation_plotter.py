"""Distance Permutation Testing Visualization."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class DistancePermutationPlotter:
    """Plots for distance permutation testing results."""

    def __init__(self, output_dir: Path, config: Dict):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.dpi = config.get('output', {}).get('dpi', 300)
        plotting = config.get('plotting', {})
        self.group_colors = plotting.get('group_colors', {})
        self.default_palette = sns.color_palette('colorblind', 10)
        sns.set_style('whitegrid')
        plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                             'savefig.dpi': self.dpi, 'figure.dpi': 150})

    def _get_group_color(self, group, idx=0):
        if group in self.group_colors:
            return self.group_colors[group]
        return self.default_palette[idx % len(self.default_palette)]

    def generate_all_plots(self, results: Dict):
        """Generate all distance permutation plots."""
        print("\n  Generating distance permutation plots...")
        for key in ['differential_tests', 'proximity_tests']:
            if key in results and not results[key].empty:
                df = results[key]
                self.plot_observed_vs_null(df, key)
                self.plot_volcano(df, key)
                self.plot_group_comparison(df, key)
        print("  Done.")

    def plot_observed_vs_null(self, df: pd.DataFrame, test_key: str):
        """Plot observed statistic vs null distribution summary per test."""
        for test_name in df['test_name'].unique():
            tdf = df[df['test_name'] == test_name].copy()
            n = len(tdf)
            if n == 0:
                continue

            fig, ax = plt.subplots(figsize=(max(8, n * 0.5), 5))
            x = np.arange(n)

            if 'observed_diff' in tdf.columns:
                obs_col = 'observed_diff'
            else:
                obs_col = 'observed_mean_dist'

            ax.bar(x, tdf['null_mean'].values, width=0.4, label='Null mean', color='gray', alpha=0.6)
            ax.errorbar(x, tdf['null_mean'].values, yerr=tdf['null_std'].values * 1.96,
                       fmt='none', color='gray', capsize=3)

            colors = ['#E41A1C' if sig else '#377EB8'
                      for sig in tdf.get('significant', tdf['p_value'] < 0.05)]
            ax.scatter(x, tdf[obs_col].values, c=colors, s=80, zorder=5, label='Observed', edgecolors='black')

            ax.set_xticks(x)
            ax.set_xticklabels(tdf['sample_id'].values, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(obs_col.replace('_', ' ').title())
            ax.set_title(f'{test_name} - Observed vs Null')
            ax.legend()
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'{test_name}_obs_vs_null.png', dpi=self.dpi, bbox_inches='tight')
            plt.close()

    def plot_volcano(self, df: pd.DataFrame, test_key: str):
        """Volcano plot: z-score vs -log10(p-value)."""
        for test_name in df['test_name'].unique():
            tdf = df[df['test_name'] == test_name].copy()
            if len(tdf) == 0:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            neg_log_p = -np.log10(tdf['p_value'].clip(lower=1e-10))
            sig = tdf.get('significant', tdf['p_value'] < 0.05).values

            ax.scatter(tdf.loc[~sig, 'z_score'], neg_log_p[~sig], c='#377EB8', alpha=0.6, label='NS')
            ax.scatter(tdf.loc[sig, 'z_score'], neg_log_p[sig], c='#E41A1C', alpha=0.8, label='Significant')
            ax.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Z-score')
            ax.set_ylabel('-log10(p-value)')
            ax.set_title(f'{test_name} - Volcano Plot')
            ax.legend()
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'{test_name}_volcano.png', dpi=self.dpi, bbox_inches='tight')
            plt.close()

    def plot_group_comparison(self, df: pd.DataFrame, test_key: str):
        """Violin plot of z-scores by group."""
        if 'group' not in df.columns:
            return
        for test_name in df['test_name'].unique():
            tdf = df[df['test_name'] == test_name].copy()
            groups = [g for g in tdf['group'].unique() if pd.notna(g) and g != '']
            if len(groups) < 2:
                continue

            fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.5), 5))
            palette = {g: self._get_group_color(g, i) for i, g in enumerate(groups)}
            sns.violinplot(data=tdf[tdf['group'].isin(groups)], x='group', y='z_score',
                          palette=palette, ax=ax, inner='box', cut=0)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(f'{test_name} - Z-scores by Group')
            ax.set_ylabel('Z-score')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'{test_name}_group_comparison.png', dpi=self.dpi, bbox_inches='tight')
            plt.close()
