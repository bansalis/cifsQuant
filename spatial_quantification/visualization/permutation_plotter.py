"""
Permutation Testing Visualization

Publication-quality plots for spatial permutation testing analysis results.
Includes null distribution plots, effect size distributions, prevalence-effect
relationships, group comparisons, and significance matrices.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
from scipy import stats

try:
    from .plot_utils import detect_plot_type, calculate_statistics
    HAS_PLOT_UTILS = True
except ImportError:
    HAS_PLOT_UTILS = False

try:
    from .styles import setup_publication_style, get_colorblind_palette
    HAS_STYLES = True
except ImportError:
    HAS_STYLES = False


class PermutationPlotter:
    """
    Visualization for spatial permutation testing results.

    Generates:
    - Null distribution histograms per tumor
    - Effect size violin plots
    - Prevalence vs effect size scatter plots
    - Group comparison violin plots
    - Significance heatmaps
    - QQ plots for p-value distribution
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
        self.null_dist_dir = self.output_dir / 'null_distributions'

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.null_dist_dir.mkdir(parents=True, exist_ok=True)

        self.config = config

        # Get plotting settings
        plotting_config = config.get('plotting', {})
        self.dpi = config.get('output', {}).get('dpi', 300)
        self.font_family = plotting_config.get('font_family', 'DejaVu Sans')
        self.font_size = plotting_config.get('font_size', 11)

        # Colors
        self.group_colors = plotting_config.get('group_colors', {})
        self.default_palette = sns.color_palette('colorblind', 10)
        self.sig_color = '#E41A1C'  # Red for significant
        self.nonsig_color = '#377EB8'  # Blue for non-significant

        # Set style
        self._setup_style()

    def _setup_style(self):
        """Set up matplotlib style for publication-quality figures."""
        plt.rcParams.update({
            'font.family': self.font_family,
            'font.size': self.font_size,
            'axes.labelsize': self.font_size + 1,
            'axes.titlesize': self.font_size + 2,
            'xtick.labelsize': self.font_size - 1,
            'ytick.labelsize': self.font_size - 1,
            'legend.fontsize': self.font_size - 1,
            'figure.dpi': 150,
            'savefig.dpi': self.dpi,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 1.5,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2
        })
        sns.set_style('whitegrid')

    def _get_group_color(self, group: str, idx: int = 0) -> str:
        """Get color for a group."""
        if group in self.group_colors:
            return self.group_colors[group]
        return self.default_palette[idx % len(self.default_palette)]

    def plot_null_distributions(self, per_tumor_df: pd.DataFrame,
                               max_plots_per_figure: int = 16):
        """
        Plot null distribution histograms for individual tumors.

        Parameters
        ----------
        per_tumor_df : pd.DataFrame
            Per-tumor results with observed and null statistics
        max_plots_per_figure : int
            Maximum number of subplots per figure
        """
        if per_tumor_df is None or len(per_tumor_df) == 0:
            return

        print("    Generating null distribution plots...")

        # Group by test type
        for test_type in per_tumor_df['test_type'].unique():
            test_data = per_tumor_df[per_tumor_df['test_type'] == test_type]

            # Create batches of plots
            n_tumors = len(test_data)
            n_figures = (n_tumors + max_plots_per_figure - 1) // max_plots_per_figure

            for fig_idx in range(n_figures):
                start_idx = fig_idx * max_plots_per_figure
                end_idx = min(start_idx + max_plots_per_figure, n_tumors)
                batch = test_data.iloc[start_idx:end_idx]

                n_plots = len(batch)
                n_cols = min(4, n_plots)
                n_rows = (n_plots + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols,
                                        figsize=(4 * n_cols, 3.5 * n_rows))

                if n_plots == 1:
                    axes = np.array([[axes]])
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                elif n_cols == 1:
                    axes = axes.reshape(-1, 1)

                for plot_idx, (_, row) in enumerate(batch.iterrows()):
                    ax_row = plot_idx // n_cols
                    ax_col = plot_idx % n_cols
                    ax = axes[ax_row, ax_col]

                    # Generate synthetic null distribution for visualization
                    # (actual null values aren't stored, regenerate from mean/std)
                    null_mean = row['null_mean']
                    null_std = row['null_std']
                    observed = row['observed_statistic']
                    p_value = row['p_value']
                    significant = row.get('significant', p_value < 0.05)

                    if pd.isna(null_std) or null_std == 0:
                        ax.text(0.5, 0.5, 'Insufficient data',
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f"{row['sample_id'][:10]}/{row['tumor_id']}")
                        continue

                    # Simulate null distribution
                    null_values = np.random.normal(null_mean, null_std, 500)

                    # Plot histogram
                    color = self.sig_color if significant else self.nonsig_color
                    ax.hist(null_values, bins=30, alpha=0.7, color='gray',
                           edgecolor='white', label='Null')

                    # Plot observed value
                    ax.axvline(observed, color=color, linewidth=2.5,
                              linestyle='-', label=f'Observed')

                    # Annotate
                    z_score = row.get('z_score', (observed - null_mean) / null_std)
                    ax.text(0.95, 0.95, f'p={p_value:.3f}\nz={z_score:.2f}',
                           transform=ax.transAxes, ha='right', va='top',
                           fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    ax.set_xlabel('Statistic')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f"{row['sample_id'][:8]}/T{row['tumor_id']}",
                                fontsize=10)

                    if plot_idx == 0:
                        ax.legend(loc='upper left', fontsize=8)

                # Hide empty subplots
                for idx in range(n_plots, n_rows * n_cols):
                    ax_row = idx // n_cols
                    ax_col = idx % n_cols
                    axes[ax_row, ax_col].set_visible(False)

                plt.suptitle(f'Null Distributions - {test_type.title()} (Page {fig_idx + 1}/{n_figures})',
                            fontsize=14, fontweight='bold')
                plt.tight_layout()

                plot_path = self.null_dist_dir / f'null_dist_{test_type}_page{fig_idx + 1}.png'
                plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()

        print(f"      Saved null distribution plots to {self.null_dist_dir.name}/")

    def plot_effect_size_distribution(self, per_tumor_df: pd.DataFrame):
        """
        Plot effect size (z-score) distributions per sample.

        Parameters
        ----------
        per_tumor_df : pd.DataFrame
            Per-tumor results
        """
        if per_tumor_df is None or len(per_tumor_df) == 0:
            return

        print("    Generating effect size distribution plots...")

        # One figure per test type
        for test_type in per_tumor_df['test_type'].unique():
            test_data = per_tumor_df[per_tumor_df['test_type'] == test_type].copy()

            samples = sorted(test_data['sample_id'].unique())
            n_samples = len(samples)

            if n_samples == 0:
                continue

            # Create figure
            fig, ax = plt.subplots(figsize=(max(10, n_samples * 0.8), 6))

            # Prepare data for violin plot
            plot_data = []
            positions = []
            colors = []

            for i, sample in enumerate(samples):
                sample_data = test_data[test_data['sample_id'] == sample]
                z_scores = sample_data['z_score'].dropna().values

                if len(z_scores) > 0:
                    plot_data.append(z_scores)
                    positions.append(i)

                    # Color by group
                    group = sample_data['group'].iloc[0] if 'group' in sample_data.columns else ''
                    colors.append(self._get_group_color(group, i))

            if not plot_data:
                plt.close()
                continue

            # Violin plot
            parts = ax.violinplot(plot_data, positions=positions, showmeans=True,
                                 showextrema=True)

            # Color violins
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)

            # Style other parts
            for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
                if partname in parts:
                    parts[partname].set_color('black')
                    parts[partname].set_linewidth(1.5)

            # Add box plots overlaid
            bp = ax.boxplot(plot_data, positions=positions, widths=0.15,
                           patch_artist=True, showfliers=False)
            for patch in bp['boxes']:
                patch.set_facecolor('white')
                patch.set_alpha(0.8)

            # Reference line at z=0
            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

            # Labels
            ax.set_xticks(range(len(samples)))
            ax.set_xticklabels([s[:12] for s in samples], rotation=45, ha='right')
            ax.set_xlabel('Sample', fontweight='bold')
            ax.set_ylabel('Effect Size (z-score)', fontweight='bold')
            ax.set_title(f'Effect Size Distribution - {test_type.title()}',
                        fontsize=14, fontweight='bold')

            # Add legend for groups
            groups = test_data['group'].unique()
            if len(groups) > 1:
                handles = [mpatches.Patch(color=self._get_group_color(g, i), label=g)
                          for i, g in enumerate(groups)]
                ax.legend(handles=handles, loc='upper right')

            plt.tight_layout()

            plot_path = self.plots_dir / f'effect_size_distribution_{test_type}.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        print(f"      Saved effect size plots to {self.plots_dir.name}/")

    def plot_prevalence_effect_relationship(self, per_tumor_df: pd.DataFrame):
        """
        Plot relationship between marker prevalence and effect size.

        Parameters
        ----------
        per_tumor_df : pd.DataFrame
            Per-tumor results
        """
        if per_tumor_df is None or len(per_tumor_df) == 0:
            return

        print("    Generating prevalence-effect relationship plots...")

        for test_type in per_tumor_df['test_type'].unique():
            test_data = per_tumor_df[per_tumor_df['test_type'] == test_type].copy()

            if 'prevalence' not in test_data.columns:
                continue

            fig, ax = plt.subplots(figsize=(10, 7))

            # Color by significance
            sig_mask = test_data.get('significant', test_data['p_value'] < 0.05)

            # Non-significant points
            nonsig_data = test_data[~sig_mask]
            ax.scatter(nonsig_data['prevalence'] * 100, nonsig_data['z_score'],
                      c=self.nonsig_color, alpha=0.5, s=40, label='Non-significant',
                      edgecolors='white', linewidth=0.5)

            # Significant points
            sig_data = test_data[sig_mask]
            ax.scatter(sig_data['prevalence'] * 100, sig_data['z_score'],
                      c=self.sig_color, alpha=0.7, s=60, label='Significant (FDR < 0.05)',
                      edgecolors='white', linewidth=0.5)

            # Trend line with 95% CI
            x = test_data['prevalence'].values * 100
            y = test_data['z_score'].values
            mask = ~(np.isnan(x) | np.isnan(y))

            if mask.sum() > 10:
                x_clean = x[mask]
                y_clean = y[mask]

                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

                x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_line = slope * x_line + intercept

                ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.7,
                       label=f'Trend (r={r_value:.2f}, p={p_value:.3f})')

                # 95% CI
                n = len(x_clean)
                se = std_err * np.sqrt(1/n + (x_line - x_clean.mean())**2 / np.sum((x_clean - x_clean.mean())**2))
                ci = 1.96 * se

                ax.fill_between(x_line, y_line - ci, y_line + ci,
                               alpha=0.2, color='gray')

            # Reference line
            ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

            ax.set_xlabel('Marker Prevalence (%)', fontweight='bold')
            ax.set_ylabel('Effect Size (z-score)', fontweight='bold')
            ax.set_title(f'Prevalence vs Effect Size - {test_type.title()}',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best')

            plt.tight_layout()

            plot_path = self.plots_dir / f'prevalence_effect_{test_type}.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        print(f"      Saved prevalence-effect plots to {self.plots_dir.name}/")

    def plot_group_comparison(self, sample_summary_df: pd.DataFrame,
                             per_tumor_df: pd.DataFrame = None):
        """
        Plot group comparisons with violin plots and statistics.

        Parameters
        ----------
        sample_summary_df : pd.DataFrame
            Sample-level summary
        per_tumor_df : pd.DataFrame, optional
            Per-tumor results for detailed visualization
        """
        if sample_summary_df is None or len(sample_summary_df) == 0:
            return

        print("    Generating group comparison plots...")

        groups = sorted(sample_summary_df['group'].unique())
        if len(groups) < 2:
            print("      Insufficient groups for comparison")
            return

        for test_name in sample_summary_df['test_name'].unique():
            test_data = sample_summary_df[sample_summary_df['test_name'] == test_name]

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Plot 1: Effect size by group
            ax1 = axes[0]
            plot_data = []
            group_labels = []

            for i, group in enumerate(groups):
                group_data = test_data[test_data['group'] == group]['mean_effect_size'].dropna()
                if len(group_data) > 0:
                    plot_data.append(group_data.values)
                    group_labels.append(group)

            if plot_data:
                colors = [self._get_group_color(g, i) for i, g in enumerate(group_labels)]

                parts = ax1.violinplot(plot_data, showmeans=True, showextrema=True)
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.7)

                # Overlay box plots
                bp = ax1.boxplot(plot_data, widths=0.15, patch_artist=True, showfliers=False)
                for patch in bp['boxes']:
                    patch.set_facecolor('white')
                    patch.set_alpha(0.8)

                # Add individual points
                for i, data in enumerate(plot_data):
                    jitter = np.random.uniform(-0.1, 0.1, len(data))
                    ax1.scatter([i + 1 + jitter[j] for j in range(len(data))], data,
                               c=colors[i], alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

                ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax1.set_xticks(range(1, len(group_labels) + 1))
                ax1.set_xticklabels(group_labels)
                ax1.set_xlabel('Group', fontweight='bold')
                ax1.set_ylabel('Mean Effect Size (z-score)', fontweight='bold')
                ax1.set_title('Effect Size by Group', fontweight='bold')

                # Add statistics if 2 groups
                if len(plot_data) == 2 and len(plot_data[0]) >= 2 and len(plot_data[1]) >= 2:
                    stat, p_val = stats.mannwhitneyu(plot_data[0], plot_data[1])
                    sig_text = f'p = {p_val:.4f}'
                    if p_val < 0.05:
                        sig_text += ' *'
                    ax1.text(0.5, 0.95, sig_text, transform=ax1.transAxes,
                            ha='center', va='top', fontsize=11,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Plot 2: Percent significant by group
            ax2 = axes[1]
            pct_sig = []

            for group in groups:
                group_data = test_data[test_data['group'] == group]
                if len(group_data) > 0:
                    pct = group_data['pct_significant'].mean()
                    pct_sig.append(pct)
                else:
                    pct_sig.append(0)

            bars = ax2.bar(groups, pct_sig,
                          color=[self._get_group_color(g, i) for i, g in enumerate(groups)],
                          edgecolor='black', linewidth=1.5)

            # Add value labels on bars
            for bar, pct in zip(bars, pct_sig):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

            ax2.set_xlabel('Group', fontweight='bold')
            ax2.set_ylabel('% Tumors Significant', fontweight='bold')
            ax2.set_title('Significance Rate by Group', fontweight='bold')
            ax2.set_ylim(0, max(pct_sig) * 1.2 if max(pct_sig) > 0 else 10)

            plt.suptitle(f'Group Comparison: {test_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()

            # Clean filename
            clean_name = test_name.replace('/', '_').replace(' ', '_')
            plot_path = self.plots_dir / f'group_comparison_{clean_name}.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        print(f"      Saved group comparison plots to {self.plots_dir.name}/")

    def plot_significance_matrix(self, sample_summary_df: pd.DataFrame):
        """
        Plot heatmap of significance rates across samples and tests.

        Parameters
        ----------
        sample_summary_df : pd.DataFrame
            Sample-level summary
        """
        if sample_summary_df is None or len(sample_summary_df) == 0:
            return

        print("    Generating significance matrix heatmap...")

        # Pivot table: samples x tests
        pivot = sample_summary_df.pivot_table(
            values='pct_significant',
            index='sample_id',
            columns='test_name',
            aggfunc='mean'
        )

        if pivot.empty or pivot.shape[0] < 2:
            print("      Insufficient data for matrix")
            return

        # Create figure
        fig_height = max(8, len(pivot) * 0.4)
        fig_width = max(10, len(pivot.columns) * 1.2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Heatmap
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
                   cbar_kws={'label': '% Tumors Significant'},
                   ax=ax, vmin=0, vmax=100, linewidths=0.5)

        ax.set_xlabel('Test Configuration', fontweight='bold')
        ax.set_ylabel('Sample', fontweight='bold')
        ax.set_title('Significance Matrix: % Significant Tumors per Sample',
                    fontsize=14, fontweight='bold')

        # Rotate x labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        plot_path = self.plots_dir / 'significance_matrix.png'
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"      Saved significance matrix to {self.plots_dir.name}/")

    def plot_qq_pvalues(self, per_tumor_df: pd.DataFrame):
        """
        Plot QQ plot of p-values against uniform distribution.

        Useful for detecting systematic biases or global effects.

        Parameters
        ----------
        per_tumor_df : pd.DataFrame
            Per-tumor results
        """
        if per_tumor_df is None or len(per_tumor_df) == 0:
            return

        print("    Generating p-value QQ plot...")

        fig, axes = plt.subplots(1, len(per_tumor_df['test_type'].unique()),
                                figsize=(6 * len(per_tumor_df['test_type'].unique()), 5))

        if len(per_tumor_df['test_type'].unique()) == 1:
            axes = [axes]

        for ax, test_type in zip(axes, per_tumor_df['test_type'].unique()):
            test_data = per_tumor_df[per_tumor_df['test_type'] == test_type]
            p_values = test_data['p_value'].dropna().sort_values()

            n = len(p_values)
            if n < 10:
                ax.text(0.5, 0.5, 'Insufficient data',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{test_type.title()}')
                continue

            # Expected uniform quantiles
            expected = np.arange(1, n + 1) / (n + 1)

            # QQ plot
            ax.scatter(expected, p_values.values, alpha=0.6, s=30,
                      edgecolors='white', linewidth=0.5)

            # Reference line
            ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Expected (uniform)')

            # KS test
            ks_stat, ks_p = stats.kstest(p_values.values, 'uniform')
            ax.text(0.05, 0.95, f'KS stat = {ks_stat:.3f}\np = {ks_p:.3f}',
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel('Expected (Uniform)', fontweight='bold')
            ax.set_ylabel('Observed P-values', fontweight='bold')
            ax.set_title(f'P-value QQ Plot: {test_type.title()}', fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.legend(loc='lower right')

        plt.tight_layout()

        plot_path = self.plots_dir / 'pvalue_qq_plot.png'
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"      Saved QQ plot to {self.plots_dir.name}/")

    def plot_temporal_trends(self, sample_summary_df: pd.DataFrame):
        """
        Plot effect sizes and significance rates over time.

        Parameters
        ----------
        sample_summary_df : pd.DataFrame
            Sample-level summary with timepoint column
        """
        if sample_summary_df is None or len(sample_summary_df) == 0:
            return

        if 'timepoint' not in sample_summary_df.columns:
            return

        # Check if there are multiple timepoints
        timepoints = sample_summary_df['timepoint'].dropna().unique()
        if len(timepoints) < 2:
            return

        print("    Generating temporal trend plots...")

        groups = sorted(sample_summary_df['group'].unique())

        for test_name in sample_summary_df['test_name'].unique():
            test_data = sample_summary_df[sample_summary_df['test_name'] == test_name]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Plot 1: Effect size over time
            ax1 = axes[0]
            for i, group in enumerate(groups):
                group_data = test_data[test_data['group'] == group]

                if len(group_data) == 0:
                    continue

                # Aggregate by timepoint
                agg = group_data.groupby('timepoint').agg({
                    'mean_effect_size': ['mean', 'sem']
                }).reset_index()
                agg.columns = ['timepoint', 'mean', 'sem']

                color = self._get_group_color(group, i)

                # Individual points
                ax1.scatter(group_data['timepoint'], group_data['mean_effect_size'],
                           alpha=0.3, s=30, color=color, edgecolors='none')

                # Line with error bars
                ax1.errorbar(agg['timepoint'], agg['mean'], yerr=agg['sem'],
                            fmt='-o', color=color, linewidth=2, markersize=8,
                            capsize=4, label=group)

            ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax1.set_xlabel('Timepoint', fontweight='bold')
            ax1.set_ylabel('Mean Effect Size', fontweight='bold')
            ax1.set_title('Effect Size Over Time', fontweight='bold')
            ax1.legend()

            # Plot 2: % significant over time
            ax2 = axes[1]
            for i, group in enumerate(groups):
                group_data = test_data[test_data['group'] == group]

                if len(group_data) == 0:
                    continue

                agg = group_data.groupby('timepoint')['pct_significant'].mean().reset_index()
                color = self._get_group_color(group, i)

                ax2.plot(agg['timepoint'], agg['pct_significant'], '-o',
                        color=color, linewidth=2, markersize=8, label=group)

            ax2.set_xlabel('Timepoint', fontweight='bold')
            ax2.set_ylabel('% Significant Tumors', fontweight='bold')
            ax2.set_title('Significance Rate Over Time', fontweight='bold')
            ax2.legend()

            plt.suptitle(f'Temporal Trends: {test_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()

            clean_name = test_name.replace('/', '_').replace(' ', '_')
            plot_path = self.plots_dir / f'temporal_trend_{clean_name}.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        print(f"      Saved temporal trend plots to {self.plots_dir.name}/")

    def generate_all_plots(self, results: Dict):
        """
        Generate all permutation testing visualization plots.

        Parameters
        ----------
        results : dict
            Results dictionary from SpatialPermutationTesting
        """
        print("  Generating permutation testing plots...")

        # 1. Null distributions
        if 'per_tumor_results' in results:
            self.plot_null_distributions(results['per_tumor_results'])

        # 2. Effect size distributions
        if 'per_tumor_results' in results:
            self.plot_effect_size_distribution(results['per_tumor_results'])

        # 3. Prevalence-effect relationship
        if 'per_tumor_results' in results:
            self.plot_prevalence_effect_relationship(results['per_tumor_results'])

        # 4. Group comparisons
        if 'sample_summary' in results:
            self.plot_group_comparison(
                results['sample_summary'],
                results.get('per_tumor_results')
            )

        # 5. Significance matrix
        if 'sample_summary' in results:
            self.plot_significance_matrix(results['sample_summary'])

        # 6. QQ plot for p-values
        if 'per_tumor_results' in results:
            self.plot_qq_pvalues(results['per_tumor_results'])

        # 7. Temporal trends
        if 'sample_summary' in results:
            self.plot_temporal_trends(results['sample_summary'])

        print(f"  All permutation plots saved to {self.plots_dir}/")
