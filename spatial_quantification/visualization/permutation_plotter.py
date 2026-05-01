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

        # Group by test name (each analysis gets its own set of pages)
        for test_name in per_tumor_df['test_name'].unique():
            test_data = per_tumor_df[per_tumor_df['test_name'] == test_name]
            test_type = test_data['test_type'].iloc[0]

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
                    observed = row['observed']
                    p_value = row['p_value']
                    significant = row.get('significant', p_value < 0.05)

                    if pd.isna(null_std) or null_std == 0:
                        ax.text(0.5, 0.5, 'Insufficient data',
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f"{row['sample_id'][:10]}/{row['structure_id']}")
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
                    ax.set_title(f"{row['sample_id'][:8]}/T{row['structure_id']}",
                                fontsize=10)

                    if plot_idx == 0:
                        ax.legend(loc='upper left', fontsize=8)

                # Hide empty subplots
                for idx in range(n_plots, n_rows * n_cols):
                    ax_row = idx // n_cols
                    ax_col = idx % n_cols
                    axes[ax_row, ax_col].set_visible(False)

                clean_name = test_name.replace('/', '_').replace(' ', '_')
                plt.suptitle(f'Null Distributions - {test_name} ({test_type}) (Page {fig_idx + 1}/{n_figures})',
                            fontsize=14, fontweight='bold')
                plt.tight_layout()

                plot_path = self.null_dist_dir / f'null_dist_{clean_name}_page{fig_idx + 1}.png'
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

        # One figure per test name
        for test_name in per_tumor_df['test_name'].unique():
            test_data = per_tumor_df[per_tumor_df['test_name'] == test_name].copy()
            test_type = test_data['test_type'].iloc[0]

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
            ax.set_title(f'Effect Size Distribution - {test_name} ({test_type})',
                        fontsize=14, fontweight='bold')

            # Add legend for groups
            groups = test_data['group'].unique()
            if len(groups) > 1:
                handles = [mpatches.Patch(color=self._get_group_color(g, i), label=g)
                          for i, g in enumerate(groups)]
                ax.legend(handles=handles, loc='upper right')

            plt.tight_layout()

            clean_name = test_name.replace('/', '_').replace(' ', '_')
            plot_path = self.plots_dir / f'effect_size_distribution_{clean_name}.png'
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

        for test_name in per_tumor_df['test_name'].unique():
            test_data = per_tumor_df[per_tumor_df['test_name'] == test_name].copy()
            test_type = test_data['test_type'].iloc[0]

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
            ax.set_title(f'Prevalence vs Effect Size - {test_name} ({test_type})',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best')

            plt.tight_layout()

            clean_name = test_name.replace('/', '_').replace(' ', '_')
            plot_path = self.plots_dir / f'prevalence_effect_{clean_name}.png'
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
                group_data = test_data[test_data['group'] == group]['mean_z_score'].dropna()
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

        Generates one subplot per test_name (e.g., pERK_clustering and
        NINJA_clustering get separate panels) so that different analyses
        within the same test type are not conflated.

        Parameters
        ----------
        per_tumor_df : pd.DataFrame
            Per-tumor results
        """
        if per_tumor_df is None or len(per_tumor_df) == 0:
            return

        print("    Generating p-value QQ plots...")

        test_names = per_tumor_df['test_name'].unique()
        n_tests = len(test_names)

        if n_tests == 0:
            return

        n_cols = min(3, n_tests)
        n_rows = (n_tests + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=(6 * n_cols, 5.5 * n_rows),
                                squeeze=False)

        for idx, test_name in enumerate(test_names):
            r, c = idx // n_cols, idx % n_cols
            ax = axes[r, c]

            test_data = per_tumor_df[per_tumor_df['test_name'] == test_name]
            test_type = test_data['test_type'].iloc[0]
            p_values = test_data['p_value'].dropna().sort_values()

            n = len(p_values)
            if n < 10:
                ax.text(0.5, 0.5, 'Insufficient data',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{test_name}')
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
            ax.text(0.05, 0.95,
                    f'n = {n}\nKS stat = {ks_stat:.3f}\np = {ks_p:.3f}',
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel('Expected (Uniform)', fontweight='bold')
            ax.set_ylabel('Observed P-values', fontweight='bold')
            ax.set_title(f'{test_name} ({test_type})', fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.legend(loc='lower right')

        # Hide empty subplots
        for idx in range(n_tests, n_rows * n_cols):
            r, c = idx // n_cols, idx % n_cols
            axes[r, c].set_visible(False)

        plt.suptitle('P-value QQ Plots (per analysis)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = self.plots_dir / 'pvalue_qq_plot.png'
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"      Saved QQ plots to {self.plots_dir.name}/")

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
                    'mean_z_score': ['mean', 'sem']
                }).reset_index()
                agg.columns = ['timepoint', 'mean', 'sem']

                color = self._get_group_color(group, i)

                # Individual points
                ax1.scatter(group_data['timepoint'], group_data['mean_z_score'],
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

    def _reconstruct_null_samples(self, df: pd.DataFrame, n_samples_per_tumor: int = 200) -> np.ndarray:
        """
        Reconstruct an aggregate null distribution by sampling from each tumor's
        null distribution (parameterized by null_mean and null_std).
        """
        null_samples = []
        for _, row in df.iterrows():
            nm, ns = row['null_mean'], row['null_std']
            if pd.notna(nm) and pd.notna(ns) and ns > 0:
                null_samples.append(np.random.normal(nm, ns, n_samples_per_tumor))
        if null_samples:
            return np.concatenate(null_samples)
        return np.array([])

    def _plot_null_vs_observed_kde(self, ax: plt.Axes, null_values: np.ndarray,
                                   observed_values: np.ndarray, title: str,
                                   show_stats: bool = False, direction_label: str = ''):
        """
        Draw overlapped KDE of aggregate null and observed distributions on one axis.

        direction_label: e.g. '← more clustered  |  more dispersed →'
        """
        if len(null_values) < 5 or len(observed_values) < 5:
            ax.text(0.5, 0.5, 'Insufficient data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontweight='bold')
            return

        # Shared x-range
        all_vals = np.concatenate([null_values, observed_values])
        x_min = np.percentile(all_vals, 0.5)
        x_max = np.percentile(all_vals, 99.5)
        x_grid = np.linspace(x_min, x_max, 300)

        # Null KDE
        null_kde = stats.gaussian_kde(null_values)
        ax.fill_between(x_grid, null_kde(x_grid), alpha=0.35, color='#999999', label='Aggregate Null')
        ax.plot(x_grid, null_kde(x_grid), color='#666666', linewidth=1.5)

        # Observed KDE
        obs_kde = stats.gaussian_kde(observed_values)
        ax.fill_between(x_grid, obs_kde(x_grid), alpha=0.35, color=self.sig_color, label='Observed')
        ax.plot(x_grid, obs_kde(x_grid), color=self.sig_color, linewidth=1.5)

        xlabel = f'Statistic Value\n{direction_label}' if direction_label else 'Statistic Value'
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title(title, fontweight='bold')

        if show_stats:
            # KS test
            ks_stat, ks_p = stats.ks_2samp(observed_values, null_values)
            # Median shift
            null_med = np.median(null_values)
            obs_med = np.median(observed_values)
            shift = obs_med - null_med
            # Cohen's d
            pooled_std = np.sqrt((np.var(null_values) + np.var(observed_values)) / 2)
            cohens_d = (np.mean(observed_values) - np.mean(null_values)) / pooled_std if pooled_std > 0 else 0

            # Median lines
            ax.axvline(null_med, color='#666666', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.axvline(obs_med, color=self.sig_color, linestyle='--', linewidth=1.5, alpha=0.8)

            stat_text = (f"KS = {ks_stat:.3f}, p = {ks_p:.2e}\n"
                         f"Median shift = {shift:+.3f}\n"
                         f"Cohen's d = {cohens_d:.2f}")
            ax.text(0.97, 0.95, stat_text, transform=ax.transAxes,
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        ax.legend(loc='upper left', fontsize=9)

    def _plot_direct_metric_dist(self, ax: plt.Axes, values: np.ndarray,
                                  null_ref: float, title: str,
                                  direction_label: str = '', color: str = '#377EB8'):
        """
        Plot observed distribution of a directly computed metric (clark_evans_r,
        morans_i) with a vertical reference line at the theoretical null value.
        No permutation null is stored for these — the null is the theoretical
        random expectation (R=1 for CE, I=0 for Moran's I).
        """
        if len(values) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title, fontweight='bold')
            return

        ax.hist(values, bins=min(20, max(5, len(values) // 3)), color=color,
                alpha=0.45, density=True, edgecolor='white', linewidth=0.5)
        try:
            kde = stats.gaussian_kde(values)
            x_range = np.linspace(values.min() - values.std() * 0.3,
                                   values.max() + values.std() * 0.3, 200)
            ax.plot(x_range, kde(x_range), color=color, linewidth=2)
        except Exception:
            pass

        # Null reference
        ax.axvline(null_ref, color='black', linestyle='--', linewidth=2,
                   alpha=0.85, label=f'Null (={null_ref})', zorder=5)

        # Median
        med = np.median(values)
        ax.axvline(med, color=color, linestyle='-', linewidth=1.8,
                   alpha=0.9, label=f'Median={med:.3f}', zorder=4)

        # 1-sample t-test vs null
        try:
            _, p_t = stats.ttest_1samp(values, null_ref)
            stars = '***' if p_t < 0.001 else ('**' if p_t < 0.01 else
                    ('*' if p_t < 0.05 else 'ns'))
            ax.text(0.97, 0.97, f'n={len(values)}\nt-test p={p_t:.3f} {stars}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        except Exception:
            ax.text(0.97, 0.97, f'n={len(values)}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=9)

        xlabel = f'Metric Value\n{direction_label}' if direction_label else 'Metric Value'
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)

    def plot_aggregate_null_vs_observed(self, per_tumor_df: pd.DataFrame):
        """
        Multi-metric null vs observed plots.

        Per test generates:
        1. All-sample KDE (mean NN statistic vs permutation null): clean + stats panels
        2. By-group figure: rows = mean NN null/obs + clark_evans_r dist + morans_i dist;
           cols = groups. All X-axes annotated with clustering direction.
        3. By-timepoint KDE (mean NN statistic).
        """
        if per_tumor_df is None or len(per_tumor_df) == 0:
            return

        print("    Generating aggregate null vs observed KDE plots...")

        ce_direction = '← clustered (R<1)  |  R=1 random  |  dispersed (R>1) →'
        mi_direction = '← dispersed (I<0)  |  I=0 random  |  clustered (I>0) →'

        # Direction labels depend on test type for the primary z_score statistic
        _nn_by_type = {
            'clustering':     '← more clustered (shorter NN)  |  more dispersed →',
            'colocalization': '← fewer co-localizing  |  more co-localizing →',
            'enrichment':     '← fewer immune near marker+  |  more immune near marker+ →',
        }

        group_col = 'main_group' if 'main_group' in per_tumor_df.columns else 'group'

        for test_name in per_tumor_df['test_name'].unique():
            test_data = per_tumor_df[per_tumor_df['test_name'] == test_name].copy()
            test_type = test_data['test_type'].iloc[0]
            clean_name = test_name.replace('/', '_').replace(' ', '_')
            observed_all = test_data['observed'].dropna().values

            if len(observed_all) < 5:
                continue

            null_all = self._reconstruct_null_samples(test_data)
            if len(null_all) < 5:
                continue

            n_tumors = len(test_data)

            nn_direction = _nn_by_type.get(test_type, '← lower  |  higher →')

            # --- 1. All samples: clean (left) + stats (right) ---
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            self._plot_null_vs_observed_kde(
                axes[0], null_all, observed_all,
                title='All Samples', show_stats=False,
                direction_label=nn_direction)
            self._plot_null_vs_observed_kde(
                axes[1], null_all, observed_all,
                title='All Samples (with statistics)', show_stats=True,
                direction_label=nn_direction)

            plt.suptitle(
                f'Mean NN: Null vs Observed — {test_name} ({test_type}) (n={n_tumors} structures)',
                fontsize=14, fontweight='bold')
            plt.tight_layout()
            plot_path = self.plots_dir / f'aggregate_null_vs_observed_{clean_name}_all.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            # --- 2. Multi-metric by group ---
            # rows: mean_NN null/obs | clark_evans_r | morans_i
            # cols: each group
            groups = [g for g in sorted(test_data[group_col].dropna().unique())
                      if str(g).strip() != '']
            if len(groups) >= 1:
                has_ce = ('clark_evans_r' in test_data.columns and
                          test_data['clark_evans_r'].notna().any())
                has_mi = ('morans_i' in test_data.columns and
                          test_data['morans_i'].notna().any())

                n_rows = 1 + int(has_ce) + int(has_mi)
                n_cols = len(groups)
                fig, axes = plt.subplots(n_rows, n_cols,
                                         figsize=(5.5 * n_cols, 4.5 * n_rows),
                                         squeeze=False)

                # Row 0: mean NN null vs observed
                for col_j, group in enumerate(groups):
                    grp_data = test_data[test_data[group_col] == group]
                    grp_obs = grp_data['observed'].dropna().values
                    grp_null = self._reconstruct_null_samples(grp_data)
                    color = self._get_group_color(group, col_j)
                    # Temporarily override sig_color for per-group colour
                    orig_sig = self.sig_color
                    self.sig_color = color
                    self._plot_null_vs_observed_kde(
                        axes[0][col_j], grp_null, grp_obs,
                        title=f'Mean NN — {group} (n={len(grp_data)})',
                        show_stats=True, direction_label=nn_direction)
                    self.sig_color = orig_sig

                # Row 1: clark_evans_r distribution
                if has_ce:
                    ce_row = 1
                    for col_j, group in enumerate(groups):
                        grp_data = test_data[test_data[group_col] == group]
                        ce_vals = grp_data['clark_evans_r'].dropna().values
                        color = self._get_group_color(group, col_j)
                        self._plot_direct_metric_dist(
                            axes[ce_row][col_j], ce_vals, null_ref=1.0,
                            title=f'Clark-Evans R — {group}',
                            direction_label=ce_direction, color=color)

                # Row 2 (or 1 if no CE): morans_i distribution
                if has_mi:
                    mi_row = 1 + int(has_ce)
                    for col_j, group in enumerate(groups):
                        grp_data = test_data[test_data[group_col] == group]
                        mi_vals = grp_data['morans_i'].dropna().values
                        color = self._get_group_color(group, col_j)
                        self._plot_direct_metric_dist(
                            axes[mi_row][col_j], mi_vals, null_ref=0.0,
                            title=f"Moran's I — {group}",
                            direction_label=mi_direction, color=color)

                plt.suptitle(
                    f'All Metrics by Group — {test_name} ({test_type})',
                    fontsize=14, fontweight='bold')
                plt.tight_layout()
                plot_path = self.plots_dir / f'aggregate_null_vs_observed_{clean_name}_by_group.png'
                plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()

            # --- 3. By timepoint (mean NN only) ---
            if 'timepoint' in test_data.columns:
                timepoints = sorted(test_data['timepoint'].dropna().unique())
                if len(timepoints) >= 2:
                    n_tp = len(timepoints)
                    n_cols_tp = min(4, n_tp)
                    n_rows_tp = (n_tp + n_cols_tp - 1) // n_cols_tp
                    fig, axes = plt.subplots(n_rows_tp, n_cols_tp,
                                            figsize=(5.5 * n_cols_tp, 4.5 * n_rows_tp))
                    if n_tp == 1:
                        axes = np.array([[axes]])
                    elif n_rows_tp == 1:
                        axes = axes.reshape(1, -1)
                    elif n_cols_tp == 1:
                        axes = axes.reshape(-1, 1)

                    for idx, tp in enumerate(timepoints):
                        r, c = idx // n_cols_tp, idx % n_cols_tp
                        ax = axes[r, c]
                        tp_data = test_data[test_data['timepoint'] == tp]
                        tp_obs = tp_data['observed'].dropna().values
                        tp_null = self._reconstruct_null_samples(tp_data)
                        self._plot_null_vs_observed_kde(
                            ax, tp_null, tp_obs,
                            title=f'Timepoint {tp} (n={len(tp_data)})',
                            show_stats=True, direction_label=nn_direction)

                    for idx in range(n_tp, n_rows_tp * n_cols_tp):
                        r, c = idx // n_cols_tp, idx % n_cols_tp
                        axes[r, c].set_visible(False)

                    plt.suptitle(
                        f'Mean NN: Null vs Observed by Timepoint — {test_name} ({test_type})',
                        fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    plot_path = self.plots_dir / f'aggregate_null_vs_observed_{clean_name}_by_timepoint.png'
                    plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
                    plt.close()

        print(f"      Saved aggregate null vs observed plots to {self.plots_dir.name}/")

    def plot_by_prevalence_group(self, per_tumor_df: pd.DataFrame):
        """
        Plot z-scores split by prevalence group (low/medium/high tertile).

        For each test, generates a violin plot panel with three groups of violins
        (low/medium/high prevalence), showing both mean_NN z-score and clark_evans_r.
        """
        if per_tumor_df is None or len(per_tumor_df) == 0:
            return

        if 'prevalence_group' not in per_tumor_df.columns:
            return

        print("    Generating prevalence group plots...")

        group_order = ['low', 'medium', 'high']
        group_palette = {'low': '#4DAF4A', 'medium': '#FF7F00', 'high': '#E41A1C'}

        for test_name in per_tumor_df['test_name'].unique():
            test_data = per_tumor_df[per_tumor_df['test_name'] == test_name].copy()
            test_type = test_data['test_type'].iloc[0]

            has_ce = 'clark_evans_r' in test_data.columns and test_data['clark_evans_r'].notna().any()
            n_panels = 2 if has_ce else 1

            fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
            if n_panels == 1:
                axes = [axes]

            # Panel 1: mean_NN z-score by prevalence group
            ax1 = axes[0]
            present_groups = [g for g in group_order if g in test_data['prevalence_group'].values]
            plot_data = [test_data[test_data['prevalence_group'] == g]['z_score'].dropna().values
                         for g in present_groups]
            plot_data = [d for d in plot_data if len(d) > 0]
            present_groups_filtered = [g for g, d in zip(present_groups,
                                        [test_data[test_data['prevalence_group'] == g]['z_score'].dropna().values
                                         for g in present_groups]) if len(d) > 0]

            if plot_data:
                positions = list(range(len(plot_data)))
                colors = [group_palette.get(g, '#999999') for g in present_groups_filtered]

                vparts = ax1.violinplot(plot_data, positions=positions, showmeans=True)
                for i, pc in enumerate(vparts['bodies']):
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.7)

                bp = ax1.boxplot(plot_data, positions=positions, widths=0.12,
                                patch_artist=True, showfliers=False)
                for patch in bp['boxes']:
                    patch.set_facecolor('white')
                    patch.set_alpha(0.8)

                ax1.set_xticks(positions)
                ax1.set_xticklabels(present_groups_filtered)
                ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6)

            ax1.set_xlabel('Prevalence Tertile', fontweight='bold')
            ax1.set_ylabel('Effect Size (z-score)', fontweight='bold')
            ax1.set_title(f'Mean NN z-score\nby Prevalence Group', fontweight='bold')

            # Panel 2: Clark-Evans R by prevalence group
            if has_ce:
                ax2 = axes[1]
                ce_data = [test_data[test_data['prevalence_group'] == g]['clark_evans_r'].dropna().values
                           for g in present_groups]
                ce_data_f = [(g, d) for g, d in zip(present_groups, ce_data) if len(d) > 0]

                if ce_data_f:
                    pos2 = list(range(len(ce_data_f)))
                    cols2 = [group_palette.get(g, '#999999') for g, _ in ce_data_f]
                    ce_vals = [d for _, d in ce_data_f]
                    ce_grps = [g for g, _ in ce_data_f]

                    vparts2 = ax2.violinplot(ce_vals, positions=pos2, showmeans=True)
                    for i, pc in enumerate(vparts2['bodies']):
                        pc.set_facecolor(cols2[i])
                        pc.set_alpha(0.7)

                    ax2.set_xticks(pos2)
                    ax2.set_xticklabels(ce_grps)
                    ax2.axhline(1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

                ax2.set_xlabel('Prevalence Tertile', fontweight='bold')
                ax2.set_ylabel('Clark-Evans R', fontweight='bold')
                ax2.set_title('Clark-Evans R\n(R<1=clustered, R>1=dispersed)', fontweight='bold')

            clean_name = test_name.replace('/', '_').replace(' ', '_')
            plt.suptitle(f'Clustering by Prevalence Group: {test_name} ({test_type})',
                        fontsize=13, fontweight='bold')
            plt.tight_layout()

            plot_path = self.plots_dir / f'prevalence_group_{clean_name}.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        print(f"      Saved prevalence group plots to {self.plots_dir.name}/")

    def plot_binned_null_vs_observed_by_prevalence(self, per_tumor_df: pd.DataFrame):
        """
        Null vs observed KDE plots separated by prevalence tertile.

        The aggregate null distribution is often trimodal because tumors with very
        different marker prevalences have different null expectations. This plot
        re-runs the comparison WITHIN each tertile (low/medium/high % positive),
        showing whether the observed statistic departs from the null at each level.

        Layout: rows = prevalence tertiles (low/medium/high), 2 cols (clean + stats).
        One figure per test. Saved as: binned_null_vs_observed_{test}.png
        """
        if per_tumor_df is None or len(per_tumor_df) == 0:
            return

        print("    Generating binned null vs observed plots by prevalence tertile...")

        _nn_by_type = {
            'clustering':     '← more clustered (shorter NN)  |  more dispersed →',
            'colocalization': '← fewer co-localizing  |  more co-localizing →',
            'enrichment':     '← fewer immune near marker+  |  more immune near marker+ →',
        }

        for test_name in per_tumor_df['test_name'].unique():
            test_data = per_tumor_df[per_tumor_df['test_name'] == test_name].copy()
            test_type = test_data['test_type'].iloc[0]
            clean_name = test_name.replace('/', '_').replace(' ', '_')
            nn_direction = _nn_by_type.get(test_type, '← lower  |  higher →')

            # Assign prevalence tertiles within this test (may already exist)
            if 'prevalence_group' not in test_data.columns or test_data['prevalence_group'].isna().all():
                prev = test_data['prevalence'].dropna()
                if len(prev) < 9:
                    continue
                try:
                    test_data['prevalence_group'] = pd.qcut(
                        test_data['prevalence'], q=3,
                        labels=['low', 'medium', 'high'], duplicates='drop')
                except Exception:
                    continue

            tertile_order = ['low', 'medium', 'high']
            present = [t for t in tertile_order if t in test_data['prevalence_group'].values]
            if not present:
                continue

            n_tertiles = len(present)
            fig, axes = plt.subplots(n_tertiles, 2,
                                     figsize=(14, 4.5 * n_tertiles),
                                     squeeze=False)

            for row_i, tertile in enumerate(present):
                bin_data = test_data[test_data['prevalence_group'] == tertile]
                bin_obs = bin_data['observed'].dropna().values
                bin_null = self._reconstruct_null_samples(bin_data)
                n_structs = len(bin_data)
                # Median prevalence in this bin
                med_prev = bin_data['prevalence'].median() * 100

                if len(bin_obs) < 5 or len(bin_null) < 5:
                    for c in range(2):
                        axes[row_i][c].text(0.5, 0.5,
                            f'{tertile} prevalence\nn={n_structs} (insufficient)',
                            ha='center', va='center', transform=axes[row_i][c].transAxes)
                        axes[row_i][c].set_title(f'{tertile.capitalize()} prevalence '
                                                 f'(median {med_prev:.0f}% positive)',
                                                 fontweight='bold')
                    continue

                title_base = (f'{tertile.capitalize()} prevalence '
                              f'(median {med_prev:.0f}% positive, n={n_structs})')
                self._plot_null_vs_observed_kde(
                    axes[row_i][0], bin_null, bin_obs,
                    title=title_base, show_stats=False,
                    direction_label=nn_direction)
                self._plot_null_vs_observed_kde(
                    axes[row_i][1], bin_null, bin_obs,
                    title=f'{title_base} + stats', show_stats=True,
                    direction_label=nn_direction)

            plt.suptitle(
                f'Null vs Observed by Prevalence Tertile — {test_name} ({test_type})',
                fontsize=14, fontweight='bold')
            plt.tight_layout()

            plot_path = self.plots_dir / f'binned_null_vs_observed_{clean_name}.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        print(f"      Saved binned null vs observed plots to {self.plots_dir.name}/")

    def plot_colocalization_double_positive(self, per_tumor_df: pd.DataFrame):
        """
        Scatter plot of observed vs expected % double-positive cells for colocalization tests.

        For each tumor, under marker independence:
            expected_pct_both = P(marker1) * P(marker2) * 100
        Observed:
            observed_pct_both = (n_cells positive for BOTH markers) / n_total * 100

        Points above the y=x line have MORE double-positives than expected (enrichment);
        points below have FEWER (mutual exclusion or spatial segregation).
        Points colored by main_group.

        Requires n_both, observed_pct_both, expected_pct_both columns (added in
        _test_colocalization in spatial_permutation_testing.py).
        """
        if per_tumor_df is None or len(per_tumor_df) == 0:
            return

        coloc_data = per_tumor_df[per_tumor_df['test_type'] == 'colocalization'].copy()
        if len(coloc_data) == 0:
            return

        required = ['observed_pct_both', 'expected_pct_both']
        if not all(c in coloc_data.columns for c in required):
            return

        coloc_data = coloc_data.dropna(subset=required)
        if len(coloc_data) < 3:
            return

        print("    Generating colocalization double-positive plots...")

        group_col = 'main_group' if 'main_group' in coloc_data.columns else 'group'
        groups = sorted(coloc_data[group_col].dropna().unique())

        for test_name in coloc_data['test_name'].unique():
            test_data = coloc_data[coloc_data['test_name'] == test_name].copy()
            clean_name = test_name.replace('/', '_').replace(' ', '_')

            if len(test_data) < 3:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

            # --- Panel 1: scatter observed vs expected per tumor ---
            ax1 = axes[0]
            xy_max = max(test_data['observed_pct_both'].max(),
                         test_data['expected_pct_both'].max()) * 1.1
            xy_max = max(xy_max, 1.0)

            ax1.plot([0, xy_max], [0, xy_max], 'k--', linewidth=1.5,
                     alpha=0.6, label='y = x (independence)')

            for i, g in enumerate(groups):
                grp = test_data[test_data[group_col].fillna('').astype(str) == str(g)]
                if len(grp) == 0:
                    continue
                color = self._get_group_color(g, i)
                ax1.scatter(grp['expected_pct_both'], grp['observed_pct_both'],
                            color=color, alpha=0.65, s=35, edgecolors='white',
                            linewidth=0.4, label=g, zorder=3)

            ax1.set_xlim(0, xy_max)
            ax1.set_ylim(0, xy_max)
            ax1.set_xlabel('Expected % double positive\n(under independence)',
                           fontweight='bold')
            ax1.set_ylabel('Observed % double positive', fontweight='bold')
            ax1.set_title('Observed vs Expected % Double Positive\n(per tumor structure)',
                          fontweight='bold')
            ax1.legend(fontsize=9, loc='upper left')
            ax1.set_aspect('equal')

            # Annotation: overall enrichment test
            obs_all = test_data['observed_pct_both'].values
            exp_all = test_data['expected_pct_both'].values
            try:
                _, p_paired = stats.wilcoxon(obs_all - exp_all)
                fold = (obs_all.mean() / exp_all.mean()) if exp_all.mean() > 0 else np.nan
                ax1.text(0.97, 0.05,
                         f'Wilcoxon p = {p_paired:.3f}\nmean fold = {fold:.2f}x',
                         transform=ax1.transAxes, ha='right', va='bottom', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
            except Exception:
                pass

            # --- Panel 2: fold enrichment distribution by group ---
            ax2 = axes[1]
            test_data = test_data.copy()
            test_data['fold_enrichment'] = np.where(
                test_data['expected_pct_both'] > 0,
                test_data['observed_pct_both'] / test_data['expected_pct_both'],
                np.nan)

            plot_data2, labels2, colors2 = [], [], []
            for i, g in enumerate(groups):
                grp = test_data[test_data[group_col].fillna('').astype(str) == str(g)]
                fe = grp['fold_enrichment'].dropna().values
                if len(fe) >= 2:
                    plot_data2.append(fe)
                    labels2.append(g)
                    colors2.append(self._get_group_color(g, i))

            if plot_data2:
                positions = list(range(len(plot_data2)))
                try:
                    vp = ax2.violinplot(plot_data2, positions=positions, showmeans=True)
                    for j, pc in enumerate(vp['bodies']):
                        pc.set_facecolor(colors2[j])
                        pc.set_alpha(0.7)
                except Exception:
                    pass
                bp = ax2.boxplot(plot_data2, positions=positions, widths=0.15,
                                 patch_artist=True, showfliers=False)
                for patch in bp['boxes']:
                    patch.set_facecolor('white')
                    patch.set_alpha(0.8)
                for j, data in enumerate(plot_data2):
                    jitter = np.random.uniform(-0.12, 0.12, len(data))
                    ax2.scatter([j + jitter[k] for k in range(len(data))], data,
                                color=colors2[j], alpha=0.55, s=25,
                                edgecolors='white', linewidth=0.4)

                ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.5,
                            alpha=0.7, label='Fold=1 (independence)')
                ax2.set_xticks(positions)
                ax2.set_xticklabels(labels2)

            ax2.set_xlabel('Group', fontweight='bold')
            ax2.set_ylabel('Fold enrichment\n(observed / expected)', fontweight='bold')
            ax2.set_title('Double-Positive Fold Enrichment by Group', fontweight='bold')
            ax2.legend(fontsize=9)

            plt.suptitle(f'Co-expression Enrichment: {test_name}',
                         fontsize=13, fontweight='bold')
            plt.tight_layout()

            plot_path = self.plots_dir / f'coloc_double_positive_{clean_name}.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        print(f"      Saved double-positive plots to {self.plots_dir.name}/")

    def plot_multi_metric_group_comparison(self, per_tumor_df: pd.DataFrame):
        """
        Per-group distributions of each clustering metric vs theoretical null.

        Layout: rows = metrics, cols = groups (KPT / KPNT).
        Each cell shows a histogram + KDE of per-tumor metric values for that group,
        with a vertical line at the null/random expectation:
          - z_score:          null = 0   (no clustering signal vs permutation)
          - Clark-Evans R:    null = 1.0 (complete spatial randomness)
          - Moran's I:        null = 0.0 (no spatial autocorrelation)
          - NN ratio z-score: null = 0

        1-sample t-test annotation shows whether each group departs significantly
        from the null; the x-axis label shows which direction means clustered.
        """
        if per_tumor_df is None or len(per_tumor_df) == 0:
            return

        group_col = 'main_group' if 'main_group' in per_tumor_df.columns else 'group'
        groups = [g for g in sorted(per_tumor_df[group_col].dropna().unique())
                  if str(g).strip() != '']
        if not groups:
            return

        print("    Generating multi-metric vs null plots...")

        _z_direction_by_type = {
            'clustering':     '← clustered (z<0)  |  dispersed (z>0) →',
            'colocalization': '← fewer co-localizing (z<0)  |  more co-localizing (z>0) →',
            'enrichment':     '← fewer immune near marker+ (z<0)  |  more (z>0) →',
        }

        # Clark-Evans and Moran's I only meaningful for clustering tests;
        # they will be absent from colocalization/enrichment results naturally.
        _ce_dlbl = '← clustered (R<1)  |  R=1 random  |  dispersed (R>1) →'
        _mi_dlbl = '← dispersed (I<0)  |  I=0 random  |  clustered (I>0) →'
        _nn_dlbl = '← segregated (z<0)  |  intermixed (z>0) →'

        for test_name in per_tumor_df['test_name'].unique():
            test_data = per_tumor_df[per_tumor_df['test_name'] == test_name].copy()
            clean_name = test_name.replace('/', '_').replace(' ', '_')
            ttype = test_data['test_type'].iloc[0] if 'test_type' in test_data.columns else 'clustering'

            metrics = [
                ('z_score',          'z-score',
                 0.0, _z_direction_by_type.get(ttype, '← lower  |  higher →')),
                ('clark_evans_r',    'Clark-Evans R',    1.0, _ce_dlbl),
                ('morans_i',         "Moran's I",        0.0, _mi_dlbl),
                ('nn_ratio_z_score', 'NN ratio z-score', 0.0, _nn_dlbl),
            ]
            available = [(col, lbl, null, dlbl) for col, lbl, null, dlbl in metrics
                         if col in test_data.columns and test_data[col].notna().any()]

            if not available:
                continue

            n_rows = len(available)
            n_cols = len(groups)
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(4.8 * n_cols, 4.0 * n_rows),
                                     squeeze=False)

            for row_i, (metric, ylabel, null_val, dir_label) in enumerate(available):
                for col_j, group in enumerate(groups):
                    ax = axes[row_i][col_j]
                    vals = test_data[test_data[group_col] == group][metric].dropna().values
                    color = self._get_group_color(group, col_j)

                    if len(vals) < 3:
                        ax.text(0.5, 0.5, f'n={len(vals)}\n(insufficient)',
                                ha='center', va='center', transform=ax.transAxes,
                                fontsize=9)
                        ax.set_title(f'{group}' if row_i == 0 else '', fontsize=11)
                        ax.set_ylabel(ylabel if col_j == 0 else '', fontweight='bold',
                                      fontsize=9)
                        continue

                    # Histogram
                    ax.hist(vals, bins=min(20, max(5, len(vals) // 3)),
                            color=color, alpha=0.45, density=True,
                            edgecolor='white', linewidth=0.5)
                    # KDE
                    try:
                        kde = stats.gaussian_kde(vals)
                        x_min = min(vals.min(), null_val) - vals.std() * 0.4
                        x_max = max(vals.max(), null_val) + vals.std() * 0.4
                        xg = np.linspace(x_min, x_max, 200)
                        ax.plot(xg, kde(xg), color=color, linewidth=2)
                    except Exception:
                        pass

                    # Null reference line
                    ax.axvline(null_val, color='black', linestyle='--', linewidth=2,
                               alpha=0.85, zorder=5, label=f'Null={null_val}')
                    # Median line
                    med = np.median(vals)
                    ax.axvline(med, color=color, linestyle='-', linewidth=1.8,
                               alpha=0.9, zorder=4, label=f'Median={med:.3f}')

                    # 1-sample t-test vs null
                    try:
                        _, p_t = stats.ttest_1samp(vals, null_val)
                        stars = ('***' if p_t < 0.001 else '**' if p_t < 0.01
                                 else '*' if p_t < 0.05 else 'ns')
                        ax.text(0.97, 0.97,
                                f'n={len(vals)}\np={p_t:.3f} {stars}',
                                transform=ax.transAxes, ha='right', va='top',
                                fontsize=8.5,
                                bbox=dict(boxstyle='round', facecolor='white',
                                          alpha=0.85))
                    except Exception:
                        ax.text(0.97, 0.97, f'n={len(vals)}',
                                transform=ax.transAxes, ha='right', va='top',
                                fontsize=8.5)

                    # Top row: group name as column header
                    if row_i == 0:
                        ax.set_title(group, fontsize=11, fontweight='bold')
                    # Left column: metric label as row header
                    ax.set_ylabel(ylabel if col_j == 0 else '', fontweight='bold',
                                  fontsize=9)
                    ax.set_xlabel(dir_label, fontsize=7.5)
                    ax.legend(fontsize=7, loc='upper left')

            plt.suptitle(
                f'Metrics vs Null ({ttype}) — {test_name}',
                fontsize=13, fontweight='bold')
            plt.tight_layout()

            plot_path = self.plots_dir / f'multi_metric_vs_null_{clean_name}.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        print(f"      Saved multi-metric vs null plots to {self.plots_dir.name}/")

    def plot_metric_correlations(self, per_tumor_df: pd.DataFrame):
        """
        Scatter plot matrix comparing clustering metrics to each other.

        Shows: clark_evans_r vs z_score, morans_i vs z_score, morans_i vs clark_evans_r.
        Points coloured by main_group (or group). Adds Pearson r and p-value.
        """
        if per_tumor_df is None or len(per_tumor_df) == 0:
            return

        metric_pairs = [
            ('z_score', 'clark_evans_r', 'Mean NN z-score', 'Clark-Evans R'),
            ('z_score', 'morans_i',      'Mean NN z-score', "Moran's I"),
            ('clark_evans_r', 'morans_i', 'Clark-Evans R',  "Moran's I"),
        ]
        available_pairs = [(x, y, xl, yl) for x, y, xl, yl in metric_pairs
                           if x in per_tumor_df.columns and y in per_tumor_df.columns]

        if not available_pairs:
            return

        group_col = 'main_group' if 'main_group' in per_tumor_df.columns else 'group'
        groups = sorted(per_tumor_df[group_col].dropna().unique())

        print("    Generating metric correlation plots...")

        for test_name in per_tumor_df['test_name'].unique():
            test_data = per_tumor_df[per_tumor_df['test_name'] == test_name].copy()
            clean_name = test_name.replace('/', '_').replace(' ', '_')

            n_pairs = len(available_pairs)
            fig, axes = plt.subplots(1, n_pairs, figsize=(5.5 * n_pairs, 5))
            if n_pairs == 1:
                axes = [axes]

            for ax, (x_col, y_col, x_label, y_label) in zip(axes, available_pairs):
                # Drop NaN only on the two metric columns so that rows with
                # missing group label don't silently empty the data
                valid = test_data.dropna(subset=[x_col, y_col]).copy()
                if len(valid) < 5:
                    ax.set_visible(False)
                    continue

                for i, g in enumerate(groups):
                    grp_mask = valid[group_col].fillna('').astype(str) == str(g)
                    grp = valid[grp_mask]
                    if len(grp) == 0:
                        continue
                    color = self._get_group_color(g, i)
                    ax.scatter(grp[x_col], grp[y_col], color=color, alpha=0.6, s=30,
                               label=g, edgecolors='white', linewidth=0.4)

                # Overall Pearson r
                x_vals = valid[x_col].values
                y_vals = valid[y_col].values
                try:
                    r, p_r = stats.pearsonr(x_vals, y_vals)
                    # Regression line
                    m, b = np.polyfit(x_vals, y_vals, 1)
                    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                    ax.plot(x_line, m * x_line + b, 'k--', linewidth=1.5, alpha=0.7)
                    ax.text(0.97, 0.03, f'r={r:.2f}, p={p_r:.3f}',
                            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
                except Exception:
                    pass

                ax.set_xlabel(x_label, fontweight='bold')
                ax.set_ylabel(y_label, fontweight='bold')
                ax.set_title(f'{x_label} vs {y_label}', fontweight='bold', fontsize=10)
                if len(groups) > 1:
                    ax.legend(fontsize=8, loc='upper left')

            plt.suptitle(f'Metric Correlations: {test_name}', fontsize=13, fontweight='bold')
            plt.tight_layout()

            plot_path = self.plots_dir / f'metric_correlations_{clean_name}.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        print(f"      Saved metric correlation plots to {self.plots_dir.name}/")

    def plot_per_tumor_metrics_overview(self, per_tumor_df: pd.DataFrame):
        """
        Summary overview: one row per test, columns = each metric.
        Shows distribution of all 4 metrics (z_score, clark_evans_r, morans_i,
        nn_ratio_z_score) across all tumors, coloured by main_group.
        """
        if per_tumor_df is None or len(per_tumor_df) == 0:
            return

        group_col = 'main_group' if 'main_group' in per_tumor_df.columns else 'group'
        groups = sorted(per_tumor_df[group_col].dropna().unique())

        metrics = [
            ('z_score',          'Mean NN z-score',  None),
            ('clark_evans_r',    'Clark-Evans R',     1.0),
            ('morans_i',         "Moran's I",         0.0),
            ('nn_ratio_z_score', 'NN Ratio z-score',  None),
        ]
        avail = [(c, l, r) for c, l, r in metrics
                 if c in per_tumor_df.columns and per_tumor_df[c].notna().any()]

        if not avail:
            return

        print("    Generating per-tumor metrics overview...")

        test_names = per_tumor_df['test_name'].unique()
        n_tests = len(test_names)
        n_metrics = len(avail)

        fig, axes = plt.subplots(n_tests, n_metrics,
                                 figsize=(4.5 * n_metrics, 4.5 * n_tests),
                                 squeeze=False)

        for row_i, test_name in enumerate(test_names):
            test_data = per_tumor_df[per_tumor_df['test_name'] == test_name].copy()

            for col_j, (metric, label, ref_line) in enumerate(avail):
                ax = axes[row_i][col_j]

                plot_data = []
                valid_groups = []
                for g in groups:
                    vals = test_data[test_data[group_col] == g][metric].dropna().values
                    if len(vals) >= 1:
                        plot_data.append(vals)
                        valid_groups.append(g)

                if not plot_data:
                    ax.set_visible(False)
                    continue

                colors = [self._get_group_color(g, i) for i, g in enumerate(valid_groups)]

                if len(plot_data[0]) >= 3:
                    try:
                        vparts = ax.violinplot(plot_data, positions=range(len(valid_groups)),
                                               showmeans=False, showextrema=True)
                        for i, pc in enumerate(vparts['bodies']):
                            pc.set_facecolor(colors[i])
                            pc.set_alpha(0.6)
                    except Exception:
                        pass

                bp = ax.boxplot(plot_data, positions=range(len(valid_groups)),
                                widths=0.25, patch_artist=True, showfliers=True,
                                flierprops=dict(marker='o', markersize=3, alpha=0.4))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.5)

                if ref_line is not None:
                    ax.axhline(ref_line, color='black', linestyle=':', linewidth=1.0,
                               alpha=0.6)

                ax.set_xticks(range(len(valid_groups)))
                ax.set_xticklabels(valid_groups, fontsize=9)
                if col_j == 0:
                    short_name = test_name[:20] + '...' if len(test_name) > 20 else test_name
                    ax.set_ylabel(f'{short_name}\n{label}', fontweight='bold', fontsize=9)
                else:
                    ax.set_ylabel(label, fontsize=9)
                if row_i == 0:
                    ax.set_title(label, fontweight='bold', fontsize=10)

        plt.suptitle('Per-Tumor Metrics Overview (all tests × all metrics)',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()

        plot_path = self.plots_dir / 'per_tumor_metrics_overview.png'
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"      Saved per-tumor metrics overview to {self.plots_dir.name}/")

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

        # 8. Aggregate null vs observed KDE
        if 'per_tumor_results' in results:
            self.plot_aggregate_null_vs_observed(results['per_tumor_results'])

        # 9. Plots split by prevalence group (low/medium/high tertile)
        if 'per_tumor_results' in results:
            self.plot_by_prevalence_group(results['per_tumor_results'])

        # 10. Multi-metric group comparison (z_score, clark_evans_r, morans_i, nn_ratio_z)
        if 'per_tumor_results' in results:
            self.plot_multi_metric_group_comparison(results['per_tumor_results'])

        # 11. Metric correlation scatter plots
        if 'per_tumor_results' in results:
            self.plot_metric_correlations(results['per_tumor_results'])

        # 12. Per-tumor metrics overview grid (tests × metrics)
        if 'per_tumor_results' in results:
            self.plot_per_tumor_metrics_overview(results['per_tumor_results'])

        # 13. Null vs observed split by prevalence tertile (binned)
        if 'per_tumor_results' in results:
            self.plot_binned_null_vs_observed_by_prevalence(results['per_tumor_results'])

        # 14. Colocalization: observed vs expected double-positive % per tumor
        if 'per_tumor_results' in results:
            self.plot_colocalization_double_positive(results['per_tumor_results'])

        print(f"  All permutation plots saved to {self.plots_dir}/")
