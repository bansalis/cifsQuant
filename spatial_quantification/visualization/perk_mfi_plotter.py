"""
pERK MFI Plotter

Publication-quality plots for pERK mean fluorescence intensity analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings


class PerkMFIPlotter:
    """
    Visualization for pERK MFI analysis results.

    Generates:
    - Violin plots of raw/normalized MFI in gated+ vs gated- cells per group
    - Per-sample IQR / CV bars (gating consistency)
    - Scatter + binned regression of MFI vs T cell distance
    - Binned mean MFI by T cell density
    - Per-tumor median MFI vs immune cell count scatter
    """

    def __init__(self, output_dir: Path, config: Dict):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        plotting_config = config.get('plotting', {})
        self.dpi = config.get('output', {}).get('dpi', 300)
        self.group_colors = plotting_config.get('group_colors', {})
        self.default_colors = sns.color_palette('colorblind', 10)

        meta_config = config.get('metadata', {})
        self.group_col = (meta_config.get('primary_grouping') or
                          meta_config.get('group_column', 'group'))

        sns.set_style('whitegrid')
        plt.rcParams.update({
            'font.size': plotting_config.get('font_size', 11),
            'axes.spines.top': False,
            'axes.spines.right': False,
            'savefig.dpi': self.dpi,
        })

    def _get_color(self, group: str, groups: List[str]) -> str:
        if group in self.group_colors:
            return self.group_colors[group]
        idx = groups.index(group) % len(self.default_colors) if group in groups else 0
        return self.default_colors[idx]

    def generate_all_plots(self, results: Dict):
        """Generate all pERK MFI plots."""
        print("  Generating pERK MFI plots...")

        if 'mfi_distribution_by_gate' in results:
            self.plot_perk_mfi_distribution(results['mfi_distribution_by_gate'])

        if 'mfi_vs_tcell_proximity' in results:
            self.plot_perk_vs_tcell_distance(results['mfi_vs_tcell_proximity'])
            self.plot_perk_vs_tcell_density(results['mfi_vs_tcell_proximity'])

        if 'per_tumor_mfi_summary' in results:
            self.plot_per_tumor_summary(results['per_tumor_mfi_summary'])

        print(f"  All pERK MFI plots saved to {self.plots_dir}/")

    def plot_perk_mfi_distribution(self, dist_df: pd.DataFrame):
        """
        Violin plots of pERK MFI distributions in gated+ vs gated- cells,
        split by group. Two rows: raw MFI (top) and normalized MFI (bottom).
        """
        if dist_df is None or len(dist_df) == 0:
            return

        groups = sorted(dist_df[self.group_col].dropna().unique()) if self.group_col in dist_df.columns else ['all']
        mfi_types = dist_df['mfi_type'].unique()

        for mfi_type in mfi_types:
            df = dist_df[dist_df['mfi_type'] == mfi_type]

            fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 6), sharey=True)
            if len(groups) == 1:
                axes = [axes]

            fig.suptitle(f'pERK MFI Distribution ({mfi_type}) by Gate Status',
                        fontsize=13, fontweight='bold')

            for ax, group in zip(axes, groups):
                gdata = df[df[self.group_col] == group] if self.group_col in df.columns else df

                pos_vals = gdata[gdata['gate_status'] == 'positive']['median'].dropna().values
                neg_vals = gdata[gdata['gate_status'] == 'negative']['median'].dropna().values

                plot_data = []
                labels = []
                colors_list = []

                if len(pos_vals) > 0:
                    plot_data.append(pos_vals)
                    labels.append('pERK+')
                    colors_list.append('#E41A1C')
                if len(neg_vals) > 0:
                    plot_data.append(neg_vals)
                    labels.append('pERK-')
                    colors_list.append('#377EB8')

                if plot_data:
                    vparts = ax.violinplot(plot_data, positions=range(len(plot_data)),
                                          showmeans=True, showmedians=False)
                    for i, pc in enumerate(vparts['bodies']):
                        pc.set_facecolor(colors_list[i])
                        pc.set_alpha(0.7)

                    bp = ax.boxplot(plot_data, positions=range(len(plot_data)), widths=0.12,
                                   patch_artist=True, showfliers=True, flierprops={'markersize': 3})
                    for patch in bp['boxes']:
                        patch.set_facecolor('white')
                        patch.set_alpha(0.8)

                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels)

                ax.set_title(group, fontweight='bold')
                ax.set_ylabel(f'pERK MFI ({mfi_type})' if ax == axes[0] else '', fontweight='bold')

            plt.tight_layout()
            plot_path = self.plots_dir / f'perk_mfi_distribution_{mfi_type}.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        print(f"    ✓ Saved pERK MFI distribution plots")

    def plot_perk_vs_tcell_distance(self, proximity_df: pd.DataFrame):
        """
        Scatter + binned line of pERK MFI vs distance to nearest T cell population.
        One panel per T cell population per group.
        """
        if proximity_df is None or len(proximity_df) == 0:
            return

        dist_cols = [c for c in proximity_df.columns if c.startswith('dist_to_nearest_')]
        if not dist_cols:
            return

        groups = sorted(proximity_df[self.group_col].dropna().unique()) if self.group_col in proximity_df.columns else ['all']
        mfi_col = 'perk_mfi_normalized'
        if mfi_col not in proximity_df.columns:
            mfi_col = 'perk_mfi_raw'

        for dist_col in dist_cols:
            pop_name = dist_col.replace('dist_to_nearest_', '')

            n_groups = len(groups)
            fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 5), sharey=True)
            if n_groups == 1:
                axes = [axes]

            fig.suptitle(f'pERK MFI vs Distance to {pop_name}',
                        fontsize=13, fontweight='bold')

            for ax, group in zip(axes, groups):
                gdata = proximity_df[proximity_df[self.group_col] == group].copy() \
                        if self.group_col in proximity_df.columns else proximity_df.copy()

                valid = gdata[[dist_col, mfi_col, 'is_perk_positive']].dropna()
                if len(valid) < 10:
                    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(group)
                    continue

                color = self._get_color(group, groups)

                # Scatter (subsample to max 2000 for readability)
                n_plot = min(2000, len(valid))
                idx_sample = np.random.choice(len(valid), n_plot, replace=False)
                ax.scatter(valid[dist_col].values[idx_sample],
                          valid[mfi_col].values[idx_sample],
                          alpha=0.15, s=10, color=color)

                # Binned line
                try:
                    distance_bins = [0, 25, 50, 100, 200, 500]
                    bin_labels = [(distance_bins[i] + distance_bins[i+1]) / 2
                                  for i in range(len(distance_bins) - 1)]
                    valid['_dist_bin'] = pd.cut(valid[dist_col], bins=distance_bins)
                    binned = valid.groupby('_dist_bin', observed=True)[mfi_col].mean()
                    ax.plot(bin_labels[:len(binned)], binned.values, 'o-',
                           color='black', linewidth=2, markersize=7, zorder=10,
                           label='Binned mean')
                except Exception:
                    pass

                ax.set_xlabel(f'Distance to {pop_name} (µm)', fontweight='bold')
                ax.set_ylabel(f'pERK MFI ({mfi_col.split("_")[-1]})', fontweight='bold')
                ax.set_title(group, fontweight='bold')
                ax.legend(fontsize=9)

            plt.tight_layout()
            safe_pop = pop_name.replace('+', 'pos').replace('-', 'neg')
            plot_path = self.plots_dir / f'perk_mfi_vs_{safe_pop}_distance.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        print(f"    ✓ Saved pERK MFI vs T cell distance plots")

    def plot_perk_vs_tcell_density(self, proximity_df: pd.DataFrame):
        """
        Binned mean pERK MFI by T cell density (count within radius window).
        """
        if proximity_df is None or len(proximity_df) == 0:
            return

        density_cols = [c for c in proximity_df.columns if 'within_' in c and c.startswith('n_')]
        if not density_cols:
            return

        groups = sorted(proximity_df[self.group_col].dropna().unique()) if self.group_col in proximity_df.columns else ['all']
        mfi_col = 'perk_mfi_normalized' if 'perk_mfi_normalized' in proximity_df.columns else 'perk_mfi_raw'

        for density_col in density_cols:
            fig, ax = plt.subplots(figsize=(8, 5))

            for group in groups:
                gdata = proximity_df[proximity_df[self.group_col] == group].copy() \
                        if self.group_col in proximity_df.columns else proximity_df.copy()

                valid = gdata[[density_col, mfi_col]].dropna()
                if len(valid) < 10:
                    continue

                color = self._get_color(group, groups)

                # Bin by T cell count (0, 1-2, 3-5, 6-10, >10)
                density_bins = [-1, 0, 2, 5, 10, valid[density_col].max() + 1]
                density_labels = ['0', '1-2', '3-5', '6-10', '>10']
                valid['_density_bin'] = pd.cut(valid[density_col], bins=density_bins, labels=density_labels)
                binned = valid.groupby('_density_bin', observed=True)[mfi_col].agg(['mean', 'sem'])

                ax.errorbar(range(len(binned)), binned['mean'].values,
                           yerr=binned['sem'].values,
                           fmt='-o', color=color, linewidth=2, markersize=8,
                           capsize=4, label=group)

            ax.set_xticks(range(len(density_labels)))
            ax.set_xticklabels(density_labels)
            pop_name = density_col.split('_within_')[0].replace('n_', '')
            radius = density_col.split('within_')[1] if 'within_' in density_col else ''
            ax.set_xlabel(f'{pop_name} count within {radius}', fontweight='bold')
            ax.set_ylabel(f'Mean pERK MFI ({mfi_col.split("_")[-1]})', fontweight='bold')
            ax.set_title(f'pERK MFI vs {pop_name} Density', fontsize=12, fontweight='bold')
            ax.legend(frameon=True)

            plt.tight_layout()
            safe_col = density_col.replace('+', 'pos').replace('-', 'neg')
            plot_path = self.plots_dir / f'perk_mfi_vs_{safe_col}_density.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        print(f"    ✓ Saved pERK MFI vs T cell density plots")

    def plot_per_tumor_summary(self, per_tumor_df: pd.DataFrame):
        """
        Per-tumor median pERK MFI vs immune cell count scatter plots.
        """
        if per_tumor_df is None or len(per_tumor_df) == 0:
            return

        mfi_col = 'median_perk_mfi_normalized' if 'median_perk_mfi_normalized' in per_tumor_df.columns \
                  else 'median_perk_mfi_raw'

        immune_count_cols = [c for c in per_tumor_df.columns if c.startswith('n_') and c.endswith('_in_tumor')]
        if not immune_count_cols:
            immune_count_cols = ['n_perk_positive']

        groups = sorted(per_tumor_df[self.group_col].dropna().unique()) if self.group_col in per_tumor_df.columns else ['all']

        for immune_col in immune_count_cols:
            if immune_col not in per_tumor_df.columns:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))

            for group in groups:
                gdata = per_tumor_df[per_tumor_df[self.group_col] == group].copy() \
                        if self.group_col in per_tumor_df.columns else per_tumor_df.copy()

                valid = gdata[[immune_col, mfi_col]].dropna()
                if len(valid) < 3:
                    continue

                color = self._get_color(group, groups)
                ax.scatter(valid[immune_col].values, valid[mfi_col].values,
                          color=color, alpha=0.7, s=60, label=group,
                          edgecolors='white', linewidth=0.5)

                # Trend line
                if len(valid) >= 5:
                    try:
                        from scipy import stats as scipy_stats
                        slope, intercept, r, p, _ = scipy_stats.linregress(
                            valid[immune_col].values, valid[mfi_col].values
                        )
                        x_line = np.linspace(valid[immune_col].min(), valid[immune_col].max(), 100)
                        ax.plot(x_line, slope * x_line + intercept, '--', color=color,
                               alpha=0.6, linewidth=1.5,
                               label=f'{group}: r={r:.2f}, p={p:.3f}')
                    except Exception:
                        pass

            pop_name = immune_col.replace('n_', '').replace('_in_tumor', '')
            ax.set_xlabel(f'{pop_name} count in tumor', fontweight='bold')
            ax.set_ylabel(f'Median pERK MFI', fontweight='bold')
            ax.set_title(f'Per-tumor: pERK MFI vs {pop_name}',
                        fontsize=12, fontweight='bold')
            ax.legend(frameon=True, fontsize=9)

            plt.tight_layout()
            safe_col = immune_col.replace('+', 'pos').replace('-', 'neg')
            plot_path = self.plots_dir / f'per_tumor_perk_mfi_vs_{safe_col}.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        print(f"    ✓ Saved per-tumor pERK MFI summary plots")
