"""Neighborhood Enrichment Permutation Testing Visualization."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class NeighborhoodPermutationPlotter:
    """Plots for neighborhood enrichment permutation testing results."""

    def __init__(self, output_dir: Path, config: Dict):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.dpi = config.get('output', {}).get('dpi', 300)
        plotting = config.get('plotting', {})
        self.group_colors = plotting.get('group_colors', {})
        sns.set_style('whitegrid')
        plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                             'savefig.dpi': self.dpi, 'figure.dpi': 150})

    def generate_all_plots(self, results: Dict):
        """Generate all neighborhood permutation plots."""
        print("\n  Generating neighborhood permutation plots...")

        if 'aggregate_enrichment_matrix' in results:
            self.plot_enrichment_heatmap(results['aggregate_enrichment_matrix'], 'all_samples')

        if 'group_enrichment_matrices' in results:
            for group, matrix in results['group_enrichment_matrices'].items():
                self.plot_enrichment_heatmap(matrix, f'group_{group}')

        if 'pairwise_enrichment' in results and not results['pairwise_enrichment'].empty:
            self.plot_per_sample_heatmaps(results['pairwise_enrichment'])
            self.plot_top_interactions(results['pairwise_enrichment'])
            self.plot_interaction_dotplot(results['pairwise_enrichment'])

        print("  Done.")

    def plot_enrichment_heatmap(self, matrix: pd.DataFrame, label: str):
        """Plot enrichment z-score heatmap (clustermap)."""
        if matrix.empty:
            return

        n = max(matrix.shape)
        figsize = (max(8, n * 0.6), max(7, n * 0.5))

        try:
            g = sns.clustermap(matrix.fillna(0), cmap='RdBu_r', center=0,
                              figsize=figsize, dendrogram_ratio=0.15,
                              cbar_kws={'label': 'Z-score (enrichment / depletion)'},
                              linewidths=0.5, linecolor='white',
                              xticklabels=True, yticklabels=True)
            g.fig.suptitle(f'Neighborhood Enrichment - {label}', y=1.02, fontsize=14)
            plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=8, rotation=45, ha='right')
            plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8)
            g.savefig(self.plots_dir / f'enrichment_heatmap_{label}.png',
                     dpi=self.dpi, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    WARNING: Could not create clustermap for {label}: {e}")
            # Fallback to simple heatmap
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(matrix.fillna(0), cmap='RdBu_r', center=0, ax=ax,
                       linewidths=0.5, cbar_kws={'label': 'Z-score'})
            ax.set_title(f'Neighborhood Enrichment - {label}')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'enrichment_heatmap_{label}.png',
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()

    def plot_per_sample_heatmaps(self, df: pd.DataFrame, max_samples: int = 12):
        """Plot enrichment heatmaps for individual samples."""
        samples = df['sample_id'].unique()
        if len(samples) > max_samples:
            samples = samples[:max_samples]

        for sample in samples:
            sdf = df[df['sample_id'] == sample]
            pivot = sdf.pivot_table(values='z_score', index='cell_type_a',
                                     columns='cell_type_b', aggfunc='mean')
            if pivot.empty:
                continue

            n = max(pivot.shape)
            fig, ax = plt.subplots(figsize=(max(6, n * 0.5), max(5, n * 0.4)))
            sns.heatmap(pivot.fillna(0), cmap='RdBu_r', center=0, ax=ax,
                       linewidths=0.5, cbar_kws={'label': 'Z-score'},
                       xticklabels=True, yticklabels=True)
            ax.set_title(f'Neighborhood Enrichment - {sample}', fontsize=11)
            plt.setp(ax.get_xticklabels(), fontsize=7, rotation=45, ha='right')
            plt.setp(ax.get_yticklabels(), fontsize=7)
            plt.tight_layout()
            clean = sample.replace('/', '_').replace(' ', '_')
            plt.savefig(self.plots_dir / f'enrichment_{clean}.png',
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()

    def plot_top_interactions(self, df: pd.DataFrame, top_n: int = 20):
        """Bar plot of top enriched and depleted interactions."""
        mean_z = df.groupby(['cell_type_a', 'cell_type_b'])['z_score'].mean().reset_index()
        mean_z['pair'] = mean_z['cell_type_a'] + ' → ' + mean_z['cell_type_b']
        # Remove self-interactions
        mean_z = mean_z[mean_z['cell_type_a'] != mean_z['cell_type_b']]

        if mean_z.empty:
            return

        # Top enriched
        top_enriched = mean_z.nlargest(top_n, 'z_score')
        top_depleted = mean_z.nsmallest(top_n, 'z_score')
        combined = pd.concat([top_enriched, top_depleted]).drop_duplicates()

        fig, ax = plt.subplots(figsize=(10, max(6, len(combined) * 0.3)))
        combined = combined.sort_values('z_score')
        colors = ['#E41A1C' if z > 0 else '#377EB8' for z in combined['z_score']]
        ax.barh(combined['pair'], combined['z_score'], color=colors, alpha=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Mean Z-score')
        ax.set_title(f'Top {top_n} Enriched/Depleted Interactions')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'top_interactions.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_interaction_dotplot(self, df: pd.DataFrame):
        """Dot plot: size = -log10(p), color = z-score, grid = cell type pairs."""
        mean_df = df.groupby(['cell_type_a', 'cell_type_b']).agg(
            z_score=('z_score', 'mean'),
            p_value=('p_value', 'mean')
        ).reset_index()
        # Remove self-interactions
        mean_df = mean_df[mean_df['cell_type_a'] != mean_df['cell_type_b']]

        if mean_df.empty or len(mean_df) < 2:
            return

        mean_df['neg_log_p'] = -np.log10(mean_df['p_value'].clip(lower=1e-10))
        # Limit size range
        max_size = mean_df['neg_log_p'].max()
        mean_df['dot_size'] = (mean_df['neg_log_p'] / max(max_size, 1)) * 200 + 20

        types_a = sorted(mean_df['cell_type_a'].unique())
        types_b = sorted(mean_df['cell_type_b'].unique())

        fig, ax = plt.subplots(figsize=(max(8, len(types_b) * 0.6), max(6, len(types_a) * 0.5)))

        # Map types to indices
        a_map = {t: i for i, t in enumerate(types_a)}
        b_map = {t: i for i, t in enumerate(types_b)}

        scatter = ax.scatter(
            [b_map[r['cell_type_b']] for _, r in mean_df.iterrows()],
            [a_map[r['cell_type_a']] for _, r in mean_df.iterrows()],
            c=mean_df['z_score'], s=mean_df['dot_size'],
            cmap='RdBu_r', vmin=-mean_df['z_score'].abs().max(),
            vmax=mean_df['z_score'].abs().max(),
            edgecolors='black', linewidths=0.5, alpha=0.8
        )

        ax.set_xticks(range(len(types_b)))
        ax.set_xticklabels(types_b, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(types_a)))
        ax.set_yticklabels(types_a, fontsize=8)
        ax.set_title('Neighborhood Interactions (size=-log10(p), color=z-score)')
        plt.colorbar(scatter, ax=ax, label='Z-score', shrink=0.8)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'interaction_dotplot.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
