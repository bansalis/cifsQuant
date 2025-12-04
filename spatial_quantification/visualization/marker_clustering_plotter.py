"""
Marker Clustering Visualization
Generate plots for marker clustering analysis with randomization testing
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings


class MarkerClusteringPlotter:
    """
    Generate visualizations for marker clustering analysis.

    Creates:
    - Observed vs random distribution plots
    - P-value heatmaps showing significance
    - Distance distribution comparisons
    - Marker overlap matrices with significance testing
    """

    def __init__(self, results: Dict, output_dir: Path, config: Dict):
        """
        Initialize plotter.

        Parameters
        ----------
        results : dict
            Results from MarkerClusteringAnalysis
        output_dir : Path
            Output directory for plots
        config : dict
            Configuration dictionary
        """
        self.results = results
        self.plots_dir = Path(output_dir) / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Get plotting settings
        plotting_config = config.get('plotting', {})
        self.group_colors = plotting_config.get('group_colors', {
            'KPT': '#E41A1C', 'KPNT': '#377EB8'
        })

    def plot_all(self):
        """Generate all clustering plots."""
        print("\n  Generating marker clustering visualizations...")

        # 1. Observed vs random clustering metrics
        if 'randomization_tests' in self.results:
            self._plot_observed_vs_random()

        # 2. Clustering significance heatmaps
        if 'randomization_tests' in self.results:
            self._plot_clustering_significance()

        # 3. Distance distributions
        if 'randomization_tests' in self.results:
            self._plot_distance_distributions()

        # 4. Marker overlap analysis
        if 'overlap_vs_random' in self.results:
            self._plot_marker_overlap()

        # 5. Co-localization heatmap
        if 'marker_colocalization' in self.results:
            self._plot_colocalization_heatmap()

        # 6. Mutual exclusivity analysis
        if 'overlap_vs_random' in self.results:
            self._plot_mutual_exclusivity()

        print("  ✓ All plots generated")

    def _plot_observed_vs_random(self):
        """Plot observed clustering metrics vs random null distribution."""
        df = self.results['randomization_tests']

        # Find metrics with both observed and random values
        metric_cols = [col for col in df.columns if col.endswith('_observed')]
        metric_names = [col.replace('_observed', '') for col in metric_cols]

        # Plot for nearest neighbor distance (most interpretable)
        if 'mean_nn_distance_observed' in df.columns:
            for group in df['main_group'].unique():
                if pd.isna(group) or group == '':
                    continue

                group_df = df[df['main_group'] == group]

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                # Plot 1: Observed vs Random Mean
                ax = axes[0]
                ax.scatter(group_df['mean_nn_distance_random_mean'],
                          group_df['mean_nn_distance_observed'],
                          alpha=0.7, s=80, edgecolors='black', linewidth=2,
                          c=group_df['mean_nn_distance_significant'].map({True: 'red', False: 'gray'}))

                # Add diagonal line
                max_val = max(group_df['mean_nn_distance_random_mean'].max(),
                            group_df['mean_nn_distance_observed'].max())
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2, label='Random expectation')

                ax.set_xlabel('Random Mean NN Distance (μm)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Observed NN Distance (μm)', fontsize=14, fontweight='bold')
                ax.set_title(f'{group}\nObserved vs Random Clustering', fontsize=14, fontweight='bold')
                ax.legend(fontsize=12)
                ax.grid(False)
                ax.tick_params(axis='both', labelsize=12)

                for spine in ax.spines.values():
                    spine.set_linewidth(2)

                # Plot 2: Z-scores
                ax = axes[1]

                z_scores = group_df['mean_nn_distance_z_score'].values
                markers = group_df['marker'].values
                colors = ['red' if sig else 'gray'
                         for sig in group_df['mean_nn_distance_significant']]

                y_pos = np.arange(len(markers))
                bars = ax.barh(y_pos, z_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

                ax.set_yticks(y_pos)
                ax.set_yticklabels(markers, fontsize=12)
                ax.set_xlabel('Z-score', fontsize=14, fontweight='bold')
                ax.set_title('Clustering Z-scores\n(positive = more clustered than random)',
                           fontsize=14, fontweight='bold')
                ax.axvline(0, color='black', linestyle='-', linewidth=2)
                ax.axvline(1.96, color='red', linestyle='--', linewidth=2, alpha=0.5, label='p=0.05')
                ax.axvline(-1.96, color='red', linestyle='--', linewidth=2, alpha=0.5)
                ax.legend(fontsize=11)
                ax.grid(False)
                ax.tick_params(axis='both', labelsize=12)

                for spine in ax.spines.values():
                    spine.set_linewidth(2)

                plt.tight_layout()
                plt.savefig(self.plots_dir / f'observed_vs_random_clustering_{group}.png',
                          dpi=300, bbox_inches='tight')
                plt.close()

        print("    ✓ Observed vs random clustering plots")

    def _plot_clustering_significance(self):
        """Create heatmap showing which markers are significantly clustered."""
        df = self.results['randomization_tests']

        # Create matrix of significance for nearest neighbor distance
        if 'mean_nn_distance_significant' not in df.columns:
            return

        for group in df['main_group'].unique():
            if pd.isna(group) or group == '':
                continue

            group_df = df[df['main_group'] == group]

            # Aggregate by marker (mean across samples)
            marker_stats = []
            for marker in group_df['marker'].unique():
                marker_df = group_df[group_df['marker'] == marker]

                prop_significant = marker_df['mean_nn_distance_significant'].mean()
                mean_z_score = marker_df['mean_nn_distance_z_score'].mean()

                marker_stats.append({
                    'marker': marker,
                    'proportion_significant': prop_significant,
                    'mean_z_score': mean_z_score
                })

            stats_df = pd.DataFrame(marker_stats)

            if len(stats_df) == 0:
                continue

            # Create horizontal bar plot
            fig, ax = plt.subplots(figsize=(10, max(6, len(stats_df) * 0.5)))

            y_pos = np.arange(len(stats_df))

            # Color by z-score
            colors = ['#d62728' if z > 0 else '#1f77b4' for z in stats_df['mean_z_score']]

            bars = ax.barh(y_pos, stats_df['mean_z_score'], color=colors, alpha=0.7,
                          edgecolor='black', linewidth=2)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(stats_df['marker'], fontsize=13)
            ax.set_xlabel('Mean Z-score (Clustering Tendency)', fontsize=14, fontweight='bold')
            ax.set_title(f'{group}\nSpatial Clustering of Tumor Markers\nvs Random Null Model',
                       fontsize=14, fontweight='bold')
            ax.axvline(0, color='black', linestyle='-', linewidth=2.5)
            ax.axvline(1.96, color='gray', linestyle='--', linewidth=2, alpha=0.5)
            ax.axvline(-1.96, color='gray', linestyle='--', linewidth=2, alpha=0.5)

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#d62728', alpha=0.7, label='Clustered'),
                Patch(facecolor='#1f77b4', alpha=0.7, label='Dispersed')
            ]
            ax.legend(handles=legend_elements, fontsize=12, loc='best')

            ax.grid(False)
            ax.tick_params(axis='both', labelsize=12)

            for spine in ax.spines.values():
                spine.set_linewidth(2)

            plt.tight_layout()
            plt.savefig(self.plots_dir / f'clustering_significance_{group}.png',
                      dpi=300, bbox_inches='tight')
            plt.close()

        print("    ✓ Clustering significance plots")

    def _plot_distance_distributions(self):
        """Plot distance distributions for observed vs random."""
        df = self.results['randomization_tests']

        if 'mean_nn_distance_observed' not in df.columns:
            return

        # For each marker, plot histogram of observed vs random
        for marker in df['marker'].unique()[:5]:  # Limit to first 5 for clarity
            marker_df = df[df['marker'] == marker]

            if len(marker_df) == 0:
                continue

            fig, axes = plt.subplots(1, min(3, len(marker_df)), figsize=(15, 5))

            if len(marker_df) == 1:
                axes = [axes]

            for idx, (_, row) in enumerate(marker_df.head(3).iterrows()):
                ax = axes[idx] if len(marker_df) > 1 else axes[0]

                observed = row['mean_nn_distance_observed']
                random_mean = row['mean_nn_distance_random_mean']
                random_std = row['mean_nn_distance_random_std']

                # Plot normal distribution for random
                x = np.linspace(max(0, random_mean - 4*random_std),
                              random_mean + 4*random_std, 100)
                from scipy.stats import norm
                y = norm.pdf(x, random_mean, random_std)

                ax.plot(x, y, 'gray', linewidth=3, label='Random null model', alpha=0.7)
                ax.fill_between(x, y, alpha=0.2, color='gray')

                # Mark observed value
                ax.axvline(observed, color='red', linewidth=3, label='Observed', linestyle='--')

                # Shade significant region
                if row['mean_nn_distance_significant']:
                    if observed < random_mean:
                        ax.axvspan(0, observed, alpha=0.2, color='red')
                    else:
                        ax.axvspan(observed, x.max(), alpha=0.2, color='red')

                ax.set_xlabel('Nearest Neighbor Distance (μm)', fontsize=13, fontweight='bold')
                ax.set_ylabel('Density', fontsize=13, fontweight='bold')
                ax.set_title(f'{row["sample_id"]}\n{marker}',
                           fontsize=13, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(False)
                ax.tick_params(labelsize=11)

                for spine in ax.spines.values():
                    spine.set_linewidth(2)

            plt.tight_layout()
            plt.savefig(self.plots_dir / f'distance_distribution_{marker}.png',
                      dpi=300, bbox_inches='tight')
            plt.close()

        print("    ✓ Distance distribution plots")

    def _plot_marker_overlap(self):
        """Plot marker overlap observed vs random."""
        df = self.results['overlap_vs_random']

        for group in df['main_group'].unique():
            if pd.isna(group) or group == '':
                continue

            group_df = df[df['main_group'] == group]

            if len(group_df) == 0:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Plot 1: Observed vs expected overlap
            ax = axes[0]

            significant = group_df['overlap_significant']
            ax.scatter(group_df['random_overlap_mean'],
                      group_df['observed_overlap'],
                      alpha=0.7, s=80, edgecolors='black', linewidth=2,
                      c=significant.map({True: 'red', False: 'gray'}))

            # Add diagonal
            max_val = max(group_df['random_overlap_mean'].max(),
                        group_df['observed_overlap'].max())
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2,
                   label='Random expectation')

            ax.set_xlabel('Random Expected Overlap (# cells)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Observed Overlap (# cells)', fontsize=14, fontweight='bold')
            ax.set_title(f'{group}\nMarker Overlap vs Random', fontsize=14, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(False)
            ax.tick_params(labelsize=12)

            for spine in ax.spines.values():
                spine.set_linewidth(2)

            # Plot 2: P-values
            ax = axes[1]

            p_values = group_df['overlap_p_value'].values
            pair_labels = [f"{r['marker1']}\nvs\n{r['marker2']}"
                          for _, r in group_df.iterrows()]

            y_pos = np.arange(len(pair_labels))
            colors = ['red' if p < 0.05 or p > 0.95 else 'gray' for p in p_values]

            bars = ax.barh(y_pos, p_values, color=colors, alpha=0.7,
                          edgecolor='black', linewidth=2)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(pair_labels, fontsize=10)
            ax.set_xlabel('P-value (overlap >= observed)', fontsize=14, fontweight='bold')
            ax.set_title('Marker Overlap Significance', fontsize=14, fontweight='bold')
            ax.axvline(0.05, color='red', linestyle='--', linewidth=2, alpha=0.5)
            ax.axvline(0.95, color='red', linestyle='--', linewidth=2, alpha=0.5)
            ax.grid(False)
            ax.tick_params(labelsize=11)

            for spine in ax.spines.values():
                spine.set_linewidth(2)

            plt.tight_layout()
            plt.savefig(self.plots_dir / f'marker_overlap_vs_random_{group}.png',
                      dpi=300, bbox_inches='tight')
            plt.close()

        print("    ✓ Marker overlap vs random plots")

    def _plot_colocalization_heatmap(self):
        """Create heatmap of marker co-localization distances."""
        df = self.results['marker_colocalization']

        for group in df['main_group'].unique():
            if pd.isna(group) or group == '':
                continue

            group_df = df[df['main_group'] == group]

            # Get unique markers
            markers = sorted(set(group_df['marker1'].unique()) | set(group_df['marker2'].unique()))

            # Create distance matrix
            n = len(markers)
            distance_matrix = np.full((n, n), np.nan)

            for _, row in group_df.iterrows():
                i = markers.index(row['marker1'])
                j = markers.index(row['marker2'])
                dist = row['mean_colocalization_distance']

                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(10, 9))

            im = ax.imshow(distance_matrix, cmap='YlOrRd_r', aspect='auto')

            # Set ticks
            ax.set_xticks(np.arange(n))
            ax.set_yticks(np.arange(n))
            ax.set_xticklabels(markers, fontsize=12, rotation=45, ha='right')
            ax.set_yticklabels(markers, fontsize=12)

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Co-localization Distance (μm)', fontsize=13, fontweight='bold')
            cbar.ax.tick_params(labelsize=11)

            # Add values
            for i in range(n):
                for j in range(n):
                    if not np.isnan(distance_matrix[i, j]):
                        text = ax.text(j, i, f'{distance_matrix[i, j]:.0f}',
                                     ha='center', va='center', color='black',
                                     fontsize=10, fontweight='bold')

            ax.set_title(f'{group}\nMarker Co-localization Distances',
                       fontsize=14, fontweight='bold')

            plt.tight_layout()
            plt.savefig(self.plots_dir / f'colocalization_heatmap_{group}.png',
                      dpi=300, bbox_inches='tight')
            plt.close()

        print("    ✓ Co-localization heatmap")

    def _plot_mutual_exclusivity(self):
        """Plot markers that are mutually exclusive."""
        df = self.results['overlap_vs_random']

        for group in df['main_group'].unique():
            if pd.isna(group) or group == '':
                continue

            group_df = df[df['main_group'] == group]

            # Calculate enrichment/depletion score
            group_df = group_df.copy()
            group_df['overlap_ratio'] = (group_df['observed_overlap'] /
                                         group_df['random_overlap_mean'])

            # Sort by overlap ratio
            group_df = group_df.sort_values('overlap_ratio')

            if len(group_df) == 0:
                continue

            fig, ax = plt.subplots(figsize=(10, max(6, len(group_df) * 0.4)))

            pair_labels = [f"{r['marker1']} + {r['marker2']}"
                          for _, r in group_df.iterrows()]
            ratios = group_df['overlap_ratio'].values

            y_pos = np.arange(len(pair_labels))

            # Color by mutual exclusivity
            colors = []
            for ratio, sig in zip(ratios, group_df['overlap_significant']):
                if sig:
                    if ratio < 1:
                        colors.append('#1f77b4')  # Blue for mutually exclusive
                    else:
                        colors.append('#d62728')  # Red for co-localized
                else:
                    colors.append('gray')

            bars = ax.barh(y_pos, ratios, color=colors, alpha=0.7,
                          edgecolor='black', linewidth=2)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(pair_labels, fontsize=11)
            ax.set_xlabel('Observed / Expected Overlap Ratio', fontsize=14, fontweight='bold')
            ax.set_title(f'{group}\nMarker Co-occurrence vs Mutual Exclusivity',
                       fontsize=14, fontweight='bold')
            ax.axvline(1, color='black', linestyle='-', linewidth=2.5, label='Random expectation')

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#d62728', alpha=0.7, label='Co-localized'),
                Patch(facecolor='#1f77b4', alpha=0.7, label='Mutually exclusive'),
                Patch(facecolor='gray', alpha=0.7, label='Not significant')
            ]
            ax.legend(handles=legend_elements, fontsize=11, loc='best')

            ax.grid(False)
            ax.tick_params(labelsize=11)

            for spine in ax.spines.values():
                spine.set_linewidth(2)

            plt.tight_layout()
            plt.savefig(self.plots_dir / f'mutual_exclusivity_{group}.png',
                      dpi=300, bbox_inches='tight')
            plt.close()

        print("    ✓ Mutual exclusivity plots")
