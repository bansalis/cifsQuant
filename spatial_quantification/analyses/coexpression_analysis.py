"""
Coexpression Analysis
Analyze coexpression patterns of tumor markers (configurable)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from spatial_quantification.visualization.plot_utils import (
        detect_plot_type,
        plot_with_stats,
        create_dual_plots
    )
    HAS_PLOT_UTILS = True
except ImportError:
    HAS_PLOT_UTILS = False
    warnings.warn("Plot utilities not available - using basic plots")


class CoexpressionAnalysis:
    """
    Analyze coexpression patterns of tumor markers.

    Key features:
    - Pairwise coexpression rates (e.g., pERK+ AND Ki67+)
    - Triple coexpression (pERK+ AND NINJA+ AND Ki67+)
    - Temporal dynamics of coexpression
    - Group comparisons
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        """
        Initialize coexpression analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data with phenotypes
        config : dict
            Configuration dictionary
        output_dir : Path
            Output directory
        """
        self.adata = adata
        self.config = config
        self.output_dir = Path(output_dir) / 'coexpression_analysis'
        self.plots_dir = self.output_dir / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Markers to analyze - read from config
        coexp_config = config.get('coexpression_analysis', {})
        marker_config = coexp_config.get('markers', {})

        if marker_config:
            self.markers = marker_config
        else:
            # Default markers
            self.markers = {
                'pERK': 'is_PERK',
                'NINJA': 'is_AGFP',
                'Ki67': 'is_KI67'
            }

        # Results storage
        self.results = {}

    def run(self):
        """Run complete coexpression analysis."""
        print("\n" + "="*80)
        print("COEXPRESSION ANALYSIS")
        print("="*80)

        # Calculate single marker frequencies
        print("\n1. Calculating single marker frequencies...")
        self._calculate_single_marker_frequencies()

        # Calculate pairwise coexpression
        print("\n2. Calculating pairwise coexpression...")
        self._calculate_pairwise_coexpression()

        # Calculate triple coexpression
        print("\n3. Calculating triple coexpression...")
        self._calculate_triple_coexpression()

        # Calculate coexpression per tumor structure
        print("\n4. Calculating per-tumor coexpression...")
        self._calculate_per_tumor_coexpression()

        # Save results
        self._save_results()

        # Generate plots
        print("\n5. Generating visualizations...")
        self._generate_plots()

        print("\n✓ Coexpression analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
        print(f"  Plots saved to: {self.plots_dir}/")
        print("="*80 + "\n")

        return self.results

    def _calculate_single_marker_frequencies(self):
        """Calculate frequency of each marker in tumor cells."""
        tumor_col = 'is_Tumor'

        if tumor_col not in self.adata.obs.columns:
            warnings.warn("Tumor phenotype not found")
            return

        results = []

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]

            # Get tumor cells
            tumor_mask = sample_data[tumor_col]
            tumor_data = sample_data[tumor_mask]

            if len(tumor_data) == 0:
                continue

            n_tumor_cells = len(tumor_data)

            result = {
                'sample_id': sample,
                'n_tumor_cells': n_tumor_cells,
                'timepoint': tumor_data['timepoint'].iloc[0] if 'timepoint' in tumor_data.columns else np.nan,
                'group': tumor_data['group'].iloc[0] if 'group' in tumor_data.columns else '',
                'main_group': tumor_data['main_group'].iloc[0] if 'main_group' in tumor_data.columns else ''
            }

            # Calculate frequency for each marker
            for marker_name, marker_col in self.markers.items():
                if marker_col in tumor_data.columns:
                    n_positive = tumor_data[marker_col].sum()
                    freq = n_positive / n_tumor_cells * 100 if n_tumor_cells > 0 else 0
                    result[f'{marker_name}_count'] = int(n_positive)
                    result[f'{marker_name}_percent'] = freq

            results.append(result)

        if results:
            df = pd.DataFrame(results)
            self.results['single_marker_frequencies'] = df
            print(f"    ✓ Calculated frequencies for {len(results)} samples")

    def _calculate_pairwise_coexpression(self):
        """Calculate pairwise coexpression rates."""
        tumor_col = 'is_Tumor'

        if tumor_col not in self.adata.obs.columns:
            return

        results = []

        marker_names = list(self.markers.keys())
        marker_pairs = list(combinations(marker_names, 2))

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]

            tumor_mask = sample_data[tumor_col]
            tumor_data = sample_data[tumor_mask]

            if len(tumor_data) == 0:
                continue

            n_tumor_cells = len(tumor_data)

            result = {
                'sample_id': sample,
                'n_tumor_cells': n_tumor_cells,
                'timepoint': tumor_data['timepoint'].iloc[0] if 'timepoint' in tumor_data.columns else np.nan,
                'group': tumor_data['group'].iloc[0] if 'group' in tumor_data.columns else '',
                'main_group': tumor_data['main_group'].iloc[0] if 'main_group' in tumor_data.columns else ''
            }

            # Calculate pairwise coexpression
            for marker1, marker2 in marker_pairs:
                col1 = self.markers[marker1]
                col2 = self.markers[marker2]

                if col1 in tumor_data.columns and col2 in tumor_data.columns:
                    # Both positive
                    both_pos = (tumor_data[col1] & tumor_data[col2]).sum()

                    # One or both positive
                    either_pos = (tumor_data[col1] | tumor_data[col2]).sum()

                    # Percent of tumor cells
                    percent_both = both_pos / n_tumor_cells * 100 if n_tumor_cells > 0 else 0

                    # Percent of marker1+ cells that are also marker2+
                    marker1_pos = tumor_data[col1].sum()
                    percent_of_marker1 = both_pos / marker1_pos * 100 if marker1_pos > 0 else 0

                    # Percent of marker2+ cells that are also marker1+
                    marker2_pos = tumor_data[col2].sum()
                    percent_of_marker2 = both_pos / marker2_pos * 100 if marker2_pos > 0 else 0

                    # Jaccard index (similarity)
                    jaccard = both_pos / either_pos if either_pos > 0 else 0

                    result[f'{marker1}_and_{marker2}_count'] = int(both_pos)
                    result[f'{marker1}_and_{marker2}_percent_of_tumor'] = percent_both
                    result[f'{marker1}_and_{marker2}_percent_of_{marker1}'] = percent_of_marker1
                    result[f'{marker1}_and_{marker2}_percent_of_{marker2}'] = percent_of_marker2
                    result[f'{marker1}_and_{marker2}_jaccard'] = jaccard

            results.append(result)

        if results:
            df = pd.DataFrame(results)
            self.results['pairwise_coexpression'] = df
            print(f"    ✓ Calculated pairwise coexpression for {len(results)} samples")

    def _calculate_triple_coexpression(self):
        """Calculate triple/multi-marker coexpression (all markers)."""
        tumor_col = 'is_Tumor'

        if tumor_col not in self.adata.obs.columns:
            return

        results = []

        marker_names = list(self.markers.keys())
        n_markers = len(marker_names)

        # If we have exactly 3 markers, use the original approach with all combinations
        # If we have more or fewer, we'll calculate multi-marker positive and individual counts

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]

            tumor_mask = sample_data[tumor_col]
            tumor_data = sample_data[tumor_mask]

            if len(tumor_data) == 0:
                continue

            n_tumor_cells = len(tumor_data)

            result = {
                'sample_id': sample,
                'n_tumor_cells': n_tumor_cells,
                'timepoint': tumor_data['timepoint'].iloc[0] if 'timepoint' in tumor_data.columns else np.nan,
                'group': tumor_data['group'].iloc[0] if 'group' in tumor_data.columns else '',
                'main_group': tumor_data['main_group'].iloc[0] if 'main_group' in tumor_data.columns else ''
            }

            # Check all markers are present
            all_cols_present = all(self.markers[m] in tumor_data.columns for m in marker_names)

            if all_cols_present:
                # Calculate multi-marker positive (all markers positive)
                multi_pos_mask = pd.Series(True, index=tumor_data.index)
                for marker_name in marker_names:
                    multi_pos_mask &= tumor_data[self.markers[marker_name]]

                multi_pos = multi_pos_mask.sum()

                result[f'all_{n_markers}_positive_count'] = int(multi_pos)
                result[f'all_{n_markers}_positive_percent'] = multi_pos / n_tumor_cells * 100

                # Calculate individual marker-only counts
                for marker_name in marker_names:
                    only_mask = tumor_data[self.markers[marker_name]].copy()
                    # Exclude other markers
                    for other_marker in marker_names:
                        if other_marker != marker_name:
                            only_mask &= ~tumor_data[self.markers[other_marker]]

                    only_count = only_mask.sum()
                    result[f'{marker_name}_only_count'] = int(only_count)

                # Calculate pairwise combinations (if n_markers >= 2)
                if n_markers >= 2:
                    for i, m1 in enumerate(marker_names):
                        for m2 in marker_names[i+1:]:
                            # Both m1 and m2, but no others
                            pair_mask = tumor_data[self.markers[m1]] & tumor_data[self.markers[m2]]
                            for other_marker in marker_names:
                                if other_marker != m1 and other_marker != m2:
                                    pair_mask &= ~tumor_data[self.markers[other_marker]]

                            pair_count = pair_mask.sum()
                            result[f'{m1}_{m2}_only_count'] = int(pair_count)

                # Calculate all negative (none of the markers)
                none_mask = pd.Series(True, index=tumor_data.index)
                for marker_name in marker_names:
                    none_mask &= ~tumor_data[self.markers[marker_name]]

                none_count = none_mask.sum()
                result['all_negative_count'] = int(none_count)

            results.append(result)

        if results:
            df = pd.DataFrame(results)
            self.results['multi_marker_coexpression'] = df
            print(f"    ✓ Calculated multi-marker coexpression for {len(results)} samples ({n_markers} markers)")

    def _calculate_per_tumor_coexpression(self):
        """Calculate coexpression per tumor structure."""
        # This requires tumor structures - check if available
        # For now, we'll skip if not implemented
        print("    ⚠ Per-tumor coexpression requires tumor structure detection (skipped for now)")

    def _save_results(self):
        """Save all results to files."""
        for name, df in self.results.items():
            output_path = self.output_dir / f'{name}.csv'
            df.to_csv(output_path, index=False)

        print(f"\n  ✓ Saved {len(self.results)} result datasets")

    def _generate_plots(self):
        """Generate visualization plots."""
        # Plot 1: Single marker frequencies over time
        if 'single_marker_frequencies' in self.results:
            self._plot_single_markers()

        # Plot 2: Pairwise coexpression over time
        if 'pairwise_coexpression' in self.results:
            self._plot_pairwise()

        # Plot 3: Multi-marker coexpression patterns
        if 'multi_marker_coexpression' in self.results:
            self._plot_multi_marker_expression()

        # Plot 4: Coexpression heatmap
        if 'pairwise_coexpression' in self.results:
            self._plot_coexpression_heatmap()

    def _plot_single_markers(self):
        """Plot single marker frequencies with adaptive plot type and stats."""
        df = self.results['single_marker_frequencies']

        marker_names = list(self.markers.keys())

        if HAS_PLOT_UTILS:
            # Use adaptive plotting for each marker
            for marker in marker_names:
                col = f'{marker}_percent'
                if col not in df.columns:
                    continue

                output_base = str(self.plots_dir / f'single_marker_{marker}_frequency')
                create_dual_plots(
                    df,
                    value_col=col,
                    group_col='main_group',
                    timepoint_col='timepoint',
                    group_colors={'KPT': '#E41A1C', 'KPNT': '#377EB8'},
                    title_base=f'{marker} Frequency',
                    ylabel='% of Tumor Cells',
                    xlabel='Time (weeks)',
                    output_path_base=output_base
                )
            print(f"    ✓ Saved single marker plots (with and without stats)")
        else:
            # Fallback to basic plotting
            fig, ax = plt.subplots(figsize=(10, 6))

            groups = sorted(df['main_group'].unique())
            colors = {'KPT': '#E41A1C', 'KPNT': '#377EB8'}
            linestyles = ['-', '--', ':', '-.']

            for group in groups:
                group_data = df[df['main_group'] == group]

                for marker, ls in zip(marker_names, linestyles):
                    col = f'{marker}_percent'
                    if col not in group_data.columns:
                        continue

                    summary = group_data.groupby('timepoint')[col].agg(['mean', 'sem'])
                    timepoints = summary.index.values
                    means = summary['mean'].values
                    sems = summary['sem'].values

                    color = colors.get(group, '#000000')
                    label = f'{group} - {marker}'

                    ax.plot(timepoints, means, linestyle=ls, color=color, linewidth=2,
                           marker='o', markersize=6, label=label)
                    ax.fill_between(timepoints, means - sems, means + sems,
                                   alpha=0.15, color=color)

            ax.set_xlabel('Time (weeks)', fontsize=12, fontweight='bold')
            ax.set_ylabel('% of Tumor Cells', fontsize=12, fontweight='bold')
            ax.set_title('Single Marker Frequencies', fontsize=14, fontweight='bold')
            ax.legend(frameon=True, loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.plots_dir / 'single_marker_frequencies.png', dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_pairwise(self):
        """Plot pairwise coexpression with adaptive plot type and stats."""
        df = self.results['pairwise_coexpression']

        marker_names = list(self.markers.keys())
        pairs = list(combinations(marker_names, 2))

        if HAS_PLOT_UTILS:
            # Use adaptive plotting for each pair
            for m1, m2 in pairs:
                col = f'{m1}_and_{m2}_percent_of_tumor'

                if col not in df.columns:
                    continue

                output_base = str(self.plots_dir / f'pairwise_{m1}_and_{m2}')
                create_dual_plots(
                    df,
                    value_col=col,
                    group_col='main_group',
                    timepoint_col='timepoint',
                    group_colors={'KPT': '#E41A1C', 'KPNT': '#377EB8'},
                    title_base=f'{m1}+ AND {m2}+ Coexpression',
                    ylabel='% of Tumor Cells',
                    xlabel='Time (weeks)',
                    output_path_base=output_base
                )
            print(f"    ✓ Saved pairwise coexpression plots (with and without stats)")
        else:
            # Fallback to basic plotting
            n_pairs = len(pairs)
            fig, axes = plt.subplots(1, n_pairs, figsize=(6*n_pairs, 5))

            if n_pairs == 1:
                axes = [axes]

            groups = sorted(df['main_group'].unique())
            colors = {'KPT': '#E41A1C', 'KPNT': '#377EB8'}

            for idx, (m1, m2) in enumerate(pairs):
                ax = axes[idx]
                col = f'{m1}_and_{m2}_percent_of_tumor'

                if col not in df.columns:
                    continue

                for group in groups:
                    group_data = df[df['main_group'] == group]

                    summary = group_data.groupby('timepoint')[col].agg(['mean', 'sem'])
                    timepoints = summary.index.values
                    means = summary['mean'].values
                    sems = summary['sem'].values

                    color = colors.get(group, '#000000')

                    ax.plot(timepoints, means, '-o', color=color, linewidth=2.5,
                           markersize=8, label=group)
                    ax.fill_between(timepoints, means - sems, means + sems,
                                   alpha=0.2, color=color)

                ax.set_xlabel('Time (weeks)', fontsize=11, fontweight='bold')
                ax.set_ylabel('% of Tumor Cells', fontsize=11, fontweight='bold')
                ax.set_title(f'{m1}+ AND {m2}+', fontsize=12, fontweight='bold')
                if idx == 0:
                    ax.legend(frameon=True, loc='best')
                ax.grid(True, alpha=0.3)

            plt.suptitle('Pairwise Coexpression', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'pairwise_coexpression.png', dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_multi_marker_expression(self):
        """Plot multi-marker coexpression patterns."""
        df = self.results['multi_marker_coexpression']

        # Average across all samples per group/timepoint
        groups = sorted(df['main_group'].unique())
        timepoints = sorted(df['timepoint'].unique())

        # Categories
        categories = [
            'triple_positive_count',
            'pERK_only_count',
            'NINJA_only_count',
            'Ki67_only_count',
            'pERK_NINJA_only_count',
            'pERK_Ki67_only_count',
            'NINJA_Ki67_only_count',
            'triple_negative_count'
        ]

        labels = [
            'All 3+',
            'pERK only',
            'NINJA only',
            'Ki67 only',
            'pERK+NINJA',
            'pERK+Ki67',
            'NINJA+Ki67',
            'None'
        ]

        # Check which categories are available
        available_cats = [cat for cat in categories if cat in df.columns]
        available_labels = [labels[i] for i, cat in enumerate(categories) if cat in df.columns]

        if not available_cats:
            return

        # Create stacked bar plot
        fig, axes = plt.subplots(1, len(groups), figsize=(8*len(groups), 6))
        if len(groups) == 1:
            axes = [axes]

        colors_plot = plt.cm.Set3(np.linspace(0, 1, len(available_cats)))

        for idx, group in enumerate(groups):
            ax = axes[idx]
            group_data = df[df['main_group'] == group]

            # Calculate mean percentages per timepoint
            plot_data = []
            for tp in timepoints:
                tp_data = group_data[group_data['timepoint'] == tp]
                if len(tp_data) == 0:
                    continue

                row = {'timepoint': tp}
                total = tp_data['n_tumor_cells'].sum()

                for cat in available_cats:
                    count = tp_data[cat].sum()
                    row[cat] = count / total * 100 if total > 0 else 0

                plot_data.append(row)

            plot_df = pd.DataFrame(plot_data)

            if len(plot_df) == 0:
                continue

            # Stacked bar
            bottom = np.zeros(len(plot_df))
            for cat, label, color in zip(available_cats, available_labels, colors_plot):
                if cat in plot_df.columns:
                    ax.bar(plot_df['timepoint'], plot_df[cat], bottom=bottom,
                          label=label, color=color, width=0.8)
                    bottom += plot_df[cat].values

            ax.set_xlabel('Time (weeks)', fontsize=12, fontweight='bold')
            ax.set_ylabel('% of Tumor Cells', fontsize=12, fontweight='bold')
            ax.set_title(f'{group}', fontsize=13, fontweight='bold')
            if idx == 0:
                ax.legend(frameon=True, loc='upper left', fontsize=9)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Marker Coexpression Patterns', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'triple_coexpression_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_coexpression_heatmap(self):
        """Plot heatmap of coexpression rates."""
        df = self.results['pairwise_coexpression']

        pairs = [('pERK', 'NINJA'), ('pERK', 'Ki67'), ('NINJA', 'Ki67')]

        # Calculate average coexpression per group across all timepoints
        groups = sorted(df['main_group'].unique())

        heatmap_data = []

        for group in groups:
            group_data = df[df['main_group'] == group]

            row = {'group': group}
            for m1, m2 in pairs:
                col = f'{m1}_and_{m2}_jaccard'
                if col in group_data.columns:
                    row[f'{m1}+{m2}'] = group_data[col].mean()

            heatmap_data.append(row)

        if not heatmap_data:
            return

        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df = heatmap_df.set_index('group')

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'Jaccard Index'}, ax=ax)
        ax.set_title('Coexpression Similarity (Jaccard Index)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Group', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'coexpression_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
