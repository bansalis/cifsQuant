"""
Coexpression Analysis
Analyze coexpression patterns of tumor markers (pERK, NINJA, Ki67)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings


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
        """Calculate triple coexpression (all three markers)."""
        tumor_col = 'is_Tumor'

        if tumor_col not in self.adata.obs.columns:
            return

        results = []

        marker_names = list(self.markers.keys())

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

            # All three markers
            all_cols_present = all(self.markers[m] in tumor_data.columns for m in marker_names)

            if all_cols_present:
                # Triple positive
                triple_pos = (tumor_data[self.markers['pERK']] &
                             tumor_data[self.markers['NINJA']] &
                             tumor_data[self.markers['Ki67']]).sum()

                # All possible combinations
                perk_only = (tumor_data[self.markers['pERK']] &
                            ~tumor_data[self.markers['NINJA']] &
                            ~tumor_data[self.markers['Ki67']]).sum()

                ninja_only = (~tumor_data[self.markers['pERK']] &
                             tumor_data[self.markers['NINJA']] &
                             ~tumor_data[self.markers['Ki67']]).sum()

                ki67_only = (~tumor_data[self.markers['pERK']] &
                            ~tumor_data[self.markers['NINJA']] &
                            tumor_data[self.markers['Ki67']]).sum()

                perk_ninja = (tumor_data[self.markers['pERK']] &
                             tumor_data[self.markers['NINJA']] &
                             ~tumor_data[self.markers['Ki67']]).sum()

                perk_ki67 = (tumor_data[self.markers['pERK']] &
                            ~tumor_data[self.markers['NINJA']] &
                            tumor_data[self.markers['Ki67']]).sum()

                ninja_ki67 = (~tumor_data[self.markers['pERK']] &
                             tumor_data[self.markers['NINJA']] &
                             tumor_data[self.markers['Ki67']]).sum()

                none = (~tumor_data[self.markers['pERK']] &
                       ~tumor_data[self.markers['NINJA']] &
                       ~tumor_data[self.markers['Ki67']]).sum()

                result['triple_positive_count'] = int(triple_pos)
                result['triple_positive_percent'] = triple_pos / n_tumor_cells * 100
                result['pERK_only_count'] = int(perk_only)
                result['NINJA_only_count'] = int(ninja_only)
                result['Ki67_only_count'] = int(ki67_only)
                result['pERK_NINJA_only_count'] = int(perk_ninja)
                result['pERK_Ki67_only_count'] = int(perk_ki67)
                result['NINJA_Ki67_only_count'] = int(ninja_ki67)
                result['triple_negative_count'] = int(none)

            results.append(result)

        if results:
            df = pd.DataFrame(results)
            self.results['triple_coexpression'] = df
            print(f"    ✓ Calculated triple coexpression for {len(results)} samples")

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

        # Plot 3: Triple coexpression pie charts
        if 'triple_coexpression' in self.results:
            self._plot_triple_expression()

        # Plot 4: Coexpression heatmap
        if 'pairwise_coexpression' in self.results:
            self._plot_coexpression_heatmap()

    def _plot_single_markers(self):
        """Plot single marker frequencies over time."""
        df = self.results['single_marker_frequencies']

        fig, ax = plt.subplots(figsize=(10, 6))

        groups = sorted(df['main_group'].unique())
        colors = {'KPT': '#E41A1C', 'KPNT': '#377EB8'}
        markers_plot = ['pERK', 'NINJA', 'Ki67']
        linestyles = ['-', '--', ':']

        for group in groups:
            group_data = df[df['main_group'] == group]

            for marker, ls in zip(markers_plot, linestyles):
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
        """Plot pairwise coexpression over time."""
        df = self.results['pairwise_coexpression']

        pairs = [('pERK', 'NINJA'), ('pERK', 'Ki67'), ('NINJA', 'Ki67')]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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

    def _plot_triple_expression(self):
        """Plot triple coexpression patterns."""
        df = self.results['triple_coexpression']

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
