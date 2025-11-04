"""
Neighborhood Plotter
Comprehensive plotting for cellular neighborhood analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class NeighborhoodPlotter:
    """
    Comprehensive plotting for neighborhood analysis.

    Generates:
    - Neighborhood composition heatmaps
    - Temporal evolution plots
    - Spatial distribution visualizations
    - Group comparison plots
    """

    def __init__(self, output_dir: Path, config: Dict):
        """Initialize neighborhood plotter."""
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300

    def plot_neighborhood_composition_heatmap(self, data: pd.DataFrame,
                                             phenotypes: List[str]):
        """
        Create heatmap showing composition of each neighborhood type.

        Rows = neighborhood types
        Columns = phenotypes
        Values = fraction of phenotype in neighborhood
        """
        print("\n  Generating neighborhood composition heatmap...")

        # Get neighborhood types
        nh_types = sorted(data['neighborhood_type'].unique())

        # Build composition matrix
        composition_matrix = []
        nh_labels = []

        for nh_type in nh_types:
            nh_data = data[data['neighborhood_type'] == nh_type]

            # Average composition across samples
            compositions = []
            for pheno in phenotypes:
                col_name = f'frac_{pheno}'
                if col_name in nh_data.columns:
                    compositions.append(nh_data[col_name].mean())
                else:
                    compositions.append(0)

            composition_matrix.append(compositions)
            nh_labels.append(f"NH-{nh_type}")

        composition_matrix = np.array(composition_matrix)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(composition_matrix, cmap='YlOrRd', aspect='auto',
                      vmin=0, vmax=1)

        # Set ticks
        ax.set_xticks(range(len(phenotypes)))
        ax.set_yticks(range(len(nh_types)))
        ax.set_xticklabels(phenotypes, rotation=45, ha='right')
        ax.set_yticklabels(nh_labels)

        # Add values to cells
        for i in range(len(nh_types)):
            for j in range(len(phenotypes)):
                val = composition_matrix[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=color, fontsize=9)

        plt.colorbar(im, ax=ax, label='Fraction in Neighborhood')

        ax.set_xlabel('Cell Phenotype', fontsize=12, fontweight='bold')
        ax.set_ylabel('Neighborhood Type', fontsize=12, fontweight='bold')
        ax.set_title('Neighborhood Composition Heatmap',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        plot_path = self.plots_dir / 'neighborhood_composition_heatmap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Saved composition heatmap")

    def plot_neighborhood_abundance_over_time(self, data: pd.DataFrame,
                                             group_col: str = 'main_group',
                                             groups: List[str] = None):
        """
        Plot how neighborhood type abundance changes over time.

        Creates line plots showing fraction of each neighborhood type.
        """
        print("\n  Generating neighborhood temporal evolution plots...")

        if 'timepoint' not in data.columns:
            print("    ✗ No timepoint column found, skipping")
            return

        if groups is None:
            groups = sorted(data[group_col].unique()) if group_col in data.columns else []

        if len(groups) == 0:
            groups = ['all']
            data[group_col] = 'all'

        nh_types = sorted(data['neighborhood_type'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(nh_types)))

        # Plot per group
        for group in groups:
            fig, ax = plt.subplots(figsize=(12, 8))

            group_data = data[data[group_col] == group] if group != 'all' else data

            for nh_type, color in zip(nh_types, colors):
                nh_data = group_data[group_data['neighborhood_type'] == nh_type]

                if len(nh_data) == 0:
                    continue

                # Aggregate by timepoint
                summary = nh_data.groupby('timepoint')['fraction_of_sample'].agg(['mean', 'sem'])
                timepoints = summary.index.values
                means = summary['mean'].values
                sems = summary['sem'].values

                ax.plot(timepoints, means, '-o', color=color, linewidth=2.5,
                       markersize=8, label=f'NH-{nh_type}', zorder=10)
                ax.fill_between(timepoints, means - sems, means + sems,
                               alpha=0.2, color=color)

            ax.set_xlabel('Timepoint (days)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Fraction of Tissue', fontsize=12, fontweight='bold')
            ax.set_title(f'Neighborhood Evolution - {group}',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', ncol=2, fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            plot_path = self.plots_dir / f'neighborhood_evolution_{group}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"    ✓ Generated {len(groups)} temporal evolution plots")

    def plot_neighborhood_comparison(self, data: pd.DataFrame,
                                    nh_type: int,
                                    group_col: str = 'main_group',
                                    groups: List[str] = None):
        """
        Compare specific neighborhood type between groups over time.

        4-panel plot:
        1. Time series
        2. Box plots per timepoint
        3. Violin plots per timepoint
        4. Cell count over time
        """
        print(f"\n  Generating comparison plots for NH-{nh_type}...")

        if groups is None:
            groups = sorted(data[group_col].unique()) if group_col in data.columns else []

        nh_data = data[data['neighborhood_type'] == nh_type]

        if len(nh_data) == 0:
            print(f"    ✗ No data for NH-{nh_type}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Neighborhood Type {nh_type} - Group Comparison',
                    fontsize=16, fontweight='bold', y=0.995)

        colors = {'KPT': '#E41A1C', 'KPNT': '#377EB8', 'cis': '#4DAF4A', 'trans': '#FF7F00'}

        # Panel 1: Time series
        ax = axes[0, 0]
        for group in groups:
            group_data = nh_data[nh_data[group_col] == group]

            if len(group_data) == 0:
                continue

            summary = group_data.groupby('timepoint')['fraction_of_sample'].agg(['mean', 'sem'])
            timepoints = summary.index.values
            means = summary['mean'].values
            sems = summary['sem'].values

            color = colors.get(group, '#000000')

            ax.plot(timepoints, means, '-o', color=color, linewidth=2.5,
                   markersize=8, label=group, zorder=10)
            ax.fill_between(timepoints, means - sems, means + sems,
                           alpha=0.2, color=color)

        ax.set_xlabel('Timepoint (days)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Fraction of Tissue', fontsize=11, fontweight='bold')
        ax.set_title('Temporal Evolution', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Panel 2: Box plots
        ax = axes[0, 1]
        if 'timepoint' in nh_data.columns:
            timepoints = sorted(nh_data['timepoint'].unique())

            box_data = []
            box_positions = []
            box_colors = []
            width = 0.35

            for i, tp in enumerate(timepoints):
                tp_data = nh_data[nh_data['timepoint'] == tp]
                for j, group in enumerate(groups):
                    group_data = tp_data[tp_data[group_col] == group]['fraction_of_sample'].values
                    if len(group_data) > 0:
                        box_data.append(group_data)
                        box_positions.append(i + j * width)
                        box_colors.append(colors.get(group, '#000000'))

            if box_data:
                bp = ax.boxplot(box_data, positions=box_positions, widths=width*0.8,
                               patch_artist=True, showfliers=False)

                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)

                ax.set_xticks([i + width/2 for i in range(len(timepoints))])
                ax.set_xticklabels([str(tp) for tp in timepoints])

        ax.set_xlabel('Timepoint (days)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Fraction of Tissue', fontsize=11, fontweight='bold')
        ax.set_title('Distribution per Timepoint', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Panel 3: Violin plots
        ax = axes[1, 0]
        if 'timepoint' in nh_data.columns:
            violin_data = []
            positions = []
            violin_colors = []

            for i, tp in enumerate(timepoints):
                tp_data = nh_data[nh_data['timepoint'] == tp]
                for j, group in enumerate(groups):
                    group_data = tp_data[tp_data[group_col] == group]['fraction_of_sample'].values
                    if len(group_data) > 0:
                        violin_data.append(group_data)
                        positions.append(i + j * width)
                        violin_colors.append(colors.get(group, '#000000'))

            if violin_data:
                parts = ax.violinplot(violin_data, positions=positions, widths=width*0.8,
                                     showmeans=True, showmedians=True)

                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(violin_colors[i])
                    pc.set_alpha(0.6)

                ax.set_xticks([i + width/2 for i in range(len(timepoints))])
                ax.set_xticklabels([str(tp) for tp in timepoints])

        ax.set_xlabel('Timepoint (days)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Fraction of Tissue', fontsize=11, fontweight='bold')
        ax.set_title('Violin Plots', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Panel 4: Cell count over time
        ax = axes[1, 1]
        for group in groups:
            group_data = nh_data[nh_data[group_col] == group]

            if len(group_data) == 0:
                continue

            summary = group_data.groupby('timepoint')['n_cells'].agg(['mean', 'sem'])
            timepoints = summary.index.values
            means = summary['mean'].values
            sems = summary['sem'].values

            color = colors.get(group, '#000000')

            ax.plot(timepoints, means, '-o', color=color, linewidth=2.5,
                   markersize=8, label=group, zorder=10)
            ax.fill_between(timepoints, means - sems, means + sems,
                           alpha=0.2, color=color)

        ax.set_xlabel('Timepoint (days)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cell Count', fontsize=11, fontweight='bold')
        ax.set_title('Cell Count Evolution', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = self.plots_dir / f'neighborhood_{nh_type}_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Saved comparison plot for NH-{nh_type}")

    def plot_all_neighborhoods_summary(self, data: pd.DataFrame,
                                      phenotypes: List[str],
                                      group_col: str = 'main_group',
                                      groups: List[str] = None):
        """
        Create comprehensive summary figure showing:
        - Composition heatmap
        - Abundance over time
        - Cell counts per neighborhood
        - Dominant phenotypes
        """
        print("\n  Generating comprehensive neighborhood summary...")

        if groups is None:
            groups = sorted(data[group_col].unique()) if group_col in data.columns else ['all']
            if len(groups) == 0:
                groups = ['all']

        # Create summary per group
        for group in groups:
            group_data = data[data[group_col] == group] if group != 'all' else data

            if len(group_data) == 0:
                continue

            fig = plt.figure(figsize=(18, 12))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # Panel 1: Composition heatmap
            ax1 = fig.add_subplot(gs[0, 0])
            nh_types = sorted(group_data['neighborhood_type'].unique())

            composition_matrix = []
            for nh_type in nh_types:
                nh_data = group_data[group_data['neighborhood_type'] == nh_type]
                compositions = []
                for pheno in phenotypes:
                    col_name = f'frac_{pheno}'
                    if col_name in nh_data.columns:
                        compositions.append(nh_data[col_name].mean())
                    else:
                        compositions.append(0)
                composition_matrix.append(compositions)

            composition_matrix = np.array(composition_matrix)
            im = ax1.imshow(composition_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            ax1.set_xticks(range(len(phenotypes)))
            ax1.set_yticks(range(len(nh_types)))
            ax1.set_xticklabels(phenotypes, rotation=45, ha='right', fontsize=9)
            ax1.set_yticklabels([f'NH-{t}' for t in nh_types], fontsize=9)
            ax1.set_title('Composition', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax1, label='Fraction')

            # Panel 2: Abundance over time
            ax2 = fig.add_subplot(gs[0, 1])
            if 'timepoint' in group_data.columns:
                colors_nh = plt.cm.tab10(np.linspace(0, 1, len(nh_types)))
                for nh_type, color in zip(nh_types, colors_nh):
                    nh_data = group_data[group_data['neighborhood_type'] == nh_type]
                    summary = nh_data.groupby('timepoint')['fraction_of_sample'].mean()
                    ax2.plot(summary.index, summary.values, '-o', color=color,
                           label=f'NH-{nh_type}', linewidth=2)

                ax2.set_xlabel('Timepoint (days)', fontsize=11, fontweight='bold')
                ax2.set_ylabel('Fraction of Tissue', fontsize=11, fontweight='bold')
                ax2.set_title('Temporal Evolution', fontsize=12, fontweight='bold')
                ax2.legend(loc='best', ncol=2, fontsize=8)
                ax2.grid(True, alpha=0.3)

            # Panel 3: Cell counts
            ax3 = fig.add_subplot(gs[1, 0])
            nh_cell_counts = group_data.groupby('neighborhood_type')['n_cells'].mean()
            bars = ax3.bar(range(len(nh_cell_counts)), nh_cell_counts.values,
                          color=plt.cm.tab10(np.linspace(0, 1, len(nh_cell_counts))))
            ax3.set_xticks(range(len(nh_cell_counts)))
            ax3.set_xticklabels([f'NH-{t}' for t in nh_cell_counts.index])
            ax3.set_xlabel('Neighborhood Type', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Mean Cell Count', fontsize=11, fontweight='bold')
            ax3.set_title('Average Size', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')

            # Panel 4: Dominant phenotypes text
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis('off')

            nh_descriptions = []
            for nh_type in nh_types:
                nh_data = group_data[group_data['neighborhood_type'] == nh_type]
                dominant = nh_data['dominant_phenotypes'].iloc[0] if len(nh_data) > 0 else 'N/A'
                n_samples = len(nh_data['sample_id'].unique()) if 'sample_id' in nh_data.columns else 0
                nh_descriptions.append(f"NH-{nh_type}: {dominant}\n  ({n_samples} samples)")

            description_text = '\n\n'.join(nh_descriptions)
            ax4.text(0.1, 0.95, description_text, transform=ax4.transAxes,
                   fontsize=9, verticalalignment='top', family='monospace')
            ax4.set_title('Dominant Phenotypes', fontsize=12, fontweight='bold',
                        loc='left', pad=10)

            fig.suptitle(f'Neighborhood Summary - {group}',
                        fontsize=16, fontweight='bold', y=0.98)

            plt.tight_layout()

            plot_path = self.plots_dir / f'neighborhood_summary_{group}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"    ✓ Generated {len(groups)} summary plots")

    def plot_neighborhood_stacked_area(self, data: pd.DataFrame,
                                      group_col: str = 'main_group',
                                      groups: List[str] = None):
        """
        Create stacked area chart showing neighborhood evolution over time.

        Shows fractional abundance of each neighborhood type stacked to 100%.

        Parameters
        ----------
        data : pd.DataFrame
            Neighborhood statistics with timepoint column
        group_col : str
            Column for grouping
        groups : List[str], optional
            Groups to plot
        """
        if 'timepoint' not in data.columns:
            print("    ✗ No timepoint column, skipping stacked area chart")
            return

        if groups is None:
            groups = sorted(data[group_col].unique()) if group_col in data.columns else ['all']
            if len(groups) == 0:
                groups = ['all']

        print("\n  Generating stacked area charts...")

        nh_types = sorted(data['neighborhood_type'].unique())
        n_neighborhoods = len(nh_types)

        # Create colormap
        colors = plt.cm.tab20(np.linspace(0, 1, n_neighborhoods))

        for group in groups:
            group_data = data[data[group_col] == group] if group != 'all' else data

            if len(group_data) == 0:
                continue

            # Pivot data: rows=timepoints, cols=neighborhoods
            pivot_data = group_data.pivot_table(
                index='timepoint',
                columns='neighborhood_type',
                values='fraction_of_sample',
                aggfunc='mean',
                fill_value=0
            )

            # Ensure all neighborhoods are present
            for nh in nh_types:
                if nh not in pivot_data.columns:
                    pivot_data[nh] = 0

            pivot_data = pivot_data[nh_types]  # Consistent order

            # Create stacked area plot
            fig, ax = plt.subplots(figsize=(12, 8))

            timepoints = pivot_data.index.values
            values = pivot_data.values.T  # Transpose for stackplot

            ax.stackplot(timepoints, values,
                        labels=[f'NH-{nh}' for nh in nh_types],
                        colors=colors,
                        alpha=0.8)

            ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
            ax.set_ylabel('Fraction of Tissue', fontsize=12, fontweight='bold')
            ax.set_title(f'Neighborhood Evolution - {group}',
                        fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                     ncol=1, fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()

            plot_path = self.plots_dir / f'stacked_area_{group}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"    ✓ Generated {len(groups)} stacked area charts")
