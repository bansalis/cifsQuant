"""
Pseudotime Visualization
Generate differentiation trajectory plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


class PseudotimePlotter:
    """
    Comprehensive plotting for pseudotime analysis.

    Generates:
    - Differentiation trajectories showing marker changes along pseudotime
    - PC space plots colored by markers
    - Spatial pseudotime maps
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
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Get plotting settings
        plotting_config = config.get('plotting', {})
        self.group_colors = plotting_config.get('group_colors', {
            'KPT': '#E41A1C', 'KPNT': '#377EB8'
        })
        self.timepoint_label = plotting_config.get('timepoint_label', 'Time (weeks)')

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300

    def plot_differentiation_trajectories(self, trajectory_df: pd.DataFrame):
        """
        Plot marker frequencies along pseudotime trajectory.

        Parameters
        ----------
        trajectory_df : pd.DataFrame
            Trajectory dynamics data with pseudotime_center and marker percentages
        """
        # Get available markers
        marker_cols = [col for col in trajectory_df.columns if col.endswith('_percent')]
        markers = [col.replace('_percent', '') for col in marker_cols]

        if not markers:
            return

        groups = sorted(trajectory_df['main_group'].unique())
        n_markers = len(markers)

        fig, axes = plt.subplots(1, n_markers, figsize=(6*n_markers, 5))

        if n_markers == 1:
            axes = [axes]

        for idx, marker in enumerate(markers):
            ax = axes[idx]
            col = f'{marker}_percent'

            for group in groups:
                group_data = trajectory_df[trajectory_df['main_group'] == group]

                if len(group_data) == 0:
                    continue

                # Calculate mean and SEM per pseudotime bin
                summary = group_data.groupby('pseudotime_center')[col].agg(['mean', 'sem'])
                pt_values = summary.index.values
                means = summary['mean'].values
                sems = summary['sem'].values

                color = self.group_colors.get(group, '#000000')

                # Plot trajectory
                ax.plot(pt_values, means, '-o', color=color, linewidth=2.5,
                       markersize=6, label=group, zorder=10)

                # Confidence band
                ax.fill_between(pt_values, means - sems, means + sems,
                               alpha=0.2, color=color)

            ax.set_xlabel('Pseudotime', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'% {marker}+ Cells', fontsize=12, fontweight='bold')
            ax.set_title(f'{marker} Along Trajectory', fontsize=13, fontweight='bold')
            ax.legend(frameon=True, loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)

        plt.suptitle('Differentiation Trajectories', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save
        plot_path = self.plots_dir / 'differentiation_trajectories.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Saved differentiation trajectory plot to {plot_path}")

    def plot_pseudotime_pc_space(self, cell_df: pd.DataFrame):
        """
        Plot cells in PC space colored by markers and pseudotime.

        Parameters
        ----------
        cell_df : pd.DataFrame
            Per-cell pseudotime data with PC1, PC2, PC3 and marker states
        """
        # Get available markers
        marker_cols = [col for col in cell_df.columns
                      if col in ['PERK', 'AGFP', 'KI67']]

        if 'PC1' not in cell_df.columns or 'PC2' not in cell_df.columns:
            return

        # Subsample for plotting if too many cells
        if len(cell_df) > 50000:
            cell_df = cell_df.sample(n=50000, random_state=42)

        n_panels = len(marker_cols) + 1  # +1 for pseudotime
        fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 5))

        if n_panels == 1:
            axes = [axes]

        # Panel 1: Pseudotime
        ax = axes[0]
        scatter = ax.scatter(cell_df['PC1'], cell_df['PC2'],
                           c=cell_df['pseudotime'], s=2, alpha=0.5,
                           cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(scatter, ax=ax, label='Pseudotime')
        ax.set_xlabel('PC1', fontsize=11, fontweight='bold')
        ax.set_ylabel('PC2', fontsize=11, fontweight='bold')
        ax.set_title('Pseudotime', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')

        # Subsequent panels: Markers
        for idx, marker in enumerate(marker_cols):
            ax = axes[idx + 1]

            # Color by marker status
            pos_mask = cell_df[marker] == 1
            neg_mask = cell_df[marker] == 0

            ax.scatter(cell_df.loc[neg_mask, 'PC1'],
                      cell_df.loc[neg_mask, 'PC2'],
                      c='lightgray', s=1, alpha=0.3, label=f'{marker}-')
            ax.scatter(cell_df.loc[pos_mask, 'PC1'],
                      cell_df.loc[pos_mask, 'PC2'],
                      c='red', s=2, alpha=0.6, label=f'{marker}+')

            ax.set_xlabel('PC1', fontsize=11, fontweight='bold')
            ax.set_ylabel('PC2', fontsize=11, fontweight='bold')
            ax.set_title(f'{marker} Expression', fontsize=12, fontweight='bold')
            ax.legend(markerscale=5, fontsize=9)
            ax.set_aspect('equal')

        plt.suptitle('Principal Component Space', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save
        plot_path = self.plots_dir / 'pseudotime_pc_space.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Saved PC space plot to {plot_path}")

    def generate_all_plots(self, results: Dict):
        """
        Generate all pseudotime plots.

        Parameters
        ----------
        results : dict
            Results dictionary from PseudotimeAnalysis
        """
        print("  Generating pseudotime plots...")

        # Plot differentiation trajectories
        if 'trajectory_dynamics' in results:
            self.plot_differentiation_trajectories(results['trajectory_dynamics'])

        # Plot PC space
        if 'cell_pseudotime' in results:
            self.plot_pseudotime_pc_space(results['cell_pseudotime'])

        print(f"  ✓ Generated pseudotime plots in {self.plots_dir}/")
