"""
Pseudotime Visualization - IMPROVED VERSION
Builds from UMAP embedding for better trajectory inference
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


class PseudotimePlotter:
    """
    Comprehensive plotting for pseudotime analysis.

    IMPROVED: Builds pseudotime from UMAP embedding
    Uses diffusion maps or principal curves for trajectory inference
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

    def compute_diffusion_pseudotime(self, umap_coords: np.ndarray,
                                     n_components: int = 10) -> np.ndarray:
        """
        Compute pseudotime using diffusion maps.

        Better than PCA for non-linear trajectories.

        Parameters
        ----------
        umap_coords : np.ndarray
            UMAP embedding (N x 2)
        n_components : int
            Number of diffusion components

        Returns
        -------
        np.ndarray
            Pseudotime values (N,)
        """
        from scipy.spatial.distance import pdist, squareform
        from scipy.linalg import eigh

        print("    Computing diffusion pseudotime...")

        # Compute pairwise distances in UMAP space
        D = squareform(pdist(umap_coords, metric='euclidean'))

        # Gaussian kernel
        epsilon = np.median(D[D > 0]) ** 2
        W = np.exp(-D ** 2 / epsilon)

        # Normalize to create Markov matrix
        D_sum = np.sum(W, axis=1)
        P = W / D_sum[:, np.newaxis]

        # Eigen decomposition
        eigenvalues, eigenvectors = eigh(P)

        # Sort by eigenvalue (largest first)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Pseudotime as first non-trivial diffusion component
        pseudotime = eigenvectors[:, 1]  # Skip first (constant) component

        # Normalize to [0, 1]
        pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min() + 1e-10)

        print(f"    ✓ Diffusion pseudotime computed")

        return pseudotime

    def plot_umap_with_pseudotime(self, umap_df: pd.DataFrame,
                                  marker_cols: list, name_suffix: str = ""):
        """
        Plot UMAP colored by pseudotime and markers.

        Parameters
        ----------
        umap_df : pd.DataFrame
            UMAP data with pseudotime column
        marker_cols : list
            Marker columns to plot (e.g., ['is_PERK', 'is_AGFP', 'is_KI67'])
        name_suffix : str
            Suffix for filename
        """
        if 'pseudotime' not in umap_df.columns:
            print("    ⚠ No pseudotime column found")
            return

        # Filter available markers
        available_markers = [m for m in marker_cols if m in umap_df.columns]

        n_panels = 1 + len(available_markers)  # +1 for pseudotime
        ncols = min(3, n_panels)
        nrows = (n_panels + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))

        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]
        else:
            axes = axes

        # Panel 1: Pseudotime
        ax = axes[0][0] if nrows > 1 or ncols > 1 else axes
        scatter = ax.scatter(umap_df['UMAP1'], umap_df['UMAP2'],
                           c=umap_df['pseudotime'], s=3, alpha=0.6,
                           cmap='viridis', vmin=0, vmax=1, rasterized=True)
        plt.colorbar(scatter, ax=ax, label='Pseudotime', fraction=0.046)
        ax.set_xlabel('UMAP1', fontsize=11, fontweight='bold')
        ax.set_ylabel('UMAP2', fontsize=11, fontweight='bold')
        ax.set_title('Pseudotime Trajectory', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')

        # Subsequent panels: Markers
        for idx, marker in enumerate(available_markers):
            panel_idx = idx + 1
            row = panel_idx // ncols
            col = panel_idx % ncols

            if nrows > 1 or ncols > 1:
                ax = axes[row][col] if nrows > 1 else axes[col]
            else:
                break  # Only one panel

            marker_name = marker.replace('is_', '')

            # Binary gate coloring
            pos_mask = umap_df[marker] == 1
            neg_mask = umap_df[marker] == 0

            ax.scatter(umap_df.loc[neg_mask, 'UMAP1'],
                      umap_df.loc[neg_mask, 'UMAP2'],
                      c='lightgray', s=1, alpha=0.2, label=f'{marker_name}-',
                      rasterized=True)
            ax.scatter(umap_df.loc[pos_mask, 'UMAP1'],
                      umap_df.loc[pos_mask, 'UMAP2'],
                      c='red', s=3, alpha=0.6, label=f'{marker_name}+',
                      rasterized=True)

            ax.set_xlabel('UMAP1', fontsize=11, fontweight='bold')
            ax.set_ylabel('UMAP2', fontsize=11, fontweight='bold')
            ax.set_title(f'{marker_name} Expression', fontsize=12, fontweight='bold')
            ax.legend(markerscale=5, fontsize=9)
            ax.set_aspect('equal')

        # Hide unused subplots
        for idx in range(n_panels, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            if nrows > 1:
                axes[row][col].axis('off')
            elif ncols > 1:
                axes[col].axis('off')

        title = f'UMAP with Pseudotime{name_suffix}'
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save
        filename = f'umap_pseudotime{name_suffix.replace(" ", "_").lower()}.png'
        plot_path = self.plots_dir / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Saved UMAP with pseudotime to {plot_path.name}")

    def plot_differentiation_trajectories(self, umap_df: pd.DataFrame,
                                         marker_cols: list, name_suffix: str = ""):
        """
        Plot marker frequencies along pseudotime trajectory.

        Parameters
        ----------
        umap_df : pd.DataFrame
            UMAP data with pseudotime and marker columns
        marker_cols : list
            Marker columns (binary gates)
        name_suffix : str
            Suffix for filename
        """
        if 'pseudotime' not in umap_df.columns:
            print("    ⚠ No pseudotime column found")
            return

        # Filter available markers
        available_markers = [m for m in marker_cols if m in umap_df.columns]

        if not available_markers:
            print("    ⚠ No markers available for trajectory plot")
            return

        # Bin pseudotime
        n_bins = 20
        umap_df['pseudotime_bin'] = pd.cut(umap_df['pseudotime'], bins=n_bins, labels=False)

        # Calculate marker frequencies per bin
        bin_data = []
        for bin_id in range(n_bins):
            bin_df = umap_df[umap_df['pseudotime_bin'] == bin_id]

            if len(bin_df) < 5:
                continue

            result = {
                'pseudotime_center': (bin_id + 0.5) / n_bins,
                'n_cells': len(bin_df)
            }

            # Add group info if available
            if 'main_group' in bin_df.columns:
                # Majority vote for group
                result['main_group'] = bin_df['main_group'].mode()[0] if len(bin_df) > 0 else ''

            # Calculate marker frequencies
            for marker in available_markers:
                marker_name = marker.replace('is_', '')
                freq = bin_df[marker].sum() / len(bin_df) * 100
                result[f'{marker_name}_percent'] = freq

            bin_data.append(result)

        if not bin_data:
            print("    ⚠ No bin data available")
            return

        df_traj = pd.DataFrame(bin_data)

        # Plot
        n_markers = len(available_markers)
        fig, axes = plt.subplots(1, n_markers, figsize=(6*n_markers, 5))

        if n_markers == 1:
            axes = [axes]

        for idx, marker in enumerate(available_markers):
            ax = axes[idx]
            marker_name = marker.replace('is_', '')
            col = f'{marker_name}_percent'

            if col not in df_traj.columns:
                continue

            # Check if we have group information
            if 'main_group' in df_traj.columns:
                groups = sorted(df_traj['main_group'].unique())

                for group in groups:
                    group_data = df_traj[df_traj['main_group'] == group]

                    if len(group_data) == 0:
                        continue

                    color = self.group_colors.get(group, '#000000')

                    ax.plot(group_data['pseudotime_center'], group_data[col],
                           '-o', color=color, linewidth=2.5, markersize=6,
                           label=group, zorder=10)
            else:
                # No group info, plot all together
                ax.plot(df_traj['pseudotime_center'], df_traj[col],
                       '-o', color='black', linewidth=2.5, markersize=6,
                       zorder=10)

            ax.set_xlabel('Pseudotime', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'% {marker_name}+ Cells', fontsize=12, fontweight='bold')
            ax.set_title(f'{marker_name} Along Trajectory', fontsize=13, fontweight='bold')
            ax.legend(frameon=True, loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 100)

        title = f'Differentiation Trajectories{name_suffix}'
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save
        filename = f'differentiation_trajectories{name_suffix.replace(" ", "_").lower()}.png'
        plot_path = self.plots_dir / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Saved differentiation trajectory plot to {plot_path.name}")

    def generate_all_plots(self, umap_results: Dict):
        """
        Generate all pseudotime plots from UMAP results.

        Parameters
        ----------
        umap_results : dict
            Dictionary containing UMAP DataFrames:
            - 'umap_all_cells': UMAP for all cells
            - 'umap_tumor_only': UMAP for tumor cells only
        """
        print("\n" + "="*80)
        print("PSEUDOTIME ANALYSIS (FROM UMAP)")
        print("="*80)
        print("  Computing diffusion pseudotime from UMAP embedding")
        print("="*80)

        marker_cols = ['is_PERK', 'is_AGFP', 'is_KI67']

        # Process all cells UMAP
        if 'umap_all_cells' in umap_results and umap_results['umap_all_cells'] is not None:
            print("\n1. Pseudotime for ALL CELLS...")

            umap_df = umap_results['umap_all_cells'].copy()

            if 'UMAP1' in umap_df.columns and 'UMAP2' in umap_df.columns:
                # Compute pseudotime
                umap_coords = umap_df[['UMAP1', 'UMAP2']].values
                pseudotime = self.compute_diffusion_pseudotime(umap_coords)
                umap_df['pseudotime'] = pseudotime

                # Save
                output_path = self.output_dir / 'pseudotime_all_cells.csv'
                umap_df.to_csv(output_path, index=False)
                print(f"  ✓ Saved to {output_path.name}")

                # Generate plots
                self.plot_umap_with_pseudotime(umap_df, marker_cols, name_suffix=" (All Cells)")
                self.plot_differentiation_trajectories(umap_df, marker_cols, name_suffix=" (All Cells)")
            else:
                print("  ⚠ No UMAP coordinates found")

        # Process tumor-only UMAP
        if 'umap_tumor_only' in umap_results and umap_results['umap_tumor_only'] is not None:
            print("\n2. Pseudotime for TUMOR CELLS ONLY...")

            umap_df = umap_results['umap_tumor_only'].copy()

            if 'UMAP1' in umap_df.columns and 'UMAP2' in umap_df.columns:
                # Compute pseudotime
                umap_coords = umap_df[['UMAP1', 'UMAP2']].values
                pseudotime = self.compute_diffusion_pseudotime(umap_coords)
                umap_df['pseudotime'] = pseudotime

                # Save
                output_path = self.output_dir / 'pseudotime_tumor_only.csv'
                umap_df.to_csv(output_path, index=False)
                print(f"  ✓ Saved to {output_path.name}")

                # Generate plots
                self.plot_umap_with_pseudotime(umap_df, marker_cols, name_suffix=" (Tumor Only)")
                self.plot_differentiation_trajectories(umap_df, marker_cols, name_suffix=" (Tumor Only)")
            else:
                print("  ⚠ No UMAP coordinates found")

        print("\n" + "="*80)
        print("✓ PSEUDOTIME ANALYSIS COMPLETE")
        print(f"  All plots saved to {self.plots_dir}/")
        print("="*80 + "\n")
