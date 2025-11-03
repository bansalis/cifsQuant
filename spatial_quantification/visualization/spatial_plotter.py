"""
Spatial Plotter
Per-sample spatial visualization of tumor structures and marker distributions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.cluster import DBSCAN


class SpatialPlotter:
    """
    Spatial visualization for tumor structures and marker distributions.

    Generates:
    - Tumor structure plots (DBSCAN clusters per sample)
    - Marker positive/negative spatial maps
    - Multi-phenotype overlay maps
    - Publication-quality spatial figures
    """

    def __init__(self, output_dir: Path, config: Dict):
        """Initialize spatial plotter."""
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'spatial_plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Set style
        sns.set_style('white')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300

    def plot_tumor_structures_per_sample(self, adata, tumor_mask_col: str = 'is_Tumor'):
        """
        Plot tumor structures identified by DBSCAN for each sample.

        Shows spatial distribution of tumor clusters with different colors.
        """
        print("\n  Generating tumor structure plots per sample...")

        tumor_config = self.config.get('tumor_definition', {}).get('structure_detection', {})
        eps = tumor_config.get('eps', 100)
        min_samples = tumor_config.get('min_samples', 10)

        samples = adata.obs['sample_id'].unique()

        for sample in samples:
            sample_mask = adata.obs['sample_id'] == sample
            sample_data = adata.obs[sample_mask]
            sample_coords = adata.obsm['spatial'][sample_mask.values]

            # Get tumor cells
            if tumor_mask_col not in sample_data.columns:
                continue

            tumor_mask = sample_data[tumor_mask_col].values
            tumor_coords = sample_coords[tumor_mask]
            non_tumor_coords = sample_coords[~tumor_mask]

            if len(tumor_coords) < min_samples:
                continue

            # Run DBSCAN
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(tumor_coords)

            # Plot
            fig, ax = plt.subplots(figsize=(12, 10))

            # Non-tumor cells (light gray background)
            if len(non_tumor_coords) > 0:
                ax.scatter(non_tumor_coords[:, 0], non_tumor_coords[:, 1],
                          c='lightgray', s=1, alpha=0.3, label='Non-tumor')

            # Tumor structures (colored by cluster)
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

            colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

            for k, col in zip(range(n_clusters), colors):
                class_member_mask = (labels == k)
                xy = tumor_coords[class_member_mask]
                ax.scatter(xy[:, 0], xy[:, 1], c=[col], s=3, alpha=0.7,
                          edgecolors='none', label=f'Structure {k+1}')

            # Noise points
            if -1 in labels:
                noise_mask = (labels == -1)
                xy = tumor_coords[noise_mask]
                ax.scatter(xy[:, 0], xy[:, 1], c='black', s=1, alpha=0.3,
                          label='Noise')

            ax.set_xlabel('X (μm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y (μm)', fontsize=12, fontweight='bold')
            ax.set_title(f'{sample} - Tumor Structures (n={n_clusters})',
                        fontsize=14, fontweight='bold')
            ax.set_aspect('equal')
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=8)

            plt.tight_layout()

            plot_path = self.plots_dir / f'{sample}_tumor_structures.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"    ✓ Generated {len(samples)} tumor structure plots")

    def plot_marker_spatial_maps(self, adata, markers: List[str]):
        """
        Plot spatial distribution of marker positive/negative cells per sample.

        Creates 2-panel plots showing marker+ vs marker- cells.
        """
        print("\n  Generating marker spatial maps...")

        samples = adata.obs['sample_id'].unique()
        n_plots = 0

        for sample in samples:
            sample_mask = adata.obs['sample_id'] == sample
            sample_data = adata.obs[sample_mask]
            sample_coords = adata.obsm['spatial'][sample_mask.values]

            for marker in markers:
                marker_col = f'is_{marker}'

                if marker_col not in sample_data.columns:
                    continue

                marker_mask = sample_data[marker_col].values
                pos_coords = sample_coords[marker_mask]
                neg_coords = sample_coords[~marker_mask]

                # Create 2-panel figure
                fig, axes = plt.subplots(1, 2, figsize=(18, 8))

                # Panel 1: Positive vs Negative
                axes[0].scatter(neg_coords[:, 0], neg_coords[:, 1],
                              c='lightgray', s=2, alpha=0.3, label=f'{marker}-')
                axes[0].scatter(pos_coords[:, 0], pos_coords[:, 1],
                              c='#E41A1C', s=3, alpha=0.7, label=f'{marker}+')
                axes[0].set_xlabel('X (μm)', fontsize=12, fontweight='bold')
                axes[0].set_ylabel('Y (μm)', fontsize=12, fontweight='bold')
                axes[0].set_title(f'{marker} Distribution', fontsize=13, fontweight='bold')
                axes[0].set_aspect('equal')
                axes[0].legend(loc='best', fontsize=10)

                # Panel 2: Positive cells only (density view)
                if len(pos_coords) > 0:
                    axes[1].scatter(pos_coords[:, 0], pos_coords[:, 1],
                                  c='#E41A1C', s=4, alpha=0.8)
                    axes[1].set_xlabel('X (μm)', fontsize=12, fontweight='bold')
                    axes[1].set_ylabel('Y (μm)', fontsize=12, fontweight='bold')
                    axes[1].set_title(f'{marker}+ Cells Only (n={len(pos_coords):,})',
                                    fontsize=13, fontweight='bold')
                    axes[1].set_aspect('equal')

                fig.suptitle(f'{sample} - {marker} Spatial Distribution',
                           fontsize=16, fontweight='bold', y=1.00)

                plt.tight_layout()

                plot_path = self.plots_dir / f'{sample}_{marker}_spatial.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

                n_plots += 1

        print(f"    ✓ Generated {n_plots} marker spatial maps")

    def plot_phenotype_overlay(self, adata, phenotypes: List[str],
                               tumor_col: str = 'is_Tumor'):
        """
        Plot multiple phenotypes overlaid on tumor structures.

        Creates multi-color spatial maps showing different cell populations.
        """
        print("\n  Generating phenotype overlay maps...")

        samples = adata.obs['sample_id'].unique()
        color_map = {
            'CD8': '#E41A1C',      # Red
            'CD45': '#377EB8',     # Blue
            'PERK': '#4DAF4A',     # Green
            'KI67': '#FF7F00',     # Orange
            'CD3': '#984EA3',      # Purple
            'Tumor': '#FFD700',    # Gold
        }

        for sample in samples:
            sample_mask = adata.obs['sample_id'] == sample
            sample_data = adata.obs[sample_mask]
            sample_coords = adata.obsm['spatial'][sample_mask.values]

            # Get tumor background
            if tumor_col in sample_data.columns:
                tumor_mask = sample_data[tumor_col].values
                tumor_coords = sample_coords[tumor_mask]
                non_tumor_coords = sample_coords[~tumor_mask]
            else:
                tumor_coords = sample_coords
                non_tumor_coords = np.array([])

            fig, ax = plt.subplots(figsize=(14, 12))

            # Background: non-tumor cells
            if len(non_tumor_coords) > 0:
                ax.scatter(non_tumor_coords[:, 0], non_tumor_coords[:, 1],
                          c='lightgray', s=1, alpha=0.2, label='Non-tumor')

            # Tumor cells (light yellow background)
            if len(tumor_coords) > 0:
                ax.scatter(tumor_coords[:, 0], tumor_coords[:, 1],
                          c='#FFF9E6', s=2, alpha=0.4, label='Tumor')

            # Overlay phenotypes
            for phenotype in phenotypes:
                pheno_col = f'is_{phenotype}'

                if pheno_col not in sample_data.columns:
                    continue

                pheno_mask = sample_data[pheno_col].values
                pheno_coords = sample_coords[pheno_mask]

                if len(pheno_coords) == 0:
                    continue

                # Get color for this phenotype
                color = color_map.get(phenotype, '#000000')

                ax.scatter(pheno_coords[:, 0], pheno_coords[:, 1],
                          c=color, s=5, alpha=0.7, edgecolors='none',
                          label=f'{phenotype}+ (n={len(pheno_coords):,})')

            ax.set_xlabel('X (μm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y (μm)', fontsize=12, fontweight='bold')
            ax.set_title(f'{sample} - Multi-Phenotype Overlay',
                        fontsize=14, fontweight='bold')
            ax.set_aspect('equal')
            ax.legend(loc='upper right', bbox_to_anchor=(1.20, 1), fontsize=9)

            plt.tight_layout()

            plot_path = self.plots_dir / f'{sample}_phenotype_overlay.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"    ✓ Generated {len(samples)} phenotype overlay plots")

    def plot_tumor_infiltration_heatmap(self, adata, immune_markers: List[str],
                                       tumor_col: str = 'is_Tumor'):
        """
        Create spatial density heatmap of immune infiltration into tumor.

        Shows where immune cells accumulate within tumor structures.
        """
        print("\n  Generating tumor infiltration heatmaps...")

        samples = adata.obs['sample_id'].unique()

        for sample in samples:
            sample_mask = adata.obs['sample_id'] == sample
            sample_data = adata.obs[sample_mask]
            sample_coords = adata.obsm['spatial'][sample_mask.values]

            # Get tumor region
            if tumor_col not in sample_data.columns:
                continue

            tumor_mask = sample_data[tumor_col].values
            tumor_coords = sample_coords[tumor_mask]

            if len(tumor_coords) < 10:
                continue

            # Create figure with subplots for each immune marker
            n_markers = len(immune_markers)
            fig, axes = plt.subplots(1, n_markers, figsize=(7*n_markers, 6))

            if n_markers == 1:
                axes = [axes]

            for ax, marker in zip(axes, immune_markers):
                marker_col = f'is_{marker}'

                if marker_col not in sample_data.columns:
                    continue

                # Get marker+ cells within tumor
                marker_mask = sample_data[marker_col].values
                infiltrating_mask = marker_mask & tumor_mask
                infiltrating_coords = sample_coords[infiltrating_mask]

                if len(infiltrating_coords) == 0:
                    continue

                # Hexbin density plot
                hexbin = ax.hexbin(infiltrating_coords[:, 0], infiltrating_coords[:, 1],
                                  gridsize=50, cmap='YlOrRd', mincnt=1)
                ax.set_xlabel('X (μm)', fontsize=11, fontweight='bold')
                ax.set_ylabel('Y (μm)', fontsize=11, fontweight='bold')
                ax.set_title(f'{marker}+ Density\n(n={len(infiltrating_coords):,})',
                           fontsize=12, fontweight='bold')
                ax.set_aspect('equal')
                plt.colorbar(hexbin, ax=ax, label='Cell Count')

            fig.suptitle(f'{sample} - Immune Infiltration Density',
                        fontsize=14, fontweight='bold', y=1.02)

            plt.tight_layout()

            plot_path = self.plots_dir / f'{sample}_infiltration_heatmap.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"    ✓ Generated {len(samples)} infiltration heatmaps")

    def plot_summary_spatial_panel(self, adata, sample: str,
                                   tumor_col: str = 'is_Tumor',
                                   key_markers: List[str] = None):
        """
        Create comprehensive 4-panel spatial summary for a single sample.

        Panel 1: Tumor structures (DBSCAN)
        Panel 2: Key marker overlay
        Panel 3: Immune infiltration density
        Panel 4: All phenotypes combined
        """
        if key_markers is None:
            key_markers = ['CD8', 'CD45', 'PERK']

        sample_mask = adata.obs['sample_id'] == sample
        sample_data = adata.obs[sample_mask]
        sample_coords = adata.obsm['spatial'][sample_mask.values]

        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        fig.suptitle(f'{sample} - Spatial Analysis Summary',
                    fontsize=18, fontweight='bold', y=0.998)

        # Panel 1: Tumor structures
        ax = axes[0, 0]
        if tumor_col in sample_data.columns:
            tumor_mask = sample_data[tumor_col].values
            tumor_coords = sample_coords[tumor_mask]

            if len(tumor_coords) >= 10:
                clusterer = DBSCAN(eps=100, min_samples=10)
                labels = clusterer.fit_predict(tumor_coords)

                unique_labels = set(labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

                colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

                for k, col in zip(range(n_clusters), colors):
                    class_member_mask = (labels == k)
                    xy = tumor_coords[class_member_mask]
                    ax.scatter(xy[:, 0], xy[:, 1], c=[col], s=3, alpha=0.7,
                             edgecolors='none')

        ax.set_title(f'Tumor Structures', fontsize=13, fontweight='bold')
        ax.set_xlabel('X (μm)', fontsize=11)
        ax.set_ylabel('Y (μm)', fontsize=11)
        ax.set_aspect('equal')

        # Panel 2: Key marker overlay
        ax = axes[0, 1]
        color_map = {'CD8': '#E41A1C', 'CD45': '#377EB8', 'PERK': '#4DAF4A'}

        for marker in key_markers:
            marker_col = f'is_{marker}'
            if marker_col in sample_data.columns:
                marker_mask = sample_data[marker_col].values
                marker_coords = sample_coords[marker_mask]
                color = color_map.get(marker, '#000000')
                ax.scatter(marker_coords[:, 0], marker_coords[:, 1],
                         c=color, s=4, alpha=0.7, label=f'{marker}+')

        ax.set_title('Key Markers', fontsize=13, fontweight='bold')
        ax.set_xlabel('X (μm)', fontsize=11)
        ax.set_ylabel('Y (μm)', fontsize=11)
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=9)

        # Panel 3: Immune density heatmap
        ax = axes[1, 0]
        if tumor_col in sample_data.columns:
            tumor_mask = sample_data[tumor_col].values
            cd45_col = 'is_CD45'
            if cd45_col in sample_data.columns:
                cd45_mask = sample_data[cd45_col].values
                infiltrating_mask = cd45_mask & tumor_mask
                infiltrating_coords = sample_coords[infiltrating_mask]

                if len(infiltrating_coords) > 0:
                    hexbin = ax.hexbin(infiltrating_coords[:, 0], infiltrating_coords[:, 1],
                                     gridsize=40, cmap='YlOrRd', mincnt=1)
                    plt.colorbar(hexbin, ax=ax, label='CD45+ Density')

        ax.set_title('CD45+ Infiltration', fontsize=13, fontweight='bold')
        ax.set_xlabel('X (μm)', fontsize=11)
        ax.set_ylabel('Y (μm)', fontsize=11)
        ax.set_aspect('equal')

        # Panel 4: Combined spatial view
        ax = axes[1, 1]
        # Background: all cells
        ax.scatter(sample_coords[:, 0], sample_coords[:, 1],
                 c='lightgray', s=1, alpha=0.3)

        # Tumor
        if tumor_col in sample_data.columns:
            tumor_coords = sample_coords[tumor_mask]
            ax.scatter(tumor_coords[:, 0], tumor_coords[:, 1],
                     c='gold', s=2, alpha=0.4, label='Tumor')

        # Immune
        cd45_col = 'is_CD45'
        if cd45_col in sample_data.columns:
            cd45_mask = sample_data[cd45_col].values
            cd45_coords = sample_coords[cd45_mask]
            ax.scatter(cd45_coords[:, 0], cd45_coords[:, 1],
                     c='#377EB8', s=3, alpha=0.7, label='CD45+')

        ax.set_title('Combined View', fontsize=13, fontweight='bold')
        ax.set_xlabel('X (μm)', fontsize=11)
        ax.set_ylabel('Y (μm)', fontsize=11)
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=9)

        plt.tight_layout()

        plot_path = self.plots_dir / f'{sample}_spatial_summary.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Generated spatial summary for {sample}")
