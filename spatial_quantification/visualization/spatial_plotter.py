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
import warnings

try:
    import spatialcells as spc
    HAS_SPATIALCELLS = True
except ImportError:
    HAS_SPATIALCELLS = False


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

    def plot_neighborhood_spatial_maps(self, neighborhood_assignments: Dict):
        """
        Plot neighborhood assignments spatially for each sample.

        Shows which cells belong to which neighborhood type with consistent colors.

        Parameters
        ----------
        neighborhood_assignments : Dict
            Dict of {sample_id: {'labels': array, 'coords': array}}
        """
        print("\n  Generating neighborhood spatial maps...")

        # Get unique neighborhood types across all samples
        all_labels = []
        for sample_data in neighborhood_assignments.values():
            all_labels.extend(np.unique(sample_data['labels']))
        unique_neighborhoods = sorted(set(all_labels))

        # Create consistent colormap
        n_neighborhoods = len(unique_neighborhoods)
        colors = plt.cm.tab20(np.linspace(0, 1, n_neighborhoods))
        nh_to_color = {nh: colors[i] for i, nh in enumerate(unique_neighborhoods)}

        for sample, data in neighborhood_assignments.items():
            labels = data['labels']
            coords = data['coords']

            if len(labels) == 0:
                continue

            fig, ax = plt.subplots(figsize=(14, 12))

            # Plot each neighborhood type
            for nh_label in unique_neighborhoods:
                nh_mask = (labels == nh_label)
                if nh_mask.sum() == 0:
                    continue

                nh_coords = coords[nh_mask]
                color = nh_to_color[nh_label]

                ax.scatter(nh_coords[:, 0], nh_coords[:, 1],
                          c=[color], s=5, alpha=0.8, edgecolors='none',
                          label=f'NH-{nh_label}')

            ax.set_xlabel('X (μm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y (μm)', fontsize=12, fontweight='bold')
            ax.set_title(f'{sample} - Neighborhood Assignments',
                        fontsize=14, fontweight='bold')
            ax.set_aspect('equal')

            # Legend
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1),
                     ncol=1, fontsize=8, markerscale=2)

            plt.tight_layout()

            plot_path = self.plots_dir / f'{sample}_neighborhoods.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"    ✓ Generated {len(neighborhood_assignments)} neighborhood spatial maps")

    def plot_individual_phenotypes(self, adata, phenotypes: List[str] = None):
        """
        Plot each cell phenotype separately showing spatial distribution.

        Creates individual spatial plots for each phenotype with all cells in background.
        Useful for visualizing where specific cell types are located.

        Parameters
        ----------
        adata : AnnData
            Annotated data object
        phenotypes : List[str], optional
            List of phenotype names to plot. If None, uses all phenotypes from config.
        """
        print("\n  Generating individual phenotype spatial plots...")

        if phenotypes is None:
            phenotypes = list(self.config.get('phenotypes', {}).keys())

        samples = adata.obs['sample_id'].unique()
        n_plots = 0

        for sample in samples:
            sample_mask = adata.obs['sample_id'] == sample
            sample_data = adata.obs[sample_mask]
            sample_coords = adata.obsm['spatial'][sample_mask.values]

            for phenotype in phenotypes:
                pheno_col = f'is_{phenotype}'

                if pheno_col not in sample_data.columns:
                    continue

                pheno_mask = sample_data[pheno_col].values
                pos_coords = sample_coords[pheno_mask]
                neg_coords = sample_coords[~pheno_mask]

                if len(pos_coords) == 0:
                    continue

                # Get phenotype color from config
                pheno_config = self.config.get('phenotypes', {}).get(phenotype, {})
                color = pheno_config.get('color', '#E41A1C')

                fig, ax = plt.subplots(figsize=(12, 10))

                # Background: all other cells
                ax.scatter(neg_coords[:, 0], neg_coords[:, 1],
                          c='lightgray', s=1, alpha=0.2, label='Other cells')

                # Phenotype cells
                ax.scatter(pos_coords[:, 0], pos_coords[:, 1],
                          c=color, s=4, alpha=0.7, label=f'{phenotype}')

                ax.set_xlabel('X (μm)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Y (μm)', fontsize=12, fontweight='bold')
                ax.set_title(f'{sample} - {phenotype} (n={len(pos_coords):,})',
                           fontsize=14, fontweight='bold')
                ax.set_aspect('equal')
                ax.legend(loc='upper right', fontsize=10)

                plt.tight_layout()

                plot_path = self.plots_dir / 'individual_phenotypes' / f'{sample}_{phenotype}_spatial.png'
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

                n_plots += 1

        print(f"    ✓ Generated {n_plots} individual phenotype spatial plots")

    def plot_tumor_zones_dbscan(self, adata, tumor_col: str = 'is_Tumor'):
        """
        Plot unique tumor zones/clusters detected by DBSCAN.

        Each tumor cluster is colored differently to visualize spatial heterogeneity
        and validate DBSCAN parameters (eps, min_samples).

        Parameters
        ----------
        adata : AnnData
            Annotated data object
        tumor_col : str
            Column name for tumor cell identification
        """
        print("\n  Generating tumor zone (DBSCAN cluster) plots...")

        tumor_config = self.config.get('tumor_definition', {}).get('structure_detection', {})
        eps = tumor_config.get('eps', 100)
        min_samples = tumor_config.get('min_samples', 10)
        min_cluster_size = tumor_config.get('min_cluster_size', 250)

        samples = adata.obs['sample_id'].unique()
        n_plots = 0

        for sample in samples:
            sample_mask = adata.obs['sample_id'] == sample
            sample_data = adata.obs[sample_mask]
            sample_coords = adata.obsm['spatial'][sample_mask.values]

            # Get tumor cells
            if tumor_col not in sample_data.columns:
                continue

            tumor_mask = sample_data[tumor_col].values
            tumor_coords = sample_coords[tumor_mask]
            non_tumor_coords = sample_coords[~tumor_mask]

            if len(tumor_coords) < min_samples:
                continue

            # Run DBSCAN
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(tumor_coords)

            # Filter by minimum cluster size
            unique_labels = set(labels)
            valid_clusters = []
            for label in unique_labels:
                if label == -1:  # Skip noise
                    continue
                cluster_size = (labels == label).sum()
                if cluster_size >= min_cluster_size:
                    valid_clusters.append(label)

            n_clusters = len(valid_clusters)

            if n_clusters == 0:
                continue

            # Create plot
            fig, ax = plt.subplots(figsize=(14, 12))

            # Background: non-tumor cells
            if len(non_tumor_coords) > 0:
                ax.scatter(non_tumor_coords[:, 0], non_tumor_coords[:, 1],
                          c='lightgray', s=1, alpha=0.2, label='Non-tumor', rasterized=True)

            # Plot each tumor zone with unique color
            colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

            for idx, cluster_label in enumerate(sorted(valid_clusters)):
                cluster_mask = (labels == cluster_label)
                cluster_coords = tumor_coords[cluster_mask]
                cluster_size = len(cluster_coords)

                ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                          c=[colors[idx]], s=4, alpha=0.8, edgecolors='none',
                          label=f'Zone {cluster_label+1} (n={cluster_size:,})',
                          rasterized=True)

            # Plot noise (if any)
            if -1 in labels:
                noise_mask = (labels == -1)
                noise_coords = tumor_coords[noise_mask]
                if len(noise_coords) > 0:
                    ax.scatter(noise_coords[:, 0], noise_coords[:, 1],
                             c='black', s=1, alpha=0.3, label=f'Noise (n={len(noise_coords):,})',
                             rasterized=True)

            ax.set_xlabel('X (μm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y (μm)', fontsize=12, fontweight='bold')
            ax.set_title(f'{sample} - Tumor Zones/Clusters (n={n_clusters})\n'
                        f'DBSCAN: eps={eps}, min_samples={min_samples}',
                        fontsize=14, fontweight='bold')
            ax.set_aspect('equal')
            ax.legend(loc='upper right', bbox_to_anchor=(1.20, 1), fontsize=9, markerscale=3)

            plt.tight_layout()

            plot_path = self.plots_dir / 'tumor_zones' / f'{sample}_tumor_zones_dbscan.png'
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            n_plots += 1

        print(f"    ✓ Generated {n_plots} tumor zone plots (DBSCAN parameters: eps={eps}, min_samples={min_samples})")

    def plot_marker_zones(self, adata, markers: List[Dict], tumor_col: str = 'is_Tumor'):
        """
        Plot spatial zones for marker +/- tumor cells (e.g., pERK+/- zones).

        Uses DBSCAN to identify spatial clusters of marker+ and marker- tumor cells
        to visualize marker heterogeneity and validate zone detection parameters.

        Parameters
        ----------
        adata : AnnData
            Annotated data object
        markers : List[Dict]
            List of marker definitions with 'name', 'positive_phenotype', 'negative_phenotype'
        tumor_col : str
            Column name for tumor cell identification
        """
        print("\n  Generating marker +/- zone plots...")

        tumor_config = self.config.get('tumor_definition', {}).get('structure_detection', {})
        eps = tumor_config.get('eps', 100)
        min_samples = tumor_config.get('min_samples', 10)
        min_cluster_size = tumor_config.get('min_cluster_size', 250)

        samples = adata.obs['sample_id'].unique()
        n_plots = 0

        for marker_info in markers:
            marker_name = marker_info.get('name', marker_info.get('marker'))
            pos_phenotype = marker_info.get('positive_phenotype')
            neg_phenotype = marker_info.get('negative_phenotype')

            for sample in samples:
                sample_mask = adata.obs['sample_id'] == sample
                sample_data = adata.obs[sample_mask]
                sample_coords = adata.obsm['spatial'][sample_mask.values]

                # Get marker+ and marker- tumor cells
                pos_col = f'is_{pos_phenotype}'
                neg_col = f'is_{neg_phenotype}'

                if pos_col not in sample_data.columns or neg_col not in sample_data.columns:
                    continue

                pos_mask = sample_data[pos_col].values
                neg_mask = sample_data[neg_col].values
                tumor_mask = sample_data.get(tumor_col, pos_mask | neg_mask).values if tumor_col in sample_data.columns else (pos_mask | neg_mask)

                pos_coords = sample_coords[pos_mask]
                neg_coords = sample_coords[neg_mask]
                non_tumor_coords = sample_coords[~tumor_mask]

                if len(pos_coords) < min_samples and len(neg_coords) < min_samples:
                    continue

                # Run DBSCAN on marker+ cells
                pos_labels = np.full(len(pos_coords), -1)
                if len(pos_coords) >= min_samples:
                    clusterer_pos = DBSCAN(eps=eps, min_samples=min_samples)
                    pos_labels = clusterer_pos.fit_predict(pos_coords)

                # Run DBSCAN on marker- cells
                neg_labels = np.full(len(neg_coords), -1)
                if len(neg_coords) >= min_samples:
                    clusterer_neg = DBSCAN(eps=eps, min_samples=min_samples)
                    neg_labels = clusterer_neg.fit_predict(neg_coords)

                # Count valid clusters (>= min_cluster_size)
                pos_clusters = []
                for label in set(pos_labels):
                    if label != -1 and (pos_labels == label).sum() >= min_cluster_size:
                        pos_clusters.append(label)

                neg_clusters = []
                for label in set(neg_labels):
                    if label != -1 and (neg_labels == label).sum() >= min_cluster_size:
                        neg_clusters.append(label)

                n_pos_zones = len(pos_clusters)
                n_neg_zones = len(neg_clusters)

                # Create plot
                fig, ax = plt.subplots(figsize=(14, 12))

                # Background: non-tumor cells
                if len(non_tumor_coords) > 0:
                    ax.scatter(non_tumor_coords[:, 0], non_tumor_coords[:, 1],
                             c='lightgray', s=1, alpha=0.2, label='Non-tumor', rasterized=True)

                # Plot marker+ zones (shades of green)
                if n_pos_zones > 0:
                    pos_colors = plt.cm.Greens(np.linspace(0.4, 0.9, max(n_pos_zones, 1)))
                    for idx, cluster_label in enumerate(sorted(pos_clusters)):
                        cluster_mask = (pos_labels == cluster_label)
                        cluster_coords = pos_coords[cluster_mask]
                        ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                                 c=[pos_colors[idx]], s=5, alpha=0.8, edgecolors='none',
                                 label=f'{marker_name}+ Zone {cluster_label+1} (n={len(cluster_coords):,})',
                                 rasterized=True)
                else:
                    # No valid clusters, plot all as one
                    if len(pos_coords) > 0:
                        ax.scatter(pos_coords[:, 0], pos_coords[:, 1],
                                 c='green', s=3, alpha=0.5,
                                 label=f'{marker_name}+ (no zones, n={len(pos_coords):,})',
                                 rasterized=True)

                # Plot marker- zones (shades of red)
                if n_neg_zones > 0:
                    neg_colors = plt.cm.Reds(np.linspace(0.4, 0.9, max(n_neg_zones, 1)))
                    for idx, cluster_label in enumerate(sorted(neg_clusters)):
                        cluster_mask = (neg_labels == cluster_label)
                        cluster_coords = neg_coords[cluster_mask]
                        ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                                 c=[neg_colors[idx]], s=5, alpha=0.8, edgecolors='none',
                                 label=f'{marker_name}- Zone {cluster_label+1} (n={len(cluster_coords):,})',
                                 rasterized=True)
                else:
                    # No valid clusters, plot all as one
                    if len(neg_coords) > 0:
                        ax.scatter(neg_coords[:, 0], neg_coords[:, 1],
                                 c='red', s=3, alpha=0.5,
                                 label=f'{marker_name}- (no zones, n={len(neg_coords):,})',
                                 rasterized=True)

                # Plot noise (if any)
                pos_noise_mask = (pos_labels == -1)
                neg_noise_mask = (neg_labels == -1)
                n_pos_noise = pos_noise_mask.sum()
                n_neg_noise = neg_noise_mask.sum()

                if n_pos_noise > 0:
                    ax.scatter(pos_coords[pos_noise_mask][:, 0], pos_coords[pos_noise_mask][:, 1],
                             c='darkgreen', s=1, alpha=0.3, rasterized=True)

                if n_neg_noise > 0:
                    ax.scatter(neg_coords[neg_noise_mask][:, 0], neg_coords[neg_noise_mask][:, 1],
                             c='darkred', s=1, alpha=0.3, rasterized=True)

                ax.set_xlabel('X (μm)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Y (μm)', fontsize=12, fontweight='bold')
                ax.set_title(f'{sample} - {marker_name}+/- Zones\n'
                           f'{marker_name}+ zones: {n_pos_zones}, {marker_name}- zones: {n_neg_zones}\n'
                           f'DBSCAN: eps={eps}, min_samples={min_samples}',
                           fontsize=13, fontweight='bold')
                ax.set_aspect('equal')
                ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), fontsize=8, markerscale=2)

                plt.tight_layout()

                plot_path = self.plots_dir / 'marker_zones' / f'{sample}_{marker_name}_zones.png'
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

                n_plots += 1

        print(f"    ✓ Generated {n_plots} marker +/- zone plots")

    def plot_sample_with_spatialcells_boundaries(self, adata, region_detector, sample: str,
                                                 show_immune: bool = True,
                                                 figsize: tuple = (20, 20)):
        """
        Plot sample with SpatialCells-detected tumor boundaries.

        Parameters
        ----------
        adata : AnnData
            Annotated data
        region_detector : SpatialCellsRegionDetector
            Region detector with detected boundaries
        sample : str
            Sample ID
        show_immune : bool
            Whether to show immune populations
        figsize : tuple
            Figure size
        """
        if not HAS_SPATIALCELLS:
            warnings.warn("SpatialCells not available")
            return

        sample_mask = adata.obs['sample_id'] == sample
        sample_adata = adata[sample_mask]

        if len(sample_adata) == 0:
            return

        tumor_col = self.config.get('tumor_definition', {}).get('base_phenotype', 'Tumor')
        tumor_col = f'is_{tumor_col}'
        immune_pops = self.config.get('immune_populations', ['CD8_T_cells', 'CD3_positive', 'CD45_positive'])

        fig, ax = plt.subplots(figsize=figsize)

        coords = sample_adata.obsm['spatial']

        # Determine cell types
        is_tumor = sample_adata.obs[tumor_col].values if tumor_col in sample_adata.obs.columns else np.zeros(len(sample_adata), dtype=bool)
        is_immune = np.zeros(len(sample_adata), dtype=bool)
        for immune_pop in immune_pops:
            immune_col = f'is_{immune_pop}'
            if immune_col in sample_adata.obs.columns:
                is_immune |= sample_adata.obs[immune_col].values

        background_mask = ~(is_tumor | is_immune)

        # Plot background
        if background_mask.sum() > 0:
            ax.scatter(coords[background_mask, 0], coords[background_mask, 1],
                      s=0.5, c='#CCCCCC', alpha=0.2, rasterized=True)

        # Plot tumor cells
        if is_tumor.sum() > 0:
            ax.scatter(coords[is_tumor, 0], coords[is_tumor, 1],
                      s=2, c='#E41A1C', alpha=0.6, rasterized=True, label='Tumor cells')

        # Plot immune populations with different colors
        immune_colors = {
            'CD8_T_cells': '#377EB8',
            'T_cells': '#4DAF4A',
            'CD3_positive': '#4DAF4A',
            'CD45_positive': '#984EA3',
            'CD4_T_cells': '#FF7F00',
            'Macrophages': '#A65628'
        }

        if show_immune:
            for immune_pop in immune_pops:
                immune_col = f'is_{immune_pop}'
                if immune_col not in sample_adata.obs.columns:
                    continue

                immune_mask = sample_adata.obs[immune_col].values
                if immune_mask.sum() > 0:
                    color = immune_colors.get(immune_pop, '#000000')
                    ax.scatter(coords[immune_mask, 0], coords[immune_mask, 1],
                              s=4, c=color, alpha=0.7, rasterized=True,
                              label=immune_pop.replace('_', ' '))

        # Plot boundaries
        if sample in region_detector.tumor_boundaries:
            boundaries = region_detector.tumor_boundaries[sample]
            for tumor_id, boundary in boundaries.items():
                # Validate boundary is a proper 2D array
                boundary_array = np.array(boundary)
                if boundary_array.ndim != 2 or boundary_array.shape[0] < 3 or boundary_array.shape[1] != 2:
                    warnings.warn(f"Invalid boundary shape for tumor {tumor_id} in {sample}: {boundary_array.shape}, skipping")
                    continue

                ax.plot(boundary_array[:, 0], boundary_array[:, 1],
                       'k-', linewidth=3, alpha=0.9)

                # Add tumor ID label
                try:
                    centroid = spc.msmt.getRegionCentroid(boundary)
                    ax.text(centroid[0], centroid[1], f'T{tumor_id}',
                           fontsize=12, fontweight='bold', color='white',
                           ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
                except Exception as e:
                    warnings.warn(f"Could not add label for tumor {tumor_id}: {e}")

        ax.set_xlabel('X (μm)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (μm)', fontsize=14, fontweight='bold')
        ax.set_title(f'Sample: {sample} - Tumor Boundaries and Immune Populations',
                    fontsize=16, fontweight='bold')
        ax.legend(loc='best', frameon=True, fontsize=11, markerscale=3)
        ax.set_aspect('equal')
        plt.tight_layout()

        plot_path = self.plots_dir / 'spatialcells_boundaries' / f'{sample}_boundaries_immune.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_individual_tumor_with_boundary(self, adata, region_detector, sample: str,
                                           tumor_id: int, context_radius: float = 500,
                                           figsize: tuple = (14, 14)):
        """
        Plot individual tumor with boundary and immune cells.

        Parameters
        ----------
        adata : AnnData
            Annotated data
        region_detector : SpatialCellsRegionDetector
            Region detector
        sample : str
            Sample ID
        tumor_id : int
            Tumor region ID
        context_radius : float
            Radius around tumor (μm)
        figsize : tuple
            Figure size
        """
        if not HAS_SPATIALCELLS:
            return

        sample_mask = adata.obs['sample_id'] == sample
        sample_adata = adata[sample_mask]

        boundary = region_detector.get_boundary(sample, tumor_id)
        if boundary is None:
            return

        # Validate boundary is a proper 2D array
        boundary_array = np.array(boundary)
        if boundary_array.ndim != 2 or boundary_array.shape[0] < 3 or boundary_array.shape[1] != 2:
            warnings.warn(f"Invalid boundary shape for tumor {tumor_id} in {sample}: {boundary_array.shape}")
            return

        tumor_col = self.config.get('tumor_definition', {}).get('base_phenotype', 'Tumor')
        tumor_col = f'is_{tumor_col}'
        immune_pops = self.config.get('immune_populations', ['CD8_T_cells', 'CD3_positive', 'CD45_positive'])

        centroid = spc.msmt.getRegionCentroid(boundary)
        coords = sample_adata.obsm['spatial']
        distances = np.sqrt((coords[:, 0] - centroid[0])**2 + (coords[:, 1] - centroid[1])**2)
        context_mask = distances <= context_radius

        if context_mask.sum() == 0:
            return

        context_adata = sample_adata[context_mask]
        context_coords = context_adata.obsm['spatial']

        fig, ax = plt.subplots(figsize=figsize)

        in_tumor = context_adata.obs['tumor_region_id'] == tumor_id if 'tumor_region_id' in context_adata.obs.columns else np.zeros(len(context_adata), dtype=bool)
        is_tumor_type = context_adata.obs[tumor_col].values if tumor_col in context_adata.obs.columns else np.zeros(len(context_adata), dtype=bool)
        is_immune = np.zeros(len(context_adata), dtype=bool)
        for immune_pop in immune_pops:
            immune_col = f'is_{immune_pop}'
            if immune_col in context_adata.obs.columns:
                is_immune |= context_adata.obs[immune_col].values

        background_mask = ~(is_tumor_type | is_immune)
        if background_mask.sum() > 0:
            ax.scatter(context_coords[background_mask, 0], context_coords[background_mask, 1],
                      s=1, c='#CCCCCC', alpha=0.2, rasterized=True)

        # Plot tumor cells in this tumor
        tumor_cells_mask = in_tumor & is_tumor_type
        if tumor_cells_mask.sum() > 0:
            ax.scatter(context_coords[tumor_cells_mask, 0], context_coords[tumor_cells_mask, 1],
                      s=6, c='#E41A1C', alpha=0.8, rasterized=True,
                      label=f'Tumor {tumor_id}', edgecolors='darkred', linewidths=0.5)

        # Other tumor cells
        other_tumor_mask = (~in_tumor) & is_tumor_type
        if other_tumor_mask.sum() > 0:
            ax.scatter(context_coords[other_tumor_mask, 0], context_coords[other_tumor_mask, 1],
                      s=2, c='#FFA500', alpha=0.3, rasterized=True, label='Other tumors')

        # Immune populations
        immune_colors = {'CD8_T_cells': '#377EB8', 'T_cells': '#4DAF4A', 'CD3_positive': '#4DAF4A',
                        'CD45_positive': '#984EA3', 'CD4_T_cells': '#FF7F00'}

        for immune_pop in immune_pops:
            immune_col = f'is_{immune_pop}'
            if immune_col not in context_adata.obs.columns:
                continue

            immune_mask = context_adata.obs[immune_col].values
            if immune_mask.sum() > 0:
                color = immune_colors.get(immune_pop, '#000000')
                ax.scatter(context_coords[immune_mask, 0], context_coords[immune_mask, 1],
                          s=10, c=color, alpha=0.9, rasterized=True,
                          label=immune_pop.replace('_', ' '),
                          edgecolors='black', linewidths=0.3)

        # Boundary (already validated as 2D array above)
        ax.plot(boundary_array[:, 0], boundary_array[:, 1],
               'k-', linewidth=4, alpha=1.0, label='Tumor boundary')

        # Scale bar
        scalebar_length = 100
        x_range = context_coords[:, 0].max() - context_coords[:, 0].min()
        y_range = context_coords[:, 1].max() - context_coords[:, 1].min()
        scalebar_x = context_coords[:, 0].min() + x_range * 0.1
        scalebar_y = context_coords[:, 1].min() + y_range * 0.05
        ax.plot([scalebar_x, scalebar_x + scalebar_length], [scalebar_y, scalebar_y],
               'k-', linewidth=5)
        ax.text(scalebar_x + scalebar_length/2, scalebar_y - y_range*0.03,
               f'{scalebar_length} μm', ha='center', fontsize=11, fontweight='bold')

        n_tumor_cells = tumor_cells_mask.sum()
        area_um2 = spc.msmt.getRegionArea(boundary)

        ax.set_xlabel('X (μm)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (μm)', fontsize=14, fontweight='bold')
        ax.set_title(f'Sample: {sample} | Tumor {tumor_id}\n'
                    f'Size: {n_tumor_cells} cells | Area: {area_um2:.0f} μm²',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, fontsize=10, markerscale=2)
        ax.set_aspect('equal')
        plt.tight_layout()

        sample_dir = self.plots_dir / 'individual_tumors' / sample
        sample_dir.mkdir(parents=True, exist_ok=True)
        plot_path = sample_dir / f'tumor_{tumor_id}_boundary_immune.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def generate_spatialcells_plots(self, adata, region_detector):
        """Generate all SpatialCells-based spatial plots."""
        if not HAS_SPATIALCELLS:
            warnings.warn("SpatialCells not available for spatial plotting")
            return

        print("\n  Generating SpatialCells-based spatial plots...")

        # Debug: Check if region_detector has boundaries
        if not hasattr(region_detector, 'tumor_boundaries'):
            print("    ⚠ region_detector has no tumor_boundaries attribute")
            return

        if not region_detector.tumor_boundaries:
            print("    ⚠ region_detector.tumor_boundaries is empty")
            return

        print(f"    Found {len(region_detector.tumor_boundaries)} samples with tumor boundaries")

        n_samples = 0
        n_tumors = 0
        errors = []

        for sample in region_detector.tumor_boundaries.keys():
            try:
                self.plot_sample_with_spatialcells_boundaries(adata, region_detector, sample)
                n_samples += 1
            except Exception as e:
                import traceback
                error_msg = f"Error plotting sample {sample}: {e}\n{traceback.format_exc()}"
                errors.append(error_msg)
                print(f"    ⚠ {error_msg}")

            boundaries = region_detector.tumor_boundaries[sample]
            print(f"    Processing {len(boundaries)} tumors for sample {sample}")

            for tumor_id in boundaries.keys():
                try:
                    self.plot_individual_tumor_with_boundary(adata, region_detector, sample, tumor_id)
                    n_tumors += 1
                except Exception as e:
                    import traceback
                    error_msg = f"Error plotting tumor {tumor_id} in {sample}: {e}\n{traceback.format_exc()}"
                    errors.append(error_msg)
                    print(f"    ⚠ {error_msg}")

        print(f"    ✓ Generated {n_samples} sample plots and {n_tumors} individual tumor plots with boundaries")
        if errors:
            print(f"    ⚠ Encountered {len(errors)} errors during plotting")
        print(f"    Saved to: {self.plots_dir}/")
