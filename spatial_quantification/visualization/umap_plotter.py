"""
UMAP Visualization
Dimensionality reduction and clustering visualization of cell populations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import warnings


class UMAPPlotter:
    """
    Comprehensive UMAP visualization for cell populations.

    Generates:
    - UMAP embeddings colored by clusters
    - UMAP embeddings colored by marker expression
    - UMAP embeddings colored by phenotypes
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

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300

    def compute_umap_embedding(self, adata, markers: list,
                              n_neighbors: int = 30,
                              min_dist: float = 0.3,
                              subsample: Optional[int] = None):
        """
        Compute UMAP embedding from marker expression.

        Parameters
        ----------
        adata : AnnData
            Annotated data
        markers : list
            Marker column names to use for UMAP
        n_neighbors : int
            UMAP n_neighbors parameter
        min_dist : float
            UMAP min_dist parameter
        subsample : int, optional
            Maximum cells to use (subsample if larger)

        Returns
        -------
        pd.DataFrame
            DataFrame with UMAP coordinates and cell metadata
        """
        try:
            import umap
        except ImportError:
            print("    ⚠ umap-learn not installed. Install with: pip install umap-learn")
            return None

        print(f"    Computing UMAP from {len(markers)} markers...")

        # Get available markers
        available_markers = [m for m in markers if m in adata.obs.columns]

        if len(available_markers) < 2:
            print(f"    ⚠ Insufficient markers available for UMAP")
            return None

        # Extract marker matrix
        X = adata.obs[available_markers].values.astype(float)

        # Subsample if needed
        if subsample and len(X) > subsample:
            indices = np.random.choice(len(X), subsample, replace=False)
            X = X[indices]
            obs_subset = adata.obs.iloc[indices]
        else:
            indices = np.arange(len(X))
            obs_subset = adata.obs

        # Compute UMAP
        print(f"    Running UMAP on {len(X)} cells...")
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric='euclidean',
            random_state=42
        )

        embedding = reducer.fit_transform(X)

        # Create result DataFrame
        result_df = pd.DataFrame({
            'UMAP1': embedding[:, 0],
            'UMAP2': embedding[:, 1],
            'sample_id': obs_subset['sample_id'].values,
            'timepoint': obs_subset['timepoint'].values if 'timepoint' in obs_subset.columns else np.nan,
            'group': obs_subset['group'].values if 'group' in obs_subset.columns else '',
            'main_group': obs_subset['main_group'].values if 'main_group' in obs_subset.columns else ''
        })

        # Add marker values
        for marker in available_markers:
            result_df[marker] = obs_subset[marker].values

        print(f"    ✓ UMAP computed successfully")

        return result_df

    def compute_clusters(self, umap_df: pd.DataFrame, n_clusters: int = 10):
        """
        Cluster cells in UMAP space.

        Parameters
        ----------
        umap_df : pd.DataFrame
            UMAP embedding DataFrame
        n_clusters : int
            Number of clusters

        Returns
        -------
        pd.DataFrame
            Input DataFrame with added 'cluster' column
        """
        from sklearn.cluster import KMeans

        print(f"    Clustering cells into {n_clusters} clusters...")

        X = umap_df[['UMAP1', 'UMAP2']].values

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        umap_df['cluster'] = clusters

        print(f"    ✓ Clustering complete")

        return umap_df

    def plot_umap_clusters(self, umap_df: pd.DataFrame):
        """
        Plot UMAP colored by clusters.

        Parameters
        ----------
        umap_df : pd.DataFrame
            UMAP data with cluster assignments
        """
        if 'cluster' not in umap_df.columns:
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        n_clusters = umap_df['cluster'].nunique()
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

        for cluster_id in sorted(umap_df['cluster'].unique()):
            cluster_data = umap_df[umap_df['cluster'] == cluster_id]

            ax.scatter(cluster_data['UMAP1'], cluster_data['UMAP2'],
                      c=[colors[cluster_id]], s=2, alpha=0.5,
                      label=f'Cluster {cluster_id}')

        ax.set_xlabel('UMAP1', fontsize=13, fontweight='bold')
        ax.set_ylabel('UMAP2', fontsize=13, fontweight='bold')
        ax.set_title('Cell Clusters in UMAP Space', fontsize=15, fontweight='bold')
        ax.legend(markerscale=5, fontsize=9, ncol=2, loc='best')
        ax.set_aspect('equal')

        plt.tight_layout()

        # Save
        plot_path = self.plots_dir / 'umap_clusters.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Saved cluster UMAP to {plot_path}")

    def plot_umap_by_markers(self, umap_df: pd.DataFrame, markers: list):
        """
        Plot UMAP colored by marker expression.

        Parameters
        ----------
        umap_df : pd.DataFrame
            UMAP embedding data
        markers : list
            Marker columns to plot
        """
        # Get available markers
        available_markers = [m for m in markers if m in umap_df.columns]

        if not available_markers:
            return

        n_markers = len(available_markers)
        ncols = min(3, n_markers)
        nrows = (n_markers + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))

        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]

        for idx, marker in enumerate(available_markers):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col]

            # Get marker name for display
            marker_name = marker.replace('is_', '')

            # Separate positive and negative cells
            pos_mask = umap_df[marker] == 1
            neg_mask = umap_df[marker] == 0

            # Plot negative cells first (background)
            ax.scatter(umap_df.loc[neg_mask, 'UMAP1'],
                      umap_df.loc[neg_mask, 'UMAP2'],
                      c='lightgray', s=1, alpha=0.2, label=f'{marker_name}-')

            # Plot positive cells on top
            ax.scatter(umap_df.loc[pos_mask, 'UMAP1'],
                      umap_df.loc[pos_mask, 'UMAP2'],
                      c='red', s=2, alpha=0.6, label=f'{marker_name}+')

            ax.set_xlabel('UMAP1', fontsize=11, fontweight='bold')
            ax.set_ylabel('UMAP2', fontsize=11, fontweight='bold')
            ax.set_title(f'{marker_name} Expression', fontsize=12, fontweight='bold')
            ax.legend(markerscale=5, fontsize=9)
            ax.set_aspect('equal')

        # Hide unused subplots
        for idx in range(n_markers, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row][col].axis('off')

        plt.suptitle('Marker Expression in UMAP Space', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save
        plot_path = self.plots_dir / 'umap_markers.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Saved marker UMAP to {plot_path}")

    def plot_umap_by_phenotypes(self, adata, umap_df: pd.DataFrame, phenotypes: list):
        """
        Plot UMAP colored by phenotypes.

        Parameters
        ----------
        adata : AnnData
            Annotated data
        umap_df : pd.DataFrame
            UMAP embedding data
        phenotypes : list
            Phenotype columns to plot
        """
        # Get available phenotypes
        available_phenotypes = [p for p in phenotypes if f'is_{p}' in adata.obs.columns]

        if not available_phenotypes:
            return

        n_phenos = len(available_phenotypes)
        ncols = min(3, n_phenos)
        nrows = (n_phenos + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))

        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]

        # Get phenotype data for the same cells as UMAP
        # Note: Assumes umap_df indices match adata.obs indices or we subsampled
        for idx, pheno in enumerate(available_phenotypes):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col]

            pheno_col = f'is_{pheno}'

            # Add phenotype to umap_df if not already there
            if pheno_col not in umap_df.columns:
                # Try to match by sample_id and add phenotype
                # This is tricky if we subsampled, so skip if not present
                continue

            pos_mask = umap_df[pheno_col] == 1
            neg_mask = umap_df[pheno_col] == 0

            # Plot negative cells
            ax.scatter(umap_df.loc[neg_mask, 'UMAP1'],
                      umap_df.loc[neg_mask, 'UMAP2'],
                      c='lightgray', s=1, alpha=0.2, label='Other')

            # Plot positive cells
            ax.scatter(umap_df.loc[pos_mask, 'UMAP1'],
                      umap_df.loc[pos_mask, 'UMAP2'],
                      c='blue', s=2, alpha=0.6, label=pheno)

            ax.set_xlabel('UMAP1', fontsize=11, fontweight='bold')
            ax.set_ylabel('UMAP2', fontsize=11, fontweight='bold')
            ax.set_title(f'{pheno} Phenotype', fontsize=12, fontweight='bold')
            ax.legend(markerscale=5, fontsize=9)
            ax.set_aspect('equal')

        # Hide unused subplots
        for idx in range(n_phenos, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row][col].axis('off')

        plt.suptitle('Phenotypes in UMAP Space', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save
        plot_path = self.plots_dir / 'umap_phenotypes.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Saved phenotype UMAP to {plot_path}")

    def generate_all_plots(self, adata):
        """
        Generate all UMAP plots.

        Parameters
        ----------
        adata : AnnData
            Annotated data with phenotypes
        """
        print("  Generating UMAP visualizations...")

        # Get markers from config
        umap_config = self.config.get('umap_visualization', {})
        markers = umap_config.get('markers', [
            'is_PERK', 'is_AGFP', 'is_KI67', 'is_Tumor',
            'is_CD45_positive', 'is_CD3_positive', 'is_CD8_T_cells'
        ])

        # Compute UMAP
        subsample = umap_config.get('subsample', 100000)  # Subsample to 100k cells for speed
        umap_df = self.compute_umap_embedding(
            adata, markers,
            n_neighbors=umap_config.get('n_neighbors', 30),
            min_dist=umap_config.get('min_dist', 0.3),
            subsample=subsample
        )

        if umap_df is None:
            print("    ⚠ UMAP computation failed")
            return

        # Add phenotype columns to umap_df
        # Match by index if indices align
        for pheno in ['Tumor', 'CD45_positive', 'CD3_positive', 'CD8_T_cells']:
            col = f'is_{pheno}'
            if col in adata.obs.columns and col not in umap_df.columns:
                # Add if we can safely match indices
                try:
                    if len(umap_df) == len(adata.obs):
                        umap_df[col] = adata.obs[col].values
                    # If subsampled, skip adding (would need index matching)
                except:
                    pass

        # Compute clusters
        n_clusters = umap_config.get('n_clusters', 10)
        umap_df = self.compute_clusters(umap_df, n_clusters=n_clusters)

        # Save UMAP coordinates
        umap_path = self.output_dir / 'umap_coordinates.csv'
        umap_df.to_csv(umap_path, index=False)
        print(f"    ✓ Saved UMAP coordinates to {umap_path}")

        # Generate plots
        self.plot_umap_clusters(umap_df)
        self.plot_umap_by_markers(umap_df, markers)
        self.plot_umap_by_phenotypes(adata, umap_df, umap_config.get('phenotypes', [
            'Tumor', 'CD45_positive', 'CD3_positive', 'CD8_T_cells', 'CD4_T_cells'
        ]))

        print(f"  ✓ All UMAP plots saved to {self.plots_dir}/")
