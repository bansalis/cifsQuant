"""
UMAP Visualization - CORRECTED VERSION
Uses raw fluorescence intensities and morphological features
NOT binary gated values (which create artificial discrete blobs)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from sklearn.preprocessing import StandardScaler
import warnings


class UMAPPlotter:
    """
    UMAP dimensionality reduction using PROPER biological features.

    CRITICAL: Uses ungated normalized fluorescence + morphology
    NOT binary gates (which create artificial separation)

    Generates TWO analyses:
    1. All cells (tumor + immune) - full tissue structure
    2. Tumor cells only - tumor heterogeneity
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

    def prepare_umap_features(self, adata, cell_subset_mask: Optional[np.ndarray] = None):
        """
        Prepare PROPER features for UMAP.

        Uses:
        1. Normalized fluorescence intensities (channels 1-10)
        2. Morphological features (area, axis lengths, shape descriptors)
        3. Local spatial density (optional)

        NOT binary gates!

        Parameters
        ----------
        adata : AnnData
            Full annotated data
        cell_subset_mask : np.ndarray, optional
            Boolean mask for cell subset (e.g., tumor only)

        Returns
        -------
        tuple
            (feature_matrix, feature_names, cell_metadata)
        """
        print("  Preparing UMAP features from raw data...")

        if cell_subset_mask is not None:
            obs = adata.obs[cell_subset_mask]
            X = adata.X[cell_subset_mask]
            coords = adata.obsm['spatial'][cell_subset_mask]
        else:
            obs = adata.obs
            X = adata.X
            coords = adata.obsm['spatial']

        feature_list = []
        feature_names = []

        # 1. Normalized fluorescence intensities
        # adata.X should contain normalized marker intensities (from manual_gating.py)
        # These are the REAL biological signals, not binary gates
        marker_names = adata.var_names.tolist()  # e.g., ['TOM', 'CD45', 'AGFP', 'PERK', ...]

        feature_list.append(X)
        feature_names.extend(marker_names)

        print(f"    ✓ Added {len(marker_names)} normalized fluorescence channels")

        # 2. Morphological features
        morph_features = []
        morph_names = []

        morph_cols = ['Area', 'MajorAxisLength', 'MinorAxisLength',
                     'Eccentricity', 'Solidity', 'Extent']

        for col in morph_cols:
            if col in obs.columns:
                values = obs[col].values.reshape(-1, 1)
                morph_features.append(values)
                morph_names.append(col)

        if morph_features:
            morph_matrix = np.hstack(morph_features)
            feature_list.append(morph_matrix)
            feature_names.extend(morph_names)
            print(f"    ✓ Added {len(morph_names)} morphological features")

        # 3. OPTIONAL: Local spatial density
        # Adds spatial context without using coordinates directly
        umap_config = self.config.get('umap_visualization', {})
        if umap_config.get('include_spatial_density', False):
            from scipy.spatial import cKDTree

            # Calculate local cell density (cells within 50μm radius)
            tree = cKDTree(coords)
            neighbor_counts = tree.query_ball_point(coords, r=50, return_length=True)
            density = np.array(neighbor_counts).reshape(-1, 1)

            feature_list.append(density)
            feature_names.append('local_density')
            print(f"    ✓ Added local spatial density")

        # Combine all features
        X_combined = np.hstack(feature_list)

        # Create metadata DataFrame
        metadata = pd.DataFrame({
            'sample_id': obs['sample_id'].values,
            'timepoint': obs['timepoint'].values if 'timepoint' in obs.columns else np.nan,
            'group': obs['group'].values if 'group' in obs.columns else '',
            'main_group': obs['main_group'].values if 'main_group' in obs.columns else ''
        })

        # Add gated phenotypes for coloring (but not for UMAP computation!)
        for col in obs.columns:
            if col.startswith('is_'):
                metadata[col] = obs[col].values

        print(f"  ✓ Prepared {X_combined.shape[1]} features for {X_combined.shape[0]} cells")

        return X_combined, feature_names, metadata

    def compute_umap_embedding(self, X: np.ndarray, feature_names: List[str],
                              metadata: pd.DataFrame,
                              n_neighbors: int = 30,
                              min_dist: float = 0.3,
                              subsample: Optional[int] = None,
                              name_suffix: str = ""):
        """
        Compute UMAP embedding from feature matrix.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (cells x features)
        feature_names : list
            Feature names
        metadata : pd.DataFrame
            Cell metadata
        n_neighbors : int
            UMAP n_neighbors parameter
        min_dist : float
            UMAP min_dist parameter
        subsample : int, optional
            Maximum cells to use
        name_suffix : str
            Suffix for output files (e.g., "_all_cells" or "_tumor_only")

        Returns
        -------
        pd.DataFrame
            UMAP coordinates + metadata
        """
        try:
            import umap
        except ImportError:
            print("    ⚠ umap-learn not installed. Install with: pip install umap-learn")
            return None

        print(f"  Computing UMAP{name_suffix}...")

        # Subsample if needed
        if subsample and len(X) > subsample:
            indices = np.random.choice(len(X), subsample, replace=False)
            X = X[indices]
            metadata = metadata.iloc[indices].reset_index(drop=True)
            print(f"    Subsampled to {subsample} cells")

        # Standardize features (CRITICAL for proper UMAP)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(f"    Running UMAP on {len(X)} cells with {X.shape[1]} features...")
        print(f"    Features: {', '.join(feature_names[:10])}{'...' if len(feature_names) > 10 else ''}")

        # Compute UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric='euclidean',
            random_state=42,
            n_jobs=-1
        )

        embedding = reducer.fit_transform(X_scaled)

        # Create result DataFrame
        result_df = metadata.copy()
        result_df['UMAP1'] = embedding[:, 0]
        result_df['UMAP2'] = embedding[:, 1]

        print(f"    ✓ UMAP computed successfully")

        return result_df

    def compute_clusters(self, umap_df: pd.DataFrame, n_clusters: int = 10,
                        name_suffix: str = ""):
        """
        Cluster cells in UMAP space.

        Parameters
        ----------
        umap_df : pd.DataFrame
            UMAP embedding DataFrame
        n_clusters : int
            Number of clusters
        name_suffix : str
            Suffix for output

        Returns
        -------
        pd.DataFrame
            Input DataFrame with added 'cluster' column
        """
        from sklearn.cluster import KMeans

        print(f"    Clustering{name_suffix} into {n_clusters} clusters...")

        X = umap_df[['UMAP1', 'UMAP2']].values

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        umap_df['cluster'] = clusters

        print(f"    ✓ Clustering complete")

        return umap_df

    def plot_umap_clusters(self, umap_df: pd.DataFrame, name_suffix: str = ""):
        """Plot UMAP colored by clusters."""
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
        title = f'Cell Clusters in UMAP Space{name_suffix}'
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.legend(markerscale=5, fontsize=9, ncol=2, loc='best')
        ax.set_aspect('equal')

        plt.tight_layout()

        # Save
        filename = f'umap_clusters{name_suffix.replace(" ", "_").lower()}.png'
        plot_path = self.plots_dir / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Saved cluster UMAP to {plot_path.name}")

    def plot_umap_by_markers(self, umap_df: pd.DataFrame, marker_names: List[str],
                            name_suffix: str = ""):
        """
        Plot UMAP colored by gated marker expression.

        Note: Markers used for COLORING only, not for UMAP computation!
        """
        # Get available gated markers
        available_markers = [f'is_{m}' for m in ['PERK', 'AGFP', 'KI67', 'Tumor',
                            'CD45_positive', 'CD3_positive', 'CD8_T_cells']
                            if f'is_{m}' in umap_df.columns]

        if not available_markers:
            print(f"    ⚠ No gated markers available for coloring{name_suffix}")
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

            marker_name = marker.replace('is_', '')

            pos_mask = umap_df[marker] == 1
            neg_mask = umap_df[marker] == 0

            # Plot negative cells first
            ax.scatter(umap_df.loc[neg_mask, 'UMAP1'],
                      umap_df.loc[neg_mask, 'UMAP2'],
                      c='lightgray', s=1, alpha=0.2, label=f'{marker_name}-', rasterized=True)

            # Plot positive cells on top
            ax.scatter(umap_df.loc[pos_mask, 'UMAP1'],
                      umap_df.loc[pos_mask, 'UMAP2'],
                      c='red', s=2, alpha=0.6, label=f'{marker_name}+', rasterized=True)

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

        title = f'Marker Expression in UMAP Space{name_suffix}'
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save
        filename = f'umap_markers{name_suffix.replace(" ", "_").lower()}.png'
        plot_path = self.plots_dir / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Saved marker UMAP to {plot_path.name}")

    def generate_all_plots(self, adata):
        """
        Generate UMAP plots for BOTH all cells and tumor-only.

        Parameters
        ----------
        adata : AnnData
            Annotated data with RAW fluorescence (not just gates!)
        """
        print("\n" + "="*80)
        print("UMAP VISUALIZATION (RAW FLUORESCENCE + MORPHOLOGY)")
        print("="*80)
        print("  IMPORTANT: Using ungated fluorescence + morphological features")
        print("  NOT using binary gates (which create artificial blobs)")
        print("="*80)

        umap_config = self.config.get('umap_visualization', {})
        subsample = umap_config.get('subsample', 100000)
        n_neighbors = umap_config.get('n_neighbors', 30)
        min_dist = umap_config.get('min_dist', 0.3)
        n_clusters = umap_config.get('n_clusters', 10)

        # =====================================================================
        # ANALYSIS 1: ALL CELLS (tumor + immune)
        # =====================================================================
        print("\n1. UMAP for ALL CELLS (tumor + immune)...")

        X_all, features_all, metadata_all = self.prepare_umap_features(adata)

        umap_all = self.compute_umap_embedding(
            X_all, features_all, metadata_all,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            subsample=subsample,
            name_suffix=" (all cells)"
        )

        if umap_all is not None:
            umap_all = self.compute_clusters(umap_all, n_clusters=n_clusters,
                                            name_suffix=" (all cells)")

            # Save coordinates
            umap_path = self.output_dir / 'umap_coordinates_all_cells.csv'
            umap_all.to_csv(umap_path, index=False)
            print(f"  ✓ Saved coordinates to {umap_path.name}")

            # Generate plots
            self.plot_umap_clusters(umap_all, name_suffix=" (All Cells)")
            self.plot_umap_by_markers(umap_all, features_all, name_suffix=" (All Cells)")

        # =====================================================================
        # ANALYSIS 2: TUMOR CELLS ONLY
        # =====================================================================
        print("\n2. UMAP for TUMOR CELLS ONLY...")

        if 'is_Tumor' in adata.obs.columns:
            tumor_mask = adata.obs['is_Tumor'].values.astype(bool)
            n_tumor = tumor_mask.sum()

            if n_tumor < 100:
                print(f"  ⚠ Only {n_tumor} tumor cells, skipping tumor-only UMAP")
            else:
                X_tumor, features_tumor, metadata_tumor = self.prepare_umap_features(
                    adata, cell_subset_mask=tumor_mask
                )

                umap_tumor = self.compute_umap_embedding(
                    X_tumor, features_tumor, metadata_tumor,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    subsample=subsample,
                    name_suffix=" (tumor only)"
                )

                if umap_tumor is not None:
                    umap_tumor = self.compute_clusters(umap_tumor, n_clusters=n_clusters,
                                                      name_suffix=" (tumor only)")

                    # Save coordinates
                    umap_path = self.output_dir / 'umap_coordinates_tumor_only.csv'
                    umap_tumor.to_csv(umap_path, index=False)
                    print(f"  ✓ Saved coordinates to {umap_path.name}")

                    # Generate plots
                    self.plot_umap_clusters(umap_tumor, name_suffix=" (Tumor Only)")
                    self.plot_umap_by_markers(umap_tumor, features_tumor, name_suffix=" (Tumor Only)")
        else:
            print("  ⚠ No tumor phenotype found, skipping tumor-only UMAP")

        print("\n" + "="*80)
        print("✓ UMAP VISUALIZATION COMPLETE")
        print(f"  All plots saved to {self.plots_dir}/")
        print("="*80 + "\n")
