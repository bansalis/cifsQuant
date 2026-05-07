"""
Spatial Lag Analysis — Tumor Cell Communities (TCCs)

Reproduces the Tumor Cell Community (TCC) methodology from:
    Nirmal et al. 2022 Cancer Discovery, Fig 5C-F

Algorithm (per Nirmal et al.):
  1. Restrict to Melanocytes + Tumor_cells (configurable via ``tumor_phenotypes``).
  2. For each tumor cell, retrieve all cells within `lag_radius` μm using a
     per-sample KDTree and compute the mean expression of each marker across
     the neighborhood — this is the "spatial lag vector".
  3. PCA on the spatial lag vectors (keep top n_pcs components).
  4. K-means (k=n_tccs) on PCA-reduced lag vectors → TCC assignments.
  5. Store TCC labels in ``adata.obs['tcc']`` (NaN for non-tumor cells).
  6. Compute per-TCC mean marker expression, % positive per marker, and
     per-stage / per-sample TCC frequency distributions.

Marker intensities are read from ``adata.X`` (dense or sparse) with marker
names indexed via ``adata.var_names``.  Spatial coordinates are read from
``adata.obsm['spatial']``.
"""

import gc
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class SpatialLagAnalysis:
    """
    Tumor Cell Communities (TCCs) via spatial lag + PCA + K-means, following
    Nirmal et al. 2022 Cancer Discovery.

    Constructor parameters
    ----------------------
    adata : AnnData
        Annotated data matrix.  Marker expression intensities must be in
        ``adata.X`` with feature names accessible via ``adata.var_names``.
        Spatial coordinates must be in ``adata.obsm['spatial']`` (n_cells × 2).
        Cell-type columns ``is_<CellType>`` (boolean) must be in ``adata.obs``.
    config : dict
        Full pipeline configuration dict.  The section consumed by this
        class is ``config['spatial_lag_analysis']``.
    output_dir : Path or str
        Root output directory; results are written to a
        ``spatial_lag_analysis/`` subdirectory.
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        self.adata = adata
        self.config = config
        self.output_dir = Path(output_dir) / 'spatial_lag_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pull analysis-specific config block
        self.analysis_config: Dict = config.get('spatial_lag_analysis', {})

        # Parameters
        self.lag_radius: float = self.analysis_config.get('lag_radius', 20.0)
        self.n_tccs: int = self.analysis_config.get('n_tccs', 10)
        self.n_pcs: int = self.analysis_config.get('n_pcs', 20)
        self.random_seed: int = self.analysis_config.get('random_seed', 42)
        self.stage_column: str = self.analysis_config.get('stage_column', 'stage_group')
        self.generate_plots: bool = self.analysis_config.get('generate_plots', True)

        # Which phenotypes constitute "tumor cells"
        default_tumor_phenotypes = ['Melanocytes', 'Tumor_cells']
        self.tumor_phenotypes: List[str] = self.analysis_config.get(
            'tumor_phenotypes', default_tumor_phenotypes
        )

        # Markers to include in spatial lag vectors
        default_markers = [
            'S100a', 'MART1', 'SOX10', 'HLADPB1', 'panCK',
            'CD163', 'CD11c', 'CD3d', 'CD8a', 'FOXP3',
            'PD1', 'PDL1', 'TIM3', 'LAG3',
            'pERK', 'CCND1', 'MYC',
        ]
        self.markers: List[str] = self.analysis_config.get('markers', default_markers)

        # Validate configuration against actual data
        self.tumor_phenotypes = self._validate_phenotypes(self.tumor_phenotypes)
        self.markers = self._validate_markers(self.markers)

        # Results store
        self.results: Dict = {}

        np.random.seed(self.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """
        Run the full TCC analysis pipeline.

        Returns
        -------
        dict with keys:
            tcc_composition        : DataFrame (markers × TCCs), mean expression
            tcc_stage_distribution : DataFrame (stage × TCC), fraction
            tcc_sample_distribution: DataFrame (sample × TCC), fraction
            tcc_marker_fractions   : DataFrame (markers × TCCs), % positive
            cell_tcc_assignments   : Series, TCC label per tumor cell
            pca_model              : fitted PCA
            kmeans_model           : fitted KMeans
            pca_variance_explained : array of explained variance ratios
        """
        print("\n" + "=" * 80)
        print("SPATIAL LAG ANALYSIS — TUMOR CELL COMMUNITIES (TCCs)")
        print("=" * 80)
        print(f"  Lag radius          : {self.lag_radius} μm")
        print(f"  n_tccs              : {self.n_tccs}")
        print(f"  n_pcs               : {self.n_pcs}")
        print(f"  Tumor phenotypes    : {self.tumor_phenotypes}")
        print(f"  Markers             : {len(self.markers)}")
        print(f"  Random seed         : {self.random_seed}")

        if len(self.tumor_phenotypes) == 0:
            warnings.warn("SpatialLagAnalysis: No valid tumor phenotypes. Aborting.")
            return self.results

        if len(self.markers) == 0:
            warnings.warn("SpatialLagAnalysis: No valid markers found. Aborting.")
            return self.results

        # Step 1 — identify tumor cells
        print("\n[1/5] Identifying tumor cells...")
        tumor_global_idx = self._get_tumor_cell_indices()

        if len(tumor_global_idx) == 0:
            warnings.warn("SpatialLagAnalysis: No tumor cells found. Aborting.")
            return self.results

        print(f"  ✓ {len(tumor_global_idx):,} tumor cells identified")

        # Step 2 — build spatial lag vectors
        print("\n[2/5] Computing spatial lag vectors...")
        lag_matrix = self._build_lag_matrix(tumor_global_idx)

        if lag_matrix is None or lag_matrix.shape[0] == 0:
            warnings.warn("SpatialLagAnalysis: Spatial lag matrix is empty. Aborting.")
            return self.results

        print(f"  ✓ Lag matrix: {lag_matrix.shape[0]:,} cells × {lag_matrix.shape[1]} markers")

        # Step 3 — PCA
        print(f"\n[3/5] PCA on lag vectors (keeping {self.n_pcs} components)...")
        pca_model, pca_coords = self._fit_pca(lag_matrix)
        variance_explained = pca_model.explained_variance_ratio_
        cumulative = np.cumsum(variance_explained)
        print(f"  ✓ PCA fitted — cumulative variance explained by {self.n_pcs} PCs: "
              f"{cumulative[-1] * 100:.1f}%")

        # Step 4 — K-means on PCA-reduced vectors
        print(f"\n[4/5] K-means clustering ({self.n_tccs} TCCs) on PCA coordinates...")
        kmeans_model, tcc_labels = self._fit_kmeans(pca_coords)
        print(f"  ✓ K-means fitted — {len(np.unique(tcc_labels))} TCCs assigned")

        # Step 5 — write labels to adata.obs, compute statistics
        print("\n[5/5] Writing TCC labels to adata.obs and computing statistics...")
        self._assign_labels_to_adata(tumor_global_idx, tcc_labels)
        self._compute_statistics(lag_matrix, tumor_global_idx, tcc_labels)

        # Store models and variance
        self.results['pca_model'] = pca_model
        self.results['kmeans_model'] = kmeans_model
        self.results['pca_variance_explained'] = variance_explained

        # Save outputs
        self._save_results()

        print("\n" + "=" * 80)
        print("SPATIAL LAG ANALYSIS COMPLETE")
        print(f"  Results saved to: {self.output_dir}/")
        print("=" * 80 + "\n")

        return self.results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_phenotypes(self, phenotypes: List[str]) -> List[str]:
        """Return only phenotypes whose is_<Phenotype> column exists."""
        valid = []
        missing = []
        for pt in phenotypes:
            col = f'is_{pt}'
            if col in self.adata.obs.columns:
                valid.append(pt)
            else:
                missing.append(pt)
        if missing:
            warnings.warn(
                f"SpatialLagAnalysis: {len(missing)} tumor-phenotype columns not found "
                f"in adata.obs and will be skipped: {missing}"
            )
        return valid

    def _validate_markers(self, markers: List[str]) -> List[str]:
        """Return only markers present in adata.var_names."""
        if not hasattr(self.adata, 'var_names') or self.adata.var_names is None:
            warnings.warn(
                "SpatialLagAnalysis: adata.var_names is not set; cannot validate markers."
            )
            return markers

        available = set(self.adata.var_names.tolist())
        valid = [m for m in markers if m in available]
        missing = [m for m in markers if m not in available]
        if missing:
            warnings.warn(
                f"SpatialLagAnalysis: {len(missing)} markers not found in adata.var_names "
                f"and will be skipped: {missing}"
            )
        return valid

    def _get_tumor_cell_indices(self) -> np.ndarray:
        """Return positional indices of all tumor cells across all samples."""
        if len(self.tumor_phenotypes) == 0:
            return np.array([], dtype=int)

        tumor_mask = np.zeros(len(self.adata), dtype=bool)
        for pt in self.tumor_phenotypes:
            col = f'is_{pt}'
            tumor_mask |= self.adata.obs[col].values.astype(bool)

        return np.where(tumor_mask)[0]

    def _get_marker_matrix(self) -> Optional[np.ndarray]:
        """
        Extract a dense float32 matrix of shape (n_cells, n_markers) from
        adata.X, restricted to self.markers columns.

        Returns None if adata.X is not accessible.
        """
        if self.adata.X is None:
            return None

        marker_indices = [self.adata.var_names.get_loc(m) for m in self.markers]

        if hasattr(self.adata.X, 'toarray'):
            # Sparse matrix
            X = self.adata.X[:, marker_indices].toarray().astype(np.float32)
        else:
            X = np.asarray(self.adata.X[:, marker_indices], dtype=np.float32)

        return X

    def _build_lag_matrix(self, tumor_global_idx: np.ndarray) -> Optional[np.ndarray]:
        """
        For each tumor cell, compute the mean expression of each marker across
        all cells (any phenotype) within `lag_radius` μm, using per-sample
        KDTree queries.

        Parameters
        ----------
        tumor_global_idx : np.ndarray
            Global (adata-level) integer positions of tumor cells.

        Returns
        -------
        lag_matrix : np.ndarray, shape (n_tumor_cells, n_markers)
            Spatial lag vectors; rows correspond to ``tumor_global_idx``.
        """
        if 'spatial' not in self.adata.obsm:
            warnings.warn(
                "SpatialLagAnalysis: 'spatial' key not found in adata.obsm; "
                "cannot compute spatial lag vectors."
            )
            return None

        # Pre-extract full marker matrix (all cells × selected markers)
        X_full = self._get_marker_matrix()
        if X_full is None:
            warnings.warn(
                "SpatialLagAnalysis: adata.X is None; cannot build lag matrix."
            )
            return None

        all_coords = self.adata.obsm['spatial']  # (n_cells, 2)
        obs = self.adata.obs

        lag_matrix = np.zeros((len(tumor_global_idx), len(self.markers)),
                              dtype=np.float32)

        samples = obs['sample_id'].unique()

        for sample in samples:
            # Global indices for this sample (all cells)
            sample_all_mask = (obs['sample_id'] == sample).values
            sample_all_global = np.where(sample_all_mask)[0]

            # Tumor cells in this sample
            tumor_in_sample_local = np.isin(tumor_global_idx, sample_all_global)
            if not tumor_in_sample_local.any():
                continue

            tumor_in_sample_global = tumor_global_idx[tumor_in_sample_local]
            # Local positions within the lag_matrix output
            lag_matrix_local_idx = np.where(tumor_in_sample_local)[0]

            sample_coords_all = all_coords[sample_all_mask]  # (n_sample, 2)
            sample_X = X_full[sample_all_global]             # (n_sample, n_markers)

            # Build KDTree on ALL cells in the sample
            tree = cKDTree(sample_coords_all)

            # Map global tumor indices → local positions within sample_all_global
            # Use searchsorted (vectorised) instead of a Python dict comprehension
            tumor_local_positions = np.searchsorted(sample_all_global,
                                                    tumor_in_sample_global)
            tumor_coords = sample_coords_all[tumor_local_positions]  # (n_tumor, 2)

            # Radius query for each tumor cell — C-level
            n_tumor_sample = len(tumor_coords)
            neighbor_lists = tree.query_ball_point(tumor_coords, r=self.lag_radius)

            # Build adjacency without per-cell Python loop:
            # rows  = tumor cell index (0..n_tumor_sample-1)
            # cols  = neighbor index in sample_coords_all (0..n_sample-1)
            neighbor_counts = np.array([len(nbrs) for nbrs in neighbor_lists],
                                       dtype=np.int32)
            total_edges = int(neighbor_counts.sum())

            if total_edges > 0:
                row_idx = np.repeat(np.arange(n_tumor_sample, dtype=np.int32),
                                    neighbor_counts)
                col_idx = np.concatenate([np.asarray(n, dtype=np.int32)
                                          for n in neighbor_lists])

                # Weighted (1/count) so that dot gives the mean
                counts_per_row = neighbor_counts[row_idx].astype(np.float32)
                weights = np.where(counts_per_row > 0, 1.0 / counts_per_row, 0.0).astype(np.float32)

                adj = csr_matrix(
                    (weights, (row_idx, col_idx)),
                    shape=(n_tumor_sample, len(sample_all_global))
                )
                # adj.dot(sample_X) gives mean expression per tumor cell — C-BLAS
                lag_block = np.asarray(adj.dot(sample_X), dtype=np.float32)

                # Fallback for tumor cells with zero neighbors: use self expression
                no_nbr = neighbor_counts == 0
                if no_nbr.any():
                    lag_block[no_nbr] = sample_X[tumor_local_positions[no_nbr]]

                del adj, row_idx, col_idx, weights
            else:
                # No neighbors at all — use self expression for every tumor cell
                lag_block = sample_X[tumor_local_positions]

            lag_matrix[lag_matrix_local_idx] = lag_block
            del lag_block, sample_X
            gc.collect()

        return lag_matrix

    def _fit_pca(self, lag_matrix: np.ndarray) -> Tuple:
        """Fit PCA on spatial lag vectors."""
        n_pcs_actual = min(self.n_pcs, lag_matrix.shape[1], lag_matrix.shape[0] - 1)
        pca = PCA(n_components=n_pcs_actual, random_state=self.random_seed)
        pca_coords = pca.fit_transform(lag_matrix)
        gc.collect()
        return pca, pca_coords

    def _fit_kmeans(self, pca_coords: np.ndarray) -> Tuple:
        """K-means on PCA-reduced lag vectors."""
        n_tccs_actual = min(self.n_tccs, len(pca_coords))
        kmeans = KMeans(
            n_clusters=n_tccs_actual,
            random_state=self.random_seed,
            n_init=10,
            max_iter=300,
        )
        labels = kmeans.fit_predict(pca_coords)
        return kmeans, labels

    def _assign_labels_to_adata(self, tumor_global_idx: np.ndarray,
                                tcc_labels: np.ndarray):
        """Write tcc column into adata.obs (NaN for non-tumor cells)."""
        self.adata.obs['tcc'] = np.nan
        self.adata.obs['tcc'] = self.adata.obs['tcc'].astype(object)

        obs_index = self.adata.obs.index
        for pos, label in zip(tumor_global_idx, tcc_labels):
            self.adata.obs.at[obs_index[pos], 'tcc'] = int(label)

        self.adata.obs['tcc'] = pd.to_numeric(
            self.adata.obs['tcc'], errors='coerce'
        ).astype('Int64')

        n_assigned = self.adata.obs['tcc'].notna().sum()
        print(f"  ✓ Assigned TCC labels to {n_assigned:,} tumor cells "
              f"({n_assigned / len(self.adata) * 100:.1f}% of all cells)")

    def _compute_statistics(self, lag_matrix: np.ndarray,
                            tumor_global_idx: np.ndarray,
                            tcc_labels: np.ndarray):
        """Compute TCC-level statistics and store in self.results."""
        obs = self.adata.obs
        unique_tccs = sorted(np.unique(tcc_labels).tolist())

        # Build a DataFrame indexing lag_matrix rows
        tumor_obs = obs.iloc[tumor_global_idx].copy()
        tumor_obs = tumor_obs.assign(
            _tcc=tcc_labels,
            _lag_row=np.arange(len(tumor_global_idx))
        )

        # --- Mean marker expression per TCC ---
        composition_rows: Dict = {}
        for tcc_id in unique_tccs:
            tcc_mask = tcc_labels == tcc_id
            mean_expr = lag_matrix[tcc_mask].mean(axis=0)
            composition_rows[f'TCC_{tcc_id}'] = mean_expr
        tcc_composition = pd.DataFrame(
            composition_rows, index=self.markers
        )
        self.results['tcc_composition'] = tcc_composition
        print(f"  ✓ TCC composition: {tcc_composition.shape}")

        # --- % positive cells per marker per TCC ---
        # Use raw expression from adata.X; define threshold as > 0
        X_tumor = self._get_tumor_marker_matrix(tumor_global_idx)
        if X_tumor is not None:
            fraction_rows: Dict = {}
            for tcc_id in unique_tccs:
                tcc_mask = tcc_labels == tcc_id
                tcc_X = X_tumor[tcc_mask]
                pct_pos = (tcc_X > 0).mean(axis=0) * 100.0
                fraction_rows[f'TCC_{tcc_id}'] = pct_pos
            tcc_fractions = pd.DataFrame(fraction_rows, index=self.markers)
            self.results['tcc_marker_fractions'] = tcc_fractions
            print(f"  ✓ TCC marker fractions: {tcc_fractions.shape}")
        else:
            self.results['tcc_marker_fractions'] = pd.DataFrame()

        # --- Per-sample TCC fraction ---
        sample_dists = self._compute_group_distribution(
            tumor_obs, '_tcc', unique_tccs, 'sample_id'
        )
        self.results['tcc_sample_distribution'] = sample_dists
        self.results['cell_tcc_assignments'] = self.adata.obs['tcc']
        print(f"  ✓ Per-sample TCC distribution: {sample_dists.shape}")

        # --- Per-stage-group TCC fraction ---
        if self.stage_column in tumor_obs.columns:
            stage_dists = self._compute_group_distribution(
                tumor_obs, '_tcc', unique_tccs, self.stage_column
            )
            self.results['tcc_stage_distribution'] = stage_dists
            print(f"  ✓ Per-stage TCC distribution: {stage_dists.shape}")
        else:
            warnings.warn(
                f"SpatialLagAnalysis: Stage column '{self.stage_column}' not found "
                "in adata.obs; skipping stage-group distribution."
            )
            self.results['tcc_stage_distribution'] = pd.DataFrame()

    def _get_tumor_marker_matrix(self, tumor_global_idx: np.ndarray) -> Optional[np.ndarray]:
        """Return adata.X slice for tumor cells and selected markers."""
        if self.adata.X is None:
            return None
        marker_indices = [self.adata.var_names.get_loc(m) for m in self.markers]
        if hasattr(self.adata.X, 'toarray'):
            X = self.adata.X[tumor_global_idx][:, marker_indices].toarray().astype(np.float32)
        else:
            X = np.asarray(self.adata.X[tumor_global_idx][:, marker_indices], dtype=np.float32)
        return X

    def _compute_group_distribution(self, tumor_obs: pd.DataFrame,
                                    tcc_col: str,
                                    unique_tccs: List[int],
                                    group_col: str) -> pd.DataFrame:
        """Compute fraction of tumor cells in each TCC per group."""
        if group_col not in tumor_obs.columns:
            return pd.DataFrame()

        rows = []
        for group in tumor_obs[group_col].unique():
            group_mask = tumor_obs[group_col] == group
            group_tccs = tumor_obs.loc[group_mask, tcc_col]
            n_total = group_mask.sum()
            row = {group_col: group}
            for tcc_id in unique_tccs:
                n_tcc = (group_tccs == tcc_id).sum()
                row[f'TCC_{tcc_id}'] = n_tcc / n_total if n_total > 0 else 0.0
            rows.append(row)
        return pd.DataFrame(rows).set_index(group_col)

    def _save_results(self):
        """Save all DataFrames to CSV in the output directory."""
        saved = []

        df_keys = [
            ('tcc_composition', 'tcc_composition.csv'),
            ('tcc_stage_distribution', 'tcc_stage_distribution.csv'),
            ('tcc_sample_distribution', 'tcc_sample_distribution.csv'),
            ('tcc_marker_fractions', 'tcc_marker_fractions.csv'),
        ]
        for key, fname in df_keys:
            df = self.results.get(key)
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                path = self.output_dir / fname
                df.to_csv(path)
                saved.append(fname)

        # Cell-level assignments
        assignments = self.results.get('cell_tcc_assignments')
        if assignments is not None:
            path = self.output_dir / 'cell_tcc_assignments.csv'
            assignments.to_csv(path, header=True)
            saved.append('cell_tcc_assignments.csv')

        # PCA variance explained
        variance = self.results.get('pca_variance_explained')
        if variance is not None:
            path = self.output_dir / 'pca_variance_explained.csv'
            pd.Series(variance, name='variance_explained').to_csv(path, header=True)
            saved.append('pca_variance_explained.csv')

        if saved:
            print(f"  ✓ Saved {len(saved)} output files to {self.output_dir}/")
        else:
            print("  ! No output files were saved (empty results).")
