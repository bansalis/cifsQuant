"""
LDA Neighborhood Analysis — Recurrent Cellular Neighborhoods (RCNs)

Reproduces the Recurrent Cellular Neighborhood (RCN) methodology from:
    Nirmal et al. 2022 Cancer Discovery, Fig 2D/E, 3A-D

Algorithm (per Nirmal et al.):
  1. Build a neighborhood composition vector for each cell (counts of each cell
     type within `proximity_radius` μm, using KDTree queried per-sample).
  2. Apply Latent Dirichlet Allocation on the composition matrix to extract
     latent topics representing co-occurring cell-type mixtures.
  3. K-means cluster (k=n_lda_clusters, default 30) on the LDA topic-weight
     matrix.
  4. Hierarchically merge the 30 fine clusters into n_rcns (default 10) RCNs
     via agglomerative clustering on mean composition profiles.
  5. Assign each cell its RCN label; store in adata.obs['rcn'] and
     adata.obs['rcn_cluster'] (before merging).
  6. Compute per-sample and per-stage-group RCN frequency distributions.
"""

import gc
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation


class LDANeighborhoodAnalysis:
    """
    Recurrent Cellular Neighborhoods (RCNs) via LDA + K-means + hierarchical
    agglomeration, following Nirmal et al. 2022 Cancer Discovery.

    Constructor parameters
    ----------------------
    adata : AnnData
        Annotated data matrix.  Spatial coordinates are expected in
        ``adata.obsm['spatial']`` (shape n_cells × 2).
        Cell-type assignments are expected as boolean ``is_<CellType>``
        columns in ``adata.obs``.
    config : dict
        Full pipeline configuration dict.  The section consumed by this
        class is ``config['lda_neighborhood_analysis']``.
    output_dir : Path or str
        Root output directory; results are written to a
        ``lda_neighborhood_analysis/`` subdirectory.
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        self.adata = adata
        self.config = config
        self.output_dir = Path(output_dir) / 'lda_neighborhood_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pull analysis-specific config block
        self.analysis_config: Dict = config.get('lda_neighborhood_analysis', {})

        # Parameters
        self.proximity_radius: float = self.analysis_config.get('proximity_radius', 20.0)
        self.n_lda_topics: int = self.analysis_config.get('n_lda_topics', 10)
        self.n_lda_clusters: int = self.analysis_config.get('n_lda_clusters', 30)
        self.n_rcns: int = self.analysis_config.get('n_rcns', 10)
        self.random_seed: int = self.analysis_config.get('random_seed', 42)
        self.min_cells: int = self.analysis_config.get(
            'min_cells', self.analysis_config.get('min_neighborhood_cells', 5)
        )
        self.stage_column: str = self.analysis_config.get('stage_column', 'stage_group')
        self.group_column: str = self.analysis_config.get('group_column', 'sample_id')
        self.generate_plots: bool = self.analysis_config.get('generate_plots', True)
        # Memory-saving options
        self.max_cells_per_sample: Optional[int] = self.analysis_config.get('max_cells_per_sample', None)
        self.learning_method: str = self.analysis_config.get('learning_method', 'online')

        # Cell types to include in composition vectors
        default_cell_types = [
            'Melanocytes', 'Tumor_cells', 'Keratinocytes', 'Endothelial_cells',
            'Langerhans_cells', 'Mast_cells', 'T_cells', 'Cytotoxic_T_cells',
            'Regulatory_T_cells', 'Partially_Exhausted_T_cells',
            'Terminally_Exhausted_T_cells', 'Myeloid_Lineage', 'Macrophages',
            'DC_like_cells', 'DC_like_PDL1_pos', 'Langerhans_in_Myeloid',
            'CD163_CD11C_pos_Macrophages', 'CD163_CD11C_neg_Macrophages',
            'CD163_CD11C_pos_PDL1_pos_Macrophages',
        ]
        self.cell_types: List[str] = self.analysis_config.get('cell_types', default_cell_types)

        # Validate and filter to present cell types
        self.cell_types = self._validate_cell_types(self.cell_types)

        # Results store
        self.results: Dict = {}

        np.random.seed(self.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """
        Run the full RCN analysis pipeline.

        Returns
        -------
        dict with keys:
            rcn_composition        : DataFrame (cell_types × RCNs), mean fraction
            rcn_stage_distribution : DataFrame (stage × RCN), mean fraction
            rcn_sample_distribution: DataFrame (sample × RCN), fraction
            cell_rcn_assignments   : Series, RCN label per cell
            lda_model              : fitted LatentDirichletAllocation
            kmeans_model           : fitted KMeans
        """
        print("\n" + "=" * 80)
        print("LDA NEIGHBORHOOD ANALYSIS — RECURRENT CELLULAR NEIGHBORHOODS (RCNs)")
        print("=" * 80)
        print(f"  Proximity radius : {self.proximity_radius} μm")
        print(f"  LDA topics       : {self.n_lda_topics}")
        print(f"  K-means clusters : {self.n_lda_clusters}")
        print(f"  Final RCNs       : {self.n_rcns}")
        print(f"  Cell types       : {len(self.cell_types)}")
        print(f"  Random seed      : {self.random_seed}")

        if len(self.cell_types) == 0:
            warnings.warn("LDANeighborhoodAnalysis: No valid cell types found. Aborting.")
            return self.results

        # Step 1 — build composition matrices per sample, then pool
        print("\n[1/5] Building neighborhood composition vectors...")
        composition_matrix, valid_indices = self._build_composition_matrix()

        if composition_matrix is None or len(composition_matrix) == 0:
            warnings.warn("LDANeighborhoodAnalysis: Could not build composition matrix. Aborting.")
            return self.results

        n_cells = len(composition_matrix)
        print(f"  ✓ Composition matrix: {n_cells:,} cells × {len(self.cell_types)} cell types")

        # Step 2 — LDA
        print("\n[2/5] Fitting Latent Dirichlet Allocation...")
        lda_model, lda_weights = self._fit_lda(composition_matrix)
        print(f"  ✓ LDA fitted — topic weights shape: {lda_weights.shape}")

        # Step 3 — K-means on LDA weights
        print(f"\n[3/5] K-means clustering ({self.n_lda_clusters} clusters) on LDA weights...")
        kmeans_model, cluster_labels = self._fit_kmeans(lda_weights)
        print(f"  ✓ K-means fitted — {len(np.unique(cluster_labels))} clusters assigned")

        # Step 4 — Hierarchical merge: n_lda_clusters → n_rcns
        print(f"\n[4/5] Merging {self.n_lda_clusters} clusters → {self.n_rcns} RCNs "
              f"via agglomerative clustering...")
        rcn_labels, cluster_to_rcn = self._merge_to_rcns(
            composition_matrix, cluster_labels
        )
        print(f"  ✓ RCN assignments complete — {len(np.unique(rcn_labels))} RCNs")

        # Step 5 — Write labels back to adata, compute statistics
        print("\n[5/5] Writing labels to adata.obs and computing statistics...")
        self._assign_labels_to_adata(valid_indices, cluster_labels, rcn_labels)
        self._compute_statistics()

        # Store models
        self.results['lda_model'] = lda_model
        self.results['kmeans_model'] = kmeans_model

        # Save outputs
        self._save_results()

        print("\n" + "=" * 80)
        print("LDA NEIGHBORHOOD ANALYSIS COMPLETE")
        print(f"  Results saved to: {self.output_dir}/")
        print("=" * 80 + "\n")

        return self.results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_cell_types(self, cell_types: List[str]) -> List[str]:
        """Return only cell types for which is_<CellType> exists in adata.obs."""
        valid = []
        missing = []
        for ct in cell_types:
            col = f'is_{ct}'
            if col in self.adata.obs.columns:
                valid.append(ct)
            else:
                missing.append(ct)
        if missing:
            warnings.warn(
                f"LDANeighborhoodAnalysis: {len(missing)} cell-type columns not found "
                f"in adata.obs and will be skipped: {missing}"
            )
        return valid

    def _build_composition_matrix(self):
        """
        For every cell, count how many cells of each type lie within
        `proximity_radius` μm (KDTree query, per sample so no cross-sample
        neighbors are included).

        Returns
        -------
        composition_matrix : np.ndarray, shape (n_valid_cells, n_cell_types)
            Raw counts (not normalised) of cell types in each neighborhood.
        valid_indices : np.ndarray
            Integer indices into adata corresponding to rows of the matrix.
        """
        if 'spatial' not in self.adata.obsm:
            warnings.warn(
                "LDANeighborhoodAnalysis: 'spatial' key not found in adata.obsm. "
                "Cannot compute spatial neighborhoods."
            )
            return None, None

        pheno_cols = [f'is_{ct}' for ct in self.cell_types]
        samples = self.adata.obs['sample_id'].unique()

        all_compositions = []
        all_global_indices = []

        for sample in samples:
            sample_mask = (self.adata.obs['sample_id'] == sample).values
            sample_global_idx = np.where(sample_mask)[0]

            # Optional subsampling to cap memory on very large samples
            if self.max_cells_per_sample and len(sample_global_idx) > self.max_cells_per_sample:
                rng = np.random.RandomState(self.random_seed)
                sub = rng.choice(len(sample_global_idx), self.max_cells_per_sample, replace=False)
                sub.sort()
                sample_global_idx = sample_global_idx[sub]
                sample_mask = np.zeros(len(self.adata), dtype=bool)
                sample_mask[sample_global_idx] = True

            sample_data = self.adata.obs.iloc[sample_global_idx]
            sample_coords = self.adata.obsm['spatial'][sample_mask]  # (n, 2)
            n_sample = len(sample_coords)

            if n_sample < self.min_cells:
                continue

            # Binary phenotype matrix for this sample — (n_sample, n_types)
            pheno_matrix = sample_data[pheno_cols].values.astype(np.float32)

            # KDTree radius search — C-level, fast
            tree = cKDTree(sample_coords)
            neighbor_lists = tree.query_ball_point(sample_coords, r=self.proximity_radius)

            # Build sparse adjacency using vectorised NumPy (no Python per-cell loop)
            # neighbor_counts[i] = total neighbors of cell i (including self)
            neighbor_counts = np.array([len(nbrs) for nbrs in neighbor_lists],
                                       dtype=np.int32)
            total_edges = int(neighbor_counts.sum())

            if total_edges == 0:
                continue

            # Flat row/col index arrays — vectorised, no Python loop over cells
            row_idx = np.repeat(np.arange(n_sample, dtype=np.int32), neighbor_counts)
            col_idx = np.concatenate([np.asarray(n, dtype=np.int32)
                                      for n in neighbor_lists])

            # Remove self-loops
            not_self = row_idx != col_idx
            row_idx = row_idx[not_self]
            col_idx = col_idx[not_self]

            if len(row_idx) == 0:
                continue

            # Sparse adjacency matrix (row = query cell, col = neighbor)
            adj = csr_matrix(
                (np.ones(len(row_idx), dtype=np.float32), (row_idx, col_idx)),
                shape=(n_sample, n_sample)
            )

            # Matrix multiply: compositions[i] = sum of phenotype bits over neighbors
            compositions_full = adj.dot(pheno_matrix)  # (n_sample, n_types), C-BLAS

            # Filter: keep cells whose neighborhood sums to >= min_cells phenotyped cells
            valid_local = np.asarray(compositions_full.sum(axis=1)).ravel() >= self.min_cells
            compositions = compositions_full[valid_local]
            valid_global = sample_global_idx[valid_local]

            all_compositions.append(np.asarray(compositions, dtype=np.float32))
            all_global_indices.append(valid_global)

            # Release per-sample temporaries promptly
            del adj, compositions_full, row_idx, col_idx, pheno_matrix
            gc.collect()

        if len(all_compositions) == 0:
            return None, None

        composition_matrix = np.vstack(all_compositions)
        valid_indices = np.concatenate(all_global_indices)
        return composition_matrix, valid_indices

    def _fit_lda(self, composition_matrix: np.ndarray):
        """Fit LDA on the pooled composition matrix."""
        # LDA requires non-negative integer-like counts; composition_matrix
        # already holds raw counts (float32), which is acceptable.
        # 'online' (default) is much faster than 'batch' for large datasets.
        lda = LatentDirichletAllocation(
            n_components=self.n_lda_topics,
            learning_method=self.learning_method,
            random_state=self.random_seed,
            max_iter=100,
        )
        lda_weights = lda.fit_transform(composition_matrix)  # (n_cells, n_topics)
        gc.collect()
        return lda, lda_weights

    def _fit_kmeans(self, lda_weights: np.ndarray):
        """K-means cluster on the LDA topic-weight matrix."""
        kmeans = KMeans(
            n_clusters=self.n_lda_clusters,
            random_state=self.random_seed,
            n_init=10,
            max_iter=300,
        )
        labels = kmeans.fit_predict(lda_weights)
        return kmeans, labels

    def _merge_to_rcns(self, composition_matrix: np.ndarray,
                       cluster_labels: np.ndarray):
        """
        Merge n_lda_clusters fine clusters into n_rcns meta-clusters (RCNs)
        using agglomerative clustering on mean composition profiles.

        Returns
        -------
        rcn_labels : np.ndarray, shape (n_cells,)
            RCN integer label (0-indexed) for each cell.
        cluster_to_rcn : dict
            Mapping from fine cluster index to RCN index.
        """
        # Compute mean composition profile per fine cluster
        n_clusters = self.n_lda_clusters
        n_features = composition_matrix.shape[1]
        mean_profiles = np.zeros((n_clusters, n_features), dtype=np.float64)

        for k in range(n_clusters):
            mask = cluster_labels == k
            if mask.sum() > 0:
                mean_profiles[k] = composition_matrix[mask].mean(axis=0)

        # Normalise rows to fractions (safe division)
        row_sums = mean_profiles.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        mean_profiles_norm = mean_profiles / row_sums

        # Agglomerative clustering on the mean profiles
        n_rcns_actual = min(self.n_rcns, n_clusters)
        Z = linkage(mean_profiles_norm, method='ward', metric='euclidean')
        agg_labels = fcluster(Z, t=n_rcns_actual, criterion='maxclust')
        # fcluster returns 1-indexed labels; convert to 0-indexed
        agg_labels = agg_labels - 1

        cluster_to_rcn = {k: int(agg_labels[k]) for k in range(n_clusters)}

        # Propagate to per-cell labels
        rcn_labels = np.array([cluster_to_rcn[c] for c in cluster_labels])
        return rcn_labels, cluster_to_rcn

    def _assign_labels_to_adata(self, valid_indices: np.ndarray,
                                cluster_labels: np.ndarray,
                                rcn_labels: np.ndarray):
        """Write rcn_cluster and rcn columns into adata.obs."""
        # Initialise with NaN / sentinel
        self.adata.obs['rcn_cluster'] = np.nan
        self.adata.obs['rcn'] = np.nan

        # Use .iloc-compatible integer-position assignment
        # valid_indices are positional indices into adata (from np.where)
        self.adata.obs['rcn_cluster'] = self.adata.obs['rcn_cluster'].astype(object)
        self.adata.obs['rcn'] = self.adata.obs['rcn'].astype(object)

        obs_index = self.adata.obs.index
        for pos, k_label, rcn_label in zip(valid_indices, cluster_labels, rcn_labels):
            self.adata.obs.at[obs_index[pos], 'rcn_cluster'] = int(k_label)
            self.adata.obs.at[obs_index[pos], 'rcn'] = int(rcn_label)

        # Convert to nullable integer
        self.adata.obs['rcn_cluster'] = pd.to_numeric(
            self.adata.obs['rcn_cluster'], errors='coerce'
        ).astype('Int64')
        self.adata.obs['rcn'] = pd.to_numeric(
            self.adata.obs['rcn'], errors='coerce'
        ).astype('Int64')

        n_assigned = self.adata.obs['rcn'].notna().sum()
        print(f"  ✓ Assigned RCN labels to {n_assigned:,} cells "
              f"({n_assigned / len(self.adata) * 100:.1f}% of all cells)")

    def _compute_statistics(self):
        """Compute RCN composition, per-sample and per-stage distributions."""
        pheno_cols = [f'is_{ct}' for ct in self.cell_types]
        obs = self.adata.obs

        # Filter to cells with RCN assignment
        assigned_mask = obs['rcn'].notna()
        assigned_obs = obs[assigned_mask]

        if len(assigned_obs) == 0:
            warnings.warn("LDANeighborhoodAnalysis: No cells with RCN assignments.")
            return

        rcn_values = assigned_obs['rcn'].astype(int)
        unique_rcns = sorted(rcn_values.unique())

        # --- RCN composition (mean cell-type fraction per RCN) ---
        composition_rows = {}
        for rcn_id in unique_rcns:
            rcn_mask = rcn_values == rcn_id
            rcn_cells = assigned_obs[rcn_mask]
            fracs = rcn_cells[pheno_cols].mean()
            composition_rows[f'RCN_{rcn_id}'] = fracs.values
        rcn_composition = pd.DataFrame(
            composition_rows, index=self.cell_types
        )
        self.results['rcn_composition'] = rcn_composition
        print(f"  ✓ RCN composition: {rcn_composition.shape}")

        # --- Per-sample RCN fraction ---
        sample_dists = self._compute_group_distribution(
            assigned_obs, rcn_values, unique_rcns, self.group_column
        )
        self.results['rcn_sample_distribution'] = sample_dists
        self.results['cell_rcn_assignments'] = self.adata.obs['rcn']
        print(f"  ✓ Per-sample distribution: {sample_dists.shape}")

        # --- Fine-cluster composition (k=30 clusters × cell types) for Fig E ---
        cluster_values = assigned_obs['rcn_cluster'].astype(int)
        unique_clusters = sorted(cluster_values.unique())
        fine_rows = {}
        for cid in unique_clusters:
            c_mask = cluster_values == cid
            c_cells = assigned_obs[c_mask]
            fracs = c_cells[pheno_cols].mean()
            fine_rows[f'cluster_{cid}'] = fracs.values
        fine_composition = pd.DataFrame(fine_rows, index=self.cell_types)
        self.results['fine_cluster_composition'] = fine_composition
        print(f"  ✓ Fine cluster composition: {fine_composition.shape}")

        # Cell-level cluster+RCN assignments
        self.results['cell_cluster_rcn_assignments'] = self.adata.obs[['rcn_cluster', 'rcn']]

        # --- Per-stage-group RCN fraction ---
        if self.stage_column in assigned_obs.columns:
            stage_dists = self._compute_group_distribution(
                assigned_obs, rcn_values, unique_rcns, self.stage_column
            )
            self.results['rcn_stage_distribution'] = stage_dists
            print(f"  ✓ Per-stage distribution: {stage_dists.shape}")
        else:
            warnings.warn(
                f"LDANeighborhoodAnalysis: Stage column '{self.stage_column}' not found "
                "in adata.obs; skipping stage-group distribution."
            )
            self.results['rcn_stage_distribution'] = pd.DataFrame()

    def _compute_group_distribution(self, assigned_obs: pd.DataFrame,
                                    rcn_values: pd.Series,
                                    unique_rcns: List[int],
                                    group_col: str) -> pd.DataFrame:
        """Compute fraction of cells in each RCN per group."""
        if group_col not in assigned_obs.columns:
            return pd.DataFrame()

        rows = []
        for group in assigned_obs[group_col].unique():
            group_mask = assigned_obs[group_col] == group
            group_rcns = rcn_values[group_mask]
            n_total = group_mask.sum()
            row = {group_col: group}
            for rcn_id in unique_rcns:
                n_rcn = (group_rcns == rcn_id).sum()
                row[f'RCN_{rcn_id}'] = n_rcn / n_total if n_total > 0 else 0.0
            rows.append(row)
        return pd.DataFrame(rows).set_index(group_col)

    def _save_results(self):
        """Save all DataFrames to CSV in the output directory."""
        saved = []

        df_keys = [
            ('rcn_composition', 'rcn_composition.csv'),
            ('rcn_stage_distribution', 'rcn_stage_distribution.csv'),
            ('rcn_sample_distribution', 'rcn_sample_distribution.csv'),
            ('fine_cluster_composition', 'fine_cluster_composition.csv'),
        ]
        for key, fname in df_keys:
            df = self.results.get(key)
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                path = self.output_dir / fname
                df.to_csv(path)
                saved.append(fname)

        # Cell-level assignments
        assignments = self.results.get('cell_rcn_assignments')
        if assignments is not None:
            path = self.output_dir / 'cell_rcn_assignments.csv'
            assignments.to_csv(path, header=True)
            saved.append('cell_rcn_assignments.csv')

        # Cell-level cluster + RCN assignments (both columns)
        cluster_rcn = self.results.get('cell_cluster_rcn_assignments')
        if cluster_rcn is not None:
            path = self.output_dir / 'cell_rcn_cluster_assignments.csv'
            cluster_rcn.to_csv(path)
            saved.append('cell_rcn_cluster_assignments.csv')

        if saved:
            print(f"  ✓ Saved {len(saved)} output files to {self.output_dir}/")
        else:
            print("  ! No output files were saved (empty results).")
