"""
Spatial Permutation Testing Analysis

Determines whether spatial patterns of tumor marker expression are
biologically meaningful or artifacts of random chance.

Test Types:
1. Single Marker Clustering - Are marker+ cells spatially clustered within tumors?
2. Two-Marker Co-localization - Do marker+ cells overlap more than chance?
3. Immune-Marker Enrichment - Are immune cells enriched near marker+ tumor cells?

Statistical Approach:
- Per-tumor Monte Carlo permutation testing
- Fixed cell coordinates with randomized marker assignments
- Benjamini-Hochberg FDR correction per sample
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from scipy.spatial import cKDTree
from scipy import stats
import warnings
from itertools import combinations
import json


class SpatialPermutationTesting:
    """
    Spatial permutation testing for marker expression patterns within tumor structures.

    IMPORTANT: This analysis runs PER TUMOR STRUCTURE.
    Question: "Within this tumor, are pERK+ cells clustered together?"
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        self.adata = adata
        self.config = config
        self.output_dir = Path(output_dir) / 'spatial_permutation'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get analysis configuration
        self.analysis_config = self.config.get('spatial_permutation', {})

        # Parameters
        params = self.analysis_config.get('parameters', {})
        self.n_permutations = params.get('n_permutations', 500)
        self.min_tumor_cells = params.get('min_tumor_cells', 20)
        self.alpha = params.get('alpha', 0.05)
        self.min_prevalence = params.get('min_prevalence', 0.05)
        self.max_prevalence = params.get('max_prevalence', 0.95)
        self.max_structures = params.get('max_structures', 500)

        # Get structure column from config
        self.structure_column = self.analysis_config.get('structure_column', 'tumor_structure_id')

        # Random seed
        self.random_seed = self.config.get('advanced', {}).get('random_seed', 42)
        np.random.seed(self.random_seed)

        # Test configurations
        self.tests = self.analysis_config.get('tests', [])

        # Results
        self.results = {}
        self.exclusion_log = []

    def run(self) -> Dict:
        """Run spatial permutation testing analysis."""
        print("\n" + "="*80)
        print("SPATIAL PERMUTATION TESTING ANALYSIS")
        print("="*80)
        print(f"  Structure column: {self.structure_column}")
        print(f"  Permutations: {self.n_permutations}")
        print(f"  Min cells per tumor: {self.min_tumor_cells}")

        # Validate structure column
        if not self._validate_structure_column():
            return self.results

        # Setup tests
        if len(self.tests) == 0:
            self._setup_default_tests()

        # Validate marker columns
        self._validate_marker_columns()

        if len(self.tests) == 0:
            print("  ERROR: No valid tests configured")
            return self.results

        print(f"\n  Tests to run: {len(self.tests)}")
        for t in self.tests:
            print(f"    - {t['name']} ({t['type']})")

        # Run per-tumor tests
        print("\n" + "-"*40)
        print("Running per-tumor permutation tests...")
        print("-"*40)

        self._run_per_tumor_tests()

        # Aggregate and save
        if 'per_tumor_results' in self.results and len(self.results['per_tumor_results']) > 0:
            self._apply_fdr_correction()
            self._aggregate_to_sample_level()
            self._run_group_comparisons()

        # Compute background immune clustering as reference distribution
        try:
            self._compute_background_immune_clustering()
        except Exception as e:
            print(f"  ⚠ Background immune clustering failed: {e}")

        self._save_results()

        print("\n" + "="*80)
        print("SPATIAL PERMUTATION TESTING COMPLETE")
        print("="*80 + "\n")

        return self.results

    def _validate_structure_column(self) -> bool:
        """Validate the structure column exists and has reasonable values."""
        print(f"\n  Validating structure column: '{self.structure_column}'")

        # List all columns for debugging
        obs_cols = list(self.adata.obs.columns)
        structure_like = [c for c in obs_cols if 'structure' in c.lower() or 'tumor' in c.lower() or 'cluster' in c.lower()]
        print(f"  Available structure-like columns: {structure_like}")

        if self.structure_column not in self.adata.obs.columns:
            print(f"\n  ERROR: Column '{self.structure_column}' not found!")
            print(f"  Please set 'structure_column' in config to one of: {structure_like}")
            return False

        # Get unique values
        col_values = self.adata.obs[self.structure_column]
        unique_values = col_values.dropna().unique()

        # Filter out -1 and string '-1'
        valid_values = [v for v in unique_values if v != -1 and str(v) != '-1' and str(v) != 'nan']

        print(f"  Total unique values: {len(unique_values)}")
        print(f"  Valid structure IDs: {len(valid_values)}")
        print(f"  Sample values: {valid_values[:10]}")

        if len(valid_values) == 0:
            print("  ERROR: No valid structure IDs found")
            return False

        if len(valid_values) > self.max_structures:
            print(f"  ERROR: Too many structures ({len(valid_values)} > {self.max_structures})")
            print("  This suggests the wrong column. Expected 10-100 tumor structures.")
            return False

        # Count cells per structure
        cells_per_struct = col_values.value_counts()
        print(f"  Cells per structure: min={cells_per_struct.min()}, max={cells_per_struct.max()}, median={cells_per_struct.median():.0f}")

        return True

    def _setup_default_tests(self):
        """Set up default tests based on available marker columns."""
        print("\n  Setting up default tests...")

        # Find marker columns
        marker_cols = [c for c in self.adata.obs.columns
                      if c.startswith('is_') and self.adata.obs[c].dtype == bool]

        # Filter to likely interesting markers
        interesting = ['GL7', 'Ki67', 'pERK', 'BCL6', 'proliferat']
        selected = []
        for col in marker_cols:
            for pattern in interesting:
                if pattern.lower() in col.lower():
                    selected.append(col)
                    break

        selected = list(set(selected))[:3]  # Max 3 markers

        self.tests = []
        for marker in selected:
            self.tests.append({
                'type': 'clustering',
                'name': f'{marker}_clustering',
                'marker': marker
            })

        print(f"  Created {len(self.tests)} default clustering tests")

    def _validate_marker_columns(self):
        """Validate that marker columns exist."""
        valid_tests = []
        for test in self.tests:
            markers_to_check = []
            if test['type'] == 'clustering':
                markers_to_check = [test.get('marker')]
            elif test['type'] == 'colocalization':
                markers_to_check = [test.get('marker1'), test.get('marker2')]
            elif test['type'] == 'enrichment':
                markers_to_check = [test.get('tumor_marker'), test.get('immune_phenotype')]

            all_exist = True
            for m in markers_to_check:
                if m and m not in self.adata.obs.columns:
                    print(f"  WARNING: Column '{m}' not found, skipping test '{test['name']}'")
                    all_exist = False

            if all_exist:
                valid_tests.append(test)

        self.tests = valid_tests

    def _run_per_tumor_tests(self):
        """Run permutation tests for each tumor structure."""
        all_results = []

        samples = self.adata.obs['sample_id'].unique()
        print(f"\n  Processing {len(samples)} samples...")

        total_structures = 0
        total_tests_run = 0

        for sample_idx, sample in enumerate(samples):
            # Get sample data
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_obs = self.adata.obs.loc[sample_mask].copy()

            # Get coordinates
            if 'spatial' in self.adata.obsm:
                sample_coords = self.adata.obsm['spatial'][sample_mask.values]
            else:
                sample_coords = sample_obs[['X_centroid', 'Y_centroid']].values

            # Get metadata
            timepoint = sample_obs['timepoint'].iloc[0] if 'timepoint' in sample_obs.columns else np.nan
            group = sample_obs['group'].iloc[0] if 'group' in sample_obs.columns else ''
            # main_group is the binary KPT/KPNT label (collapses cis/trans sub-categories)
            main_group = sample_obs['main_group'].iloc[0] if 'main_group' in sample_obs.columns else group

            # Get tumor structure IDs for this sample
            struct_ids = sample_obs[self.structure_column].unique()
            struct_ids = [s for s in struct_ids if pd.notna(s) and s != -1 and str(s) != '-1']

            n_structures = len(struct_ids)
            total_structures += n_structures

            print(f"\n  [{sample_idx+1}/{len(samples)}] {sample}: {n_structures} structures")

            if n_structures == 0:
                continue

            # Process each structure
            for struct_idx, struct_id in enumerate(struct_ids):
                # Get cells in this structure
                struct_mask = sample_obs[self.structure_column] == struct_id
                struct_obs = sample_obs.loc[struct_mask]
                struct_coords = sample_coords[struct_mask.values]

                n_cells = len(struct_obs)

                if n_cells < self.min_tumor_cells:
                    self.exclusion_log.append({
                        'sample_id': sample, 'structure_id': struct_id,
                        'reason': f'Too few cells ({n_cells})'
                    })
                    continue

                # Run each test
                for test in self.tests:
                    result = self._run_single_test(
                        struct_obs, struct_coords, test,
                        sample, struct_id, timepoint, group, main_group,
                        sample_obs, sample_coords  # For enrichment tests
                    )

                    if result is not None:
                        all_results.append(result)
                        total_tests_run += 1

                # Progress update every 10 structures
                if (struct_idx + 1) % 10 == 0:
                    print(f"    Processed {struct_idx + 1}/{n_structures} structures...")

        print(f"\n  Total: {total_structures} structures, {total_tests_run} tests completed")

        if all_results:
            self.results['per_tumor_results'] = pd.DataFrame(all_results)
        else:
            self.results['per_tumor_results'] = pd.DataFrame()

    def _run_single_test(self, struct_obs: pd.DataFrame, struct_coords: np.ndarray,
                         test: Dict, sample: str, struct_id, timepoint, group, main_group='',
                         sample_obs: pd.DataFrame = None, sample_coords: np.ndarray = None) -> Optional[Dict]:
        """Run a single permutation test on one structure."""

        test_type = test['type']

        if test_type == 'clustering':
            return self._test_clustering(struct_obs, struct_coords, test, sample, struct_id, timepoint, group, main_group)
        elif test_type == 'colocalization':
            return self._test_colocalization(struct_obs, struct_coords, test, sample, struct_id, timepoint, group, main_group)
        elif test_type == 'enrichment':
            return self._test_enrichment(struct_obs, struct_coords, test, sample, struct_id, timepoint, group, main_group,
                                        sample_obs, sample_coords)
        return None

    def _test_clustering(self, struct_obs: pd.DataFrame, struct_coords: np.ndarray,
                        test: Dict, sample: str, struct_id, timepoint, group, main_group='') -> Optional[Dict]:
        """
        Test if marker+ cells form spatial sub-clusters within a tumor structure.

        Uses two complementary statistics under random labeling:
        1. Mean NN distance among marker+ cells (lower = more sub-clustered)
        2. Cross-type NN ratio: mean(pos→pos) / mean(pos→neg), R<1 = segregation

        Permutation: Shuffle marker labels among cells, keeping positions fixed.
        """
        marker_col = test['marker']

        # Get marker status
        marker_status = struct_obs[marker_col].values.astype(bool)
        n_cells = len(marker_status)
        n_positive = marker_status.sum()
        prevalence = n_positive / n_cells

        # Check prevalence bounds
        if prevalence < self.min_prevalence or prevalence > self.max_prevalence:
            return None
        if n_positive < 10:  # Need enough positive cells
            return None
        if (n_cells - n_positive) < 5:  # Need enough negative cells
            return None

        # Calculate observed statistics
        observed_nn = self._mean_nn_distance(struct_coords, marker_status)
        observed_ratio = self._cross_type_nn_ratio(struct_coords, marker_status)
        if observed_nn is None:
            return None

        # Permutation test
        null_nn = []
        null_ratio = []
        for _ in range(self.n_permutations):
            # Shuffle marker assignments (keeps same number of positive cells)
            perm_status = np.random.permutation(marker_status)
            perm_nn = self._mean_nn_distance(struct_coords, perm_status)
            if perm_nn is not None:
                null_nn.append(perm_nn)
            perm_r = self._cross_type_nn_ratio(struct_coords, perm_status)
            if perm_r is not None:
                null_ratio.append(perm_r)

        if len(null_nn) < self.n_permutations // 2:
            return None

        # Mean NN distance statistics (lower observed = more clustered)
        null_nn = np.array(null_nn)
        null_mean_nn = null_nn.mean()
        null_std_nn = null_nn.std()
        z_score_nn = (observed_nn - null_mean_nn) / null_std_nn if null_std_nn > 0 else 0
        p_value_nn = (null_nn <= observed_nn).sum() / len(null_nn)

        # Cross-type NN ratio statistics (lower observed = more segregated)
        nn_ratio_z = np.nan
        nn_ratio_p = np.nan
        if observed_ratio is not None and len(null_ratio) > 0:
            null_ratio = np.array(null_ratio)
            null_mean_r = null_ratio.mean()
            null_std_r = null_ratio.std()
            nn_ratio_z = (observed_ratio - null_mean_r) / null_std_r if null_std_r > 0 else 0
            nn_ratio_p = (null_ratio <= observed_ratio).sum() / len(null_ratio)

        # Clark-Evans R statistic: R = observed_mean_NN / expected_mean_NN_under_CSR
        # expected_mean_NN = 0.5 / sqrt(density), where density = n_positive / approx_area
        # R < 1 = clustered, R > 1 = dispersed, R = 1 = random
        clark_evans_r = self._clark_evans_r(struct_coords, marker_status)

        # Moran's I for spatial autocorrelation of the binary marker (range -1 to +1)
        morans_i = self._morans_i(struct_coords, marker_status)

        return {
            'sample_id': sample,
            'structure_id': struct_id,
            'test_type': 'clustering',
            'test_name': test['name'],
            'marker': marker_col,
            'n_cells': n_cells,
            'n_positive': int(n_positive),
            'prevalence': prevalence,
            'observed': observed_nn,
            'observed_nn_ratio': observed_ratio if observed_ratio is not None else np.nan,
            'null_mean': null_mean_nn,
            'null_std': null_std_nn,
            'z_score': z_score_nn,
            'p_value': p_value_nn,
            'nn_ratio_z_score': nn_ratio_z,
            'nn_ratio_p_value': nn_ratio_p,
            'clark_evans_r': clark_evans_r,
            'morans_i': morans_i,
            'timepoint': timepoint,
            'group': group,
            'main_group': main_group if main_group else group
        }

    def _mean_nn_distance(self, coords: np.ndarray, positive_mask: np.ndarray) -> Optional[float]:
        """
        Mean nearest-neighbor distance among marker+ cells.

        Lower values indicate that marker+ cells are closer together
        (sub-clustered) than a random subset of the same size.
        """
        pos_coords = coords[positive_mask]
        n_pos = len(pos_coords)

        if n_pos < 10:
            return None

        tree = cKDTree(pos_coords)
        dists, _ = tree.query(pos_coords, k=2)
        nn_dists = dists[:, 1]  # second nearest (first is self)

        return float(np.mean(nn_dists))

    def _cross_type_nn_ratio(self, coords: np.ndarray, positive_mask: np.ndarray) -> Optional[float]:
        """
        Cross-type nearest-neighbor ratio.

        R = mean(NN dist: marker+ to marker+) / mean(NN dist: marker+ to marker-)

        R < 1: marker+ cells are closer to each other than to marker- (segregation)
        R = 1: random labeling (no spatial preference)
        R > 1: marker+ cells are closer to marker- than to each other (intermixing)
        """
        pos_coords = coords[positive_mask]
        neg_coords = coords[~positive_mask]

        if len(pos_coords) < 10 or len(neg_coords) < 5:
            return None

        # NN distance: marker+ to marker+ (excluding self)
        tree_pos = cKDTree(pos_coords)
        dists_pp, _ = tree_pos.query(pos_coords, k=2)
        mean_pp = np.mean(dists_pp[:, 1])

        # NN distance: marker+ to nearest marker-
        tree_neg = cKDTree(neg_coords)
        dists_pn, _ = tree_neg.query(pos_coords, k=1)
        mean_pn = np.mean(dists_pn)

        if mean_pn == 0:
            return None

        return float(mean_pp / mean_pn)

    def _clark_evans_r(self, coords: np.ndarray, positive_mask: np.ndarray) -> float:
        """
        Clark-Evans R statistic for the positive cells.

        R = observed_mean_NN / expected_mean_NN_under_CSR
        where expected_mean_NN = 0.5 / sqrt(density).

        R < 1: clustered (cells closer together than random)
        R = 1: complete spatial randomness
        R > 1: dispersed (cells more evenly spaced than random)

        Uses the convex hull area as the study area estimate.
        """
        pos_coords = coords[positive_mask]
        n_pos = len(pos_coords)

        if n_pos < 10:
            return np.nan

        tree = cKDTree(pos_coords)
        dists, _ = tree.query(pos_coords, k=2)
        observed_mean_nn = float(np.mean(dists[:, 1]))

        # Estimate study area from convex hull
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pos_coords)
            area = hull.volume  # In 2D, ConvexHull.volume is the area
        except Exception:
            # Fallback: bounding box area
            area = (pos_coords[:, 0].max() - pos_coords[:, 0].min()) * \
                   (pos_coords[:, 1].max() - pos_coords[:, 1].min())

        if area <= 0:
            return np.nan

        density = n_pos / area
        expected_mean_nn = 0.5 / np.sqrt(density)

        if expected_mean_nn <= 0:
            return np.nan

        return float(observed_mean_nn / expected_mean_nn)

    def _morans_i(self, coords: np.ndarray, positive_mask: np.ndarray,
                  k_neighbors: int = 8) -> float:
        """
        Moran's I spatial autocorrelation for the binary marker.

        Range: -1 (dispersed) to +1 (clustered), 0 = random.
        Computed using inverse-distance weights from k-nearest neighbors.
        """
        n = len(coords)
        if n < 20:
            return np.nan

        # Binary indicator as float
        x = positive_mask.astype(float)
        x_mean = np.mean(x)
        x_dev = x - x_mean

        # Build spatial weights: k-NN inverse distance
        tree = cKDTree(coords)
        k = min(k_neighbors, n - 1)
        dists, indices = tree.query(coords, k=k + 1)  # +1 because self is included

        # Exclude self (first neighbor at distance 0)
        dists = dists[:, 1:]
        indices = indices[:, 1:]

        # Inverse distance weights
        with np.errstate(divide='ignore'):
            weights = np.where(dists > 0, 1.0 / dists, 0.0)

        # Row-normalise
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        weights = weights / row_sums

        # W_total = sum of all weights (after row normalisation = n)
        W = float(n)

        # Numerator: sum_i sum_j w_ij * (x_i - x_mean) * (x_j - x_mean)
        numerator = 0.0
        for i in range(n):
            for j_idx, j in enumerate(indices[i]):
                numerator += weights[i, j_idx] * x_dev[i] * x_dev[j]

        denominator = np.sum(x_dev ** 2)
        if denominator == 0:
            return np.nan

        return float((n / W) * (numerator / denominator))

    def _test_colocalization(self, struct_obs: pd.DataFrame, struct_coords: np.ndarray,
                            test: Dict, sample: str, struct_id, timepoint, group, main_group='') -> Optional[Dict]:
        """
        Test if two markers spatially co-localize using cross-K function.

        Permutation: Independently shuffle both marker labels.
        """
        marker1_col = test['marker1']
        marker2_col = test['marker2']
        radius = test.get('radius', 30)

        mask1 = struct_obs[marker1_col].values.astype(bool)
        mask2 = struct_obs[marker2_col].values.astype(bool)

        n1 = mask1.sum()
        n2 = mask2.sum()

        if n1 < 5 or n2 < 5:
            return None

        # Observed cross-K
        observed = self._cross_k(struct_coords, mask1, mask2, radius)
        if observed is None:
            return None

        # Permutation test
        null_dist = []
        for _ in range(self.n_permutations):
            perm1 = np.random.permutation(mask1)
            perm2 = np.random.permutation(mask2)
            perm_stat = self._cross_k(struct_coords, perm1, perm2, radius)
            if perm_stat is not None:
                null_dist.append(perm_stat)

        if len(null_dist) < self.n_permutations // 2:
            return None

        null_dist = np.array(null_dist)
        null_mean = null_dist.mean()
        null_std = null_dist.std()

        z_score = (observed - null_mean) / null_std if null_std > 0 else 0

        # Two-tailed p-value
        p_lower = (null_dist <= observed).sum() / len(null_dist)
        p_upper = (null_dist >= observed).sum() / len(null_dist)
        p_value = 2 * min(p_lower, p_upper)

        n_total = len(struct_obs)
        n_both = int((mask1 & mask2).sum())
        observed_pct_both = n_both / n_total * 100 if n_total > 0 else 0.0
        expected_pct_both = (n1 / n_total) * (n2 / n_total) * 100 if n_total > 0 else 0.0

        return {
            'sample_id': sample,
            'structure_id': struct_id,
            'test_type': 'colocalization',
            'test_name': test['name'],
            'marker': f'{marker1_col}_vs_{marker2_col}',
            'n_cells': n_total,
            'n_marker1': int(n1),
            'n_marker2': int(n2),
            'n_both': n_both,
            'observed_pct_both': observed_pct_both,
            'expected_pct_both': expected_pct_both,
            'prevalence': (n1 + n2) / (2 * n_total),
            'observed': observed,
            'null_mean': null_mean,
            'null_std': null_std,
            'z_score': z_score,
            'p_value': p_value,
            'timepoint': timepoint,
            'group': group,
            'main_group': main_group if main_group else group
        }

    def _cross_k(self, coords: np.ndarray, mask1: np.ndarray, mask2: np.ndarray, radius: float) -> Optional[float]:
        """Calculate cross-K function at given radius."""
        coords1 = coords[mask1]
        coords2 = coords[mask2]

        if len(coords1) == 0 or len(coords2) == 0:
            return None

        tree2 = cKDTree(coords2)
        counts = tree2.query_ball_point(coords1, r=radius, return_length=True)

        return np.mean(counts)

    def _test_enrichment(self, struct_obs: pd.DataFrame, struct_coords: np.ndarray,
                        test: Dict, sample: str, struct_id, timepoint, group, main_group='',
                        sample_obs: pd.DataFrame = None, sample_coords: np.ndarray = None) -> Optional[Dict]:
        """
        Test if immune cells are enriched near marker+ tumor cells.

        Permutation: Shuffle marker status among tumor cells.
        """
        tumor_marker = test['tumor_marker']
        immune_col = test['immune_phenotype']
        radius = test.get('radius', 50)

        # Tumor cells and their marker status
        marker_status = struct_obs[tumor_marker].values.astype(bool)
        n_positive = marker_status.sum()

        if n_positive < 5 or n_positive == len(marker_status):
            return None

        # Immune cells from entire sample
        immune_mask = sample_obs[immune_col].values.astype(bool)
        immune_coords = sample_coords[immune_mask]

        if len(immune_coords) < 5:
            return None

        # Observed enrichment
        observed = self._enrichment_score(struct_coords, marker_status, immune_coords, radius)
        if observed is None:
            return None

        # Permutation test
        null_dist = []
        for _ in range(self.n_permutations):
            perm_status = np.random.permutation(marker_status)
            perm_stat = self._enrichment_score(struct_coords, perm_status, immune_coords, radius)
            if perm_stat is not None:
                null_dist.append(perm_stat)

        if len(null_dist) < self.n_permutations // 2:
            return None

        null_dist = np.array(null_dist)
        null_mean = null_dist.mean()
        null_std = null_dist.std()

        z_score = (observed - null_mean) / null_std if null_std > 0 else 0
        p_value = (null_dist >= observed).sum() / len(null_dist)

        return {
            'sample_id': sample,
            'structure_id': struct_id,
            'test_type': 'enrichment',
            'test_name': test['name'],
            'marker': f'{immune_col}_near_{tumor_marker}',
            'n_cells': len(struct_obs),
            'n_positive': int(n_positive),
            'n_immune': len(immune_coords),
            'prevalence': n_positive / len(struct_obs),
            'observed': observed,
            'null_mean': null_mean,
            'null_std': null_std,
            'z_score': z_score,
            'p_value': p_value,
            'timepoint': timepoint,
            'group': group,
            'main_group': main_group if main_group else group
        }

    def _enrichment_score(self, tumor_coords: np.ndarray, marker_positive: np.ndarray,
                         immune_coords: np.ndarray, radius: float) -> Optional[float]:
        """Mean immune count within radius of marker+ tumor cells."""
        pos_coords = tumor_coords[marker_positive]

        if len(pos_coords) == 0:
            return None

        tree = cKDTree(immune_coords)
        counts = tree.query_ball_point(pos_coords, r=radius, return_length=True)

        return np.mean(counts)

    def _apply_fdr_correction(self):
        """Apply Benjamini-Hochberg FDR correction per sample."""
        if 'per_tumor_results' not in self.results:
            return

        df = self.results['per_tumor_results']

        corrected = []
        for sample in df['sample_id'].unique():
            sample_df = df[df['sample_id'] == sample].copy()
            p_values = sample_df['p_value'].values
            n = len(p_values)

            if n > 1:
                # BH correction
                sorted_idx = np.argsort(p_values)
                ranks = np.empty(n)
                ranks[sorted_idx] = np.arange(1, n + 1)
                adjusted = np.minimum(1, p_values * n / ranks)

                # Ensure monotonicity
                sorted_adj = adjusted[sorted_idx]
                for i in range(n - 2, -1, -1):
                    sorted_adj[i] = min(sorted_adj[i], sorted_adj[i + 1])
                adjusted[sorted_idx] = sorted_adj

                sample_df['p_adjusted'] = adjusted
            else:
                sample_df['p_adjusted'] = sample_df['p_value']

            sample_df['significant'] = sample_df['p_adjusted'] < self.alpha
            corrected.append(sample_df)

        df_out = pd.concat(corrected, ignore_index=True)

        # Add prevalence_group (tertile of % positive within each test)
        if 'prevalence' in df_out.columns:
            for test_name in df_out['test_name'].unique():
                idx = df_out['test_name'] == test_name
                prev = df_out.loc[idx, 'prevalence']
                if prev.nunique() >= 3:
                    labels = ['low', 'medium', 'high']
                    df_out.loc[idx, 'prevalence_group'] = pd.qcut(
                        prev, q=3, labels=labels, duplicates='drop'
                    ).astype(str)
                else:
                    df_out.loc[idx, 'prevalence_group'] = 'medium'

        self.results['per_tumor_results'] = df_out

    def _aggregate_to_sample_level(self):
        """Aggregate per-tumor results to sample level."""
        df = self.results['per_tumor_results']

        summaries = []
        for sample in df['sample_id'].unique():
            sample_df = df[df['sample_id'] == sample]
            timepoint = sample_df['timepoint'].iloc[0]
            group = sample_df['group'].iloc[0]
            main_group = sample_df['main_group'].iloc[0] if 'main_group' in sample_df.columns else group

            for test_name in sample_df['test_name'].unique():
                test_df = sample_df[sample_df['test_name'] == test_name]
                n_tumors = len(test_df)
                n_sig = test_df['significant'].sum()

                row = {
                    'sample_id': sample,
                    'group': group,
                    'main_group': main_group,
                    'timepoint': timepoint,
                    'test_name': test_name,
                    'test_type': test_df['test_type'].iloc[0],
                    'marker': test_df['marker'].iloc[0],
                    'n_tumors': n_tumors,
                    'n_significant': int(n_sig),
                    'pct_significant': 100 * n_sig / n_tumors if n_tumors > 0 else 0,
                    'mean_z_score': test_df['z_score'].mean(),
                    'median_p_value': test_df['p_value'].median()
                }
                # Aggregate additional metrics if present
                for extra_col in ('clark_evans_r', 'morans_i', 'nn_ratio_z_score'):
                    if extra_col in test_df.columns:
                        row[f'mean_{extra_col}'] = test_df[extra_col].mean()
                summaries.append(row)

        self.results['sample_summary'] = pd.DataFrame(summaries)
        print(f"\n  Aggregated to {len(summaries)} sample-level results")

    def _run_group_comparisons(self):
        """Compare results between groups using both group and main_group columns."""
        if 'sample_summary' not in self.results:
            return

        df = self.results['sample_summary']

        # Metrics to compare
        metric_cols = ['mean_z_score']
        for extra in ('mean_clark_evans_r', 'mean_morans_i', 'mean_nn_ratio_z_score'):
            if extra in df.columns:
                metric_cols.append(extra)

        comparisons = []

        # Run comparisons on main_group (binary KPT/KPNT) and group (4-category)
        for group_col in ('main_group', 'group'):
            if group_col not in df.columns:
                continue
            groups = [g for g in df[group_col].dropna().unique() if str(g) != '']
            if len(groups) < 2:
                continue

            for test_name in df['test_name'].unique():
                test_df = df[df['test_name'] == test_name]

                for g1, g2 in combinations(groups, 2):
                    row = {
                        'test_name': test_name,
                        'comparison_level': group_col,
                        'group1': g1,
                        'group2': g2
                    }
                    for metric in metric_cols:
                        if metric not in test_df.columns:
                            continue
                        g1_vals = test_df[test_df[group_col] == g1][metric].dropna()
                        g2_vals = test_df[test_df[group_col] == g2][metric].dropna()
                        if len(g1_vals) < 2 or len(g2_vals) < 2:
                            continue
                        try:
                            _, p = stats.mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')
                        except Exception:
                            continue
                        row[f'{metric}_g1'] = g1_vals.mean()
                        row[f'{metric}_g2'] = g2_vals.mean()
                        row[f'{metric}_p'] = p
                    if len(row) > 4:  # at least one metric was compared
                        comparisons.append(row)

        if comparisons:
            self.results['group_comparisons'] = pd.DataFrame(comparisons)

    def _compute_background_immune_clustering(self):
        """
        Compute clustering statistics for immune cells in background lung tissue
        (cells where tumor_region_id is -1 or NaN, i.e., outside any tumor boundary).

        Samples K random 500-µm-radius windows centred on background immune cells.
        Runs the same mean-NN and Clark-Evans R statistics used in the main clustering tests.
        Results are saved as a reference distribution to compare against intra-tumoral clustering.

        Output: background_immune_clustering.csv
        """
        background_config = self.analysis_config.get('background_immune', {})
        n_windows = background_config.get('n_windows', 20)
        window_radius = background_config.get('window_radius', 500)

        cd45_col = 'is_CD45_positive'
        if cd45_col not in self.adata.obs.columns:
            # Try to find any immune marker
            immune_candidates = [c for c in self.adata.obs.columns
                                  if c.startswith('is_') and 'CD45' in c]
            if not immune_candidates:
                print("  ⚠ No CD45 column found; skipping background immune clustering")
                return
            cd45_col = immune_candidates[0]

        print(f"\n  Computing background immune clustering (using '{cd45_col}')...")

        background_rows = []
        rng = np.random.default_rng(self.random_seed)

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_obs = self.adata.obs.loc[sample_mask]

            if 'spatial' in self.adata.obsm:
                sample_coords = self.adata.obsm['spatial'][sample_mask.values]
            else:
                sample_coords = sample_obs[['X_centroid', 'Y_centroid']].values

            # Background = immune cells outside any tumor region
            if self.structure_column in sample_obs.columns:
                in_background = (sample_obs[self.structure_column].isna() |
                                 (sample_obs[self.structure_column] == -1) |
                                 (sample_obs[self.structure_column].astype(str) == '-1')).values
            else:
                in_background = np.ones(len(sample_obs), dtype=bool)

            immune_mask = sample_obs[cd45_col].values.astype(bool) if cd45_col in sample_obs.columns else np.zeros(len(sample_obs), dtype=bool)
            bg_immune_mask = immune_mask & in_background

            bg_immune_coords = sample_coords[bg_immune_mask]

            if len(bg_immune_coords) < 20:
                continue

            timepoint = sample_obs['timepoint'].iloc[0] if 'timepoint' in sample_obs.columns else np.nan
            group = sample_obs['group'].iloc[0] if 'group' in sample_obs.columns else ''

            # Sample K random centres from background immune cells
            n_centres = min(n_windows, len(bg_immune_coords))
            centre_indices = rng.choice(len(bg_immune_coords), size=n_centres, replace=False)
            tree_all = cKDTree(bg_immune_coords)

            for centre_idx in centre_indices:
                centre = bg_immune_coords[centre_idx]

                # Find all background immune cells within radius
                within_idx = tree_all.query_ball_point(centre, r=window_radius)
                window_coords = bg_immune_coords[within_idx]

                n_w = len(window_coords)
                if n_w < 10:
                    continue

                # Mean NN distance among cells in window
                tree_w = cKDTree(window_coords)
                dists, _ = tree_w.query(window_coords, k=2)
                mean_nn = float(np.mean(dists[:, 1]))

                # Clark-Evans R for window
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(window_coords)
                    area = hull.volume
                except Exception:
                    area = float(window_radius ** 2 * np.pi)

                if area > 0:
                    density = n_w / area
                    expected_nn = 0.5 / np.sqrt(density) if density > 0 else np.nan
                    ce_r = mean_nn / expected_nn if expected_nn and expected_nn > 0 else np.nan
                else:
                    ce_r = np.nan

                background_rows.append({
                    'sample_id': sample,
                    'window_centre_x': centre[0],
                    'window_centre_y': centre[1],
                    'window_radius_um': window_radius,
                    'n_immune_cells': n_w,
                    'mean_nn_distance': mean_nn,
                    'clark_evans_r': ce_r,
                    'timepoint': timepoint,
                    'group': group
                })

        if background_rows:
            bg_df = pd.DataFrame(background_rows)
            self.results['background_immune_clustering'] = bg_df
            print(f"    ✓ Computed background immune clustering: {len(bg_df)} windows across {self.adata.obs['sample_id'].nunique()} samples")
        else:
            print("    ⚠ No background immune clustering windows computed")

    def _save_results(self):
        """Save all results."""
        for name, data in self.results.items():
            if isinstance(data, pd.DataFrame) and len(data) > 0:
                path = self.output_dir / f'{name}.csv'
                data.to_csv(path, index=False)
                print(f"  Saved {name}.csv ({len(data)} rows)")

        # Save config
        config = {
            'structure_column': self.structure_column,
            'n_permutations': self.n_permutations,
            'min_tumor_cells': self.min_tumor_cells,
            'alpha': self.alpha,
            'tests': self.tests
        }
        with open(self.output_dir / 'config_used.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Save exclusions
        if self.exclusion_log:
            pd.DataFrame(self.exclusion_log).to_csv(
                self.output_dir / 'exclusion_log.csv', index=False
            )


def run_spatial_permutation_testing(adata, config: Dict, output_dir: Path) -> Dict:
    """Convenience function to run spatial permutation testing."""
    analysis = SpatialPermutationTesting(adata, config, output_dir)
    return analysis.run()
