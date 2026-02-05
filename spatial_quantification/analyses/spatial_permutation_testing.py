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
                        sample, struct_id, timepoint, group,
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
                         test: Dict, sample: str, struct_id, timepoint, group,
                         sample_obs: pd.DataFrame = None, sample_coords: np.ndarray = None) -> Optional[Dict]:
        """Run a single permutation test on one structure."""

        test_type = test['type']

        if test_type == 'clustering':
            return self._test_clustering(struct_obs, struct_coords, test, sample, struct_id, timepoint, group)
        elif test_type == 'colocalization':
            return self._test_colocalization(struct_obs, struct_coords, test, sample, struct_id, timepoint, group)
        elif test_type == 'enrichment':
            return self._test_enrichment(struct_obs, struct_coords, test, sample, struct_id, timepoint, group,
                                        sample_obs, sample_coords)
        return None

    def _test_clustering(self, struct_obs: pd.DataFrame, struct_coords: np.ndarray,
                        test: Dict, sample: str, struct_id, timepoint, group) -> Optional[Dict]:
        """
        Test if marker+ cells are spatially clustered using Hopkins statistic.

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

        # Calculate observed Hopkins statistic
        observed = self._hopkins_statistic(struct_coords, marker_status)
        if observed is None:
            return None

        # Permutation test
        null_dist = []
        for _ in range(self.n_permutations):
            # Shuffle marker assignments (keeps same number of positive cells)
            perm_status = np.random.permutation(marker_status)
            perm_stat = self._hopkins_statistic(struct_coords, perm_status)
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
            'test_type': 'clustering',
            'test_name': test['name'],
            'marker': marker_col,
            'n_cells': n_cells,
            'n_positive': int(n_positive),
            'prevalence': prevalence,
            'observed': observed,
            'null_mean': null_mean,
            'null_std': null_std,
            'z_score': z_score,
            'p_value': p_value,
            'timepoint': timepoint,
            'group': group
        }

    def _hopkins_statistic(self, coords: np.ndarray, positive_mask: np.ndarray) -> Optional[float]:
        """
        Calculate Hopkins statistic for clustering tendency of positive cells.

        H > 0.5 indicates clustering, H = 0.5 is random, H < 0.5 is uniform.
        """
        pos_coords = coords[positive_mask]
        n_pos = len(pos_coords)

        if n_pos < 10:
            return None

        # Sample size (10% of positive cells, max 30)
        m = min(max(5, int(n_pos * 0.1)), 30)

        # Build KDTree for positive cells
        tree = cKDTree(pos_coords)

        # Sample m points from positive cells
        sample_idx = np.random.choice(n_pos, m, replace=False)
        sample_pts = pos_coords[sample_idx]

        # W: distances from sampled points to their nearest neighbor (excluding self)
        w_dist, _ = tree.query(sample_pts, k=2)
        w_dist = w_dist[:, 1]  # Second nearest (first is self)

        # U: distances from random points to nearest positive cell
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        random_pts = np.column_stack([
            np.random.uniform(x_min, x_max, m),
            np.random.uniform(y_min, y_max, m)
        ])
        u_dist, _ = tree.query(random_pts, k=1)

        # Hopkins statistic
        sum_u = np.sum(u_dist)
        sum_w = np.sum(w_dist)

        if sum_u + sum_w == 0:
            return None

        return sum_u / (sum_u + sum_w)

    def _test_colocalization(self, struct_obs: pd.DataFrame, struct_coords: np.ndarray,
                            test: Dict, sample: str, struct_id, timepoint, group) -> Optional[Dict]:
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

        return {
            'sample_id': sample,
            'structure_id': struct_id,
            'test_type': 'colocalization',
            'test_name': test['name'],
            'marker': f'{marker1_col}_vs_{marker2_col}',
            'n_cells': len(struct_obs),
            'n_marker1': int(n1),
            'n_marker2': int(n2),
            'prevalence': (n1 + n2) / (2 * len(struct_obs)),
            'observed': observed,
            'null_mean': null_mean,
            'null_std': null_std,
            'z_score': z_score,
            'p_value': p_value,
            'timepoint': timepoint,
            'group': group
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
                        test: Dict, sample: str, struct_id, timepoint, group,
                        sample_obs: pd.DataFrame, sample_coords: np.ndarray) -> Optional[Dict]:
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
            'group': group
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

        self.results['per_tumor_results'] = pd.concat(corrected, ignore_index=True)

    def _aggregate_to_sample_level(self):
        """Aggregate per-tumor results to sample level."""
        df = self.results['per_tumor_results']

        summaries = []
        for sample in df['sample_id'].unique():
            sample_df = df[df['sample_id'] == sample]
            timepoint = sample_df['timepoint'].iloc[0]
            group = sample_df['group'].iloc[0]

            for test_name in sample_df['test_name'].unique():
                test_df = sample_df[sample_df['test_name'] == test_name]
                n_tumors = len(test_df)
                n_sig = test_df['significant'].sum()

                summaries.append({
                    'sample_id': sample,
                    'group': group,
                    'timepoint': timepoint,
                    'test_name': test_name,
                    'test_type': test_df['test_type'].iloc[0],
                    'marker': test_df['marker'].iloc[0],
                    'n_tumors': n_tumors,
                    'n_significant': int(n_sig),
                    'pct_significant': 100 * n_sig / n_tumors if n_tumors > 0 else 0,
                    'mean_z_score': test_df['z_score'].mean(),
                    'median_p_value': test_df['p_value'].median()
                })

        self.results['sample_summary'] = pd.DataFrame(summaries)
        print(f"\n  Aggregated to {len(summaries)} sample-level results")

    def _run_group_comparisons(self):
        """Compare results between groups."""
        if 'sample_summary' not in self.results:
            return

        df = self.results['sample_summary']
        groups = df['group'].unique()

        if len(groups) < 2:
            return

        comparisons = []
        for test_name in df['test_name'].unique():
            test_df = df[df['test_name'] == test_name]

            for g1, g2 in combinations(groups, 2):
                g1_z = test_df[test_df['group'] == g1]['mean_z_score'].dropna()
                g2_z = test_df[test_df['group'] == g2]['mean_z_score'].dropna()

                if len(g1_z) < 2 or len(g2_z) < 2:
                    continue

                try:
                    _, p = stats.mannwhitneyu(g1_z, g2_z)
                except:
                    continue

                comparisons.append({
                    'test_name': test_name,
                    'group1': g1,
                    'group2': g2,
                    'mean_z_g1': g1_z.mean(),
                    'mean_z_g2': g2_z.mean(),
                    'p_value': p
                })

        if comparisons:
            self.results['group_comparisons'] = pd.DataFrame(comparisons)

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
