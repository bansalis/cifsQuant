"""
Distance Permutation Testing

Determines whether observed distances between cell populations are
statistically significant or could arise by chance.

Test Types:
1. Differential Test - Are source cells closer to marker+ than marker- targets?
2. Proximity Test - Are source cells specifically close to a target population?

Statistical Approach:
- Keep all cell positions fixed
- Shuffle marker labels (differential) or cell-type labels (proximity)
- Recompute distances under permuted labels
- Compare observed to null distribution
- Benjamini-Hochberg FDR correction per sample
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from scipy.spatial import cKDTree
from scipy import stats
import warnings


class DistancePermutationTesting:
    """
    Permutation testing for cell-to-cell distance analyses.

    Tests whether observed distance differences between cell populations
    are statistically significant beyond random chance.
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        self.adata = adata
        self.config = config
        self.output_dir = Path(output_dir) / 'distance_permutation'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get analysis configuration
        self.analysis_config = self.config.get('distance_permutation_testing', {})

        # Parameters
        params = self.analysis_config.get('parameters', {})
        self.n_permutations = params.get('n_permutations', 999)
        self.min_cells = params.get('min_cells', 10)
        self.alpha = params.get('alpha', 0.05)

        # Random seed
        self.random_seed = self.config.get('advanced', {}).get('random_seed', 42)
        np.random.seed(self.random_seed)

        # Test configurations
        self.differential_tests = self.analysis_config.get('differential_tests', [])
        self.proximity_tests = self.analysis_config.get('proximity_tests', [])

        # Results
        self.results = {}

    def run(self) -> Dict:
        """Run distance permutation testing analysis."""
        print("\n" + "=" * 80)
        print("DISTANCE PERMUTATION TESTING")
        print("=" * 80)
        print(f"  Permutations: {self.n_permutations}")
        print(f"  Min cells: {self.min_cells}")

        # Run differential tests
        if self.differential_tests:
            print(f"\n  Differential tests: {len(self.differential_tests)}")
            diff_results = self._run_differential_tests()
            if diff_results:
                self.results['differential_tests'] = pd.DataFrame(diff_results)

        # Run proximity tests
        if self.proximity_tests:
            print(f"\n  Proximity tests: {len(self.proximity_tests)}")
            prox_results = self._run_proximity_tests()
            if prox_results:
                self.results['proximity_tests'] = pd.DataFrame(prox_results)

        # Apply FDR correction
        self._apply_fdr_correction()

        # Save results
        self._save_results()

        print("\n" + "=" * 80)
        print("DISTANCE PERMUTATION TESTING COMPLETE")
        print("=" * 80 + "\n")

        return self.results

    def _run_differential_tests(self) -> List[Dict]:
        """
        Run differential distance tests.

        For each test: compute mean NN distance from source to marker+ targets
        vs source to marker- targets. Shuffle marker labels to build null.
        """
        all_results = []

        for test in self.differential_tests:
            test_name = test['name']
            source_pop = test['source']
            target_base = test['target_base']
            target_marker = test['target_marker']

            source_col = f'is_{source_pop}'
            base_col = f'is_{target_base}'
            marker_col = target_marker if target_marker.startswith('is_') else f'is_{target_marker}'

            print(f"\n    [{test_name}] {source_pop} → {target_base} (by {target_marker})")

            # Validate columns
            missing = [c for c in [source_col, base_col, marker_col]
                       if c not in self.adata.obs.columns]
            if missing:
                print(f"      WARNING: Missing columns {missing}, skipping")
                continue

            for sample in self.adata.obs['sample_id'].unique():
                result = self._differential_test_sample(
                    sample, test_name, source_col, base_col, marker_col,
                    source_pop, target_base, target_marker
                )
                if result is not None:
                    all_results.append(result)

        return all_results

    def _differential_test_sample(self, sample: str, test_name: str,
                                   source_col: str, base_col: str, marker_col: str,
                                   source_pop: str, target_base: str,
                                   target_marker: str) -> Optional[Dict]:
        """Run differential test for a single sample."""
        sample_mask = self.adata.obs['sample_id'] == sample
        sample_obs = self.adata.obs.loc[sample_mask]
        sample_coords = self.adata.obsm['spatial'][sample_mask.values]

        # Get source cells
        source_mask = sample_obs[source_col].values.astype(bool)
        n_source = source_mask.sum()
        if n_source < self.min_cells:
            return None

        source_coords = sample_coords[source_mask]

        # Get target base cells (e.g., all Tumor cells)
        base_mask = sample_obs[base_col].values.astype(bool)
        if base_mask.sum() < self.min_cells:
            return None

        # Get marker status among base cells
        marker_status = sample_obs[marker_col].values.astype(bool)
        # Marker positive = base AND marker
        pos_mask = base_mask & marker_status
        neg_mask = base_mask & ~marker_status

        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()

        if n_pos < self.min_cells or n_neg < self.min_cells:
            return None

        pos_coords = sample_coords[pos_mask]
        neg_coords = sample_coords[neg_mask]

        # Observed: mean NN distance to positive vs negative
        tree_pos = cKDTree(pos_coords)
        tree_neg = cKDTree(neg_coords)
        dist_to_pos, _ = tree_pos.query(source_coords, k=1)
        dist_to_neg, _ = tree_neg.query(source_coords, k=1)

        observed_mean_pos = float(np.mean(dist_to_pos))
        observed_mean_neg = float(np.mean(dist_to_neg))
        observed_diff = observed_mean_pos - observed_mean_neg

        # Permutation: shuffle marker labels among base cells
        base_indices = np.where(base_mask)[0]
        base_coords = sample_coords[base_mask]
        n_base_pos = pos_mask[base_mask].sum()  # number of marker+ among base

        null_diffs = []
        for _ in range(self.n_permutations):
            # Shuffle which base cells are marker+
            perm_pos_idx = np.random.choice(len(base_indices), size=n_base_pos, replace=False)
            perm_neg_idx = np.setdiff1d(np.arange(len(base_indices)), perm_pos_idx)

            perm_pos_coords = base_coords[perm_pos_idx]
            perm_neg_coords = base_coords[perm_neg_idx]

            perm_tree_pos = cKDTree(perm_pos_coords)
            perm_tree_neg = cKDTree(perm_neg_coords)

            perm_dist_pos, _ = perm_tree_pos.query(source_coords, k=1)
            perm_dist_neg, _ = perm_tree_neg.query(source_coords, k=1)

            perm_diff = float(np.mean(perm_dist_pos)) - float(np.mean(perm_dist_neg))
            null_diffs.append(perm_diff)

        null_diffs = np.array(null_diffs)
        null_mean = null_diffs.mean()
        null_std = null_diffs.std()

        # Two-tailed p-value
        if null_std > 0:
            z_score = (observed_diff - null_mean) / null_std
        else:
            z_score = 0.0
        p_value = (np.abs(null_diffs - null_mean) >= np.abs(observed_diff - null_mean)).sum() / len(null_diffs)
        p_value = max(p_value, 1.0 / (self.n_permutations + 1))

        # Metadata
        timepoint = sample_obs['timepoint'].iloc[0] if 'timepoint' in sample_obs.columns else np.nan
        group = sample_obs['group'].iloc[0] if 'group' in sample_obs.columns else ''

        return {
            'sample_id': sample,
            'test_name': test_name,
            'test_type': 'differential',
            'source_population': source_pop,
            'target_base': target_base,
            'target_marker': target_marker,
            'n_source': int(n_source),
            'n_target_pos': int(n_pos),
            'n_target_neg': int(n_neg),
            'observed_mean_dist_to_pos': observed_mean_pos,
            'observed_mean_dist_to_neg': observed_mean_neg,
            'observed_diff': observed_diff,
            'null_mean': null_mean,
            'null_std': null_std,
            'z_score': z_score,
            'p_value': p_value,
            'timepoint': timepoint,
            'group': group
        }

    def _run_proximity_tests(self) -> List[Dict]:
        """
        Run proximity tests.

        For each test: compute mean NN distance from source to target.
        Shuffle cell-type labels among shuffle_pool to build null.
        """
        all_results = []

        for test in self.proximity_tests:
            test_name = test['name']
            source_pop = test['source']
            target_pop = test['target']
            shuffle_pool = test.get('shuffle_pool', [source_pop, target_pop])

            source_col = f'is_{source_pop}'
            target_col = f'is_{target_pop}'

            print(f"\n    [{test_name}] {source_pop} → {target_pop}")

            # Validate columns
            missing = [c for c in [source_col, target_col]
                       if c not in self.adata.obs.columns]
            if missing:
                print(f"      WARNING: Missing columns {missing}, skipping")
                continue

            for sample in self.adata.obs['sample_id'].unique():
                result = self._proximity_test_sample(
                    sample, test_name, source_col, target_col,
                    shuffle_pool, source_pop, target_pop
                )
                if result is not None:
                    all_results.append(result)

        return all_results

    def _proximity_test_sample(self, sample: str, test_name: str,
                                source_col: str, target_col: str,
                                shuffle_pool: List[str], source_pop: str,
                                target_pop: str) -> Optional[Dict]:
        """Run proximity test for a single sample."""
        sample_mask = self.adata.obs['sample_id'] == sample
        sample_obs = self.adata.obs.loc[sample_mask]
        sample_coords = self.adata.obsm['spatial'][sample_mask.values]

        # Get source and target cells
        source_mask = sample_obs[source_col].values.astype(bool)
        target_mask = sample_obs[target_col].values.astype(bool)

        n_source = source_mask.sum()
        n_target = target_mask.sum()

        if n_source < self.min_cells or n_target < self.min_cells:
            return None

        source_coords = sample_coords[source_mask]
        target_coords = sample_coords[target_mask]

        # Observed mean NN distance
        tree = cKDTree(target_coords)
        dists, _ = tree.query(source_coords, k=1)
        observed_mean = float(np.mean(dists))

        # Build pool mask: union of all shuffle_pool populations
        pool_mask = np.zeros(len(sample_obs), dtype=bool)
        for pop in shuffle_pool:
            pop_col = f'is_{pop}'
            if pop_col in sample_obs.columns:
                pool_mask |= sample_obs[pop_col].values.astype(bool)

        pool_indices = np.where(pool_mask)[0]
        pool_coords = sample_coords[pool_mask]
        n_pool = len(pool_indices)

        if n_pool < (n_source + n_target):
            return None

        # Permutation: randomly assign source/target labels within pool
        null_dists = []
        for _ in range(self.n_permutations):
            perm = np.random.permutation(n_pool)
            perm_source_coords = pool_coords[perm[:n_source]]
            perm_target_coords = pool_coords[perm[n_source:n_source + n_target]]

            if len(perm_target_coords) < 1:
                continue

            perm_tree = cKDTree(perm_target_coords)
            perm_dists, _ = perm_tree.query(perm_source_coords, k=1)
            null_dists.append(float(np.mean(perm_dists)))

        if len(null_dists) < self.n_permutations // 2:
            return None

        null_dists = np.array(null_dists)
        null_mean = null_dists.mean()
        null_std = null_dists.std()

        # One-tailed p-value (lower = source specifically close to target)
        if null_std > 0:
            z_score = (observed_mean - null_mean) / null_std
        else:
            z_score = 0.0
        p_value = (null_dists <= observed_mean).sum() / len(null_dists)
        p_value = max(p_value, 1.0 / (self.n_permutations + 1))

        # Metadata
        timepoint = sample_obs['timepoint'].iloc[0] if 'timepoint' in sample_obs.columns else np.nan
        group = sample_obs['group'].iloc[0] if 'group' in sample_obs.columns else ''

        return {
            'sample_id': sample,
            'test_name': test_name,
            'test_type': 'proximity',
            'source_population': source_pop,
            'target_population': target_pop,
            'n_source': int(n_source),
            'n_target': int(n_target),
            'observed_mean_dist': observed_mean,
            'null_mean': null_mean,
            'null_std': null_std,
            'z_score': z_score,
            'p_value': p_value,
            'timepoint': timepoint,
            'group': group
        }

    def _apply_fdr_correction(self):
        """Apply Benjamini-Hochberg FDR correction per test per sample."""
        for key in ['differential_tests', 'proximity_tests']:
            if key not in self.results:
                continue
            df = self.results[key]
            if df.empty:
                continue

            df['p_adjusted'] = np.nan
            df['significant'] = False

            for test_name in df['test_name'].unique():
                mask = df['test_name'] == test_name
                p_values = df.loc[mask, 'p_value'].values
                n = len(p_values)
                if n == 0:
                    continue

                # BH correction
                sorted_idx = np.argsort(p_values)
                sorted_p = p_values[sorted_idx]
                bh_adjusted = np.minimum(1.0, sorted_p * n / (np.arange(1, n + 1)))

                # Enforce monotonicity
                for i in range(n - 2, -1, -1):
                    bh_adjusted[i] = min(bh_adjusted[i], bh_adjusted[i + 1])

                # Map back
                adj_p = np.empty(n)
                adj_p[sorted_idx] = bh_adjusted

                df.loc[mask, 'p_adjusted'] = adj_p
                df.loc[mask, 'significant'] = adj_p < self.alpha

            self.results[key] = df

    def _save_results(self):
        """Save results to CSV files."""
        for key, df in self.results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                path = self.output_dir / f'{key}.csv'
                df.to_csv(path, index=False)
                print(f"  Saved {key}: {len(df)} rows → {path.name}")

        print(f"\n  Results saved to: {self.output_dir}/")
