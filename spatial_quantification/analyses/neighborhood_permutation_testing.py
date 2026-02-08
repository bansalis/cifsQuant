"""
Neighborhood Enrichment Permutation Testing

Tests whether cell-type pairs are spatial neighbors more or less often
than expected by chance, following the approach from:
- squidpy (gr.nhood_enrichment)
- imcRtools (testInteractions)
- histoCAT

Statistical Approach:
- Build k-NN spatial graph (positions fixed)
- Count edges between each cell-type pair
- Permute cell-type labels N times (graph unchanged)
- Z-score: (observed - mean_perm) / std_perm
- Positive z = enrichment (attraction), Negative z = avoidance (depletion)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.spatial import cKDTree
from scipy import stats
import warnings


class NeighborhoodPermutationTesting:
    """
    Permutation testing for neighborhood enrichment between cell types.

    Tests whether cell-type pairs co-occur as spatial neighbors more
    (enrichment) or less (depletion) than random chance.
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        self.adata = adata
        self.config = config
        self.output_dir = Path(output_dir) / 'neighborhood_permutation'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get analysis configuration
        self.analysis_config = self.config.get('neighborhood_permutation_testing', {})

        # Parameters
        params = self.analysis_config.get('parameters', {})
        self.n_permutations = params.get('n_permutations', 1000)
        self.k_neighbors = params.get('k_neighbors', 30)
        self.min_cells_per_type = params.get('min_cells_per_type', 10)
        self.alpha = params.get('alpha', 0.05)

        # Cell types to include
        self.cell_types = self.analysis_config.get('cell_types', [])

        # Differential enrichment tests
        self.diff_enrichment_tests = self.analysis_config.get('differential_enrichment_tests', [])

        # Random seed
        self.random_seed = self.config.get('advanced', {}).get('random_seed', 42)
        np.random.seed(self.random_seed)

        # Results
        self.results = {}

    def run(self) -> Dict:
        """Run neighborhood enrichment permutation testing."""
        print("\n" + "=" * 80)
        print("NEIGHBORHOOD ENRICHMENT PERMUTATION TESTING")
        print("=" * 80)
        print(f"  k-neighbors: {self.k_neighbors}")
        print(f"  Permutations: {self.n_permutations}")

        # Resolve cell types
        if not self.cell_types:
            self.cell_types = self._detect_cell_types()
        print(f"  Cell types: {len(self.cell_types)}")

        if len(self.cell_types) < 2:
            print("  ERROR: Need at least 2 cell types for enrichment testing")
            return self.results

        # Run per-sample
        pairwise_results = []
        samples = self.adata.obs['sample_id'].unique()
        print(f"\n  Processing {len(samples)} samples...")

        for sample_idx, sample in enumerate(samples):
            print(f"\n  [{sample_idx + 1}/{len(samples)}] {sample}")
            sample_results = self._run_sample(sample)
            pairwise_results.extend(sample_results)

        if pairwise_results:
            self.results['pairwise_enrichment'] = pd.DataFrame(pairwise_results)
            self._build_aggregate_matrix()
        else:
            self.results['pairwise_enrichment'] = pd.DataFrame()

        # Run differential enrichment tests
        if self.diff_enrichment_tests:
            print(f"\n  Differential enrichment tests: {len(self.diff_enrichment_tests)}")
            diff_results = self._run_differential_enrichment_tests()
            if diff_results:
                self.results['differential_enrichment'] = pd.DataFrame(diff_results)

        # Save results
        self._save_results()

        print("\n" + "=" * 80)
        print("NEIGHBORHOOD ENRICHMENT PERMUTATION TESTING COMPLETE")
        print("=" * 80 + "\n")

        return self.results

    def _detect_cell_types(self) -> List[str]:
        """Auto-detect cell types from is_ columns."""
        is_cols = [c for c in self.adata.obs.columns
                   if c.startswith('is_') and self.adata.obs[c].dtype == bool]

        # Filter to those with reasonable prevalence
        valid = []
        for col in is_cols:
            frac = self.adata.obs[col].mean()
            if 0.001 < frac < 0.5:
                name = col[3:]  # strip 'is_'
                valid.append(name)

        # Limit to manageable number
        if len(valid) > 20:
            # Prioritize common populations
            counts = [(name, self.adata.obs[f'is_{name}'].sum()) for name in valid]
            counts.sort(key=lambda x: x[1], reverse=True)
            valid = [name for name, _ in counts[:20]]

        print(f"  Auto-detected {len(valid)} cell types")
        return valid

    def _run_sample(self, sample: str) -> List[Dict]:
        """Run enrichment testing for a single sample."""
        sample_mask = self.adata.obs['sample_id'] == sample
        sample_obs = self.adata.obs.loc[sample_mask].copy()
        sample_coords = self.adata.obsm['spatial'][sample_mask.values]

        n_cells = len(sample_obs)
        if n_cells < self.k_neighbors + 1:
            print(f"    Skipping: too few cells ({n_cells})")
            return []

        # Metadata
        timepoint = sample_obs['timepoint'].iloc[0] if 'timepoint' in sample_obs.columns else np.nan
        group = sample_obs['group'].iloc[0] if 'group' in sample_obs.columns else ''

        # Assign integer cell-type labels
        # Each cell gets the index of its most specific type
        # Cells not matching any type get label -1
        labels = np.full(n_cells, -1, dtype=int)
        type_names = []
        type_counts = []

        for i, ct in enumerate(self.cell_types):
            col = f'is_{ct}'
            if col not in sample_obs.columns:
                continue
            mask = sample_obs[col].values.astype(bool)
            count = mask.sum()
            if count < self.min_cells_per_type:
                continue
            # Later types overwrite earlier (more specific overrides general)
            labels[mask] = len(type_names)
            type_names.append(ct)
            type_counts.append(count)

        n_types = len(type_names)
        if n_types < 2:
            print(f"    Skipping: fewer than 2 valid cell types")
            return []

        print(f"    {n_types} cell types, {n_cells} cells")

        # Build k-NN graph
        tree = cKDTree(sample_coords)
        _, neighbor_indices = tree.query(sample_coords, k=self.k_neighbors + 1)
        # Remove self (first column)
        neighbor_indices = neighbor_indices[:, 1:]

        # Count observed interactions
        observed_counts = self._count_interactions(labels, neighbor_indices, n_types)

        # Permutation testing
        perm_counts = np.zeros((self.n_permutations, n_types, n_types))
        for p in range(self.n_permutations):
            perm_labels = np.random.permutation(labels)
            perm_counts[p] = self._count_interactions(perm_labels, neighbor_indices, n_types)

        # Compute z-scores and p-values
        null_mean = perm_counts.mean(axis=0)
        null_std = perm_counts.std(axis=0)

        results = []
        for i in range(n_types):
            for j in range(n_types):
                obs = observed_counts[i, j]
                nm = null_mean[i, j]
                ns = null_std[i, j]

                if ns > 0:
                    z = (obs - nm) / ns
                else:
                    z = 0.0

                # Two-tailed p-value from permutation distribution
                perm_vals = perm_counts[:, i, j]
                p_val = (np.abs(perm_vals - nm) >= np.abs(obs - nm)).sum() / self.n_permutations
                p_val = max(p_val, 1.0 / (self.n_permutations + 1))

                results.append({
                    'sample_id': sample,
                    'cell_type_a': type_names[i],
                    'cell_type_b': type_names[j],
                    'n_type_a': type_counts[i],
                    'n_type_b': type_counts[j],
                    'observed_count': float(obs),
                    'null_mean': float(nm),
                    'null_std': float(ns),
                    'z_score': float(z),
                    'p_value': float(p_val),
                    'timepoint': timepoint,
                    'group': group
                })

        return results

    def _count_interactions(self, labels: np.ndarray, neighbor_indices: np.ndarray,
                            n_types: int) -> np.ndarray:
        """
        Count interactions between cell types in the k-NN graph.

        Vectorized: for each cell with label i, count how many of its
        k neighbors have label j, for all (i, j) pairs.
        """
        counts = np.zeros((n_types, n_types), dtype=float)
        neighbor_labels = labels[neighbor_indices]  # (n_cells, k)

        for i in range(n_types):
            # Cells of type i
            type_mask = labels == i
            if not type_mask.any():
                continue
            # Their neighbors' labels
            nb_labels = neighbor_labels[type_mask]  # (n_type_i, k)
            for j in range(n_types):
                counts[i, j] = (nb_labels == j).sum()

        return counts

    def _run_differential_enrichment_tests(self) -> List[Dict]:
        """
        Differential enrichment: is immune pop X more enriched around
        marker+ vs marker- tumor cells?

        For each test: count immune neighbors of marker+ tumor cells vs
        marker- tumor cells. Shuffle marker labels among tumor cells to build null.
        """
        all_results = []

        for test in self.diff_enrichment_tests:
            test_name = test['name']
            immune_pop = test['immune_population']
            tumor_base = test['tumor_base']
            tumor_marker = test['tumor_marker']

            immune_col = f'is_{immune_pop}'
            base_col = f'is_{tumor_base}'
            marker_col = tumor_marker if tumor_marker.startswith('is_') else f'is_{tumor_marker}'

            print(f"\n    [{test_name}] {immune_pop} near {tumor_marker}")

            missing = [c for c in [immune_col, base_col, marker_col]
                       if c not in self.adata.obs.columns]
            if missing:
                print(f"      WARNING: Missing columns {missing}, skipping")
                continue

            for sample in self.adata.obs['sample_id'].unique():
                result = self._diff_enrichment_sample(
                    sample, test_name, immune_col, base_col, marker_col,
                    immune_pop, tumor_base, tumor_marker
                )
                if result is not None:
                    all_results.append(result)

        return all_results

    def _diff_enrichment_sample(self, sample: str, test_name: str,
                                 immune_col: str, base_col: str, marker_col: str,
                                 immune_pop: str, tumor_base: str,
                                 tumor_marker: str) -> Optional[Dict]:
        """Differential enrichment test for a single sample."""
        sample_mask = self.adata.obs['sample_id'] == sample
        sample_obs = self.adata.obs.loc[sample_mask]
        sample_coords = self.adata.obsm['spatial'][sample_mask.values]

        n_cells = len(sample_obs)
        if n_cells < self.k_neighbors + 1:
            return None

        # Get populations
        immune_mask = sample_obs[immune_col].values.astype(bool)
        base_mask = sample_obs[base_col].values.astype(bool)
        marker_status = sample_obs[marker_col].values.astype(bool)

        pos_mask = base_mask & marker_status
        neg_mask = base_mask & ~marker_status

        n_immune = immune_mask.sum()
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()

        if n_immune < self.min_cells_per_type or n_pos < self.min_cells_per_type or n_neg < self.min_cells_per_type:
            return None

        # Build k-NN graph
        tree = cKDTree(sample_coords)
        _, neighbor_indices = tree.query(sample_coords, k=self.k_neighbors + 1)
        neighbor_indices = neighbor_indices[:, 1:]

        # Count: how many immune neighbors does each tumor+ / tumor- cell have?
        neighbor_is_immune = immune_mask[neighbor_indices]  # (n_cells, k) bool

        obs_count_pos = neighbor_is_immune[pos_mask].sum() / n_pos  # mean immune neighbors per marker+ cell
        obs_count_neg = neighbor_is_immune[neg_mask].sum() / n_neg
        observed_diff = float(obs_count_pos - obs_count_neg)

        # Permutation: shuffle marker labels among base (tumor) cells
        base_indices = np.where(base_mask)[0]
        n_base_pos = pos_mask[base_mask].sum()

        null_diffs = []
        for _ in range(self.n_permutations):
            perm = np.random.permutation(len(base_indices))
            perm_pos_idx = base_indices[perm[:n_base_pos]]
            perm_neg_idx = base_indices[perm[n_base_pos:]]

            perm_count_pos = neighbor_is_immune[perm_pos_idx].sum() / len(perm_pos_idx)
            perm_count_neg = neighbor_is_immune[perm_neg_idx].sum() / len(perm_neg_idx)
            null_diffs.append(float(perm_count_pos - perm_count_neg))

        null_diffs = np.array(null_diffs)
        null_mean = null_diffs.mean()
        null_std = null_diffs.std()

        z_score = (observed_diff - null_mean) / null_std if null_std > 0 else 0.0
        p_value = (np.abs(null_diffs - null_mean) >= np.abs(observed_diff - null_mean)).sum() / len(null_diffs)
        p_value = max(p_value, 1.0 / (self.n_permutations + 1))

        # Prevalence and effect size
        marker_prevalence = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else np.nan
        cohens_d = observed_diff / null_std if null_std > 0 else 0.0

        timepoint = sample_obs['timepoint'].iloc[0] if 'timepoint' in sample_obs.columns else np.nan
        group = sample_obs['group'].iloc[0] if 'group' in sample_obs.columns else ''

        return {
            'sample_id': sample,
            'test_name': test_name,
            'immune_population': immune_pop,
            'tumor_base': tumor_base,
            'tumor_marker': tumor_marker,
            'n_immune': int(n_immune),
            'n_marker_pos': int(n_pos),
            'n_marker_neg': int(n_neg),
            'marker_prevalence': marker_prevalence,
            'mean_immune_neighbors_pos': float(obs_count_pos),
            'mean_immune_neighbors_neg': float(obs_count_neg),
            'observed_diff': observed_diff,
            'null_mean': null_mean,
            'null_std': null_std,
            'z_score': z_score,
            'cohens_d': cohens_d,
            'p_value': p_value,
            'timepoint': timepoint,
            'group': group
        }

    def _build_aggregate_matrix(self):
        """Build aggregate enrichment matrix (mean z-score across samples)."""
        df = self.results['pairwise_enrichment']
        if df.empty:
            return

        # Pivot to matrix: mean z-score across samples
        pivot = df.pivot_table(
            values='z_score',
            index='cell_type_a',
            columns='cell_type_b',
            aggfunc='mean'
        )

        self.results['aggregate_enrichment_matrix'] = pivot

        # Also build per-group matrices
        if 'group' in df.columns:
            group_matrices = {}
            for group in df['group'].unique():
                if pd.isna(group) or group == '':
                    continue
                group_df = df[df['group'] == group]
                group_pivot = group_df.pivot_table(
                    values='z_score',
                    index='cell_type_a',
                    columns='cell_type_b',
                    aggfunc='mean'
                )
                group_matrices[group] = group_pivot
            self.results['group_enrichment_matrices'] = group_matrices

    def _save_results(self):
        """Save results to CSV files."""
        for key, val in self.results.items():
            if key == 'group_enrichment_matrices':
                for group, matrix in val.items():
                    path = self.output_dir / f'enrichment_matrix_{group}.csv'
                    matrix.to_csv(path)
                continue

            if isinstance(val, pd.DataFrame) and not val.empty:
                path = self.output_dir / f'{key}.csv'
                val.to_csv(path, index=True if isinstance(val.index, pd.MultiIndex) or val.index.name else False)
                print(f"  Saved {key}: {val.shape} → {path.name}")

        print(f"\n  Results saved to: {self.output_dir}/")
