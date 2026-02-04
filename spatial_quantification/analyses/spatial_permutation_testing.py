"""
Spatial Permutation Testing Analysis

Determines whether spatial patterns of tumor marker expression and immune-tumor
marker associations are biologically meaningful or artifacts of random chance.

Test Types:
1. Single Marker Clustering - Are marker+ cells spatially clustered within tumors?
2. Two-Marker Co-localization - Do marker+ cells overlap more than chance within tumors?
3. Immune-Marker Enrichment - Are immune cells enriched near marker+ tumor cells?

Statistical Approach:
- Per-tumor Monte Carlo permutation testing
- Fixed cell coordinates with randomized marker assignments
- Benjamini-Hochberg FDR correction per sample

Key Concept:
- Analysis runs WITHIN each tumor structure (e.g., B cell cluster, tumor mass)
- Tests whether marker+ cells (e.g., pERK+) are spatially clustered within that tumor
- Permutes marker status among cells while keeping positions fixed
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.spatial import cKDTree
from scipy import stats
import warnings
from itertools import combinations
import json

try:
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn not available - some metrics will be limited")


class SpatialPermutationTesting:
    """
    Spatial permutation testing for tumor marker expression patterns.

    Implements Monte Carlo permutation tests to determine if observed spatial
    patterns are statistically significant compared to random null distributions.

    IMPORTANT: This analysis runs PER TUMOR STRUCTURE, not per marker region.
    The question is: "Within this tumor, are pERK+ cells clustered together?"
    NOT: "Is this pERK+ region significant?"
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        """
        Initialize spatial permutation testing.

        Parameters
        ----------
        adata : AnnData
            Annotated data with spatial coordinates and phenotype columns
        config : dict
            Configuration dictionary
        output_dir : Path
            Output directory for results
        """
        self.adata = adata
        self.config = config
        self.output_dir = Path(output_dir) / 'spatial_permutation'
        self.plots_dir = self.output_dir / 'plots'
        self.null_dist_dir = self.output_dir / 'null_distributions'

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.null_dist_dir.mkdir(parents=True, exist_ok=True)

        # Get spatial coordinates
        if 'spatial' in self.adata.obsm:
            self.adata.obs['X_centroid'] = self.adata.obsm['spatial'][:, 0]
            self.adata.obs['Y_centroid'] = self.adata.obsm['spatial'][:, 1]

        # Get analysis configuration
        self.analysis_config = self.config.get('spatial_permutation', {})

        # Parameters
        params = self.analysis_config.get('parameters', {})
        self.n_permutations = params.get('n_permutations', 500)
        self.min_tumor_cells = params.get('min_tumor_cells', 20)
        self.enrichment_radius = params.get('enrichment_radius', 50)
        self.clustering_radius = params.get('clustering_radius', 30)
        self.colocalization_radius = params.get('colocalization_radius', 30)
        self.alpha = params.get('alpha', 0.05)
        self.min_prevalence = params.get('min_prevalence', 0.05)
        self.max_prevalence = params.get('max_prevalence', 0.95)

        # Maximum number of structures to analyze (safety limit)
        self.max_structures = params.get('max_structures', 500)

        # CRITICAL: Get the structure column from config
        # This should match what per_structure_analysis creates
        structure_config = self.config.get('per_structure_analysis',
                                           self.config.get('per_tumor_analysis', {}))
        self.structure_column = self.analysis_config.get(
            'structure_column',
            structure_config.get('structure_column', 'tumor_structure_id')
        )

        # Random seed for reproducibility
        self.random_seed = self.config.get('advanced', {}).get('random_seed', 42)
        np.random.seed(self.random_seed)

        # Get test configurations
        self.tests = self.analysis_config.get('tests', [])

        # Grouping configuration
        grouping = self.analysis_config.get('grouping', {})
        self.split_by = grouping.get('split_by', ['group', 'timepoint'])
        self.aggregate = grouping.get('aggregate', True)

        # Results storage
        self.results = {}

        # Track excluded tumors
        self.exclusion_log = []

    def run(self) -> Dict:
        """Run complete spatial permutation testing analysis."""
        print("\n" + "="*80)
        print("SPATIAL PERMUTATION TESTING ANALYSIS")
        print("="*80)
        print(f"  Permutations: {self.n_permutations}")
        print(f"  Min tumor cells: {self.min_tumor_cells}")
        print(f"  Significance threshold: {self.alpha}")
        print(f"  Structure column: {self.structure_column}")
        print(f"  Tests configured: {len(self.tests)}")

        if len(self.tests) == 0:
            print("  No tests configured. Using default tests...")
            self._setup_default_tests()

        # Validate inputs
        print("\n1. Validating inputs...")
        if not self._validate_inputs():
            print("  Input validation failed")
            return self.results

        # Run per-tumor tests
        print("\n2. Running per-tumor permutation tests...")
        self._run_per_tumor_tests()

        # Aggregate to sample level
        print("\n3. Aggregating results to sample level...")
        self._aggregate_to_sample_level()

        # Cross-sample group comparisons
        print("\n4. Running cross-sample group comparisons...")
        self._run_group_comparisons()

        # Quality control
        print("\n5. Running quality control checks...")
        self._run_quality_control()

        # Save results
        print("\n6. Saving results...")
        self._save_results()

        print("\n" + "="*80)
        print("SPATIAL PERMUTATION TESTING COMPLETE")
        print(f"  Results saved to: {self.output_dir}/")
        print("="*80 + "\n")

        return self.results

    def _setup_default_tests(self):
        """Set up default test configurations based on available columns."""
        # Find potential marker columns (pERK, NINJA, Ki67, etc.)
        marker_patterns = ['pERK', 'NINJA', 'Ki67', 'MHC', 'PD-L1', 'GL7', 'BCL6']
        marker_cols = []
        for pattern in marker_patterns:
            cols = [col for col in self.adata.obs.columns
                   if pattern.lower() in col.lower() and col.startswith('is_')]
            marker_cols.extend(cols)

        # Find immune phenotype columns
        immune_patterns = ['CD8', 'CD4', 'T_cell', 'Tcell', 'Tfh']
        immune_cols = []
        for pattern in immune_patterns:
            cols = [col for col in self.adata.obs.columns
                   if pattern.lower() in col.lower() and col.startswith('is_')]
            immune_cols.extend(cols)

        # Remove duplicates
        marker_cols = list(dict.fromkeys(marker_cols))
        immune_cols = list(dict.fromkeys(immune_cols))

        # Set up default tests
        self.tests = []

        # Add clustering tests for each marker (limit to 3)
        for marker_col in marker_cols[:3]:
            self.tests.append({
                'type': 'clustering',
                'name': f'{marker_col}_clustering',
                'marker': marker_col
            })

        # Add colocalization tests for marker pairs (limit to 2)
        if len(marker_cols) >= 2:
            for m1, m2 in list(combinations(marker_cols, 2))[:2]:
                self.tests.append({
                    'type': 'colocalization',
                    'name': f'{m1}_vs_{m2}',
                    'marker1': m1,
                    'marker2': m2
                })

        # Add enrichment tests (limit to 2)
        for immune_col in immune_cols[:1]:
            for marker_col in marker_cols[:2]:
                self.tests.append({
                    'type': 'enrichment',
                    'name': f'{immune_col}_near_{marker_col}',
                    'tumor_marker': marker_col,
                    'immune_phenotype': immune_col,
                    'radius': self.enrichment_radius
                })

        print(f"    Set up {len(self.tests)} default tests")
        for t in self.tests:
            print(f"      - {t['type']}: {t['name']}")

    def _validate_inputs(self) -> bool:
        """Validate required inputs exist."""
        valid = True

        # Check for the specified structure column
        print(f"    Looking for structure column: '{self.structure_column}'")

        if self.structure_column not in self.adata.obs.columns:
            # List available columns that might be structure-related
            potential_cols = [col for col in self.adata.obs.columns
                            if 'tumor' in col.lower() or 'structure' in col.lower()]

            print(f"    ERROR: Structure column '{self.structure_column}' not found!")
            print(f"    Available structure-like columns: {potential_cols}")
            print(f"    ")
            print(f"    Please either:")
            print(f"    1. Run per_structure_analysis first to create tumor structures")
            print(f"    2. Set 'structure_column' in spatial_permutation config to the correct column")

            # Check if we should fall back to sample-level analysis
            if self.analysis_config.get('allow_sample_level', False):
                print(f"    Falling back to sample-level analysis (no per-tumor breakdown)")
                self.tumor_structure_col = None
            else:
                valid = False
                return valid
        else:
            self.tumor_structure_col = self.structure_column

            # Count unique structures
            n_structures = self.adata.obs[self.tumor_structure_col].nunique()
            valid_structures = self.adata.obs[self.tumor_structure_col].dropna()
            valid_structures = valid_structures[valid_structures != -1]
            n_valid = valid_structures.nunique()

            print(f"    Found {n_valid} valid tumor structures (excluding -1/NaN)")

            # Safety check: too many structures suggests wrong column
            if n_valid > self.max_structures:
                print(f"    ERROR: Too many structures ({n_valid} > {self.max_structures})")
                print(f"    This suggests the wrong column is being used.")
                print(f"    Expected: tumor/cluster IDs (typically 10-100 per sample)")
                print(f"    Got: {n_valid} unique values")

                # Show sample of values
                sample_values = valid_structures.head(20).tolist()
                print(f"    Sample values: {sample_values}")
                valid = False
                return valid

            if n_valid == 0:
                print(f"    ERROR: No valid tumor structures found")
                valid = False
                return valid

            if n_valid < 5:
                print(f"    WARNING: Very few structures ({n_valid}) - limited statistical power")

        # Check spatial coordinates
        if 'X_centroid' not in self.adata.obs.columns:
            if 'spatial' in self.adata.obsm:
                self.adata.obs['X_centroid'] = self.adata.obsm['spatial'][:, 0]
                self.adata.obs['Y_centroid'] = self.adata.obsm['spatial'][:, 1]
                print("    Extracted coordinates from obsm['spatial']")
            else:
                print("    ERROR: No spatial coordinates found")
                valid = False

        # Check marker columns exist
        for test in self.tests:
            test_type = test.get('type')
            if test_type == 'clustering':
                marker = test.get('marker')
                if marker and marker not in self.adata.obs.columns:
                    print(f"    WARNING: Marker column '{marker}' not found, skipping test")
                    test['skip'] = True
            elif test_type == 'colocalization':
                for m in ['marker1', 'marker2']:
                    marker = test.get(m)
                    if marker and marker not in self.adata.obs.columns:
                        print(f"    WARNING: Marker column '{marker}' not found, skipping test")
                        test['skip'] = True
            elif test_type == 'enrichment':
                for m in ['tumor_marker', 'immune_phenotype']:
                    marker = test.get(m)
                    if marker and marker not in self.adata.obs.columns:
                        print(f"    WARNING: Column '{marker}' not found, skipping test")
                        test['skip'] = True

        # Filter out skipped tests
        self.tests = [t for t in self.tests if not t.get('skip', False)]

        if len(self.tests) == 0:
            print("    ERROR: No valid tests remaining")
            valid = False
        else:
            print(f"    {len(self.tests)} valid tests configured")

        return valid

    def _run_per_tumor_tests(self):
        """Run permutation tests for each tumor structure."""
        all_results = []

        samples = self.adata.obs['sample_id'].unique()
        total_tests = 0
        total_structures = 0

        for sample in samples:
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask].copy()

            if 'spatial' in self.adata.obsm:
                sample_coords = self.adata.obsm['spatial'][sample_mask.values]
            else:
                sample_coords = sample_data[['X_centroid', 'Y_centroid']].values

            # Get metadata
            timepoint = sample_data['timepoint'].iloc[0] if 'timepoint' in sample_data.columns else np.nan
            group = sample_data['group'].iloc[0] if 'group' in sample_data.columns else ''

            # Get tumor structures in this sample
            if self.tumor_structure_col:
                tumor_ids = sample_data[self.tumor_structure_col].unique()
                # Filter out invalid IDs
                tumor_ids = [t for t in tumor_ids if pd.notna(t) and t != -1 and t != '-1']
                n_tumors = len(tumor_ids)
            else:
                tumor_ids = ['whole_sample']
                n_tumors = 1

            print(f"    {sample}: {n_tumors} tumor structures")

            if n_tumors == 0:
                print(f"      No valid structures, skipping sample")
                continue

            total_structures += n_tumors
            structures_processed = 0

            for tumor_id in tumor_ids:
                if self.tumor_structure_col and tumor_id != 'whole_sample':
                    tumor_mask = sample_data[self.tumor_structure_col] == tumor_id
                else:
                    tumor_mask = pd.Series(True, index=sample_data.index)

                tumor_data = sample_data[tumor_mask]
                tumor_coords_subset = sample_coords[tumor_mask.values]

                n_cells = len(tumor_data)

                # Check minimum cell count
                if n_cells < self.min_tumor_cells:
                    self.exclusion_log.append({
                        'sample_id': sample,
                        'tumor_id': tumor_id,
                        'reason': f'Insufficient cells ({n_cells} < {self.min_tumor_cells})'
                    })
                    continue

                structures_processed += 1

                # Run each test type
                for test in self.tests:
                    test_type = test.get('type')
                    test_name = test.get('name', f'{test_type}_{test.get("marker", "")}')

                    try:
                        if test_type == 'clustering':
                            result = self._run_clustering_test(
                                tumor_data, tumor_coords_subset, test, sample, tumor_id,
                                timepoint, group
                            )
                        elif test_type == 'colocalization':
                            result = self._run_colocalization_test(
                                tumor_data, tumor_coords_subset, test, sample, tumor_id,
                                timepoint, group
                            )
                        elif test_type == 'enrichment':
                            # For enrichment, we need immune cells from entire sample
                            result = self._run_enrichment_test(
                                sample_data, sample_coords, tumor_mask,
                                test, sample, tumor_id, timepoint, group
                            )
                        else:
                            continue

                        if result:
                            all_results.append(result)
                            total_tests += 1

                    except Exception as e:
                        print(f"      Error in test {test_name} for tumor {tumor_id}: {e}")
                        continue

            print(f"      Processed {structures_processed}/{n_tumors} structures")

        if all_results:
            self.results['per_tumor_results'] = pd.DataFrame(all_results)
            print(f"\n  Completed {total_tests} tests across {total_structures} structures")
        else:
            print("\n  WARNING: No test results generated")
            self.results['per_tumor_results'] = pd.DataFrame()

    def _run_clustering_test(self, tumor_data: pd.DataFrame, tumor_coords: np.ndarray,
                            test: Dict, sample: str, tumor_id: Union[str, int],
                            timepoint: float, group: str) -> Optional[Dict]:
        """
        Run single marker clustering test using Hopkins statistic.

        Tests whether marker+ cells are spatially clustered within the tumor.
        Null hypothesis: Marker+ cells are randomly distributed among tumor cells.
        """
        marker_col = test.get('marker')
        if marker_col not in tumor_data.columns:
            return None

        marker_positive = tumor_data[marker_col].values.astype(bool)
        n_positive = marker_positive.sum()
        n_cells = len(tumor_data)
        prevalence = n_positive / n_cells if n_cells > 0 else 0

        # Check prevalence bounds
        if prevalence < self.min_prevalence or prevalence > self.max_prevalence:
            self.exclusion_log.append({
                'sample_id': sample,
                'tumor_id': tumor_id,
                'test': test.get('name'),
                'reason': f'Prevalence out of bounds ({prevalence:.1%})'
            })
            return None

        if n_positive < 5:
            return None

        # Calculate observed Hopkins statistic
        observed_hopkins = self._calculate_hopkins(tumor_coords, marker_positive)

        if observed_hopkins is None:
            return None

        # Run permutations
        null_distribution = []
        for _ in range(self.n_permutations):
            # Shuffle marker assignments while preserving prevalence
            permuted_positive = np.random.permutation(marker_positive)
            perm_hopkins = self._calculate_hopkins(tumor_coords, permuted_positive)
            if perm_hopkins is not None:
                null_distribution.append(perm_hopkins)

        null_distribution = np.array(null_distribution)

        if len(null_distribution) < self.n_permutations * 0.5:
            return None

        # Calculate statistics
        null_mean = np.mean(null_distribution)
        null_std = np.std(null_distribution)

        # Z-score (effect size)
        z_score = (observed_hopkins - null_mean) / null_std if null_std > 0 else 0

        # P-value (one-tailed: is observed > null? i.e., more clustered)
        p_value = (null_distribution >= observed_hopkins).sum() / len(null_distribution)

        return {
            'sample_id': sample,
            'tumor_id': tumor_id,
            'test_type': 'clustering',
            'test_name': test.get('name'),
            'marker': marker_col,
            'n_cells': n_cells,
            'n_positive': int(n_positive),
            'prevalence': prevalence,
            'observed_statistic': observed_hopkins,
            'null_mean': null_mean,
            'null_std': null_std,
            'z_score': z_score,
            'p_value': p_value,
            'timepoint': timepoint,
            'group': group
        }

    def _calculate_hopkins(self, coords: np.ndarray, positive_mask: np.ndarray) -> Optional[float]:
        """
        Calculate Hopkins statistic for clustering tendency.

        H > 0.5 indicates clustering, H < 0.5 indicates uniformity, H ~ 0.5 is random.
        """
        pos_coords = coords[positive_mask]
        n_positive = len(pos_coords)

        if n_positive < 10:
            return None

        # Sample size
        m = min(int(n_positive * 0.1), 50)
        if m < 5:
            m = min(5, n_positive)

        # Sample points from positive cells
        sample_idx = np.random.choice(n_positive, m, replace=False)
        sample_coords = pos_coords[sample_idx]

        # Build KDTree for positive cells
        tree = cKDTree(pos_coords)

        # Calculate distances from sample to nearest neighbor in positive cells (excluding self)
        w_distances, _ = tree.query(sample_coords, k=2)
        w_distances = w_distances[:, 1]  # Exclude self

        # Generate random points within bounding box
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        random_coords = np.column_stack([
            np.random.uniform(x_min, x_max, m),
            np.random.uniform(y_min, y_max, m)
        ])

        # Calculate distances from random points to nearest positive cell
        u_distances, _ = tree.query(random_coords, k=1)

        # Hopkins statistic
        sum_u = np.sum(u_distances)
        sum_w = np.sum(w_distances)

        if sum_u + sum_w == 0:
            return None

        hopkins = sum_u / (sum_u + sum_w)

        return hopkins

    def _run_colocalization_test(self, tumor_data: pd.DataFrame, tumor_coords: np.ndarray,
                                 test: Dict, sample: str, tumor_id: Union[str, int],
                                 timepoint: float, group: str) -> Optional[Dict]:
        """
        Run two-marker co-localization test using Cross-K function.

        Tests whether two markers spatially overlap more than expected by chance.
        Null hypothesis: Markers are independently distributed among tumor cells.
        """
        marker1_col = test.get('marker1')
        marker2_col = test.get('marker2')

        if marker1_col not in tumor_data.columns or marker2_col not in tumor_data.columns:
            return None

        marker1_positive = tumor_data[marker1_col].values.astype(bool)
        marker2_positive = tumor_data[marker2_col].values.astype(bool)

        n1 = marker1_positive.sum()
        n2 = marker2_positive.sum()
        n_cells = len(tumor_data)

        if n1 < 5 or n2 < 5:
            return None

        # Calculate observed cross-K at specified radius
        radius = test.get('radius', self.colocalization_radius)
        observed_cross_k = self._calculate_cross_k(tumor_coords, marker1_positive,
                                                   marker2_positive, radius)

        if observed_cross_k is None:
            return None

        # Run permutations - INDEPENDENTLY permute each marker
        null_distribution = []
        for _ in range(self.n_permutations):
            perm_marker1 = np.random.permutation(marker1_positive)
            perm_marker2 = np.random.permutation(marker2_positive)
            perm_cross_k = self._calculate_cross_k(tumor_coords, perm_marker1,
                                                   perm_marker2, radius)
            if perm_cross_k is not None:
                null_distribution.append(perm_cross_k)

        null_distribution = np.array(null_distribution)

        if len(null_distribution) < self.n_permutations * 0.5:
            return None

        # Calculate statistics
        null_mean = np.mean(null_distribution)
        null_std = np.std(null_distribution)

        z_score = (observed_cross_k - null_mean) / null_std if null_std > 0 else 0

        # Two-tailed p-value
        p_lower = (null_distribution <= observed_cross_k).sum() / len(null_distribution)
        p_upper = (null_distribution >= observed_cross_k).sum() / len(null_distribution)
        p_value = 2 * min(p_lower, p_upper)

        # Calculate Jaccard overlap
        both_positive = (marker1_positive & marker2_positive).sum()
        either_positive = (marker1_positive | marker2_positive).sum()
        jaccard = both_positive / either_positive if either_positive > 0 else 0

        return {
            'sample_id': sample,
            'tumor_id': tumor_id,
            'test_type': 'colocalization',
            'test_name': test.get('name'),
            'marker': f'{marker1_col}_vs_{marker2_col}',
            'marker1': marker1_col,
            'marker2': marker2_col,
            'n_cells': n_cells,
            'n_marker1': int(n1),
            'n_marker2': int(n2),
            'n_double_positive': int(both_positive),
            'jaccard_overlap': jaccard,
            'prevalence': (n1 + n2) / (2 * n_cells),
            'observed_statistic': observed_cross_k,
            'null_mean': null_mean,
            'null_std': null_std,
            'z_score': z_score,
            'p_value': p_value,
            'radius': radius,
            'timepoint': timepoint,
            'group': group
        }

    def _calculate_cross_k(self, coords: np.ndarray, mask1: np.ndarray,
                          mask2: np.ndarray, radius: float) -> Optional[float]:
        """
        Calculate Cross-K function value at given radius.

        Measures the number of marker2+ cells within radius of marker1+ cells,
        normalized by the expected number under CSR (complete spatial randomness).
        """
        coords1 = coords[mask1]
        coords2 = coords[mask2]

        n1 = len(coords1)
        n2 = len(coords2)

        if n1 == 0 or n2 == 0:
            return None

        # Build KDTree for marker2 cells
        tree2 = cKDTree(coords2)

        # Count marker2 cells within radius of each marker1 cell
        counts = tree2.query_ball_point(coords1, r=radius, return_length=True)
        total_count = np.sum(counts)

        # Calculate area of the region
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        area = (x_max - x_min) * (y_max - y_min)

        if area == 0:
            return None

        # Cross-K function: normalized count
        cross_k = area * total_count / (n1 * n2) if n1 * n2 > 0 else 0

        return cross_k

    def _run_enrichment_test(self, sample_data: pd.DataFrame, sample_coords: np.ndarray,
                            tumor_mask: pd.Series, test: Dict, sample: str,
                            tumor_id: Union[str, int], timepoint: float,
                            group: str) -> Optional[Dict]:
        """
        Run immune-marker enrichment test.

        Tests whether immune cells are enriched near marker+ tumor cells.
        Null hypothesis: Immune cells are equally distributed near marker+ and marker- cells.
        """
        tumor_marker_col = test.get('tumor_marker')
        immune_col = test.get('immune_phenotype')
        radius = test.get('radius', self.enrichment_radius)

        if tumor_marker_col not in sample_data.columns or immune_col not in sample_data.columns:
            return None

        # Get tumor cells and their marker status
        tumor_data = sample_data[tumor_mask]
        tumor_coords_subset = sample_coords[tumor_mask.values]

        marker_positive = tumor_data[tumor_marker_col].values.astype(bool)
        n_positive = marker_positive.sum()
        n_tumor = len(tumor_data)

        if n_positive < 5 or n_positive == n_tumor:
            return None

        # Get immune cells (from entire sample, not just tumor)
        immune_mask = sample_data[immune_col].values.astype(bool)
        immune_coords = sample_coords[immune_mask]
        n_immune = len(immune_coords)

        if n_immune < 5:
            return None

        # Calculate observed enrichment: mean immune count near marker+ cells
        observed_enrichment = self._calculate_enrichment(
            tumor_coords_subset, marker_positive, immune_coords, radius
        )

        if observed_enrichment is None:
            return None

        # Run permutations - permute marker status among tumor cells
        null_distribution = []
        for _ in range(self.n_permutations):
            perm_positive = np.random.permutation(marker_positive)
            perm_enrichment = self._calculate_enrichment(
                tumor_coords_subset, perm_positive, immune_coords, radius
            )
            if perm_enrichment is not None:
                null_distribution.append(perm_enrichment)

        null_distribution = np.array(null_distribution)

        if len(null_distribution) < self.n_permutations * 0.5:
            return None

        # Calculate statistics
        null_mean = np.mean(null_distribution)
        null_std = np.std(null_distribution)

        z_score = (observed_enrichment - null_mean) / null_std if null_std > 0 else 0

        # One-tailed p-value (enrichment = observed > null)
        p_value = (null_distribution >= observed_enrichment).sum() / len(null_distribution)

        # Fold change
        fold_change = observed_enrichment / null_mean if null_mean > 0 else np.inf

        return {
            'sample_id': sample,
            'tumor_id': tumor_id,
            'test_type': 'enrichment',
            'test_name': test.get('name'),
            'marker': f'{immune_col}_near_{tumor_marker_col}',
            'tumor_marker': tumor_marker_col,
            'immune_phenotype': immune_col,
            'n_cells': n_tumor,
            'n_positive': int(n_positive),
            'n_immune': n_immune,
            'prevalence': n_positive / n_tumor,
            'observed_statistic': observed_enrichment,
            'null_mean': null_mean,
            'null_std': null_std,
            'z_score': z_score,
            'p_value': p_value,
            'fold_change': fold_change,
            'radius': radius,
            'timepoint': timepoint,
            'group': group
        }

    def _calculate_enrichment(self, tumor_coords: np.ndarray, marker_positive: np.ndarray,
                             immune_coords: np.ndarray, radius: float) -> Optional[float]:
        """Calculate mean immune count within radius of marker+ tumor cells."""
        pos_coords = tumor_coords[marker_positive]

        if len(pos_coords) == 0 or len(immune_coords) == 0:
            return None

        # Build KDTree for immune cells
        tree = cKDTree(immune_coords)

        # Count immune cells within radius of each marker+ tumor cell
        counts = tree.query_ball_point(pos_coords, r=radius, return_length=True)

        return np.mean(counts)

    def _aggregate_to_sample_level(self):
        """Aggregate per-tumor results to sample level with FDR correction."""
        if 'per_tumor_results' not in self.results or len(self.results['per_tumor_results']) == 0:
            return

        per_tumor_df = self.results['per_tumor_results']

        # Apply FDR correction per sample
        corrected_results = []

        for sample in per_tumor_df['sample_id'].unique():
            sample_data = per_tumor_df[per_tumor_df['sample_id'] == sample].copy()

            # FDR correction using Benjamini-Hochberg
            p_values = sample_data['p_value'].values
            n_tests = len(p_values)

            if n_tests > 1:
                # Sort p-values
                sorted_idx = np.argsort(p_values)
                sorted_p = p_values[sorted_idx]

                # BH correction
                adjusted_p = np.zeros(n_tests)
                for i, idx in enumerate(sorted_idx):
                    rank = i + 1
                    adjusted_p[idx] = min(sorted_p[i] * n_tests / rank, 1.0)

                # Ensure monotonicity
                for i in range(n_tests - 2, -1, -1):
                    if i + 1 < n_tests:
                        adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])

                sample_data['p_adjusted'] = adjusted_p
            else:
                sample_data['p_adjusted'] = sample_data['p_value']

            sample_data['significant'] = sample_data['p_adjusted'] < self.alpha
            corrected_results.append(sample_data)

        if corrected_results:
            self.results['per_tumor_results'] = pd.concat(corrected_results, ignore_index=True)

        # Aggregate to sample level
        sample_summaries = []
        per_tumor_df = self.results['per_tumor_results']

        for sample in per_tumor_df['sample_id'].unique():
            sample_data = per_tumor_df[per_tumor_df['sample_id'] == sample]

            # Get metadata
            timepoint = sample_data['timepoint'].iloc[0]
            group = sample_data['group'].iloc[0]

            # Summarize by test type/name
            for test_name in sample_data['test_name'].unique():
                test_data = sample_data[sample_data['test_name'] == test_name]

                n_tumors = len(test_data)
                n_significant = test_data['significant'].sum()

                summary = {
                    'sample_id': sample,
                    'group': group,
                    'timepoint': timepoint,
                    'test_type': test_data['test_type'].iloc[0],
                    'test_name': test_name,
                    'marker': test_data['marker'].iloc[0],
                    'n_tumors_tested': n_tumors,
                    'n_significant': int(n_significant),
                    'pct_significant': n_significant / n_tumors * 100 if n_tumors > 0 else 0,
                    'mean_effect_size': test_data['z_score'].mean(),
                    'std_effect_size': test_data['z_score'].std(),
                    'median_p_value': test_data['p_value'].median(),
                    'mean_observed': test_data['observed_statistic'].mean(),
                    'mean_null': test_data['null_mean'].mean()
                }

                sample_summaries.append(summary)

        if sample_summaries:
            self.results['sample_summary'] = pd.DataFrame(sample_summaries)
            print(f"    Aggregated results for {len(per_tumor_df['sample_id'].unique())} samples")

    def _run_group_comparisons(self):
        """Run statistical comparisons between groups."""
        if 'sample_summary' not in self.results:
            return

        summary_df = self.results['sample_summary']

        # Get unique groups
        groups = summary_df['group'].unique()

        if len(groups) < 2:
            print("    Insufficient groups for comparison")
            return

        comparisons = []

        # Compare all pairs of groups
        for test_name in summary_df['test_name'].unique():
            test_data = summary_df[summary_df['test_name'] == test_name]

            for g1, g2 in combinations(groups, 2):
                g1_data = test_data[test_data['group'] == g1]['mean_effect_size'].dropna()
                g2_data = test_data[test_data['group'] == g2]['mean_effect_size'].dropna()

                if len(g1_data) < 2 or len(g2_data) < 2:
                    continue

                # Wilcoxon rank-sum test for effect sizes
                try:
                    stat, p_value = stats.mannwhitneyu(g1_data, g2_data, alternative='two-sided')
                except Exception:
                    continue

                # Fisher's exact test for proportion significant
                g1_sig = test_data[test_data['group'] == g1]['n_significant'].sum()
                g1_total = test_data[test_data['group'] == g1]['n_tumors_tested'].sum()
                g2_sig = test_data[test_data['group'] == g2]['n_significant'].sum()
                g2_total = test_data[test_data['group'] == g2]['n_tumors_tested'].sum()

                if g1_total > 0 and g2_total > 0:
                    table = [[g1_sig, g1_total - g1_sig], [g2_sig, g2_total - g2_sig]]
                    try:
                        _, fisher_p = stats.fisher_exact(table)
                    except Exception:
                        fisher_p = np.nan
                else:
                    fisher_p = np.nan

                comparison = {
                    'test_type': test_data['test_type'].iloc[0],
                    'test_name': test_name,
                    'marker': test_data['marker'].iloc[0],
                    'group1': g1,
                    'group2': g2,
                    'n_samples_g1': len(g1_data),
                    'n_samples_g2': len(g2_data),
                    'mean_effect_g1': g1_data.mean(),
                    'mean_effect_g2': g2_data.mean(),
                    'effect_difference': g1_data.mean() - g2_data.mean(),
                    'wilcoxon_p': p_value,
                    'pct_sig_g1': g1_sig / g1_total * 100 if g1_total > 0 else 0,
                    'pct_sig_g2': g2_sig / g2_total * 100 if g2_total > 0 else 0,
                    'fisher_p': fisher_p
                }

                comparisons.append(comparison)

        if comparisons:
            self.results['group_comparison'] = pd.DataFrame(comparisons)
            print(f"    Completed {len(comparisons)} group comparisons")

    def _run_quality_control(self):
        """Run quality control checks on results."""
        qc_results = {
            'n_tests_run': 0,
            'n_tumors_excluded': len(self.exclusion_log),
            'exclusion_reasons': {},
            'p_value_distribution': {},
            'power_warnings': []
        }

        if 'per_tumor_results' in self.results and len(self.results['per_tumor_results']) > 0:
            per_tumor_df = self.results['per_tumor_results']
            qc_results['n_tests_run'] = len(per_tumor_df)

            # P-value distribution check
            p_values = per_tumor_df['p_value'].dropna()
            if len(p_values) > 10:
                ks_stat, ks_p = stats.kstest(p_values, 'uniform')
                qc_results['p_value_distribution'] = {
                    'ks_statistic': float(ks_stat),
                    'ks_p_value': float(ks_p),
                    'interpretation': 'Uniform (no global effect)' if ks_p > 0.05 else 'Non-uniform (global effect present)'
                }

            # Check for degenerate tests
            zero_variance = (per_tumor_df['null_std'] == 0).sum()
            if zero_variance > 0:
                qc_results['power_warnings'].append(
                    f'{zero_variance} tests had zero variance in null distribution'
                )

        # Summarize exclusion reasons
        for entry in self.exclusion_log:
            reason = entry.get('reason', 'Unknown')
            qc_results['exclusion_reasons'][reason] = qc_results['exclusion_reasons'].get(reason, 0) + 1

        self.results['quality_control'] = qc_results

        # Save exclusion log
        if self.exclusion_log:
            exclusion_df = pd.DataFrame(self.exclusion_log)
            exclusion_df.to_csv(self.output_dir / 'exclusion_log.csv', index=False)

        # Print summary
        print(f"    Tests run: {qc_results['n_tests_run']}")
        print(f"    Tumors excluded: {qc_results['n_tumors_excluded']}")
        if qc_results['exclusion_reasons']:
            print("    Exclusion reasons:")
            for reason, count in qc_results['exclusion_reasons'].items():
                print(f"      - {reason}: {count}")

    def _save_results(self):
        """Save all results to files."""
        # Save DataFrames
        for name, data in self.results.items():
            if isinstance(data, pd.DataFrame):
                if len(data) > 0:
                    output_path = self.output_dir / f'{name}.csv'
                    data.to_csv(output_path, index=False)
                    print(f"    Saved {name}.csv ({len(data)} rows)")
            elif isinstance(data, dict):
                output_path = self.output_dir / f'{name}.json'
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                print(f"    Saved {name}.json")

        # Save configuration used
        config_used = {
            'n_permutations': self.n_permutations,
            'min_tumor_cells': self.min_tumor_cells,
            'alpha': self.alpha,
            'min_prevalence': self.min_prevalence,
            'max_prevalence': self.max_prevalence,
            'enrichment_radius': self.enrichment_radius,
            'clustering_radius': self.clustering_radius,
            'colocalization_radius': self.colocalization_radius,
            'structure_column': self.structure_column,
            'tests': self.tests,
            'random_seed': self.random_seed
        }

        with open(self.output_dir / 'configuration_used.json', 'w') as f:
            json.dump(config_used, f, indent=2)

        print(f"    Saved configuration_used.json")


def run_spatial_permutation_testing(adata, config: Dict, output_dir: Path) -> Dict:
    """
    Convenience function to run spatial permutation testing.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    config : dict
        Configuration dictionary
    output_dir : Path
        Output directory

    Returns
    -------
    dict
        Results dictionary
    """
    analysis = SpatialPermutationTesting(adata, config, output_dir)
    return analysis.run()
