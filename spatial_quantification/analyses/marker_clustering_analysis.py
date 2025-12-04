"""
Marker Clustering Analysis with Randomization Testing
Analyzes spatial clustering of tumor markers and tests against random null model
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ks_2samp, mannwhitneyu
import warnings
from itertools import combinations

try:
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MarkerClusteringAnalysis:
    """
    Analyze spatial clustering of tumor markers with randomization testing.

    Key features:
    - Measure spatial clustering of marker+ cells (e.g., pERK+, Ki67+, MHC-II+)
    - Distance-based metrics (nearest neighbor distances, Hopkins statistic)
    - Ripley's K function for point pattern analysis
    - Randomization testing: generate N random permutations per tumor
    - Test if observed clustering is due to chance
    - Analyze marker co-localization and mutual exclusivity
    - Test if marker overlap is greater/less than expected by chance
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        """
        Initialize marker clustering analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data with tumor cells and markers
        config : dict
            Configuration dictionary
        output_dir : Path
            Output directory
        """
        self.adata = adata
        self.config = config
        self.output_dir = Path(output_dir) / 'marker_clustering_analysis'
        self.plots_dir = self.output_dir / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Get spatial coordinates
        if 'spatial' in self.adata.obsm:
            self.adata.obs['X_centroid'] = self.adata.obsm['spatial'][:, 0]
            self.adata.obs['Y_centroid'] = self.adata.obsm['spatial'][:, 1]

        # Get tumor markers from config
        self.tumor_markers = self._extract_tumor_markers()
        print(f"  Found {len(self.tumor_markers)} tumor markers for analysis")

        # Randomization parameters
        analysis_config = self.config.get('marker_clustering_analysis', {})
        self.n_permutations = analysis_config.get('n_permutations', 500)
        self.alpha = analysis_config.get('alpha', 0.05)

        # Results storage
        self.results = {}

    def _extract_tumor_markers(self) -> List[str]:
        """Extract tumor markers of interest from config."""
        markers = []

        # Look for tumor-specific markers in config
        if 'marker_clustering_analysis' in self.config:
            markers = self.config['marker_clustering_analysis'].get('tumor_markers', [])

        # If not specified, look for common tumor markers in phenotypes
        if not markers:
            common_tumor_markers = ['pERK', 'Ki67', 'NINJA', 'MHC-II', 'PD-L1']
            for marker in common_tumor_markers:
                col_name = f'is_{marker}'
                if col_name in self.adata.obs.columns:
                    markers.append(marker)

        return markers

    def run(self) -> Dict:
        """Run complete marker clustering analysis."""
        print("\n" + "="*80)
        print("MARKER CLUSTERING ANALYSIS WITH RANDOMIZATION TESTING")
        print("="*80)
        print(f"  Using {self.n_permutations} random permutations per tumor")

        if len(self.tumor_markers) == 0:
            print("  ⚠ No tumor markers found for clustering analysis")
            return {}

        # 1. Calculate observed clustering metrics
        print("\n1. Calculating observed spatial clustering metrics...")
        self._calculate_observed_clustering()

        # 2. Run randomization tests
        print("\n2. Running randomization tests...")
        self._run_randomization_tests()

        # 3. Analyze marker co-localization
        print("\n3. Analyzing marker co-localization...")
        self._analyze_marker_colocalization()

        # 4. Test marker overlap vs random
        print("\n4. Testing marker overlap against random null model...")
        self._test_marker_overlap_random()

        # 5. Save results
        print("\n5. Saving results...")
        self._save_results()

        print("\n✓ Marker clustering analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
        print("="*80 + "\n")

        return self.results

    def _calculate_observed_clustering(self):
        """Calculate observed spatial clustering metrics for each marker."""
        clustering_results = []

        # Identify tumor cells
        tumor_mask = self.adata.obs.get('is_tumor', True)
        if isinstance(tumor_mask, bool):
            # If no tumor column, try to identify tumor phenotype
            tumor_pheno_cols = [col for col in self.adata.obs.columns if 'tumor' in col.lower() and col.startswith('is_')]
            if tumor_pheno_cols:
                tumor_mask = self.adata.obs[tumor_pheno_cols[0]]
            else:
                tumor_mask = np.ones(len(self.adata), dtype=bool)

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = (self.adata.obs['sample_id'] == sample) & tumor_mask
            sample_data = self.adata.obs[sample_mask]

            if len(sample_data) < 10:
                continue

            # Get coordinates
            coords = self.adata.obsm['spatial'][sample_mask]

            for marker in self.tumor_markers:
                marker_col = f'is_{marker}'

                if marker_col not in sample_data.columns:
                    continue

                marker_positive = sample_data[marker_col].values
                n_positive = marker_positive.sum()

                if n_positive < 3:
                    continue

                proportion_positive = n_positive / len(marker_positive)

                # Get coordinates of marker+ cells
                pos_coords = coords[marker_positive]

                # Calculate clustering metrics
                metrics = self._calculate_clustering_metrics(coords, pos_coords, marker_positive)

                clustering_results.append({
                    'sample_id': sample,
                    'marker': marker,
                    'n_tumor_cells': len(marker_positive),
                    'n_marker_positive': int(n_positive),
                    'proportion_positive': proportion_positive,
                    **metrics,
                    'timepoint': sample_data['timepoint'].iloc[0] if 'timepoint' in sample_data.columns else np.nan,
                    'group': sample_data['group'].iloc[0] if 'group' in sample_data.columns else '',
                    'main_group': sample_data['main_group'].iloc[0] if 'main_group' in sample_data.columns else ''
                })

        if clustering_results:
            df = pd.DataFrame(clustering_results)
            self.results['observed_clustering'] = df
            print(f"  ✓ Calculated clustering metrics for {len(clustering_results)} sample-marker combinations")

    def _calculate_clustering_metrics(self, all_coords: np.ndarray,
                                     positive_coords: np.ndarray,
                                     marker_positive: np.ndarray) -> Dict:
        """Calculate various clustering metrics."""
        metrics = {}

        n_positive = len(positive_coords)
        n_total = len(all_coords)

        # 1. Nearest neighbor distances (marker+ to marker+)
        if n_positive >= 2:
            nn_model = NearestNeighbors(n_neighbors=min(2, n_positive))
            nn_model.fit(positive_coords)
            distances, _ = nn_model.kneighbors(positive_coords)

            # Distance to nearest neighbor (excluding self)
            if distances.shape[1] > 1:
                nn_distances = distances[:, 1]
                metrics['mean_nn_distance'] = np.mean(nn_distances)
                metrics['median_nn_distance'] = np.median(nn_distances)
                metrics['std_nn_distance'] = np.std(nn_distances)

        # 2. Mean distance to other marker+ cells
        if n_positive >= 2:
            pos_dists = pdist(positive_coords)
            metrics['mean_positive_distance'] = np.mean(pos_dists)
            metrics['median_positive_distance'] = np.median(pos_dists)

        # 3. Hopkins statistic (clustering tendency)
        # Sample m points from positive cells and m random points
        # Compare distances to nearest neighbors
        if n_positive >= 10:
            m = min(int(n_positive * 0.1), 50)

            # Sample from positive cells
            sample_idx = np.random.choice(n_positive, m, replace=False)
            sample_coords = positive_coords[sample_idx]

            # Calculate distances to nearest neighbor in positive cells
            nn_model = NearestNeighbors(n_neighbors=2)
            nn_model.fit(positive_coords)
            distances, _ = nn_model.kneighbors(sample_coords)
            w_distances = distances[:, 1]  # Exclude self

            # Generate m random points within the same bounding box
            x_min, y_min = all_coords.min(axis=0)
            x_max, y_max = all_coords.max(axis=0)
            random_coords = np.column_stack([
                np.random.uniform(x_min, x_max, m),
                np.random.uniform(y_min, y_max, m)
            ])

            # Calculate distances from random points to nearest positive cell
            distances, _ = nn_model.kneighbors(random_coords)
            u_distances = distances[:, 0]

            # Hopkins statistic
            hopkins = np.sum(u_distances) / (np.sum(u_distances) + np.sum(w_distances))
            metrics['hopkins_statistic'] = hopkins
            # hopkins ~ 0.5: random, > 0.5: clustered, < 0.5: uniform

        # 4. Ripley's K function at multiple radii
        radii = [50, 100, 150, 200]
        for r in radii:
            k_value = self._calculate_ripleys_k(positive_coords, all_coords, r)
            metrics[f'ripleys_k_{r}um'] = k_value

        return metrics

    def _calculate_ripleys_k(self, positive_coords: np.ndarray,
                            all_coords: np.ndarray, radius: float) -> float:
        """Calculate Ripley's K function at a given radius."""
        n = len(positive_coords)

        if n < 2:
            return 0.0

        # Calculate area of the region
        x_min, y_min = all_coords.min(axis=0)
        x_max, y_max = all_coords.max(axis=0)
        area = (x_max - x_min) * (y_max - y_min)

        if area == 0:
            return 0.0

        # Count pairs within radius
        dists = squareform(pdist(positive_coords))
        n_within_radius = (dists < radius).sum() - n  # Exclude diagonal

        # Ripley's K
        k = area * n_within_radius / (n * (n - 1))

        return k

    def _run_randomization_tests(self):
        """Run randomization tests for each marker in each sample."""
        if 'observed_clustering' not in self.results:
            return

        observed_df = self.results['observed_clustering']
        randomization_results = []

        tumor_mask = self.adata.obs.get('is_tumor', True)
        if isinstance(tumor_mask, bool):
            tumor_pheno_cols = [col for col in self.adata.obs.columns if 'tumor' in col.lower() and col.startswith('is_')]
            if tumor_pheno_cols:
                tumor_mask = self.adata.obs[tumor_pheno_cols[0]]
            else:
                tumor_mask = np.ones(len(self.adata), dtype=bool)

        for idx, row in observed_df.iterrows():
            sample = row['sample_id']
            marker = row['marker']

            print(f"    Randomizing {sample} - {marker}...")

            sample_mask = (self.adata.obs['sample_id'] == sample) & tumor_mask
            coords = self.adata.obsm['spatial'][sample_mask]

            marker_col = f'is_{marker}'
            marker_positive = self.adata.obs[sample_mask][marker_col].values
            n_positive = marker_positive.sum()

            if n_positive < 3:
                continue

            # Run N permutations
            random_metrics = []

            for perm in range(self.n_permutations):
                # Randomly shuffle marker+ status while preserving proportion
                random_positive = np.random.permutation(marker_positive)
                random_pos_coords = coords[random_positive]

                # Calculate metrics for random
                metrics = self._calculate_clustering_metrics(coords, random_pos_coords, random_positive)
                random_metrics.append(metrics)

            # Aggregate random metrics
            random_df = pd.DataFrame(random_metrics)

            # Compare observed to random distribution
            comparison = {
                'sample_id': sample,
                'marker': marker,
                'n_permutations': self.n_permutations
            }

            # For each metric, calculate p-value and z-score
            for metric_name in random_df.columns:
                if metric_name in row.index:
                    observed_val = row[metric_name]
                    random_vals = random_df[metric_name].values

                    # Remove NaN values
                    random_vals = random_vals[~np.isnan(random_vals)]

                    if len(random_vals) > 0 and not np.isnan(observed_val):
                        # Calculate p-value (two-tailed)
                        random_mean = np.mean(random_vals)
                        random_std = np.std(random_vals)

                        # P-value: proportion of random values more extreme than observed
                        p_value_lower = (random_vals <= observed_val).sum() / len(random_vals)
                        p_value_upper = (random_vals >= observed_val).sum() / len(random_vals)
                        p_value = 2 * min(p_value_lower, p_value_upper)

                        # Z-score
                        if random_std > 0:
                            z_score = (observed_val - random_mean) / random_std
                        else:
                            z_score = 0.0

                        comparison[f'{metric_name}_observed'] = observed_val
                        comparison[f'{metric_name}_random_mean'] = random_mean
                        comparison[f'{metric_name}_random_std'] = random_std
                        comparison[f'{metric_name}_p_value'] = p_value
                        comparison[f'{metric_name}_z_score'] = z_score
                        comparison[f'{metric_name}_significant'] = p_value < self.alpha

            comparison['timepoint'] = row['timepoint']
            comparison['group'] = row['group']
            comparison['main_group'] = row['main_group']

            randomization_results.append(comparison)

        if randomization_results:
            df = pd.DataFrame(randomization_results)
            self.results['randomization_tests'] = df
            print(f"  ✓ Completed randomization testing for {len(randomization_results)} cases")

    def _analyze_marker_colocalization(self):
        """Analyze spatial co-localization between marker pairs."""
        if len(self.tumor_markers) < 2:
            return

        colocalization_results = []

        marker_pairs = list(combinations(self.tumor_markers, 2))
        print(f"  Analyzing {len(marker_pairs)} marker pairs...")

        tumor_mask = self.adata.obs.get('is_tumor', True)
        if isinstance(tumor_mask, bool):
            tumor_pheno_cols = [col for col in self.adata.obs.columns if 'tumor' in col.lower() and col.startswith('is_')]
            if tumor_pheno_cols:
                tumor_mask = self.adata.obs[tumor_pheno_cols[0]]
            else:
                tumor_mask = np.ones(len(self.adata), dtype=bool)

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = (self.adata.obs['sample_id'] == sample) & tumor_mask
            sample_data = self.adata.obs[sample_mask]

            if len(sample_data) < 10:
                continue

            coords = self.adata.obsm['spatial'][sample_mask]

            for marker1, marker2 in marker_pairs:
                col1 = f'is_{marker1}'
                col2 = f'is_{marker2}'

                if col1 not in sample_data.columns or col2 not in sample_data.columns:
                    continue

                pos1 = sample_data[col1].values
                pos2 = sample_data[col2].values

                n1 = pos1.sum()
                n2 = pos2.sum()

                if n1 < 3 or n2 < 3:
                    continue

                # Calculate co-localization metrics
                coords1 = coords[pos1]
                coords2 = coords[pos2]

                # Mean distance from marker1+ cells to nearest marker2+ cell
                if len(coords2) > 0:
                    nn_model = NearestNeighbors(n_neighbors=1)
                    nn_model.fit(coords2)
                    distances, _ = nn_model.kneighbors(coords1)
                    mean_distance_1_to_2 = np.mean(distances)
                else:
                    mean_distance_1_to_2 = np.nan

                # Mean distance from marker2+ cells to nearest marker1+ cell
                if len(coords1) > 0:
                    nn_model = NearestNeighbors(n_neighbors=1)
                    nn_model.fit(coords1)
                    distances, _ = nn_model.kneighbors(coords2)
                    mean_distance_2_to_1 = np.mean(distances)
                else:
                    mean_distance_2_to_1 = np.nan

                # Average co-localization distance
                mean_coloc_distance = np.nanmean([mean_distance_1_to_2, mean_distance_2_to_1])

                # Cellular overlap
                both_positive = (pos1 & pos2).sum()
                either_positive = (pos1 | pos2).sum()

                jaccard_cellular = both_positive / either_positive if either_positive > 0 else 0

                colocalization_results.append({
                    'sample_id': sample,
                    'marker1': marker1,
                    'marker2': marker2,
                    'n_marker1': int(n1),
                    'n_marker2': int(n2),
                    'n_double_positive': int(both_positive),
                    'jaccard_cellular': jaccard_cellular,
                    'mean_distance_1_to_2': mean_distance_1_to_2,
                    'mean_distance_2_to_1': mean_distance_2_to_1,
                    'mean_colocalization_distance': mean_coloc_distance,
                    'timepoint': sample_data['timepoint'].iloc[0] if 'timepoint' in sample_data.columns else np.nan,
                    'group': sample_data['group'].iloc[0] if 'group' in sample_data.columns else '',
                    'main_group': sample_data['main_group'].iloc[0] if 'main_group' in sample_data.columns else ''
                })

        if colocalization_results:
            df = pd.DataFrame(colocalization_results)
            self.results['marker_colocalization'] = df
            print(f"  ✓ Analyzed co-localization for {len(colocalization_results)} pairs")

    def _test_marker_overlap_random(self):
        """Test if marker overlap is greater/less than expected by random chance."""
        if 'marker_colocalization' not in self.results:
            return

        coloc_df = self.results['marker_colocalization']
        overlap_random_results = []

        tumor_mask = self.adata.obs.get('is_tumor', True)
        if isinstance(tumor_mask, bool):
            tumor_pheno_cols = [col for col in self.adata.obs.columns if 'tumor' in col.lower() and col.startswith('is_')]
            if tumor_pheno_cols:
                tumor_mask = self.adata.obs[tumor_pheno_cols[0]]
            else:
                tumor_mask = np.ones(len(self.adata), dtype=bool)

        for idx, row in coloc_df.iterrows():
            sample = row['sample_id']
            marker1 = row['marker1']
            marker2 = row['marker2']

            print(f"    Testing overlap: {sample} - {marker1} vs {marker2}...")

            sample_mask = (self.adata.obs['sample_id'] == sample) & tumor_mask
            sample_data = self.adata.obs[sample_mask]

            col1 = f'is_{marker1}'
            col2 = f'is_{marker2}'

            pos1 = sample_data[col1].values
            pos2 = sample_data[col2].values

            # Observed overlap
            observed_overlap = (pos1 & pos2).sum()
            observed_distance = row['mean_colocalization_distance']

            # Run permutations
            random_overlaps = []
            random_distances = []

            coords = self.adata.obsm['spatial'][sample_mask]

            for perm in range(self.n_permutations):
                # Randomly shuffle marker2 status (keep marker1 fixed)
                random_pos2 = np.random.permutation(pos2)

                # Calculate overlap
                random_overlap = (pos1 & random_pos2).sum()
                random_overlaps.append(random_overlap)

                # Calculate distance
                if pos1.sum() > 0 and random_pos2.sum() > 0:
                    coords1 = coords[pos1]
                    coords2 = coords[random_pos2]

                    nn_model = NearestNeighbors(n_neighbors=1)
                    nn_model.fit(coords2)
                    distances, _ = nn_model.kneighbors(coords1)
                    mean_dist = np.mean(distances)
                    random_distances.append(mean_dist)

            random_overlaps = np.array(random_overlaps)
            random_distances = np.array(random_distances)

            # Calculate p-values
            p_value_overlap = (random_overlaps >= observed_overlap).sum() / len(random_overlaps)

            if len(random_distances) > 0 and not np.isnan(observed_distance):
                p_value_distance = (random_distances <= observed_distance).sum() / len(random_distances)
            else:
                p_value_distance = np.nan

            # Is overlap greater than random?
            more_overlap_than_random = observed_overlap > np.mean(random_overlaps)

            # Are markers mutually exclusive? (less overlap than random)
            mutually_exclusive = observed_overlap < np.mean(random_overlaps) and p_value_overlap > (1 - self.alpha)

            overlap_random_results.append({
                'sample_id': sample,
                'marker1': marker1,
                'marker2': marker2,
                'observed_overlap': int(observed_overlap),
                'random_overlap_mean': np.mean(random_overlaps),
                'random_overlap_std': np.std(random_overlaps),
                'overlap_p_value': p_value_overlap,
                'observed_distance': observed_distance,
                'random_distance_mean': np.mean(random_distances) if len(random_distances) > 0 else np.nan,
                'random_distance_std': np.std(random_distances) if len(random_distances) > 0 else np.nan,
                'distance_p_value': p_value_distance,
                'more_overlap_than_random': more_overlap_than_random,
                'mutually_exclusive': mutually_exclusive,
                'overlap_significant': p_value_overlap < self.alpha or p_value_overlap > (1 - self.alpha),
                'timepoint': row['timepoint'],
                'group': row['group'],
                'main_group': row['main_group']
            })

        if overlap_random_results:
            df = pd.DataFrame(overlap_random_results)
            self.results['overlap_vs_random'] = df
            print(f"  ✓ Tested {len(overlap_random_results)} marker pairs against random null model")

    def _save_results(self):
        """Save all results to CSV files."""
        for name, df in self.results.items():
            if isinstance(df, pd.DataFrame):
                output_path = self.output_dir / f'{name}.csv'
                df.to_csv(output_path, index=False)

        print(f"  ✓ Saved {len(self.results)} result datasets")
