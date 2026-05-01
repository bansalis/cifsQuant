"""
Cell-to-Cell Distance Analysis
Calculate distances between cell populations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial import cKDTree
import warnings
from ..visualization.distance_analysis_plotter import DistanceAnalysisPlotter


class DistanceAnalysis:
    """
    Analyze distances between cell populations.

    Key features:
    - Nearest neighbor distances
    - Distance distributions
    - Per-sample and per-structure analysis
    - Temporal and group comparisons
    - Within-tumor filtering to avoid background lung bias
    - Differential distances (dist_to_pos - dist_to_neg) to cancel density effects
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        """
        Initialize distance analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data with phenotypes and spatial coordinates
        config : dict
            Configuration dictionary
        output_dir : Path
            Output directory
        """
        self.adata = adata
        self.full_config = config
        self.config = config['distance_analysis']
        self.output_dir = Path(output_dir) / 'distance_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Within-tumor-only filtering: restricts source cells to those inside a tumor boundary
        # (tumor_region_id != -1 and not NaN). This prevents background lung immune cells
        # from inflating mean distances to tumor-expressed markers.
        self.within_tumor_only = self.config.get('within_tumor_only', False)

        # k-nearest neighbors for distance computation.
        # k=1 is noisy and biased when target populations differ in density.
        # k=5 computes mean distance to the 5 nearest targets — more robust.
        self.k_neighbors = int(self.config.get('k_neighbors', 5))

        # Maximum distance cap (µm). Source cells further than this from ALL target
        # cells are excluded as non-adjacent. Prevents very distal immune cells
        # from dominating mean distance statistics.
        self.max_distance = self.config.get('max_distance', None)

        # Storage
        self.results = {}

    def run(self):
        """Run complete distance analysis."""
        print("\n" + "="*80)
        print("DISTANCE ANALYSIS")
        print("="*80)

        # Get pairings to analyze
        pairings = self.config.get('pairings', [])
        print(f"\nAnalyzing {len(pairings)} cell population pairings...")

        for pairing in pairings:
            source = pairing['source']
            targets = pairing['targets']

            for target in targets:
                print(f"\n  {source} → {target}")
                self._analyze_pairing(source, target)

        # Save results
        self._save_results()

        # Generate plots
        self._generate_plots()

        print("\n✓ Distance analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
        print("="*80 + "\n")

        return self.results

    def _analyze_pairing(self, source_pop: str, target_pop: str):
        """Analyze distance from source to target population."""
        # Check populations exist
        source_col = f'is_{source_pop}'
        target_col = f'is_{target_pop}'

        if source_col not in self.adata.obs.columns:
            warnings.warn(f"Source population '{source_pop}' not found, skipping")
            return

        if target_col not in self.adata.obs.columns:
            warnings.warn(f"Target population '{target_pop}' not found, skipping")
            return

        # Analyze per sample
        results_per_sample = []

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]
            sample_coords = self.adata.obsm['spatial'][sample_mask.values]

            # Get source and target cells
            source_mask = sample_data[source_col].values.astype(bool)
            target_mask = sample_data[target_col].values.astype(bool)

            # Apply within-tumor-only filter to source cells
            # This restricts to immune cells inside a tumor boundary, avoiding
            # background lung immune cells that inflate mean distances to tumor markers.
            if self.within_tumor_only and 'tumor_region_id' in sample_data.columns:
                valid_tumor = (sample_data['tumor_region_id'].notna() &
                               (sample_data['tumor_region_id'] != -1)).values
                source_mask = source_mask & valid_tumor

            if source_mask.sum() == 0 or target_mask.sum() == 0:
                continue

            source_coords = sample_coords[source_mask]
            target_coords = sample_coords[target_mask]

            # Apply max_distance cap: exclude source cells with no target within cap.
            # This removes very distal immune cells not adjacent to the tumor.
            if self.max_distance is not None and len(target_coords) > 0:
                cap_tree = cKDTree(target_coords)
                nn1, _ = cap_tree.query(source_coords, k=1)
                within_cap = nn1 <= self.max_distance
                source_coords = source_coords[within_cap]
                if len(source_coords) == 0:
                    continue

            # Calculate distances
            distances_dict = self._calculate_distances(source_coords, target_coords)

            # Add metadata
            distances_dict['sample_id'] = sample
            distances_dict['source_population'] = source_pop
            distances_dict['target_population'] = target_pop
            distances_dict['n_source_cells'] = int(source_mask.sum())
            distances_dict['n_target_cells'] = int(target_mask.sum())
            distances_dict['within_tumor_only'] = self.within_tumor_only

            # Add sample metadata
            for col in ['timepoint', 'group', 'genotype', 'treatment']:
                if col in sample_data.columns:
                    distances_dict[col] = sample_data[col].iloc[0]

            results_per_sample.append(distances_dict)

        # Convert to DataFrame
        if results_per_sample:
            df = pd.DataFrame(results_per_sample)
            pairing_name = f'{source_pop}_to_{target_pop}'
            self.results[pairing_name] = df

    def _calculate_distances(self, source_coords: np.ndarray,
                            target_coords: np.ndarray) -> Dict:
        """
        Calculate distance metrics from source to target.

        Uses k-nearest neighbours (k=self.k_neighbors) and returns the mean
        distance to the k nearest targets per source cell.  Using k>1 reduces
        noise when target populations differ in abundance: with k=1, a dense
        NINJA- field will always appear closer than sparse NINJA+ cells simply
        because the very nearest cell is more likely to be NINJA-.  Averaging
        over the k nearest targets gives a fairer density-corrected estimate.

        Parameters
        ----------
        source_coords : np.ndarray
            Coordinates of source cells (N x 2)
        target_coords : np.ndarray
            Coordinates of target cells (M x 2)

        Returns
        -------
        dict
            Distance metrics
        """
        tree = cKDTree(target_coords)

        k = min(self.k_neighbors, len(target_coords))
        raw_dists, _ = tree.query(source_coords, k=k)

        # For k=1 the result is 1-D; for k>1 average across the k neighbours.
        if raw_dists.ndim == 1:
            distances = raw_dists
        else:
            distances = raw_dists.mean(axis=1)

        metrics = {
            'mean_distance': float(np.mean(distances)),
            'median_distance': float(np.median(distances)),
            'std_distance': float(np.std(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances)),
            'q25_distance': float(np.percentile(distances, 25)),
            'q75_distance': float(np.percentile(distances, 75)),
            'n_source_used': int(len(distances)),
            'k_neighbors_used': int(k),
        }

        return metrics

    def _calculate_distance_distribution(self, source_coords: np.ndarray,
                                        target_coords: np.ndarray,
                                        bins: np.ndarray) -> np.ndarray:
        """
        Calculate distance distribution histogram.

        Parameters
        ----------
        source_coords : np.ndarray
            Coordinates of source cells
        target_coords : np.ndarray
            Coordinates of target cells
        bins : np.ndarray
            Distance bins

        Returns
        -------
        np.ndarray
            Histogram counts
        """
        tree = cKDTree(target_coords)
        distances, _ = tree.query(source_coords, k=1)

        hist, _ = np.histogram(distances, bins=bins)

        return hist

    def _compute_paired_differentials(self) -> pd.DataFrame:
        """
        Compute differential distances for paired positive/negative targets.

        For each source population and marker, if both {marker}_positive_tumor and
        {marker}_negative_tumor targets exist, compute:
            differential_distance = mean_dist_to_pos - mean_dist_to_neg

        Negative differential = source cells are closer to marker+ zone (biologically meaningful).
        Cancels pERK+ cell density effects because both targets are from the same tumor.

        Returns
        -------
        pd.DataFrame
            Differential distance table with one row per sample per source/marker pair.
        """
        paired_rows = []

        # Find all (source, marker) pairs where both _positive_tumor and _negative_tumor exist
        sources = set()
        for name in self.results:
            parts = name.split('_to_')
            if len(parts) == 2:
                sources.add(parts[0])

        for source in sources:
            # Collect targets for this source
            targets = {}
            for name, df in self.results.items():
                parts = name.split('_to_')
                if len(parts) == 2 and parts[0] == source:
                    targets[parts[1]] = df

            # Identify complete pos/neg pairs
            markers_found: Dict[str, Dict] = {}
            for target_name in targets:
                if target_name.endswith('_positive_tumor'):
                    marker = target_name[:-len('_positive_tumor')]
                    markers_found.setdefault(marker, {})['pos'] = target_name
                elif target_name.endswith('_negative_tumor'):
                    marker = target_name[:-len('_negative_tumor')]
                    markers_found.setdefault(marker, {})['neg'] = target_name

            for marker, pair_info in markers_found.items():
                if 'pos' not in pair_info or 'neg' not in pair_info:
                    continue

                pos_df = targets[pair_info['pos']]
                neg_df = targets[pair_info['neg']]

                # Merge on sample_id
                merge_on = ['sample_id']
                merged = pos_df.merge(neg_df, on=merge_on, suffixes=('_pos', '_neg'))

                if len(merged) == 0:
                    continue

                for _, row in merged.iterrows():
                    entry = {
                        'sample_id': row['sample_id'],
                        'source_population': source,
                        'marker': marker,
                        'mean_dist_to_pos': row.get('mean_distance_pos', np.nan),
                        'mean_dist_to_neg': row.get('mean_distance_neg', np.nan),
                        'n_source_cells': row.get('n_source_cells_pos', np.nan),
                        'n_pos_cells': row.get('n_target_cells_pos', np.nan),
                        'n_neg_cells': row.get('n_target_cells_neg', np.nan),
                        'within_tumor_only': self.within_tumor_only,
                    }
                    # Differential: negative means closer to positive zone
                    entry['differential_distance'] = entry['mean_dist_to_pos'] - entry['mean_dist_to_neg']

                    # Carry over metadata
                    for col in ['timepoint', 'group', 'genotype', 'treatment']:
                        for suffix in ['_pos', '_neg', '']:
                            cname = col + suffix
                            if cname in row.index and pd.notna(row[cname]):
                                entry[col] = row[cname]
                                break

                    paired_rows.append(entry)

        if paired_rows:
            return pd.DataFrame(paired_rows)
        return pd.DataFrame()

    def _save_results(self):
        """Save all results to files."""
        # Save each pairing's data
        for name, df in self.results.items():
            output_path = self.output_dir / f'{name}_distances.csv'
            df.to_csv(output_path, index=False)

        # Compute and save differential distances
        diff_df = self._compute_paired_differentials()
        if len(diff_df) > 0:
            diff_path = self.output_dir / 'differential_distances.csv'
            diff_df.to_csv(diff_path, index=False)
            self.results['differential_distances'] = diff_df
            print(f"\n  ✓ Saved differential_distances.csv ({len(diff_df)} rows)")

        # Create summary table
        summary_rows = []
        for name, df in self.results.items():
            if 'mean_distance' not in df.columns:
                continue
            summary_rows.append({
                'pairing': name,
                'n_samples': len(df),
                'mean_distance_overall': df['mean_distance'].mean(),
                'median_distance_overall': df['median_distance'].median()
            })

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = self.output_dir / 'distance_summary.csv'
            summary_df.to_csv(summary_path, index=False)

        print(f"\n  ✓ Saved {len(self.results)} distance datasets")

    def _generate_plots(self):
        """Generate comprehensive plots for all distance pairings."""
        if not self.results:
            return

        print("\n  Generating distance plots...")

        # Initialize plotter
        plotter = DistanceAnalysisPlotter(self.output_dir, self.full_config)

        # Get grouping config - use primary_grouping if set, otherwise fall back to group_column
        meta_config = self.full_config.get('metadata', {})
        group_col = meta_config.get('primary_grouping') or meta_config.get('group_column', 'group')
        groups = meta_config.get('groups_to_compare', None)

        # Plot each pairing
        for pairing_name, data in self.results.items():
            # Parse pairing name
            parts = pairing_name.split('_to_')
            if len(parts) == 2:
                source, target = parts
                print(f"    {source} → {target}")
                plotter.plot_distance_comprehensive(data, source, target, group_col, groups)

        # Generate heatmap of all pairings
        plotter.plot_all_distances_heatmap(self.results, group_col)

        # Plot differential distances if available
        if 'differential_distances' in self.results:
            plotter.plot_differential_distances(
                self.results['differential_distances'], group_col, groups
            )

        print(f"  ✓ Generated plots for {len(self.results)} pairings")

    def get_pairing_data(self, source: str, target: str) -> pd.DataFrame:
        """Get distance data for a specific pairing."""
        pairing_name = f'{source}_to_{target}'
        return self.results.get(pairing_name, pd.DataFrame())
