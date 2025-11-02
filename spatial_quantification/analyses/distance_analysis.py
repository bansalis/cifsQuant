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


class DistanceAnalysis:
    """
    Analyze distances between cell populations.

    Key features:
    - Nearest neighbor distances
    - Distance distributions
    - Per-sample and per-structure analysis
    - Temporal and group comparisons
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
        self.config = config['distance_analysis']
        self.output_dir = Path(output_dir) / 'distance_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
            source_mask = sample_data[source_col].values
            target_mask = sample_data[target_col].values

            if source_mask.sum() == 0 or target_mask.sum() == 0:
                continue

            source_coords = sample_coords[source_mask]
            target_coords = sample_coords[target_mask]

            # Calculate distances
            distances_dict = self._calculate_distances(source_coords, target_coords)

            # Add metadata
            distances_dict['sample_id'] = sample
            distances_dict['source_population'] = source_pop
            distances_dict['target_population'] = target_pop
            distances_dict['n_source_cells'] = int(source_mask.sum())
            distances_dict['n_target_cells'] = int(target_mask.sum())

            # Add sample metadata
            for col in ['timepoint', 'group', 'main_group', 'genotype', 'treatment']:
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
        # Build KDTree for fast nearest neighbor search
        tree = cKDTree(target_coords)

        # Find nearest neighbor for each source cell
        distances, indices = tree.query(source_coords, k=1)

        # Calculate metrics
        metrics = {
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'q25_distance': np.percentile(distances, 25),
            'q75_distance': np.percentile(distances, 75)
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

    def _save_results(self):
        """Save all results to files."""
        # Save each pairing's data
        for name, df in self.results.items():
            output_path = self.output_dir / f'{name}_distances.csv'
            df.to_csv(output_path, index=False)

        # Create summary table
        summary_rows = []
        for name, df in self.results.items():
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

    def get_pairing_data(self, source: str, target: str) -> pd.DataFrame:
        """Get distance data for a specific pairing."""
        pairing_name = f'{source}_to_{target}'
        return self.results.get(pairing_name, pd.DataFrame())
