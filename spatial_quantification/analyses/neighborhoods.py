"""
Cellular Neighborhood Analysis
Identify and analyze recurrent cellular neighborhoods
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
import warnings


class NeighborhoodAnalysis:
    """
    Analyze cellular neighborhoods.

    Key features:
    - Define neighborhoods across all samples
    - Composition analysis
    - Temporal evolution
    - Group comparisons
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        """
        Initialize neighborhood analysis.

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
        self.config = config['cellular_neighborhoods']
        self.output_dir = Path(output_dir) / 'neighborhoods'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.results = {}

    def run(self):
        """Run complete neighborhood analysis."""
        print("\n" + "="*80)
        print("CELLULAR NEIGHBORHOOD ANALYSIS")
        print("="*80)

        # Get populations to include
        populations = self.config.get('populations', [])
        window_size = self.config.get('window_size', 100)
        n_clusters = self.config.get('n_clusters', 8)

        print(f"\nAnalyzing neighborhoods with:")
        print(f"  - {len(populations)} populations")
        print(f"  - Window size: {window_size} μm")
        print(f"  - Number of neighborhood types: {n_clusters}")

        # Calculate neighborhood compositions
        print("\n1. Calculating neighborhood compositions...")
        compositions = self._calculate_compositions(populations, window_size)

        # Cluster neighborhoods
        print("\n2. Clustering neighborhood types...")
        self._cluster_neighborhoods(compositions, n_clusters)

        # Analyze composition
        print("\n3. Analyzing neighborhood composition...")
        self._analyze_composition()

        # Temporal evolution
        print("\n4. Analyzing temporal evolution...")
        self._analyze_temporal_evolution()

        # Save results
        self._save_results()

        print("\n✓ Neighborhood analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
        print("="*80 + "\n")

        return self.results

    def _calculate_compositions(self, populations: List[str],
                               window_size: float) -> pd.DataFrame:
        """Calculate neighborhood composition for each cell."""
        compositions = []

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]
            sample_coords = self.adata.obsm['spatial'][sample_mask.values]

            if len(sample_coords) < 10:
                continue

            # Build KDTree
            tree = cKDTree(sample_coords)

            # For each cell, find neighbors within window_size
            neighbors = tree.query_ball_point(sample_coords, window_size)

            # Calculate composition
            for cell_idx, neighbor_indices in enumerate(neighbors):
                if len(neighbor_indices) < 2:  # At least self + 1 neighbor
                    continue

                # Count each population type in neighborhood
                composition = {'sample_id': sample, 'cell_idx': cell_idx}

                for pop in populations:
                    pop_col = f'is_{pop}'
                    if pop_col in sample_data.columns:
                        neighbor_data = sample_data.iloc[neighbor_indices]
                        count = neighbor_data[pop_col].sum()
                        composition[pop] = count

                # Add metadata
                for col in ['timepoint', 'group', 'main_group']:
                    if col in sample_data.columns:
                        composition[col] = sample_data[col].iloc[cell_idx]

                compositions.append(composition)

        df = pd.DataFrame(compositions)
        self.results['compositions'] = df

        print(f"    ✓ Calculated compositions for {len(df):,} cells")

        return df

    def _cluster_neighborhoods(self, compositions: pd.DataFrame, n_clusters: int):
        """Cluster neighborhoods into types."""
        # Get population columns
        pop_cols = [col for col in compositions.columns
                   if col not in ['sample_id', 'cell_idx', 'timepoint', 'group', 'main_group']]

        if len(pop_cols) == 0:
            warnings.warn("No population columns found for clustering")
            return

        # Normalize compositions
        X = compositions[pop_cols].values
        X_sum = X.sum(axis=1, keepdims=True)
        X_sum[X_sum == 0] = 1  # Avoid division by zero
        X_norm = X / X_sum

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_norm)

        compositions['neighborhood_type'] = labels
        self.results['compositions'] = compositions

        print(f"    ✓ Identified {n_clusters} neighborhood types")

    def _analyze_composition(self):
        """Analyze composition of each neighborhood type."""
        compositions = self.results['compositions']

        # Get population columns
        pop_cols = [col for col in compositions.columns
                   if col not in ['sample_id', 'cell_idx', 'timepoint', 'group',
                                 'main_group', 'neighborhood_type']]

        # Calculate mean composition per neighborhood type
        composition_summary = []

        for nh_type in compositions['neighborhood_type'].unique():
            nh_data = compositions[compositions['neighborhood_type'] == nh_type]

            summary = {'neighborhood_type': int(nh_type), 'n_cells': len(nh_data)}

            for pop in pop_cols:
                summary[f'{pop}_mean'] = nh_data[pop].mean()

            composition_summary.append(summary)

        df = pd.DataFrame(composition_summary)
        self.results['neighborhood_composition'] = df

        print(f"    ✓ Analyzed composition of {len(df)} neighborhood types")

    def _analyze_temporal_evolution(self):
        """Analyze how neighborhoods evolve over time."""
        compositions = self.results['compositions']

        if 'timepoint' not in compositions.columns:
            print("    ⚠ No timepoint data available")
            return

        # Count neighborhood types per sample/timepoint
        evolution = compositions.groupby(
            ['sample_id', 'timepoint', 'neighborhood_type', 'group', 'main_group']
        ).size().reset_index(name='count')

        self.results['temporal_evolution'] = evolution

        print(f"    ✓ Analyzed temporal evolution")

    def _save_results(self):
        """Save all results."""
        for name, df in self.results.items():
            output_path = self.output_dir / f'{name}.csv'
            df.to_csv(output_path, index=False)

        print(f"\n  ✓ Saved {len(self.results)} result datasets")
