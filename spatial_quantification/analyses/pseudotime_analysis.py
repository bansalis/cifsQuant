"""
Pseudotime Differentiation Analysis
Infer differentiation trajectories from spatial and phenotypic data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings


class PseudotimeAnalysis:
    """
    Pseudotime differentiation analysis.

    Key features:
    - Infer differentiation trajectories based on marker expression
    - Diffusion pseudotime approach
    - Temporal ordering of tumor cell states
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        """
        Initialize pseudotime analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data with phenotypes
        config : dict
            Configuration dictionary
        output_dir : Path
            Output directory
        """
        self.adata = adata
        self.config = config
        self.output_dir = Path(output_dir) / 'pseudotime_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Markers for trajectory
        self.trajectory_markers = ['is_PERK', 'is_AGFP', 'is_KI67']

        # Results storage
        self.results = {}

    def run(self):
        """Run complete pseudotime analysis."""
        print("\n" + "="*80)
        print("PSEUDOTIME ANALYSIS")
        print("="*80)

        # Calculate marker-based pseudotime
        print("\n1. Calculating marker-based pseudotime...")
        self._calculate_marker_pseudotime()

        # Analyze trajectory dynamics
        print("\n2. Analyzing trajectory dynamics...")
        self._analyze_trajectory_dynamics()

        # Save results
        self._save_results()

        print("\n✓ Pseudotime analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
        print("="*80 + "\n")

        return self.results

    def _calculate_marker_pseudotime(self):
        """Calculate pseudotime based on marker expression patterns."""
        tumor_col = 'is_Tumor'

        if tumor_col not in self.adata.obs.columns:
            warnings.warn("Tumor phenotype not found")
            return

        # Check if trajectory markers are available
        available_markers = [m for m in self.trajectory_markers if m in self.adata.obs.columns]

        if len(available_markers) < 2:
            print("    ⚠ Insufficient markers for pseudotime analysis")
            return

        pseudotime_results = []

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]
            sample_coords = self.adata.obsm['spatial'][sample_mask.values]

            # Get tumor cells
            tumor_mask = sample_data[tumor_col]
            tumor_data = sample_data[tumor_mask]
            tumor_coords = sample_coords[tumor_mask.values]

            if len(tumor_data) < 50:
                continue

            # Create feature matrix from markers
            X = tumor_data[available_markers].values.astype(float)

            # Add spatial features (normalized)
            coords_norm = tumor_coords - tumor_coords.mean(axis=0)
            coords_norm = coords_norm / (coords_norm.std(axis=0) + 1e-10)

            X_combined = np.hstack([X, coords_norm * 0.1])  # Small weight on spatial

            # PCA for dimensionality reduction
            if X_combined.shape[1] > 1:
                pca = PCA(n_components=min(3, X_combined.shape[1]))
                X_reduced = pca.fit_transform(X_combined)

                # Pseudotime as first principal component
                pseudotime = X_reduced[:, 0]

                # Normalize to [0, 1]
                pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min() + 1e-10)

                # Store per cell
                for idx, (cell_idx, pt) in enumerate(zip(tumor_data.index, pseudotime)):
                    result = {
                        'sample_id': sample,
                        'cell_id': cell_idx,
                        'pseudotime': pt,
                        'PC1': X_reduced[idx, 0],
                        'PC2': X_reduced[idx, 1] if X_reduced.shape[1] > 1 else 0,
                        'PC3': X_reduced[idx, 2] if X_reduced.shape[1] > 2 else 0,
                        'timepoint': tumor_data['timepoint'].iloc[0] if 'timepoint' in tumor_data.columns else np.nan,
                        'group': tumor_data['group'].iloc[0] if 'group' in tumor_data.columns else '',
                        'main_group': tumor_data['main_group'].iloc[0] if 'main_group' in tumor_data.columns else ''
                    }

                    # Add marker states
                    for marker in available_markers:
                        marker_name = marker.replace('is_', '')
                        result[marker_name] = tumor_data[marker].iloc[idx]

                    pseudotime_results.append(result)

        if pseudotime_results:
            df = pd.DataFrame(pseudotime_results)
            self.results['cell_pseudotime'] = df
            print(f"    ✓ Calculated pseudotime for {len(pseudotime_results)} cells")

    def _analyze_trajectory_dynamics(self):
        """Analyze how markers change along pseudotime trajectory."""
        if 'cell_pseudotime' not in self.results:
            return

        df = self.results['cell_pseudotime']

        # Bin pseudotime into intervals
        n_bins = 20
        df['pseudotime_bin'] = pd.cut(df['pseudotime'], bins=n_bins, labels=False)

        trajectory_results = []

        # Calculate marker frequencies per bin
        available_markers = ['PERK', 'AGFP', 'KI67']
        available_markers = [m for m in available_markers if m in df.columns]

        for sample in df['sample_id'].unique():
            sample_df = df[df['sample_id'] == sample]

            for bin_id in range(n_bins):
                bin_df = sample_df[sample_df['pseudotime_bin'] == bin_id]

                if len(bin_df) < 5:
                    continue

                result = {
                    'sample_id': sample,
                    'pseudotime_bin': bin_id,
                    'pseudotime_center': (bin_id + 0.5) / n_bins,
                    'n_cells': len(bin_df),
                    'timepoint': bin_df['timepoint'].iloc[0],
                    'group': bin_df['group'].iloc[0],
                    'main_group': bin_df['main_group'].iloc[0]
                }

                # Calculate marker frequencies
                for marker in available_markers:
                    if marker in bin_df.columns:
                        freq = bin_df[marker].sum() / len(bin_df) * 100
                        result[f'{marker}_percent'] = freq

                trajectory_results.append(result)

        if trajectory_results:
            df_traj = pd.DataFrame(trajectory_results)
            self.results['trajectory_dynamics'] = df_traj
            print(f"    ✓ Analyzed trajectory dynamics across {n_bins} pseudotime bins")

    def _save_results(self):
        """Save all results to files."""
        for name, df in self.results.items():
            output_path = self.output_dir / f'{name}.csv'
            df.to_csv(output_path, index=False)

        print(f"\n  ✓ Saved {len(self.results)} result datasets")
