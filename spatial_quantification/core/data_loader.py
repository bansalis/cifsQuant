"""
Data Loader for Spatial Quantification
Loads gated h5ad file and metadata
"""

import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Load and validate spatial quantification data."""

    def __init__(self, config: Dict):
        """
        Initialize data loader with configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary from YAML
        """
        self.config = config
        self.adata = None
        self.metadata = None

    def load(self) -> tuple:
        """
        Load gated data and metadata.

        Returns
        -------
        tuple
            (adata, metadata) - AnnData object and metadata DataFrame
        """
        # Load h5ad file
        h5ad_path = Path(self.config['input']['gated_data'])
        if not h5ad_path.exists():
            raise FileNotFoundError(f"Gated data file not found: {h5ad_path}")

        print(f"\nLoading gated data from: {h5ad_path}")
        self.adata = ad.read_h5ad(h5ad_path)
        print(f"  ✓ Loaded {len(self.adata):,} cells")
        print(f"  ✓ Samples: {self.adata.obs['sample_id'].nunique()}")

        # Load metadata
        metadata_path = Path(self.config['input']['metadata'])
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        print(f"\nLoading metadata from: {metadata_path}")
        self.metadata = pd.read_csv(metadata_path)
        print(f"  ✓ Loaded metadata for {len(self.metadata)} samples")

        # Validate data
        self._validate_data()

        # Extract spatial coordinates
        self._extract_coordinates()

        return self.adata, self.metadata

    def _validate_data(self):
        """Validate required columns and data integrity."""
        # Check required metadata columns
        meta_config = self.config['metadata']
        required_cols = [
            meta_config['sample_column'],
            meta_config['group_column'],
            meta_config['timepoint_column']
        ]

        missing_cols = [col for col in required_cols if col not in self.metadata.columns]
        if missing_cols:
            raise ValueError(f"Missing required metadata columns: {missing_cols}")

        # Check sample_id exists in adata
        if 'sample_id' not in self.adata.obs.columns:
            raise ValueError("adata.obs must have 'sample_id' column")

        # Standardize sample IDs (uppercase)
        self.adata.obs['sample_id'] = self.adata.obs['sample_id'].str.upper()
        self.metadata[meta_config['sample_column']] = \
            self.metadata[meta_config['sample_column']].str.upper()

        # Check all samples in adata have metadata
        adata_samples = set(self.adata.obs['sample_id'].unique())
        metadata_samples = set(self.metadata[meta_config['sample_column']].unique())

        missing_metadata = adata_samples - metadata_samples
        if missing_metadata:
            warnings.warn(f"Samples in adata without metadata: {missing_metadata}")

        # Check for gated columns (is_* columns)
        gated_cols = [col for col in self.adata.obs.columns if col.startswith('is_')]

        if not gated_cols:
            # Try to load gates from manual_gating output
            print("\n  ℹ No is_* columns found, attempting to load from manual_gating layers...")
            self._apply_gates_from_layers()
            gated_cols = [col for col in self.adata.obs.columns if col.startswith('is_')]

            if not gated_cols:
                raise ValueError(
                    "No gated columns (is_*) found in adata. "
                    "Run manual_gating.py first or ensure gated_data.h5ad has 'gated' layer."
                )

        print(f"\n  ✓ Found {len(gated_cols)} gated cell populations:")
        for col in sorted(gated_cols):
            count = self.adata.obs[col].sum()
            pct = 100 * count / len(self.adata)
            print(f"    - {col}: {count:,} cells ({pct:.1f}%)")

    def _extract_coordinates(self):
        """Extract and validate spatial coordinates."""
        if 'spatial' in self.adata.obsm:
            coords = self.adata.obsm['spatial']
            print(f"\n  ✓ Spatial coordinates found in obsm['spatial']")
        elif 'X_centroid' in self.adata.obs and 'Y_centroid' in self.adata.obs:
            coords = np.column_stack([
                self.adata.obs['X_centroid'].values,
                self.adata.obs['Y_centroid'].values
            ])
            self.adata.obsm['spatial'] = coords
            print(f"\n  ✓ Extracted spatial coordinates from X_centroid, Y_centroid")
        else:
            raise ValueError("No spatial coordinates found in adata")

        # Validate coordinates
        if np.any(np.isnan(coords)):
            n_nan = np.sum(np.isnan(coords).any(axis=1))
            warnings.warn(f"{n_nan} cells have NaN coordinates and will be excluded")

            # Remove cells with NaN coordinates
            valid_mask = ~np.isnan(coords).any(axis=1)
            self.adata = self.adata[valid_mask, :].copy()
            print(f"  ✓ Removed {n_nan} cells with invalid coordinates")
            print(f"  ✓ Remaining cells: {len(self.adata):,}")

    def _apply_gates_from_layers(self):
        """
        Apply gates from manual_gating output.
        Converts adata.layers['gated'] to is_* columns in obs.
        """
        if 'gated' not in self.adata.layers:
            print("  ✗ No 'gated' layer found in adata")
            return

        print("  ✓ Found 'gated' layer from manual_gating")
        print("  Creating is_* columns from gates...")

        # The 'gated' layer has shape (n_cells, n_markers)
        # with 0/1 values for negative/positive
        gated_layer = self.adata.layers['gated']

        # Create is_* column for each marker
        for i, marker in enumerate(self.adata.var_names):
            col_name = f'is_{marker}'
            # Convert to boolean
            self.adata.obs[col_name] = gated_layer[:, i].astype(bool)

            n_positive = self.adata.obs[col_name].sum()
            pct = 100 * n_positive / len(self.adata)
            print(f"    - {col_name}: {n_positive:,} cells ({pct:.1f}%)")
