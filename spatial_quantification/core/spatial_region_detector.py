"""
SpatialCells-based Region Detector
Uses SpatialCells library for robust tumor region identification and boundary detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings

try:
    import spatialcells as spc
    HAS_SPATIALCELLS = True
except ImportError:
    HAS_SPATIALCELLS = False
    warnings.warn("SpatialCells not available. Please install: pip install spatialcells")


class SpatialCellsRegionDetector:
    """
    Detect and characterize spatial regions using SpatialCells library.

    This provides superior region detection compared to raw DBSCAN by:
    - Creating proper geometric boundaries using alpha shapes
    - Providing accurate area and density calculations
    - Enabling distance-based infiltration analysis from boundaries
    - Supporting spatial heterogeneity analysis via sliding windows
    """

    def __init__(self, adata, config: Dict):
        """
        Initialize region detector.

        Parameters
        ----------
        adata : AnnData
            Annotated data with spatial coordinates in adata.obsm['spatial']
        config : dict
            Configuration dictionary with tumor_definition settings
        """
        if not HAS_SPATIALCELLS:
            raise ImportError("SpatialCells library required. Install with: pip install spatialcells")

        self.adata = adata
        self.config = config
        # Support both 'structure_definition' (generic) and 'tumor_definition' (legacy)
        self.tumor_config = config.get('structure_definition', config.get('tumor_definition', {}))

        # Prepare coordinate columns for SpatialCells (expects X_centroid, Y_centroid)
        if 'spatial' in self.adata.obsm:
            self.adata.obs['X_centroid'] = self.adata.obsm['spatial'][:, 0]
            self.adata.obs['Y_centroid'] = self.adata.obsm['spatial'][:, 1]

        # Storage for detected regions
        self.tumor_boundaries = {}  # sample -> {tumor_id: boundary}
        self.tumor_communities = {}  # sample -> community labels
        self.tumor_region_assignments = {}  # sample -> region assignments

    def detect_tumor_regions(self, sample: str = None,
                            eps: int = None,
                            min_samples: int = 20,
                            alpha: int = 100,
                            min_cluster_size: int = 50,
                            min_edges: int = 20,
                            holes_min_edges: int = 200) -> Dict:
        """
        Detect tumor regions using SpatialCells getCommunities and getBoundary.

        Parameters
        ----------
        sample : str, optional
            Sample ID to process. If None, processes all samples
        eps : int
            DBSCAN epsilon parameter for community detection
        min_samples : int
            Minimum samples for DBSCAN
        alpha : int
            Alpha parameter for alpha shape boundary creation
        min_cluster_size : int
            Minimum cells per tumor region
        min_edges : int
            Minimum edges for boundary polygon filtering
        holes_min_edges : int
            Minimum edges for holes in boundaries

        Returns
        -------
        dict
            Dictionary with detected regions per sample
        """
        # Get parameters from config if not specified
        if eps is None:
            eps = self.tumor_config.get('structure_detection', {}).get('eps', 100)

        tumor_pheno = self.tumor_config.get('base_phenotype', 'Tumor')
        tumor_col = f'is_{tumor_pheno}'

        if tumor_col not in self.adata.obs.columns:
            raise ValueError(f"Tumor phenotype column '{tumor_col}' not found")

        results = {}
        samples = [sample] if sample else self.adata.obs['sample_id'].unique()

        print(f"  Using SpatialCells with eps={eps}, alpha={alpha}, min_samples={min_samples}")

        for sample_id in samples:
            sample_mask = self.adata.obs['sample_id'] == sample_id
            sample_adata = self.adata[sample_mask].copy()

            # Get tumor cells for this sample
            tumor_mask = sample_adata.obs[tumor_col].values
            n_tumor_cells = tumor_mask.sum()

            if n_tumor_cells < min_samples:
                print(f"    ⚠ {sample_id}: Too few tumor cells ({n_tumor_cells}), skipping")
                continue

            # Use SpatialCells to detect communities
            markers_of_interest = [tumor_col]
            communitycolumn = f'{tumor_pheno}_community'

            community_results = spc.spatial.getCommunities(
                sample_adata,
                markers_of_interest,
                eps=eps,
                min_samples=min_samples,
                newcolumn=communitycolumn,
                core_only=False
            )

            if community_results is None:
                print(f"    ⚠ {sample_id}: No communities detected")
                continue

            labels_sorted, db = community_results

            # Filter by size and create boundaries for each community
            sample_boundaries = {}
            sample_regions = []
            valid_tumor_count = 0

            for cell_count, community_idx in labels_sorted:
                if cell_count < min_cluster_size:
                    continue

                try:
                    # Create boundary using alpha shapes
                    boundary = spc.spa.getBoundary(
                        sample_adata,
                        communitycolumn,
                        [community_idx],
                        alpha=alpha,
                        debug=False
                    )

                    # Prune small components
                    pruned_boundary = spc.spa.pruneSmallComponents(
                        boundary,
                        min_edges=min_edges,
                        holes_min_edges=holes_min_edges
                    )

                    # Calculate area to validate
                    area = spc.msmt.getRegionArea(pruned_boundary)

                    if area > 0:
                        sample_boundaries[community_idx] = pruned_boundary
                        sample_regions.append({
                            'sample_id': sample_id,
                            'tumor_id': community_idx,
                            'n_cells': cell_count,
                            'area_um2': area
                        })
                        valid_tumor_count += 1

                except Exception as e:
                    warnings.warn(f"Error creating boundary for community {community_idx}: {e}")
                    continue

            # Store results
            self.tumor_boundaries[sample_id] = sample_boundaries
            self.tumor_communities[sample_id] = sample_adata.obs[communitycolumn].values

            results[sample_id] = {
                'n_communities': len(labels_sorted),
                'n_valid_tumors': valid_tumor_count,
                'tumor_regions': sample_regions,
                'boundaries': sample_boundaries
            }

            print(f"    ✓ {sample_id}: Detected {valid_tumor_count} tumor regions from {len(labels_sorted)} communities")

        return results

    def assign_cells_to_regions(self, sample: str = None) -> None:
        """
        Assign cells to detected tumor regions using boundary geometries.

        Parameters
        ----------
        sample : str, optional
            Sample ID to process. If None, processes all samples with detected boundaries
        """
        samples = [sample] if sample else list(self.tumor_boundaries.keys())

        for sample_id in samples:
            if sample_id not in self.tumor_boundaries:
                continue

            sample_mask = self.adata.obs['sample_id'] == sample_id
            sample_adata = self.adata[sample_mask].copy()

            boundaries = self.tumor_boundaries[sample_id]
            if not boundaries:
                continue

            # Assign cells to each tumor region
            region_assignments = np.full(len(sample_adata), -1, dtype=int)

            for tumor_id, boundary in boundaries.items():
                region_name = f'tumor_{tumor_id}'

                # Use SpatialCells to assign points to this region
                temp_adata = sample_adata.copy()
                spc.spatial.assignPointsToRegions(
                    temp_adata,
                    [boundary],
                    [region_name],
                    assigncolumn='temp_region',
                    default='background'
                )

                # Extract assignments
                in_region = temp_adata.obs['temp_region'] == region_name
                region_assignments[in_region.values] = tumor_id

            # Store in main adata
            self.adata.obs.loc[sample_mask, 'tumor_region_id'] = region_assignments
            self.tumor_region_assignments[sample_id] = region_assignments

    def calculate_region_metrics(self, sample: str, tumor_id: int,
                                phenotype_cols: List[str] = None) -> Dict:
        """
        Calculate comprehensive metrics for a tumor region using SpatialCells measurements.

        Parameters
        ----------
        sample : str
            Sample ID
        tumor_id : int
            Tumor region ID
        phenotype_cols : list of str, optional
            Phenotype columns to analyze for composition

        Returns
        -------
        dict
            Metrics including area, density, composition
        """
        if sample not in self.tumor_boundaries:
            raise ValueError(f"No boundaries detected for sample {sample}")

        if tumor_id not in self.tumor_boundaries[sample]:
            raise ValueError(f"Tumor {tumor_id} not found in sample {sample}")

        boundary = self.tumor_boundaries[sample][tumor_id]

        sample_mask = self.adata.obs['sample_id'] == sample
        sample_adata = self.adata[sample_mask].copy()

        # Assign to region for SpatialCells functions
        region_name = f'tumor_{tumor_id}'
        spc.spatial.assignPointsToRegions(
            sample_adata,
            [boundary],
            [region_name],
            assigncolumn='region',
            default='background'
        )

        metrics = {}

        # Area
        metrics['area_um2'] = spc.msmt.getRegionArea(boundary)

        # Overall density
        density_result = spc.msmt.getRegionDensity(
            sample_adata,
            boundary,
            region_subset=[region_name]
        )
        metrics['cell_density'] = density_result.values[0] if len(density_result) > 0 else 0

        # Composition by phenotype
        if phenotype_cols:
            for pheno_col in phenotype_cols:
                if pheno_col in sample_adata.obs.columns:
                    comp = spc.msmt.getRegionComposition(
                        sample_adata,
                        pheno_col,
                        regions=[region_name]
                    )
                    metrics[f'{pheno_col}_composition'] = comp

        # Centroid
        centroid = spc.msmt.getRegionCentroid(boundary)
        metrics['centroid_x'] = centroid[0]
        metrics['centroid_y'] = centroid[1]

        return metrics

    def calculate_infiltration_distances(self, sample: str, tumor_id: int,
                                        immune_col: str,
                                        max_distance: float = 200) -> np.ndarray:
        """
        Calculate distances from immune cells to tumor boundary.

        Parameters
        ----------
        sample : str
            Sample ID
        tumor_id : int
            Tumor region ID
        immune_col : str
            Column name for immune cell phenotype
        max_distance : float
            Maximum distance to calculate (um)

        Returns
        -------
        np.ndarray
            Distances for each immune cell
        """
        if sample not in self.tumor_boundaries:
            raise ValueError(f"No boundaries for sample {sample}")

        boundary = self.tumor_boundaries[sample][tumor_id]

        sample_mask = self.adata.obs['sample_id'] == sample
        sample_adata = self.adata[sample_mask].copy()

        # Assign to region
        region_name = f'tumor_{tumor_id}'
        spc.spatial.assignPointsToRegions(
            sample_adata,
            [boundary],
            [region_name],
            assigncolumn='region',
            default='background'
        )

        # Calculate distances from tumor boundary
        spc.msmt.getDistanceFromObject(
            sample_adata,
            boundary,
            region_col='region',
            region_subset=None,  # All cells
            name='distance_from_tumor',
            inplace=True,
            binned=False
        )

        # Get immune cells and their distances
        immune_mask = sample_adata.obs[immune_col].values if immune_col in sample_adata.obs.columns else np.zeros(len(sample_adata), dtype=bool)
        distances = sample_adata.obs['distance_from_tumor'].values

        # Filter to immune cells within max distance
        immune_distances = distances[immune_mask]
        immune_distances = immune_distances[immune_distances <= max_distance]

        return immune_distances

    def analyze_spatial_heterogeneity(self, sample: str, tumor_id: int,
                                     phenotype_col: str,
                                     window_size: int = 300,
                                     step_size: int = 300,
                                     min_cells: int = 10) -> pd.DataFrame:
        """
        Analyze spatial heterogeneity using sliding window composition.

        Parameters
        ----------
        sample : str
            Sample ID
        tumor_id : int
            Tumor region ID
        phenotype_col : str
            Phenotype column to analyze
        window_size : int
            Size of sliding window in um
        step_size : int
            Step size for sliding window in um
        min_cells : int
            Minimum cells per window

        Returns
        -------
        pd.DataFrame
            Sliding window composition results
        """
        if sample not in self.tumor_boundaries:
            raise ValueError(f"No boundaries for sample {sample}")

        boundary = self.tumor_boundaries[sample][tumor_id]

        sample_mask = self.adata.obs['sample_id'] == sample
        sample_adata = self.adata[sample_mask].copy()

        # Assign to region
        region_name = f'tumor_{tumor_id}'
        spc.spatial.assignPointsToRegions(
            sample_adata,
            [boundary],
            [region_name],
            assigncolumn='region',
            default='background'
        )

        # Sliding window analysis
        comp_df = spc.msmt.getSlidingWindowsComposition(
            sample_adata,
            window_size=window_size,
            step_size=step_size,
            phenotype_col=phenotype_col,
            region_col='region',
            region_subset=[region_name],
            min_cells=min_cells
        )

        return comp_df

    def get_tumor_structures_for_legacy_analyses(self) -> Dict:
        """
        Convert SpatialCells region assignments to legacy format for compatibility.

        Returns
        -------
        dict
            Dictionary mapping sample_id to structure labels array
        """
        legacy_structures = {}

        for sample_id in self.tumor_region_assignments.keys():
            legacy_structures[sample_id] = self.tumor_region_assignments[sample_id]

        return legacy_structures

    def get_boundary(self, sample: str, tumor_id: int):
        """
        Get the Shapely boundary object for a tumor region.

        Parameters
        ----------
        sample : str
            Sample ID
        tumor_id : int
            Tumor region ID

        Returns
        -------
        shapely.geometry
            Boundary geometry
        """
        if sample not in self.tumor_boundaries:
            raise ValueError(f"No boundaries for sample {sample}")

        if tumor_id not in self.tumor_boundaries[sample]:
            raise ValueError(f"Tumor {tumor_id} not found in sample {sample}")

        return self.tumor_boundaries[sample][tumor_id]
