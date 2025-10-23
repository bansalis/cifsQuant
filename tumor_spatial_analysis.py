#!/usr/bin/env python3
"""
Comprehensive Spatial Analysis Framework for Tumor Immunology

This module provides a complete, publication-ready spatial analysis pipeline for
cyclic immunofluorescence data, focusing on tumor-immune interactions with support for:

1. Tumor and immune cell identification with customizable subpopulations
2. Tumor structure detection (aggregates/clusters)
3. Immune infiltration boundary detection and quantification
4. Temporal analysis of tumor size, marker expression, and immune infiltration
5. Co-enrichment analysis for immune-tumor proximity
6. Spatial heterogeneity detection in tumor regions
7. Comparative analysis across regions and timepoints
8. Publication-quality visualizations

Author: Automated generation for spatial tumor analysis
Date: 2025-10-22
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from scipy.spatial import distance_matrix, ConvexHull, Delaunay
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu, ks_2samp
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class TumorSpatialAnalysis:
    """
    Comprehensive spatial analysis for tumor immunology studies.

    This class provides a complete workflow for analyzing spatial relationships
    between tumor cells and immune populations, with temporal tracking and
    publication-ready visualizations.
    """

    def __init__(self, adata, tumor_markers: List[str], immune_markers: List[str],
                 output_dir: str = 'tumor_spatial_analysis'):
        """
        Initialize the spatial analysis framework.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix with spatial coordinates in adata.obsm['spatial']
            and phenotype information in adata.obs
        tumor_markers : list of str
            List of markers that define tumor cells (e.g., ['TOM', 'AGFP'])
        immune_markers : list of str
            List of markers that define immune cells (e.g., ['CD45', 'CD3', 'CD8B'])
        output_dir : str
            Directory for saving outputs
        """
        self.adata = adata
        self.tumor_markers = tumor_markers
        self.immune_markers = immune_markers
        self.output_dir = output_dir

        # Extract spatial coordinates
        if 'spatial' in adata.obsm:
            self.coords = adata.obsm['spatial']
        else:
            # Try to construct from X_centroid, Y_centroid
            if 'X_centroid' in adata.obs and 'Y_centroid' in adata.obs:
                self.coords = np.column_stack([adata.obs['X_centroid'].values,
                                               adata.obs['Y_centroid'].values])
                adata.obsm['spatial'] = self.coords
            else:
                raise ValueError("No spatial coordinates found in adata.obsm['spatial'] or adata.obs")

        # Storage for computed features
        self.tumor_structures = None
        self.infiltration_boundaries = None
        self.subpopulations = {}
        self.temporal_metrics = {}
        self.heterogeneity_regions = None

        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)
        os.makedirs(f"{output_dir}/data", exist_ok=True)

        print(f"Initialized TumorSpatialAnalysis with {len(adata)} cells")
        print(f"  Tumor markers: {tumor_markers}")
        print(f"  Immune markers: {immune_markers}")


    # ==================== SECTION 1: CELL IDENTIFICATION ====================

    def define_cell_populations(self, population_config: Dict[str, Dict[str, any]]):
        """
        Define tumor and immune cell populations based on marker combinations.

        Parameters
        ----------
        population_config : dict
            Configuration dictionary with population definitions.

            Example:
            {
                'Tumor': {
                    'markers': {'TOM': True},
                    'color': '#E41A1C'
                },
                'Tumor_AGFP+': {
                    'markers': {'TOM': True, 'AGFP': True},
                    'parent': 'Tumor',
                    'color': '#377EB8'
                },
                'CD8_T_cells': {
                    'markers': {'CD3': True, 'CD8B': True},
                    'color': '#4DAF4A'
                },
                'CD8_Ki67+': {
                    'markers': {'CD3': True, 'CD8B': True, 'KI67': True},
                    'parent': 'CD8_T_cells',
                    'color': '#984EA3'
                }
            }
        """
        print("\n" + "="*70)
        print("DEFINING CELL POPULATIONS")
        print("="*70)

        for pop_name, pop_config in population_config.items():
            marker_conditions = pop_config['markers']

            # Build boolean mask
            mask = np.ones(len(self.adata), dtype=bool)

            for marker, required_state in marker_conditions.items():
                # Check if marker exists in gated layer
                if marker not in self.adata.var_names:
                    print(f"  WARNING: Marker {marker} not found, skipping {pop_name}")
                    mask = np.zeros(len(self.adata), dtype=bool)
                    break

                marker_idx = self.adata.var_names.get_loc(marker)

                # Get marker positivity
                if 'gated' in self.adata.layers:
                    marker_positive = self.adata.layers['gated'][:, marker_idx] > 0
                else:
                    # Fallback: use raw values with percentile threshold
                    marker_values = self.adata.X[:, marker_idx]
                    threshold = np.percentile(marker_values[marker_values > 0], 90)
                    marker_positive = marker_values > threshold

                if required_state:
                    mask &= marker_positive
                else:
                    mask &= ~marker_positive

            # Store population
            self.adata.obs[f'is_{pop_name}'] = mask
            n_cells = mask.sum()
            pct = 100 * n_cells / len(self.adata)

            print(f"  {pop_name}: {n_cells:,} cells ({pct:.1f}%)")

            # Store in subpopulations
            self.subpopulations[pop_name] = {
                'mask': mask,
                'count': n_cells,
                'percentage': pct,
                'color': pop_config.get('color', '#999999'),
                'parent': pop_config.get('parent', None)
            }

        print("="*70)
        return self.subpopulations


    # ==================== SECTION 2: TUMOR STRUCTURE DETECTION ====================

    def detect_tumor_structures(self, min_cluster_size: int = 50,
                               eps: float = 30, min_samples: int = 10,
                               tumor_population: str = 'Tumor'):
        """
        Detect tumor structures as spatial aggregates/clusters of tumor cells.

        Uses DBSCAN clustering to identify contiguous tumor regions based on
        spatial proximity.

        Parameters
        ----------
        min_cluster_size : int
            Minimum number of cells to be considered a tumor structure
        eps : float
            Maximum distance (in microns) between cells in same cluster
        min_samples : int
            Minimum samples for DBSCAN core point
        tumor_population : str
            Name of tumor population to cluster

        Returns
        -------
        dict
            Dictionary with tumor structure information
        """
        print("\n" + "="*70)
        print("DETECTING TUMOR STRUCTURES")
        print("="*70)

        # Get tumor cell coordinates
        tumor_mask = self.adata.obs[f'is_{tumor_population}'].values
        tumor_coords = self.coords[tumor_mask]

        print(f"Clustering {tumor_mask.sum():,} {tumor_population} cells")
        print(f"  Parameters: eps={eps}μm, min_samples={min_samples}")

        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clustering.fit_predict(tumor_coords)

        # Filter out noise (-1) and small clusters
        unique_labels = set(labels)
        unique_labels.discard(-1)

        structures = {}
        valid_structure_id = 0

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_size = cluster_mask.sum()

            if cluster_size >= min_cluster_size:
                cluster_coords = tumor_coords[cluster_mask]

                # Calculate structure properties
                centroid = cluster_coords.mean(axis=0)
                area = self._calculate_cluster_area(cluster_coords)
                perimeter = self._calculate_cluster_perimeter(cluster_coords)
                compactness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

                structures[valid_structure_id] = {
                    'cell_indices': np.where(tumor_mask)[0][cluster_mask],
                    'size': cluster_size,
                    'centroid': centroid,
                    'area': area,
                    'perimeter': perimeter,
                    'compactness': compactness,
                    'coords': cluster_coords
                }

                valid_structure_id += 1

        print(f"  Detected {len(structures)} tumor structures")
        print(f"  Size range: {min([s['size'] for s in structures.values()])} - "
              f"{max([s['size'] for s in structures.values()])} cells")

        # Store results
        self.tumor_structures = structures

        # Add structure ID to adata.obs
        structure_ids = np.full(len(self.adata), -1, dtype=int)
        for struct_id, struct_info in structures.items():
            structure_ids[struct_info['cell_indices']] = struct_id

        self.adata.obs['tumor_structure_id'] = structure_ids

        print("="*70)
        return structures


    def _calculate_cluster_area(self, coords: np.ndarray) -> float:
        """
        Calculate area of cell cluster.

        For small clusters (<10k cells), use ConvexHull for accuracy.
        For large regions, use bounding box to avoid memory crashes.
        """
        if len(coords) < 3:
            return 0.0

        # For large regions, use bounding box instead of ConvexHull
        # ConvexHull on millions of points causes WSL crashes
        if len(coords) > 10000:
            # Use bounding box area as approximation
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            return (x_max - x_min) * (y_max - y_min)

        # For small clusters, ConvexHull is fine and more accurate
        try:
            hull = ConvexHull(coords)
            return hull.volume  # In 2D, volume is actually area
        except:
            # Fallback to bounding box if ConvexHull fails
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            return (x_max - x_min) * (y_max - y_min)


    def _calculate_cluster_perimeter(self, coords: np.ndarray) -> float:
        """
        Calculate perimeter of cell cluster.

        For small clusters (<10k cells), use ConvexHull for accuracy.
        For large regions, use bounding box to avoid memory crashes.
        """
        if len(coords) < 3:
            return 0.0

        # For large regions, use bounding box perimeter
        if len(coords) > 10000:
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            return 2 * ((x_max - x_min) + (y_max - y_min))

        # For small clusters, ConvexHull is fine
        try:
            hull = ConvexHull(coords)
            return hull.area  # In 2D, area is actually perimeter
        except:
            # Fallback to bounding box
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            return 2 * ((x_max - x_min) + (y_max - y_min))


    # ==================== SECTION 3: INFILTRATION BOUNDARY DETECTION ====================

    def define_infiltration_boundaries(self, boundary_widths: List[float] = [30, 100, 200],
                                      tumor_population: str = 'Tumor'):
        """
        Define infiltration boundaries around tumor structures at multiple scales.

        Creates concentric zones around tumor structures:
        - Tumor Core: Inside tumor structure
        - Tumor Margin: 0 - boundary_widths[0] μm
        - Peri-tumor: boundary_widths[0] - boundary_widths[1] μm
        - Distal: boundary_widths[1] - boundary_widths[2] μm
        - Far: > boundary_widths[2] μm

        Parameters
        ----------
        boundary_widths : list of float
            Boundary distances in microns
        tumor_population : str
            Name of tumor population

        Returns
        -------
        pd.DataFrame
            DataFrame with cell-level boundary assignments
        """
        print("\n" + "="*70)
        print("DEFINING INFILTRATION BOUNDARIES")
        print("="*70)

        if self.tumor_structures is None:
            print("  ERROR: No tumor structures detected. Run detect_tumor_structures() first.")
            return None

        # Calculate distance of each cell to nearest tumor structure
        tumor_mask = self.adata.obs[f'is_{tumor_population}'].values
        tumor_coords = self.coords[tumor_mask]
        all_coords = self.coords

        print(f"  Calculating distances to {len(self.tumor_structures)} tumor structures")
        print(f"  Using memory-efficient KD-Tree approach...")

        # For each cell, find distance to nearest tumor cell
        # Use KD-Tree with memory-efficient batching
        from scipy.spatial import cKDTree
        import gc

        # Convert to float32 to reduce memory usage (saves ~50% memory)
        tumor_coords_f32 = tumor_coords.astype(np.float32)
        all_coords_f32 = all_coords.astype(np.float32)

        # Build KD-Tree from tumor cell coordinates
        print(f"  Building KD-Tree from {len(tumor_coords_f32):,} tumor cells...")
        tree = cKDTree(tumor_coords_f32)

        # Free up original tumor coords to save memory
        del tumor_coords_f32
        gc.collect()

        # Query distances for all cells in batches (smaller batches for memory efficiency)
        distances_to_tumor = np.zeros(len(all_coords_f32), dtype=np.float32)
        batch_size = 50000  # Smaller batches to reduce memory pressure

        n_batches = (len(all_coords_f32) + batch_size - 1) // batch_size
        print(f"  Processing {len(all_coords_f32):,} cells in {n_batches} batches...")

        for i in range(0, len(all_coords_f32), batch_size):
            end_idx = min(i + batch_size, len(all_coords_f32))
            batch_coords = all_coords_f32[i:end_idx]

            # Query nearest tumor cell for each cell in batch
            dists, _ = tree.query(batch_coords, k=1, workers=1)  # workers=1 to control memory
            distances_to_tumor[i:end_idx] = dists

            # Free batch memory
            del batch_coords, dists

            # Progress reporting every batch
            batch_num = i // batch_size + 1
            progress = 100 * end_idx / len(all_coords_f32)
            print(f"    Batch {batch_num}/{n_batches} - Progress: {progress:.1f}% ({end_idx:,}/{len(all_coords_f32):,} cells)")

            # Periodic garbage collection to free memory
            if batch_num % 10 == 0:
                gc.collect()

        # Convert back to float64 for consistency
        distances_to_tumor = distances_to_tumor.astype(np.float64)

        # Cleanup
        del all_coords_f32, tree
        gc.collect()

        print(f"  Distance calculation complete!")

        # Assign boundary regions
        boundary_region = np.full(len(all_coords), 'Far', dtype=object)
        boundary_region[tumor_mask] = 'Tumor_Core'

        # Non-tumor cells
        non_tumor_mask = ~tumor_mask
        non_tumor_distances = distances_to_tumor[non_tumor_mask]

        # Create boundary labels
        region_labels_temp = np.full(non_tumor_mask.sum(), 'Far', dtype=object)
        region_labels_temp[non_tumor_distances <= boundary_widths[0]] = 'Tumor_Margin'
        region_labels_temp[(non_tumor_distances > boundary_widths[0]) &
                          (non_tumor_distances <= boundary_widths[1])] = 'Peri_Tumor'
        region_labels_temp[(non_tumor_distances > boundary_widths[1]) &
                          (non_tumor_distances <= boundary_widths[2])] = 'Distal'

        boundary_region[non_tumor_mask] = region_labels_temp

        # Store in adata
        self.adata.obs['boundary_region'] = boundary_region
        self.adata.obs['distance_to_tumor'] = distances_to_tumor

        # Summary statistics
        print(f"\n  Boundary Region Distribution:")
        for region in ['Tumor_Core', 'Tumor_Margin', 'Peri_Tumor', 'Distal', 'Far']:
            n = (boundary_region == region).sum()
            pct = 100 * n / len(boundary_region)
            print(f"    {region}: {n:,} cells ({pct:.1f}%)")

        # Create summary dataframe
        boundary_df = pd.DataFrame({
            'cell_id': np.arange(len(all_coords)),
            'boundary_region': boundary_region,
            'distance_to_tumor': distances_to_tumor,
            'X': all_coords[:, 0],
            'Y': all_coords[:, 1]
        })

        self.infiltration_boundaries = boundary_df

        print("="*70)
        return boundary_df


    # ==================== SECTION 4: INFILTRATION QUANTIFICATION ====================

    def quantify_immune_infiltration(self, immune_populations: List[str],
                                    by_sample: bool = True,
                                    by_timepoint: bool = False):
        """
        Quantify immune infiltration in each boundary region.

        Parameters
        ----------
        immune_populations : list of str
            List of immune population names to quantify
        by_sample : bool
            Calculate metrics per sample
        by_timepoint : bool
            Calculate metrics per timepoint (requires 'timepoint' column)

        Returns
        -------
        pd.DataFrame
            Infiltration metrics by region and population
        """
        print("\n" + "="*70)
        print("QUANTIFYING IMMUNE INFILTRATION")
        print("="*70)

        if self.infiltration_boundaries is None:
            print("  ERROR: Boundaries not defined. Run define_infiltration_boundaries() first.")
            return None

        results = []

        grouping_cols = []
        if by_sample and 'sample_id' in self.adata.obs:
            grouping_cols.append('sample_id')
        if by_timepoint and 'timepoint' in self.adata.obs:
            grouping_cols.append('timepoint')

        if not grouping_cols:
            grouping_cols = [None]  # Single global analysis

        # Get unique group combinations
        if grouping_cols[0] is None:
            groups = [(None,)]
        else:
            groups = self.adata.obs[grouping_cols].drop_duplicates().values

        for group_vals in groups:
            # Create group mask
            if group_vals[0] is None:
                group_mask = np.ones(len(self.adata), dtype=bool)
                group_label = 'All'
            else:
                group_mask = np.ones(len(self.adata), dtype=bool)
                group_dict = {}
                for i, col in enumerate(grouping_cols):
                    group_mask &= (self.adata.obs[col] == group_vals[i])
                    group_dict[col] = group_vals[i]
                group_label = '_'.join([str(v) for v in group_vals])

            # For each boundary region
            for region in ['Tumor_Core', 'Tumor_Margin', 'Peri_Tumor', 'Distal', 'Far']:
                region_mask = group_mask & (self.adata.obs['boundary_region'] == region)
                n_total = region_mask.sum()

                if n_total == 0:
                    continue

                # For each immune population
                for pop_name in immune_populations:
                    if f'is_{pop_name}' not in self.adata.obs:
                        continue

                    pop_mask = region_mask & self.adata.obs[f'is_{pop_name}']
                    n_pop = pop_mask.sum()
                    pct_pop = 100 * n_pop / n_total if n_total > 0 else 0

                    # Calculate density (cells per mm²)
                    if n_total > 0:
                        region_coords = self.coords[region_mask]
                        area_mm2 = self._calculate_cluster_area(region_coords) / 1e6  # μm² to mm²
                        density = n_pop / area_mm2 if area_mm2 > 0 else 0
                    else:
                        density = 0

                    result_dict = {
                        'region': region,
                        'population': pop_name,
                        'n_cells': n_pop,
                        'total_cells': n_total,
                        'percentage': pct_pop,
                        'density_per_mm2': density
                    }

                    # Add grouping information
                    if group_vals[0] is not None:
                        for i, col in enumerate(grouping_cols):
                            result_dict[col] = group_vals[i]

                    results.append(result_dict)

        infiltration_df = pd.DataFrame(results)

        # Print summary
        print(f"\n  Calculated infiltration for {len(immune_populations)} populations")
        print(f"  Across {len(infiltration_df['region'].unique())} regions")
        if by_sample:
            print(f"  Across {len(infiltration_df.get('sample_id', [None]).unique())} samples")

        # Save
        infiltration_df.to_csv(f"{self.output_dir}/data/immune_infiltration_metrics.csv", index=False)
        print(f"\n  Saved: {self.output_dir}/data/immune_infiltration_metrics.csv")

        print("="*70)
        return infiltration_df


    # ==================== SECTION 5: TEMPORAL ANALYSIS ====================

    def analyze_temporal_changes(self, timepoint_col: str = 'timepoint',
                                 populations: List[str] = None,
                                 marker_trends: List[str] = None):
        """
        Analyze temporal changes in tumor size, marker expression, and infiltration.

        Parameters
        ----------
        timepoint_col : str
            Column name in adata.obs containing timepoint information
        populations : list of str
            Cell populations to track over time
        marker_trends : list of str
            Markers to track expression changes over time

        Returns
        -------
        dict
            Dictionary with temporal metrics for each analysis type
        """
        print("\n" + "="*70)
        print("ANALYZING TEMPORAL CHANGES")
        print("="*70)

        if timepoint_col not in self.adata.obs:
            print(f"  ERROR: Column '{timepoint_col}' not found in adata.obs")
            return None

        temporal_results = {
            'tumor_size': [],
            'marker_expression': [],
            'infiltration': []
        }

        timepoints = sorted(self.adata.obs[timepoint_col].unique())
        print(f"  Analyzing {len(timepoints)} timepoints: {timepoints}")

        # 1. TUMOR SIZE CHANGES
        print("\n  1. Tumor Size Analysis")
        if self.tumor_structures is not None:
            for tp in timepoints:
                tp_mask = self.adata.obs[timepoint_col] == tp

                # Recompute structures for this timepoint
                tp_tumor_mask = tp_mask & self.adata.obs[f'is_{list(self.subpopulations.keys())[0]}']
                n_tumor_cells = tp_tumor_mask.sum()

                # Get structure information for this timepoint
                tp_structure_ids = self.adata.obs.loc[tp_mask, 'tumor_structure_id']
                n_structures = len(set(tp_structure_ids[tp_structure_ids >= 0]))

                if n_structures > 0:
                    # Calculate average structure size
                    structure_sizes = []
                    for struct_id in set(tp_structure_ids[tp_structure_ids >= 0]):
                        size = (tp_structure_ids == struct_id).sum()
                        structure_sizes.append(size)

                    avg_size = np.mean(structure_sizes)
                    total_area = sum([self.tumor_structures.get(int(sid), {}).get('area', 0)
                                     for sid in set(tp_structure_ids[tp_structure_ids >= 0])])
                else:
                    avg_size = 0
                    total_area = 0

                temporal_results['tumor_size'].append({
                    'timepoint': tp,
                    'n_tumor_cells': n_tumor_cells,
                    'n_structures': n_structures,
                    'avg_structure_size': avg_size,
                    'total_tumor_area': total_area
                })

                print(f"    {tp}: {n_tumor_cells:,} tumor cells in {n_structures} structures")

        # 2. MARKER EXPRESSION CHANGES
        if marker_trends:
            print("\n  2. Marker Expression Analysis")
            for marker in marker_trends:
                if marker not in self.adata.var_names:
                    continue

                marker_idx = self.adata.var_names.get_loc(marker)

                for tp in timepoints:
                    tp_mask = self.adata.obs[timepoint_col] == tp

                    # Get marker values for this timepoint
                    if 'gated' in self.adata.layers:
                        marker_values = self.adata.layers['gated'][tp_mask, marker_idx]
                        pct_positive = (marker_values > 0).mean() * 100
                    else:
                        marker_values = self.adata.X[tp_mask, marker_idx]
                        pct_positive = np.nan

                    # Calculate mean expression in positive cells
                    pos_values = marker_values[marker_values > 0]
                    mean_expr = pos_values.mean() if len(pos_values) > 0 else 0

                    temporal_results['marker_expression'].append({
                        'timepoint': tp,
                        'marker': marker,
                        'pct_positive': pct_positive,
                        'mean_expression': mean_expr,
                        'n_positive': len(pos_values)
                    })

                print(f"    {marker}: Tracked across timepoints")

        # 3. INFILTRATION CHANGES
        if populations:
            print("\n  3. Infiltration Analysis")
            for pop in populations:
                if f'is_{pop}' not in self.adata.obs:
                    continue

                for tp in timepoints:
                    tp_mask = self.adata.obs[timepoint_col] == tp

                    # For each boundary region
                    for region in ['Tumor_Margin', 'Peri_Tumor']:
                        region_mask = tp_mask & (self.adata.obs['boundary_region'] == region)
                        n_total = region_mask.sum()

                        if n_total == 0:
                            continue

                        pop_mask = region_mask & self.adata.obs[f'is_{pop}']
                        n_pop = pop_mask.sum()
                        pct = 100 * n_pop / n_total

                        temporal_results['infiltration'].append({
                            'timepoint': tp,
                            'population': pop,
                            'region': region,
                            'n_cells': n_pop,
                            'percentage': pct
                        })

                print(f"    {pop}: Tracked infiltration across timepoints")

        # Convert to DataFrames and save
        for key, data in temporal_results.items():
            if data:
                df = pd.DataFrame(data)
                temporal_results[key] = df
                df.to_csv(f"{self.output_dir}/data/temporal_{key}.csv", index=False)
                print(f"\n  Saved: {self.output_dir}/data/temporal_{key}.csv")

        self.temporal_metrics = temporal_results
        print("="*70)
        return temporal_results


    # ==================== SECTION 6: CO-ENRICHMENT ANALYSIS ====================

    def analyze_coenrichment(self, population_pairs: List[Tuple[str, str]],
                            radius: float = 50, n_permutations: int = 100):
        """
        Analyze spatial co-enrichment between population pairs.

        Calculates whether two cell types are found closer together than expected
        by chance, using permutation testing.

        Parameters
        ----------
        population_pairs : list of tuple
            Pairs of populations to test (e.g., [('CD8_T_cells', 'Tumor_AGFP+')])
        radius : float
            Search radius in microns
        n_permutations : int
            Number of permutations for statistical testing

        Returns
        -------
        pd.DataFrame
            Co-enrichment metrics with p-values
        """
        print("\n" + "="*70)
        print("ANALYZING SPATIAL CO-ENRICHMENT")
        print("="*70)

        results = []

        for pop_a, pop_b in population_pairs:
            print(f"\n  Analyzing: {pop_a} <-> {pop_b}")

            if f'is_{pop_a}' not in self.adata.obs or f'is_{pop_b}' not in self.adata.obs:
                print(f"    ERROR: Population not found, skipping")
                continue

            mask_a = self.adata.obs[f'is_{pop_a}'].values
            mask_b = self.adata.obs[f'is_{pop_b}'].values

            coords_a = self.coords[mask_a]
            coords_b = self.coords[mask_b]

            # Calculate observed co-enrichment
            observed_score = self._calculate_enrichment_score(coords_a, coords_b, radius)

            # Permutation test
            null_scores = []
            for perm in range(n_permutations):
                # Randomly shuffle population B coordinates
                shuffled_coords_b = coords_b[np.random.permutation(len(coords_b))]
                null_score = self._calculate_enrichment_score(coords_a, shuffled_coords_b, radius)
                null_scores.append(null_score)

            null_scores = np.array(null_scores)
            p_value = (null_scores >= observed_score).mean()

            # Calculate z-score
            z_score = (observed_score - null_scores.mean()) / (null_scores.std() + 1e-10)

            results.append({
                'population_a': pop_a,
                'population_b': pop_b,
                'enrichment_score': observed_score,
                'null_mean': null_scores.mean(),
                'null_std': null_scores.std(),
                'z_score': z_score,
                'p_value': p_value,
                'significant': p_value < 0.05
            })

            sig_str = "SIGNIFICANT" if p_value < 0.05 else "not significant"
            print(f"    Score: {observed_score:.3f}, Z: {z_score:.2f}, "
                  f"p={p_value:.4f} ({sig_str})")

        enrichment_df = pd.DataFrame(results)
        enrichment_df.to_csv(f"{self.output_dir}/data/coenrichment_analysis.csv", index=False)
        print(f"\n  Saved: {self.output_dir}/data/coenrichment_analysis.csv")

        print("="*70)
        return enrichment_df


    def _calculate_enrichment_score(self, coords_a: np.ndarray, coords_b: np.ndarray,
                                   radius: float) -> float:
        """Calculate enrichment score as average number of B cells within radius of A cells."""
        if len(coords_a) == 0 or len(coords_b) == 0:
            return 0.0

        # For each A cell, count B cells within radius
        dists = cdist(coords_a, coords_b, metric='euclidean')
        n_neighbors = (dists <= radius).sum(axis=1)

        return n_neighbors.mean()


    # ==================== SECTION 7: SPATIAL HETEROGENEITY ====================

    def detect_spatial_heterogeneity(self, tumor_population: str = 'Tumor',
                                     heterogeneity_markers: List[str] = None,
                                     n_regions: int = 3,
                                     min_region_size: int = 100):
        """
        Identify spatially distinct tumor regions based on marker expression.

        Uses k-means clustering on spatial coordinates weighted by marker expression
        to identify tumor regions with distinct molecular profiles.

        Parameters
        ----------
        tumor_population : str
            Name of tumor population
        heterogeneity_markers : list of str
            Markers defining heterogeneity (e.g., ['AGFP', 'PERK', 'KI67'])
        n_regions : int
            Number of distinct regions to identify
        min_region_size : int
            Minimum cells per region

        Returns
        -------
        pd.DataFrame
            Region assignments and marker profiles
        """
        print("\n" + "="*70)
        print("DETECTING SPATIAL HETEROGENEITY")
        print("="*70)

        if heterogeneity_markers is None:
            heterogeneity_markers = self.tumor_markers

        print(f"  Using markers: {heterogeneity_markers}")

        # Get tumor cells
        tumor_mask = self.adata.obs[f'is_{tumor_population}'].values
        tumor_coords = self.coords[tumor_mask]

        # Extract marker expression for tumor cells
        marker_data = []
        for marker in heterogeneity_markers:
            if marker not in self.adata.var_names:
                continue
            marker_idx = self.adata.var_names.get_loc(marker)

            if 'gated' in self.adata.layers:
                marker_vals = self.adata.layers['gated'][tumor_mask, marker_idx]
            else:
                marker_vals = self.adata.X[tumor_mask, marker_idx]

            marker_data.append(marker_vals)

        marker_matrix = np.column_stack(marker_data)

        # Combine spatial coordinates with marker expression
        # Weight spatial coordinates to balance with marker expression
        coords_scaled = tumor_coords / tumor_coords.std(axis=0)
        markers_scaled = StandardScaler().fit_transform(marker_matrix)

        # Combined feature matrix (spatial + molecular)
        combined_features = np.hstack([coords_scaled * 0.3, markers_scaled * 0.7])

        # K-means clustering
        kmeans = KMeans(n_clusters=n_regions, random_state=42, n_init=50)
        region_labels = kmeans.fit_predict(combined_features)

        # Filter small regions
        valid_regions = []
        for region_id in range(n_regions):
            region_size = (region_labels == region_id).sum()
            if region_size >= min_region_size:
                valid_regions.append(region_id)

        print(f"\n  Identified {len(valid_regions)} valid regions (min size: {min_region_size})")

        # Remap to valid regions
        region_map = {old_id: new_id for new_id, old_id in enumerate(valid_regions)}
        region_labels_filtered = np.array([region_map.get(r, -1) for r in region_labels])

        # Add to adata
        heterogeneity_regions = np.full(len(self.adata), -1, dtype=int)
        heterogeneity_regions[tumor_mask] = region_labels_filtered
        self.adata.obs['heterogeneity_region'] = heterogeneity_regions

        # Calculate region profiles
        region_profiles = []
        for region_id in valid_regions:
            region_mask_local = region_labels == region_id
            region_size = region_mask_local.sum()

            profile = {
                'region_id': region_map[region_id],
                'n_cells': region_size,
                'pct_of_tumor': 100 * region_size / len(tumor_coords)
            }

            # Calculate mean marker expression
            for i, marker in enumerate(heterogeneity_markers):
                if i < marker_matrix.shape[1]:
                    mean_expr = marker_matrix[region_mask_local, i].mean()
                    pct_positive = (marker_matrix[region_mask_local, i] > 0).mean() * 100
                    profile[f'{marker}_mean'] = mean_expr
                    profile[f'{marker}_pct_pos'] = pct_positive

            region_profiles.append(profile)

            print(f"    Region {region_map[region_id]}: {region_size:,} cells ({profile['pct_of_tumor']:.1f}%)")

        heterogeneity_df = pd.DataFrame(region_profiles)
        heterogeneity_df.to_csv(f"{self.output_dir}/data/heterogeneity_regions.csv", index=False)
        print(f"\n  Saved: {self.output_dir}/data/heterogeneity_regions.csv")

        self.heterogeneity_regions = heterogeneity_df

        print("="*70)
        return heterogeneity_df


    def compare_region_infiltration(self, immune_populations: List[str],
                                   region_col: str = 'heterogeneity_region'):
        """
        Compare immune infiltration across tumor heterogeneity regions.

        Parameters
        ----------
        immune_populations : list of str
            Immune populations to compare
        region_col : str
            Column containing region assignments

        Returns
        -------
        pd.DataFrame
            Comparative infiltration metrics by region
        """
        print("\n" + "="*70)
        print("COMPARING INFILTRATION ACROSS TUMOR REGIONS")
        print("="*70)

        if region_col not in self.adata.obs:
            print(f"  ERROR: Column '{region_col}' not found")
            return None

        results = []

        valid_regions = [r for r in self.adata.obs[region_col].unique() if r >= 0]

        for region_id in valid_regions:
            # Get cells in this tumor region
            region_tumor_mask = self.adata.obs[region_col] == region_id

            # Get nearby cells (within infiltration boundary)
            region_cell_indices = np.where(region_tumor_mask)[0]

            if len(region_cell_indices) == 0:
                continue

            # For infiltration, look at cells near this region
            region_coords = self.coords[region_tumor_mask]

            # Define "near" as within 50μm of this region
            dists_to_region = cdist(self.coords, region_coords, metric='euclidean').min(axis=1)
            near_region_mask = dists_to_region <= 50

            # Exclude the tumor cells themselves
            near_region_mask &= ~region_tumor_mask

            n_near = near_region_mask.sum()

            if n_near == 0:
                continue

            # Quantify each immune population
            for pop in immune_populations:
                if f'is_{pop}' not in self.adata.obs:
                    continue

                pop_mask = near_region_mask & self.adata.obs[f'is_{pop}']
                n_pop = pop_mask.sum()
                pct = 100 * n_pop / n_near if n_near > 0 else 0

                results.append({
                    'region_id': region_id,
                    'population': pop,
                    'n_cells': n_pop,
                    'total_near': n_near,
                    'pct_infiltration': pct
                })

        infiltration_comparison = pd.DataFrame(results)

        # Statistical comparison
        print(f"\n  Comparing {len(immune_populations)} populations across {len(valid_regions)} regions")

        infiltration_comparison.to_csv(
            f"{self.output_dir}/data/region_infiltration_comparison.csv", index=False)
        print(f"  Saved: {self.output_dir}/data/region_infiltration_comparison.csv")

        print("="*70)
        return infiltration_comparison


    # ==================== SECTION 8: PUBLICATION-READY VISUALIZATIONS ====================

    def plot_spatial_overview(self, figsize=(20, 5), save=True):
        """
        Create comprehensive spatial overview figure.

        4-panel figure:
        1. All cells with population colors
        2. Tumor structures with boundaries
        3. Immune infiltration heatmap
        4. Heterogeneity regions
        """
        fig, axes = plt.subplots(1, 4, figsize=figsize, dpi=300)

        # Panel 1: Cell populations
        ax = axes[0]
        for pop_name, pop_info in self.subpopulations.items():
            mask = pop_info['mask']
            if mask.sum() > 0:
                ax.scatter(self.coords[mask, 0], self.coords[mask, 1],
                          c=pop_info['color'], s=0.5, alpha=0.6, label=pop_name)
        ax.set_title('Cell Populations', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.legend(markerscale=5, frameon=True, fontsize=8)
        ax.set_aspect('equal')

        # Panel 2: Tumor structures and boundaries
        ax = axes[1]
        if self.tumor_structures is not None:
            # Plot boundaries
            boundary_colors = {
                'Tumor_Core': '#E41A1C',
                'Tumor_Margin': '#377EB8',
                'Peri_Tumor': '#4DAF4A',
                'Distal': '#984EA3',
                'Far': '#CCCCCC'
            }

            for region, color in boundary_colors.items():
                mask = self.adata.obs['boundary_region'] == region
                if mask.sum() > 0:
                    ax.scatter(self.coords[mask, 0], self.coords[mask, 1],
                             c=color, s=0.5, alpha=0.4, label=region)

        ax.set_title('Tumor Structures & Boundaries', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.legend(markerscale=5, frameon=True, fontsize=8)
        ax.set_aspect('equal')

        # Panel 3: Immune infiltration density
        ax = axes[2]
        # Create 2D histogram of immune cell density
        immune_mask = np.zeros(len(self.adata), dtype=bool)
        for marker in self.immune_markers:
            if f'is_CD45' in self.adata.obs:  # Use pan-immune as proxy
                immune_mask = self.adata.obs['is_CD45'].values if 'is_CD45' in self.adata.obs else immune_mask
                break

        if immune_mask.sum() > 0:
            immune_coords = self.coords[immune_mask]

            # Create hexbin plot
            hb = ax.hexbin(immune_coords[:, 0], immune_coords[:, 1],
                          gridsize=50, cmap='YlOrRd', mincnt=1)
            plt.colorbar(hb, ax=ax, label='Cell density')

        ax.set_title('Immune Cell Density', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_aspect('equal')

        # Panel 4: Heterogeneity regions
        ax = axes[3]
        if 'heterogeneity_region' in self.adata.obs:
            regions = self.adata.obs['heterogeneity_region'].values
            valid_regions = regions >= 0

            if valid_regions.sum() > 0:
                scatter = ax.scatter(self.coords[valid_regions, 0],
                                   self.coords[valid_regions, 1],
                                   c=regions[valid_regions],
                                   s=0.5, alpha=0.7, cmap='tab10')
                plt.colorbar(scatter, ax=ax, label='Region ID')

        ax.set_title('Tumor Heterogeneity', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_aspect('equal')

        plt.tight_layout()

        if save:
            plt.savefig(f"{self.output_dir}/figures/spatial_overview.png", dpi=300, bbox_inches='tight')
            print(f"Saved: {self.output_dir}/figures/spatial_overview.png")

        return fig


    def plot_temporal_trends(self, save=True):
        """Create temporal trend visualizations."""
        if not self.temporal_metrics:
            print("No temporal metrics available. Run analyze_temporal_changes() first.")
            return None

        n_plots = sum([1 for v in self.temporal_metrics.values() if len(v) > 0])
        if n_plots == 0:
            return None

        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5), dpi=300)
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Tumor size trends
        if len(self.temporal_metrics.get('tumor_size', [])) > 0:
            df = self.temporal_metrics['tumor_size']
            ax = axes[plot_idx]

            ax.plot(df['timepoint'], df['n_tumor_cells'], marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Timepoint', fontsize=12)
            ax.set_ylabel('Number of Tumor Cells', fontsize=12)
            ax.set_title('Tumor Growth Over Time', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            plot_idx += 1

        # Marker expression trends
        if len(self.temporal_metrics.get('marker_expression', [])) > 0:
            df = self.temporal_metrics['marker_expression']
            ax = axes[plot_idx]

            for marker in df['marker'].unique():
                marker_df = df[df['marker'] == marker]
                ax.plot(marker_df['timepoint'], marker_df['pct_positive'],
                       marker='o', label=marker, linewidth=2, markersize=8)

            ax.set_xlabel('Timepoint', fontsize=12)
            ax.set_ylabel('% Positive Cells', fontsize=12)
            ax.set_title('Marker Expression Over Time', fontsize=14, fontweight='bold')
            ax.legend(frameon=True)
            ax.grid(True, alpha=0.3)

            plot_idx += 1

        # Infiltration trends
        if len(self.temporal_metrics.get('infiltration', [])) > 0:
            df = self.temporal_metrics['infiltration']
            ax = axes[plot_idx]

            for pop in df['population'].unique():
                for region in df['region'].unique():
                    subset = df[(df['population'] == pop) & (df['region'] == region)]
                    if len(subset) > 0:
                        ax.plot(subset['timepoint'], subset['percentage'],
                               marker='o', label=f'{pop} ({region})', linewidth=2, markersize=8)

            ax.set_xlabel('Timepoint', fontsize=12)
            ax.set_ylabel('% Infiltration', fontsize=12)
            ax.set_title('Immune Infiltration Over Time', fontsize=14, fontweight='bold')
            ax.legend(frameon=True, fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(f"{self.output_dir}/figures/temporal_trends.png", dpi=300, bbox_inches='tight')
            print(f"Saved: {self.output_dir}/figures/temporal_trends.png")

        return fig


    def plot_infiltration_heatmap(self, infiltration_df: pd.DataFrame, save=True):
        """Create heatmap of infiltration across regions and populations."""
        if infiltration_df is None or len(infiltration_df) == 0:
            return None

        # Pivot to create matrix
        pivot_df = infiltration_df.pivot_table(
            values='percentage',
            index='population',
            columns='region',
            aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='YlOrRd',
                   cbar_kws={'label': '% of Region'},
                   linewidths=0.5, ax=ax)

        ax.set_title('Immune Infiltration by Region', fontsize=14, fontweight='bold')
        ax.set_xlabel('Boundary Region', fontsize=12)
        ax.set_ylabel('Cell Population', fontsize=12)

        plt.tight_layout()

        if save:
            plt.savefig(f"{self.output_dir}/figures/infiltration_heatmap.png", dpi=300, bbox_inches='tight')
            print(f"Saved: {self.output_dir}/figures/infiltration_heatmap.png")

        return fig


    def generate_comprehensive_report(self):
        """Generate a complete analysis report with all metrics and figures."""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*70)

        report_lines = []
        report_lines.append("# Comprehensive Tumor Spatial Analysis Report")
        report_lines.append(f"\nGenerated: {pd.Timestamp.now()}")
        report_lines.append(f"\nDataset: {len(self.adata)} cells")

        # Cell populations
        report_lines.append("\n## Cell Population Summary")
        for pop_name, pop_info in self.subpopulations.items():
            report_lines.append(f"- **{pop_name}**: {pop_info['count']:,} cells ({pop_info['percentage']:.1f}%)")

        # Tumor structures
        if self.tumor_structures:
            report_lines.append("\n## Tumor Structures")
            report_lines.append(f"- Total structures detected: {len(self.tumor_structures)}")
            sizes = [s['size'] for s in self.tumor_structures.values()]
            report_lines.append(f"- Size range: {min(sizes)} - {max(sizes)} cells")
            report_lines.append(f"- Mean size: {np.mean(sizes):.1f} cells")

        # Infiltration boundaries
        if self.infiltration_boundaries is not None:
            report_lines.append("\n## Infiltration Boundaries")
            for region in ['Tumor_Core', 'Tumor_Margin', 'Peri_Tumor', 'Distal', 'Far']:
                n = (self.adata.obs['boundary_region'] == region).sum()
                pct = 100 * n / len(self.adata)
                report_lines.append(f"- **{region}**: {n:,} cells ({pct:.1f}%)")

        # Heterogeneity
        if self.heterogeneity_regions is not None:
            report_lines.append("\n## Spatial Heterogeneity")
            report_lines.append(f"- Distinct tumor regions: {len(self.heterogeneity_regions)}")

        # Save report
        report_path = f"{self.output_dir}/analysis_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"\n  Saved: {report_path}")
        print("="*70)

        return '\n'.join(report_lines)


def main():
    """Example usage of the TumorSpatialAnalysis framework."""
    import scanpy as sc

    # Load data
    print("Loading data...")
    adata = sc.read_h5ad('path/to/gated_data.h5ad')

    # Initialize analysis
    tsa = TumorSpatialAnalysis(
        adata,
        tumor_markers=['TOM', 'AGFP'],
        immune_markers=['CD45', 'CD3', 'CD8B'],
        output_dir='tumor_spatial_analysis'
    )

    # Define cell populations
    population_config = {
        'Tumor': {'markers': {'TOM': True}, 'color': '#E41A1C'},
        'Tumor_AGFP+': {'markers': {'TOM': True, 'AGFP': True}, 'parent': 'Tumor', 'color': '#377EB8'},
        'Tumor_PERK+': {'markers': {'TOM': True, 'PERK': True}, 'parent': 'Tumor', 'color': '#4DAF4A'},
        'CD45': {'markers': {'CD45': True}, 'color': '#984EA3'},
        'CD3_T_cells': {'markers': {'CD3': True}, 'color': '#FF7F00'},
        'CD8_T_cells': {'markers': {'CD3': True, 'CD8B': True}, 'parent': 'CD3_T_cells', 'color': '#FFFF33'},
        'CD8_Ki67+': {'markers': {'CD3': True, 'CD8B': True, 'KI67': True}, 'parent': 'CD8_T_cells', 'color': '#A65628'}
    }

    tsa.define_cell_populations(population_config)

    # Detect tumor structures
    tsa.detect_tumor_structures(min_cluster_size=50, eps=30)

    # Define infiltration boundaries
    tsa.define_infiltration_boundaries(boundary_widths=[30, 100, 200])

    # Quantify infiltration
    immune_pops = ['CD3_T_cells', 'CD8_T_cells', 'CD8_Ki67+']
    infiltration_df = tsa.quantify_immune_infiltration(immune_pops, by_sample=True)

    # Temporal analysis (if timepoint data available)
    if 'timepoint' in adata.obs:
        tsa.analyze_temporal_changes(
            timepoint_col='timepoint',
            populations=immune_pops,
            marker_trends=['AGFP', 'PERK', 'KI67']
        )

    # Co-enrichment analysis
    tsa.analyze_coenrichment([
        ('CD8_T_cells', 'Tumor_AGFP+'),
        ('CD8_Ki67+', 'Tumor_PERK+')
    ])

    # Spatial heterogeneity
    tsa.detect_spatial_heterogeneity(
        heterogeneity_markers=['AGFP', 'PERK', 'KI67'],
        n_regions=3
    )

    tsa.compare_region_infiltration(immune_pops)

    # Generate visualizations
    tsa.plot_spatial_overview()
    tsa.plot_temporal_trends()
    tsa.plot_infiltration_heatmap(infiltration_df)

    # Generate report
    tsa.generate_comprehensive_report()

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
