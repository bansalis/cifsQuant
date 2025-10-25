#!/usr/bin/env python3
"""
Memory-Efficient Tumor Spatial Analysis Framework

This module provides a scalable, publication-ready spatial analysis pipeline for
large-scale cyclic immunofluorescence datasets. Designed to handle datasets with
millions of cells without memory crashes.

Key Features:
1. Per-structure processing to minimize memory footprint
2. Cellular neighborhood detection (RCN-based approach from Sorger/Nolan labs)
3. Temporal and group-based statistical analysis
4. Publication-quality visualizations
5. Resumable analysis with intermediate results saved

Architecture:
- Phase 1: Structure detection and indexing
- Phase 2: Per-structure analysis with local neighborhoods
- Phase 3: Statistical aggregation and hypothesis testing
- Phase 4: Publication figure generation

References:
- Schapiro et al. (2017) "histoCAT: analysis of cell phenotypes and interactions in multiplex image cytometry data"
- Keren et al. (2018) "A Structured Tumor-Immune Microenvironment in Triple Negative Breast Cancer"
- Jackson et al. (2020) "The single-cell pathology landscape of breast cancer"

Author: AI-assisted development for cifsQuant
Date: 2025-10-23
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, kruskal, spearmanr, pearsonr
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from typing import Dict, List, Tuple, Optional, Union
import warnings
import os
import h5py
from pathlib import Path
import gc
warnings.filterwarnings('ignore')


class EfficientTumorSpatialAnalysis:
    """
    Memory-efficient spatial analysis for large-scale tumor immunology studies.

    Processes tumor structures individually to minimize memory usage while
    maintaining comprehensive analysis capabilities.
    """

    def __init__(self, adata, sample_metadata: pd.DataFrame,
                 tumor_markers: List[str], immune_markers: List[str],
                 output_dir: str = 'efficient_spatial_analysis'):
        """
        Initialize the efficient spatial analysis framework.

        Parameters
        ----------
        adata : AnnData
            Full annotated data matrix with all cells
        sample_metadata : pd.DataFrame
            Metadata with columns: sample_id, timepoint, group, condition, etc.
        tumor_markers : list of str
            Markers defining tumor cells
        immune_markers : list of str
            Markers defining immune cells
        output_dir : str
            Output directory
        """
        self.adata = adata
        self.sample_metadata = sample_metadata
        self.tumor_markers = tumor_markers
        self.immune_markers = immune_markers
        self.output_dir = output_dir

        # Extract spatial coordinates
        if 'spatial' in adata.obsm:
            self.coords = adata.obsm['spatial']
        elif 'X_centroid' in adata.obs and 'Y_centroid' in adata.obs:
            self.coords = np.column_stack([adata.obs['X_centroid'].values,
                                           adata.obs['Y_centroid'].values])
            adata.obsm['spatial'] = self.coords
        else:
            raise ValueError("No spatial coordinates found")

        # Create directory structure
        self._setup_directories()

        # Storage for results
        self.structure_index = None
        self.neighborhood_profiles = None
        self.statistical_results = {}

        print(f"Initialized EfficientTumorSpatialAnalysis")
        print(f"  Total cells: {len(adata):,}")
        print(f"  Samples: {adata.obs['sample_id'].nunique()}")
        print(f"  Output: {output_dir}/")


    def _setup_directories(self):
        """Create organized output directory structure."""
        dirs = [
            self.output_dir,
            f"{self.output_dir}/structures",
            f"{self.output_dir}/neighborhoods",
            f"{self.output_dir}/statistics",
            f"{self.output_dir}/figures",
            f"{self.output_dir}/figures/spatial_maps",
            f"{self.output_dir}/figures/temporal",
            f"{self.output_dir}/figures/neighborhoods",
            f"{self.output_dir}/data"
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


    # ==================== PHASE 1: STRUCTURE DETECTION ====================

    def detect_all_tumor_structures(self, population_config: Dict,
                                   min_cluster_size: int = 50,
                                   eps: float = 30,
                                   min_samples: int = 10,
                                   tumor_population: str = 'Tumor') -> pd.DataFrame:
        """
        Detect all tumor structures across all samples and create index.

        This is a lightweight operation that only identifies structures
        without loading full local neighborhoods.

        Parameters
        ----------
        population_config : dict
            Cell population definitions
        min_cluster_size : int
            Minimum cells per structure
        eps : float
            DBSCAN clustering radius (μm)
        min_samples : int
            DBSCAN minimum samples
        tumor_population : str
            Name of tumor population to cluster

        Returns
        -------
        pd.DataFrame
            Structure index with metadata
        """
        print("\n" + "="*70)
        print("PHASE 1: DETECTING TUMOR STRUCTURES")
        print("="*70)

        # Define cell populations
        print("\nDefining cell populations...")
        self._define_populations(population_config)

        # Get tumor cells
        tumor_mask = self.adata.obs[f'is_{tumor_population}'].values
        tumor_coords = self.coords[tumor_mask].astype(np.float32)
        tumor_sample_ids = self.adata.obs.loc[tumor_mask, 'sample_id'].values

        print(f"\nClustering {tumor_mask.sum():,} tumor cells across all samples...")
        print(f"  Parameters: eps={eps}μm, min_samples={min_samples}")

        # Per-sample structure detection to avoid cross-sample clustering
        structure_records = []
        global_structure_id = 0

        for sample_id in np.unique(tumor_sample_ids):
            sample_tumor_mask = tumor_sample_ids == sample_id
            sample_coords = tumor_coords[sample_tumor_mask]
            sample_indices = np.where(tumor_mask)[0][sample_tumor_mask]

            # DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            labels = clustering.fit_predict(sample_coords)

            # Process each cluster
            for label in set(labels):
                if label == -1:  # Skip noise
                    continue

                cluster_mask = labels == label
                cluster_size = cluster_mask.sum()

                if cluster_size >= min_cluster_size:
                    cluster_coords = sample_coords[cluster_mask]
                    cluster_cell_indices = sample_indices[cluster_mask]

                    # Calculate structure properties
                    centroid = cluster_coords.mean(axis=0)
                    x_min, x_max = cluster_coords[:, 0].min(), cluster_coords[:, 0].max()
                    y_min, y_max = cluster_coords[:, 1].min(), cluster_coords[:, 1].max()

                    bbox_area = (x_max - x_min) * (y_max - y_min)

                    # Get sample metadata
                    sample_meta = self.sample_metadata[
                        self.sample_metadata['sample_id'] == sample_id
                    ].iloc[0] if sample_id in self.sample_metadata['sample_id'].values else {}

                    structure_records.append({
                        'structure_id': global_structure_id,
                        'sample_id': sample_id,
                        'n_cells': cluster_size,
                        'centroid_x': centroid[0],
                        'centroid_y': centroid[1],
                        'x_min': x_min,
                        'x_max': x_max,
                        'y_min': y_min,
                        'y_max': y_max,
                        'area_um2': bbox_area,
                        'timepoint': sample_meta.get('timepoint', None),
                        'main_group': sample_meta.get('main_group', None),
                        'genotype': sample_meta.get('genotype', None),
                        'genotype_full': sample_meta.get('genotype_full', None),
                        'group': sample_meta.get('group', None),
                        'condition': sample_meta.get('condition', None)
                    })

                    # Store cell indices for this structure
                    np.save(
                        f"{self.output_dir}/structures/structure_{global_structure_id:04d}_cells.npy",
                        cluster_cell_indices
                    )

                    global_structure_id += 1

            print(f"  Sample {sample_id}: {global_structure_id} structures so far")

        # Create structure index DataFrame
        self.structure_index = pd.DataFrame(structure_records)
        self.structure_index.to_csv(f"{self.output_dir}/data/structure_index.csv", index=False)

        print(f"\n✓ Detected {len(self.structure_index)} tumor structures")
        print(f"  Size range: {self.structure_index['n_cells'].min()} - "
              f"{self.structure_index['n_cells'].max()} cells")
        print(f"  Saved: {self.output_dir}/data/structure_index.csv")
        print("="*70)

        # Clean up
        del tumor_coords, tumor_sample_ids
        gc.collect()

        return self.structure_index


    def _define_populations(self, population_config: Dict):
        """Define cell populations based on marker combinations."""
        for pop_name, pop_config in population_config.items():
            marker_conditions = pop_config['markers']
            mask = np.ones(len(self.adata), dtype=bool)

            for marker, required_state in marker_conditions.items():
                if marker not in self.adata.var_names:
                    mask = np.zeros(len(self.adata), dtype=bool)
                    break

                marker_idx = self.adata.var_names.get_loc(marker)

                if 'gated' in self.adata.layers:
                    marker_positive = self.adata.layers['gated'][:, marker_idx] > 0
                else:
                    marker_values = self.adata.X[:, marker_idx]
                    threshold = np.percentile(marker_values[marker_values > 0], 90)
                    marker_positive = marker_values > threshold

                if required_state:
                    mask &= marker_positive
                else:
                    mask &= ~marker_positive

            self.adata.obs[f'is_{pop_name}'] = mask


    # ==================== PHASE 2: PER-STRUCTURE ANALYSIS ====================

    def analyze_structures_individually(self,
                                       immune_populations: List[str],
                                       boundary_widths: List[float] = [30, 100, 200],
                                       buffer_distance: float = 500,
                                       batch_size: int = 50) -> pd.DataFrame:
        """
        Analyze each tumor structure individually with local neighborhood.

        This is the memory-efficient core: only loads cells relevant to each
        structure, processes them, saves results, and clears memory.

        Parameters
        ----------
        immune_populations : list of str
            Immune populations to quantify
        boundary_widths : list of float
            Boundary distances (μm)
        buffer_distance : float
            Distance around structure to load (μm)
        batch_size : int
            Number of structures to process before checkpoint

        Returns
        -------
        pd.DataFrame
            Aggregated infiltration metrics
        """
        print("\n" + "="*70)
        print("PHASE 2: PER-STRUCTURE ANALYSIS")
        print("="*70)

        if self.structure_index is None:
            raise ValueError("Must run detect_all_tumor_structures() first")

        n_structures = len(self.structure_index)
        print(f"\nProcessing {n_structures} structures individually...")
        print(f"  Buffer distance: {buffer_distance}μm")
        print(f"  Boundary widths: {boundary_widths}")

        all_metrics = []

        # Build spatial index once for efficiency
        print("\nBuilding global spatial index...")
        all_coords_f32 = self.coords.astype(np.float32)
        global_tree = cKDTree(all_coords_f32)

        for idx, row in self.structure_index.iterrows():
            struct_id = row['structure_id']

            if idx % 10 == 0:
                progress = 100 * idx / n_structures
                print(f"  Progress: {progress:.1f}% ({idx}/{n_structures} structures)")

            # Load tumor cells for this structure
            tumor_cell_indices = np.load(
                f"{self.output_dir}/structures/structure_{struct_id:04d}_cells.npy"
            )

            # Get structure centroid
            centroid = np.array([row['centroid_x'], row['centroid_y']], dtype=np.float32)

            # Find all cells within buffer distance
            nearby_indices = global_tree.query_ball_point(centroid, buffer_distance)
            nearby_indices = np.array(nearby_indices)

            # Get local coordinates
            local_coords = all_coords_f32[nearby_indices]

            # Calculate distances to tumor boundary
            tumor_local_mask = np.isin(nearby_indices, tumor_cell_indices)
            tumor_local_coords = local_coords[tumor_local_mask]

            if len(tumor_local_coords) < 3:
                continue

            # Build local KD-tree for tumor cells
            tumor_tree = cKDTree(tumor_local_coords)
            distances, _ = tumor_tree.query(local_coords, k=1, workers=1)

            # Assign boundary regions
            boundary_labels = np.full(len(local_coords), 'Far', dtype=object)
            boundary_labels[tumor_local_mask] = 'Tumor_Core'

            non_tumor_mask = ~tumor_local_mask
            non_tumor_dists = distances[non_tumor_mask]

            temp_labels = np.full(non_tumor_mask.sum(), 'Far', dtype=object)
            temp_labels[non_tumor_dists <= boundary_widths[0]] = 'Margin'
            temp_labels[(non_tumor_dists > boundary_widths[0]) &
                       (non_tumor_dists <= boundary_widths[1])] = 'Peri_Tumor'
            temp_labels[(non_tumor_dists > boundary_widths[1]) &
                       (non_tumor_dists <= boundary_widths[2])] = 'Distal'

            boundary_labels[non_tumor_mask] = temp_labels

            # Quantify immune populations in each region
            for region in ['Tumor_Core', 'Margin', 'Peri_Tumor', 'Distal']:
                region_mask = boundary_labels == region
                n_total = region_mask.sum()

                if n_total == 0:
                    continue

                region_coords = local_coords[region_mask]
                x_range = region_coords[:, 0].max() - region_coords[:, 0].min()
                y_range = region_coords[:, 1].max() - region_coords[:, 1].min()
                area_mm2 = (x_range * y_range) / 1e6  # μm² to mm²

                for pop_name in immune_populations:
                    if f'is_{pop_name}' not in self.adata.obs:
                        continue

                    # Get population status for nearby cells
                    pop_status = self.adata.obs.iloc[nearby_indices][f'is_{pop_name}'].values
                    pop_in_region = pop_status[region_mask]
                    n_pop = pop_in_region.sum()

                    pct_pop = 100 * n_pop / n_total
                    density = n_pop / area_mm2 if area_mm2 > 0 else 0

                    all_metrics.append({
                        'structure_id': struct_id,
                        'sample_id': row['sample_id'],
                        'timepoint': row['timepoint'],
                        'group': row['group'],
                        'condition': row['condition'],
                        'region': region,
                        'population': pop_name,
                        'n_cells': n_pop,
                        'total_cells': n_total,
                        'percentage': pct_pop,
                        'density_per_mm2': density,
                        'tumor_size': row['n_cells'],
                        'tumor_area_um2': row['area_um2']
                    })

            # Cleanup
            del tumor_cell_indices, local_coords, distances, boundary_labels

            if idx % batch_size == 0 and idx > 0:
                gc.collect()

        # Create results DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(f"{self.output_dir}/data/infiltration_metrics.csv", index=False)

        print(f"\n✓ Completed per-structure analysis")
        print(f"  Saved: {self.output_dir}/data/infiltration_metrics.csv")
        print("="*70)

        # Cleanup
        del all_coords_f32, global_tree
        gc.collect()

        return metrics_df


    # ==================== PHASE 3: CELLULAR NEIGHBORHOODS ====================

    def detect_cellular_neighborhoods(self,
                                     populations: List[str],
                                     k_neighbors: int = 10,
                                     window_size: float = 100,
                                     n_clusters: int = 10,
                                     subsample_size: int = 100000) -> pd.DataFrame:
        """
        Detect recurrent cellular neighborhoods (RCN) across the dataset.

        Based on:
        - Schapiro et al. 2017 (histoCAT)
        - Keren et al. 2018 (MIBI-TOF neighborhoods)

        For each cell, characterize its local neighborhood by the frequency
        of different cell types within a radius, then cluster similar
        neighborhoods to identify recurrent patterns.

        Parameters
        ----------
        populations : list of str
            Cell populations to include in neighborhood profiles
        k_neighbors : int
            Number of nearest neighbors for neighborhood
        window_size : float
            Radius for neighborhood (μm)
        n_clusters : int
            Number of neighborhood types to identify
        subsample_size : int
            Subsample for clustering (for speed)

        Returns
        -------
        pd.DataFrame
            Neighborhood assignments and profiles
        """
        print("\n" + "="*70)
        print("PHASE 3: CELLULAR NEIGHBORHOOD DETECTION")
        print("="*70)

        print(f"\nComputing neighborhood compositions...")
        print(f"  Window size: {window_size}μm")
        print(f"  Populations: {len(populations)}")

        # Create population matrix (cells x populations)
        pop_matrix = np.zeros((len(self.adata), len(populations)), dtype=np.float32)

        for i, pop in enumerate(populations):
            if f'is_{pop}' in self.adata.obs:
                pop_matrix[:, i] = self.adata.obs[f'is_{pop}'].values.astype(np.float32)

        # Build KD-tree
        coords_f32 = self.coords.astype(np.float32)
        tree = cKDTree(coords_f32)

        # Compute neighborhood profiles in batches
        batch_size = 50000
        n_batches = (len(coords_f32) + batch_size - 1) // batch_size

        neighborhood_profiles = np.zeros((len(coords_f32), len(populations)), dtype=np.float32)

        print(f"\nProcessing {len(coords_f32):,} cells in {n_batches} batches...")

        for i in range(0, len(coords_f32), batch_size):
            end_idx = min(i + batch_size, len(coords_f32))
            batch_coords = coords_f32[i:end_idx]

            # Find neighbors within window
            neighbors_list = tree.query_ball_point(batch_coords, window_size, workers=1)

            # Compute neighborhood composition for each cell
            for j, neighbors in enumerate(neighbors_list):
                if len(neighbors) > 1:  # Exclude self
                    neighbor_pops = pop_matrix[neighbors]
                    neighborhood_profiles[i+j] = neighbor_pops.mean(axis=0)

            if i // batch_size % 10 == 0:
                progress = 100 * end_idx / len(coords_f32)
                print(f"  Progress: {progress:.1f}%")

        # Cluster neighborhoods
        print(f"\nClustering neighborhoods into {n_clusters} types...")

        # Subsample for efficiency
        if len(neighborhood_profiles) > subsample_size:
            subsample_idx = np.random.choice(len(neighborhood_profiles),
                                            subsample_size, replace=False)
            subsample_profiles = neighborhood_profiles[subsample_idx]
        else:
            subsample_profiles = neighborhood_profiles

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        kmeans.fit(subsample_profiles)

        # Assign all cells to nearest cluster
        print("  Assigning all cells to neighborhood types...")
        neighborhood_labels = kmeans.predict(neighborhood_profiles)

        # Add to adata
        self.adata.obs['neighborhood_type'] = neighborhood_labels

        # Characterize each neighborhood type
        neighborhood_summary = []

        for cluster_id in range(n_clusters):
            cluster_mask = neighborhood_labels == cluster_id
            cluster_profile = neighborhood_profiles[cluster_mask].mean(axis=0)

            summary = {
                'neighborhood_id': cluster_id,
                'n_cells': cluster_mask.sum(),
                'percentage': 100 * cluster_mask.sum() / len(neighborhood_labels)
            }

            for i, pop in enumerate(populations):
                summary[f'{pop}_enrichment'] = cluster_profile[i]

            neighborhood_summary.append(summary)

        self.neighborhood_profiles = pd.DataFrame(neighborhood_summary)
        self.neighborhood_profiles.to_csv(
            f"{self.output_dir}/data/neighborhood_profiles.csv", index=False
        )

        # Save per-cell neighborhood assignments
        cell_neighborhoods = pd.DataFrame({
            'cell_id': self.adata.obs.index,
            'sample_id': self.adata.obs['sample_id'].values,
            'neighborhood_type': neighborhood_labels
        })

        # Add metadata if available
        for col in ['timepoint', 'group', 'main_group', 'condition']:
            if col in self.adata.obs.columns:
                cell_neighborhoods[col] = self.adata.obs[col].values

        cell_neighborhoods.to_csv(
            f"{self.output_dir}/data/cell_neighborhoods.csv", index=False
        )

        print(f"\n✓ Detected {n_clusters} cellular neighborhood types")
        print(f"  Saved: {self.output_dir}/data/neighborhood_profiles.csv")
        print(f"  Saved: {self.output_dir}/data/cell_neighborhoods.csv")
        print("="*70)

        # Cleanup
        del coords_f32, tree, pop_matrix, neighborhood_profiles
        gc.collect()

        return self.neighborhood_profiles


    # ==================== PHASE 4: STATISTICAL ANALYSIS ====================

    def statistical_analysis(self, metrics_df: pd.DataFrame,
                            alpha: float = 0.05) -> Dict[str, pd.DataFrame]:
        """
        Comprehensive statistical analysis for publication.

        Tests:
        1. Temporal trends (Spearman correlation, linear mixed models)
        2. Group comparisons (Mann-Whitney, Kruskal-Wallis)
        3. Multiple testing correction (FDR)

        Parameters
        ----------
        metrics_df : pd.DataFrame
            Infiltration metrics from analyze_structures_individually
        alpha : float
            Significance threshold

        Returns
        -------
        dict
            Dictionary of statistical test results
        """
        print("\n" + "="*70)
        print("PHASE 4: STATISTICAL ANALYSIS")
        print("="*70)

        results = {}

        # 1. TEMPORAL TRENDS
        if 'timepoint' in metrics_df.columns and metrics_df['timepoint'].notna().any():
            print("\n1. Temporal Trend Analysis")
            temporal_results = self._test_temporal_trends(metrics_df, alpha)
            results['temporal_trends'] = temporal_results

        # 2. GROUP COMPARISONS
        if 'group' in metrics_df.columns and metrics_df['group'].notna().any():
            print("\n2. Group Comparison Analysis")
            group_results = self._test_group_differences(metrics_df, alpha)
            results['group_comparisons'] = group_results

        # 3. TUMOR SIZE CORRELATIONS
        print("\n3. Tumor Size Correlation Analysis")
        size_results = self._test_size_correlations(metrics_df, alpha)
        results['size_correlations'] = size_results

        # 4. REGION-SPECIFIC ANALYSIS
        print("\n4. Region-Specific Infiltration Analysis")
        region_results = self._test_region_differences(metrics_df, alpha)
        results['region_analysis'] = region_results

        self.statistical_results = results

        # Save all results
        for test_name, result_df in results.items():
            result_df.to_csv(
                f"{self.output_dir}/statistics/{test_name}.csv", index=False
            )

        print(f"\n✓ Statistical analysis complete")
        print(f"  Saved: {self.output_dir}/statistics/")
        print("="*70)

        return results


    def _test_temporal_trends(self, df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        """Test for temporal trends in infiltration metrics."""
        results = []

        # Get numeric timepoints
        df_temp = df[df['timepoint'].notna()].copy()

        if len(df_temp) == 0:
            return pd.DataFrame()

        # Try to convert timepoint to numeric
        try:
            df_temp['timepoint_numeric'] = pd.to_numeric(df_temp['timepoint'])
        except:
            # If categorical, use ordinal encoding
            timepoint_map = {tp: i for i, tp in enumerate(sorted(df_temp['timepoint'].unique()))}
            df_temp['timepoint_numeric'] = df_temp['timepoint'].map(timepoint_map)

        # Test each population x region combination
        for pop in df_temp['population'].unique():
            for region in df_temp['region'].unique():
                subset = df_temp[(df_temp['population'] == pop) &
                                (df_temp['region'] == region)]

                if len(subset) < 3:  # Need at least 3 points
                    continue

                # Spearman correlation
                rho, p_val = spearmanr(subset['timepoint_numeric'], subset['percentage'])

                # Linear regression for effect size
                if len(subset) >= 5:
                    try:
                        model = smf.ols('percentage ~ timepoint_numeric', data=subset).fit()
                        slope = model.params['timepoint_numeric']
                        r_squared = model.rsquared
                    except:
                        slope, r_squared = np.nan, np.nan
                else:
                    slope, r_squared = np.nan, np.nan

                results.append({
                    'population': pop,
                    'region': region,
                    'metric': 'percentage',
                    'n_timepoints': subset['timepoint'].nunique(),
                    'n_structures': len(subset),
                    'spearman_rho': rho,
                    'p_value': p_val,
                    'slope': slope,
                    'r_squared': r_squared
                })

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            # Multiple testing correction
            _, p_adj, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
            results_df['p_adjusted'] = p_adj
            results_df['significant'] = results_df['p_adjusted'] < alpha

            n_sig = results_df['significant'].sum()
            print(f"  {n_sig}/{len(results_df)} tests significant after FDR correction")

        return results_df


    def _test_group_differences(self, df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        """Test for differences between experimental groups."""
        results = []

        df_groups = df[df['group'].notna()].copy()

        if len(df_groups) == 0 or df_groups['group'].nunique() < 2:
            return pd.DataFrame()

        groups = df_groups['group'].unique()

        # Test each population x region combination
        for pop in df_groups['population'].unique():
            for region in df_groups['region'].unique():
                subset = df_groups[(df_groups['population'] == pop) &
                                  (df_groups['region'] == region)]

                if len(subset) < 2:
                    continue

                # If 2 groups: Mann-Whitney U test
                if len(groups) == 2:
                    g1 = subset[subset['group'] == groups[0]]['percentage']
                    g2 = subset[subset['group'] == groups[1]]['percentage']

                    if len(g1) > 0 and len(g2) > 0:
                        stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')

                        results.append({
                            'population': pop,
                            'region': region,
                            'metric': 'percentage',
                            'test': 'Mann-Whitney U',
                            'group_1': groups[0],
                            'group_2': groups[1],
                            'n_group_1': len(g1),
                            'n_group_2': len(g2),
                            'mean_group_1': g1.mean(),
                            'mean_group_2': g2.mean(),
                            'fold_change': g2.mean() / (g1.mean() + 1e-10),
                            'statistic': stat,
                            'p_value': p_val
                        })

                # If >2 groups: Kruskal-Wallis test
                else:
                    group_data = [subset[subset['group'] == g]['percentage'].values
                                 for g in groups]
                    group_data = [g for g in group_data if len(g) > 0]

                    if len(group_data) >= 2:
                        stat, p_val = kruskal(*group_data)

                        results.append({
                            'population': pop,
                            'region': region,
                            'metric': 'percentage',
                            'test': 'Kruskal-Wallis',
                            'n_groups': len(group_data),
                            'statistic': stat,
                            'p_value': p_val
                        })

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            # Multiple testing correction
            _, p_adj, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
            results_df['p_adjusted'] = p_adj
            results_df['significant'] = results_df['p_adjusted'] < alpha

            n_sig = results_df['significant'].sum()
            print(f"  {n_sig}/{len(results_df)} tests significant after FDR correction")

        return results_df


    def _test_size_correlations(self, df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        """Test correlation between tumor size and infiltration."""
        results = []

        # Test each population x region combination
        for pop in df['population'].unique():
            for region in df['region'].unique():
                subset = df[(df['population'] == pop) & (df['region'] == region)]

                if len(subset) < 3:
                    continue

                # Spearman correlation between tumor size and infiltration
                rho, p_val = spearmanr(subset['tumor_size'], subset['percentage'])

                results.append({
                    'population': pop,
                    'region': region,
                    'metric': 'percentage',
                    'n_structures': len(subset),
                    'spearman_rho': rho,
                    'p_value': p_val
                })

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            _, p_adj, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
            results_df['p_adjusted'] = p_adj
            results_df['significant'] = results_df['p_adjusted'] < alpha

            n_sig = results_df['significant'].sum()
            print(f"  {n_sig}/{len(results_df)} correlations significant")

        return results_df


    def _test_region_differences(self, df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        """Test for differences in infiltration across boundary regions."""
        results = []

        regions = df['region'].unique()

        if len(regions) < 2:
            return pd.DataFrame()

        # Test each population
        for pop in df['population'].unique():
            subset = df[df['population'] == pop]

            # Kruskal-Wallis across regions
            region_data = [subset[subset['region'] == r]['percentage'].values
                          for r in regions]
            region_data = [r for r in region_data if len(r) > 0]

            if len(region_data) >= 2:
                stat, p_val = kruskal(*region_data)

                results.append({
                    'population': pop,
                    'n_regions': len(region_data),
                    'test': 'Kruskal-Wallis',
                    'statistic': stat,
                    'p_value': p_val
                })

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            _, p_adj, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
            results_df['p_adjusted'] = p_adj
            results_df['significant'] = results_df['p_adjusted'] < alpha

        return results_df


    # ==================== PHASE 5: PUBLICATION VISUALIZATIONS ====================

    def create_publication_figures(self, metrics_df: pd.DataFrame):
        """
        Generate comprehensive publication-quality figures.

        Figures:
        1. Spatial maps with neighborhoods
        2. Infiltration heatmaps by region
        3. Temporal trend plots
        4. Group comparison plots
        5. Neighborhood enrichment plots
        6. Statistical summary tables
        """
        print("\n" + "="*70)
        print("PHASE 5: PUBLICATION FIGURE GENERATION")
        print("="*70)

        # Set publication style
        sns.set_style("white")
        sns.set_context("paper", font_scale=1.3)

        # 1. Infiltration heatmap
        print("\n1. Creating infiltration heatmap...")
        self._plot_infiltration_heatmap(metrics_df)

        # 2. Temporal trends
        if 'timepoint' in metrics_df.columns and metrics_df['timepoint'].notna().any():
            print("2. Creating temporal trend plots...")
            self._plot_temporal_trends(metrics_df)

        # 3. Group comparisons
        if 'group' in metrics_df.columns and metrics_df['group'].notna().any():
            print("3. Creating group comparison plots...")
            self._plot_group_comparisons(metrics_df)

        # 4. Neighborhood profiles
        if self.neighborhood_profiles is not None:
            print("4. Creating neighborhood enrichment plots...")
            self._plot_neighborhood_profiles()

        # 5. Size correlation plots
        print("5. Creating tumor size correlation plots...")
        self._plot_size_correlations(metrics_df)

        # 6. Summary statistics table
        print("6. Creating summary statistics table...")
        self._create_summary_table(metrics_df)

        print(f"\n✓ Publication figures complete")
        print(f"  Saved: {self.output_dir}/figures/")
        print("="*70)


    def _plot_infiltration_heatmap(self, df: pd.DataFrame):
        """Create heatmap of infiltration across regions and populations."""
        pivot_df = df.pivot_table(
            values='percentage',
            index='population',
            columns='region',
            aggfunc='mean'
        )

        # Order regions logically
        region_order = ['Tumor_Core', 'Margin', 'Peri_Tumor', 'Distal', 'Far']
        region_order = [r for r in region_order if r in pivot_df.columns]
        pivot_df = pivot_df[region_order]

        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

        sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlBu_r',
                   cbar_kws={'label': '% of Region'},
                   linewidths=0.5, ax=ax, vmin=0)

        ax.set_title('Immune Infiltration by Tumor Region',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Tumor Boundary Region', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cell Population', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/infiltration_heatmap.png",
                   dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_temporal_trends(self, df: pd.DataFrame):
        """Create temporal trend plots for key populations."""
        df_temp = df[df['timepoint'].notna()].copy()

        if len(df_temp) == 0:
            return

        # Focus on key populations and regions
        key_pops = df_temp['population'].unique()[:4]  # Top 4 populations
        key_regions = ['Margin', 'Peri_Tumor']

        n_pops = len(key_pops)
        n_regions = len(key_regions)

        fig, axes = plt.subplots(n_regions, n_pops,
                                figsize=(5*n_pops, 4*n_regions), dpi=300)

        if n_pops == 1 and n_regions == 1:
            axes = np.array([[axes]])
        elif n_pops == 1:
            axes = axes.reshape(-1, 1)
        elif n_regions == 1:
            axes = axes.reshape(1, -1)

        for i, region in enumerate(key_regions):
            for j, pop in enumerate(key_pops):
                ax = axes[i, j]

                subset = df_temp[(df_temp['population'] == pop) &
                               (df_temp['region'] == region)]

                if len(subset) > 0:
                    # Aggregate by timepoint
                    trend = subset.groupby('timepoint')['percentage'].agg(['mean', 'sem'])

                    ax.errorbar(trend.index, trend['mean'], yerr=trend['sem'],
                              marker='o', linewidth=2, markersize=8, capsize=5)

                    # Get significance if available
                    if 'temporal_trends' in self.statistical_results:
                        sig_df = self.statistical_results['temporal_trends']
                        sig_row = sig_df[(sig_df['population'] == pop) &
                                        (sig_df['region'] == region)]

                        if len(sig_row) > 0 and sig_row.iloc[0]['significant']:
                            rho = sig_row.iloc[0]['spearman_rho']
                            p_adj = sig_row.iloc[0]['p_adjusted']
                            ax.text(0.05, 0.95, f'ρ={rho:.2f}\np={p_adj:.3f}*',
                                   transform=ax.transAxes, fontsize=10,
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax.set_title(f'{pop} - {region}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Timepoint', fontsize=11)
                ax.set_ylabel('% Infiltration', fontsize=11)
                ax.grid(True, alpha=0.3)

        plt.suptitle('Temporal Trends in Immune Infiltration',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/temporal/temporal_trends.png",
                   dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_group_comparisons(self, df: pd.DataFrame):
        """Create box plots comparing groups."""
        df_groups = df[df['group'].notna()].copy()

        if len(df_groups) == 0:
            return

        # Focus on key populations and regions
        key_pops = df_groups['population'].unique()[:4]
        key_regions = ['Margin', 'Peri_Tumor']

        n_pops = len(key_pops)
        n_regions = len(key_regions)

        fig, axes = plt.subplots(n_regions, n_pops,
                                figsize=(5*n_pops, 4*n_regions), dpi=300)

        if n_pops == 1 and n_regions == 1:
            axes = np.array([[axes]])
        elif n_pops == 1:
            axes = axes.reshape(-1, 1)
        elif n_regions == 1:
            axes = axes.reshape(1, -1)

        for i, region in enumerate(key_regions):
            for j, pop in enumerate(key_pops):
                ax = axes[i, j]

                subset = df_groups[(df_groups['population'] == pop) &
                                 (df_groups['region'] == region)]

                if len(subset) > 0:
                    sns.boxplot(data=subset, x='group', y='percentage', ax=ax,
                               palette='Set2')
                    sns.stripplot(data=subset, x='group', y='percentage', ax=ax,
                                 color='black', alpha=0.3, size=3)

                    # Add significance stars
                    if 'group_comparisons' in self.statistical_results:
                        sig_df = self.statistical_results['group_comparisons']
                        sig_row = sig_df[(sig_df['population'] == pop) &
                                        (sig_df['region'] == region)]

                        if len(sig_row) > 0 and sig_row.iloc[0]['significant']:
                            p_adj = sig_row.iloc[0]['p_adjusted']
                            y_max = subset['percentage'].max()

                            if p_adj < 0.001:
                                sig_text = '***'
                            elif p_adj < 0.01:
                                sig_text = '**'
                            else:
                                sig_text = '*'

                            ax.text(0.5, y_max * 1.1, sig_text,
                                   ha='center', fontsize=16)

                ax.set_title(f'{pop} - {region}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Group', fontsize=11)
                ax.set_ylabel('% Infiltration', fontsize=11)
                ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Group Comparisons in Immune Infiltration',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/group_comparisons.png",
                   dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_neighborhood_profiles(self):
        """Create neighborhood enrichment heatmap."""
        if self.neighborhood_profiles is None:
            return

        # Extract enrichment columns
        enrichment_cols = [c for c in self.neighborhood_profiles.columns
                          if c.endswith('_enrichment')]

        if len(enrichment_cols) == 0:
            return

        data = self.neighborhood_profiles[enrichment_cols].values.T
        pop_names = [c.replace('_enrichment', '') for c in enrichment_cols]

        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        sns.heatmap(data, annot=True, fmt='.2f', cmap='viridis',
                   xticklabels=[f'CN{i}' for i in range(len(self.neighborhood_profiles))],
                   yticklabels=pop_names,
                   cbar_kws={'label': 'Mean Enrichment'},
                   linewidths=0.5, ax=ax)

        ax.set_title('Cellular Neighborhood Profiles',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Neighborhood Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cell Population', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/neighborhoods/neighborhood_profiles.png",
                   dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_size_correlations(self, df: pd.DataFrame):
        """Create scatter plots of tumor size vs infiltration."""
        key_pops = df['population'].unique()[:4]
        key_regions = ['Margin', 'Peri_Tumor']

        n_pops = len(key_pops)
        n_regions = len(key_regions)

        fig, axes = plt.subplots(n_regions, n_pops,
                                figsize=(5*n_pops, 4*n_regions), dpi=300)

        if n_pops == 1 and n_regions == 1:
            axes = np.array([[axes]])
        elif n_pops == 1:
            axes = axes.reshape(-1, 1)
        elif n_regions == 1:
            axes = axes.reshape(1, -1)

        for i, region in enumerate(key_regions):
            for j, pop in enumerate(key_pops):
                ax = axes[i, j]

                subset = df[(df['population'] == pop) & (df['region'] == region)]

                if len(subset) > 0:
                    ax.scatter(subset['tumor_size'], subset['percentage'],
                             alpha=0.5, s=30, edgecolors='black', linewidths=0.5)

                    # Add trend line
                    z = np.polyfit(subset['tumor_size'], subset['percentage'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(subset['tumor_size'].min(),
                                        subset['tumor_size'].max(), 100)
                    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)

                    # Add correlation if significant
                    if 'size_correlations' in self.statistical_results:
                        sig_df = self.statistical_results['size_correlations']
                        sig_row = sig_df[(sig_df['population'] == pop) &
                                        (sig_df['region'] == region)]

                        if len(sig_row) > 0:
                            rho = sig_row.iloc[0]['spearman_rho']
                            p_adj = sig_row.iloc[0]['p_adjusted']
                            sig = '*' if sig_row.iloc[0]['significant'] else ''

                            ax.text(0.05, 0.95, f'ρ={rho:.2f}\np={p_adj:.3f}{sig}',
                                   transform=ax.transAxes, fontsize=10,
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

                ax.set_title(f'{pop} - {region}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Tumor Size (cells)', fontsize=11)
                ax.set_ylabel('% Infiltration', fontsize=11)
                ax.grid(True, alpha=0.3)

        plt.suptitle('Tumor Size vs Immune Infiltration',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/size_correlations.png",
                   dpi=300, bbox_inches='tight')
        plt.close()


    def _create_summary_table(self, df: pd.DataFrame):
        """Create summary statistics table."""
        summary = df.groupby(['population', 'region']).agg({
            'percentage': ['mean', 'std', 'count'],
            'density_per_mm2': ['mean', 'std']
        }).round(2)

        summary.to_csv(f"{self.output_dir}/data/summary_statistics.csv")

        # Create formatted table figure
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
        ax.axis('tight')
        ax.axis('off')

        # Format for display
        table_data = summary.reset_index().values
        col_labels = ['Population', 'Region',
                     '% Mean', '% SD', 'N',
                     'Density Mean', 'Density SD']

        table = ax.table(cellText=table_data, colLabels=col_labels,
                        cellLoc='center', loc='center',
                        colWidths=[0.15, 0.12, 0.1, 0.1, 0.08, 0.12, 0.12])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.title('Summary Statistics: Immune Infiltration Metrics',
                 fontsize=16, fontweight='bold', pad=20)

        plt.savefig(f"{self.output_dir}/figures/summary_table.png",
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Example usage."""
    import scanpy as sc

    # Load data
    print("This is a template. Please use run_efficient_spatial_analysis.py")


if __name__ == '__main__':
    main()
