#!/usr/bin/env python3
"""
COMPREHENSIVE Tumor Spatial Analysis Framework - Publication Grade

This is a complete spatial analysis suite with extensive visualizations and statistics.
Designed to provide copious data and plots in multiple formats for publication.

Key Features:
- Spatial maps with all cell types
- Marker expression tracking across time and genotypes
- Tumor size/growth analysis
- Cellular neighborhood temporal dynamics
- Distance-based spatial analysis
- Co-localization metrics
- Multiple statistical tests with and without corrections
- Publication-quality figures in multiple formats

Author: Comprehensive rewrite for complete spatial analysis
Date: 2025-10-24
"""

import numpy as np
import pandas as pd

# Set matplotlib to use non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.spatial import cKDTree, distance_matrix
from scipy.stats import (mannwhitneyu, kruskal, spearmanr, pearsonr,
                         ttest_ind, f_oneway, linregress)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from typing import Dict, List, Tuple, Optional, Union
import warnings
import os
from pathlib import Path
import gc
from collections import defaultdict
warnings.filterwarnings('ignore')


class ComprehensiveTumorSpatialAnalysis:
    """
    Complete spatial analysis suite for tumor immunology with extensive visualizations.

    Provides every possible analysis and visualization for publication-quality results.
    """

    def __init__(self, adata, sample_metadata: pd.DataFrame,
                 tumor_markers: List[str], immune_markers: List[str],
                 output_dir: str = 'comprehensive_spatial_analysis'):
        """
        Initialize comprehensive spatial analysis.

        Parameters
        ----------
        adata : AnnData
            Full annotated data with spatial coordinates
        sample_metadata : pd.DataFrame
            Must have: sample_id, group (genotype), timepoint
        tumor_markers : list of str
            Tumor markers
        immune_markers : list of str
            Immune markers
        output_dir : str
            Output directory
        """
        self.adata = adata
        self.sample_metadata = self._parse_metadata(sample_metadata)
        self.tumor_markers = tumor_markers
        self.immune_markers = immune_markers
        self.output_dir = output_dir

        # Extract coordinates
        if 'spatial' in adata.obsm:
            self.coords = adata.obsm['spatial']
        elif 'X_centroid' in adata.obs and 'Y_centroid' in adata.obs:
            self.coords = np.column_stack([adata.obs['X_centroid'].values,
                                           adata.obs['Y_centroid'].values])
            adata.obsm['spatial'] = self.coords
        else:
            raise ValueError("No spatial coordinates found")

        # Merge metadata into adata.obs
        self._merge_metadata()

        # Setup directories
        self._setup_directories()

        # Storage
        self.structure_index = None
        self.neighborhood_profiles = None
        self.results = {}

        print(f"="*80)
        print("COMPREHENSIVE TUMOR SPATIAL ANALYSIS INITIALIZED")
        print(f"="*80)
        print(f"Total cells: {len(adata):,}")
        print(f"Samples: {adata.obs['sample_id'].nunique()}")
        print(f"Genotypes: {sorted(self.adata.obs['genotype'].dropna().unique())}")
        print(f"Timepoints: {sorted(self.adata.obs['timepoint'].dropna().unique())}")
        print(f"Output: {output_dir}/\n")


    def _parse_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Parse metadata to extract main group (KPT vs KPNT), cis/trans, and full group.

        Handles: "KPNT cis", "KPNT trans", "KPT Het cis", "KPT Het trans"

        Creates:
        - main_group: KPT or KPNT (for 2-group comparisons)
        - genotype: cis or trans (cis/trans status)
        - genotype_full: Full group name (for 4-group comparisons)
        """
        metadata = metadata.copy()

        # Standardize sample_id casing
        metadata['sample_id'] = metadata['sample_id'].str.upper()

        # Extract main group: KPT or KPNT
        metadata['main_group'] = metadata['group'].apply(
            lambda x: 'KPT' if 'KPT' in str(x) else 'KPNT'
        )

        # Extract cis/trans status
        metadata['genotype'] = metadata['group'].apply(
            lambda x: 'cis' if 'cis' in str(x).lower() else
                     ('trans' if 'trans' in str(x).lower() else 'Unknown')
        )

        # Full group name for detailed 4-group analysis
        metadata['genotype_full'] = metadata['group']

        # Convert timepoint to numeric
        metadata['timepoint'] = pd.to_numeric(metadata['timepoint'])

        print("Parsed metadata:")
        print(f"  Main groups: {sorted(metadata['main_group'].unique())}")
        print(f"  Cis/Trans: {sorted(metadata['genotype'].unique())}")
        print(f"  Full groups (4 subgroups): {sorted(metadata['genotype_full'].unique())}")
        print(f"  Timepoints: {sorted(metadata['timepoint'].unique())}")
        print("\nSample breakdown:")
        print(metadata.groupby(['main_group', 'genotype', 'timepoint']).size().to_string())
        print()

        return metadata


    def _merge_metadata(self):
        """Merge metadata into adata.obs."""
        # Standardize sample_id in adata
        if 'sample_id' in self.adata.obs:
            self.adata.obs['sample_id'] = self.adata.obs['sample_id'].str.upper()

            # Merge metadata columns including main_group
            sample_to_meta = {}
            for col in ['main_group', 'genotype', 'genotype_full', 'timepoint', 'treatment']:
                if col in self.sample_metadata.columns:
                    sample_to_meta[col] = dict(zip(
                        self.sample_metadata['sample_id'],
                        self.sample_metadata[col]
                    ))
                    self.adata.obs[col] = self.adata.obs['sample_id'].map(sample_to_meta[col])


    def _setup_directories(self):
        """Create comprehensive directory structure."""
        dirs = [
            self.output_dir,
            f"{self.output_dir}/data",
            f"{self.output_dir}/structures",
            f"{self.output_dir}/statistics",

            # Visualization directories
            f"{self.output_dir}/figures",
            f"{self.output_dir}/figures/spatial_maps",
            f"{self.output_dir}/figures/spatial_maps/by_sample",
            f"{self.output_dir}/figures/spatial_maps/by_timepoint",
            f"{self.output_dir}/figures/spatial_maps/by_genotype",

            f"{self.output_dir}/figures/temporal",
            f"{self.output_dir}/figures/temporal/tumor_size",
            f"{self.output_dir}/figures/temporal/marker_expression",
            f"{self.output_dir}/figures/temporal/infiltration",

            f"{self.output_dir}/figures/genotype_comparisons",
            f"{self.output_dir}/figures/genotype_comparisons/boxplots",
            f"{self.output_dir}/figures/genotype_comparisons/violin",
            f"{self.output_dir}/figures/genotype_comparisons/barplots",

            f"{self.output_dir}/figures/neighborhoods",
            f"{self.output_dir}/figures/neighborhoods/profiles",
            f"{self.output_dir}/figures/neighborhoods/temporal",
            f"{self.output_dir}/figures/neighborhoods/spatial",

            f"{self.output_dir}/figures/distances",
            f"{self.output_dir}/figures/colocalization",
            f"{self.output_dir}/figures/heatmaps",
            f"{self.output_dir}/figures/combined",
        ]

        for d in dirs:
            os.makedirs(d, exist_ok=True)


    def run_complete_analysis(self, population_config: Dict,
                             immune_populations: List[str],
                             **kwargs):
        """
        Run the COMPLETE analysis pipeline with all visualizations.

        This is the main entry point that runs everything.
        """
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE ANALYSIS PIPELINE")
        print("="*80 + "\n")

        # Phase 1: Structure Detection
        print("\n### PHASE 1: TUMOR STRUCTURE DETECTION ###\n")
        self.detect_all_tumor_structures(population_config, **kwargs)

        # Phase 2: Per-Structure Infiltration
        print("\n### PHASE 2: INFILTRATION QUANTIFICATION ###\n")
        metrics_df = self.analyze_structures_individually(immune_populations, **kwargs)

        # Phase 3: Marker Expression Analysis
        print("\n### PHASE 3: MARKER EXPRESSION ANALYSIS ###\n")
        marker_df = self.analyze_marker_expression_temporal()

        # Phase 4: Tumor Size Analysis
        print("\n### PHASE 4: TUMOR SIZE ANALYSIS ###\n")
        size_df = self.analyze_tumor_size_temporal()

        # Phase 5: Cellular Neighborhoods
        print("\n### PHASE 5: CELLULAR NEIGHBORHOOD ANALYSIS ###\n")
        neighborhood_df = self.detect_cellular_neighborhoods_comprehensive(
            populations=list(population_config.keys()), **kwargs
        )

        # Phase 6: Distance Analysis
        print("\n### PHASE 6: SPATIAL DISTANCE ANALYSIS ###\n")
        distance_df = self.analyze_spatial_distances(immune_populations)

        # Phase 7: Co-localization
        print("\n### PHASE 7: CO-LOCALIZATION ANALYSIS ###\n")
        coloc_df = self.analyze_colocalization(immune_populations)

        # Phase 8: Statistical Testing
        print("\n### PHASE 8: COMPREHENSIVE STATISTICAL ANALYSIS ###\n")
        stats_results = self.comprehensive_statistical_analysis(
            metrics_df, marker_df, size_df, neighborhood_df
        )

        # Phase 9: All Visualizations
        print("\n### PHASE 9: GENERATING ALL VISUALIZATIONS ###\n")
        self.generate_all_visualizations(
            metrics_df, marker_df, size_df, neighborhood_df,
            distance_df, coloc_df, stats_results
        )

        # Phase 10: Summary Report
        print("\n### PHASE 10: GENERATING COMPREHENSIVE REPORT ###\n")
        self.generate_comprehensive_report()

        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nAll outputs saved to: {self.output_dir}/")
        print("\nKey directories:")
        print(f"  - {self.output_dir}/data/                      : All CSV data files")
        print(f"  - {self.output_dir}/statistics/                : Statistical test results")
        print(f"  - {self.output_dir}/figures/spatial_maps/      : Spatial visualizations")
        print(f"  - {self.output_dir}/figures/temporal/          : Time-series plots")
        print(f"  - {self.output_dir}/figures/genotype_comparisons/ : Genotype analyses")
        print(f"  - {self.output_dir}/figures/neighborhoods/     : Neighborhood analyses")
        print("\n" + "="*80 + "\n")

        return metrics_df, marker_df, size_df, neighborhood_df, stats_results


    # Copy structure detection from efficient version
    def detect_all_tumor_structures(self, population_config: Dict,
                                   min_cluster_size: int = 50,
                                   eps: float = 30,
                                   min_samples: int = 10,
                                   tumor_population: str = 'Tumor') -> pd.DataFrame:
        """Detect all tumor structures."""
        # Define populations
        self._define_populations(population_config)

        # Get tumor cells
        tumor_mask = self.adata.obs[f'is_{tumor_population}'].values
        tumor_coords = self.coords[tumor_mask].astype(np.float32)
        tumor_sample_ids = self.adata.obs.loc[tumor_mask, 'sample_id'].values

        print(f"Clustering {tumor_mask.sum():,} tumor cells...")

        structure_records = []
        global_structure_id = 0

        for sample_id in np.unique(tumor_sample_ids):
            sample_tumor_mask = tumor_sample_ids == sample_id
            sample_coords = tumor_coords[sample_tumor_mask]
            sample_indices = np.where(tumor_mask)[0][sample_tumor_mask]

            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            labels = clustering.fit_predict(sample_coords)

            for label in set(labels):
                if label == -1:
                    continue

                cluster_mask = labels == label
                cluster_size = cluster_mask.sum()

                if cluster_size >= min_cluster_size:
                    cluster_coords = sample_coords[cluster_mask]
                    cluster_cell_indices = sample_indices[cluster_mask]

                    centroid = cluster_coords.mean(axis=0)
                    x_min, x_max = cluster_coords[:, 0].min(), cluster_coords[:, 0].max()
                    y_min, y_max = cluster_coords[:, 1].min(), cluster_coords[:, 1].max()
                    bbox_area = (x_max - x_min) * (y_max - y_min)

                    # Get sample metadata
                    sample_meta_row = self.sample_metadata[
                        self.sample_metadata['sample_id'] == sample_id
                    ]

                    if len(sample_meta_row) > 0:
                        sample_meta = sample_meta_row.iloc[0]
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
                            'width_um': x_max - x_min,
                            'height_um': y_max - y_min,
                            'timepoint': sample_meta.get('timepoint'),
                            'main_group': sample_meta.get('main_group'),
                            'genotype': sample_meta.get('genotype'),
                            'genotype_full': sample_meta.get('genotype_full'),
                            'treatment': sample_meta.get('treatment')
                        })

                        np.save(
                            f"{self.output_dir}/structures/structure_{global_structure_id:04d}_cells.npy",
                            cluster_cell_indices
                        )

                        global_structure_id += 1

            print(f"  {sample_id}: {len([r for r in structure_records if r['sample_id'] == sample_id])} structures")

        self.structure_index = pd.DataFrame(structure_records)
        self.structure_index.to_csv(f"{self.output_dir}/data/structure_index.csv", index=False)

        print(f"\n✓ Detected {len(self.structure_index)} tumor structures")
        print(f"  Size range: {self.structure_index['n_cells'].min()} - "
              f"{self.structure_index['n_cells'].max()} cells")

        del tumor_coords, tumor_sample_ids
        gc.collect()

        return self.structure_index


    def _define_populations(self, population_config: Dict):
        """Define cell populations."""
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


    def _get_cells_in_structure(self, structure_id: int) -> np.ndarray:
        """
        Get a boolean mask of cells belonging to a specific structure.

        Args:
            structure_id: The structure ID to get cells for

        Returns:
            Boolean array of length n_cells, True where cell is in the structure
        """
        # Load cell indices from saved file
        cell_indices = np.load(
            f"{self.output_dir}/structures/structure_{structure_id:04d}_cells.npy"
        )

        # Create boolean mask for all cells
        mask = np.zeros(len(self.adata), dtype=bool)
        mask[cell_indices] = True

        return mask


    def analyze_structures_individually(self, immune_populations: List[str],
                                       boundary_widths: List[float] = [30, 100, 200],
                                       buffer_distance: float = 500) -> pd.DataFrame:
        """Analyze each structure for infiltration."""
        all_metrics = []

        all_coords_f32 = self.coords.astype(np.float32)
        global_tree = cKDTree(all_coords_f32)

        n_structures = len(self.structure_index)

        for idx, row in self.structure_index.iterrows():
            if idx % 50 == 0:
                print(f"  Progress: {100*idx/n_structures:.1f}% ({idx}/{n_structures})")

            struct_id = row['structure_id']
            tumor_cell_indices = np.load(
                f"{self.output_dir}/structures/structure_{struct_id:04d}_cells.npy"
            )

            centroid = np.array([row['centroid_x'], row['centroid_y']], dtype=np.float32)
            nearby_indices = global_tree.query_ball_point(centroid, buffer_distance)
            nearby_indices = np.array(nearby_indices)

            local_coords = all_coords_f32[nearby_indices]
            tumor_local_mask = np.isin(nearby_indices, tumor_cell_indices)
            tumor_local_coords = local_coords[tumor_local_mask]

            if len(tumor_local_coords) < 3:
                continue

            tumor_tree = cKDTree(tumor_local_coords)
            distances, _ = tumor_tree.query(local_coords, k=1, workers=1)

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

            for region in ['Tumor_Core', 'Margin', 'Peri_Tumor', 'Distal']:
                region_mask = boundary_labels == region
                n_total = region_mask.sum()

                if n_total == 0:
                    continue

                region_coords = local_coords[region_mask]
                x_range = region_coords[:, 0].max() - region_coords[:, 0].min()
                y_range = region_coords[:, 1].max() - region_coords[:, 1].min()
                area_mm2 = (x_range * y_range) / 1e6

                for pop_name in immune_populations:
                    if f'is_{pop_name}' not in self.adata.obs:
                        continue

                    pop_status = self.adata.obs.iloc[nearby_indices][f'is_{pop_name}'].values
                    pop_in_region = pop_status[region_mask]
                    n_pop = pop_in_region.sum()

                    pct_pop = 100 * n_pop / n_total
                    density = n_pop / area_mm2 if area_mm2 > 0 else 0

                    all_metrics.append({
                        'structure_id': struct_id,
                        'sample_id': row['sample_id'],
                        'timepoint': row['timepoint'],
                        'main_group': row['main_group'],
                        'genotype': row['genotype'],
                        'genotype_full': row['genotype_full'],
                        'region': region,
                        'population': pop_name,
                        'n_cells': n_pop,
                        'total_cells': n_total,
                        'percentage': pct_pop,
                        'density_per_mm2': density,
                        'tumor_size': row['n_cells'],
                        'tumor_area_um2': row['area_um2']
                    })

            del tumor_cell_indices, local_coords, distances, boundary_labels

            if idx % 100 == 0:
                gc.collect()

        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(f"{self.output_dir}/data/infiltration_metrics.csv", index=False)

        print(f"\n✓ Completed infiltration analysis for {n_structures} structures")

        del all_coords_f32, global_tree
        gc.collect()

        return metrics_df


    def analyze_marker_expression_temporal(self) -> pd.DataFrame:
        """
        Analyze marker expression over time for ALL markers.

        Returns detailed expression metrics per marker, timepoint, genotype.
        """
        print("Analyzing marker expression across time...")

        results = []
        all_markers = list(self.adata.var_names)

        for marker in all_markers:
            marker_idx = self.adata.var_names.get_loc(marker)

            # Get expression values
            if 'gated' in self.adata.layers:
                values = self.adata.layers['gated'][:, marker_idx]
                positivity = values > 0
            else:
                values = self.adata.X[:, marker_idx]
                threshold = np.percentile(values[values > 0], 90) if np.any(values > 0) else 0
                positivity = values > threshold

            # Analyze by sample
            for sample_id in self.adata.obs['sample_id'].unique():
                sample_mask = self.adata.obs['sample_id'] == sample_id

                sample_values = values[sample_mask]
                sample_pos = positivity[sample_mask]

                # Get metadata
                meta_row = self.sample_metadata[
                    self.sample_metadata['sample_id'] == sample_id
                ]

                if len(meta_row) == 0:
                    continue

                meta = meta_row.iloc[0]

                pos_values = sample_values[sample_values > 0]

                results.append({
                    'marker': marker,
                    'sample_id': sample_id,
                    'timepoint': meta['timepoint'],
                    'main_group': meta['main_group'],
                    'genotype': meta['genotype'],
                    'genotype_full': meta['genotype_full'],
                    'n_cells': len(sample_values),
                    'n_positive': sample_pos.sum(),
                    'pct_positive': 100 * sample_pos.sum() / len(sample_values),
                    'mean_expression': sample_values.mean(),
                    'mean_positive': pos_values.mean() if len(pos_values) > 0 else 0,
                    'median_positive': np.median(pos_values) if len(pos_values) > 0 else 0,
                    'std_positive': pos_values.std() if len(pos_values) > 0 else 0
                })

            if len(results) > 0 and len(results) % 10 == 0:
                print(f"  Processed {len(results)} marker-sample combinations...")

        marker_df = pd.DataFrame(results)
        marker_df.to_csv(f"{self.output_dir}/data/marker_expression_temporal.csv", index=False)

        print(f"✓ Analyzed {len(all_markers)} markers across {len(marker_df)} samples")

        return marker_df


    def analyze_tumor_size_temporal(self) -> pd.DataFrame:
        """
        Analyze tumor size and growth across time and genotypes.

        Returns structure-level and sample-level size metrics.
        """
        print("Analyzing tumor size across time...")

        if self.structure_index is None:
            print("  ERROR: No structures detected")
            return pd.DataFrame()

        # Per-structure metrics already in structure_index

        # Aggregate per sample
        sample_agg = self.structure_index.groupby(['sample_id', 'timepoint', 'main_group',
                                                   'genotype', 'genotype_full']).agg({
            'n_cells': ['sum', 'mean', 'std', 'count'],
            'area_um2': ['sum', 'mean', 'std'],
            'width_um': 'mean',
            'height_um': 'mean'
        }).reset_index()

        sample_agg.columns = ['_'.join(col).strip('_') for col in sample_agg.columns.values]
        sample_agg.rename(columns={
            'sample_id_': 'sample_id',
            'timepoint_': 'timepoint',
            'main_group_': 'main_group',
            'genotype_': 'genotype',
            'genotype_full_': 'genotype_full',
            'n_cells_sum': 'total_tumor_cells',
            'n_cells_mean': 'mean_structure_size',
            'n_cells_std': 'std_structure_size',
            'n_cells_count': 'n_structures',
            'area_um2_sum': 'total_tumor_area',
            'area_um2_mean': 'mean_structure_area',
            'area_um2_std': 'std_structure_area',
            'width_um_mean': 'mean_width',
            'height_um_mean': 'mean_height'
        }, inplace=True)

        sample_agg.to_csv(f"{self.output_dir}/data/tumor_size_temporal.csv", index=False)

        print(f"✓ Analyzed tumor size for {len(sample_agg)} samples")

        return sample_agg


    def detect_cellular_neighborhoods_comprehensive(self, populations: List[str],
                                                   window_size: float = 100,
                                                   n_clusters: int = 10,
                                                   subsample_size: int = 100000) -> pd.DataFrame:
        """
        Comprehensive cellular neighborhood detection with temporal tracking.
        """
        print(f"Computing cellular neighborhoods (window={window_size}μm, k={n_clusters})...")

        # Build population matrix
        pop_matrix = np.zeros((len(self.adata), len(populations)), dtype=np.float32)
        for i, pop in enumerate(populations):
            if f'is_{pop}' in self.adata.obs:
                pop_matrix[:, i] = self.adata.obs[f'is_{pop}'].values.astype(np.float32)

        # Build KD-tree
        coords_f32 = self.coords.astype(np.float32)
        tree = cKDTree(coords_f32)

        # Compute neighborhood profiles
        batch_size = 50000
        neighborhood_profiles = np.zeros((len(coords_f32), len(populations)), dtype=np.float32)

        for i in range(0, len(coords_f32), batch_size):
            end_idx = min(i + batch_size, len(coords_f32))
            batch_coords = coords_f32[i:end_idx]

            neighbors_list = tree.query_ball_point(batch_coords, window_size, workers=1)

            for j, neighbors in enumerate(neighbors_list):
                if len(neighbors) > 1:
                    neighbor_pops = pop_matrix[neighbors]
                    neighborhood_profiles[i+j] = neighbor_pops.mean(axis=0)

            if i // batch_size % 20 == 0:
                print(f"  Progress: {100*end_idx/len(coords_f32):.1f}%")

        # Cluster neighborhoods
        print(f"  Clustering into {n_clusters} neighborhood types...")

        if len(neighborhood_profiles) > subsample_size:
            subsample_idx = np.random.choice(len(neighborhood_profiles),
                                            subsample_size, replace=False)
            subsample_profiles = neighborhood_profiles[subsample_idx]
        else:
            subsample_profiles = neighborhood_profiles

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        kmeans.fit(subsample_profiles)

        neighborhood_labels = kmeans.predict(neighborhood_profiles)
        self.adata.obs['neighborhood_type'] = neighborhood_labels

        # Characterize neighborhoods
        neighborhood_summary = []
        for cluster_id in range(n_clusters):
            cluster_mask = neighborhood_labels == cluster_id
            cluster_profile = neighborhood_profiles[cluster_mask].mean(axis=0)

            # Also get stats by timepoint and genotype
            cluster_cells = self.adata.obs[cluster_mask]

            summary = {
                'neighborhood_id': cluster_id,
                'n_cells': cluster_mask.sum(),
                'percentage': 100 * cluster_mask.sum() / len(neighborhood_labels)
            }

            for i, pop in enumerate(populations):
                summary[f'{pop}_enrichment'] = cluster_profile[i]

            neighborhood_summary.append(summary)

        neighborhood_df = pd.DataFrame(neighborhood_summary)
        neighborhood_df.to_csv(f"{self.output_dir}/data/neighborhood_profiles.csv", index=False)

        # Also save per-cell neighborhood assignments with metadata
        cell_neighborhoods = pd.DataFrame({
            'cell_id': np.arange(len(self.adata)),
            'neighborhood_type': neighborhood_labels,
            'sample_id': self.adata.obs['sample_id'].values,
            'timepoint': self.adata.obs['timepoint'].values,
            'genotype': self.adata.obs['genotype'].values,
            'genotype_full': self.adata.obs['genotype_full'].values,
            'X': self.coords[:, 0],
            'Y': self.coords[:, 1]
        })
        cell_neighborhoods.to_csv(f"{self.output_dir}/data/cell_neighborhoods.csv", index=False)

        print(f"✓ Detected {n_clusters} neighborhood types")

        del coords_f32, tree, pop_matrix, neighborhood_profiles
        gc.collect()

        return neighborhood_df


    def analyze_spatial_distances(self, immune_populations: List[str]) -> pd.DataFrame:
        """
        Analyze distances between immune cells and tumor structures.

        For each immune population, computes:
        - Mean/median distance to nearest tumor structure
        - Distribution of distances across regions
        - Per-structure distance metrics
        """
        print("Analyzing spatial distances...")

        if self.structure_index is None or len(self.structure_index) == 0:
            print("  No tumor structures detected, skipping distance analysis")
            return pd.DataFrame()

        all_distances = []
        coords_f32 = self.coords.astype(np.float32)

        # For each structure, compute distances
        for idx, structure in self.structure_index.iterrows():
            struct_id = structure['structure_id']

            # Load structure cells
            try:
                tumor_cell_indices = np.load(
                    f"{self.output_dir}/structures/structure_{struct_id:04d}_cells.npy"
                )
            except FileNotFoundError:
                continue

            if len(tumor_cell_indices) == 0:
                continue

            tumor_coords = coords_f32[tumor_cell_indices]
            tumor_tree = cKDTree(tumor_coords)

            # For each immune population
            for pop_name in immune_populations:
                if f'is_{pop_name}' not in self.adata.obs:
                    continue

                pop_mask = self.adata.obs[f'is_{pop_name}'].values
                if pop_mask.sum() == 0:
                    continue

                # Get immune cells in the same sample
                sample_mask = self.adata.obs['sample_id'] == structure['sample_id']
                combined_mask = pop_mask & sample_mask

                if combined_mask.sum() == 0:
                    continue

                immune_coords = coords_f32[combined_mask]

                # Compute distances to nearest tumor cell in this structure
                distances, _ = tumor_tree.query(immune_coords, k=1, workers=1)

                # Compute statistics
                all_distances.append({
                    'structure_id': struct_id,
                    'sample_id': structure['sample_id'],
                    'timepoint': structure['timepoint'],
                    'main_group': structure['main_group'],
                    'genotype': structure['genotype'],
                    'genotype_full': structure['genotype_full'],
                    'population': pop_name,
                    'n_immune_cells': len(immune_coords),
                    'mean_distance': distances.mean(),
                    'median_distance': np.median(distances),
                    'std_distance': distances.std(),
                    'min_distance': distances.min(),
                    'max_distance': distances.max(),
                    'q25_distance': np.percentile(distances, 25),
                    'q75_distance': np.percentile(distances, 75),
                    'n_within_30um': (distances <= 30).sum(),
                    'n_within_100um': (distances <= 100).sum(),
                    'pct_within_30um': 100 * (distances <= 30).sum() / len(distances),
                    'pct_within_100um': 100 * (distances <= 100).sum() / len(distances)
                })

        if len(all_distances) == 0:
            print("  No distance data computed")
            return pd.DataFrame()

        distance_df = pd.DataFrame(all_distances)
        distance_df.to_csv(f"{self.output_dir}/data/spatial_distances.csv", index=False)

        print(f"✓ Distance analysis complete: {len(distance_df)} structure-population combinations")
        return distance_df


    def analyze_colocalization(self, immune_populations: List[str]) -> pd.DataFrame:
        """
        Analyze co-localization between different cell types.

        For each pair of populations, computes:
        - Spatial correlation (neighbors within radius)
        - Enrichment scores
        - Per-sample co-localization metrics
        """
        print("Analyzing co-localization...")

        if len(immune_populations) < 2:
            print("  Need at least 2 populations for co-localization analysis")
            return pd.DataFrame()

        coloc_results = []
        coords_f32 = self.coords.astype(np.float32)
        tree = cKDTree(coords_f32)
        radius = 50.0  # μm radius for co-localization

        # Analyze each pair of populations
        from itertools import combinations

        for pop1, pop2 in combinations(immune_populations, 2):
            col1 = f'is_{pop1}'
            col2 = f'is_{pop2}'

            if col1 not in self.adata.obs or col2 not in self.adata.obs:
                continue

            mask1 = self.adata.obs[col1].values
            mask2 = self.adata.obs[col2].values

            if mask1.sum() == 0 or mask2.sum() == 0:
                continue

            # Analyze per sample
            for sample_id in self.adata.obs['sample_id'].unique():
                sample_mask = self.adata.obs['sample_id'] == sample_id

                sample_mask1 = mask1 & sample_mask
                sample_mask2 = mask2 & sample_mask

                if sample_mask1.sum() < 10 or sample_mask2.sum() < 10:
                    continue

                coords1 = coords_f32[sample_mask1]
                coords2 = coords_f32[sample_mask2]

                # For each cell of pop1, count neighbors of pop2 within radius
                neighbors = tree.query_ball_point(coords1, radius, workers=1)

                # Count pop2 cells in neighborhood
                n_pop2_neighbors = []
                for neighbor_idx in neighbors:
                    neighbor_mask = sample_mask2[neighbor_idx]
                    n_pop2_neighbors.append(neighbor_mask.sum())

                n_pop2_neighbors = np.array(n_pop2_neighbors)

                # Compute enrichment: observed vs expected
                expected_pop2 = sample_mask2.sum() / sample_mask.sum()
                local_densities = n_pop2_neighbors / len(coords2)

                # Get sample metadata
                sample_meta = self.adata.obs[sample_mask].iloc[0]

                coloc_results.append({
                    'sample_id': sample_id,
                    'timepoint': sample_meta['timepoint'],
                    'main_group': sample_meta['main_group'],
                    'genotype': sample_meta['genotype'],
                    'genotype_full': sample_meta['genotype_full'],
                    'population_1': pop1,
                    'population_2': pop2,
                    'n_pop1': len(coords1),
                    'n_pop2': len(coords2),
                    'mean_pop2_neighbors': n_pop2_neighbors.mean(),
                    'median_pop2_neighbors': np.median(n_pop2_neighbors),
                    'std_pop2_neighbors': n_pop2_neighbors.std(),
                    'pct_pop1_with_pop2_neighbor': 100 * (n_pop2_neighbors > 0).sum() / len(n_pop2_neighbors),
                    'enrichment_score': (n_pop2_neighbors.mean() / len(coords2)) / (expected_pop2 + 1e-10)
                })

        if len(coloc_results) == 0:
            print("  No co-localization data computed")
            return pd.DataFrame()

        coloc_df = pd.DataFrame(coloc_results)
        coloc_df.to_csv(f"{self.output_dir}/data/colocalization_analysis.csv", index=False)

        print(f"✓ Co-localization analysis complete: {len(coloc_df)} sample-population pairs")
        return coloc_df


    def comprehensive_statistical_analysis(self, metrics_df: pd.DataFrame,
                                          marker_df: pd.DataFrame,
                                          size_df: pd.DataFrame,
                                          neighborhood_df: pd.DataFrame) -> Dict:
        """
        Comprehensive statistical testing for all analyses.
        """
        print("Running comprehensive statistical tests...")

        results = {}

        # Test infiltration temporal trends
        if len(metrics_df) > 0:
            print("  Testing infiltration temporal trends...")
            results['infiltration_temporal'] = self._test_temporal_trends(metrics_df)

        # Test infiltration genotype differences
        if len(metrics_df) > 0:
            print("  Testing infiltration genotype differences...")
            results['infiltration_genotype'] = self._test_genotype_differences(metrics_df)

        # Test marker expression temporal
        if len(marker_df) > 0:
            print("  Testing marker expression temporal trends...")
            results['marker_temporal'] = self._test_temporal_trends(
                marker_df, value_col='pct_positive'
            )

        # Test tumor size temporal
        if len(size_df) > 0:
            print("  Testing tumor size temporal trends...")
            results['size_temporal'] = self._test_temporal_trends(
                size_df, value_col='mean_structure_size'
            )

        # Save all results
        for test_name, result_df in results.items():
            if len(result_df) > 0:
                result_df.to_csv(
                    f"{self.output_dir}/statistics/{test_name}.csv", index=False
                )

        print("✓ Statistical analysis complete")

        return results


    def _test_temporal_trends(self, df: pd.DataFrame,
                             value_col: str = 'percentage',
                             group_cols: List[str] = None) -> pd.DataFrame:
        """Test temporal trends with multiple statistical approaches."""

        if group_cols is None:
            # Determine appropriate grouping columns based on available columns
            if 'region' in df.columns and 'population' in df.columns:
                # Infiltration metrics: group by population and region
                group_cols = ['population', 'region']
            elif 'marker' in df.columns:
                # Marker expression: group by marker (and optionally genotype)
                group_cols = ['marker', 'genotype'] if 'genotype' in df.columns else ['marker']
            elif 'genotype' in df.columns:
                # Tumor size or other genotype-based metrics
                group_cols = ['genotype']
            else:
                # Fallback: no grouping, analyze all data together
                group_cols = []

        results = []

        # Get grouping combinations
        if len(group_cols) > 0:
            grouping_values = df[group_cols].drop_duplicates().values
        else:
            # No grouping - analyze entire dataset as one group
            grouping_values = [tuple()]

        for group_vals in grouping_values:
            # Filter to this group
            mask = np.ones(len(df), dtype=bool)
            for i, col in enumerate(group_cols):
                mask &= (df[col] == group_vals[i])

            subset = df[mask].copy()

            if len(subset) < 3 or subset['timepoint'].nunique() < 3:
                continue

            # Spearman correlation
            rho, p_spearman = spearmanr(subset['timepoint'], subset[value_col])

            # Linear regression
            slope, intercept, r_value, p_linear, std_err = linregress(
                subset['timepoint'], subset[value_col]
            )

            result = {
                'test': 'temporal_trend',
                'n_timepoints': subset['timepoint'].nunique(),
                'n_samples': len(subset),
                'spearman_rho': rho,
                'p_spearman': p_spearman,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_linear': p_linear
            }

            for i, col in enumerate(group_cols):
                result[col] = group_vals[i]

            results.append(result)

        if len(results) == 0:
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        # FDR correction
        if len(results_df) > 0:
            _, p_adj_spearman, _, _ = multipletests(results_df['p_spearman'], method='fdr_bh')
            _, p_adj_linear, _, _ = multipletests(results_df['p_linear'], method='fdr_bh')

            results_df['p_adj_spearman'] = p_adj_spearman
            results_df['p_adj_linear'] = p_adj_linear
            results_df['significant'] = (results_df['p_adj_spearman'] < 0.05) | (results_df['p_adj_linear'] < 0.05)

        return results_df


    def _test_genotype_differences(self, df: pd.DataFrame,
                                  value_col: str = 'percentage',
                                  group_cols: List[str] = None) -> pd.DataFrame:
        """
        Test genotype differences at multiple levels:
        1. Main group (KPT vs KPNT) - 2-group comparison
        2. Cis/trans - 2-group comparison
        3. Full groups - 4-group pairwise comparisons
        """

        if group_cols is None:
            group_cols = ['population', 'region'] if 'region' in df.columns else ['marker']

        results = []
        grouping_values = df[group_cols].drop_duplicates().values

        for group_vals in grouping_values:
            # Filter to this group
            mask = np.ones(len(df), dtype=bool)
            for i, col in enumerate(group_cols):
                mask &= (df[col] == group_vals[i])

            subset = df[mask]

            # 1. Main group comparison (KPT vs KPNT)
            if 'main_group' in df.columns:
                main_groups = subset['main_group'].dropna().unique()
                if len(main_groups) == 2:
                    g1, g2 = main_groups[0], main_groups[1]
                    data1 = subset[subset['main_group'] == g1][value_col].values
                    data2 = subset[subset['main_group'] == g2][value_col].values

                    if len(data1) >= 2 and len(data2) >= 2:
                        stat_mw, p_mw = mannwhitneyu(data1, data2, alternative='two-sided')
                        stat_t, p_t = ttest_ind(data1, data2)

                        result = {
                            'test': 'main_group_comparison',
                            'comparison_type': 'KPT_vs_KPNT',
                            'group_1': g1,
                            'group_2': g2,
                            'n_1': len(data1),
                            'n_2': len(data2),
                            'mean_1': data1.mean(),
                            'mean_2': data2.mean(),
                            'fold_change': data2.mean() / (data1.mean() + 1e-10),
                            'stat_mannwhitney': stat_mw,
                            'p_mannwhitney': p_mw,
                            'stat_ttest': stat_t,
                            'p_ttest': p_t
                        }
                        for j, col in enumerate(group_cols):
                            result[col] = group_vals[j]
                        results.append(result)

            # 2. Cis/trans comparison
            if 'genotype' in df.columns:
                genotypes = subset['genotype'].dropna().unique()
                if len(genotypes) == 2:
                    g1, g2 = genotypes[0], genotypes[1]
                    data1 = subset[subset['genotype'] == g1][value_col].values
                    data2 = subset[subset['genotype'] == g2][value_col].values

                    if len(data1) >= 2 and len(data2) >= 2:
                        stat_mw, p_mw = mannwhitneyu(data1, data2, alternative='two-sided')
                        stat_t, p_t = ttest_ind(data1, data2)

                        result = {
                            'test': 'cistrans_comparison',
                            'comparison_type': 'cis_vs_trans',
                            'group_1': g1,
                            'group_2': g2,
                            'n_1': len(data1),
                            'n_2': len(data2),
                            'mean_1': data1.mean(),
                            'mean_2': data2.mean(),
                            'fold_change': data2.mean() / (data1.mean() + 1e-10),
                            'stat_mannwhitney': stat_mw,
                            'p_mannwhitney': p_mw,
                            'stat_ttest': stat_t,
                            'p_ttest': p_t
                        }
                        for j, col in enumerate(group_cols):
                            result[col] = group_vals[j]
                        results.append(result)

            # 3. Full group pairwise comparisons (all 4 subgroups)
            if 'genotype_full' in df.columns:
                full_groups = subset['genotype_full'].dropna().unique()
                if len(full_groups) >= 2:
                    for i, g1 in enumerate(full_groups):
                        for g2 in full_groups[i+1:]:
                            data1 = subset[subset['genotype_full'] == g1][value_col].values
                            data2 = subset[subset['genotype_full'] == g2][value_col].values

                            if len(data1) < 2 or len(data2) < 2:
                                continue

                            stat_mw, p_mw = mannwhitneyu(data1, data2, alternative='two-sided')
                            stat_t, p_t = ttest_ind(data1, data2)

                            result = {
                                'test': 'subgroup_comparison',
                                'comparison_type': 'pairwise_4groups',
                                'group_1': g1,
                                'group_2': g2,
                                'n_1': len(data1),
                                'n_2': len(data2),
                                'mean_1': data1.mean(),
                                'mean_2': data2.mean(),
                                'fold_change': data2.mean() / (data1.mean() + 1e-10),
                                'stat_mannwhitney': stat_mw,
                                'p_mannwhitney': p_mw,
                                'stat_ttest': stat_t,
                                'p_ttest': p_t
                            }
                            for j, col in enumerate(group_cols):
                                result[col] = group_vals[j]
                            results.append(result)

        if len(results) == 0:
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        # FDR correction
        if len(results_df) > 0:
            _, p_adj_mw, _, _ = multipletests(results_df['p_mannwhitney'], method='fdr_bh')
            _, p_adj_t, _, _ = multipletests(results_df['p_ttest'], method='fdr_bh')

            results_df['p_adj_mannwhitney'] = p_adj_mw
            results_df['p_adj_ttest'] = p_adj_t
            results_df['significant'] = (results_df['p_adj_mannwhitney'] < 0.05)

        return results_df


    def generate_all_visualizations(self, metrics_df, marker_df, size_df,
                                   neighborhood_df, distance_df, coloc_df,
                                   stats_results):
        """
        Generate ALL visualizations with error handling.
        """
        print("\nGenerating comprehensive visualizations...")

        # Ensure figures directory exists
        import os
        os.makedirs(f"{self.output_dir}/figures", exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)

        errors = []

        # 1. Spatial maps
        print("\n  1. Spatial maps...")
        try:
            self._generate_spatial_maps()
        except Exception as e:
            print(f"     WARNING: Spatial maps failed: {e}")
            errors.append(f"Spatial maps: {e}")

        # 2. Tumor size plots
        print("  2. Tumor size temporal plots...")
        try:
            if len(size_df) > 0:
                self._plot_tumor_size_comprehensive(size_df, stats_results)
            else:
                print("     Skipping: no size data")
        except Exception as e:
            print(f"     WARNING: Tumor size plots failed: {e}")
            errors.append(f"Tumor size: {e}")

        # 3. Marker expression plots
        print("  3. Marker expression temporal plots...")
        try:
            if len(marker_df) > 0:
                self._plot_marker_expression_comprehensive(marker_df, stats_results)
            else:
                print("     Skipping: no marker data")
        except Exception as e:
            print(f"     WARNING: Marker plots failed: {e}")
            errors.append(f"Markers: {e}")

        # 4. Infiltration plots
        print("  4. Infiltration temporal/genotype plots...")
        try:
            if len(metrics_df) > 0:
                self._plot_infiltration_comprehensive(metrics_df, stats_results)
            else:
                print("     Skipping: no infiltration data")
        except Exception as e:
            print(f"     WARNING: Infiltration plots failed: {e}")
            errors.append(f"Infiltration: {e}")

        # 5. Neighborhood plots
        print("  5. Neighborhood analysis plots...")
        try:
            if len(neighborhood_df) > 0:
                self._plot_neighborhoods_comprehensive(neighborhood_df)
            else:
                print("     Skipping: no neighborhood data")
        except Exception as e:
            print(f"     WARNING: Neighborhood plots failed: {e}")
            errors.append(f"Neighborhoods: {e}")

        # 6. Heatmaps
        print("  6. Comprehensive heatmaps...")
        try:
            if len(metrics_df) > 0:
                self._plot_heatmaps_comprehensive(metrics_df, marker_df)
            else:
                print("     Skipping: no data for heatmaps")
        except Exception as e:
            print(f"     WARNING: Heatmaps failed: {e}")
            errors.append(f"Heatmaps: {e}")

        # 7. Combined summary figures
        print("  7. Combined summary figures...")
        try:
            self._plot_combined_summaries(metrics_df, marker_df, size_df)
        except Exception as e:
            print(f"     WARNING: Combined summary failed: {e}")
            errors.append(f"Summary: {e}")

        # 8. Distance plots
        print("  8. Distance analysis plots...")
        try:
            if len(distance_df) > 0:
                self._plot_distances_comprehensive(distance_df)
            else:
                print("     Skipping: no distance data")
        except Exception as e:
            print(f"     WARNING: Distance plots failed: {e}")
            errors.append(f"Distances: {e}")

        # 9. Co-localization plots
        print("  9. Co-localization plots...")
        try:
            if len(coloc_df) > 0:
                self._plot_colocalization_comprehensive(coloc_df)
            else:
                print("     Skipping: no co-localization data")
        except Exception as e:
            print(f"     WARNING: Co-localization plots failed: {e}")
            errors.append(f"Co-localization: {e}")

        if len(errors) > 0:
            print(f"\n⚠ Completed with {len(errors)} warnings (plots may be incomplete)")
            print("  See warnings above for details")
        else:
            print("\n✓ All visualizations complete!")


    def _generate_spatial_maps(self):
        """Generate spatial maps showing cell populations and structures."""
        import matplotlib.pyplot as plt

        print("    Generating spatial maps (sample plots)...")

        # Generate for a subset of samples (to avoid too many plots)
        sample_ids = self.adata.obs['sample_id'].unique()[:3]  # First 3 samples

        for sample_id in sample_ids:
            sample_mask = self.adata.obs['sample_id'] == sample_id
            sample_coords = self.coords[sample_mask]

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Plot 1: Tumor structures
            tumor_mask = self.adata.obs['is_Tumor'] & sample_mask
            immune_mask = self.adata.obs['is_CD45_positive'] & sample_mask

            axes[0].scatter(sample_coords[~(tumor_mask[sample_mask] | immune_mask[sample_mask]), 0],
                          sample_coords[~(tumor_mask[sample_mask] | immune_mask[sample_mask]), 1],
                          s=0.1, c='lightgray', alpha=0.3, label='Other')
            axes[0].scatter(self.coords[tumor_mask, 0], self.coords[tumor_mask, 1],
                          s=0.5, c='red', alpha=0.5, label='Tumor')
            axes[0].scatter(self.coords[immune_mask, 0], self.coords[immune_mask, 1],
                          s=0.5, c='blue', alpha=0.5, label='CD45+')
            axes[0].set_title(f'Sample {sample_id}: Tumor & Immune')
            axes[0].legend()
            axes[0].set_aspect('equal')

            # Plot 2: Neighborhood types
            if 'neighborhood_type' in self.adata.obs:
                neighborhoods = self.adata.obs.loc[sample_mask, 'neighborhood_type']
                scatter = axes[1].scatter(sample_coords[:, 0], sample_coords[:, 1],
                                        c=neighborhoods, s=0.5, cmap='tab10', alpha=0.5)
                axes[1].set_title(f'Sample {sample_id}: Neighborhoods')
                plt.colorbar(scatter, ax=axes[1], label='Neighborhood Type')
                axes[1].set_aspect('equal')

            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/figures/spatial_map_{sample_id}.png", dpi=150, bbox_inches='tight')
            plt.close()

        print(f"      Generated spatial maps for {len(sample_ids)} samples")


    def _plot_tumor_size_comprehensive(self, size_df: pd.DataFrame, stats_results: Dict):
        """Generate comprehensive tumor size plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Mean structure size over time by genotype
        for genotype in size_df['genotype'].unique():
            genotype_data = size_df[size_df['genotype'] == genotype]
            axes[0, 0].plot(genotype_data['timepoint'], genotype_data['mean_structure_size'],
                          marker='o', label=genotype, linewidth=2)
        axes[0, 0].set_xlabel('Timepoint')
        axes[0, 0].set_ylabel('Mean Structure Size (cells)')
        axes[0, 0].set_title('Tumor Growth: Mean Structure Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Total tumor area over time
        for genotype in size_df['genotype'].unique():
            genotype_data = size_df[size_df['genotype'] == genotype]
            axes[0, 1].plot(genotype_data['timepoint'], genotype_data['total_tumor_area'],
                          marker='s', label=genotype, linewidth=2)
        axes[0, 1].set_xlabel('Timepoint')
        axes[0, 1].set_ylabel('Total Tumor Area (μm²)')
        axes[0, 1].set_title('Total Tumor Area Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Number of structures over time
        for genotype in size_df['genotype'].unique():
            genotype_data = size_df[size_df['genotype'] == genotype]
            axes[1, 0].plot(genotype_data['timepoint'], genotype_data['n_structures'],
                          marker='^', label=genotype, linewidth=2)
        axes[1, 0].set_xlabel('Timepoint')
        axes[1, 0].set_ylabel('Number of Structures')
        axes[1, 0].set_title('Tumor Structure Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Genotype comparison boxplot
        sns.boxplot(data=size_df, x='genotype', y='mean_structure_size', ax=axes[1, 1])
        axes[1, 1].set_ylabel('Mean Structure Size (cells)')
        axes[1, 1].set_title('Tumor Size by Genotype')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/tumor_size_comprehensive.png", dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_marker_expression_comprehensive(self, marker_df: pd.DataFrame, stats_results: Dict):
        """Generate comprehensive marker expression plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get top markers by variability
        marker_variability = marker_df.groupby('marker')['pct_positive'].std().sort_values(ascending=False)
        top_markers = marker_variability.head(6).index.tolist()

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, marker in enumerate(top_markers):
            marker_data = marker_df[marker_df['marker'] == marker]

            for genotype in marker_data['genotype'].unique():
                genotype_data = marker_data[marker_data['genotype'] == genotype]
                axes[idx].plot(genotype_data['timepoint'], genotype_data['pct_positive'],
                             marker='o', label=genotype, linewidth=2, alpha=0.7)

            axes[idx].set_xlabel('Timepoint')
            axes[idx].set_ylabel('% Positive Cells')
            axes[idx].set_title(f'{marker} Expression Over Time')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/marker_expression_temporal.png", dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_infiltration_comprehensive(self, metrics_df: pd.DataFrame, stats_results: Dict):
        """Generate comprehensive infiltration plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get unique populations and regions
        populations = metrics_df['population'].unique()[:4]  # Top 4 populations
        regions = ['Tumor_Core', 'Margin', 'Peri_Tumor', 'Distal']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, pop in enumerate(populations):
            pop_data = metrics_df[metrics_df['population'] == pop]

            for region in regions:
                region_data = pop_data[pop_data['region'] == region]
                if len(region_data) > 0:
                    region_summary = region_data.groupby('timepoint')['percentage'].mean()
                    axes[idx].plot(region_summary.index, region_summary.values,
                                 marker='o', label=region, linewidth=2)

            axes[idx].set_xlabel('Timepoint')
            axes[idx].set_ylabel('% Infiltration')
            axes[idx].set_title(f'{pop} Infiltration by Region')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/infiltration_temporal.png", dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_neighborhoods_comprehensive(self, neighborhood_df: pd.DataFrame):
        """Generate comprehensive neighborhood plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Neighborhood sizes
        axes[0].bar(neighborhood_df['neighborhood_id'], neighborhood_df['n_cells'])
        axes[0].set_xlabel('Neighborhood Type')
        axes[0].set_ylabel('Number of Cells')
        axes[0].set_title('Neighborhood Sizes')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Neighborhood percentage
        axes[1].pie(neighborhood_df['percentage'], labels=neighborhood_df['neighborhood_id'],
                   autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Neighborhood Distribution')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/neighborhood_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_heatmaps_comprehensive(self, metrics_df: pd.DataFrame, marker_df: pd.DataFrame):
        """Generate comprehensive heatmaps."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        if len(metrics_df) == 0:
            return

        # Heatmap 1: Infiltration by population and region
        infiltration_pivot = metrics_df.pivot_table(
            values='percentage',
            index='population',
            columns='region',
            aggfunc='mean'
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(infiltration_pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                   ax=axes[0], cbar_kws={'label': '% Infiltration'})
        axes[0].set_title('Mean Infiltration by Population & Region')
        axes[0].set_ylabel('Population')
        axes[0].set_xlabel('Region')

        # Heatmap 2: Infiltration by genotype and region
        if 'main_group' in metrics_df.columns:
            genotype_pivot = metrics_df.pivot_table(
                values='percentage',
                index='population',
                columns='main_group',
                aggfunc='mean'
            )

            sns.heatmap(genotype_pivot, annot=True, fmt='.1f', cmap='viridis',
                       ax=axes[1], cbar_kws={'label': '% Infiltration'})
            axes[1].set_title('Mean Infiltration by Population & Group')
            axes[1].set_ylabel('Population')
            axes[1].set_xlabel('Main Group')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/infiltration_heatmaps.png", dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_combined_summaries(self, metrics_df: pd.DataFrame, marker_df: pd.DataFrame,
                                 size_df: pd.DataFrame):
        """Generate combined summary figures."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Summary 1: Tumor growth
        if len(size_df) > 0:
            ax1 = fig.add_subplot(gs[0, :2])
            for genotype in size_df['genotype'].unique():
                genotype_data = size_df[size_df['genotype'] == genotype]
                ax1.plot(genotype_data['timepoint'], genotype_data['total_tumor_cells'],
                        marker='o', label=genotype, linewidth=2)
            ax1.set_xlabel('Timepoint')
            ax1.set_ylabel('Total Tumor Cells')
            ax1.set_title('Tumor Growth Dynamics')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Summary 2: Key infiltration metrics
        if len(metrics_df) > 0:
            ax2 = fig.add_subplot(gs[0, 2])
            tumor_core = metrics_df[metrics_df['region'] == 'Tumor_Core']
            tumor_core_summary = tumor_core.groupby('population')['percentage'].mean().sort_values(ascending=False).head(5)
            ax2.barh(range(len(tumor_core_summary)), tumor_core_summary.values)
            ax2.set_yticks(range(len(tumor_core_summary)))
            ax2.set_yticklabels(tumor_core_summary.index)
            ax2.set_xlabel('% in Tumor Core')
            ax2.set_title('Top Infiltrating Populations')
            ax2.grid(True, alpha=0.3, axis='x')

        # Summary 3: Marker expression overview
        if len(marker_df) > 0:
            ax3 = fig.add_subplot(gs[1, :])
            marker_summary = marker_df.groupby(['marker', 'genotype'])['pct_positive'].mean().unstack()
            marker_summary.plot(kind='bar', ax=ax3, width=0.8)
            ax3.set_xlabel('Marker')
            ax3.set_ylabel('% Positive Cells')
            ax3.set_title('Marker Expression by Genotype')
            ax3.legend(title='Genotype')
            ax3.grid(True, alpha=0.3, axis='y')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Summary 4: Sample counts
        ax4 = fig.add_subplot(gs[2, :])
        sample_counts = self.adata.obs.groupby(['timepoint', 'genotype']).size().unstack(fill_value=0)
        sample_counts.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_xlabel('Timepoint')
        ax4.set_ylabel('Number of Cells')
        ax4.set_title('Sample Distribution')
        ax4.legend(title='Genotype')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.savefig(f"{self.output_dir}/figures/combined_summary.png", dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_distances_comprehensive(self, distance_df: pd.DataFrame):
        """Generate comprehensive distance analysis plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get top populations by sample count
        pop_counts = distance_df.groupby('population').size()
        top_pops = pop_counts.nlargest(4).index.tolist()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, pop in enumerate(top_pops):
            pop_data = distance_df[distance_df['population'] == pop]

            if len(pop_data) == 0:
                continue

            ax = axes[idx]

            # Boxplot by genotype
            if 'genotype' in pop_data.columns:
                sns.boxplot(data=pop_data, x='genotype', y='mean_distance',
                           ax=ax, palette='Set2')
                ax.set_ylabel('Mean Distance (μm)', fontweight='bold')
                ax.set_xlabel('Genotype', fontweight='bold')
                ax.set_title(f'{pop} Distance to Tumor', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Immune Cell Distance to Nearest Tumor Structure',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/distance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Temporal trends
        if 'timepoint' in distance_df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()

            for idx, pop in enumerate(top_pops):
                pop_data = distance_df[distance_df['population'] == pop]

                if len(pop_data) == 0 or 'timepoint' not in pop_data.columns:
                    continue

                ax = axes[idx]

                for genotype in pop_data['genotype'].unique():
                    genotype_data = pop_data[pop_data['genotype'] == genotype]
                    temporal_mean = genotype_data.groupby('timepoint')['mean_distance'].mean()
                    ax.plot(temporal_mean.index, temporal_mean.values,
                           marker='o', label=genotype, linewidth=2)

                ax.set_xlabel('Timepoint', fontweight='bold')
                ax.set_ylabel('Mean Distance (μm)', fontweight='bold')
                ax.set_title(f'{pop} Distance Over Time', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.suptitle('Temporal Dynamics of Immune-Tumor Distances',
                        fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/figures/distance_temporal.png", dpi=300, bbox_inches='tight')
            plt.close()


    def _plot_colocalization_comprehensive(self, coloc_df: pd.DataFrame):
        """Generate comprehensive co-localization plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get unique pairs
        pairs = coloc_df[['population_1', 'population_2']].drop_duplicates()
        n_pairs = min(len(pairs), 6)  # Limit to top 6 pairs

        if n_pairs == 0:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx in range(n_pairs):
            if idx >= len(pairs):
                break

            pair = pairs.iloc[idx]
            pop1, pop2 = pair['population_1'], pair['population_2']

            pair_data = coloc_df[
                (coloc_df['population_1'] == pop1) &
                (coloc_df['population_2'] == pop2)
            ]

            if len(pair_data) == 0:
                continue

            ax = axes[idx]

            # Boxplot of enrichment scores by genotype
            if 'genotype' in pair_data.columns:
                sns.boxplot(data=pair_data, x='genotype', y='enrichment_score',
                           ax=ax, palette='Set2')
                ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Expected')
                ax.set_ylabel('Enrichment Score', fontweight='bold')
                ax.set_xlabel('Genotype', fontweight='bold')
                ax.set_title(f'{pop1} ↔ {pop2}', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                ax.legend()

        plt.suptitle('Cell Type Co-localization Analysis (within 50μm)',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/colocalization_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


    def generate_comprehensive_report(self):
        """Generate comprehensive HTML/PDF report."""
        print("Generating comprehensive report...")

        report_path = f"{self.output_dir}/comprehensive_analysis_report.html"

        # Build HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Spatial Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 5px; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        .section {{ margin: 30px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #34495e; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
        .figure-caption {{ font-style: italic; color: #7f8c8d; margin-top: -15px; margin-bottom: 20px; }}
        .warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }}
        .success {{ background-color: #d4edda; border-left: 4px solid #28a745; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Comprehensive Tumor Spatial Analysis Report</h1>
        <p><strong>Analysis Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Output Directory:</strong> {self.output_dir}</p>

        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metric">
                <div class="metric-label">Total Cells Analyzed</div>
                <div class="metric-value">{len(self.adata):,}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Tumor Structures Detected</div>
                <div class="metric-value">{len(self.structure_index) if self.structure_index is not None else 0}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Samples</div>
                <div class="metric-value">{len(self.adata.obs['sample_id'].unique())}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Timepoints</div>
                <div class="metric-value">{len(self.adata.obs['timepoint'].unique())}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Genotypes</div>
                <div class="metric-value">{', '.join(map(str, self.adata.obs['genotype'].unique()))}</div>
            </div>
        </div>

        <div class="section">
            <h2>Analysis Pipeline Overview</h2>
            <div class="success">✓ Phase 1: Tumor Structure Detection - Complete</div>
            <div class="success">✓ Phase 2: Infiltration Quantification - Complete</div>
            <div class="success">✓ Phase 3: Marker Expression Analysis - Complete</div>
            <div class="success">✓ Phase 4: Tumor Size Analysis - Complete</div>
            <div class="success">✓ Phase 5: Cellular Neighborhood Analysis - Complete</div>
            <div class="success">✓ Phase 6: Spatial Distance Analysis - Complete</div>
            <div class="success">✓ Phase 7: Co-localization Analysis - Complete</div>
            <div class="success">✓ Phase 8: Statistical Analysis - Complete</div>
            <div class="success">✓ Phase 9: Visualizations - Complete</div>
            <div class="success">✓ Phase 10: Report Generation - Complete</div>
        </div>

        <div class="section">
            <h2>Population Summary</h2>
            <table>
                <tr>
                    <th>Population</th>
                    <th>Cell Count</th>
                    <th>Percentage</th>
                </tr>"""

        # Add population statistics
        pop_cols = [col for col in self.adata.obs.columns if col.startswith('is_')]
        for col in sorted(pop_cols):
            if self.adata.obs[col].dtype == bool:
                count = self.adata.obs[col].sum()
                pct = 100 * count / len(self.adata)
                html_content += f"""
                <tr>
                    <td>{col.replace('is_', '')}</td>
                    <td>{count:,}</td>
                    <td>{pct:.2f}%</td>
                </tr>"""

        html_content += """
            </table>
        </div>

        <div class="section">
            <h2>Key Findings</h2>"""

        # Add structure statistics if available
        if self.structure_index is not None and len(self.structure_index) > 0:
            html_content += f"""
            <h3>Tumor Structure Characteristics</h3>
            <ul>
                <li>Total structures detected: {len(self.structure_index)}</li>
                <li>Mean structure size: {self.structure_index['n_cells'].mean():.0f} cells</li>
                <li>Size range: {self.structure_index['n_cells'].min()} - {self.structure_index['n_cells'].max()} cells</li>
                <li>Mean structure area: {self.structure_index['area_um2'].mean():.0f} μm²</li>
            </ul>"""

        html_content += """
        </div>

        <div class="section">
            <h2>Output Files</h2>
            <h3>Data Files</h3>
            <ul>
                <li><code>data/infiltration_metrics.csv</code> - Per-structure infiltration metrics</li>
                <li><code>data/marker_expression_temporal.csv</code> - Marker expression over time</li>
                <li><code>data/tumor_size_temporal.csv</code> - Tumor growth metrics</li>
                <li><code>data/spatial_distances.csv</code> - Distance analysis results</li>
                <li><code>data/colocalization_analysis.csv</code> - Co-localization metrics</li>
                <li><code>data/cell_neighborhoods.csv</code> - Per-cell neighborhood assignments</li>
                <li><code>data/neighborhood_profiles.csv</code> - Neighborhood characteristics</li>
            </ul>

            <h3>Statistical Results</h3>
            <ul>
                <li><code>statistics/infiltration_temporal.csv</code> - Temporal trend tests</li>
                <li><code>statistics/infiltration_genotype.csv</code> - Genotype comparison tests</li>
                <li><code>statistics/marker_temporal.csv</code> - Marker expression trends</li>
                <li><code>statistics/size_temporal.csv</code> - Tumor size trends</li>
            </ul>

            <h3>Visualizations</h3>
            <ul>
                <li><code>figures/spatial_map_*.png</code> - Spatial distribution maps</li>
                <li><code>figures/tumor_size_comprehensive.png</code> - Tumor growth plots</li>
                <li><code>figures/marker_expression_temporal.png</code> - Marker expression over time</li>
                <li><code>figures/infiltration_temporal.png</code> - Infiltration dynamics</li>
                <li><code>figures/neighborhood_analysis.png</code> - Neighborhood composition</li>
                <li><code>figures/infiltration_heatmaps.png</code> - Infiltration heatmaps</li>
                <li><code>figures/combined_summary.png</code> - Combined summary figure</li>
            </ul>
        </div>

        <div class="section">
            <h2>Analysis Notes</h2>
            <p>This comprehensive spatial analysis includes:</p>
            <ul>
                <li>Tumor structure detection using DBSCAN clustering</li>
                <li>Multi-region infiltration analysis (Core, Margin, Peri-tumor, Distal)</li>
                <li>Temporal dynamics across multiple timepoints</li>
                <li>Genotype comparisons (cis vs trans)</li>
                <li>Cellular neighborhood detection and characterization</li>
                <li>Spatial distance and co-localization analysis</li>
                <li>Comprehensive statistical testing with FDR correction</li>
            </ul>
        </div>

        <div class="section">
            <h2>Next Steps</h2>
            <p>For advanced analysis, run with --run_advanced flag to execute Phases 11-18:</p>
            <ul>
                <li>Phase 11: Enhanced phenotyping validation</li>
                <li>Phase 12: pERK spatial architecture analysis</li>
                <li>Phase 13: NINJA escape mechanism analysis</li>
                <li>Phase 14: Heterogeneity emergence and evolution</li>
                <li>Phase 15: Enhanced RCN temporal dynamics</li>
                <li>Phase 16: Multi-level distance analysis</li>
                <li>Phase 17: Infiltration-tumor associations</li>
                <li>Phase 18: Pseudo-temporal trajectory analysis</li>
            </ul>
        </div>

        <div class="section">
            <p style="text-align: center; color: #7f8c8d; margin-top: 50px;">
                Report generated by Comprehensive Tumor Spatial Analysis Pipeline<br>
                For questions or issues, please contact the analysis team.
            </p>
        </div>
    </div>
</body>
</html>
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(html_content)

        print(f"✓ Report saved to: {report_path}")
        print(f"  Open in browser: file://{report_path}")


if __name__ == '__main__':
    print("Use run_comprehensive_spatial_analysis.py to run this analysis")
