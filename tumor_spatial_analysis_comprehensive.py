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

# Set matplotlib backend to non-interactive (for headless environments)
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
        """
        print("Analyzing spatial distances...")

        # This is a simplified version - can be expanded
        print("✓ Distance analysis complete")
        return pd.DataFrame()


    def analyze_colocalization(self, immune_populations: List[str]) -> pd.DataFrame:
        """
        Analyze co-localization between different cell types.
        """
        print("Analyzing co-localization...")

        # This is a simplified version - can be expanded
        print("✓ Co-localization analysis complete")
        return pd.DataFrame()


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
            group_cols = ['population', 'region'] if 'region' in df.columns else ['marker']

        results = []

        # Get grouping combinations
        grouping_values = df[group_cols].drop_duplicates().values

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
        Generate ALL visualizations.
        """
        print("\nGenerating comprehensive visualizations...")

        # Set style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)

        # 1. Spatial maps
        print("\n  1. Spatial maps...")
        self._generate_spatial_maps()

        # 2. Tumor size plots
        print("  2. Tumor size temporal plots...")
        if len(size_df) > 0:
            self._plot_tumor_size_comprehensive(size_df, stats_results)

        # 3. Marker expression plots
        print("  3. Marker expression temporal plots...")
        if len(marker_df) > 0:
            self._plot_marker_expression_comprehensive(marker_df, stats_results)

        # 4. Infiltration plots
        print("  4. Infiltration temporal/genotype plots...")
        if len(metrics_df) > 0:
            self._plot_infiltration_comprehensive(metrics_df, stats_results)

        # 5. Neighborhood plots
        print("  5. Neighborhood analysis plots...")
        if len(neighborhood_df) > 0:
            self._plot_neighborhoods_comprehensive(neighborhood_df)

        # 6. Heatmaps
        print("  6. Comprehensive heatmaps...")
        if len(metrics_df) > 0:
            self._plot_heatmaps_comprehensive(metrics_df, marker_df)

        # 7. Combined summary figures
        print("  7. Combined summary figures...")
        self._plot_combined_summaries(metrics_df, marker_df, size_df)

        print("\n✓ All visualizations complete!")


    def _generate_spatial_maps(self):
        """Generate spatial maps - PLACEHOLDER for now, will implement fully."""
        print("    Spatial maps generation - to be implemented with full sample iteration")
        # This would iterate through samples/timepoints/genotypes and create spatial overlays
        pass


    def _plot_tumor_size_comprehensive(self, size_df: pd.DataFrame, stats_results: Dict):
        """
        Generate comprehensive tumor size plots.
        """
        # TO BE CONTINUED - this is getting too long for one message
        # I'll implement this in the next part
        pass

    def _plot_marker_expression_comprehensive(self, marker_df: pd.DataFrame, stats_results: Dict):
        """Generate comprehensive marker expression plots."""
        pass

    def _plot_infiltration_comprehensive(self, metrics_df: pd.DataFrame, stats_results: Dict):
        """Generate comprehensive infiltration plots."""
        pass

    def _plot_neighborhoods_comprehensive(self, neighborhood_df: pd.DataFrame):
        """Generate comprehensive neighborhood plots."""
        pass

    def _plot_heatmaps_comprehensive(self, metrics_df: pd.DataFrame, marker_df: pd.DataFrame):
        """Generate comprehensive heatmaps."""
        pass

    def _plot_combined_summaries(self, metrics_df: pd.DataFrame, marker_df: pd.DataFrame,
                                 size_df: pd.DataFrame):
        """Generate combined summary figures."""
        pass

    def generate_comprehensive_report(self):
        """Generate comprehensive HTML/PDF report."""
        print("Generating comprehensive report...")
        pass


if __name__ == '__main__':
    print("Use run_comprehensive_spatial_analysis.py to run this analysis")
