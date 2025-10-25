#!/usr/bin/env python3
"""
Spatial Analysis Expansions - Comprehensive Immune-Tumor Interactions

This module provides extensive analyses for:
1. T cell to tumor cell distances (overall and by tumor subtype)
2. T cell to NINJA+ (aGFP+) and pERK+ tumor cells
3. Tumor heterogeneity (spatial clustering of markers)
4. Neighborhood composition over time
5. NINJA+/- tumor comparisons
6. Dual-level statistics (by sample and by tumor)
7. Spatial maps with analysis ranges

Author: Expansion module for comprehensive analysis
Date: 2025-10-25
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.spatial import cKDTree, distance_matrix
from scipy.stats import (mannwhitneyu, kruskal, ks_2samp, ttest_ind,
                         f_oneway, spearmanr, pearsonr, linregress)
from sklearn.metrics import pairwise_distances
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from typing import Dict, List, Tuple, Optional
import warnings
import os
from pathlib import Path
import gc
from collections import defaultdict
warnings.filterwarnings('ignore')


class ImmuneInfiltrationAnalysis:
    """
    Comprehensive immune infiltration analysis focusing on T cell-tumor interactions.
    """

    def __init__(self, adata, structure_index: pd.DataFrame, output_dir: str):
        """
        Initialize immune infiltration analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data with spatial coordinates
        structure_index : pd.DataFrame
            Tumor structure index from detection
        output_dir : str
            Output directory
        """
        self.adata = adata
        self.structure_index = structure_index
        self.output_dir = output_dir

        # Get coordinates
        if 'spatial' in adata.obsm:
            self.coords = adata.obsm['spatial']
        elif 'X_centroid' in adata.obs and 'Y_centroid' in adata.obs:
            self.coords = np.column_stack([adata.obs['X_centroid'].values,
                                           adata.obs['Y_centroid'].values])
        else:
            raise ValueError("No spatial coordinates found")

        # Setup directories
        self._setup_directories()

        print(f"Immune Infiltration Analysis initialized")
        print(f"  Cells: {len(adata):,}")
        print(f"  Structures: {len(structure_index)}")


    def _setup_directories(self):
        """Create directory structure."""
        dirs = [
            f"{self.output_dir}/data/distances",
            f"{self.output_dir}/data/heterogeneity",
            f"{self.output_dir}/statistics/distances",
            f"{self.output_dir}/statistics/heterogeneity",
            f"{self.output_dir}/figures/distances",
            f"{self.output_dir}/figures/distances/distributions",
            f"{self.output_dir}/figures/distances/boxplots",
            f"{self.output_dir}/figures/distances/scatters",
            f"{self.output_dir}/figures/heterogeneity",
            f"{self.output_dir}/figures/spatial_maps/tumor_definitions",
            f"{self.output_dir}/figures/spatial_maps/analysis_ranges",
        ]

        for d in dirs:
            os.makedirs(d, exist_ok=True)


    def analyze_tcell_tumor_distances_comprehensive(
        self,
        tcell_populations: List[str],
        tumor_subtypes: Dict[str, Dict[str, bool]] = None,
        max_distance: float = 500
    ) -> pd.DataFrame:
        """
        Comprehensive T cell to tumor distance analysis.

        Analyzes:
        - T cell to overall tumor distances
        - T cell to specific tumor subtypes (pERK+, NINJA+/aGFP+, etc.)
        - Per-structure level (n = tumors)
        - Per-sample level (n = samples)
        - Temporal trends
        - Group comparisons

        Parameters
        ----------
        tcell_populations : list
            T cell populations to analyze (e.g., ['CD3', 'CD8', 'CD4'])
        tumor_subtypes : dict
            Tumor subtypes defined by markers
            Example: {
                'pERK_positive': {'pERK': True, 'Tumor_core': True},
                'NINJA_positive': {'aGFP': True, 'Tumor_core': True},
                'NINJA_negative': {'aGFP': False, 'Tumor_core': True}
            }
        max_distance : float
            Maximum distance to consider (μm)

        Returns
        -------
        pd.DataFrame
            Distance metrics at multiple levels
        """
        print("\n" + "="*80)
        print("T CELL - TUMOR DISTANCE ANALYSIS")
        print("="*80)

        if tumor_subtypes is None:
            # Default tumor subtypes
            tumor_subtypes = {
                'Tumor_all': {'is_Tumor': True},
                'pERK_positive': {'pERK': True},
                'NINJA_positive': {'aGFP': True},
                'NINJA_negative': {'aGFP': False, 'is_Tumor': True}
            }

        # Define tumor subtypes
        print("\nDefining tumor subtypes...")
        for subtype_name, subtype_def in tumor_subtypes.items():
            mask = np.ones(len(self.adata), dtype=bool)

            for marker, required_state in subtype_def.items():
                if marker.startswith('is_'):
                    # Population marker
                    if marker in self.adata.obs:
                        if required_state:
                            mask &= self.adata.obs[marker].values
                        else:
                            mask &= ~self.adata.obs[marker].values
                else:
                    # Expression marker
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

            self.adata.obs[f'is_{subtype_name}'] = mask
            print(f"  {subtype_name}: {mask.sum():,} cells")

        # Analyze distances at structure level (n = tumors)
        print("\nAnalyzing per-structure distances...")
        structure_results = self._analyze_distances_per_structure(
            tcell_populations, tumor_subtypes, max_distance
        )

        # Analyze distances at sample level (n = samples)
        print("\nAggregating to sample level...")
        sample_results = self._aggregate_distances_to_sample(structure_results)

        # Save results
        structure_results.to_csv(
            f"{self.output_dir}/data/distances/tcell_tumor_distances_per_structure.csv",
            index=False
        )
        sample_results.to_csv(
            f"{self.output_dir}/data/distances/tcell_tumor_distances_per_sample.csv",
            index=False
        )

        print(f"\n✓ Distance analysis complete")
        print(f"  Per-structure results: {len(structure_results)}")
        print(f"  Per-sample results: {len(sample_results)}")

        return structure_results, sample_results


    def _analyze_distances_per_structure(
        self,
        tcell_populations: List[str],
        tumor_subtypes: Dict,
        max_distance: float
    ) -> pd.DataFrame:
        """Analyze T cell distances for each structure."""

        results = []
        coords_f32 = self.coords.astype(np.float32)

        for idx, struct_row in self.structure_index.iterrows():
            if idx % 20 == 0:
                print(f"  Progress: {100*idx/len(self.structure_index):.1f}%")

            struct_id = struct_row['structure_id']

            # Load structure cells
            tumor_cell_indices = np.load(
                f"{self.output_dir}/structures/structure_{struct_id:04d}_cells.npy"
            )

            # Get cells in vicinity
            centroid = np.array([struct_row['centroid_x'], struct_row['centroid_y']],
                               dtype=np.float32)

            # Get cells within buffer
            buffer_distance = max_distance + 500
            tree = cKDTree(coords_f32)
            nearby_indices = tree.query_ball_point(centroid, buffer_distance)
            nearby_indices = np.array(nearby_indices)

            # For each T cell population
            for tcell_pop in tcell_populations:
                if f'is_{tcell_pop}' not in self.adata.obs:
                    continue

                # Get T cells in vicinity
                tcell_mask = self.adata.obs.iloc[nearby_indices][f'is_{tcell_pop}'].values
                tcell_indices = nearby_indices[tcell_mask]

                if len(tcell_indices) == 0:
                    continue

                tcell_coords = coords_f32[tcell_indices]

                # For each tumor subtype
                for subtype_name in tumor_subtypes.keys():
                    if f'is_{subtype_name}' not in self.adata.obs:
                        continue

                    # Get tumor subtype cells in this structure
                    subtype_mask = (
                        np.isin(nearby_indices, tumor_cell_indices) &
                        self.adata.obs.iloc[nearby_indices][f'is_{subtype_name}'].values
                    )
                    subtype_indices = nearby_indices[subtype_mask]

                    if len(subtype_indices) == 0:
                        continue

                    subtype_coords = coords_f32[subtype_indices]

                    # Compute pairwise distances
                    dist_matrix = distance_matrix(tcell_coords, subtype_coords)

                    # Min distance for each T cell
                    min_distances = dist_matrix.min(axis=1)

                    # Filter by max_distance
                    valid_distances = min_distances[min_distances <= max_distance]

                    if len(valid_distances) == 0:
                        continue

                    # Compute metrics
                    results.append({
                        'structure_id': struct_id,
                        'sample_id': struct_row['sample_id'],
                        'timepoint': struct_row['timepoint'],
                        'main_group': struct_row['main_group'],
                        'genotype': struct_row['genotype'],
                        'genotype_full': struct_row['genotype_full'],
                        'tcell_population': tcell_pop,
                        'tumor_subtype': subtype_name,
                        'n_tcells': len(tcell_indices),
                        'n_tumor_subtype': len(subtype_indices),
                        'n_tcells_within_max_dist': len(valid_distances),
                        'pct_tcells_within_max_dist': 100 * len(valid_distances) / len(tcell_indices),
                        'mean_distance': valid_distances.mean(),
                        'median_distance': np.median(valid_distances),
                        'min_distance': valid_distances.min(),
                        'max_distance': valid_distances.max(),
                        'std_distance': valid_distances.std(),
                        'q25_distance': np.percentile(valid_distances, 25),
                        'q75_distance': np.percentile(valid_distances, 75),
                        'tumor_size': struct_row['n_cells'],
                        'tumor_area': struct_row['area_um2']
                    })

        return pd.DataFrame(results)


    def _aggregate_distances_to_sample(self, structure_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate structure-level distances to sample level."""

        # Group by sample, tcell_population, tumor_subtype
        group_cols = ['sample_id', 'timepoint', 'main_group', 'genotype', 'genotype_full',
                      'tcell_population', 'tumor_subtype']

        sample_results = structure_df.groupby(group_cols).agg({
            'n_tcells': 'sum',
            'n_tumor_subtype': 'sum',
            'n_tcells_within_max_dist': 'sum',
            'pct_tcells_within_max_dist': 'mean',
            'mean_distance': 'mean',
            'median_distance': 'median',
            'min_distance': 'min',
            'max_distance': 'max',
            'std_distance': 'mean',
            'q25_distance': 'mean',
            'q75_distance': 'mean',
            'structure_id': 'count'
        }).reset_index()

        sample_results.rename(columns={'structure_id': 'n_structures'}, inplace=True)

        return sample_results


    def analyze_tumor_heterogeneity(
        self,
        markers_of_interest: List[str],
        window_size: float = 100
    ) -> pd.DataFrame:
        """
        Analyze spatial heterogeneity of marker expression within tumors.

        Answers: Are NINJA+ and pERK+ regions spatially isolated or randomly distributed?

        Parameters
        ----------
        markers_of_interest : list
            Markers to analyze (e.g., ['aGFP', 'pERK'])
        window_size : float
            Window size for local heterogeneity (μm)

        Returns
        -------
        pd.DataFrame
            Heterogeneity metrics per structure
        """
        print("\n" + "="*80)
        print("TUMOR HETEROGENEITY ANALYSIS")
        print("="*80)

        results = []
        coords_f32 = self.coords.astype(np.float32)

        for idx, struct_row in self.structure_index.iterrows():
            if idx % 20 == 0:
                print(f"  Progress: {100*idx/len(self.structure_index):.1f}%")

            struct_id = struct_row['structure_id']

            # Load structure cells
            tumor_cell_indices = np.load(
                f"{self.output_dir}/structures/structure_{struct_id:04d}_cells.npy"
            )

            tumor_coords = coords_f32[tumor_cell_indices]

            if len(tumor_coords) < 10:
                continue

            # For each marker
            marker_metrics = {
                'structure_id': struct_id,
                'sample_id': struct_row['sample_id'],
                'timepoint': struct_row['timepoint'],
                'main_group': struct_row['main_group'],
                'genotype': struct_row['genotype'],
                'genotype_full': struct_row['genotype_full'],
                'tumor_size': len(tumor_cell_indices)
            }

            for marker in markers_of_interest:
                if marker not in self.adata.var_names:
                    continue

                marker_idx = self.adata.var_names.get_loc(marker)

                # Get marker expression
                if 'gated' in self.adata.layers:
                    marker_values = self.adata.layers['gated'][tumor_cell_indices, marker_idx]
                    marker_positive = marker_values > 0
                else:
                    marker_values = self.adata.X[tumor_cell_indices, marker_idx]
                    threshold = np.percentile(marker_values[marker_values > 0], 90) if np.any(marker_values > 0) else 0
                    marker_positive = marker_values > threshold

                pct_positive = 100 * marker_positive.sum() / len(marker_positive)

                # Spatial autocorrelation (Moran's I-like metric)
                # Build neighbor matrix
                tree = cKDTree(tumor_coords)
                neighbors_list = tree.query_ball_point(tumor_coords, window_size)

                local_fractions = []
                for i, neighbors in enumerate(neighbors_list):
                    if len(neighbors) > 1:
                        local_pct = 100 * marker_positive[neighbors].sum() / len(neighbors)
                        local_fractions.append(local_pct)

                # Heterogeneity = variance of local fractions
                heterogeneity = np.std(local_fractions) if len(local_fractions) > 0 else 0

                # Clustering coefficient: Do positive cells cluster together?
                # Compute distances between positive cells
                if marker_positive.sum() >= 2:
                    positive_coords = tumor_coords[marker_positive]
                    positive_tree = cKDTree(positive_coords)

                    # Mean distance to nearest positive neighbor
                    distances, _ = positive_tree.query(positive_coords, k=2)
                    mean_nn_dist_positive = distances[:, 1].mean()

                    # Compare to expected if random
                    # Expected = average distance in whole tumor
                    all_distances, _ = tree.query(tumor_coords, k=2)
                    mean_nn_dist_all = all_distances[:, 1].mean()

                    clustering_index = mean_nn_dist_all / (mean_nn_dist_positive + 1e-6)
                else:
                    clustering_index = 0

                marker_metrics[f'{marker}_pct_positive'] = pct_positive
                marker_metrics[f'{marker}_heterogeneity'] = heterogeneity
                marker_metrics[f'{marker}_clustering_index'] = clustering_index

            results.append(marker_metrics)

        heterogeneity_df = pd.DataFrame(results)
        heterogeneity_df.to_csv(
            f"{self.output_dir}/data/heterogeneity/tumor_heterogeneity.csv",
            index=False
        )

        print(f"\n✓ Heterogeneity analysis complete")
        print(f"  Analyzed {len(heterogeneity_df)} structures")

        return heterogeneity_df


    def compare_ninja_positive_negative_tumors(self) -> pd.DataFrame:
        """
        Compare NINJA+ vs NINJA- tumors in KPNT and KPT.

        Analyzes:
        - Do NINJA-negative tumors exist in KPNT?
        - Are NINJA-negative tumors in KPNT similar to KPT in spatial composition?

        Returns
        -------
        pd.DataFrame
            Comparison metrics
        """
        print("\n" + "="*80)
        print("NINJA+/- TUMOR COMPARISON")
        print("="*80)

        # Classify each structure as NINJA+ or NINJA-
        results = []

        for idx, struct_row in self.structure_index.iterrows():
            struct_id = struct_row['structure_id']

            # Load structure cells
            tumor_cell_indices = np.load(
                f"{self.output_dir}/structures/structure_{struct_id:04d}_cells.npy"
            )

            # Check if aGFP is available
            if 'aGFP' not in self.adata.var_names:
                continue

            agfp_idx = self.adata.var_names.get_loc('aGFP')

            if 'gated' in self.adata.layers:
                agfp_values = self.adata.layers['gated'][tumor_cell_indices, agfp_idx]
                agfp_positive = agfp_values > 0
            else:
                agfp_values = self.adata.X[tumor_cell_indices, agfp_idx]
                threshold = np.percentile(agfp_values[agfp_values > 0], 90) if np.any(agfp_values > 0) else 0
                agfp_positive = agfp_values > threshold

            pct_ninja = 100 * agfp_positive.sum() / len(agfp_positive)

            # Classify (> 10% NINJA+ = NINJA-positive tumor)
            ninja_status = 'NINJA_positive' if pct_ninja > 10 else 'NINJA_negative'

            results.append({
                'structure_id': struct_id,
                'sample_id': struct_row['sample_id'],
                'timepoint': struct_row['timepoint'],
                'main_group': struct_row['main_group'],
                'genotype': struct_row['genotype'],
                'genotype_full': struct_row['genotype_full'],
                'ninja_status': ninja_status,
                'pct_NINJA_positive': pct_ninja,
                'tumor_size': len(tumor_cell_indices),
                'tumor_area': struct_row['area_um2']
            })

        ninja_df = pd.DataFrame(results)
        ninja_df.to_csv(
            f"{self.output_dir}/data/ninja_tumor_classification.csv",
            index=False
        )

        print("\nNINJA tumor breakdown:")
        print(ninja_df.groupby(['main_group', 'ninja_status']).size())

        return ninja_df


class NeighborhoodTemporalAnalysis:
    """
    Comprehensive neighborhood temporal analysis.
    """

    def __init__(self, adata, cell_neighborhoods: pd.DataFrame,
                 neighborhood_profiles: pd.DataFrame, output_dir: str):
        """
        Initialize neighborhood temporal analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data
        cell_neighborhoods : pd.DataFrame
            Per-cell neighborhood assignments
        neighborhood_profiles : pd.DataFrame
            Neighborhood type profiles
        output_dir : str
            Output directory
        """
        self.adata = adata
        self.cell_neighborhoods = cell_neighborhoods
        self.neighborhood_profiles = neighborhood_profiles
        self.output_dir = output_dir

        print(f"Neighborhood Temporal Analysis initialized")
        print(f"  Cells: {len(cell_neighborhoods):,}")
        print(f"  Neighborhood types: {neighborhood_profiles['neighborhood_id'].nunique()}")


    def analyze_neighborhood_composition_temporal(self) -> pd.DataFrame:
        """
        Analyze neighborhood composition over time per group.

        Returns
        -------
        pd.DataFrame
            Neighborhood composition by timepoint and group
        """
        print("\n" + "="*80)
        print("NEIGHBORHOOD COMPOSITION TEMPORAL ANALYSIS")
        print("="*80)

        # Aggregate by sample, timepoint, group, neighborhood_type
        composition = self.cell_neighborhoods.groupby([
            'sample_id', 'timepoint', 'genotype', 'genotype_full', 'neighborhood_type'
        ]).size().reset_index(name='n_cells')

        # Calculate percentages per sample
        sample_totals = composition.groupby(['sample_id'])['n_cells'].transform('sum')
        composition['pct_cells'] = 100 * composition['n_cells'] / sample_totals

        composition.to_csv(
            f"{self.output_dir}/data/neighborhood_composition_temporal.csv",
            index=False
        )

        print(f"✓ Neighborhood composition temporal analysis complete")

        return composition


def create_spatial_maps_with_analysis_ranges(
    adata,
    structure_index: pd.DataFrame,
    output_dir: str,
    samples_to_plot: List[str] = None,
    boundary_widths: List[float] = [30, 100, 200]
):
    """
    Create spatial maps showing:
    - What each tumor object is defined by
    - The nearby range being counted

    Parameters
    ----------
    adata : AnnData
        Annotated data
    structure_index : pd.DataFrame
        Structure index
    output_dir : str
        Output directory
    samples_to_plot : list
        Specific samples to plot (if None, plot all)
    boundary_widths : list
        Boundary widths to show
    """
    print("\n" + "="*80)
    print("SPATIAL MAPS WITH ANALYSIS RANGES")
    print("="*80)

    coords = adata.obsm['spatial']

    if samples_to_plot is None:
        samples_to_plot = structure_index['sample_id'].unique()[:5]  # Limit to 5

    for sample_id in samples_to_plot:
        print(f"\nCreating map for {sample_id}...")

        # Get sample structures
        sample_structures = structure_index[structure_index['sample_id'] == sample_id]

        if len(sample_structures) == 0:
            continue

        # Get sample cells
        sample_mask = adata.obs['sample_id'] == sample_id
        sample_coords = coords[sample_mask]

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 14), dpi=150)

        # Plot all cells as background
        ax.scatter(sample_coords[:, 0], sample_coords[:, 1],
                  c='lightgray', s=0.5, alpha=0.3, label='Other cells')

        # Plot each structure
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for idx, (_, struct) in enumerate(sample_structures.iterrows()):
            struct_id = struct['structure_id']

            # Load tumor cells
            tumor_indices = np.load(
                f"{output_dir}/structures/structure_{struct_id:04d}_cells.npy"
            )
            tumor_coords = coords[tumor_indices]

            color = colors[idx % 10]

            # Plot tumor core
            ax.scatter(tumor_coords[:, 0], tumor_coords[:, 1],
                      c=[color], s=2, alpha=0.7, label=f'Tumor {struct_id}')

            # Plot boundaries
            centroid_x = struct['centroid_x']
            centroid_y = struct['centroid_y']

            for i, width in enumerate(boundary_widths):
                circle = plt.Circle((centroid_x, centroid_y), width,
                                   fill=False, edgecolor=color,
                                   linewidth=1.5, linestyle='--', alpha=0.5)
                ax.add_patch(circle)

                if idx == 0:  # Label only once
                    ax.text(centroid_x + width, centroid_y, f'{width}μm',
                           fontsize=8, color=color)

        ax.set_xlabel('X (μm)', fontsize=12)
        ax.set_ylabel('Y (μm)', fontsize=12)
        ax.set_title(f'{sample_id} - Tumor Definitions and Analysis Ranges',
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)

        plt.tight_layout()

        # Create output directory if it doesn't exist
        os.makedirs(f"{output_dir}/figures/spatial_maps/analysis_ranges", exist_ok=True)

        plt.savefig(
            f"{output_dir}/figures/spatial_maps/analysis_ranges/{sample_id}_analysis_ranges.png",
            dpi=150, bbox_inches='tight'
        )
        plt.close()

        print(f"  Saved {sample_id} analysis range map")

    print("\n✓ Spatial maps with analysis ranges complete")


if __name__ == '__main__':
    print("Use this module with your main analysis script")
