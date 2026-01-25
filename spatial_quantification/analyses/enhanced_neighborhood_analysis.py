"""
Enhanced Neighborhood Analysis
Focus on pERK+/- and NINJA+/- region-specific neighborhood composition
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import warnings


class EnhancedNeighborhoodAnalysis:
    """
    Enhanced neighborhood analysis focusing on marker-specific regions.

    Key features:
    - Identify distinct pERK+/- and NINJA+/- regions within tumors
    - Two-step infiltration: % immune in region + distance/depth
    - Neighborhood composition within regions
    - Per-cell neighborhood comparison (pERK+ vs pERK- cells)
    """

    def __init__(self, adata, config: Dict, output_dir: Path, tumor_structures: Dict = None):
        """
        Initialize enhanced neighborhood analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data with phenotypes and spatial coordinates
        config : dict
            Configuration dictionary
        output_dir : Path
            Output directory
        tumor_structures : dict
            Pre-computed tumor structures (from PerTumorAnalysis)
        """
        self.adata = adata
        self.config = config
        self.output_dir = Path(output_dir) / 'enhanced_neighborhoods'
        self.plots_dir = self.output_dir / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Use provided tumor structures or detect new ones
        self.tumor_structures = tumor_structures or {}
        self.tumor_config = config.get('structure_definition', config.get('tumor_definition', {}))

        # Markers to analyze - read from config
        enh_config = config.get('enhanced_neighborhoods', {})
        marker_configs = enh_config.get('markers', [])

        # Use config if available, otherwise default
        if marker_configs:
            self.markers_config = marker_configs
        else:
            # Default markers
            self.markers_config = [
                {'name': 'pERK', 'pos_col': 'is_pERK_positive_tumor', 'neg_col': 'is_pERK_negative_tumor'},
                {'name': 'NINJA', 'pos_col': 'is_AGFP_positive_tumor', 'neg_col': 'is_AGFP_negative_tumor'}
            ]

        # Cell types for neighborhood analysis - read from config
        cell_type_list = enh_config.get('cell_types', [])

        if cell_type_list:
            self.cell_types = [f'is_{ct}' if not ct.startswith('is_') else ct for ct in cell_type_list]
        else:
            # Default cell types
            self.cell_types = [
                'is_Tumor', 'is_pERK_positive_tumor', 'is_AGFP_positive_tumor', 'is_Ki67_positive_tumor',
                'is_CD45_positive', 'is_CD3_positive', 'is_CD8_T_cells', 'is_CD4_T_cells'
            ]

        # Immune cell types for neighborhood analysis - read from config
        immune_type_list = enh_config.get('immune_cells', [])

        if immune_type_list:
            self.immunetypes = [f'is_{ct}' if not ct.startswith('is_') else ct for ct in immune_type_list]
        else:
            # Default cell types
            self.immunetypes = [
                'is_CD45_positive', 'is_CD3_positive', 'is_CD8_T_cells', 'is_CD4_T_cells'
            ]

        # Neighborhood radius
        self.neighborhood_radius = 50  # μm

        # Storage
        self.results = {}
        self.marker_regions = {}  # Store identified regions

    def run(self):
        """Run complete enhanced neighborhood analysis."""
        print("\n" + "="*80)
        print("ENHANCED NEIGHBORHOOD ANALYSIS")
        print("="*80)

        # Ensure tumor structures are available
        if not self.tumor_structures:
            print("\n1. Detecting tumor structures...")
            self._detect_tumor_structures()
        else:
            print("\n1. Using provided tumor structures...")

        # Identify marker-specific regions
        print("\n2. Identifying marker-specific regions...")
        self._identify_marker_regions()

        # Calculate infiltration within regions
        print("\n3. Calculating regional infiltration...")
        self._calculate_regional_infiltration()

        # Calculate neighborhood composition within regions
        print("\n4. Calculating regional neighborhood composition...")
        self._calculate_regional_neighborhoods()

        # Per-cell neighborhood comparison
        print("\n5. Comparing per-cell neighborhoods (marker+ vs marker-)...")
        self._calculate_per_cell_neighborhoods()

        # Save results
        self._save_results()

        print("\n✓ Enhanced neighborhood analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
        print("="*80 + "\n")

        return self.results

    def _detect_tumor_structures(self):
        """Detect tumor structures using DBSCAN if not provided."""
        tumor_def = self.tumor_config
        tumor_pheno = tumor_def.get('base_phenotype', 'Tumor')
        tumor_col = f'is_{tumor_pheno}'

        if tumor_col not in self.adata.obs.columns:
            raise ValueError(f"Tumor phenotype '{tumor_pheno}' not found")

        struct_config = tumor_def.get('structure_detection', {})
        eps = struct_config.get('eps', 800)
        min_samples = struct_config.get('min_samples', 250)
        min_cluster_size = struct_config.get('min_cluster_size', 250)

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]
            sample_coords = self.adata.obsm['spatial'][sample_mask.values]

            tumor_mask = sample_data[tumor_col].values
            if tumor_mask.sum() < min_samples:
                continue

            tumor_coords = sample_coords[tumor_mask]
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(tumor_coords)
            labels = clustering.labels_

            valid_labels = [label for label in set(labels) - {-1}
                          if (labels == label).sum() >= min_cluster_size]

            structure_labels = np.full(len(sample_data), -1)
            tumor_indices = np.where(tumor_mask)[0]
            for label in valid_labels:
                cluster_mask = labels == label
                structure_labels[tumor_indices[cluster_mask]] = label

            self.tumor_structures[sample] = structure_labels

    def _identify_marker_regions(self):
        """Identify distinct marker+ and marker- regions within tumor structures."""
        for marker_def in self.markers_config:
            marker_name = marker_def['name']
            pos_col = marker_def['pos_col']
            neg_col = marker_def['neg_col']

            print(f"    Processing {marker_name} regions...")

            region_results = []

            for sample in self.tumor_structures.keys():
                sample_mask = self.adata.obs['sample_id'] == sample
                sample_data = self.adata.obs[sample_mask]
                sample_coords = self.adata.obsm['spatial'][sample_mask.values]

                structure_labels = self.tumor_structures[sample]
                unique_structures = set(structure_labels) - {-1}

                for structure_id in unique_structures:
                    structure_mask = structure_labels == structure_id
                    structure_data = sample_data[structure_mask]
                    structure_coords = sample_coords[structure_mask]

                    # Get marker+ and marker- cells
                    if pos_col not in structure_data.columns or neg_col not in structure_data.columns:
                        continue

                    pos_mask = structure_data[pos_col].values
                    neg_mask = structure_data[neg_col].values

                    if pos_mask.sum() < 10 or neg_mask.sum() < 10:
                        continue

                    # Cluster marker+ cells into regions
                    pos_coords = structure_coords[pos_mask]
                    pos_clustering = DBSCAN(eps=30, min_samples=10).fit(pos_coords)
                    pos_regions = pos_clustering.labels_

                    # Count regions
                    n_pos_regions = len(set(pos_regions) - {-1})

                    # Cluster marker- cells into regions
                    neg_coords = structure_coords[neg_mask]
                    neg_clustering = DBSCAN(eps=30, min_samples=10).fit(neg_coords)
                    neg_regions = neg_clustering.labels_

                    n_neg_regions = len(set(neg_regions) - {-1})

                    # Store region info
                    key = (sample, structure_id, marker_name)
                    self.marker_regions[key] = {
                        'pos_coords': pos_coords,
                        'pos_regions': pos_regions,
                        'neg_coords': neg_coords,
                        'neg_regions': neg_regions,
                        'structure_coords': structure_coords,
                        'structure_data': structure_data
                    }

                    region_results.append({
                        'sample_id': sample,
                        'tumor_id': int(structure_id),
                        'marker': marker_name,
                        'n_marker_positive_cells': int(pos_mask.sum()),
                        'n_marker_negative_cells': int(neg_mask.sum()),
                        'n_positive_regions': int(n_pos_regions),
                        'n_negative_regions': int(n_neg_regions),
                        'percent_marker_positive': pos_mask.sum() / len(structure_data) * 100,
                        'timepoint': structure_data['timepoint'].iloc[0] if 'timepoint' in structure_data.columns else np.nan,
                        'group': structure_data['group'].iloc[0] if 'group' in structure_data.columns else '',
                        'main_group': structure_data['main_group'].iloc[0] if 'main_group' in structure_data.columns else ''
                    })

            if region_results:
                df = pd.DataFrame(region_results)
                self.results[f'{marker_name}_region_identification'] = df
                print(f"      ✓ Identified regions for {len(region_results)} tumor structures")

    def _calculate_regional_infiltration(self):
        """Calculate infiltration within marker+ and marker- regions."""
        #immune_pops = ['CD8_T_cells', 'CD3_positive', 'CD45_positive', 'CD4_T_cells']
        immune_pops = self.immunetypes
        for marker_def in self.markers_config:
            marker_name = marker_def['name']

            print(f"    Processing {marker_name} infiltration...")

            infiltration_results = []

            for key, region_data in self.marker_regions.items():
                sample, structure_id, reg_marker = key

                if reg_marker != marker_name:
                    continue

                structure_data = region_data['structure_data']
                structure_coords = region_data['structure_coords']
                pos_coords = region_data['pos_coords']
                neg_coords = region_data['neg_coords']

                # Build KDTrees for distance calculations
                if len(pos_coords) > 0:
                    pos_tree = cKDTree(pos_coords)
                else:
                    pos_tree = None

                if len(neg_coords) > 0:
                    neg_tree = cKDTree(neg_coords)
                else:
                    neg_tree = None

                result = {
                    'sample_id': sample,
                    'tumor_id': int(structure_id),
                    'marker': marker_name,
                    'n_marker_positive': len(pos_coords),
                    'n_marker_negative': len(neg_coords),
                    'timepoint': structure_data['timepoint'].iloc[0] if 'timepoint' in structure_data.columns else np.nan,
                    'group': structure_data['group'].iloc[0] if 'group' in structure_data.columns else '',
                    'main_group': structure_data['main_group'].iloc[0] if 'main_group' in structure_data.columns else ''
                }

                # For each immune population
                for immune_pop in immune_pops:
                    #immune_col = f'is_{immune_pop}'
                    immune_col = immune_pop
                    if immune_col not in structure_data.columns:
                        continue

                    immune_mask = structure_data[immune_col].values
                    immune_coords = structure_coords[immune_mask]

                    if len(immune_coords) == 0:
                        result[f'{immune_pop}_in_pos_region_count'] = 0
                        result[f'{immune_pop}_in_pos_region_percent'] = 0.0
                        result[f'{immune_pop}_in_neg_region_count'] = 0
                        result[f'{immune_pop}_in_neg_region_percent'] = 0.0
                        result[f'{immune_pop}_mean_dist_to_pos'] = np.nan
                        result[f'{immune_pop}_mean_dist_to_neg'] = np.nan
                        continue

                    # Count immune cells near marker+ region (within 50 μm)
                    if pos_tree is not None:
                        dists_to_pos, _ = pos_tree.query(immune_coords, k=1)
                        n_near_pos = (dists_to_pos < 50).sum()
                        mean_dist_to_pos = np.mean(dists_to_pos)
                    else:
                        n_near_pos = 0
                        mean_dist_to_pos = np.nan

                    # Count immune cells near marker- region (within 50 μm)
                    if neg_tree is not None:
                        dists_to_neg, _ = neg_tree.query(immune_coords, k=1)
                        n_near_neg = (dists_to_neg < 50).sum()
                        mean_dist_to_neg = np.mean(dists_to_neg)
                    else:
                        n_near_neg = 0
                        mean_dist_to_neg = np.nan

                    # Percentages
                    total_immune = len(immune_coords)
                    percent_near_pos = n_near_pos / total_immune * 100 if total_immune > 0 else 0
                    percent_near_neg = n_near_neg / total_immune * 100 if total_immune > 0 else 0

                    result[f'{immune_pop}_in_pos_region_count'] = int(n_near_pos)
                    result[f'{immune_pop}_in_pos_region_percent'] = percent_near_pos
                    result[f'{immune_pop}_in_neg_region_count'] = int(n_near_neg)
                    result[f'{immune_pop}_in_neg_region_percent'] = percent_near_neg
                    result[f'{immune_pop}_mean_dist_to_pos'] = mean_dist_to_pos
                    result[f'{immune_pop}_mean_dist_to_neg'] = mean_dist_to_neg

                infiltration_results.append(result)

            if infiltration_results:
                df = pd.DataFrame(infiltration_results)
                self.results[f'{marker_name}_regional_infiltration'] = df
                print(f"      ✓ Calculated infiltration for {len(infiltration_results)} structures")

    def _calculate_regional_neighborhoods(self):
        """Calculate neighborhood composition within marker+ and marker- regions."""
        for marker_def in self.markers_config:
            marker_name = marker_def['name']

            print(f"    Processing {marker_name} neighborhoods...")

            neighborhood_results = []

            for key, region_data in self.marker_regions.items():
                sample, structure_id, reg_marker = key

                if reg_marker != marker_name:
                    continue

                structure_data = region_data['structure_data']
                structure_coords = region_data['structure_coords']
                pos_coords = region_data['pos_coords']
                neg_coords = region_data['neg_coords']

                # Calculate neighborhood composition within marker+ region
                pos_composition = self._calculate_neighborhood_composition(
                    structure_coords, structure_data, pos_coords
                )

                # Calculate neighborhood composition within marker- region
                neg_composition = self._calculate_neighborhood_composition(
                    structure_coords, structure_data, neg_coords
                )

                result = {
                    'sample_id': sample,
                    'tumor_id': int(structure_id),
                    'marker': marker_name,
                    'timepoint': structure_data['timepoint'].iloc[0] if 'timepoint' in structure_data.columns else np.nan,
                    'group': structure_data['group'].iloc[0] if 'group' in structure_data.columns else '',
                    'main_group': structure_data['main_group'].iloc[0] if 'main_group' in structure_data.columns else ''
                }

                # Add composition metrics
                for cell_type in self.cell_types:
                    clean_name = cell_type.replace('is_', '')
                    result[f'pos_region_{clean_name}_fraction'] = pos_composition.get(cell_type, 0)
                    result[f'neg_region_{clean_name}_fraction'] = neg_composition.get(cell_type, 0)

                neighborhood_results.append(result)

            if neighborhood_results:
                df = pd.DataFrame(neighborhood_results)
                self.results[f'{marker_name}_regional_neighborhoods'] = df
                print(f"      ✓ Calculated neighborhood composition for {len(neighborhood_results)} structures")

    def _calculate_neighborhood_composition(self, all_coords: np.ndarray,
                                          all_data: pd.DataFrame,
                                          region_coords: np.ndarray) -> Dict:
        """Calculate cell type composition in neighborhood around region."""
        if len(region_coords) == 0:
            return {}

        # Build KDTree for all cells
        tree = cKDTree(all_coords)

        # For each cell in region, find neighbors within radius
        neighbor_indices = tree.query_ball_point(region_coords, self.neighborhood_radius)

        # Flatten to get all neighbor indices
        all_neighbor_idx = set()
        for indices in neighbor_indices:
            all_neighbor_idx.update(indices)

        if len(all_neighbor_idx) == 0:
            return {}

        # Get cell types of neighbors
        neighbor_data = all_data.iloc[list(all_neighbor_idx)]

        # Calculate composition
        composition = {}
        for cell_type in self.cell_types:
            if cell_type in neighbor_data.columns:
                fraction = neighbor_data[cell_type].sum() / len(neighbor_data)
                composition[cell_type] = fraction

        return composition

    def _calculate_per_cell_neighborhoods(self):
        """Calculate per-cell neighborhood composition for marker+ vs marker- cells."""
        for marker_def in self.markers_config:
            marker_name = marker_def['name']
            pos_col = marker_def['pos_col']
            neg_col = marker_def['neg_col']

            print(f"    Processing {marker_name} per-cell neighborhoods...")

            per_cell_results = []

            for key, region_data in self.marker_regions.items():
                sample, structure_id, reg_marker = key

                if reg_marker != marker_name:
                    continue

                structure_data = region_data['structure_data']
                structure_coords = region_data['structure_coords']

                # Build KDTree for all cells in structure
                tree = cKDTree(structure_coords)

                # For each cell, calculate neighborhood composition
                for idx, (coord, row) in enumerate(zip(structure_coords, structure_data.itertuples())):
                    # Find neighbors
                    neighbor_idx = tree.query_ball_point(coord, self.neighborhood_radius)

                    if len(neighbor_idx) <= 1:  # Only self
                        continue

                    # Get neighbor data
                    neighbor_data = structure_data.iloc[neighbor_idx]

                    # Cell type
                    is_marker_pos = getattr(row, pos_col.replace('is_', ''), False)

                    result = {
                        'sample_id': sample,
                        'tumor_id': int(structure_id),
                        'marker': marker_name,
                        'cell_marker_status': 'positive' if is_marker_pos else 'negative',
                        'n_neighbors': len(neighbor_idx),
                        'timepoint': structure_data['timepoint'].iloc[0] if 'timepoint' in structure_data.columns else np.nan,
                        'group': structure_data['group'].iloc[0] if 'group' in structure_data.columns else '',
                        'main_group': structure_data['main_group'].iloc[0] if 'main_group' in structure_data.columns else ''
                    }

                    # Calculate neighbor composition
                    for cell_type in self.cell_types:
                        if cell_type in neighbor_data.columns:
                            fraction = neighbor_data[cell_type].sum() / len(neighbor_data)
                            clean_name = cell_type.replace('is_', '')
                            result[f'neighbor_{clean_name}_fraction'] = fraction

                    per_cell_results.append(result)

                    # Limit to avoid excessive output
                    if len(per_cell_results) > 100000:
                        break

            if per_cell_results:
                # Save INDIVIDUAL cell data (not aggregated)
                df = pd.DataFrame(per_cell_results)

                # Add cell index for tracking
                df['cell_index'] = range(len(df))

                self.results[f'{marker_name}_per_cell_neighborhoods_individual'] = df
                print(f"      ✓ Calculated per-cell neighborhoods for {len(per_cell_results)} individual cells")

                # Also save aggregated summary for quick overview
                agg_df = df.groupby(['sample_id', 'tumor_id', 'marker', 'cell_marker_status',
                                    'timepoint', 'group', 'main_group']).agg({
                    'n_neighbors': 'mean',
                    **{col: 'mean' for col in df.columns if col.startswith('neighbor_')}
                }).reset_index()

                self.results[f'{marker_name}_per_cell_neighborhoods_summary'] = agg_df

    def _save_results(self):
        """Save all results to files."""
        for name, df in self.results.items():
            output_path = self.output_dir / f'{name}.csv'
            df.to_csv(output_path, index=False)

        print(f"\n  ✓ Saved {len(self.results)} result datasets")
