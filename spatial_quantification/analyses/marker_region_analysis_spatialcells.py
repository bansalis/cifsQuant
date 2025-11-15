"""
Marker-Based Regional Analysis with SpatialCells
Detect and analyze marker-defined spatial regions (e.g., pERK+, pERK-, Ki67+/-)
and characterize immune cell enrichment within these heterogeneous zones
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

try:
    import spatialcells as spc
    HAS_SPATIALCELLS = True
except ImportError:
    HAS_SPATIALCELLS = False


class MarkerRegionAnalysisSpatialCells:
    """
    Detect and analyze marker-defined spatial regions using SpatialCells.

    Key features:
    - Detect pERK+, pERK-, Ki67+, Ki67- spatial communities
    - Create geometric boundaries for marker regions
    - Analyze immune cell enrichment in marker+ vs marker- regions
    - Compare regional composition and immune infiltration
    - Detect holes/gaps within marker regions
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        """
        Initialize marker region analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data with phenotypes and spatial coordinates
        config : dict
            Configuration dictionary
        output_dir : Path
            Output directory
        """
        if not HAS_SPATIALCELLS:
            raise ImportError("SpatialCells required. Install with: pip install spatialcells")

        self.adata = adata
        self.config = config.get('marker_region_analysis', {})
        self.output_dir = Path(output_dir) / 'marker_region_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare coordinate columns for SpatialCells
        if 'spatial' in self.adata.obsm:
            self.adata.obs['X_centroid'] = self.adata.obsm['spatial'][:, 0]
            self.adata.obs['Y_centroid'] = self.adata.obsm['spatial'][:, 1]

        # Storage
        self.results = {}
        self.marker_boundaries = {}  # sample -> marker -> {region_id: boundary}
        self.marker_regions = {}  # sample -> marker -> region assignments

    def run(self):
        """Run complete marker region analysis."""
        print("\n" + "="*80)
        print("MARKER REGION ANALYSIS (SpatialCells)")
        print("="*80)

        if not self.config.get('enabled', True):
            print("  ⚠ Marker region analysis disabled in config")
            return {}

        # Detect marker-defined regions
        print("\n1. Detecting marker-defined spatial regions...")
        self._detect_marker_regions()

        # Analyze regional composition
        print("\n2. Analyzing regional composition...")
        self._analyze_regional_composition()

        # Compare marker+ vs marker- regions
        print("\n3. Comparing marker+ vs marker- regions...")
        self._compare_marker_regions()

        # Analyze immune enrichment
        print("\n4. Analyzing immune cell enrichment...")
        self._analyze_immune_enrichment()

        # Detect and analyze holes in marker regions
        if self.config.get('analyze_holes', False):
            print("\n5. Analyzing holes in marker regions...")
            self._analyze_region_holes()

        # Save results
        self._save_results()

        print("\n✓ Marker region analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
        print("="*80 + "\n")

        return self.results

    def _detect_marker_regions(self):
        """Detect spatial regions defined by marker expression."""
        markers = self.config.get('markers', [])

        if not markers:
            # Default markers
            markers = [
                {'name': 'pERK', 'positive_col': 'is_pERK_positive_tumor', 'negative_col': 'is_pERK_negative_tumor'},
                {'name': 'Ki67', 'positive_col': 'is_Ki67_positive_tumor', 'negative_col': 'is_Ki67_negative_tumor'},
                {'name': 'NINJA', 'positive_col': 'is_AGFP_positive_tumor', 'negative_col': 'is_AGFP_negative_tumor'}
            ]

        detection_config = self.config.get('region_detection', {})
        eps = detection_config.get('eps', 55)
        min_samples = detection_config.get('min_samples', 5)
        alpha = detection_config.get('alpha', 27)
        core_only = detection_config.get('core_only', True)
        min_area = detection_config.get('min_area', 0)
        min_edges = detection_config.get('min_edges', 20)
        holes_min_area = detection_config.get('holes_min_area', 10000)
        holes_min_edges = detection_config.get('holes_min_edges', 10)

        print(f"  Detecting regions for {len(markers)} markers...")
        print(f"  Parameters: eps={eps}, alpha={alpha}, min_samples={min_samples}, core_only={core_only}")

        all_region_info = []

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_adata = self.adata[sample_mask].copy()

            if sample not in self.marker_boundaries:
                self.marker_boundaries[sample] = {}
            if sample not in self.marker_regions:
                self.marker_regions[sample] = {}

            for marker_def in markers:
                marker_name = marker_def['name']
                pos_col = marker_def['positive_col']
                neg_col = marker_def.get('negative_col', None)

                # Analyze positive regions
                if pos_col in sample_adata.obs.columns:
                    pos_info = self._detect_single_marker_region(
                        sample_adata, sample, marker_name, 'positive', pos_col,
                        eps, min_samples, alpha, core_only,
                        min_area, min_edges, holes_min_area, holes_min_edges
                    )
                    all_region_info.extend(pos_info)

                # Analyze negative regions
                if neg_col and neg_col in sample_adata.obs.columns:
                    neg_info = self._detect_single_marker_region(
                        sample_adata, sample, marker_name, 'negative', neg_col,
                        eps, min_samples, alpha, core_only,
                        min_area, min_edges, holes_min_area, holes_min_edges
                    )
                    all_region_info.extend(neg_info)

        if all_region_info:
            df = pd.DataFrame(all_region_info)
            self.results['detected_marker_regions'] = df
            print(f"    ✓ Detected {len(all_region_info)} marker-defined regions")

    def _detect_single_marker_region(self, sample_adata, sample: str, marker_name: str,
                                    polarity: str, marker_col: str,
                                    eps: int, min_samples: int, alpha: int, core_only: bool,
                                    min_area: float, min_edges: int,
                                    holes_min_area: float, holes_min_edges: int) -> List[Dict]:
        """Detect regions for a single marker (positive or negative)."""
        region_info = []

        # Check if marker column exists and has enough positive cells
        if marker_col not in sample_adata.obs.columns:
            return region_info

        n_marker_cells = sample_adata.obs[marker_col].sum()
        if n_marker_cells < min_samples:
            return region_info

        try:
            # Detect communities
            communitycolumn = f'{marker_name}_{polarity}_community'
            community_results = spc.spatial.getCommunities(
                sample_adata,
                [marker_col],
                eps=eps,
                min_samples=min_samples,
                newcolumn=communitycolumn,
                core_only=core_only
            )

            if community_results is None:
                return region_info

            labels_sorted, db = community_results
            n_communities = len(labels_sorted)

            if n_communities == 0:
                return region_info

            # Get all community indices
            communityIndexList = [idx for _, idx in labels_sorted]

            # Create boundaries for all communities combined
            try:
                boundaries = spc.spatial.getBoundary(
                    sample_adata,
                    communitycolumn,
                    communityIndexList,
                    alpha=alpha,
                    debug=False
                )

                # Prune boundaries
                pruned_boundaries = spc.spatial.pruneSmallComponents(
                    boundaries,
                    min_area=min_area,
                    min_edges=min_edges,
                    holes_min_area=holes_min_area,
                    holes_min_edges=holes_min_edges
                )

                # Get individual components
                boundary_components = spc.spa.getComponents(pruned_boundaries)
                n_components = len(boundary_components)

                # Store boundaries
                marker_key = f'{marker_name}_{polarity}'
                if marker_key not in self.marker_boundaries[sample]:
                    self.marker_boundaries[sample][marker_key] = {}

                for component_idx, boundary in enumerate(boundary_components):
                    area = spc.msmt.getRegionArea(boundary)
                    centroid = spc.msmt.getRegionCentroid(boundary)

                    self.marker_boundaries[sample][marker_key][component_idx] = boundary

                    region_info.append({
                        'sample_id': sample,
                        'marker': marker_name,
                        'polarity': polarity,
                        'region_id': component_idx,
                        'n_communities': n_communities,
                        'n_components': n_components,
                        'area_um2': area,
                        'centroid_x': centroid[0],
                        'centroid_y': centroid[1],
                        'n_marker_cells': n_marker_cells,
                        'timepoint': sample_adata.obs['timepoint'].iloc[0] if 'timepoint' in sample_adata.obs.columns else np.nan,
                        'group': sample_adata.obs['group'].iloc[0] if 'group' in sample_adata.obs.columns else '',
                        'main_group': sample_adata.obs['main_group'].iloc[0] if 'main_group' in sample_adata.obs.columns else ''
                    })

                print(f"      ✓ {sample}: {marker_name} {polarity} - {n_communities} communities, {n_components} components")

            except Exception as e:
                warnings.warn(f"Error creating boundaries for {sample} {marker_name} {polarity}: {e}")
                return region_info

        except Exception as e:
            warnings.warn(f"Error detecting communities for {sample} {marker_name} {polarity}: {e}")
            return region_info

        return region_info

    def _analyze_regional_composition(self):
        """Analyze cell type composition within each marker-defined region."""
        composition_results = []

        # Get phenotype columns to analyze
        phenotype_cols = self.config.get('phenotype_columns', [])
        if not phenotype_cols:
            # Default: analyze immune populations
            phenotype_cols = [
                'is_CD8_T_cells',
                'is_CD3_positive',
                'is_CD45_positive',
                'is_Tumor'
            ]

        for sample, marker_dict in self.marker_boundaries.items():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_adata = self.adata[sample_mask].copy()

            for marker_key, boundaries in marker_dict.items():
                marker_name, polarity = marker_key.rsplit('_', 1)

                for region_id, boundary in boundaries.items():
                    # Assign cells to this region
                    region_name = f'{marker_key}_{region_id}'
                    spc.spatial.assignPointsToRegions(
                        sample_adata,
                        [boundary],
                        [region_name],
                        assigncolumn='temp_region',
                        default='background'
                    )

                    region_cells = sample_adata.obs[sample_adata.obs['temp_region'] == region_name]
                    n_total_cells = len(region_cells)

                    if n_total_cells < 10:
                        continue

                    # Count each phenotype
                    for pheno_col in phenotype_cols:
                        if pheno_col in region_cells.columns:
                            n_positive = region_cells[pheno_col].sum()
                            composition = n_positive / n_total_cells if n_total_cells > 0 else 0

                            composition_results.append({
                                'sample_id': sample,
                                'marker': marker_name,
                                'polarity': polarity,
                                'region_id': region_id,
                                'phenotype': pheno_col,
                                'n_cells': int(n_positive),
                                'n_total_cells': int(n_total_cells),
                                'composition': composition,
                                'percent': composition * 100,
                                'timepoint': region_cells['timepoint'].iloc[0] if 'timepoint' in region_cells.columns else np.nan,
                                'group': region_cells['group'].iloc[0] if 'group' in region_cells.columns else '',
                                'main_group': region_cells['main_group'].iloc[0] if 'main_group' in region_cells.columns else ''
                            })

        if composition_results:
            df = pd.DataFrame(composition_results)
            self.results['regional_composition'] = df
            print(f"    ✓ Analyzed composition for {len(composition_results)} region-phenotype combinations")

    def _compare_marker_regions(self):
        """Compare marker+ vs marker- regions for each marker."""
        if 'regional_composition' not in self.results:
            print("    ⚠ No regional composition data available")
            return

        comp_df = self.results['regional_composition']
        comparison_results = []

        # Group by sample, marker, and phenotype
        for (sample, marker, phenotype), group_df in comp_df.groupby(['sample_id', 'marker', 'phenotype']):
            pos_data = group_df[group_df['polarity'] == 'positive']
            neg_data = group_df[group_df['polarity'] == 'negative']

            if len(pos_data) == 0 or len(neg_data) == 0:
                continue

            # Calculate mean composition for positive and negative regions
            pos_mean = pos_data['composition'].mean()
            pos_std = pos_data['composition'].std()
            pos_n_regions = len(pos_data)
            pos_total_cells = pos_data['n_cells'].sum()

            neg_mean = neg_data['composition'].mean()
            neg_std = neg_data['composition'].std()
            neg_n_regions = len(neg_data)
            neg_total_cells = neg_data['n_cells'].sum()

            # Calculate fold change and difference
            fold_change = pos_mean / neg_mean if neg_mean > 0 else np.nan
            difference = pos_mean - neg_mean

            comparison_results.append({
                'sample_id': sample,
                'marker': marker,
                'phenotype': phenotype,
                'positive_mean_composition': pos_mean,
                'positive_std_composition': pos_std,
                'positive_n_regions': pos_n_regions,
                'positive_total_cells': int(pos_total_cells),
                'negative_mean_composition': neg_mean,
                'negative_std_composition': neg_std,
                'negative_n_regions': neg_n_regions,
                'negative_total_cells': int(neg_total_cells),
                'fold_change_pos_vs_neg': fold_change,
                'difference_pos_minus_neg': difference,
                'timepoint': pos_data['timepoint'].iloc[0] if 'timepoint' in pos_data.columns else np.nan,
                'group': pos_data['group'].iloc[0] if 'group' in pos_data.columns else '',
                'main_group': pos_data['main_group'].iloc[0] if 'main_group' in pos_data.columns else ''
            })

        if comparison_results:
            df = pd.DataFrame(comparison_results)
            self.results['marker_region_comparison'] = df
            print(f"    ✓ Compared {len(comparison_results)} marker-phenotype combinations")

    def _analyze_immune_enrichment(self):
        """Analyze immune cell enrichment in marker+ vs marker- regions."""
        enrichment_results = []

        # Get immune populations
        immune_pops = self.config.get('immune_populations', [])
        if not immune_pops:
            immune_pops = ['is_CD8_T_cells', 'is_CD3_positive', 'is_CD45_positive']

        for sample, marker_dict in self.marker_boundaries.items():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_adata = self.adata[sample_mask].copy()

            # Group by marker
            marker_groups = {}
            for marker_key in marker_dict.keys():
                marker_name, polarity = marker_key.rsplit('_', 1)
                if marker_name not in marker_groups:
                    marker_groups[marker_name] = {'positive': [], 'negative': []}
                marker_groups[marker_name][polarity] = marker_dict[marker_key]

            # Compare positive vs negative for each marker
            for marker_name, polarities in marker_groups.items():
                if 'positive' not in polarities or 'negative' not in polarities:
                    continue

                pos_boundaries = polarities['positive']
                neg_boundaries = polarities['negative']

                if not pos_boundaries or not neg_boundaries:
                    continue

                # Combine all positive region boundaries
                pos_boundary_list = list(pos_boundaries.values())
                neg_boundary_list = list(neg_boundaries.values())

                # Assign cells to positive vs negative regions
                spc.spatial.assignPointsToRegions(
                    sample_adata,
                    pos_boundary_list + neg_boundary_list,
                    [f'pos_{i}' for i in range(len(pos_boundary_list))] +
                    [f'neg_{i}' for i in range(len(neg_boundary_list))],
                    assigncolumn='temp_marker_region',
                    default='background'
                )

                # Get cells in positive vs negative regions
                pos_mask = sample_adata.obs['temp_marker_region'].str.startswith('pos_')
                neg_mask = sample_adata.obs['temp_marker_region'].str.startswith('neg_')

                pos_cells = sample_adata.obs[pos_mask]
                neg_cells = sample_adata.obs[neg_mask]

                n_pos_total = len(pos_cells)
                n_neg_total = len(neg_cells)

                if n_pos_total < 10 or n_neg_total < 10:
                    continue

                # Calculate immune enrichment for each population
                for immune_col in immune_pops:
                    if immune_col not in sample_adata.obs.columns:
                        continue

                    # Count immune cells in each region type
                    n_immune_pos = pos_cells[immune_col].sum()
                    n_immune_neg = neg_cells[immune_col].sum()

                    # Calculate densities
                    density_pos = n_immune_pos / n_pos_total if n_pos_total > 0 else 0
                    density_neg = n_immune_neg / n_neg_total if n_neg_total > 0 else 0

                    # Calculate enrichment
                    enrichment = density_pos / density_neg if density_neg > 0 else np.nan
                    difference = density_pos - density_neg

                    enrichment_results.append({
                        'sample_id': sample,
                        'marker': marker_name,
                        'immune_population': immune_col,
                        'n_positive_regions': len(pos_boundaries),
                        'n_negative_regions': len(neg_boundaries),
                        'n_cells_positive_regions': int(n_pos_total),
                        'n_cells_negative_regions': int(n_neg_total),
                        'n_immune_in_positive': int(n_immune_pos),
                        'n_immune_in_negative': int(n_immune_neg),
                        'immune_density_positive': density_pos,
                        'immune_density_negative': density_neg,
                        'enrichment_fold_change': enrichment,
                        'enrichment_difference': difference,
                        'percent_immune_positive': density_pos * 100,
                        'percent_immune_negative': density_neg * 100,
                        'timepoint': pos_cells['timepoint'].iloc[0] if 'timepoint' in pos_cells.columns else np.nan,
                        'group': pos_cells['group'].iloc[0] if 'group' in pos_cells.columns else '',
                        'main_group': pos_cells['main_group'].iloc[0] if 'main_group' in pos_cells.columns else ''
                    })

        if enrichment_results:
            df = pd.DataFrame(enrichment_results)
            self.results['immune_enrichment'] = df
            print(f"    ✓ Analyzed immune enrichment for {len(enrichment_results)} marker-immune combinations")

    def _analyze_region_holes(self):
        """Detect and analyze holes within marker regions."""
        hole_results = []

        for sample, marker_dict in self.marker_boundaries.items():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_adata = self.adata[sample_mask].copy()

            for marker_key, boundaries in marker_dict.items():
                marker_name, polarity = marker_key.rsplit('_', 1)

                for region_id, boundary in boundaries.items():
                    # Get holes in this boundary
                    try:
                        holes = spc.spa.getHoles(boundary)

                        if len(holes) == 0:
                            continue

                        for hole_idx, hole in enumerate(holes):
                            hole_area = spc.msmt.getRegionArea(hole)
                            hole_centroid = spc.msmt.getRegionCentroid(hole)

                            # Optionally create buffer regions around holes
                            buffer_distance = self.config.get('hole_buffer_distance', 30)
                            shrunken_hole = spc.spatial.bufferBoundary(hole, -buffer_distance)
                            extended_hole = spc.spatial.bufferBoundary(hole, buffer_distance)

                            # Assign cells to hole zones (inner, boundary, outer)
                            hole_region_name = f'{marker_key}_{region_id}_hole_{hole_idx}'
                            spc.spa.assignPointsToRegions(
                                sample_adata,
                                [shrunken_hole, hole, extended_hole],
                                [f'{hole_region_name}_inner', f'{hole_region_name}_boundary', f'{hole_region_name}_outer'],
                                assigncolumn=f'temp_hole_{marker_key}_{region_id}_{hole_idx}',
                                default='background'
                            )

                            # Count cells in each zone
                            hole_col = f'temp_hole_{marker_key}_{region_id}_{hole_idx}'
                            n_inner = (sample_adata.obs[hole_col] == f'{hole_region_name}_inner').sum()
                            n_boundary = (sample_adata.obs[hole_col] == f'{hole_region_name}_boundary').sum()
                            n_outer = (sample_adata.obs[hole_col] == f'{hole_region_name}_outer').sum()

                            hole_results.append({
                                'sample_id': sample,
                                'marker': marker_name,
                                'polarity': polarity,
                                'region_id': region_id,
                                'hole_id': hole_idx,
                                'hole_area_um2': hole_area,
                                'hole_centroid_x': hole_centroid[0],
                                'hole_centroid_y': hole_centroid[1],
                                'n_cells_inner_zone': int(n_inner),
                                'n_cells_boundary_zone': int(n_boundary),
                                'n_cells_outer_zone': int(n_outer),
                                'timepoint': sample_adata.obs['timepoint'].iloc[0] if 'timepoint' in sample_adata.obs.columns else np.nan,
                                'group': sample_adata.obs['group'].iloc[0] if 'group' in sample_adata.obs.columns else '',
                                'main_group': sample_adata.obs['main_group'].iloc[0] if 'main_group' in sample_adata.obs.columns else ''
                            })

                    except Exception as e:
                        warnings.warn(f"Error analyzing holes for {sample} {marker_key} region {region_id}: {e}")
                        continue

        if hole_results:
            df = pd.DataFrame(hole_results)
            self.results['region_holes'] = df
            print(f"    ✓ Analyzed {len(hole_results)} holes in marker regions")

    def _save_results(self):
        """Save all results to files."""
        for name, df in self.results.items():
            output_path = self.output_dir / f'{name}.csv'
            df.to_csv(output_path, index=False)

        print(f"\n  ✓ Saved {len(self.results)} result datasets")

    def get_marker_boundaries(self) -> Dict:
        """Return detected marker region boundaries."""
        return self.marker_boundaries

    def get_boundary(self, sample: str, marker: str, polarity: str, region_id: int):
        """
        Get boundary for a specific marker region.

        Parameters
        ----------
        sample : str
            Sample ID
        marker : str
            Marker name (e.g., 'pERK', 'Ki67')
        polarity : str
            'positive' or 'negative'
        region_id : int
            Region ID

        Returns
        -------
        shapely.geometry
            Boundary geometry
        """
        marker_key = f'{marker}_{polarity}'
        if sample not in self.marker_boundaries:
            raise ValueError(f"No boundaries for sample {sample}")
        if marker_key not in self.marker_boundaries[sample]:
            raise ValueError(f"No boundaries for {marker_key} in sample {sample}")
        if region_id not in self.marker_boundaries[sample][marker_key]:
            raise ValueError(f"Region {region_id} not found for {marker_key} in sample {sample}")

        return self.marker_boundaries[sample][marker_key][region_id]
