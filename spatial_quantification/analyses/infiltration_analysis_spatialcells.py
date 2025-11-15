"""
Immune Infiltration Analysis with SpatialCells
Analyze immune infiltration into tumor structures using SpatialCells for accurate distance-based metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree
import warnings

try:
    import spatialcells as spc
    HAS_SPATIALCELLS = True
except ImportError:
    HAS_SPATIALCELLS = False

from spatial_quantification.core import SpatialCellsRegionDetector


class InfiltrationAnalysisSpatialCells:
    """
    Analyze immune infiltration into tumor structures using SpatialCells.

    Key improvements over DBSCAN-based approach:
    - Distance calculations from actual geometric boundaries (not cluster centroids)
    - Proper infiltration metrics using boundary-based distances
    - Regional immune cell community detection
    - Immune-isolated vs immune-rich tumor region identification
    """

    def __init__(self, adata, config: Dict, output_dir: Path, region_detector: SpatialCellsRegionDetector = None):
        """
        Initialize infiltration analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data with phenotypes and spatial coordinates
        config : dict
            Configuration dictionary
        output_dir : Path
            Output directory
        region_detector : SpatialCellsRegionDetector, optional
            Pre-initialized region detector. If None, creates new one.
        """
        if not HAS_SPATIALCELLS:
            raise ImportError("SpatialCells required. Install with: pip install spatialcells")

        self.adata = adata
        self.config = config['immune_infiltration']
        self.tumor_config = config.get('tumor_definition', {})
        self.output_dir = Path(output_dir) / 'infiltration_analysis_spatialcells'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Region detector
        if region_detector is None:
            self.region_detector = SpatialCellsRegionDetector(adata, config)
            self.own_detector = True
        else:
            self.region_detector = region_detector
            self.own_detector = False

        # Storage
        self.results = {}
        self.immune_boundaries = {}  # sample -> immune cell region boundaries

    def run(self):
        """Run complete infiltration analysis."""
        print("\n" + "="*80)
        print("INFILTRATION ANALYSIS (SpatialCells)")
        print("="*80)

        # Detect tumor structures if needed
        if self.own_detector:
            print("\n1. Detecting tumor structures...")
            self._detect_tumor_structures()
        else:
            print("\n1. Using pre-detected tumor structures...")

        # Calculate basic infiltration
        print("\n2. Calculating immune infiltration from tumor boundaries...")
        self._calculate_infiltration_from_boundaries()

        # Detect immune-rich regions
        print("\n3. Detecting immune-rich regions...")
        self._detect_immune_rich_regions()

        # Calculate immune isolation metrics
        print("\n4. Calculating immune isolation metrics...")
        self._calculate_immune_isolation()

        # Marker zone analysis
        if self.config.get('marker_zone_analysis', {}).get('enabled', False):
            print("\n5. Analyzing marker zones...")
            self._analyze_marker_zones()

        # Save results
        self._save_results()

        print("\n✓ Infiltration analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
        print("="*80 + "\n")

        return self.results

    def _detect_tumor_structures(self):
        """Detect tumor structures using SpatialCells."""
        struct_config = self.tumor_config.get('structure_detection', {})
        self.region_detector.detect_tumor_regions(
            eps=struct_config.get('eps', 100),
            min_samples=struct_config.get('min_samples', 20),
            alpha=struct_config.get('alpha', 100),
            min_cluster_size=struct_config.get('min_cluster_size', 50)
        )
        self.region_detector.assign_cells_to_regions()

    def _calculate_infiltration_from_boundaries(self):
        """Calculate immune infiltration using distance from tumor boundaries."""
        immune_pops = self.config.get('immune_populations', [])
        boundaries = self.config.get('boundaries', [0, 50, 100, 200])

        print(f"  Analyzing {len(immune_pops)} immune populations...")
        print(f"  Distance boundaries: {boundaries} μm from tumor edge")

        infiltration_results = []
        detected_regions = self.region_detector.tumor_boundaries

        for sample, tumor_boundaries in detected_regions.items():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_adata = self.adata[sample_mask].copy()

            for tumor_id, boundary in tumor_boundaries.items():
                # Assign cells to region
                spc.spatial.assignPointsToRegions(
                    sample_adata,
                    [boundary],
                    ['tumor'],
                    assigncolumn='temp_region',
                    default='background'
                )

                # Calculate distance from boundary for all cells
                spc.msmt.getDistanceFromObject(
                    sample_adata,
                    boundary,
                    region_col='temp_region',
                    region_subset=None,
                    name='distance_to_tumor',
                    inplace=True,
                    binned=False
                )

                # Count cells in tumor
                tumor_region_mask = sample_adata.obs['temp_region'] == 'tumor'
                structure_size = tumor_region_mask.sum()

                if structure_size < 10:
                    continue

                # Analyze each immune population
                for immune_pop in immune_pops:
                    immune_col = f'is_{immune_pop}'
                    if immune_col not in sample_adata.obs.columns:
                        continue

                    immune_mask = sample_adata.obs[immune_col].values
                    immune_distances = sample_adata.obs['distance_to_tumor'].values[immune_mask]

                    # Count in each distance zone
                    for i in range(len(boundaries)):
                        if i == 0:
                            # Within tumor (distance ~ 0, using <= 5 to account for boundary tolerance)
                            count = (immune_distances <= 5).sum()
                            zone_name = 'within_tumor'
                            boundary_lower = 0
                            boundary_upper = 5
                        else:
                            # Between boundaries
                            lower = boundaries[i-1]
                            upper = boundaries[i]
                            count = ((immune_distances > lower) & (immune_distances <= upper)).sum()
                            zone_name = f'{lower}_{upper}um'
                            boundary_lower = lower
                            boundary_upper = upper

                        infiltration_results.append({
                            'sample_id': sample,
                            'structure_id': int(tumor_id),
                            'immune_population': immune_pop,
                            'zone': zone_name,
                            'boundary_lower': boundary_lower,
                            'boundary_upper': boundary_upper,
                            'count': int(count),
                            'structure_size': int(structure_size),
                            'infiltration_density': count / structure_size if structure_size > 0 else 0,
                            'timepoint': sample_adata.obs['timepoint'].iloc[0] if 'timepoint' in sample_adata.obs.columns else np.nan,
                            'group': sample_adata.obs['group'].iloc[0] if 'group' in sample_adata.obs.columns else '',
                            'main_group': sample_adata.obs['main_group'].iloc[0] if 'main_group' in sample_adata.obs.columns else ''
                        })

        if infiltration_results:
            df = pd.DataFrame(infiltration_results)
            self.results['infiltration'] = df
            print(f"    ✓ Calculated infiltration for {len(infiltration_results)} structure-population-zone combinations")

    def _detect_immune_rich_regions(self):
        """Detect immune-rich regions using SpatialCells community detection on immune cells."""
        immune_pops = self.config.get('immune_populations', [])
        immune_detection_config = self.config.get('immune_community_detection', {})

        if not immune_detection_config.get('enabled', True):
            print("    ⚠ Immune community detection disabled")
            return

        eps = immune_detection_config.get('eps', 130)
        min_samples = immune_detection_config.get('min_samples', 20)
        alpha = immune_detection_config.get('alpha', 130)
        min_area = immune_detection_config.get('min_area', 30000)

        immune_region_results = []

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_adata = self.adata[sample_mask].copy()

            # Combine all immune populations
            immune_combined = np.zeros(len(sample_adata), dtype=bool)
            for immune_pop in immune_pops:
                immune_col = f'is_{immune_pop}'
                if immune_col in sample_adata.obs.columns:
                    immune_combined |= sample_adata.obs[immune_col].values

            if immune_combined.sum() < min_samples:
                continue

            # Create temporary column for immune cells
            sample_adata.obs['immune_combined'] = immune_combined

            try:
                # Detect immune communities
                community_results = spc.spatial.getCommunities(
                    sample_adata,
                    ['immune_combined'],
                    eps=eps,
                    min_samples=min_samples,
                    newcolumn='immune_community',
                    core_only=False
                )

                if community_results is None:
                    continue

                labels_sorted, db = community_results

                # Create boundaries for immune regions
                sample_immune_boundaries = {}

                for cell_count, community_idx in labels_sorted:
                    if cell_count < min_samples:
                        continue

                    try:
                        # Create boundary
                        boundary = spc.spa.getBoundary(
                            sample_adata,
                            'immune_community',
                            [community_idx],
                            alpha=alpha,
                            debug=False
                        )

                        # Prune
                        pruned_boundary = spc.spa.pruneSmallComponents(
                            boundary,
                            min_edges=25,
                            holes_min_edges=30,
                            min_area=min_area
                        )

                        # Calculate area
                        area = spc.msmt.getRegionArea(pruned_boundary)

                        if area >= min_area:
                            sample_immune_boundaries[community_idx] = pruned_boundary

                            immune_region_results.append({
                                'sample_id': sample,
                                'immune_region_id': community_idx,
                                'n_cells': cell_count,
                                'area_um2': area
                            })

                    except Exception as e:
                        warnings.warn(f"Error creating immune boundary for community {community_idx}: {e}")
                        continue

                self.immune_boundaries[sample] = sample_immune_boundaries
                print(f"    ✓ {sample}: Detected {len(sample_immune_boundaries)} immune-rich regions")

            except Exception as e:
                warnings.warn(f"Error detecting immune communities for {sample}: {e}")
                continue

        if immune_region_results:
            df = pd.DataFrame(immune_region_results)
            self.results['immune_rich_regions'] = df

    def _calculate_immune_isolation(self):
        """Calculate immune isolation metrics: tumor cells in immune-poor vs immune-rich regions."""
        if not self.immune_boundaries:
            print("    ⚠ No immune boundaries detected, skipping isolation analysis")
            return

        tumor_pheno = self.tumor_config.get('base_phenotype', 'Tumor')
        tumor_col = f'is_{tumor_pheno}'

        isolation_results = []
        detected_tumor_regions = self.region_detector.tumor_boundaries

        for sample, tumor_boundaries in detected_tumor_regions.items():
            if sample not in self.immune_boundaries:
                continue

            sample_mask = self.adata.obs['sample_id'] == sample
            sample_adata = self.adata[sample_mask].copy()

            immune_boundaries = self.immune_boundaries[sample]

            # Combine all immune boundaries for this sample
            if len(immune_boundaries) > 0:
                immune_boundary_list = list(immune_boundaries.values())

                for tumor_id, tumor_boundary in tumor_boundaries.items():
                    # Assign tumor cells to regions
                    spc.spatial.assignPointsToRegions(
                        sample_adata,
                        [tumor_boundary],
                        ['tumor'],
                        assigncolumn='temp_tumor_region',
                        default='background'
                    )

                    # Assign to immune regions
                    spc.spatial.assignPointsToRegions(
                        sample_adata,
                        immune_boundary_list,
                        [f'immune_{i}' for i in range(len(immune_boundary_list))],
                        assigncolumn='temp_immune_region',
                        default='immune_poor'
                    )

                    # Get tumor cells
                    tumor_mask = (sample_adata.obs['temp_tumor_region'] == 'tumor')
                    if tumor_col in sample_adata.obs.columns:
                        tumor_mask &= sample_adata.obs[tumor_col]

                    tumor_cells = sample_adata.obs[tumor_mask]
                    n_tumor_cells = len(tumor_cells)

                    if n_tumor_cells < 10:
                        continue

                    # Count immune-isolated vs immune-rich
                    immune_isolated = (tumor_cells['temp_immune_region'] == 'immune_poor').sum()
                    immune_rich = (tumor_cells['temp_immune_region'] != 'immune_poor').sum()

                    # Calculate area overlap
                    tumor_area = spc.msmt.getRegionArea(tumor_boundary)

                    # Calculate overlap with immune regions
                    total_overlap_area = 0
                    for immune_boundary in immune_boundary_list:
                        try:
                            overlap = tumor_boundary.intersection(immune_boundary)
                            overlap_area = spc.msmt.getRegionArea(overlap)
                            total_overlap_area += overlap_area
                        except:
                            continue

                    isolation_results.append({
                        'sample_id': sample,
                        'tumor_id': int(tumor_id),
                        'n_tumor_cells': int(n_tumor_cells),
                        'n_immune_isolated': int(immune_isolated),
                        'n_immune_rich': int(immune_rich),
                        'percent_immune_isolated': (immune_isolated / n_tumor_cells * 100) if n_tumor_cells > 0 else 0,
                        'percent_immune_rich': (immune_rich / n_tumor_cells * 100) if n_tumor_cells > 0 else 0,
                        'tumor_area_um2': tumor_area,
                        'immune_overlap_area_um2': total_overlap_area,
                        'percent_area_overlap': (total_overlap_area / tumor_area * 100) if tumor_area > 0 else 0,
                        'timepoint': sample_adata.obs['timepoint'].iloc[0] if 'timepoint' in sample_adata.obs.columns else np.nan,
                        'group': sample_adata.obs['group'].iloc[0] if 'group' in sample_adata.obs.columns else '',
                        'main_group': sample_adata.obs['main_group'].iloc[0] if 'main_group' in sample_adata.obs.columns else ''
                    })

        if isolation_results:
            df = pd.DataFrame(isolation_results)
            self.results['immune_isolation'] = df
            print(f"    ✓ Calculated isolation metrics for {len(isolation_results)} tumors")

    def _analyze_marker_zones(self):
        """Analyze spatial heterogeneity of marker zones using SpatialCells."""
        marker_config = self.config['marker_zone_analysis']
        markers = marker_config.get('markers', [])

        print(f"  Analyzing {len(markers)} marker zones...")

        for marker_def in markers:
            marker = marker_def['marker']
            pos_pheno = marker_def['positive_phenotype']
            neg_pheno = marker_def['negative_phenotype']

            print(f"    Processing {marker} zones...")
            self._analyze_single_marker_zone(marker, pos_pheno, neg_pheno, marker_config)

    def _analyze_single_marker_zone(self, marker: str, pos_pheno: str,
                                    neg_pheno: str, config: Dict):
        """Analyze a single marker zone using SpatialCells measurements."""
        pos_col = f'is_{pos_pheno}'
        neg_col = f'is_{neg_pheno}'

        if pos_col not in self.adata.obs.columns or neg_col not in self.adata.obs.columns:
            warnings.warn(f"Phenotypes for {marker} not found, skipping")
            return

        heterogeneity_results = []
        detected_regions = self.region_detector.tumor_boundaries

        for sample, tumor_boundaries in detected_regions.items():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_adata = self.adata[sample_mask].copy()

            for tumor_id, boundary in tumor_boundaries.items():
                # Assign to region
                spc.spatial.assignPointsToRegions(
                    sample_adata,
                    [boundary],
                    ['tumor'],
                    assigncolumn='temp_region',
                    default='background'
                )

                tumor_cells = sample_adata.obs[sample_adata.obs['temp_region'] == 'tumor']

                # Get marker composition
                pos_mask = tumor_cells[pos_col].values if pos_col in tumor_cells.columns else np.zeros(len(tumor_cells), dtype=bool)
                neg_mask = tumor_cells[neg_col].values if neg_col in tumor_cells.columns else np.zeros(len(tumor_cells), dtype=bool)

                n_pos = pos_mask.sum()
                n_neg = neg_mask.sum()
                n_total = len(tumor_cells)

                if n_pos < 5 or n_neg < 5:
                    continue

                fraction_pos = n_pos / n_total if n_total > 0 else 0

                # Use SpatialCells to get composition
                comp = spc.msmt.getRegionComposition(
                    sample_adata,
                    [pos_col, neg_col],
                    regions=['tumor']
                )

                heterogeneity_results.append({
                    'sample_id': sample,
                    'structure_id': int(tumor_id),
                    'marker': marker,
                    'n_positive': int(n_pos),
                    'n_negative': int(n_neg),
                    'fraction_positive': fraction_pos,
                    'timepoint': tumor_cells['timepoint'].iloc[0] if 'timepoint' in tumor_cells.columns else np.nan,
                    'group': tumor_cells['group'].iloc[0] if 'group' in tumor_cells.columns else '',
                    'main_group': tumor_cells['main_group'].iloc[0] if 'main_group' in tumor_cells.columns else ''
                })

        if heterogeneity_results:
            df = pd.DataFrame(heterogeneity_results)
            self.results[f'{marker}_zone_heterogeneity'] = df
            print(f"      ✓ Analyzed heterogeneity for {len(heterogeneity_results)} structures")

    def _save_results(self):
        """Save all results to files."""
        for name, df in self.results.items():
            output_path = self.output_dir / f'{name}.csv'
            df.to_csv(output_path, index=False)

        print(f"\n  ✓ Saved {len(self.results)} result datasets")

    def get_immune_boundaries(self) -> Dict:
        """Return detected immune region boundaries."""
        return self.immune_boundaries
