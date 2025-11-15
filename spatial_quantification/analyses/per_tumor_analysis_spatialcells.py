"""
Per-Tumor Analysis with SpatialCells
Analyze metrics at the individual tumor structure level using SpatialCells for superior region detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial import cKDTree
import warnings

try:
    import spatialcells as spc
    HAS_SPATIALCELLS = True
except ImportError:
    HAS_SPATIALCELLS = False

from spatial_quantification.core import SpatialCellsRegionDetector


class PerTumorAnalysisSpatialCells:
    """
    Analyze metrics at the individual tumor structure level using SpatialCells.

    Key improvements over DBSCAN-based approach:
    - Proper geometric boundaries using alpha shapes
    - Accurate area and density calculations
    - Distance-based metrics from actual boundaries
    - Spatial heterogeneity analysis via sliding windows
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        """
        Initialize per-tumor analysis.

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
        self.config = config
        self.tumor_config = config.get('tumor_definition', {})
        self.output_dir = Path(output_dir) / 'per_tumor_analysis_spatialcells'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SpatialCells region detector
        self.region_detector = SpatialCellsRegionDetector(adata, config)

        # Storage
        self.results = {}
        self.detected_regions = {}

    def run(self):
        """Run complete per-tumor analysis."""
        print("\n" + "="*80)
        print("PER-TUMOR ANALYSIS (SpatialCells)")
        print("="*80)

        # Detect tumor structures
        print("\n1. Detecting tumor structures using SpatialCells...")
        self._detect_tumor_structures()

        # Assign cells to regions
        print("\n2. Assigning cells to tumor regions...")
        self._assign_cells_to_regions()

        # Calculate per-tumor metrics
        print("\n3. Calculating per-tumor metrics...")
        self._calculate_per_tumor_metrics()

        # Calculate marker percentages per tumor
        print("\n4. Calculating marker percentages per tumor...")
        self._calculate_marker_percentages()

        # Growth rate normalization
        print("\n5. Calculating growth-rate normalized metrics...")
        self._calculate_growth_normalized_metrics()

        # Per-tumor infiltration
        print("\n6. Calculating per-tumor infiltration...")
        self._calculate_per_tumor_infiltration()

        # Spatial heterogeneity (new!)
        print("\n7. Analyzing spatial heterogeneity...")
        self._analyze_spatial_heterogeneity()

        # Save results
        self._save_results()

        print("\n✓ Per-tumor analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
        print("="*80 + "\n")

        return self.results

    def _detect_tumor_structures(self):
        """Detect tumor structures using SpatialCells."""
        struct_config = self.tumor_config.get('structure_detection', {})
        eps = struct_config.get('eps', 100)
        min_samples = struct_config.get('min_samples', 20)
        min_cluster_size = struct_config.get('min_cluster_size', 50)
        alpha = struct_config.get('alpha', 100)

        self.detected_regions = self.region_detector.detect_tumor_regions(
            eps=eps,
            min_samples=min_samples,
            alpha=alpha,
            min_cluster_size=min_cluster_size
        )

    def _assign_cells_to_regions(self):
        """Assign cells to detected regions."""
        self.region_detector.assign_cells_to_regions()

    def _calculate_per_tumor_metrics(self):
        """Calculate basic metrics per tumor structure using SpatialCells measurements."""
        tumor_pheno = self.tumor_config.get('base_phenotype', 'Tumor')
        tumor_col = f'is_{tumor_pheno}'

        per_tumor_results = []

        for sample, region_info in self.detected_regions.items():
            for region_data in region_info['tumor_regions']:
                tumor_id = region_data['tumor_id']

                # Get cells in this tumor region
                sample_mask = self.adata.obs['sample_id'] == sample
                tumor_region_mask = self.adata.obs['tumor_region_id'] == tumor_id
                region_cells = self.adata.obs[sample_mask & tumor_region_mask]

                if len(region_cells) == 0:
                    continue

                # Get boundary
                boundary = self.region_detector.get_boundary(sample, tumor_id)

                # Calculate metrics using SpatialCells
                area_um2 = spc.msmt.getRegionArea(boundary)
                centroid = spc.msmt.getRegionCentroid(boundary)

                # Count tumor cells
                tumor_mask = region_cells[tumor_col].values if tumor_col in region_cells.columns else np.zeros(len(region_cells), dtype=bool)
                n_tumor_cells = tumor_mask.sum()

                # Cell density
                density = n_tumor_cells / area_um2 if area_um2 > 0 else 0

                per_tumor_results.append({
                    'sample_id': sample,
                    'tumor_id': int(tumor_id),
                    'n_tumor_cells': int(n_tumor_cells),
                    'n_total_cells': int(len(region_cells)),
                    'area_um2': area_um2,
                    'density_cells_per_um2': density,
                    'centroid_x': centroid[0],
                    'centroid_y': centroid[1],
                    'timepoint': region_cells['timepoint'].iloc[0] if 'timepoint' in region_cells.columns else np.nan,
                    'group': region_cells['group'].iloc[0] if 'group' in region_cells.columns else '',
                    'main_group': region_cells['main_group'].iloc[0] if 'main_group' in region_cells.columns else ''
                })

        if per_tumor_results:
            df = pd.DataFrame(per_tumor_results)
            self.results['per_tumor_metrics'] = df
            print(f"    ✓ Calculated metrics for {len(per_tumor_results)} tumor structures")

    def _calculate_marker_percentages(self):
        """Calculate marker percentages per tumor structure."""
        per_tumor_config = self.config.get('per_tumor_analysis', {})
        marker_configs = per_tumor_config.get('markers', [])

        if marker_configs:
            markers = marker_configs
        else:
            # Default markers
            markers = [
                {'name': 'pERK', 'phenotype': 'pERK_positive_tumor'},
                {'name': 'NINJA', 'phenotype': 'AGFP_positive_tumor'},
                {'name': 'Ki67', 'phenotype': 'Ki67_positive_tumor'}
            ]

        marker_results = []
        tumor_pheno = self.tumor_config.get('base_phenotype', 'Tumor')
        tumor_col = f'is_{tumor_pheno}'

        for sample, region_info in self.detected_regions.items():
            for region_data in region_info['tumor_regions']:
                tumor_id = region_data['tumor_id']

                # Get cells in this tumor region
                sample_mask = self.adata.obs['sample_id'] == sample
                tumor_region_mask = self.adata.obs['tumor_region_id'] == tumor_id
                region_cells = self.adata.obs[sample_mask & tumor_region_mask]

                # Count tumor cells
                tumor_mask = region_cells[tumor_col].values if tumor_col in region_cells.columns else np.zeros(len(region_cells), dtype=bool)
                n_tumor_cells = tumor_mask.sum()

                if n_tumor_cells < 10:  # Skip very small tumors
                    continue

                result = {
                    'sample_id': sample,
                    'tumor_id': int(tumor_id),
                    'n_tumor_cells': int(n_tumor_cells),
                    'timepoint': region_cells['timepoint'].iloc[0] if 'timepoint' in region_cells.columns else np.nan,
                    'group': region_cells['group'].iloc[0] if 'group' in region_cells.columns else '',
                    'main_group': region_cells['main_group'].iloc[0] if 'main_group' in region_cells.columns else ''
                }

                # Calculate percentage for each marker
                for marker_def in markers:
                    marker_name = marker_def['name']
                    marker_pheno = marker_def['phenotype']
                    marker_col = f'is_{marker_pheno}'

                    if marker_col in region_cells.columns:
                        marker_mask = region_cells[marker_col].values
                        n_marker_positive = marker_mask.sum()
                        percent_marker = (n_marker_positive / n_tumor_cells * 100) if n_tumor_cells > 0 else 0

                        result[f'n_{marker_name}_positive'] = int(n_marker_positive)
                        result[f'percent_{marker_name}_positive'] = percent_marker

                marker_results.append(result)

        if marker_results:
            df = pd.DataFrame(marker_results)
            self.results['per_tumor_marker_percentages'] = df
            print(f"    ✓ Calculated marker percentages for {len(marker_results)} tumors")

    def _calculate_growth_normalized_metrics(self):
        """Calculate pERK+ normalized by tumor growth rate (Ki67+)."""
        if 'per_tumor_marker_percentages' not in self.results:
            print("    ⚠ No marker percentages available, skipping")
            return

        df = self.results['per_tumor_marker_percentages'].copy()

        # Check if required columns exist
        required_cols = ['percent_pERK_positive', 'percent_Ki67_positive']
        if not all(col in df.columns for col in required_cols):
            print("    ⚠ Required marker data not available, skipping")
            return

        # Calculate growth-normalized pERK
        df['pERK_per_Ki67_ratio'] = df['percent_pERK_positive'] / (df['percent_Ki67_positive'] + 0.01)
        df['pERK_minus_Ki67'] = df['percent_pERK_positive'] - df['percent_Ki67_positive']

        # Residual after Ki67 regression
        growth_norm_results = []

        for (timepoint, group), group_df in df.groupby(['timepoint', 'main_group']):
            if len(group_df) < 5:
                continue

            ki67 = group_df['percent_Ki67_positive'].values
            perk = group_df['percent_pERK_positive'].values

            valid_mask = ~(np.isnan(ki67) | np.isnan(perk))
            if valid_mask.sum() < 3:
                continue

            ki67_valid = ki67[valid_mask]
            perk_valid = perk[valid_mask]

            # Fit line
            slope, intercept = np.polyfit(ki67_valid, perk_valid, 1)
            predicted_perk = slope * ki67 + intercept
            residuals = perk - predicted_perk

            for idx, (sample_id, tumor_id) in enumerate(zip(group_df['sample_id'], group_df['tumor_id'])):
                growth_norm_results.append({
                    'sample_id': sample_id,
                    'tumor_id': tumor_id,
                    'timepoint': timepoint,
                    'main_group': group,
                    'pERK_residual_from_Ki67': residuals[idx],
                    'predicted_pERK_from_Ki67': predicted_perk[idx]
                })

        if growth_norm_results:
            residuals_df = pd.DataFrame(growth_norm_results)
            df = df.merge(residuals_df[['sample_id', 'tumor_id', 'pERK_residual_from_Ki67', 'predicted_pERK_from_Ki67']],
                         on=['sample_id', 'tumor_id'], how='left')

        self.results['per_tumor_growth_normalized'] = df
        print(f"    ✓ Calculated growth-normalized metrics")

    def _calculate_per_tumor_infiltration(self):
        """Calculate immune infiltration per tumor structure using distance from boundaries."""
        immune_pops = ['CD8_T_cells', 'CD3_positive', 'CD45_positive']
        infiltration_results = []
        tumor_pheno = self.tumor_config.get('base_phenotype', 'Tumor')
        tumor_col = f'is_{tumor_pheno}'

        for sample, region_info in self.detected_regions.items():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_adata = self.adata[sample_mask].copy()

            for region_data in region_info['tumor_regions']:
                tumor_id = region_data['tumor_id']

                # Get boundary
                boundary = self.region_detector.get_boundary(sample, tumor_id)

                # Assign cells to region for analysis
                spc.spatial.assignPointsToRegions(
                    sample_adata,
                    [boundary],
                    ['tumor'],
                    assigncolumn='temp_region',
                    default='background'
                )

                # Get cells within tumor
                tumor_region_mask = sample_adata.obs['temp_region'] == 'tumor'
                tumor_cells = sample_adata.obs[tumor_region_mask]
                n_tumor_cells = (tumor_cells[tumor_col].values if tumor_col in tumor_cells.columns else np.zeros(len(tumor_cells), dtype=bool)).sum()

                if n_tumor_cells < 10:
                    continue

                result = {
                    'sample_id': sample,
                    'tumor_id': int(tumor_id),
                    'n_tumor_cells': int(n_tumor_cells),
                    'timepoint': tumor_cells['timepoint'].iloc[0] if 'timepoint' in tumor_cells.columns else np.nan,
                    'group': tumor_cells['group'].iloc[0] if 'group' in tumor_cells.columns else '',
                    'main_group': tumor_cells['main_group'].iloc[0] if 'main_group' in tumor_cells.columns else ''
                }

                # Calculate infiltration using distance from boundary
                spc.msmt.getDistanceFromObject(
                    sample_adata,
                    boundary,
                    region_col='temp_region',
                    region_subset=None,
                    name='distance_to_tumor',
                    inplace=True,
                    binned=False
                )

                for immune_pop in immune_pops:
                    immune_col = f'is_{immune_pop}'
                    if immune_col not in sample_adata.obs.columns:
                        continue

                    immune_mask = sample_adata.obs[immune_col].values

                    # Within tumor (distance close to 0 or negative)
                    n_immune_within = (immune_mask & (sample_adata.obs['distance_to_tumor'] <= 5)).sum()

                    # Near tumor (within 50um)
                    n_immune_near = (immune_mask & (sample_adata.obs['distance_to_tumor'] <= 50)).sum()

                    # Density metrics
                    immune_density = n_immune_within / n_tumor_cells if n_tumor_cells > 0 else 0

                    result[f'n_{immune_pop}_within_tumor'] = int(n_immune_within)
                    result[f'n_{immune_pop}_near_tumor_50um'] = int(n_immune_near)
                    result[f'{immune_pop}_density_per_tumor_cell'] = immune_density

                infiltration_results.append(result)

        if infiltration_results:
            df = pd.DataFrame(infiltration_results)
            self.results['per_tumor_infiltration'] = df
            print(f"    ✓ Calculated infiltration for {len(infiltration_results)} tumors")

    def _analyze_spatial_heterogeneity(self):
        """Analyze spatial heterogeneity using sliding window composition."""
        per_tumor_config = self.config.get('per_tumor_analysis', {})
        heterogeneity_config = per_tumor_config.get('spatial_heterogeneity', {})

        if not heterogeneity_config.get('enabled', False):
            print("    ⚠ Spatial heterogeneity analysis disabled in config")
            return

        window_size = heterogeneity_config.get('window_size', 300)
        step_size = heterogeneity_config.get('step_size', 300)
        min_cells = heterogeneity_config.get('min_cells', 10)
        phenotype_col = heterogeneity_config.get('phenotype_col', 'is_Ki67_positive_tumor')

        heterogeneity_results = []

        for sample, region_info in self.detected_regions.items():
            for region_data in region_info['tumor_regions']:
                tumor_id = region_data['tumor_id']

                try:
                    comp_df = self.region_detector.analyze_spatial_heterogeneity(
                        sample=sample,
                        tumor_id=tumor_id,
                        phenotype_col=phenotype_col,
                        window_size=window_size,
                        step_size=step_size,
                        min_cells=min_cells
                    )

                    if len(comp_df) > 0:
                        comp_df['sample_id'] = sample
                        comp_df['tumor_id'] = tumor_id
                        heterogeneity_results.append(comp_df)

                except Exception as e:
                    warnings.warn(f"Error in heterogeneity analysis for {sample} tumor {tumor_id}: {e}")
                    continue

        if heterogeneity_results:
            combined_df = pd.concat(heterogeneity_results, ignore_index=True)
            self.results['spatial_heterogeneity'] = combined_df
            print(f"    ✓ Analyzed spatial heterogeneity for {len(heterogeneity_results)} tumors")
        else:
            print("    ⚠ No heterogeneity results generated")

    def _save_results(self):
        """Save all results to files."""
        for name, df in self.results.items():
            output_path = self.output_dir / f'{name}.csv'
            df.to_csv(output_path, index=False)

        print(f"\n  ✓ Saved {len(self.results)} result datasets")

    def get_tumor_structures(self) -> Dict:
        """Return tumor structure labels for use by other analyses."""
        return self.region_detector.get_tumor_structures_for_legacy_analyses()

    def get_region_detector(self) -> SpatialCellsRegionDetector:
        """Return the region detector for advanced analyses."""
        return self.region_detector
