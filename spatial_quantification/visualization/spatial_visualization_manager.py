"""
Spatial Visualization Manager
Orchestrates all spatial plotting functions for comprehensive visualization
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings

from .spatial_plotter import SpatialPlotter


class SpatialVisualizationManager:
    """
    Manages and orchestrates all spatial visualizations.

    Coordinates:
    - Individual phenotype plots
    - Tumor zone plots (DBSCAN validation)
    - Marker +/- zone plots
    - Multi-phenotype overlays
    - Tumor structure visualizations
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        """
        Initialize spatial visualization manager.

        Parameters
        ----------
        adata : AnnData
            Annotated data object with cell coordinates and phenotypes
        config : Dict
            Configuration dictionary
        output_dir : Path
            Base output directory
        """
        self.adata = adata
        self.config = config
        self.output_dir = Path(output_dir)

        # Create spatial plotter
        self.plotter = SpatialPlotter(output_dir / 'spatial_visualizations', config)

    def generate_all_spatial_plots(self):
        """Generate all configured spatial visualizations."""
        print("\n" + "="*80)
        print("SPATIAL VISUALIZATIONS")
        print("="*80)

        # 1. Individual phenotype plots
        if self.config.get('spatial_visualization', {}).get('individual_phenotypes', True):
            print("\n[1/5] Generating individual phenotype spatial plots...")
            phenotypes_to_plot = self._get_phenotypes_for_plotting()
            self.plotter.plot_individual_phenotypes(self.adata, phenotypes_to_plot)

        # 2. Tumor zone plots (DBSCAN validation)
        if self.config.get('spatial_visualization', {}).get('tumor_zones', True):
            print("\n[2/5] Generating tumor zone (DBSCAN) plots...")
            tumor_col = self.config.get('tumor_definition', {}).get('base_phenotype', 'Tumor')
            self.plotter.plot_tumor_zones_dbscan(self.adata, f'is_{tumor_col}')

        # 3. Marker +/- zone plots
        if self.config.get('spatial_visualization', {}).get('marker_zones', True):
            print("\n[3/5] Generating marker +/- zone plots...")
            markers = self._get_marker_zone_definitions()
            if markers:
                self.plotter.plot_marker_zones(self.adata, markers)

        # 4. Tumor structure plots
        if self.config.get('spatial_visualization', {}).get('tumor_structures', True):
            print("\n[4/5] Generating tumor structure plots...")
            tumor_col = self.config.get('tumor_definition', {}).get('base_phenotype', 'Tumor')
            self.plotter.plot_tumor_structures_per_sample(self.adata, f'is_{tumor_col}')

        # 5. Multi-phenotype overlays
        if self.config.get('spatial_visualization', {}).get('phenotype_overlays', True):
            print("\n[5/5] Generating phenotype overlay plots...")
            key_phenotypes = self._get_key_phenotypes_for_overlay()
            if key_phenotypes:
                tumor_col = self.config.get('tumor_definition', {}).get('base_phenotype', 'Tumor')
                self.plotter.plot_phenotype_overlay(self.adata, key_phenotypes, f'is_{tumor_col}')

        print("\n" + "="*80)
        print("SPATIAL VISUALIZATIONS COMPLETE")
        print("="*80)

    def _get_phenotypes_for_plotting(self) -> List[str]:
        """
        Get list of phenotypes to plot individually.

        Returns
        -------
        List[str]
            Phenotype names
        """
        # Check config for explicit list
        explicit_list = self.config.get('spatial_visualization', {}).get('phenotype_list', None)

        if explicit_list:
            return explicit_list

        # Otherwise, use key populations from various analyses
        phenotypes = []

        # Population dynamics
        pop_dynamics_config = self.config.get('population_dynamics', {})
        if pop_dynamics_config.get('enabled', False):
            phenotypes.extend(pop_dynamics_config.get('populations', []))

        # Per-tumor analysis
        per_tumor_config = self.config.get('per_tumor_analysis', {})
        if per_tumor_config.get('enabled', False):
            for marker in per_tumor_config.get('markers', []):
                pheno = marker.get('phenotype')
                if pheno and pheno not in phenotypes:
                    phenotypes.append(pheno)

        # Distance analysis (sources)
        distance_config = self.config.get('distance_analysis', {})
        if distance_config.get('enabled', False):
            for pairing in distance_config.get('pairings', []):
                source = pairing.get('source')
                if source and source not in phenotypes:
                    phenotypes.append(source)

        # Limit to avoid too many plots
        if len(phenotypes) > 50:
            warnings.warn(f"Too many phenotypes ({len(phenotypes)}) for individual plotting. Limiting to first 50.")
            phenotypes = phenotypes[:50]

        return phenotypes

    def _get_marker_zone_definitions(self) -> List[Dict]:
        """
        Get marker zone definitions for spatial plotting.

        Returns
        -------
        List[Dict]
            List of marker definitions with keys: 'name', 'positive_phenotype', 'negative_phenotype'
        """
        markers = []

        # Check for explicit marker zone config in spatial_visualization
        explicit_markers = self.config.get('spatial_visualization', {}).get('marker_zones_list', None)

        if explicit_markers:
            return explicit_markers

        # Otherwise, extract from immune_infiltration config
        infiltration_config = self.config.get('immune_infiltration', {})
        marker_zone_config = infiltration_config.get('marker_zone_analysis', {})

        if marker_zone_config.get('enabled', False):
            for marker_def in marker_zone_config.get('markers', []):
                markers.append({
                    'name': marker_def.get('marker'),
                    'positive_phenotype': marker_def.get('positive_phenotype'),
                    'negative_phenotype': marker_def.get('negative_phenotype')
                })

        # Also extract from per_tumor_analysis
        per_tumor_config = self.config.get('per_tumor_analysis', {})
        if per_tumor_config.get('enabled', False):
            for marker_def in per_tumor_config.get('markers', []):
                marker_name = marker_def.get('name')
                pos_pheno = marker_def.get('phenotype')

                # Try to infer negative phenotype
                neg_pheno = pos_pheno.replace('_positive_', '_negative_')

                # Check if it exists in phenotypes
                if f'is_{neg_pheno}' in self.adata.obs.columns:
                    marker_info = {
                        'name': marker_name,
                        'positive_phenotype': pos_pheno,
                        'negative_phenotype': neg_pheno
                    }

                    # Avoid duplicates
                    if marker_info not in markers:
                        markers.append(marker_info)

        return markers

    def _get_key_phenotypes_for_overlay(self) -> List[str]:
        """
        Get key phenotypes for multi-phenotype overlay plots.

        Returns
        -------
        List[str]
            Phenotype names (without 'is_' prefix)
        """
        # Check for explicit list in config
        explicit_list = self.config.get('spatial_visualization', {}).get('overlay_phenotypes', None)

        if explicit_list:
            return explicit_list

        # Default key phenotypes: tumor + key immune markers
        key_phenotypes = []

        # Tumor
        tumor_base = self.config.get('tumor_definition', {}).get('base_phenotype', 'Tumor')
        key_phenotypes.append(tumor_base)

        # Key tumor markers (from per_tumor_analysis)
        per_tumor_config = self.config.get('per_tumor_analysis', {})
        if per_tumor_config.get('enabled', False):
            for marker in per_tumor_config.get('markers', [])[:3]:  # Limit to 3
                pheno = marker.get('phenotype')
                if pheno:
                    key_phenotypes.append(pheno)

        # Key immune populations
        immune_pops = ['CD8_T_cells', 'CD45_positive', 'T_cells']
        for pop in immune_pops:
            if f'is_{pop}' in self.adata.obs.columns:
                key_phenotypes.append(pop)
                if len(key_phenotypes) >= 6:  # Limit to 6 for readability
                    break

        return key_phenotypes
