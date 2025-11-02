#!/usr/bin/env python3
"""
Spatial Quantification - Main Orchestrator
==========================================

Clean, comprehensive spatial analysis workflow for cyclic IF data.

Usage:
    python run_spatial_quantification.py --config config/spatial_config.yaml

Author: cifsQuant refactor
Date: 2025-11-02
"""

import argparse
import yaml
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spatial_quantification.core import DataLoader, PhenotypeBuilder, MetadataManager
from spatial_quantification.analyses import (
    PopulationDynamics,
    DistanceAnalysis,
    InfiltrationAnalysis,
    NeighborhoodAnalysis,
    AdvancedAnalysis
)
from spatial_quantification.visualization import PlotManager


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Run complete spatial quantification workflow."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Spatial Quantification Analysis Pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='spatial_quantification/config/spatial_config.yaml',
        help='Path to configuration YAML file'
    )
    args = parser.parse_args()

    # Load configuration
    print("\n" + "="*80)
    print("SPATIAL QUANTIFICATION PIPELINE")
    print("="*80)
    print(f"\nLoading configuration from: {args.config}")

    config = load_config(args.config)

    # Create output directory
    output_dir = Path(config['output']['base_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}/")
    print("="*80)

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING")
    print("="*80)

    data_loader = DataLoader(config)
    adata, metadata = data_loader.load()

    # =========================================================================
    # STEP 2: Process Metadata
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: METADATA PROCESSING")
    print("="*80)

    metadata_manager = MetadataManager(metadata, config)
    metadata = metadata_manager.process()
    adata = metadata_manager.merge_with_adata(adata)

    # =========================================================================
    # STEP 3: Build Phenotypes
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: PHENOTYPE BUILDING")
    print("="*80)

    phenotype_builder = PhenotypeBuilder(adata, config)
    adata = phenotype_builder.build_all_phenotypes()

    # =========================================================================
    # STEP 4: Run Analyses
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: RUNNING ANALYSES")
    print("="*80)

    all_results = {}

    # Population Dynamics
    if config.get('population_dynamics', {}).get('enabled', False):
        pop_dynamics = PopulationDynamics(adata, config, output_dir)
        all_results['population_dynamics'] = pop_dynamics.run()

    # Distance Analysis
    if config.get('distance_analysis', {}).get('enabled', False):
        distance_analysis = DistanceAnalysis(adata, config, output_dir)
        all_results['distance_analysis'] = distance_analysis.run()

    # Infiltration Analysis
    if config.get('immune_infiltration', {}).get('enabled', False):
        infiltration_analysis = InfiltrationAnalysis(adata, config, output_dir)
        all_results['infiltration_analysis'] = infiltration_analysis.run()

    # Neighborhood Analysis
    if config.get('cellular_neighborhoods', {}).get('enabled', False):
        neighborhood_analysis = NeighborhoodAnalysis(adata, config, output_dir)
        all_results['neighborhood_analysis'] = neighborhood_analysis.run()

    # Advanced Analysis
    if config.get('advanced_analyses', {}).get('enabled', False):
        advanced_analysis = AdvancedAnalysis(adata, config, output_dir)
        all_results['advanced_analysis'] = advanced_analysis.run()

    # =========================================================================
    # STEP 5: Generate Plots
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("="*80)

    if config['visualization'].get('enabled', True):
        plot_manager = PlotManager(config, output_dir)

        # Plot each analysis
        if 'population_dynamics' in all_results:
            plot_manager.plot_population_dynamics(all_results['population_dynamics'])

        if 'distance_analysis' in all_results:
            plot_manager.plot_distance_analysis(all_results['distance_analysis'])

        if 'infiltration_analysis' in all_results:
            plot_manager.plot_infiltration_analysis(all_results['infiltration_analysis'])

        if 'neighborhood_analysis' in all_results:
            plot_manager.plot_neighborhood_analysis(all_results['neighborhood_analysis'])

    # =========================================================================
    # COMPLETE
    # =========================================================================
    print("\n" + "="*80)
    print("SPATIAL QUANTIFICATION COMPLETE")
    print("="*80)
    print(f"\n✓ All results saved to: {output_dir}/")
    print("\nAnalyses completed:")
    for analysis_name in all_results.keys():
        print(f"  - {analysis_name}")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
