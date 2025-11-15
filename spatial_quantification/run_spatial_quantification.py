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
    InfiltrationAnalysisOptimized,
    NeighborhoodAnalysis,
    NeighborhoodAnalysisOptimized,
    AdvancedAnalysis,
    TumorMicroenvironmentAnalysis
)
# Import new analyses
try:
    from spatial_quantification.analyses.per_tumor_analysis import PerTumorAnalysis
    from spatial_quantification.analyses.coexpression_analysis import CoexpressionAnalysis
    from spatial_quantification.analyses.enhanced_neighborhood_analysis import EnhancedNeighborhoodAnalysis
    from spatial_quantification.analyses.pseudotime_analysis import PseudotimeAnalysis
    HAS_NEW_ANALYSES = True
except ImportError:
    HAS_NEW_ANALYSES = False
    print("  ⚠ New analyses not available (per_tumor, coexpression, enhanced_neighborhoods, pseudotime)")
from spatial_quantification.visualization import PlotManager


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file and resolve paths."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get project root (parent of spatial_quantification/)
    project_root = Path(__file__).parent.parent

    # Resolve input paths relative to project root
    if 'input' in config:
        if 'gated_data' in config['input']:
            gated_path = Path(config['input']['gated_data'])
            if not gated_path.is_absolute():
                config['input']['gated_data'] = str(project_root / gated_path)

        if 'metadata' in config['input']:
            metadata_path = Path(config['input']['metadata'])
            if not metadata_path.is_absolute():
                config['input']['metadata'] = str(project_root / metadata_path)

    # Resolve output directory relative to project root
    if 'output' in config:
        if 'base_directory' in config['output']:
            output_path = Path(config['output']['base_directory'])
            if not output_path.is_absolute():
                config['output']['base_directory'] = str(project_root / output_path)

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
        default=None,
        help='Path to configuration YAML file'
    )
    args = parser.parse_args()

    # Determine config path
    if args.config is None:
        # Default: look for config in same directory as script
        script_dir = Path(__file__).parent
        config_path = script_dir / 'config' / 'spatial_config.yaml'
    else:
        config_path = Path(args.config)

    # Load configuration
    print("\n" + "="*80)
    print("SPATIAL QUANTIFICATION PIPELINE")
    print("="*80)
    print(f"\nLoading configuration from: {config_path}")

    config = load_config(str(config_path))

    # Print resolved paths
    print(f"\nResolved paths:")
    print(f"  Gated data: {config['input']['gated_data']}")
    print(f"  Metadata: {config['input']['metadata']}")
    print(f"  Output: {config['output']['base_directory']}")

    # Create output directory
    output_dir = Path(config['output']['base_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)

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
    tumor_structures = None  # Will be populated by per_tumor_analysis
    region_detector = None  # For SpatialCells analyses

    # NEW: Per-Tumor Analysis (run first to detect tumor structures)
    if HAS_NEW_ANALYSES and config.get('per_tumor_analysis', {}).get('enabled', False):
        use_spatialcells = config.get('per_tumor_analysis', {}).get('use_spatialcells', False)

        if use_spatialcells:
            print("  ✓ Using SpatialCells-based per-tumor analysis (alpha shapes + boundaries)")
            from spatial_quantification.analyses import PerTumorAnalysisSpatialCells
            per_tumor = PerTumorAnalysisSpatialCells(adata, config, output_dir)
            all_results['per_tumor_analysis'] = per_tumor.run()
            tumor_structures = per_tumor.get_tumor_structures()
            region_detector = per_tumor.get_region_detector()  # Reuse for other analyses
        else:
            print("  Using DBSCAN-based per-tumor analysis (legacy)")
            per_tumor = PerTumorAnalysis(adata, config, output_dir)
            all_results['per_tumor_analysis'] = per_tumor.run()
            tumor_structures = per_tumor.get_tumor_structures()

        # Generate plots if configured
        if config.get('per_tumor_analysis', {}).get('generate_plots', True):
            try:
                from spatial_quantification.visualization.per_tumor_plotter import PerTumorPlotter
                # Use the actual output directory from per_tumor analysis
                plotter = PerTumorPlotter(per_tumor.output_dir, config)
                plotter.generate_all_plots(all_results['per_tumor_analysis'])
            except Exception as e:
                import traceback
                print(f"  ⚠ Could not generate per-tumor plots: {e}")
                print(f"     Traceback: {traceback.format_exc()}")

        # Generate comprehensive spatial plots (samples + individual tumors)
        if use_spatialcells and region_detector is not None:
            # Debug: Check region detector state
            print(f"\n  DEBUG: region_detector type: {type(region_detector)}")
            print(f"  DEBUG: region_detector has tumor_boundaries: {hasattr(region_detector, 'tumor_boundaries')}")
            if hasattr(region_detector, 'tumor_boundaries'):
                print(f"  DEBUG: tumor_boundaries keys: {list(region_detector.tumor_boundaries.keys())}")
                for sample, boundaries in region_detector.tumor_boundaries.items():
                    print(f"  DEBUG: Sample {sample} has {len(boundaries)} tumor boundaries")

            if config.get('per_tumor_analysis', {}).get('generate_spatial_plots', True):
                try:
                    from spatial_quantification.visualization.spatial_plotter import SpatialPlotter
                    spatial_plotter = SpatialPlotter(per_tumor.output_dir, config)
                    spatial_plotter.generate_spatialcells_plots(adata, region_detector)
                except Exception as e:
                    import traceback
                    print(f"  ⚠ Could not generate spatial plots: {e}")
                    print(f"     Traceback: {traceback.format_exc()}")

    # Comprehensive Coexpression Analysis
    if config.get('coexpression_analysis', {}).get('enabled', True):
        try:
            from spatial_quantification.analyses.coexpression_analysis_comprehensive import CoexpressionAnalysisComprehensive
            print("\n  Running comprehensive coexpression analysis...")
            coexpression = CoexpressionAnalysisComprehensive(adata, config, output_dir)
            all_results['coexpression_analysis'] = coexpression.run()
        except ImportError as e:
            print(f"  ⚠ Coexpression analysis module not available: {e}")
        except Exception as e:
            import traceback
            print(f"  ⚠ Error in coexpression analysis: {e}")
            print(f"     Traceback: {traceback.format_exc()}")

    # Spatial Overlap Analysis (marker regions)
    if region_detector is not None and config.get('spatial_overlap_analysis', {}).get('enabled', True):
        try:
            from spatial_quantification.analyses.spatial_overlap_analysis import SpatialOverlapAnalysis
            print("\n  Running spatial overlap analysis...")
            overlap_analysis = SpatialOverlapAnalysis(adata, config, output_dir, region_detector)
            all_results['spatial_overlap_analysis'] = overlap_analysis.run()
        except ImportError as e:
            print(f"  ⚠ Spatial overlap analysis module not available: {e}")
        except Exception as e:
            import traceback
            print(f"  ⚠ Error in spatial overlap analysis: {e}")
            print(f"     Traceback: {traceback.format_exc()}")

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
        use_spatialcells = config.get('immune_infiltration', {}).get('use_spatialcells', False)
        use_optimized = config.get('immune_infiltration', {}).get('use_optimized', True)

        if use_spatialcells:
            print("  ✓ Using SpatialCells-based infiltration analysis (boundary distances + immune regions)")
            from spatial_quantification.analyses import InfiltrationAnalysisSpatialCells
            infiltration_analysis = InfiltrationAnalysisSpatialCells(
                adata, config, output_dir, region_detector=region_detector
            )
        elif use_optimized:
            print("  Using OPTIMIZED infiltration analysis (Getis-Ord Gi* + Ripley's K)")
            infiltration_analysis = InfiltrationAnalysisOptimized(adata, config, output_dir)
        else:
            print("  Using standard infiltration analysis (Moran's I)")
            infiltration_analysis = InfiltrationAnalysis(adata, config, output_dir)

        all_results['infiltration_analysis'] = infiltration_analysis.run()

        # Generate comprehensive summary plots
        print("\n  Generating infiltration summary visualizations...")
        try:
            from spatial_quantification.visualization.infiltration_plotter import InfiltrationPlotter
            infiltration_plotter = InfiltrationPlotter(output_dir / 'infiltration_analysis', config)
            infiltration_plotter.generate_all_plots(all_results['infiltration_analysis'])
        except Exception as e:
            print(f"  ⚠ Could not generate infiltration plots: {e}")

    # Neighborhood Analysis
    if config.get('cellular_neighborhoods', {}).get('enabled', False):
        # Use optimized version if specified (RECOMMENDED)
        use_optimized = config.get('cellular_neighborhoods', {}).get('use_optimized', True)

        if use_optimized:
            print("  Using OPTIMIZED neighborhood analysis (windowed + HNSW)")
            neighborhood_analysis = NeighborhoodAnalysisOptimized(adata, config, output_dir)
        else:
            print("  Using standard neighborhood analysis")
            neighborhood_analysis = NeighborhoodAnalysis(adata, config, output_dir)

        all_results['neighborhood_analysis'] = neighborhood_analysis.run()

    # Advanced Analysis
    if config.get('advanced_analyses', {}).get('enabled', False):
        advanced_analysis = AdvancedAnalysis(adata, config, output_dir)
        all_results['advanced_analysis'] = advanced_analysis.run()

    # NEW: Coexpression Analysis
    if HAS_NEW_ANALYSES and config.get('coexpression_analysis', {}).get('enabled', False):
        coexpression = CoexpressionAnalysis(adata, config, output_dir, tumor_structures=tumor_structures)
        all_results['coexpression_analysis'] = coexpression.run()

    # NEW: Enhanced Neighborhood Analysis (marker-specific regions)
    if HAS_NEW_ANALYSES and config.get('enhanced_neighborhoods', {}).get('enabled', False):
        enhanced_neighborhoods = EnhancedNeighborhoodAnalysis(
            adata, config, output_dir, tumor_structures=tumor_structures
        )
        all_results['enhanced_neighborhoods'] = enhanced_neighborhoods.run()

        # Generate comprehensive summary plots
        print("\n  Generating enhanced neighborhood visualizations...")
        try:
            from spatial_quantification.visualization.enhanced_neighborhood_plotter import EnhancedNeighborhoodPlotter
            enh_plotter = EnhancedNeighborhoodPlotter(output_dir / 'enhanced_neighborhoods', config)
            enh_plotter.generate_all_plots(all_results['enhanced_neighborhoods'])
        except Exception as e:
            print(f"  ⚠ Could not generate enhanced neighborhood plots: {e}")

    # NEW: Tumor Microenvironment Analysis (per-phenotype immune composition)
    if config.get('tumor_microenvironment', {}).get('enabled', False):
        print("\n  Running tumor microenvironment analysis...")
        tumor_microenv = TumorMicroenvironmentAnalysis(
            adata, config, output_dir, tumor_structures=tumor_structures
        )
        all_results['tumor_microenvironment'] = tumor_microenv.run()

        # Generate comprehensive plots
        if config.get('tumor_microenvironment', {}).get('generate_plots', True):
            print("\n  Generating tumor microenvironment visualizations...")
            try:
                from spatial_quantification.visualization.tumor_microenvironment_plotter import TumorMicroenvironmentPlotter
                tm_plotter = TumorMicroenvironmentPlotter(output_dir / 'tumor_microenvironment', config)
                tm_plotter.generate_all_plots(all_results['tumor_microenvironment'])
            except Exception as e:
                print(f"  ⚠ Could not generate tumor microenvironment plots: {e}")
                import traceback
                traceback.print_exc()

    # NEW: Pseudotime Analysis
    if HAS_NEW_ANALYSES and config.get('pseudotime_analysis', {}).get('enabled', False):
        pseudotime = PseudotimeAnalysis(adata, config, output_dir)
        all_results['pseudotime_analysis'] = pseudotime.run()

        # Generate differentiation plots
        if config.get('pseudotime_analysis', {}).get('generate_plots', True):
            print("\n  Generating pseudotime visualizations...")
            try:
                from spatial_quantification.visualization.pseudotime_plotter import PseudotimePlotter
                pseudotime_plotter = PseudotimePlotter(output_dir / 'pseudotime_analysis', config)
                pseudotime_plotter.generate_all_plots(all_results['pseudotime_analysis'])
            except Exception as e:
                print(f"  ⚠ Could not generate pseudotime plots: {e}")

    # NEW: Marker Region Analysis (SpatialCells)
    if config.get('marker_region_analysis', {}).get('enabled', False):
        print("\n  ✓ Running marker region analysis (pERK+/-, Ki67+/-, etc.)")
        try:
            from spatial_quantification.analyses import MarkerRegionAnalysisSpatialCells
            marker_region = MarkerRegionAnalysisSpatialCells(adata, config, output_dir)
            all_results['marker_region_analysis'] = marker_region.run()

            # Generate plots if configured
            if config.get('marker_region_analysis', {}).get('generate_plots', True):
                print("\n  Generating marker region visualizations...")
                try:
                    from spatial_quantification.visualization.marker_region_plotter import MarkerRegionPlotter
                    mr_plotter = MarkerRegionPlotter(output_dir / 'marker_region_analysis', config)
                    mr_plotter.generate_all_plots(all_results['marker_region_analysis'], marker_region)
                except Exception as e:
                    print(f"  ⚠ Could not generate marker region plots: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"  ⚠ Could not run marker region analysis: {e}")
            import traceback
            traceback.print_exc()

    # NEW: UMAP Visualization
    if config.get('umap_visualization', {}).get('enabled', False):
        print("\n  Generating UMAP visualizations...")
        try:
            from spatial_quantification.visualization.umap_plotter import UMAPPlotter
            umap_plotter = UMAPPlotter(output_dir / 'umap_visualization', config)
            umap_plotter.generate_all_plots(adata)
        except Exception as e:
            print(f"  ⚠ Could not generate UMAP plots: {e}")
            import traceback
            traceback.print_exc()

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
    # STEP 6: Generate Spatial Visualizations
    # =========================================================================
    if config.get('spatial_visualization', {}).get('enabled', True):
        print("\n" + "="*80)
        print("STEP 6: GENERATING SPATIAL VISUALIZATIONS")
        print("="*80)

        try:
            from spatial_quantification.visualization import SpatialVisualizationManager

            spatial_viz = SpatialVisualizationManager(adata, config, output_dir)
            spatial_viz.generate_all_spatial_plots()
        except Exception as e:
            print(f"  ⚠ Could not generate spatial visualizations: {e}")
            import traceback
            traceback.print_exc()

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
