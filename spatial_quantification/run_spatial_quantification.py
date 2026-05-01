#!/usr/bin/env python3
"""
Spatial Quantification - Main Orchestrator

Usage:
    python run_spatial_quantification.py --config config/spatial_config.yaml
"""

import argparse
import yaml
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from spatial_quantification.core import DataLoader, PhenotypeBuilder, MetadataManager
from spatial_quantification.analyses import (
    PopulationDynamics,
    DistanceAnalysis,
    InfiltrationAnalysisOptimized,
    InfiltrationAnalysisSpatialCells,
    NeighborhoodAnalysisOptimized,
    AdvancedAnalysis,
    TumorMicroenvironmentAnalysis,
    PerTumorAnalysisSpatialCells,
    CoexpressionAnalysisComprehensive,
)
from spatial_quantification.analyses.enhanced_neighborhood_analysis import EnhancedNeighborhoodAnalysis
from spatial_quantification.analyses.pseudotime_analysis import PseudotimeAnalysis
from spatial_quantification.visualization import PlotManager


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML and resolve paths.
    Accepts either a standalone spatial_config.yaml or a project.yaml
    (which has the spatial analysis config nested under a 'spatial:' key).
    """
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

    # project.yaml has spatial config under 'spatial:' key
    config = raw.get('spatial', raw)

    project_root = Path(__file__).parent.parent

    if 'input' in config:
        if 'gated_data' in config['input']:
            gated_path = Path(config['input']['gated_data'])
            if not gated_path.is_absolute():
                config['input']['gated_data'] = str(project_root / gated_path)
        if 'metadata' in config['input']:
            metadata_path = Path(config['input']['metadata'])
            if not metadata_path.is_absolute():
                config['input']['metadata'] = str(project_root / metadata_path)

    if 'output' in config:
        if 'base_directory' in config['output']:
            output_path = Path(config['output']['base_directory'])
            if not output_path.is_absolute():
                config['output']['base_directory'] = str(project_root / output_path)

    return config


def main():
    """Run complete spatial quantification workflow."""
    parser = argparse.ArgumentParser(description='Spatial Quantification Analysis Pipeline')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration YAML file')
    args = parser.parse_args()

    if args.config is None:
        config_path = Path(__file__).parent / 'config' / 'spatial_config.yaml'
    else:
        config_path = Path(args.config)

    print("\n" + "="*80)
    print("SPATIAL QUANTIFICATION PIPELINE")
    print("="*80)
    print(f"\nLoading configuration from: {config_path}")

    config = load_config(str(config_path))

    print(f"\nResolved paths:")
    print(f"  Gated data: {config['input']['gated_data']}")
    print(f"  Metadata:   {config['input']['metadata']}")
    print(f"  Output:     {config['output']['base_directory']}")

    output_dir = Path(config['output']['base_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)

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
    tumor_structures = None
    region_detector = None

    # Per-Structure Analysis (run first — detects structures reused by downstream analyses)
    per_structure_config = config.get('per_structure_analysis', config.get('per_tumor_analysis', {}))
    if per_structure_config.get('enabled', False):
        use_spatialcells = per_structure_config.get('use_spatialcells', True)

        print("  Using SpatialCells-based per-structure analysis (alpha shapes + boundaries)")
        per_structure = PerTumorAnalysisSpatialCells(adata, config, output_dir)
        all_results['per_structure_analysis'] = per_structure.run()
        tumor_structures = per_structure.get_tumor_structures()
        region_detector = per_structure.get_region_detector()

        if per_structure_config.get('generate_plots', True):
            try:
                from spatial_quantification.visualization.per_tumor_plotter import PerTumorPlotter
                plotter = PerTumorPlotter(per_structure.output_dir, config)
                plotter.generate_all_plots(all_results['per_structure_analysis'])
            except Exception as e:
                import traceback
                print(f"  ⚠ Could not generate per-structure plots: {e}")
                print(f"     {traceback.format_exc()}")

        if use_spatialcells and region_detector is not None:
            if per_structure_config.get('generate_spatial_plots', True):
                try:
                    from spatial_quantification.visualization.spatial_plotter import SpatialPlotter
                    spatial_plotter = SpatialPlotter(per_structure.output_dir, config)
                    spatial_plotter.generate_spatialcells_plots(adata, region_detector)
                    spatial_plotter.plot_raw_fluorescence_spatial(adata, markers=['B220', 'GL7', 'BCL6', 'CD3'])
                except Exception as e:
                    import traceback
                    print(f"  ⚠ Could not generate spatial plots: {e}")
                    print(f"     {traceback.format_exc()}")

    # pERK MFI Analysis
    if config.get('perk_mfi_analysis', {}).get('enabled', False):
        print("\n  Running pERK MFI analysis")
        try:
            from spatial_quantification.analyses.perk_mfi_analysis import PerkMFIAnalysis
            perk_mfi = PerkMFIAnalysis(adata, config, output_dir)
            all_results['perk_mfi_analysis'] = perk_mfi.run()
        except Exception as e:
            import traceback
            print(f"  ⚠ pERK MFI analysis failed: {e}")
            traceback.print_exc()

    # UMAP Visualization
    if config.get('umap_visualization', {}).get('enabled', False):
        print("\n  Generating UMAP visualizations...")
        try:
            from spatial_quantification.visualization.umap_plotter import UMAPPlotter
            umap_plotter = UMAPPlotter(output_dir / 'umap_visualization', config)
            umap_plotter.generate_all_plots(adata)
        except Exception as e:
            import traceback
            print(f"  ⚠ Could not generate UMAP plots: {e}")
            traceback.print_exc()

    # Spatial Permutation Testing
    if config.get('spatial_permutation', {}).get('enabled', False):
        print("\n  Running spatial permutation testing")
        try:
            from spatial_quantification.analyses.spatial_permutation_testing import SpatialPermutationTesting
            permutation_testing = SpatialPermutationTesting(adata, config, output_dir)
            all_results['spatial_permutation'] = permutation_testing.run()

            if config.get('spatial_permutation', {}).get('generate_plots', True):
                print("\n  Generating permutation testing visualizations...")
                try:
                    from spatial_quantification.visualization.permutation_plotter import PermutationPlotter
                    perm_plotter = PermutationPlotter(output_dir / 'spatial_permutation', config)
                    perm_plotter.generate_all_plots(all_results['spatial_permutation'])
                except Exception as e:
                    import traceback
                    print(f"  ⚠ Could not generate permutation plots: {e}")
                    traceback.print_exc()
        except Exception as e:
            import traceback
            print(f"  ⚠ Could not run spatial permutation testing: {e}")
            traceback.print_exc()

    # Temporal Analysis
    if config.get('temporal_analysis', {}).get('enabled', False):
        print("\n  Running temporal analysis...")
        try:
            from spatial_quantification.stats.temporal import TemporalAnalysis
            temporal = TemporalAnalysis(adata, config, output_dir)
            all_results['temporal_analysis'] = temporal.run()
        except Exception as e:
            import traceback
            print(f"  ⚠ Error in temporal analysis: {e}")
            print(f"     {traceback.format_exc()}")

    # Cluster Composition Analysis
    if config.get('cluster_composition_analysis', {}).get('enabled', False):
        print("\n  Running cluster composition analysis...")
        try:
            from spatial_quantification.analyses.cluster_composition_analysis import ClusterCompositionAnalysis
            cluster_comp = ClusterCompositionAnalysis(adata, config, output_dir, tumor_structures=tumor_structures)
            all_results['cluster_composition_analysis'] = cluster_comp.run()
        except Exception as e:
            import traceback
            print(f"  ⚠ Error in cluster composition analysis: {e}")
            print(f"     {traceback.format_exc()}")

    # KPNT Correlation Analysis
    if config.get('kpnt_correlation_analysis', {}).get('enabled', False):
        print("\n  Running KPNT correlation analysis...")
        try:
            from spatial_quantification.analyses.kpnt_correlation_analysis import KPNTCorrelationAnalysis
            kpnt_corr = KPNTCorrelationAnalysis(
                adata, config, output_dir,
                per_tumor_results=all_results.get('per_structure_analysis')
            )
            all_results['kpnt_correlation_analysis'] = kpnt_corr.run()
        except Exception as e:
            import traceback
            print(f"  ⚠ Error in KPNT correlation analysis: {e}")
            print(f"     {traceback.format_exc()}")

    # Comprehensive Coexpression Analysis
    if config.get('coexpression_analysis', {}).get('enabled', True):
        print("\n  Running comprehensive coexpression analysis...")
        try:
            coexpression = CoexpressionAnalysisComprehensive(adata, config, output_dir)
            all_results['coexpression_analysis'] = coexpression.run()
        except Exception as e:
            import traceback
            print(f"  ⚠ Error in coexpression analysis: {e}")
            print(f"     {traceback.format_exc()}")

    # Spatial Overlap Analysis
    if config.get('spatial_overlap_analysis', {}).get('enabled', True):
        print("\n  Running spatial overlap analysis...")
        try:
            from spatial_quantification.analyses.spatial_overlap_analysis import SpatialOverlapAnalysis
            overlap_analysis = SpatialOverlapAnalysis(adata, config, output_dir)
            all_results['spatial_overlap_analysis'] = overlap_analysis.run()
        except Exception as e:
            import traceback
            print(f"  ⚠ Error in spatial overlap analysis: {e}")
            print(f"     {traceback.format_exc()}")

    # Population Dynamics
    if config.get('population_dynamics', {}).get('enabled', False):
        pop_dynamics = PopulationDynamics(adata, config, output_dir)
        all_results['population_dynamics'] = pop_dynamics.run()

    # Distance Analysis
    if config.get('distance_analysis', {}).get('enabled', False):
        distance_analysis = DistanceAnalysis(adata, config, output_dir)
        all_results['distance_analysis'] = distance_analysis.run()

    # Distance Permutation Testing
    if config.get('distance_permutation_testing', {}).get('enabled', False):
        print("\n  Running distance permutation testing")
        try:
            from spatial_quantification.analyses.distance_permutation_testing import DistancePermutationTesting
            dist_perm = DistancePermutationTesting(adata, config, output_dir)
            all_results['distance_permutation'] = dist_perm.run()

            if config.get('distance_permutation_testing', {}).get('generate_plots', True):
                print("\n  Generating distance permutation visualizations...")
                try:
                    from spatial_quantification.visualization.distance_permutation_plotter import DistancePermutationPlotter
                    dist_perm_plotter = DistancePermutationPlotter(output_dir / 'distance_permutation', config)
                    dist_perm_plotter.generate_all_plots(all_results['distance_permutation'])
                except Exception as e:
                    import traceback
                    print(f"  ⚠ Could not generate distance permutation plots: {e}")
                    traceback.print_exc()
        except Exception as e:
            import traceback
            print(f"  ⚠ Could not run distance permutation testing: {e}")
            traceback.print_exc()

    # Infiltration Analysis
    if config.get('immune_infiltration', {}).get('enabled', False):
        use_spatialcells = config.get('immune_infiltration', {}).get('use_spatialcells', False)

        if use_spatialcells:
            print("  Using SpatialCells-based infiltration analysis (boundary distances + immune regions)")
            infiltration_analysis = InfiltrationAnalysisSpatialCells(
                adata, config, output_dir, region_detector=region_detector
            )
        else:
            print("  Using optimized infiltration analysis (Getis-Ord Gi* + Ripley's K)")
            infiltration_analysis = InfiltrationAnalysisOptimized(adata, config, output_dir)

        all_results['infiltration_analysis'] = infiltration_analysis.run()

        print("\n  Generating infiltration summary visualizations...")
        try:
            if use_spatialcells:
                from spatial_quantification.visualization.infiltration_spatialcells_plotter import InfiltrationSpatialCellsPlotter
                infiltration_plotter = InfiltrationSpatialCellsPlotter(
                    output_dir / 'infiltration_analysis_spatialcells', config
                )
            else:
                from spatial_quantification.visualization.infiltration_plotter import InfiltrationPlotter
                infiltration_plotter = InfiltrationPlotter(output_dir / 'infiltration_analysis', config)
            infiltration_plotter.generate_all_plots(all_results['infiltration_analysis'])
        except Exception as e:
            import traceback
            print(f"  ⚠ Could not generate infiltration plots: {e}")
            print(f"  {traceback.format_exc()}")

    # Neighborhood Analysis
    if config.get('cellular_neighborhoods', {}).get('enabled', False):
        print("  Using optimized neighborhood analysis (windowed + HNSW)")
        neighborhood_analysis = NeighborhoodAnalysisOptimized(adata, config, output_dir)
        all_results['neighborhood_analysis'] = neighborhood_analysis.run()

    # Neighborhood Enrichment Permutation Testing
    if config.get('neighborhood_permutation_testing', {}).get('enabled', False):
        print("\n  Running neighborhood enrichment permutation testing")
        try:
            from spatial_quantification.analyses.neighborhood_permutation_testing import NeighborhoodPermutationTesting
            nhood_perm = NeighborhoodPermutationTesting(adata, config, output_dir)
            all_results['neighborhood_permutation'] = nhood_perm.run()

            if config.get('neighborhood_permutation_testing', {}).get('generate_plots', True):
                print("\n  Generating neighborhood permutation visualizations...")
                try:
                    from spatial_quantification.visualization.neighborhood_permutation_plotter import NeighborhoodPermutationPlotter
                    nhood_plotter = NeighborhoodPermutationPlotter(output_dir / 'neighborhood_permutation', config)
                    nhood_plotter.generate_all_plots(all_results['neighborhood_permutation'])
                except Exception as e:
                    import traceback
                    print(f"  ⚠ Could not generate neighborhood permutation plots: {e}")
                    traceback.print_exc()
        except Exception as e:
            import traceback
            print(f"  ⚠ Could not run neighborhood permutation testing: {e}")
            traceback.print_exc()

    # Advanced Analysis
    if config.get('advanced_analyses', {}).get('enabled', False):
        advanced_analysis = AdvancedAnalysis(adata, config, output_dir)
        all_results['advanced_analysis'] = advanced_analysis.run()

    # Enhanced Neighborhood Analysis (marker-specific regions)
    if config.get('enhanced_neighborhoods', {}).get('enabled', False):
        enhanced_neighborhoods = EnhancedNeighborhoodAnalysis(
            adata, config, output_dir, tumor_structures=tumor_structures
        )
        all_results['enhanced_neighborhoods'] = enhanced_neighborhoods.run()

        if config.get('enhanced_neighborhoods', {}).get('generate_plots', True):
            print("\n  Generating enhanced neighborhood visualizations...")
            try:
                from spatial_quantification.visualization.enhanced_neighborhood_plotter import EnhancedNeighborhoodPlotter
                enh_plotter = EnhancedNeighborhoodPlotter(output_dir / 'enhanced_neighborhoods', config)
                enh_plotter.generate_all_plots(all_results['enhanced_neighborhoods'])
            except Exception as e:
                print(f"  ⚠ Could not generate enhanced neighborhood plots: {e}")

    # Tumor Microenvironment Analysis
    if config.get('tumor_microenvironment', {}).get('enabled', False):
        print("\n  Running tumor microenvironment analysis...")
        tumor_microenv = TumorMicroenvironmentAnalysis(
            adata, config, output_dir, tumor_structures=tumor_structures
        )
        all_results['tumor_microenvironment'] = tumor_microenv.run()

        if config.get('tumor_microenvironment', {}).get('generate_plots', True):
            print("\n  Generating tumor microenvironment visualizations...")
            try:
                from spatial_quantification.visualization.tumor_microenvironment_plotter import TumorMicroenvironmentPlotter
                tm_plotter = TumorMicroenvironmentPlotter(output_dir / 'tumor_microenvironment', config)
                tm_plotter.generate_all_plots(all_results['tumor_microenvironment'])
            except Exception as e:
                import traceback
                print(f"  ⚠ Could not generate tumor microenvironment plots: {e}")
                traceback.print_exc()

    # Pseudotime Analysis
    if config.get('pseudotime_analysis', {}).get('enabled', False):
        pseudotime = PseudotimeAnalysis(adata, config, output_dir)
        all_results['pseudotime_analysis'] = pseudotime.run()

        if config.get('pseudotime_analysis', {}).get('generate_plots', True):
            print("\n  Generating pseudotime visualizations...")
            try:
                from spatial_quantification.visualization.pseudotime_plotter import PseudotimePlotter
                pseudotime_plotter = PseudotimePlotter(output_dir / 'pseudotime_analysis', config)
                pseudotime_plotter.generate_all_plots(all_results['pseudotime_analysis'])
            except Exception as e:
                print(f"  ⚠ Could not generate pseudotime plots: {e}")

    # Marker Region Analysis (SpatialCells)
    if config.get('marker_region_analysis', {}).get('enabled', False):
        print("\n  Running marker region analysis (pERK+/-, Ki67+/-, etc.)")
        try:
            from spatial_quantification.analyses import MarkerRegionAnalysisSpatialCells
            marker_region = MarkerRegionAnalysisSpatialCells(adata, config, output_dir)
            all_results['marker_region_analysis'] = marker_region.run()

            if config.get('marker_region_analysis', {}).get('generate_plots', True):
                print("\n  Generating marker region visualizations...")
                try:
                    from spatial_quantification.visualization.marker_region_plotter import MarkerRegionPlotter
                    mr_plotter = MarkerRegionPlotter(output_dir / 'marker_region_analysis', config)
                    mr_plotter.generate_all_plots(all_results['marker_region_analysis'], marker_region)
                except Exception as e:
                    import traceback
                    print(f"  ⚠ Could not generate marker region plots: {e}")
                    traceback.print_exc()
        except Exception as e:
            import traceback
            print(f"  ⚠ Could not run marker region analysis: {e}")
            traceback.print_exc()

    # Marker Clustering Analysis
    if config.get('marker_clustering_analysis', {}).get('enabled', False):
        print("\n  Running marker clustering analysis with randomization testing")
        try:
            from spatial_quantification.analyses.marker_clustering_analysis import MarkerClusteringAnalysis
            marker_clustering = MarkerClusteringAnalysis(adata, config, output_dir)
            all_results['marker_clustering_analysis'] = marker_clustering.run()

            if config.get('marker_clustering_analysis', {}).get('generate_plots', True):
                print("\n  Generating marker clustering visualizations...")
                try:
                    from spatial_quantification.visualization.marker_clustering_plotter import MarkerClusteringPlotter
                    mc_plotter = MarkerClusteringPlotter(
                        all_results['marker_clustering_analysis'],
                        output_dir / 'marker_clustering_analysis',
                        config
                    )
                    mc_plotter.plot_all()
                except Exception as e:
                    import traceback
                    print(f"  ⚠ Could not generate marker clustering plots: {e}")
                    traceback.print_exc()
        except Exception as e:
            import traceback
            print(f"  ⚠ Could not run marker clustering analysis: {e}")
            traceback.print_exc()

    # =========================================================================
    # STEP 5: Generate Plots
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("="*80)

    if config['visualization'].get('enabled', True):
        plot_manager = PlotManager(config, output_dir)

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
            import traceback
            print(f"  ⚠ Could not generate spatial visualizations: {e}")
            traceback.print_exc()

    # =========================================================================
    # COMPLETE
    # =========================================================================
    print("\n" + "="*80)
    print("SPATIAL QUANTIFICATION COMPLETE")
    print("="*80)
    print(f"\n  Results saved to: {output_dir}/")
    print("\n  Analyses completed:")
    for analysis_name in all_results.keys():
        print(f"    - {analysis_name}")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
