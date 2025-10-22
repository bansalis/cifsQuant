#!/usr/bin/env python3
"""
Runner script for comprehensive tumor spatial analysis.

This script loads configuration from YAML and runs the complete spatial analysis
pipeline including tumor structure detection, infiltration quantification, temporal
analysis, co-enrichment, and spatial heterogeneity detection.

Usage:
    python run_tumor_spatial_analysis.py --config configs/tumor_spatial_config.yaml
    python run_tumor_spatial_analysis.py --config configs/tumor_spatial_config.yaml --output custom_output_dir
"""

import argparse
import yaml
import sys
from pathlib import Path
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

from tumor_spatial_analysis import TumorSpatialAnalysis


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: dict) -> bool:
    """Validate that required configuration fields are present."""
    required_fields = ['input_data', 'tumor_markers', 'immune_markers', 'populations']

    for field in required_fields:
        if field not in config:
            print(f"ERROR: Required field '{field}' missing from configuration")
            return False

    return True


def parse_population_config(config: dict) -> dict:
    """Convert YAML population config to format expected by TumorSpatialAnalysis."""
    population_config = {}

    for pop_name, pop_def in config['populations'].items():
        population_config[pop_name] = {
            'markers': pop_def['markers'],
            'color': pop_def.get('color', '#999999'),
            'parent': pop_def.get('parent', None)
        }

    return population_config


def run_analysis(config: dict, output_dir: str = None):
    """Run complete spatial analysis pipeline based on configuration."""

    # Validate config
    if not validate_config(config):
        sys.exit(1)

    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    input_path = config['input_data']
    if not Path(input_path).exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    adata = sc.read_h5ad(input_path)
    print(f"Loaded {len(adata)} cells from {input_path}")
    print(f"  Markers: {list(adata.var_names)}")
    if 'sample_id' in adata.obs:
        print(f"  Samples: {adata.obs['sample_id'].nunique()}")
    if 'timepoint' in adata.obs:
        print(f"  Timepoints: {sorted(adata.obs['timepoint'].unique())}")

    # Override output directory if specified
    if output_dir is None:
        output_dir = config.get('output_directory', 'tumor_spatial_analysis')

    # Initialize analysis framework
    print("\n" + "="*70)
    print("INITIALIZING ANALYSIS FRAMEWORK")
    print("="*70)

    tsa = TumorSpatialAnalysis(
        adata,
        tumor_markers=config['tumor_markers'],
        immune_markers=config['immune_markers'],
        output_dir=output_dir
    )

    # Step 1: Define cell populations
    print("\n" + "="*70)
    print("STEP 1: DEFINING CELL POPULATIONS")
    print("="*70)

    population_config = parse_population_config(config)
    tsa.define_cell_populations(population_config)

    # Step 2: Detect tumor structures
    print("\n" + "="*70)
    print("STEP 2: DETECTING TUMOR STRUCTURES")
    print("="*70)

    struct_config = config.get('tumor_structure_detection', {})
    tsa.detect_tumor_structures(
        min_cluster_size=struct_config.get('min_cluster_size', 50),
        eps=struct_config.get('eps', 30),
        min_samples=struct_config.get('min_samples', 10),
        tumor_population=struct_config.get('tumor_population', 'Tumor')
    )

    # Step 3: Define infiltration boundaries
    print("\n" + "="*70)
    print("STEP 3: DEFINING INFILTRATION BOUNDARIES")
    print("="*70)

    boundary_config = config.get('infiltration_boundaries', {})
    tsa.define_infiltration_boundaries(
        boundary_widths=boundary_config.get('boundary_widths', [30, 100, 200]),
        tumor_population=boundary_config.get('tumor_population', 'Tumor')
    )

    # Step 4: Quantify immune infiltration
    print("\n" + "="*70)
    print("STEP 4: QUANTIFYING IMMUNE INFILTRATION")
    print("="*70)

    infiltration_config = config.get('immune_infiltration', {})
    infiltration_df = tsa.quantify_immune_infiltration(
        immune_populations=infiltration_config.get('populations', []),
        by_sample=infiltration_config.get('by_sample', True),
        by_timepoint=infiltration_config.get('by_timepoint', False)
    )

    # Step 5: Temporal analysis (if enabled)
    temporal_config = config.get('temporal_analysis', {})
    if temporal_config.get('enabled', False) and 'timepoint' in adata.obs:
        print("\n" + "="*70)
        print("STEP 5: TEMPORAL ANALYSIS")
        print("="*70)

        tsa.analyze_temporal_changes(
            timepoint_col=temporal_config.get('timepoint_column', 'timepoint'),
            populations=temporal_config.get('populations_to_track', []),
            marker_trends=temporal_config.get('marker_trends', [])
        )
    else:
        print("\n[STEP 5: Temporal analysis SKIPPED - not enabled or no timepoint data]")

    # Step 6: Co-enrichment analysis
    print("\n" + "="*70)
    print("STEP 6: CO-ENRICHMENT ANALYSIS")
    print("="*70)

    coenrich_config = config.get('coenrichment_analysis', {})
    if coenrich_config.get('population_pairs', []):
        enrichment_df = tsa.analyze_coenrichment(
            population_pairs=coenrich_config['population_pairs'],
            radius=coenrich_config.get('radius', 50),
            n_permutations=coenrich_config.get('n_permutations', 100)
        )
    else:
        print("[SKIPPED - no population pairs defined]")
        enrichment_df = None

    # Step 7: Spatial heterogeneity detection
    print("\n" + "="*70)
    print("STEP 7: SPATIAL HETEROGENEITY DETECTION")
    print("="*70)

    heterogeneity_config = config.get('spatial_heterogeneity', {})
    heterogeneity_df = tsa.detect_spatial_heterogeneity(
        tumor_population=heterogeneity_config.get('tumor_population', 'Tumor'),
        heterogeneity_markers=heterogeneity_config.get('heterogeneity_markers', []),
        n_regions=heterogeneity_config.get('n_regions', 3),
        min_region_size=heterogeneity_config.get('min_region_size', 100)
    )

    # Step 8: Region infiltration comparison
    print("\n" + "="*70)
    print("STEP 8: COMPARING INFILTRATION ACROSS REGIONS")
    print("="*70)

    region_config = config.get('region_infiltration_comparison', {})
    region_infiltration_df = tsa.compare_region_infiltration(
        immune_populations=region_config.get('immune_populations', []),
        region_col=region_config.get('region_column', 'heterogeneity_region')
    )

    # Step 9: Generate visualizations
    print("\n" + "="*70)
    print("STEP 9: GENERATING VISUALIZATIONS")
    print("="*70)

    viz_config = config.get('visualizations', {})

    if viz_config.get('spatial_overview', True):
        print("\n  Creating spatial overview...")
        tsa.plot_spatial_overview(save=True)

    if viz_config.get('temporal_trends', True) and tsa.temporal_metrics:
        print("\n  Creating temporal trends...")
        tsa.plot_temporal_trends(save=True)

    if viz_config.get('infiltration_heatmap', True) and infiltration_df is not None:
        print("\n  Creating infiltration heatmap...")
        tsa.plot_infiltration_heatmap(infiltration_df, save=True)

    # Step 10: Generate comprehensive report
    print("\n" + "="*70)
    print("STEP 10: GENERATING COMPREHENSIVE REPORT")
    print("="*70)

    report = tsa.generate_comprehensive_report()

    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - Data files: {output_dir}/data/")
    print(f"  - Figures: {output_dir}/figures/")
    print(f"  - Report: {output_dir}/analysis_report.md")

    return tsa


def main():
    """Main entry point for the analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive tumor spatial analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python run_tumor_spatial_analysis.py

  # Run with custom config
  python run_tumor_spatial_analysis.py --config my_config.yaml

  # Run with custom output directory
  python run_tumor_spatial_analysis.py --output my_analysis_results

For more information, see: https://github.com/yourusername/cifsQuant
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/tumor_spatial_config.yaml',
        help='Path to YAML configuration file (default: configs/tumor_spatial_config.yaml)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (overrides config file setting)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML configuration: {e}")
        sys.exit(1)

    # Run analysis
    try:
        tsa = run_analysis(config, output_dir=args.output)
        print("\n✓ Analysis completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Analysis failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
