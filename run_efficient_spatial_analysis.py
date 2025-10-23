#!/usr/bin/env python3
"""
Runner script for memory-efficient tumor spatial analysis.

This script orchestrates the complete spatial analysis pipeline with minimal
memory footprint, suitable for datasets with millions of cells.

Usage:
    python run_efficient_spatial_analysis.py --config configs/efficient_spatial_config.yaml
    python run_efficient_spatial_analysis.py --config configs/efficient_spatial_config.yaml --resume

The --resume flag allows you to resume from a previous run if structure detection
is already complete.

Author: AI-assisted development for cifsQuant
Date: 2025-10-23
"""

import argparse
import yaml
import sys
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from tumor_spatial_analysis_efficient import EfficientTumorSpatialAnalysis


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    print(f"Loading configuration from: {config_path}\n")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_sample_metadata(metadata_path: str) -> pd.DataFrame:
    """Load sample metadata CSV."""
    if not Path(metadata_path).exists():
        print(f"WARNING: Metadata file not found: {metadata_path}")
        print("  Continuing without temporal/group analysis\n")
        return pd.DataFrame()

    metadata = pd.read_csv(metadata_path)
    print(f"Loaded metadata for {len(metadata)} samples")

    # Display metadata structure
    print(f"  Columns: {list(metadata.columns)}")

    if 'timepoint' in metadata.columns:
        print(f"  Timepoints: {sorted(metadata['timepoint'].unique())}")

    if 'group' in metadata.columns:
        print(f"  Groups: {metadata['group'].unique()}")

    print()
    return metadata


def validate_config(config: dict) -> bool:
    """Validate configuration."""
    required_fields = ['input_data', 'sample_metadata', 'tumor_markers',
                      'immune_markers', 'populations']

    for field in required_fields:
        if field not in config:
            print(f"ERROR: Required field '{field}' missing from configuration")
            return False

    return True


def parse_population_config(config: dict) -> dict:
    """Convert YAML population config to expected format."""
    population_config = {}

    for pop_name, pop_def in config['populations'].items():
        population_config[pop_name] = {
            'markers': pop_def['markers'],
            'color': pop_def.get('color', '#999999'),
            'parent': pop_def.get('parent', None)
        }

    return population_config


def run_analysis(config: dict, resume: bool = False):
    """
    Run complete memory-efficient spatial analysis pipeline.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    resume : bool
        If True, resume from existing structure detection
    """

    # Validate config
    if not validate_config(config):
        sys.exit(1)

    # Load data
    print("="*70)
    print("LOADING DATA")
    print("="*70 + "\n")

    input_path = config['input_data']
    if not Path(input_path).exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading: {input_path}")
    adata = sc.read_h5ad(input_path)
    print(f"  Loaded {len(adata):,} cells")
    print(f"  Markers: {list(adata.var_names)}")

    if 'sample_id' in adata.obs:
        print(f"  Samples: {adata.obs['sample_id'].nunique()}")
    else:
        print("  ERROR: 'sample_id' column not found in adata.obs")
        sys.exit(1)

    # Load sample metadata
    metadata_path = config['sample_metadata']
    sample_metadata = load_sample_metadata(metadata_path)

    # Merge timepoint/group info into adata if not already present
    if len(sample_metadata) > 0:
        if 'timepoint' not in adata.obs and 'timepoint' in sample_metadata.columns:
            sample_to_timepoint = dict(zip(sample_metadata['sample_id'],
                                          sample_metadata['timepoint']))
            adata.obs['timepoint'] = adata.obs['sample_id'].map(sample_to_timepoint)

        if 'group' not in adata.obs and 'group' in sample_metadata.columns:
            sample_to_group = dict(zip(sample_metadata['sample_id'],
                                      sample_metadata['group']))
            adata.obs['group'] = adata.obs['sample_id'].map(sample_to_group)

        if 'condition' not in adata.obs and 'condition' in sample_metadata.columns:
            sample_to_condition = dict(zip(sample_metadata['sample_id'],
                                          sample_metadata['condition']))
            adata.obs['condition'] = adata.obs['sample_id'].map(sample_to_condition)

    # Output directory
    output_dir = config.get('output_directory', 'efficient_spatial_analysis')

    # Initialize framework
    print("\n" + "="*70)
    print("INITIALIZING FRAMEWORK")
    print("="*70 + "\n")

    etsa = EfficientTumorSpatialAnalysis(
        adata,
        sample_metadata=sample_metadata,
        tumor_markers=config['tumor_markers'],
        immune_markers=config['immune_markers'],
        output_dir=output_dir
    )

    # PHASE 1: Structure Detection
    structure_index_path = Path(output_dir) / 'data' / 'structure_index.csv'

    if resume and structure_index_path.exists():
        print("\n" + "="*70)
        print("RESUMING FROM EXISTING STRUCTURE DETECTION")
        print("="*70 + "\n")
        etsa.structure_index = pd.read_csv(structure_index_path)
        print(f"Loaded {len(etsa.structure_index)} existing structures")
    else:
        population_config = parse_population_config(config)
        struct_config = config.get('tumor_structure_detection', {})

        etsa.detect_all_tumor_structures(
            population_config=population_config,
            min_cluster_size=struct_config.get('min_cluster_size', 50),
            eps=struct_config.get('eps', 30),
            min_samples=struct_config.get('min_samples', 10),
            tumor_population=struct_config.get('tumor_population', 'Tumor')
        )

    # PHASE 2: Per-Structure Analysis
    infiltration_config = config.get('immune_infiltration', {})
    boundary_config = config.get('infiltration_boundaries', {})

    metrics_df = etsa.analyze_structures_individually(
        immune_populations=infiltration_config.get('populations', []),
        boundary_widths=boundary_config.get('boundary_widths', [30, 100, 200]),
        buffer_distance=config.get('buffer_distance', 500),
        batch_size=config.get('checkpoint_batch_size', 50)
    )

    # PHASE 3: Cellular Neighborhoods
    neighborhood_config = config.get('cellular_neighborhoods', {})

    if neighborhood_config.get('enabled', True):
        etsa.detect_cellular_neighborhoods(
            populations=neighborhood_config.get('populations', []),
            k_neighbors=neighborhood_config.get('k_neighbors', 10),
            window_size=neighborhood_config.get('window_size', 100),
            n_clusters=neighborhood_config.get('n_clusters', 10),
            subsample_size=neighborhood_config.get('subsample_size', 100000)
        )

    # PHASE 4: Statistical Analysis
    statistical_config = config.get('statistical_analysis', {})

    if statistical_config.get('enabled', True):
        etsa.statistical_analysis(
            metrics_df=metrics_df,
            alpha=statistical_config.get('alpha', 0.05)
        )

    # PHASE 5: Publication Figures
    viz_config = config.get('visualizations', {})

    if viz_config.get('enabled', True):
        etsa.create_publication_figures(metrics_df)

    # Final Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nDirectory structure:")
    print(f"  {output_dir}/structures/        - Per-structure cell indices")
    print(f"  {output_dir}/data/              - All analysis results (CSV)")
    print(f"  {output_dir}/statistics/        - Statistical test results")
    print(f"  {output_dir}/figures/           - Publication figures")
    print(f"  {output_dir}/neighborhoods/     - Neighborhood analysis")

    print("\nKey outputs:")
    print(f"  - structure_index.csv           : Tumor structure metadata")
    print(f"  - infiltration_metrics.csv      : Per-structure infiltration")
    print(f"  - neighborhood_profiles.csv     : Cellular neighborhood types")
    print(f"  - temporal_trends.csv           : Temporal analysis results")
    print(f"  - group_comparisons.csv         : Group statistical tests")
    print(f"  - summary_statistics.csv        : Overall summary stats")

    print("\nFigures:")
    print(f"  - infiltration_heatmap.png      : Heatmap of infiltration")
    print(f"  - temporal_trends.png           : Temporal changes")
    print(f"  - group_comparisons.png         : Group differences")
    print(f"  - size_correlations.png         : Tumor size effects")
    print(f"  - neighborhood_profiles.png     : Cellular neighborhoods")

    print("\n" + "="*70)

    return etsa, metrics_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run memory-efficient tumor spatial analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete analysis
  python run_efficient_spatial_analysis.py --config configs/efficient_spatial_config.yaml

  # Resume from existing structure detection
  python run_efficient_spatial_analysis.py --config configs/efficient_spatial_config.yaml --resume

  # Run with custom output directory
  python run_efficient_spatial_analysis.py --config myconfig.yaml --output my_results

For more information: https://github.com/bansalis/cifsQuant
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )

    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume from existing structure detection'
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

    # Override output directory if specified
    if args.output:
        config['output_directory'] = args.output

    # Run analysis
    try:
        etsa, metrics_df = run_analysis(config, resume=args.resume)
        print("\n✓ Analysis completed successfully!\n")
        return 0
    except Exception as e:
        print(f"\n✗ Analysis failed with error: {e}\n")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
