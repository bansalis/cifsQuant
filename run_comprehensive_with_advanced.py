#!/usr/bin/env python3
"""
COMPREHENSIVE Spatial Analysis with Advanced Extensions Runner

This script demonstrates how to run the complete spatial analysis pipeline
(Phases 1-10) followed by the advanced analysis (Phases 11-18).

Usage:
    # Run comprehensive analysis only (Phases 1-10)
    python run_comprehensive_with_advanced.py \
        --config configs/comprehensive_config.yaml \
        --metadata sample_metadata.csv

    # Run comprehensive + advanced analysis (Phases 1-18)
    python run_comprehensive_with_advanced.py \
        --config configs/advanced_spatial_config.yaml \
        --metadata sample_metadata.csv \
        --run-advanced

Author: Integrated pipeline runner
Date: 2025-10-29
"""

import argparse
import yaml
import sys
from pathlib import Path
import scanpy as sc
import pandas as pd

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Import existing comprehensive analysis framework
from tumor_spatial_analysis_comprehensive import ComprehensiveTumorSpatialAnalysis

# Import advanced extensions
from advanced_spatial_extensions import add_advanced_methods


def parse_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Parse metadata with proper group extraction.
    """
    metadata = pd.read_csv(metadata_path)

    # Standardize sample_id
    metadata['sample_id'] = metadata['sample_id'].str.upper()

    # Extract main group (KPT vs KPNT)
    metadata['main_group'] = metadata['group'].apply(
        lambda x: 'KPT' if 'KPT' in str(x) else 'KPNT'
    )

    # Extract cis/trans
    metadata['genotype'] = metadata['group'].apply(
        lambda x: 'cis' if 'cis' in str(x).lower() else
                 ('trans' if 'trans' in str(x).lower() else 'Unknown')
    )

    # Full group name
    metadata['genotype_full'] = metadata['group']

    # Convert timepoint to numeric
    metadata['timepoint'] = pd.to_numeric(metadata['timepoint'])

    print("=" * 80)
    print("PARSED SAMPLE METADATA")
    print("=" * 80)
    print(f"Samples: {len(metadata)}")
    print(f"Main groups: {sorted(metadata['main_group'].unique())}")
    print(f"Cis/Trans: {sorted(metadata['genotype'].unique())}")
    print(f"Timepoints: {sorted(metadata['timepoint'].unique())}")
    print("=" * 80 + "\n")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Spatial Analysis with Advanced Extensions'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file')
    parser.add_argument('--metadata', type=str, required=True,
                       help='Path to sample metadata CSV')
    parser.add_argument('--run-advanced', action='store_true',
                       help='Run advanced analysis phases (11-18) after comprehensive analysis')
    parser.add_argument('--skip-comprehensive', action='store_true',
                       help='Skip comprehensive analysis (1-10), only run advanced if specified')

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    print(f"\nLoading data from: {config['input_data']}")
    adata = sc.read_h5ad(config['input_data'])

    # Load and parse metadata
    print(f"\nLoading metadata from: {args.metadata}")
    metadata = parse_metadata(args.metadata)

    # Initialize comprehensive analysis
    print("\n" + "=" * 80)
    print("INITIALIZING COMPREHENSIVE TUMOR SPATIAL ANALYSIS")
    print("=" * 80 + "\n")

    analysis = ComprehensiveTumorSpatialAnalysis(
        adata=adata,
        sample_metadata=metadata,
        tumor_markers=config['tumor_markers'],
        immune_markers=config['immune_markers'],
        output_dir=config['output_directory']
    )

    # Run comprehensive analysis (Phases 1-10)
    if not args.skip_comprehensive:
        print("\n" + "=" * 80)
        print("RUNNING COMPREHENSIVE ANALYSIS (PHASES 1-10)")
        print("=" * 80 + "\n")

        analysis.run_complete_analysis(
            population_config=config['populations'],
            immune_populations=config['immune_infiltration']['populations'],
            **config.get('tumor_structure_detection', {})
        )

        print("\n✓ Comprehensive analysis (Phases 1-10) complete")

    # Run advanced analysis (Phases 11-18)
    if args.run_advanced:
        print("\n" + "=" * 80)
        print("RUNNING ADVANCED ANALYSIS (PHASES 11-18)")
        print("=" * 80 + "\n")

        # Add advanced methods to the analysis instance
        add_advanced_methods(analysis)

        # Run advanced analysis
        analysis.run_advanced_analysis(config)

        print("\n✓ Advanced analysis (Phases 11-18) complete")

    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {config['output_directory']}/")

    if not args.skip_comprehensive:
        print("\nComprehensive Analysis Outputs (Phases 1-10):")
        print(f"  - {config['output_directory']}/data/")
        print(f"  - {config['output_directory']}/statistics/")
        print(f"  - {config['output_directory']}/figures/")

    if args.run_advanced:
        print("\nAdvanced Analysis Outputs (Phases 11-18):")
        print(f"  - {config['output_directory']}/advanced_perk_analysis/")
        print(f"  - {config['output_directory']}/advanced_ninja_analysis/")
        print(f"  - {config['output_directory']}/advanced_heterogeneity/")
        print(f"  - {config['output_directory']}/advanced_rcn/")
        print(f"  - {config['output_directory']}/advanced_distances/")
        print(f"  - {config['output_directory']}/advanced_infiltration/")
        print(f"  - {config['output_directory']}/advanced_pseudotime/")

    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
