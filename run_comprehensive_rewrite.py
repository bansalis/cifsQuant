#!/usr/bin/env python3
"""
Runner Script for Comprehensive Spatial Analysis Rewrite

This script runs the completely redesigned spatial analysis pipeline that:

1. Analyzes at DUAL levels: per-sample AND per-tumor-structure
2. Calculates distances from EVERY tumor type to EVERY immune type
3. Analyzes EVERY marker region over time
4. ONLY compares KPT vs KPNT (never cis vs trans alone)
5. Generates individual + combo plots for everything
6. Generates raw scatter + summary bar plots
7. Only creates directories with actual content
8. Validates all outputs

Usage:
    python run_comprehensive_rewrite.py \
        --input manual_gating_output/gated_data.h5ad \
        --metadata sample_metadata.csv \
        --output comprehensive_spatial_output_new

Author: Complete pipeline rewrite
Date: 2025-10-31
"""

import argparse
import sys
from pathlib import Path
import scanpy as sc
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from comprehensive_spatial_rewrite import (
    ComprehensiveSpatialAnalysisRewrite,
    ComprehensivePlotGenerator
)


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Spatial Analysis - Complete Rewrite'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Path to h5ad file with gated data')
    parser.add_argument('--metadata', type=str, required=True,
                       help='Path to sample metadata CSV')
    parser.add_argument('--output', type=str,
                       default='comprehensive_spatial_output_new',
                       help='Output directory')
    parser.add_argument('--tumor-eps', type=float, default=30,
                       help='DBSCAN eps for tumor structure detection')
    parser.add_argument('--tumor-min-samples', type=int, default=10,
                       help='DBSCAN min_samples for tumor structure detection')
    parser.add_argument('--tumor-min-size', type=int, default=50,
                       help='Minimum tumor structure size')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("COMPREHENSIVE SPATIAL ANALYSIS - COMPLETE REWRITE")
    print("="*80 + "\n")

    # Load data
    print(f"Loading data from: {args.input}")
    adata = sc.read_h5ad(args.input)
    print(f"  Loaded {len(adata):,} cells")

    # Load metadata
    print(f"\nLoading metadata from: {args.metadata}")
    metadata = pd.read_csv(args.metadata)
    print(f"  Loaded {len(metadata)} samples")

    # Initialize analysis
    print("\nInitializing comprehensive spatial analysis...")
    analysis = ComprehensiveSpatialAnalysisRewrite(
        adata=adata,
        sample_metadata=metadata,
        output_dir=args.output
    )

    # Detect tumor structures
    print("\n" + "="*80)
    print("PHASE 1: TUMOR STRUCTURE DETECTION")
    print("="*80)
    analysis.detect_tumor_structures(
        tumor_population='Tumor',
        eps=args.tumor_eps,
        min_samples=args.tumor_min_samples,
        min_cluster_size=args.tumor_min_size
    )

    # Run comprehensive distance analysis
    print("\n" + "="*80)
    print("PHASE 2: COMPREHENSIVE DISTANCE ANALYSIS")
    print("="*80)
    print("\nCalculating distances from ALL tumor types to ALL immune types...")
    print("At BOTH per-sample AND per-tumor-structure levels...")
    distance_results = analysis.analyze_distances_comprehensive()

    # Run marker region analysis
    print("\n" + "="*80)
    print("PHASE 3: PER-MARKER REGION TEMPORAL ANALYSIS")
    print("="*80)
    print("\nAnalyzing ALL markers over time...")
    marker_results = analysis.analyze_marker_regions_temporal()

    # Generate all plots
    print("\n" + "="*80)
    print("PHASE 4: COMPREHENSIVE PLOT GENERATION")
    print("="*80)
    print("\nGenerating ALL plots (individual + combo, scatter + bar)...")

    plot_generator = ComprehensivePlotGenerator(
        results=analysis.results,
        output_dir=analysis.output_dir
    )
    plot_generator.generate_all_plots()

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")

    print("Results Summary:")
    print(f"  Output directory: {args.output}/")
    print(f"\n  Tumor structures detected: {len(analysis.tumor_structures)}")
    print(f"  Distance measurements: {len(distance_results):,}")
    print(f"    - Structure-level: {len(distance_results[distance_results['level']=='structure']):,}")
    print(f"    - Sample-level: {len(distance_results[distance_results['level']=='sample']):,}")
    print(f"  Marker measurements: {len(marker_results):,}")
    print(f"    - Structure-level: {len(marker_results[marker_results['level']=='structure']):,}")
    print(f"    - Sample-level: {len(marker_results[marker_results['level']=='sample']):,}")

    print("\n  Output structure:")
    print(f"    {args.output}/")
    print(f"      ├── distance_analysis/")
    print(f"      │   └── comprehensive_distances.csv")
    print(f"      ├── marker_regions/")
    print(f"      │   └── marker_regions_temporal.csv")
    print(f"      └── figures/")
    print(f"          ├── distance_analysis/")
    print(f"          │   ├── individual_plots/")
    print(f"          │   │   ├── *_temporal.png")
    print(f"          │   │   ├── *_boxplot.png")
    print(f"          │   │   └── *_scatter.png")
    print(f"          │   ├── combo_plots/")
    print(f"          │   │   └── *_combo.png (3x3 grids)")
    print(f"          │   └── distance_summary_heatmaps.png")
    print(f"          └── marker_regions/")
    print(f"              ├── individual_plots/")
    print(f"              │   ├── *_temporal.png")
    print(f"              │   ├── *_boxplot.png")
    print(f"              │   └── *_scatter.png")
    print(f"              └── combo_plots/")
    print(f"                  └── *_combo.png (3x3 grids)")

    print("\n" + "="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
