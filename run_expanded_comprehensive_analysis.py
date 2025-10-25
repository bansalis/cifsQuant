#!/usr/bin/env python3
"""
EXPANDED Comprehensive Spatial Analysis Runner

This script runs the complete spatial analysis with ALL the requested expansions:

1. T cell to tumor cell distances (overall and by tumor subtype)
   - pERK+ vs other tumor cells
   - NINJA+/aGFP+ vs NINJA- tumor cells
   - Boxplots, distributions, scatters with stats over time and between groups

2. Tumor heterogeneity analysis
   - Are NINJA+ and pERK+ regions spatially isolated or randomly distributed?

3. RCNs (cellular neighborhoods) over time
   - KPT vs KPNT comparison
   - NINJA+/- tumor comparisons
   - Neighborhood composition per group across time

4. Dual-level statistics (by sample AND by tumor)

5. Spatial maps with tumor definitions and analysis ranges

6. Extensive marker expression temporal tracking

Usage:
    python run_expanded_comprehensive_analysis.py --config configs/comprehensive_config.yaml

Author: Expanded comprehensive analysis runner
Date: 2025-10-25
"""

import argparse
import yaml
import sys
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import analysis modules
from tumor_spatial_analysis_efficient import EfficientTumorSpatialAnalysis
from spatial_analysis_expansions import (
    ImmuneInfiltrationAnalysis,
    NeighborhoodTemporalAnalysis,
    create_spatial_maps_with_analysis_ranges
)
from comprehensive_visualizations_stats import ComprehensiveVisualizationsStats

# Also import existing comprehensive analysis functions
from run_comprehensive_analysis import (
    parse_metadata_properly,
    merge_metadata_into_adata,
    create_spatial_maps,
    analyze_and_plot_tumor_size,
    analyze_and_plot_marker_expression
)


def main():
    """Main entry point for expanded comprehensive analysis."""

    parser = argparse.ArgumentParser(
        description='Run EXPANDED comprehensive tumor spatial analysis'
    )

    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Configuration file')
    parser.add_argument('--metadata', '-m', type=str, default='sample_metadata.csv',
                       help='Sample metadata CSV')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup
    output_dir = config.get('output_directory', 'expanded_comprehensive_analysis')
    import os
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/structures", exist_ok=True)

    print("\n" + "="*80)
    print("EXPANDED COMPREHENSIVE SPATIAL ANALYSIS")
    print("="*80)
    print(f"\nOutput directory: {output_dir}/\n")

    # Load and parse metadata
    print("="*80)
    print("PHASE 1: DATA LOADING AND METADATA PARSING")
    print("="*80)

    metadata = parse_metadata_properly(args.metadata)

    print("\nLoading spatial data...")
    adata = sc.read_h5ad(config['input_data'])
    print(f"Loaded {len(adata):,} cells")

    merge_metadata_into_adata(adata, metadata)

    # Initialize core analysis framework
    etsa = EfficientTumorSpatialAnalysis(
        adata,
        sample_metadata=metadata,
        tumor_markers=config['tumor_markers'],
        immune_markers=config['immune_markers'],
        output_dir=output_dir
    )

    # Population config
    population_config = {}
    for pop_name, pop_def in config['populations'].items():
        population_config[pop_name] = {
            'markers': pop_def['markers'],
            'color': pop_def.get('color', '#999999')
        }

    # ========================================================================
    # PHASE 2: TUMOR STRUCTURE DETECTION
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: TUMOR STRUCTURE DETECTION")
    print("="*80)

    struct_config = config.get('tumor_structure_detection', {})
    etsa.detect_all_tumor_structures(
        population_config=population_config,
        **struct_config
    )

    # ========================================================================
    # PHASE 3: STANDARD INFILTRATION ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 3: STANDARD INFILTRATION ANALYSIS")
    print("="*80)

    infil_config = config.get('immune_infiltration', {})
    boundary_config = config.get('infiltration_boundaries', {})

    metrics_df = etsa.analyze_structures_individually(
        immune_populations=infil_config.get('populations', []),
        boundary_widths=boundary_config.get('boundary_widths', [30, 100, 200]),
        buffer_distance=config.get('buffer_distance', 500)
    )

    # ========================================================================
    # PHASE 4: SPATIAL MAPS (Basic and with Analysis Ranges)
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 4: SPATIAL MAPS")
    print("="*80)

    # 4a. Basic spatial maps
    create_spatial_maps(adata, output_dir, list(population_config.keys()))

    # 4b. Spatial maps with tumor definitions and analysis ranges
    # Plot ALL samples, not just a subset
    create_spatial_maps_with_analysis_ranges(
        adata,
        etsa.structure_index,
        output_dir,
        samples_to_plot=adata.obs['sample_id'].unique(),  # Plot ALL samples
        boundary_widths=boundary_config.get('boundary_widths', [30, 100, 200])
    )

    # ========================================================================
    # PHASE 5: TUMOR SIZE ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 5: TUMOR SIZE ANALYSIS")
    print("="*80)

    size_df = analyze_and_plot_tumor_size(etsa.structure_index, output_dir)

    # ========================================================================
    # PHASE 6: MARKER EXPRESSION TEMPORAL ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 6: MARKER EXPRESSION TEMPORAL ANALYSIS")
    print("="*80)

    # Include ALL markers
    all_markers = config['tumor_markers'] + config['immune_markers']

    # Add specific markers of interest if not already included
    # NOTE: Use actual marker names from data (AGFP, PERK - uppercase)
    # CD8B is the actual marker name for CD8, CD4 is not available in this dataset
    markers_of_interest = ['AGFP', 'PERK', 'CD3', 'CD8B']
    for marker in markers_of_interest:
        if marker not in all_markers and marker in adata.var_names:
            all_markers.append(marker)

    marker_df = analyze_and_plot_marker_expression(adata, output_dir, all_markers)

    # ========================================================================
    # PHASE 7: EXPANDED IMMUNE INFILTRATION ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 7: EXPANDED IMMUNE INFILTRATION ANALYSIS")
    print("="*80)

    # Initialize immune infiltration analyzer
    immune_analyzer = ImmuneInfiltrationAnalysis(
        adata,
        etsa.structure_index,
        output_dir
    )

    # Define T cell populations
    # Note: CD4 marker is not available in this dataset, using CD3 and CD8 (CD8B)
    tcell_populations = ['CD3', 'CD8']

    # Define tumor subtypes
    # NOTE: Use actual marker names from data (AGFP, PERK - uppercase)
    tumor_subtypes = {
        'Tumor_all': {'is_Tumor': True},
        'pERK_positive': {'PERK': True, 'is_Tumor': True},
        'pERK_negative': {'PERK': False, 'is_Tumor': True},
        'NINJA_positive': {'AGFP': True, 'is_Tumor': True},
        'NINJA_negative': {'AGFP': False, 'is_Tumor': True}
    }

    # 7a. T cell-tumor distance analysis (dual-level: structure and sample)
    print("\n7a. T cell-tumor distance analysis...")
    structure_distances, sample_distances = immune_analyzer.analyze_tcell_tumor_distances_comprehensive(
        tcell_populations=tcell_populations,
        tumor_subtypes=tumor_subtypes,
        max_distance=500
    )

    # 7b. Tumor heterogeneity analysis
    print("\n7b. Tumor heterogeneity analysis...")
    heterogeneity_df = immune_analyzer.analyze_tumor_heterogeneity(
        markers_of_interest=['AGFP', 'PERK'],
        window_size=100
    )

    # 7c. NINJA+/- tumor comparison
    print("\n7c. NINJA+/- tumor comparison...")
    ninja_df = immune_analyzer.compare_ninja_positive_negative_tumors()

    # ========================================================================
    # PHASE 8: CELLULAR NEIGHBORHOODS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 8: CELLULAR NEIGHBORHOODS")
    print("="*80)

    neighborhood_config = config.get('cellular_neighborhoods', {})

    if neighborhood_config.get('enabled', True):
        # Detect neighborhoods
        etsa.detect_cellular_neighborhoods(
            populations=list(population_config.keys()),
            **{k: v for k, v in neighborhood_config.items() if k not in ['enabled', 'populations']}
        )

        # Load neighborhood results
        cell_neighborhoods = pd.read_csv(f"{output_dir}/data/cell_neighborhoods.csv")
        neighborhood_profiles = pd.read_csv(f"{output_dir}/data/neighborhood_profiles.csv")

        # Temporal neighborhood analysis
        neighborhood_analyzer = NeighborhoodTemporalAnalysis(
            adata,
            cell_neighborhoods,
            neighborhood_profiles,
            output_dir
        )

        neighborhood_composition = neighborhood_analyzer.analyze_neighborhood_composition_temporal()
    else:
        neighborhood_composition = pd.DataFrame()

    # ========================================================================
    # PHASE 9: COMPREHENSIVE VISUALIZATIONS AND STATISTICS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 9: COMPREHENSIVE VISUALIZATIONS AND STATISTICS")
    print("="*80)

    # Initialize visualization generator
    viz_gen = ComprehensiveVisualizationsStats(output_dir)

    # 9a. T cell-tumor distance plots
    print("\n9a. T cell-tumor distance visualizations...")
    viz_gen.plot_tcell_tumor_distances_comprehensive(
        structure_distances,
        sample_distances
    )

    # 9b. Tumor heterogeneity plots
    print("\n9b. Tumor heterogeneity visualizations...")
    viz_gen.plot_tumor_heterogeneity(
        heterogeneity_df,
        markers=['AGFP', 'PERK']
    )

    # 9c. Neighborhood composition temporal plots
    if len(neighborhood_composition) > 0:
        print("\n9c. Neighborhood composition temporal visualizations...")
        viz_gen.plot_neighborhood_composition_temporal(neighborhood_composition)

    # ========================================================================
    # PHASE 10: ADDITIONAL STATISTICAL ANALYSES
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 10: ADDITIONAL STATISTICAL ANALYSES")
    print("="*80)

    # Standard statistical analysis from efficient framework
    if config.get('statistical_analysis', {}).get('enabled', True):
        etsa.statistical_analysis(metrics_df)

    # Additional heterogeneity statistics
    print("\nComputing heterogeneity statistics...")
    _compute_heterogeneity_statistics(heterogeneity_df, output_dir)

    # NINJA+/- comparison statistics
    print("\nComputing NINJA+/- comparison statistics...")
    _compute_ninja_statistics(ninja_df, output_dir)

    # ========================================================================
    # PHASE 11: PUBLICATION FIGURES
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 11: PUBLICATION FIGURES")
    print("="*80)

    if config.get('visualizations', {}).get('enabled', True):
        etsa.create_publication_figures(metrics_df)

    # ========================================================================
    # COMPLETE
    # ========================================================================
    print("\n" + "="*80)
    print("EXPANDED COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nKey directories:")
    print(f"  - {output_dir}/data/                       : All CSV data files")
    print(f"  - {output_dir}/data/distances/             : T cell-tumor distance data")
    print(f"  - {output_dir}/data/heterogeneity/         : Tumor heterogeneity data")
    print(f"  - {output_dir}/statistics/                 : Statistical test results")
    print(f"  - {output_dir}/statistics/distances/       : Distance statistics")
    print(f"  - {output_dir}/figures/spatial_maps/       : Spatial visualizations")
    print(f"  - {output_dir}/figures/distances/          : Distance plots")
    print(f"  - {output_dir}/figures/heterogeneity/      : Heterogeneity plots")
    print(f"  - {output_dir}/figures/neighborhoods/      : Neighborhood analyses")
    print(f"  - {output_dir}/figures/temporal/           : Time-series plots")
    print("\n" + "="*80 + "\n")

    # Generate summary report
    _generate_summary_report(output_dir)


def _compute_heterogeneity_statistics(heterogeneity_df: pd.DataFrame, output_dir: str):
    """Compute statistics for tumor heterogeneity."""

    from scipy.stats import mannwhitneyu, spearmanr
    from statsmodels.stats.multitest import multipletests

    results = []

    # NOTE: Use actual marker names from data (AGFP, PERK - uppercase)
    for marker in ['AGFP', 'PERK']:
        het_col = f'{marker}_heterogeneity'
        clust_col = f'{marker}_clustering_index'

        if het_col not in heterogeneity_df.columns:
            continue

        # Main group comparison
        main_groups = heterogeneity_df['main_group'].dropna().unique()
        if len(main_groups) == 2:
            g1, g2 = main_groups[0], main_groups[1]

            # Heterogeneity
            data1_het = heterogeneity_df[heterogeneity_df['main_group'] == g1][het_col].dropna().values
            data2_het = heterogeneity_df[heterogeneity_df['main_group'] == g2][het_col].dropna().values

            if len(data1_het) >= 2 and len(data2_het) >= 2:
                stat_het, p_het = mannwhitneyu(data1_het, data2_het)

                results.append({
                    'marker': marker,
                    'metric': 'heterogeneity',
                    'test': 'main_group_comparison',
                    'group_1': g1,
                    'group_2': g2,
                    'n_1': len(data1_het),
                    'n_2': len(data2_het),
                    'mean_1': data1_het.mean(),
                    'mean_2': data2_het.mean(),
                    'statistic': stat_het,
                    'p_value': p_het
                })

            # Clustering
            data1_clust = heterogeneity_df[heterogeneity_df['main_group'] == g1][clust_col].dropna().values
            data2_clust = heterogeneity_df[heterogeneity_df['main_group'] == g2][clust_col].dropna().values

            if len(data1_clust) >= 2 and len(data2_clust) >= 2:
                stat_clust, p_clust = mannwhitneyu(data1_clust, data2_clust)

                results.append({
                    'marker': marker,
                    'metric': 'clustering_index',
                    'test': 'main_group_comparison',
                    'group_1': g1,
                    'group_2': g2,
                    'n_1': len(data1_clust),
                    'n_2': len(data2_clust),
                    'mean_1': data1_clust.mean(),
                    'mean_2': data2_clust.mean(),
                    'statistic': stat_clust,
                    'p_value': p_clust
                })

        # Temporal trend
        if 'timepoint' in heterogeneity_df.columns:
            for group in heterogeneity_df['main_group'].dropna().unique():
                group_data = heterogeneity_df[heterogeneity_df['main_group'] == group]

                if len(group_data) >= 3:
                    # Heterogeneity trend
                    rho_het, p_het = spearmanr(
                        group_data['timepoint'], group_data[het_col].fillna(0)
                    )

                    results.append({
                        'marker': marker,
                        'metric': 'heterogeneity',
                        'test': 'temporal_trend',
                        'group_1': group,
                        'group_2': None,
                        'n_1': len(group_data),
                        'n_2': None,
                        'mean_1': None,
                        'mean_2': None,
                        'statistic': rho_het,
                        'p_value': p_het
                    })

    if len(results) > 0:
        results_df = pd.DataFrame(results)

        # FDR correction
        _, p_adj, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_adjusted'] = p_adj
        results_df['significant'] = p_adj < 0.05

        results_df.to_csv(
            f"{output_dir}/statistics/heterogeneity/heterogeneity_statistics.csv",
            index=False
        )

        print(f"  Saved {len(results_df)} heterogeneity statistical tests")


def _compute_ninja_statistics(ninja_df: pd.DataFrame, output_dir: str):
    """Compute statistics for NINJA+/- tumor comparison."""

    from scipy.stats import chi2_contingency, mannwhitneyu

    results = []

    # 1. Frequency of NINJA+ vs NINJA- tumors per group
    contingency = pd.crosstab(ninja_df['main_group'], ninja_df['ninja_status'])
    chi2, p, dof, expected = chi2_contingency(contingency)

    results.append({
        'test': 'NINJA_status_frequency',
        'comparison': 'KPT_vs_KPNT',
        'statistic': chi2,
        'p_value': p,
        'df': dof,
        'note': 'Chi-square test of NINJA+/- frequency'
    })

    # 2. Tumor size comparison within KPNT: NINJA+ vs NINJA-
    kpnt_data = ninja_df[ninja_df['main_group'] == 'KPNT']

    if len(kpnt_data) > 0:
        ninja_pos = kpnt_data[kpnt_data['ninja_status'] == 'NINJA_positive']['tumor_size'].values
        ninja_neg = kpnt_data[kpnt_data['ninja_status'] == 'NINJA_negative']['tumor_size'].values

        if len(ninja_pos) >= 2 and len(ninja_neg) >= 2:
            stat, p = mannwhitneyu(ninja_pos, ninja_neg)

            results.append({
                'test': 'NINJA_pos_vs_neg_tumor_size',
                'comparison': 'KPNT_only',
                'statistic': stat,
                'p_value': p,
                'df': None,
                'note': f'NINJA+ (n={len(ninja_pos)}) vs NINJA- (n={len(ninja_neg)})'
            })

    # 3. KPT vs KPNT NINJA-negative tumor comparison
    kpt_ninja_neg = ninja_df[
        (ninja_df['main_group'] == 'KPT') &
        (ninja_df['ninja_status'] == 'NINJA_negative')
    ]['tumor_size'].values

    kpnt_ninja_neg = ninja_df[
        (ninja_df['main_group'] == 'KPNT') &
        (ninja_df['ninja_status'] == 'NINJA_negative')
    ]['tumor_size'].values

    if len(kpt_ninja_neg) >= 2 and len(kpnt_ninja_neg) >= 2:
        stat, p = mannwhitneyu(kpt_ninja_neg, kpnt_ninja_neg)

        results.append({
            'test': 'KPT_vs_KPNT_NINJA_negative_only',
            'comparison': 'NINJA_negative_tumors',
            'statistic': stat,
            'p_value': p,
            'df': None,
            'note': f'KPT (n={len(kpt_ninja_neg)}) vs KPNT (n={len(kpnt_ninja_neg)})'
        })

    if len(results) > 0:
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            f"{output_dir}/statistics/ninja_comparison_statistics.csv",
            index=False
        )

        print(f"  Saved {len(results_df)} NINJA comparison statistical tests")


def _generate_summary_report(output_dir: str):
    """Generate a summary report of all analyses."""

    report_path = f"{output_dir}/ANALYSIS_SUMMARY.txt"

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPANDED COMPREHENSIVE SPATIAL ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write("This analysis includes:\n\n")

        f.write("1. TUMOR STRUCTURE DETECTION\n")
        f.write("   - Automated tumor clustering\n")
        f.write("   - Size and morphology metrics\n\n")

        f.write("2. STANDARD INFILTRATION ANALYSIS\n")
        f.write("   - Immune infiltration at tumor margin, peri-tumor, and distal regions\n")
        f.write("   - Density and percentage calculations\n\n")

        f.write("3. T CELL-TUMOR DISTANCE ANALYSIS (DUAL-LEVEL)\n")
        f.write("   - Per-structure level (n = tumors)\n")
        f.write("   - Per-sample level (n = samples)\n")
        f.write("   - T cell distances to:\n")
        f.write("     * All tumor cells\n")
        f.write("     * pERK+ tumor cells\n")
        f.write("     * pERK- tumor cells\n")
        f.write("     * NINJA+ (aGFP+) tumor cells\n")
        f.write("     * NINJA- tumor cells\n")
        f.write("   - Boxplots, distributions, scatters with statistics\n\n")

        f.write("4. TUMOR HETEROGENEITY ANALYSIS\n")
        f.write("   - Spatial clustering of aGFP+ (NINJA) and pERK+ regions\n")
        f.write("   - Heterogeneity metrics\n")
        f.write("   - Clustering indices\n\n")

        f.write("5. NINJA+/- TUMOR COMPARISON\n")
        f.write("   - Classification of tumors as NINJA+ or NINJA-\n")
        f.write("   - Comparison within KPNT\n")
        f.write("   - Comparison of NINJA- tumors between KPT and KPNT\n\n")

        f.write("6. CELLULAR NEIGHBORHOODS TEMPORAL ANALYSIS\n")
        f.write("   - Neighborhood composition over time\n")
        f.write("   - KPT vs KPNT comparison\n")
        f.write("   - Temporal dynamics\n\n")

        f.write("7. MARKER EXPRESSION TEMPORAL TRACKING\n")
        f.write("   - All markers tracked over time\n")
        f.write("   - Group comparisons\n\n")

        f.write("8. TUMOR SIZE ANALYSIS\n")
        f.write("   - Growth curves\n")
        f.write("   - Group comparisons\n")
        f.write("   - Statistical tests\n\n")

        f.write("9. SPATIAL MAPS\n")
        f.write("   - Basic cell type distributions\n")
        f.write("   - Tumor definitions with analysis ranges\n\n")

        f.write("10. COMPREHENSIVE STATISTICS\n")
        f.write("    - Dual-level statistics (by sample and by tumor)\n")
        f.write("    - Temporal trends (Spearman correlation, linear regression)\n")
        f.write("    - Group comparisons (Mann-Whitney, t-tests)\n")
        f.write("    - FDR correction for multiple testing\n")
        f.write("    - Effect sizes (Cohen's d)\n\n")

        f.write("="*80 + "\n")
        f.write("For questions, see individual CSV files and plots in subdirectories.\n")
        f.write("="*80 + "\n")

    print(f"\n  Generated summary report: {report_path}")


if __name__ == '__main__':
    main()
