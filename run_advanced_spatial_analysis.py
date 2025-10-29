#!/usr/bin/env python3
"""
Master Orchestration Script for Advanced Multi-Level Spatial Analysis

This script orchestrates the complete advanced spatial immunofluorescence analysis
pipeline for cyclic IF data from murine lung adenocarcinoma samples.

Usage:
    python run_advanced_spatial_analysis.py --input_dir results/ \
                                             --metadata sample_metadata.csv \
                                             --markers markers.csv \
                                             --output spatial_analysis_comprehensive

Phases:
    1. Enhanced Phenotyping & Tumor Detection
    2. Critical Research Questions (pERK, NINJA, Heterogeneity, RCN)
    3. Advanced Spatial Metrics
    4. Statistical Analysis & Reporting

Author: Advanced spatial analysis expansion
Date: 2025-10-29
"""

import argparse
import sys
from pathlib import Path
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Import the advanced spatial analysis framework
from advanced_spatial_analysis import AdvancedSpatialAnalysis, AnalysisConfig


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Advanced Multi-Level Spatial Immunofluorescence Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_advanced_spatial_analysis.py --input_dir results/ \\
                                           --metadata sample_metadata.csv \\
                                           --markers markers.csv

  # Run with custom configuration
  python run_advanced_spatial_analysis.py --input_dir results/ \\
                                           --metadata sample_metadata.csv \\
                                           --markers markers.csv \\
                                           --tumor_eps 60 \\
                                           --perk_eps 35
        """
    )

    # Required arguments
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing combined_quantification.csv files')
    parser.add_argument('--metadata', type=str, required=True,
                       help='Path to sample_metadata.csv')
    parser.add_argument('--markers', type=str, required=True,
                       help='Path to markers.csv')

    # Optional arguments
    parser.add_argument('--output', type=str, default='spatial_analysis_comprehensive',
                       help='Output directory (default: spatial_analysis_comprehensive)')

    # Configuration parameters
    parser.add_argument('--tumor_eps', type=float, default=50.0,
                       help='DBSCAN epsilon for tumor detection (μm, default: 50)')
    parser.add_argument('--tumor_min_samples', type=int, default=30,
                       help='DBSCAN min_samples for tumor detection (default: 30)')
    parser.add_argument('--perk_eps', type=float, default=30.0,
                       help='DBSCAN epsilon for pERK clustering (μm, default: 30)')
    parser.add_argument('--n_neighbors', type=int, default=10,
                       help='Number of neighbors for RCN analysis (default: 10)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for figure output (default: 300)')

    # Analysis control
    parser.add_argument('--skip_phase1', action='store_true',
                       help='Skip Phase 1 (phenotyping/tumor detection)')
    parser.add_argument('--skip_phase2', action='store_true',
                       help='Skip Phase 2 (research questions)')
    parser.add_argument('--skip_phase3', action='store_true',
                       help='Skip Phase 3 (advanced metrics)')
    parser.add_argument('--only_phase', type=str, choices=['1', '2', '3'],
                       help='Run only specified phase')

    return parser.parse_args()


def create_config(args):
    """Create analysis configuration from arguments."""
    config = AnalysisConfig(
        tumor_dbscan_eps=args.tumor_eps,
        tumor_dbscan_min_samples=args.tumor_min_samples,
        perk_cluster_eps=args.perk_eps,
        n_neighbors=args.n_neighbors,
        dpi=args.dpi,
    )
    return config


def run_phase1(analysis):
    """Run Phase 1: Enhanced Phenotyping & Tumor Detection."""
    print("\n" + "=" * 80)
    print("PHASE 1: ENHANCED PHENOTYPING & TUMOR DETECTION")
    print("=" * 80)

    # 1.1: Phenotyping
    print("\n### Phase 1.1: Cell Phenotyping ###")
    analysis.phase1_phenotype_cells()

    # 1.2: Tumor structure detection
    print("\n### Phase 1.2: Tumor Structure Detection ###")
    analysis.phase12_detect_tumor_structures()

    print("\n✓ Phase 1 complete")


def run_phase2(analysis):
    """Run Phase 2: Critical Research Questions."""
    print("\n" + "=" * 80)
    print("PHASE 2: CRITICAL RESEARCH QUESTIONS")
    print("=" * 80)

    # 2.1: pERK analysis
    print("\n### Phase 2.1: pERK Spatial Architecture ###")
    analysis.phase21_perk_spatial_architecture()

    # 2.2: NINJA analysis
    print("\n### Phase 2.2: NINJA Escape Mechanism ###")
    analysis.phase22_ninja_escape_analysis()

    # 2.3: Heterogeneity
    print("\n### Phase 2.3: Heterogeneity Emergence & Evolution ###")
    analysis.phase23_heterogeneity_analysis()

    # 2.4: RCN dynamics
    print("\n### Phase 2.4: Cellular Neighborhood Dynamics ###")
    analysis.phase24_rcn_temporal_dynamics()

    print("\n✓ Phase 2 complete")


def run_phase3(analysis):
    """Run Phase 3: Advanced Spatial Metrics."""
    print("\n" + "=" * 80)
    print("PHASE 3: ADVANCED SPATIAL METRICS")
    print("=" * 80)

    # 3.1: Distance analysis
    print("\n### Phase 3.1: Multi-Level Distance Analysis ###")
    analysis.phase31_multilevel_distance_analysis()

    # 3.2: Infiltration associations
    print("\n### Phase 3.2: Infiltration-Tumor Associations ###")
    analysis.phase32_infiltration_tumor_associations()

    # 3.3: Pseudo-temporal trajectories
    print("\n### Phase 3.3: Pseudo-Temporal Trajectory Analysis ###")
    analysis.phase33_pseudotemporal_analysis()

    print("\n✓ Phase 3 complete")


def generate_report(analysis, output_dir):
    """Generate comprehensive HTML report."""
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("=" * 80)

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Advanced Spatial Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .summary-box {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #bdc3c7;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .figure {{
            margin: 20px 0;
            text-align: center;
        }}
        .figure img {{
            max-width: 100%;
            border: 1px solid #ddd;
            padding: 5px;
        }}
        .figure-caption {{
            font-style: italic;
            color: #7f8c8d;
            margin-top: 10px;
        }}
        .timestamp {{
            color: #95a5a6;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>Advanced Multi-Level Spatial Immunofluorescence Analysis Report</h1>

    <div class="summary-box">
        <h3>Analysis Summary</h3>
        <p><strong>Date:</strong> <span class="timestamp">{time.strftime('%Y-%m-%d %H:%M:%S')}</span></p>
        <p><strong>Output Directory:</strong> {output_dir}</p>
        <p><strong>Total Cells Analyzed:</strong> {len(analysis.cells_df):,}</p>
        <p><strong>Number of Samples:</strong> {analysis.cells_df['sample_id'].nunique()}</p>
        <p><strong>Number of Tumors Detected:</strong> {len(analysis.tumors_df) if analysis.tumors_df is not None else 0}</p>
    </div>

    <div class="section">
        <h2>Phase 1: Enhanced Phenotyping & Tumor Detection</h2>

        <h3>1.1 Cell Phenotyping</h3>
        <p>Intensity-based gating was performed to identify cell phenotypes based on marker expression.</p>

        <div class="figure">
            <img src="01_phenotyping/aggregated_panels/phenotype_composition_overview.pdf" alt="Phenotype Composition">
            <div class="figure-caption">Figure 1.1: Overall phenotype composition across samples</div>
        </div>

        <h3>1.2 Tumor Structure Detection</h3>
        <p>Tumor structures were detected using DBSCAN clustering on TOM+ cells with morphological validation.</p>

        <div class="figure">
            <img src="01_phenotyping/aggregated_panels/tumor_size_distribution.pdf" alt="Tumor Size Distribution">
            <div class="figure-caption">Figure 1.2: Tumor size distribution across samples</div>
        </div>
    </div>

    <div class="section">
        <h2>Phase 2: Critical Research Questions</h2>

        <h3>2.1 pERK Spatial Architecture</h3>
        <p><strong>Q1:</strong> Are pERK+ regions spatially clustered or stochastic?</p>
        <p><strong>Q2:</strong> pERK+ region growth dynamics over time</p>
        <p><strong>Q3:</strong> Differential T cell infiltration in pERK+ vs pERK- regions</p>

        <div class="figure">
            <img src="02_perk_analysis/aggregated_panels/perk_clustering_statistics.pdf" alt="pERK Clustering">
            <div class="figure-caption">Figure 2.1: pERK+ spatial clustering analysis</div>
        </div>

        <h3>2.2 NINJA Escape Mechanism</h3>
        <p><strong>Q1:</strong> NINJA+ spatial clustering patterns</p>
        <p><strong>Q2:</strong> NINJA+ region growth independent of total tumor growth</p>
        <p><strong>Q3:</strong> Cell type enrichment near NINJA+ regions</p>

        <h3>2.3 Heterogeneity Emergence & Evolution</h3>
        <p><strong>Q1:</strong> Marker diversification: random vs regional</p>
        <p><strong>Q2:</strong> Tumor diversity per sample (intra-sample heterogeneity)</p>

        <h3>2.4 Cellular Neighborhood (RCN) Dynamics</h3>
        <p>Recurrent cellular neighborhood analysis showing temporal evolution of tumor-immune microenvironments.</p>
    </div>

    <div class="section">
        <h2>Phase 3: Advanced Spatial Metrics</h2>

        <h3>3.1 Multi-Level Distance Analysis</h3>
        <p>Quantification of spatial distances between immune cells and different tumor subtypes.</p>

        <h3>3.2 Infiltration-Tumor Association Analysis</h3>
        <p>Analysis of relationships between tumor characteristics and immune infiltration patterns.</p>

        <h3>3.3 Pseudo-Temporal Trajectory Analysis</h3>
        <p>Inference of tumor evolution trajectories based on spatial organization and phenotype composition.</p>
    </div>

    <div class="section">
        <h2>Statistical Summary</h2>
        <p>All statistical tests were performed with multiple comparison correction (Benjamini-Hochberg FDR).</p>
        <p>Effect sizes (Cohen's d) are reported alongside p-values for all comparisons.</p>
    </div>

    <div class="section">
        <h2>References</h2>
        <ul>
            <li>Nirmal et al. 2021. The spatial landscape of progression and immunoediting in primary melanoma at single-cell resolution. <i>Cancer Discovery</i>.</li>
            <li>Schapiro et al. 2017. histoCAT: analysis of cell phenotypes and interactions in multiplex image cytometry data. <i>Nature Methods</i>.</li>
            <li>Jackson et al. 2020. The single-cell pathology landscape of breast cancer. <i>Nature</i>.</li>
        </ul>
    </div>

    <footer>
        <hr>
        <p class="timestamp">Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Advanced Spatial Analysis Pipeline v1.0</p>
    </footer>
</body>
</html>
"""

    # Save HTML report
    report_file = Path(output_dir) / 'master_summary_report.html'
    with open(report_file, 'w') as f:
        f.write(html_content)

    print(f"\n✓ Report saved: {report_file}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("ADVANCED MULTI-LEVEL SPATIAL IMMUNOFLUORESCENCE ANALYSIS")
    print("=" * 80)
    print()

    # Parse arguments
    args = parse_arguments()

    # Validate inputs
    if not Path(args.input_dir).exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    if not Path(args.metadata).exists():
        print(f"Error: Metadata file not found: {args.metadata}")
        sys.exit(1)

    if not Path(args.markers).exists():
        print(f"Error: Markers file not found: {args.markers}")
        sys.exit(1)

    # Create configuration
    config = create_config(args)

    # Initialize analysis
    print("Initializing advanced spatial analysis...")
    analysis = AdvancedSpatialAnalysis(
        output_dir=args.output,
        config=config
    )

    # Load data
    print("\nLoading data...")
    analysis.load_data(
        quantification_files=args.input_dir,
        sample_metadata_file=args.metadata,
        markers_file=args.markers
    )

    # Determine which phases to run
    run_phases = []
    if args.only_phase:
        run_phases = [args.only_phase]
    else:
        if not args.skip_phase1:
            run_phases.append('1')
        if not args.skip_phase2:
            run_phases.append('2')
        if not args.skip_phase3:
            run_phases.append('3')

    # Execute analysis phases
    start_time = time.time()

    if '1' in run_phases:
        run_phase1(analysis)

    if '2' in run_phases:
        run_phase2(analysis)

    if '3' in run_phases:
        run_phase3(analysis)

    # Generate comprehensive report
    generate_report(analysis, args.output)

    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total execution time: {elapsed_time/60:.2f} minutes")
    print(f"Output directory: {args.output}")
    print(f"HTML report: {args.output}/master_summary_report.html")
    print("\nDirectory structure:")
    print(f"  {args.output}/")
    print(f"    ├── data/                    # CSV files with results")
    print(f"    ├── statistics/              # Statistical test results")
    print(f"    ├── 01_phenotyping/          # Phase 1 outputs")
    print(f"    ├── 02_perk_analysis/        # Phase 2.1 outputs")
    print(f"    ├── 03_ninja_analysis/       # Phase 2.2 outputs")
    print(f"    ├── 04_heterogeneity/        # Phase 2.3 outputs")
    print(f"    ├── 05_rcn_dynamics/         # Phase 2.4 outputs")
    print(f"    ├── 06_distance_analysis/    # Phase 3.1 outputs")
    print(f"    ├── 07_infiltration_associations/  # Phase 3.2 outputs")
    print(f"    ├── 08_pseudotime/           # Phase 3.3 outputs")
    print(f"    └── master_summary_report.html")
    print("\n✓ All analyses completed successfully!")


if __name__ == '__main__':
    main()
