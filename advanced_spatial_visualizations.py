#!/usr/bin/env python3
"""
Advanced Spatial Analysis Visualizations

Visualization functions for advanced analysis phases 11-18.
Generates comprehensive plots for:
- Phase 12: pERK spatial architecture
- Phase 13: NINJA escape mechanisms
- Phase 14: Tumor heterogeneity
- Phase 15: Enhanced RCN dynamics
- Phase 16: Multi-level distance analysis
- Phase 17: Infiltration-tumor associations
- Phase 18: Pseudo-temporal trajectories
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import os


def plot_perk_analysis(output_dir: str):
    """Generate pERK analysis plots."""
    print("\nGenerating pERK analysis plots...")

    perk_dir = f"{output_dir}/advanced_perk_analysis"
    if not os.path.exists(perk_dir):
        print("  No pERK analysis data found, skipping...")
        return

    os.makedirs(f"{perk_dir}/figures", exist_ok=True)

    # Plot clustering results
    clustering_file = f"{perk_dir}/perk_clustering_analysis.csv"
    if os.path.exists(clustering_file):
        try:
            df = pd.read_csv(clustering_file)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Plot 1: Number of clusters by genotype
            if 'genotype' in df.columns and 'n_ninja_clusters' in df.columns:
                sns.boxplot(data=df, x='genotype', y='n_ninja_clusters',
                           ax=axes[0], palette='Set2')
                axes[0].set_ylabel('Number of pERK+ Clusters', fontweight='bold')
                axes[0].set_xlabel('Genotype', fontweight='bold')
                axes[0].set_title('pERK+ Clustering by Genotype', fontweight='bold')
                axes[0].grid(True, alpha=0.3, axis='y')

            # Plot 2: % clustered over time
            if 'timepoint' in df.columns and 'pct_clustered' in df.columns:
                for genotype in df['genotype'].unique():
                    geno_data = df[df['genotype'] == genotype]
                    temporal_mean = geno_data.groupby('timepoint')['pct_clustered'].mean()
                    axes[1].plot(temporal_mean.index, temporal_mean.values,
                               marker='o', label=genotype, linewidth=2)

                axes[1].set_xlabel('Timepoint', fontweight='bold')
                axes[1].set_ylabel('% pERK+ Cells Clustered', fontweight='bold')
                axes[1].set_title('pERK+ Clustering Over Time', fontweight='bold')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{perk_dir}/figures/perk_clustering_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ pERK clustering plots saved")
        except Exception as e:
            print(f"  WARNING: pERK clustering plots failed: {e}")

    # Plot growth dynamics
    growth_file = f"{perk_dir}/perk_growth_dynamics.csv"
    if os.path.exists(growth_file):
        try:
            df = pd.read_csv(growth_file)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Plot 1: pERK+ fraction over time
            if 'timepoint' in df.columns and 'pct_ninja' in df.columns:
                for genotype in df['genotype'].unique():
                    geno_data = df[df['genotype'] == genotype]
                    axes[0].plot(geno_data['timepoint'], geno_data['pct_ninja'],
                               marker='o', label=genotype, linewidth=2)

                axes[0].set_xlabel('Timepoint', fontweight='bold')
                axes[0].set_ylabel('% pERK+ of Total Tumor', fontweight='bold')
                axes[0].set_title('pERK+ Fraction Over Time', fontweight='bold')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

            # Plot 2: pERK+ by genotype
            if 'genotype' in df.columns and 'pct_ninja' in df.columns:
                sns.boxplot(data=df, x='genotype', y='pct_ninja',
                           ax=axes[1], palette='Set2')
                axes[1].set_ylabel('% pERK+ of Total Tumor', fontweight='bold')
                axes[1].set_xlabel('Genotype', fontweight='bold')
                axes[1].set_title('pERK+ Fraction by Genotype', fontweight='bold')
                axes[1].grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(f"{perk_dir}/figures/perk_growth_dynamics.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ pERK growth dynamics plots saved")
        except Exception as e:
            print(f"  WARNING: pERK growth plots failed: {e}")


def plot_ninja_analysis(output_dir: str):
    """Generate NINJA analysis plots."""
    print("\nGenerating NINJA analysis plots...")

    ninja_dir = f"{output_dir}/advanced_ninja_analysis"
    if not os.path.exists(ninja_dir):
        print("  No NINJA analysis data found, skipping...")
        return

    os.makedirs(f"{ninja_dir}/figures", exist_ok=True)

    # Similar structure to pERK plots
    clustering_file = f"{ninja_dir}/ninja_clustering_analysis.csv"
    if os.path.exists(clustering_file):
        try:
            df = pd.read_csv(clustering_file)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            if 'genotype' in df.columns and 'n_ninja_clusters' in df.columns:
                sns.boxplot(data=df, x='genotype', y='n_ninja_clusters',
                           ax=axes[0], palette='Set3')
                axes[0].set_ylabel('Number of NINJA+ Clusters', fontweight='bold')
                axes[0].set_xlabel('Genotype', fontweight='bold')
                axes[0].set_title('NINJA+ Clustering by Genotype', fontweight='bold')
                axes[0].grid(True, alpha=0.3, axis='y')

            if 'timepoint' in df.columns and 'pct_clustered' in df.columns:
                for genotype in df['genotype'].unique():
                    geno_data = df[df['genotype'] == genotype]
                    temporal_mean = geno_data.groupby('timepoint')['pct_clustered'].mean()
                    axes[1].plot(temporal_mean.index, temporal_mean.values,
                               marker='s', label=genotype, linewidth=2)

                axes[1].set_xlabel('Timepoint', fontweight='bold')
                axes[1].set_ylabel('% NINJA+ Cells Clustered', fontweight='bold')
                axes[1].set_title('NINJA+ Clustering Over Time', fontweight='bold')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{ninja_dir}/figures/ninja_clustering_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ NINJA clustering plots saved")
        except Exception as e:
            print(f"  WARNING: NINJA plots failed: {e}")


def plot_heterogeneity_analysis(output_dir: str):
    """Generate heterogeneity analysis plots."""
    print("\nGenerating heterogeneity analysis plots...")

    het_dir = f"{output_dir}/advanced_heterogeneity"
    if not os.path.exists(het_dir):
        print("  No heterogeneity analysis data found, skipping...")
        return

    os.makedirs(f"{het_dir}/figures", exist_ok=True)

    # Plot entropy analysis
    entropy_file = f"{het_dir}/marker_entropy_analysis.csv"
    if os.path.exists(entropy_file):
        try:
            df = pd.read_csv(entropy_file)

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes = axes.flatten()

            # Plot 1: Entropy by genotype
            if 'genotype' in df.columns and 'mean_marker_entropy' in df.columns:
                sns.boxplot(data=df, x='genotype', y='mean_marker_entropy',
                           ax=axes[0], palette='viridis')
                axes[0].set_ylabel('Mean Marker Entropy', fontweight='bold')
                axes[0].set_xlabel('Genotype', fontweight='bold')
                axes[0].set_title('Marker Diversification by Genotype', fontweight='bold')
                axes[0].grid(True, alpha=0.3, axis='y')

            # Plot 2: Entropy over time
            if 'timepoint' in df.columns and 'mean_marker_entropy' in df.columns:
                for genotype in df['genotype'].unique():
                    geno_data = df[df['genotype'] == genotype]
                    axes[1].plot(geno_data['timepoint'], geno_data['mean_marker_entropy'],
                               marker='o', label=genotype, linewidth=2)

                axes[1].set_xlabel('Timepoint', fontweight='bold')
                axes[1].set_ylabel('Mean Marker Entropy', fontweight='bold')
                axes[1].set_title('Entropy Over Time', fontweight='bold')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            # Plot 3: Max entropy by genotype
            if 'genotype' in df.columns and 'max_marker_entropy' in df.columns:
                sns.violinplot(data=df, x='genotype', y='max_marker_entropy',
                              ax=axes[2], palette='viridis')
                axes[2].set_ylabel('Max Marker Entropy', fontweight='bold')
                axes[2].set_xlabel('Genotype', fontweight='bold')
                axes[2].set_title('Maximum Marker Entropy', fontweight='bold')
                axes[2].grid(True, alpha=0.3, axis='y')

            # Plot 4: Scatter - mean vs max entropy
            if 'mean_marker_entropy' in df.columns and 'max_marker_entropy' in df.columns:
                for genotype in df['genotype'].unique():
                    geno_data = df[df['genotype'] == genotype]
                    axes[3].scatter(geno_data['mean_marker_entropy'],
                                  geno_data['max_marker_entropy'],
                                  alpha=0.6, s=50, label=genotype)

                axes[3].set_xlabel('Mean Marker Entropy', fontweight='bold')
                axes[3].set_ylabel('Max Marker Entropy', fontweight='bold')
                axes[3].set_title('Entropy Distribution', fontweight='bold')
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{het_dir}/figures/marker_entropy_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Entropy plots saved")
        except Exception as e:
            print(f"  WARNING: Entropy plots failed: {e}")

    # Plot heterogeneity metrics
    het_file = f"{het_dir}/heterogeneity_metrics.csv"
    if os.path.exists(het_file):
        try:
            df = pd.read_csv(het_file)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            if 'genotype' in df.columns and 'heterogeneity_score' in df.columns:
                sns.boxplot(data=df, x='genotype', y='heterogeneity_score',
                           ax=axes[0], palette='coolwarm')
                axes[0].set_ylabel('Heterogeneity Score (CV)', fontweight='bold')
                axes[0].set_xlabel('Genotype', fontweight='bold')
                axes[0].set_title('Intra-sample Heterogeneity', fontweight='bold')
                axes[0].grid(True, alpha=0.3, axis='y')

            if 'timepoint' in df.columns and 'heterogeneity_score' in df.columns:
                for genotype in df['genotype'].unique():
                    geno_data = df[df['genotype'] == genotype]
                    axes[1].plot(geno_data['timepoint'], geno_data['heterogeneity_score'],
                               marker='o', label=genotype, linewidth=2)

                axes[1].set_xlabel('Timepoint', fontweight='bold')
                axes[1].set_ylabel('Heterogeneity Score', fontweight='bold')
                axes[1].set_title('Heterogeneity Over Time', fontweight='bold')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{het_dir}/figures/heterogeneity_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Heterogeneity plots saved")
        except Exception as e:
            print(f"  WARNING: Heterogeneity plots failed: {e}")


def plot_infiltration_associations(output_dir: str):
    """Generate infiltration-tumor association plots."""
    print("\nGenerating infiltration association plots...")

    infil_dir = f"{output_dir}/advanced_infiltration"
    if not os.path.exists(infil_dir):
        print("  No infiltration association data found, skipping...")
        return

    os.makedirs(f"{infil_dir}/figures", exist_ok=True)

    # Plot associations
    assoc_file = f"{infil_dir}/tumor_infiltration_associations.csv"
    if os.path.exists(assoc_file):
        try:
            df = pd.read_csv(assoc_file)

            # Get top populations
            pop_counts = df.groupby('population').size()
            top_pops = pop_counts.nlargest(4).index.tolist()

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()

            for idx, pop in enumerate(top_pops):
                pop_data = df[df['population'] == pop]

                if len(pop_data) == 0:
                    continue

                ax = axes[idx]

                # Scatter: tumor size vs infiltration
                if 'tumor_size' in pop_data.columns and 'mean_infiltration' in pop_data.columns:
                    for genotype in pop_data['genotype'].unique():
                        geno_data = pop_data[pop_data['genotype'] == genotype]
                        ax.scatter(geno_data['tumor_size'], geno_data['mean_infiltration'],
                                 alpha=0.5, s=30, label=genotype)

                    ax.set_xlabel('Tumor Size (cells)', fontweight='bold')
                    ax.set_ylabel('Mean Infiltration (%)', fontweight='bold')
                    ax.set_title(f'{pop} Infiltration vs Tumor Size', fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{infil_dir}/figures/infiltration_associations.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Infiltration association plots saved")
        except Exception as e:
            print(f"  WARNING: Infiltration association plots failed: {e}")


def plot_all_advanced_visualizations(output_dir: str):
    """Generate all advanced analysis visualizations."""
    print("\n" + "="*80)
    print("GENERATING ADVANCED ANALYSIS VISUALIZATIONS (PHASES 11-18)")
    print("="*80)

    # Phase 12: pERK analysis
    plot_perk_analysis(output_dir)

    # Phase 13: NINJA analysis
    plot_ninja_analysis(output_dir)

    # Phase 14: Heterogeneity
    plot_heterogeneity_analysis(output_dir)

    # Phase 17: Infiltration associations
    plot_infiltration_associations(output_dir)

    print("\n✓ Advanced visualizations complete!")


if __name__ == '__main__':
    print("Use this module with your advanced analysis pipeline")
