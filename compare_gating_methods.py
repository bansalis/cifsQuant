#!/usr/bin/env python3
"""
Comparison Script: Original vs UniFORM-based Gating
====================================================

Compares results from two gating methods:
1. Original gating (manual_gating.py)
2. UniFORM-based hierarchical gating (uniform_gating.py)

Creates publication-quality figures showing:
- Gate differences
- Population percentage differences
- Agreement metrics
- Method concordance plots

Usage:
    python compare_gating_methods.py \
        --original manual_gating_output/gated_data.h5ad \
        --uniform uniform_gating_output/gated_data.h5ad \
        --output comparison_output

Author: Automated generation
Date: 2025-10-22
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from pathlib import Path
import argparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'


class GatingComparison:
    """Compare two gating methods."""

    def __init__(self, adata_original, adata_uniform, output_dir):
        """
        Initialize comparison.

        Parameters
        ----------
        adata_original : AnnData
            Results from original gating method
        adata_uniform : AnnData
            Results from UniFORM gating method
        output_dir : str
            Output directory for figures and reports
        """
        self.adata_orig = adata_original
        self.adata_unif = adata_uniform
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Verify same cells
        if len(adata_original) != len(adata_uniform):
            raise ValueError(f"Cell counts don't match: {len(adata_original)} vs {len(adata_uniform)}")

        # Get common markers
        self.markers = [m for m in adata_original.var_names if m in adata_uniform.var_names]

        print(f"Comparing {len(self.markers)} markers across {len(adata_original):,} cells")


    def compare_gates(self) -> pd.DataFrame:
        """
        Compare gate values between methods.

        Returns
        -------
        pd.DataFrame
            Gate comparison table
        """
        print("\nComparing gate values...")

        # Load gates if available
        orig_gates = {}
        unif_gates = {}

        # Try to load from files
        orig_gates_file = Path('manual_gating_output/gates.csv')
        unif_gates_file = Path('uniform_gating_output/gates.csv')

        if orig_gates_file.exists():
            df = pd.read_csv(orig_gates_file)
            orig_gates = dict(zip(df['marker'], df['gate']))

        if unif_gates_file.exists():
            df = pd.read_csv(unif_gates_file)
            unif_gates = dict(zip(df['marker'], df['gate']))

        # Build comparison table
        results = []
        for marker in self.markers:
            orig_gate = orig_gates.get(marker, np.nan)
            unif_gate = unif_gates.get(marker, np.nan)

            # Calculate difference
            if not np.isnan(orig_gate) and not np.isnan(unif_gate):
                abs_diff = unif_gate - orig_gate
                rel_diff = 100 * abs_diff / orig_gate if orig_gate > 0 else np.nan
            else:
                abs_diff = np.nan
                rel_diff = np.nan

            results.append({
                'marker': marker,
                'original_gate': orig_gate,
                'uniform_gate': unif_gate,
                'absolute_difference': abs_diff,
                'relative_difference_pct': rel_diff
            })

        comparison_df = pd.DataFrame(results)
        comparison_df.to_csv(self.output_dir / 'gate_comparison.csv', index=False)
        print(f"  Saved: {self.output_dir / 'gate_comparison.csv'}")

        return comparison_df


    def compare_populations(self) -> pd.DataFrame:
        """
        Compare positive percentages between methods.

        Returns
        -------
        pd.DataFrame
            Population comparison table
        """
        print("\nComparing population percentages...")

        results = []

        for marker in self.markers:
            marker_idx = self.adata_orig.var_names.get_loc(marker)

            # Get gated calls
            orig_gated = self.adata_orig.layers['gated'][:, marker_idx]
            unif_gated = self.adata_unif.layers['gated'][:, marker_idx]

            # Calculate percentages
            orig_pct = (orig_gated > 0).mean() * 100
            unif_pct = (unif_gated > 0).mean() * 100

            # Agreement metrics
            agreement = (orig_gated == unif_gated).mean() * 100

            # Cohen's kappa
            kappa = cohen_kappa_score(orig_gated, unif_gated)

            # Correlation (on normalized values)
            if 'uniform_normalized' in self.adata_unif.layers:
                unif_values = self.adata_unif.layers['uniform_normalized'][:, marker_idx]
            else:
                unif_values = self.adata_unif.X[:, marker_idx]

            if 'aligned' in self.adata_orig.layers:
                orig_values = self.adata_orig.layers['aligned'][:, marker_idx]
            else:
                orig_values = self.adata_orig.X[:, marker_idx]

            # Remove zeros for correlation
            mask = (orig_values > 0) & (unif_values > 0)
            if mask.sum() > 100:
                corr, _ = pearsonr(orig_values[mask], unif_values[mask])
            else:
                corr = np.nan

            results.append({
                'marker': marker,
                'original_pct_positive': orig_pct,
                'uniform_pct_positive': unif_pct,
                'difference_pct': unif_pct - orig_pct,
                'agreement_pct': agreement,
                'cohens_kappa': kappa,
                'intensity_correlation': corr
            })

        comparison_df = pd.DataFrame(results)
        comparison_df.to_csv(self.output_dir / 'population_comparison.csv', index=False)
        print(f"  Saved: {self.output_dir / 'population_comparison.csv'}")

        return comparison_df


    def plot_gate_comparison(self, gate_comparison_df):
        """Bar plot comparing gates."""
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(self.markers))
        width = 0.35

        # Get gates
        orig_gates = [gate_comparison_df[gate_comparison_df['marker'] == m]['original_gate'].values[0]
                     for m in self.markers]
        unif_gates = [gate_comparison_df[gate_comparison_df['marker'] == m]['uniform_gate'].values[0]
                     for m in self.markers]

        # Plot bars
        ax.bar(x - width/2, orig_gates, width, label='Original', alpha=0.8, color='steelblue')
        ax.bar(x + width/2, unif_gates, width, label='UniFORM', alpha=0.8, color='coral')

        ax.set_xlabel('Marker', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gate Threshold', fontsize=12, fontweight='bold')
        ax.set_title('Gate Comparison: Original vs UniFORM', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.markers, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'gate_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {self.output_dir / 'gate_comparison.png'}")


    def plot_population_comparison(self, pop_comparison_df):
        """Scatter plot comparing positive percentages."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Scatter plot
        ax = axes[0]

        orig_pcts = pop_comparison_df['original_pct_positive'].values
        unif_pcts = pop_comparison_df['uniform_pct_positive'].values

        ax.scatter(orig_pcts, unif_pcts, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

        # Add marker labels
        for i, marker in enumerate(pop_comparison_df['marker']):
            ax.annotate(marker, (orig_pcts[i], unif_pcts[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)

        # Identity line
        max_val = max(orig_pcts.max(), unif_pcts.max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.5, label='Identity')

        ax.set_xlabel('Original Method (% Positive)', fontsize=12, fontweight='bold')
        ax.set_ylabel('UniFORM Method (% Positive)', fontsize=12, fontweight='bold')
        ax.set_title('Population Percentage Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Panel 2: Difference plot
        ax = axes[1]

        differences = pop_comparison_df['difference_pct'].values
        colors = ['green' if d < 0 else 'red' for d in differences]

        ax.barh(self.markers, differences, color=colors, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='black', linestyle='-', linewidth=1.5)
        ax.set_xlabel('Difference (UniFORM - Original) %', fontsize=12, fontweight='bold')
        ax.set_ylabel('Marker', fontsize=12, fontweight='bold')
        ax.set_title('Population Differences', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add annotation
        ax.text(0.02, 0.98, 'Green: UniFORM lower\nRed: UniFORM higher',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'population_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {self.output_dir / 'population_comparison.png'}")


    def plot_agreement_heatmap(self, pop_comparison_df):
        """Heatmap showing agreement metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Agreement percentage
        ax = axes[0]

        agreement_data = pop_comparison_df[['marker', 'agreement_pct']].set_index('marker')
        sns.heatmap(agreement_data, annot=True, fmt='.1f', cmap='RdYlGn',
                   vmin=80, vmax=100, cbar_kws={'label': '% Agreement'},
                   ax=ax, linewidths=0.5)
        ax.set_title('Cell-Level Agreement', fontsize=13, fontweight='bold')
        ax.set_ylabel('')

        # Panel 2: Cohen's Kappa
        ax = axes[1]

        kappa_data = pop_comparison_df[['marker', 'cohens_kappa']].set_index('marker')
        sns.heatmap(kappa_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0, vmax=1, cbar_kws={'label': "Cohen's κ"},
                   ax=ax, linewidths=0.5)
        ax.set_title("Cohen's Kappa (Agreement)", fontsize=13, fontweight='bold')
        ax.set_ylabel('')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'agreement_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {self.output_dir / 'agreement_heatmap.png'}")


    def plot_confusion_matrices(self):
        """Confusion matrices for each marker."""
        n_markers = len(self.markers)
        ncols = 3
        nrows = (n_markers + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        axes = axes.flatten() if n_markers > 1 else [axes]

        fig.suptitle('Confusion Matrices: Original vs UniFORM', fontsize=16, fontweight='bold')

        for i, marker in enumerate(self.markers):
            marker_idx_orig = self.adata_orig.var_names.get_loc(marker)
            marker_idx_unif = self.adata_unif.var_names.get_loc(marker)

            orig_gated = self.adata_orig.layers['gated'][:, marker_idx_orig]
            unif_gated = self.adata_unif.layers['gated'][:, marker_idx_unif]

            # Confusion matrix
            cm = confusion_matrix(orig_gated, unif_gated)

            # Normalize to percentages
            cm_pct = 100 * cm / cm.sum()

            ax = axes[i]
            sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                       xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
                       cbar=False, ax=ax, linewidths=1)

            ax.set_title(marker, fontsize=11, fontweight='bold')
            ax.set_xlabel('UniFORM', fontsize=10)
            ax.set_ylabel('Original', fontsize=10)

        # Hide unused axes
        for i in range(n_markers, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {self.output_dir / 'confusion_matrices.png'}")


    def plot_intensity_distributions(self, marker: str):
        """
        Compare intensity distributions for a specific marker.

        Shows raw, normalized, and gated populations.
        """
        marker_idx_orig = self.adata_orig.var_names.get_loc(marker)
        marker_idx_unif = self.adata_unif.var_names.get_loc(marker)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Intensity Distribution Comparison: {marker}', fontsize=14, fontweight='bold')

        # Get data
        orig_raw = self.adata_orig.layers['raw'][:, marker_idx_orig]
        unif_raw = self.adata_unif.layers['raw'][:, marker_idx_unif]

        if 'aligned' in self.adata_orig.layers:
            orig_norm = self.adata_orig.layers['aligned'][:, marker_idx_orig]
        else:
            orig_norm = self.adata_orig.X[:, marker_idx_orig]

        if 'uniform_normalized' in self.adata_unif.layers:
            unif_norm = self.adata_unif.layers['uniform_normalized'][:, marker_idx_unif]
        else:
            unif_norm = self.adata_unif.X[:, marker_idx_unif]

        orig_gated = self.adata_orig.layers['gated'][:, marker_idx_orig]
        unif_gated = self.adata_unif.layers['gated'][:, marker_idx_unif]

        # Panel 1: Raw distributions
        ax = axes[0, 0]
        ax.hist(orig_raw[orig_raw > 0], bins=100, alpha=0.5, label='Original', color='blue')
        ax.hist(unif_raw[unif_raw > 0], bins=100, alpha=0.5, label='UniFORM', color='red')
        ax.set_xlabel('Raw Intensity', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Raw Data', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Panel 2: Normalized distributions
        ax = axes[0, 1]
        ax.hist(orig_norm[orig_norm > 0], bins=100, alpha=0.5, label='Original', color='blue')
        ax.hist(unif_norm[unif_norm > 0], bins=100, alpha=0.5, label='UniFORM', color='red')
        ax.set_xlabel('Normalized Intensity', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('After Normalization', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Panel 3: Original method gated
        ax = axes[1, 0]
        orig_neg = orig_norm[(orig_norm > 0) & (orig_gated == 0)]
        orig_pos = orig_norm[orig_gated > 0]
        ax.hist(orig_neg, bins=50, alpha=0.6, label='Negative', color='blue')
        ax.hist(orig_pos, bins=50, alpha=0.6, label='Positive', color='red')
        ax.set_xlabel('Intensity', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Original Method Gating', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Panel 4: UniFORM method gated
        ax = axes[1, 1]
        unif_neg = unif_norm[(unif_norm > 0) & (unif_gated == 0)]
        unif_pos = unif_norm[unif_gated > 0]
        ax.hist(unif_neg, bins=50, alpha=0.6, label='Negative', color='blue')
        ax.hist(unif_pos, bins=50, alpha=0.6, label='Positive', color='red')
        ax.set_xlabel('Intensity', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('UniFORM Method Gating', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'intensity_comparison_{marker}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {self.output_dir / f'intensity_comparison_{marker}.png'}")


    def generate_summary_report(self, gate_comp_df, pop_comp_df):
        """Generate comprehensive text report."""
        report_lines = []

        report_lines.append("="*70)
        report_lines.append("GATING METHOD COMPARISON REPORT")
        report_lines.append("="*70)
        report_lines.append(f"\nGenerated: {pd.Timestamp.now()}")
        report_lines.append(f"\nDatasets:")
        report_lines.append(f"  - Original: {len(self.adata_orig):,} cells")
        report_lines.append(f"  - UniFORM: {len(self.adata_unif):,} cells")
        report_lines.append(f"  - Markers: {len(self.markers)}")

        report_lines.append("\n\n" + "="*70)
        report_lines.append("GATE VALUE COMPARISON")
        report_lines.append("="*70)

        for _, row in gate_comp_df.iterrows():
            report_lines.append(f"\n{row['marker']}:")
            report_lines.append(f"  Original gate:  {row['original_gate']:.2f}")
            report_lines.append(f"  UniFORM gate:   {row['uniform_gate']:.2f}")
            report_lines.append(f"  Difference:     {row['absolute_difference']:.2f} ({row['relative_difference_pct']:.1f}%)")

        report_lines.append("\n\n" + "="*70)
        report_lines.append("POPULATION COMPARISON")
        report_lines.append("="*70)

        for _, row in pop_comp_df.iterrows():
            report_lines.append(f"\n{row['marker']}:")
            report_lines.append(f"  Original % positive:  {row['original_pct_positive']:.2f}%")
            report_lines.append(f"  UniFORM % positive:   {row['uniform_pct_positive']:.2f}%")
            report_lines.append(f"  Difference:           {row['difference_pct']:.2f}%")
            report_lines.append(f"  Agreement:            {row['agreement_pct']:.2f}%")
            report_lines.append(f"  Cohen's κ:            {row['cohens_kappa']:.3f}")
            report_lines.append(f"  Intensity correlation: {row['intensity_correlation']:.3f}")

        report_lines.append("\n\n" + "="*70)
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("="*70)

        report_lines.append(f"\nMean agreement: {pop_comp_df['agreement_pct'].mean():.2f}%")
        report_lines.append(f"Mean Cohen's κ: {pop_comp_df['cohens_kappa'].mean():.3f}")
        report_lines.append(f"Mean intensity correlation: {pop_comp_df['intensity_correlation'].mean():.3f}")

        report_lines.append(f"\nLargest population difference:")
        max_diff_idx = pop_comp_df['difference_pct'].abs().idxmax()
        max_diff_row = pop_comp_df.loc[max_diff_idx]
        report_lines.append(f"  {max_diff_row['marker']}: {max_diff_row['difference_pct']:.2f}%")

        report_lines.append(f"\nLowest agreement:")
        min_agree_idx = pop_comp_df['agreement_pct'].idxmin()
        min_agree_row = pop_comp_df.loc[min_agree_idx]
        report_lines.append(f"  {min_agree_row['marker']}: {min_agree_row['agreement_pct']:.2f}%")

        report_lines.append("\n\n" + "="*70)
        report_lines.append("INTERPRETATION")
        report_lines.append("="*70)

        mean_kappa = pop_comp_df['cohens_kappa'].mean()
        if mean_kappa > 0.8:
            interp = "Excellent agreement"
        elif mean_kappa > 0.6:
            interp = "Good agreement"
        elif mean_kappa > 0.4:
            interp = "Moderate agreement"
        else:
            interp = "Poor agreement"

        report_lines.append(f"\nOverall method agreement: {interp} (κ = {mean_kappa:.3f})")

        report_lines.append("\n\nMarkers with substantial differences (>10% population change):")
        large_diff = pop_comp_df[pop_comp_df['difference_pct'].abs() > 10]
        if len(large_diff) > 0:
            for _, row in large_diff.iterrows():
                report_lines.append(f"  - {row['marker']}: {row['difference_pct']:.1f}% difference")
        else:
            report_lines.append("  None")

        report_lines.append("\n\n" + "="*70)
        report_lines.append("END OF REPORT")
        report_lines.append("="*70)

        # Save report
        report_path = self.output_dir / 'comparison_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"\n  Saved: {report_path}")

        return '\n'.join(report_lines)


    def run_complete_comparison(self):
        """Run complete comparison pipeline."""
        print("\n" + "="*70)
        print("GATING METHOD COMPARISON")
        print("="*70)

        # Compare gates
        gate_comp_df = self.compare_gates()

        # Compare populations
        pop_comp_df = self.compare_populations()

        # Generate plots
        print("\nGenerating comparison visualizations...")
        self.plot_gate_comparison(gate_comp_df)
        self.plot_population_comparison(pop_comp_df)
        self.plot_agreement_heatmap(pop_comp_df)
        self.plot_confusion_matrices()

        # Per-marker intensity comparisons
        print("\nGenerating per-marker intensity comparisons...")
        for marker in self.markers:
            self.plot_intensity_distributions(marker)

        # Generate report
        print("\nGenerating summary report...")
        self.generate_summary_report(gate_comp_df, pop_comp_df)

        print("\n" + "="*70)
        print("COMPARISON COMPLETE!")
        print("="*70)
        print(f"\nAll outputs saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Compare gating methods')
    parser.add_argument('--original', type=str, required=True,
                       help='Path to original gating output (.h5ad)')
    parser.add_argument('--uniform', type=str, required=True,
                       help='Path to UniFORM gating output (.h5ad)')
    parser.add_argument('--output', type=str, default='comparison_output',
                       help='Output directory')

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    adata_orig = sc.read_h5ad(args.original)
    adata_unif = sc.read_h5ad(args.uniform)

    print(f"  Original: {len(adata_orig):,} cells, {len(adata_orig.var_names)} markers")
    print(f"  UniFORM:  {len(adata_unif):,} cells, {len(adata_unif.var_names)} markers")

    # Run comparison
    comparer = GatingComparison(adata_orig, adata_unif, args.output)
    comparer.run_complete_comparison()

    print("\n✓ Comparison complete!")


if __name__ == '__main__':
    main()
