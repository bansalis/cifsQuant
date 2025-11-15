"""
Marker Region Visualization
Generate spatial plots showing marker-defined regions and immune enrichment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import warnings

try:
    import spatialcells as spc
    HAS_SPATIALCELLS = True
except ImportError:
    HAS_SPATIALCELLS = False


class MarkerRegionPlotter:
    """
    Visualize marker-defined spatial regions and immune enrichment.

    Creates comprehensive plots showing:
    - Marker+ and marker- region boundaries
    - Spatial overlays with immune cells
    - Immune enrichment comparisons
    - Regional composition
    """

    def __init__(self, output_dir: Path, config: Dict):
        """
        Initialize plotter.

        Parameters
        ----------
        output_dir : Path
            Output directory for plots
        config : dict
            Configuration dictionary
        """
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Set style
        sns.set_style('white')
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 11

    def generate_all_plots(self, results: Dict, marker_region_analysis):
        """
        Generate all marker region plots.

        Parameters
        ----------
        results : dict
            Results dictionary from MarkerRegionAnalysisSpatialCells
        marker_region_analysis : MarkerRegionAnalysisSpatialCells
            Analysis object with boundary data
        """
        print("  Generating marker region plots...")

        # Plot regional boundaries
        if 'detected_marker_regions' in results:
            self._plot_marker_boundaries(results, marker_region_analysis)

        # Plot immune enrichment
        if 'immune_enrichment' in results:
            self._plot_immune_enrichment(results['immune_enrichment'])

        # Plot regional comparison
        if 'marker_region_comparison' in results:
            self._plot_marker_comparison(results['marker_region_comparison'])

        # Plot composition
        if 'regional_composition' in results:
            self._plot_regional_composition(results['regional_composition'])

        print(f"    ✓ Saved plots to {self.plots_dir}/")

    def _plot_marker_boundaries(self, results: Dict, marker_region_analysis):
        """Plot spatial boundaries for each marker."""
        if not HAS_SPATIALCELLS:
            print("    ⚠ SpatialCells not available, skipping boundary plots")
            return

        detected_regions = results['detected_marker_regions']
        marker_boundaries = marker_region_analysis.get_marker_boundaries()

        # Plot each sample
        for sample in detected_regions['sample_id'].unique():
            sample_regions = detected_regions[detected_regions['sample_id'] == sample]

            if sample not in marker_boundaries:
                continue

            # Get markers for this sample
            markers = sample_regions['marker'].unique()

            for marker in markers:
                marker_regions_df = sample_regions[sample_regions['marker'] == marker]

                if marker_regions_df.empty:
                    continue

                # Create figure
                fig, axes = plt.subplots(1, 2, figsize=(20, 10))

                # Get all cells for this sample
                adata = marker_region_analysis.adata
                sample_mask = adata.obs['sample_id'] == sample
                sample_coords = adata.obsm['spatial'][sample_mask.values]

                # Plot all cells as background
                for ax in axes:
                    ax.scatter(sample_coords[:, 0], sample_coords[:, 1],
                             s=0.5, c='lightgray', alpha=0.2, rasterized=True)
                    ax.set_aspect('equal')
                    ax.invert_yaxis()

                # Left: Marker+ regions
                ax = axes[0]
                marker_key_pos = f'{marker}_positive'
                if marker_key_pos in marker_boundaries[sample]:
                    pos_boundaries = marker_boundaries[sample][marker_key_pos]

                    for region_id, boundary in pos_boundaries.items():
                        try:
                            spc.plt.plotBoundary(boundary, ax=ax, color='red',
                                               linewidth=2, alpha=0.7,
                                               label=f'{marker}+ region {region_id}')
                        except:
                            pass

                ax.set_title(f'{sample}: {marker}+ Regions', fontsize=14, fontweight='bold')
                ax.set_xlabel('X (μm)', fontsize=12)
                ax.set_ylabel('Y (μm)', fontsize=12)
                if len(ax.get_legend_handles_labels()[0]) > 0:
                    ax.legend(loc='upper right', fontsize=10)

                # Right: Marker- regions
                ax = axes[1]
                marker_key_neg = f'{marker}_negative'
                if marker_key_neg in marker_boundaries[sample]:
                    neg_boundaries = marker_boundaries[sample][marker_key_neg]

                    for region_id, boundary in neg_boundaries.items():
                        try:
                            spc.plt.plotBoundary(boundary, ax=ax, color='blue',
                                               linewidth=2, alpha=0.7,
                                               label=f'{marker}- region {region_id}')
                        except:
                            pass

                ax.set_title(f'{sample}: {marker}- Regions', fontsize=14, fontweight='bold')
                ax.set_xlabel('X (μm)', fontsize=12)
                ax.set_ylabel('Y (μm)', fontsize=12)
                if len(ax.get_legend_handles_labels()[0]) > 0:
                    ax.legend(loc='upper right', fontsize=10)

                plt.tight_layout()
                plt.savefig(self.plots_dir / f'{sample}_{marker}_spatial_regions.png',
                           dpi=300, bbox_inches='tight')
                plt.close()

                # Create overlay plot showing both
                fig, ax = plt.subplots(figsize=(15, 12))
                ax.scatter(sample_coords[:, 0], sample_coords[:, 1],
                         s=0.5, c='lightgray', alpha=0.2, rasterized=True)

                # Plot marker+ in red
                if marker_key_pos in marker_boundaries[sample]:
                    for region_id, boundary in marker_boundaries[sample][marker_key_pos].items():
                        try:
                            spc.plt.plotBoundary(boundary, ax=ax, color='red',
                                               linewidth=2.5, alpha=0.8,
                                               label=f'{marker}+' if region_id == 0 else None)
                        except:
                            pass

                # Plot marker- in blue
                if marker_key_neg in marker_boundaries[sample]:
                    for region_id, boundary in marker_boundaries[sample][marker_key_neg].items():
                        try:
                            spc.plt.plotBoundary(boundary, ax=ax, color='blue',
                                               linewidth=2.5, alpha=0.8,
                                               label=f'{marker}-' if region_id == 0 else None)
                        except:
                            pass

                ax.set_title(f'{sample}: {marker} Spatial Heterogeneity', fontsize=16, fontweight='bold')
                ax.set_xlabel('X (μm)', fontsize=13)
                ax.set_ylabel('Y (μm)', fontsize=13)
                ax.set_aspect('equal')
                ax.invert_yaxis()
                ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)

                plt.tight_layout()
                plt.savefig(self.plots_dir / f'{sample}_{marker}_overlay.png',
                           dpi=300, bbox_inches='tight')
                plt.close()

        print(f"      ✓ Created boundary plots for {len(markers)} markers")

    def _plot_immune_enrichment(self, enrichment_df: pd.DataFrame):
        """Plot immune cell enrichment in marker+ vs marker- regions."""
        if enrichment_df.empty:
            return

        # Enrichment bar plot for each marker
        markers = enrichment_df['marker'].unique()

        for marker in markers:
            marker_data = enrichment_df[enrichment_df['marker'] == marker]

            if marker_data.empty:
                continue

            fig, axes = plt.subplots(2, 1, figsize=(12, 10))

            # Top: Enrichment fold-change
            ax = axes[0]
            plot_data = marker_data.pivot_table(
                index='immune_population',
                values='enrichment_fold_change',
                aggfunc='mean'
            ).sort_values(ascending=False)

            colors = ['red' if x > 1 else 'blue' for x in plot_data.values]
            plot_data.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=1.5)
            ax.axvline(1, color='black', linestyle='--', linewidth=2, label='No enrichment')
            ax.set_xlabel('Enrichment Fold-Change (Positive/Negative)', fontsize=12)
            ax.set_ylabel('Immune Population', fontsize=12)
            ax.set_title(f'{marker}: Immune Enrichment in Positive vs Negative Regions',
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(axis='x', alpha=0.3)

            # Bottom: Percent immune in each region type
            ax = axes[1]
            percent_data = marker_data.groupby('immune_population').agg({
                'percent_immune_positive': 'mean',
                'percent_immune_negative': 'mean'
            }).sort_values('percent_immune_positive', ascending=False)

            x = np.arange(len(percent_data))
            width = 0.35

            ax.barh(x - width/2, percent_data['percent_immune_positive'],
                   width, label=f'{marker}+ regions', color='red', alpha=0.7, edgecolor='black')
            ax.barh(x + width/2, percent_data['percent_immune_negative'],
                   width, label=f'{marker}- regions', color='blue', alpha=0.7, edgecolor='black')

            ax.set_yticks(x)
            ax.set_yticklabels(percent_data.index)
            ax.set_xlabel('Percent Immune Cells (%)', fontsize=12)
            ax.set_ylabel('Immune Population', fontsize=12)
            ax.set_title(f'{marker}: Immune Cell Abundance', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.plots_dir / f'{marker}_immune_enrichment.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

        # Summary heatmap across all markers
        if len(markers) > 1:
            pivot_data = enrichment_df.pivot_table(
                index='immune_population',
                columns='marker',
                values='enrichment_fold_change',
                aggfunc='mean'
            )

            fig, ax = plt.subplots(figsize=(len(markers)*2 + 4, len(pivot_data)*0.5 + 2))
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=1, vmin=0.5, vmax=2,
                       cbar_kws={'label': 'Enrichment Fold-Change'},
                       linewidths=1, linecolor='white', ax=ax)
            ax.set_title('Immune Enrichment Across All Markers', fontsize=16, fontweight='bold')
            ax.set_xlabel('Marker', fontsize=13)
            ax.set_ylabel('Immune Population', fontsize=13)

            plt.tight_layout()
            plt.savefig(self.plots_dir / 'immune_enrichment_heatmap.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

        print(f"      ✓ Created enrichment plots for {len(markers)} markers")

    def _plot_marker_comparison(self, comparison_df: pd.DataFrame):
        """Plot marker+ vs marker- comparison."""
        if comparison_df.empty:
            return

        # Focus on CD8 and immune populations
        immune_phenotypes = [p for p in comparison_df['phenotype'].unique()
                            if 'CD8' in p or 'CD4' in p or 'CD45' in p or 'T_cells' in p]

        if not immune_phenotypes:
            return

        comparison_subset = comparison_df[comparison_df['phenotype'].isin(immune_phenotypes)]

        # Enrichment plot
        fig, ax = plt.subplots(figsize=(14, 8))

        markers = comparison_subset['marker'].unique()
        x = np.arange(len(immune_phenotypes))
        width = 0.8 / len(markers)

        for i, marker in enumerate(markers):
            marker_data = comparison_subset[comparison_subset['marker'] == marker]
            fold_changes = [marker_data[marker_data['phenotype'] == p]['fold_change_pos_vs_neg'].values[0]
                          if len(marker_data[marker_data['phenotype'] == p]) > 0 else 0
                          for p in immune_phenotypes]

            ax.bar(x + i*width - 0.4 + width/2, fold_changes,
                  width, label=marker, alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.axhline(1, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Immune Population', fontsize=13)
        ax.set_ylabel('Fold-Change (Positive/Negative)', fontsize=13)
        ax.set_title('Immune Cell Enrichment: Marker+ vs Marker- Regions',
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(immune_phenotypes, rotation=45, ha='right')
        ax.legend(title='Marker', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'marker_comparison_immune.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"      ✓ Created comparison plots")

    def _plot_regional_composition(self, composition_df: pd.DataFrame):
        """Plot cell type composition within marker regions."""
        if composition_df.empty:
            return

        # Composition by marker and polarity
        markers = composition_df['marker'].unique()

        for marker in markers:
            marker_data = composition_df[composition_df['marker'] == marker]

            # Group by polarity
            pos_data = marker_data[marker_data['polarity'] == 'positive']
            neg_data = marker_data[marker_data['polarity'] == 'negative']

            if pos_data.empty or neg_data.empty:
                continue

            # Get top phenotypes
            phenotypes = marker_data.groupby('phenotype')['composition'].mean().sort_values(ascending=False).head(10).index

            fig, ax = plt.subplots(figsize=(12, 8))

            x = np.arange(len(phenotypes))
            width = 0.35

            pos_means = [pos_data[pos_data['phenotype'] == p]['composition'].mean() * 100
                        if len(pos_data[pos_data['phenotype'] == p]) > 0 else 0
                        for p in phenotypes]
            neg_means = [neg_data[neg_data['phenotype'] == p]['composition'].mean() * 100
                        if len(neg_data[neg_data['phenotype'] == p]) > 0 else 0
                        for p in phenotypes]

            ax.bar(x - width/2, pos_means, width, label=f'{marker}+', color='red', alpha=0.7, edgecolor='black')
            ax.bar(x + width/2, neg_means, width, label=f'{marker}-', color='blue', alpha=0.7, edgecolor='black')

            ax.set_xlabel('Phenotype', fontsize=12)
            ax.set_ylabel('Percent Composition (%)', fontsize=12)
            ax.set_title(f'{marker}: Regional Cell Composition', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(phenotypes, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.plots_dir / f'{marker}_composition.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

        print(f"      ✓ Created composition plots for {len(markers)} markers")
