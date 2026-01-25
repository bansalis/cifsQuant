"""
Cluster Composition Analysis
Analyze immune cell composition within B cell clusters/tumor structures
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


class ClusterCompositionAnalysis:
    """
    Analyze and visualize the cellular composition of clusters/structures.

    Generates:
    - Stacked bar charts showing cell type composition per structure
    - Composition over time
    - Group comparisons
    """

    def __init__(self, adata, config: Dict, output_dir: Path, tumor_structures: Dict = None):
        """
        Initialize cluster composition analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data with phenotypes
        config : dict
            Configuration dictionary
        output_dir : Path
            Output directory
        tumor_structures : dict, optional
            Pre-computed structure labels per sample
        """
        self.adata = adata
        self.full_config = config
        self.config = config.get('cluster_composition_analysis', {})
        self.output_dir = Path(output_dir) / 'cluster_composition_analysis'
        self.plots_dir = self.output_dir / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.tumor_structures = tumor_structures

        # Get structure config
        self.structure_config = config.get('structure_definition', config.get('tumor_definition', {}))

        # Get metadata config
        meta_config = config.get('metadata', {})
        self.group_col = meta_config.get('group_column', 'group')

        # Results storage
        self.results = {}

    def run(self):
        """Run cluster composition analysis."""
        print("\n" + "="*80)
        print("CLUSTER COMPOSITION ANALYSIS")
        print("="*80)

        # Get populations to analyze
        populations = self.config.get('populations', [])
        if not populations:
            print("  ⚠ No populations configured for composition analysis")
            return self.results

        print(f"\nAnalyzing composition of {len(populations)} populations...")

        # Detect structures if not provided
        if self.tumor_structures is None:
            self._detect_structures()

        if not self.tumor_structures:
            print("  ⚠ No structures detected, skipping composition analysis")
            return self.results

        # Calculate composition per structure
        self._calculate_composition(populations)

        # Save results
        self._save_results()

        # Generate plots
        if self.config.get('generate_plots', True):
            self._generate_plots()

        print("\n✓ Cluster composition analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
        print("="*80 + "\n")

        return self.results

    def _detect_structures(self):
        """Detect structures using DBSCAN if not provided."""
        from sklearn.cluster import DBSCAN

        struct_config = self.structure_config
        base_pheno = struct_config.get('base_phenotype', 'B_cells')
        pheno_col = f'is_{base_pheno}'

        if pheno_col not in self.adata.obs.columns:
            print(f"  ⚠ Structure phenotype column '{pheno_col}' not found")
            return

        detect_config = struct_config.get('structure_detection', {})
        eps = detect_config.get('eps', 100)
        min_samples = detect_config.get('min_samples', 50)
        min_cluster_size = detect_config.get('min_cluster_size', 100)

        self.tumor_structures = {}

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]
            sample_coords = self.adata.obsm['spatial'][sample_mask.values]

            # Get structure cells
            pheno_mask = sample_data[pheno_col].values
            if pheno_mask.sum() < min_samples:
                continue

            pheno_coords = sample_coords[pheno_mask]

            # Run DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pheno_coords)
            labels = clustering.labels_

            # Filter by minimum size
            valid_labels = [label for label in set(labels) - {-1}
                          if (labels == label).sum() >= min_cluster_size]

            # Create full sample labels
            structure_labels = np.full(len(sample_data), -1)
            pheno_indices = np.where(pheno_mask)[0]
            for label in valid_labels:
                cluster_mask = labels == label
                structure_labels[pheno_indices[cluster_mask]] = label

            self.tumor_structures[sample] = structure_labels

        print(f"  ✓ Detected structures in {len(self.tumor_structures)} samples")

    def _calculate_composition(self, populations: List[str]):
        """Calculate cell type composition per structure."""
        results = []

        for sample, structure_labels in self.tumor_structures.items():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]

            unique_structures = set(structure_labels) - {-1}

            for structure_id in unique_structures:
                structure_mask = structure_labels == structure_id
                structure_data = sample_data[structure_mask]

                if len(structure_data) == 0:
                    continue

                n_total = len(structure_data)

                result = {
                    'sample_id': sample,
                    'structure_id': int(structure_id),
                    'n_cells': n_total,
                    'timepoint': structure_data['timepoint'].iloc[0] if 'timepoint' in structure_data.columns else np.nan,
                    'group': structure_data[self.group_col].iloc[0] if self.group_col in structure_data.columns else ''
                }

                # Count each population
                for pop in populations:
                    pop_col = f'is_{pop}'
                    if pop_col in structure_data.columns:
                        n_pop = structure_data[pop_col].sum()
                        result[f'{pop}_count'] = int(n_pop)
                        result[f'{pop}_percent'] = (n_pop / n_total) * 100 if n_total > 0 else 0
                    else:
                        result[f'{pop}_count'] = 0
                        result[f'{pop}_percent'] = 0

                results.append(result)

        if results:
            df = pd.DataFrame(results)
            self.results['composition'] = df
            print(f"  ✓ Calculated composition for {len(results)} structures")

    def _save_results(self):
        """Save results to files."""
        for name, df in self.results.items():
            output_path = self.output_dir / f'{name}.csv'
            df.to_csv(output_path, index=False)

        print(f"  ✓ Saved {len(self.results)} result files")

    def _generate_plots(self):
        """Generate composition plots."""
        if 'composition' not in self.results:
            return

        df = self.results['composition']
        populations = self.config.get('populations', [])

        # Get percent columns
        percent_cols = [f'{pop}_percent' for pop in populations if f'{pop}_percent' in df.columns]

        if not percent_cols:
            return

        # Stacked bar by timepoint
        if 'timepoint' in df.columns:
            self._plot_stacked_bar_by_timepoint(df, percent_cols, populations)

        # Mean composition per group
        self._plot_mean_composition(df, percent_cols, populations)

        print(f"  ✓ Generated composition plots")

    def _plot_stacked_bar_by_timepoint(self, df: pd.DataFrame, percent_cols: List[str], populations: List[str]):
        """Create stacked bar chart by timepoint."""
        timepoints = sorted(df['timepoint'].unique())

        # Calculate mean percentages per timepoint
        means = df.groupby('timepoint')[percent_cols].mean()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Use colormap
        colors = plt.cm.tab20(np.linspace(0, 1, len(percent_cols)))

        bottom = np.zeros(len(timepoints))

        for idx, (col, pop) in enumerate(zip(percent_cols, populations)):
            if col in means.columns:
                values = means[col].values
                ax.bar(range(len(timepoints)), values, bottom=bottom,
                      label=pop, color=colors[idx], width=0.7)
                bottom += values

        ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
        ax.set_ylabel('% of Structure Cells', fontsize=12, fontweight='bold')
        ax.set_title('Cluster Composition Over Time', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(timepoints)))
        ax.set_xticklabels([str(tp) for tp in timepoints])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'composition_stacked_bar_by_timepoint.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_mean_composition(self, df: pd.DataFrame, percent_cols: List[str], populations: List[str]):
        """Create mean composition pie/bar chart."""
        means = df[percent_cols].mean()

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.tab20(np.linspace(0, 1, len(percent_cols)))

        bars = ax.bar(range(len(populations)), means.values, color=colors, width=0.7)

        ax.set_xlabel('Population', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean % of Structure Cells', fontsize=12, fontweight='bold')
        ax.set_title('Mean Cluster Composition', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(populations)))
        ax.set_xticklabels(populations, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'mean_composition.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
