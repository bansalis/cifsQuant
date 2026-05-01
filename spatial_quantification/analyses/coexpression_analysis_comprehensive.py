"""
Comprehensive Coexpression Analysis
Analyze coexpression patterns for ALL phenotypes defined in config
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings


class CoexpressionAnalysisComprehensive:
    """
    Comprehensive coexpression analysis for all phenotypes in config.

    Key features:
    - Dynamically analyzes ALL phenotypes from config
    - Calculates pairwise and multi-marker coexpression
    - Creates comprehensive coexpression matrices
    - Generates heatmaps and visualization plots
    - Exports results as CSV files
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        """
        Initialize comprehensive coexpression analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data with phenotypes
        config : dict
            Configuration dictionary
        output_dir : Path
            Output directory
        """
        self.adata = adata
        self.config = config
        self.output_dir = Path(output_dir) / 'coexpression_analysis_comprehensive'
        self.plots_dir = self.output_dir / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Get all phenotypes from config
        self.phenotypes = self._extract_phenotypes_from_config()
        print(f"  Found {len(self.phenotypes)} phenotypes in config")

        # Results storage
        self.results = {}

    def _extract_phenotypes_from_config(self) -> List[str]:
        """Extract all phenotype names from config."""
        phenotypes = []

        if 'phenotypes' in self.config:
            phenotypes = list(self.config['phenotypes'].keys())

        return phenotypes

    def run(self) -> Dict:
        """Run complete comprehensive coexpression analysis."""
        print("\n" + "="*80)
        print("COMPREHENSIVE COEXPRESSION ANALYSIS")
        print("="*80)

        if len(self.phenotypes) < 2:
            print("  ⚠ Need at least 2 phenotypes for coexpression analysis")
            return {}

        # Calculate single phenotype frequencies
        print("\n1. Calculating single phenotype frequencies...")
        self._calculate_single_frequencies()

        # Calculate pairwise coexpression
        print("\n2. Calculating pairwise coexpression...")
        self._calculate_pairwise_coexpression()

        # Create coexpression matrices
        print("\n3. Creating coexpression matrices...")
        self._create_coexpression_matrices()

        # Calculate triple+ coexpression (up to 5-way)
        print("\n4. Calculating multi-marker coexpression...")
        self._calculate_multi_marker_coexpression()

        # Save results
        print("\n5. Saving results...")
        self._save_results()

        # Generate plots
        print("\n6. Generating visualizations...")
        self._generate_plots()

        print("\n✓ Comprehensive coexpression analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
        print(f"  Plots saved to: {self.plots_dir}/")
        print("="*80 + "\n")

        return self.results

    def _calculate_single_frequencies(self):
        """Calculate frequency of each phenotype."""
        results = []

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]

            n_total_cells = len(sample_data)

            result = {
                'sample_id': sample,
                'n_total_cells': n_total_cells,
                'timepoint': sample_data['timepoint'].iloc[0] if 'timepoint' in sample_data.columns else np.nan,
                'group': sample_data['group'].iloc[0] if 'group' in sample_data.columns else '',
                'main_group': sample_data['main_group'].iloc[0] if 'main_group' in sample_data.columns else ''
            }

            # Calculate frequency for each phenotype
            for pheno in self.phenotypes:
                pheno_col = f'is_{pheno}'
                if pheno_col in sample_data.columns:
                    n_positive = sample_data[pheno_col].sum()
                    freq = n_positive / n_total_cells * 100 if n_total_cells > 0 else 0
                    result[f'{pheno}_count'] = int(n_positive)
                    result[f'{pheno}_percent'] = freq

            results.append(result)

        if results:
            df = pd.DataFrame(results)
            self.results['single_phenotype_frequencies'] = df
            print(f"    ✓ Calculated frequencies for {len(results)} samples")

    def _calculate_pairwise_coexpression(self):
        """Calculate pairwise coexpression for all phenotype combinations."""
        results = []

        # Generate all pairwise combinations
        pheno_pairs = list(combinations(self.phenotypes, 2))
        print(f"    Analyzing {len(pheno_pairs)} pairwise combinations...")

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]

            n_total_cells = len(sample_data)

            result = {
                'sample_id': sample,
                'n_total_cells': n_total_cells,
                'timepoint': sample_data['timepoint'].iloc[0] if 'timepoint' in sample_data.columns else np.nan,
                'group': sample_data['group'].iloc[0] if 'group' in sample_data.columns else '',
                'main_group': sample_data['main_group'].iloc[0] if 'main_group' in sample_data.columns else ''
            }

            # Calculate pairwise coexpression
            for pheno1, pheno2 in pheno_pairs:
                col1 = f'is_{pheno1}'
                col2 = f'is_{pheno2}'

                if col1 in sample_data.columns and col2 in sample_data.columns:
                    # Both positive
                    both_pos = (sample_data[col1] & sample_data[col2]).sum()

                    # Either positive (union)
                    either_pos = (sample_data[col1] | sample_data[col2]).sum()

                    # Individual counts
                    pheno1_pos = sample_data[col1].sum()
                    pheno2_pos = sample_data[col2].sum()

                    # Percent of total cells
                    percent_both = both_pos / n_total_cells * 100 if n_total_cells > 0 else 0

                    # Percent of pheno1+ cells that are also pheno2+
                    percent_of_pheno1 = both_pos / pheno1_pos * 100 if pheno1_pos > 0 else 0

                    # Percent of pheno2+ cells that are also pheno1+
                    percent_of_pheno2 = both_pos / pheno2_pos * 100 if pheno2_pos > 0 else 0

                    # Jaccard index (similarity coefficient)
                    jaccard = both_pos / either_pos if either_pos > 0 else 0

                    # Expected coexpression under independence: P(pheno1+) * P(pheno2+) * 100
                    pheno1_pct = pheno1_pos / n_total_cells * 100 if n_total_cells > 0 else 0
                    pheno2_pct = pheno2_pos / n_total_cells * 100 if n_total_cells > 0 else 0
                    expected_pct = (pheno1_pct / 100) * (pheno2_pct / 100) * 100

                    # Fold enrichment: observed / expected
                    epsilon = 1e-6
                    fold_enrichment = percent_both / (expected_pct + epsilon)
                    log2_fold_enrichment = np.log2(fold_enrichment + epsilon)

                    # Store results
                    result[f'{pheno1}_AND_{pheno2}_count'] = int(both_pos)
                    result[f'{pheno1}_AND_{pheno2}_percent_of_total'] = percent_both
                    result[f'{pheno1}_AND_{pheno2}_percent_of_{pheno1}'] = percent_of_pheno1
                    result[f'{pheno1}_AND_{pheno2}_percent_of_{pheno2}'] = percent_of_pheno2
                    result[f'{pheno1}_AND_{pheno2}_jaccard'] = jaccard
                    result[f'{pheno1}_AND_{pheno2}_expected_pct'] = expected_pct
                    result[f'{pheno1}_AND_{pheno2}_fold_enrichment'] = fold_enrichment
                    result[f'{pheno1}_AND_{pheno2}_log2_fold_enrichment'] = log2_fold_enrichment

            results.append(result)

        if results:
            df = pd.DataFrame(results)
            self.results['pairwise_coexpression'] = df
            print(f"    ✓ Calculated pairwise coexpression for {len(results)} samples")

    def _create_coexpression_matrices(self):
        """Create coexpression matrices showing all pairwise relationships."""
        if 'pairwise_coexpression' not in self.results:
            print("    ⚠ No pairwise coexpression data available")
            return

        df = self.results['pairwise_coexpression']

        # Create matrices for different metrics
        metrics = ['percent_of_total', 'jaccard']

        for metric in metrics:
            matrices_by_group = {}

            for group in df['main_group'].unique():
                if pd.isna(group) or group == '':
                    continue

                group_data = df[df['main_group'] == group]

                # Initialize matrix
                n_phenos = len(self.phenotypes)
                matrix = np.zeros((n_phenos, n_phenos))

                # Fill matrix
                for i, pheno1 in enumerate(self.phenotypes):
                    for j, pheno2 in enumerate(self.phenotypes):
                        if i == j:
                            matrix[i, j] = np.nan  # Diagonal
                        elif i < j:
                            # Upper triangle
                            col_name = f'{pheno1}_AND_{pheno2}_{metric}'
                            if col_name in group_data.columns:
                                matrix[i, j] = group_data[col_name].mean()
                        else:
                            # Lower triangle (symmetric)
                            col_name = f'{pheno2}_AND_{pheno1}_{metric}'
                            if col_name in group_data.columns:
                                matrix[i, j] = group_data[col_name].mean()

                # Convert to DataFrame
                matrix_df = pd.DataFrame(
                    matrix,
                    index=self.phenotypes,
                    columns=self.phenotypes
                )

                matrices_by_group[group] = matrix_df

            if matrices_by_group:
                self.results[f'coexpression_matrix_{metric}'] = matrices_by_group
                print(f"    ✓ Created {metric} matrices for {len(matrices_by_group)} groups")

    def _calculate_multi_marker_coexpression(self):
        """Calculate multi-marker coexpression (3-way, 4-way, 5-way)."""
        results_3way = []
        results_4way = []
        results_5way = []

        # Limit to prevent combinatorial explosion
        max_markers_for_analysis = min(10, len(self.phenotypes))
        phenos_subset = self.phenotypes[:max_markers_for_analysis]

        # Generate combinations
        triplets = list(combinations(phenos_subset, 3))
        quadruplets = list(combinations(phenos_subset, 4))
        quintuplets = list(combinations(phenos_subset, 5))

        print(f"    Analyzing {len(triplets)} 3-way, {len(quadruplets)} 4-way, {len(quintuplets)} 5-way combinations...")

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]

            n_total_cells = len(sample_data)

            base_result = {
                'sample_id': sample,
                'n_total_cells': n_total_cells,
                'timepoint': sample_data['timepoint'].iloc[0] if 'timepoint' in sample_data.columns else np.nan,
                'group': sample_data['group'].iloc[0] if 'group' in sample_data.columns else '',
                'main_group': sample_data['main_group'].iloc[0] if 'main_group' in sample_data.columns else ''
            }

            # 3-way combinations
            for phenos in triplets:
                result = base_result.copy()
                result['combination'] = '_AND_'.join(phenos)

                # Check all phenotypes are present
                all_present = all(f'is_{p}' in sample_data.columns for p in phenos)

                if all_present:
                    # All three positive
                    mask = pd.Series(True, index=sample_data.index)
                    for p in phenos:
                        mask &= sample_data[f'is_{p}']

                    count = mask.sum()
                    result['count'] = int(count)
                    result['percent'] = count / n_total_cells * 100 if n_total_cells > 0 else 0

                    results_3way.append(result)

            # 4-way combinations
            for phenos in quadruplets:
                result = base_result.copy()
                result['combination'] = '_AND_'.join(phenos)

                all_present = all(f'is_{p}' in sample_data.columns for p in phenos)

                if all_present:
                    mask = pd.Series(True, index=sample_data.index)
                    for p in phenos:
                        mask &= sample_data[f'is_{p}']

                    count = mask.sum()
                    result['count'] = int(count)
                    result['percent'] = count / n_total_cells * 100 if n_total_cells > 0 else 0

                    results_4way.append(result)

            # 5-way combinations
            for phenos in quintuplets:
                result = base_result.copy()
                result['combination'] = '_AND_'.join(phenos)

                all_present = all(f'is_{p}' in sample_data.columns for p in phenos)

                if all_present:
                    mask = pd.Series(True, index=sample_data.index)
                    for p in phenos:
                        mask &= sample_data[f'is_{p}']

                    count = mask.sum()
                    result['count'] = int(count)
                    result['percent'] = count / n_total_cells * 100 if n_total_cells > 0 else 0

                    results_5way.append(result)

        # Store results
        if results_3way:
            df = pd.DataFrame(results_3way)
            self.results['triple_coexpression'] = df
            print(f"    ✓ Calculated {len(results_3way)} 3-way combinations")

        if results_4way:
            df = pd.DataFrame(results_4way)
            self.results['quadruple_coexpression'] = df
            print(f"    ✓ Calculated {len(results_4way)} 4-way combinations")

        if results_5way:
            df = pd.DataFrame(results_5way)
            self.results['quintuple_coexpression'] = df
            print(f"    ✓ Calculated {len(results_5way)} 5-way combinations")

    def _save_results(self):
        """Save all results to CSV files."""
        for name, data in self.results.items():
            if isinstance(data, pd.DataFrame):
                output_path = self.output_dir / f'{name}.csv'
                data.to_csv(output_path, index=False)
            elif isinstance(data, dict):
                # Handle matrices
                for subname, df in data.items():
                    output_path = self.output_dir / f'{name}_{subname}.csv'
                    df.to_csv(output_path)

        print(f"  ✓ Saved {len(self.results)} result datasets")

    def _generate_plots(self):
        """Generate visualization plots."""
        # Coexpression heatmaps
        self._plot_coexpression_heatmaps()

        # Conditional probability heatmaps (NEW!)
        self._plot_conditional_coexpression_heatmaps()

        # Top coexpressing pairs
        self._plot_top_coexpressing_pairs()

        # Multi-marker coexpression
        self._plot_multi_marker_coexpression()

    def _plot_coexpression_heatmaps(self):
        """Generate heatmaps for coexpression matrices."""
        if 'coexpression_matrix_jaccard' not in self.results:
            return

        matrices = self.results['coexpression_matrix_jaccard']

        for group, matrix_df in matrices.items():
            fig, ax = plt.subplots(figsize=(12, 10))

            # Create heatmap
            sns.heatmap(
                matrix_df,
                annot=False,
                fmt='.2f',
                cmap='RdYlBu_r',
                center=0.5,
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Jaccard Index'},
                ax=ax,
                square=True
            )

            ax.set_title(f'Coexpression Matrix - {group}\n(Jaccard Similarity Index)',
                        fontsize=14, fontweight='bold')

            # Rotate labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

            plt.tight_layout()
            plt.savefig(self.plots_dir / f'coexpression_heatmap_{group}.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

        print(f"    ✓ Generated coexpression heatmaps for {len(matrices)} groups")

    def _plot_conditional_coexpression_heatmaps(self):
        """
        Generate conditional coexpression heatmaps showing:
        "Given phenotype X is positive, what % are also positive for phenotype Y?"

        This creates asymmetric matrices showing directional conditional probabilities.
        Useful for understanding dependencies like "If pERK+, what % are NINJA+?"
        """
        if 'pairwise_coexpression' not in self.results:
            print("    ⚠ No pairwise coexpression data for conditional plots")
            return

        df = self.results['pairwise_coexpression']

        # Create conditional probability matrices for each group
        for group in df['main_group'].unique():
            if pd.isna(group) or group == '':
                continue

            group_data = df[df['main_group'] == group]

            # Initialize matrix: rows = "given X", columns = "also Y"
            n_phenos = len(self.phenotypes)
            cond_matrix = np.zeros((n_phenos, n_phenos))

            # Fill matrix with conditional probabilities
            for i, pheno_x in enumerate(self.phenotypes):
                for j, pheno_y in enumerate(self.phenotypes):
                    if i == j:
                        # Diagonal: 100% (if X+, then X+ is trivially 100%)
                        cond_matrix[i, j] = 100.0
                    else:
                        # Find the column for "percent of X that are also Y"
                        col_name = f'{pheno_x}_AND_{pheno_y}_percent_of_{pheno_x}'

                        if col_name in group_data.columns:
                            # Mean across all samples in this group
                            cond_matrix[i, j] = group_data[col_name].mean()

            # Create DataFrame
            cond_df = pd.DataFrame(
                cond_matrix,
                index=[f'{p}+' for p in self.phenotypes],
                columns=[f'also\n{p}+' for p in self.phenotypes]
            )

            # Create heatmap
            fig, ax = plt.subplots(figsize=(14, 12))

            sns.heatmap(
                cond_df,
                annot=True,
                fmt='.1f',
                cmap='YlOrRd',
                vmin=0,
                vmax=100,
                cbar_kws={'label': '% of Row Phenotype\nthat are also Column Phenotype'},
                ax=ax,
                square=True,
                linewidths=0.5,
                linecolor='gray'
            )

            ax.set_title(
                f'Conditional Coexpression: {group}\n'
                f'"If row marker is positive, what % are also column marker positive?"',
                fontsize=14, fontweight='bold'
            )
            ax.set_xlabel('ALSO Positive For →', fontsize=12, fontweight='bold')
            ax.set_ylabel('← GIVEN Positive For', fontsize=12, fontweight='bold')

            # Rotate labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

            plt.tight_layout()
            plt.savefig(self.plots_dir / f'conditional_coexpression_{group}.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

            # Also create a focused version showing only high conditional probabilities (>25%)
            # This highlights strong dependencies
            fig, ax = plt.subplots(figsize=(14, 12))

            # Mask low values
            masked_matrix = cond_matrix.copy()
            masked_matrix[masked_matrix < 25.0] = np.nan

            masked_df = pd.DataFrame(
                masked_matrix,
                index=[f'{p}+' for p in self.phenotypes],
                columns=[f'also\n{p}+' for p in self.phenotypes]
            )

            sns.heatmap(
                masked_df,
                annot=True,
                fmt='.1f',
                cmap='YlOrRd',
                vmin=25,
                vmax=100,
                cbar_kws={'label': '% Conditional Probability'},
                ax=ax,
                square=True,
                linewidths=0.5,
                linecolor='gray',
                cbar=True
            )

            ax.set_title(
                f'Strong Conditional Coexpression: {group}\n'
                f'(Showing only >25% conditional probability)',
                fontsize=14, fontweight='bold'
            )
            ax.set_xlabel('ALSO Positive For →', fontsize=12, fontweight='bold')
            ax.set_ylabel('← GIVEN Positive For', fontsize=12, fontweight='bold')

            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

            plt.tight_layout()
            plt.savefig(self.plots_dir / f'conditional_coexpression_strong_{group}.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

        print(f"    ✓ Generated conditional coexpression heatmaps")

    def _plot_top_coexpressing_pairs(self):
        """Plot top coexpressing phenotype pairs."""
        if 'pairwise_coexpression' not in self.results:
            return

        df = self.results['pairwise_coexpression']

        # Extract all jaccard columns
        jaccard_cols = [col for col in df.columns if col.endswith('_jaccard')]

        if not jaccard_cols:
            return

        # Calculate mean jaccard for each pair across all samples
        mean_jaccard = {}
        for col in jaccard_cols:
            pair_name = col.replace('_jaccard', '').replace('_AND_', ' + ')
            mean_jaccard[pair_name] = df[col].mean()

        # Sort and get top 20
        sorted_pairs = sorted(mean_jaccard.items(), key=lambda x: x[1], reverse=True)[:20]

        if not sorted_pairs:
            return

        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 8))

        pairs = [p[0] for p in sorted_pairs]
        values = [p[1] for p in sorted_pairs]

        bars = ax.barh(pairs, values, color='steelblue')

        # Color code by value
        for bar, val in zip(bars, values):
            if val > 0.7:
                bar.set_color('#d62728')  # Red for high
            elif val > 0.4:
                bar.set_color('#ff7f0e')  # Orange for medium
            else:
                bar.set_color('#1f77b4')  # Blue for low

        ax.set_xlabel('Jaccard Index', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Coexpressing Phenotype Pairs', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'top_coexpressing_pairs.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Generated top coexpressing pairs plot")

    def _plot_multi_marker_coexpression(self):
        """Plot multi-marker coexpression frequencies."""
        for result_key in ['triple_coexpression', 'quadruple_coexpression', 'quintuple_coexpression']:
            if result_key not in self.results:
                continue

            df = self.results[result_key]

            # Group by combination and calculate mean
            combo_means = df.groupby('combination')['percent'].mean().sort_values(ascending=False).head(15)

            if len(combo_means) == 0:
                continue

            # Create bar plot
            fig, ax = plt.subplots(figsize=(14, 8))

            combos = combo_means.index.tolist()
            # Shorten names if too long
            combos_short = [c.replace('_AND_', '+') for c in combos]
            values = combo_means.values

            bars = ax.barh(combos_short, values, color='forestgreen')

            ax.set_xlabel('% of Total Cells', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {len(combos_short)} {result_key.replace("_coexpression", "").title()} Coexpression Patterns',
                        fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.plots_dir / f'{result_key}_top_combinations.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

        print(f"    ✓ Generated multi-marker coexpression plots")
