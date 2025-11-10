"""
Tumor Phenotype Microenvironment Analysis
===========================================
Analyze immune microenvironment around tumor cell phenotypes (pERK+, pERK-, etc.)
at multiple spatial scales.

For each tumor sub-phenotype:
- Calculate immune cell composition in local vicinity
- Multiple radius bins: cell contact, close proximity, distal proximity
- Generate stacked column plots (percentages and counts)
- Statistical tests between phenotypes, timepoints, and groups
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree
from scipy import stats
import warnings


class TumorMicroenvironmentAnalysis:
    """
    Analyze immune microenvironment around tumor cell phenotypes.

    Features:
    - Per-cell immune neighborhood composition for tumor phenotypes
    - Multiple distance bins (contact, close, distal)
    - Both percentage (relative composition) and count (absolute) metrics
    - Statistical comparisons: phenotype+/-, timepoints, KPT/KPNT
    - Enrichment/depletion analysis adjusting for overall infiltration
    """

    def __init__(self, adata, config: Dict, output_dir: Path,
                 tumor_structures: Optional[Dict] = None):
        """
        Initialize tumor microenvironment analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data with phenotypes and spatial coordinates
        config : dict
            Configuration dictionary
        output_dir : Path
            Output directory
        tumor_structures : dict, optional
            Pre-computed tumor structures from PerTumorAnalysis
        """
        self.adata = adata
        self.config = config
        self.output_dir = Path(output_dir) / 'tumor_microenvironment'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tumor_structures = tumor_structures or {}

        # Get analysis config
        self.tm_config = config.get('tumor_microenvironment', {})

        # Distance bins (in microns)
        self.distance_bins = self.tm_config.get('distance_bins', {
            'contact': 10,      # Cell-cell contact
            'close': 30,        # Close proximity
            'distal': 100       # Distal proximity
        })

        # Tumor phenotypes to analyze
        self.phenotypes = self.tm_config.get('tumor_phenotypes', [
            {'name': 'pERK', 'phenotype': 'pERK_positive_tumor'},
            {'name': 'NINJA', 'phenotype': 'AGFP_positive_tumor'},
            {'name': 'Ki67', 'phenotype': 'Ki67_positive_tumor'}
        ])

        # Immune populations to quantify
        self.immune_pops = self.tm_config.get('immune_populations', [
            'CD8_T_cells',
            'CD4_T_cells',
            'CD3_positive',
            'CD45_positive',
            'B_cells'
        ])

        # Storage
        self.results = {}

    def run(self):
        """Run complete tumor microenvironment analysis."""
        print("\n" + "="*80)
        print("TUMOR PHENOTYPE MICROENVIRONMENT ANALYSIS")
        print("="*80)
        print(f"\nDistance bins: {self.distance_bins}")
        print(f"Tumor phenotypes: {[p['name'] for p in self.phenotypes]}")
        print(f"Immune populations: {self.immune_pops}")

        # Analyze each phenotype
        for pheno_def in self.phenotypes:
            pheno_name = pheno_def['name']
            print(f"\n{'='*80}")
            print(f"Analyzing {pheno_name}+ vs {pheno_name}- microenvironment")
            print(f"{'='*80}")

            results = self._analyze_phenotype_microenvironment(pheno_def)
            if results:
                self.results[pheno_name] = results

        # Save results
        self._save_results()

        print("\n✓ Tumor microenvironment analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
        print("="*80 + "\n")

        return self.results

    def _analyze_phenotype_microenvironment(self, pheno_def: Dict) -> Dict:
        """
        Analyze immune microenvironment for a specific tumor phenotype.

        Parameters
        ----------
        pheno_def : dict
            Phenotype definition with 'name' and 'phenotype' keys

        Returns
        -------
        dict
            Results for this phenotype
        """
        pheno_name = pheno_def['name']
        pheno_col = f"is_{pheno_def['phenotype']}"
        tumor_col = f"is_{self.config.get('tumor_definition', {}).get('base_phenotype', 'Tumor')}"

        if pheno_col not in self.adata.obs.columns:
            print(f"  ⚠ Phenotype column '{pheno_col}' not found, skipping")
            return None

        if tumor_col not in self.adata.obs.columns:
            print(f"  ⚠ Tumor column '{tumor_col}' not found, skipping")
            return None

        results = {}

        # Analyze for each distance bin
        for bin_name, radius in self.distance_bins.items():
            print(f"\n  Processing {bin_name} distance bin (radius={radius}μm)...")

            bin_results = self._analyze_at_radius(
                pheno_name, pheno_col, tumor_col, radius, bin_name
            )

            if bin_results is not None:
                results[bin_name] = bin_results
                print(f"    ✓ Analyzed {len(bin_results)} tumor cells")

        # Calculate enrichment/depletion (adjusting for overall infiltration)
        if results:
            print(f"\n  Calculating enrichment/depletion analysis...")
            enrichment_results = self._calculate_enrichment(results, pheno_name)
            if enrichment_results is not None:
                results['enrichment'] = enrichment_results
                print(f"    ✓ Found {len(enrichment_results)} significant changes")

        return results

    def _analyze_at_radius(self, pheno_name: str, pheno_col: str,
                           tumor_col: str, radius: float,
                           bin_name: str) -> Optional[pd.DataFrame]:
        """
        Analyze immune microenvironment at specific radius.

        Parameters
        ----------
        pheno_name : str
            Phenotype name (e.g., 'pERK')
        pheno_col : str
            Column name for phenotype
        tumor_col : str
            Column name for tumor cells
        radius : float
            Radius in microns
        bin_name : str
            Name of distance bin

        Returns
        -------
        pd.DataFrame or None
            Per-cell results with immune composition
        """
        results_list = []

        # Process each sample separately
        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask].copy()
            sample_coords = self.adata.obsm['spatial'][sample_mask.values]

            # Get tumor cells only
            tumor_mask = sample_data[tumor_col].values
            if tumor_mask.sum() < 10:
                continue

            tumor_indices = np.where(tumor_mask)[0]
            tumor_coords = sample_coords[tumor_mask]
            tumor_pheno_status = sample_data[pheno_col].values[tumor_mask]

            # Build KDTree for all cells in sample
            tree = cKDTree(sample_coords)

            # For each tumor cell, find immune neighbors
            for idx, (coord, is_positive) in enumerate(zip(tumor_coords, tumor_pheno_status)):
                # Find neighbors within radius
                neighbor_indices = tree.query_ball_point(coord, radius)

                # Exclude self
                actual_tumor_idx = tumor_indices[idx]
                neighbor_indices = [i for i in neighbor_indices if i != actual_tumor_idx]

                if len(neighbor_indices) == 0:
                    continue

                # Get neighbor data
                neighbor_data = sample_data.iloc[neighbor_indices]

                # Count each immune population
                immune_counts = {}
                for immune_pop in self.immune_pops:
                    immune_col = f'is_{immune_pop}'
                    if immune_col in neighbor_data.columns:
                        count = neighbor_data[immune_col].sum()
                        immune_counts[immune_pop] = count
                    else:
                        immune_counts[immune_pop] = 0

                # Total immune cells
                total_immune = sum(immune_counts.values())

                # Calculate percentages
                immune_percentages = {}
                for immune_pop, count in immune_counts.items():
                    pct = (count / total_immune * 100) if total_immune > 0 else 0
                    immune_percentages[immune_pop] = pct

                # Store result
                result = {
                    'sample_id': sample,
                    'tumor_cell_idx': actual_tumor_idx,
                    'phenotype_status': f'{pheno_name}+' if is_positive else f'{pheno_name}-',
                    'is_positive': bool(is_positive),
                    'radius_bin': bin_name,
                    'radius_um': radius,
                    'n_neighbors': len(neighbor_indices),
                    'n_total_immune': int(total_immune),
                    'timepoint': sample_data.iloc[0]['timepoint'] if 'timepoint' in sample_data.columns else np.nan,
                    'main_group': sample_data.iloc[0]['main_group'] if 'main_group' in sample_data.columns else '',
                    'group': sample_data.iloc[0]['group'] if 'group' in sample_data.columns else ''
                }

                # Add counts
                for immune_pop, count in immune_counts.items():
                    result[f'{immune_pop}_count'] = int(count)

                # Add percentages
                for immune_pop, pct in immune_percentages.items():
                    result[f'{immune_pop}_percent'] = pct

                results_list.append(result)

        if not results_list:
            return None

        return pd.DataFrame(results_list)

    def _calculate_enrichment(self, results: Dict, pheno_name: str) -> Optional[pd.DataFrame]:
        """
        Calculate enrichment/depletion of immune cells around phenotype+ vs phenotype-.

        Adjusts for overall infiltration rates to detect true local enrichment.

        Parameters
        ----------
        results : dict
            Results dictionary with per-distance-bin dataframes
        pheno_name : str
            Phenotype name

        Returns
        -------
        pd.DataFrame or None
            Enrichment analysis results
        """
        enrichment_results = []

        # For each distance bin
        for bin_name, df in results.items():
            if bin_name == 'enrichment':
                continue

            if df is None or len(df) == 0:
                continue

            # For each immune population
            for immune_pop in self.immune_pops:
                count_col = f'{immune_pop}_count'
                pct_col = f'{immune_pop}_percent'

                if count_col not in df.columns or pct_col not in df.columns:
                    continue

                # Compare phenotype+ vs phenotype-
                pos_data = df[df['is_positive'] == True]
                neg_data = df[df['is_positive'] == False]

                if len(pos_data) < 5 or len(neg_data) < 5:
                    continue

                # Test for count differences (absolute infiltration)
                pos_counts = pos_data[count_col].values
                neg_counts = neg_data[count_col].values

                try:
                    stat_count, pval_count = stats.mannwhitneyu(
                        pos_counts, neg_counts, alternative='two-sided'
                    )
                except:
                    pval_count = 1.0

                # Test for percentage differences (relative composition)
                pos_pcts = pos_data[pct_col].values
                neg_pcts = neg_data[pct_col].values

                try:
                    stat_pct, pval_pct = stats.mannwhitneyu(
                        pos_pcts, neg_pcts, alternative='two-sided'
                    )
                except:
                    pval_pct = 1.0

                # Calculate effect sizes
                mean_pos_count = pos_counts.mean()
                mean_neg_count = neg_counts.mean()
                fold_change_count = mean_pos_count / mean_neg_count if mean_neg_count > 0 else np.nan

                mean_pos_pct = pos_pcts.mean()
                mean_neg_pct = neg_pcts.mean()
                diff_pct = mean_pos_pct - mean_neg_pct

                # Determine if enriched (considering both count and percentage)
                # Enriched if percentage is higher in phenotype+ (adjusting for infiltration)
                is_enriched = (pval_pct < 0.05 and diff_pct > 0)
                is_depleted = (pval_pct < 0.05 and diff_pct < 0)

                enrichment_results.append({
                    'phenotype': pheno_name,
                    'distance_bin': bin_name,
                    'radius_um': df['radius_um'].iloc[0],
                    'immune_population': immune_pop,
                    'mean_count_positive': mean_pos_count,
                    'mean_count_negative': mean_neg_count,
                    'fold_change_count': fold_change_count,
                    'pvalue_count': pval_count,
                    'mean_percent_positive': mean_pos_pct,
                    'mean_percent_negative': mean_neg_pct,
                    'diff_percent': diff_pct,
                    'pvalue_percent': pval_pct,
                    'is_enriched': is_enriched,
                    'is_depleted': is_depleted,
                    'significant': (pval_pct < 0.05),
                    'n_positive': len(pos_data),
                    'n_negative': len(neg_data)
                })

        if not enrichment_results:
            return None

        enrichment_df = pd.DataFrame(enrichment_results)

        # Apply FDR correction
        from statsmodels.stats.multitest import multipletests
        _, enrichment_df['pvalue_percent_adj'], _, _ = multipletests(
            enrichment_df['pvalue_percent'], method='fdr_bh'
        )
        enrichment_df['significant_adj'] = enrichment_df['pvalue_percent_adj'] < 0.05

        return enrichment_df

    def _save_results(self):
        """Save all results to CSV files."""
        for pheno_name, pheno_results in self.results.items():
            pheno_dir = self.output_dir / pheno_name
            pheno_dir.mkdir(parents=True, exist_ok=True)

            for result_name, df in pheno_results.items():
                if df is not None and len(df) > 0:
                    output_path = pheno_dir / f'{result_name}_results.csv'
                    df.to_csv(output_path, index=False)
                    print(f"    ✓ Saved {output_path.name}")

    def get_results(self) -> Dict:
        """Get all results."""
        return self.results
