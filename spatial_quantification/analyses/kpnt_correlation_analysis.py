"""
KPNT Correlation Analysis
Analyze correlations between tumor size/infiltration and marker positivity for KPNT samples
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings


class KPNTCorrelationAnalysis:
    """
    Correlation analysis for KPNT samples.

    Analyzes:
    1. Tumor size (# cells) vs marker positivity (pERK, NINJA, Ki67, etc.)
    2. Within-tumor infiltration (%) vs marker positivity

    Creates box plots compatible with single or multiple timepoints.
    Generates both statistical and non-statistical versions.
    """

    def __init__(self, adata, config: Dict, output_dir: Path, per_tumor_results: Dict = None):
        """
        Initialize KPNT correlation analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data
        config : dict
            Configuration dictionary
        output_dir : Path
            Output directory
        per_tumor_results : dict, optional
            Results from per-tumor analysis
        """
        self.adata = adata
        self.config = config
        self.output_dir = Path(output_dir) / 'kpnt_correlation_analysis'
        self.plots_dir = self.output_dir / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.per_tumor_results = per_tumor_results
        self.results = {}

        # Get plotting settings
        plotting_config = config.get('plotting', {})
        self.timepoint_label = plotting_config.get('timepoint_label', 'Time (weeks)')

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300

    def run(self) -> Dict:
        """Run complete correlation analysis."""
        print("\n" + "="*80)
        print("KPNT CORRELATION ANALYSIS")
        print("="*80)

        # Filter to KPNT samples only
        kpnt_mask = self.adata.obs['main_group'] == 'KPNT'
        if kpnt_mask.sum() == 0:
            print("  ⚠ No KPNT samples found - skipping analysis")
            return {}

        print(f"  Found {kpnt_mask.sum()} cells from KPNT samples")

        # 1. Prepare tumor-level data
        print("\n1. Preparing tumor-level data...")
        tumor_data = self._prepare_tumor_data()

        if tumor_data is None or len(tumor_data) == 0:
            print("  ⚠ No tumor data available")
            return {}

        print(f"    ✓ Prepared data for {len(tumor_data)} tumors")
        self.results['tumor_data'] = tumor_data

        # 2. Calculate correlations
        print("\n2. Calculating correlations...")
        self._calculate_correlations(tumor_data)

        # 3. Generate visualizations
        print("\n3. Generating visualizations...")
        self._generate_plots(tumor_data)

        # 4. Save results
        print("\n4. Saving results...")
        self._save_results()

        print("\n✓ KPNT correlation analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
        print(f"  Plots saved to: {self.plots_dir}/")
        print("="*80 + "\n")

        return self.results

    def _prepare_tumor_data(self) -> pd.DataFrame:
        """
        Prepare tumor-level data combining size, marker positivity, and infiltration.

        Returns
        -------
        pd.DataFrame
            Tumor-level data with columns: sample_id, tumor_id, size, marker percentages, infiltration
        """
        # Get KPNT samples
        kpnt_samples = self.adata.obs[self.adata.obs['main_group'] == 'KPNT']['sample_id'].unique()

        tumor_records = []

        for sample in kpnt_samples:
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]

            # Get unique tumors in this sample
            if 'tumor_region_id' not in sample_data.columns:
                continue

            tumor_ids = sample_data['tumor_region_id'].unique()
            tumor_ids = tumor_ids[~pd.isna(tumor_ids)]

            for tumor_id in tumor_ids:
                tumor_mask = sample_data['tumor_region_id'] == tumor_id
                tumor_cells = sample_data[tumor_mask]

                # Get only tumor cells (not immune)
                if 'is_Tumor' in tumor_cells.columns:
                    actual_tumor_cells = tumor_cells[tumor_cells['is_Tumor'] == True]
                else:
                    actual_tumor_cells = tumor_cells

                if len(actual_tumor_cells) < 10:
                    continue

                record = {
                    'sample_id': sample,
                    'tumor_id': int(tumor_id),
                    'tumor_size': len(actual_tumor_cells),
                    'timepoint': sample_data['timepoint'].iloc[0] if 'timepoint' in sample_data.columns else np.nan
                }

                # Get marker positivity percentages
                markers = ['pERK', 'NINJA', 'Ki67', 'PDL1', 'EPCAM', 'TTF1', 'CC3', 'MHCII']
                for marker in markers:
                    col = f'is_{marker}_positive_tumor'
                    if col in actual_tumor_cells.columns:
                        positive = actual_tumor_cells[col].sum()
                        percent = (positive / len(actual_tumor_cells)) * 100
                        record[f'percent_{marker}_positive'] = percent

                # Get infiltration metrics (from all cells in tumor region, not just tumor cells)
                immune_pops = ['CD45_positive', 'T_cells', 'CD4_T_cells', 'CD8_T_cells',
                              'CD4_Tregs', 'PD1_positive_CD8', 'B_cells', 'Macrophages']

                for immune_pop in immune_pops:
                    col = f'is_{immune_pop}'
                    if col in tumor_cells.columns:
                        # Count immune cells within this tumor region
                        immune_count = tumor_cells[col].sum()
                        # Density = immune cells / tumor cells
                        infiltration_density = (immune_count / len(actual_tumor_cells)) * 100
                        record[f'infiltration_{immune_pop}_percent'] = infiltration_density

                tumor_records.append(record)

        if not tumor_records:
            return None

        return pd.DataFrame(tumor_records)

    def _calculate_correlations(self, tumor_data: pd.DataFrame):
        """Calculate Pearson and Spearman correlations."""
        # Get marker columns
        marker_cols = [col for col in tumor_data.columns if col.startswith('percent_') and col.endswith('_positive')]

        # Get infiltration columns
        infiltration_cols = [col for col in tumor_data.columns if col.startswith('infiltration_')]

        # Correlation: Tumor size vs marker positivity
        size_corr = []
        for marker_col in marker_cols:
            valid_mask = ~tumor_data[marker_col].isna() & ~tumor_data['tumor_size'].isna()
            if valid_mask.sum() < 3:
                continue

            x = tumor_data.loc[valid_mask, 'tumor_size'].values
            y = tumor_data.loc[valid_mask, marker_col].values

            pearson_r, pearson_p = stats.pearsonr(x, y)
            spearman_r, spearman_p = stats.spearmanr(x, y)

            size_corr.append({
                'marker': marker_col.replace('percent_', '').replace('_positive', ''),
                'n': valid_mask.sum(),
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            })

        if size_corr:
            self.results['size_marker_correlation'] = pd.DataFrame(size_corr)

        # Correlation: Infiltration vs marker positivity
        infiltration_corr = []
        for marker_col in marker_cols:
            for infiltration_col in infiltration_cols:
                valid_mask = (~tumor_data[marker_col].isna() &
                             ~tumor_data[infiltration_col].isna())
                if valid_mask.sum() < 3:
                    continue

                x = tumor_data.loc[valid_mask, infiltration_col].values
                y = tumor_data.loc[valid_mask, marker_col].values

                pearson_r, pearson_p = stats.pearsonr(x, y)
                spearman_r, spearman_p = stats.spearmanr(x, y)

                infiltration_corr.append({
                    'marker': marker_col.replace('percent_', '').replace('_positive', ''),
                    'immune_pop': infiltration_col.replace('infiltration_', '').replace('_percent', ''),
                    'n': valid_mask.sum(),
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p
                })

        if infiltration_corr:
            self.results['infiltration_marker_correlation'] = pd.DataFrame(infiltration_corr)

    def _generate_plots(self, tumor_data: pd.DataFrame):
        """Generate all visualization plots."""
        # Check if single or multiple timepoints
        timepoints = tumor_data['timepoint'].unique()
        timepoints = timepoints[~pd.isna(timepoints)]
        single_timepoint = len(timepoints) <= 1

        # Plot 1: Tumor size vs marker positivity
        self._plot_size_vs_markers(tumor_data, single_timepoint, with_stats=True)
        self._plot_size_vs_markers(tumor_data, single_timepoint, with_stats=False)

        # Plot 2: Infiltration vs marker positivity
        self._plot_infiltration_vs_markers(tumor_data, single_timepoint, with_stats=True)
        self._plot_infiltration_vs_markers(tumor_data, single_timepoint, with_stats=False)

    def _plot_size_vs_markers(self, tumor_data: pd.DataFrame, single_timepoint: bool,
                              with_stats: bool = True):
        """
        Plot tumor size vs marker positivity.

        Creates box plots stratified by tumor size quartiles.
        """
        marker_cols = [col for col in tumor_data.columns
                      if col.startswith('percent_') and col.endswith('_positive')]

        if len(marker_cols) == 0:
            return

        # Create size quartiles
        tumor_data['size_quartile'] = pd.qcut(tumor_data['tumor_size'], q=4,
                                               labels=['Q1\n(Smallest)', 'Q2', 'Q3', 'Q4\n(Largest)'],
                                               duplicates='drop')

        # Create subplots
        n_markers = len(marker_cols)
        n_cols = 3
        n_rows = int(np.ceil(n_markers / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = axes.flatten() if n_markers > 1 else [axes]

        for idx, marker_col in enumerate(marker_cols):
            ax = axes[idx]
            marker_name = marker_col.replace('percent_', '').replace('_positive', '')

            # Prepare data
            plot_data = tumor_data[['size_quartile', marker_col]].dropna()

            if len(plot_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{marker_name}', fontsize=12, fontweight='bold')
                continue

            # Box plot
            bp = ax.boxplot([plot_data[plot_data['size_quartile'] == q][marker_col].values
                            for q in ['Q1\n(Smallest)', 'Q2', 'Q3', 'Q4\n(Largest)']
                            if q in plot_data['size_quartile'].values],
                           labels=[q for q in ['Q1\n(Smallest)', 'Q2', 'Q3', 'Q4\n(Largest)']
                                  if q in plot_data['size_quartile'].values],
                           patch_artist=True, showfliers=True, widths=0.6)

            # Color boxes
            for patch in bp['boxes']:
                patch.set_facecolor('steelblue')
                patch.set_alpha(0.6)

            # Add statistics if requested
            if with_stats:
                # Kruskal-Wallis test (non-parametric ANOVA)
                groups = [plot_data[plot_data['size_quartile'] == q][marker_col].values
                         for q in plot_data['size_quartile'].unique()]
                groups = [g for g in groups if len(g) >= 3]

                if len(groups) >= 2:
                    try:
                        stat, pval = stats.kruskal(*groups)
                        sig_text = self._format_pvalue(pval)
                        ax.text(0.98, 0.98, f'Kruskal-Wallis: {sig_text}',
                               transform=ax.transAxes, ha='right', va='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                               fontsize=10, fontweight='bold')
                    except:
                        pass

            ax.set_xlabel('Tumor Size Quartile', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'% {marker_name}+ Tumor Cells', fontsize=11, fontweight='bold')
            ax.set_title(f'{marker_name} vs Tumor Size', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        # Hide unused subplots
        for idx in range(len(marker_cols), len(axes)):
            axes[idx].axis('off')

        stats_suffix = '_with_stats' if with_stats else '_no_stats'
        plt.suptitle(f'KPNT: Tumor Size vs Marker Positivity{" (with statistics)" if with_stats else ""}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        plot_path = self.plots_dir / f'size_vs_markers{stats_suffix}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved: {plot_path.name}")

    def _plot_infiltration_vs_markers(self, tumor_data: pd.DataFrame, single_timepoint: bool,
                                      with_stats: bool = True):
        """
        Plot infiltration vs marker positivity.

        Creates box plots stratified by infiltration quartiles.
        """
        marker_cols = [col for col in tumor_data.columns
                      if col.startswith('percent_') and col.endswith('_positive')]

        # Focus on key immune populations
        key_immune = ['CD8_T_cells', 'CD4_T_cells', 'T_cells']
        infiltration_cols = [f'infiltration_{pop}_percent' for pop in key_immune
                            if f'infiltration_{pop}_percent' in tumor_data.columns]

        if len(marker_cols) == 0 or len(infiltration_cols) == 0:
            return

        for infiltration_col in infiltration_cols:
            immune_pop = infiltration_col.replace('infiltration_', '').replace('_percent', '')

            # Create infiltration quartiles
            tumor_data[f'{immune_pop}_quartile'] = pd.qcut(
                tumor_data[infiltration_col], q=4,
                labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'],
                duplicates='drop'
            )

            # Create subplots
            n_markers = len(marker_cols)
            n_cols = 3
            n_rows = int(np.ceil(n_markers / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            axes = axes.flatten() if n_markers > 1 else [axes]

            for idx, marker_col in enumerate(marker_cols):
                ax = axes[idx]
                marker_name = marker_col.replace('percent_', '').replace('_positive', '')

                # Prepare data
                plot_data = tumor_data[[f'{immune_pop}_quartile', marker_col]].dropna()

                if len(plot_data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'{marker_name}', fontsize=12, fontweight='bold')
                    continue

                # Box plot
                quartile_labels = ['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)']
                available_quartiles = [q for q in quartile_labels
                                      if q in plot_data[f'{immune_pop}_quartile'].values]

                bp = ax.boxplot([plot_data[plot_data[f'{immune_pop}_quartile'] == q][marker_col].values
                                for q in available_quartiles],
                               labels=available_quartiles,
                               patch_artist=True, showfliers=True, widths=0.6)

                # Color boxes with gradient
                colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(bp['boxes'])))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # Add statistics if requested
                if with_stats:
                    groups = [plot_data[plot_data[f'{immune_pop}_quartile'] == q][marker_col].values
                             for q in plot_data[f'{immune_pop}_quartile'].unique()]
                    groups = [g for g in groups if len(g) >= 3]

                    if len(groups) >= 2:
                        try:
                            stat, pval = stats.kruskal(*groups)
                            sig_text = self._format_pvalue(pval)
                            ax.text(0.98, 0.98, f'Kruskal-Wallis: {sig_text}',
                                   transform=ax.transAxes, ha='right', va='top',
                                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                                   fontsize=10, fontweight='bold')
                        except:
                            pass

                ax.set_xlabel(f'{immune_pop} Infiltration Quartile', fontsize=11, fontweight='bold')
                ax.set_ylabel(f'% {marker_name}+ Tumor Cells', fontsize=11, fontweight='bold')
                ax.set_title(f'{marker_name} vs {immune_pop} Infiltration',
                            fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

            # Hide unused subplots
            for idx in range(len(marker_cols), len(axes)):
                axes[idx].axis('off')

            stats_suffix = '_with_stats' if with_stats else '_no_stats'
            plt.suptitle(f'KPNT: {immune_pop} Infiltration vs Marker Positivity{" (with statistics)" if with_stats else ""}',
                        fontsize=16, fontweight='bold')
            plt.tight_layout()

            plot_path = self.plots_dir / f'infiltration_{immune_pop}_vs_markers{stats_suffix}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"    ✓ Saved: {plot_path.name}")

    def _format_pvalue(self, pval: float) -> str:
        """Format p-value with significance stars."""
        if np.isnan(pval):
            return 'ns'
        elif pval < 0.001:
            return 'p<0.001 ***'
        elif pval < 0.01:
            return 'p<0.01 **'
        elif pval < 0.05:
            return 'p<0.05 *'
        else:
            return f'p={pval:.3f} ns'

    def _save_results(self):
        """Save correlation results to CSV files."""
        for key, df in self.results.items():
            if isinstance(df, pd.DataFrame):
                output_path = self.output_dir / f'{key}.csv'
                df.to_csv(output_path, index=False)
                print(f"    ✓ Saved: {output_path.name}")
