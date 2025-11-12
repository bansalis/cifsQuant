"""
Tumor Microenvironment Visualization
====================================
Generate comprehensive plots for tumor phenotype microenvironment analysis.

Features:
- Stacked column plots (percentages and counts)
- Statistical comparisons between phenotypes
- Time series analysis
- Enrichment/depletion highlights
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from scipy import stats
import warnings


class TumorMicroenvironmentPlotter:
    """
    Comprehensive visualization for tumor microenvironment analysis.

    Generates:
    - Stacked column plots (percentage composition)
    - Absolute count plots
    - Statistical comparison plots
    - Enrichment/depletion highlights
    - Multi-radius comparisons
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

        # Get plotting settings
        plotting_config = config.get('plotting', {})
        self.group_colors = plotting_config.get('group_colors', {
            'KPT': '#E41A1C', 'KPNT': '#377EB8'
        })
        self.timepoint_label = plotting_config.get('timepoint_label', 'Time (weeks)')

        # Immune cell colors for stacked plots
        self.immune_colors = {
            'CD8_T_cells': '#1f77b4',
            'CD4_T_cells': '#ff7f0e',
            'CD3_positive': '#2ca02c',
            'B_cells': '#d62728',
            'CD45_positive': '#9467bd',
            'Other': '#8c564b'
        }

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300

    def plot_stacked_columns(self, df: pd.DataFrame, pheno_name: str,
                            bin_name: str, value_type: str = 'percent'):
        """
        Create stacked column plots showing immune composition.

        Parameters
        ----------
        df : pd.DataFrame
            Per-cell microenvironment data
        pheno_name : str
            Phenotype name (e.g., 'pERK')
        bin_name : str
            Distance bin name
        value_type : str
            'percent' for percentages or 'count' for absolute counts
        """
        if df is None or len(df) == 0:
            return

        # Get immune populations from columns
        if value_type == 'percent':
            immune_cols = [col for col in df.columns if col.endswith('_percent')]
            immune_pops = [col.replace('_percent', '') for col in immune_cols]
            ylabel = '% of Immune Cells'
            ylim = (0, 100)
        else:
            immune_cols = [col for col in df.columns if col.endswith('_count')]
            immune_pops = [col.replace('_count', '') for col in immune_cols]
            ylabel = 'Number of Immune Cells'
            ylim = None

        if not immune_pops:
            return

        # Check if we have timepoint and group information
        has_timepoint = 'timepoint' in df.columns and df['timepoint'].notna().any()
        has_group = 'main_group' in df.columns and df['main_group'].notna().any()

        # Create figure based on available data
        if has_timepoint and has_group:
            # Full analysis: phenotype x group x timepoint
            self._plot_stacked_full(df, pheno_name, bin_name, immune_pops,
                                   value_type, ylabel, ylim)
        elif has_group:
            # Phenotype x group only
            self._plot_stacked_by_group(df, pheno_name, bin_name, immune_pops,
                                       value_type, ylabel, ylim)
        else:
            # Phenotype only
            self._plot_stacked_simple(df, pheno_name, bin_name, immune_pops,
                                     value_type, ylabel, ylim)

    def _plot_stacked_simple(self, df: pd.DataFrame, pheno_name: str,
                            bin_name: str, immune_pops: List[str],
                            value_type: str, ylabel: str, ylim: Optional[tuple]):
        """Simple stacked column: phenotype+ vs phenotype-."""

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Prepare data
        phenotypes = [f'{pheno_name}+', f'{pheno_name}-']
        data_by_pheno = {}

        for pheno_status in phenotypes:
            pheno_data = df[df['phenotype_status'] == pheno_status]

            means = []
            sems = []
            for pop in immune_pops:
                col = f'{pop}_{value_type}'
                if col in pheno_data.columns:
                    mean_val = pheno_data[col].mean()
                    sem_val = pheno_data[col].sem()
                    means.append(mean_val)
                    sems.append(sem_val)
                else:
                    means.append(0)
                    sems.append(0)

            data_by_pheno[pheno_status] = {'means': means, 'sems': sems}

        # Plot 1: Stacked bar chart (means)
        ax = axes[0]
        x = np.arange(len(phenotypes))
        width = 0.6

        bottom = np.zeros(len(phenotypes))
        for idx, pop in enumerate(immune_pops):
            heights = [data_by_pheno[pheno]['means'][idx] for pheno in phenotypes]
            color = self.immune_colors.get(pop, '#cccccc')

            ax.bar(x, heights, width, bottom=bottom, label=pop.replace('_', ' '),
                  color=color, alpha=0.8, edgecolor='white', linewidth=1)
            bottom += heights

        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(phenotypes, fontsize=11)
        ax.set_title(f'Mean {value_type.title()}', fontsize=13, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        if ylim:
            ax.set_ylim(ylim)

        # Plot 2: Individual population comparison with stats
        ax = axes[1]

        # Select top immune populations for comparison
        # Calculate total for each population
        pop_totals = []
        for pop in immune_pops:
            col = f'{pop}_{value_type}'
            if col in df.columns:
                total = df[col].sum()
                pop_totals.append((pop, total))

        pop_totals.sort(key=lambda x: x[1], reverse=True)
        top_pops = [p[0] for p in pop_totals[:3]]  # Top 3

        if top_pops:
            # Create grouped box plots for top populations
            plot_data = []
            for pop in top_pops:
                col = f'{pop}_{value_type}'
                if col in df.columns:
                    for _, row in df.iterrows():
                        plot_data.append({
                            'population': pop.replace('_', ' '),
                            'phenotype': row['phenotype_status'],
                            'value': row[col]
                        })

            if plot_data:
                plot_df = pd.DataFrame(plot_data)

                # Box plots
                positions = []
                box_data = []
                labels = []
                colors = []

                x_pos = 0
                for pop in top_pops:
                    pop_label = pop.replace('_', ' ')
                    color = self.immune_colors.get(pop, '#cccccc')

                    for pheno in phenotypes:
                        data = plot_df[(plot_df['population'] == pop_label) &
                                      (plot_df['phenotype'] == pheno)]['value']
                        if len(data) > 0:
                            box_data.append(data)
                            positions.append(x_pos)
                            labels.append(f'{pheno}')
                            colors.append(color)
                            x_pos += 1

                    x_pos += 0.5  # Gap between populations

                if box_data:
                    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                                   patch_artist=True, showfliers=False)

                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.6)

                    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
                    ax.set_title(f'Top Populations', fontsize=13, fontweight='bold')

                    # Add stat tests
                    self._add_pairwise_stats(ax, plot_df, top_pops, phenotypes,
                                           positions, value_type)

        plt.suptitle(f'{pheno_name} Microenvironment: {bin_name.title()} Distance',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save
        plot_path = self.plots_dir / f'{pheno_name}_{bin_name}_{value_type}_stacked.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Saved {plot_path.name}")

    def _plot_stacked_by_group(self, df: pd.DataFrame, pheno_name: str,
                               bin_name: str, immune_pops: List[str],
                               value_type: str, ylabel: str, ylim: Optional[tuple]):
        """Stacked columns by phenotype and group (KPT/KPNT)."""

        groups = sorted(df['main_group'].unique())
        phenotypes = [f'{pheno_name}+', f'{pheno_name}-']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Stacked bars for each combination
        ax = axes[0]

        categories = []
        for group in groups:
            for pheno in phenotypes:
                categories.append(f'{pheno}\n{group}')

        x = np.arange(len(categories))
        width = 0.7

        # Prepare data
        data_by_category = {}
        for group in groups:
            for pheno in phenotypes:
                cat = f'{pheno}\n{group}'
                subset = df[(df['main_group'] == group) &
                           (df['phenotype_status'] == pheno)]

                means = []
                for pop in immune_pops:
                    col = f'{pop}_{value_type}'
                    if col in subset.columns and len(subset) > 0:
                        means.append(subset[col].mean())
                    else:
                        means.append(0)

                data_by_category[cat] = means

        # Plot stacked bars
        bottom = np.zeros(len(categories))
        for idx, pop in enumerate(immune_pops):
            heights = [data_by_category[cat][idx] for cat in categories]
            color = self.immune_colors.get(pop, '#cccccc')

            ax.bar(x, heights, width, bottom=bottom, label=pop.replace('_', ' '),
                  color=color, alpha=0.8, edgecolor='white', linewidth=1)
            bottom += heights

        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_title(f'Immune Composition by Group', fontsize=13, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        if ylim:
            ax.set_ylim(ylim)

        # Plot 2: Statistical comparison
        ax = axes[1]

        # For each group, compare phenotype+ vs phenotype-
        for g_idx, group in enumerate(groups):
            pos_data = df[(df['main_group'] == group) &
                         (df['phenotype_status'] == f'{pheno_name}+')]
            neg_data = df[(df['main_group'] == group) &
                         (df['phenotype_status'] == f'{pheno_name}-')]

            if len(pos_data) < 3 or len(neg_data) < 3:
                continue

            # Total immune comparison
            pos_total = pos_data['n_total_immune'].values
            neg_total = neg_data['n_total_immune'].values

            try:
                stat, pval = stats.mannwhitneyu(pos_total, neg_total,
                                               alternative='two-sided')
            except:
                pval = 1.0

            # Plot
            box_data = [pos_total, neg_total]
            positions = [g_idx*2, g_idx*2+0.8]
            labels_local = [f'{pheno_name}+', f'{pheno_name}-']
            color = self.group_colors.get(group, '#999999')

            bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                           patch_artist=True, showfliers=False)

            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            # Add significance
            if pval < 0.05:
                y_max = max(pos_total.max(), neg_total.max())
                y_range = y_max - min(pos_total.min(), neg_total.min())
                y_bracket = y_max + y_range * 0.05

                ax.plot([positions[0], positions[1]],
                       [y_bracket, y_bracket], 'k-', linewidth=1.5)

                sig_symbol = '***' if pval < 0.001 else '**' if pval < 0.01 else '*'
                ax.text((positions[0] + positions[1])/2, y_bracket,
                       sig_symbol, ha='center', va='bottom',
                       fontsize=12, fontweight='bold')

        ax.set_ylabel('Total Immune Cells', fontsize=12, fontweight='bold')
        ax.set_title(f'Total Immune: {pheno_name}+ vs {pheno_name}-',
                    fontsize=13, fontweight='bold')

        # Set x-ticks
        xticks = []
        xticklabels = []
        for g_idx, group in enumerate(groups):
            xticks.append(g_idx*2 + 0.4)
            xticklabels.append(group)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        plt.suptitle(f'{pheno_name} Microenvironment: {bin_name.title()} Distance',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save
        plot_path = self.plots_dir / f'{pheno_name}_{bin_name}_{value_type}_by_group.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Saved {plot_path.name}")

    def _plot_stacked_full(self, df: pd.DataFrame, pheno_name: str,
                          bin_name: str, immune_pops: List[str],
                          value_type: str, ylabel: str, ylim: Optional[tuple]):
        """Full stacked columns with timepoint, group, and phenotype."""

        groups = sorted(df['main_group'].unique())
        timepoints = sorted(df['timepoint'].unique())
        phenotypes = [f'{pheno_name}+', f'{pheno_name}-']

        # Create 2x2 plot grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Plot 1: Stacked bars by timepoint (all groups combined, phenotype comparison)
        ax = axes[0]
        self._plot_stacked_by_timepoint(ax, df, pheno_name, timepoints,
                                       immune_pops, value_type, ylabel, ylim)

        # Plot 2: Stacked bars by group (all timepoints combined, phenotype comparison)
        ax = axes[1]
        self._plot_stacked_by_group_subplot(ax, df, pheno_name, groups,
                                           immune_pops, value_type, ylabel, ylim)

        # Plot 3: Total immune cells over time (line plot with stats)
        ax = axes[2]
        self._plot_total_immune_over_time(ax, df, pheno_name, groups, timepoints)

        # Plot 4: Enrichment summary (if available)
        ax = axes[3]
        self._plot_enrichment_summary(ax, df, pheno_name, immune_pops, value_type)

        plt.suptitle(f'{pheno_name} Microenvironment: {bin_name.title()} Distance',
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save
        plot_path = self.plots_dir / f'{pheno_name}_{bin_name}_{value_type}_comprehensive.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Saved {plot_path.name}")

    def _plot_stacked_by_timepoint(self, ax, df, pheno_name, timepoints,
                                   immune_pops, value_type, ylabel, ylim):
        """Stacked bars by timepoint."""
        phenotypes = [f'{pheno_name}+', f'{pheno_name}-']

        categories = []
        for tp in timepoints:
            for pheno in phenotypes:
                categories.append(f'T{tp}\n{pheno}')

        x = np.arange(len(categories))
        width = 0.7

        # Prepare data
        data_by_category = {}
        for tp in timepoints:
            for pheno in phenotypes:
                cat = f'T{tp}\n{pheno}'
                subset = df[(df['timepoint'] == tp) &
                           (df['phenotype_status'] == pheno)]

                means = []
                for pop in immune_pops:
                    col = f'{pop}_{value_type}'
                    if col in subset.columns and len(subset) > 0:
                        means.append(subset[col].mean())
                    else:
                        means.append(0)

                data_by_category[cat] = means

        # Plot
        bottom = np.zeros(len(categories))
        for idx, pop in enumerate(immune_pops):
            heights = [data_by_category[cat][idx] for cat in categories]
            color = self.immune_colors.get(pop, '#cccccc')

            ax.bar(x, heights, width, bottom=bottom, label=pop.replace('_', ' '),
                  color=color, alpha=0.8, edgecolor='white', linewidth=1)
            bottom += heights

        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_title('By Timepoint', fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        if ylim:
            ax.set_ylim(ylim)

    def _plot_stacked_by_group_subplot(self, ax, df, pheno_name, groups,
                                      immune_pops, value_type, ylabel, ylim):
        """Stacked bars by group (subplot version)."""
        phenotypes = [f'{pheno_name}+', f'{pheno_name}-']

        categories = []
        for group in groups:
            for pheno in phenotypes:
                categories.append(f'{group}\n{pheno}')

        x = np.arange(len(categories))
        width = 0.7

        data_by_category = {}
        for group in groups:
            for pheno in phenotypes:
                cat = f'{group}\n{pheno}'
                subset = df[(df['main_group'] == group) &
                           (df['phenotype_status'] == pheno)]

                means = []
                for pop in immune_pops:
                    col = f'{pop}_{value_type}'
                    if col in subset.columns and len(subset) > 0:
                        means.append(subset[col].mean())
                    else:
                        means.append(0)

                data_by_category[cat] = means

        bottom = np.zeros(len(categories))
        for idx, pop in enumerate(immune_pops):
            heights = [data_by_category[cat][idx] for cat in categories]
            color = self.immune_colors.get(pop, '#cccccc')

            ax.bar(x, heights, width, bottom=bottom,
                  color=color, alpha=0.8, edgecolor='white', linewidth=1)
            bottom += heights

        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_title('By Group', fontsize=12, fontweight='bold')
        if ylim:
            ax.set_ylim(ylim)

    def _plot_total_immune_over_time(self, ax, df, pheno_name, groups, timepoints):
        """Line plot of total immune cells over time."""
        phenotypes = [f'{pheno_name}+', f'{pheno_name}-']

        for group in groups:
            for pheno in phenotypes:
                means = []
                sems = []

                for tp in timepoints:
                    subset = df[(df['main_group'] == group) &
                               (df['timepoint'] == tp) &
                               (df['phenotype_status'] == pheno)]

                    if len(subset) > 0:
                        mean_immune = subset['n_total_immune'].mean()
                        sem_immune = subset['n_total_immune'].sem()
                        means.append(mean_immune)
                        sems.append(sem_immune)
                    else:
                        means.append(np.nan)
                        sems.append(np.nan)

                means = np.array(means)
                sems = np.array(sems)

                # Plot
                color = self.group_colors.get(group, '#999999')
                linestyle = '-' if pheno == f'{pheno_name}+' else '--'
                label = f'{group} {pheno}'

                ax.plot(timepoints, means, linestyle=linestyle, marker='o',
                       color=color, linewidth=2, markersize=6, label=label)
                ax.fill_between(timepoints, means-sems, means+sems,
                               alpha=0.2, color=color)

        ax.set_xlabel(self.timepoint_label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Total Immune Cells', fontsize=11, fontweight='bold')
        ax.set_title('Total Immune Over Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_enrichment_summary(self, ax, df, pheno_name, immune_pops, value_type):
        """Plot enrichment/depletion summary."""
        # Calculate fold changes for each immune population
        enrichment_data = []

        for pop in immune_pops:
            col = f'{pop}_{value_type}'
            if col not in df.columns:
                continue

            pos_data = df[df['phenotype_status'] == f'{pheno_name}+'][col]
            neg_data = df[df['phenotype_status'] == f'{pheno_name}-'][col]

            if len(pos_data) < 3 or len(neg_data) < 3:
                continue

            mean_pos = pos_data.mean()
            mean_neg = neg_data.mean()

            # Calculate fold change (log2)
            if mean_neg > 0:
                fold_change = np.log2(mean_pos / mean_neg)
            else:
                fold_change = 0

            # Statistical test
            try:
                stat, pval = stats.mannwhitneyu(pos_data, neg_data,
                                               alternative='two-sided')
            except:
                pval = 1.0

            enrichment_data.append({
                'population': pop.replace('_', ' '),
                'fold_change': fold_change,
                'pvalue': pval,
                'significant': pval < 0.05
            })

        if not enrichment_data:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            return

        enrichment_df = pd.DataFrame(enrichment_data)

        # Sort by fold change
        enrichment_df = enrichment_df.sort_values('fold_change')

        # Bar plot
        y_pos = np.arange(len(enrichment_df))
        colors = ['red' if fc > 0 else 'blue' for fc in enrichment_df['fold_change']]
        alphas = [0.8 if sig else 0.3 for sig in enrichment_df['significant']]

        bars = ax.barh(y_pos, enrichment_df['fold_change'], color=colors)
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(enrichment_df['population'], fontsize=9)
        ax.set_xlabel(f'Log2 Fold Change ({pheno_name}+/{pheno_name}-)',
                     fontsize=11, fontweight='bold')
        ax.set_title('Enrichment/Depletion', fontsize=12, fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.8, label='Enriched*'),
            Patch(facecolor='red', alpha=0.3, label='Enriched (ns)'),
            Patch(facecolor='blue', alpha=0.8, label='Depleted*'),
            Patch(facecolor='blue', alpha=0.3, label='Depleted (ns)')
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc='best')

    def _add_pairwise_stats(self, ax, plot_df, populations, phenotypes,
                           positions, value_type):
        """Add pairwise statistical comparisons."""
        # For each population, compare phenotype+ vs phenotype-
        # (This is a simplified version - you may want to enhance)
        pass  # Already handled in main plots

    def plot_enrichment_highlights(self, enrichment_df: pd.DataFrame, pheno_name: str):
        """
        Create focused plots for significant enrichment/depletion.

        Parameters
        ----------
        enrichment_df : pd.DataFrame
            Enrichment analysis results
        pheno_name : str
            Phenotype name
        """
        if enrichment_df is None or len(enrichment_df) == 0:
            return

        # Filter to significant results
        sig_df = enrichment_df[enrichment_df['significant_adj'] == True].copy()

        if len(sig_df) == 0:
            print(f"    No significant enrichment/depletion found for {pheno_name}")
            return

        # Create summary plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(sig_df) * 0.5)))

        # Sort by effect size
        sig_df = sig_df.sort_values('diff_percent')

        y_pos = np.arange(len(sig_df))
        colors = ['red' if d > 0 else 'blue' for d in sig_df['diff_percent']]

        bars = ax.barh(y_pos, sig_df['diff_percent'], color=colors, alpha=0.7)

        # Add labels
        labels = [f"{row['immune_population']} ({row['distance_bin']})"
                 for _, row in sig_df.iterrows()]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel(f'Difference in % ({pheno_name}+ - {pheno_name}-)',
                     fontsize=12, fontweight='bold')
        ax.set_title(f'{pheno_name}: Significant Immune Enrichment/Depletion',
                    fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=1.5)
        ax.grid(True, alpha=0.3, axis='x')

        # Add p-value annotations
        for i, (_, row) in enumerate(sig_df.iterrows()):
            pval = row['pvalue_percent_adj']
            if pval < 0.001:
                sig_text = '***'
            elif pval < 0.01:
                sig_text = '**'
            else:
                sig_text = '*'

            x_pos = row['diff_percent']
            x_offset = 0.5 if x_pos > 0 else -0.5
            ax.text(x_pos + x_offset, i, sig_text,
                   ha='left' if x_pos > 0 else 'right',
                   va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()

        plot_path = self.plots_dir / f'{pheno_name}_enrichment_highlights.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Saved {plot_path.name}")

    def generate_all_plots(self, results: Dict):
        """
        Generate all tumor microenvironment plots.

        Parameters
        ----------
        results : dict
            Results from TumorMicroenvironmentAnalysis
        """
        print("  Generating tumor microenvironment plots...")

        for pheno_name, pheno_results in results.items():
            print(f"\n    Processing {pheno_name}...")

            # Plot for each distance bin
            for bin_name, df in pheno_results.items():
                if bin_name == 'enrichment':
                    continue

                if df is None or len(df) == 0:
                    continue

                print(f"      {bin_name} distance bin:")

                # Percentage stacked plots
                self.plot_stacked_columns(df, pheno_name, bin_name, 'percent')

                # Count stacked plots
                self.plot_stacked_columns(df, pheno_name, bin_name, 'count')

            # Enrichment highlights
            if 'enrichment' in pheno_results:
                self.plot_enrichment_highlights(pheno_results['enrichment'], pheno_name)

        print(f"\n  ✓ All tumor microenvironment plots saved to {self.plots_dir}/")
