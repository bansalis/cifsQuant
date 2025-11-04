"""
Population Dynamics Analysis
Analyze cell population counts/fractions over time between groups
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
from ..visualization.population_dynamics_plotter import PopulationDynamicsPlotter


class PopulationDynamics:
    """
    Analyze population dynamics over time.

    Key features:
    - Raw counts per sample
    - Density calculations
    - Fractional populations (subset/parent)
    - Temporal trends
    - Group comparisons
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        """
        Initialize population dynamics analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data with phenotypes and metadata
        config : dict
            Configuration dictionary
        output_dir : Path
            Output directory
        """
        self.adata = adata
        self.config = config['population_dynamics']
        self.stats_config = config.get('statistics', {})
        self.output_dir = Path(output_dir) / 'population_dynamics'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.results = {}

    def run(self):
        """Run complete population dynamics analysis."""
        print("\n" + "="*80)
        print("POPULATION DYNAMICS ANALYSIS")
        print("="*80)

        # Get populations to analyze
        populations = self.config.get('populations', [])
        print(f"\nAnalyzing {len(populations)} populations...")

        # Calculate metrics for each population
        for pop in populations:
            print(f"\n  Processing {pop}...")
            self._analyze_population(pop)

        # Calculate fractional populations
        if 'fractional_populations' in self.config:
            print("\n  Calculating fractional populations...")
            self._calculate_fractions()

        # Save results
        self._save_results()

        # Generate comprehensive plots
        print("\n  Generating comprehensive plots...")
        self._generate_plots()

        print("\n✓ Population dynamics analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
        print(f"  Plots saved to: {self.output_dir}/plots/")
        print("="*80 + "\n")

        return self.results

    def _analyze_population(self, population: str):
        """Analyze a single population."""
        # Check if population exists
        pop_col = f'is_{population}'
        if pop_col not in self.adata.obs.columns:
            warnings.warn(f"Population '{population}' not found, skipping")
            return

        # Calculate metrics per sample
        metrics_per_sample = []

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]

            # Count cells
            pop_mask = sample_data[pop_col]
            count = pop_mask.sum()
            total = len(sample_data)

            # Calculate density (cells per mm²)
            # Get spatial area for this sample
            coords = self.adata.obsm['spatial'][sample_mask.values]
            area_pixels = self._calculate_sample_area(coords)
            # Assume 1 pixel = 0.325 μm (common for 20x imaging)
            area_mm2 = area_pixels * (0.325e-3) ** 2  # Convert μm² to mm²
            density = count / area_mm2 if area_mm2 > 0 else 0

            # Get metadata
            meta_dict = {
                'sample_id': sample,
                'population': population,
                'count': int(count),
                'total_cells': int(total),
                'fraction_of_total': count / total if total > 0 else 0,
                'density_per_mm2': density
            }

            # Add metadata columns
            for col in ['timepoint', 'group', 'main_group', 'genotype', 'treatment']:
                if col in sample_data.columns:
                    meta_dict[col] = sample_data[col].iloc[0]

            metrics_per_sample.append(meta_dict)

        # Convert to DataFrame
        df = pd.DataFrame(metrics_per_sample)

        # Store results
        self.results[population] = df

    def _calculate_fractions(self):
        """Calculate fractional populations (subset/parent)."""
        fractional_pops = self.config.get('fractional_populations', {})

        for subset, parent in fractional_pops.items():
            if subset not in self.results or parent not in self.results:
                warnings.warn(f"Cannot calculate fraction for {subset}/{parent}, missing data")
                continue

            # Merge subset and parent data
            subset_df = self.results[subset][['sample_id', 'count']].rename(
                columns={'count': 'subset_count'}
            )
            parent_df = self.results[parent][['sample_id', 'count']].rename(
                columns={'count': 'parent_count'}
            )

            merged = subset_df.merge(parent_df, on='sample_id')

            # Calculate fraction
            merged['fraction'] = merged['subset_count'] / merged['parent_count']
            merged['fraction'] = merged['fraction'].fillna(0)  # Handle 0/0 as 0

            # Add metadata from subset
            merged = merged.merge(
                self.results[subset][['sample_id', 'timepoint', 'group', 'main_group']],
                on='sample_id',
                how='left'
            )

            # Store
            fraction_name = f'{subset}_fraction_of_{parent}'
            self.results[fraction_name] = merged

            print(f"    ✓ {fraction_name}")

    def _calculate_sample_area(self, coords: np.ndarray) -> float:
        """
        Calculate sample area from spatial coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Spatial coordinates (N x 2)

        Returns
        -------
        float
            Area in pixels²
        """
        if len(coords) < 3:
            return 0

        # Use bounding box as approximate area
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()

        area = x_range * y_range

        return area

    def _save_results(self):
        """Save all results to files."""
        # Save each population's data
        for name, df in self.results.items():
            output_path = self.output_dir / f'{name}_metrics.csv'
            df.to_csv(output_path, index=False)

        # Create summary table
        summary_rows = []
        for pop in self.config.get('populations', []):
            if pop in self.results:
                df = self.results[pop]
                summary_rows.append({
                    'population': pop,
                    'total_cells': df['count'].sum(),
                    'mean_per_sample': df['count'].mean(),
                    'median_per_sample': df['count'].median(),
                    'n_samples': len(df)
                })

        summary_df = pd.DataFrame(summary_rows)
        summary_path = self.output_dir / 'population_summary.csv'
        summary_df.to_csv(summary_path, index=False)

        print(f"\n  ✓ Saved {len(self.results)} population datasets")
        print(f"  ✓ Saved summary table")

    def _generate_plots(self):
        """Generate comprehensive plots for all populations."""
        # Initialize plotter
        try:
            plotter = PopulationDynamicsPlotter(self.output_dir, self.config)
        except Exception as e:
            print(f"  ⚠ Could not initialize plotter: {e}")
            return

        # Get comparison settings
        comparisons = self.config.get('comparisons', [])
        if not comparisons:
            print("  ⚠ No comparisons specified, skipping plots")
            return

        comparison = comparisons[0]  # Use first comparison
        group_col = 'main_group'
        groups = comparison.get('groups', ['KPT', 'KPNT'])

        # Plot each population - counts and density
        plot_count = 0
        for pop_name, pop_data in self.results.items():
            if 'fraction' in pop_name:
                # Skip fraction populations for now, plot them separately
                continue

            if 'count' not in pop_data.columns:
                continue

            try:
                # Plot counts
                plotter.plot_population_over_time(
                    pop_data, pop_name, value_col='count',
                    group_col=group_col, groups=groups
                )
                plot_count += 1

                # Plot density if available
                if 'density_per_mm2' in pop_data.columns:
                    plotter.plot_population_over_time(
                        pop_data, pop_name, value_col='density_per_mm2',
                        group_col=group_col, groups=groups
                    )
                    plot_count += 1

            except Exception as e:
                print(f"  ⚠ Error plotting {pop_name}: {e}")

        # Plot fractional populations
        for pop_name, pop_data in self.results.items():
            if 'fraction' not in pop_name:
                continue

            if 'fraction' not in pop_data.columns:
                continue

            try:
                plotter.plot_population_over_time(
                    pop_data, pop_name, value_col='fraction',
                    group_col=group_col, groups=groups
                )
                plot_count += 1
            except Exception as e:
                print(f"  ⚠ Error plotting {pop_name}: {e}")

        # Create overview plots
        try:
            # Count overview
            plotter.plot_all_populations_overview(
                self.results, value_col='count', group_col=group_col
            )
            plot_count += 1

            # Density overview if available
            has_density = any('density_per_mm2' in df.columns
                            for df in self.results.values())
            if has_density:
                plotter.plot_all_populations_overview(
                    self.results, value_col='density_per_mm2', group_col=group_col
                )
                plot_count += 1

        except Exception as e:
            print(f"  ⚠ Error creating overview plots: {e}")

        print(f"  ✓ Generated {plot_count} comprehensive plots")

    def get_population_data(self, population: str) -> pd.DataFrame:
        """Get data for a specific population."""
        return self.results.get(population, pd.DataFrame())
