"""
Spatial Overlap Analysis
Analyze spatial region overlap between marker-defined regions using SpatialCells
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings

try:
    import spatialcells as spc
    HAS_SPATIALCELLS = True
except ImportError:
    HAS_SPATIALCELLS = False


class SpatialOverlapAnalysis:
    """
    Analyze spatial overlap between marker-defined regions.

    Key features:
    - Detect spatial regions for each phenotype using SpatialCells
    - Calculate overlap metrics (Jaccard index, overlap area, etc.)
    - Analyze region overlap even when cells aren't double positive
    - Generate spatial plots showing overlapping regions
    - Export comprehensive overlap statistics
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        """
        Initialize spatial overlap analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data with phenotypes and spatial coordinates
        config : dict
            Configuration dictionary
        output_dir : Path
            Output directory
        """
        if not HAS_SPATIALCELLS:
            raise ImportError("SpatialCells required. Install with: pip install spatialcells")

        self.adata = adata
        self.config = config
        self.output_dir = Path(output_dir) / 'spatial_overlap_analysis'
        self.plots_dir = self.output_dir / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Prepare coordinate columns for SpatialCells
        if 'spatial' in self.adata.obsm:
            self.adata.obs['X_centroid'] = self.adata.obsm['spatial'][:, 0]
            self.adata.obs['Y_centroid'] = self.adata.obsm['spatial'][:, 1]

        # Get phenotypes from config
        self.phenotypes = self._extract_phenotypes_from_config()
        print(f"  Found {len(self.phenotypes)} phenotypes in config")

        # Storage
        self.results = {}
        self.region_boundaries = {}  # sample -> phenotype -> {region_id: boundary}

    def _extract_phenotypes_from_config(self) -> List[str]:
        """Extract all phenotype names from config."""
        phenotypes = []

        if 'phenotypes' in self.config:
            phenotypes = list(self.config['phenotypes'].keys())

        return phenotypes

    def run(self) -> Dict:
        """Run complete spatial overlap analysis."""
        print("\n" + "="*80)
        print("SPATIAL OVERLAP ANALYSIS")
        print("="*80)

        if len(self.phenotypes) < 2:
            print("  ⚠ Need at least 2 phenotypes for overlap analysis")
            return {}

        # Detect spatial regions for each phenotype
        print("\n1. Detecting spatial regions for each phenotype...")
        self._detect_phenotype_regions()

        # Calculate pairwise spatial overlap
        print("\n2. Calculating pairwise spatial overlap...")
        self._calculate_pairwise_overlap()

        # Calculate multi-region overlap
        print("\n3. Calculating multi-region overlap...")
        self._calculate_multi_region_overlap()

        # Analyze overlap vs coexpression
        print("\n4. Analyzing spatial overlap vs cellular coexpression...")
        self._analyze_overlap_vs_coexpression()

        # Save results
        print("\n5. Saving results...")
        self._save_results()

        # Generate plots
        print("\n6. Generating visualizations...")
        self._generate_plots()

        print("\n✓ Spatial overlap analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
        print(f"  Plots saved to: {self.plots_dir}/")
        print("="*80 + "\n")

        return self.results

    def _detect_phenotype_regions(self):
        """Detect spatial regions for each phenotype using SpatialCells."""
        # Get detection parameters from config or use defaults
        detection_config = self.config.get('spatial_overlap_analysis', {}).get('region_detection', {})
        eps = detection_config.get('eps', 50)
        min_samples = detection_config.get('min_samples', 10)
        alpha = detection_config.get('alpha', 50)
        core_only = detection_config.get('core_only', True)
        min_area = detection_config.get('min_area', 5000)
        min_edges = detection_config.get('min_edges', 15)

        print(f"  Detection parameters: eps={eps}, alpha={alpha}, min_samples={min_samples}")

        all_region_info = []

        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_adata = self.adata[sample_mask].copy()

            if sample not in self.region_boundaries:
                self.region_boundaries[sample] = {}

            for phenotype in self.phenotypes:
                pheno_col = f'is_{phenotype}'

                if pheno_col not in sample_adata.obs.columns:
                    continue

                n_pheno_cells = sample_adata.obs[pheno_col].sum()

                if n_pheno_cells < min_samples:
                    continue

                try:
                    # Detect communities for this phenotype
                    community_col = f'{phenotype}_community'
                    community_results = spc.spatial.getCommunities(
                        sample_adata,
                        [pheno_col],
                        eps=eps,
                        min_samples=min_samples,
                        newcolumn=community_col,
                        core_only=core_only
                    )

                    if community_results is None:
                        continue

                    labels_sorted, db = community_results
                    n_communities = len(labels_sorted)

                    if n_communities == 0:
                        continue

                    # Get community indices
                    community_indices = [idx for _, idx in labels_sorted]

                    # Create boundaries
                    boundaries = spc.spatial.getBoundary(
                        sample_adata,
                        community_col,
                        community_indices,
                        alpha=alpha,
                        debug=False
                    )

                    # Prune small components
                    pruned_boundaries = spc.spatial.pruneSmallComponents(
                        boundaries,
                        min_area=min_area,
                        min_edges=min_edges,
                        holes_min_area=10000,
                        holes_min_edges=10
                    )

                    # Get individual components
                    boundary_components = spc.spa.getComponents(pruned_boundaries)
                    n_components = len(boundary_components)

                    # Store boundaries
                    if phenotype not in self.region_boundaries[sample]:
                        self.region_boundaries[sample][phenotype] = {}

                    for component_idx, boundary in enumerate(boundary_components):
                        area = spc.msmt.getRegionArea(boundary)
                        centroid = spc.msmt.getRegionCentroid(boundary)

                        self.region_boundaries[sample][phenotype][component_idx] = boundary

                        all_region_info.append({
                            'sample_id': sample,
                            'phenotype': phenotype,
                            'region_id': component_idx,
                            'n_communities': n_communities,
                            'n_components': n_components,
                            'area_um2': area,
                            'centroid_x': centroid[0],
                            'centroid_y': centroid[1],
                            'n_cells': n_pheno_cells,
                            'timepoint': sample_adata.obs['timepoint'].iloc[0] if 'timepoint' in sample_adata.obs.columns else np.nan,
                            'group': sample_adata.obs['group'].iloc[0] if 'group' in sample_adata.obs.columns else '',
                            'main_group': sample_adata.obs['main_group'].iloc[0] if 'main_group' in sample_adata.obs.columns else ''
                        })

                    print(f"    ✓ {sample}: {phenotype} - {n_communities} communities, {n_components} regions")

                except Exception as e:
                    warnings.warn(f"Error detecting regions for {sample} {phenotype}: {e}")
                    continue

        if all_region_info:
            df = pd.DataFrame(all_region_info)
            self.results['detected_regions'] = df
            print(f"  ✓ Detected {len(all_region_info)} spatial regions total")

    def _calculate_pairwise_overlap(self):
        """Calculate pairwise spatial overlap between phenotype regions."""
        overlap_results = []

        # Generate all phenotype pairs
        pheno_pairs = list(combinations(self.phenotypes, 2))
        print(f"  Analyzing {len(pheno_pairs)} phenotype pairs...")

        for sample, pheno_dict in self.region_boundaries.items():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_adata = self.adata[sample_mask].copy()

            for pheno1, pheno2 in pheno_pairs:
                if pheno1 not in pheno_dict or pheno2 not in pheno_dict:
                    continue

                regions1 = pheno_dict[pheno1]
                regions2 = pheno_dict[pheno2]

                # Calculate overlap for all region pairs
                for region1_id, boundary1 in regions1.items():
                    for region2_id, boundary2 in regions2.items():
                        try:
                            # Calculate intersection
                            intersection = boundary1.intersection(boundary2)
                            intersection_area = spc.msmt.getRegionArea(intersection)

                            # Calculate union
                            union = boundary1.union(boundary2)
                            union_area = spc.msmt.getRegionArea(union)

                            # Calculate areas
                            area1 = spc.msmt.getRegionArea(boundary1)
                            area2 = spc.msmt.getRegionArea(boundary2)

                            # Calculate metrics
                            jaccard_index = intersection_area / union_area if union_area > 0 else 0
                            overlap_coefficient = intersection_area / min(area1, area2) if min(area1, area2) > 0 else 0
                            dice_coefficient = (2 * intersection_area) / (area1 + area2) if (area1 + area2) > 0 else 0

                            # Percent overlap
                            percent_of_region1 = intersection_area / area1 * 100 if area1 > 0 else 0
                            percent_of_region2 = intersection_area / area2 * 100 if area2 > 0 else 0

                            # Check if overlapping
                            is_overlapping = intersection_area > 0

                            overlap_results.append({
                                'sample_id': sample,
                                'phenotype1': pheno1,
                                'phenotype2': pheno2,
                                'region1_id': region1_id,
                                'region2_id': region2_id,
                                'is_overlapping': is_overlapping,
                                'intersection_area_um2': intersection_area,
                                'union_area_um2': union_area,
                                'region1_area_um2': area1,
                                'region2_area_um2': area2,
                                'jaccard_index': jaccard_index,
                                'overlap_coefficient': overlap_coefficient,
                                'dice_coefficient': dice_coefficient,
                                'percent_of_region1': percent_of_region1,
                                'percent_of_region2': percent_of_region2,
                                'timepoint': sample_adata.obs['timepoint'].iloc[0] if 'timepoint' in sample_adata.obs.columns else np.nan,
                                'group': sample_adata.obs['group'].iloc[0] if 'group' in sample_adata.obs.columns else '',
                                'main_group': sample_adata.obs['main_group'].iloc[0] if 'main_group' in sample_adata.obs.columns else ''
                            })

                        except Exception as e:
                            warnings.warn(f"Error calculating overlap for {sample} {pheno1}-{pheno2}: {e}")
                            continue

        if overlap_results:
            df = pd.DataFrame(overlap_results)
            self.results['pairwise_overlap'] = df
            print(f"  ✓ Calculated {len(overlap_results)} pairwise region overlaps")

            # Calculate summary statistics per phenotype pair
            self._calculate_overlap_summary(df)

    def _calculate_overlap_summary(self, overlap_df: pd.DataFrame):
        """Calculate summary statistics for pairwise overlaps."""
        summary_results = []

        for (sample, pheno1, pheno2), group_df in overlap_df.groupby(['sample_id', 'phenotype1', 'phenotype2']):
            # Count overlapping vs non-overlapping region pairs
            n_total_pairs = len(group_df)
            n_overlapping = group_df['is_overlapping'].sum()
            n_non_overlapping = n_total_pairs - n_overlapping

            # Calculate mean overlap metrics (only for overlapping pairs)
            overlapping_pairs = group_df[group_df['is_overlapping']]

            if len(overlapping_pairs) > 0:
                mean_jaccard = overlapping_pairs['jaccard_index'].mean()
                mean_overlap_coef = overlapping_pairs['overlap_coefficient'].mean()
                mean_dice = overlapping_pairs['dice_coefficient'].mean()
                total_intersection_area = overlapping_pairs['intersection_area_um2'].sum()
            else:
                mean_jaccard = 0
                mean_overlap_coef = 0
                mean_dice = 0
                total_intersection_area = 0

            summary_results.append({
                'sample_id': sample,
                'phenotype1': pheno1,
                'phenotype2': pheno2,
                'n_region_pairs': n_total_pairs,
                'n_overlapping_pairs': int(n_overlapping),
                'n_non_overlapping_pairs': int(n_non_overlapping),
                'percent_overlapping': n_overlapping / n_total_pairs * 100 if n_total_pairs > 0 else 0,
                'mean_jaccard_index': mean_jaccard,
                'mean_overlap_coefficient': mean_overlap_coef,
                'mean_dice_coefficient': mean_dice,
                'total_intersection_area_um2': total_intersection_area,
                'timepoint': group_df['timepoint'].iloc[0] if 'timepoint' in group_df.columns else np.nan,
                'group': group_df['group'].iloc[0] if 'group' in group_df.columns else '',
                'main_group': group_df['main_group'].iloc[0] if 'main_group' in group_df.columns else ''
            })

        if summary_results:
            df = pd.DataFrame(summary_results)
            self.results['pairwise_overlap_summary'] = df
            print(f"  ✓ Generated overlap summary for {len(summary_results)} sample-phenotype pairs")

    def _calculate_multi_region_overlap(self):
        """Calculate overlap for 3+ phenotype regions."""
        multi_overlap_results = []

        # Limit to prevent combinatorial explosion
        max_phenos = min(5, len(self.phenotypes))
        phenos_subset = self.phenotypes[:max_phenos]

        # Generate triplets
        pheno_triplets = list(combinations(phenos_subset, 3))
        print(f"  Analyzing {len(pheno_triplets)} triplet overlaps...")

        for sample, pheno_dict in self.region_boundaries.items():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_adata = self.adata[sample_mask].copy()

            for phenos in pheno_triplets:
                # Check all phenotypes present
                if not all(p in pheno_dict for p in phenos):
                    continue

                # Get all region combinations
                region_sets = [list(pheno_dict[p].items()) for p in phenos]

                # For simplicity, calculate overlap for largest region of each phenotype
                if all(len(rs) > 0 for rs in region_sets):
                    # Get largest region for each phenotype
                    boundaries = []
                    region_ids = []

                    for pheno, region_set in zip(phenos, region_sets):
                        # Find largest region
                        largest_region_id = max(region_set, key=lambda x: spc.msmt.getRegionArea(x[1]))[0]
                        boundaries.append(pheno_dict[pheno][largest_region_id])
                        region_ids.append(largest_region_id)

                    try:
                        # Calculate intersection of all three
                        intersection = boundaries[0]
                        for boundary in boundaries[1:]:
                            intersection = intersection.intersection(boundary)

                        intersection_area = spc.msmt.getRegionArea(intersection)

                        # Calculate union
                        union = boundaries[0]
                        for boundary in boundaries[1:]:
                            union = union.union(boundary)

                        union_area = spc.msmt.getRegionArea(union)

                        # Calculate Jaccard index
                        jaccard_index = intersection_area / union_area if union_area > 0 else 0

                        multi_overlap_results.append({
                            'sample_id': sample,
                            'phenotypes': '_AND_'.join(phenos),
                            'region_ids': '_'.join(map(str, region_ids)),
                            'intersection_area_um2': intersection_area,
                            'union_area_um2': union_area,
                            'jaccard_index': jaccard_index,
                            'is_overlapping': intersection_area > 0,
                            'timepoint': sample_adata.obs['timepoint'].iloc[0] if 'timepoint' in sample_adata.obs.columns else np.nan,
                            'group': sample_adata.obs['group'].iloc[0] if 'group' in sample_adata.obs.columns else '',
                            'main_group': sample_adata.obs['main_group'].iloc[0] if 'main_group' in sample_adata.obs.columns else ''
                        })

                    except Exception as e:
                        warnings.warn(f"Error calculating multi-region overlap: {e}")
                        continue

        if multi_overlap_results:
            df = pd.DataFrame(multi_overlap_results)
            self.results['multi_region_overlap'] = df
            print(f"  ✓ Calculated {len(multi_overlap_results)} multi-region overlaps")

    def _analyze_overlap_vs_coexpression(self):
        """Compare spatial overlap with cellular coexpression."""
        if 'pairwise_overlap_summary' not in self.results:
            return

        overlap_df = self.results['pairwise_overlap_summary']
        comparison_results = []

        for _, row in overlap_df.iterrows():
            sample = row['sample_id']
            pheno1 = row['phenotype1']
            pheno2 = row['phenotype2']

            # Get sample data
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]

            # Check for coexpression
            col1 = f'is_{pheno1}'
            col2 = f'is_{pheno2}'

            if col1 in sample_data.columns and col2 in sample_data.columns:
                # Calculate cellular coexpression
                n_total = len(sample_data)
                both_positive = (sample_data[col1] & sample_data[col2]).sum()
                either_positive = (sample_data[col1] | sample_data[col2]).sum()

                cell_jaccard = both_positive / either_positive if either_positive > 0 else 0
                percent_coexpressing = both_positive / n_total * 100 if n_total > 0 else 0

                # Compare with spatial overlap
                spatial_jaccard = row['mean_jaccard_index']
                percent_overlapping = row['percent_overlapping']

                comparison_results.append({
                    'sample_id': sample,
                    'phenotype1': pheno1,
                    'phenotype2': pheno2,
                    # Spatial metrics
                    'spatial_jaccard_index': spatial_jaccard,
                    'percent_regions_overlapping': percent_overlapping,
                    # Cellular metrics
                    'cellular_jaccard_index': cell_jaccard,
                    'percent_cells_coexpressing': percent_coexpressing,
                    # Comparison
                    'jaccard_difference': spatial_jaccard - cell_jaccard,
                    'spatial_exceeds_cellular': spatial_jaccard > cell_jaccard,
                    'timepoint': row['timepoint'],
                    'group': row['group'],
                    'main_group': row['main_group']
                })

        if comparison_results:
            df = pd.DataFrame(comparison_results)
            self.results['overlap_vs_coexpression'] = df
            print(f"  ✓ Compared spatial overlap vs coexpression for {len(comparison_results)} pairs")

    def _save_results(self):
        """Save all results to CSV files."""
        for name, df in self.results.items():
            if isinstance(df, pd.DataFrame):
                output_path = self.output_dir / f'{name}.csv'
                df.to_csv(output_path, index=False)

        print(f"  ✓ Saved {len(self.results)} result datasets")

    def _generate_plots(self):
        """Generate visualization plots."""
        # Overlap heatmap
        self._plot_overlap_heatmap()

        # Overlap vs coexpression scatter
        self._plot_overlap_vs_coexpression()

        # Top overlapping pairs
        self._plot_top_overlapping_pairs()

    def _plot_overlap_heatmap(self):
        """Generate heatmap of spatial overlap (Jaccard index)."""
        if 'pairwise_overlap_summary' not in self.results:
            return

        df = self.results['pairwise_overlap_summary']

        for group in df['main_group'].unique():
            if pd.isna(group) or group == '':
                continue

            group_data = df[df['main_group'] == group]

            # Create matrix
            n_phenos = len(self.phenotypes)
            matrix = np.zeros((n_phenos, n_phenos))
            matrix[:] = np.nan

            for _, row in group_data.iterrows():
                pheno1 = row['phenotype1']
                pheno2 = row['phenotype2']

                if pheno1 in self.phenotypes and pheno2 in self.phenotypes:
                    i = self.phenotypes.index(pheno1)
                    j = self.phenotypes.index(pheno2)

                    matrix[i, j] = row['mean_jaccard_index']
                    matrix[j, i] = row['mean_jaccard_index']  # Symmetric

            # Create DataFrame
            matrix_df = pd.DataFrame(
                matrix,
                index=self.phenotypes,
                columns=self.phenotypes
            )

            # Plot
            fig, ax = plt.subplots(figsize=(12, 10))

            sns.heatmap(
                matrix_df,
                annot=False,
                fmt='.2f',
                cmap='YlOrRd',
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Jaccard Index'},
                ax=ax,
                square=True
            )

            ax.set_title(f'Spatial Region Overlap - {group}\n(Jaccard Index)',
                        fontsize=14, fontweight='bold')

            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

            plt.tight_layout()
            plt.savefig(self.plots_dir / f'spatial_overlap_heatmap_{group}.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

        print(f"  ✓ Generated spatial overlap heatmaps")

    def _plot_overlap_vs_coexpression(self):
        """Plot spatial overlap vs cellular coexpression comparison."""
        if 'overlap_vs_coexpression' not in self.results:
            return

        df = self.results['overlap_vs_coexpression']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Jaccard index comparison
        ax = axes[0]
        ax.scatter(df['cellular_jaccard_index'], df['spatial_jaccard_index'],
                  alpha=0.6, s=50, c='steelblue', edgecolors='black', linewidth=0.5)

        # Add diagonal line
        max_val = max(df['cellular_jaccard_index'].max(), df['spatial_jaccard_index'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal overlap')

        ax.set_xlabel('Cellular Coexpression (Jaccard Index)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Spatial Region Overlap (Jaccard Index)', fontsize=11, fontweight='bold')
        ax.set_title('Spatial Overlap vs Cellular Coexpression', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 2: Percent comparison
        ax = axes[1]
        ax.scatter(df['percent_cells_coexpressing'], df['percent_regions_overlapping'],
                  alpha=0.6, s=50, c='forestgreen', edgecolors='black', linewidth=0.5)

        max_val = max(df['percent_cells_coexpressing'].max(), df['percent_regions_overlapping'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal percent')

        ax.set_xlabel('% Cells Coexpressing', fontsize=11, fontweight='bold')
        ax.set_ylabel('% Regions Overlapping', fontsize=11, fontweight='bold')
        ax.set_title('Region Overlap vs Cell Coexpression', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'overlap_vs_coexpression_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Generated overlap vs coexpression comparison plot")

    def _plot_top_overlapping_pairs(self):
        """Plot top spatially overlapping phenotype pairs."""
        if 'pairwise_overlap_summary' not in self.results:
            return

        df = self.results['pairwise_overlap_summary']

        # Calculate mean Jaccard across samples for each pair
        pair_jaccard = {}
        for (pheno1, pheno2), group_df in df.groupby(['phenotype1', 'phenotype2']):
            pair_name = f'{pheno1} + {pheno2}'
            pair_jaccard[pair_name] = group_df['mean_jaccard_index'].mean()

        # Sort and get top 20
        sorted_pairs = sorted(pair_jaccard.items(), key=lambda x: x[1], reverse=True)[:20]

        if not sorted_pairs:
            return

        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 8))

        pairs = [p[0] for p in sorted_pairs]
        values = [p[1] for p in sorted_pairs]

        bars = ax.barh(pairs, values)

        # Color code by value
        for bar, val in zip(bars, values):
            if val > 0.5:
                bar.set_color('#d62728')  # Red for high
            elif val > 0.25:
                bar.set_color('#ff7f0e')  # Orange for medium
            else:
                bar.set_color('#1f77b4')  # Blue for low

        ax.set_xlabel('Spatial Overlap (Jaccard Index)', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Spatially Overlapping Phenotype Pairs', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'top_overlapping_pairs.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Generated top overlapping pairs plot")

    def get_region_boundaries(self) -> Dict:
        """Return detected region boundaries."""
        return self.region_boundaries

    def get_boundary(self, sample: str, phenotype: str, region_id: int):
        """
        Get boundary for a specific phenotype region.

        Parameters
        ----------
        sample : str
            Sample ID
        phenotype : str
            Phenotype name
        region_id : int
            Region ID

        Returns
        -------
        shapely.geometry
            Boundary geometry
        """
        if sample not in self.region_boundaries:
            raise ValueError(f"No boundaries for sample {sample}")
        if phenotype not in self.region_boundaries[sample]:
            raise ValueError(f"No boundaries for {phenotype} in sample {sample}")
        if region_id not in self.region_boundaries[sample][phenotype]:
            raise ValueError(f"Region {region_id} not found for {phenotype} in sample {sample}")

        return self.region_boundaries[sample][phenotype][region_id]
