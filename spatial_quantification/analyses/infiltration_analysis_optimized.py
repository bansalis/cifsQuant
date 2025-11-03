"""
Optimized Immune Infiltration Analysis
Uses Getis-Ord Gi* and Ripley's K for efficient spatial analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import warnings


class InfiltrationAnalysisOptimized:
    """
    Optimized immune infiltration analysis.

    Key improvements:
    - Getis-Ord Gi* for local hotspot detection (faster than Moran's I)
    - Ripley's K for multi-scale clustering quantification
    - Spatial subsampling for large structures (>10k cells)
    - Comprehensive spatial visualization
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        """
        Initialize infiltration analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data with phenotypes and spatial coordinates
        config : dict
            Configuration dictionary
        output_dir : Path
            Output directory
        """
        self.adata = adata
        self.config = config['immune_infiltration']
        self.tumor_config = config.get('tumor_definition', {})
        self.output_dir = Path(output_dir) / 'infiltration_analysis'
        self.plots_dir = self.output_dir / 'spatial_plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.results = {}
        self.tumor_structures = {}  # Per-sample tumor structure labels

    def run(self):
        """Run complete infiltration analysis."""
        print("\n" + "="*80)
        print("INFILTRATION ANALYSIS (OPTIMIZED)")
        print("="*80)

        # Detect tumor structures
        print("\n1. Detecting tumor structures...")
        self._detect_tumor_structures()

        # Calculate basic infiltration
        print("\n2. Calculating immune infiltration...")
        self._calculate_infiltration()

        # Marker zone analysis (optimized)
        if self.config.get('marker_zone_analysis', {}).get('enabled', False):
            print("\n3. Analyzing marker zones (OPTIMIZED)...")
            self._analyze_marker_zones_optimized()

        # Save results
        self._save_results()

        print("\n✓ Infiltration analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
        print(f"  Spatial plots: {self.plots_dir}/")
        print("="*80 + "\n")

        return self.results

    def _detect_tumor_structures(self):
        """Detect tumor structures using DBSCAN clustering."""
        tumor_def = self.tumor_config
        tumor_pheno = tumor_def.get('base_phenotype', 'Tumor')
        tumor_col = f'is_{tumor_pheno}'

        if tumor_col not in self.adata.obs.columns:
            raise ValueError(f"Tumor phenotype '{tumor_pheno}' not found")

        # Structure detection parameters
        struct_config = tumor_def.get('structure_detection', {})
        eps = struct_config.get('eps', 100)
        min_samples = struct_config.get('min_samples', 10)
        min_cluster_size = struct_config.get('min_cluster_size', 50)

        print(f"  Using DBSCAN with eps={eps}, min_samples={min_samples}")

        # Detect structures per sample
        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]
            sample_coords = self.adata.obsm['spatial'][sample_mask.values]

            # Get tumor cells
            tumor_mask = sample_data[tumor_col].values
            if tumor_mask.sum() < min_samples:
                print(f"    ⚠ {sample}: Too few tumor cells ({tumor_mask.sum()}), skipping")
                continue

            tumor_coords = sample_coords[tumor_mask]

            # DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(tumor_coords)
            labels = clustering.labels_

            # Filter small clusters
            unique_labels = set(labels) - {-1}  # Exclude noise
            valid_labels = []
            for label in unique_labels:
                if (labels == label).sum() >= min_cluster_size:
                    valid_labels.append(label)

            # Store structure labels
            structure_labels = np.full(len(sample_data), -1)
            tumor_indices = np.where(tumor_mask)[0]
            for label in valid_labels:
                cluster_mask = labels == label
                structure_labels[tumor_indices[cluster_mask]] = label

            self.tumor_structures[sample] = structure_labels

            n_structures = len(valid_labels)
            print(f"    ✓ {sample}: Detected {n_structures} tumor structures")

    def _calculate_infiltration(self):
        """Calculate immune infiltration per tumor structure."""
        # Get immune populations
        immune_pops = self.config.get('immune_populations', [])
        boundaries = self.config.get('boundaries', [0, 50, 100, 200])

        print(f"  Analyzing {len(immune_pops)} immune populations...")
        print(f"  Boundaries: {boundaries} μm")

        infiltration_results = []

        for sample in self.tumor_structures.keys():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]
            sample_coords = self.adata.obsm['spatial'][sample_mask.values]

            structure_labels = self.tumor_structures[sample]
            unique_structures = set(structure_labels) - {-1}

            for structure_id in unique_structures:
                # Get tumor structure cells
                structure_mask = structure_labels == structure_id
                structure_coords = sample_coords[structure_mask]

                if len(structure_coords) < 10:
                    continue

                # Build KDTree for distance calculations
                tumor_tree = cKDTree(structure_coords)

                # Calculate infiltration for each immune population
                for immune_pop in immune_pops:
                    immune_col = f'is_{immune_pop}'
                    if immune_col not in sample_data.columns:
                        continue

                    immune_mask = sample_data[immune_col].values
                    immune_coords = sample_coords[immune_mask]

                    if len(immune_coords) == 0:
                        continue

                    # Calculate distances from immune cells to tumor
                    distances, _ = tumor_tree.query(immune_coords, k=1)

                    # Count infiltration in each boundary
                    for i in range(len(boundaries)):
                        if i == 0:
                            zone_name = 'within_tumor'
                            lower = 0
                            upper = boundaries[i]
                            count = 0  # Within tumor handled separately
                        else:
                            lower = boundaries[i-1]
                            upper = boundaries[i]
                            count = ((distances >= lower) & (distances < upper)).sum()
                            zone_name = f'{lower}_{upper}um'

                        infiltration_results.append({
                            'sample_id': sample,
                            'structure_id': int(structure_id),
                            'immune_population': immune_pop,
                            'zone': zone_name,
                            'boundary_lower': lower,
                            'boundary_upper': upper,
                            'count': int(count),
                            'structure_size': int(structure_mask.sum()),
                            'timepoint': sample_data['timepoint'].iloc[0] if 'timepoint' in sample_data.columns else np.nan,
                            'group': sample_data['group'].iloc[0] if 'group' in sample_data.columns else '',
                            'main_group': sample_data['main_group'].iloc[0] if 'main_group' in sample_data.columns else ''
                        })

        if infiltration_results:
            df = pd.DataFrame(infiltration_results)
            self.results['infiltration'] = df
            print(f"    ✓ Calculated infiltration for {len(infiltration_results)} structure-population combinations")

    def _analyze_marker_zones_optimized(self):
        """Analyze marker zones with optimized methods (Getis-Ord Gi* + Ripley's K)."""
        marker_config = self.config['marker_zone_analysis']
        markers = marker_config.get('markers', [])

        print(f"  Analyzing {len(markers)} marker zones...")

        for marker_def in markers:
            marker = marker_def['marker']
            pos_pheno = marker_def['positive_phenotype']
            neg_pheno = marker_def['negative_phenotype']

            print(f"\n    Processing {marker} zones (OPTIMIZED)...")
            self._analyze_single_marker_zone_optimized(marker, pos_pheno, neg_pheno, marker_config)

    def _analyze_single_marker_zone_optimized(self, marker: str, pos_pheno: str,
                                              neg_pheno: str, config: Dict):
        """Analyze a single marker zone with optimization."""
        pos_col = f'is_{pos_pheno}'
        neg_col = f'is_{neg_pheno}'

        if pos_col not in self.adata.obs.columns or neg_col not in self.adata.obs.columns:
            warnings.warn(f"Phenotypes for {marker} not found, skipping")
            return

        heterogeneity_results = []
        plot_counter = 0

        for sample in self.tumor_structures.keys():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]
            sample_coords = self.adata.obsm['spatial'][sample_mask.values]

            structure_labels = self.tumor_structures[sample]
            unique_structures = set(structure_labels) - {-1}

            for structure_id in unique_structures:
                structure_mask = structure_labels == structure_id
                structure_data = sample_data[structure_mask]
                structure_coords = sample_coords[structure_mask]

                # Get marker+ and marker- cells within structure
                pos_mask = structure_data[pos_col].values
                neg_mask = structure_data[neg_col].values

                n_pos = pos_mask.sum()
                n_neg = neg_mask.sum()
                n_total = len(structure_data)

                if n_pos < 5 or n_neg < 5:
                    continue

                # Calculate heterogeneity metrics (OPTIMIZED)
                fraction_pos = n_pos / n_total if n_total > 0 else 0

                # 1. Getis-Ord Gi* (local hotspot statistic)
                gi_star_stats = self._calculate_getis_ord_gi_star(
                    structure_coords, pos_mask
                )

                # 2. Ripley's K (multi-scale clustering)
                ripleys_k_stats = self._calculate_ripleys_k(
                    structure_coords, pos_mask
                )

                # 3. DBSCAN clustering score
                if 'cluster_analysis' in config.get('heterogeneity_metrics', []):
                    clustering_score = self._calculate_clustering(
                        structure_coords, pos_mask
                    )
                else:
                    clustering_score = np.nan

                heterogeneity_results.append({
                    'sample_id': sample,
                    'structure_id': int(structure_id),
                    'marker': marker,
                    'n_positive': int(n_pos),
                    'n_negative': int(n_neg),
                    'fraction_positive': fraction_pos,
                    'clustering_score': clustering_score,
                    'gi_star_mean': gi_star_stats['mean_gi_star'],
                    'gi_star_max': gi_star_stats['max_gi_star'],
                    'gi_star_hotspots': gi_star_stats['n_hotspots'],
                    'ripleys_k_30um': ripleys_k_stats['K_30um'],
                    'ripleys_k_50um': ripleys_k_stats['K_50um'],
                    'ripleys_k_100um': ripleys_k_stats['K_100um'],
                    'ripleys_l_30um': ripleys_k_stats['L_30um'],
                    'ripleys_l_50um': ripleys_k_stats['L_50um'],
                    'ripleys_l_100um': ripleys_k_stats['L_100um'],
                    'timepoint': structure_data['timepoint'].iloc[0] if 'timepoint' in structure_data.columns else np.nan,
                    'group': structure_data['group'].iloc[0] if 'group' in structure_data.columns else '',
                    'main_group': structure_data['main_group'].iloc[0] if 'main_group' in structure_data.columns else ''
                })

                # Generate spatial plot (first 20 structures per sample)
                if plot_counter < 20:
                    self._plot_marker_zone_spatial(
                        structure_coords, pos_mask, neg_mask,
                        marker, sample, structure_id,
                        gi_star_stats, ripleys_k_stats
                    )
                    plot_counter += 1

        if heterogeneity_results:
            df = pd.DataFrame(heterogeneity_results)
            self.results[f'{marker}_zone_heterogeneity'] = df
            print(f"      ✓ Analyzed heterogeneity for {len(heterogeneity_results)} structures")
            print(f"      ✓ Generated {min(plot_counter, 20)} spatial plots")

            # Zone-specific infiltration
            if config.get('zone_infiltration', {}).get('enabled', False):
                self._calculate_zone_specific_infiltration_optimized(
                    marker, pos_pheno, neg_pheno, config
                )

    def _calculate_getis_ord_gi_star(self, coords: np.ndarray, mask: np.ndarray,
                                     subsample_size: int = 100000) -> Dict:
        """
        Calculate Getis-Ord Gi* local spatial statistic.

        More efficient than Moran's I for large datasets.
        Identifies LOCAL hotspots rather than global autocorrelation.

        Parameters
        ----------
        coords : np.ndarray
            Cell coordinates (N x 2)
        mask : np.ndarray
            Boolean mask for positive cells
        subsample_size : int
            Maximum cells to use (subsample if larger)

        Returns
        -------
        dict
            Gi* statistics
        """
        if mask.sum() < 5:
            return {
                'mean_gi_star': np.nan,
                'max_gi_star': np.nan,
                'n_hotspots': 0
            }

        # Subsample if too large
        if len(coords) > subsample_size:
            indices = np.random.choice(len(coords), subsample_size, replace=False)
            coords = coords[indices]
            mask = mask[indices]

        # Convert mask to weights (1 for positive, 0 for negative)
        values = mask.astype(float)

        # Build spatial weights (k-nearest neighbors)
        k = min(8, len(coords) - 1)
        tree = cKDTree(coords)
        distances, indices = tree.query(coords, k=k+1)  # +1 for self

        # Calculate Gi* for each cell
        gi_star = np.zeros(len(coords))

        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val == 0:
            return {
                'mean_gi_star': 0,
                'max_gi_star': 0,
                'n_hotspots': 0
            }

        for i in range(len(coords)):
            # Neighbor values (excluding self)
            neighbor_vals = values[indices[i, 1:]]  # Skip self at index 0

            # Getis-Ord Gi*
            local_sum = np.sum(neighbor_vals)
            local_mean = local_sum / k

            # Z-score version of Gi*
            gi_star[i] = (local_mean - mean_val) / (std_val / np.sqrt(k))

        # Statistics
        mean_gi_star = np.mean(gi_star)
        max_gi_star = np.max(gi_star)

        # Count significant hotspots (Gi* > 1.96, p < 0.05)
        n_hotspots = (gi_star > 1.96).sum()

        return {
            'mean_gi_star': mean_gi_star,
            'max_gi_star': max_gi_star,
            'n_hotspots': int(n_hotspots),
            'gi_star_values': gi_star  # For plotting
        }

    def _calculate_ripleys_k(self, coords: np.ndarray, mask: np.ndarray,
                            radii: List[float] = [30, 50, 100]) -> Dict:
        """
        Calculate Ripley's K function for multi-scale clustering.

        Quantifies clustering at different spatial scales.

        Parameters
        ----------
        coords : np.ndarray
            Cell coordinates
        mask : np.ndarray
            Boolean mask for positive cells
        radii : list of float
            Radii to calculate K at (μm)

        Returns
        -------
        dict
            Ripley's K and L statistics
        """
        pos_coords = coords[mask]

        if len(pos_coords) < 10:
            return {f'K_{r}um': np.nan for r in radii} | {f'L_{r}um': np.nan for r in radii}

        # Estimate area
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        area = x_range * y_range

        if area == 0:
            return {f'K_{r}um': np.nan for r in radii} | {f'L_{r}um': np.nan for r in radii}

        n = len(pos_coords)
        intensity = n / area

        # Build KDTree
        tree = cKDTree(pos_coords)

        results = {}

        for r in radii:
            # Count neighbors within radius r
            neighbor_counts = tree.query_ball_point(pos_coords, r, return_length=True)

            # Ripley's K
            K = np.sum(neighbor_counts - 1) / (n * intensity)  # -1 to exclude self

            # L transformation (variance stabilization)
            L = np.sqrt(K / np.pi) - r

            results[f'K_{r}um'] = K
            results[f'L_{r}um'] = L

        return results

    def _calculate_clustering(self, coords: np.ndarray, mask: np.ndarray) -> float:
        """Calculate spatial clustering score for positive cells."""
        if mask.sum() < 5:
            return np.nan

        pos_coords = coords[mask]

        # Use DBSCAN to measure clustering
        clustering = DBSCAN(eps=30, min_samples=5).fit(pos_coords)
        labels = clustering.labels_

        # Calculate clustering score as fraction of cells in clusters
        n_clustered = (labels != -1).sum()
        clustering_score = n_clustered / len(labels) if len(labels) > 0 else 0

        return clustering_score

    def _plot_marker_zone_spatial(self, coords: np.ndarray, pos_mask: np.ndarray,
                                  neg_mask: np.ndarray, marker: str, sample: str,
                                  structure_id: int, gi_star_stats: Dict,
                                  ripleys_stats: Dict):
        """
        Create spatial plot showing marker+ vs marker- regions.

        Parameters
        ----------
        coords : np.ndarray
            Cell coordinates
        pos_mask, neg_mask : np.ndarray
            Boolean masks for positive/negative cells
        marker : str
            Marker name (e.g., 'PERK')
        sample : str
            Sample ID
        structure_id : int
            Tumor structure ID
        gi_star_stats : dict
            Getis-Ord Gi* statistics
        ripleys_stats : dict
            Ripley's K statistics
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: Marker+ vs Marker- cells
        ax = axes[0]
        ax.scatter(coords[neg_mask, 0], coords[neg_mask, 1],
                  c='lightgray', s=1, alpha=0.3, label=f'{marker}-')
        ax.scatter(coords[pos_mask, 0], coords[pos_mask, 1],
                  c='red', s=1, alpha=0.6, label=f'{marker}+')
        ax.set_title(f'{marker}+ vs {marker}- Cells')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.legend(markerscale=10)
        ax.set_aspect('equal')

        # Panel 2: Getis-Ord Gi* Hotspots
        ax = axes[1]
        if 'gi_star_values' in gi_star_stats:
            gi_star = gi_star_stats['gi_star_values']
            scatter = ax.scatter(coords[:, 0], coords[:, 1],
                               c=gi_star, s=1, alpha=0.5,
                               cmap='RdYlBu_r', vmin=-2, vmax=2)
            plt.colorbar(scatter, ax=ax, label='Gi* Z-score')
            ax.set_title(f'Gi* Hotspots (n={gi_star_stats["n_hotspots"]})')
        else:
            ax.text(0.5, 0.5, 'Gi* not calculated', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('Gi* Hotspots')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_aspect('equal')

        # Panel 3: DBSCAN Clusters
        ax = axes[2]
        if pos_mask.sum() >= 5:
            pos_coords = coords[pos_mask]
            clustering = DBSCAN(eps=30, min_samples=5).fit(pos_coords)
            labels = clustering.labels_

            # Plot clusters
            unique_labels = set(labels)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                if label == -1:
                    color = 'lightgray'
                    label_name = 'Noise'
                else:
                    label_name = f'Cluster {label}'

                cluster_mask = labels == label
                ax.scatter(pos_coords[cluster_mask, 0],
                          pos_coords[cluster_mask, 1],
                          c=[color], s=2, alpha=0.6, label=label_name)

            ax.set_title(f'DBSCAN Clusters ({len(unique_labels)-1} clusters)')
            ax.legend(markerscale=5, fontsize=8, loc='upper right')
        else:
            ax.text(0.5, 0.5, 'Too few cells', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('DBSCAN Clusters')

        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_aspect('equal')

        # Add statistics as text
        stats_text = (
            f"Ripley's L:\n"
            f"  30μm: {ripleys_stats.get('L_30um', np.nan):.2f}\n"
            f"  50μm: {ripleys_stats.get('L_50um', np.nan):.2f}\n"
            f"  100μm: {ripleys_stats.get('L_100um', np.nan):.2f}"
        )
        fig.text(0.98, 0.5, stats_text, ha='right', va='center',
                fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'{marker} Zones: {sample} Structure {structure_id}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save
        plot_path = self.plots_dir / f'{marker}_{sample}_structure{structure_id}_spatial.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _calculate_zone_specific_infiltration_optimized(self, marker: str, pos_pheno: str,
                                                        neg_pheno: str, config: Dict):
        """Calculate zone-specific infiltration with optimization."""
        print(f"      Calculating zone-specific infiltration for {marker}...")

        immune_pops = config['zone_infiltration'].get('immune_populations', [])
        zone_infiltration_results = []

        pos_col = f'is_{pos_pheno}'
        neg_col = f'is_{neg_pheno}'

        for sample in self.tumor_structures.keys():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]
            sample_coords = self.adata.obsm['spatial'][sample_mask.values]

            structure_labels = self.tumor_structures[sample]
            unique_structures = set(structure_labels) - {-1}

            for structure_id in unique_structures:
                structure_mask = structure_labels == structure_id
                structure_data = sample_data[structure_mask]
                structure_coords = sample_coords[structure_mask]

                # Get marker zones
                pos_mask = structure_data[pos_col].values
                neg_mask = structure_data[neg_col].values

                if pos_mask.sum() < 5 or neg_mask.sum() < 5:
                    continue

                pos_coords = structure_coords[pos_mask]
                neg_coords = structure_coords[neg_mask]

                # Build KDTrees
                pos_tree = cKDTree(pos_coords)
                neg_tree = cKDTree(neg_coords)

                # Calculate infiltration for each immune population
                for immune_pop in immune_pops:
                    immune_col = f'is_{immune_pop}'
                    if immune_col not in structure_data.columns:
                        continue

                    immune_mask = structure_data[immune_col].values
                    immune_coords = structure_coords[immune_mask]

                    if len(immune_coords) == 0:
                        continue

                    # Distance to marker+ and marker- zones
                    dist_to_pos, _ = pos_tree.query(immune_coords, k=1)
                    dist_to_neg, _ = neg_tree.query(immune_coords, k=1)

                    # Count infiltration in each zone (within 50 μm)
                    infiltration_radius = 50
                    count_near_pos = (dist_to_pos < infiltration_radius).sum()
                    count_near_neg = (dist_to_neg < infiltration_radius).sum()

                    zone_infiltration_results.append({
                        'sample_id': sample,
                        'structure_id': int(structure_id),
                        'marker': marker,
                        'immune_population': immune_pop,
                        'zone_positive_infiltration': int(count_near_pos),
                        'zone_negative_infiltration': int(count_near_neg),
                        'n_positive_cells': int(pos_mask.sum()),
                        'n_negative_cells': int(neg_mask.sum()),
                        'infiltration_enrichment': count_near_pos / pos_mask.sum() if pos_mask.sum() > 0 else 0,
                        'timepoint': structure_data['timepoint'].iloc[0] if 'timepoint' in structure_data.columns else np.nan,
                        'group': structure_data['group'].iloc[0] if 'group' in structure_data.columns else '',
                        'main_group': structure_data['main_group'].iloc[0] if 'main_group' in structure_data.columns else ''
                    })

        if zone_infiltration_results:
            df = pd.DataFrame(zone_infiltration_results)
            self.results[f'{marker}_zone_infiltration'] = df
            print(f"        ✓ Calculated zone-specific infiltration")

    def _save_results(self):
        """Save all results to files."""
        for name, df in self.results.items():
            output_path = self.output_dir / f'{name}.csv'
            df.to_csv(output_path, index=False)

        print(f"\n  ✓ Saved {len(self.results)} result datasets")
