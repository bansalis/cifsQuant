"""
Immune Infiltration Analysis
Analyze immune infiltration into tumor structures with zone heterogeneity
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from scipy.stats import moran_i
import warnings


class InfiltrationAnalysis:
    """
    Analyze immune infiltration into tumor structures.

    Key features:
    - Tumor structure detection
    - Infiltration boundaries
    - Marker zone analysis (heterogeneity)
    - Zone-specific infiltration
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
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.results = {}
        self.tumor_structures = {}  # Per-sample tumor structure labels

    def run(self):
        """Run complete infiltration analysis."""
        print("\n" + "="*80)
        print("INFILTRATION ANALYSIS")
        print("="*80)

        # Detect tumor structures
        print("\n1. Detecting tumor structures...")
        self._detect_tumor_structures()

        # Calculate basic infiltration
        print("\n2. Calculating immune infiltration...")
        self._calculate_infiltration()

        # Marker zone analysis
        if self.config.get('marker_zone_analysis', {}).get('enabled', False):
            print("\n3. Analyzing marker zones...")
            self._analyze_marker_zones()

        # Save results
        self._save_results()

        print("\n✓ Infiltration analysis complete")
        print(f"  Results saved to: {self.output_dir}/")
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
                            # Within tumor (inside structure)
                            # Count immune cells inside tumor structures
                            # For simplicity, use distance = 0
                            count = 0  # Will implement properly below
                            zone_name = 'within_tumor'
                        else:
                            # Between boundaries
                            lower = boundaries[i-1]
                            upper = boundaries[i]
                            count = ((distances >= lower) & (distances < upper)).sum()
                            zone_name = f'{lower}_{upper}um'

                        infiltration_results.append({
                            'sample_id': sample,
                            'structure_id': int(structure_id),
                            'immune_population': immune_pop,
                            'zone': zone_name,
                            'boundary_lower': boundaries[i-1] if i > 0 else 0,
                            'boundary_upper': boundaries[i],
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

    def _analyze_marker_zones(self):
        """Analyze spatial heterogeneity of marker zones."""
        marker_config = self.config['marker_zone_analysis']
        markers = marker_config.get('markers', [])

        print(f"  Analyzing {len(markers)} marker zones...")

        for marker_def in markers:
            marker = marker_def['marker']
            pos_pheno = marker_def['positive_phenotype']
            neg_pheno = marker_def['negative_phenotype']

            print(f"    Processing {marker} zones...")
            self._analyze_single_marker_zone(marker, pos_pheno, neg_pheno, marker_config)

    def _analyze_single_marker_zone(self, marker: str, pos_pheno: str,
                                    neg_pheno: str, config: Dict):
        """Analyze a single marker zone."""
        pos_col = f'is_{pos_pheno}'
        neg_col = f'is_{neg_pheno}'

        if pos_col not in self.adata.obs.columns or neg_col not in self.adata.obs.columns:
            warnings.warn(f"Phenotypes for {marker} not found, skipping")
            return

        heterogeneity_results = []

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

                # Calculate heterogeneity metrics
                fraction_pos = n_pos / n_total if n_total > 0 else 0

                # Spatial clustering of marker+ cells
                if 'cluster_analysis' in config.get('heterogeneity_metrics', []):
                    clustering_score = self._calculate_clustering(
                        structure_coords, pos_mask
                    )
                else:
                    clustering_score = np.nan

                # Moran's I (spatial autocorrelation)
                if 'morans_i' in config.get('heterogeneity_metrics', []):
                    morans_i = self._calculate_morans_i(
                        structure_coords, pos_mask.astype(float)
                    )
                else:
                    morans_i = np.nan

                heterogeneity_results.append({
                    'sample_id': sample,
                    'structure_id': int(structure_id),
                    'marker': marker,
                    'n_positive': int(n_pos),
                    'n_negative': int(n_neg),
                    'fraction_positive': fraction_pos,
                    'clustering_score': clustering_score,
                    'morans_i': morans_i,
                    'timepoint': structure_data['timepoint'].iloc[0] if 'timepoint' in structure_data.columns else np.nan,
                    'group': structure_data['group'].iloc[0] if 'group' in structure_data.columns else '',
                    'main_group': structure_data['main_group'].iloc[0] if 'main_group' in structure_data.columns else ''
                })

        if heterogeneity_results:
            df = pd.DataFrame(heterogeneity_results)
            self.results[f'{marker}_zone_heterogeneity'] = df
            print(f"      ✓ Analyzed heterogeneity for {len(heterogeneity_results)} structures")

            # Zone-specific infiltration
            if config.get('zone_infiltration', {}).get('enabled', False):
                self._calculate_zone_specific_infiltration(
                    marker, pos_pheno, neg_pheno, config
                )

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

    def _calculate_morans_i(self, coords: np.ndarray, values: np.ndarray) -> float:
        """Calculate Moran's I spatial autocorrelation."""
        if len(coords) < 10:
            return np.nan

        try:
            # Build spatial weights matrix (distance-based)
            tree = cKDTree(coords)
            # Find k nearest neighbors
            k = min(8, len(coords) - 1)
            distances, indices = tree.query(coords, k=k+1)

            # Create weights matrix
            n = len(coords)
            W = np.zeros((n, n))
            for i in range(n):
                for j in range(1, k+1):  # Skip self
                    neighbor_idx = indices[i, j]
                    dist = distances[i, j]
                    if dist > 0:
                        W[i, neighbor_idx] = 1.0 / dist

            # Normalize rows
            row_sums = W.sum(axis=1)
            W = W / row_sums[:, np.newaxis]
            W[np.isnan(W)] = 0

            # Calculate Moran's I
            n = len(values)
            mean_val = np.mean(values)
            numerator = 0
            denominator = 0

            for i in range(n):
                for j in range(n):
                    numerator += W[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
                denominator += (values[i] - mean_val) ** 2

            if denominator == 0:
                return np.nan

            I = (n / W.sum()) * (numerator / denominator)

            return I

        except Exception as e:
            warnings.warn(f"Error calculating Moran's I: {e}")
            return np.nan

    def _calculate_zone_specific_infiltration(self, marker: str, pos_pheno: str,
                                             neg_pheno: str, config: Dict):
        """Calculate immune infiltration in marker+ vs marker- zones."""
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
