"""
Per-Tumor Analysis
Analyze metrics at the individual tumor structure level
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import warnings


class PerTumorAnalysis:
    """
    Analyze metrics at the individual tumor structure level.

    Key features:
    - Number of tumor cells per tumor (structure)
    - Percent pERK+, Ki67+, NINJA+ per tumor
    - Growth rate normalization (Ki67 correlation)
    - Per-tumor infiltration metrics
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        """
        Initialize per-tumor analysis.

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
        self.config = config
        self.tumor_config = config.get('tumor_definition', {})
        self.output_dir = Path(output_dir) / 'per_tumor_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.results = {}
        self.tumor_structures = {}  # Per-sample tumor structure labels

    def run(self):
        """Run complete per-tumor analysis."""
        print("\n" + "="*80)
        print("PER-TUMOR ANALYSIS")
        print("="*80)

        # Detect tumor structures
        print("\n1. Detecting tumor structures...")
        self._detect_tumor_structures()

        # Calculate per-tumor metrics
        print("\n2. Calculating per-tumor metrics...")
        self._calculate_per_tumor_metrics()

        # Calculate marker percentages per tumor
        print("\n3. Calculating marker percentages per tumor...")
        self._calculate_marker_percentages()

        # Growth rate normalization
        print("\n4. Calculating growth-rate normalized metrics...")
        self._calculate_growth_normalized_metrics()

        # Per-tumor infiltration
        print("\n5. Calculating per-tumor infiltration...")
        self._calculate_per_tumor_infiltration()

        # Save results
        self._save_results()

        print("\n✓ Per-tumor analysis complete")
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

    def _calculate_per_tumor_metrics(self):
        """Calculate basic metrics per tumor structure."""
        tumor_pheno = self.tumor_config.get('base_phenotype', 'Tumor')
        tumor_col = f'is_{tumor_pheno}'

        per_tumor_results = []

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

                # Count tumor cells in this structure
                tumor_mask = structure_data[tumor_col].values
                n_tumor_cells = tumor_mask.sum()

                # Calculate area
                if len(structure_coords) > 0:
                    x_range = structure_coords[:, 0].max() - structure_coords[:, 0].min()
                    y_range = structure_coords[:, 1].max() - structure_coords[:, 1].min()
                    area_um2 = x_range * y_range
                    density = n_tumor_cells / area_um2 if area_um2 > 0 else 0
                else:
                    area_um2 = 0
                    density = 0

                per_tumor_results.append({
                    'sample_id': sample,
                    'tumor_id': int(structure_id),
                    'n_tumor_cells': int(n_tumor_cells),
                    'n_total_cells': int(structure_mask.sum()),
                    'area_um2': area_um2,
                    'density_cells_per_um2': density,
                    'timepoint': structure_data['timepoint'].iloc[0] if 'timepoint' in structure_data.columns else np.nan,
                    'group': structure_data['group'].iloc[0] if 'group' in structure_data.columns else '',
                    'main_group': structure_data['main_group'].iloc[0] if 'main_group' in structure_data.columns else ''
                })

        if per_tumor_results:
            df = pd.DataFrame(per_tumor_results)
            self.results['per_tumor_metrics'] = df
            print(f"    ✓ Calculated metrics for {len(per_tumor_results)} tumor structures")

    def _calculate_marker_percentages(self):
        """Calculate marker percentages per tumor structure."""
        # Markers to analyze - read from config
        per_tumor_config = self.config.get('per_tumor_analysis', {})
        marker_configs = per_tumor_config.get('markers', [])

        if marker_configs:
            markers = marker_configs
        else:
            # Default markers
            markers = [
                {'name': 'pERK', 'phenotype': 'pERK_positive_tumor'},
                {'name': 'NINJA', 'phenotype': 'AGFP_positive_tumor'},
                {'name': 'Ki67', 'phenotype': 'Ki67_positive_tumor'}
            ]

        marker_results = []

        tumor_pheno = self.tumor_config.get('base_phenotype', 'Tumor')
        tumor_col = f'is_{tumor_pheno}'

        for sample in self.tumor_structures.keys():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.obs[sample_mask]

            structure_labels = self.tumor_structures[sample]
            unique_structures = set(structure_labels) - {-1}

            for structure_id in unique_structures:
                structure_mask = structure_labels == structure_id
                structure_data = sample_data[structure_mask]

                # Count tumor cells in this structure
                tumor_mask = structure_data[tumor_col].values
                n_tumor_cells = tumor_mask.sum()

                if n_tumor_cells < 10:  # Skip very small tumors
                    continue

                result = {
                    'sample_id': sample,
                    'tumor_id': int(structure_id),
                    'n_tumor_cells': int(n_tumor_cells),
                    'timepoint': structure_data['timepoint'].iloc[0] if 'timepoint' in structure_data.columns else np.nan,
                    'group': structure_data['group'].iloc[0] if 'group' in structure_data.columns else '',
                    'main_group': structure_data['main_group'].iloc[0] if 'main_group' in structure_data.columns else ''
                }

                # Calculate percentage for each marker
                for marker_def in markers:
                    marker_name = marker_def['name']
                    marker_pheno = marker_def['phenotype']
                    marker_col = f'is_{marker_pheno}'

                    if marker_col in structure_data.columns:
                        marker_mask = structure_data[marker_col].values
                        n_marker_positive = marker_mask.sum()
                        percent_marker = (n_marker_positive / n_tumor_cells * 100) if n_tumor_cells > 0 else 0

                        result[f'n_{marker_name}_positive'] = int(n_marker_positive)
                        result[f'percent_{marker_name}_positive'] = percent_marker

                marker_results.append(result)

        if marker_results:
            df = pd.DataFrame(marker_results)
            self.results['per_tumor_marker_percentages'] = df
            print(f"    ✓ Calculated marker percentages for {len(marker_results)} tumors")

    def _calculate_growth_normalized_metrics(self):
        """Calculate pERK+ normalized by tumor growth rate (Ki67+)."""
        if 'per_tumor_marker_percentages' not in self.results:
            print("    ⚠ No marker percentages available, skipping")
            return

        df = self.results['per_tumor_marker_percentages'].copy()

        # Check if required columns exist
        required_cols = ['percent_pERK_positive', 'percent_Ki67_positive']
        if not all(col in df.columns for col in required_cols):
            print("    ⚠ Required marker data not available, skipping")
            return

        # Calculate growth-normalized pERK
        # Method 1: Simple ratio (pERK / Ki67)
        df['pERK_per_Ki67_ratio'] = df['percent_pERK_positive'] / (df['percent_Ki67_positive'] + 0.01)  # Add small constant to avoid division by zero

        # Method 2: Difference (pERK - Ki67) - shows excess pERK beyond proliferation
        df['pERK_minus_Ki67'] = df['percent_pERK_positive'] - df['percent_Ki67_positive']

        # Method 3: Residual after Ki67 regression (per timepoint/group)
        growth_norm_results = []

        for (timepoint, group), group_df in df.groupby(['timepoint', 'main_group']):
            if len(group_df) < 5:  # Need enough tumors for regression
                continue

            # Simple linear regression: pERK ~ Ki67
            ki67 = group_df['percent_Ki67_positive'].values
            perk = group_df['percent_pERK_positive'].values

            # Calculate regression manually
            valid_mask = ~(np.isnan(ki67) | np.isnan(perk))
            if valid_mask.sum() < 3:
                continue

            ki67_valid = ki67[valid_mask]
            perk_valid = perk[valid_mask]

            # Fit line
            slope, intercept = np.polyfit(ki67_valid, perk_valid, 1)

            # Calculate residuals for all tumors in this group
            predicted_perk = slope * ki67 + intercept
            residuals = perk - predicted_perk

            # Store results
            for idx, (sample_id, tumor_id) in enumerate(zip(group_df['sample_id'], group_df['tumor_id'])):
                growth_norm_results.append({
                    'sample_id': sample_id,
                    'tumor_id': tumor_id,
                    'timepoint': timepoint,
                    'main_group': group,
                    'pERK_residual_from_Ki67': residuals[idx],
                    'predicted_pERK_from_Ki67': predicted_perk[idx]
                })

        # Merge back to main df
        if growth_norm_results:
            residuals_df = pd.DataFrame(growth_norm_results)
            df = df.merge(residuals_df[['sample_id', 'tumor_id', 'pERK_residual_from_Ki67', 'predicted_pERK_from_Ki67']],
                         on=['sample_id', 'tumor_id'], how='left')

        self.results['per_tumor_growth_normalized'] = df
        print(f"    ✓ Calculated growth-normalized metrics")

    def _calculate_per_tumor_infiltration(self):
        """Calculate immune infiltration per tumor structure."""
        # Immune populations to analyze
        immune_pops = ['CD8_T_cells', 'CD3_positive', 'CD45_positive']

        infiltration_results = []

        tumor_pheno = self.tumor_config.get('base_phenotype', 'Tumor')
        tumor_col = f'is_{tumor_pheno}'

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

                # Count tumor cells
                tumor_mask = structure_data[tumor_col].values
                n_tumor_cells = tumor_mask.sum()

                if n_tumor_cells < 10:
                    continue

                result = {
                    'sample_id': sample,
                    'tumor_id': int(structure_id),
                    'n_tumor_cells': int(n_tumor_cells),
                    'timepoint': structure_data['timepoint'].iloc[0] if 'timepoint' in structure_data.columns else np.nan,
                    'group': structure_data['group'].iloc[0] if 'group' in structure_data.columns else '',
                    'main_group': structure_data['main_group'].iloc[0] if 'main_group' in structure_data.columns else ''
                }

                # Calculate infiltration for each immune population
                for immune_pop in immune_pops:
                    immune_col = f'is_{immune_pop}'
                    if immune_col not in structure_data.columns:
                        continue

                    immune_mask = structure_data[immune_col].values
                    n_immune_within = immune_mask.sum()

                    # Calculate density
                    immune_density = n_immune_within / n_tumor_cells if n_tumor_cells > 0 else 0

                    result[f'n_{immune_pop}_within_tumor'] = int(n_immune_within)
                    result[f'{immune_pop}_density_per_tumor_cell'] = immune_density
                    result[f'percent_{immune_pop}_of_total_cells'] = (n_immune_within / len(structure_data) * 100) if len(structure_data) > 0 else 0

                infiltration_results.append(result)

        if infiltration_results:
            df = pd.DataFrame(infiltration_results)
            self.results['per_tumor_infiltration'] = df
            print(f"    ✓ Calculated infiltration for {len(infiltration_results)} tumors")

    def _save_results(self):
        """Save all results to files."""
        for name, df in self.results.items():
            output_path = self.output_dir / f'{name}.csv'
            df.to_csv(output_path, index=False)

        print(f"\n  ✓ Saved {len(self.results)} result datasets")

    def get_tumor_structures(self) -> Dict:
        """Return tumor structure labels for use by other analyses."""
        return self.tumor_structures
