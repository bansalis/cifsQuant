"""
Phenotype Builder for Spatial Quantification
Build custom phenotypes from manual_gating gates
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import warnings


class PhenotypeBuilder:
    """Build custom cell phenotypes from gated populations."""

    def __init__(self, adata, config: Dict):
        """
        Initialize phenotype builder.

        Parameters
        ----------
        adata : AnnData
            AnnData object with gated populations (is_* columns)
        config : dict
            Configuration dictionary
        """
        self.adata = adata
        self.config = config
        self.phenotypes_config = config['phenotypes']

        # Detect available gates
        self.available_gates = self._detect_gates()

    def _detect_gates(self) -> Dict[str, str]:
        """
        Detect available gates from adata.obs.

        Returns
        -------
        dict
            Mapping of marker name to is_* column name
        """
        gates = {}
        for col in self.adata.obs.columns:
            if col.startswith('is_'):
                marker = col.replace('is_', '')
                gates[marker] = col

        return gates

    def build_all_phenotypes(self):
        """Build all phenotypes defined in config."""
        print("\n" + "="*80)
        print("BUILDING PHENOTYPES")
        print("="*80)
        print(f"Available gates: {list(self.available_gates.keys())}")
        print(f"Phenotypes to build: {len(self.phenotypes_config)}")
        print()

        for phenotype_name, phenotype_def in self.phenotypes_config.items():
            self._build_phenotype(phenotype_name, phenotype_def)

        # Print summary
        self._print_summary()

        return self.adata

    def _build_phenotype(self, name: str, definition: Dict):
        """
        Build a single phenotype.

        Parameters
        ----------
        name : str
            Phenotype name
        definition : dict
            Phenotype definition with positive/negative markers
        """
        # Check if phenotype already exists
        pheno_col = f'is_{name}'
        if pheno_col in self.adata.obs.columns:
            print(f"  ⚠ Phenotype '{name}' already exists, skipping")
            return

        # Get positive and negative markers
        positive_markers = definition.get('positive', [])
        negative_markers = definition.get('negative', [])
        base_phenotype = definition.get('base', None)

        # Validate markers exist
        all_markers = positive_markers + negative_markers
        missing_markers = [m for m in all_markers if m not in self.available_gates]
        if missing_markers:
            warnings.warn(f"Phenotype '{name}': markers not found: {missing_markers}. Skipping.")
            return

        # Start with all cells
        mask = np.ones(len(self.adata), dtype=bool)

        # Apply base phenotype if specified
        if base_phenotype:
            base_col = f'is_{base_phenotype}'
            if base_col in self.adata.obs.columns:
                mask &= self.adata.obs[base_col].values
            else:
                warnings.warn(f"Base phenotype '{base_phenotype}' not found for '{name}'")
                return

        # Apply positive markers
        for marker in positive_markers:
            gate_col = self.available_gates[marker]
            mask &= self.adata.obs[gate_col].values

        # Apply negative markers
        for marker in negative_markers:
            gate_col = self.available_gates[marker]
            mask &= ~self.adata.obs[gate_col].values

        # Add phenotype to adata
        self.adata.obs[pheno_col] = mask

        # Count and report
        count = mask.sum()
        pct = 100 * count / len(self.adata)

        # Build description string
        pos_str = ' '.join([f"{m}+" for m in positive_markers])
        neg_str = ' '.join([f"{m}-" for m in negative_markers])
        marker_str = ' '.join([pos_str, neg_str]).strip()

        print(f"  ✓ {name:30s} ({marker_str:30s}): {count:>8,} cells ({pct:>5.1f}%)")

        # Store color if provided
        if 'color' in definition:
            # Store in uns for later use
            if 'phenotype_colors' not in self.adata.uns:
                self.adata.uns['phenotype_colors'] = {}
            self.adata.uns['phenotype_colors'][name] = definition['color']

    def _print_summary(self):
        """Print phenotype building summary."""
        print("\n" + "="*80)

        # Count phenotypes
        phenotype_cols = [col for col in self.adata.obs.columns if col.startswith('is_')]
        print(f"Total phenotypes available: {len(phenotype_cols)}")

        # Check for tumor phenotypes
        tumor_phenos = [col for col in phenotype_cols
                       if 'tumor' in col.lower() or col == 'is_Tumor']
        immune_phenos = [col for col in phenotype_cols if col not in tumor_phenos]

        print(f"  - Tumor phenotypes: {len(tumor_phenos)}")
        print(f"  - Immune phenotypes: {len(immune_phenos)}")
        print("="*80 + "\n")

    def get_phenotype_hierarchy(self) -> Dict:
        """
        Get phenotype hierarchy (parent-child relationships).

        Returns
        -------
        dict
            Hierarchy of phenotypes
        """
        hierarchy = {}

        for name, definition in self.phenotypes_config.items():
            if 'base' in definition:
                parent = definition['base']
                if parent not in hierarchy:
                    hierarchy[parent] = []
                hierarchy[parent].append(name)

        return hierarchy

    def validate_phenotype_exists(self, phenotype_name: str) -> bool:
        """
        Check if a phenotype exists in adata.

        Parameters
        ----------
        phenotype_name : str
            Name of phenotype to check

        Returns
        -------
        bool
            True if phenotype exists
        """
        pheno_col = f'is_{phenotype_name}'
        return pheno_col in self.adata.obs.columns

    def get_tumor_definition(self) -> Dict:
        """
        Get the tumor definition from config.

        Returns
        -------
        dict
            Tumor definition parameters
        """
        if 'tumor_definition' in self.config:
            return self.config['tumor_definition']
        else:
            # Default: TOM+ defines tumor
            return {
                'base_phenotype': 'Tumor',
                'required_positive': ['TOM'],
                'required_negative': []
            }
