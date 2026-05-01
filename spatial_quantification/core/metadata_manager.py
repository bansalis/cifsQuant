"""
Metadata Manager for Spatial Quantification
Handles flexible metadata with custom groupings
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings


class MetadataManager:
    """Manage and process metadata with flexible groupings."""

    def __init__(self, metadata: pd.DataFrame, config: Dict):
        """
        Initialize metadata manager.

        Parameters
        ----------
        metadata : pd.DataFrame
            Sample metadata
        config : dict
            Configuration dictionary
        """
        self.metadata = metadata.copy()
        self.config = config
        self.meta_config = config['metadata']

    def process(self) -> pd.DataFrame:
        """
        Process metadata: extract groupings, add custom columns.

        Returns
        -------
        pd.DataFrame
            Processed metadata with all grouping columns
        """
        print("\nProcessing metadata...")

        # Standardize column names
        sample_col = self.meta_config['sample_column']
        group_col = self.meta_config['group_column']
        timepoint_col = self.meta_config['timepoint_column']
        treatment_col = self.meta_config.get('treatment_column', 'treatment')

        # Ensure sample_id is uppercase
        self.metadata[sample_col] = self.metadata[sample_col].str.upper()

        # Extract additional groupings
        if 'additional_groupings' in self.meta_config:
            for grouping in self.meta_config['additional_groupings']:
                if grouping not in self.metadata.columns:
                    self._extract_grouping(grouping, group_col)

        # Process custom groupings if specified
        if 'custom_groupings' in self.meta_config:
            self._add_custom_groupings()

        # Convert timepoint to numeric
        self.metadata[timepoint_col] = pd.to_numeric(
            self.metadata[timepoint_col],
            errors='coerce'
        )

        # Convert treatment column to string category if present
        if treatment_col in self.metadata.columns:
            self.metadata[treatment_col] = self.metadata[treatment_col].astype(str)
            print(f"  ✓ Treatment column '{treatment_col}' detected: "
                  f"{sorted(self.metadata[treatment_col].unique())}")

        # Validate
        self._validate()

        # Print summary
        self._print_summary()

        return self.metadata

    def _extract_grouping(self, grouping: str, group_col: str):
        """Extract a grouping from the main group column."""
        if grouping == 'main_group':
            # Extract KPT vs KPNT
            self.metadata['main_group'] = self.metadata[group_col].apply(
                lambda x: 'KPT' if 'KPT' in str(x).upper() else 'KPNT'
            )
            print(f"  ✓ Extracted main_group (KPT/KPNT) from {group_col}")

        elif grouping == 'genotype':
            # Extract cis vs trans
            self.metadata['genotype'] = self.metadata[group_col].apply(
                lambda x: 'cis' if 'cis' in str(x).lower() else
                         ('trans' if 'trans' in str(x).lower() else 'unknown')
            )
            print(f"  ✓ Extracted genotype (cis/trans) from {group_col}")

        elif grouping == 'genotype_full':
            # Create full genotype: KPT-cis, KPT-trans, KPNT-cis, KPNT-trans
            if 'main_group' not in self.metadata.columns:
                self._extract_grouping('main_group', group_col)
            if 'genotype' not in self.metadata.columns:
                self._extract_grouping('genotype', group_col)

            self.metadata['genotype_full'] = self.metadata.apply(
                lambda row: f"{row['main_group']}-{row['genotype']}", axis=1
            )
            print(f"  ✓ Created genotype_full (4-way comparison)")

        else:
            # Check if column already exists
            if grouping not in self.metadata.columns:
                warnings.warn(f"Grouping '{grouping}' not found in metadata and cannot be auto-extracted")

    def _add_custom_groupings(self):
        """Add custom groupings from config."""
        custom_groupings = self.meta_config['custom_groupings']

        for grouping_name, grouping_config in custom_groupings.items():
            if 'sample_mapping' in grouping_config:
                mapping = grouping_config['sample_mapping']
                sample_col = self.meta_config['sample_column']

                self.metadata[grouping_name] = self.metadata[sample_col].map(mapping)
                print(f"  ✓ Added custom grouping: {grouping_name}")

    def _validate(self):
        """Validate processed metadata."""
        # Check for missing values in critical columns
        sample_col = self.meta_config['sample_column']
        group_col = self.meta_config['group_column']
        timepoint_col = self.meta_config['timepoint_column']

        for col in [sample_col, group_col, timepoint_col]:
            if self.metadata[col].isna().any():
                n_missing = self.metadata[col].isna().sum()
                warnings.warn(f"{n_missing} missing values in {col}")

        # Check for duplicate sample IDs
        duplicates = self.metadata[sample_col].duplicated()
        if duplicates.any():
            dup_samples = self.metadata[sample_col][duplicates].unique()
            raise ValueError(f"Duplicate sample IDs found: {dup_samples}")

    def _print_summary(self):
        """Print metadata summary."""
        sample_col = self.meta_config['sample_column']
        group_col = self.meta_config['group_column']
        timepoint_col = self.meta_config['timepoint_column']
        treatment_col = self.meta_config.get('treatment_column', 'treatment')

        print("\n" + "="*80)
        print("METADATA SUMMARY")
        print("="*80)
        print(f"Total samples: {len(self.metadata)}")
        print(f"\nGroups ({group_col}):")
        for group, count in self.metadata[group_col].value_counts().items():
            print(f"  - {group}: {count} samples")

        if 'main_group' in self.metadata.columns:
            print(f"\nMain groups (KPT/KPNT):")
            for group, count in self.metadata['main_group'].value_counts().items():
                print(f"  - {group}: {count} samples")

        print(f"\nTimepoints ({timepoint_col}):")
        timepoints = sorted(self.metadata[timepoint_col].dropna().unique())
        print(f"  - {timepoints}")

        if treatment_col in self.metadata.columns:
            print(f"\nTreatment ({treatment_col}):")
            for treatment, count in self.metadata[treatment_col].value_counts().items():
                print(f"  - {treatment}: {count} samples")

        print("\nAvailable grouping columns:")
        grouping_cols = [col for col in self.metadata.columns
                        if col not in [sample_col, 'X_centroid', 'Y_centroid']]
        for col in grouping_cols:
            n_unique = self.metadata[col].nunique()
            print(f"  - {col}: {n_unique} unique values")

        print("="*80 + "\n")

    def merge_with_adata(self, adata):
        """
        Merge metadata into adata.obs.

        Parameters
        ----------
        adata : AnnData
            AnnData object to merge metadata into

        Returns
        -------
        AnnData
            AnnData with merged metadata
        """
        print("\nMerging metadata into adata.obs...")

        sample_col = self.meta_config['sample_column']

        # Ensure sample_id column exists and is uppercase
        if 'sample_id' not in adata.obs.columns:
            raise ValueError("adata.obs must have 'sample_id' column")

        adata.obs['sample_id'] = adata.obs['sample_id'].str.upper()

        # Create mappings for each metadata column
        for col in self.metadata.columns:
            if col != sample_col:
                mapping = dict(zip(
                    self.metadata[sample_col],
                    self.metadata[col]
                ))
                adata.obs[col] = adata.obs['sample_id'].map(mapping)

        # Validate merge
        n_missing = adata.obs[self.meta_config['group_column']].isna().sum()
        if n_missing > 0:
            pct_missing = 100 * n_missing / len(adata)
            warnings.warn(f"{n_missing} cells ({pct_missing:.1f}%) have no metadata")

        print(f"  ✓ Merged {len(self.metadata.columns)} metadata columns")

        return adata
