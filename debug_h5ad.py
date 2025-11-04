#!/usr/bin/env python3
"""Quick debug script to inspect gated h5ad file structure"""

import anndata as ad
import sys

h5ad_path = sys.argv[1] if len(sys.argv) > 1 else 'manual_gating_output/gated_data.h5ad'

print(f"\nInspecting: {h5ad_path}")
print("="*80)

# Load data
adata = ad.read_h5ad(h5ad_path)

print(f"\nTotal cells: {len(adata):,}")
print(f"Total features: {adata.n_vars}")

print("\n" + "="*80)
print("adata.obs columns (first 50):")
print("="*80)
for i, col in enumerate(adata.obs.columns[:50], 1):
    dtype = adata.obs[col].dtype
    n_unique = adata.obs[col].nunique()
    print(f"{i:3d}. {col:40s} | dtype: {str(dtype):10s} | unique: {n_unique}")

print("\n" + "="*80)
print("Looking for gated columns (is_*):")
print("="*80)
gated_cols = [col for col in adata.obs.columns if col.startswith('is_')]
if gated_cols:
    print(f"✓ Found {len(gated_cols)} gated columns:")
    for col in gated_cols:
        count = adata.obs[col].sum() if adata.obs[col].dtype == bool else 'N/A'
        print(f"  - {col}: {count:,} cells" if isinstance(count, int) else f"  - {col}: {count}")
else:
    print("✗ NO gated columns found!")
    print("\nSearching for potential marker columns:")
    marker_names = ['TOM', 'CD45', 'AGFP', 'PERK', 'CD8B', 'KI67', 'CD3']
    for marker in marker_names:
        matches = [col for col in adata.obs.columns if marker.upper() in col.upper()]
        if matches:
            print(f"  {marker}: {matches}")

print("\n" + "="*80)
print("adata.var_names (markers):")
print("="*80)
print(list(adata.var_names))

print("\n" + "="*80)
print("Available attributes:")
print("="*80)
print(f"obsm keys: {list(adata.obsm.keys())}")
print(f"uns keys: {list(adata.uns.keys())}")
print(f"varm keys: {list(adata.varm.keys())}")

print("\n" + "="*80)
print("Sample IDs:")
print("="*80)
if 'sample_id' in adata.obs.columns:
    print(f"Unique samples: {adata.obs['sample_id'].nunique()}")
    print(f"Samples: {sorted(adata.obs['sample_id'].unique())}")
else:
    print("✗ No 'sample_id' column found")
    print("Available columns that might be sample IDs:")
    for col in adata.obs.columns:
        if 'sample' in col.lower() or 'id' in col.lower():
            print(f"  - {col}")

print("\n" + "="*80 + "\n")
