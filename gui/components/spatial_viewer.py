"""Load and cache AnnData for the GUI. Provides data access helpers."""
import numpy as np
from pathlib import Path

try:
    import streamlit as st
    _cache = st.cache_data
except ImportError:
    # Allow importing outside Streamlit (e.g. in tests)
    def _cache(func=None, **kwargs):
        if func is not None:
            return func
        return lambda f: f


@_cache(show_spinner='Loading gated data…', ttl=300)
def load_adata(h5ad_path: str):
    import anndata
    return anndata.read_h5ad(h5ad_path)


@_cache(show_spinner='Loading checkpoint…', ttl=300)
def load_normalized(h5ad_path: str):
    import anndata
    return anndata.read_h5ad(h5ad_path)


def get_samples(adata) -> list[str]:
    if 'sample_id' in adata.obs.columns:
        return sorted(adata.obs['sample_id'].unique().tolist())
    return ['all']


def get_marker_values(adata, marker: str, sample_id: str | None = None) -> np.ndarray:
    """Return aligned (normalized) intensities for a marker, optionally per sample."""
    mask = slice(None)
    if sample_id and sample_id != 'all' and 'sample_id' in adata.obs.columns:
        mask = adata.obs['sample_id'] == sample_id

    if 'aligned' in adata.layers:
        layer = adata.layers['aligned']
    else:
        layer = adata.X

    if marker in adata.var_names:
        idx = list(adata.var_names).index(marker)
        vals = np.asarray(layer[mask, idx]).flatten()
    else:
        return np.array([])
    return vals


def get_spatial_coords(adata, sample_id: str | None = None):
    """Return (x, y) arrays for cells, optionally filtered by sample."""
    if sample_id and sample_id != 'all' and 'sample_id' in adata.obs.columns:
        mask = adata.obs['sample_id'] == sample_id
        obs = adata.obs[mask]
    else:
        obs = adata.obs

    if 'X_centroid' in obs.columns and 'Y_centroid' in obs.columns:
        return obs['X_centroid'].values, obs['Y_centroid'].values
    elif hasattr(adata, 'obsm') and 'spatial' in adata.obsm:
        spatial = adata.obsm['spatial']
        if sample_id and sample_id != 'all' and 'sample_id' in adata.obs.columns:
            mask = (adata.obs['sample_id'] == sample_id).values
            spatial = spatial[mask]
        return spatial[:, 0], spatial[:, 1]
    return np.zeros(len(obs)), np.zeros(len(obs))


def get_gate_mask(adata, marker: str, threshold: float, sample_id: str | None = None) -> np.ndarray:
    """Return boolean array: True where marker >= threshold."""
    values = get_marker_values(adata, marker, sample_id)
    return values >= threshold


def get_phenotype_mask(adata, phenotype_col: str, sample_id: str | None = None) -> np.ndarray:
    """Return boolean array for a precomputed is_{phenotype} column."""
    if sample_id and sample_id != 'all' and 'sample_id' in adata.obs.columns:
        mask = adata.obs['sample_id'] == sample_id
        obs = adata.obs[mask]
    else:
        obs = adata.obs

    if phenotype_col in obs.columns:
        return obs[phenotype_col].astype(bool).values
    return np.zeros(len(obs), dtype=bool)


def find_h5ad(project_dir: Path, name: str) -> Path | None:
    candidates = [
        project_dir / 'manual_gating_output' / name,
        project_dir / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None
