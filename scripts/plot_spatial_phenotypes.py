"""
High-resolution spatial phenotype plots for batch25.

Produces:
  1. Per-sample merged plot — all phenotypes color-coded on one axes
  2. Per-sample per-phenotype plots — each phenotype highlighted vs grey background

Output: /mnt/e/IB/batch25quant/spatial_pipeline/spatial_phenotype_plots/
"""

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
H5AD   = Path('/mnt/e/IB/batch25quant/manual_gating_output/gated_data.h5ad')
OUTDIR = Path('/mnt/e/IB/batch25quant/spatial_pipeline/spatial_phenotype_plots')
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── phenotype hierarchy (ordered: first match wins) ──────────────────────────
# Each entry: (label, {marker: required_value, ...})
# required_value: 1 = positive, 0 = negative
PHENOTYPE_HIERARCHY = [
    # T cell subsets (specific first)
    ('Tregs',          {'CD3E': 1, 'CD4': 1, 'FOXP3': 1}),
    ('Exhausted CD8',  {'CD3E': 1, 'CD8A': 1, 'PD1': 1}),
    ('CD8 T',          {'CD3E': 1, 'CD8A': 1}),
    ('CD4 T',          {'CD3E': 1, 'CD4': 1}),
    ('T cells',        {'CD3E': 1}),
    # Myeloid
    ('MHCII+ Mac',     {'F480': 1, 'MHCII': 1}),
    ('Macrophages',    {'F480': 1}),
    # B cells
    ('B cells',        {'B220': 1}),
    # Tumor subsets (specific first)
    ('pERK+ Tumor',    {'TOM': 1, 'PERK': 1}),
    ('NINJA+ Tumor',   {'TOM': 1, 'NINJA': 1}),
    ('Ki67+ Tumor',    {'TOM': 1, 'KI67': 1}),
    ('PDL1+ Tumor',    {'TOM': 1, 'PDL1': 1}),
    ('Tumor',          {'TOM': 1}),
    # Catch-all immune
    ('CD45+',          {'CD45': 1}),
    # Everything else
    ('Other',          {}),
]

# Colorblind-friendly palette
PHENOTYPE_COLORS = {
    'Tregs':          '#7B2D8B',   # purple
    'Exhausted CD8':  '#E41A1C',   # red
    'CD8 T':          '#FF7F00',   # orange
    'CD4 T':          '#377EB8',   # blue
    'T cells':        '#A6CEE3',   # light blue
    'MHCII+ Mac':     '#33A02C',   # dark green
    'Macrophages':    '#B2DF8A',   # light green
    'B cells':        '#1F78B4',   # deep blue
    'pERK+ Tumor':    '#FB9A99',   # pink
    'NINJA+ Tumor':   '#FDBF6F',   # peach
    'Ki67+ Tumor':    '#CAB2D6',   # lavender
    'PDL1+ Tumor':    '#FFFF99',   # yellow
    'Tumor':          '#E31A1C',   # crimson
    'CD45+':          '#6A3D9A',   # dark purple
    'Other':          '#CCCCCC',   # grey
}

# Per-phenotype solo plots (channel-level, simpler definitions)
SOLO_PHENOTYPES = [
    ('Tumor',         {'TOM': 1}),
    ('pERK+ Tumor',   {'TOM': 1, 'PERK': 1}),
    ('NINJA+ Tumor',  {'TOM': 1, 'NINJA': 1}),
    ('Ki67+ Tumor',   {'TOM': 1, 'KI67': 1}),
    ('PDL1+ Tumor',   {'TOM': 1, 'PDL1': 1}),
    ('T cells',       {'CD3E': 1}),
    ('CD4 T',         {'CD3E': 1, 'CD4': 1}),
    ('CD8 T',         {'CD3E': 1, 'CD8A': 1}),
    ('Tregs',         {'CD3E': 1, 'CD4': 1, 'FOXP3': 1}),
    ('Exhausted CD8', {'CD3E': 1, 'CD8A': 1, 'PD1': 1}),
    ('Macrophages',   {'F480': 1}),
    ('MHCII+ Mac',    {'F480': 1, 'MHCII': 1}),
    ('B cells',       {'B220': 1}),
    ('CD45+',         {'CD45': 1}),
    ('KI67',          {'KI67': 1}),
    ('PD1',           {'PD1': 1}),
    ('BCL6',          {'BCL6': 1}),
    ('GZMB',          {'GZMB': 1}),
]

POINT_SIZE   = 0.15   # scatter point size — tiny so dense regions stay legible
ALPHA_FG     = 0.6    # foreground (phenotype-positive) alpha
ALPHA_BG     = 0.04   # background (other cells) alpha in solo plots
BG_COLOR     = '#AAAAAA'

# DPI and figure inches — sized so full-tissue width ≈ 12,000 px
# tissue max width ≈ 120,000 units; aspect ≈ 2:1
FIG_W_INCHES = 40
DPI          = 300    # → 12,000 px wide


def load_data():
    print("Loading h5ad …")
    adata = ad.read_h5ad(H5AD)
    print(f"  {adata.n_obs:,} cells × {adata.n_vars} markers")

    # Build gated dataframe
    gated = pd.DataFrame(
        adata.layers['gated'].astype(np.int8),
        index=adata.obs_names,
        columns=adata.var_names,
    )
    obs = adata.obs[['sample_id', 'X_centroid', 'Y_centroid']].copy()
    obs = obs.join(gated)
    return obs


def assign_phenotype(obs: pd.DataFrame) -> pd.Series:
    """Assign each cell its first-match phenotype from PHENOTYPE_HIERARCHY."""
    phenotype = pd.Series('Other', index=obs.index)
    assigned   = pd.Series(False, index=obs.index)

    for label, reqs in PHENOTYPE_HIERARCHY:
        if not reqs:  # 'Other' catch-all
            phenotype[~assigned] = 'Other'
            break
        mask = pd.Series(True, index=obs.index)
        for marker, val in reqs.items():
            if marker in obs.columns:
                mask &= (obs[marker] == val)
        new = mask & ~assigned
        phenotype[new] = label
        assigned |= new

    return phenotype


def sample_fig_size(x, y):
    """Return (w, h) in inches keeping FIG_W_INCHES wide."""
    w_range = x.max() - x.min()
    h_range = y.max() - y.min()
    ratio   = h_range / w_range if w_range > 0 else 0.5
    return FIG_W_INCHES, max(FIG_W_INCHES * ratio, 4)


def plot_merged(obs: pd.DataFrame, sample: str):
    """All phenotypes on one axes, colour-coded."""
    sub = obs[obs['sample_id'] == sample].copy()
    sub['phenotype'] = assign_phenotype(sub)

    x, y = sub['X_centroid'].values, sub['Y_centroid'].values
    fw, fh = sample_fig_size(pd.Series(x), pd.Series(y))

    fig, ax = plt.subplots(figsize=(fw, fh), dpi=DPI)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Plot rarest phenotypes last (on top)
    counts = sub['phenotype'].value_counts()
    order  = counts.index.tolist()  # most common first → rarest on top
    order.reverse()

    for ph in reversed(order):           # most common drawn first (bottom)
        mask = sub['phenotype'] == ph
        c    = PHENOTYPE_COLORS.get(ph, '#FFFFFF')
        ax.scatter(
            sub.loc[mask, 'X_centroid'],
            sub.loc[mask, 'Y_centroid'],
            c=c, s=POINT_SIZE, alpha=ALPHA_FG,
            linewidths=0, rasterized=True,
        )

    # Legend
    handles = [
        mpatches.Patch(color=PHENOTYPE_COLORS.get(ph, '#FFFFFF'), label=f"{ph} ({counts.get(ph, 0):,})")
        for ph in PHENOTYPE_COLORS if ph in sub['phenotype'].values
    ]
    leg = ax.legend(
        handles=handles, loc='upper right',
        fontsize=7, framealpha=0.6,
        facecolor='#222222', labelcolor='white',
        markerscale=4,
    )

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title(f"{sample} — all phenotypes  ({len(sub):,} cells)", color='white', fontsize=14, pad=8)

    outpath = OUTDIR / f"{sample}_merged_phenotypes.png"
    fig.savefig(outpath, dpi=DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {outpath.name}")


def plot_solo(obs: pd.DataFrame, sample: str, label: str, reqs: dict):
    """Single phenotype highlighted on grey/black background."""
    sub = obs[obs['sample_id'] == sample].copy()

    # Build phenotype mask
    mask = pd.Series(True, index=sub.index)
    for marker, val in reqs.items():
        if marker in sub.columns:
            mask &= (sub[marker] == val)
        else:
            print(f"    WARNING: marker {marker} not in data, skipping {label}")
            return

    n_pos = mask.sum()
    n_tot = len(sub)

    x, y = sub['X_centroid'].values, sub['Y_centroid'].values
    fw, fh = sample_fig_size(pd.Series(x), pd.Series(y))

    fig, ax = plt.subplots(figsize=(fw, fh), dpi=DPI)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Background cells
    ax.scatter(
        sub.loc[~mask, 'X_centroid'],
        sub.loc[~mask, 'Y_centroid'],
        c=BG_COLOR, s=POINT_SIZE, alpha=ALPHA_BG,
        linewidths=0, rasterized=True,
    )
    # Foreground cells
    if n_pos > 0:
        c = PHENOTYPE_COLORS.get(label, '#FF4444')
        ax.scatter(
            sub.loc[mask, 'X_centroid'],
            sub.loc[mask, 'Y_centroid'],
            c=c, s=POINT_SIZE * 2, alpha=min(ALPHA_FG * 1.5, 1.0),
            linewidths=0, rasterized=True,
        )

    pct = 100 * n_pos / n_tot if n_tot > 0 else 0
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title(
        f"{sample} — {label}  ({n_pos:,} / {n_tot:,} cells, {pct:.1f}%)",
        color='white', fontsize=14, pad=8,
    )

    safe_label = label.replace(' ', '_').replace('+', 'pos').replace('/', '_')
    outpath = OUTDIR / f"{sample}_{safe_label}.png"
    fig.savefig(outpath, dpi=DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {outpath.name}  ({n_pos:,} cells)")


def main():
    obs = load_data()
    samples = obs['sample_id'].unique().tolist()
    print(f"\nSamples: {samples}")

    for sample in samples:
        print(f"\n{'='*60}")
        print(f"Sample: {sample}")
        print(f"{'='*60}")

        print("  → merged phenotype plot")
        plot_merged(obs, sample)

        for label, reqs in SOLO_PHENOTYPES:
            print(f"  → {label}")
            plot_solo(obs, sample, label, reqs)

    print(f"\nDone. All plots in: {OUTDIR}")


if __name__ == '__main__':
    main()
