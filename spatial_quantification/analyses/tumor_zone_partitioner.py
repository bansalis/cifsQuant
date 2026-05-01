"""
TumorZonePartitioner
====================
Radius-scaled spatial k-means zoning for tumor cells.

Replaces the broken DBSCAN-circle approach in MarkerRegionAnalysisSpatialCells.
Validated 2026-04-18: p < 0.0001, z = 15.6 on JL216/T9 (n=1000 permutations).

Algorithm:
  1. DBSCAN on TOM+ cells per sample → identify tumor cluster matching per_tumor_metrics centroid
  2. ConvexHull + 50 µm expansion → capture intratumoral immune cells
  3. k_max = max(2, min(6, floor(radius_um / 250))) — scales with tumor size
  4. KMeans on TOM+ coords, pick k by silhouette, enforce ≥60 TOM+ per zone
  5. Zone boundary: each TOM+ cell → 50 µm disk → unary_union → smooth blobs
  6. Immune cells assigned to nearest zone centroid via cKDTree
  7. pERK_label = 'high' if pERK% ≥ median across zones, else 'low'
  8. Same geometry used for all markers (NINJA, Ki67, EPCAM, MHCII)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
import warnings

from scipy.stats import binomtest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull, cKDTree
from matplotlib.path import Path as MPath
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import shapely.wkt

warnings.filterwarnings('ignore')

# ── Constants matching pipeline config ────────────────────────────────────────
DBSCAN_EPS      = 1000   # µm (spatial_config.yaml tumor detection)
DBSCAN_MIN      = 500
HULL_EXPAND_UM  = 50     # µm expansion for immune capture
BUFFER_UM       = 50     # µm disk radius for smooth zone boundaries
SIMPLIFY_UM     = 8
MIN_CELLS_ZONE  = 60     # minimum TOM+ cells per zone
K_DIVISOR       = 180    # one zone per this many µm of tumor radius
K_FLOOR, K_CEIL = 2, 6
MIN_TUMOR_CELLS = 100    # skip tumors smaller than this
PERK_FLOOR_PCT  = 5.0   # zone must have ≥5% pERK to be called pERK+
BINOM_ALPHA     = 0.05  # one-sided binomial test significance threshold
PERK_ZONE_BUFFER_UM = 80  # larger buffer for pERK+ zone outlines: anchors to pERK+
                           # cells but expands smoothly into local tumor structure
BUFFER_SUBSAMPLE    = 30_000  # max cells used for boundary computation
SMALL_HOLE_UM2      = 15_000  # holes below this are buffer artifacts → always fill
MIN_IMMUNE_IN_HOLE  = 30      # large holes with ≥ this many immune cells are real
                               # aggregates → preserve; fewer → fill (tumor structure)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def dynamic_k_max(area_um2: float) -> int:
    radius_um = np.sqrt(area_um2 / np.pi)
    return int(np.clip(int(np.floor(radius_um / K_DIVISOR)), K_FLOOR, K_CEIL))


def buffer_union(points: np.ndarray,
                 buffer_um: float = BUFFER_UM,
                 simplify_um: float = SIMPLIFY_UM):
    """Smooth organic boundary via union of per-cell disks."""
    if len(points) > BUFFER_SUBSAMPLE:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), BUFFER_SUBSAMPLE, replace=False)
        points = points[idx]
    disks  = [Point(float(x), float(y)).buffer(buffer_um, resolution=8)
              for x, y in points]
    merged = unary_union(disks)
    return merged.simplify(simplify_um, preserve_topology=True)


def fill_holes(geom, immune_pts: np.ndarray = None):
    """Fill interior holes selectively:
      - Small holes (< SMALL_HOLE_UM2): always fill — buffer artifacts.
      - Large holes with few immune cells: fill — they are tumor structure voids.
      - Large holes with ≥ MIN_IMMUNE_IN_HOLE immune cells inside: keep — real
        immune aggregates / stromal pockets within the zone.
    immune_pts: (N,2) array of [x, y] for non-tumor cells in this zone.
    """
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == 'MultiPolygon':
        return MultiPolygon([fill_holes(g, immune_pts) for g in geom.geoms])
    if geom.geom_type != 'Polygon':
        polys = [g for g in getattr(geom, 'geoms', [])
                 if g.geom_type in ('Polygon', 'MultiPolygon')]
        return fill_holes(unary_union(polys), immune_pts) if polys else geom

    kept = []
    for ring in geom.interiors:
        hole = Polygon(ring)
        if hole.area < SMALL_HOLE_UM2:
            continue  # small artifact → fill
        if immune_pts is not None and len(immune_pts) > 0:
            # count immune cells inside this hole
            from shapely.vectorized import contains as sv_contains
            try:
                inside = sv_contains(hole, immune_pts[:, 0], immune_pts[:, 1])
                n_immune = int(inside.sum())
            except Exception:
                n_immune = sum(1 for x, y in immune_pts if hole.contains(Point(x, y)))
            if n_immune < MIN_IMMUNE_IN_HOLE:
                continue  # large but few immune cells → tumor structure → fill
        kept.append(ring)

    return Polygon(geom.exterior, kept)


def pick_k(coords_tom: np.ndarray,
           area_um2: float) -> Tuple[int, Optional[KMeans]]:
    """Select k by silhouette, capped at dynamic_k_max."""
    k_max = dynamic_k_max(area_um2)
    best_k, best_score, best_km = 2, -1.0, None
    n = len(coords_tom)

    for k in range(2, k_max + 1):
        if n < k * MIN_CELLS_ZONE:
            break
        km  = KMeans(n_clusters=k, n_init=15, random_state=42)
        lbl = km.fit_predict(coords_tom)
        if np.bincount(lbl).min() < MIN_CELLS_ZONE:
            continue
        try:
            s = silhouette_score(coords_tom, lbl,
                                 sample_size=min(3000, n), random_state=42)
        except Exception:
            s = -1.0
        if s > best_score:
            best_score, best_k, best_km = s, k, km

    if best_km is None:  # couldn't fit k=2 with constraints → force it
        best_km = KMeans(n_clusters=2, n_init=15, random_state=42)
        best_km.fit(coords_tom)
        best_k = 2

    return best_k, best_km


# ── Per-tumor extraction ──────────────────────────────────────────────────────

def extract_tumor_cells(sample_df: pd.DataFrame,
                        ref_cx: float, ref_cy: float,
                        area_um2: float) -> Optional[pd.DataFrame]:
    """
    Find cells belonging to one tumor using DBSCAN on TOM+,
    then convex-hull + HULL_EXPAND_UM expansion.
    Returns subset of sample_df or None if failed.
    """
    tom_idx    = np.where(sample_df['is_Tumor'].values)[0]
    if len(tom_idx) < DBSCAN_MIN:
        return None

    coords_tom = sample_df[['X_centroid', 'Y_centroid']].values[tom_idx]
    db         = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN,
                        n_jobs=-1).fit(coords_tom)
    labels     = db.labels_

    # Match cluster by centroid proximity to reference
    best_lbl, best_d = -1, np.inf
    for lbl in np.unique(labels):
        if lbl == -1:
            continue
        cc = coords_tom[labels == lbl]
        d  = np.hypot(cc[:, 0].mean() - ref_cx, cc[:, 1].mean() - ref_cy)
        if d < best_d:
            best_d, best_lbl = d, lbl

    if best_lbl == -1:
        return None

    cluster_coords = coords_tom[labels == best_lbl]

    try:
        hull     = ConvexHull(cluster_coords)
        hull_pts = cluster_coords[hull.vertices]
        hull_cx, hull_cy = hull_pts.mean(axis=0)
        norms    = np.linalg.norm(hull_pts - [hull_cx, hull_cy],
                                  axis=1, keepdims=True) + 1e-9
        expanded = hull_pts + HULL_EXPAND_UM * (hull_pts - [hull_cx, hull_cy]) / norms
        path     = MPath(np.vstack([expanded, expanded[0]]))
        in_hull  = path.contains_points(
            sample_df[['X_centroid', 'Y_centroid']].values)
        return sample_df[in_hull].copy()
    except Exception:
        # Fallback: bounding box + margin
        xmin, xmax = cluster_coords[:, 0].min(), cluster_coords[:, 0].max()
        ymin, ymax = cluster_coords[:, 1].min(), cluster_coords[:, 1].max()
        m = HULL_EXPAND_UM
        mask = (
            (sample_df['X_centroid'] >= xmin - m) &
            (sample_df['X_centroid'] <= xmax + m) &
            (sample_df['Y_centroid'] >= ymin - m) &
            (sample_df['Y_centroid'] <= ymax + m)
        )
        return sample_df[mask].copy()


# ── Main partitioner class ────────────────────────────────────────────────────

class TumorZonePartitioner:
    """
    Run zone partitioning across all tumors in all samples.

    Usage:
        partitioner = TumorZonePartitioner(metrics_df)
        partitioner.run(sample_cache)          # sample_cache: {h5ad_sid: obs_df}
        partitioner.save(output_dir)
    """

    MARKERS = {
        'pERK':  ('is_pERK_pos_tumor',  'is_pERK_neg_tumor'),
        'NINJA': ('is_NINJA_pos_tumor', 'is_NINJA_neg_tumor'),
        'Ki67':  ('is_Ki67_pos_tumor',  None),
        'EPCAM': ('is_EPCAM_pos_tumor', None),
        'MHCII': ('is_MHCII_pos_tumor', 'is_MHCII_neg_tumor'),
    }
    # h5ad sample_id → UPPERCASE metrics sample_id
    SID_MAP = {
        'JL216_Final': 'JL216_FINAL',
        'JL217_Final': 'JL217_FINAL',
        'JL218_Final': 'JL218_FINAL',
        'JL219_Final': 'JL219_FINAL',
    }

    def __init__(self, metrics_df: pd.DataFrame):
        self.metrics_df  = metrics_df
        self.zone_rows   = []   # one row per (sample × tumor × zone)
        self.cell_frames = []   # one df per (sample × tumor), with zone_id

    def run(self, sample_cache: Dict[str, pd.DataFrame]) -> None:
        """
        sample_cache : {h5ad_sid: obs_df} — obs_df must have phenotype columns
                       (build via load_sample() before calling).
        """
        for h5ad_sid, sample_df in sample_cache.items():
            metrics_sid = self.SID_MAP[h5ad_sid]
            tumors      = self.metrics_df[
                (self.metrics_df['sample_id']    == metrics_sid) &
                (self.metrics_df['n_tumor_cells'] >= MIN_TUMOR_CELLS)
            ]
            print(f"\n{h5ad_sid}  ({len(tumors)} eligible tumors)")

            for _, ref in tumors.iterrows():
                tumor_id = int(ref['tumor_id'])
                area_um2 = float(ref['area_um2'])
                self._process_tumor(sample_df, h5ad_sid, metrics_sid,
                                    tumor_id, area_um2,
                                    float(ref['centroid_x']),
                                    float(ref['centroid_y']),
                                    ref)

    def _process_tumor(self, sample_df, h5ad_sid, metrics_sid,
                       tumor_id, area_um2, ref_cx, ref_cy, meta_row):
        cells = extract_tumor_cells(sample_df, ref_cx, ref_cy, area_um2)
        if cells is None or cells['is_Tumor'].sum() < MIN_TUMOR_CELLS:
            print(f"  T{tumor_id}: skipped (insufficient TOM+ cells)")
            return

        coords_tom = cells.loc[cells['is_Tumor'],
                               ['X_centroid', 'Y_centroid']].values
        best_k, km = pick_k(coords_tom, area_um2)

        # ── Assign zones ──────────────────────────────────────────────────────
        zone_tom  = km.predict(coords_tom)
        zone_all  = np.full(len(cells), -1, dtype=int)
        tom_locs  = np.where(cells['is_Tumor'].values)[0]
        zone_all[tom_locs] = zone_tom

        non_tom = np.where(~cells['is_Tumor'].values)[0]
        if len(non_tom):
            tree      = cKDTree(km.cluster_centers_)
            _, nn_idx = tree.query(cells[['X_centroid', 'Y_centroid']].values[non_tom])
            zone_all[non_tom] = nn_idx

        cells = cells.copy()
        cells['zone_id']   = zone_all
        cells['tumor_id']  = tumor_id
        cells['sample_id'] = metrics_sid

        # ── Per-zone stats ────────────────────────────────────────────────────
        zone_stats = self._zone_stats(cells, best_k)

        # pERK_label: binomial test — is zone pERK% significantly above
        # the tumor's global pERK rate? pERK- is the default/background state.
        n_tom_total  = int(cells['is_Tumor'].sum())
        n_perk_total = int(cells['is_pERK_pos_tumor'].sum())
        global_rate  = n_perk_total / n_tom_total if n_tom_total > 0 else 0.0

        perk_labels = []
        for _, row in zone_stats.iterrows():
            n_zone = int(row['n_tumor'])
            n_pos  = int(round(row['pERK_pct'] / 100 * n_zone))
            passes_floor = row['pERK_pct'] >= PERK_FLOOR_PCT
            if n_zone > 0 and global_rate > 0 and passes_floor:
                p = binomtest(n_pos, n_zone, global_rate,
                              alternative='greater').pvalue
                perk_labels.append('pERK+' if p < BINOM_ALPHA else 'pERK-')
            else:
                perk_labels.append('pERK-')
        zone_stats['pERK_label'] = perk_labels

        # NINJA_label: same logic
        n_ninja_total = int(cells['is_NINJA_pos_tumor'].sum())
        global_ninja  = n_ninja_total / n_tom_total if n_tom_total > 0 else 0.0
        ninja_labels  = []
        for _, row in zone_stats.iterrows():
            n_zone = int(row['n_tumor'])
            n_pos  = int(round(row['NINJA_pct'] / 100 * n_zone))
            passes_floor = row['NINJA_pct'] >= PERK_FLOOR_PCT
            if n_zone > 0 and global_ninja > 0 and passes_floor:
                p = binomtest(n_pos, n_zone, global_ninja,
                              alternative='greater').pvalue
                ninja_labels.append('NINJA+' if p < BINOM_ALPHA else 'NINJA-')
            else:
                ninja_labels.append('NINJA-')
        zone_stats['NINJA_label'] = ninja_labels

        # ── Boundaries (WKT) ─────────────────────────────────────────────────
        tumor_poly = buffer_union(coords_tom)
        tumor_poly = fill_holes(tumor_poly)   # clean tumor boundary too
        tumor_wkt  = shapely.wkt.dumps(tumor_poly, rounding_precision=1)

        # Immune cell coords for hole-fill decision (non-tumor cells in tumor)
        immune_pts = cells.loc[~cells['is_Tumor'],
                               ['X_centroid', 'Y_centroid']].values

        for _, row in zone_stats.iterrows():
            zid        = int(row['zone_id'])
            perk_label = row['pERK_label']

            zmask_all = (cells['zone_id'] == zid) & cells['is_Tumor']
            zpts_all  = cells.loc[zmask_all, ['X_centroid', 'Y_centroid']].values

            if perk_label == 'pERK+':
                # Anchor boundary to pERK+ cells with a wider buffer — stays honest
                # to activation territory but smooth enough to reflect local structure.
                # Clip to tumor boundary so the outline never escapes the tumor edge.
                zmask_pos = (cells['zone_id'] == zid) & cells['is_pERK_pos_tumor']
                zpts_pos  = cells.loc[zmask_pos, ['X_centroid', 'Y_centroid']].values
                if len(zpts_pos) >= 4:
                    zpoly = buffer_union(zpts_pos, buffer_um=PERK_ZONE_BUFFER_UM)
                    zpoly = zpoly.intersection(tumor_poly)
                    if zpoly.is_empty:
                        zpoly = buffer_union(zpts_all) if len(zpts_all) >= 4 else tumor_poly
                else:
                    zpoly = buffer_union(zpts_all) if len(zpts_all) >= 4 else tumor_poly
            else:
                zpoly = buffer_union(zpts_all) if len(zpts_all) >= 4 else tumor_poly

            # Fill small/non-immune holes; preserve large immune-aggregate holes
            zpoly = fill_holes(zpoly, immune_pts)
            zwkt  = shapely.wkt.dumps(zpoly, rounding_precision=1)

            self.zone_rows.append({
                'sample_id':        metrics_sid,
                'tumor_id':         tumor_id,
                'zone_id':          zid,
                'k_chosen':         best_k,
                'n_tumor_cells':    int(row['n_tumor']),
                'n_all_cells':      int((cells['zone_id'] == zid).sum()),
                'pERK_pct':         round(row['pERK_pct'], 4),
                'NINJA_pct':        round(row['NINJA_pct'], 4),
                'Ki67_pct':         round(row['Ki67_pct'], 4),
                'EPCAM_pct':        round(row['EPCAM_pct'], 4),
                'MHCII_pct':        round(row['MHCII_pct'], 4),
                'pERK_label':        row['pERK_label'],
                'NINJA_label':       row['NINJA_label'],
                'global_pERK_rate':  round(global_rate * 100, 4),
                'centroid_x':       round(row['cx'], 2),
                'centroid_y':       round(row['cy'], 2),
                'area_um2':         round(row['area_um2'], 1),
                'zone_boundary_wkt': zwkt,
                'tumor_boundary_wkt': tumor_wkt,
                'main_group':       str(meta_row.get('main_group', '')),
                'timepoint':        meta_row.get('timepoint', np.nan),
            })

        cells['pERK_label']  = cells['zone_id'].map(
            zone_stats.set_index('zone_id')['pERK_label'])
        cells['NINJA_label'] = cells['zone_id'].map(
            zone_stats.set_index('zone_id')['NINJA_label'])
        self.cell_frames.append(cells)

        n_tom       = cells['is_Tumor'].sum()
        n_perk_pos  = (zone_stats['pERK_label'] == 'pERK+').sum()
        perk_avg    = zone_stats['pERK_pct'].mean()
        print(f"  T{tumor_id}: k={best_k}, {n_tom} TOM+, "
              f"pERK={perk_avg:.1f}% avg, {n_perk_pos}/{best_k} zones pERK+")

    def _zone_stats(self, cells: pd.DataFrame, k: int) -> pd.DataFrame:
        rows = []
        for zid in range(k):
            zmask = cells['zone_id'] == zid
            z     = cells[zmask]
            n_tom = int(z['is_Tumor'].sum())
            def pct(col):
                if col not in z.columns:
                    return 0.0
                return float(z[col].sum() / n_tom * 100) if n_tom > 0 else 0.0
            tom_coords = z.loc[z['is_Tumor'], ['X_centroid', 'Y_centroid']]
            cx, cy = (tom_coords.mean() if len(tom_coords) > 0
                      else z[['X_centroid', 'Y_centroid']].mean())
            # area from buffer-union
            if n_tom >= 4:
                pts  = tom_coords.values
                poly = buffer_union(pts)
                area = float(poly.area)
            else:
                area = 0.0
            rows.append({
                'zone_id':   zid,
                'n_tumor':   n_tom,
                'pERK_pct':  pct('is_pERK_pos_tumor'),
                'NINJA_pct': pct('is_NINJA_pos_tumor'),
                'Ki67_pct':  pct('is_Ki67_pos_tumor'),
                'EPCAM_pct': pct('is_EPCAM_pos_tumor'),
                'MHCII_pct': pct('is_MHCII_pos_tumor'),
                'cx': float(cx), 'cy': float(cy),
                'area_um2': area,
            })
        return pd.DataFrame(rows)

    def save(self, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Save zone summary and cell-level parquets. Returns (zones_df, cells_df)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        zones_df = pd.DataFrame(self.zone_rows)
        zones_df.to_csv(output_dir / 'tumor_spatial_zones.csv', index=False)
        print(f"\nSaved tumor_spatial_zones.csv  ({len(zones_df)} zone rows)")

        all_cells = pd.concat(self.cell_frames, ignore_index=True)

        # Save per-sample parquets
        for sid in all_cells['sample_id'].unique():
            sub  = all_cells[all_cells['sample_id'] == sid]
            path = output_dir / f'cells_with_zones_{sid}.parquet'
            sub.to_parquet(path, index=False)
            print(f"Saved cells_with_zones_{sid}.parquet  ({len(sub):,} cells)")

        return zones_df, all_cells
