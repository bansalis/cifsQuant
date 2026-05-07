"""Load, save, and validate project.yaml configs."""
import yaml
from pathlib import Path


def load_project(path: str | Path) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def save_project(config: dict, path: str | Path):
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def validate_project(config: dict, stages: list | None = None) -> list[dict]:
    """Return list of {check, status, message} dicts.

    stages: which pipeline stages to validate for. Defaults to all three.
    Segmentation-only runs skip gating/spatial section checks.
    """
    if stages is None:
        stages = ['segmentation', 'gating', 'spatial']

    results = []

    def ok(check, msg=''):
        results.append({'check': check, 'status': 'ok', 'message': msg})

    def fail(check, msg):
        results.append({'check': check, 'status': 'error', 'message': msg})

    def warn(check, msg):
        results.append({'check': check, 'status': 'warn', 'message': msg})

    markers = config.get('markers', {})
    display_names = set(markers.values())

    if not markers:
        fail('Panel defined', 'No markers found. Define your panel on Page 1.')
    else:
        ok('Panel defined', f'{len(markers)} channels mapped')

    # Gating checks — only relevant when gating or spatial stages run
    if 'gating' in stages or 'spatial' in stages:
        gating = config.get('gating', {})
        if not gating:
            warn('Gating section', 'No gating: block — will use auto-calculated gates')
        else:
            gates = gating.get('gates', {})
            bad_gates = [k for k in gates if k not in display_names]
            if bad_gates:
                fail('Gate keys match panel', f'Unknown markers in gates: {bad_gates}. Update on Page 1 or Page 2.')
            else:
                ok('Gate keys match panel', f'{len(gates)} gates configured')

            tc = gating.get('tile_correction', {})
            if tc.get('enabled'):
                bad_tc = [m for m in tc.get('markers', []) if m not in display_names]
                if bad_tc:
                    fail('Tile correction markers', f'Unknown: {bad_tc}')
                else:
                    ok('Tile correction markers')

            lg = gating.get('liberal_gating', {})
            if lg.get('enabled'):
                bad_lg = [m for m in lg.get('liberal_markers', []) if m not in display_names]
                if bad_lg:
                    fail('Liberal gating markers', f'Unknown: {bad_lg}')
                else:
                    ok('Liberal gating markers')

        hierarchy = config.get('marker_hierarchy', {})
        bad_hier = []
        for child, parent in hierarchy.items():
            if child not in display_names:
                bad_hier.append(f"child '{child}' not in panel")
            if parent is not None and parent not in display_names:
                bad_hier.append(f"parent '{parent}' not in panel")
        if bad_hier:
            fail('Marker hierarchy', '; '.join(bad_hier))
        elif hierarchy:
            ok('Marker hierarchy', f'{len(hierarchy)} constraints')

    # Spatial checks — only when running spatial analysis
    if 'spatial' in stages:
        spatial = config.get('spatial', {})
        phenotypes = spatial.get('phenotypes', {})
        bad_pheno = []
        for name, defn in phenotypes.items():
            for key in ('positive', 'negative', 'anypos'):
                for m in defn.get(key, []):
                    if m not in display_names:
                        bad_pheno.append(f"'{name}.{key}': unknown marker '{m}'")
            if 'base' in defn and defn['base'] not in phenotypes:
                bad_pheno.append(f"'{name}.base': unknown phenotype '{defn['base']}'")
        if bad_pheno:
            fail('Phenotype markers', '; '.join(bad_pheno[:3]) + ('…' if len(bad_pheno) > 3 else ''))
        elif phenotypes:
            ok('Phenotypes', f'{len(phenotypes)} defined')

        per_tumor = spatial.get('per_tumor_analysis', {}).get('enabled', False)
        for dep in ('immune_infiltration', 'tumor_microenvironment', 'cluster_composition_analysis', 'enhanced_neighborhoods'):
            if spatial.get(dep, {}).get('enabled', False) and not per_tumor:
                warn('Analysis dependencies', f"'{dep}' requires per_tumor_analysis — enable it first")

    return results


def detect_channels_from_results(project_dir: Path) -> list[str]:
    """Sniff column names from the first segmentation CSV found in results/."""
    results_dir = project_dir / 'results'
    if not results_dir.exists():
        return []
    for csv_path in results_dir.rglob('combined_quantification.csv'):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, nrows=0)
            cols = [c for c in df.columns if c not in ('X_centroid', 'Y_centroid', 'cell_id', 'area', 'tile_x', 'tile_y')]
            return cols
        except Exception:
            continue
    return []


def get_display_names(config: dict) -> list[str]:
    return list(config.get('markers', {}).values())


def get_analysis_list() -> list[dict]:
    """Return metadata for all 22 analysis modules."""
    return [
        {'key': 'per_tumor_analysis',            'label': 'Per-Structure Analysis',               'requires': []},
        {'key': 'population_dynamics',            'label': 'Population Dynamics',                  'requires': []},
        {'key': 'distance_analysis',              'label': 'Distance Analysis',                    'requires': []},
        {'key': 'immune_infiltration',            'label': 'Immune Infiltration',                  'requires': ['per_tumor_analysis']},
        {'key': 'spatial_permutation',            'label': 'Spatial Permutation Testing',          'requires': []},
        {'key': 'distance_permutation_testing',   'label': 'Distance Permutation Testing',         'requires': []},
        {'key': 'neighborhood_permutation_testing','label': 'Neighborhood Permutation Testing',    'requires': []},
        {'key': 'cellular_neighborhoods',         'label': 'Cellular Neighborhoods',               'requires': []},
        {'key': 'tumor_microenvironment',         'label': 'Tumor Microenvironment (zones)',        'requires': ['per_tumor_analysis']},
        {'key': 'enhanced_neighborhoods',         'label': 'Enhanced Neighborhoods',               'requires': ['per_tumor_analysis']},
        {'key': 'marker_region_analysis',         'label': 'Marker Region Analysis',               'requires': []},
        {'key': 'cluster_composition_analysis',   'label': 'Cluster Composition Analysis',         'requires': ['per_tumor_analysis']},
        {'key': 'temporal_analysis',              'label': 'Temporal Analysis',                    'requires': []},
        {'key': 'lda_neighborhood_analysis',      'label': 'LDA Recurrent Cellular Neighborhoods', 'requires': []},
        {'key': 'spatial_lag_analysis',           'label': 'Spatial Lag / Tumor Cell Communities', 'requires': []},
        {'key': 'shift_plot_analysis',            'label': 'Shift Plot Analysis',                  'requires': []},
        {'key': 'perk_mfi_analysis',              'label': 'pERK MFI Analysis',                   'requires': []},
        {'key': 'coexpression_analysis',          'label': 'Coexpression Analysis',                'requires': []},
        {'key': 'spatial_overlap_analysis',       'label': 'Spatial Overlap Analysis',             'requires': []},
        {'key': 'kpnt_correlation_analysis',      'label': 'KPNT Correlation Analysis',            'requires': []},
        {'key': 'pseudotime_analysis',            'label': 'Pseudotime Analysis',                  'requires': []},
        {'key': 'marker_clustering_analysis',     'label': 'Marker Clustering Analysis',           'requires': []},
    ]
