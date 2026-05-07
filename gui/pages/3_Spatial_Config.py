"""Page 3: Phenotype builder + analysis toggles + sample metadata editor."""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'gui'))

from components.config_io import load_project, save_project, get_display_names, get_analysis_list
from components.gating_plots import plot_phenotype_scatter
from components.spatial_viewer import load_adata, find_h5ad, get_samples, get_spatial_coords


def _render_analysis_params(key: str, spatial: dict, display_names: list, phenotype_names: list):
    """Render parameter inputs for each analysis type."""
    cfg = spatial.setdefault(key, {})

    if key == 'per_tumor_analysis':
        cfg['use_spatialcells'] = st.toggle('Use SpatialCells', value=cfg.get('use_spatialcells', True), key=f'{key}_sc')
        cfg['generate_plots'] = st.toggle('Generate plots', value=cfg.get('generate_plots', True), key=f'{key}_plots')

    elif key == 'population_dynamics':
        cfg['populations'] = st.multiselect('Populations to quantify', phenotype_names,
                                             default=cfg.get('populations', []), key=f'{key}_pops')
        cfg.setdefault('metrics', ['count', 'density', 'fraction'])
        cfg.setdefault('statistics', {})['test_per_timepoint'] = st.toggle(
            'Test per timepoint', value=cfg.get('statistics', {}).get('test_per_timepoint', False), key=f'{key}_tpt'
        )

    elif key == 'distance_analysis':
        st.caption('Define source → target pairings. Configure in project.yaml for full control.')
        max_dist = st.number_input('Max distance for plots (px)', min_value=50, max_value=5000,
                                    value=int(cfg.get('max_distance_plot', 500)), key=f'{key}_maxdist')
        cfg['max_distance_plot'] = max_dist

    elif key == 'cellular_neighborhoods':
        col1, col2 = st.columns(2)
        with col1:
            cfg['n_clusters'] = st.number_input('N clusters (RCNs)', min_value=2, max_value=30,
                                                  value=int(cfg.get('n_clusters', 8)), key=f'{key}_nc')
            cfg['window_size'] = st.number_input('Window size (px)', min_value=20, max_value=500,
                                                   value=int(cfg.get('window_size', 100)), key=f'{key}_ws')
        with col2:
            cfg['k_neighbors'] = st.number_input('K neighbors', min_value=5, max_value=100,
                                                   value=int(cfg.get('k_neighbors', 30)), key=f'{key}_kn')
            cfg['define_globally'] = st.toggle('Global clustering', value=cfg.get('define_globally', True), key=f'{key}_global')

    elif key == 'lda_neighborhood_analysis':
        col1, col2 = st.columns(2)
        with col1:
            cfg['n_lda_topics'] = st.number_input('LDA topics', min_value=2, max_value=50,
                                                    value=int(cfg.get('n_lda_topics', 10)), key=f'{key}_topics')
            cfg['n_rcns'] = st.number_input('N RCNs (meta-clusters)', min_value=2, max_value=30,
                                             value=int(cfg.get('n_rcns', 10)), key=f'{key}_rcns')
        with col2:
            cfg['proximity_radius'] = st.number_input('Neighborhood radius (µm)', min_value=5, max_value=200,
                                                        value=int(cfg.get('proximity_radius', 20)), key=f'{key}_radius')
            cfg['n_lda_clusters'] = st.number_input('K-means clusters (LDA)', min_value=5, max_value=100,
                                                      value=int(cfg.get('n_lda_clusters', 30)), key=f'{key}_kc')

    elif key == 'spatial_permutation':
        cfg['n_permutations'] = st.number_input('N permutations', min_value=100, max_value=5000,
                                                  value=int(cfg.get('n_permutations', 500)), key=f'{key}_nperm')

    elif key in ('tumor_microenvironment', 'immune_infiltration'):
        cfg['immune_populations'] = st.multiselect('Immune populations', phenotype_names,
                                                    default=cfg.get('immune_populations', []), key=f'{key}_immune')

    elif key == 'temporal_analysis':
        st.caption('Requires `timepoint` column in sample_metadata.csv.')

    elif key == 'shift_plot_analysis':
        cfg['n_bootstrap'] = st.number_input('Bootstrap samples', min_value=100, max_value=5000,
                                              value=int(cfg.get('n_bootstrap', 500)), key=f'{key}_bs')
        cfg['proximity_radius'] = st.number_input('Max search distance (µm)', min_value=100, max_value=10000,
                                                   value=int(cfg.get('proximity_radius', 2000)), key=f'{key}_pr')


st.set_page_config(page_title='Spatial Config · cifsQuant', layout='wide')
st.title('3 · Spatial Config')

project_dir = Path(st.session_state.get('project_dir', REPO_ROOT))
yaml_path = project_dir / 'project.yaml'
config = st.session_state.get('project_config') or load_project(yaml_path)
spatial = config.setdefault('spatial', {})
display_names = get_display_names(config)

# Load gated data for live previews (optional)
gated_path = find_h5ad(project_dir, 'gated_data.h5ad')
adata = None
if gated_path:
    try:
        adata = load_adata(str(gated_path))
    except Exception:
        pass

tab_meta, tab_pheno, tab_analyses, tab_structure = st.tabs(
    ['Sample Metadata', 'Phenotype Builder', 'Analyses', 'Structure Detection']
)

# ── Tab: Sample Metadata ──────────────────────────────────────────────────────
with tab_meta:
    st.subheader('Sample metadata')
    st.markdown('One row per sample. `sample_id` must match your image filenames (without extension).')

    meta_path = project_dir / 'sample_metadata.csv'
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
    else:
        meta_df = pd.DataFrame(columns=['sample_id', 'group', 'timepoint', 'treatment'])

    # Column management
    optional_cols = ['timepoint', 'treatment']
    active_cols = list(meta_df.columns)
    for col in optional_cols:
        add = st.toggle(f'Include `{col}` column', value=col in active_cols, key=f'meta_col_{col}')
        if add and col not in active_cols:
            meta_df[col] = ''
        elif not add and col in active_cols:
            meta_df = meta_df.drop(columns=[col])

    edited_meta = st.data_editor(meta_df, num_rows='dynamic', use_container_width=True, key='meta_editor')

    # Update metadata config keys
    meta_cfg = spatial.setdefault('metadata', {})
    meta_cfg['sample_column'] = 'sample_id'
    col1, col2, col3 = st.columns(3)
    with col1:
        meta_cfg['group_column'] = st.text_input('Group column name', value=meta_cfg.get('group_column', 'group'))
    with col2:
        if 'timepoint' in edited_meta.columns:
            meta_cfg['timepoint_column'] = st.text_input('Timepoint column', value=meta_cfg.get('timepoint_column', 'timepoint'))
    with col3:
        if 'treatment' in edited_meta.columns:
            meta_cfg['treatment_column'] = st.text_input('Treatment column', value=meta_cfg.get('treatment_column', 'treatment'))

    if st.button('Save sample_metadata.csv'):
        edited_meta.to_csv(meta_path, index=False)
        st.success('Saved.')

# ── Tab: Phenotype Builder ────────────────────────────────────────────────────
with tab_pheno:
    st.subheader('Phenotype definitions')
    st.markdown(
        'Define cell populations as marker combinations. '
        'Each phenotype creates an `is_{name}` boolean column in the gated data.'
    )

    phenotypes = spatial.setdefault('phenotypes', {})
    phenotype_names = list(phenotypes.keys())

    # Phenotype tree
    if phenotypes:
        st.markdown('**Current phenotypes:**')
        for name, defn in phenotypes.items():
            base = defn.get('base', '')
            pos = ', '.join(defn.get('positive', []))
            neg = ', '.join(defn.get('negative', []))
            anyp = ', '.join(defn.get('anypos', []))
            parts = []
            if base:
                parts.append(f'base: {base}')
            if pos:
                parts.append(f'+ [{pos}]')
            if neg:
                parts.append(f'− [{neg}]')
            if anyp:
                parts.append(f'any [{anyp}]')
            desc = '  ·  '.join(parts) if parts else '(no definition)'
            indent = '&nbsp;&nbsp;&nbsp;&nbsp;' if base else ''
            st.markdown(f'{indent}**{name}** — {desc}', unsafe_allow_html=True)

    st.divider()
    st.markdown('**Add / Edit phenotype:**')
    col_name, col_base = st.columns(2)
    with col_name:
        new_name = st.text_input('Phenotype name', placeholder='e.g. CD8_T_cells')
    with col_base:
        base_options = ['— none —'] + phenotype_names
        new_base = st.selectbox('Base (parent phenotype)', base_options)

    col_pos, col_neg, col_any = st.columns(3)
    with col_pos:
        new_pos = st.multiselect('Positive markers (AND)', display_names, key='new_pos')
    with col_neg:
        new_neg = st.multiselect('Negative markers (AND)', display_names, key='new_neg')
    with col_any:
        new_any = st.multiselect('Any positive (OR)', display_names, key='new_any')

    col_add, col_del = st.columns(2)
    with col_add:
        if st.button('Add / Update phenotype', disabled=not new_name):
            defn = {}
            if new_base != '— none —':
                defn['base'] = new_base
            if new_pos:
                defn['positive'] = new_pos
            if new_neg:
                defn['negative'] = new_neg
            if new_any:
                defn['anypos'] = new_any
            phenotypes[new_name] = defn
            spatial['phenotypes'] = phenotypes
            st.rerun()
    with col_del:
        del_target = st.selectbox('Remove phenotype', ['—'] + phenotype_names, key='del_pheno')
        if st.button('Remove', disabled=del_target == '—'):
            phenotypes.pop(del_target, None)
            st.rerun()

    # Live preview: n_cells matching definition
    if adata is not None and new_name and new_pos:
        preview_col = f'is_{new_pos[0]}'
        if preview_col in adata.obs.columns:
            mask = adata.obs[preview_col].astype(bool).values
            for m in new_pos[1:]:
                col = f'is_{m}'
                if col in adata.obs.columns:
                    mask = mask & adata.obs[col].astype(bool).values
            if new_base and new_base != '— none —':
                base_col = f'is_{new_base}'
                if base_col in adata.obs.columns:
                    mask = mask & adata.obs[base_col].astype(bool).values
            st.info(f'Preview: {mask.sum():,} cells would match this definition')

            if st.toggle('Show spatial preview', key='preview_toggle'):
                samples = get_samples(adata)
                sample_sel = st.selectbox('Preview sample', ['all'] + samples, key='preview_sample')
                import numpy as np
                x, y = get_spatial_coords(adata, sample_sel if sample_sel != 'all' else None)
                if sample_sel != 'all' and 'sample_id' in adata.obs.columns:
                    smask = (adata.obs['sample_id'] == sample_sel).values
                    pheno_mask = mask[smask]
                else:
                    pheno_mask = mask
                fig = plot_phenotype_scatter(x, y, pheno_mask, new_name)
                st.plotly_chart(fig, use_container_width=True, key='pheno_preview_fig')

# ── Tab: Analyses ─────────────────────────────────────────────────────────────
with tab_analyses:
    st.subheader('Analysis modules')
    st.markdown('Enable the analyses you want to run. Dependent analyses require `Per-Structure Analysis` to be enabled first.')

    enabled_analyses = set()
    analyses = get_analysis_list()

    for analysis in analyses:
        key = analysis['key']
        label = analysis['label']
        requires = analysis['requires']

        # Dependency warning
        deps_missing = [r for r in requires if not spatial.get(r, {}).get('enabled', False)]

        current_enabled = spatial.get(key, {}).get('enabled', False)
        col_toggle, col_label = st.columns([1, 5])
        with col_toggle:
            enabled = st.toggle('', value=current_enabled, key=f'toggle_{key}',
                                disabled=bool(deps_missing))
        with col_label:
            dep_warn = f' ⚠ requires: {", ".join(deps_missing)}' if deps_missing else ''
            st.markdown(f'**{label}**{dep_warn}')

        spatial.setdefault(key, {})['enabled'] = enabled

        if enabled:
            enabled_analyses.add(key)
            with st.expander(f'{label} — parameters', expanded=False):
                _render_analysis_params(key, spatial, display_names, list(phenotypes.keys()))

    # Dependency validation
    if 'immune_infiltration' in enabled_analyses and 'per_tumor_analysis' not in enabled_analyses:
        st.error('⚠ `immune_infiltration` requires `per_tumor_analysis` to be enabled and run first.')
    if 'tumor_microenvironment' in enabled_analyses and 'per_tumor_analysis' not in enabled_analyses:
        st.error('⚠ `tumor_microenvironment` requires `per_tumor_analysis`.')

# ── Tab: Structure Detection ──────────────────────────────────────────────────
with tab_structure:
    st.subheader('Spatial structure detection')
    st.markdown(
        'Defines the primary spatial structure (tumor, B cell cluster, follicle) '
        'used by per-structure analysis, infiltration, and zone analyses.'
    )

    td = spatial.setdefault('tumor_definition', {})
    col1, col2 = st.columns(2)
    with col1:
        td['base_phenotype'] = st.selectbox(
            'Base phenotype', [''] + list(phenotypes.keys()),
            index=(list(phenotypes.keys()).index(td['base_phenotype'])
                   if td.get('base_phenotype') in phenotypes else 0)
        )
        td['required_positive'] = st.multiselect(
            'Required positive markers', display_names,
            default=[m for m in td.get('required_positive', []) if m in display_names],
        )
        td['required_negative'] = st.multiselect(
            'Required negative markers', display_names,
            default=[m for m in td.get('required_negative', []) if m in display_names],
        )

    sd = td.setdefault('structure_detection', {})
    with col2:
        sd['eps'] = st.number_input('DBSCAN eps (pixels)', min_value=10, max_value=5000,
                                    value=int(sd.get('eps', 1000)),
                                    help='Neighborhood radius. Larger = more cells merged per structure. Typical: tumors=1000, TLS=250')
        sd['min_samples'] = st.number_input('Min cells per cluster', min_value=5, max_value=5000,
                                             value=int(sd.get('min_samples', 500)),
                                             help='Minimum cells for DBSCAN to form a cluster')
        sd['min_cluster_size'] = st.number_input('Min cells per structure (post-filter)', min_value=5,
                                                   value=int(sd.get('min_cluster_size', 250)))
        sd['boundary_buffer'] = st.number_input('Boundary buffer (pixels)', min_value=0, max_value=500,
                                                  value=int(sd.get('boundary_buffer', 100)))
        sd['alpha'] = st.number_input('Alpha shape alpha', min_value=10, max_value=500,
                                       value=int(sd.get('alpha', 100)),
                                       help='Concavity of alpha-shape boundary. Smaller = tighter fit.')

# ── Save ─────────────────────────────────────────────────────────────────────
st.divider()
if st.button('💾 Save to project.yaml', type='primary'):
    config['spatial'] = spatial
    from components.config_io import save_project
    save_project(config, yaml_path)
    st.session_state.project_config = config
    st.success('Saved. Proceed to **4 · Run Pipeline**.')
