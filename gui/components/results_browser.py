"""Browse output directories: PNG gallery + CSV tables."""
import streamlit as st
from pathlib import Path
import pandas as pd


ANALYSIS_DESCRIPTIONS = {
    'per_structure_analysis':   'Per-tumor/structure metrics: cell composition, area, and immune infiltration per detected spatial structure.',
    'population_dynamics':      'Cell population frequencies (count, density, fraction) compared across experimental groups.',
    'distance_analysis':        'Nearest-neighbor distances between cell populations across groups.',
    'immune_infiltration':      'Immune cell density at defined distances from structure boundaries.',
    'spatial_permutation':      'Permutation test: are observed spatial co-localization patterns statistically significant?',
    'distance_permutation':     'Permutation test: are distance patterns between populations beyond chance?',
    'neighborhood_permutation': 'Permutation test for neighborhood enrichment between cell types.',
    'neighborhood_analysis':    'K-means classification of cells by local neighborhood composition (RCNs).',
    'tumor_microenvironment':   'Zone analysis: immune composition at contact / close / distal distances from structure boundary.',
    'enhanced_neighborhoods':   'Marker-stratified neighborhood composition inside vs outside marker-defined regions.',
    'marker_region_analysis':   'Alpha-shape communities of marker+ cells and immune enrichment inside vs outside.',
    'cluster_composition_analysis': 'Stacked bar charts of immune composition per detected structure.',
    'temporal_analysis':        'Structure-level metrics across timepoints (longitudinal studies).',
    'lda_neighborhood_analysis':'LDA Recurrent Cellular Neighborhoods (Nirmal et al. 2022 methodology).',
    'spatial_lag_analysis':     'Spatial Lag Tumor Cell Communities (Nirmal et al. 2022 methodology).',
    'shift_plot_analysis':      'Harrell-Davis shift plots comparing distance distributions between groups.',
    'perk_mfi_analysis':        'pERK mean fluorescence intensity per phenotype and per structure.',
    'coexpression_analysis':    'Pairwise co-occurrence analysis between cell phenotypes.',
    'spatial_overlap_analysis': 'Spatial overlap (Dice coefficient) between cell population density maps.',
}


def render_results_browser(output_dir: Path):
    if not output_dir.exists():
        st.info(f'No results yet at `{output_dir}`. Run the pipeline first.')
        return

    subdirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    if not subdirs:
        st.info('Output directory exists but no analysis subdirectories found.')
        return

    # Sidebar-style selection
    analysis_names = [d.name for d in subdirs]
    selected = st.selectbox('Select analysis', analysis_names)
    analysis_dir = output_dir / selected

    if selected in ANALYSIS_DESCRIPTIONS:
        st.caption(ANALYSIS_DESCRIPTIONS[selected])
    st.divider()

    # Tabs: Plots | Data
    tab_plots, tab_data = st.tabs(['Plots', 'Data'])

    with tab_plots:
        _render_png_gallery(analysis_dir)

    with tab_data:
        _render_csv_viewer(analysis_dir)


def _render_png_gallery(directory: Path):
    pngs = list(directory.rglob('*.png'))
    if not pngs:
        st.info('No plots found in this directory.')
        return

    # Group by subdirectory
    by_subdir: dict[str, list[Path]] = {}
    for p in sorted(pngs):
        key = str(p.parent.relative_to(directory)) if p.parent != directory else '.'
        by_subdir.setdefault(key, []).append(p)

    for subdir_key, files in by_subdir.items():
        if subdir_key != '.':
            st.subheader(subdir_key)
        cols = st.columns(min(3, len(files)))
        for i, png_path in enumerate(files):
            with cols[i % 3]:
                st.image(str(png_path), caption=png_path.stem, use_container_width=True)


def _render_csv_viewer(directory: Path):
    csvs = list(directory.rglob('*.csv'))
    if not csvs:
        st.info('No CSV files found in this directory.')
        return

    csv_names = [str(p.relative_to(directory)) for p in csvs]
    selected_csv = st.selectbox('Select table', csv_names)
    csv_path = directory / selected_csv

    try:
        df = pd.read_csv(csv_path)
        st.caption(f'{len(df):,} rows × {len(df.columns)} columns')

        # Filter by sample if sample_id present
        if 'sample_id' in df.columns:
            samples = ['All'] + sorted(df['sample_id'].unique().tolist())
            chosen = st.selectbox('Filter by sample', samples, key=f'sample_filter_{selected_csv}')
            if chosen != 'All':
                df = df[df['sample_id'] == chosen]

        st.dataframe(df, use_container_width=True, height=400)

        csv_bytes = df.to_csv(index=False).encode()
        st.download_button('Download CSV', csv_bytes, file_name=Path(selected_csv).name, mime='text/csv')
    except Exception as e:
        st.error(f'Could not load CSV: {e}')
