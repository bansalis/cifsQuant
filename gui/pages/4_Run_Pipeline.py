"""Page 4: Run pipeline stages with live log streaming and config validation."""
import streamlit as st
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'gui'))

from components.config_io import load_project, validate_project

st.set_page_config(page_title='Run Pipeline · cifsQuant', layout='wide')
st.title('4 · Run Pipeline')
st.caption('Validate your configuration and execute pipeline stages.')

project_dir = Path(st.session_state.get('project_dir', REPO_ROOT))
yaml_path = project_dir / 'project.yaml'

if not yaml_path.exists():
    st.warning('No project.yaml found. Create one from the home page.')
    st.stop()

config = st.session_state.get('project_config') or load_project(yaml_path)

# ── Config validation ─────────────────────────────────────────────────────────
st.subheader('Configuration validation')

col_validate, col_dryrun = st.columns([1, 1])
with col_validate:
    if st.button('Validate config', help='Check for missing or inconsistent settings'):
        checks = validate_project(config)
        all_ok = all(c['status'] == 'ok' for c in checks)
        for check in checks:
            icon = '✓' if check['status'] == 'ok' else ('⚠' if check['status'] == 'warn' else '✗')
            fn = st.success if check['status'] == 'ok' else (st.warning if check['status'] == 'warn' else st.error)
            fn(f'{icon} **{check["check"]}** — {check["message"]}')
        if all_ok:
            st.success('All checks passed. Ready to run.')

with col_dryrun:
    if st.button('Dry run', help='Show what would execute without running anything'):
        with st.spinner('Running dry-run…'):
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / 'run_cifsquant.py'),
                 '--project', str(yaml_path), '--dry-run'],
                capture_output=True, text=True, cwd=str(project_dir)
            )
        st.code(result.stdout + result.stderr, language='text')

st.divider()

# ── Stage selection ───────────────────────────────────────────────────────────
st.subheader('Select stages to run')

col1, col2, col3 = st.columns(3)
with col1:
    run_seg = st.checkbox(
        'Stage 1 · Segmentation',
        help='Nextflow + Cellpose segmentation. Requires Docker and Nextflow installed.',
    )
with col2:
    run_gating = st.checkbox(
        'Stage 2 · Gating',
        value=True,
        help='Normalize intensities and apply per-marker thresholds → gated_data.h5ad',
    )
with col3:
    run_spatial = st.checkbox(
        'Stage 3 · Spatial analysis',
        value=True,
        help='Run all enabled analysis modules → spatial_quantification_results/',
    )

if run_seg:
    st.info(
        'Stage 1 requires **Docker** and **Nextflow ≥ 23.10** on your PATH. '
        'If running locally without these, uncheck Stage 1 and run it separately:\n'
        '```bash\nnextflow run mcmicro-tiled.nf -c nextflow.config -params-file project.yaml\n```'
    )

# Build stage list for CLI
selected_stages = []
if run_seg:
    selected_stages.append('segmentation')
if run_gating:
    selected_stages.append('gating')
if run_spatial:
    selected_stages.append('spatial')

# ── Additional options ────────────────────────────────────────────────────────
with st.expander('Advanced options'):
    n_jobs = st.number_input('Parallel jobs (gating)', min_value=1, max_value=64,
                              value=8, help='Number of CPU cores for normalization step')
    skip_norm = st.checkbox('Skip normalization (reuse existing checkpoint)',
                             help='Reuse normalized_data.h5ad from a previous run. '
                                  'Only valid if panel and samples have not changed.')
    force_rerun = st.checkbox('Force re-run (ignore existing outputs)',
                               help='Re-run even if output files already exist.')

# ── Run button ────────────────────────────────────────────────────────────────
st.divider()

can_run = bool(selected_stages)
run_btn = st.button(
    '▶ Run Pipeline',
    type='primary',
    disabled=not can_run,
    help='Run selected stages' if can_run else 'Select at least one stage to run',
)

if not can_run and not run_btn:
    st.caption('Select at least one stage above to enable the Run button.')

if run_btn:
    cmd = [
        sys.executable, str(REPO_ROOT / 'run_cifsquant.py'),
        '--project', str(yaml_path),
        '--stages', *selected_stages,
        '--n_jobs', str(n_jobs),
    ]
    if skip_norm:
        cmd.append('--skip_normalization')
    if force_rerun:
        cmd.append('--force')

    st.markdown(f'**Command:** `{" ".join(cmd)}`')
    st.divider()

    log_placeholder = st.empty()
    status_placeholder = st.empty()
    lines: list[str] = []
    start_time = time.time()

    with st.spinner('Pipeline running…'):
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(project_dir),
                bufsize=1,
            )

            for raw_line in proc.stdout:
                line = raw_line.rstrip()
                elapsed = time.time() - start_time
                lines.append(line)
                # Keep last 200 lines for display
                display = '\n'.join(lines[-200:])
                log_placeholder.code(display, language='text')
                status_placeholder.caption(f'Elapsed: {elapsed:.0f}s')

            proc.wait()
            elapsed = time.time() - start_time

            if proc.returncode == 0:
                st.success(f'Pipeline completed successfully in {elapsed:.0f}s. Proceed to **5 · Results Browser**.')
            else:
                st.error(f'Pipeline exited with code {proc.returncode}. Check log above for errors.')

        except FileNotFoundError:
            st.error(
                'Could not find `run_cifsquant.py`. Make sure the project directory is set correctly '
                'on the home page and that `run_cifsquant.py` exists in the repository root.'
            )
        except Exception as e:
            st.error(f'Unexpected error: {e}')

# ── Status overview ───────────────────────────────────────────────────────────
st.divider()
st.subheader('Current output status')

results_dir = project_dir / 'results'
norm_h5ad = project_dir / 'manual_gating_output' / 'normalized_data.h5ad'
gated_h5ad = project_dir / 'manual_gating_output' / 'gated_data.h5ad'
output_dir = project_dir / 'spatial_quantification_results'

col1, col2, col3, col4 = st.columns(4)
with col1:
    n = len(list(results_dir.glob('*/final/combined_quantification.csv'))) if results_dir.exists() else 0
    st.metric('Segmented samples', n)
with col2:
    st.metric('Normalization checkpoint', '✓' if norm_h5ad.exists() else '—')
with col3:
    st.metric('Gated data', '✓' if gated_h5ad.exists() else '—')
with col4:
    n_analyses = len(list(output_dir.glob('*/'))) if output_dir.exists() else 0
    st.metric('Completed analyses', n_analyses)

# CLI reminder
st.divider()
with st.expander('CLI equivalent (for advanced users)'):
    stage_str = ' '.join(selected_stages) if selected_stages else 'gating spatial'
    st.code(
        f'python run_cifsquant.py \\\n'
        f'  --project {yaml_path} \\\n'
        f'  --stages {stage_str} \\\n'
        f'  --n_jobs {n_jobs}',
        language='bash'
    )
    st.caption('All GUI functionality is available via the CLI above. The GUI is a wrapper — no lock-in.')
