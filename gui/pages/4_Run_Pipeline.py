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
st.caption('Validate your configuration and execute one or more pipeline stages with live log output.')

project_dir = Path(st.session_state.get('project_dir', REPO_ROOT))
yaml_path = project_dir / 'project.yaml'

if not yaml_path.exists():
    st.warning('No project.yaml found. Create one from the home page.')
    st.page_link('app.py', label='Go to Home')
    st.stop()

config = st.session_state.get('project_config') or load_project(yaml_path)

# ── Pipeline status ───────────────────────────────────────────────────────────
results_dir = project_dir / 'results'
norm_h5ad = project_dir / 'manual_gating_output' / 'normalized_data.h5ad'
gated_h5ad = project_dir / 'manual_gating_output' / 'gated_data.h5ad'
output_dir = project_dir / 'spatial_quantification_results'

n_seg = len(list(results_dir.glob('*/final/combined_quantification.csv'))) if results_dir.exists() else 0
n_analyses = len(list(output_dir.glob('*/'))) if output_dir.exists() else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('Stage 1 · Segmentation', f'{n_seg} samples' if n_seg else '—',
              help='CSVs found in results/')
with col2:
    st.metric('Normalization checkpoint', '✓' if norm_h5ad.exists() else '—',
              help=str(norm_h5ad))
with col3:
    st.metric('Stage 2 · Gating', '✓ done' if gated_h5ad.exists() else '—',
              help=str(gated_h5ad))
with col4:
    st.metric('Stage 3 · Spatial', f'{n_analyses} analyses' if n_analyses else '—')

st.divider()

# ── Config validation ─────────────────────────────────────────────────────────
st.subheader('Configuration validation')

col_validate, col_dryrun = st.columns(2)
with col_validate:
    if st.button('Validate config', help='Check for missing or inconsistent settings'):
        checks = validate_project(config, stages=selected_stages or ['segmentation', 'gating', 'spatial'])
        for check in checks:
            icon = '✓' if check['status'] == 'ok' else ('⚠' if check['status'] == 'warn' else '✗')
            fn = st.success if check['status'] == 'ok' else (st.warning if check['status'] == 'warn' else st.error)
            fn(f'{icon} **{check["check"]}** — {check["message"]}')
        if all(c['status'] == 'ok' for c in checks):
            st.success('All checks passed.')

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
        help='Nextflow + Cellpose segmentation. Requires Docker and Nextflow on your PATH.',
    )
    if run_seg:
        st.caption('Requires Docker + Nextflow ≥ 23.10')
with col2:
    run_gating = st.checkbox(
        'Stage 2 · Gating',
        value=True,
        help='Normalize intensities and apply per-marker thresholds → gated_data.h5ad',
    )
with col3:
    run_spatial = st.checkbox(
        'Stage 3 · Spatial analysis',
        help='Run all enabled analysis modules → spatial_quantification_results/',
    )

selected_stages = []
if run_seg:
    selected_stages.append('segmentation')
if run_gating:
    selected_stages.append('gating')
if run_spatial:
    selected_stages.append('spatial')

# Advanced options
with st.expander('Advanced options'):
    n_jobs = st.number_input('Parallel jobs (gating stage)', min_value=1, max_value=64,
                              value=8, help='CPU cores for normalization step — only used for Stage 2')
    skip_norm = st.checkbox('Skip normalization (reuse existing checkpoint)',
                             help='Reuse normalized_data.h5ad from a previous run. '
                                  'Only valid if panel and samples have not changed.')
    force_norm = st.checkbox('Force re-normalization',
                              help='Re-run normalization even if checkpoint already exists.')

# ── Run button ────────────────────────────────────────────────────────────────
st.divider()

can_run = bool(selected_stages)
run_btn = st.button(
    '▶ Run Pipeline',
    type='primary',
    disabled=not can_run,
    help='Run selected stages' if can_run else 'Select at least one stage above',
)

if not can_run:
    st.caption('Select at least one stage above to enable Run.')

if run_btn:
    # Build command — use hyphen flags as expected by argparse
    cmd = [
        sys.executable, str(REPO_ROOT / 'run_cifsquant.py'),
        '--project', str(yaml_path),
        '--stages', *selected_stages,
    ]
    # Only pass --n-jobs when gating is included (it's irrelevant otherwise)
    if run_gating:
        cmd += ['--n-jobs', str(n_jobs)]
    if skip_norm:
        cmd.append('--skip-normalization')
    if force_norm:
        cmd.append('--force-normalization')

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
                log_placeholder.code('\n'.join(lines[-200:]), language='text')
                status_placeholder.caption(f'Elapsed: {elapsed:.0f}s')

            proc.wait()
            elapsed = time.time() - start_time

            if proc.returncode == 0:
                st.success(f'Done in {elapsed:.0f}s. Proceed to the next step.')
                # Invalidate cached h5ad so the next page reads fresh data
                try:
                    from components.spatial_viewer import load_normalized, load_adata
                    load_normalized.clear()
                    load_adata.clear()
                except Exception:
                    pass
            else:
                st.error(f'Exited with code {proc.returncode}. See log above.')

        except FileNotFoundError:
            st.error('Could not find `run_cifsquant.py`. Check that the project directory is set correctly on the home page.')
        except Exception as e:
            st.error(f'Unexpected error: {e}')

st.divider()

# CLI equivalent
with st.expander('CLI equivalent'):
    stage_str = ' '.join(selected_stages) if selected_stages else 'gating spatial'
    st.code(
        f'conda activate cifsquant\n'
        f'cd {project_dir}\n'
        f'python run_cifsquant.py \\\n'
        f'  --project project.yaml \\\n'
        f'  --stages {stage_str}',
        language='bash'
    )
    st.caption('All GUI functionality is available via CLI. No lock-in.')
