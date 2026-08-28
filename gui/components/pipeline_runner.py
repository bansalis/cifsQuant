"""Shared helper for running a single pipeline stage from any page."""
import subprocess
import sys
import time
import streamlit as st
from pathlib import Path


def run_stage_inline(stage: str, project_dir: Path, yaml_path: Path, extra_flags: list | None = None):
    """
    Run a single pipeline stage with a spinner and collapsed log output.
    Returns True on success, False on failure.
    """
    REPO_ROOT = yaml_path.parent  # project dir IS the repo root in standard setup
    # Walk up to find run_cifsquant.py (works whether project is in repo root or subdir)
    runner = project_dir / 'run_cifsquant.py'
    if not runner.exists():
        # Try repo root two levels up from gui/components/
        runner = Path(__file__).parent.parent.parent / 'run_cifsquant.py'

    cmd = [
        sys.executable, str(runner),
        '--project', str(yaml_path),
        '--stages', stage,
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    with st.spinner(f'Running Stage {_stage_num(stage)}: {stage}…'):
        start = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(project_dir)
            )
            elapsed = time.time() - start

            if result.returncode == 0:
                st.success(f'Stage complete in {elapsed:.0f}s.')
                with st.expander('Show log'):
                    st.code(result.stdout[-4000:], language='text')
                return True
            else:
                st.error(f'Stage failed (exit {result.returncode}).')
                with st.expander('Show error log'):
                    st.code((result.stdout + result.stderr)[-4000:], language='text')
                return False

        except FileNotFoundError:
            st.error(f'Could not find run_cifsquant.py at {runner}')
            return False


def _stage_num(stage: str) -> str:
    return {'segmentation': '1', 'gating': '2', 'spatial': '3'}.get(stage, '?')
