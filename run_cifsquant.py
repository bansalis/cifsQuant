#!/usr/bin/env python3
"""
cifsQuant Universal Orchestrator

Runs the full three-stage pipeline from a single project.yaml:
  Stage 1 (segmentation) — Nextflow + Cellpose
  Stage 2 (gating)       — manual_gating.py
  Stage 3 (spatial)      — spatial_quantification

Usage:
    python run_cifsquant.py                               # full pipeline, default project.yaml
    python run_cifsquant.py --project my_study.yaml       # specify config
    python run_cifsquant.py --stages gating spatial       # skip segmentation
    python run_cifsquant.py --dry-run                     # validate config only
"""

import argparse
import csv
import subprocess
import sys
import yaml
from pathlib import Path


STAGES = ['segmentation', 'gating', 'spatial']


def load_project(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def validate_project(project: dict, path: Path):
    """Check for common configuration errors before wasting a run."""
    errors = []

    if 'markers' not in project:
        errors.append("Missing required key: 'markers'")

    marker_display_names = set(project.get('markers', {}).values())

    # Validate gate keys match panel
    gating = project.get('gating', {})
    gate_keys = set(gating.get('gates', {}).keys())
    unknown_gates = gate_keys - marker_display_names
    if unknown_gates:
        errors.append(f"Gate keys not in marker panel: {sorted(unknown_gates)}")

    # Validate tile correction marker names
    tc_markers = set(gating.get('tile_correction', {}).get('markers', []))
    unknown_tc = tc_markers - marker_display_names
    if unknown_tc:
        errors.append(f"Tile correction markers not in panel: {sorted(unknown_tc)}")

    # Validate liberal gating marker names
    lib_markers = set(gating.get('liberal_gating', {}).get('liberal_markers', []))
    unknown_lib = lib_markers - marker_display_names
    if unknown_lib:
        errors.append(f"Liberal gating markers not in panel: {sorted(unknown_lib)}")

    # Validate hierarchy references exist
    hierarchy = project.get('marker_hierarchy', {})
    for child, parent in hierarchy.items():
        if child not in marker_display_names:
            errors.append(f"marker_hierarchy child '{child}' not in panel")
        if parent is not None and parent not in marker_display_names:
            errors.append(f"marker_hierarchy parent '{parent}' not in panel")

    # Validate spatial phenotype marker references
    spatial = project.get('spatial', {})
    for pheno_name, pheno_def in spatial.get('phenotypes', {}).items():
        if not isinstance(pheno_def, dict):
            continue
        for marker in pheno_def.get('positive', []) + pheno_def.get('negative', []):
            if marker not in marker_display_names and marker not in ('STRUCTURE_MARKER', 'IMMUNE_MARKER', 'T_CELL_MARKER', 'CD8_MARKER', 'CD4_MARKER'):
                errors.append(f"Phenotype '{pheno_name}' references unknown marker '{marker}'")

    if errors:
        print(f"\n  Validation errors in {path}:")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)

    print(f"  Validated: {path}")


def generate_markers_csv(project: dict, output_path: Path = Path('markers.csv')):
    """Generate markers.csv from the markers dict (overwrites existing)."""
    markers = project.get('markers', {})
    if not markers:
        print("  Skipping markers.csv generation: no 'markers' key in project config")
        return

    rows = []
    cycle = 1
    for channel_name, display_name in markers.items():
        rows.append({'cycle': cycle, 'channel_number': cycle, 'marker_name': display_name})
        cycle += 1

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['cycle', 'channel_number', 'marker_name'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Generated {output_path} ({len(rows)} markers)")


def run_segmentation(project_path: Path, dry_run: bool):
    cmd = ['nextflow', 'run', 'mcmicro-tiled.nf', '-params-file', str(project_path), '-resume']
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return
    subprocess.run(cmd, check=True)


def run_gating(project_path: Path, results_dir: str, n_jobs: int, extra_flags: list, dry_run: bool):
    script = Path(__file__).parent / 'manual_gating.py'
    cmd = [
        sys.executable, str(script),
        '--results_dir', results_dir,
        '--project', str(project_path),
        '--n_jobs', str(n_jobs),
    ] + extra_flags
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return
    subprocess.run(cmd, check=True)


def run_spatial(project_path: Path, dry_run: bool):
    script = Path(__file__).parent / 'spatial_quantification' / 'run_spatial_quantification.py'
    cmd = [sys.executable, str(script), '--config', str(project_path)]
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description='cifsQuant Universal Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cifsquant.py                                   # full pipeline
  python run_cifsquant.py --stages gating spatial           # skip segmentation
  python run_cifsquant.py --stages spatial                  # spatial analysis only
  python run_cifsquant.py --dry-run                         # validate config
  python run_cifsquant.py --force-normalization             # re-run gating from scratch
        """
    )
    parser.add_argument('--project', default='project.yaml',
                        help='Path to project.yaml (default: project.yaml)')
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=STAGES,
                        help='Stages to run (default: all three)')
    parser.add_argument('--results-dir', default='results',
                        help='Nextflow results directory with per-cell CSVs (default: results)')
    parser.add_argument('--n-jobs', type=int, default=8,
                        help='Parallel jobs for gating stage (default: 8)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate config and print commands without executing')
    parser.add_argument('--force-normalization', action='store_true',
                        help='Force re-normalization even if checkpoint exists (gating stage)')
    parser.add_argument('--skip-normalization', action='store_true',
                        help='Load from normalization checkpoint and re-gate only')
    args = parser.parse_args()

    project_path = Path(args.project)
    if not project_path.exists():
        print(f"  Project config not found: {project_path}")
        print(f"  Copy and edit the template: cp project.yaml {project_path}")
        sys.exit(1)

    print("\n" + "="*70)
    print("cifsQuant PIPELINE")
    print("="*70)
    print(f"\nProject config: {project_path.resolve()}")
    print(f"Stages:         {', '.join(args.stages)}")
    if args.dry_run:
        print("Mode:           DRY RUN (validate only)")

    project = load_project(project_path)
    validate_project(project, project_path)
    generate_markers_csv(project)

    gating_flags = []
    if args.force_normalization:
        gating_flags.append('--force_normalization')
    if args.skip_normalization:
        gating_flags.append('--skip_normalization')

    for stage in args.stages:
        print(f"\n{'='*70}")
        print(f"STAGE: {stage.upper()}")
        print('='*70)

        if stage == 'segmentation':
            run_segmentation(project_path, args.dry_run)
        elif stage == 'gating':
            run_gating(project_path, args.results_dir, args.n_jobs, gating_flags, args.dry_run)
        elif stage == 'spatial':
            run_spatial(project_path, args.dry_run)

    print(f"\n{'='*70}")
    if args.dry_run:
        print("Dry run complete. No stages were executed.")
    else:
        print("Pipeline complete.")

        spatial_cfg = project.get('spatial', {})
        out_dir = spatial_cfg.get('output', {}).get('base_directory', 'spatial_quantification_results')
        print(f"\nOutputs:")
        print(f"  Gated data:     manual_gating_output/gated_data.h5ad")
        print(f"  Spatial results: {out_dir}/")
    print('='*70 + '\n')


if __name__ == '__main__':
    main()
