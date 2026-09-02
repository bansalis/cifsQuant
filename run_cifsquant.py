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


def validate_project(project: dict, path: Path = Path('project.yaml'),
                     stages: list | None = None) -> bool:
    """Check for configuration errors relevant to the stages being run.

    Only gating/spatial sections are validated when those stages are included;
    segmentation only requires markers to be defined. Returns True if the
    config is valid, False otherwise (errors are printed).
    """
    if stages is None:
        stages = list(STAGES)

    errors = []

    if 'markers' not in project:
        errors.append("Missing required key: 'markers'")

    marker_display_names = set(project.get('markers', {}).values())

    # Gating and hierarchy checks are only relevant when those stages run
    if 'gating' in stages or 'spatial' in stages:
        gating = project.get('gating', {})

        # Gate keys must map to panel display names
        gate_keys = set(gating.get('gates', {}).keys())
        if gate_keys:
            unknown_gates = gate_keys - marker_display_names
            if unknown_gates:
                errors.append(f"Gate keys not in marker panel: {sorted(unknown_gates)}")

        # Tile correction markers
        tc_markers = set(gating.get('tile_correction', {}).get('markers', []))
        unknown_tc = tc_markers - marker_display_names
        if unknown_tc:
            errors.append(f"Tile correction markers not in panel: {sorted(unknown_tc)}")

        # Liberal gating markers
        lib_markers = set(gating.get('liberal_gating', {}).get('liberal_markers', []))
        unknown_lib = lib_markers - marker_display_names
        if unknown_lib:
            errors.append(f"Liberal gating markers not in panel: {sorted(unknown_lib)}")

        # Marker hierarchy
        hierarchy = project.get('marker_hierarchy', {})
        for child, parent in hierarchy.items():
            if child not in marker_display_names:
                errors.append(f"marker_hierarchy child '{child}' not in panel")
            if parent is not None and parent not in marker_display_names:
                errors.append(f"marker_hierarchy parent '{parent}' not in panel")

    # Spatial phenotype checks only when running spatial analysis
    if 'spatial' in stages:
        spatial = project.get('spatial', {})
        for pheno_name, pheno_def in spatial.get('phenotypes', {}).items():
            if not isinstance(pheno_def, dict):
                continue
            for marker in pheno_def.get('positive', []) + pheno_def.get('negative', []):
                # Template placeholders stay valid so --dry-run works on a fresh clone
                if marker not in marker_display_names and marker not in (
                        'STRUCTURE_MARKER', 'IMMUNE_MARKER', 'T_CELL_MARKER', 'CD8_MARKER', 'CD4_MARKER'):
                    errors.append(f"Phenotype '{pheno_name}' references unknown marker '{marker}'")

    if errors:
        print(f"\n  Validation errors in {path}:")
        for e in errors:
            print(f"    - {e}")
        return False

    print(f"  Validated: {path} (stages: {', '.join(stages)})")
    return True


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


def generate_channel_markers_csv(project: dict, output_path: Path):
    """markers csv keyed by CHANNEL names, for matching per-channel raw files.

    Quantification columns then carry channel names, which Stage 2 renames to
    display names via the project's markers map (load_and_combine).
    """
    rows = []
    for cycle, channel_name in enumerate(project.get('markers', {}), start=1):
        rows.append({'cycle': cycle, 'channel_number': cycle, 'marker_name': channel_name})
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['cycle', 'channel_number', 'marker_name'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Generated {output_path} ({len(rows)} channels)")


_channel_matcher = None


def match_channels_report(sample_dir, channel_names) -> list:
    """Soft-match each channel name against the raw files in one sample folder.

    Uses the tiler's own matcher (round/fluorophore/protein substring matching),
    so this report shows exactly what a real tiling run would pick up.
    Returns [{'channel', 'file', 'matched'}] in channel order.
    """
    global _channel_matcher
    if _channel_matcher is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'tile_from_channels',
            Path(__file__).parent / 'scripts' / 'tile_from_channels.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _channel_matcher = mod.find_matching_channels

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        files = _channel_matcher(str(sample_dir), list(channel_names), allow_missing=True)
    return [{'channel': name, 'file': (f.name if f is not None else None),
             'matched': f is not None}
            for name, f in zip(channel_names, files)]


def print_match_report(sample: str, report: list):
    matched = sum(r['matched'] for r in report)
    print(f"  {sample}: {matched}/{len(report)} channels matched")
    for r in report:
        if r['matched']:
            print(f"    ✓ {r['channel']:24s} -> {r['file']}")
        else:
            print(f"    ⚠ {r['channel']:24s} -> NO MATCH (will be zero-filled)")


def discover_rawdata_samples(rawdata_dir) -> list:
    """Sample folders under rawdata/: one subfolder per sample, each holding
    per-channel .ome.tif files."""
    rawdata_dir = Path(rawdata_dir)
    if not rawdata_dir.is_dir():
        return []
    return sorted(p for p in rawdata_dir.iterdir()
                  if p.is_dir() and not p.name.startswith('.'))


def run_segmentation(project: dict, project_path: Path, dry_run: bool):
    repo = Path(__file__).parent

    # Per-channel mode: rawdata/<sample>/ folders of per-channel .ome.tif files.
    # Explicit via 'rawdata_dir' in project.yaml, or auto-detected at ./rawdata
    # when no input_image is set.
    rawdata_dir = project.get('rawdata_dir')
    if not rawdata_dir and not project.get('input_image') and discover_rawdata_samples(repo / 'rawdata'):
        rawdata_dir = repo / 'rawdata'

    if rawdata_dir:
        samples = discover_rawdata_samples(rawdata_dir)
        if not samples:
            print(f"  No sample folders found in {rawdata_dir}")
            sys.exit(1)
        print(f"  Per-channel mode: {len(samples)} sample(s) in {rawdata_dir}")

        outdir = Path(project.get('outdir', './results'))
        channel_csv = Path('markers_channels.csv')
        if not dry_run:
            generate_channel_markers_csv(project, channel_csv)

        # The tiler's dapi index refers to the generated csv order (= the
        # markers: mapping order), so derive it from the DAPI entry directly
        display_names = [str(n).upper() for n in project.get('markers', {}).values()]
        if 'DAPI' in display_names:
            dapi_idx = display_names.index('DAPI')
        else:
            dapi_idx = int(project.get('dapi_channel', 0))

        for sample_dir in samples:
            sample = sample_dir.name
            # Show how each channel name soft-matched a raw file BEFORE running,
            # so a bad panel/filename mismatch is caught here, not hours later
            print_match_report(sample, match_channels_report(
                sample_dir, list(project.get('markers', {}).keys())))
            tiles_dir = outdir / sample / 'tiles'
            tile_cmd = [
                sys.executable, str(repo / 'scripts' / 'tile_from_channels.py'),
                '--sample_dir', str(sample_dir),
                '--markers_csv', str(channel_csv),
                '--output_dir', str(tiles_dir),
                '--tile_size', str(project.get('tile_size', 4096)),
                '--overlap', str(project.get('overlap', 512)),
                '--dapi_channel', str(dapi_idx),
                '--allow_missing_channels',
            ]
            nf_cmd = ['nextflow', 'run', str(repo / 'mcmicro-tiled.nf'),
                      '-params-file', str(project_path),
                      '--skip_tiling', 'true', '--tiles_dir', str(tiles_dir),
                      '--sample_name', sample, '--outdir', str(outdir / sample),
                      '-resume']
            if dry_run:
                print(f"  [dry-run] {' '.join(tile_cmd)}")
                print(f"  [dry-run] {' '.join(nf_cmd)}")
                continue
            if (tiles_dir / 'tile_info.json').exists():
                print(f"  {sample}: reusing existing tiles in {tiles_dir}")
            else:
                subprocess.run(tile_cmd, check=True)
            subprocess.run(nf_cmd, check=True)
        return

    # Stacked mode: one multi-channel OME-TIFF per run (input_image)
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
    if not validate_project(project, project_path, stages=args.stages):
        sys.exit(1)
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
            run_segmentation(project, project_path, args.dry_run)
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
