"""Smoke tests: per-channel rawdata/<sample>/ segmentation mode."""
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_discover_rawdata_samples(tmp_path):
    """Sample folders are discovered sorted; files and hidden dirs are ignored."""
    from run_cifsquant import discover_rawdata_samples

    (tmp_path / 'sampleB').mkdir()
    (tmp_path / 'sampleA').mkdir()
    (tmp_path / '.hidden').mkdir()
    (tmp_path / 'stray.ome.tif').touch()

    samples = discover_rawdata_samples(tmp_path)
    assert [s.name for s in samples] == ['sampleA', 'sampleB']


def test_discover_rawdata_samples_missing_dir(tmp_path):
    from run_cifsquant import discover_rawdata_samples
    assert discover_rawdata_samples(tmp_path / 'nope') == []


def test_generate_channel_markers_csv_uses_channel_names(tmp_path):
    """The tiling csv must carry CHANNEL names (matched against raw filenames),
    not display names."""
    from run_cifsquant import generate_channel_markers_csv

    project = {'markers': {'DAPI': 'DAPI', 'Cy3_CD3': 'CD3', 'Cy5_CD8': 'CD8'}}
    out = tmp_path / 'markers_channels.csv'
    generate_channel_markers_csv(project, out)

    with open(out) as f:
        names = [row['marker_name'] for row in csv.DictReader(f)]
    assert names == ['DAPI', 'Cy3_CD3', 'Cy5_CD8']


def test_dry_run_emits_per_sample_commands(tmp_path, capsys):
    """rawdata mode plans one tiling + one nextflow command per sample folder,
    each with skip_tiling and its own tiles_dir."""
    from run_cifsquant import run_segmentation

    rawdata = tmp_path / 'rawdata'
    (rawdata / 'JL216').mkdir(parents=True)
    (rawdata / 'JL217').mkdir()

    project = {
        # DAPI deliberately NOT first, and dapi_channel deliberately stale:
        # the tiler's dapi index must be derived from the markers map order
        'markers': {'Cy3_CD3': 'CD3', 'DAPI': 'DAPI'},
        'rawdata_dir': str(rawdata),
        'outdir': str(tmp_path / 'results'),
        'tile_size': 2048,
        'overlap': 256,
        'dapi_channel': 0,
    }
    run_segmentation(project, tmp_path / 'project.yaml', dry_run=True)

    out = capsys.readouterr().out
    for sample in ('JL216', 'JL217'):
        assert f'--sample_dir {rawdata / sample}' in out
        assert f'--sample_name {sample}' in out
    assert out.count('tile_from_channels.py') == 2
    assert out.count('--skip_tiling true') == 2
    assert '--tiles_dir' in out
    assert '--tile_size 2048' in out
    assert '--dapi_channel 1' in out


def test_stacked_mode_unchanged(tmp_path, capsys):
    """With input_image set and no rawdata_dir, the single nextflow command is planned."""
    from run_cifsquant import run_segmentation

    project = {'markers': {'DAPI': 'DAPI'}, 'input_image': 'sample.ome.tif'}
    run_segmentation(project, tmp_path / 'project.yaml', dry_run=True)

    out = capsys.readouterr().out
    assert 'tile_from_channels.py' not in out
    assert out.count('nextflow run') == 1
