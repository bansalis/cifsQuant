"""Smoke tests: project.yaml config validation logic."""
import pytest
import yaml
import tempfile
import os


def test_minimal_project_yaml_is_valid_yaml(minimal_project_yaml):
    """Round-trip a minimal config through YAML serialization."""
    dumped = yaml.dump(minimal_project_yaml)
    loaded = yaml.safe_load(dumped)
    assert loaded['markers']['Cy3_CD3'] == 'CD3'


def test_required_keys_present(minimal_project_yaml):
    """Config must have markers, gating, and spatial sections."""
    assert 'markers' in minimal_project_yaml
    assert 'gating' in minimal_project_yaml
    assert 'spatial' in minimal_project_yaml


def test_gate_keys_match_display_names(minimal_project_yaml):
    """Every gate key must correspond to a display name in the panel."""
    display_names = set(minimal_project_yaml['markers'].values())
    gates = minimal_project_yaml['gating']['gates']
    for gate_key in gates:
        assert gate_key in display_names, (
            f"Gate key '{gate_key}' not found in panel display names: {display_names}"
        )


def test_hierarchy_keys_match_display_names(minimal_project_yaml):
    """Marker hierarchy child/parent must both be valid display names or None."""
    display_names = set(minimal_project_yaml['markers'].values())
    hierarchy = minimal_project_yaml.get('marker_hierarchy', {})
    for child, parent in hierarchy.items():
        assert child in display_names, f"Hierarchy child '{child}' not in panel"
        if parent is not None:
            assert parent in display_names, f"Hierarchy parent '{parent}' not in panel"


def test_phenotype_markers_in_panel(minimal_project_yaml):
    """All markers referenced in phenotype definitions must exist in the panel."""
    display_names = set(minimal_project_yaml['markers'].values())
    phenotypes = minimal_project_yaml['spatial']['phenotypes']
    for pheno_name, pheno_def in phenotypes.items():
        for marker_list_key in ('positive', 'negative', 'anypos'):
            for marker in pheno_def.get(marker_list_key, []):
                assert marker in display_names, (
                    f"Phenotype '{pheno_name}' references marker '{marker}' not in panel"
                )


def test_phenotype_base_references_valid_phenotype(minimal_project_yaml):
    """'base' keys in phenotypes must reference an earlier phenotype."""
    phenotypes = minimal_project_yaml['spatial']['phenotypes']
    for pheno_name, pheno_def in phenotypes.items():
        if 'base' in pheno_def:
            assert pheno_def['base'] in phenotypes, (
                f"Phenotype '{pheno_name}' references unknown base '{pheno_def['base']}'"
            )


def test_load_config_accepts_project_yaml_format(minimal_project_yaml, tmp_path):
    """load_config() in run_spatial_quantification.py should handle project.yaml format."""
    import sys
    sys.path.insert(0, str(tmp_path.parent.parent))

    config_file = tmp_path / 'project.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(minimal_project_yaml, f)

    # Import load_config from the production module
    from spatial_quantification.run_spatial_quantification import load_config
    config = load_config(str(config_file))

    # Should return the spatial subsection
    assert 'phenotypes' in config
    assert 'tumor_definition' in config


def test_load_config_accepts_standalone_spatial_config(tmp_path):
    """load_config() should also handle a flat spatial_config.yaml (no 'spatial:' key)."""
    from spatial_quantification.run_spatial_quantification import load_config

    standalone_config = {
        'input': {'gated_data': 'gated.h5ad', 'metadata': 'meta.csv'},
        'output': {'base_directory': 'results'},
        'phenotypes': {'T_cells': {'positive': ['CD3']}},
        'statistics': {'test': 'mann_whitney'},
        'visualization': {'enabled': False},
    }

    config_file = tmp_path / 'spatial_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(standalone_config, f)

    config = load_config(str(config_file))
    assert 'phenotypes' in config
    assert 'T_cells' in config['phenotypes']


def test_validate_project_passes_on_valid_config(minimal_project_yaml, tmp_path):
    """validate_project() should return True for a valid config."""
    import yaml
    from run_cifsquant import validate_project

    config_file = tmp_path / 'project.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(minimal_project_yaml, f)

    assert validate_project(minimal_project_yaml, config_file) is True


def test_validate_project_catches_bad_gate_key(minimal_project_yaml, tmp_path):
    """validate_project() should return False on a gate key not in the panel."""
    import copy
    import yaml
    from run_cifsquant import validate_project

    bad_config = copy.deepcopy(minimal_project_yaml)
    bad_config['gating']['gates']['NONEXISTENT_MARKER'] = None

    config_file = tmp_path / 'project.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(bad_config, f)

    assert validate_project(bad_config, config_file) is False


def test_validate_project_segmentation_only_skips_gating_checks(minimal_project_yaml):
    """Segmentation-only runs should not fail on gating/spatial config errors."""
    import copy
    from run_cifsquant import validate_project

    bad_config = copy.deepcopy(minimal_project_yaml)
    bad_config['gating']['gates']['NONEXISTENT_MARKER'] = None

    assert validate_project(bad_config, stages=['segmentation']) is True
    assert validate_project(bad_config, stages=['gating']) is False
