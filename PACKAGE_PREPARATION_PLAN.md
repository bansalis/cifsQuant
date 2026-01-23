# cifsQuant Package Preparation Plan

**Goal:** Transform cifsQuant from development repository into a clean, deployable package suitable for distribution and publication.

---

## Phase 1: Code Organization & Cleanup

### 1.1 Archive Deprecated/Experimental Files

Create `archive/` directory to store non-essential files:

#### **Root Directory - Archive Candidates**

```
archive/
├── workflows/
│   ├── mcmicro-tiled-archive.nf          # Archived Nextflow workflow
│   ├── mcmicro-tiled-2stagedidntwork.nf  # Failed two-stage experiment
│   ├── nextflow-archive.config           # Old configuration
│   ├── run_pipeline_archive.sh           # Old pipeline script
│   └── archiverun_phenotyping.sh         # Old phenotyping runner
│
├── scripts_backup/
│   ├── manual_gating_copy.py             # Backup copy
│   ├── scimap_pipeline_copy.py           # Backup copy
│   └── archive_cleanup.sh                # Maintenance script
│
├── experimental/
│   ├── test_tile_correction.py           # Development testing
│   ├── debug_h5ad.py                     # Debugging utility
│   └── diagnose_segmentation.py          # Development diagnostic
│
└── documentation_dev/
    ├── FIXES_APPLIED.md                  # Development history
    ├── INTEGRATION_INSTRUCTIONS.md       # Old integration notes
    └── [other development docs]
```

#### **Scripts Directory - Archive Candidates**

Most scripts in `scripts/` are utility/setup scripts that are not part of the core workflow:

```
archive/scripts/
├── visualization_setup/
│   ├── setup_minerva_viewer.py
│   ├── setup_histocat_complete.py
│   ├── setup_cellprofiler_analyst.py
│   ├── launch_cpa.sh
│   └── create_cellprofiler_project.py
│
├── preprocessing_utilities/
│   ├── realignscript.py                  # Image realignment
│   ├── prep_cellpose_data.py             # Cellpose data prep
│   └── raw_tomato_tumor_detection.py     # Raw tumor detection
│
└── development/
    ├── visual_threshold_validation.py    # Threshold validation
    └── batch_load_slides.py              # Batch processing
```

**Keep in scripts/ (core utilities):**
- `partition_samples.py` - Essential preprocessing
- `tile_from_channels.py` - Core tiling functionality
- `tile_artifact_correction.py` - Core correction
- `phenotype_cells.py` - Core phenotyping

### 1.2 Clean Up manual_gating.py

**Remove deprecated functions:**

```python
# DEPRECATED - to remove
def correct_illumination(adata)                    # Old BaSiC illumination (replaced by UniFORM)
def normalize_tiles_by_background(adata, n_jobs)  # Old normalization
def normalize_samples_after_tiles(adata)          # Old normalization
def landmark_quantile_normalization(adata)        # Old normalization
def normalize_data(adata, method)                 # Generic old method
def integrated_normalization(adata)               # Replaced by hierarchical
def uniform_normalization(adata)                  # Replaced by hierarchical version
def rolling_ball_background(adata, ...)           # Old background correction
def slow_remove_tiling_artifacts(adata)           # Replaced by fast version
def remove_tiling_artifacts(adata)                # Old fast version
def spatial_local_background_correction(adata)    # Old spatial correction
def two_stage_spatial_correction(adata, ...)      # Old two-stage approach
def detect_physical_tiles(adata, sample)          # Deprecated detection
def stratified_subsample(adata, n_per_sample)     # Not used
def auto_suggest_gates(adata)                     # Replaced by density_based_gating
def gmm_gating(adata)                             # Replaced by density_based_gating
def quantile_normalize_tiles(adata)               # Old normalization
def fast_detect_tile_size(adata, ...)             # Deprecated
def load_or_detect_tile_config(adata, ...)        # Deprecated
def assign_tiles_fast(adata, tile_size)           # Deprecated
def detect_and_assign_tiles(adata, ...)           # Deprecated
```

**Remove deprecated configuration:**
```python
OLD_MARKER_HIERARCHY  # Replaced by MARKER_HIERARCHY
```

**Current active pipeline:**
1. `load_and_combine()` - Load data
2. `correct_tile_artifacts_per_marker()` - Tile boundary correction
3. `hierarchical_uniform_normalization()` - Main normalization with UniFORM
4. `density_based_gating()` - Gate determination
5. `apply_hierarchical_gating()` - Apply marker hierarchies
6. `apply_gates()` - Final gate application

### 1.3 Consolidate Documentation

**Keep as core documentation:**
- `README.md` - Main documentation (newly updated)
- `spatial_quantification/README.md` - Spatial analysis documentation
- `spatial_quantification/QUICKSTART.md` - Quick start guide
- `SAMPLE_PARTITIONING_README.md` - Preprocessing guide
- `TILE_ARTIFACT_GUIDE.md` - Technical details

**Archive development documentation:**
- `FIXES_APPLIED.md` → `archive/documentation_dev/`
- `LATEST_UPDATES.md` → `archive/documentation_dev/`
- `SPATIAL_QUANTIFICATION_ENHANCEMENTS.md` → `archive/documentation_dev/`
- `INTEGRATION_INSTRUCTIONS.md` → `archive/documentation_dev/`
- All `IMPLEMENTATION_SUMMARY*.md` files → `archive/documentation_dev/`
- All `OPTIMIZATION*.md` files → `archive/documentation_dev/`
- Various migration/update docs → `archive/documentation_dev/`

---

## Phase 2: Create Package Structure

### 2.1 Modular Package Layout

Reorganize into proper Python package structure:

```
cifsquant/                           # Main package directory
├── __init__.py
├── __version__.py
│
├── segmentation/                    # Stage 1: Segmentation
│   ├── __init__.py
│   ├── tiling.py                    # tile_from_channels functionality
│   ├── workflows/
│   │   ├── mcmicro-tiled.nf         # Nextflow workflow
│   │   └── nextflow.config          # Configuration template
│   └── run_pipeline.sh              # Main launcher
│
├── phenotyping/                     # Stage 2: Cell phenotyping
│   ├── __init__.py
│   ├── gating.py                    # Manual gating core
│   ├── normalization.py             # UniFORM normalization
│   ├── tile_correction.py           # Tile artifact correction
│   └── config_templates/
│       ├── markers_template.csv
│       └── gate_definitions_template.py
│
├── spatial/                         # Stage 3: Spatial quantification
│   ├── __init__.py
│   ├── core/                        # Already well-organized
│   ├── analyses/
│   ├── visualization/
│   ├── stats/
│   ├── config/
│   └── run_spatial_quantification.py
│
├── preprocessing/                   # Stage 0: Optional preprocessing
│   ├── __init__.py
│   └── partition_samples.py
│
├── utils/                           # Shared utilities
│   ├── __init__.py
│   ├── io.py                        # File I/O helpers
│   ├── validation.py                # Input validation
│   └── config_loader.py             # Configuration management
│
└── cli/                             # Command-line interface
    ├── __init__.py
    ├── cifsquant_segment.py         # CLI for segmentation
    ├── cifsquant_phenotype.py       # CLI for phenotyping
    └── cifsquant_spatial.py         # CLI for spatial analysis
```

### 2.2 Configuration Management

**Create centralized configuration system:**

```python
# cifsquant/config/default_config.yaml
segmentation:
  tile_size: 8192
  overlap: 1024
  dapi_channel: 3
  nuc_diameter: 15
  cyto_diameter: 28

phenotyping:
  normalization_method: 'hierarchical_uniform'
  tile_correction:
    enabled: true
    markers: ['GZMB', 'FOXP3', 'KLRG1']

  marker_hierarchy:
    FOXP3: CD4
    GZMB: CD8A
    CD8A: CD3E

spatial:
  tumor_definition:
    base_phenotype: 'Tumor'
    required_positive: ['TOM']
```

---

## Phase 3: Add Package Infrastructure

### 3.1 Setup Files

**Create `setup.py` or `pyproject.toml`:**

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name='cifsquant',
    version='1.0.0',
    description='Spatial immunofluorescence analysis pipeline',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20',
        'pandas>=1.3',
        'scipy>=1.7',
        'scikit-learn>=1.0',
        'matplotlib>=3.4',
        'seaborn>=0.11',
        'anndata>=0.8',
        'pyyaml>=5.4',
        'tifffile>=2021.7',
        'scikit-image>=0.18',
    ],
    extras_require={
        'gpu': ['cupy>=10.0'],
        'visualization': ['napari>=0.4', 'squidpy>=1.2'],
    },
    entry_points={
        'console_scripts': [
            'cifsquant=cifsquant.cli.main:main',
            'cifsquant-segment=cifsquant.cli.cifsquant_segment:main',
            'cifsquant-phenotype=cifsquant.cli.cifsquant_phenotype:main',
            'cifsquant-spatial=cifsquant.cli.cifsquant_spatial:main',
        ],
    },
    python_requires='>=3.8',
)
```

### 3.2 Testing Infrastructure

**Create `tests/` directory:**

```
tests/
├── __init__.py
├── test_segmentation/
│   ├── test_tiling.py
│   └── test_pipeline.py
├── test_phenotyping/
│   ├── test_normalization.py
│   ├── test_gating.py
│   └── test_tile_correction.py
├── test_spatial/
│   ├── test_population_dynamics.py
│   ├── test_infiltration.py
│   └── test_neighborhoods.py
└── fixtures/
    ├── sample_data.h5ad
    └── test_config.yaml
```

### 3.3 Continuous Integration

**Create `.github/workflows/tests.yml`:**

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - run: pip install -e .[dev]
      - run: pytest tests/
```

### 3.4 Documentation Infrastructure

**Create `docs/` directory:**

```
docs/
├── index.md                         # Landing page
├── installation.md                  # Installation guide
├── quickstart.md                    # Quick start tutorial
├── user_guide/
│   ├── segmentation.md
│   ├── phenotyping.md              # Detailed methodology
│   └── spatial_analysis.md
├── methodology/
│   ├── tile_correction.md          # UniFORM normalization details
│   ├── hierarchical_gating.md      # Novel gating approach
│   └── normalization.md
├── api_reference/
│   └── [auto-generated from docstrings]
└── examples/
    ├── basic_workflow.ipynb
    └── advanced_analysis.ipynb
```

---

## Phase 4: Methodology Documentation

### 4.1 Enhanced Manual Gating Documentation

Create detailed methodology documentation highlighting novel approaches:

**`docs/methodology/hierarchical_gating.md`:**

#### **Novel Multi-Level Normalization Pipeline**

cifsQuant implements a sophisticated three-tier normalization strategy:

1. **Tile Boundary Correction (Pre-normalization)**
   - Gradient-based detection of tile artifacts
   - Corrects intensity discontinuities at MCMICRO tile boundaries
   - Applied BEFORE UniFORM to ensure clean input

2. **Hierarchical UniFORM Normalization (Within-sample)**
   - Microscope tile detection via spatial intensity patterns
   - UniFORM (Uniform Manifold Approximation) quantile normalization
   - Corrects illumination variations within each sample
   - Handles both dim and bright tile artifacts
   - Optional radial artifact correction (vignetting)

3. **Cross-Sample Percentile Normalization (Between-samples)**
   - 99th percentile normalization for marker alignment
   - Enables shared gates across all samples
   - Preserves biological variation while normalizing technical noise

#### **Hierarchical Marker Relationships**

Enforces biological parent-child marker relationships:

```python
MARKER_HIERARCHY = {
    'FOXP3': 'CD4',      # Tregs are subset of CD4+ T cells
    'GZMB': 'CD8A',      # Cytotoxic markers require CD8+ T cells
    'CD8A': 'CD3E',      # CD8+ requires T cell marker
    'CD4': 'CD3E',       # CD4+ requires T cell marker
}
```

This ensures logical consistency (e.g., FOXP3+ cells MUST also be CD4+).

#### **Liberal Gating for Rare Markers**

Configurable gating stringency for rare/functional markers:
- **Conservative (default)**: High confidence for common markers
- **Liberal**: Relaxed thresholds for rare markers (GZMB, FOXP3, etc.)
- Prevents under-calling of important rare populations

### 4.2 Add Method Comparison Figures

Create figures demonstrating:
- Before/after tile correction
- UniFORM normalization effects
- Cross-sample gate consistency
- Hierarchical enforcement examples

---

## Phase 5: Deployment Preparation

### 5.1 Container Images

**Create Dockerfile for complete pipeline:**

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3.8 python3-pip \
    openjdk-11-jre \
    && rm -rf /var/lib/apt/lists/*

# Install Nextflow
RUN curl -s https://get.nextflow.io | bash \
    && mv nextflow /usr/local/bin/

# Install cifsQuant
COPY . /opt/cifsquant
RUN pip3 install /opt/cifsquant

WORKDIR /data
ENTRYPOINT ["cifsquant"]
```

### 5.2 Example Datasets

**Create `examples/` with minimal test data:**

```
examples/
├── sample_dataset/
│   ├── rawdata/
│   │   └── SAMPLE1/
│   │       └── [subset of channels]
│   ├── markers.csv
│   ├── sample_metadata.csv
│   └── README.md
│
└── tutorial_notebooks/
    ├── 01_basic_workflow.ipynb
    ├── 02_custom_phenotypes.ipynb
    └── 03_spatial_analysis.ipynb
```

### 5.3 Release Checklist

**Pre-release tasks:**
- [ ] All deprecated code removed
- [ ] Tests passing (>80% coverage)
- [ ] Documentation complete
- [ ] Example notebooks working
- [ ] Docker image builds successfully
- [ ] CLI tools functional
- [ ] Version numbers updated
- [ ] CHANGELOG.md created
- [ ] LICENSE file added (MIT/BSD/GPL)
- [ ] CITATION.cff created

---

## Phase 6: Publication Preparation

### 6.1 Manuscript Materials

**Prepare supplementary materials:**
- Detailed methods section (from methodology docs)
- Benchmarking comparisons (vs. SCIMAP, Squidpy)
- Performance metrics (speed, memory usage)
- Validation on public datasets (if available)

### 6.2 Zenodo Archival

- Create Zenodo deposit for code versioning
- Generate DOI for citation
- Link to GitHub releases

### 6.3 Community Guidelines

**Create contribution guides:**
- `CONTRIBUTING.md` - How to contribute
- `CODE_OF_CONDUCT.md` - Community standards
- Issue templates for bugs/features
- Pull request template

---

## Implementation Timeline

### Week 1-2: Code Cleanup
- Archive deprecated files
- Clean up manual_gating.py
- Consolidate documentation

### Week 3-4: Package Restructuring
- Create modular package structure
- Set up configuration system
- Build CLI interfaces

### Week 5-6: Testing & Documentation
- Write unit tests
- Create methodology documentation
- Build example notebooks

### Week 7-8: Deployment & Release
- Create Docker images
- Final testing
- Prepare release materials

---

## File Movement Summary

### Archive (move to `archive/`)

**Root level:**
- `mcmicro-tiled-archive.nf`
- `mcmicro-tiled-2stagedidntwork.nf`
- `nextflow-archive.config`
- `run_pipeline_archive.sh`
- `archiverun_phenotyping.sh`
- `manual_gating copy.py`
- `archive_cleanup.sh`
- `test_tile_correction.py`
- `debug_h5ad.py`
- `diagnose_segmentation.py`

**Documentation:**
- `FIXES_APPLIED.md`
- `LATEST_UPDATES.md`
- `SPATIAL_QUANTIFICATION_ENHANCEMENTS.md`
- `INTEGRATION_INSTRUCTIONS.md`
- All `IMPLEMENTATION_SUMMARY*.md`
- All `OPTIMIZATION*.md`
- `SPATIALCELLS_MIGRATION.md`
- `REMAINING_TASKS.md`
- Various other development docs

**Scripts:**
- `scripts/setup_minerva_viewer.py`
- `scripts/setup_histocat_complete.py`
- `scripts/setup_cellprofiler_analyst.py`
- `scripts/launch_cpa.sh`
- `scripts/create_cellprofiler_project.py`
- `scripts/realignscript.py`
- `scripts/prep_cellpose_data.py`
- `scripts/raw_tomato_tumor_detection.py`
- `scripts/visual_threshold_validation.py`
- `scripts/batch_load_slides.py`
- `scripts/scimap_pipeline copy.py`

### Keep (core functionality)

**Root level:**
- `README.md` (updated)
- `run_pipeline.sh`
- `manual_gating.py` (cleaned)
- `nextflow.config`
- `mcmicro-tiled.nf`
- `markers.csv`
- `sample_metadata.csv`
- Configuration templates

**Scripts (essential):**
- `scripts/partition_samples.py`
- `scripts/tile_from_channels.py`
- `scripts/tile_artifact_correction.py`
- `scripts/phenotype_cells.py`

**Documentation:**
- `SAMPLE_PARTITIONING_README.md`
- `TILE_ARTIFACT_GUIDE.md`
- `spatial_quantification/README.md`
- `spatial_quantification/QUICKSTART.md`

**Spatial quantification (entire directory):**
- Keep as-is (already well-organized)

---

## Success Criteria

Package is ready for deployment when:

1. **Usability**: New user can install and run pipeline in <30 minutes
2. **Documentation**: All major features documented with examples
3. **Testing**: Core functionality has automated tests
4. **Modularity**: Can use individual components independently
5. **Performance**: Pipeline runs efficiently on standard hardware
6. **Reproducibility**: Same input → same output every time
7. **Community**: Clear contribution guidelines established

---

## Notes

- **Do NOT delete anything yet** - only archive
- Keep git history intact during reorganization
- Tag current state before major restructuring: `git tag pre-packaging`
- Create feature branch for package preparation work
- Consider backward compatibility during refactoring

