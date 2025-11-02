# Quick Start Guide

## Running the Pipeline

The pipeline can be run from **any directory** - paths are automatically resolved relative to the project root.

### Option 1: Run from project root (recommended)

```bash
# From /home/user/cifsQuant/
python spatial_quantification/run_spatial_quantification.py
```

### Option 2: Run from spatial_quantification directory

```bash
cd spatial_quantification
python run_spatial_quantification.py
```

### Option 3: Run with custom config

```bash
python spatial_quantification/run_spatial_quantification.py --config /path/to/my_config.yaml
```

## Path Resolution

The pipeline automatically resolves paths:

**In config file (`spatial_config.yaml`):**
```yaml
input:
  gated_data: 'manual_gating_output/gated_data.h5ad'  # Relative to project root
  metadata: 'sample_metadata.csv'                      # Relative to project root

output:
  base_directory: 'spatial_quantification_results'     # Relative to project root
```

**Absolute paths also work:**
```yaml
input:
  gated_data: '/full/path/to/gated_data.h5ad'
  metadata: '/full/path/to/metadata.csv'
```

## Verification

When you run the pipeline, it will print resolved paths:

```
Resolved paths:
  Gated data: /home/user/cifsQuant/manual_gating_output/gated_data.h5ad
  Metadata: /home/user/cifsQuant/sample_metadata.csv
  Output: /home/user/cifsQuant/spatial_quantification_results
```

**Always check these paths** to ensure they point to the correct files!

## Troubleshooting

### "File not found" errors

If you see `FileNotFoundError`, check the resolved paths printed at the start.

Common issues:
- **Wrong working directory**: The pipeline expects to be run from project root or spatial_quantification/
- **Incorrect config paths**: Update paths in `spatial_config.yaml` to match your file locations
- **Missing gated data**: Ensure `manual_gating.py` has been run first

### Fix path issues

1. Check your current directory:
   ```bash
   pwd
   ```

2. Check where your gated data actually is:
   ```bash
   find . -name "gated_data.h5ad"
   ```

3. Update config accordingly:
   ```yaml
   input:
     gated_data: 'path/from/project/root/to/gated_data.h5ad'
   ```

## Example: Full Workflow

```bash
# 1. Navigate to project root
cd /home/user/cifsQuant

# 2. Check your data exists
ls manual_gating_output/gated_data.h5ad
ls sample_metadata.csv

# 3. Run pipeline
python spatial_quantification/run_spatial_quantification.py

# 4. Check results
ls spatial_quantification_results/
```

## Dependencies

Make sure you have all required packages:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn pyyaml anndata
```

Or use your existing cifsQuant environment.
