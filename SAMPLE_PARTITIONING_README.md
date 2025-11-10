# Sample Partitioning Preprocessor

A preprocessing tool for partitioning multi-condition samples on the same slide before running the mcmicro pipeline.

## Overview

Sometimes multiple experimental samples/conditions are present on the same physical slide. This tool allows you to:
- Visualize your sample with coordinate grids
- Define custom partitions (regions of interest)
- Automatically split all channels along the same boundaries
- Output partitioned samples ready for the mcmicro pipeline

## When to Use This Tool

Use this preprocessor when:
- You have multiple conditions/samples on one physical slide
- You need to analyze each condition separately
- You want to partition before tiling and segmentation
- Each region needs its own sample name (e.g., JL216 → JL216A, JL216B)

## Directory Structure

```
.
├── rawdata_prepartition/          # Input: Pre-partition samples
│   └── JL216/                     # Your original sample
│       ├── JL216_1.0.1_R000_Cy3_AF_I.ome.tif
│       ├── JL216_1.0.1_R000_Cy5_AF_I.ome.tif
│       ├── JL216_1.0.1_R000_DAPI_AF_I.ome.tif
│       └── ...
│
├── partition_visualizations/       # Generated visualizations
│   └── JL216/
│       ├── JL216_coordinate_grid.png      # Reference for coordinates
│       ├── JL216_partitions.png           # Final partition layout
│       └── JL216_partition_metadata.json  # Partition details
│
└── rawdata/                       # Output: Partitioned samples
    ├── JL216A/                    # Partition A
    │   ├── JL216_1.0.1_R000_Cy3_AF_I.ome.tif
    │   ├── JL216_1.0.1_R000_Cy5_AF_I.ome.tif
    │   └── ...
    └── JL216B/                    # Partition B
        ├── JL216_1.0.1_R000_Cy3_AF_I.ome.tif
        ├── JL216_1.0.1_R000_Cy5_AF_I.ome.tif
        └── ...
```

## Requirements

```bash
pip install tifffile numpy matplotlib scikit-image
```

## Usage

### Step 1: Prepare Your Data

Place your pre-partition samples in `rawdata_prepartition/`:

```bash
mkdir -p rawdata_prepartition/JL216
# Copy your original sample files to rawdata_prepartition/JL216/
```

### Step 2: Run the Partitioning Script

```bash
python scripts/partition_samples.py --sample_name JL216
```

### Step 3: Interactive Partitioning

The script will:

1. **Generate a coordinate grid image**
   - Opens the DAPI channel
   - Creates a PNG with coordinate overlays
   - Saves to `partition_visualizations/JL216/JL216_coordinate_grid.png`

2. **Prompt you for partition information**
   ```
   How many partitions do you want to create? 2

   --- Partition 1/2 (will be saved as JL216A) ---
   Enter the bounding box coordinates for this partition:
     X minimum (0 to 12000): 0
     X maximum (0 to 12000): 6000
     Y minimum (0 to 8000): 0
     Y maximum (0 to 8000): 8000

   --- Partition 2/2 (will be saved as JL216B) ---
   Enter the bounding box coordinates for this partition:
     X minimum (0 to 12000): 6000
     X maximum (6000 to 12000): 12000
     Y minimum (0 to 8000): 0
     Y maximum (0 to 8000): 8000
   ```

3. **Show partition summary and ask for confirmation**
   ```
   Partition Summary:
     JL216A: x=[0:6000], y=[0:8000], size=6000x8000
     JL216B: x=[6000:12000], y=[0:8000], size=6000x8000

   Proceed with partitioning? (yes/no): yes
   ```

4. **Process all channels**
   - Applies the same bounds to every channel file
   - Saves partitioned channels to `rawdata/JL216A/` and `rawdata/JL216B/`
   - Creates metadata files documenting the partition parameters

### Step 4: Run the Pipeline

After partitioning, run the mcmicro pipeline on each partition:

```bash
# Process partition A
bash run_mcmicro_tiled.sh

# Process partition B
bash run_mcmicro_tiled.sh
```

## Advanced Usage

### Custom Directories

```bash
# Use custom source directory
python scripts/partition_samples.py \
    --sample_name JL216 \
    --source_dir /path/to/custom_rawdata \
    --output_dir /path/to/output

# Custom visualization directory
python scripts/partition_samples.py \
    --sample_name JL216 \
    --visualization_dir /path/to/visualizations
```

### Multiple Samples

Process multiple samples in a batch:

```bash
for sample in JL216 JL217 JL218; do
    python scripts/partition_samples.py --sample_name $sample
done
```

## Understanding the Coordinate System

The coordinate grid image shows:
- **Cyan dashed lines**: Vertical grid lines (X coordinates)
- **Yellow dashed lines**: Horizontal grid lines (Y coordinates)
- **Red stars**: Corner coordinates (0,0), (width,0), (0,height), (width,height)
- **Grid spacing**: Lines every 1000 pixels

### Coordinate System
```
(0,0) ────────────────────► X
  │
  │    Your tissue sample
  │
  │
  ▼
  Y
```

- **X increases** from left to right
- **Y increases** from top to bottom
- All coordinates are in pixels

## Tips for Accurate Partitioning

1. **Open the coordinate grid PNG** in an image viewer that shows pixel coordinates
2. **Identify tissue boundaries** using the DAPI signal
3. **Choose non-overlapping regions** for clean separation
4. **Use round numbers** (multiples of 100) for easier tracking
5. **Document your partitions** - the script saves metadata automatically

## Example Partitioning Scenarios

### Horizontal Split (Left/Right)
```
Original: 12000 x 8000 pixels

Partition A (Left half):
  x=[0:6000], y=[0:8000]

Partition B (Right half):
  x=[6000:12000], y=[0:8000]
```

### Vertical Split (Top/Bottom)
```
Original: 12000 x 8000 pixels

Partition A (Top half):
  x=[0:12000], y=[0:4000]

Partition B (Bottom half):
  x=[0:12000], y=[4000:8000]
```

### Quadrant Split
```
Original: 12000 x 8000 pixels

Partition A (Top-left):
  x=[0:6000], y=[0:4000]

Partition B (Top-right):
  x=[6000:12000], y=[0:4000]

Partition C (Bottom-left):
  x=[0:6000], y=[4000:8000]

Partition D (Bottom-right):
  x=[6000:12000], y=[4000:8000]
```

### Custom ROI (Region of Interest)
```
Original: 12000 x 8000 pixels

Partition A (Custom region):
  x=[2000:8000], y=[1500:6500]

Partition B (Another custom region):
  x=[8500:11500], y=[2000:7000]
```

## Output Files

### Per Partition
```
rawdata/JL216A/
├── JL216_1.0.1_R000_Cy3_AF_I.ome.tif    # Partitioned channel
├── JL216_1.0.1_R000_Cy5_AF_I.ome.tif    # Partitioned channel
├── ...
└── partition_metadata.json               # Partition parameters
```

### Visualizations
```
partition_visualizations/JL216/
├── JL216_coordinate_grid.png             # Coordinate reference
├── JL216_partitions.png                  # Final partition overlay
└── JL216_partition_metadata.json         # Complete partition info
```

## Metadata Format

The `partition_metadata.json` contains:
```json
{
  "original_sample": "JL216",
  "partition_name": "JL216A",
  "suffix": "A",
  "bounds": {
    "x_min": 0,
    "x_max": 6000,
    "y_min": 0,
    "y_max": 8000
  },
  "dimensions": {
    "width": 6000,
    "height": 8000
  },
  "num_channels": 45
}
```

## Troubleshooting

### DAPI Channel Not Found
```
Error: No DAPI channel found in /path/to/sample
```
**Solution**: Ensure you have a file with "DAPI" in its name. The script looks for:
- `*DAPI*.ome.tif`
- `*Hoechst*.ome.tif`

### Image Too Large to Display
The script automatically downsamples very large images (>4000 pixels) for visualization. The coordinates shown are still from the original full-resolution image.

### Invalid Coordinates
```
⚠ Invalid X bounds. Must have: 0 <= x_min < x_max <= 12000
```
**Solution**: Check that:
- Minimum is less than maximum
- Values are within image dimensions
- All values are positive integers

### Out of Memory
If processing very large images with many channels:
- Reduce the number of partitions
- Process samples one at a time
- Close other memory-intensive applications

## Integration with MCMICRO Pipeline

After partitioning, each partition is treated as an independent sample:

```bash
# Your directory structure is now ready for the pipeline
rawdata/
├── JL216A/   # Will be processed as "JL216A"
└── JL216B/   # Will be processed as "JL216B"

# Run the pipeline on each partition
bash run_mcmicro_tiled.sh
```

Results will be organized by partition:
```
results/
├── JL216A/
│   └── final/
│       └── combined_quantification.csv
└── JL216B/
    └── final/
        └── combined_quantification.csv
```

## Notes

- **Non-overlapping partitions**: Make sure your partitions don't overlap to avoid duplicate analysis
- **Preserve all channels**: The script processes ALL `.ome.tif` files in the sample directory
- **File naming**: Original filenames are preserved within each partition
- **Metadata tracking**: Partition parameters are saved for reproducibility
- **Quality control**: Always review the partition visualization PNG before confirming

## Support

For issues or questions:
1. Check the visualization PNGs to verify coordinates
2. Review the partition metadata JSON files
3. Ensure source directory structure matches expected format
4. Verify sufficient disk space for partitioned outputs
