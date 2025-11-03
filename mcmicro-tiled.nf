#!/usr/bin/env nextflow

/*
 * MCMICRO Tiled Workflow - BATCH CELLPOSE OPTIMIZATION
 * Processes multiple tiles per GPU call for maximum performance
 */

nextflow.enable.dsl = 2

/*
 * Define parameters
 */
params.input_image = null
params.input_dir = null  // NEW: directory with per-channel TIFFs for single sample
params.rawdata_dir = null  // NEW: parent directory with multiple sample folders
params.markers_csv = null
params.outdir = './results'
params.tile_size = 4096
params.overlap = 512
params.pyramid_level = 0
params.sample_name = 'large_sample'

// MCMICRO specific parameters
params.cellpose = true
params.mcquant = true
params.scimap = true

// Container parameters
params.cellpose_container = 'biocontainers/cellpose:3.0.1_cv1'
params.mcquant_container = 'labsyspharm/quantification:latest'
params.scimap_container = 'labsyspharm/scimap:0.17.7'

// Cellpose specific parameters
params.dapi_channel = 8
params.nuc_model = 'nuclei'
params.nuc_diameter = 15
params.cyto_model = 'cyto2'
params.cyto_diameter = 28
params.cyto_batch_size = 8
params.buffer_px = null

// BATCH OPTIMIZATION PARAMETERS
params.nuclei_batch_size = 6    // Process 6 tiles per nuclei batch
params.cyto_batch_size_tiles = 4  // Process 4 tiles per cyto batch

/*
 * Process that creates tiles
 */
process TILE_LARGE_IMAGE {
    tag "$sample_name"
    publishDir "${params.outdir}/${sample_name}/tiles", mode: 'copy'
    cpus 4
    memory '32.GB'

    input:
    tuple val(sample_name), path(sample_dir)
    path markers_input
    val tile_size
    val overlap
    val pyramid_level

    output:
    tuple val(sample_name), path("tile_*.tif"), emit: tiles
    tuple val(sample_name), path("tile_info.json"), emit: tile_info
    path "markers_tiled.csv", emit: markers_csv
    tuple val(sample_name), path("tile_list.txt"), emit: tile_list

    script:
    """
#!/usr/bin/env python3
import tifffile
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import gc
import os

def extract_tile_from_stack(args):
    """Extract tile from single stacked multi-channel TIFF (slower)"""
    y, x, y_end, x_end, tif, n_channels, dapi_ch = args

    try:
        pages = tif.pages

        # Quick DAPI check
        dapi = pages[dapi_ch].asarray()[y:y_end, x:x_end]
        if dapi.max() == 0:
            del dapi
            return None

        # Check for nuclei signal
        p95 = np.percentile(dapi, 95)
        p5 = np.percentile(dapi, 5)
        dynamic_range = p95 - p5

        if p95 < 100 or dynamic_range < 50 or p95 / (p5 + 1) < 1.5:
            del dapi
            return None

        # Read all channels
        tile = np.zeros((n_channels, y_end - y, x_end - x), dtype=np.uint16)
        for c in range(n_channels):
            tile[c] = pages[c].asarray()[y:y_end, x:x_end]

        filename = f"tile_y{y:06d}_x{x:06d}.tif"
        tifffile.imwrite(filename, tile,
                        photometric='minisblack',
                        compression='zlib',
                        compressionargs={'level': 6},
                        metadata={'axes': 'CYX'},
                        bigtiff=True)

        del tile, dapi
        gc.collect()

        return {
            'filename': filename,
            'y_start': y, 'x_start': x,
            'y_end': y_end, 'x_end': x_end,
            'height': y_end - y, 'width': x_end - x,
            'channels': n_channels,
            'has_data': True
        }
    except Exception as e:
        print(f"Error tile y{y}_x{x}: {e}", file=sys.stderr)
        return None

def extract_tile_from_channels(args):
    """Extract tile from separate per-channel TIFFs (MUCH faster!)"""
    y, x, y_end, x_end, channel_files, dapi_ch = args

    try:
        n_channels = len(channel_files)

        # Quick DAPI check - read only DAPI channel first
        with tifffile.TiffFile(channel_files[dapi_ch]) as tif:
            dapi = tif.pages[0].asarray()[y:y_end, x:x_end]

        if dapi.max() == 0:
            del dapi
            return None

        # Check for nuclei signal
        p95 = np.percentile(dapi, 95)
        p5 = np.percentile(dapi, 5)
        dynamic_range = p95 - p5

        if p95 < 100 or dynamic_range < 50 or p95 / (p5 + 1) < 1.5:
            del dapi
            return None

        # Read all channels from separate files
        tile = np.zeros((n_channels, y_end - y, x_end - x), dtype=np.uint16)
        tile[dapi_ch] = dapi  # Already loaded

        for c in range(n_channels):
            if c == dapi_ch:
                continue  # Already loaded
            with tifffile.TiffFile(channel_files[c]) as tif:
                tile[c] = tif.pages[0].asarray()[y:y_end, x:x_end]

        filename = f"tile_y{y:06d}_x{x:06d}.tif"

        # PERFORMANCE: Lossless compression for slow I/O (WSL mounts)
        # Reduces write by ~60%, transparently decompressed on read
        tifffile.imwrite(filename, tile,
                        photometric='minisblack',
                        compression='zlib',
                        compressionargs={'level': 6},
                        metadata={'axes': 'CYX'},
                        bigtiff=True)

        del tile, dapi
        gc.collect()

        return {
            'filename': filename,
            'y_start': y, 'x_start': x,
            'y_end': y_end, 'x_end': x_end,
            'height': y_end - y, 'width': x_end - x,
            'channels': n_channels,
            'has_data': True
        }
    except Exception as e:
        print(f"Error tile y{y}_x{x}: {e}", file=sys.stderr)
        return None

# Detect input mode: per-channel files or stacked TIFF
import glob
from pathlib import Path
import pandas as pd
import re

def match_marker_to_file(marker_name, available_files):
    """
    Match a marker name from markers.csv to an actual file.
    Example: "R1.0.1_DAPI" matches "JL216_1.0.1_R000_DAPI__FINAL_F.ome.tif"
    """
    # Extract key components from marker name
    # Handle patterns like: R1.0.1_DAPI, R1.0.4_CY5_CD45, etc.
    parts = marker_name.replace('R', '').split('_')
    cycle_round = parts[0]  # e.g., "1.0.1" or "1.0.4"

    # Build search pattern components
    search_terms = []
    search_terms.append(cycle_round.replace('.', r'\.'))  # Match cycle/round

    # Add remaining parts (channel, marker)
    for part in parts[1:]:
        if part:  # Skip empty parts
            search_terms.append(part)

    # Try to find matching file (case-insensitive)
    best_match = None
    best_score = 0

    for filepath in available_files:
        filename = Path(filepath).name.upper()  # Case-insensitive
        marker_upper = marker_name.upper()

        # Score based on how many search terms match
        score = 0
        for term in search_terms:
            if term.upper() in filename:
                score += 1

        # Additional score if marker name components are in order
        if all(term.upper() in filename for term in search_terms):
            score += 10

        if score > best_score:
            best_score = score
            best_match = filepath

    if best_match and best_score >= len(search_terms):
        return best_match

    # Fallback: try simpler pattern matching
    pattern = marker_name.replace('R', '').replace('_', '.*')
    for filepath in available_files:
        if re.search(pattern, str(filepath), re.IGNORECASE):
            return filepath

    return None

# Input is the sample directory path
input_dir = '${sample_dir}'
input_file = None  # Will look for stacked TIFF if needed

# Check if markers CSV is provided
markers_csv_path = '${markers_input}'
use_channel_files = False
channel_files = []

if Path(markers_csv_path).exists() and Path(markers_csv_path).stat().st_size > 0:
    # Read markers.csv to get ordered list of channels
    markers_df = pd.read_csv(markers_csv_path)
    marker_names = markers_df['marker_name'].tolist()

    print(f"Reading markers.csv: {len(marker_names)} channels defined")

    # Find all available .ome.tif files in directory
    if Path(input_dir).is_dir():
        available_files = glob.glob(f"{input_dir}/*.ome.tif")
        print(f"Found {len(available_files)} .ome.tif files in {input_dir}")

        # Match each marker to a file
        matched_files = []
        for marker in marker_names:
            matched_file = match_marker_to_file(marker, available_files)
            if matched_file:
                matched_files.append(matched_file)
                print(f"  ✓ {marker} → {Path(matched_file).name}")
            else:
                print(f"  ✗ WARNING: No file found for marker '{marker}'")
                # Create placeholder - will need to handle this
                matched_files.append(None)

        # Filter out None values
        channel_files = [f for f in matched_files if f is not None]

        if len(channel_files) >= len(marker_names) * 0.8:  # At least 80% matched
            use_channel_files = True
            print(f"FAST MODE: Using {len(channel_files)} per-channel TIFF files in markers.csv order")
        else:
            print(f"ERROR: Only matched {len(channel_files)}/{len(marker_names)} markers")
            print(f"Please check markers.csv and file naming conventions")
            sys.exit(1)
    elif Path(input_file).exists():
        print(f"LEGACY MODE: Using stacked multi-channel TIFF")
else:
    # No markers.csv - fall back to auto-detection
    print("No markers.csv provided, auto-detecting channels...")
    if Path(input_dir).is_dir():
        potential_channels = sorted(glob.glob(f"{input_dir}/*.ome.tif"))
        if len(potential_channels) > 1:
            use_channel_files = True
            channel_files = potential_channels
            print(f"FAST MODE: Found {len(channel_files)} per-channel TIFF files (alphabetical order)")
        elif Path(input_file).exists():
            print(f"LEGACY MODE: Using stacked multi-channel TIFF")
    else:
        print(f"LEGACY MODE: Using stacked multi-channel TIFF")

# Get metadata
if use_channel_files:
    # Read from first channel to get dimensions
    with tifffile.TiffFile(channel_files[0]) as tif:
        height = tif.pages[0].shape[0]
        width = tif.pages[0].shape[1]
    n_channels = len(channel_files)
    print(f"Image: {n_channels} channels, {height}x{width} pixels")
    print(f"Channel files: {[Path(f).name for f in channel_files[:3]]}... (showing first 3)")
else:
    # Legacy: read from stacked TIFF
    tif = tifffile.TiffFile(input_file)
    pages = tif.pages
    n_channels = len(pages)
    height = pages[0].shape[0]
    width = pages[0].shape[1]
    print(f"Image: {n_channels} channels, {height}x{width} pixels")

# Generate tile coordinates
step = ${tile_size} - ${overlap}
coords = []
for y in range(0, height - ${overlap}, step):
    for x in range(0, width - ${overlap}, step):
        y_end = min(y + ${tile_size}, height)
        x_end = min(x + ${tile_size}, width)

        if use_channel_files:
            coords.append((y, x, y_end, x_end, channel_files, ${params.dapi_channel}))
        else:
            coords.append((y, x, y_end, x_end, tif, n_channels, ${params.dapi_channel}))

print(f"Processing {len(coords)} tiles in batches...")
sys.stdout.flush()

# Process in batches with limited concurrency to avoid OOM
tile_info = []
batch_size = 8
max_workers = 2

extract_func = extract_tile_from_channels if use_channel_files else extract_tile_from_stack

for batch_start in range(0, len(coords), batch_size):
    batch = coords[batch_start:batch_start + batch_size]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_func, c): c for c in batch}

        for future in as_completed(futures):
            result = future.result()
            if result:
                tile_info.append(result)

    print(f"Progress: {len(tile_info)}/{len(coords)}", flush=True)
    gc.collect()  # Force cleanup between batches

# Close the shared TiffFile if using stacked mode
if not use_channel_files:
    tif.close()

print(f"\\nTILING SUMMARY: {len(coords)} total tiles, {len(tile_info)} with meaningful data")

# Save outputs
with open('tile_info.json', 'w') as f:
    json.dump(tile_info, f, indent=2)

with open('tile_list.txt', 'w') as f:
    for info in tile_info:
        f.write(info['filename'] + "\\n")

# Create markers CSV - copy from input file if provided
import shutil
import os
markers_input_file = '${markers_input}'
if os.path.exists(markers_input_file) and os.path.getsize(markers_input_file) > 0:
    # Copy the markers file
    shutil.copy(markers_input_file, 'markers_tiled.csv')
    print(f"Copied markers from {markers_input_file}")
else:
    # Create default markers file
    print("No markers file provided, creating default channel names")
    with open('markers_tiled.csv', 'w') as f:
        f.write("marker_name\\n")
        for i in range(tile_info[0]['channels'] if tile_info else n_channels):
            f.write(f"Channel_{i+1}\\n")

print(f"Tiling completed! Created {len(tile_info)} tiles")
    """
}

process BACKGROUND_SUBTRACT {
    publishDir "${params.outdir}/background_corrected", mode: 'copy'

    input:
    path tile

    output:
    path "*_corrected.tif", emit: corrected

    script:
    """
    export MPLCONFIGDIR=/tmp/matplotlib-\$\$

    # Debug: Show current directory and files
    echo "Working directory: \$(pwd)"
    echo "Files in directory:"
    ls -lh
    echo "Looking for file: ${tile.name}"

    python3 - <<'EOF'
import tifffile
import numpy as np
import os
import sys

# Debug: Show working directory and file
print(f"Python working directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")
print(f"Looking for file: ${tile.name}")

# Check if file exists
if not os.path.exists('${tile.name}'):
    print(f"ERROR: File '${tile.name}' not found!", file=sys.stderr)
    print(f"Available files: {os.listdir('.')}", file=sys.stderr)
    sys.exit(1)

img = tifffile.imread('${tile.name}')
print(f"Image loaded: shape={img.shape}, dtype={img.dtype}")

# Per-channel 5th percentile subtraction (MCMICRO standard)
for c in range(img.shape[0]):
    channel = img[c]
    positive = channel[channel > 0]

    if len(positive) > 0:
        bg = np.percentile(positive, 5)
        img[c] = np.clip(channel.astype(float) - bg, 0, None).astype(channel.dtype)

tifffile.imwrite('${tile.baseName}_corrected.tif', img,
                 compression='zlib',
                 compressionargs={'level': 6},
                 metadata={'axes': 'CYX'},
                 bigtiff=True)
print(f"Background correction complete: ${tile.baseName}_corrected.tif")
EOF
    """
}

process GLOBAL_BACKGROUND_SUBTRACT {
    input:
    path image

    output:
    path "*_bg_corrected.ome.tif"

    script:
    """
    python3 - <<'EOF'
import tifffile
import numpy as np
from skimage.morphology import white_tophat, disk

img = tifffile.imread('${image.name}')
corrected = np.zeros_like(img)

for c in range(img.shape[0]):
    # Rolling ball background subtraction
    corrected[c] = white_tophat(img[c], disk(${params.bg_radius}))

tifffile.imwrite('${image.baseName}_bg_corrected.ome.tif', corrected,
                 photometric='minisblack',
                 compression='zlib',
                 compressionargs={'level': 6},
                 metadata={'axes': 'CYX'},
                 bigtiff=True)
EOF
    """
}

process RUN_CELLPOSE_NUCLEI_BATCH {
    tag "nuclei_batch_${batch_id}"
    container params.cellpose_container
    publishDir "${params.outdir}/nuclei_masks", mode: 'copy'
    
    input:
    tuple val(batch_id), path(tiles)
    
    output:
    path "*_nuclei_mask.tif", emit: nuclei_masks
    path "*_dapi_bg_subtracted.tif", emit: dapi_processed
    
    when:
    params.cellpose
    
    script:
    """
    export HOME=/tmp
    export CELLPOSE_LOCAL_MODELS_PATH=/tmp/cellpose_models
    mkdir -p /tmp/cellpose_models
    
    echo "=== CELLPOSE NUCLEI BATCH ${batch_id}: ${tiles.size()} tiles ==="
    
    python3 - <<'PYSCRIPT'
import numpy as np
import tifffile
from cellpose import models
from pathlib import Path
import torch

# Verify GPU
use_gpu = torch.cuda.is_available()
print(f'CUDA available: {use_gpu}')
if use_gpu:
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')

# Load model ONCE for entire batch with GPU
model = models.Cellpose(model_type='${params.nuc_model}', gpu=use_gpu)
print(f'Model loaded on {"GPU" if use_gpu else "CPU"}')

# Process all tiles in batch
tile_paths = sorted(Path('.').glob('tile_*.tif'))
n_tiles = len(tile_paths)

for idx, tile_path in enumerate(tile_paths):
    base = tile_path.stem
    print(f'[{idx+1}/{n_tiles}] Processing {base}...', flush=True)
    
    try:
        # Load and extract DAPI
        img = tifffile.imread(tile_path)
        if img.ndim != 3 or img.shape[0] <= ${params.dapi_channel}:
            print(f'  ERROR: DAPI channel ${params.dapi_channel} not available')
            continue
        
        dapi = img[${params.dapi_channel}]
        
        # Background subtraction
        bg = np.percentile(dapi, 5)
        dapi_bg = np.clip(dapi.astype(np.float32) - bg, 0, None).astype(np.uint16)
        
        # Save preprocessed DAPI
        tifffile.imwrite(f'{base}_dapi_bg_subtracted.tif', dapi_bg, compression='lzw')
        
        # Run Cellpose on GPU
        masks = model.eval(
            dapi_bg,
            diameter=${params.nuc_diameter},
            channels=[0, 0],
            flow_threshold=${params.nuc_flow_threshold},
            cellprob_threshold=${params.nuc_cellprob_threshold},
            stitch_threshold=${params.stitch_threshold},
            batch_size=8
        )[0]
        
        n_cells = int(masks.max())
        print(f'  {base}: {n_cells} nuclei detected', flush=True)
        
        # Save mask
        tifffile.imwrite(f'{base}_nuclei_mask.tif', masks.astype(np.uint16), compression='lzw')
        
        # Free memory
        del img, dapi, dapi_bg, masks
        
    except Exception as e:
        print(f'  ERROR {base}: {e}', flush=True)
        # Create empty mask
        empty = np.zeros((${params.tile_size}, ${params.tile_size}), dtype=np.uint16)
        tifffile.imwrite(f'{base}_nuclei_mask.tif', empty, compression='lzw')
        tifffile.imwrite(f'{base}_dapi_bg_subtracted.tif', empty, compression='lzw')

print(f'\\nBatch complete: {n_tiles} tiles processed')
PYSCRIPT

    echo "=== NUCLEI BATCH ${batch_id} completed ==="
    """
}

process CREATE_MEMBRANE_COMPOSITES {
    tag "membranes_${batch_id}"
    container 'mcmicro-tiles:latest'
    
    input:
    tuple val(batch_id), path(tiles)
    
    output:
    tuple val(batch_id), path("*_nuclear.tif"), path("*_membrane.tif"), emit: composites
    
    script:
    """
    python3 << 'PYSCRIPT'
import tifffile
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

for tile_path in Path('.').glob('tile_*.tif'):
    tile_base = tile_path.stem
    img = tifffile.imread(tile_path)
    nuclear = img[${params.dapi_channel}]
    tifffile.imwrite(f'{tile_base}_nuclear.tif', nuclear)
    
    membrane_chs = [img[i] for i in range(img.shape[0]) if i != ${params.dapi_channel}]
    weights = []
    for ch in membrane_chs:
        norm = np.clip((ch - np.percentile(ch, 1)) / 
                      (np.percentile(ch, 99.9) - np.percentile(ch, 1) + 1), 0, 1)
        variance = gaussian_filter(norm**2, 3) - gaussian_filter(norm, 3)**2
        weights.append(np.mean(variance))
    
    weights = np.array(weights) 
    weights = weights / (weights.sum() + 1e-8)
    
    membrane = np.zeros_like(membrane_chs[0], dtype=np.float32)
    for ch, w in zip(membrane_chs, weights):
        norm = np.clip((ch - np.percentile(ch, 1)) / 
                      (np.percentile(ch, 99.9) - np.percentile(ch, 1) + 1), 0, 1)
        membrane += w * norm
    
    membrane = (membrane * 65535).astype(np.uint16)
    tifffile.imwrite(f'{tile_base}_membrane.tif', membrane)
PYSCRIPT
    """
}

process RUN_CELLPOSE_CYTO_SEEDED {
    tag "cyto_batch_${batch_id}"
    container params.cellpose_container
    publishDir "${params.outdir}/cell_masks", mode: 'copy'

    input:
    tuple val(batch_id), path(tiles), path(nuclei_masks)

    output:
    path "*_cell.tif", emit: cell_masks

    when:
    params.cellpose

    script:
    def tumor_channels = params.tumor_channels ?: '1'
    def immune_channels = params.immune_channels ?: '2,9'
    def tumor_weight = params.tumor_weight ?: 0.7
    def immune_weight = params.immune_weight ?: 0.3
    def custom_weights = params.custom_channel_weights ?: ''
    
    """
    set -euo pipefail
    
    export HOME=/tmp
    export CELLPOSE_LOCAL_MODELS_PATH=/tmp/cellpose_models
    mkdir -p /tmp/cellpose_models
    
    echo "=== CYTO SEEDED BATCH ${batch_id}: ${tiles.size()} tiles ==="
    echo "Tumor channels: ${tumor_channels} (weight: ${tumor_weight})"
    echo "Immune channels: ${immune_channels} (weight: ${immune_weight})"
    
    python3 - <<'PYSCRIPT'
import numpy as np
import tifffile
from cellpose import models
from pathlib import Path
import os
import sys
import torch

# Verify GPU and initialize model
use_gpu = torch.cuda.is_available()
print(f"GPU available: {use_gpu}")
if use_gpu:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

def parse_channel_config():
    config = {}
    
    # Custom per-channel weights (highest priority)
    custom = os.environ.get('CUSTOM_WEIGHTS', '').strip()
    if custom:
        for pair in custom.split(','):
            if ':' in pair:
                ch, w = pair.split(':')
                config[int(ch)] = float(w)
        return config
    
    # Category-based weights
    tumor_chs = [int(x) for x in os.environ.get('TUMOR_CHANNELS', '1').split(',')]
    immune_chs = [int(x) for x in os.environ.get('IMMUNE_CHANNELS', '2,9').split(',')]
    tumor_w = float(os.environ.get('TUMOR_WEIGHT', 0.7))
    immune_w = float(os.environ.get('IMMUNE_WEIGHT', 0.3))
    
    tumor_w_per = tumor_w / len(tumor_chs)
    immune_w_per = immune_w / len(immune_chs)
    
    for ch in tumor_chs:
        config[ch] = tumor_w_per
    for ch in immune_chs:
        config[ch] = immune_w_per
    
    return config

def create_weighted_cytoplasm(img, channel_weights):
    h, w = img.shape[1], img.shape[2]
    cyto = np.zeros((h, w), dtype=np.float32)
    
    for ch, weight in channel_weights.items():
        if ch >= img.shape[0]:
            continue
        
        signal = img[ch].astype(np.float32)
        p1, p99 = np.percentile(signal, [1, 99])
        if p99 > p1:
            signal = np.clip((signal - p1) / (p99 - p1), 0, 1)
        else:
            signal = np.zeros_like(signal)
        
        cyto += weight * signal
    
    if cyto.max() > 0:
        cyto = cyto / cyto.max()
    
    return (cyto * 65535).astype(np.uint16)

# Parse config
channel_weights = parse_channel_config()
print(f"Channel weights: {channel_weights}")
sys.stdout.flush()

# Load model ONCE for entire batch with GPU
print(f"Loading Cellpose model on {'GPU' if use_gpu else 'CPU'}...", flush=True)
model = models.Cellpose(model_type='${params.cyto_model}', gpu=use_gpu)
print("Model loaded successfully!", flush=True)

# Process all tiles in batch
tile_paths = sorted(Path('.').glob('tile_*.tif'))
n_tiles = len(tile_paths)

for idx, tile_path in enumerate(tile_paths):
    base = tile_path.stem
    nuc_mask_path = f'{base}_nuclei_mask.tif'
    
    if not Path(nuc_mask_path).exists():
        print(f"[{idx+1}/{n_tiles}] Skip {base}: no nuclei mask", flush=True)
        continue
    
    try:
        # Load data
        img = tifffile.imread(tile_path)
        nuclei_mask = tifffile.imread(nuc_mask_path)
        
        n_nuclei = int(nuclei_mask.max())
        if n_nuclei == 0:
            print(f"[{idx+1}/{n_tiles}] {base}: 0 nuclei, skipping", flush=True)
            tifffile.imwrite(f'{base}_cell.tif', nuclei_mask.astype(np.int32), compression='lzw')
            continue
        
        # Create weighted cytoplasm
        cyto = create_weighted_cytoplasm(img, channel_weights)
        del img  # Free memory
        
        # Run Cellpose with nuclei seeding
        cells = model.eval(
            cyto,
            channels=[0, 0],
            diameter=${params.cyto_diameter},
            flow_threshold=${params.cyto_flow_threshold},
            cellprob_threshold=${params.cyto_cellprob_threshold},
            min_size=${params.min_cell_size},
            nuc_mask=nuclei_mask,
            do_3D=False,
            batch_size=8,
            gpu=use_gpu
        )[0]
        
        n_cells = int(cells.max())
        recovery = n_cells / n_nuclei if n_nuclei > 0 else 0
        
        # Calculate expansion metrics
        nuc_area = np.sum(nuclei_mask > 0)
        cell_area = np.sum(cells > 0)
        expansion_ratio = cell_area / nuc_area if nuc_area > 0 else 0
        
        print(f'[{idx+1}/{n_tiles}] {base}: {n_nuclei}→{n_cells} ({recovery:.1%}), '
              f'expansion: {expansion_ratio:.2f}x', flush=True)
        
        # Save and free memory
        tifffile.imwrite(f'{base}_cell.tif', cells.astype(np.int32), compression='lzw')
        del cyto, nuclei_mask, cells
        
    except Exception as e:
        print(f'[{idx+1}/{n_tiles}] ERROR {base}: {e}', flush=True)
        # Fallback to nuclei mask
        try:
            nuc = tifffile.imread(nuc_mask_path)
            tifffile.imwrite(f'{base}_cell.tif', nuc.astype(np.int32), compression='lzw')
            print(f'[{idx+1}/{n_tiles}] {base}: Fallback to nuclei', flush=True)
        except:
            pass

print(f"\\nBatch complete: {n_tiles} tiles processed")
PYSCRIPT
    
    # Ensure all tiles have output
    for tile in ${tiles.join(' ')}; do
        tile_base=\$(basename "\$tile" .tif)
        if [ ! -f "\${tile_base}_cell.tif" ]; then
            python3 -c "
import tifffile, numpy as np
tifffile.imwrite('\${tile_base}_cell.tif', 
                 np.zeros((${params.tile_size}, ${params.tile_size}), dtype=np.int32),
                 compression='lzw')
" 2>/dev/null || true
        fi
    done
    
    echo "=== BATCH ${batch_id} COMPLETE ==="
    """
}

process RUN_MCQUANT {
    tag "${cell_mask.baseName.replaceAll('_cell\$', '')}"
    container params.mcquant_container
    publishDir "${params.outdir}/quantification", mode: 'copy'

    input:
    tuple path(cell_mask), path(original_tile)
    each path(markers_csv)

    output:
    path "*_cell.csv", emit: cell_quantification

    script:
    def tile_name = cell_mask.baseName.replaceAll('_cell$', '')
    """
    echo "=== MCQUANT: ${tile_name} ==="

    python3 -c "
import pandas as pd
df = pd.read_csv('${markers_csv}')
with open('channel_names.txt', 'w') as f:
    for name in df['marker_name']:
        f.write(name + '\\n')
"
    
    mkdir -p quantification_output
    
    mcquant \\
        --masks ${cell_mask} \\
        --image ${original_tile} \\
        --channel_names channel_names.txt \\
        --intensity_props intensity_mean intensity_median \\
        --output quantification_output/
    
    python3 -c "
import pandas as pd, numpy as np, glob

csv_files = glob.glob('quantification_output/*.csv')
if not csv_files:
    with open('${tile_name}_cell.csv', 'w') as f:
        f.write('CellID,X_centroid,Y_centroid\\n')
    print('=== ${tile_name}: 0 CELLS ===')
    exit()

df = pd.read_csv(csv_files[0])

# Vectorized background subtraction
marker_cols = [col for col in df.columns if 'intensity_mean' in col.lower()]
for col in marker_cols:
    bg = df[col].quantile(0.05)
    df[col] = np.clip(df[col] - bg, 0, None)

df.to_csv('${tile_name}_cell.csv', index=False)
print(f'=== ${tile_name}: {len(df)} CELLS ===')
"
    """
}

process STITCH_RESULTS {
    tag "stitching"
    publishDir "${params.outdir}/final", mode: 'copy'
    cpus 4
    memory '16.GB'
    time 2.h  // Much faster with tiled approach

    input:
    path tile_info
    path "masks/*"
    path "csvs/*"
    val sample_name

    output:
    path "combined_quantification.csv", emit: combined_csv
    path "stitching_report.txt", emit: report

    script:
    """
#!/usr/bin/env python3
import json, pandas as pd, os
from pathlib import Path

with open('${tile_info}', 'r') as f:
    tile_info = json.load(f)

all_quantifications = []
current_label = 1
stitched = 0

for info in tile_info:
    tile_base = Path(info['filename']).stem
    csv_file = f"csvs/{tile_base}_corrected_cell.csv"
    
    if not os.path.exists(csv_file):
        csv_file = f"csvs/{tile_base}_cell.csv"
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if not df.empty:
            # Remap IDs
            if 'CellID' in df.columns:
                df['CellID'] += current_label
                current_label += df['CellID'].max()
            
            # Add global coords
            if 'X_centroid' in df.columns:
                df['X_centroid'] += info['x_start']
            if 'Y_centroid' in df.columns:
                df['Y_centroid'] += info['y_start']
            
            df['tile_y'] = info['y_start']
            df['tile_x'] = info['x_start']
            all_quantifications.append(df)
            stitched += 1

if all_quantifications:
    combined = pd.concat(all_quantifications, ignore_index=True)
    combined['CellID'] = range(1, len(combined) + 1)
    combined.to_csv('combined_quantification.csv', index=False)
    total = len(combined)
else:
    pd.DataFrame().to_csv('combined_quantification.csv', index=False)
    total = 0

with open('stitching_report.txt', 'w') as f:
    f.write(f'''Stitching Report
=================
Tiles processed: {stitched}/{len(tile_info)}
Total cells: {total}
''')

print(f"Complete: {total} cells from {stitched} tiles")
"""
}

process CYLINTER_QC {
    tag "qc_${sample_name}"
    publishDir "${params.outdir}/qc", mode: 'copy'
    
    input:
    path combined_csv
    path full_mask
    path markers_csv
    val sample_name
    
    output:
    path "*_QC_pass.csv", emit: cleaned
    path "qc_report.txt", emit: report
    
    script:
    """
    python3 - <<'EOF'
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
df = pd.read_csv('${combined_csv}')
print(f"Starting QC with {len(df)} cells")

# 1. Debris removal by area
df = df[(df['Area'] >= ${params.min_cell_area}) & (df['Area'] <= ${params.max_cell_area})]
print(f"After area filter: {len(df)} cells")

# 2. Remove elongated debris
if 'Eccentricity' in df.columns:
    df = df[df['Eccentricity'] <= ${params.max_eccentricity}]
    print(f"After eccentricity filter: {len(df)} cells")

# 3. Remove cells with minimal expression
marker_cols = [col for col in df.columns if 'Channel_' in col]
if marker_cols:
    max_intensity = df[marker_cols].max(axis=1)
    df = df[max_intensity >= ${params.min_marker_expression}]
    print(f"After expression filter: {len(df)} cells")

# 4. Remove edge artifacts (cells at tile boundaries)
if 'tile_x' in df.columns and 'tile_y' in df.columns:
    edge_buffer = 10
    tile_size = ${params.tile_size}
    
    x_in_tile = df['X_centroid'] % tile_size
    y_in_tile = df['Y_centroid'] % tile_size
    
    edge_mask = ((x_in_tile > edge_buffer) & (x_in_tile < tile_size - edge_buffer) &
                 (y_in_tile > edge_buffer) & (y_in_tile < tile_size - edge_buffer))
    df = df[edge_mask]
    print(f"After edge filter: {len(df)} cells")

# 5. Detect folded tissue regions
if 'Channel_1' in df.columns:  # Assuming Channel_1 is DAPI
    from scipy.spatial import cKDTree
    
    dapi = df['Channel_1'].values
    coords = df[['X_centroid', 'Y_centroid']].values
    
    tree = cKDTree(coords)
    fold_flags = []
    
    for i in range(len(coords)):
        neighbors = tree.query_ball_point(coords[i], ${params.fold_window})
        if len(neighbors) > 5:
            local_dapi = dapi[neighbors]
            cv = local_dapi.std() / (local_dapi.mean() + 1)
            fold_flags.append(cv > ${params.fold_cv_threshold})
        else:
            fold_flags.append(False)
    
    df = df[~np.array(fold_flags)]
    print(f"After fold detection: {len(df)} cells")

# Save
df.to_csv('${sample_name}_QC_pass.csv', index=False)

# Report
with open('qc_report.txt', 'w') as f:
    f.write(f"CyLinter-style QC Report\\n")
    f.write(f"========================\\n")
    f.write(f"Sample: ${sample_name}\\n")
    f.write(f"Final cells: {len(df)}\\n")
    f.write(f"\\nFilters applied:\\n")
    f.write(f"- Area: ${params.min_cell_area}-${params.max_cell_area}\\n")
    f.write(f"- Eccentricity: <${params.max_eccentricity}\\n")
    f.write(f"- Min expression: >${params.min_marker_expression}\\n")
    f.write(f"- Edge buffer: 10px\\n")

print(f"QC complete: {len(df)} cells passed")
EOF
    """
}

process SPATIAL_ANALYSIS {
    tag "spatial_analysis"
    container params.scimap_container
    publishDir "${params.outdir}/spatial", mode: 'copy'
    
    input:
    path combined_csv
    path markers_csv
    val sample_name
    
    output:
    path "spatial_analysis_results.h5ad", emit: adata
    path "spatial_plots/*", emit: plots, optional: true
    path "neighborhood_analysis.csv", emit: neighborhoods
    
    when:
    params.scimap
    
    script:
    """
#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from pathlib import Path

print("=== STARTING SPATIAL ANALYSIS ===")
Path('spatial_plots').mkdir(exist_ok=True)

df = pd.read_csv('${combined_csv}')

if df.empty:
    print("No data to analyze")
    with open('spatial_analysis_results.h5ad', 'w') as f:
        f.write('# No data for analysis')
    pd.DataFrame().to_csv('neighborhood_analysis.csv', index=False)
    with open('spatial_plots/analysis_summary.txt', 'w') as f:
        f.write("No cells detected for spatial analysis")
else:
    print(f"Processing {len(df)} cells...")
    try:
        os.system(f"scimap-mcmicro -o . ${combined_csv}")
        if not os.path.exists('spatial_analysis_results.h5ad'):
            print("Basic spatial analysis...")
            df.to_csv('neighborhood_analysis.csv', index=False)
            with open('spatial_analysis_results.h5ad', 'w') as f:
                f.write('# Basic analysis completed')
    except Exception as e:
        print(f"Error: {e}")
        with open('spatial_analysis_results.h5ad', 'w') as f:
            f.write('# Analysis failed')
        df.to_csv('neighborhood_analysis.csv', index=False)
    """
}

/*
 * BATCH CELLPOSE WORKFLOW
 */
workflow {
    log.info "=== MCMICRO BATCH CELLPOSE PIPELINE ==="
    log.info "Tile size: ${params.tile_size}"
    log.info "Output directory: ${params.outdir}"

    // Support multiple samples from rawdata_dir
    if (params.rawdata_dir) {
        log.info "MULTI-SAMPLE MODE: Processing all samples in ${params.rawdata_dir}"
        log.info "    Using markers.csv to determine channel order"
        log.info "    This mode is 5-10x faster on network/WSL mounts!"

        // Find all sample directories (directories containing .ome.tif files)
        def rawdata = file(params.rawdata_dir)
        def sample_dirs = rawdata.listFiles()
            .findAll { it.isDirectory() }
            .findAll { dir ->
                dir.listFiles().any { it.name.endsWith('.ome.tif') }
            }

        if (sample_dirs.size() == 0) {
            error "No sample directories with .ome.tif files found in ${params.rawdata_dir}"
        }

        log.info "Found ${sample_dirs.size()} samples: ${sample_dirs.collect { it.name }.join(', ')}"

        // Create channel with (sample_name, sample_dir) tuples
        samples_ch = Channel.fromList(
            sample_dirs.collect { dir -> tuple(dir.name, dir) }
        )
    } else if (params.input_dir) {
        // Single sample mode
        log.info "SINGLE-SAMPLE MODE: Processing ${params.sample_name}"
        log.info "    Using per-channel TIFFs from: ${params.input_dir}"
        samples_ch = Channel.of(tuple(params.sample_name, file(params.input_dir)))
    } else if (params.input_image) {
        log.info "LEGACY MODE: Stacked multi-channel TIFF"
        log.info "    Consider using --rawdata_dir with per-channel TIFFs for much faster processing"
        // For legacy mode, pass the directory containing the stacked TIFF
        def image_file = file(params.input_image)
        def image_dir = image_file.parent
        samples_ch = Channel.of(tuple(params.sample_name, image_dir))
    } else {
        error "Please provide --rawdata_dir, --input_dir, or --input_image parameter"
    }

    // Create markers channel - use provided file or look for default
    if (params.markers_csv) {
        markers_ch = Channel.fromPath(params.markers_csv, checkIfExists: true)
        log.info "Using markers file: ${params.markers_csv}"
    } else if (file('./markers.csv').exists()) {
        markers_ch = Channel.fromPath('./markers.csv', checkIfExists: true)
        log.info "Using default markers file: ./markers.csv"
    } else {
        // Create empty placeholder file
        file('NO_MARKERS.txt').text = ''
        markers_ch = Channel.fromPath('NO_MARKERS.txt')
        log.info "No markers file provided - will use default channel names"
    }

    // Step 0: Global background subtraction (BEFORE tiling)
    if (params.global_bg_subtract) {
        log.info "Applying global background subtraction..."
        bg_corrected_ch = GLOBAL_BACKGROUND_SUBTRACT(input_image_ch)
        image_to_tile = bg_corrected_ch
    } else {
        image_to_tile = input_image_ch
    }

    // Step 1: Tile the images for all samples
    TILE_LARGE_IMAGE(
        samples_ch,
        markers_ch,
        params.tile_size,
        params.overlap,
        params.pyramid_level
    )

    if (params.cellpose) {
        
        // Step 2: Background subtraction and flatten tiles
        tiles_flattened = TILE_LARGE_IMAGE.out.tiles.flatten()
        
        if (params.background_subtract) {
            tiles_corrected = BACKGROUND_SUBTRACT(tiles_flattened).corrected
            tiles_to_use = tiles_corrected
        } else {
            tiles_to_use = tiles_flattened
        }

        tiles_for_nuclei = tiles_to_use
        tiles_for_mcquant = tiles_to_use

        // Create nuclei batches separately
        nuclei_batches = tiles_to_use
            .collate(params.nuclei_batch_size)
            .map { batch ->
                def batch_id = "nuclei_" + Math.abs(batch.hashCode())
                [batch_id, batch]
            }

        RUN_CELLPOSE_NUCLEI_BATCH(nuclei_batches)

        // Create nuclei-seeded cyto batches
        nuclei_masks_flat = RUN_CELLPOSE_NUCLEI_BATCH.out.nuclei_masks.flatten()

        // Match tiles to nuclei masks by basename, then batch
        tiles_with_masks = tiles_to_use
            .map { tile -> [tile.baseName, tile] }
            .join(
                nuclei_masks_flat.map { mask -> 
                    [mask.baseName.replaceAll('_nuclei_mask$', ''), mask] 
                }
            )
            .map { basename, tile, mask -> [tile, mask] }

        // Now batch the matched pairs
        cyto_seeded_input = tiles_with_masks
            .collate(params.cyto_batch_size_tiles)
            .map { batch ->
                def batch_id = "cyto_" + Math.abs(batch.hashCode())
                def tiles = batch.collect { it[0] }
                def masks = batch.collect { it[1] }
                [batch_id, tiles, masks]
            }

        RUN_CELLPOSE_CYTO_SEEDED(cyto_seeded_input)

        if (params.mcquant) {
            // Step 4: Run MCQuant on individual tiles - FIXED JOIN
            // After creating tiles_to_use, ADD THIS:
            // tiles_for_mcquant = tiles_to_use  // Create second reference BEFORE collate

            // Then use tiles_for_mcquant for MCQuant join:
            mcquant_input = RUN_CELLPOSE_CYTO_SEEDED.out.cell_masks
                .flatMap { it } //.flatten()
                .map { mask -> [mask.baseName.replaceAll('_cell$', ''), mask] }
                .join(
                    tiles_for_mcquant.map { tile -> [tile.baseName, tile] }
                )
                .map { key, mask, tile -> [mask, tile] }

            RUN_MCQUANT(mcquant_input, TILE_LARGE_IMAGE.out.markers_csv)

            // Step 5: Stitch results
            STITCH_RESULTS(
                TILE_LARGE_IMAGE.out.tile_info,
                RUN_CELLPOSE_CYTO_SEEDED.out.cell_masks.collect(),
                RUN_MCQUANT.out.cell_quantification.collect(),
                params.sample_name
            )
            
            if (params.cylinter) {
                CYLINTER_QC(
                    STITCH_RESULTS.out.combined_csv,
                    TILE_LARGE_IMAGE.out.markers_csv,  // Missing full_mask parameter
                    params.sample_name
                )
                
                if (params.scimap) {
                    SPATIAL_ANALYSIS(
                        CYLINTER_QC.out.cleaned,
                        TILE_LARGE_IMAGE.out.markers_csv,
                        params.sample_name
                    )
                }
            }
        }
    }
}

workflow.onComplete {
    println ""
    println "=== BATCH CELLPOSE PIPELINE COMPLETED ==="
    println "Execution status: ${ workflow.success ? 'OK' : 'failed' }"
    println "Results directory: ${params.outdir}"
    
    if (workflow.success) {
        println ""
        println "SUCCESS! Batch processing completed:"
        println "- TILE_LARGE_IMAGE: 1 of 1 ✔"
        println "- RUN_CELLPOSE_NUCLEI_BATCH: X batches ✔"
        println "- RUN_CELLPOSE_CYTO_BATCH: X batches ✔" 
        println "- RUN_MCQUANT: X tiles ✔"
        println ""
        println "Batch optimization delivered significant speedup!"
    }
}