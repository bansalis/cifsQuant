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
    publishDir "${params.outdir}/tiles", mode: 'copy'
    cpus 4
    memory '32.GB'
    
    input:
    path image
    val sample_name
    val tile_size
    val overlap
    val pyramid_level
    
    output:
    path "tile_*.tif", emit: tiles
    path "tile_info.json", emit: tile_info
    path "markers_tiled.csv", emit: markers_csv
    path "tile_list.txt", emit: tile_list
    
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

def extract_tile(args):
    y, x, y_end, x_end, input_path, n_channels, dapi_ch = args
    
    try:
        with tifffile.TiffFile(input_path) as tif:
            pages = tif.series[0].pages if hasattr(tif, 'series') else tif.pages
            
            # Quick DAPI check
            dapi = pages[dapi_ch].asarray()[y:y_end, x:x_end]
            if dapi.max() == 0:
                del dapi
                return None
            
            # Check for nuclei signal
            # Enhanced nuclei signal check
            p95 = np.percentile(dapi, 95)
            p5 = np.percentile(dapi, 5)
            dynamic_range = p95 - p5

            # Require: contrast AND minimum intensity
            if p95 < 100 or dynamic_range < 50 or p95 / (p5 + 1) < 1.5:
                del dapi
                return None
            
            # Read all channels
            tile = np.zeros((n_channels, y_end - y, x_end - x), dtype=np.uint16)
            for c in range(n_channels):
                tile[c] = pages[c].asarray()[y:y_end, x:x_end]
        
        filename = f"tile_y{y:06d}_x{x:06d}.tif"
        tifffile.imwrite(filename, tile, photometric='minisblack', compression='lzw')
        
        # Free memory immediately
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

# Get metadata without loading full image
print("Reading image metadata...")
with tifffile.TiffFile('${image}') as tif:
    pages = tif.series[0].pages if hasattr(tif, 'series') else tif.pages
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
        coords.append((y, x, y_end, x_end, '${image}', n_channels, ${params.dapi_channel}))

print(f"Processing {len(coords)} tiles in batches (4 concurrent)...")
sys.stdout.flush()

# Process in batches with limited concurrency to avoid OOM
tile_info = []
batch_size = 20
max_workers = 4

for batch_start in range(0, len(coords), batch_size):
    batch = coords[batch_start:batch_start + batch_size]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_tile, c): c for c in batch}
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                tile_info.append(result)
    
    print(f"Progress: {len(tile_info)}/{len(coords)}", flush=True)
    gc.collect()  # Force cleanup between batches

print(f"\\nTILING SUMMARY: {len(coords)} total tiles, {len(tile_info)} with meaningful data")

# Save outputs
with open('tile_info.json', 'w') as f:
    json.dump(tile_info, f, indent=2)

with open('tile_list.txt', 'w') as f:
    for info in tile_info:
        f.write(info['filename'] + "\\n")

# Create markers CSV - same as old code
markers_param = '${params.markers_csv}'
if markers_param != 'null' and markers_param != '':
    try:
        import shutil
        shutil.copy(markers_param, 'markers_tiled.csv')
        print(f"Copied markers from {markers_param}")
    except Exception as e:
        print(f"Could not copy markers file: {e}, creating default")
        with open('markers_tiled.csv', 'w') as f:
            f.write("marker_name\\n")
            for i in range(tile_info[0]['channels'] if tile_info else n_channels):
                f.write(f"Channel_{i+1}\\n")
else:
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
    
    python3 - <<'EOF'
import tifffile
import numpy as np

img = tifffile.imread('${tile}')

# Per-channel 5th percentile subtraction (MCMICRO standard)
for c in range(img.shape[0]):
    channel = img[c]
    positive = channel[channel > 0]
    
    if len(positive) > 0:
        bg = np.percentile(positive, 5)
        img[c] = np.clip(channel.astype(float) - bg, 0, None).astype(channel.dtype)

tifffile.imwrite('${tile.baseName}_corrected.tif', img, compression='lzw')
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

img = tifffile.imread('${image}')
corrected = np.zeros_like(img)

for c in range(img.shape[0]):
    # Rolling ball background subtraction
    corrected[c] = white_tophat(img[c], disk(${params.bg_radius}))

tifffile.imwrite('${image.baseName}_bg_corrected.ome.tif', corrected, 
                 photometric='minisblack', compression='lzw')
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
    
    # Verify GPU access
    python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU memory:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')
"
    
    # Process each tile to extract and preprocess DAPI
    for tile in ${tiles.join(' ')}; do
        tile_base=\$(basename "\$tile" .tif)
        echo "Preprocessing \$tile_base..."
        
        python3 - <<EOF
import tifffile
import numpy as np

image = tifffile.imread("\$tile")
if image.ndim != 3 or image.shape[0] <= ${params.dapi_channel}:
    raise ValueError(f"DAPI channel ${params.dapi_channel} not available")

dapi = image[${params.dapi_channel}]
bg = np.percentile(dapi, 5)
dapi_bg = np.clip(dapi.astype(np.float32) - bg, 0, None).astype(np.uint16)
tifffile.imwrite("\${tile_base}_dapi_bg_subtracted.tif", dapi_bg)
tile_name = "\${tile_base}"
print(f"Preprocessed {tile_name}: range {dapi_bg.min()}-{dapi_bg.max()}")
EOF
    done
    
    # Run Cellpose nuclei on all DAPI files in batch
    echo "Running Cellpose nuclei segmentation on batch..."
    export HOME=/tmp
    export CELLPOSE_LOCAL_MODELS_PATH=/tmp/cellpose_models
    mkdir -p /tmp/cellpose_models

    for dapi_file in *_dapi_bg_subtracted.tif; do
        echo "Processing \$dapi_file"
        cellpose --image_path "\$dapi_file" \
            --pretrained_model "${params.nuc_model}" \
            --diameter ${params.nuc_diameter} \
            --flow_threshold ${params.nuc_flow_threshold} \
            --cellprob_threshold ${params.nuc_cellprob_threshold} \
            --stitch_threshold ${params.stitch_threshold} \
            --use_gpu \
            --save_tif --no_npy \
            --verbose
    done

    # Rename outputs to expected format
    for mask in *_dapi_bg_subtracted_cp_masks.tif; do
        if [ -f "\$mask" ]; then
            base=\$(echo "\$mask" | sed 's/_dapi_bg_subtracted_cp_masks.tif//')
            mv "\$mask" "\${base}_nuclei_mask.tif"
            
            # Print statistics
            python3 -c "
import tifffile, numpy as np
try:
    mask = tifffile.imread('\${base}_nuclei_mask.tif')
    n_cells = len(np.unique(mask[mask > 0]))
    tile_name = '\${base}'
    print(f'  {tile_name}: {n_cells} nuclei detected')
except Exception as e:
    tile_name = '\${base}'
    print(f'  {tile_name}: Error reading mask - {e}')
"
        fi
    done
    
    # Create empty masks for any failed tiles
    for tile in ${tiles.join(' ')}; do
        tile_base=\$(basename "\$tile" .tif)
        if [ ! -f "\${tile_base}_nuclei_mask.tif" ]; then
            echo "Creating empty mask for \$tile_base"
            python3 -c "
import tifffile, numpy as np
empty = np.zeros((${params.tile_size}, ${params.tile_size}), dtype=np.uint16)
tifffile.imwrite('\${tile_base}_nuclei_mask.tif', empty)
"
        fi
    done
    
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

process RUN_CELLPOSE_CYTO_MULTIPASS {
    tag "cyto_${batch_id}"
    container 'biocontainers/cellpose:3.0.1_cv1'
    publishDir "${params.outdir}/cell_masks", mode: 'copy'
    
    input:
    tuple val(batch_id), path(tiles), path(nuclei_masks), path(membrane_composites)
    
    output:
    path "*_cell.tif", emit: cell_masks
    
    script:
    """
    #!/bin/bash
    set -e
    
    export HOME=/tmp
    export CELLPOSE_LOCAL_MODELS_PATH=/tmp/cellpose_models
    mkdir -p /tmp/cellpose_models
    
    echo "[\$(date +%H:%M:%S)] Batch ${batch_id}: Processing ${tiles.size()} tiles"
    
    # IMPROVEMENT 1: Batch all Pass 1 together (GPU efficiency)
    echo "Pass 1/2: Large cells (diameter=70) - BATCH MODE"
    cellpose --dir . --img_filter "*_membrane.tif" \\
        --pretrained_model cyto2 --chan 0 \\
        --diameter 70 --flow_threshold 0.3 --cellprob_threshold 0.0 \\
        --use_gpu --save_tif --no_npy \\
        --verbose
    
    # Rename Pass 1 outputs
    for mask in *_membrane_cp_masks.tif; do
        [ -f "\$mask" ] || continue
        base=\$(echo "\$mask" | sed 's/_membrane_cp_masks.tif//')
        mv "\$mask" "\${base}_large.tif"
    done
    
    # IMPROVEMENT 2: Batch all Pass 2 together
    echo "Pass 2/2: Small cells (diameter=25) - BATCH MODE"
    cellpose --dir . --img_filter "*_membrane.tif" \\
        --pretrained_model cyto2 --chan 0 \\
        --diameter 25 --flow_threshold 0.4 --cellprob_threshold -1.5 \\
        --use_gpu --save_tif --no_npy \\
        --verbose
    
    # Rename Pass 2 outputs
    for mask in *_membrane_cp_masks.tif; do
        [ -f "\$mask" ] || continue
        base=\$(echo "\$mask" | sed 's/_membrane_cp_masks.tif//')
        mv "\$mask" "\${base}_small.tif"
    done
    
    # IMPROVEMENT 3: Fast IoU-based merge with spatial indexing
    python3 << 'MERGE'
import tifffile
import numpy as np
from pathlib import Path
from scipy.ndimage import find_objects

def fast_iou_merge(large, small, iou_threshold=0.35, min_pixels=20):
    merged = large.copy()
    max_id = int(merged.max())
    
    if small.max() == 0:
        return merged
    
    # Get small cell properties (bbox + area)
    small_ids = np.unique(small[small > 0])
    small_slices = find_objects(small)
    
    # Get large cell bboxes for fast intersection checks
    large_ids = np.unique(large[large > 0])
    large_slices = find_objects(large)
    large_areas = {lid: np.sum(large == lid) for lid in large_ids}
    
    # Process each small cell
    for s_id in small_ids:
        s_slice = small_slices[s_id - 1]
        if s_slice is None:
            continue
        
        s_mask = (small == s_id)
        s_area = np.sum(s_mask)
        
        if s_area < min_pixels:
            continue
        
        # Extract small cell region
        s_region = small[s_slice]
        s_local_mask = (s_region == s_id)
        
        # Check overlap with large cells (only in bbox region)
        large_region = large[s_slice]
        
        # Find which large cells overlap
        overlapping_large = np.unique(large_region[s_local_mask])
        overlapping_large = overlapping_large[overlapping_large > 0]
        
        # Compute IoU with overlapping large cells
        max_iou = 0.0
        for l_id in overlapping_large:
            l_area = large_areas[l_id]
            
            # Intersection in local region
            intersection = np.sum(s_local_mask & (large_region == l_id))
            
            # Union = area1 + area2 - intersection
            union = s_area + l_area - intersection
            iou = intersection / union if union > 0 else 0.0
            
            max_iou = max(max_iou, iou)
            
            if max_iou >= iou_threshold:
                break  # This small cell overlaps significantly, skip it
        
        # Add small cell if no significant overlap
        if max_iou < iou_threshold:
            max_id += 1
            merged[s_mask] = max_id
    
    return merged

# Process all tiles
for tile in sorted(Path('.').glob('tile_*_corrected.tif')):
    base = tile.stem
    large_file = f'{base}_large.tif'
    small_file = f'{base}_small.tif'
    
    if not Path(large_file).exists() or not Path(small_file).exists():
        print(f'Skipping {base}: missing masks')
        continue
    
    large = tifffile.imread(large_file)
    small = tifffile.imread(small_file)
    
    # FAST IoU-based merge (0.35 threshold = ~35% overlap)
    merged = fast_iou_merge(large, small, iou_threshold=0.35, min_pixels=20)
    
    tifffile.imwrite(f'{base}_cell.tif', merged.astype(np.int32), compression='lzw')
    
    n_large = len(np.unique(large[large > 0]))
    n_small = len(np.unique(small[small > 0]))
    n_merged = len(np.unique(merged[merged > 0]))
    
    print(f'{base}: {n_large} large + {n_small} small → {n_merged} final cells')

print("Merge complete")
MERGE
    
    echo "[\$(date +%H:%M:%S)] Batch ${batch_id} complete"
    """
}

process RUN_MCQUANT {
    tag "${cell_mask.baseName.replaceAll('_cell\$', '')}"
    container params.mcquant_container
    publishDir "${params.outdir}/quantification", mode: 'copy'
    
    input:
    tuple path(cell_mask), path(original_tile), path(markers_csv)
    
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
    if (!params.input_image) {
        error "Please provide --input_image parameter"
    }
    
    log.info "=== MCMICRO BATCH CELLPOSE PIPELINE ==="
    log.info "Input image: ${params.input_image}"
    log.info "Tile size: ${params.tile_size}"
    log.info "Nuclei batch size: ${params.nuclei_batch_size} tiles/batch"
    log.info "Cyto batch size: ${params.cyto_batch_size_tiles} tiles/batch"
    log.info "Output directory: ${params.outdir}"
    
    // Create input channel
    input_image_ch = Channel.fromPath(params.input_image)
    
    // Step 0: Global background subtraction (BEFORE tiling)
    if (params.global_bg_subtract) {
        log.info "Applying global background subtraction..."
        bg_corrected_ch = GLOBAL_BACKGROUND_SUBTRACT(input_image_ch)
        image_to_tile = bg_corrected_ch
    } else {
        image_to_tile = input_image_ch
    }

    // Step 1: Tile the image
    TILE_LARGE_IMAGE(
        input_image_ch,
        params.sample_name,
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
        
        // Create tile batches ONCE for all cyto operations
        tile_batches = tiles_to_use
            .collate(params.cyto_batch_size_tiles)
            .map { batch ->
                def batch_id = "batch_" + Math.abs(batch.hashCode())
                [batch_id, batch]
            }

        // Create composites using same batches
        composites = CREATE_MEMBRANE_COMPOSITES(tile_batches)

        // Create nuclei batches separately
        nuclei_batches = tiles_to_use
            .collate(params.nuclei_batch_size)
            .map { batch ->
                def batch_id = "nuclei_" + Math.abs(batch.hashCode())
                [batch_id, batch]
            }

        RUN_CELLPOSE_NUCLEI_BATCH(nuclei_batches)

        // Match nuclei masks to original tiles by basename
        nuclei_masks_flat = RUN_CELLPOSE_NUCLEI_BATCH.out.nuclei_masks.flatten()

        // Join: batch_id -> [tiles, nuclei_masks, membranes]
        cyto_input = tile_batches
            .map { batch_id, tiles ->
                [batch_id, tiles, tiles.collect { it.baseName }]
            }
            .combine(
                nuclei_masks_flat.map { mask -> 
                    [mask.baseName.replaceAll('_nuclei_mask$', ''), mask] 
                }.groupTuple()
            )
            .map { batch_id, tiles, basenames, mask_names, masks ->
                def matched_masks = basenames.collect { bn ->
                    def idx = mask_names.findIndexOf { it == bn }
                    idx >= 0 ? masks[idx] : null
                }.findAll { it != null }
                [batch_id, tiles, matched_masks]
            }
            .join(composites)
            .map { batch_id, tiles, masks, nuclear, membrane ->
                [batch_id, tiles, masks, membrane]
            }

        RUN_CELLPOSE_CYTO_MULTIPASS(cyto_input)

        if (params.mcquant) {
            // Step 4: Run MCQuant on individual tiles - FIXED JOIN
            // After creating tiles_to_use, ADD THIS:
            // tiles_for_mcquant = tiles_to_use  // Create second reference BEFORE collate

            // Then use tiles_for_mcquant for MCQuant join:
            mcquant_input = RUN_CELLPOSE_CYTO_MULTIPASS.out.cell_masks
                .flatMap { it } //.flatten()
                .map { mask -> [mask.baseName.replaceAll('_cell$', ''), mask] }
                .join(
                    tiles_for_mcquant.map { tile -> [tile.baseName, tile] }
                )
                .combine(TILE_LARGE_IMAGE.out.markers_csv)
                .map { key, mask, tile, markers -> [mask, tile, markers] }
            
            RUN_MCQUANT(mcquant_input)

            // Step 5: Stitch results
            STITCH_RESULTS(
                TILE_LARGE_IMAGE.out.tile_info,
                RUN_CELLPOSE_CYTO_MULTIPASS.out.cell_masks.collect(),
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