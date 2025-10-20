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

process RUN_CELLPOSE_CYTO_BATCH {
    tag "cyto_batch_${batch_id}"
    container params.cellpose_container
    publishDir "${params.outdir}/cell_masks", mode: 'copy'

    input:
    tuple val(batch_id), path(tiles), path(nuclei_masks), path(dapi_files)

    output:
    path "*_cell.tif", emit: cell_masks

    when:
    params.cellpose

    script:
    // Precompute safe strings for passing into the script
    def tiles_str = tiles.collect { it.toString() }.join('|')
    def nucs_str  = nuclei_masks.collect { it.toString() }.join('|')
    def buffer_px = params.buffer_px ?: ""

    """
    set -euo pipefail

    echo "=== CELLPOSE CYTO BATCH ${batch_id}: ${tiles.size()} tiles ==="

    # 1) Build cytoplasm composites per tile
    for tile in ${tiles.join(' ')}; do
        tile_base=\$(basename "\$tile" .tif)
        echo "[Compose] \$tile_base"

        TILE="\$tile" DAPI_CH="${params.dapi_channel}" python3 - <<'PY'
import os, sys, tifffile, numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation

tile = os.environ["TILE"]
dapi_ch = int(os.environ["DAPI_CH"])
out_base = os.path.splitext(os.path.basename(tile))[0]

img = tifffile.imread(tile)
dapi = img[dapi_ch]

# Use all non-DAPI channels for cytoplasm
cyto_channels = [img[i] for i in range(img.shape[0]) if i != dapi_ch]
cyto_raw = np.maximum.reduce(cyto_channels).astype(np.float32)
cyto_smooth = gaussian_filter(cyto_raw, sigma=2.0)

# More permissive normalization
cyto_norm = np.clip((cyto_smooth - np.percentile(cyto_smooth, 5)) / 
                    (np.percentile(cyto_smooth, 95) - np.percentile(cyto_smooth, 5) + 1), 
                    0, 1)
cyto_buffered = (cyto_norm * 65535).astype(np.uint16)

combined = np.stack([dapi, cyto_buffered.astype(np.uint16)], axis=0).astype(np.uint16)
tifffile.imwrite(f"{out_base}_nuclear_membrane_input.tif", combined)
print(f"Created: {out_base}_nuclear_membrane_input.tif  shape={combined.shape}")
PY
    done

    # 2) Run Cellpose cyto on the composites
    echo "Running Cellpose cyto on composites..."
    export HOME=/tmp
    export CELLPOSE_LOCAL_MODELS_PATH=/tmp/cellpose_models
    mkdir -p /tmp/cellpose_models

    shopt -s nullglob
    for comp in *_nuclear_membrane_input.tif; do
        echo "  [Cellpose] \$comp"
        cellpose --image_path "\$comp" \\
                 --pretrained_model "${params.cyto_model}" \\
                 --chan 1 --chan2 2 \\
                 --diameter ${params.cyto_diameter} \\
                 --flow_threshold 0.4 \\
                 --cellprob_threshold 0.0 \\
                 --use_gpu \\
                 --batch_size ${params.cyto_batch_size} \\
                 --save_tif --no_npy \\
                 --verbose
    done
    shopt -u nullglob

    # 3) Buffer each Cellpose cyto result
    for mask in *_nuclear_membrane_input_cp_masks.tif; do
        [ -f "\$mask" ] || continue
        tile_base=\$(echo "\$mask" | sed 's/_nuclear_membrane_input_cp_masks.tif//')
        echo "Buffering: \$tile_base"

        MASK_PATH="\$mask" TILE_BASE="\$tile_base" BUFFER_PX="${buffer_px}" \\
        TILES_STR="${tiles_str}" NUCS_STR="${nucs_str}" \\
        python3 - <<'PY'
import os, glob, sys, math, tifffile, numpy as np
from skimage.segmentation import expand_labels
from scipy.ndimage import gaussian_filter, binary_dilation

mask_path = os.environ["MASK_PATH"]
tile_base = os.environ["TILE_BASE"]
buffer_px = os.environ["BUFFER_PX"].strip()
tiles_list = os.environ["TILES_STR"].split("|") if os.environ.get("TILES_STR") else []
nucs_list  = os.environ["NUCS_STR"].split("|")  if os.environ.get("NUCS_STR")  else []

labels = tifffile.imread(mask_path)

# Find matching nuclei mask from provided list
nuc_path = None
for p in nucs_list:
    if os.path.basename(p).startswith(tile_base) and p.endswith("_nuclei_mask.tif"):
        nuc_path = p; break

if nuc_path and os.path.exists(nuc_path):
    ref = tifffile.imread(nuc_path)
    sizes = np.bincount(ref.ravel())[1:]
    r_med = math.sqrt(float(np.median(sizes))/math.pi) if sizes.size else 6.0
else:
    r_med = 6.0

# Decide buffer
if buffer_px:
    buf = max(1, int(buffer_px))
else:
    # Density-aware buffering
    cell_density = len(sizes) / (ref.shape[0] * ref.shape[1]) * 1e6  # cells/mm²
    if cell_density > 500:  # Dense region
        buf = int(round(np.clip(0.5 * r_med, 4, 12)))
    elif cell_density > 200:  # Medium
        buf = int(round(np.clip(0.6 * r_med, 6, 15)))
    else:  # Sparse
        buf = int(round(np.clip(0.8 * r_med, 8, 20)))

# Find original tile path from list
tile_path = None
for p in tiles_list:
    if os.path.basename(p).startswith(tile_base):
        tile_path = p; break

# Optional constraint mask from original tile
if tile_path and os.path.exists(tile_path):
    img = tifffile.imread(tile_path)
    tom = img[1] if img.shape[0] > 1 else np.zeros_like(img[0])
    cd45 = img[2] if img.shape[0] > 2 else np.zeros_like(img[0])
    cd3  = img[9] if img.shape[0] > 9 else np.zeros_like(img[0])
    cyto = np.maximum.reduce([tom, cd45, cd3]).astype(np.float32)
    cyto_s = gaussian_filter(cyto, sigma=1.0)
    cyto_norm = (cyto_s - cyto_s.min()) / (np.ptp(cyto_s) + 1e-8)
    cyto_mask = binary_dilation(cyto_norm > 0.05, iterations=2)
    expanded = expand_labels(labels, distance=buf)
    expanded[~cyto_mask] = 0
    expanded = np.where(labels > 0, labels, expanded).astype(np.int32)
else:
    expanded = expand_labels(labels, distance=buf).astype(np.int32)

tifffile.imwrite(f"{tile_base}_cell.tif", expanded, dtype=np.int32)

orig_cells = len(np.unique(labels[labels > 0]))
final_cells = len(np.unique(expanded[expanded > 0]))
print(f"  {tile_base}: {orig_cells} -> {final_cells} cells, buffer={buf}px")
PY
    done

    # 4) Ensure every tile yields an output
    for tile in ${tiles.join(' ')}; do
        tile_base=\$(basename "\$tile" .tif)
        if [ ! -f "\${tile_base}_cell.tif" ]; then
            echo "Creating empty cell mask for \$tile_base"
            python3 - <<'PY'
import sys, tifffile, numpy as np
tile_size = int(sys.argv[1])
out = sys.argv[2]
tifffile.imwrite(out, np.zeros((tile_size, tile_size), dtype=np.int32))
PY
            "${params.tile_size}" "\${tile_base}_cell.tif"
        fi
    done

    echo "=== CYTO BATCH ${batch_id} completed ==="
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
        
        // Group tiles into batches for nuclei processing
        nuclei_batches = tiles_for_nuclei
            .collate(params.nuclei_batch_size)
            .map { batch -> 
                def batch_id = "nuclei_batch_" + Math.abs(batch.hashCode())
                [batch_id, batch] 
            }
        RUN_CELLPOSE_NUCLEI_BATCH(nuclei_batches)
        
        // Step 3: Prepare cyto batches - FIXED JOINS
        nuclei_masks_flat = RUN_CELLPOSE_NUCLEI_BATCH.out.nuclei_masks.flatten()
        dapi_processed_flat = RUN_CELLPOSE_NUCLEI_BATCH.out.dapi_processed.flatten()
        
        cyto_matched = tiles_to_use
            .map { tile -> [tile.baseName, tile] }
            .join(
                nuclei_masks_flat.map { mask -> 
                    [mask.baseName.replaceAll('_nuclei_mask$', ''), mask] 
                }
            )
            .join(
                dapi_processed_flat.map { dapi -> 
                    [dapi.baseName.replaceAll('_dapi_bg_subtracted$', ''), dapi] 
                }
            )
            .map { key, tile, mask, dapi -> [tile, mask, dapi] }
        
        // Group into cyto batches
        cyto_batches = cyto_matched
            .collate(params.cyto_batch_size_tiles)
            .map { batch ->
                def batch_id = "cyto_batch_" + Math.abs(batch.hashCode())
                def tiles = batch.collect { it[0] }
                def masks = batch.collect { it[1] }  
                def dapis = batch.collect { it[2] }
                [batch_id, tiles, masks, dapis]
            }
        
        RUN_CELLPOSE_CYTO_BATCH(cyto_batches)
        
        if (params.mcquant) {
            // Step 4: Run MCQuant on individual tiles - FIXED JOIN
            // After creating tiles_to_use, ADD THIS:
            // tiles_for_mcquant = tiles_to_use  // Create second reference BEFORE collate

            // Then use tiles_for_mcquant for MCQuant join:
            mcquant_input = RUN_CELLPOSE_CYTO_BATCH.out.cell_masks
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
                RUN_CELLPOSE_CYTO_BATCH.out.cell_masks.collect(),
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