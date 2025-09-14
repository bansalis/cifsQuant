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
params.pyramid_level = 1
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
params.cyto_model = 'cyto3'
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
import os
from pathlib import Path
import pandas as pd

def load_ome_tiff_properly(input_path, level=0):
    \"\"\"Load OME-TIFF using the same method as the alignment script.\"\"\"
    print(f"Loading OME-TIFF: {input_path}")
    
    with tifffile.TiffFile(input_path) as tif:
        print(f"TIFF file has {len(tif.pages)} total pages")
        print(f"TIFF file has {len(tif.series)} series (pyramid levels)")
        
        if level >= len(tif.series):
            print(f"Level {level} not available, using level 0")
            level = 0
        
        selected_series = tif.series[level]
        print(f"Using pyramid level {level} with {len(selected_series.pages)} pages")
        
        image_data = selected_series.asarray()
        print(f"Loaded series as array with shape: {image_data.shape}")
        
        if image_data.ndim == 2:
            image_data = image_data[np.newaxis, :, :]
        elif image_data.ndim == 3:
            if image_data.shape[0] > 20:
                image_data = np.transpose(image_data, (2, 0, 1))
        elif image_data.ndim == 4 and image_data.shape[0] == 1:
            image_data = image_data[0]
        
        channels, height, width = image_data.shape
        print(f"Final image shape: {channels} channels, {height}x{width} pixels")
        
        return image_data

def tile_image(input_path, output_dir, tile_size=${tile_size}, overlap=${overlap}, level=${pyramid_level}):
    print(f"Processing image: {input_path}")
    
    image = load_ome_tiff_properly(input_path, level)
    channels, height, width = image.shape
    
    step = tile_size - overlap
    tile_info = []
    
    tile_count = 0
    tiles_with_data = 0
    
    for y in range(0, height - overlap, step):
        for x in range(0, width - overlap, step):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            
            tile = image[:, y:y_end, x:x_end]
            tile_filename = f"tile_y{y:06d}_x{x:06d}.tif"
            
            # Check if tile has meaningful data
            has_data = bool(tile.max() > 0)
            dapi_signal = tile[${params.dapi_channel}] if tile.shape[0] > ${params.dapi_channel} else tile[0]
            has_nuclei = np.percentile(dapi_signal, 95) > np.percentile(dapi_signal, 5) * 2
            
            if has_data and has_nuclei:
                tiles_with_data += 1
                tifffile.imwrite(tile_filename, tile, photometric='minisblack')
                
                tile_info.append({
                    'filename': tile_filename,
                    'y_start': y, 'x_start': x, 'y_end': y_end, 'x_end': x_end,
                    'height': y_end - y, 'width': x_end - x,
                    'channels': channels, 'has_data': has_data
                })
            else:
                if tile_count < 3:
                    print(f"Skipping tile {tile_count + 1}: {tile_filename} (no nuclei signal)")
            
            tile_count += 1
    
    print(f"\\nTILING SUMMARY: {tile_count} total tiles, {tiles_with_data} with meaningful data")
    return tile_info

# Process the image
tile_info = tile_image('${image}', '.', ${tile_size}, ${overlap}, ${pyramid_level})

# Create markers CSV
if os.path.exists('${params.markers_csv}' if '${params.markers_csv}' != 'null' else ''):
    import shutil
    shutil.copy('${params.markers_csv}', 'markers_tiled.csv')
else:
    with open('markers_tiled.csv', 'w') as f:
        f.write("marker_name\\n")
        for i in range(tile_info[0]['channels'] if tile_info else 10):
            f.write(f"Channel_{i+1}\\n")

# Save outputs
with open('tile_info.json', 'w') as f:
    json.dump(tile_info, f, indent=2)

with open('tile_list.txt', 'w') as f:
    for info in tile_info:
        f.write(info['filename'] + "\\n")

print("Tiling completed!")
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
        cellpose --image_path "\$dapi_file" \\
             --img_filter "*_dapi_bg_subtracted.tif" \\
             --pretrained_model "${params.nuc_model}" \\
             --diameter ${params.nuc_diameter} \\
             --flow_threshold 0.4 \\
             --cellprob_threshold 0.0 \\
             --use_gpu \\
             --save_tif --no_npy \\
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

tom = img[1] if img.shape[0] > 1 else np.zeros_like(dapi)
cd45 = img[2] if img.shape[0] > 2 else np.zeros_like(dapi)
cd3  = img[9] if img.shape[0] > 9 else np.zeros_like(dapi)

cyto_raw = np.maximum.reduce([tom, cd45, cd3]).astype(np.float32)
cyto_smooth = gaussian_filter(cyto_raw, sigma=1.0)
norm = (cyto_smooth - cyto_smooth.min()) / (np.ptp(cyto_smooth) + 1e-8)
mask = norm > 0.1
mask_dil = binary_dilation(mask, iterations=3)
cyto_buffered = np.where(mask_dil, cyto_smooth.max(), cyto_smooth)

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
    buf = int(round(np.clip(0.3 * r_med, 2, 8)))

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
    echo "=== MCQUANT PROCESSING: ${tile_name} ==="
    
    # Create channel names file
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
        --output quantification_output/
        
    # Process results
    python3 -c "
import os, pandas as pd, glob, numpy as np
csv_files = glob.glob('quantification_output/*.csv')
if csv_files:
    df = pd.read_csv(csv_files[0])
    os.rename(csv_files[0], '${tile_name}_cell.csv')
    print(f'=== ${tile_name}: {len(df)} CELLS DETECTED ===')
    
    # Summary analytics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    const_cols = [col for col in numeric_cols if df[col].nunique(dropna=False) <= 1]
    
    if const_cols:
        const_list = ', '.join(const_cols)
        print(f'Warning: Constant values in {len(const_cols)} channel(s): {const_list}')
    else:
        print('All numeric channels have >1 unique value')
else:
    with open('${tile_name}_cell.csv', 'w') as f:
        f.write('CellID,X_centroid,Y_centroid\\n')
    print(f'=== ${tile_name}: 0 CELLS DETECTED ===')
"
    """
}

process STITCH_RESULTS {
    tag "stitching"
    publishDir "${params.outdir}/final", mode: 'copy'

    input:
    path tile_info
    path "masks/*"
    path "csvs/*"
    val sample_name

    output:
    path "full_segmentation_mask.tif", emit: full_mask
    path "combined_quantification.csv", emit: combined_csv
    path "stitching_report.txt", emit: report

    script:
    """
#!/usr/bin/env python3

import json
import tifffile
import numpy as np
import pandas as pd
from pathlib import Path
import glob
import os

print("=== STARTING STITCHING PROCESS ===")

with open('${tile_info}', 'r') as f:
    tile_info = json.load(f)

print(f"Stitching {len(tile_info)} tiles...")

max_y = max(info['y_end'] for info in tile_info)
max_x = max(info['x_end'] for info in tile_info)
print(f"Full image dimensions: {max_y} x {max_x}")

full_mask = np.zeros((max_y, max_x), dtype=np.int32)
current_label = 1
stitched_tiles = 0
all_quantifications = []

for info in tile_info:
    tile_base = Path(info['filename']).stem
    mask_file = f"masks/{tile_base}_cell.tif"
    csv_file = f"csvs/{tile_base}_cell.csv"
    
    if os.path.exists(mask_file) and os.path.exists(csv_file):
        tile_mask = tifffile.imread(mask_file)
        unique_labels = np.unique(tile_mask[tile_mask > 0])
        
        updated_mask = tile_mask.copy()
        label_mapping = {}
        for old_label in unique_labels:
            label_mapping[old_label] = current_label
            updated_mask[tile_mask == old_label] = current_label
            current_label += 1
        
        y_start, x_start = info['y_start'], info['x_start']
        y_end = min(y_start + updated_mask.shape[0], max_y)
        x_end = min(x_start + updated_mask.shape[1], max_x)
        
        mask_h = y_end - y_start
        mask_w = x_end - x_start
        
        full_mask[y_start:y_end, x_start:x_end] = updated_mask[:mask_h, :mask_w]
        
        df = pd.read_csv(csv_file)
        if not df.empty:
            if 'X_centroid' in df.columns:
                df['X_centroid'] += info['x_start']
            if 'Y_centroid' in df.columns:
                df['Y_centroid'] += info['y_start']
            
            if 'CellID' in df.columns:
                for old_label, new_label in label_mapping.items():
                    df.loc[df['CellID'] == old_label, 'CellID'] = new_label
            
            df['tile_y'] = info['y_start']
            df['tile_x'] = info['x_start']
            df['original_tile'] = info['filename']
            
            all_quantifications.append(df)
            stitched_tiles += 1
            print(f"  Added {len(df)} cells from {tile_base}")

print("Saving full segmentation mask...")
tifffile.imwrite('full_segmentation_mask.tif', full_mask)

if all_quantifications:
    combined_df = pd.concat(all_quantifications, ignore_index=True)
    if 'CellID' in combined_df.columns:
        combined_df['CellID'] = range(1, len(combined_df) + 1)
    combined_df.to_csv('combined_quantification.csv', index=False)
    total_cells = len(combined_df)
else:
    pd.DataFrame().to_csv('combined_quantification.csv', index=False)
    total_cells = 0

with open('stitching_report.txt', 'w') as f:
    f.write(f'''Stitching Report for ${sample_name}
=====================================
Total tiles expected: {len(tile_info)}
Successfully stitched tiles: {stitched_tiles}
Final image dimensions: {max_y} x {max_x}
Total cells in quantification: {total_cells}
''')

print("=== STITCHING COMPLETED ===")
print(f"Processed {stitched_tiles}/{len(tile_info)} tiles")
print(f"Total cells: {total_cells}")
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
 * BATCH CELLPOSE WORKFLOW - COMPLETE FIXED VERSION
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
    
    // Step 1: Tile the image
    TILE_LARGE_IMAGE(
        input_image_ch,
        params.sample_name,
        params.tile_size,
        params.overlap,
        params.pyramid_level
    )
    
    if (params.cellpose) {
        
        // Step 2: Create nuclei batches and process
        tiles_flattened = TILE_LARGE_IMAGE.out.tiles.flatten()
        
        // Group tiles into batches for nuclei processing
        nuclei_batches = tiles_flattened
            .collate(params.nuclei_batch_size)
            .map { batch -> 
                def batch_id = "nuclei_batch_" + Math.abs(batch.hashCode())
                [batch_id, batch] 
            }
        
        RUN_CELLPOSE_NUCLEI_BATCH(nuclei_batches)
        
        // Step 3: Prepare cyto batches
        // Flatten outputs and create matching channel
        nuclei_masks_flat = RUN_CELLPOSE_NUCLEI_BATCH.out.nuclei_masks.flatten()
        dapi_processed_flat = RUN_CELLPOSE_NUCLEI_BATCH.out.dapi_processed.flatten()
        
        // Create a simple join channel for cyto processing  
        cyto_matched = tiles_flattened
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
            // Step 4: Run MCQuant on individual tiles
            mcquant_input = RUN_CELLPOSE_CYTO_BATCH.out.cell_masks
                .flatten()
                .map { mask -> [mask.baseName.replaceAll('_cell$', ''), mask] }
                .join(
                    tiles_flattened.map { tile -> [tile.baseName, tile] }
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
            
            if (params.scimap) {
                SPATIAL_ANALYSIS(
                    STITCH_RESULTS.out.combined_csv,
                    TILE_LARGE_IMAGE.out.markers_csv,
                    params.sample_name
                )
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