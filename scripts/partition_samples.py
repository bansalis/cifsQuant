#!/usr/bin/env python3
"""
Sample Partitioning Preprocessor for MCMICRO Pipeline

This script allows you to partition multi-condition samples on the same slide
before running the main mcmicro pipeline. It creates coordinate-based
partitions that are saved as separate samples in rawdata/.

Usage:
    # Single sample
    python scripts/partition_samples.py --sample_names JL216

    # Multiple samples (batch mode)
    python scripts/partition_samples.py --sample_names JL216 JL217 JL218

    # Custom source directory
    python scripts/partition_samples.py --sample_names JL216 --source_dir custom_rawdata
"""

import argparse
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Dict, Tuple
import shutil
import re
import gc
import os

def find_dapi_channel(sample_dir: Path) -> Path:
    """Find the DAPI channel file in the sample directory."""
    # Look for files with DAPI in the name
    dapi_files = list(sample_dir.glob('*DAPI*.ome.tif'))
    if dapi_files:
        return dapi_files[0]

    # Fallback: look for common DAPI patterns
    patterns = ['*DAPI*', '*dapi*', '*Hoechst*', '*hoechst*']
    for pattern in patterns:
        files = list(sample_dir.glob(f'{pattern}.ome.tif'))
        if files:
            return files[0]

    raise FileNotFoundError(f"No DAPI channel found in {sample_dir}")

def create_coordinate_grid(image_data: np.ndarray, output_path: Path, sample_name: str):
    """Create a visualization of the DAPI image with coordinate grid overlay."""
    print(f"\n📐 Creating coordinate grid for {sample_name}")
    print(f"   Image dimensions: {image_data.shape[1]} x {image_data.shape[0]} pixels")

    # Downsample aggressively to prevent memory issues
    max_display_size = 800  # Reduced from 2000 to save memory
    height, width = image_data.shape[:2]

    # Always downsample for safety
    scale = min(max_display_size / max(height, width), 1.0)
    display_height = int(height * scale)
    display_width = int(width * scale)

    if scale < 1.0:
        from skimage.transform import resize
        display_image = resize(image_data, (display_height, display_width),
                             preserve_range=True, anti_aliasing=True)
        print(f"   Display size: {display_width} x {display_height} pixels (scaled {scale:.2%} for memory)")
    else:
        display_image = image_data.copy()

    # Delete original to free memory immediately
    del image_data
    gc.collect()

    # Normalize for display
    p2, p98 = np.percentile(display_image, (2, 98))
    display_image = np.clip((display_image - p2) / (p98 - p2), 0, 1)

    # Convert to uint8 to save memory
    display_image = (display_image * 255).astype(np.uint8)

    # Create smaller figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(display_image, cmap='gray', interpolation='nearest')

    # Add coordinate grid (every 2000 pixels)
    grid_spacing = 2000

    # Vertical lines
    for x in range(0, width, grid_spacing):
        x_display = x * scale
        ax.axvline(x=x_display, color='cyan', alpha=0.4, linewidth=1, linestyle='--')
        ax.text(x_display, 50, f'x={x}', color='cyan', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Horizontal lines
    for y in range(0, height, grid_spacing):
        y_display = y * scale
        ax.axhline(y=y_display, color='yellow', alpha=0.4, linewidth=1, linestyle='--')
        ax.text(50, y_display, f'y={y}', color='yellow', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Add corner coordinates
    corners = [
        (0, 0, 'topleft'),
        (width, 0, 'topright'),
        (0, height, 'bottomleft'),
        (width, height, 'bottomright')
    ]

    for x, y, label in corners:
        x_display = x * scale
        y_display = y * scale
        ax.plot(x_display, y_display, 'r*', markersize=15, markeredgewidth=2,
               markeredgecolor='white')
        ax.text(x_display, y_display - 100, f'({x}, {y})', color='red',
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    ax.set_title(f'{sample_name} - DAPI Channel with Coordinate Grid\n'
                f'Original size: {width} x {height} pixels', fontsize=14, fontweight='bold')
    ax.set_xlabel('X coordinate (pixels)', fontsize=12)
    ax.set_ylabel('Y coordinate (pixels)', fontsize=12)

    # Save high-res version
    plt.tight_layout()
    plt.savefig(output_path, dpi=75, bbox_inches='tight')  # Reduced DPI further
    print(f"   ✓ Saved coordinate grid to: {output_path}")
    plt.close()

    # Clean up memory
    del display_image, fig, ax
    gc.collect()

    return width, height

def get_partition_info(sample_name: str, image_width: int, image_height: int) -> List[Dict]:
    """Interactively get partition information from user."""
    print(f"\n🔪 Partition Setup for {sample_name}")
    print(f"   Image dimensions: {image_width} x {image_height} pixels")
    print(f"   Refer to the coordinate grid PNG to select partition boundaries.")

    while True:
        try:
            num_partitions = int(input(f"\nHow many partitions do you want to create? "))
            if num_partitions < 2:
                print("   ⚠ Please enter at least 2 partitions")
                continue
            break
        except ValueError:
            print("   ⚠ Please enter a valid number")

    partitions = []
    suffixes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    for i in range(num_partitions):
        suffix = suffixes[i] if i < len(suffixes) else str(i + 1)
        print(f"\n--- Partition {i + 1}/{num_partitions} (will be saved as {sample_name}{suffix}) ---")

        while True:
            try:
                print("Enter the bounding box coordinates for this partition:")
                x_min = int(input(f"  X minimum (0 to {image_width}): "))
                x_max = int(input(f"  X maximum ({x_min} to {image_width}): "))
                y_min = int(input(f"  Y minimum (0 to {image_height}): "))
                y_max = int(input(f"  Y maximum ({y_min} to {image_height}): "))

                # Validate bounds
                if not (0 <= x_min < x_max <= image_width):
                    print(f"   ⚠ Invalid X bounds. Must have: 0 <= x_min < x_max <= {image_width}")
                    continue
                if not (0 <= y_min < y_max <= image_height):
                    print(f"   ⚠ Invalid Y bounds. Must have: 0 <= y_min < y_max <= {image_height}")
                    continue

                # Calculate dimensions
                width = x_max - x_min
                height = y_max - y_min
                print(f"   ✓ Partition {suffix}: {width} x {height} pixels")

                partitions.append({
                    'suffix': suffix,
                    'name': f"{sample_name}{suffix}",
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                    'width': width,
                    'height': height
                })
                break

            except ValueError:
                print("   ⚠ Please enter valid integer coordinates")

    return partitions

def partition_channel_file(source_file: Path, partition: Dict, output_dir: Path,
                          original_sample_name: str):
    """Partition a single channel file and save to output directory with renamed filename."""
    # Read the image
    image_data = tifffile.imread(source_file)

    # Extract the partition
    partitioned_data = image_data[
        partition['y_min']:partition['y_max'],
        partition['x_min']:partition['x_max']
    ]

    # Replace original sample name with partition name in filename
    # e.g., JL98_1.0.1_R000_Cy3_AF_I.ome.tif -> JL98A_1.0.1_R000_Cy3_AF_I.ome.tif
    output_filename = source_file.name.replace(original_sample_name, partition['name'], 1)

    # Fallback: if the sample name doesn't appear at the start, try to handle it
    if output_filename == source_file.name:
        # Try to replace the first occurrence of the pattern
        output_filename = re.sub(f'^{re.escape(original_sample_name)}',
                                partition['name'], source_file.name)

    # Save the partitioned image
    output_path = output_dir / output_filename
    tifffile.imwrite(output_path, partitioned_data,
                    photometric='minisblack',
                    compression='lzw',
                    metadata={'axes': 'YX'})

    return output_path

def create_partition_visualization(dapi_data: np.ndarray, partitions: List[Dict],
                                  output_path: Path, sample_name: str):
    """Create a visualization showing all partitions on the DAPI image."""
    height, width = dapi_data.shape[:2]

    # Downsample aggressively to prevent memory issues
    max_display_size = 800  # Reduced from 2000 to save memory
    scale = min(max_display_size / max(height, width), 1.0)
    display_height = int(height * scale)
    display_width = int(width * scale)

    if scale < 1.0:
        from skimage.transform import resize
        display_image = resize(dapi_data, (display_height, display_width),
                             preserve_range=True, anti_aliasing=True)
    else:
        display_image = dapi_data.copy()

    # Delete original to free memory
    del dapi_data
    gc.collect()

    # Normalize for display
    p2, p98 = np.percentile(display_image, (2, 98))
    display_image = np.clip((display_image - p2) / (p98 - p2), 0, 1)

    # Convert to uint8 to save memory
    display_image = (display_image * 255).astype(np.uint8)

    # Create smaller figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(display_image, cmap='gray', interpolation='nearest')

    # Draw partition rectangles
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple']

    for i, partition in enumerate(partitions):
        color = colors[i % len(colors)]

        # Scale coordinates for display
        x_min = partition['x_min'] * scale
        x_max = partition['x_max'] * scale
        y_min = partition['y_min'] * scale
        y_max = partition['y_max'] * scale

        # Draw rectangle
        rect_width = x_max - x_min
        rect_height = y_max - y_min

        from matplotlib.patches import Rectangle
        rect = Rectangle((x_min, y_min), rect_width, rect_height,
                        linewidth=3, edgecolor=color, facecolor='none',
                        linestyle='-', alpha=0.8)
        ax.add_patch(rect)

        # Add label
        label_x = x_min + rect_width / 2
        label_y = y_min + rect_height / 2
        ax.text(label_x, label_y, partition['suffix'],
               color=color, fontsize=24, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    ax.set_title(f'{sample_name} - Partition Plan\n'
                f'{len(partitions)} partitions defined',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X coordinate (pixels)', fontsize=12)
    ax.set_ylabel('Y coordinate (pixels)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=75, bbox_inches='tight')  # Reduced DPI further
    print(f"   ✓ Saved partition visualization to: {output_path}")
    plt.close()

    # Clean up memory
    del display_image, fig, ax
    gc.collect()

def collect_sample_partitions(sample_name: str, source_dir: Path,
                             visualization_dir: Path) -> Dict:
    """Collect partition information for a single sample."""
    print(f"\n{'='*70}")
    print(f"🔬 Collecting Partitions for Sample: {sample_name}")
    print(f"{'='*70}")

    # Setup paths
    sample_source_dir = source_dir / sample_name
    if not sample_source_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_source_dir}")

    # Find DAPI channel
    dapi_file = find_dapi_channel(sample_source_dir)
    print(f"✓ Found DAPI channel: {dapi_file.name}")

    # Check file size before loading
    file_size_mb = os.path.getsize(dapi_file) / (1024 * 1024)
    print(f"✓ DAPI file size: {file_size_mb:.1f} MB")

    # Load DAPI data with memory-efficient approach
    print(f"⏳ Loading DAPI image (this may take a moment for large files)...")
    dapi_data = tifffile.imread(dapi_file)
    print(f"✓ Loaded DAPI image: {dapi_data.shape}")

    # Get dimensions before we potentially delete dapi_data
    image_height, image_width = dapi_data.shape[:2]

    # Create coordinate grid visualization
    vis_dir = visualization_dir / sample_name
    vis_dir.mkdir(parents=True, exist_ok=True)

    grid_path = vis_dir / f"{sample_name}_coordinate_grid.png"
    _ = create_coordinate_grid(dapi_data.copy(), grid_path, sample_name)

    print(f"\n📍 Please open the coordinate grid image to view coordinates:")
    print(f"   {grid_path}")
    print(f"\n   Use these coordinates to define your partitions.")

    # Get partition information from user
    partitions = get_partition_info(sample_name, image_width, image_height)

    # Create partition visualization
    partition_vis_path = vis_dir / f"{sample_name}_partitions.png"
    create_partition_visualization(dapi_data.copy(), partitions, partition_vis_path, sample_name)

    # Delete DAPI data to free memory immediately
    del dapi_data
    gc.collect()
    print(f"✓ Released DAPI image from memory")

    # Get all channel files
    channel_files = sorted(sample_source_dir.glob('*.ome.tif'))

    return {
        'sample_name': sample_name,
        'source_dir': sample_source_dir,
        'partitions': partitions,
        'channel_files': channel_files,
        'image_width': image_width,
        'image_height': image_height,
        'vis_dir': vis_dir
    }

def process_all_partitions(sample_partitions_list: List[Dict], output_dir: Path):
    """Process and save all partitions for all samples."""

    # Show summary of all partitions
    print(f"\n{'='*70}")
    print(f"📋 PARTITION SUMMARY - All Samples")
    print(f"{'='*70}")

    total_partitions = 0
    for sample_info in sample_partitions_list:
        sample_name = sample_info['sample_name']
        partitions = sample_info['partitions']
        num_channels = len(sample_info['channel_files'])

        print(f"\n{sample_name} ({num_channels} channels):")
        for p in partitions:
            print(f"   • {p['name']}: x=[{p['x_min']}:{p['x_max']}], "
                  f"y=[{p['y_min']}:{p['y_max']}], size={p['width']}x{p['height']}")
            total_partitions += 1

    print(f"\nTotal partitions to create: {total_partitions}")

    # Confirm with user
    response = input(f"\nProceed with partitioning all samples? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ Partitioning cancelled by user")
        return False

    # Process each sample
    all_metadata = []

    for sample_info in sample_partitions_list:
        sample_name = sample_info['sample_name']
        partitions = sample_info['partitions']
        channel_files = sample_info['channel_files']
        vis_dir = sample_info['vis_dir']

        print(f"\n{'='*70}")
        print(f"🔄 Processing Sample: {sample_name}")
        print(f"{'='*70}")
        print(f"Processing {len(channel_files)} channel files...")

        # Process each partition
        partition_metadata = []

        for partition in partitions:
            print(f"\n--- Creating partition: {partition['name']} ---")

            # Create output directory
            partition_output_dir = output_dir / partition['name']
            partition_output_dir.mkdir(parents=True, exist_ok=True)

            # Process each channel
            for i, channel_file in enumerate(channel_files, 1):
                output_path = partition_channel_file(
                    channel_file, partition, partition_output_dir, sample_name
                )
                print(f"   [{i}/{len(channel_files)}] {channel_file.name} → {output_path.name}")

            # Save partition metadata
            metadata = {
                'original_sample': sample_name,
                'partition_name': partition['name'],
                'suffix': partition['suffix'],
                'bounds': {
                    'x_min': partition['x_min'],
                    'x_max': partition['x_max'],
                    'y_min': partition['y_min'],
                    'y_max': partition['y_max']
                },
                'dimensions': {
                    'width': partition['width'],
                    'height': partition['height']
                },
                'num_channels': len(channel_files)
            }

            metadata_path = partition_output_dir / 'partition_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            partition_metadata.append(metadata)
            print(f"   ✓ Partition {partition['name']} complete!")

        # Save overall metadata for this sample
        overall_metadata = {
            'original_sample': sample_name,
            'partitions': partition_metadata,
            'total_channels': len(channel_files),
            'original_dimensions': {
                'width': sample_info['image_width'],
                'height': sample_info['image_height']
            }
        }

        metadata_path = vis_dir / f"{sample_name}_partition_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(overall_metadata, f, indent=2)

        all_metadata.append(overall_metadata)

        print(f"\n✅ Sample {sample_name} partitioning complete!")
        print(f"Created {len(partitions)} partitions:")
        for p in partitions:
            print(f"   • {p['name']}: {p['width']}x{p['height']} pixels")

    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ ALL SAMPLES PARTITIONING COMPLETE!")
    print(f"{'='*70}")
    print(f"Processed {len(sample_partitions_list)} samples:")
    for metadata in all_metadata:
        print(f"   • {metadata['original_sample']}: {len(metadata['partitions'])} partitions")
    print(f"\nOutput directory: {output_dir}")

    return True

def main():
    parser = argparse.ArgumentParser(
        description='Partition samples before running mcmicro pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Single sample
  python scripts/partition_samples.py --sample_names JL216

  # Multiple samples (batch mode)
  python scripts/partition_samples.py --sample_names JL216 JL217 JL218

  # Custom directories
  python scripts/partition_samples.py --sample_names JL216 --source_dir custom_rawdata

This will:
  1. For each sample, generate a coordinate grid image of the DAPI channel
  2. Prompt you to define partition boundaries for each sample
  3. Show a summary of ALL partitions across ALL samples
  4. Ask for confirmation, then create all partitions at once
  5. Each partition will have correctly renamed channel files (e.g., JL216A_*.ome.tif)
        """
    )

    parser.add_argument('--sample_names', nargs='+', required=True,
                       help='Names of samples to partition (e.g., JL216 JL217 JL218)')
    parser.add_argument('--source_dir', default='rawdata_prepartition',
                       help='Source directory containing pre-partition samples (default: rawdata_prepartition)')
    parser.add_argument('--output_dir', default='rawdata',
                       help='Output directory for partitioned samples (default: rawdata)')
    parser.add_argument('--visualization_dir', default='partition_visualizations',
                       help='Directory to save coordinate grids and partition visualizations')

    args = parser.parse_args()

    # Convert to Path objects
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    visualization_dir = Path(args.visualization_dir)

    # Verify source directory exists
    if not source_dir.exists():
        print(f"❌ Error: Source directory does not exist: {source_dir}")
        print(f"   Please create it and place your samples inside")
        return 1

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Check for scikit-image (needed for downsampling large images)
        try:
            import skimage
        except ImportError:
            print("⚠ Warning: scikit-image not found. Install with: pip install scikit-image")
            print("   (Required for displaying very large images)")

        # Collect partition info for all samples
        sample_partitions_list = []

        for sample_name in args.sample_names:
            try:
                sample_info = collect_sample_partitions(
                    sample_name, source_dir, visualization_dir
                )
                sample_partitions_list.append(sample_info)
            except Exception as e:
                print(f"\n❌ Error collecting partitions for {sample_name}: {e}")
                import traceback
                traceback.print_exc()
                print("\nContinuing with remaining samples...")
                continue

        if not sample_partitions_list:
            print("\n❌ No samples were successfully processed")
            return 1

        # Process all partitions
        success = process_all_partitions(sample_partitions_list, output_dir)

        if success:
            print(f"\n🎉 Partitioning complete!")
            print(f"\nNext steps:")
            print(f"  1. Verify the partitioned samples in: {output_dir}")
            print(f"  2. Run the mcmicro pipeline on each partition")
            print(f"     bash run_mcmicro_tiled.sh")
            return 0
        else:
            return 1

    except Exception as e:
        print(f"\n❌ Error during partitioning: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
