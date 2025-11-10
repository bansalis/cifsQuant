#!/usr/bin/env python3
"""
Sample Partitioning Preprocessor for MCMICRO Pipeline

This script allows you to partition multi-condition samples on the same slide
before running the main mcmicro pipeline. It creates coordinate-based
partitions that are saved as separate samples in rawdata/.

Usage:
    python scripts/partition_samples.py --sample_name JL216
    python scripts/partition_samples.py --sample_name JL216 --source_dir custom_rawdata
"""

import argparse
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Dict, Tuple
import shutil

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

    # Downsample if image is very large to make visualization manageable
    max_display_size = 4000
    height, width = image_data.shape[:2]

    if max(height, width) > max_display_size:
        scale = max_display_size / max(height, width)
        display_height = int(height * scale)
        display_width = int(width * scale)
        from skimage.transform import resize
        display_image = resize(image_data, (display_height, display_width),
                             preserve_range=True, anti_aliasing=True)
        print(f"   Display size: {display_width} x {display_height} pixels (scaled for visualization)")
    else:
        display_image = image_data
        display_height, display_width = height, width
        scale = 1.0

    # Normalize for display
    p2, p98 = np.percentile(display_image, (2, 98))
    display_image = np.clip((display_image - p2) / (p98 - p2), 0, 1)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(display_image, cmap='gray', interpolation='nearest')

    # Add coordinate grid (every 1000 pixels)
    grid_spacing = 1000

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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved coordinate grid to: {output_path}")
    plt.close()

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

def partition_channel_file(source_file: Path, partition: Dict, output_dir: Path):
    """Partition a single channel file and save to output directory."""
    # Read the image
    image_data = tifffile.imread(source_file)

    # Extract the partition
    partitioned_data = image_data[
        partition['y_min']:partition['y_max'],
        partition['x_min']:partition['x_max']
    ]

    # Generate output filename (replace original sample name with partitioned name)
    output_filename = source_file.name

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

    # Downsample for display if needed
    max_display_size = 4000
    if max(height, width) > max_display_size:
        scale = max_display_size / max(height, width)
        display_height = int(height * scale)
        display_width = int(width * scale)
        from skimage.transform import resize
        display_image = resize(dapi_data, (display_height, display_width),
                             preserve_range=True, anti_aliasing=True)
    else:
        display_image = dapi_data
        scale = 1.0

    # Normalize for display
    p2, p98 = np.percentile(display_image, (2, 98))
    display_image = np.clip((display_image - p2) / (p98 - p2), 0, 1)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved partition visualization to: {output_path}")
    plt.close()

def partition_sample(sample_name: str, source_dir: Path, output_dir: Path,
                    visualization_dir: Path):
    """Main function to partition a sample."""
    print(f"\n{'='*70}")
    print(f"🔬 Processing Sample: {sample_name}")
    print(f"{'='*70}")

    # Setup paths
    sample_source_dir = source_dir / sample_name
    if not sample_source_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_source_dir}")

    # Find DAPI channel
    dapi_file = find_dapi_channel(sample_source_dir)
    print(f"✓ Found DAPI channel: {dapi_file.name}")

    # Load DAPI data
    dapi_data = tifffile.imread(dapi_file)
    print(f"✓ Loaded DAPI image: {dapi_data.shape}")

    # Create coordinate grid visualization
    vis_dir = visualization_dir / sample_name
    vis_dir.mkdir(parents=True, exist_ok=True)

    grid_path = vis_dir / f"{sample_name}_coordinate_grid.png"
    image_width, image_height = create_coordinate_grid(dapi_data, grid_path, sample_name)

    print(f"\n📍 Please open the coordinate grid image to view coordinates:")
    print(f"   {grid_path}")
    print(f"\n   Use these coordinates to define your partitions.")

    # Get partition information from user
    partitions = get_partition_info(sample_name, image_width, image_height)

    # Create partition visualization
    partition_vis_path = vis_dir / f"{sample_name}_partitions.png"
    create_partition_visualization(dapi_data, partitions, partition_vis_path, sample_name)

    # Confirm with user
    print(f"\n📋 Partition Summary:")
    for p in partitions:
        print(f"   {p['name']}: x=[{p['x_min']}:{p['x_max']}], "
              f"y=[{p['y_min']}:{p['y_max']}], size={p['width']}x{p['height']}")

    response = input(f"\nProceed with partitioning? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ Partitioning cancelled by user")
        return False

    # Get all channel files
    channel_files = sorted(sample_source_dir.glob('*.ome.tif'))
    print(f"\n🔄 Processing {len(channel_files)} channel files...")

    # Process each partition
    partition_metadata = []

    for partition in partitions:
        print(f"\n--- Creating partition: {partition['name']} ---")

        # Create output directory
        partition_output_dir = output_dir / partition['name']
        partition_output_dir.mkdir(parents=True, exist_ok=True)

        # Process each channel
        for i, channel_file in enumerate(channel_files, 1):
            output_path = partition_channel_file(channel_file, partition, partition_output_dir)
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

    # Save overall metadata
    overall_metadata = {
        'original_sample': sample_name,
        'partitions': partition_metadata,
        'total_channels': len(channel_files),
        'original_dimensions': {
            'width': image_width,
            'height': image_height
        }
    }

    metadata_path = vis_dir / f"{sample_name}_partition_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(overall_metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ Sample {sample_name} partitioning complete!")
    print(f"{'='*70}")
    print(f"Created {len(partitions)} partitions:")
    for p in partitions:
        print(f"   • {p['name']}: {p['width']}x{p['height']} pixels")
    print(f"\nOutput directory: {output_dir}")
    print(f"Visualizations: {vis_dir}")

    return True

def main():
    parser = argparse.ArgumentParser(
        description='Partition samples before running mcmicro pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/partition_samples.py --sample_name JL216
  python scripts/partition_samples.py --sample_name JL216 --source_dir custom_rawdata

This will:
  1. Generate a coordinate grid image of the DAPI channel
  2. Prompt you to define partition boundaries
  3. Create partitioned samples in rawdata/ (e.g., JL216A/, JL216B/)
  4. Each partition will contain all channels with the same naming structure
        """
    )

    parser.add_argument('--sample_name', required=True,
                       help='Name of the sample to partition (e.g., JL216)')
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
        print(f"   Please create it and place your samples inside (e.g., {source_dir}/{args.sample_name}/)")
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

        # Partition the sample
        success = partition_sample(
            args.sample_name,
            source_dir,
            output_dir,
            visualization_dir
        )

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
