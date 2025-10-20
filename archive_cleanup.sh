#!/bin/bash
# archive_cleanup.sh - Prepare spatial pipeline results for archival
# ./archive_cleanup.sh results

set -euo pipefail

RESULTS_DIR="${1:-results}"

echo "=== SPATIAL PIPELINE ARCHIVE & CLEANUP ==="
echo "Results directory: $RESULTS_DIR"
echo ""

# Check for pigz (parallel gzip), fallback to gzip
if command -v pigz &> /dev/null; then
    COMPRESSOR="pigz"
    echo "Using pigz (parallel compression)"
else
    COMPRESSOR="gzip"
    echo "Using gzip (install pigz for faster compression: apt install pigz)"
fi
echo ""

# Process each sample directory
for SAMPLE_DIR in "$RESULTS_DIR"/*/; do
    SAMPLE=$(basename "$SAMPLE_DIR")
    
    # Skip if not a sample directory
    [[ ! -d "$SAMPLE_DIR/final" ]] && continue
    
    echo "Processing: $SAMPLE"
    cd "$SAMPLE_DIR"
    
    # Calculate current size
    BEFORE=$(du -sh . | cut -f1)
    echo "  Current size: $BEFORE"
    
    # 1. Archive masks (keep for publication)
    if [[ -d "nuclei_masks" ]]; then
        echo "  Archiving nuclei_masks..."
        tar -I "$COMPRESSOR" -cf nuclei_masks.tar.gz nuclei_masks/
        rm -rf nuclei_masks/
    fi
    
    if [[ -d "cell_masks" ]]; then
        echo "  Archiving cell_masks..."
        tar -I "$COMPRESSOR" -cf cell_masks.tar.gz cell_masks/
        rm -rf cell_masks/
    fi
    
    # 2. Delete reconstructable/intermediate data
    echo "  Removing intermediate files..."
    rm -rf tiles/ 2>/dev/null || true
    rm -rf background_corrected/ 2>/dev/null || true
    rm -rf quantification/ 2>/dev/null || true
    rm -rf spatial/ 2>/dev/null || true
    rm -f *_dapi_bg_subtracted.tif 2>/dev/null || true
    rm -f *_nuclear_membrane_input.tif 2>/dev/null || true
    rm -f *_cp_masks.tif 2>/dev/null || true
    
    # 3. Compress large CSVs in final/ (optional)
    if [[ -f "final/combined_quantification.csv" ]]; then
        SIZE=$(stat -f%z "final/combined_quantification.csv" 2>/dev/null || stat -c%s "final/combined_quantification.csv")
        if (( SIZE > 10485760 )); then  # > 10MB
            echo "  Compressing large CSV..."
            $COMPRESSOR -k final/combined_quantification.csv
        fi
    fi
    
    # Calculate savings
    AFTER=$(du -sh . | cut -f1)
    echo "  Final size: $AFTER"
    echo "  ✓ Complete"
    echo ""
    
    cd - > /dev/null
done

echo "=== CLEANUP COMPLETE ==="
echo ""
echo "Archived files can be extracted with:"
echo "  tar -xzf nuclei_masks.tar.gz"
echo "  tar -xzf cell_masks.tar.gz"