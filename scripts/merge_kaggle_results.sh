#!/bin/bash

# Merge downloaded Kaggle results with local results
# Usage: bash scripts/merge_kaggle_results.sh /path/to/downloaded/results.zip

echo "=================================================================="
echo "MERGE KAGGLE RESULTS WITH LOCAL RESULTS"
echo "=================================================================="

if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/merge_kaggle_results.sh /path/to/results.zip"
    echo ""
    echo "Example:"
    echo "  bash scripts/merge_kaggle_results.sh ~/Downloads/results.zip"
    echo ""
    exit 1
fi

ZIP_FILE="$1"

if [ ! -f "$ZIP_FILE" ]; then
    echo "❌ Error: File not found: $ZIP_FILE"
    exit 1
fi

# Create temp directory
TEMP_DIR=$(mktemp -d)
echo "📦 Extracting to temporary directory..."
unzip -q "$ZIP_FILE" -d "$TEMP_DIR"

# Count new results
NEW_RESULTS=$(find "$TEMP_DIR/results" -maxdepth 1 -type d -name "*_*" | wc -l)

echo "✓ Found $NEW_RESULTS new experiment(s)"
echo ""

# Copy to local results, avoiding duplicates
echo "📋 Merging results:"
for folder in "$TEMP_DIR/results"/*_*; do
    if [ -d "$folder" ]; then
        basename=$(basename "$folder")
        
        if [ -d "results/$basename" ]; then
            echo "  ⚠️  Skipping $basename (already exists)"
        else
            echo "  ✓ Adding $basename"
            cp -r "$folder" results/
        fi
    fi
done

# Copy CSV files if they exist
if [ -f "$TEMP_DIR/results/overall_comparison.csv" ]; then
    echo ""
    echo "📊 Copying comparison tables..."
    cp "$TEMP_DIR/results"/*.csv results/ 2>/dev/null || true
    echo "  ✓ Updated CSV files"
fi

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "=================================================================="
echo "✅ MERGE COMPLETE"
echo "=================================================================="
echo ""
echo "Current results:"
ls -1 results/ | grep "_" | head -10
echo ""
echo "Run analysis:"
echo "  python experiments/quick_compare.py"
echo "  bash experiments/run_all_analysis.sh"
echo ""
