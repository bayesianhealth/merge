#!/usr/bin/env bash
# Usage: bash data_preprocess/preprocess_mimic.sh <notes_file_path> [gpu] [batch_size]
#
# notes_file_path: path to radiology notes CSV
#                  (e.g. /Volumes/mimiciv/note/radiology.csv.gz)
# gpu:            GPU device ID for embedding steps (default: 0)
# batch_size:     Step 1 admissions-per-batch (default: 40000). Larger = faster
#                 but more RAM. Tune to the instance you run Step 1 on.
#
# Data is read from Unity Catalog (mimiciv catalog) via databricks_utils.py.
# Override the catalog name with the MIMICIV_CATALOG env var if needed.
# All intermediate and final files are written to ./data/
# Run from the mimiciv/ directory.
#
# Steps are individually resumable: each writes a completion marker and is
# skipped on rerun (use --force on a step to redo it). This means you can run
# the non-GPU steps on a high-memory instance, then switch to a GPU instance
# and rerun this script -- finished steps are skipped automatically.

set -e

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash data_preprocess/preprocess_mimic.sh <notes_file_path> [gpu] [batch_size]"
    echo ""
    echo "  notes_file_path: path to radiology notes CSV (e.g. /Volumes/mimiciv/note/radiology.csv.gz)"
    echo "  gpu:             GPU device ID (default: 0)"
    echo "  batch_size:      Step 1 admissions per batch (default: 40000)"
    exit 1
fi

NOTES_FILE_PATH=$1
GPU=${2:-0}
BATCH_SIZE=${3:-40000}
OUTPUT_DIR=./data

echo "=== Step 1/6: Irregular time series (labs + vitals) [batch_size=$BATCH_SIZE] ==="
python data_preprocess/preprocess_irg_time_series.py \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE"

echo "=== Step 2/6: Imputed regular time series ==="
python data_preprocess/preprocess_imputed_time_series.py \
    --output_dir "$OUTPUT_DIR"

echo "=== Step 3/6: Radiology notes text ==="
python data_preprocess/preprocess_notes.py \
    --notes_file_path "$NOTES_FILE_PATH" \
    --output_dir "$OUTPUT_DIR"

echo "=== Step 4/6: BioBERT note embeddings (GPU-intensive) ==="
python data_preprocess/preprocess_notes_embeddings.py \
    --output_dir "$OUTPUT_DIR" \
    --device_number "$GPU"

echo "=== Step 5/6: Create IHM task (train/val/test pkl files) ==="
python data_preprocess/create_ihm_task.py \
    --output_dir "$OUTPUT_DIR" \
    --restrict_hours 48 \
    --include_notes \
    --include_missing \
    --standardize_features \
    --seed 42

echo "=== Step 6/6: Create LOS task (train/val/test pkl files) ==="
python data_preprocess/create_los_task.py \
    --output_dir "$OUTPUT_DIR" \
    --include_notes \
    --include_missing \
    --standardize_features \
    --seed 42

echo ""
echo "Preprocessing complete. Output files:"
echo "  IHM: $OUTPUT_DIR/ihm/{train,val,test}_ihm-48-notes-missingInd-standardized_stays.pkl"
echo "  LOS: $OUTPUT_DIR/los/{train,val,test}_los-notes-missingInd-standardized_stays.pkl"
