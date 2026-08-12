#!/usr/bin/env bash
# Orchestrates the deterioration preprocessing pipeline (TEST.PUBLIC -> TEST.SILVER).
# Co-authored with CoCo
#
# Runs the six steps in order. Each step is idempotent (skipped if its marker exists);
# pass --force to a step to redo it. Step 5 is the heavy per-encounter Snowpark job.
#
# Usage:
#   bash preprocess_deterioration.sh            # full run
#   LIMIT_ENCOUNTERS=200 bash preprocess_deterioration.sh   # quick trial of step 5
set -euo pipefail

cd "$(dirname "$0")"

LIMIT_ARG=""
if [[ -n "${LIMIT_ENCOUNTERS:-}" ]]; then
    LIMIT_ARG="--limit_encounters ${LIMIT_ENCOUNTERS}"
fi

echo "==== Step 1/6: care-unit intervals (_DET_LOC) ===="
python3 01_build_loc.py

echo "==== Step 2/6: event timestamps (_DET_EVENTS) ===="
python3 02_build_events.py

echo "==== Step 3/6: hourly anchors + split (_DET_ANCHORS) ===="
python3 03_build_anchors.py

echo "==== Step 4/6: train scaler (DET_FEATURE_SPEC) ===="
python3 04_build_scaler.py

echo "==== Step 5/6: build examples (_DET_ALL) ===="
python3 05_build_examples.py ${LIMIT_ARG}

echo "==== Step 6/6: write DET_TRAIN/VAL/TEST ===="
python3 06_write_splits.py

echo "Deterioration preprocessing complete."
