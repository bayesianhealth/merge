import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import det_constants as C
import pipeline_utils as pu
from snowflake_utils import get_connection, run_sql


def build_split_sql():
    train_hi = C.TRAIN_PCT
    val_hi = C.TRAIN_PCT + C.VAL_PCT
    # ABS(HASH(enc_id || seed)) % 100 -> stable bucket independent of row order
    return f"""
CREATE OR REPLACE TABLE {C.T_ENC_SPLIT} AS
SELECT
    enc_id,
    CASE
        WHEN bucket < {train_hi} THEN 'train'
        WHEN bucket < {val_hi}   THEN 'val'
        ELSE 'test'
    END AS split
FROM (
    SELECT DISTINCT enc_id,
           ABS(HASH(enc_id || '_{C.SPLIT_SEED}')) % 100 AS bucket
    FROM {C.T_LOC}
)
"""


def build_anchors_sql():
    ward_list = ", ".join(f"'{w}'" for w in C.WARD_LOCS)
    return f"""
CREATE OR REPLACE TABLE {C.T_ANCHORS} AS
WITH ward AS (
    SELECT enc_id, enter_time, leave_time
    FROM {C.T_LOC}
    WHERE level_of_care IN ({ward_list})
      AND leave_time IS NOT NULL
      AND leave_time > enter_time
),
-- Merge contiguous eligible intervals into spans (gaps-and-islands). A ward->stepdown
-- move is continuous eligible time, so truncating the 48h window at the *interval*
-- boundary would discard real ward history; truncate at the start of the merged span
-- instead. Gaps up to ELIGIBLE_MERGE_GAP_HOURS are treated as continuous (ADT churn).
ordered AS (
    SELECT enc_id, enter_time, leave_time,
           MAX(leave_time) OVER (
               PARTITION BY enc_id ORDER BY enter_time
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
           ) AS prev_max_leave
    FROM ward
),
marked AS (
    SELECT *,
           CASE WHEN prev_max_leave IS NULL
                  OR enter_time > DATEADD('hour', {C.ELIGIBLE_MERGE_GAP_HOURS}, prev_max_leave)
                THEN 1 ELSE 0 END AS is_span_start
    FROM ordered
),
grouped AS (
    SELECT *,
           SUM(is_span_start) OVER (
               PARTITION BY enc_id ORDER BY enter_time ROWS UNBOUNDED PRECEDING
           ) AS span_id
    FROM marked
),
spans AS (
    SELECT enc_id, span_id,
           MIN(enter_time) AS span_start,
           MAX(leave_time) AS span_end
    FROM grouped
    GROUP BY enc_id, span_id
),
grid AS (
    SELECT
        s.enc_id,
        DATEADD('hour', g.value::int, DATE_TRUNC('hour', s.span_start)) AS anchor_tsp,
        s.span_start,
        s.span_end
    FROM spans s,
         LATERAL FLATTEN(INPUT => ARRAY_GENERATE_RANGE(
             0,
             DATEDIFF('hour', DATE_TRUNC('hour', s.span_start), s.span_end) + 1
         )) g
)
SELECT DISTINCT
    grid.enc_id,
    grid.anchor_tsp,
    -- start of the eligible span this anchor sits in; step 5 clips the trailing
    -- 48h window at this timestamp so pre-ward (ICU/ED) bins are never fed as real data
    grid.span_start AS eligible_start,
    sp.split
FROM grid
JOIN {C.T_ENC_SPLIT} sp ON sp.enc_id = grid.enc_id
WHERE grid.anchor_tsp >= grid.span_start
  AND grid.anchor_tsp <= grid.span_end
"""


def main(args):
    marker_dir = args.marker_dir
    if not args.force and pu.is_done(marker_dir, pu.STEP3_ANCHORS):
        print("Step 3 (_DET_ANCHORS) already complete; skipping. Use --force to redo.")
        return

    conn = get_connection()
    try:
        print(f"Building {C.T_ENC_SPLIT} ...")
        run_sql(build_split_sql(), conn)
        splits = run_sql(
            f"SELECT split, COUNT(*) FROM {C.T_ENC_SPLIT} GROUP BY split ORDER BY split",
            conn, fetch=True,
        )
        print(f"  - encounters per split: {dict(splits)}")

        print(f"Building {C.T_ANCHORS} ...")
        run_sql(build_anchors_sql(), conn)
        n_anchors = run_sql(f"SELECT COUNT(*) FROM {C.T_ANCHORS}", conn, fetch=True)[0][0]
        print(f"  - {C.T_ANCHORS}: {n_anchors} hourly anchors (pre min-obs filter)")
    finally:
        conn.close()

    pu.write_marker(marker_dir, pu.STEP3_ANCHORS, {"table": C.T_ANCHORS, "anchors": int(n_anchors)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker_dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
