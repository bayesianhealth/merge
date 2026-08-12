import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import det_constants as C
import pipeline_utils as pu
from snowflake_utils import get_connection, run_sql


def stats_sql():
    fid_list = ", ".join(f"'{f}'" for f in C.FEATURE_FIDS)
    ward_list = ", ".join(f"'{w}'" for w in C.WARD_LOCS)
    return f"""
WITH train_enc AS (
    SELECT enc_id FROM {C.T_ENC_SPLIT} WHERE split = 'train'
),
ward AS (
    -- prediction-eligible time only: the same ward levels that generate anchors.
    -- Intervals with a NULL level_of_care are excluded, so an observation counts
    -- only if it provably falls inside an eligible unit.
    SELECT enc_id, enter_time, leave_time
    FROM {C.T_LOC}
    WHERE level_of_care IN ({ward_list})
      AND leave_time IS NOT NULL
      AND leave_time > enter_time
),
obs AS (
    -- EXISTS (not a join) so overlapping ward intervals cannot duplicate an
    -- observation and skew the mean/std.
    SELECT t.fid AS fid, TRY_TO_DOUBLE(t.value) AS v
    FROM {C.T_CDM_T} t
    JOIN train_enc e ON e.enc_id = t.enc_id
    WHERE t.fid IN ({fid_list})
      AND TRY_TO_DOUBLE(t.value) IS NOT NULL
      AND {C.not_deleted('t')}
      AND EXISTS (
          SELECT 1 FROM ward w
          WHERE w.enc_id = t.enc_id
            AND t.tsp >= w.enter_time
            AND t.tsp <= w.leave_time
      )
)
SELECT fid, AVG(v) AS mean, STDDEV_POP(v) AS std, COUNT(*) AS n
FROM obs
GROUP BY fid
"""


def main(args):
    marker_dir = args.marker_dir
    if not args.force and pu.is_done(marker_dir, pu.STEP4_SCALER):
        print("Step 4 (DET_FEATURE_SPEC) already complete; skipping. Use --force to redo.")
        return

    conn = get_connection()
    try:
        print("Computing train-only feature statistics ...")
        rows = run_sql(stats_sql(), conn, fetch=True)
        stats = {fid: (mean, std, n) for fid, mean, std, n in rows}

        # assemble rows in canonical feature order; fall back to mean=0/std=1 when a
        # feature has no usable train observations (avoids divide-by-zero downstream)
        values = []
        for idx, fid in enumerate(C.FEATURE_FIDS):
            mean, std, n = stats.get(fid, (None, None, 0))
            mean = float(mean) if mean is not None else 0.0
            std = float(std) if (std is not None and std not in (0.0,)) else 1.0
            modality = C.MODALITY_OF[fid]
            values.append(f"('{fid}', '{modality}', {idx}, {mean}, {std}, {C.SEQ_LEN})")
            if n == 0:
                print(f"  ! warning: no train observations for '{fid}' (mean=0, std=1 fallback)")

        run_sql(f"""
CREATE OR REPLACE TABLE {C.T_FEATURE_SPEC} (
    feature_name STRING,
    modality STRING,
    feature_index INT,
    mean FLOAT,
    std FLOAT,
    seq_len INT
)""", conn)
        run_sql(
            f"INSERT INTO {C.T_FEATURE_SPEC} "
            f"(feature_name, modality, feature_index, mean, std, seq_len) VALUES "
            + ", ".join(values),
            conn,
        )
        print(f"  - {C.T_FEATURE_SPEC}: {len(values)} features written")
        print(f"  - fit on {sum(int(v[2] or 0) for v in stats.values()):,} ward-only train observations")
    finally:
        conn.close()

    pu.write_marker(marker_dir, pu.STEP4_SCALER, {"table": C.T_FEATURE_SPEC, "features": len(C.FEATURE_FIDS)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker_dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
