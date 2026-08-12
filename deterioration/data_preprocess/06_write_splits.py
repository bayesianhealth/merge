import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import det_constants as C
import pipeline_utils as pu
from snowflake_utils import get_connection, run_sql


def main(args):
    marker_dir = args.marker_dir
    if not args.force and pu.is_done(marker_dir, pu.STEP6_SPLITS):
        print("Step 6 (DET_TRAIN/VAL/TEST) already complete; skipping. Use --force to redo.")
        return

    targets = {"train": C.T_TRAIN, "val": C.T_VAL, "test": C.T_TEST}
    conn = get_connection()
    try:
        counts = {}
        for split, tbl in targets.items():
            print(f"Building {tbl} (split={split}) ...")
            run_sql(
                f"CREATE OR REPLACE TABLE {tbl} AS "
                f"SELECT * FROM {C.T_ALL} WHERE SPLIT = '{split}'",
                conn,
            )
            n = run_sql(f"SELECT COUNT(*) FROM {tbl}", conn, fetch=True)[0][0]
            pos = run_sql(f"SELECT COALESCE(SUM(LABEL),0) FROM {tbl}", conn, fetch=True)[0][0]
            counts[split] = (int(n), int(pos))
            print(f"  - {tbl}: {n} examples, {pos} positive")
    finally:
        conn.close()

    pu.write_marker(marker_dir, pu.STEP6_SPLITS, {
        "tables": list(targets.values()),
        "counts": {k: {"rows": v[0], "positive": v[1]} for k, v in counts.items()},
    })
    print("\nDeterioration task tables ready for the DataConnector:")
    for tbl in targets.values():
        print(f"  - {tbl}")
    print(f"  - {C.T_FEATURE_SPEC} (feature->modality->index + scaler)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker_dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
