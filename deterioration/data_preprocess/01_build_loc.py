import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import det_constants as C
import pipeline_utils as pu
from snowflake_utils import get_connection, run_sql


def build_loc_sql():
    ward_list = ", ".join(f"'{w}'" for w in C.WARD_LOCS)
    return f"""
CREATE OR REPLACE TABLE {C.T_LOC} AS
WITH ecu AS (
    SELECT
        enc_id,
        care_unit,
        unmapped_hl7_care_unit,
        enter_time,
        leave_time
    FROM {C.T_ENC_CARE_UNIT}
    WHERE enc_id IS NOT NULL
      AND enter_time IS NOT NULL
      AND {C.NOT_DELETED}
),
cu AS (
    -- CARE_UNITS is keyed three ways (care_unit / adt_name / care_unit_ref) and
    -- ENC_CARE_UNIT.care_unit can match any of them, so all three are lookup keys.
    -- Matching on care_unit alone leaves ~21% of intervals with a NULL level_of_care
    -- and silently drops ~37k ward encounters from the cohort.
    SELECT care_unit_key, ANY_VALUE(level_of_care) AS level_of_care
    FROM (
        SELECT LOWER(care_unit) AS care_unit_key, LOWER(level_of_care) AS level_of_care
        FROM {C.T_CARE_UNITS} WHERE care_unit IS NOT NULL AND {C.NOT_DELETED}
        UNION ALL
        SELECT LOWER(adt_name), LOWER(level_of_care)
        FROM {C.T_CARE_UNITS} WHERE adt_name IS NOT NULL AND {C.NOT_DELETED}
        UNION ALL
        SELECT LOWER(care_unit_ref), LOWER(level_of_care)
        FROM {C.T_CARE_UNITS} WHERE care_unit_ref IS NOT NULL AND {C.NOT_DELETED}
    )
    WHERE level_of_care IS NOT NULL
    GROUP BY care_unit_key
),
joined AS (
    SELECT
        ecu.enc_id,
        ecu.care_unit,
        ecu.enter_time,
        ecu.leave_time,
        COALESCE(cu_primary.level_of_care, cu_fallback.level_of_care) AS level_of_care
    FROM ecu
    LEFT JOIN cu AS cu_primary
        ON LOWER(ecu.care_unit) = cu_primary.care_unit_key
    LEFT JOIN cu AS cu_fallback
        ON LOWER(ecu.unmapped_hl7_care_unit) = cu_fallback.care_unit_key
)
SELECT *
FROM joined
WHERE enc_id IN (
    -- cohort: encounters with >=1 ward interval
    SELECT DISTINCT enc_id
    FROM joined
    WHERE level_of_care IN ({ward_list})
)
"""


def main(args):
    marker_dir = args.marker_dir
    if not args.force and pu.is_done(marker_dir, pu.STEP1_LOC):
        print("Step 1 (_DET_LOC) already complete; skipping. Use --force to redo.")
        return

    conn = get_connection()
    try:
        print(f"Building {C.T_LOC} ...")
        run_sql(build_loc_sql(), conn)
        n_rows = run_sql(f"SELECT COUNT(*) FROM {C.T_LOC}", conn, fetch=True)[0][0]
        n_enc = run_sql(f"SELECT COUNT(DISTINCT enc_id) FROM {C.T_LOC}", conn, fetch=True)[0][0]
        print(f"  - {C.T_LOC}: {n_rows} intervals across {n_enc} cohort encounters")
    finally:
        conn.close()

    pu.write_marker(marker_dir, pu.STEP1_LOC, {"table": C.T_LOC, "rows": int(n_rows), "encounters": int(n_enc)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker_dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
