
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import det_constants as C
import pipeline_utils as pu
from snowflake_utils import get_connection, run_sql


def build_events_sql():
    ward_list = ", ".join(f"'{w}'" for w in C.WARD_LOCS)
    prior_list = ", ".join(f"'{w}'" for w in C.UNPLANNED_PRIOR_LOCS)
    next_list = ", ".join(f"'{w}'" for w in C.UNPLANNED_NEXT_LOCS)
    return f"""
CREATE OR REPLACE TABLE {C.T_EVENTS} AS
WITH loc AS (
    SELECT enc_id, care_unit, enter_time, leave_time, level_of_care
    FROM {C.T_LOC}
),
-- ---- hospice / palliative encounters (excluded from mortality events) ----
hospice_enc AS (
    SELECT DISTINCT enc_id
    FROM {C.T_CDM_S}
    WHERE fid = '{C.SERVICE_TYPE_FID}'
      AND value RLIKE '.*({C.HOSPICE_REGEX}).*'
      AND {C.NOT_DELETED}
),
-- ---- deaths: 'discharge' events whose VALUE indicates Expired ----
deaths AS (
    SELECT enc_id, tsp
    FROM {C.T_CDM_T}
    WHERE fid = '{C.DISCHARGE_FID}'
      AND CONTAINS(value, '{C.DEATH_VALUE_SUBSTR}')
      AND {C.NOT_DELETED}
),
-- map each death to the last care unit with a KNOWN level_of_care before death.
-- At the moment of death patients are moved to an unmapped "expired" pseudo care-unit
-- (entered minutes before death, level_of_care NULL); skipping NULLs attributes the
-- death to the real ward/ICU the patient was deteriorating in.
death_loc AS (
    SELECT d.enc_id, d.tsp AS event_tsp, l.level_of_care
    FROM deaths d
    JOIN loc l
      ON l.enc_id = d.enc_id
     AND l.enter_time <= d.tsp
     AND l.level_of_care IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY d.enc_id, d.tsp ORDER BY l.enter_time DESC
    ) = 1
),
mortality_events AS (
    SELECT enc_id, event_tsp, 'mortality' AS outcome
    FROM death_loc
    WHERE level_of_care IN ({ward_list})
      AND enc_id NOT IN (SELECT enc_id FROM hospice_enc)
),
-- ---- ICU transfers with neighboring-interval context ----
loc_seq AS (
    SELECT
        enc_id, care_unit, enter_time, leave_time, level_of_care,
        LAG(level_of_care) OVER (PARTITION BY enc_id ORDER BY enter_time) AS loc_prev,
        LEAD(level_of_care) OVER (PARTITION BY enc_id ORDER BY enter_time) AS loc_next,
        LEAD(care_unit) OVER (PARTITION BY enc_id ORDER BY enter_time) AS care_unit_next
    FROM loc
),
icu_events AS (
    SELECT enc_id, enter_time AS event_tsp, 'icu' AS outcome
    FROM loc_seq
    WHERE level_of_care = '{C.ICU_LOC}'
      AND loc_prev IN ({prior_list})
      AND (
            DATEDIFF('hour', enter_time,
                     COALESCE(leave_time, TO_TIMESTAMP_NTZ('2125-01-01'))) >= {C.MIN_ICU_STAY_HOURS}
            OR care_unit_next = '{C.DISCHARGED_SENTINEL}'
            OR loc_next IN ({next_list})
      )
)
SELECT enc_id, event_tsp, outcome FROM mortality_events
UNION ALL
SELECT enc_id, event_tsp, outcome FROM icu_events
"""


def main(args):
    marker_dir = args.marker_dir
    if not args.force and pu.is_done(marker_dir, pu.STEP2_EVENTS):
        print("Step 2 (_DET_EVENTS) already complete; skipping. Use --force to redo.")
        return

    conn = get_connection()
    try:
        print(f"Building {C.T_EVENTS} ...")
        run_sql(build_events_sql(), conn)
        rows = run_sql(
            f"SELECT outcome, COUNT(*) FROM {C.T_EVENTS} GROUP BY outcome ORDER BY outcome",
            conn, fetch=True,
        )
        total = run_sql(f"SELECT COUNT(*) FROM {C.T_EVENTS}", conn, fetch=True)[0][0]
        print(f"  - {C.T_EVENTS}: {total} events ({dict(rows)})")
    finally:
        conn.close()

    pu.write_marker(marker_dir, pu.STEP2_EVENTS, {"table": C.T_EVENTS, "rows": int(total)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker_dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
