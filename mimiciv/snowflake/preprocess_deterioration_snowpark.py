from deterioration_config import (
    HOSP, ICU, TS_TABLE, SCALER_TABLE,
    SRC_TRANSFERS, SRC_ADMISSIONS, SRC_D_LABITEMS, SRC_D_ITEMS,
    SRC_LABEVENTS, SRC_CHARTEVENTS,
    ICU_UNITS, STEPDOWN_UNITS, GENERAL_UNITS, OBSERVATION_UNITS,
    ELIGIBLE_LEVELS, LAB_EVENT_LIST, VITAL_EVENT_LIST, RENAME_MAP,
    FEATURE_NAMES, SEQ_LEN, LOOKBACK_HOURS, HORIZON_HOURS,
    POS_DECAY_RATE, NEG_DECAY_RATE, MIN_WEIGHT_FLOOR, WASHOUT_HOURS,
    TRAIN_PCT, VAL_PCT, flat_col, sql_str_list,
)

RAW_TABLE = f"{TS_TABLE}_RAW"
SLOT_TABLE = f"{TS_TABLE}_SLOTS"


def _level_of_care_case(col: str = "careunit") -> str:
    return (
        "CASE "
        f"WHEN {col} IN ({sql_str_list(ICU_UNITS)}) THEN 'icu' "
        f"WHEN {col} IN ({sql_str_list(STEPDOWN_UNITS)}) THEN 'stepdown' "
        f"WHEN {col} IN ({sql_str_list(GENERAL_UNITS)}) THEN 'general' "
        f"WHEN {col} IN ({sql_str_list(OBSERVATION_UNITS)}) THEN 'observation' "
        "ELSE 'other' END"
    )


def _feature_name_case() -> str:
    """SQL CASE mapping a d_items/d_labitems label to the snake_case feature name."""
    whens = " ".join(
        f"WHEN label = '{lbl.replace(chr(39), chr(39)*2)}' THEN '{feat}'"
        for lbl, feat in RENAME_MAP.items()
    )
    return f"CASE {whens} ELSE NULL END"


def build_slots_sql() -> str:
    """SQL building the long forward-filled slot table (one row per feature-hour slot).

    Strategy that avoids a full obs x 48 x 31 cross join:
      - find each measured (hadm, obs, feature, hours_before) with the latest value,
      - each measurement covers the contiguous hour range (prev_hb, hb] going forward
        in time (forward-fill), expanded with a LATERAL FLATTEN over only that range.
    Slot index k (0=oldest, 47=newest) maps to hours_before = 47 - k.
    """
    elig = sql_str_list(list(ELIGIBLE_LEVELS))
    return f"""
CREATE OR REPLACE TEMPORARY TABLE {SLOT_TABLE} AS
WITH transfers_loc AS (
    SELECT subject_id, hadm_id, careunit, intime, outtime,
           {_level_of_care_case()} AS level_of_care
    FROM {SRC_TRANSFERS}
),
ward_stays AS (
    SELECT subject_id, hadm_id, intime, outtime, level_of_care
    FROM transfers_loc
    WHERE level_of_care IN ({elig})
),
lab_itemids AS (
    SELECT itemid, label FROM (
        SELECT d.itemid, d.label,
               ROW_NUMBER() OVER (PARTITION BY d.label ORDER BY c.cnt DESC) AS rn
        FROM {SRC_D_LABITEMS} d
        JOIN (SELECT itemid, COUNT(*) cnt FROM {SRC_LABEVENTS} GROUP BY itemid) c
          ON d.itemid = c.itemid
        WHERE d.label IN ({sql_str_list(LAB_EVENT_LIST)})
    ) WHERE rn = 1
),
vital_itemids AS (
    SELECT itemid, label FROM (
        SELECT d.itemid, d.label,
               ROW_NUMBER() OVER (PARTITION BY d.label ORDER BY c.cnt DESC) AS rn
        FROM {SRC_D_ITEMS} d
        JOIN (SELECT itemid, COUNT(*) cnt FROM {SRC_CHARTEVENTS} GROUP BY itemid) c
          ON d.itemid = c.itemid
        WHERE d.label IN ({sql_str_list(VITAL_EVENT_LIST)})
    ) WHERE rn = 1
),
lab_meas AS (
    SELECT le.subject_id, le.hadm_id, le.charttime AS tsp, li.label, le.valuenum AS value
    FROM {SRC_LABEVENTS} le JOIN lab_itemids li ON le.itemid = li.itemid
    WHERE le.valuenum IS NOT NULL
),
vital_meas AS (
    SELECT ce.subject_id, ce.hadm_id, ce.charttime AS tsp, vi.label, ce.valuenum AS value
    FROM {SRC_CHARTEVENTS} ce JOIN vital_itemids vi ON ce.itemid = vi.itemid
    WHERE ce.valuenum IS NOT NULL
),
all_meas AS (
    SELECT subject_id, hadm_id, tsp, {_feature_name_case()} AS feature_name, value
    FROM (SELECT * FROM lab_meas UNION ALL SELECT * FROM vital_meas)
),
meas_on_ward AS (
    SELECT m.subject_id, m.hadm_id, m.tsp, m.feature_name, m.value
    FROM all_meas m
    JOIN ward_stays w
      ON m.hadm_id = w.hadm_id
     AND m.tsp BETWEEN w.intime AND w.outtime
    WHERE m.feature_name IS NOT NULL
),
obs_times AS (
    SELECT DISTINCT subject_id, hadm_id, tsp FROM meas_on_ward
),
obs_meas AS (
    SELECT * FROM (
        SELECT o.subject_id, o.hadm_id, o.tsp AS obs_tsp,
               m.feature_name, m.value, m.tsp AS meas_tsp,
               FLOOR(DATEDIFF('second', m.tsp, o.tsp) / 3600) AS hours_before
        FROM obs_times o
        JOIN meas_on_ward m
          ON o.hadm_id = m.hadm_id
         AND m.tsp BETWEEN DATEADD('hour', -{LOOKBACK_HOURS}, o.tsp) AND o.tsp
    ) WHERE hours_before BETWEEN 0 AND {SEQ_LEN - 1}
),
deduped AS (
    -- latest measurement per hour bucket
    SELECT subject_id, hadm_id, obs_tsp, feature_name, hours_before, value
    FROM obs_meas
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY hadm_id, obs_tsp, feature_name, hours_before
        ORDER BY meas_tsp DESC
    ) = 1
),
ranged AS (
    -- forward-fill: this value covers hours_before in (prev_hb, hb]  ->  slots [47-hb, 47-prev_hb-1]
    SELECT subject_id, hadm_id, obs_tsp, feature_name, value,
           hours_before AS hb,
           COALESCE(
               LAG(hours_before) OVER (
                   PARTITION BY hadm_id, obs_tsp, feature_name ORDER BY hours_before ASC
               ), -1
           ) AS prev_hb
    FROM deduped
)
SELECT subject_id, hadm_id, obs_tsp AS tsp, feature_name,
       ({SEQ_LEN - 1} - hb + g.value::INT) AS slot,   -- slot from (47-hb) .. (47-prev_hb-1)
       ranged.value AS value
FROM ranged,
     LATERAL FLATTEN(INPUT => ARRAY_GENERATE_RANGE(0, hb - prev_hb)) g
"""


def build_raw_pivot_sql() -> str:
    """Pivot the long slot table to flattened columns + attach label, weight, split.

    Labels/weights/split are computed on the narrow (hadm_id, tsp) key set first,
    then joined to the wide pivoted feature table - this avoids a GROUP BY over all
    1488 feature columns.
    """
    feat_cols = ",\n    ".join(
        f"MAX(CASE WHEN feature_name = '{f}' AND slot = {h} THEN value END) AS {flat_col(f, h)}"
        for f in FEATURE_NAMES for h in range(SEQ_LEN)
    )
    select_feats = ", ".join(f"f.{flat_col(f, h)}" for f in FEATURE_NAMES for h in range(SEQ_LEN))
    return f"""
CREATE OR REPLACE TEMPORARY TABLE {RAW_TABLE} AS
WITH keys AS (
    SELECT DISTINCT subject_id, hadm_id, tsp FROM {SLOT_TABLE}
),
icu_events AS (
    SELECT hadm_id, intime AS event_tsp FROM (
        SELECT hadm_id, intime,
               {_level_of_care_case()} AS level_of_care,
               LAG({_level_of_care_case()}) OVER (PARTITION BY hadm_id ORDER BY intime) AS prev_loc
        FROM {SRC_TRANSFERS}
    ) WHERE level_of_care = 'icu' AND prev_loc IN ({sql_str_list(list(ELIGIBLE_LEVELS))})
),
death_events AS (
    SELECT hadm_id, dischtime AS event_tsp FROM {SRC_ADMISSIONS}
    WHERE hospital_expire_flag = 1
),
events AS (SELECT * FROM icu_events UNION ALL SELECT * FROM death_events),
adm AS (SELECT hadm_id, dischtime FROM {SRC_ADMISSIONS}),
labeled AS (
    SELECT k.subject_id, k.hadm_id, k.tsp,
           MAX(CASE WHEN e.event_tsp IS NOT NULL THEN 1 ELSE 0 END) AS label,
           MIN(e.event_tsp) AS next_event_tsp
    FROM keys k
    LEFT JOIN events e
      ON k.hadm_id = e.hadm_id
     AND e.event_tsp BETWEEN k.tsp AND DATEADD('hour', {HORIZON_HOURS}, k.tsp)
    GROUP BY k.subject_id, k.hadm_id, k.tsp
),
prev AS (
    SELECT k.hadm_id, k.tsp, MAX(pe.event_tsp) AS prev_event_tsp
    FROM keys k
    LEFT JOIN events pe ON k.hadm_id = pe.hadm_id AND pe.event_tsp < k.tsp
    GROUP BY k.hadm_id, k.tsp
),
label_weight AS (
    SELECT l.subject_id, l.hadm_id, l.tsp, l.label,
           CASE WHEN l.next_event_tsp IS NOT NULL
                THEN DATEDIFF('second', l.tsp, l.next_event_tsp) / 60.0 END AS t2e_mins,
           DATEDIFF('second', l.tsp, a.dischtime) / 60.0 AS t2end_mins,
           CASE WHEN p.prev_event_tsp IS NOT NULL
                THEN DATEDIFF('second', p.prev_event_tsp, l.tsp) / 60.0 END AS mins_after_event
    FROM labeled l
    LEFT JOIN prev p ON l.hadm_id = p.hadm_id AND l.tsp = p.tsp
    LEFT JOIN adm a ON l.hadm_id = a.hadm_id
),
features AS (
    SELECT subject_id, hadm_id, tsp,
    {feat_cols}
    FROM {SLOT_TABLE}
    GROUP BY subject_id, hadm_id, tsp
)
SELECT f.subject_id, f.hadm_id, f.tsp, lw.label,
       {select_feats},
       CASE
           WHEN lw.mins_after_event IS NOT NULL AND lw.mins_after_event < {WASHOUT_HOURS * 60}
               THEN 0.0
           WHEN lw.t2e_mins IS NOT NULL
               THEN GREATEST(EXP(-{POS_DECAY_RATE} * GREATEST(lw.t2e_mins, -120)), {MIN_WEIGHT_FLOOR})
           ELSE GREATEST(EXP(-{NEG_DECAY_RATE} * GREATEST(lw.t2end_mins, -120)), {MIN_WEIGHT_FLOOR})
       END AS sample_weight,
       CASE WHEN ABS(HASH(f.hadm_id)) % 100 < {TRAIN_PCT} THEN 'train'
            WHEN ABS(HASH(f.hadm_id)) % 100 < {TRAIN_PCT + VAL_PCT} THEN 'val'
            ELSE 'test' END AS split
FROM features f
JOIN label_weight lw ON f.hadm_id = lw.hadm_id AND f.tsp = lw.tsp
"""


def build_scaler_sql() -> str:
    """Compute per-feature mean/std over the TRAIN split in a single pass.

    Stats are taken over the forward-filled (non-null) slot values for train
    admissions - matching the Databricks behavior of exploding the per-feature
    arrays and dropping NaNs before fitting the StandardScaler. The train filter
    uses the same hadm_id hash as the split assignment in the pivot step.
    """
    return f"""
CREATE OR REPLACE TABLE {SCALER_TABLE} AS
SELECT feature_name AS feature,
       AVG(value) AS mean_val,
       COALESCE(NULLIF(STDDEV(value), 0), 1.0) AS std_val
FROM {SLOT_TABLE}
WHERE value IS NOT NULL
  AND ABS(HASH(hadm_id)) % 100 < {TRAIN_PCT}
GROUP BY feature_name
"""


def build_normalized_sql(scaler: dict) -> str:
    """Apply (x-mean)/std per feature with NaN/NULL -> 0, write final clustered table."""
    norm_cols = []
    for f in FEATURE_NAMES:
        mean_val, std_val = scaler[f]
        std_val = std_val if std_val and std_val != 0 else 1.0
        for h in range(SEQ_LEN):
            c = flat_col(f, h)
            norm_cols.append(
                f"COALESCE(({c} - {mean_val}) / {std_val}, 0.0) AS {c}"
            )
    norm_sql = ",\n    ".join(norm_cols)
    return f"""
CREATE OR REPLACE TABLE {TS_TABLE}
CLUSTER BY (split) AS
SELECT subject_id, hadm_id, tsp, label, sample_weight, split,
    {norm_sql}
FROM {RAW_TABLE}
"""


def run(session, compute_scaler_in_python: bool = True):
    """Execute the full Phase 1 pipeline. `session` is a Snowpark Session."""
    print("Phase 1.1  building forward-filled slot table ...")
    session.sql(build_slots_sql()).collect()

    print("Phase 1.2  pivoting to flattened columns + labels + weights + split ...")
    session.sql(build_raw_pivot_sql()).collect()

    print("Phase 1.3  fitting StandardScaler on train split ...")
    session.sql(build_scaler_sql()).collect()
    scaler_rows = session.table(SCALER_TABLE).collect()
    scaler = {r["FEATURE"]: (r["MEAN_VAL"], r["STD_VAL"]) for r in scaler_rows}

    print("Phase 1.4  applying normalization -> final table ...")
    session.sql(build_normalized_sql(scaler)).collect()

    counts = session.sql(
        f"SELECT split, COUNT(*) n, SUM(label) pos, AVG(label) prevalence "
        f"FROM {TS_TABLE} GROUP BY split ORDER BY split"
    ).collect()
    print(f"Done. {TS_TABLE} written.")
    for r in counts:
        print(f"  {r['SPLIT']}: n={r['N']:,} pos={r['POS']} prevalence={r['PREVALENCE']:.4f}")
    return scaler


if __name__ == "__main__":
    from snowflake_utils import get_snowpark_session
    run(get_snowpark_session())
