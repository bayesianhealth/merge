import os
import json
import shutil
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from databricks_utils import get_spark, read_sql, CATALOG
import pipeline_utils as pu


# Number of distinct admissions (hadm_id) processed per batch. Bounds peak
# memory: the full filtered LABEVENTS table is ~43M rows, so we never load it
# all at once.
BATCH_SIZE = 40000

# Key columns carried through the time-series arrays (used for stable parquet
# schema across batches).
TS_KEYS = ['subject_id', 'hadm_id', 'stay_id', 'hosp_time_delta', 'icu_time_delta']


LAB_EVENT_LIST = ['Glucose', 'Potassium', 'Sodium', 'Chloride', 'Creatinine',
       'Urea Nitrogen', 'Bicarbonate', 'Anion Gap', 'Hemoglobin', 'Hematocrit',
       'Magnesium', 'Platelet Count', 'Phosphate', 'White Blood Cells',
       'Calcium, Total', 'MCH', 'Red Blood Cells', 'MCHC', 'MCV', 'RDW',
       'Neutrophils', 'Vancomycin']

VITAL_EVENT_LIST = ['Heart Rate','Non Invasive Blood Pressure systolic',
            'Non Invasive Blood Pressure diastolic', 'Non Invasive Blood Pressure mean',
            'Respiratory Rate','O2 saturation pulseoxymetry',
            'GCS - Verbal Response', 'GCS - Eye Opening', 'GCS - Motor Response']

VITAL_RENAME_DICT = {
    'Non Invasive Blood Pressure systolic': 'Systolic BP',
    'Non Invasive Blood Pressure diastolic': 'Diastolic BP',
    'Non Invasive Blood Pressure mean': 'Mean BP',
    'O2 saturation pulseoxymetry': 'O2 Saturation'
}


def _most_populated_itemid(spark, table, candidate_itemids):
    """Among itemids that share a D_*ITEMS label, return the one with the most rows
    in `table`.

    D_LABITEMS / D_ITEMS labels are NOT unique (e.g. 'Potassium' maps to itemids
    50971 [4.1M events], 52610 [1310], 50833 [0]). Blindly taking the first match
    can select a near-empty itemid, yielding an all-NaN feature column downstream.
    Selecting by event count picks the canonical, populated itemid.
    """
    candidate_itemids = [int(x) for x in candidate_itemids]
    if len(candidate_itemids) == 1:
        return candidate_itemids[0]
    ids = ','.join(str(x) for x in candidate_itemids)
    counts = read_sql(
        f"SELECT itemid, COUNT(*) AS n FROM {table} "
        f"WHERE itemid IN ({ids}) GROUP BY itemid ORDER BY n DESC", spark)
    if counts.empty:
        return candidate_itemids[0]
    return int(counts.iloc[0]['itemid'])


def get_lab_event_mapping(spark):
    """Return (event_id_df, d_lab_items_df). Small lookup only -- no event rows."""
    d_lab_items_df = read_sql(f'SELECT itemid, label FROM {CATALOG}.hosp.d_labitems', spark)
    d_lab_items_df = d_lab_items_df.dropna()

    rows = []
    for event in dict.fromkeys(LAB_EVENT_LIST):  # dedupe labels, preserve order
        candidates = d_lab_items_df[d_lab_items_df['label'] == event]['itemid'].tolist()
        if not candidates:
            raise ValueError(f"No itemid found in d_labitems for lab '{event}'")
        event_item_id = _most_populated_itemid(spark, f'{CATALOG}.hosp.labevents', candidates)
        rows.append({'itemid': event_item_id, 'event': event})

    event_id_df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    return event_id_df, d_lab_items_df


def get_vitals_event_mapping(spark):
    """Return (event_id_df, d_items_df). Small lookup only -- no event rows."""
    d_items_df = read_sql(f'SELECT itemid, label FROM {CATALOG}.icu.d_items', spark)

    rows = []
    for event in dict.fromkeys(VITAL_EVENT_LIST):  # dedupe labels, preserve order
        candidates = d_items_df[d_items_df['label'] == event]['itemid'].tolist()
        if not candidates:
            raise ValueError(f"No itemid found in d_items for vital '{event}'")
        event_item_id = _most_populated_itemid(spark, f'{CATALOG}.icu.chartevents', candidates)
        rows.append({'itemid': event_item_id, 'event': event})

    event_id_df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    return event_id_df, d_items_df


def _item_ids_str(event_id_df):
    return ','.join(str(int(x)) for x in event_id_df['itemid'].values)


def get_distinct_hadm_ids(spark, table, item_ids):
    """Distinct hadm_ids (as a set of ints) for the given itemids in a table."""
    query = f"""
        SELECT DISTINCT hadm_id
        FROM {table}
        WHERE itemid IN ({item_ids})
          AND hadm_id IS NOT NULL
    """
    df = read_sql(query, spark)
    return set(int(x) for x in pd.to_numeric(df['hadm_id'], errors='coerce').dropna().unique())


def _fetch_events(spark, select_cols, table, item_ids, hadm_batch):
    """Fetch events for a batch of hadm_ids. Spark has no IN-list size limit."""
    ids = ','.join(str(h) for h in hadm_batch)
    query = f"""
        SELECT {select_cols}
        FROM {table}
        WHERE itemid IN ({item_ids})
          AND hadm_id IN ({ids})
    """
    return read_sql(query, spark)


def fetch_lab_batch(spark, item_ids, hadm_batch, event_id_df):
    """Fetch + tag one hadm_id batch of labevents. Returns a DataFrame (may be empty)."""
    df = _fetch_events(
        spark, "subject_id, hadm_id, itemid, charttime, storetime, valuenum, valueuom",
        f"{CATALOG}.hosp.labevents", item_ids, hadm_batch)
    if df.empty:
        return df
    df['hadm_id'] = pd.to_numeric(df['hadm_id'], errors='coerce')
    df = df.dropna(subset=['hadm_id'])
    df['hadm_id'] = df['hadm_id'].astype('int64')
    # Databricks stores timestamps natively — no epoch conversion needed
    df['charttime'] = pd.to_datetime(df['charttime'])
    df['storetime'] = pd.to_datetime(df['storetime'])
    df = df.merge(event_id_df, on='itemid', how='left')
    return df


def fetch_vitals_batch(spark, item_ids, hadm_batch, event_id_df):
    """Fetch + tag one hadm_id batch of chartevents. Returns a DataFrame (may be empty)."""
    df = _fetch_events(
        spark, "subject_id, hadm_id, stay_id, itemid, charttime, storetime, valuenum, valueuom",
        f"{CATALOG}.icu.chartevents", item_ids, hadm_batch)
    if df.empty:
        return df
    df['hadm_id'] = pd.to_numeric(df['hadm_id'], errors='coerce')
    df = df.dropna(subset=['hadm_id'])
    df['hadm_id'] = df['hadm_id'].astype('int64')
    df['charttime'] = pd.to_datetime(df['charttime'])
    df['storetime'] = pd.to_datetime(df['storetime'])
    df = df.merge(event_id_df, on='itemid', how='left')
    df['event'] = df['event'].replace(VITAL_RENAME_DICT)
    return df

def add_time_delta_vectorized(df, admissions_df, icustays_df):
    """
    Add event time with respect to hospital admission time and ICU stay time.
    Args:
        df: labevents or chartevents DataFrame
        admissions_df: admissions DataFrame with hospital admission time
        icustays_df: icustays DataFrame with ICU stay time
    Returns:
        df: DataFrame with event times with respect to hospital admission time and ICU stay time
    """

    df = df.copy()
    
    # Determine reference time column
    if 'charttime' in df.columns:
        ref_time_col = 'charttime'
    elif 'storetime' in df.columns:
        ref_time_col = 'storetime'
    else:
        raise ValueError('DataFrame must contain either charttime or storetime column')
    
    # Check if stay_id exists
    stay_id_in_cols = 'stay_id' in df.columns
    
    # Merge with admissions to get hospital admission times
    admission_cols = ['subject_id', 'hadm_id', 'admittime']
    df = df.merge(admissions_df[admission_cols], on=['subject_id', 'hadm_id'], how='left')
    
    # Calculate hospital time delta vectorized
    df['hosp_time_delta'] = (df[ref_time_col] - df['admittime']).dt.total_seconds() / 3600
    
    # Handle ICU time delta
    if stay_id_in_cols:
        # If stay_id exists, merge directly with ICU stays
        icu_cols = ['subject_id', 'stay_id', 'intime']
        df = df.merge(icustays_df[icu_cols], on=['subject_id', 'stay_id'], how='left')
        df['icu_time_delta'] = (df[ref_time_col] - df['intime']).dt.total_seconds() / 3600
    else:
        # If no stay_id, need to find which ICU stay each event belongs to
        
        # Prepare ICU stays data
        icu_stays = icustays_df[['subject_id', 'stay_id', 'intime', 'outtime']].copy()
        
        # Create a cartesian product of events and ICU stays for the same subject
        df_with_idx = df.reset_index().rename(columns={'index': 'original_idx'})
        merged = df_with_idx.merge(icu_stays, on='subject_id', how='left')
        
        # Filter to events that fall within ICU stay time windows
        mask = (merged[ref_time_col] >= merged['intime']) & (merged[ref_time_col] <= merged['outtime'])
        valid_matches = merged[mask].copy()
        
        # Calculate ICU time delta for valid matches
        valid_matches['icu_time_delta'] = (valid_matches[ref_time_col] - valid_matches['intime']).dt.total_seconds() / 3600
        
        # Handle multiple ICU stays for same event (take the first match)
        # Sort by intime to ensure consistent selection
        valid_matches = valid_matches.sort_values(['original_idx', 'intime'])
        valid_matches = valid_matches.drop_duplicates('original_idx', keep='first')
        # Merge back the stay_id and icu_time_delta
        if not valid_matches.empty:
            df = df_with_idx.merge(
                valid_matches[['original_idx', 'stay_id', 'icu_time_delta']], 
                on='original_idx', 
                how='left'
            )
            df = df.drop(['original_idx'], axis=1)
        else:
            # If no valid matches, just add stay_id and icu_time_delta columns with None
            df = df_with_idx
            df['stay_id'] = None
            df['icu_time_delta'] = None
            df = df.drop(['original_idx'], axis=1)
    
    # Clean up temporary columns
    df = df.drop(['admittime'], axis=1, errors='ignore')
    if 'intime' in df.columns:
        df = df.drop(['intime'], axis=1)
    
    # Sort the DataFrame
    df = df.sort_values(by=['subject_id', 'hadm_id', 'stay_id', 'hosp_time_delta'])
    
    return df

def convert_events_table_to_ts_array(df):
    # Ensure 'valuenum' or 'value' columns exist
    value_column = 'valuenum' if 'valuenum' in df.columns else 'value'

    # Create a pivot table
    pivot_df = df.pivot_table(index=['hadm_id', 'hosp_time_delta'], 
                              columns='event', 
                              values=value_column, 
                              aggfunc='first').reset_index()

    # Join with the original DataFrame to get other required columns
    keys = ['subject_id', 'hadm_id', 'stay_id', 'hosp_time_delta', 'icu_time_delta']
    merged_df = pd.merge(df[keys].drop_duplicates(), pivot_df, on=['hadm_id', 'hosp_time_delta'])

    # Reorder the columns
    cols = merged_df.columns.tolist()
    cols = [col for col in keys if col in cols] + [col for col in cols if col not in keys]
    merged_df = merged_df[cols]

    # Sort the DataFrame
    merged_df.sort_values(by=['subject_id', 'hadm_id', 'stay_id', 'hosp_time_delta'], inplace=True)

    return merged_df

def create_event_uom_map(df):
    """
    Create a pd.Dataframe with event, itemid, and valueuom from labevents_df or vitals_df
    """
    df = df.copy()
    df = df[['event', 'itemid', 'valueuom']]
    df.drop_duplicates(inplace=True)
    return df


def _finalize_ts(ts, event_cols):
    """
    Reindex a per-batch time-series array to a fixed column set and stable dtypes
    so that every batch produces an identical parquet schema (required for
    incremental ParquetWriter appends).
    """
    ts = ts.reindex(columns=TS_KEYS + event_cols)
    ts['subject_id'] = ts['subject_id'].astype('int64')
    ts['hadm_id'] = ts['hadm_id'].astype('int64')
    for col in ['stay_id', 'hosp_time_delta', 'icu_time_delta'] + event_cols:
        ts[col] = pd.to_numeric(ts[col], errors='coerce').astype('float64')
    return ts


# ---------------------------------------------------------------------------
# Batch-level resume: each hadm_id batch writes part files under
# <output_dir>/_step1_parts/<subset>/bNNNNN.parquet and records its index in
# completed.txt only after all of its parts are safely written. An interrupted
# run resumes from the first unfinished batch; finished batches are never redone.
# ---------------------------------------------------------------------------

def _atomic_parquet(df, path):
    pu.atomic_to_parquet(df, path, index=False)


def _reset_parts(parts_dir):
    if os.path.isdir(parts_dir):
        shutil.rmtree(parts_dir)
    os.makedirs(parts_dir, exist_ok=True)


def _load_completed(parts_dir):
    p = os.path.join(parts_dir, "completed.txt")
    if not os.path.exists(p):
        return set()
    with open(p) as f:
        return set(int(x) for x in f.read().split() if x.strip())


def _mark_completed(parts_dir, completed_set, idx):
    """Record a finished batch. Rewrites completed.txt atomically (append mode is
    unsupported on some stage-backed filesystems)."""
    completed_set.add(idx)
    data = "\n".join(str(i) for i in sorted(completed_set)).encode()
    pu._atomic_write(os.path.join(parts_dir, "completed.txt"), data)


def _load_or_build_plan(spark, parts_dir, lab_item_ids, vit_item_ids, batch_size):
    plan_path = os.path.join(parts_dir, "plan.json")
    if os.path.exists(plan_path):
        with open(plan_path) as f:
            plan = json.load(f)
        if plan.get("batch_size") == batch_size:
            print(f'Resuming from existing plan: {len(plan["batches"])} batches.')
            return plan
        print("Batch size changed since last run; rebuilding plan and clearing partial progress.")
        _reset_parts(parts_dir)

    print('Collecting distinct hadm_ids...')
    lab_hadm = get_distinct_hadm_ids(spark, f'{CATALOG}.hosp.labevents', lab_item_ids)
    vit_hadm = get_distinct_hadm_ids(spark, f'{CATALOG}.icu.chartevents', vit_item_ids)
    all_hadm = sorted(lab_hadm | vit_hadm)
    batches = [all_hadm[i:i + batch_size] for i in range(0, len(all_hadm), batch_size)]
    plan = {"batch_size": batch_size, "n_hadm": len(all_hadm), "batches": batches}
    pu._atomic_write(plan_path, json.dumps(plan).encode())
    return plan


def _combine_ts(parts_dir, subset, full_cols, out_path):
    """Stream-concat all part files for a subset into one parquet (bounded memory)."""
    sub_dir = os.path.join(parts_dir, subset)
    writer = None
    tmp_path = out_path + ".tmp"
    if os.path.isdir(sub_dir):
        for fname in sorted(os.listdir(sub_dir)):
            if not fname.endswith(".parquet"):
                continue
            table = pq.read_table(os.path.join(sub_dir, fname))
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table)
    if writer is not None:
        writer.close()
        os.replace(tmp_path, out_path)
    else:
        # No batch produced rows for this subset -- emit an empty, typed file.
        pu.atomic_to_parquet(pd.DataFrame(columns=full_cols), out_path, index=False)


def _combine_uom(parts_dir, subset, out_path):
    sub_dir = os.path.join(parts_dir, subset)
    frames = []
    if os.path.isdir(sub_dir):
        for fname in sorted(os.listdir(sub_dir)):
            if fname.endswith(".parquet"):
                frames.append(pd.read_parquet(os.path.join(sub_dir, fname)))
    if frames:
        df = pd.concat(frames, axis=0, ignore_index=True).drop_duplicates()
    else:
        df = pd.DataFrame(columns=['event', 'itemid', 'valueuom'])
    pu.atomic_to_parquet(df, out_path, index=False)


def main(args):
    out = args.output_dir
    if not args.force and pu.is_done(out, pu.STEP1_IRG):
        print("Step 1 already complete (marker present); skipping. Use --force to redo.")
        return

    parts_dir = os.path.join(out, "_step1_parts")
    if args.force:
        _reset_parts(parts_dir)
        pu.clear_marker(out, pu.STEP1_IRG)
    os.makedirs(parts_dir, exist_ok=True)

    spark = get_spark()

    # small reference tables, kept in memory for the per-batch time-delta joins
    print('Loading admissions table...')
    admissions_df = read_sql(f"SELECT * FROM {CATALOG}.hosp.admissions", spark)
    admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])
    admissions_df['dischtime'] = pd.to_datetime(admissions_df['dischtime'])

    print('Loading icustays table...')
    icustays_df = read_sql(f"SELECT * FROM {CATALOG}.icu.icustays", spark)
    icustays_df['intime'] = pd.to_datetime(icustays_df['intime'])
    icustays_df['outtime'] = pd.to_datetime(icustays_df['outtime'])

    print('Building lab/vitals event mappings...')
    lab_map_df, _ = get_lab_event_mapping(spark)
    vit_map_df, _ = get_vitals_event_mapping(spark)
    lab_item_ids = _item_ids_str(lab_map_df)
    vit_item_ids = _item_ids_str(vit_map_df)

    # fixed event-column sets => stable parquet schema across all batches
    lab_event_cols = sorted(set(lab_map_df['event']))
    vit_event_cols = sorted(set(pd.Series(vit_map_df['event'].unique()).replace(VITAL_RENAME_DICT)))
    concat_event_cols = sorted(set(lab_event_cols) | set(vit_event_cols))

    plan = _load_or_build_plan(spark, parts_dir, lab_item_ids, vit_item_ids, args.batch_size)
    batches = plan["batches"]
    n_batches = len(batches)
    completed = _load_completed(parts_dir)
    print(f'{plan["n_hadm"]} distinct admissions; {n_batches} batch(es) of {args.batch_size}; '
          f'{len(completed)} already complete.')

    for idx, batch in enumerate(batches):
        if idx in completed:
            continue
        print(f'Batch {idx + 1}/{n_batches} ({len(batch)} admissions)...')
        bname = f"b{idx:05d}.parquet"

        labs_b = fetch_lab_batch(spark, lab_item_ids, batch, lab_map_df)
        if not labs_b.empty:
            labs_b = add_time_delta_vectorized(labs_b, admissions_df, icustays_df)
            _atomic_parquet(_finalize_ts(convert_events_table_to_ts_array(labs_b), lab_event_cols),
                            os.path.join(parts_dir, "ts_labs_icu", bname))
            _atomic_parquet(create_event_uom_map(labs_b),
                            os.path.join(parts_dir, "uom_labs_icu", bname))

        vitals_b = fetch_vitals_batch(spark, vit_item_ids, batch, vit_map_df)
        if not vitals_b.empty:
            vitals_b = add_time_delta_vectorized(vitals_b, admissions_df, icustays_df)
            _atomic_parquet(_finalize_ts(convert_events_table_to_ts_array(vitals_b), vit_event_cols),
                            os.path.join(parts_dir, "ts_vitals_icu", bname))
            _atomic_parquet(create_event_uom_map(vitals_b),
                            os.path.join(parts_dir, "uom_vitals_icu", bname))

        parts = [d for d in (labs_b, vitals_b) if not d.empty]
        if parts:
            concat_b = pd.concat(parts, axis=0, ignore_index=True)
            _atomic_parquet(_finalize_ts(convert_events_table_to_ts_array(concat_b), concat_event_cols),
                            os.path.join(parts_dir, "ts_labs_vitals", bname))
            _atomic_parquet(create_event_uom_map(concat_b),
                            os.path.join(parts_dir, "uom_labs_vitals", bname))
            del concat_b
        del labs_b, vitals_b

        # record progress only after all of this batch's parts are durably written
        _mark_completed(parts_dir, completed, idx)

    print('All batches done; combining part files into final outputs...')
    _combine_ts(parts_dir, "ts_labs_icu", TS_KEYS + lab_event_cols, os.path.join(out, "ts_labs_icu.parquet"))
    _combine_ts(parts_dir, "ts_vitals_icu", TS_KEYS + vit_event_cols, os.path.join(out, "ts_vitals_icu.parquet"))
    _combine_ts(parts_dir, "ts_labs_vitals", TS_KEYS + concat_event_cols, os.path.join(out, "ts_labs_vitals.parquet"))
    _combine_uom(parts_dir, "uom_labs_icu", os.path.join(out, "uom_labs_icu.parquet"))
    _combine_uom(parts_dir, "uom_vitals_icu", os.path.join(out, "uom_vitals_icu.parquet"))
    _combine_uom(parts_dir, "uom_labs_vitals", os.path.join(out, "uom_labs_vitals.parquet"))

    pu.write_marker(out, pu.STEP1_IRG, {
        "n_hadm": plan["n_hadm"], "n_batches": n_batches, "batch_size": args.batch_size,
        "outputs": ["ts_labs_icu.parquet", "ts_vitals_icu.parquet", "ts_labs_vitals.parquet",
                    "uom_labs_icu.parquet", "uom_vitals_icu.parquet", "uom_labs_vitals.parquet"],
    })

    if not args.keep_parts:
        shutil.rmtree(parts_dir, ignore_errors=True)

    print('Done. Output saved to:', out)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help='Path to output directory', default='data')
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help='Number of admissions (hadm_id) processed per batch')
    parser.add_argument("--force", action='store_true',
                        help='Ignore completion marker and partial progress; redo from scratch')
    parser.add_argument("--keep_parts", action='store_true',
                        help='Keep the intermediate _step1_parts/ directory after combining')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
