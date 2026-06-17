import os
import gzip
import argparse
import tempfile
import pandas as pd
from snowflake.snowpark import Session
from snowflake_utils import get_connection, read_sql, get_snowpark_session
import pipeline_utils as pu


def _epoch_to_datetime(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit='s')
    return df

def add_time_delta_notes_vectorized(notes_df, admissions_df, icustays_df):
    """
    Add event time with respect to hospital admission time and ICU stay time to the notes dataframe.
    """
    df = notes_df.copy()
    
    # Initialize new columns
    df['hadm_id'] = None
    df['stay_id'] = None
    
    # Step 1: Handle hospital admissions
    # Create cartesian product of notes and admissions for the same subject
    df_with_idx = df.reset_index().rename(columns={'index': 'original_idx'})
    
    # Merge with admissions data
    admissions_subset = admissions_df[['subject_id', 'hadm_id', 'admittime', 'dischtime']].copy()
    merged_adm = df_with_idx.merge(admissions_subset, on='subject_id', how='left', suffixes=('_orig', '_adm'))
    
    # Filter to notes that fall within admission time windows
    mask_adm = (merged_adm['charttime'] >= merged_adm['admittime']) & (merged_adm['charttime'] <= merged_adm['dischtime'])
    valid_adm_matches = merged_adm[mask_adm].copy()
    
    if not valid_adm_matches.empty:
        # Calculate hospital time delta for valid matches
        valid_adm_matches['hosp_time_delta'] = (valid_adm_matches['charttime'] - valid_adm_matches['admittime']).dt.total_seconds() / 3600
        
        # Handle multiple admissions for same note (take the first match by admit time)
        valid_adm_matches = valid_adm_matches.sort_values(['original_idx', 'admittime'])
        valid_adm_matches = valid_adm_matches.drop_duplicates('original_idx', keep='first')
        
        # Merge back the hadm_id and hosp_time_delta
        df = df_with_idx.merge(
            valid_adm_matches[['original_idx', 'hadm_id_adm', 'hosp_time_delta']], 
            on='original_idx', 
            how='left'
        )
        df['hadm_id'] = df['hadm_id_adm']
        df = df.drop(['hadm_id_adm'], axis=1)
    else:
        df = df_with_idx
    
    # Step 2: Handle ICU stays
    # Create cartesian product of notes and ICU stays for the same subject
    icu_stays_subset = icustays_df[['subject_id', 'stay_id', 'intime', 'outtime']].copy()
    merged_icu = df.merge(icu_stays_subset, on='subject_id', how='left', suffixes=('_orig', '_icu'))
    
    # Filter to notes that fall within ICU stay time windows
    mask_icu = (merged_icu['charttime'] >= merged_icu['intime']) & (merged_icu['charttime'] <= merged_icu['outtime'])
    valid_icu_matches = merged_icu[mask_icu].copy()
    
    if not valid_icu_matches.empty:
        # Calculate ICU time delta for valid matches
        valid_icu_matches['icu_time_delta'] = (valid_icu_matches['charttime'] - valid_icu_matches['intime']).dt.total_seconds() / 3600
        
        # Handle multiple ICU stays for same note (take the first match by intime)
        valid_icu_matches = valid_icu_matches.sort_values(['original_idx', 'intime'])
        valid_icu_matches = valid_icu_matches.drop_duplicates('original_idx', keep='first')
        
        # Update the stay_id and icu_time_delta columns
        df = df.merge(valid_icu_matches[['original_idx', 'stay_id_icu', 'icu_time_delta']], on='original_idx', how='left')
        df['stay_id'] = df['stay_id_icu']
        df = df.drop(['stay_id_icu'], axis=1)


    # Clean up temporary columns
    df = df.drop(['original_idx'], axis=1, errors='ignore')
    # Sort the DataFrame
    df = df.sort_values(by=['subject_id', 'hadm_id', 'hosp_time_delta', 'stay_id', 'icu_time_delta'])
    
    return df


def main(args):
    if not args.force and pu.is_done(args.output_dir, pu.STEP3_NOTES):
        print("Step 3 already complete (marker present); skipping. Use --force to redo.")
        return

    conn = get_connection()
    session = get_snowpark_session(conn)

    print('Downloading radiology notes from stage...')
    stage_path = '@"TEST"."SILVER"."UDTF_FEATURE_STAGE"/radiology.csv.gz'
    local_dir = tempfile.mkdtemp()
    session.file.get(stage_path, local_dir)
    local_gz_path = os.path.join(local_dir, "radiology.csv.gz")

    print('Decompressing and loading radiology notes...')
    with gzip.open(local_gz_path, 'rt') as f:
        rad_notes_df = pd.read_csv(f)
    rad_notes_df['charttime'] = pd.to_datetime(rad_notes_df['charttime'])
    rad_notes_df['storetime'] = pd.to_datetime(rad_notes_df['storetime'])

    print('Loading icustays...')
    icustays_df = read_sql("SELECT * FROM MIMICIV.ICU.ICUSTAYS", conn)
    icustays_df = _epoch_to_datetime(icustays_df, ['intime', 'outtime'])

    print('Loading admissions...')
    admissions_df = read_sql("SELECT * FROM MIMICIV.HOSP.ADMISSIONS", conn)
    admissions_df = _epoch_to_datetime(admissions_df, ['admittime', 'dischtime'])

    conn.close()

    print('Adding time delta...')
    rad_notes_df = add_time_delta_notes_vectorized(rad_notes_df, admissions_df, icustays_df)

    print('Saving radiology notes...')
    pu.atomic_to_parquet(rad_notes_df, os.path.join(args.output_dir, "rad_notes_text.parquet"))
    pu.write_marker(args.output_dir, pu.STEP3_NOTES,
                    {"rows": int(len(rad_notes_df)), "output": "rad_notes_text.parquet"})



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help='Path to output directory', default='data')
    parser.add_argument("--notes_file_path", type=str, default=None, help='Path to radiology notes CSV (informational)')
    parser.add_argument("--force", action='store_true', help='Ignore completion marker and redo')
    args = parser.parse_args()
    main(args)