import os
import time

import numpy as np

from deterioration_config import (
    OUT_SCHEMA, OUT_DB, OUT_SCHEMA_NAME, RADIOLOGY_TABLE, TS_TABLE, NOTES_TABLE,
    BIOBERT_STAGE_DIR, NOTE_EMB_DIM, LOOKBACK_HOURS,
    ICU_UNITS, STEPDOWN_UNITS, GENERAL_UNITS, OBSERVATION_UNITS,
    ELIGIBLE_LEVELS, SRC_TRANSFERS, sql_str_list,
)

RAW_NOTES_TABLE = f"{OUT_SCHEMA}.DETERIORATION_NOTE_EMBEDDINGS_RAW"
LOCAL_MODEL_DIR = "/tmp/biobert-v1.1"
MAX_TOKENS = 512

def download_model(session, stage_dir: str = BIOBERT_STAGE_DIR, local_dir: str = LOCAL_MODEL_DIR):
    """Pull the BioBERT files from the stage to local disk for transformers."""
    os.makedirs(local_dir, exist_ok=True)
    # session.file.get downloads every file under the stage directory.
    session.file.get(stage_dir, f"file://{local_dir}")
    # session.file.get nests the stage subfolder; flatten if needed.
    nested = os.path.join(local_dir, "biobert-v1.1")
    if os.path.isdir(nested):
        for fn in os.listdir(nested):
            os.replace(os.path.join(nested, fn), os.path.join(local_dir, fn))
    print(f"BioBERT files in {local_dir}: {sorted(os.listdir(local_dir))}")
    return local_dir


def load_biobert(local_dir: str = LOCAL_MODEL_DIR):
    """Load tokenizer + model onto GPU (eval mode)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModel.from_pretrained(local_dir).to(device).eval()
    print(f"BioBERT loaded on {device}")
    return tokenizer, model, device


def _embed_texts(texts, tokenizer, model, device, gpu_batch: int):
    """Return a list of 768-d float32 vectors, mean-pooled over 512-token chunks.

    Each note is split into consecutive 512-token chunks; we mean-pool the CLS
    embedding across chunks so arbitrarily long notes map to a single vector
    (matches the Databricks `biobert_embeddings` -> mean aggregation).
    """
    import torch

    out = []
    # Tokenize each note independently to know its chunk boundaries.
    chunk_texts, chunk_owner = [], []
    for i, txt in enumerate(texts):
        if not txt:
            continue
        ids = tokenizer.encode(txt, add_special_tokens=True, truncation=False)
        # Split into <=MAX_TOKENS chunks (keep CLS/SEP per chunk via re-decoding window).
        body = ids[1:-1] if len(ids) > 2 else ids
        step = MAX_TOKENS - 2
        for start in range(0, max(len(body), 1), step):
            window = body[start:start + step]
            chunk_ids = [tokenizer.cls_token_id] + window + [tokenizer.sep_token_id]
            chunk_texts.append(chunk_ids)
            chunk_owner.append(i)

    # Per-note running sum + count for mean pooling.
    sums = np.zeros((len(texts), NOTE_EMB_DIM), dtype=np.float64)
    counts = np.zeros(len(texts), dtype=np.int64)

    with torch.no_grad():
        for b in range(0, len(chunk_texts), gpu_batch):
            batch_ids = chunk_texts[b:b + gpu_batch]
            owners = chunk_owner[b:b + gpu_batch]
            maxlen = max(len(x) for x in batch_ids)
            input_ids = torch.zeros((len(batch_ids), maxlen), dtype=torch.long)
            attn = torch.zeros((len(batch_ids), maxlen), dtype=torch.long)
            for r, ids in enumerate(batch_ids):
                input_ids[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                attn[r, :len(ids)] = 1
            input_ids, attn = input_ids.to(device), attn.to(device)
            cls = model(input_ids=input_ids, attention_mask=attn).last_hidden_state[:, 0, :]
            cls = cls.float().cpu().numpy()
            for r, owner in enumerate(owners):
                sums[owner] += cls[r]
                counts[owner] += 1

    for i in range(len(texts)):
        if counts[i] > 0:
            out.append((sums[i] / counts[i]).astype(np.float32).tolist())
        else:
            out.append([0.0] * NOTE_EMB_DIM)
    return out


def _eligible_notes_query() -> str:
    """Radiology notes for admissions present in the training table, with text.

    Carries note_id (stable PK) instead of charttime: timestamps are sourced from
    RADIOLOGY at join time to avoid corrupting them through a pandas/write_pandas
    datetime round-trip.
    """
    return f"""
SELECT r.note_id, r.hadm_id, r.text
FROM {RADIOLOGY_TABLE} r
WHERE r.text IS NOT NULL
  AND r.hadm_id IS NOT NULL
  AND r.hadm_id IN (SELECT DISTINCT hadm_id FROM {TS_TABLE})
"""


def build_raw_embeddings(session, batch_rows: int = 20000, gpu_batch: int = 64):
    """Embed every eligible radiology note and write the raw per-note table."""
    import pandas as pd

    tokenizer, model, device = load_biobert()

    session.sql(f"""
        CREATE OR REPLACE TABLE {RAW_NOTES_TABLE} (
            note_id VARCHAR,
            hadm_id NUMBER(38,0),
            note_embedding ARRAY
        )
    """).collect()

    src = session.sql(_eligible_notes_query())
    total = src.count()
    print(f"Embedding {total:,} radiology notes in batches of {batch_rows} ...")

    processed = 0
    # to_local_iterator streams partitions without materializing everything in memory.
    buf_id, buf_h, buf_x = [], [], []

    def flush():
        nonlocal processed, buf_id, buf_h, buf_x
        if not buf_id:
            return
        embs = _embed_texts(buf_x, tokenizer, model, device, gpu_batch)
        out = pd.DataFrame({
            "NOTE_ID": buf_id,
            "HADM_ID": pd.array(buf_h, dtype="Int64"),
            "NOTE_EMBEDDING": embs,
        })
        session.write_pandas(
            out, RAW_NOTES_TABLE.split(".")[-1],
            database=OUT_DB, schema=OUT_SCHEMA_NAME,
            auto_create_table=False, overwrite=False, quote_identifiers=False,
        )
        processed += len(out)
        print(f"  wrote {len(out)} (total {processed:,}) at {time.strftime('%X')}")
        buf_id, buf_h, buf_x = [], [], []

    for row in src.to_local_iterator():
        buf_id.append(row["NOTE_ID"])
        buf_h.append(row["HADM_ID"])
        buf_x.append(row["TEXT"])
        if len(buf_id) >= batch_rows:
            flush()
    flush()

    print(f"Raw note embeddings written to {RAW_NOTES_TABLE}: {processed:,} rows")
    return processed


def _level_of_care_case(col: str = "careunit") -> str:
    return (
        "CASE "
        f"WHEN {col} IN ({sql_str_list(ICU_UNITS)}) THEN 'icu' "
        f"WHEN {col} IN ({sql_str_list(STEPDOWN_UNITS)}) THEN 'stepdown' "
        f"WHEN {col} IN ({sql_str_list(GENERAL_UNITS)}) THEN 'general' "
        f"WHEN {col} IN ({sql_str_list(OBSERVATION_UNITS)}) THEN 'observation' "
        "ELSE 'other' END"
    )


def build_obs_window_notes(session):
    """Join raw note embeddings to (hadm_id, obs_tsp) 48h windows on eligible wards.

    Produces the table the training dataloader reads:
        hadm_id, obs_tsp, note_charttime, hours_before_obs, note_embedding
    """
    elig = sql_str_list(list(ELIGIBLE_LEVELS))
    session.sql(f"""
CREATE OR REPLACE TABLE {NOTES_TABLE} AS
WITH ward_stays AS (
    SELECT * FROM (
        SELECT hadm_id, intime, outtime, {_level_of_care_case()} AS level_of_care
        FROM {SRC_TRANSFERS}
    ) WHERE level_of_care IN ({elig})
),
obs AS (
    SELECT DISTINCT hadm_id, tsp AS obs_tsp FROM {TS_TABLE}
),
note_emb AS (
    -- attach the CORRECT charttime from RADIOLOGY via the stable note_id
    SELECT n.hadm_id, rad.charttime, n.note_embedding
    FROM {RAW_NOTES_TABLE} n
    JOIN {RADIOLOGY_TABLE} rad ON n.note_id = rad.note_id
),
joined AS (
    SELECT o.hadm_id, o.obs_tsp, n.charttime AS note_charttime,
           FLOOR(DATEDIFF('second', n.charttime, o.obs_tsp) / 3600)::INT AS hours_before_obs,
           n.note_embedding
    FROM obs o
    JOIN note_emb n
      ON o.hadm_id = n.hadm_id
     AND n.charttime BETWEEN DATEADD('hour', -{LOOKBACK_HOURS}, o.obs_tsp) AND o.obs_tsp
)
SELECT j.hadm_id, j.obs_tsp, j.note_charttime, j.hours_before_obs, j.note_embedding
FROM joined j
JOIN ward_stays w
  ON j.hadm_id = w.hadm_id
 AND j.note_charttime BETWEEN w.intime AND w.outtime
""").collect()

    stats = session.sql(f"""
        SELECT COUNT(*) AS n_notes,
               COUNT(DISTINCT hadm_id || '|' || obs_tsp) AS obs_with_notes
        FROM {NOTES_TABLE}
    """).collect()[0]
    total_obs = session.sql(f"SELECT COUNT(DISTINCT hadm_id || '|' || tsp) c FROM {TS_TABLE}").collect()[0]["C"]
    cov = stats["OBS_WITH_NOTES"] / max(total_obs, 1) * 100
    print(f"{NOTES_TABLE}: {stats['N_NOTES']:,} note rows, "
          f"{stats['OBS_WITH_NOTES']:,}/{total_obs:,} obs have >=1 note ({cov:.1f}%)")


def run(session, batch_rows: int = 20000, gpu_batch: int = 64, skip_embedding: bool = False):
    """Full Phase 2: download model -> embed notes -> obs-window join."""
    if not skip_embedding:
        download_model(session)
        build_raw_embeddings(session, batch_rows=batch_rows, gpu_batch=gpu_batch)
    build_obs_window_notes(session)
    print("Phase 2 complete.")


if __name__ == "__main__":
    from snowflake_utils import get_snowpark_session
    run(get_snowpark_session())
