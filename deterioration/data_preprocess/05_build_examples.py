import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import det_constants as C
import pipeline_utils as pu
from snowflake_utils import get_connection, get_snowpark_session, run_sql


def load_scaler(conn):
    rows = run_sql(
        f"SELECT feature_name, mean, std FROM {C.T_FEATURE_SPEC}", conn, fetch=True
    )
    return {fid: (float(m), float(s)) for fid, m, s in rows}


def build_input_df(session, limit=None):
    """Union obs / anchor / event rows into one enc_id-keyed Snowpark DataFrame.

    If limit is set, restrict to a deterministic subset of encounters (same set across
    all three sources) for a quick trial run.
    """
    fid_list = ", ".join(f"'{f}'" for f in C.FEATURE_FIDS)
    # deterministic, identical enc subset across the three sources (avoids a
    # Snowpark self-join, which mangles duplicate ENC_ID column names)
    enc_filter = ""
    if limit:
        enc_filter = (
            f"enc_id IN (SELECT enc_id FROM {C.T_ENC_SPLIT} "
            f"ORDER BY enc_id LIMIT {int(limit)})"
        )

    obs = session.sql(f"""
        SELECT t.enc_id::STRING AS ENC_ID, 'OBS' AS KIND, t.fid AS FID,
               t.tsp AS TSP, TRY_TO_DOUBLE(t.value) AS VAL,
               CAST(NULL AS STRING) AS OUTCOME, CAST(NULL AS STRING) AS SPLIT,
               CAST(NULL AS TIMESTAMP_NTZ) AS ELIGIBLE_START
        FROM {C.T_CDM_T} t
        JOIN {C.T_ENC_SPLIT} s ON s.enc_id = t.enc_id
        WHERE t.fid IN ({fid_list})
          AND TRY_TO_DOUBLE(t.value) IS NOT NULL
          AND {C.not_deleted('t')}
          {f'AND t.{enc_filter}' if enc_filter else ''}
    """)
    anchors = session.sql(f"""
        SELECT enc_id::STRING AS ENC_ID, 'ANCHOR' AS KIND, CAST(NULL AS STRING) AS FID,
               anchor_tsp AS TSP, CAST(NULL AS DOUBLE) AS VAL,
               CAST(NULL AS STRING) AS OUTCOME, split AS SPLIT,
               eligible_start AS ELIGIBLE_START
        FROM {C.T_ANCHORS}
        {f'WHERE {enc_filter}' if enc_filter else ''}
    """)
    events = session.sql(f"""
        SELECT enc_id::STRING AS ENC_ID, 'EVENT' AS KIND, CAST(NULL AS STRING) AS FID,
               event_tsp AS TSP, CAST(NULL AS DOUBLE) AS VAL,
               outcome AS OUTCOME, CAST(NULL AS STRING) AS SPLIT,
               CAST(NULL AS TIMESTAMP_NTZ) AS ELIGIBLE_START
        FROM {C.T_EVENTS}
        {f'WHERE {enc_filter}' if enc_filter else ''}
    """)
    return obs.union_all(anchors).union_all(events)


def make_output_struct():
    from snowflake.snowpark.types import (
        StructType, StructField, StringType, IntegerType, FloatType,
        TimestampType, ArrayType,
    )
    fields = [
        StructField("ENC_ID", StringType()),
        StructField("ANCHOR_TSP", TimestampType()),
        StructField("SPLIT", StringType()),
        StructField("LABEL", IntegerType()),
        StructField("SAMPLE_WEIGHT", FloatType()),
        StructField("EVENT_TSP", TimestampType()),
        StructField("OUTCOME", StringType()),
    ]
    for fid in C.FEATURE_FIDS:
        fields.append(StructField(C.col_value(fid), ArrayType(FloatType())))
        fields.append(StructField(C.col_mask(fid), ArrayType(IntegerType())))
    return StructType(fields)


def make_processor(scaler):
    """Return the per-encounter pandas function (closure captures scaler + config)."""
    import numpy as np
    import pandas as pd

    feature_fids = C.FEATURE_FIDS
    seq_len = C.SEQ_LEN
    horizon = np.timedelta64(C.HORIZON_HOURS, "h")
    washout = pd.Timedelta(hours=C.WASHOUT_HOURS)
    min_obs = C.MIN_OBS_IN_WINDOW
    pos_decay, neg_decay = C.POS_DECAY_RATE, C.NEG_DECAY_RATE
    wfloor = C.MIN_WEIGHT_FLOOR
    val_cols = [C.col_value(f) for f in feature_fids]
    mask_cols = [C.col_mask(f) for f in feature_fids]
    # precompute name maps so the UDTF closure never references det_constants at runtime
    vcol_map = {f: v for f, v in zip(feature_fids, val_cols)}
    mcol_map = {f: m for f, m in zip(feature_fids, mask_cols)}
    out_cols = (["ENC_ID", "ANCHOR_TSP", "SPLIT", "LABEL", "SAMPLE_WEIGHT",
                 "EVENT_TSP", "OUTCOME"]
                + [c for pair in zip(val_cols, mask_cols) for c in pair])
    global_means = {f: scaler.get(f, (0.0, 1.0))[0] for f in feature_fids}

    def _impute(series, min_bin=0):
        # hourly-mean series indexed 0..seq_len-1, IHM-style fill. Interpolation is
        # confined to bins at/after min_bin so pre-eligible bins are never back-filled
        # from ward data; they stay NaN and become the fill value (z=0) with mask 0.
        s = series.copy()
        if min_bin > 0:
            s.iloc[min_bin:] = s.iloc[min_bin:].interpolate(
                method="linear", limit_direction="both"
            )
            return s
        return s.interpolate(method="linear", limit_direction="both")

    def process(pdf: pd.DataFrame) -> pd.DataFrame:
        enc_id = pdf["ENC_ID"].iloc[0]
        obs = pdf[pdf["KIND"] == "OBS"][["FID", "TSP", "VAL"]].dropna(subset=["TSP"])
        anchors = pdf[pdf["KIND"] == "ANCHOR"][["TSP", "SPLIT", "ELIGIBLE_START"]].dropna(subset=["TSP"])
        events = pdf[pdf["KIND"] == "EVENT"][["TSP", "OUTCOME"]].dropna(subset=["TSP"])

        if len(anchors) == 0:
            return pd.DataFrame(columns=out_cols)

        obs = obs.copy()
        obs["TSP"] = pd.to_datetime(obs["TSP"])
        anchors = anchors.sort_values("TSP")
        anchors["TSP"] = pd.to_datetime(anchors["TSP"])
        anchors["ELIGIBLE_START"] = pd.to_datetime(anchors["ELIGIBLE_START"])
        split = anchors["SPLIT"].iloc[0]
        ev_tsps = np.sort(pd.to_datetime(events["TSP"]).values) if len(events) else np.array([], dtype="datetime64[ns]")
        ev_lookup = {pd.Timestamp(t): o for t, o in zip(events["TSP"], events["OUTCOME"])}

        # per-FID sorted observation arrays for fast windowing
        by_fid = {}
        for fid in feature_fids:
            sub = obs[obs["FID"] == fid]
            if len(sub):
                ts = sub["TSP"].values.astype("datetime64[ns]")
                order = np.argsort(ts)
                by_fid[fid] = (ts[order], sub["VAL"].values[order])

        anchor_ts = anchors["TSP"].values.astype("datetime64[ns]")
        elig_ts = anchors["ELIGIBLE_START"].values.astype("datetime64[ns]")
        win = np.timedelta64(seq_len, "h")

        rows = []
        for a, elig in zip(anchor_ts, elig_ts):
            a_ts = pd.Timestamp(a)
            # clip the trailing window at the start of the eligible span: bins before
            # ward entry hold no real data, so they must not be imputed from ICU/ED values
            start = a - win
            if not np.isnat(elig) and elig > start:
                start = elig
            # first bin that contains any eligible time; earlier bins are forced to
            # mask=0 and the fill value (z=0) rather than carried-forward pre-ward data
            min_bin = 0
            if not np.isnat(elig):
                h_elig = (a - elig) / np.timedelta64(1, "h")
                min_bin = int(max(0, min(seq_len - 1, seq_len - 1 - np.floor(h_elig))))
            # ---- build per-feature 48-bin sequences over (start, a] ----
            feat_vals, feat_masks, total_obs = {}, {}, 0
            for fid in feature_fids:
                seq = np.full(seq_len, np.nan)
                msk = np.zeros(seq_len, dtype=int)
                if fid in by_fid:
                    ts, vals = by_fid[fid]
                    lo = np.searchsorted(ts, start, side="right")
                    hi = np.searchsorted(ts, a, side="right")
                    if hi > lo:
                        # bin index 0..47 (oldest..newest); bin 47 = (a-1h, a]
                        hours_before = (a - ts[lo:hi]) / np.timedelta64(1, "h")
                        bins = (seq_len - 1 - np.floor(hours_before)).astype(int)
                        bins = np.clip(bins, min_bin, seq_len - 1)
                        v = vals[lo:hi]
                        acc = pd.Series(v).groupby(bins).mean()
                        for b, mv in acc.items():
                            seq[b] = mv
                            msk[b] = 1
                        total_obs += int(hi - lo)
                feat_vals[fid] = seq
                feat_masks[fid] = msk
            if total_obs < min_obs:
                continue

            # ---- label: any event in (a, a+horizon] ----
            label, event_tsp, outcome = 0, None, None
            if ev_tsps.size:
                j = np.searchsorted(ev_tsps, a, side="right")  # first event > a
                if j < ev_tsps.size and (ev_tsps[j] - a) <= horizon:
                    label = 1
                    event_tsp = pd.Timestamp(ev_tsps[j])
                    outcome = ev_lookup.get(event_tsp)

            row = {"ENC_ID": enc_id, "ANCHOR_TSP": a_ts, "SPLIT": split,
                   "LABEL": label, "EVENT_TSP": event_tsp, "OUTCOME": outcome}
            for fid in feature_fids:
                s = _impute(pd.Series(feat_vals[fid]), min_bin)
                s = s.fillna(global_means[fid])
                mean, std = scaler.get(fid, (0.0, 1.0))
                std = std if std else 1.0
                row[vcol_map[fid]] = [float(x) for x in ((s.values - mean) / std)]
                row[mcol_map[fid]] = [int(x) for x in feat_masks[fid]]
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=out_cols)
        out = pd.DataFrame(rows)

        # ---- sample weights (templatized_training_notebook.calculate_sample_weights) ----
        out = out.sort_values("ANCHOR_TSP").reset_index(drop=True)
        next_event = out["EVENT_TSP"].bfill()
        last_tsp = out["ANCHOR_TSP"].max()
        is_pos = next_event.notna()
        raw = np.zeros(len(out))
        if is_pos.any():
            t2e = (next_event[is_pos] - out.loc[is_pos, "ANCHOR_TSP"]).dt.total_seconds() / 60
            raw[is_pos.values] = np.maximum(np.exp(-pos_decay * np.maximum(t2e, -120)), wfloor)
        if (~is_pos).any():
            t2end = (last_tsp - out.loc[~is_pos, "ANCHOR_TSP"]).dt.total_seconds() / 60
            raw[(~is_pos).values] = np.maximum(np.exp(-neg_decay * np.maximum(t2end, -120)), wfloor)
        # washout: zero weight within WASHOUT_HOURS after the previous event
        prev_event = pd.Series(pd.NaT, index=out.index)
        if ev_tsps.size:
            ev_sorted = pd.Series(pd.to_datetime(ev_tsps)).sort_values().values
            idx = np.searchsorted(ev_sorted, out["ANCHOR_TSP"].values, side="right") - 1
            has_prev = idx >= 0
            prev_event[has_prev] = pd.to_datetime(ev_sorted[idx[has_prev]])
        t_after = (out["ANCHOR_TSP"] - prev_event).dt.total_seconds() / 60
        washout_mask = np.where(t_after < (washout.total_seconds() / 60), 0, 1)
        washout_mask = np.where(prev_event.isna(), 1, washout_mask)
        out["SAMPLE_WEIGHT"] = (raw * washout_mask).astype(float)

        return out[out_cols]

    return process


def main(args):
    marker_dir = args.marker_dir
    if not args.force and pu.is_done(marker_dir, pu.STEP5_EXAMPLES):
        print("Step 5 (_DET_ALL) already complete; skipping. Use --force to redo.")
        return

    conn = get_connection()
    session = get_snowpark_session(conn)
    # applyInPandas registers a temporary UDTF, which requires a current db+schema.
    session.sql(f"USE DATABASE {C.OUT_DB}").collect()
    session.sql(f"USE SCHEMA {C.OUT_DB}.{C.OUT_SCHEMA}").collect()
    try:
        scaler = load_scaler(conn)
        print(f"Loaded scaler for {len(scaler)} features")

        inp = build_input_df(session, limit=args.limit_encounters)
        if args.limit_encounters:
            print(f"  - limiting to {args.limit_encounters} encounters (trial run)")

        processor = make_processor(scaler)
        out_schema = make_output_struct()

        print("Running per-encounter applyInPandas (this is the heavy step) ...")
        result = inp.group_by("ENC_ID").applyInPandas(processor, output_schema=out_schema)
        result.write.mode("overwrite").save_as_table(C.T_ALL)

        n = run_sql(f"SELECT COUNT(*) FROM {C.T_ALL}", conn, fetch=True)[0][0]
        pos = run_sql(f"SELECT SUM(LABEL) FROM {C.T_ALL}", conn, fetch=True)[0][0]
        print(f"  - {C.T_ALL}: {n} examples, {pos} positive")
    finally:
        session.close()
        conn.close()

    pu.write_marker(marker_dir, pu.STEP5_EXAMPLES, {"table": C.T_ALL, "rows": int(n)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker_dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--limit_encounters", type=int, default=None,
                    help="Process only N encounters for a quick trial run.")
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
