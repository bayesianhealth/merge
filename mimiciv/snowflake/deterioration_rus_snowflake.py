import os
import sys
import argparse
import itertools

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_MERGE = os.path.dirname(os.path.dirname(_HERE))
if _MERGE not in sys.path:
    sys.path.insert(0, _MERGE)

from deterioration_config import (
    MULTIMODAL_TABLE, FEATURE_NAMES, NOTE_EMB_DIM, SEQ_LEN, all_flat_columns,
)

OUTPUT_DIR = os.path.join(_HERE, "results", "deterioration")
FLAT_COLUMNS = [c.upper() for c in all_flat_columns()]

def load_rus_tensors(rus_filepath, modality_names, seq_len):
    """Load a saved RUS .npy into U(M,T), R(M,M,T), S(M,M,T) tensors.

    If `rus_filepath` is None or missing, returns zero tensors (RUS losses become
    inert), so training still runs end-to-end before RUS has been computed.
    """
    M, T = len(modality_names), seq_len
    U = torch.zeros(M, T); R = torch.zeros(M, M, T); S = torch.zeros(M, M, T)
    if not rus_filepath or not os.path.exists(rus_filepath):
        print(f"RUS file not found ({rus_filepath}); using zero RUS tensors.")
        return {"U": U, "R": R, "S": S}

    all_pid = np.load(rus_filepath, allow_pickle=True)
    name_to_idx = {n: i for i, n in enumerate(modality_names)}
    seen = set()
    for result in all_pid:
        m1, m2 = result["feature_pair"]
        if m1 not in name_to_idx or m2 not in name_to_idx:
            continue
        i, j = name_to_idx[m1], name_to_idx[m2]
        if i == j or (min(i, j), max(i, j)) in seen:
            continue
        seen.add((min(i, j), max(i, j)))
        lags = result["lag_results"]
        lag_times = [min(d["lag"], T - 1) for d in lags]
        full = np.arange(T, dtype=np.float32)

        def interp(vals):
            if len(lag_times) == 1:
                return torch.full((T,), float(vals[0]))
            return torch.from_numpy(np.interp(full, lag_times, vals).astype(np.float32))

        R_i = interp([d["R_norm"] for d in lags]); R[i, j] = R_i; R[j, i] = R_i
        S_i = interp([d["S_norm"] for d in lags]); S[i, j] = S_i; S[j, i] = S_i
        U[i] = torch.maximum(U[i], interp([d["U1_norm"] for d in lags]))
        U[j] = torch.maximum(U[j], interp([d["U2_norm"] for d in lags]))
    return {"U": U, "R": R, "S": S}


def load_data_from_snowflake(session, num_subsample=5000, seed=42, require_notes=False):
    """Sample train examples and reconstruct TS (48,31) + notes (48,768) sequences."""
    df = session.table(MULTIMODAL_TABLE).filter("split = 'train'")
    if require_notes:
        df = df.filter("notes_packed IS NOT NULL")
    total = df.count()
    frac = min(1.0, (num_subsample * 1.5) / max(total, 1))  # over-sample then trim
    cols = FLAT_COLUMNS + ["LABEL", "NOTES_PACKED"]
    pdf = df.sample(frac=frac).limit(num_subsample).select(*cols).to_pandas()
    n = len(pdf)
    print(f"Sampled {n} of {total} train examples")

    flat = pdf[FLAT_COLUMNS].to_numpy(dtype=np.float64)
    ts = flat.reshape(n, len(FEATURE_NAMES), SEQ_LEN).transpose(0, 2, 1)
    X_ts_list = [ts[i] for i in range(n)]
    labels = pdf["LABEL"].astype(int).tolist()
    ts_masks = [np.ones(SEQ_LEN, dtype=bool) for _ in range(n)]

    X_notes_list, notes_masks = [], []
    for packed in pdf["NOTES_PACKED"].tolist():
        arr = np.zeros((SEQ_LEN, NOTE_EMB_DIM), dtype=np.float64)
        mask = np.zeros(SEQ_LEN, dtype=bool)
        if packed is not None:
            if isinstance(packed, str):
                import json
                packed = json.loads(packed)
            counts = np.zeros(SEQ_LEN)
            for obj in packed:
                slot = int(obj["slot"])
                if 0 <= slot < SEQ_LEN:
                    arr[slot] += np.asarray(obj["emb"], dtype=np.float64)
                    counts[slot] += 1
                    mask[slot] = True
            nz = counts > 0
            arr[nz] /= counts[nz][:, None]
        X_notes_list.append(arr)
        notes_masks.append(mask)

    n_with = sum(1 for m in notes_masks if m.any())
    print(f"Examples with >=1 note: {n_with}/{n} ({n_with/max(n,1)*100:.1f}%)")
    return X_ts_list, X_notes_list, ts_masks, notes_masks, labels


def run_rus_analysis(X_ts_list, X_notes_list, ts_masks, notes_masks, labels,
                     seq_len=48, num_lags=6, batch_size=256, n_batches=10,
                     discrim_epochs=20, ce_epochs=10, seed=42, device=None,
                     hidden_dim=32, layers=2, activation="relu", lr=1e-3,
                     embed_dim=10, sequence_pooling="timestep",
                     dominance_threshold=0.3, dominance_percentage=0.5):
    from pid.temporal_pid_multi_sequence import temporal_pid_label_multi_sequence_multi_lag
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modality_dim = {"labs_vitals": len(FEATURE_NAMES), "notes": NOTE_EMB_DIM}
    names = list(modality_dim.keys())
    all_pid, dominant = [], []

    for mod1, mod2 in itertools.combinations(names, 2):
        X1, X1m = (X_ts_list, ts_masks) if mod1 == "labs_vitals" else (X_notes_list, notes_masks)
        X2, X2m = (X_notes_list, notes_masks) if mod2 == "notes" else (X_ts_list, ts_masks)
        print(f"--- {mod1} vs {mod2} ---")
        pid = temporal_pid_label_multi_sequence_multi_lag(
            X1, X2, labels, X1m, X2m, seq_len=seq_len, num_lags=num_lags,
            batch_size=batch_size, n_batches=n_batches, discrim_epochs=discrim_epochs,
            ce_epochs=ce_epochs, seed=seed, device=device, hidden_dim=hidden_dim,
            layers=layers, activation=activation, lr=lr, embed_dim=embed_dim,
            n_labels=len(np.unique(labels)), sequence_pooling=sequence_pooling,
        )
        lag_results, counts, valid = [], {"R": 0, "U1": 0, "U2": 0, "S": 0}, 0
        for li, lag in enumerate(pid.get("lag", [])):
            try:
                r, u1, u2, s = (pid["redundancy"][li], pid["unique_x1"][li],
                                pid["unique_x2"][li], pid["synergy"][li])
                mi = pid["total_di"][li]
            except (IndexError, KeyError):
                continue
            if mi <= 1e-9:
                continue
            valid += 1
            norm = {"R": r / mi, "U1": u1 / mi, "U2": u2 / mi, "S": s / mi}
            top = max(norm, key=norm.get)
            if norm[top] > dominance_threshold:
                counts[top] += 1
            lag_results.append({"lag": lag, "R_value": r, "U1_value": u1, "U2_value": u2,
                                "S_value": s, "MI_value": mi, "R_norm": norm["R"],
                                "U1_norm": norm["U1"], "U2_norm": norm["U2"], "S_norm": norm["S"]})
        if valid:
            avg = {k: float(np.mean([lr_[k] for lr_ in lag_results])) for k in
                   ["R_value", "U1_value", "U2_value", "S_value", "MI_value",
                    "R_norm", "U1_norm", "U2_norm", "S_norm"]}
            rec = {"feature_pair": (mod1, mod2), "avg_metrics": avg, "lag_results": lag_results,
                   "n_features_mod1": X1[0].shape[1], "n_features_mod2": X2[0].shape[1]}
            all_pid.append(rec)
            for term, c in counts.items():
                if c / valid >= dominance_percentage:
                    dominant.append({**rec, "dominant_term": term, "dominance_ratio": c / valid})
                    break
            print(f"  R={avg['R_norm']:.3f} U1={avg['U1_norm']:.3f} "
                  f"U2={avg['U2_norm']:.3f} S={avg['S_norm']:.3f} MI={avg['MI_value']:.5f}")
    return all_pid, dominant


def save_results(all_pid, dominant, args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if all_pid:
        path = os.path.join(
            OUTPUT_DIR,
            f"rus_multimodal_all_seq{args.seq_len}_lags{args.num_lags}_{args.sequence_pooling}pool.npy")
        np.save(path, all_pid, allow_pickle=True)
        print(f"Saved {path}")
    return path if all_pid else None


def main(session=None, args=None, **overrides):
    if args is None:
        defaults = dict(num_subsample=5000, seq_len=48, num_lags=6, batch_size=256,
                        n_batches=10, discrim_epochs=20, ce_epochs=10, seed=42, gpu=0,
                        hidden_dim=32, layers=2, activation="relu", lr=1e-3, embed_dim=10,
                        sequence_pooling="timestep", dominance_threshold=0.3,
                        dominance_percentage=0.5, require_notes=False)
        defaults.update(overrides)
        args = argparse.Namespace(**defaults)

    device = torch.device(f"cuda:{getattr(args, 'gpu', 0)}") if torch.cuda.is_available() \
        else torch.device("cpu")
    np.random.seed(args.seed)

    if session is None:
        from snowflake_utils import get_snowpark_session
        session = get_snowpark_session()

    data = load_data_from_snowflake(session, num_subsample=args.num_subsample,
                                    seed=args.seed, require_notes=args.require_notes)
    all_pid, dominant = run_rus_analysis(
        *data, seq_len=args.seq_len, num_lags=args.num_lags, batch_size=args.batch_size,
        n_batches=args.n_batches, discrim_epochs=args.discrim_epochs, ce_epochs=args.ce_epochs,
        seed=args.seed, device=device, hidden_dim=args.hidden_dim, layers=args.layers,
        activation=args.activation, lr=args.lr, embed_dim=args.embed_dim,
        sequence_pooling=args.sequence_pooling, dominance_threshold=args.dominance_threshold,
        dominance_percentage=args.dominance_percentage)
    path = save_results(all_pid, dominant, args)
    print("Phase 5 (RUS) complete.")
    return path


if __name__ == "__main__":
    main()
