"""Sanity-check the tensors that feed the PID estimator for the labs_vitals<->notes pair.
Self-contained (numpy only): replicates align_multimodal_irg_ts (interval_length=1,
no interpolation) + the 'mean' sequence pooling from temporal_pid_label_multi_sequence_batch,
then reports NaN/inf/extreme-value stats. Mirrors mimiciv_rus_multimodal.py exactly.
"""
import pickle
import numpy as np

PKL = "./data/ihm/train_ihm-48-notes-missingInd-standardized_stays.pkl"
NOTES_DIM = 768
LAGS = [0, 6, 12, 18, 24, 30, 36, 42]  # seq_len=48, num_lags=8


def align_notes(reg_ts, text_time, text_embeddings, num_features=NOTES_DIM, interval_length=1):
    """Faithful copy of align_multimodal_irg_ts for one irregular modality (no interp)."""
    T = len(reg_ts)
    aligned = np.zeros((T, num_features))
    mask = np.zeros(T, dtype=bool)
    feats_list = [(text_time[j], text_embeddings[j]) for j in range(len(text_time))]
    if not feats_list:
        return aligned, mask
    by_index = {}
    for time, feats in feats_list:
        idx = int(time / interval_length)
        if idx >= T or idx < 0:
            continue
        by_index.setdefault(idx, []).append(feats)
    for idx, fl in by_index.items():
        aligned[idx, :] = np.mean(fl, axis=0)
        mask[idx] = True
    return aligned, mask


def stats(name, arr):
    arr = np.asarray(arr, dtype=np.float64)
    n_nan = int(np.isnan(arr).sum()); n_inf = int(np.isinf(arr).sum())
    finite = arr[np.isfinite(arr)]
    if finite.size:
        amin, amax, amean, astd = finite.min(), finite.max(), finite.mean(), finite.std()
        amaxabs = np.abs(finite).max()
    else:
        amin = amax = amean = astd = amaxabs = float("nan")
    print(f"  {name:18s} shape={str(arr.shape):14s} nan={n_nan} inf={n_inf} "
          f"min={amin:.4g} max={amax:.4g} mean={amean:.4g} std={astd:.4g} maxabs={amaxabs:.4g}")
    return n_nan, n_inf, amaxabs


def main():
    print(f"Loading {PKL} ...")
    stays = pickle.load(open(PKL, "rb"))
    print(f"Loaded {len(stays)} stays")

    # reg_ts dimensionality (the 31 vs 30 question)
    dims = {}
    for s in stays:
        d = s["reg_ts"].shape[1]
        dims[d] = dims.get(d, 0) + 1
    print(f"\n[reg_ts column counts over ALL stays]: {dims}")
    print("[modality_dim_dict says labs_vitals = 30]")

    sample = np.concatenate([s["reg_ts"] for s in stays[:500]], axis=0)
    print("\n[raw reg_ts values, first 500 stays]")
    stats("raw reg_ts", sample)
    # notes embeddings raw health
    note_vecs = []
    for s in stays[:2000]:
        if len(s.get("text_embeddings", [])) > 0:
            note_vecs.extend(list(s["text_embeddings"]))
    if note_vecs:
        print("[raw note embeddings, first 2000 stays]")
        stats("raw notes", np.asarray(note_vecs))

    # eligibility filter (matches preprocess_mimiciv_data: len(ts_tt) > 12)
    eligible = [s for s in stays if len(s["ts_tt"]) > 12]
    print(f"\nEligible stays (len(ts_tt) > 12): {len(eligible)} / {len(stays)}")

    # build aligned per-stay (labs_vitals = reg_ts + full mask; notes = aligned)
    built = []
    for s in eligible:
        reg = np.asarray(s["reg_ts"])
        nt, nm = align_notes(reg, s["text_time"], s["text_embeddings"])
        built.append((reg, np.ones(len(reg), dtype=bool), nt, nm, s["label"]))

    labels = np.array([b[4] for b in built])
    uniq, cnt = np.unique(labels, return_counts=True)
    print(f"Label distribution: {dict(zip(uniq.tolist(), cnt.tolist()))}")

    for lag in LAGS:
        X1p, X2p, kept, skipped = [], [], 0, 0
        for reg, m1, nt, m2, _ in built:
            if lag == 0:
                a1, a2, k1, k2 = reg, nt, m1, m2
            else:
                a1, a2, k1, k2 = reg[:-lag], nt[:-lag], m1[:-lag], m2[:-lag]
            if len(a1) == 0:
                skipped += 1; continue
            if np.any(k1) and np.any(k2):
                X1p.append(np.mean(a1[k1], axis=0))
                X2p.append(np.mean(a2[k2], axis=0))
                kept += 1
            else:
                skipped += 1
        pct = 100 * kept / max(1, kept + skipped)
        print(f"\n=== lag {lag}: kept={kept} skipped={skipped} ({pct:.1f}% retained) ===")
        if kept == 0:
            print("  *** ZERO samples retained -> empty tensor -> NaN downstream ***"); continue
        _, _, mx1 = stats("X1 labs_vitals", X1p)
        _, _, mx2 = stats("X2 notes", X2p)
        x1 = np.asarray(X1p); x2 = np.asarray(X2p)
        zc1 = int((x1.std(axis=0) < 1e-8).sum()); zc2 = int((x2.std(axis=0) < 1e-8).sum())
        print(f"  zero-variance cols: labs_vitals={zc1}/{x1.shape[1]} notes={zc2}/{x2.shape[1]}")
        if max(mx1, mx2) > 1e3:
            print(f"  *** EXTREME magnitude (maxabs={max(mx1, mx2):.4g}) ***")


if __name__ == "__main__":
    main()
