import os
import sys
import argparse
import itertools
import numpy as np
import torch
from tqdm import tqdm

MERGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MERGE_DIR)

from pid.temporal_pid_multi_sequence import temporal_pid_label_multi_sequence_multi_lag

TS_TABLE = "mimiciv.hosp.deterioration_training_data"
NOTES_TABLE = "mimiciv.hosp.deterioration_note_embeddings"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "deterioration")

FEATURE_NAMES = [
    'anion_gap', 'bicarbonate', 'calcium_total', 'chloride', 'creatinine',
    'diastolic_bp', 'gcs_eye', 'gcs_motor', 'gcs_verbal', 'glucose',
    'heart_rate', 'hematocrit', 'hemoglobin', 'magnesium', 'mch',
    'mchc', 'mcv', 'mean_bp', 'neutrophils', 'o2_saturation',
    'phosphate', 'platelet_count', 'potassium', 'rdw', 'red_blood_cells',
    'respiratory_rate', 'sodium', 'systolic_bp', 'urea_nitrogen',
    'vancomycin', 'white_blood_cells',
]

NOTE_EMB_DIM = 768
SEQ_LEN = 48


def load_data_from_delta(spark, num_subsample=5000, seed=42, require_notes=False):
    """Load a subsample of training data from Delta tables.

    Args:
        spark: Active SparkSession.
        num_subsample: Number of examples to sample for RUS analysis.
        seed: Random seed for sampling.
        require_notes: If True, only include examples that have at least one note.

    Returns:
        X_ts_list: List of (48, 31) numpy arrays (time series features).
        X_notes_list: List of (48, 768) numpy arrays (aligned note embeddings).
        ts_masks: List of (48,) all-ones arrays.
        notes_masks: List of (48,) binary masks.
        labels: List of int labels.
    """
    from pyspark.sql import functions as F

    print(f"Loading data from {TS_TABLE} (train split)...")

    # Get training examples
    ts_df = spark.table(TS_TABLE).filter("split = 'train'")

    if require_notes:
        # Only keep examples that have notes
        notes_keys = (
            spark.table(NOTES_TABLE)
            .select("hadm_id", "obs_tsp")
            .distinct()
        )
        ts_df = ts_df.join(
            notes_keys,
            on=[ts_df["hadm_id"] == notes_keys["hadm_id"],
                ts_df["tsp"] == notes_keys["obs_tsp"]],
            how="inner"
        ).select(ts_df["*"])

    # Sample
    total = ts_df.count()
    fraction = min(1.0, num_subsample / total)
    sampled_df = ts_df.sample(fraction=fraction, seed=seed).limit(num_subsample)

    # Collect time series features
    ts_cols = FEATURE_NAMES + ["hadm_id", "tsp", "label"]
    ts_pdf = sampled_df.select(*ts_cols).toPandas()
    n_samples = len(ts_pdf)
    print(f"Sampled {n_samples} examples from {total} total train examples")

    # Parse TS arrays into numpy
    X_ts_list = []
    for _, row in ts_pdf.iterrows():
        ts_arr = np.zeros((SEQ_LEN, len(FEATURE_NAMES)), dtype=np.float64)
        for feat_idx, feat_name in enumerate(FEATURE_NAMES):
            if row[feat_name] is not None:
                ts_arr[:, feat_idx] = np.array(row[feat_name], dtype=np.float64)
        X_ts_list.append(ts_arr)

    labels = ts_pdf["label"].values.tolist()
    ts_masks = [np.ones(SEQ_LEN, dtype=bool) for _ in range(n_samples)]

    # Fetch notes for sampled examples and align to 48-step grid
    print(f"Fetching notes from {NOTES_TABLE}...")
    hadm_ids = ts_pdf["hadm_id"].unique().tolist()
    tsp_min = ts_pdf["tsp"].min()
    tsp_max = ts_pdf["tsp"].max()

    notes_pdf = (
        spark.table(NOTES_TABLE)
        .filter(F.col("hadm_id").isin(hadm_ids))
        .filter(F.col("obs_tsp").between(tsp_min, tsp_max))
        .select("hadm_id", "obs_tsp", "hours_before_obs", "note_embedding")
        .toPandas()
    )
    print(f"Retrieved {len(notes_pdf)} note rows for {n_samples} examples")

    # Build (hadm_id, tsp) -> row_index lookup
    key_to_idx = {}
    for idx, row in ts_pdf.iterrows():
        key = (int(row["hadm_id"]), row["tsp"])
        key_to_idx[key] = idx

    # Align notes to 48-step grid
    X_notes_list = [np.zeros((SEQ_LEN, NOTE_EMB_DIM), dtype=np.float64) for _ in range(n_samples)]
    notes_masks = [np.zeros(SEQ_LEN, dtype=bool) for _ in range(n_samples)]
    note_counts = [np.zeros(SEQ_LEN, dtype=int) for _ in range(n_samples)]  # for averaging

    for _, note_row in notes_pdf.iterrows():
        key = (int(note_row["hadm_id"]), note_row["obs_tsp"])
        if key not in key_to_idx:
            continue

        row_idx = key_to_idx[key]
        hours_before = int(note_row["hours_before_obs"])

        # Map: hours_before_obs=0 -> index 47 (current), hours_before_obs=47 -> index 0
        arr_idx = SEQ_LEN - 1 - hours_before
        if arr_idx < 0 or arr_idx >= SEQ_LEN:
            continue

        emb = np.array(note_row["note_embedding"], dtype=np.float64)
        note_counts[row_idx][arr_idx] += 1
        # Running sum for averaging
        X_notes_list[row_idx][arr_idx, :] += emb
        notes_masks[row_idx][arr_idx] = True

    # Finalize averages where multiple notes land in same slot
    for i in range(n_samples):
        for t in range(SEQ_LEN):
            if note_counts[i][t] > 1:
                X_notes_list[i][t, :] /= note_counts[i][t]

    n_with_notes = sum(1 for m in notes_masks if m.any())
    print(f"Examples with at least 1 note: {n_with_notes}/{n_samples} ({n_with_notes/n_samples*100:.1f}%)")

    return X_ts_list, X_notes_list, ts_masks, notes_masks, labels


def run_rus_analysis(
    X_ts_list, X_notes_list, ts_masks, notes_masks, labels,
    seq_len=48, num_lags=6, batch_size=256, n_batches=10,
    discrim_epochs=20, ce_epochs=10, seed=42, device=None,
    hidden_dim=32, layers=2, activation='relu', lr=1e-3,
    embed_dim=10, sequence_pooling='timestep',
    dominance_threshold=0.3, dominance_percentage=0.5,
):
    """Run PID analysis between labs_vitals and notes modalities.

    Returns:
        all_pid_results: List of result dicts for all modality pairs.
        dominant_pid_results: List of result dicts where a dominant term was found.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    modality_dim_dict = {
        'labs_vitals': len(FEATURE_NAMES),  # 31
        'notes': NOTE_EMB_DIM,              # 768
    }

    modality_names = list(modality_dim_dict.keys())
    modality_pairs = list(itertools.combinations(modality_names, 2))
    print(f"Modality pairs: {modality_pairs}")

    dominant_pid_results = []
    all_pid_results = []

    for mod1, mod2 in modality_pairs:
        print(f"\n--- Analyzing: {mod1} vs {mod2} ---")

        # Select data for each modality
        if mod1 == 'labs_vitals':
            X1_list, X1_masks = X_ts_list, ts_masks
        else:
            X1_list, X1_masks = X_notes_list, notes_masks

        if mod2 == 'notes':
            X2_list, X2_masks = X_notes_list, notes_masks
        else:
            X2_list, X2_masks = X_ts_list, ts_masks

        print(f"  X1 ({mod1}): {len(X1_list)} seqs, shape {X1_list[0].shape}")
        print(f"  X2 ({mod2}): {len(X2_list)} seqs, shape {X2_list[0].shape}")
        print(f"  Labels: {len(labels)} (positive rate: {np.mean(labels):.4f})")

        # Compute PID
        pid_results = temporal_pid_label_multi_sequence_multi_lag(
            X1_list, X2_list, labels, X1_masks, X2_masks,
            seq_len=seq_len,
            num_lags=num_lags,
            batch_size=batch_size,
            n_batches=n_batches,
            discrim_epochs=discrim_epochs,
            ce_epochs=ce_epochs,
            seed=seed,
            device=device,
            hidden_dim=hidden_dim,
            layers=layers,
            activation=activation,
            lr=lr,
            embed_dim=embed_dim,
            n_labels=len(np.unique(labels)),
            sequence_pooling=sequence_pooling,
        )
        print(f"  PID computed: {len(pid_results['lag'])} lags")

        # --- Analyze dominance ---
        lags = pid_results.get('lag', [])
        dominant_counts = {'R': 0, 'U1': 0, 'U2': 0, 'S': 0}
        total_valid_lags = 0
        lag_results = []

        for lag_idx, lag in enumerate(lags):
            try:
                r = pid_results['redundancy'][lag_idx]
                u1 = pid_results['unique_x1'][lag_idx]
                u2 = pid_results['unique_x2'][lag_idx]
                s = pid_results['synergy'][lag_idx]
                mi = pid_results['total_di'][lag_idx]

                if mi > 1e-9:
                    total_valid_lags += 1
                    r_norm = r / mi
                    u1_norm = u1 / mi
                    u2_norm = u2 / mi
                    s_norm = s / mi

                    norm_values = {'R': r_norm, 'U1': u1_norm, 'U2': u2_norm, 'S': s_norm}
                    max_term = max(norm_values, key=norm_values.get)
                    max_value = norm_values[max_term]

                    if max_value > dominance_threshold:
                        dominant_counts[max_term] += 1

                    lag_results.append({
                        'lag': lag,
                        'R_value': r, 'U1_value': u1, 'U2_value': u2, 'S_value': s,
                        'MI_value': mi,
                        'R_norm': r_norm, 'U1_norm': u1_norm, 'U2_norm': u2_norm, 'S_norm': s_norm,
                    })
            except (IndexError, KeyError) as e:
                print(f"  Warning: {e} at lag {lag}")
                continue

        if total_valid_lags > 0:
            avg_metrics = {
                k: np.mean([r[k] for r in lag_results])
                for k in ['R_value', 'U1_value', 'U2_value', 'S_value', 'MI_value',
                           'R_norm', 'U1_norm', 'U2_norm', 'S_norm']
            }

            all_pid_results.append({
                'feature_pair': (mod1, mod2),
                'avg_metrics': avg_metrics,
                'lag_results': lag_results,
                'n_features_mod1': X1_list[0].shape[1],
                'n_features_mod2': X2_list[0].shape[1],
            })

            # Check for dominant term
            for term, count in dominant_counts.items():
                dominance_ratio = count / total_valid_lags
                if dominance_ratio >= dominance_percentage:
                    print(f"  Dominant: {term} ({dominance_ratio:.1%} of lags)")
                    dominant_pid_results.append({
                        'feature_pair': (mod1, mod2),
                        'dominant_term': term,
                        'dominance_ratio': dominance_ratio,
                        'lags_analyzed': total_valid_lags,
                        'avg_metrics': avg_metrics,
                        'lag_results': lag_results,
                        'n_features_mod1': X1_list[0].shape[1],
                        'n_features_mod2': X2_list[0].shape[1],
                    })
                    break

            # Print summary
            print(f"  Avg MI: {avg_metrics['MI_value']:.6f}")
            print(f"  Avg R: {avg_metrics['R_norm']:.3f}, U1({mod1}): {avg_metrics['U1_norm']:.3f}, "
                  f"U2({mod2}): {avg_metrics['U2_norm']:.3f}, S: {avg_metrics['S_norm']:.3f}")

    return all_pid_results, dominant_pid_results


def save_results(all_pid_results, dominant_pid_results, args):
    """Save PID results to numpy files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if all_pid_results:
        fname = f"rus_multimodal_all_seq{args.seq_len}_lags{args.num_lags}_{args.sequence_pooling}pool.npy"
        path = os.path.join(OUTPUT_DIR, fname)
        np.save(path, all_pid_results, allow_pickle=True)
        print(f"Saved all PID results to {path}")

    if dominant_pid_results:
        fname = (f"rus_multimodal_dominant_seq{args.seq_len}_lags{args.num_lags}"
                 f"_thresh{args.dominance_threshold:.1f}_pct{int(args.dominance_percentage*100)}"
                 f"_{args.sequence_pooling}pool.npy")
        path = os.path.join(OUTPUT_DIR, fname)
        np.save(path, dominant_pid_results, allow_pickle=True)
        print(f"Saved dominant PID results to {path}")


def main(spark=None, args=None):
    """Main entry point. Can be called from notebook or CLI."""
    if args is None:
        parser = argparse.ArgumentParser(description='Deterioration RUS computation')
        parser.add_argument('--num_subsample', type=int, default=5000)
        parser.add_argument('--seq_len', type=int, default=48)
        parser.add_argument('--num_lags', type=int, default=6)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--n_batches', type=int, default=10)
        parser.add_argument('--discrim_epochs', type=int, default=20)
        parser.add_argument('--ce_epochs', type=int, default=10)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--gpu', type=int, default=0)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--layers', type=int, default=2)
        parser.add_argument('--activation', type=str, default='relu')
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--embed_dim', type=int, default=10)
        parser.add_argument('--sequence_pooling', type=str, default='timestep',
                            choices=['timestep', 'mean'])
        parser.add_argument('--dominance_threshold', type=float, default=0.3)
        parser.add_argument('--dominance_percentage', type=float, default=0.5)
        parser.add_argument('--require_notes', action='store_true',
                            help='Only include examples that have notes')
        parser.add_argument('--linear_interpolation', action='store_true',
                            help='Apply linear interpolation to fill missing note timesteps')
        args = parser.parse_args()

    # Device setup
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{getattr(args, "gpu", 0)}')
    else:
        device = torch.device('cpu')

    np.random.seed(args.seed)

    # Get or create Spark session
    if spark is None:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

    # Load data
    X_ts_list, X_notes_list, ts_masks, notes_masks, labels = load_data_from_delta(
        spark,
        num_subsample=args.num_subsample,
        seed=args.seed,
        require_notes=getattr(args, 'require_notes', False),
    )

    # Optionally apply linear interpolation to notes
    if getattr(args, 'linear_interpolation', False):
        print("Applying linear interpolation to note embeddings...")
        for i in range(len(X_notes_list)):
            valid_indices = np.where(notes_masks[i])[0]
            if len(valid_indices) >= 1:
                for feat_idx in range(NOTE_EMB_DIM):
                    valid_times = valid_indices.astype(float)
                    valid_values = X_notes_list[i][valid_indices, feat_idx]
                    all_times = np.arange(SEQ_LEN, dtype=float)
                    X_notes_list[i][:, feat_idx] = np.interp(all_times, valid_times, valid_values)
                notes_masks[i] = np.ones(SEQ_LEN, dtype=bool)

    # Run PID analysis
    all_pid_results, dominant_pid_results = run_rus_analysis(
        X_ts_list, X_notes_list, ts_masks, notes_masks, labels,
        seq_len=args.seq_len,
        num_lags=args.num_lags,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        discrim_epochs=args.discrim_epochs,
        ce_epochs=args.ce_epochs,
        seed=args.seed,
        device=device,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        activation=args.activation,
        lr=args.lr,
        embed_dim=args.embed_dim,
        sequence_pooling=args.sequence_pooling,
        dominance_threshold=args.dominance_threshold,
        dominance_percentage=args.dominance_percentage,
    )

    # Save results
    save_results(all_pid_results, dominant_pid_results, args)

    # Print summary
    print("\n=== RUS Analysis Complete ===")
    print(f"Modality pair: labs_vitals ({len(FEATURE_NAMES)}D) vs notes ({NOTE_EMB_DIM}D)")
    print(f"Samples: {len(labels)}, Positive rate: {np.mean(labels):.4f}")
    if all_pid_results:
        m = all_pid_results[0]['avg_metrics']
        print(f"\nAvg PID decomposition (normalized):")
        print(f"  Redundancy (R):    {m['R_norm']:.4f}")
        print(f"  Unique TS (U1):    {m['U1_norm']:.4f}")
        print(f"  Unique Notes (U2): {m['U2_norm']:.4f}")
        print(f"  Synergy (S):       {m['S_norm']:.4f}")
        print(f"  Total MI:          {m['MI_value']:.6f}")

    return all_pid_results, dominant_pid_results


if __name__ == '__main__':
    # When called via runpy.run_path(), 'spark' may be injected as a global
    _spark = globals().get('spark', None)
    main(spark=_spark)
