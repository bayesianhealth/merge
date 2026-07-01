# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # Train Deterioration Task — Multimodal TRUS-MoE
# MAGIC
# MAGIC **Task**: Predict ICU transfer or death within 24h for general ward patients in MIMIC-IV.
# MAGIC
# MAGIC **Architecture**: TRUS-MoE (Temporal Relevance-Uniqueness-Synergy Mixture of Experts)
# MAGIC - Modality 1: Labs + Vitals time series (48h × 31 features)
# MAGIC - Modality 2: Radiology note embeddings (48h × 768-dim BioBERT)
# MAGIC - RUS-aware expert routing (population-level PID informs gating)
# MAGIC
# MAGIC **Data**: Streaming from Delta tables via `DeteriorationDataLoader`
# MAGIC - Training: 1.72M examples
# MAGIC - Validation: 369K examples
# MAGIC - ~29.8% of examples have ≥1 note in 48h window
# MAGIC - 2.3% positive rate, sample weights with decay+washout
# MAGIC
# MAGIC **Tracking**: MLflow (Databricks-native)

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install torch mlflow -q

# COMMAND ----------

# DBTITLE 1,Configuration
import sys
import os

# Path setup
sys.path.insert(0, "/Workspace/Users/patrick.kasl@bayesianhealth.com/merge/mimiciv")
sys.path.insert(0, "/Workspace/Users/patrick.kasl@bayesianhealth.com/merge")

import torch
import numpy as np
import random
from types import SimpleNamespace

# --- Hyperparameters ---
args = SimpleNamespace(
    # Model architecture
    d_model=128,
    nhead=4,
    d_ff=256,
    num_encoder_layers=2,
    num_moe_layers=1,
    moe_num_experts=4,
    moe_k=2,
    moe_num_synergy_experts=1,
    dropout=0.1,
    modality_encoder_layers=2,
    use_cnn_encoders=False,
    # MoE router config
    moe_router_gru_hidden_dim=64,
    moe_router_token_processed_dim=64,
    moe_router_attn_key_dim=32,
    moe_router_attn_value_dim=32,
    moe_expert_hidden_dim=128,
    moe_capacity_factor=1.5,
    moe_drop_tokens=False,
    # Training
    lr=1e-4,
    weight_decay=0.01,
    epochs=10,
    batch_size=256,
    seq_len=48,
    clip_grad_norm=1.0,
    partition_size=10000,
    use_lr_scheduler=True,
    seed=42,
    patience=3,
    # RUS loss
    threshold_u=0.5,
    threshold_r=0.3,
    threshold_s=0.3,
    lambda_u=0.1,
    lambda_r=0.1,
    lambda_s=0.1,
    lambda_load=0.01,
    epsilon_loss=1e-8,
    # Task
    num_classes=2,
    modality_names=['labs_vitals', 'notes'],
)

# Paths
BASE_DIR = "/Workspace/Users/patrick.kasl@bayesianhealth.com/merge/mimiciv"
RUS_PATH = os.path.join(BASE_DIR, "results/deterioration/rus_multimodal_all_seq48_lags6_timesteppool_10k_seed42.npy")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "results/deterioration/checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(args.seed)
print(f"Config loaded. {args.epochs} epochs, bs={args.batch_size}, lr={args.lr}")

# COMMAND ----------

# DBTITLE 1,Load RUS data
def load_mimiciv_rus_data(rus_filepath, modality_names, seq_len):
    """Load RUS data and interpolate across sequence length."""
    all_pid_results = np.load(rus_filepath, allow_pickle=True)
    num_modalities = len(modality_names)
    modality_to_idx = {name: idx for idx, name in enumerate(modality_names)}
    T = seq_len

    U = torch.zeros(num_modalities, T, dtype=torch.float32)
    R = torch.zeros(num_modalities, num_modalities, T, dtype=torch.float32)
    S = torch.zeros(num_modalities, num_modalities, T, dtype=torch.float32)

    processed_pairs = set()
    for result in all_pid_results:
        mod1, mod2 = result['feature_pair']
        if mod1 not in modality_to_idx or mod2 not in modality_to_idx:
            continue
        m1_idx, m2_idx = modality_to_idx[mod1], modality_to_idx[mod2]
        if m1_idx == m2_idx:
            continue
        pair_key = (min(m1_idx, m2_idx), max(m1_idx, m2_idx))
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)

        lag_results = result['lag_results']
        lag_times = [min(d['lag'], T - 1) for d in lag_results]
        full_times = np.arange(T, dtype=np.float32)

        def interp(vals):
            if len(lag_times) == 1:
                return torch.full((T,), vals[0])
            return torch.from_numpy(np.interp(full_times, lag_times, vals).astype(np.float32))

        # Use normalized values (0-1 range) so thresholds work correctly
        R_interp = interp([d['R_norm'] for d in lag_results])
        R[m1_idx, m2_idx, :] = R_interp
        R[m2_idx, m1_idx, :] = R_interp

        S_interp = interp([d['S_norm'] for d in lag_results])
        S[m1_idx, m2_idx, :] = S_interp
        S[m2_idx, m1_idx, :] = S_interp

        U1_interp = interp([d['U1_norm'] for d in lag_results])
        U[m1_idx, :] = torch.maximum(U[m1_idx, :], U1_interp)
        U2_interp = interp([d['U2_norm'] for d in lag_results])
        U[m2_idx, :] = torch.maximum(U[m2_idx, :], U2_interp)

    return {'U': U, 'R': R, 'S': S}


# Load population-level RUS tensors: U(M,T), R(M,M,T), S(M,M,T)
rus_data = load_mimiciv_rus_data(RUS_PATH, args.modality_names, args.seq_len)
print(f"RUS loaded: U{rus_data['U'].shape}, R{rus_data['R'].shape}, S{rus_data['S'].shape}")
print(f"  U mean: {rus_data['U'].mean():.4f}")
print(f"  R mean: {rus_data['R'].mean():.4f}")
print(f"  S mean: {rus_data['S'].mean():.4f}")

# COMMAND ----------

# DBTITLE 1,Initialize model
from model.trus_moe_multimodal import MultimodalTRUSMoEModel

# Modality configs: labs_vitals (31D) and notes (768D)
modality_configs = [
    {
        'input_dim': 31,
        'num_layers': args.modality_encoder_layers,
        'nhead': args.nhead,
        'd_ff': args.d_ff,
        'use_cnn': args.use_cnn_encoders,
    },
    {
        'input_dim': 768,
        'num_layers': args.modality_encoder_layers,
        'nhead': args.nhead,
        'd_ff': args.d_ff,
        'use_cnn': args.use_cnn_encoders,
    },
]

# MoE config
moe_config = {
    'num_experts': args.moe_num_experts,
    'num_synergy_experts': args.moe_num_synergy_experts,
    'k': args.moe_k,
    'expert_hidden_dim': args.moe_expert_hidden_dim,
    'synergy_expert_nhead': args.nhead,
    'router_config': {
        'gru_hidden_dim': args.moe_router_gru_hidden_dim,
        'token_processed_dim': args.moe_router_token_processed_dim,
        'attn_key_dim': args.moe_router_attn_key_dim,
        'attn_value_dim': args.moe_router_attn_value_dim,
    },
    'use_load_balancing': True,
    'capacity_factor': args.moe_capacity_factor,
    'drop_tokens': args.moe_drop_tokens,
}

model = MultimodalTRUSMoEModel(
    modality_configs=modality_configs,
    d_model=args.d_model,
    nhead=args.nhead,
    d_ff=args.d_ff,
    num_encoder_layers=args.num_encoder_layers,
    num_moe_layers=args.num_moe_layers,
    moe_config=moe_config,
    num_classes=args.num_classes,
    max_seq_len=args.seq_len,
    dropout=args.dropout,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model initialized on {device}")
print(f"  Total params: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")

# COMMAND ----------

# DBTITLE 1,Training loop
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import mlflow
import time

from deterioration_dataloader import FEATURE_NAMES, NOTES_TABLE, TS_TABLE, SEQ_LEN, NOTE_EMB_DIM, collate_deterioration
from model.trus_moe_model import calculate_rus_losses, calculate_load_balancing_loss


def broadcast_rus_to_batch(rus_data, batch_size, device):
    """Expand population-level RUS tensors to batch dimension."""
    return {
        'U': rus_data['U'].unsqueeze(0).expand(batch_size, -1, -1).to(device),
        'R': rus_data['R'].unsqueeze(0).expand(batch_size, -1, -1, -1).to(device),
        'S': rus_data['S'].unsqueeze(0).expand(batch_size, -1, -1, -1).to(device),
    }


def train_one_epoch(model, loader, optimizer, scheduler, rus_data, args, device, epoch):
    model.train()
    metrics = {'loss': 0, 'task_loss': 0, 'rus_loss': 0, 'load_loss': 0,
               'correct': 0, 'total': 0, 'scores': [], 'labels': []}
    n_batches = 0

    for ts, notes, note_mask, labels, weights in loader:
        B = ts.shape[0]
        ts = ts.to(device)
        notes = notes.to(device)
        labels = labels.long().to(device)
        weights = weights.to(device)

        # Prepare model inputs
        modality_inputs = [ts, notes]  # List of (B, 48, D_m)
        rus_batch = broadcast_rus_to_batch(rus_data, B, device)

        optimizer.zero_grad()

        # Forward
        logits, aux_outputs = model(modality_inputs, rus_batch)

        # Task loss with sample weights
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # (B,)
        task_loss = (ce_loss * weights).mean()

        # Auxiliary RUS + load balancing losses
        total_L_unique = torch.tensor(0.0, device=device)
        total_L_redundancy = torch.tensor(0.0, device=device)
        total_L_synergy = torch.tensor(0.0, device=device)
        total_L_load = torch.tensor(0.0, device=device)

        for aux in aux_outputs:
            gating_probs = aux['gating_probs']
            expert_indices = aux['expert_indices']
            synergy_indices = set(range(args.moe_num_synergy_experts))

            L_u, L_r, L_s = calculate_rus_losses(
                gating_probs, rus_batch, synergy_indices,
                args.threshold_u, args.threshold_r, args.threshold_s,
                args.lambda_u, args.lambda_r, args.lambda_s,
                args.epsilon_loss
            )
            L_load = calculate_load_balancing_loss(
                gating_probs, expert_indices, args.moe_k, args.lambda_load
            )
            total_L_unique += L_u
            total_L_redundancy += L_r
            total_L_synergy += L_s
            total_L_load += L_load

        if len(aux_outputs) > 0:
            n_moe = len(aux_outputs)
            total_L_unique /= n_moe
            total_L_redundancy /= n_moe
            total_L_synergy /= n_moe
            total_L_load /= n_moe

        total_loss = task_loss + total_L_unique + total_L_redundancy + total_L_synergy + total_L_load

        # Backward
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"  Warning: NaN/Inf loss at batch {n_batches}, skipping")
            continue

        total_loss.backward()
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()

        # Accumulate metrics
        metrics['loss'] += total_loss.item()
        metrics['task_loss'] += task_loss.item()
        metrics['rus_loss'] += (total_L_unique + total_L_redundancy + total_L_synergy).item()
        metrics['load_loss'] += total_L_load.item()
        preds = torch.argmax(logits, dim=1)
        metrics['correct'] += (preds == labels).sum().item()
        metrics['total'] += B
        metrics['scores'].extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist())
        metrics['labels'].extend(labels.cpu().numpy().tolist())
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    # Compute epoch metrics
    acc = metrics['correct'] / max(metrics['total'], 1)
    try:
        auroc = roc_auc_score(metrics['labels'], metrics['scores'])
    except ValueError:
        auroc = 0.0

    return {
        'loss': metrics['loss'] / max(n_batches, 1),
        'task_loss': metrics['task_loss'] / max(n_batches, 1),
        'rus_loss': metrics['rus_loss'] / max(n_batches, 1),
        'load_loss': metrics['load_loss'] / max(n_batches, 1),
        'acc': acc,
        'auroc': auroc,
        'n_batches': n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, rus_data, args, device):
    model.eval()
    metrics = {'loss': 0, 'correct': 0, 'total': 0, 'scores': [], 'labels': []}
    n_batches = 0

    for ts, notes, note_mask, labels, weights in loader:
        B = ts.shape[0]
        ts = ts.to(device)
        notes = notes.to(device)
        labels = labels.long().to(device)
        weights = weights.to(device)

        modality_inputs = [ts, notes]
        rus_batch = broadcast_rus_to_batch(rus_data, B, device)

        logits, _ = model(modality_inputs, rus_batch)
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        loss = (ce_loss * weights).mean()

        metrics['loss'] += loss.item()
        preds = torch.argmax(logits, dim=1)
        metrics['correct'] += (preds == labels).sum().item()
        metrics['total'] += B
        metrics['scores'].extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
        metrics['labels'].extend(labels.cpu().numpy().tolist())
        n_batches += 1

    acc = metrics['correct'] / max(metrics['total'], 1)
    try:
        auroc = roc_auc_score(metrics['labels'], metrics['scores'])
    except ValueError:
        auroc = 0.0

    return {
        'loss': metrics['loss'] / max(n_batches, 1),
        'acc': acc,
        'auroc': auroc,
        'n_batches': n_batches,
    }


import pyspark.sql.functions as SF  # Spark functions (avoid collision with torch.nn.functional)


def load_partition(spark, split, hadm_tsp_pairs, include_notes=True):
    """Load one partition of data using a fresh Spark query (no cursor reuse).
    
    Args:
        hadm_tsp_pairs: list of (hadm_id, tsp_str) tuples for this partition
    Returns:
        ts_features (N,48,31), note_features (N,48,768), note_mask (N,48),
        labels (N,), weights (N,)
    """
    import pandas as pd

    # Build filter values — convert numpy datetime64 to python datetime for PySpark
    import pandas as pd
    hadm_ids = list(set(int(h) for h, _ in hadm_tsp_pairs))
    tsp_values = list(set(
        pd.Timestamp(t).to_pydatetime() if not isinstance(t, str) else t
        for _, t in hadm_tsp_pairs
    ))

    # Fresh query for this partition
    ts_pdf = (
        spark.table(TS_TABLE)
        .filter(SF.col("split") == split)
        .filter(SF.col("hadm_id").isin(hadm_ids))
        .filter(SF.col("tsp").isin(tsp_values))
        .select(*(FEATURE_NAMES + ["hadm_id", "tsp", "label", "sample_weight"]))
        .toPandas()
    )

    if len(ts_pdf) == 0:
        return None

    n_rows = len(ts_pdf)
    ts_features = np.zeros((n_rows, SEQ_LEN, len(FEATURE_NAMES)), dtype=np.float32)
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        for row_idx, arr in enumerate(ts_pdf[feat_name].values):
            if arr is not None:
                ts_features[row_idx, :, feat_idx] = np.array(arr, dtype=np.float32)

    labels = ts_pdf["label"].values.astype(np.float32)
    weights = ts_pdf["sample_weight"].values.astype(np.float32)

    # Notes
    if include_notes:
        note_features = np.zeros((n_rows, SEQ_LEN, NOTE_EMB_DIM), dtype=np.float32)
        note_mask = np.zeros((n_rows, SEQ_LEN), dtype=np.float32)

        key_to_idx = {}
        for idx in range(n_rows):
            key = (int(ts_pdf.iloc[idx]["hadm_id"]), ts_pdf.iloc[idx]["tsp"])
            key_to_idx[key] = idx

        tsp_min, tsp_max = ts_pdf["tsp"].min(), ts_pdf["tsp"].max()
        notes_pdf = (
            spark.table(NOTES_TABLE)
            .filter(SF.col("hadm_id").isin(hadm_ids))
            .filter(SF.col("obs_tsp").between(tsp_min, tsp_max))
            .select("hadm_id", "obs_tsp", "hours_before_obs", "note_embedding")
            .toPandas()
        )

        for _, note_row in notes_pdf.iterrows():
            key = (int(note_row["hadm_id"]), note_row["obs_tsp"])
            if key not in key_to_idx:
                continue
            row_idx = key_to_idx[key]
            hours_before = int(note_row["hours_before_obs"])
            arr_idx = SEQ_LEN - 1 - hours_before
            if arr_idx < 0 or arr_idx >= SEQ_LEN:
                continue
            emb = np.array(note_row["note_embedding"], dtype=np.float32)
            if note_mask[row_idx, arr_idx] == 0:
                note_features[row_idx, arr_idx, :] = emb
                note_mask[row_idx, arr_idx] = 1.0
            else:
                note_features[row_idx, arr_idx, :] = (
                    note_features[row_idx, arr_idx, :] + emb
                ) / 2.0
    else:
        note_features = np.zeros((n_rows, SEQ_LEN, NOTE_EMB_DIM), dtype=np.float32)
        note_mask = np.zeros((n_rows, SEQ_LEN), dtype=np.float32)

    return ts_features, note_features, note_mask, labels, weights


def iter_batches_from_partition(ts_features, note_features, note_mask, labels, weights, batch_size, shuffle, rng):
    """Yield (B, ...) tensor batches from pre-loaded numpy arrays."""
    n = len(labels)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        yield (
            torch.from_numpy(ts_features[idx]),
            torch.from_numpy(note_features[idx]),
            torch.from_numpy(note_mask[idx]),
            torch.from_numpy(labels[idx]),
            torch.from_numpy(weights[idx]),
        )


def train(model, rus_data, args, device):
    """Full training loop with cursor-free Spark reads and early stopping."""
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.use_lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Pre-collect keys for partitioning (one Spark query, fast)
    print("Collecting partition keys...")
    train_keys_pdf = (
        spark.table(TS_TABLE)
        .filter("split = 'train'")
        .select("hadm_id", "tsp")
        .toPandas()
    )
    val_keys_pdf = (
        spark.table(TS_TABLE)
        .filter("split = 'val'")
        .select("hadm_id", "tsp")
        .toPandas()
    )
    train_keys = list(zip(train_keys_pdf["hadm_id"].values, train_keys_pdf["tsp"].values))
    val_keys = list(zip(val_keys_pdf["hadm_id"].values, val_keys_pdf["tsp"].values))
    print(f"Train: {len(train_keys):,} examples | Val: {len(val_keys):,} examples")

    # MLflow
    mlflow.set_experiment("/Users/patrick.kasl@bayesianhealth.com/deterioration_trus_moe")
    
    best_auroc = 0.0
    patience_counter = 0

    with mlflow.start_run(run_name=f"trus_moe_d{args.d_model}_e{args.moe_num_experts}"):
        mlflow.log_params(vars(args))

        for epoch in range(args.epochs):
            t0 = time.time()
            rng = np.random.default_rng(args.seed + epoch)

            # --- TRAIN ---
            model.train()
            # Shuffle and partition keys
            train_idx = np.arange(len(train_keys))
            rng.shuffle(train_idx)
            n_parts = (len(train_keys) + args.partition_size - 1) // args.partition_size

            metrics = {'loss': 0, 'task_loss': 0, 'rus_loss': 0, 'load_loss': 0,
                       'correct': 0, 'total': 0, 'scores': [], 'labels': []}
            n_batches = 0

            for part_i in range(n_parts):
                part_start = part_i * args.partition_size
                part_end = min(part_start + args.partition_size, len(train_keys))
                part_keys = [train_keys[train_idx[i]] for i in range(part_start, part_end)]

                # Fresh Spark query for this partition
                part_data = load_partition(spark, 'train', part_keys, include_notes=True)
                if part_data is None:
                    continue
                ts_feat, note_feat, note_mask_arr, labs, wts = part_data

                # Iterate mini-batches from this partition
                for ts, notes, nmask, lab, w in iter_batches_from_partition(
                    ts_feat, note_feat, note_mask_arr, labs, wts,
                    args.batch_size, shuffle=True, rng=rng
                ):
                    B = ts.shape[0]
                    ts = ts.to(device)
                    notes = notes.to(device)
                    lab_t = lab.long().to(device)
                    w_t = w.to(device)

                    modality_inputs = [ts, notes]
                    rus_batch = broadcast_rus_to_batch(rus_data, B, device)

                    optimizer.zero_grad()
                    logits, aux_outputs = model(modality_inputs, rus_batch)

                    ce_loss = F.cross_entropy(logits, lab_t, reduction='none')
                    task_loss = (ce_loss * w_t).mean()

                    # Aux losses
                    total_L_rus = torch.tensor(0.0, device=device)
                    total_L_load = torch.tensor(0.0, device=device)
                    for aux in aux_outputs:
                        gp = aux['gating_probs']
                        ei = aux['expert_indices']
                        L_u, L_r, L_s = calculate_rus_losses(
                            gp, rus_batch, set(range(args.moe_num_synergy_experts)),
                            args.threshold_u, args.threshold_r, args.threshold_s,
                            args.lambda_u, args.lambda_r, args.lambda_s, args.epsilon_loss
                        )
                        total_L_rus += L_u + L_r + L_s
                        total_L_load += calculate_load_balancing_loss(gp, ei, args.moe_k, args.lambda_load)
                    if len(aux_outputs) > 0:
                        total_L_rus /= len(aux_outputs)
                        total_L_load /= len(aux_outputs)

                    total_loss = task_loss + total_L_rus + total_L_load
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        continue

                    total_loss.backward()
                    if args.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()

                    metrics['loss'] += total_loss.item()
                    metrics['task_loss'] += task_loss.item()
                    metrics['rus_loss'] += total_L_rus.item()
                    metrics['load_loss'] += total_L_load.item()
                    metrics['correct'] += (torch.argmax(logits, 1) == lab_t).sum().item()
                    metrics['total'] += B
                    metrics['scores'].extend(torch.softmax(logits, 1)[:, 1].detach().cpu().numpy().tolist())
                    metrics['labels'].extend(lab_t.cpu().numpy().tolist())
                    n_batches += 1

                # Progress
                if (part_i + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1} train: {part_i+1}/{n_parts} partitions, {n_batches} batches")

            if scheduler is not None:
                scheduler.step()

            train_auroc = roc_auc_score(metrics['labels'], metrics['scores']) if metrics['total'] > 0 else 0.0
            train_loss = metrics['loss'] / max(n_batches, 1)

            # --- VALIDATE ---
            val_metrics = evaluate_partitioned(model, spark, val_keys, rus_data, args, device)

            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]['lr']

            mlflow.log_metrics({
                'train/loss': train_loss,
                'train/auroc': train_auroc,
                'train/acc': metrics['correct'] / max(metrics['total'], 1),
                'val/loss': val_metrics['loss'],
                'val/auroc': val_metrics['auroc'],
                'val/acc': val_metrics['acc'],
                'lr': lr_now,
            }, step=epoch)

            print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.0f}s) — "
                  f"Train: loss={train_loss:.4f} AUROC={train_auroc:.4f} ({n_batches} batches) | "
                  f"Val: loss={val_metrics['loss']:.4f} AUROC={val_metrics['auroc']:.4f} | "
                  f"LR={lr_now:.2e}")

            if val_metrics['auroc'] > best_auroc:
                best_auroc = val_metrics['auroc']
                patience_counter = 0
                ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auroc': best_auroc,
                    'args': vars(args),
                }, ckpt_path)
                print(f"  ✓ New best AUROC={best_auroc:.4f}, saved to {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping: no improvement for {args.patience} epochs")
                    break

        mlflow.log_metric('best_val_auroc', best_auroc)
        print(f"\nTraining complete. Best val AUROC: {best_auroc:.4f}")

    return best_auroc


@torch.no_grad()
def evaluate_partitioned(model, spark, val_keys, rus_data, args, device):
    """Evaluate using the same cursor-free partition approach."""
    model.eval()
    metrics = {'loss': 0, 'correct': 0, 'total': 0, 'scores': [], 'labels': []}
    n_batches = 0
    n_parts = (len(val_keys) + args.partition_size - 1) // args.partition_size
    rng = np.random.default_rng(args.seed)

    for part_i in range(n_parts):
        part_start = part_i * args.partition_size
        part_end = min(part_start + args.partition_size, len(val_keys))
        part_keys = val_keys[part_start:part_end]

        part_data = load_partition(spark, 'val', part_keys, include_notes=True)
        if part_data is None:
            continue
        ts_feat, note_feat, note_mask_arr, labs, wts = part_data

        for ts, notes, nmask, lab, w in iter_batches_from_partition(
            ts_feat, note_feat, note_mask_arr, labs, wts,
            args.batch_size, shuffle=False, rng=rng
        ):
            B = ts.shape[0]
            ts = ts.to(device)
            notes = notes.to(device)
            lab_t = lab.long().to(device)
            w_t = w.to(device)

            modality_inputs = [ts, notes]
            rus_batch = broadcast_rus_to_batch(rus_data, B, device)
            logits, _ = model(modality_inputs, rus_batch)

            ce_loss = F.cross_entropy(logits, lab_t, reduction='none')
            loss = (ce_loss * w_t).mean()

            metrics['loss'] += loss.item()
            metrics['correct'] += (torch.argmax(logits, 1) == lab_t).sum().item()
            metrics['total'] += B
            metrics['scores'].extend(torch.softmax(logits, 1)[:, 1].cpu().numpy().tolist())
            metrics['labels'].extend(lab_t.cpu().numpy().tolist())
            n_batches += 1

    auroc = roc_auc_score(metrics['labels'], metrics['scores']) if metrics['total'] > 0 else 0.0
    return {
        'loss': metrics['loss'] / max(n_batches, 1),
        'acc': metrics['correct'] / max(metrics['total'], 1),
        'auroc': auroc,
    }

# COMMAND ----------

# DBTITLE 1,Run training
"""Write training data to UC Volume as Parquet.

Databricks-native pattern for ML training at scale:
  1. ONE-TIME: Spark writes Delta table -> Parquet on UC Volume (parallel, handles OOM)
  2. TRAINING: PyArrow reads directly from Volume (no Spark, no cursors, streams on-demand)

The UC Volume is persistent S3 storage — no file size limits, accessible from any compute.
Spark handles the write parallelism. PyArrow handles training reads without Spark overhead.
"""
import pyspark.sql.functions as SF

# UC Volume (S3-backed, no 2GB file limit, persistent across sessions)
VOLUME_PATH = "/Volumes/mimiciv/hosp/ml_training_data"
PARQUET_DIR = f"{VOLUME_PATH}/deterioration"

# Create volume if it doesn't exist
spark.sql("CREATE VOLUME IF NOT EXISTS mimiciv.hosp.ml_training_data")
print(f"Volume ready: {VOLUME_PATH}")


def write_split_to_volume(spark, split):
    """Write a split to Parquet on the UC Volume using Spark's native parallel writer."""
    output_path = f"{PARQUET_DIR}/{split}"
    
    # Check if already written
    try:
        files = os.listdir(output_path)
        parquet_files = [f for f in files if f.endswith('.parquet')]
        if len(parquet_files) > 0:
            print(f"  {split} already exists ({len(parquet_files)} parquet files), skipping.")
            return
    except (FileNotFoundError, OSError):
        pass
    
    print(f"  Writing {split} to {output_path}...")
    
    # Write directly from Delta — Spark handles partitioning and memory automatically
    # Keep the array columns as-is (Parquet handles arrays natively)
    cols = FEATURE_NAMES + ["hadm_id", "tsp", "label", "sample_weight"]
    (
        spark.table(TS_TABLE)
        .filter(SF.col("split") == split)
        .select(*cols)
        .repartition(32)  # Control file count for efficient reading
        .write
        .mode("overwrite")
        .parquet(output_path)
    )
    
    # Verify
    files = [f for f in os.listdir(output_path) if f.endswith('.parquet')]
    print(f"  {split} written: {len(files)} parquet files")


print("=== Writing training data to UC Volume (Parquet) ===")
write_split_to_volume(spark, 'train')
write_split_to_volume(spark, 'val')
print(f"\nDone! Data at {PARQUET_DIR}/")
print("Training will read directly via PyArrow — no Spark needed.")

# COMMAND ----------

# DBTITLE 1,Train from local cache
"""Train using PyArrow to read directly from UC Volume — no Spark during training.

PyArrow reads Parquet files from the Volume with zero Spark overhead:
  - Memory-efficient: reads one row group at a time
  - Fast: columnar format, predicate pushdown
  - No cursors to expire
  - Multi-file support for parallel reads
"""
VOLUME_PATH = "/Volumes/mimiciv/hosp/ml_training_data"
PARQUET_DIR = f"{VOLUME_PATH}/deterioration"

import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader


class ParquetStreamingDataset(IterableDataset):
    """PyTorch IterableDataset that streams from Parquet files on UC Volume.
    
    Reads one row group at a time for memory efficiency. Handles shuffling
    at both the file level and within each row group.
    """
    
    def __init__(self, parquet_dir, feature_names, shuffle=True, seed=42, epoch=0):
        self.parquet_dir = parquet_dir
        self.feature_names = feature_names
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch
        
        # Discover parquet files
        self.files = sorted([
            os.path.join(parquet_dir, f) 
            for f in os.listdir(parquet_dir) 
            if f.endswith('.parquet')
        ])
        
        # Get total row count from metadata
        self._total_rows = 0
        for f in self.files:
            pf = pq.ParquetFile(f)
            self._total_rows += pf.metadata.num_rows
    
    def __len__(self):
        return self._total_rows
    
    def set_epoch(self, epoch):
        """Update epoch for deterministic shuffling."""
        self.epoch = epoch
    
    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        
        # Shuffle file order each epoch
        file_order = np.arange(len(self.files))
        if self.shuffle:
            rng.shuffle(file_order)
        
        for file_idx in file_order:
            pf = pq.ParquetFile(self.files[file_idx])
            
            # Read row groups one at a time (memory efficient)
            for rg_idx in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg_idx)
                
                # Parse features into numpy
                n = len(table)
                ts_features = np.zeros((n, SEQ_LEN, len(self.feature_names)), dtype=np.float32)
                for feat_idx, feat_name in enumerate(self.feature_names):
                    col = table.column(feat_name)
                    for row_idx in range(n):
                        arr = col[row_idx].as_py()
                        if arr is not None:
                            ts_features[row_idx, :, feat_idx] = np.array(arr, dtype=np.float32)
                
                labels = table.column('label').to_numpy().astype(np.float32)
                weights = table.column('sample_weight').to_numpy().astype(np.float32)
                
                # Shuffle within row group
                indices = np.arange(n)
                if self.shuffle:
                    rng.shuffle(indices)
                
                # Yield individual samples
                for i in indices:
                    # Notes: zeros for TS-only baseline
                    note_features = np.zeros((SEQ_LEN, NOTE_EMB_DIM), dtype=np.float32)
                    yield ts_features[i], note_features, labels[i], weights[i]


def collate_fn(batch):
    ts_list, notes_list, labels_list, weights_list = zip(*batch)
    return (
        torch.from_numpy(np.stack(ts_list)),
        torch.from_numpy(np.stack(notes_list)),
        torch.from_numpy(np.array(labels_list)),
        torch.from_numpy(np.array(weights_list)),
    )


def train_streaming(model, rus_data, args, device):
    """Training loop streaming from UC Volume via PyArrow."""
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.use_lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_ds = ParquetStreamingDataset(
        f"{PARQUET_DIR}/train", FEATURE_NAMES, shuffle=True, seed=args.seed
    )
    val_ds = ParquetStreamingDataset(
        f"{PARQUET_DIR}/val", FEATURE_NAMES, shuffle=False, seed=args.seed
    )
    
    # DataLoader wraps the IterableDataset with batching + prefetch
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, collate_fn=collate_fn,
        num_workers=0, pin_memory=True,  # num_workers=0 since PyArrow handles I/O
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, collate_fn=collate_fn,
        num_workers=0, pin_memory=True,
    )
    
    print(f"Train: {len(train_ds):,} samples from {len(train_ds.files)} parquet files")
    print(f"Val:   {len(val_ds):,} samples from {len(val_ds.files)} parquet files")

    # MLflow
    mlflow.set_experiment("/Users/patrick.kasl@bayesianhealth.com/deterioration_trus_moe")
    best_auroc = 0.0
    patience_counter = 0

    with mlflow.start_run(run_name=f"trus_moe_d{args.d_model}_volume"):
        mlflow.log_params(vars(args))

        global_step = 0
        for epoch in range(args.epochs):
            t0 = time.time()

            # --- TRAIN ---
            model.train()
            metrics = {'loss': 0, 'task_loss': 0, 'rus_loss': 0, 'load_loss': 0,
                       'correct': 0, 'total': 0, 'scores': [], 'labels': []}
            n_batches = 0

            train_ds.set_epoch(epoch)
            for ts, notes, labels, weights in train_loader:
                B = ts.shape[0]
                ts = ts.to(device)
                notes = notes.to(device)
                lab_t = labels.long().to(device)
                w_t = weights.to(device)

                modality_inputs = [ts, notes]
                rus_batch = broadcast_rus_to_batch(rus_data, B, device)

                optimizer.zero_grad()
                logits, aux_outputs = model(modality_inputs, rus_batch)

                # Weighted task loss
                ce_loss = F.cross_entropy(logits, lab_t, reduction='none')
                task_loss = (ce_loss * w_t).mean()

                # Auxiliary RUS + load balancing losses
                total_L_rus = torch.tensor(0.0, device=device)
                total_L_load = torch.tensor(0.0, device=device)
                for aux in aux_outputs:
                    gp = aux['gating_probs']
                    ei = aux['expert_indices']
                    L_u, L_r, L_s = calculate_rus_losses(
                        gp, rus_batch, set(range(args.moe_num_synergy_experts)),
                        args.threshold_u, args.threshold_r, args.threshold_s,
                        args.lambda_u, args.lambda_r, args.lambda_s, args.epsilon_loss
                    )
                    total_L_rus += L_u + L_r + L_s
                    total_L_load += calculate_load_balancing_loss(gp, ei, args.moe_k, args.lambda_load)
                if len(aux_outputs) > 0:
                    total_L_rus /= len(aux_outputs)
                    total_L_load /= len(aux_outputs)

                total_loss = task_loss + total_L_rus + total_L_load
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    continue

                total_loss.backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

                metrics['loss'] += total_loss.item()
                metrics['task_loss'] += task_loss.item()
                metrics['rus_loss'] += total_L_rus.item()
                metrics['load_loss'] += total_L_load.item()
                metrics['correct'] += (torch.argmax(logits, 1) == lab_t).sum().item()
                metrics['total'] += B
                metrics['scores'].extend(torch.softmax(logits, 1)[:, 1].detach().cpu().numpy().tolist())
                metrics['labels'].extend(lab_t.cpu().numpy().tolist())
                n_batches += 1
                global_step += 1

                if n_batches % 1000 == 0:
                    running_auroc = roc_auc_score(metrics['labels'], metrics['scores'])
                    print(f"  Epoch {epoch+1} [{n_batches} batches, {metrics['total']:,} ex] "
                          f"loss={metrics['loss']/n_batches:.4f} AUROC={running_auroc:.4f}")
                    mlflow.log_metrics({
                        'train/batch_loss': metrics['loss'] / n_batches,
                        'train/batch_auroc': running_auroc,
                    }, step=global_step)

            if scheduler is not None:
                scheduler.step()

            train_auroc = roc_auc_score(metrics['labels'], metrics['scores']) if metrics['total'] > 0 else 0.0
            train_loss = metrics['loss'] / max(n_batches, 1)

            # --- VALIDATE ---
            model.eval()
            val_m = {'loss': 0, 'correct': 0, 'total': 0, 'scores': [], 'labels': []}
            val_batches = 0
            with torch.no_grad():
                for ts, notes, labels, weights in val_loader:
                    B = ts.shape[0]
                    ts, notes = ts.to(device), notes.to(device)
                    lab_t, w_t = labels.long().to(device), weights.to(device)
                    modality_inputs = [ts, notes]
                    rus_batch = broadcast_rus_to_batch(rus_data, B, device)
                    logits, _ = model(modality_inputs, rus_batch)
                    ce_loss = F.cross_entropy(logits, lab_t, reduction='none')
                    val_m['loss'] += (ce_loss * w_t).mean().item()
                    val_m['correct'] += (torch.argmax(logits, 1) == lab_t).sum().item()
                    val_m['total'] += B
                    val_m['scores'].extend(torch.softmax(logits, 1)[:, 1].cpu().numpy().tolist())
                    val_m['labels'].extend(lab_t.cpu().numpy().tolist())
                    val_batches += 1

            val_auroc = roc_auc_score(val_m['labels'], val_m['scores']) if val_m['total'] > 0 else 0.0
            val_loss = val_m['loss'] / max(val_batches, 1)

            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]['lr']

            mlflow.log_metrics({
                'train/loss': train_loss,
                'train/task_loss': metrics['task_loss'] / max(n_batches, 1),
                'train/rus_loss': metrics['rus_loss'] / max(n_batches, 1),
                'train/load_loss': metrics['load_loss'] / max(n_batches, 1),
                'train/auroc': train_auroc,
                'train/acc': metrics['correct'] / max(metrics['total'], 1),
                'val/loss': val_loss,
                'val/auroc': val_auroc,
                'val/acc': val_m['correct'] / max(val_m['total'], 1),
                'lr': lr_now,
            }, step=epoch)

            print(f"\nEpoch {epoch+1}/{args.epochs} ({elapsed:.0f}s, {n_batches} batches) — "
                  f"Train: loss={train_loss:.4f} AUROC={train_auroc:.4f} | "
                  f"Val: loss={val_loss:.4f} AUROC={val_auroc:.4f} | LR={lr_now:.2e}")

            if val_auroc > best_auroc:
                best_auroc = val_auroc
                patience_counter = 0
                ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auroc': best_auroc,
                    'args': vars(args),
                }, ckpt_path)
                print(f"  ✓ New best AUROC={best_auroc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping after {args.patience} epochs")
                    break

        mlflow.log_metric('best_val_auroc', best_auroc)
        print(f"\nTraining complete. Best val AUROC: {best_auroc:.4f}")
    return best_auroc


# Re-initialize model with fresh weights
seed_everything(args.seed)
model = MultimodalTRUSMoEModel(
    modality_configs=modality_configs,
    d_model=args.d_model, nhead=args.nhead, d_ff=args.d_ff,
    num_encoder_layers=args.num_encoder_layers,
    num_moe_layers=args.num_moe_layers,
    moe_config=moe_config, num_classes=args.num_classes,
    max_seq_len=args.seq_len, dropout=args.dropout,
).to(device)
print(f"Model re-initialized: {sum(p.numel() for p in model.parameters()):,} params")

# Run training
best_auroc = train_streaming(model, rus_data, args, device)
print(f"\nFinal best validation AUROC: {best_auroc:.4f}")