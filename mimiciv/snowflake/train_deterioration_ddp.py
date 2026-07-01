import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MERGE = os.path.dirname(os.path.dirname(_HERE))
for p in (_MERGE, os.path.dirname(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

LARGE_DEFAULTS = dict(
    # Scaled architecture
    d_model=1024, nhead=16, d_ff=4096, num_encoder_layers=4, num_moe_layers=2,
    moe_num_experts=8, moe_k=2, moe_num_synergy_experts=2, dropout=0.1,
    modality_encoder_layers=3, use_cnn_encoders=False,
    moe_router_gru_hidden_dim=128, moe_router_token_processed_dim=128,
    moe_router_attn_key_dim=64, moe_router_attn_value_dim=64,
    moe_expert_hidden_dim=1024, moe_capacity_factor=1.5, moe_drop_tokens=False,
    # Training
    lr=5e-5, weight_decay=0.01, epochs=10, batch_size=128, seq_len=48,
    clip_grad_norm=1.0, chunk_rows=20000, use_lr_scheduler=True, seed=42, patience=5,
    threshold_u=0.5, threshold_r=0.3, threshold_s=0.3,
    lambda_u=0.1, lambda_r=0.1, lambda_s=0.1, lambda_load=0.01, epsilon_loss=1e-8,
    # Task
    num_classes=2, modality_names=["labs_vitals", "notes"],
    pos_weight=1.0,
    use_mixed_precision=True, use_gradient_checkpointing=False,
    num_gpus=4, num_nodes=1,
)


def train_func():
    """Training function executed inside each distributed worker."""
    import time
    import random
    from types import SimpleNamespace

    import numpy as np
    import torch
    import torch.nn.functional as F
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score, average_precision_score

    from snowflake.ml.modeling.distributors.pytorch import get_context
    context = get_context()
    rank = context.get_rank()
    local_rank = context.get_local_rank()
    world_size = context.get_world_size()
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    hyper_params = context.get_hyper_params()
    cfg = {k: _cast_param(k, v) for k, v in hyper_params.items()}
    args = SimpleNamespace(**cfg)

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    if rank == 0:
        print(f"Distributed training (DDP): world_size={world_size}, device={device}")
        print(f"Model config: d_model={args.d_model}, d_ff={args.d_ff}, nhead={args.nhead}")
        print(f"  MoE: {args.moe_num_experts} experts, k={args.moe_k}")
        _estimate_params(args)

    from model.trus_moe_multimodal import MultimodalTRUSMoEModel
    from deterioration_config import FEATURE_NAMES, NOTE_EMB_DIM, SEQ_LEN

    modality_configs = [
        {"input_dim": len(FEATURE_NAMES), "num_layers": args.modality_encoder_layers,
         "nhead": args.nhead, "d_ff": args.d_ff, "use_cnn": args.use_cnn_encoders},
        {"input_dim": NOTE_EMB_DIM, "num_layers": args.modality_encoder_layers,
         "nhead": args.nhead, "d_ff": args.d_ff, "use_cnn": args.use_cnn_encoders},
    ]
    moe_config = {
        "num_experts": args.moe_num_experts,
        "num_synergy_experts": args.moe_num_synergy_experts,
        "k": args.moe_k,
        "expert_hidden_dim": args.moe_expert_hidden_dim,
        "synergy_expert_nhead": args.nhead,
        "router_config": {
            "gru_hidden_dim": args.moe_router_gru_hidden_dim,
            "token_processed_dim": args.moe_router_token_processed_dim,
            "attn_key_dim": args.moe_router_attn_key_dim,
            "attn_value_dim": args.moe_router_attn_value_dim,
        },
        "use_load_balancing": True,
        "capacity_factor": args.moe_capacity_factor,
        "drop_tokens": args.moe_drop_tokens,
    }

    model = MultimodalTRUSMoEModel(
        modality_configs=modality_configs, d_model=args.d_model, nhead=args.nhead,
        d_ff=args.d_ff, num_encoder_layers=args.num_encoder_layers,
        num_moe_layers=args.num_moe_layers, moe_config=moe_config,
        num_classes=args.num_classes, max_seq_len=args.seq_len, dropout=args.dropout,
        use_checkpoint=args.use_gradient_checkpointing,
    ).to(device)

    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        if rank == 0:
            print("Model wrapped with DistributedDataParallel (DDP)")

    use_amp = bool(getattr(args, "use_mixed_precision", False)) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    from deterioration_dataconnector import (
        FLAT_COLUMNS, LABEL_COL, WEIGHT_COL, flat_to_ts, collate_multimodal,
        MultimodalDeteriorationDataset, make_multimodal_loader,
    )
    from deterioration_config import SEQ_LEN as _SEQ_LEN

    dataset_map = context.get_dataset_map()
    use_dataset_map = dataset_map and "train" in dataset_map

    if use_dataset_map:
        from deterioration_dataconnector import ts_torch_loader
        if rank == 0:
            print("Using ShardedDataConnector (TS-only) for data loading")

        def _make_loader(split, shuffle):
            shard = dataset_map[split].get_shard()
            return ts_torch_loader(shard, batch_size=args.batch_size, shuffle=shuffle)
    else:
        if rank == 0:
            print(f"Using multimodal streaming loader (world_size sharding: "
                  f"each of {world_size} rank(s) reads a ~1/{world_size} shard)")

        # Distributed Ray workers have no active Snowpark session, so build one
        # from the SPCS OAuth token (get_active_session() raises here).
        worker_session = _get_worker_session()

        def _make_loader(split, shuffle):
            return make_multimodal_loader(
                worker_session, split=split, batch_size=args.batch_size,
                chunk_rows=args.chunk_rows, shuffle=shuffle,
                seed=args.seed, include_notes=True,
                rank=rank, world_size=world_size,
            )

    # --- RUS data ---
    from deterioration_rus_snowflake import load_rus_tensors
    rus_path = hyper_params.get("rus_path", "results/deterioration")
    rus_data = load_rus_tensors(rus_path, args.modality_names, args.seq_len)

    # --- Training loop ---
    from model.trus_moe_model import calculate_rus_losses, calculate_load_balancing_loss

    best_auroc = 0.0
    patience_counter = 0
    synergy_idx = set(range(args.moe_num_synergy_experts))

    pw = float(getattr(args, "pos_weight", 1.0) or 1.0)
    class_weight = torch.tensor([1.0, pw], device=device) if pw != 1.0 else None

    log_every = getattr(args, "log_every", 20)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()

        # Train
        train_loader = _make_loader("train", shuffle=True)
        tr_agg = _init_agg()
        batch_idx = 0
        for ts, notes, note_mask, labels, weights in train_loader:
            b = ts.shape[0]
            ts, notes = ts.to(device), notes.to(device)
            lab, w = labels.long().to(device), weights.to(device)
            rus_batch = _broadcast_rus(rus_data, b, device)

            optimizer.zero_grad()
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, aux_outputs = model([ts, notes], rus_batch)
                    loss = _compute_loss(logits, lab, w, aux_outputs, rus_batch, synergy_idx, args, class_weight)
            else:
                logits, aux_outputs = model([ts, notes], rus_batch)
                loss = _compute_loss(logits, lab, w, aux_outputs, rus_batch, synergy_idx, args, class_weight)

            # Clamp nan/inf instead of skipping to avoid NCCL deadlock across ranks
            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    print(f"  [batch {batch_idx}] WARNING: nan/inf loss detected, clamping to 0")
                loss = torch.zeros_like(loss, requires_grad=True)
            loss.backward()
            if args.clip_grad_norm > 0:
                # DDP exposes the underlying parameters directly; no special API needed
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            _update_agg(tr_agg, loss, logits, lab, b)

            if rank == 0 and batch_idx % log_every == 0:
                elapsed = time.time() - t0
                print(f"  [epoch {epoch+1} batch {batch_idx}] loss={loss.item():.4f} "
                      f"elapsed={elapsed:.0f}s")
            batch_idx += 1

        if rank == 0:
            print(f"  Training phase complete: {batch_idx} batches in {time.time()-t0:.0f}s")

        if scheduler:
            scheduler.step()

        # Validate
        model.eval()
        val_loader = _make_loader("val", shuffle=False)
        va_agg = _init_agg()
        val_batch_idx = 0
        with torch.no_grad():
            for ts, notes, note_mask, labels, weights in val_loader:
                b = ts.shape[0]
                ts, notes = ts.to(device), notes.to(device)
                lab, w = labels.long().to(device), weights.to(device)
                rus_batch = _broadcast_rus(rus_data, b, device)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits, aux_outputs = model([ts, notes], rus_batch)
                        loss = _compute_loss(logits, lab, w, aux_outputs, rus_batch, synergy_idx, args, class_weight)
                else:
                    logits, aux_outputs = model([ts, notes], rus_batch)
                    loss = _compute_loss(logits, lab, w, aux_outputs, rus_batch, synergy_idx, args, class_weight)
                _update_agg(va_agg, loss, logits, lab, b)
                val_batch_idx += 1

        if rank == 0:
            print(f"  Validation phase complete: {val_batch_idx} batches")

        # Metrics
        tr_metrics = _finalize_agg(tr_agg)
        va_metrics = _finalize_agg(va_agg)
        epoch_time = time.time() - t0

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.0f}s) "
                  f"train loss={tr_metrics['loss']:.4f} auroc={tr_metrics['auroc']:.4f} | "
                  f"val loss={va_metrics['loss']:.4f} auroc={va_metrics['auroc']:.4f}")
            context.get_metrics_reporter().log_metrics({
                "epoch": epoch, "train_loss": tr_metrics["loss"],
                "train_auroc": tr_metrics["auroc"], "val_loss": va_metrics["loss"],
                "val_auroc": va_metrics["auroc"], "epoch_seconds": int(epoch_time),
            })

            if va_metrics["auroc"] > best_auroc:
                best_auroc = va_metrics["auroc"]
                patience_counter = 0
                # Save checkpoint: unwrap DDP via .module on rank 0
                model_dir = context.get_model_dir()
                state = (model.module if is_distributed else model).state_dict()
                torch.save({"epoch": epoch, "model_state_dict": state,
                            "val_auroc": best_auroc, "config": cfg},
                           os.path.join(model_dir, "best_model.pt"))
                print(f"  new best val auroc={best_auroc:.4f} (saved)")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  early stopping after {args.patience} epochs without improvement")
                    break

    if rank == 0:
        print(f"\nTraining complete. Best val AUROC: {best_auroc:.4f}")

    if is_distributed:
        dist.destroy_process_group()


# --- Helper functions (must be importable by train_func's closure) ---

def _broadcast_rus(rus_data, batch_size, device):
    return {
        "U": rus_data["U"].unsqueeze(0).expand(batch_size, -1, -1).to(device),
        "R": rus_data["R"].unsqueeze(0).expand(batch_size, -1, -1, -1).to(device),
        "S": rus_data["S"].unsqueeze(0).expand(batch_size, -1, -1, -1).to(device),
    }


def _compute_loss(logits, lab, w, aux_outputs, rus_batch, synergy_idx, args, class_weight):
    import torch
    import torch.nn.functional as F
    from model.trus_moe_model import calculate_rus_losses, calculate_load_balancing_loss

    ce = F.cross_entropy(logits, lab, weight=class_weight, reduction="none")
    task_loss = (ce * w).sum() / (w.sum() + args.epsilon_loss)

    total_rus = torch.tensor(0.0, device=logits.device)
    total_load = torch.tensor(0.0, device=logits.device)
    for aux in aux_outputs:
        L_u, L_r, L_s = calculate_rus_losses(
            aux["gating_probs"], rus_batch, synergy_idx,
            args.threshold_u, args.threshold_r, args.threshold_s,
            args.lambda_u, args.lambda_r, args.lambda_s, args.epsilon_loss,
        )
        total_rus += L_u + L_r + L_s
        total_load += calculate_load_balancing_loss(
            aux["gating_probs"], aux["expert_indices"], args.moe_k, args.lambda_load)
    if aux_outputs:
        total_rus /= len(aux_outputs)
        total_load /= len(aux_outputs)

    return task_loss + total_rus + total_load


def _init_agg():
    return {"loss": 0.0, "n_batches": 0, "correct": 0, "total": 0, "scores": [], "labels": []}


def _update_agg(agg, loss, logits, lab, b):
    import torch
    agg["loss"] += loss.item()
    agg["n_batches"] += 1
    agg["correct"] += (torch.argmax(logits, 1) == lab).sum().item()
    agg["total"] += b
    agg["scores"].extend(torch.softmax(logits.float(), 1)[:, 1].detach().cpu().numpy().tolist())
    agg["labels"].extend(lab.cpu().numpy().tolist())


def _finalize_agg(agg):
    from sklearn.metrics import roc_auc_score, average_precision_score
    nb = max(agg["n_batches"], 1)
    has_both = len(set(agg["labels"])) > 1
    return {
        "loss": agg["loss"] / nb,
        "acc": agg["correct"] / max(agg["total"], 1),
        "auroc": roc_auc_score(agg["labels"], agg["scores"]) if has_both else 0.0,
        "auprc": average_precision_score(agg["labels"], agg["scores"]) if has_both else 0.0,
    }


def _cast_param(key, value):
    """Cast hyper_params (all strings from context) back to their original types."""
    if key in LARGE_DEFAULTS:
        ref = LARGE_DEFAULTS[key]
        if isinstance(ref, bool):
            return value.lower() in ("true", "1", "yes") if isinstance(value, str) else bool(value)
        if isinstance(ref, int):
            return int(float(value))
        if isinstance(ref, float):
            return float(value)
        if isinstance(ref, (list, tuple)):
            if isinstance(value, str):
                import ast
                try:
                    return ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    return ref
            return value
    return value


def _get_worker_session():
    """Build a Snowpark session inside a distributed worker.

    Ray workers have no active session, so get_active_session() raises. Build
    one from the SPCS OAuth token instead, falling back to the active session
    if one happens to exist (e.g. single-process run).
    """
    import os
    from snowflake.snowpark import Session
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except Exception:
        pass
    token_path = os.environ.get("SNOWFLAKE_TOKEN_FILE_PATH", "/snowflake/session/token")
    with open(token_path) as f:
        token = f.read().strip()
    account = os.environ.get("SNOWFLAKE_ACCOUNT", "dj34030")
    host = os.environ.get("SNOWFLAKE_HOST", f"{account}.snowflakecomputing.com")
    return Session.builder.configs({
        "account": account,
        "host": host,
        "authenticator": "oauth",
        "token": token,
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "DEIDENTIFIED_LOAD_MEDIUM"),
        "role": os.environ.get("SNOWFLAKE_ROLE", "DS_ROLE"),
        "database": "TEST",
        "schema": "SILVER",
    }).create()


def _estimate_params(args):
    """Print rough parameter count estimate."""
    from deterioration_config import FEATURE_NAMES, NOTE_EMB_DIM
    d = args.d_model
    enc_params = 2 * (max(len(FEATURE_NAMES), NOTE_EMB_DIM) * d + args.modality_encoder_layers * 4 * d * d)
    shared_params = args.num_encoder_layers * 4 * d * d
    moe_params = args.num_moe_layers * args.moe_num_experts * 2 * d * args.moe_expert_hidden_dim
    total = enc_params + shared_params + moe_params
    print(f"  ~{total / 1e6:.0f}M parameters (rough estimate)")


# --- Public launch function ---

def launch(session, ts_only=False, rus_path="results/deterioration",
           experiment_name="DETERIORATION_TRUS_MOE_DDP", **overrides):
    """Launch distributed DDP training from a notebook cell.

    Args:
        session: Snowpark session
        ts_only: If True, uses ShardedDataConnector (TS-only, no notes).
                 If False, each rank streams multimodal data independently.
        rus_path: Path to RUS .npy files
        experiment_name: For logging (rank 0 only)
        **overrides: Override any key in LARGE_DEFAULTS
    """
    from snowflake.ml.modeling.distributors.pytorch import (
        PyTorchDistributor, PyTorchScalingConfig, WorkerResourceConfig,
    )

    cfg = {**LARGE_DEFAULTS, **overrides}
    num_gpus = cfg.pop("num_gpus")
    num_nodes = cfg.pop("num_nodes")
    cfg["rus_path"] = rus_path

    # Prepare data connectors
    dataset_map = {}
    if ts_only:
        from deterioration_dataconnector import make_sharded_ts_connector
        dataset_map["train"] = make_sharded_ts_connector(session, "train")
        dataset_map["val"] = make_sharded_ts_connector(session, "val")
        print("Using ShardedDataConnector for TS-only distributed data loading")
    else:
        print("Using per-rank multimodal streaming (notes included)")

    # Scaling config: one worker (rank) per GPU. DDP replicates the full model
    # on each GPU; world_size = num_nodes * num_gpus.
    scaling_config = PyTorchScalingConfig(
        num_nodes=num_nodes,
        num_workers_per_node=num_gpus,
        resource_requirements_per_worker=WorkerResourceConfig(num_cpus=0, num_gpus=1),
    )

    trainer = PyTorchDistributor(
        train_func=train_func,
        scaling_config=scaling_config,
    )

    print(f"Launching DDP training: {num_nodes} node(s), {num_gpus} worker(s)/node, "
          f"1 GPU/worker (world_size={num_nodes * num_gpus})")
    print(f"Config: d_model={cfg['d_model']}, d_ff={cfg['d_ff']}, nhead={cfg['nhead']}, "
          f"experts={cfg['moe_num_experts']}, epochs={cfg['epochs']}, batch_size={cfg['batch_size']}")

    # Convert all config values to strings for hyper_params transport
    hyper_params = {k: str(v) for k, v in cfg.items()}

    response = trainer.run(
        dataset_map=dataset_map if dataset_map else None,
        hyper_params=hyper_params,
    )

    print("Distributed DDP training complete.")
    return response
