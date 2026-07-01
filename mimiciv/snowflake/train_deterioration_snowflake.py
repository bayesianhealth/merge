import os
import sys
import time
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

# merge/ root on path for model + pid packages
_HERE = os.path.dirname(os.path.abspath(__file__))
_MERGE = os.path.dirname(os.path.dirname(_HERE))
for p in (_MERGE, os.path.dirname(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from deterioration_config import SEQ_LEN, NOTE_EMB_DIM, FEATURE_NAMES
import deterioration_dataconnector as dl
from deterioration_rus_snowflake import load_rus_tensors  # RUS .npy loader


DEFAULTS = dict(
    # Model architecture (identical to Databricks config)
    d_model=128, nhead=4, d_ff=256, num_encoder_layers=2, num_moe_layers=1,
    moe_num_experts=4, moe_k=2, moe_num_synergy_experts=1, dropout=0.1,
    modality_encoder_layers=2, use_cnn_encoders=False,
    moe_router_gru_hidden_dim=64, moe_router_token_processed_dim=64,
    moe_router_attn_key_dim=32, moe_router_attn_value_dim=32,
    moe_expert_hidden_dim=128, moe_capacity_factor=1.5, moe_drop_tokens=False,
    # Training
    lr=1e-4, weight_decay=0.01, epochs=10, batch_size=256, seq_len=SEQ_LEN,
    clip_grad_norm=1.0, chunk_rows=20000, use_lr_scheduler=True, seed=42, patience=3,
    # RUS / load losses
    threshold_u=0.5, threshold_r=0.3, threshold_s=0.3,
    lambda_u=0.1, lambda_r=0.1, lambda_s=0.1, lambda_load=0.01, epsilon_loss=1e-8,
    # Task
    num_classes=2, modality_names=["labs_vitals", "notes"],
    # Class-imbalance handling: weight applied to the positive class in the CE
    # term (1.0 == disabled, current behavior). Try ~ neg/pos prevalence ratio.
    pos_weight=1.0,
)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args, device):
    from model.trus_moe_multimodal import MultimodalTRUSMoEModel
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
    ).to(device)
    return model


def broadcast_rus_to_batch(rus_data, batch_size, device):
    return {
        "U": rus_data["U"].unsqueeze(0).expand(batch_size, -1, -1).to(device),
        "R": rus_data["R"].unsqueeze(0).expand(batch_size, -1, -1, -1).to(device),
        "S": rus_data["S"].unsqueeze(0).expand(batch_size, -1, -1, -1).to(device),
    }


def _make_loader(session, split, args, ts_only, shuffle):
    if ts_only:
        dc = dl.make_ts_data_connector(session, split=split)
        return dl.ts_torch_loader(dc, batch_size=args.batch_size,
                                  shuffle=shuffle, drop_last=False)
    return dl.make_multimodal_loader(
        session, split=split, batch_size=args.batch_size,
        chunk_rows=args.chunk_rows, shuffle=shuffle, seed=args.seed, include_notes=True,
    )


def _run_epoch(model, loader, rus_data, args, device, optimizer=None):
    from model.trus_moe_model import calculate_rus_losses, calculate_load_balancing_loss
    train = optimizer is not None
    model.train() if train else model.eval()
    agg = {"loss": 0.0, "task": 0.0, "rus": 0.0, "load": 0.0, "grad_norm": 0.0,
           "correct": 0, "total": 0, "scores": [], "labels": []}
    n_batches = 0
    synergy_idx = set(range(args.moe_num_synergy_experts))

    # Optional positive-class weighting to counter the low deterioration prevalence.
    class_weight = None
    pw = float(getattr(args, "pos_weight", 1.0) or 1.0)
    if pw != 1.0:
        class_weight = torch.tensor([1.0, pw], device=device)

    torch.set_grad_enabled(train)
    for ts, notes, note_mask, labels, weights in loader:
        b = ts.shape[0]
        ts = ts.to(device); notes = notes.to(device)
        lab = labels.long().to(device); w = weights.to(device)
        rus_batch = broadcast_rus_to_batch(rus_data, b, device)

        if train:
            optimizer.zero_grad()
        logits, aux_outputs = model([ts, notes], rus_batch)
        ce = F.cross_entropy(logits, lab, weight=class_weight, reduction="none")
        # Weighted MEAN (normalize by the weight mass, not the batch size) so the
        # task term keeps a stable scale when sample weights are small/decayed and
        # is not swamped by the fixed-lambda RUS / load-balancing losses.
        task_loss = (ce * w).sum() / (w.sum() + args.epsilon_loss)

        total_rus = torch.tensor(0.0, device=device)
        total_load = torch.tensor(0.0, device=device)
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
            total_rus /= len(aux_outputs); total_load /= len(aux_outputs)

        loss = task_loss + total_rus + total_load
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        if train:
            loss.backward()
            if args.clip_grad_norm > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            else:
                gnorm = torch.sqrt(sum(
                    (p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None))
            optimizer.step()
            agg["grad_norm"] += float(gnorm)

        agg["loss"] += loss.item(); agg["task"] += task_loss.item()
        agg["rus"] += float(total_rus); agg["load"] += float(total_load)
        agg["correct"] += (torch.argmax(logits, 1) == lab).sum().item()
        agg["total"] += b
        agg["scores"].extend(torch.softmax(logits, 1)[:, 1].detach().cpu().numpy().tolist())
        agg["labels"].extend(lab.cpu().numpy().tolist())
        n_batches += 1
    torch.set_grad_enabled(True)

    has_both = len(set(agg["labels"])) > 1
    auroc = roc_auc_score(agg["labels"], agg["scores"]) if has_both else 0.0
    auprc = average_precision_score(agg["labels"], agg["scores"]) if has_both else 0.0
    nb = max(n_batches, 1)
    return {
        "loss": agg["loss"] / nb, "task_loss": agg["task"] / nb,
        "rus_loss": agg["rus"] / nb, "load_loss": agg["load"] / nb,
        "grad_norm": agg["grad_norm"] / nb,
        "acc": agg["correct"] / max(agg["total"], 1),
        "auroc": auroc, "auprc": auprc, "n_batches": n_batches,
    }


def main(session, ts_only=False, experiment_name="DETERIORATION_TRUS_MOE",
         rus_path=None, checkpoint_dir=None, register_model=True, **overrides):
    from snowflake.ml.experiment import ExperimentTracking

    cfg = {**DEFAULTS, **overrides}
    args = SimpleNamespace(**cfg)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | ts_only={ts_only}")

    rus_data = load_rus_tensors(rus_path, args.modality_names, args.seq_len)
    print(f"RUS: U{tuple(rus_data['U'].shape)} R{tuple(rus_data['R'].shape)} S{tuple(rus_data['S'].shape)}")

    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) \
        if args.use_lr_scheduler else None

    checkpoint_dir = checkpoint_dir or os.path.join(_HERE, "results", "deterioration", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")

    exp = ExperimentTracking(session=session)
    exp.set_experiment(experiment_name)
    best_auroc, patience = 0.0, 0

    run_name = f"trus_moe_d{args.d_model}_e{args.moe_num_experts}_{'ts' if ts_only else 'mm'}"
    with exp.start_run(run_name=run_name):
        exp.log_params({k: str(v) for k, v in cfg.items()})

        for epoch in range(args.epochs):
            t0 = time.time()
            train_loader = _make_loader(session, "train", args, ts_only, shuffle=True)
            tr = _run_epoch(model, train_loader, rus_data, args, device, optimizer)
            if scheduler is not None:
                scheduler.step()

            val_loader = _make_loader(session, "val", args, ts_only, shuffle=False)
            va = _run_epoch(model, val_loader, rus_data, args, device, optimizer=None)

            lr_now = optimizer.param_groups[0]["lr"]
            epoch_seconds = time.time() - t0
            gen_gap = tr["auroc"] - va["auroc"]
            exp.log_metrics({
                "epoch": epoch,
                "train_loss": tr["loss"], "train_task_loss": tr["task_loss"],
                "train_rus_loss": tr["rus_loss"], "train_load_loss": tr["load_loss"],
                "train_auroc": tr["auroc"], "train_auprc": tr["auprc"], "train_acc": tr["acc"],
                "train_grad_norm": tr["grad_norm"],
                "val_loss": va["loss"], "val_task_loss": va["task_loss"],
                "val_rus_loss": va["rus_loss"], "val_load_loss": va["load_loss"],
                "val_auroc": va["auroc"], "val_auprc": va["auprc"], "val_acc": va["acc"],
                "gen_gap": gen_gap, "lr": lr_now, "epoch_seconds": epoch_seconds,
            }, step=epoch)

            print(f"Epoch {epoch+1}/{args.epochs} ({epoch_seconds:.0f}s, {tr['n_batches']} batches) "
                  f"train loss={tr['loss']:.4f} auroc={tr['auroc']:.4f} auprc={tr['auprc']:.4f} | "
                  f"val loss={va['loss']:.4f} auroc={va['auroc']:.4f} auprc={va['auprc']:.4f} | "
                  f"gap={gen_gap:+.4f} gnorm={tr['grad_norm']:.2f} lr={lr_now:.2e}")

            if va["auroc"] > best_auroc:
                best_auroc, patience = va["auroc"], 0
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_auroc": best_auroc, "args": cfg}, ckpt_path)
                print(f"  new best val auroc={best_auroc:.4f}")
            else:
                patience += 1
                if patience >= args.patience:
                    print(f"  early stopping after {args.patience} epochs without improvement")
                    break

        exp.log_metric("best_val_auroc", best_auroc)

        if register_model and os.path.exists(ckpt_path):
            try:
                model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
                model.eval()
                sample = [torch.zeros(1, SEQ_LEN, len(FEATURE_NAMES)),
                          torch.zeros(1, SEQ_LEN, NOTE_EMB_DIM)]
                exp.log_model(
                    model, model_name=f"deterioration_trus_moe_{'ts' if ts_only else 'mm'}",
                    sample_input_data=sample,
                )
                print("  best model logged to Model Registry via Experiment Tracking")
            except Exception as e:  # registry logging is best-effort
                print(f"  log_model skipped: {e}")

    print(f"\nTraining complete. Best val AUROC: {best_auroc:.4f}")
    return best_auroc


if __name__ == "__main__":
    from snowflake_utils import get_snowpark_session
    main(get_snowpark_session())
