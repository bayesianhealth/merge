# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_PROJECT_ROOT)
for _path in (_PROJECT_ROOT, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DEFAULT_RUS_PATH = os.path.join(
    _PROJECT_ROOT,
    "results",
    "deterioration",
    "rus_multimodal_all_seq48_lags6_timesteppool_10k_seed42.npy",
)
DEFAULT_CACHE_DIR = "dbfs:/tmp/merge/mimiciv/deterioration_multigpu_cache"
DEFAULT_OUTPUT_DIR = "dbfs:/tmp/merge/mimiciv/deterioration_multigpu_runs"

DEFAULTS = dict(
    d_model=1024,
    nhead=16,
    d_ff=4096,
    num_encoder_layers=4,
    num_moe_layers=2,
    moe_num_experts=8,
    moe_k=2,
    moe_num_synergy_experts=2,
    dropout=0.1,
    modality_encoder_layers=3,
    use_cnn_encoders=False,
    moe_router_gru_hidden_dim=128,
    moe_router_token_processed_dim=128,
    moe_router_attn_key_dim=64,
    moe_router_attn_value_dim=64,
    moe_expert_hidden_dim=1024,
    moe_capacity_factor=1.5,
    moe_drop_tokens=False,
    lr=5e-5,
    weight_decay=0.01,
    epochs=10,
    batch_size=128,
    seq_len=48,
    clip_grad_norm=1.0,
    patience=5,
    seed=42,
    threshold_u=0.5,
    threshold_r=0.3,
    threshold_s=0.3,
    lambda_u=0.1,
    lambda_r=0.1,
    lambda_s=0.1,
    lambda_load=0.01,
    epsilon_loss=1e-8,
    num_classes=2,
    modality_names=["labs_vitals", "notes"],
    pos_weight=1.0,
    use_mixed_precision=True,
    use_gradient_checkpointing=False,
    strategy="ddp",
    num_workers=0,
    pin_memory=True,
    mlflow_experiment="/Users/patrick.kasl@bayesianhealth.com/deterioration_trus_moe",
)


def _to_local_path(path: str) -> str:
    if path.startswith("dbfs:/"):
        return "/dbfs/" + path[len("dbfs:/") :].lstrip("/")
    return path


def _ensure_dir(path: str) -> str:
    local_path = _to_local_path(path)
    os.makedirs(local_path, exist_ok=True)
    return local_path


def _write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _materialized_split_dir(cache_dir: str, split: str) -> str:
    return os.path.join(_to_local_path(cache_dir), split)


def _manifest_path(cache_dir: str, split: str) -> str:
    return os.path.join(_materialized_split_dir(cache_dir, split), "manifest.json")


def _default_run_name(strategy: str) -> str:
    return f"deterioration_{strategy}_run"


def load_rus_tensors(
    rus_filepath: str,
    modality_names: Sequence[str],
    seq_len: int,
) -> Dict[str, torch.Tensor]:
    """Load and interpolate precomputed RUS tensors for the requested sequence length."""
    if not os.path.exists(rus_filepath):
        raise FileNotFoundError(f"RUS data file not found: {rus_filepath}")

    all_pid_results = np.load(rus_filepath, allow_pickle=True)
    num_modalities = len(modality_names)
    modality_to_idx = {name: idx for idx, name in enumerate(modality_names)}

    U = torch.zeros(num_modalities, seq_len, dtype=torch.float32)
    R = torch.zeros(num_modalities, num_modalities, seq_len, dtype=torch.float32)
    S = torch.zeros(num_modalities, num_modalities, seq_len, dtype=torch.float32)

    processed_pairs = set()

    for result in all_pid_results:
        mod1, mod2 = result["feature_pair"]
        if mod1 not in modality_to_idx or mod2 not in modality_to_idx:
            continue

        m1_idx = modality_to_idx[mod1]
        m2_idx = modality_to_idx[mod2]
        if m1_idx == m2_idx:
            continue

        pair_key = (min(m1_idx, m2_idx), max(m1_idx, m2_idx))
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)

        lag_results = result["lag_results"]
        lag_values = [min(item["lag"], seq_len - 1) for item in lag_results]
        r_values = [item["R_value"] for item in lag_results]
        s_values = [item["S_value"] for item in lag_results]
        u1_values = [item["U1_value"] for item in lag_results]
        u2_values = [item["U2_value"] for item in lag_results]

        full_times = np.arange(seq_len, dtype=np.float32)

        def _interp(values: Sequence[float]) -> torch.Tensor:
            if len(lag_values) == 1:
                return torch.full((seq_len,), float(values[0]), dtype=torch.float32)
            return torch.from_numpy(np.interp(full_times, lag_values, values).astype(np.float32))

        r_interp = _interp(r_values)
        s_interp = _interp(s_values)
        u1_interp = _interp(u1_values)
        u2_interp = _interp(u2_values)

        R[m1_idx, m2_idx, :] = r_interp
        R[m2_idx, m1_idx, :] = r_interp
        S[m1_idx, m2_idx, :] = s_interp
        S[m2_idx, m1_idx, :] = s_interp
        U[m1_idx, :] = torch.maximum(U[m1_idx, :], u1_interp)
        U[m2_idx, :] = torch.maximum(U[m2_idx, :], u2_interp)

    return {"U": U, "R": R, "S": S}


def _flush_shard(
    split_dir: str,
    split: str,
    shard_idx: int,
    ts_chunks: List[torch.Tensor],
    notes_chunks: List[torch.Tensor],
    masks_chunks: List[torch.Tensor],
    labels_chunks: List[torch.Tensor],
    weights_chunks: List[torch.Tensor],
) -> Tuple[str, int]:
    payload = {
        "ts": torch.cat(ts_chunks, dim=0).contiguous(),
        "notes": torch.cat(notes_chunks, dim=0).contiguous(),
        "note_mask": torch.cat(masks_chunks, dim=0).contiguous(),
        "labels": torch.cat(labels_chunks, dim=0).contiguous(),
        "weights": torch.cat(weights_chunks, dim=0).contiguous(),
    }
    shard_path = os.path.join(split_dir, f"{split}_shard_{shard_idx:05d}.pt")
    torch.save(payload, shard_path)
    return shard_path, int(payload["labels"].shape[0])


def materialize_split_cache(
    spark,
    split: str,
    cache_dir: str,
    batch_size: int = 256,
    partition_size: int = 10000,
    examples_per_shard: int = 4096,
    include_notes: bool = True,
    overwrite: bool = False,
    seed: int = 42,
    num_workers: int = 0,
) -> Dict:
    """Materialize a file-backed cache for one split from the saved Delta tables."""
    from deterioration_dataloader import DeteriorationDataLoader

    split_dir = _materialized_split_dir(cache_dir, split)
    manifest_path = _manifest_path(cache_dir, split)

    if os.path.exists(manifest_path) and not overwrite:
        return _read_json(manifest_path)

    if overwrite and os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_dir, exist_ok=True)

    loader = DeteriorationDataLoader(
        spark=spark,
        split=split,
        batch_size=batch_size,
        partition_size=partition_size,
        shuffle=(split == "train"),
        seed=seed,
        include_notes=include_notes,
        num_workers=num_workers,
        pin_memory=False,
    )

    ts_chunks: List[torch.Tensor] = []
    notes_chunks: List[torch.Tensor] = []
    masks_chunks: List[torch.Tensor] = []
    labels_chunks: List[torch.Tensor] = []
    weights_chunks: List[torch.Tensor] = []

    shard_paths: List[str] = []
    shard_sizes: List[int] = []
    shard_idx = 0
    buffered_examples = 0
    total_examples = 0

    for ts, notes, note_mask, labels, weights in loader:
        ts_chunks.append(ts.float().cpu())
        notes_chunks.append(notes.float().cpu())
        masks_chunks.append(note_mask.float().cpu())
        labels_chunks.append(labels.long().cpu())
        weights_chunks.append(weights.float().cpu())
        buffered_examples += int(labels.shape[0])

        if buffered_examples >= examples_per_shard:
            shard_path, shard_size = _flush_shard(
                split_dir,
                split,
                shard_idx,
                ts_chunks,
                notes_chunks,
                masks_chunks,
                labels_chunks,
                weights_chunks,
            )
            shard_paths.append(shard_path)
            shard_sizes.append(shard_size)
            total_examples += shard_size
            shard_idx += 1
            ts_chunks, notes_chunks, masks_chunks, labels_chunks, weights_chunks = [], [], [], [], []
            buffered_examples = 0

    if buffered_examples > 0:
        shard_path, shard_size = _flush_shard(
            split_dir,
            split,
            shard_idx,
            ts_chunks,
            notes_chunks,
            masks_chunks,
            labels_chunks,
            weights_chunks,
        )
        shard_paths.append(shard_path)
        shard_sizes.append(shard_size)
        total_examples += shard_size

    manifest = {
        "split": split,
        "include_notes": include_notes,
        "batch_size": batch_size,
        "partition_size": partition_size,
        "examples_per_shard": examples_per_shard,
        "num_shards": len(shard_paths),
        "total_examples": total_examples,
        "shard_paths": shard_paths,
        "shard_sizes": shard_sizes,
    }
    _write_json(manifest_path, manifest)
    return manifest


def prepare_training_cache(
    spark,
    cache_dir: str = DEFAULT_CACHE_DIR,
    materialize_batch_size: int = 256,
    partition_size: int = 10000,
    examples_per_shard: int = 4096,
    include_notes: bool = True,
    overwrite: bool = False,
    prepare_test: bool = False,
    seed: int = 42,
    num_workers: int = 0,
) -> Dict[str, Dict]:
    _ensure_dir(cache_dir)
    manifests = {
        "train": materialize_split_cache(
            spark,
            "train",
            cache_dir,
            batch_size=materialize_batch_size,
            partition_size=partition_size,
            examples_per_shard=examples_per_shard,
            include_notes=include_notes,
            overwrite=overwrite,
            seed=seed,
            num_workers=num_workers,
        ),
        "val": materialize_split_cache(
            spark,
            "val",
            cache_dir,
            batch_size=materialize_batch_size,
            partition_size=partition_size,
            examples_per_shard=examples_per_shard,
            include_notes=include_notes,
            overwrite=overwrite,
            seed=seed,
            num_workers=num_workers,
        ),
    }
    if prepare_test:
        manifests["test"] = materialize_split_cache(
            spark,
            "test",
            cache_dir,
            batch_size=materialize_batch_size,
            partition_size=partition_size,
            examples_per_shard=examples_per_shard,
            include_notes=include_notes,
            overwrite=overwrite,
            seed=seed,
            num_workers=num_workers,
        )
    return manifests


class CachedShardDataset(torch.utils.data.IterableDataset):
    """Streams one or more cached shard files created by ``prepare_training_cache``."""

    def __init__(
        self,
        shard_paths: Sequence[str],
        shuffle: bool,
        seed: int,
        epoch: int = 0,
    ):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        worker_info = torch.utils.data.get_worker_info()
        shard_paths = list(self.shard_paths)

        if worker_info is not None:
            shard_paths = shard_paths[worker_info.id :: worker_info.num_workers]

        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(shard_paths)

        for shard_path in shard_paths:
            payload = torch.load(shard_path, map_location="cpu")
            n = int(payload["labels"].shape[0])
            indices = list(range(n))
            if self.shuffle:
                rng.shuffle(indices)
            for idx in indices:
                yield (
                    payload["ts"][idx],
                    payload["notes"][idx],
                    payload["note_mask"][idx],
                    payload["labels"][idx],
                    payload["weights"][idx],
                )


def collate_cached_batch(batch: Sequence[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    ts_list, notes_list, masks_list, labels_list, weights_list = zip(*batch)
    return (
        torch.stack(ts_list),
        torch.stack(notes_list),
        torch.stack(masks_list),
        torch.stack(labels_list),
        torch.stack(weights_list),
    )


def _list_shards(cache_dir: str, split: str) -> List[str]:
    manifest = _read_json(_manifest_path(cache_dir, split))
    shard_paths = [path for path in manifest["shard_paths"] if os.path.exists(path)]
    if not shard_paths:
        raise FileNotFoundError(
            f"No shard files found for split '{split}'. Expected manifest in {cache_dir}."
        )
    return shard_paths


def _rank_shards(shard_paths: Sequence[str], rank: int, world_size: int) -> List[str]:
    # Truncate to equal count per rank to avoid DDP collective desync
    shards_per_rank = len(shard_paths) // world_size
    assigned = list(shard_paths)[rank::world_size][:shards_per_rank]
    if not assigned:
        raise ValueError(
            "Not enough materialized shard files for the requested world size. "
            f"Create more shards or reduce num_processes. rank={rank}, world_size={world_size}, "
            f"num_shards={len(shard_paths)}"
        )
    return assigned


def _distributed_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _setup_distributed() -> Tuple[int, int, int]:
    import torch.distributed as dist

    rank, local_rank, world_size = _distributed_info()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def _cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _aggregate_epoch_metrics(local_metrics: Dict[str, object]) -> Dict[str, float]:
    import torch.distributed as dist
    from sklearn.metrics import average_precision_score, roc_auc_score

    numeric = torch.tensor(
        [
            float(local_metrics["loss_sum"]),
            float(local_metrics["correct"]),
            float(local_metrics["n_samples"]),
        ],
        device="cuda",
        dtype=torch.float64,
    )

    if _is_distributed():
        dist.all_reduce(numeric, op=dist.ReduceOp.SUM)
        gathered_scores: List[List[float]] = [None] * dist.get_world_size()
        gathered_labels: List[List[int]] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_scores, list(local_metrics["scores"]))
        dist.all_gather_object(gathered_labels, list(local_metrics["labels"]))
        scores = [item for sublist in gathered_scores for item in sublist]
        labels = [item for sublist in gathered_labels for item in sublist]
    else:
        scores = list(local_metrics["scores"])
        labels = list(local_metrics["labels"])

    total_loss = float(numeric[0].item())
    total_correct = float(numeric[1].item())
    total_samples = max(int(numeric[2].item()), 1)
    has_both = len(set(labels)) > 1

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
        "auroc": roc_auc_score(labels, scores) if has_both else 0.0,
        "auprc": average_precision_score(labels, scores) if has_both else 0.0,
    }


def _broadcast_rus(
    rus_tensors: Dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        "U": rus_tensors["U"].unsqueeze(0).expand(batch_size, -1, -1).to(device, non_blocking=True),
        "R": rus_tensors["R"].unsqueeze(0).expand(batch_size, -1, -1, -1).to(device, non_blocking=True),
        "S": rus_tensors["S"].unsqueeze(0).expand(batch_size, -1, -1, -1).to(device, non_blocking=True),
    }


def _build_model(args: SimpleNamespace, device: torch.device):
    from deterioration_dataloader import FEATURE_NAMES, NOTE_EMB_DIM
    from model.trus_moe_multimodal import MultimodalTRUSMoEModel

    modality_configs = [
        {
            "input_dim": len(FEATURE_NAMES),
            "num_layers": args.modality_encoder_layers,
            "nhead": args.nhead,
            "d_ff": args.d_ff,
            "use_cnn": args.use_cnn_encoders,
        },
        {
            "input_dim": NOTE_EMB_DIM,
            "num_layers": args.modality_encoder_layers,
            "nhead": args.nhead,
            "d_ff": args.d_ff,
            "use_cnn": args.use_cnn_encoders,
        },
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
        use_checkpoint=args.use_gradient_checkpointing,
    )
    return model.to(device)


def _wrap_model(model, args: SimpleNamespace, device: torch.device, local_rank: int):
    if not _is_distributed():
        return model

    strategy = args.strategy.lower()
    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

        auto_wrap = None
        policy_fn = getattr(type(model), "fsdp_auto_wrap_policy", None)
        if callable(policy_fn):
            auto_wrap = policy_fn()

        mixed_precision = None
        if args.use_mixed_precision:
            mixed_precision = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.float32,
            )

        return FSDP(
            model,
            auto_wrap_policy=auto_wrap,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )

    from torch.nn.parallel import DistributedDataParallel as DDP

    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )


def _compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    aux_outputs: Sequence[Dict[str, torch.Tensor]],
    rus_batch: Dict[str, torch.Tensor],
    args: SimpleNamespace,
    class_weight: torch.Tensor | None,
) -> torch.Tensor:
    import torch.nn.functional as F
    from model.trus_moe_model import calculate_load_balancing_loss, calculate_rus_losses

    ce = F.cross_entropy(logits.float(), labels, weight=class_weight, reduction="none")
    task_loss = (ce * weights).sum() / (weights.sum() + args.epsilon_loss)

    total_rus = torch.tensor(0.0, device=logits.device)
    total_load = torch.tensor(0.0, device=logits.device)
    synergy_idx = set(range(args.moe_num_synergy_experts))

    for aux in aux_outputs:
        l_u, l_r, l_s = calculate_rus_losses(
            aux["gating_probs"],
            rus_batch,
            synergy_idx,
            args.threshold_u,
            args.threshold_r,
            args.threshold_s,
            args.lambda_u,
            args.lambda_r,
            args.lambda_s,
            args.epsilon_loss,
        )
        total_rus += l_u + l_r + l_s
        total_load += calculate_load_balancing_loss(
            aux["gating_probs"],
            aux["expert_indices"],
            args.moe_k,
            args.lambda_load,
        )

    if aux_outputs:
        total_rus /= len(aux_outputs)
        total_load /= len(aux_outputs)

    return task_loss + total_rus + total_load


def _run_one_epoch(
    model,
    dataloader,
    optimizer,
    args: SimpleNamespace,
    rus_tensors: Dict[str, torch.Tensor],
    device: torch.device,
    class_weight: torch.Tensor | None,
    train: bool,
    epoch: int = 0,
    log_interval: int = 20,
    world_size: int = 1,
    global_batch_offset: int = 0,
    mlflow_log: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Run one training or validation epoch.

    Returns
    -------
    epoch_metrics : aggregated loss / accuracy / AUROC / AUPRC
    timing        : num_batches, total_batch_seconds, avg_sec_per_batch
    """
    model.train(mode=train)
    rank = _distributed_info()[0]
    phase = "train" if train else "val"
    try:
        num_batches = len(dataloader)
    except TypeError:
        num_batches = None  # IterableDataset has no len()

    local_metrics = {
        "loss_sum": 0.0,
        "correct": 0.0,
        "n_samples": 0.0,
        "scores": [],
        "labels": [],
    }
    use_amp = bool(args.use_mixed_precision) and torch.cuda.is_available()
    autocast_dtype = torch.bfloat16

    # Throughput tracking
    window_t0 = time.time()
    window_batches = 0
    total_batch_seconds = 0.0
    num_batches_done = 0

    for batch_idx, (ts, notes, note_mask, labels, weights) in enumerate(dataloader):
        batch_t0 = time.time()
        batch_size = int(labels.shape[0])
        ts = ts.to(device, non_blocking=True)
        notes = notes.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        weights = weights.float().to(device, non_blocking=True)
        rus_batch = _broadcast_rus(rus_tensors, batch_size, device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
                logits, aux_outputs = model([ts, notes], rus_batch)
                loss = _compute_loss(logits, labels, weights, aux_outputs, rus_batch, args, class_weight)

            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    print(f"  Warning: NaN/Inf loss at batch {batch_idx}, skipping")
                continue

            if train:
                loss.backward()
                if args.clip_grad_norm > 0:
                    if _is_distributed() and args.strategy.lower() == "fsdp":
                        model.clip_grad_norm_(args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

        logits_for_metrics = logits.detach().float()
        # Clamp NaN logits to zero before softmax to prevent NaN propagation into metrics
        logits_for_metrics = torch.nan_to_num(logits_for_metrics, nan=0.0)
        probs = torch.softmax(logits_for_metrics, dim=1)[:, 1]
        preds = torch.argmax(logits_for_metrics, dim=1)

        local_metrics["loss_sum"] += float(loss.item()) * batch_size
        local_metrics["correct"] += float((preds == labels).sum().item())
        local_metrics["n_samples"] += batch_size
        local_metrics["scores"].extend(probs.cpu().tolist())
        local_metrics["labels"].extend(labels.cpu().tolist())

        batch_elapsed = time.time() - batch_t0
        total_batch_seconds += batch_elapsed
        window_batches += 1
        num_batches_done += 1

        if rank == 0 and (batch_idx + 1) % log_interval == 0:
            running_loss = local_metrics["loss_sum"] / local_metrics["n_samples"]
            running_acc = local_metrics["correct"] / local_metrics["n_samples"]
            window_elapsed = time.time() - window_t0
            sec_per_batch = window_elapsed / max(window_batches, 1)
            global_samples_per_sec = world_size * batch_size / max(sec_per_batch, 1e-9)
            global_step = global_batch_offset + batch_idx + 1

            print(
                f"  [{phase}] epoch {epoch + 1} | "
                f"batch {batch_idx + 1}/{num_batches} | "
                f"loss={running_loss:.4f} acc={running_acc:.4f} | "
                f"sec/batch={sec_per_batch:.3f} samples/sec={global_samples_per_sec:.0f}"
            )

            if mlflow_log:
                import mlflow as _mlflow
                _mlflow.log_metrics(
                    {
                        f"{phase}_batch_loss": running_loss,
                        f"{phase}_sec_per_batch": sec_per_batch,
                        f"{phase}_samples_per_sec": global_samples_per_sec,
                    },
                    step=global_step,
                )

            # Reset throughput window
            window_t0 = time.time()
            window_batches = 0

    if rank == 0:
        final_loss = local_metrics["loss_sum"] / max(local_metrics["n_samples"], 1)
        final_acc = local_metrics["correct"] / max(local_metrics["n_samples"], 1)
        print(
            f"  [{phase}] epoch {epoch + 1} done | "
            f"{int(local_metrics['n_samples'])} samples | "
            f"loss={final_loss:.4f} acc={final_acc:.4f}"
        )

    epoch_metrics = _aggregate_epoch_metrics(local_metrics)
    timing = {
        "num_batches": num_batches_done,
        "total_batch_seconds": total_batch_seconds,
        "avg_sec_per_batch": total_batch_seconds / max(num_batches_done, 1),
    }
    return epoch_metrics, timing


def train_worker(
    config: Dict,
    rus_tensors_cpu: Dict[str, torch.Tensor],
    cache_dir: str,
    output_dir: str,
) -> Dict[str, float]:
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("GPU compute is required for distributed deterioration training.")

    rank, local_rank, world_size = _setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = SimpleNamespace(**config)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    # ── MLflow setup (rank 0 only) ────────────────────────────────────────────────────
    mlflow_log = False
    if rank == 0:
        import mlflow
        # set_experiment must be called in the notebook before .distributed() —
        # the subprocess inherits the active experiment from the parent process.
        num_nodes = max(1, world_size // max(torch.cuda.device_count(), 1))
        mlflow.start_run(run_name=f"{args.strategy}_{world_size}gpu")
        mlflow_log = True
        # ── Infrastructure params ─────────────────────────────────────────────────────
        mlflow.log_params({
            "num_gpus": world_size,
            "num_nodes": num_nodes,
            "world_size": world_size,
            "parallel_strategy": args.strategy,
        })
        # ── Full config params ────────────────────────────────────────────────────────
        for k, v in config.items():
            if k == "mlflow_experiment":
                continue
            if isinstance(v, (list, tuple)):
                v = ",".join(str(x) for x in v)
            try:
                mlflow.log_param(k, str(v))
            except Exception:
                pass

    train_shards = _rank_shards(_list_shards(cache_dir, "train"), rank, world_size)
    val_shards = _rank_shards(_list_shards(cache_dir, "val"), rank, world_size)

    model = _build_model(args, device)
    model = _wrap_model(model, args, device, local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    class_weight = None
    pos_weight = float(getattr(args, "pos_weight", 1.0) or 1.0)
    if pos_weight != 1.0:
        class_weight = torch.tensor([1.0, pos_weight], device=device)

    best_auroc = -math.inf
    best_metrics = None
    patience_counter = 0
    local_output_dir = _ensure_dir(output_dir)
    global_train_batches = 0  # cross-epoch step counter for throughput time series

    try:
        for epoch in range(args.epochs):
            epoch_t0 = time.time()
            train_dataset = CachedShardDataset(
                train_shards,
                shuffle=True,
                seed=args.seed + rank,
                epoch=epoch,
            )
            val_dataset = CachedShardDataset(
                val_shards,
                shuffle=False,
                seed=args.seed + rank,
                epoch=epoch,
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                collate_fn=collate_cached_batch,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                collate_fn=collate_cached_batch,
            )

            train_t0 = time.time()
            train_metrics, train_timing = _run_one_epoch(
                model,
                train_loader,
                optimizer,
                args,
                rus_tensors_cpu,
                device,
                class_weight,
                train=True,
                epoch=epoch,
                world_size=world_size,
                global_batch_offset=global_train_batches,
                mlflow_log=mlflow_log,
            )
            train_seconds = time.time() - train_t0
            global_train_batches += train_timing["num_batches"]

            val_metrics, _val_timing = _run_one_epoch(
                model,
                val_loader,
                optimizer,
                args,
                rus_tensors_cpu,
                device,
                class_weight,
                train=False,
                epoch=epoch,
                world_size=world_size,
                global_batch_offset=0,
                mlflow_log=mlflow_log,
            )
            epoch_seconds = time.time() - epoch_t0
            scheduler.step()

            if rank == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"train loss={train_metrics['loss']:.4f} auroc={train_metrics['auroc']:.4f} | "
                    f"val loss={val_metrics['loss']:.4f} auroc={val_metrics['auroc']:.4f} | "
                    f"epoch={epoch_seconds:.1f}s"
                )

                if mlflow_log:
                    import mlflow
                    mlflow.log_metrics(
                        {
                            "train_loss": train_metrics["loss"],
                            "train_auroc": train_metrics["auroc"],
                            "train_auprc": train_metrics["auprc"],
                            "train_acc": train_metrics["acc"],
                            "val_loss": val_metrics["loss"],
                            "val_auroc": val_metrics["auroc"],
                            "val_auprc": val_metrics["auprc"],
                            "val_acc": val_metrics["acc"],
                            "epoch_seconds": epoch_seconds,
                            "train_seconds": train_seconds,
                            "avg_sec_per_batch": train_timing["avg_sec_per_batch"],
                            "num_train_batches": float(train_timing["num_batches"]),
                        },
                        step=epoch + 1,
                    )

                if val_metrics["auroc"] > best_auroc:
                    best_auroc = val_metrics["auroc"]
                    best_metrics = {
                        "epoch": epoch,
                        "train": train_metrics,
                        "val": val_metrics,
                        "strategy": args.strategy,
                        "world_size": world_size,
                    }
                    patience_counter = 0

                    if _is_distributed() and args.strategy.lower() == "fsdp":
                        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
                        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
                            state_dict = model.state_dict()
                    elif _is_distributed():
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()

                    torch.save(
                        {
                            "epoch": epoch,
                            "config": config,
                            "model_state_dict": state_dict,
                            "best_metrics": best_metrics,
                        },
                        os.path.join(local_output_dir, "best_model.pt"),
                    )
                    _write_json(os.path.join(local_output_dir, "best_metrics.json"), best_metrics)
                else:
                    patience_counter += 1

            if _is_distributed():
                stop_tensor = torch.tensor(
                    [1 if patience_counter >= args.patience else 0],
                    device=device,
                    dtype=torch.int32,
                )
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())
            else:
                should_stop = patience_counter >= args.patience

            if should_stop:
                if rank == 0:
                    print(f"Early stopping after {args.patience} epochs without improvement")
                break

        if rank == 0 and best_metrics is not None:
            print(f"Best validation AUROC: {best_metrics['val']['auroc']:.4f}")
            if mlflow_log:
                import mlflow
                mlflow.log_metrics({
                    "best_val_auroc": best_metrics["val"]["auroc"],
                    "best_val_auprc": best_metrics["val"]["auprc"],
                    "best_epoch": float(best_metrics["epoch"] + 1),
                })

    finally:
        if rank == 0 and mlflow_log:
            import mlflow
            mlflow.end_run()

    _cleanup_distributed()
    return best_metrics or {}


def launch(
    spark,
    strategy: str = "ddp",
    cache_dir: str = DEFAULT_CACHE_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    rus_path: str = DEFAULT_RUS_PATH,
    num_processes: int = 4,
    local_mode: bool = True,
    prepare_cache_first: bool = True,
    overwrite_cache: bool = False,
    materialize_batch_size: int = 256,
    materialize_partition_size: int = 10000,
    examples_per_shard: int = 4096,
    prepare_test: bool = False,
    run_name: str | None = None,
    **overrides,
):
    """Launch Databricks multi-GPU training with TorchDistributor.

    DDP is the recommended default for the current deterioration TRUS-MoE setup.
    Switch to ``strategy='fsdp'`` only when a larger configuration no longer fits
    in one GPU's memory.
    """
    from pyspark.ml.torch.distributor import TorchDistributor

    strategy = strategy.lower()
    if strategy not in {"ddp", "fsdp"}:
        raise ValueError("strategy must be 'ddp' or 'fsdp'")

    config = {**DEFAULTS, **overrides}
    config["strategy"] = strategy
    if strategy == "fsdp" and "use_gradient_checkpointing" not in overrides:
        config["use_gradient_checkpointing"] = True

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    run_name = run_name or _default_run_name(strategy)

    if prepare_cache_first:
        manifests = prepare_training_cache(
            spark,
            cache_dir=cache_dir,
            materialize_batch_size=materialize_batch_size,
            partition_size=materialize_partition_size,
            examples_per_shard=examples_per_shard,
            include_notes=True,
            overwrite=overwrite_cache,
            prepare_test=prepare_test,
            seed=config["seed"],
            num_workers=config["num_workers"],
        )
        if manifests["train"]["num_shards"] < num_processes:
            raise ValueError(
                "The training cache has fewer shards than requested processes. "
                f"num_shards={manifests['train']['num_shards']}, num_processes={num_processes}. "
                "Decrease examples_per_shard or reduce num_processes."
            )

    rus_tensors = load_rus_tensors(rus_path, config["modality_names"], config["seq_len"])

    local_output_dir = _ensure_dir(output_dir)
    run_output_dir = os.path.join(local_output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    _write_json(os.path.join(run_output_dir, "config.json"), config)

    print(
        f"Launching Databricks {strategy.upper()} training with TorchDistributor | "
        f"num_processes={num_processes}, local_mode={local_mode}, cache_dir={cache_dir}"
    )

    distributor = TorchDistributor(
        num_processes=num_processes,
        local_mode=local_mode,
        use_gpu=True,
    )
    return distributor.run(train_worker, config, rus_tensors, cache_dir, run_output_dir)


def launch_ddp(spark, **kwargs):
    return launch(spark, strategy="ddp", **kwargs)


def launch_fsdp(spark, **kwargs):
    return launch(spark, strategy="fsdp", **kwargs)


if __name__ == "__main__":
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    launch_ddp(spark)
