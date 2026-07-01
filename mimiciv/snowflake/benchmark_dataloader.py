import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MERGE = os.path.dirname(os.path.dirname(_HERE))
for p in (_MERGE, os.path.dirname(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


DEFAULTS = dict(
    batch_size=512,
    seq_len=48,
    shuffle=True,
    max_batches=200,
    warmup_batches=5,
    num_gpus=4,
    num_nodes=1,
)


def bench_func():
    """Benchmark function executed inside each distributed worker."""
    import time
    import numpy as np

    from snowflake.ml.modeling.distributors.pytorch import get_context

    context = get_context()
    rank = context.get_rank()
    world_size = context.get_world_size()
    hyper_params = context.get_hyper_params()

    batch_size = int(float(hyper_params.get("batch_size", 512)))
    max_batches = int(float(hyper_params.get("max_batches", 200)))
    warmup_batches = int(float(hyper_params.get("warmup_batches", 5)))
    shuffle = str(hyper_params.get("shuffle", "True")).lower() in ("true", "1", "yes")

    from deterioration_dataconnector import FLAT_COLUMNS, LABEL_COL, WEIGHT_COL, flat_to_ts
    from deterioration_config import SEQ_LEN

    dataset_map = context.get_dataset_map()
    if not (dataset_map and "train" in dataset_map):
        if rank == 0:
            print("ERROR: no ShardedDataConnector dataset_map['train'] available. "
                  "Launch with ts_only data connectors.")
        return

    if rank == 0:
        print(f"Data-loader benchmark: world_size={world_size}, batch_size={batch_size}, "
              f"max_batches={max_batches}, warmup={warmup_batches}")
        print(f"  FLAT_COLUMNS={len(FLAT_COLUMNS)} columns, SEQ_LEN={SEQ_LEN}")

    shard = dataset_map["train"].get_shard()
    ds = shard.to_torch_dataset(batch_size=batch_size, shuffle=shuffle)

    t_fetch = 0.0
    t_assemble = 0.0
    t_tensor = 0.0
    n_batches = 0
    n_rows = 0

    wall_start = None  # set after warmup
    it = iter(ds)

    while True:
        if max_batches and n_batches >= max_batches + warmup_batches:
            break

        # --- fetch stage ---
        t = time.time()
        try:
            batch = next(it)
        except StopIteration:
            break
        fetch_dt = time.time() - t

        # --- assemble stage (the 1,488-column np.stack) ---
        t = time.time()
        flat = np.stack([np.asarray(batch[c]).reshape(-1) for c in FLAT_COLUMNS], axis=1)
        assemble_dt = time.time() - t

        # --- tensor stage (reshape to (B, 48, 31) + label/weight extraction) ---
        t = time.time()
        import torch
        ts = torch.from_numpy(flat_to_ts(flat))
        b = ts.shape[0]
        labels = torch.from_numpy(np.asarray(batch[LABEL_COL]).reshape(-1)).float()
        weights = torch.from_numpy(np.asarray(batch[WEIGHT_COL]).reshape(-1)).float()
        tensor_dt = time.time() - t

        # Count only post-warmup batches in the aggregate timing
        if n_batches >= warmup_batches:
            if wall_start is None:
                wall_start = time.time() - (fetch_dt + assemble_dt + tensor_dt)
            t_fetch += fetch_dt
            t_assemble += assemble_dt
            t_tensor += tensor_dt
            n_rows += b

        n_batches += 1

        if rank == 0 and n_batches % 20 == 0:
            timed = max(n_batches - warmup_batches, 1)
            print(f"  [batch {n_batches}] fetch={t_fetch/timed*1000:.1f}ms "
                  f"assemble={t_assemble/timed*1000:.1f}ms tensor={t_tensor/timed*1000:.1f}ms (avg/batch)")

    wall = time.time() - wall_start if wall_start else 0.0
    timed_batches = max(n_batches - warmup_batches, 1)

    if rank == 0:
        per_batch_ms = wall / timed_batches * 1000 if timed_batches else 0.0
        rows_per_sec = n_rows / wall if wall > 0 else 0.0
        print("\n" + "=" * 60)
        print(f"DATA-LOADER BENCHMARK RESULTS (rank 0, per worker)")
        print("=" * 60)
        print(f"  Timed batches      : {timed_batches}")
        print(f"  Rows processed     : {n_rows}")
        print(f"  Wall time          : {wall:.1f}s")
        print(f"  Throughput         : {rows_per_sec:,.0f} rows/sec/worker "
              f"(~{rows_per_sec * world_size:,.0f} rows/sec total)")
        print(f"  Per-batch (total)  : {per_batch_ms:.1f}ms")
        print(f"  Stage breakdown (avg/batch):")
        print(f"    fetch            : {t_fetch/timed_batches*1000:.1f}ms "
              f"({t_fetch/(t_fetch+t_assemble+t_tensor)*100:.0f}%)")
        print(f"    assemble (1,488) : {t_assemble/timed_batches*1000:.1f}ms "
              f"({t_assemble/(t_fetch+t_assemble+t_tensor)*100:.0f}%)")
        print(f"    tensor           : {t_tensor/timed_batches*1000:.1f}ms "
              f"({t_tensor/(t_fetch+t_assemble+t_tensor)*100:.0f}%)")
        print("=" * 60)
        print("Compare 'Per-batch (total)' here against the training loop's "
              "per-batch time. If they match, the loader is the bottleneck.")


def launch(session, rus_path="results/deterioration", **overrides):
    """Launch the distributed data-loader benchmark from a notebook cell.

    Args:
        session: Snowpark session
        **overrides: Override any key in DEFAULTS (batch_size, max_batches,
                     warmup_batches, shuffle, num_gpus, num_nodes)
    """
    from snowflake.ml.modeling.distributors.pytorch import (
        PyTorchDistributor, PyTorchScalingConfig, WorkerResourceConfig,
    )

    cfg = {**DEFAULTS, **overrides}
    num_gpus = cfg.pop("num_gpus")
    num_nodes = cfg.pop("num_nodes")

    # Same TS-only ShardedDataConnector as training
    from deterioration_dataconnector import make_sharded_ts_connector
    dataset_map = {
        "train": make_sharded_ts_connector(session, "train"),
    }
    print("Using ShardedDataConnector (TS-only) for data-loader benchmark")

    scaling_config = PyTorchScalingConfig(
        num_nodes=num_nodes,
        num_workers_per_node=num_gpus,
        resource_requirements_per_worker=WorkerResourceConfig(num_cpus=0, num_gpus=1),
    )

    trainer = PyTorchDistributor(
        train_func=bench_func,
        scaling_config=scaling_config,
    )

    print(f"Launching data-loader benchmark: {num_nodes} node(s), {num_gpus} worker(s)/node "
          f"(world_size={num_nodes * num_gpus}), batch_size={cfg['batch_size']}")

    hyper_params = {k: str(v) for k, v in cfg.items()}
    response = trainer.run(
        dataset_map=dataset_map,
        hyper_params=hyper_params,
    )

    print("Data-loader benchmark complete.")
    return response
