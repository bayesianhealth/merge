# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # Multi-GPU Training
# MAGIC
# MAGIC **DDP** the default **FSDP** for large models that don't fit on a single GPU.
# MAGIC
# MAGIC Prerequisites:
# MAGIC - Precomputed RUS file present at `results/deterioration/`

# COMMAND ----------

# Can run this 
dbutils.library.restartPython()


# COMMAND ----------

# DBTITLE 1,Setup — sys.path and Spark check
import os, sys

_DATABRICKS_DIR = "/Workspace/Users/patrick.kasl@bayesianhealth.com/merge/mimiciv/databricks"
_PROJECT_DIR    = "/Workspace/Users/patrick.kasl@bayesianhealth.com/merge/mimiciv"
_REPO_DIR       = "/Workspace/Users/patrick.kasl@bayesianhealth.com/merge"

for _p in (_DATABRICKS_DIR, _PROJECT_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Verify the active Spark session is accessible
print(f"Spark version : {spark.version}")
print(f"DATABRICKS_DIR: {_DATABRICKS_DIR}")

# COMMAND ----------

# DBTITLE 1,Configuration header
# MAGIC %md
# MAGIC ## Configuration
# MAGIC

# COMMAND ----------

# DBTITLE 1,Config — infrastructure, paths, and training overrides
# ── Infrastructure ────────────────────────────────────────────────────────────────────────────
NUM_PROCESSES   = 8          # Number of GPU workers
LOCAL_MODE      = True       # True: all workers on the same node; False: multi-node
STRATEGY        = "ddp"      # "ddp" (default) or "fsdp" (large-model memory scaling)

# ── Paths ────────────────────────────────────────────────────────────────────────────
CACHE_DIR       = "dbfs:/tmp/merge/mimiciv/deterioration_multigpu_cache"
OUTPUT_DIR      = "dbfs:/tmp/merge/mimiciv/deterioration_multigpu_runs"
RUS_PATH        = os.path.join(_PROJECT_DIR, "results", "deterioration",
                               "rus_multimodal_all_seq48_lags6_timesteppool_10k_seed42.npy")

# ── Cache materialisation ───────────────────────────────────────────────────────────────────────
# Set OVERWRITE_CACHE=True to re-read from Delta and rebuild the shard files.
# Leave False to reuse an existing cache (much faster on repeated runs).
OVERWRITE_CACHE       = False
EXAMPLES_PER_SHARD    = 4096   # Rows per shard file; must be > batch_size
MATERIALIZE_PARTITION = 10_000 # Rows per Spark round-trip during materialisation

# ── Model / training overrides ───────────────────────────────────────────────────────────────────────
OVERRIDES = dict(
    epochs      = 1,
    batch_size  = 512,
    lr          = 5e-5,
    patience    = 5,
    seed        = 42,
    num_workers = 2,
    # Mixed precision: bfloat16 causes NaN in RUS loss at init; keep False until
    # calculate_rus_losses has epsilon-clamping on gating_probs.
    use_mixed_precision = False,
    # Uncomment to change architecture defaults:
    # d_model   = 1024,
    # nhead     = 16,
    # d_ff      = 4096,
)

print(f"Strategy        : {STRATEGY}")
print(f"Num processes   : {NUM_PROCESSES}")
print(f"Cache dir       : {CACHE_DIR}")
print(f"Output dir      : {OUTPUT_DIR}")
print(f"RUS path exists : {os.path.exists(RUS_PATH)}")

# COMMAND ----------

# DBTITLE 1,Step 1 header — materialise cache
# MAGIC %md
# MAGIC ## Materialise training cache

# COMMAND ----------

# DBTITLE 1,Step 1 — Build DBFS shard cache from Delta tables
from deterioration_distributed_common import prepare_training_cache

manifests = prepare_training_cache(
    spark,
    cache_dir             = "/Volumes/mimiciv/hosp/ml_training_data/deterioration_multigpu_cache",
    materialize_batch_size= 256,
    partition_size        = MATERIALIZE_PARTITION,
    examples_per_shard    = EXAMPLES_PER_SHARD,
    include_notes         = True,
    overwrite             = OVERWRITE_CACHE,
    prepare_test          = False,
    seed                  = OVERRIDES.get("seed", 42),
)

print(f"Train shards : {manifests['train']['num_shards']}  ({manifests['train']['total_examples']} examples)")
print(f"Val   shards : {manifests['val']['num_shards']}  ({manifests['val']['total_examples']} examples)")
assert manifests["train"]["num_shards"] >= NUM_PROCESSES, (
    f"Too few shards ({manifests['train']['num_shards']}) for {NUM_PROCESSES} processes. "
    "Decrease EXAMPLES_PER_SHARD or reduce NUM_PROCESSES."
)

# COMMAND ----------

# DBTITLE 1,Step 2 header — launch training
# MAGIC %md
# MAGIC ## multi-GPU training

# COMMAND ----------

# DBTITLE 1,Step 2a — DDP launch (default)
from serverless_gpu import distributed

_cache_dir = "/Volumes/mimiciv/hosp/ml_training_data/deterioration_multigpu_cache"
_output_dir = "/Volumes/mimiciv/hosp/ml_training_data/deterioration_multigpu_runs"
_run_name = "deterioration_ddp_run"

@distributed(gpus=NUM_PROCESSES, gpu_type="H100")
def run_ddp_training():
    import os
    import sys
    for _p in (
        "/Workspace/Users/patrick.kasl@bayesianhealth.com/merge/mimiciv/databricks",
        "/Workspace/Users/patrick.kasl@bayesianhealth.com/merge/mimiciv",
        "/Workspace/Users/patrick.kasl@bayesianhealth.com/merge",
    ):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from deterioration_distributed_common import (
        DEFAULTS, _ensure_dir, _write_json, load_rus_tensors, train_worker,
    )

    config = {**DEFAULTS, **OVERRIDES, "strategy": "ddp", "batch_size": 512}
    rus_tensors = load_rus_tensors(RUS_PATH, config["modality_names"], config["seq_len"])

    run_output_dir = os.path.join(_ensure_dir(_output_dir), _run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    _write_json(os.path.join(run_output_dir, "config.json"), config)

    return train_worker(config, rus_tensors, _cache_dir, run_output_dir)

import gc, torch, mlflow
gc.collect()
torch.cuda.empty_cache()

# Must be set in the parent process before .distributed() so the experiment
# is inherited by the subprocess (not set inside train_worker).
_mlflow_experiment = OVERRIDES.get("mlflow_experiment", "/Users/patrick.kasl@bayesianhealth.com/deterioration_trus_moe")
mlflow.set_experiment(_mlflow_experiment)
print(f"MLflow experiment : {_mlflow_experiment}")

print(
    f"Launching DDP training (Serverless {NUM_PROCESSES}×H100) | "
    f"num_processes={NUM_PROCESSES}, batch_size={OVERRIDES.get('batch_size', 512)}, "
    f"mixed_precision={OVERRIDES.get('use_mixed_precision', True)}, cache_dir={_cache_dir}"
)
results = run_ddp_training.distributed()
print(results)

# COMMAND ----------

# DBTITLE 1,Step 2c — Single-GPU benchmark (large batch)
# ── Single-GPU training with large batch_size ────────────────────────────────────────────
# Hypothesis: for this model size, single-GPU with batch_size=1024 is faster wall-clock
# than 8-GPU DDP due to zero communication overhead.
import gc, os, sys, time, torch

# Clear all stale GPU memory from previous runs
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

for _p in (
    "/Workspace/Users/patrick.kasl@bayesianhealth.com/merge/mimiciv/databricks",
    "/Workspace/Users/patrick.kasl@bayesianhealth.com/merge/mimiciv",
    "/Workspace/Users/patrick.kasl@bayesianhealth.com/merge",
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib
import deterioration_distributed_common as ddc
ddc = importlib.reload(ddc)
from deterioration_distributed_common import (
    DEFAULTS, _ensure_dir, _write_json, load_rus_tensors, train_worker,
)

_cache_dir = "/Volumes/mimiciv/hosp/ml_training_data/deterioration_multigpu_cache"
_output_dir = "/Volumes/mimiciv/hosp/ml_training_data/deterioration_multigpu_runs"
_run_name = "deterioration_single_gpu_run"

_config = {**DEFAULTS, **OVERRIDES, "strategy": "ddp", "batch_size": 768, "use_mixed_precision": False}
_rus_tensors = load_rus_tensors(RUS_PATH, _config["modality_names"], _config["seq_len"])

_run_output_dir = os.path.join(_ensure_dir(_output_dir), _run_name)
os.makedirs(_run_output_dir, exist_ok=True)
_write_json(os.path.join(_run_output_dir, "config.json"), _config)

# Single-process env (WORLD_SIZE=1 skips dist.init_process_group)
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

print(
    f"Single-GPU training | GPU: {torch.cuda.get_device_name(0)} | "
    f"batch_size={_config['batch_size']}, epochs={_config['epochs']}"
)
t0 = time.time()
results = train_worker(_config, _rus_tensors, _cache_dir, _run_output_dir)
elapsed = time.time() - t0
print(f"\nTraining complete in {elapsed/60:.1f} min")
print(results)

# COMMAND ----------

# DBTITLE 1,Step 2b — FSDP launch (large-model fallback)
# ── FSDP (use when a larger model no longer fits on one GPU) ────────────────────────────
import train_deterioration_distributed as td_fsdp

results = td_fsdp.launch(
    spark,
    num_processes         = NUM_PROCESSES,
    local_mode            = LOCAL_MODE,
    cache_dir             = CACHE_DIR,
    output_dir            = OUTPUT_DIR,
    rus_path              = RUS_PATH,
    prepare_cache_first   = False,
    overwrite_cache       = False,
    run_name              = "deterioration_fsdp_run",
    **OVERRIDES,
)
print(results)

# COMMAND ----------

# DBTITLE 1,Step 3 header — inspect results
# MAGIC %md
# MAGIC ## Step 3 — Inspect results

# COMMAND ----------

# DBTITLE 1,Step 3 — Load and print best checkpoint metrics
import json, torch
from deterioration_distributed_common import _to_local_path

run_name    = "deterioration_ddp_run"   # change to fsdp_run if you ran FSDP
run_dir     = _to_local_path(f"{OUTPUT_DIR}/{run_name}")
metrics_path = os.path.join(run_dir, "best_metrics.json")
ckpt_path    = os.path.join(run_dir, "best_model.pt")

if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        best = json.load(f)
    print(f"Best epoch   : {best['epoch'] + 1}")
    print(f"Val  AUROC   : {best['val']['auroc']:.4f}")
    print(f"Val  AUPRC   : {best['val']['auprc']:.4f}")
    print(f"Val  loss    : {best['val']['loss']:.4f}")
    print(f"Train AUROC  : {best['train']['auroc']:.4f}")
    print(f"Strategy     : {best.get('strategy', 'ddp')}")
    print(f"World size   : {best.get('world_size', NUM_PROCESSES)}")
else:
    print(f"No metrics file found at: {metrics_path}")

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    keys = list(ckpt["model_state_dict"].keys())
    print(f"\nCheckpoint keys (first 5): {keys[:5]}")
    print(f"Checkpoint epoch: {ckpt['epoch']}")
