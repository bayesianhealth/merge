"""Databricks FSDP launcher for deterioration TRUS-MoE training.

Use this when scaling the deterioration model beyond what fits comfortably on a
single GPU. For the current 77M-parameter class of models, DDP is usually the
faster Databricks path, but this launcher keeps the Snowflake FSDP workflow
available with a Databricks-native TorchDistributor backend.

Notebook usage:
    import sys
    sys.path.insert(0, "/Workspace/Users/patrick.kasl@bayesianhealth.com/merge/mimiciv/databricks")
    import train_deterioration_distributed as td
    td.launch(spark, num_processes=4, local_mode=True)
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from deterioration_distributed_common import DEFAULTS, launch_fsdp

LARGE_DEFAULTS = {**DEFAULTS, "strategy": "fsdp", "use_gradient_checkpointing": True}


def launch(spark, **overrides):
    return launch_fsdp(spark, **overrides)


if __name__ == "__main__":
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    launch(spark)
