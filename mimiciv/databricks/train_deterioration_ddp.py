from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from deterioration_distributed_common import DEFAULTS, launch_ddp

LARGE_DEFAULTS = {**DEFAULTS, "strategy": "ddp", "use_gradient_checkpointing": False}


def launch(spark, **overrides):
    return launch_ddp(spark, **overrides)


if __name__ == "__main__":
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    launch(spark)
