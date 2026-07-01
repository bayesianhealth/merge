import os
import pandas as pd
from pyspark.sql import SparkSession


# Default catalog.  Override with env var if your UC layout differs.
CATALOG = os.environ.get("MIMICIV_CATALOG", "mimiciv")


def get_spark():
    """Return the active SparkSession.

    On Databricks the session is pre-created by the runtime (available as the
    global `spark`). We first try to retrieve that active session; only if it
    doesn't exist do we attempt to create one (which only succeeds when running
    directly on a cluster, not in a subprocess spawned by !bash).
    """
    # Prefer the already-active session (avoids Spark Connect issues in subprocesses)
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    return SparkSession.builder.getOrCreate()


def read_sql(query, spark=None):
    """Execute a SQL query via Spark and return a pandas DataFrame.

    Drop-in replacement for the old snowflake_utils.read_sql().  Column names
    are lowercased for downstream compatibility.

    The query can use fully-qualified table names (catalog.schema.table).
    """
    if spark is None:
        spark = get_spark()
    df = spark.sql(query).toPandas()
    df.columns = df.columns.str.lower()
    return df


def read_table(table_name, spark=None):
    """Read an entire Unity Catalog table into a pandas DataFrame.

    `table_name` should be fully qualified (catalog.schema.table) or just
    schema.table (the catalog set by USE CATALOG / spark conf will be used).
    """
    if spark is None:
        spark = get_spark()
    df = spark.table(table_name).toPandas()
    df.columns = df.columns.str.lower()
    return df
