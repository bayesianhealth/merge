import os
import pandas as pd
import snowflake.connector


def get_connection(database=None, warehouse=None, schema=None):
    """OAuth/token connection, matching the MIMIC-IV pipeline's auth approach.

    A warehouse must be set explicitly: the session this runs in has no default
    warehouse, so every query would otherwise fail with "No active warehouse selected".
    """
    import det_constants as C

    token_path = os.environ.get("SNOWFLAKE_TOKEN_FILE_PATH", "/snowflake/session/token")
    with open(token_path) as f:
        token = f.read().strip()

    conn = snowflake.connector.connect(
        host=os.environ.get("SNOWFLAKE_HOST"),
        account=os.environ.get("SNOWFLAKE_ACCOUNT"),
        token=token,
        authenticator="oauth",
        database=database or C.OUT_DB,
        schema=schema or C.OUT_SCHEMA,
        warehouse=warehouse or C.WAREHOUSE,
    )
    # The OAuth session token ignores the warehouse/database connect params, so set
    # them explicitly; otherwise every query fails with "No active warehouse selected".
    cur = conn.cursor()
    cur.execute(f"USE WAREHOUSE {warehouse or C.WAREHOUSE}")
    cur.execute(f"USE SCHEMA {database or C.OUT_DB}.{schema or C.OUT_SCHEMA}")
    cur.close()
    return conn


def get_snowpark_session(conn=None):
    """Build a Snowpark Session reusing the python-connector connection."""
    from snowflake.snowpark import Session
    if conn is None:
        conn = get_connection()
    return Session.builder.configs({"connection": conn}).create()


def run_sql(sql, conn=None, fetch=False):
    """Execute a single statement. If fetch=True, return rows as a list of tuples."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        cur = conn.cursor()
        cur.execute(sql)
        if fetch:
            return cur.fetchall()
        return None
    finally:
        if close_conn:
            conn.close()


def read_sql(query, conn=None):
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        df = pd.read_sql(query, conn)
        df.columns = df.columns.str.lower()
        return df
    finally:
        if close_conn:
            conn.close()
