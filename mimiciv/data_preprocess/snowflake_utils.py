import snowflake.connector
import os
import pandas as pd


def get_connection():
    token_path = os.environ.get("SNOWFLAKE_TOKEN_FILE_PATH", "/snowflake/session/token")
    with open(token_path) as f:
        token = f.read().strip()

    conn = snowflake.connector.connect(
        host=os.environ.get("SNOWFLAKE_HOST"),
        account=os.environ.get("SNOWFLAKE_ACCOUNT"),
        token=token,
        authenticator="oauth",
        database="MIMICIV",
    )
    return conn


def get_snowpark_session(conn=None):
    """Build a Snowpark Session from the (token/oauth) python-connector connection.

    Avoids relying on a named connection ('default'), which is not configured in
    all runtimes (e.g. a plain terminal). Reuses get_connection() so auth is
    identical to the rest of the pipeline.
    """
    from snowflake.snowpark import Session
    if conn is None:
        conn = get_connection()
    return Session.builder.configs({"connection": conn}).create()


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
