import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snowflake.snowpark import Session
import importlib
import xgb_deterioration_baseline as B

token_path = os.environ.get('SNOWFLAKE_TOKEN_FILE_PATH', '/snowflake/session/token')
with open(token_path) as f:
    token = f.read().strip()

session = Session.builder.configs({
    "account": os.environ.get("SNOWFLAKE_ACCOUNT", "dj34030"),
    "host": os.environ.get("SNOWFLAKE_HOST", "dj34030.snowflakecomputing.com"),
    "authenticator": "oauth",
    "token": token,
    "warehouse": "DEIDENTIFIED_LOAD_MEDIUM",
    "role": "DS_ROLE",
    "database": "TEST",
    "schema": "SILVER",
}).create()
importlib.reload(B)
xgb_clf = B.main(session)
print("Done — xgb_clf:", xgb_clf)
