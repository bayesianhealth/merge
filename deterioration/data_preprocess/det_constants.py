import os

# ---------------------------------------------------------------------------
# Snowflake locations
# ---------------------------------------------------------------------------
# Sources are the deduplicated SILVER dynamic tables (read-only to CCF_PROD_ROLE).
# Outputs MUST go to CCF_PROD.PUBLIC: CCF_PROD_ROLE has USAGE+SELECT on SILVER but
# no CREATE privilege there, so writing to SILVER fails on the first CREATE TABLE.
SRC_DB = "CCF_PROD"
SRC_SCHEMA = "SILVER"
OUT_DB = "CCF_PROD"
OUT_SCHEMA = "PUBLIC"

# CCF_PROD_ROLE has direct USAGE on SMALL/MEDIUM (and XSMALL/LARGE via DEVELOPER_ROLE).
# The connection sets no warehouse by default, so this must be supplied explicitly.
WAREHOUSE = os.environ.get("DET_WAREHOUSE", "MEDIUM")

# The SILVER dynamic tables carry a soft-delete flag; every read must exclude it.
NOT_DELETED = "NOT _IS_DELETED"


def not_deleted(alias=None):
    """Soft-delete predicate, optionally qualified by a table alias."""
    return f"NOT {alias}._IS_DELETED" if alias else NOT_DELETED

# source tables
T_CDM_T = f"{SRC_DB}.{SRC_SCHEMA}.CDM_T"
T_CDM_S = f"{SRC_DB}.{SRC_SCHEMA}.CDM_S"
T_PAT_ENC = f"{SRC_DB}.{SRC_SCHEMA}.PAT_ENC"
T_ENC_CARE_UNIT = f"{SRC_DB}.{SRC_SCHEMA}.ENC_CARE_UNIT"
T_CARE_UNITS = f"{SRC_DB}.{SRC_SCHEMA}.CARE_UNITS"

# intermediate + output tables (all live in OUT_DB.OUT_SCHEMA)
def _out(name):
    return f"{OUT_DB}.{OUT_SCHEMA}.{name}"

T_LOC = _out("_DET_LOC")              # care-unit intervals w/ level_of_care
T_EVENTS = _out("_DET_EVENTS")        # (enc_id, event_tsp, outcome)
T_ENC_SPLIT = _out("_DET_ENC_SPLIT")  # (enc_id, split)
T_ANCHORS = _out("_DET_ANCHORS")      # (enc_id, anchor_tsp, split)
T_FEATURE_SPEC = _out("DET_FEATURE_SPEC")  # (feature_name, modality, feature_index, mean, std, seq_len)
T_ALL = _out("_DET_ALL")              # all examples w/ split column
T_TRAIN = _out("DET_TRAIN")
T_VAL = _out("DET_VAL")
T_TEST = _out("DET_TEST")

# ---------------------------------------------------------------------------
# Task / window parameters
# ---------------------------------------------------------------------------
SEQ_LEN = 48          # trailing window length, in hourly bins
HORIZON_HOURS = 24    # forward prediction horizon for the label
WASHOUT_HOURS = 48    # sample_weight=0 for anchors within this window after an event
MIN_OBS_IN_WINDOW = 1 # drop anchors whose trailing 48h has fewer than this many obs

# sample-weight decay (matches templatized_training_notebook.calculate_sample_weights)
POS_DECAY_RATE = 1e-3
NEG_DECAY_RATE = 5e-4
MIN_WEIGHT_FLOOR = 0.016

# ---------------------------------------------------------------------------
# Level-of-care semantics (compared lowercase)
# ---------------------------------------------------------------------------
WARD_LOCS = ["general", "observation", "stepdown"]   # cohort / labeling ward
ICU_LOC = "icu"
# a transfer counts as "unplanned" if the preceding care unit was a ward
UNPLANNED_PRIOR_LOCS = ["general", "stepdown", "observation"]
# next-unit conditions that qualify a >=24h-equivalent (terminal) unplanned xfer
UNPLANNED_NEXT_LOCS = ["icu", "surgery", "procedure"]
DISCHARGED_SENTINEL = "discharged"
MIN_ICU_STAY_HOURS = 24

# death detection + hospice exclusion
DISCHARGE_FID = "discharge"
DEATH_VALUE_SUBSTR = "Expired"
SERVICE_TYPE_FID = "service_type"
HOSPICE_REGEX = "Hospice|Palliative|Comfort"

# ---------------------------------------------------------------------------
# Feature set: two modalities built from numeric CDM_T FIDs
# (derived from the CDM_T FID frequency profile; numeric-valued FIDs only)
# ---------------------------------------------------------------------------
VITALS_FIDS = [
    "heart_rate",
    "resp_rate",
    "spo2",
    "temperature",
    "nbp_sys",
    "nbp_dias",
    "map",
    "gcs",
    "fio2",
]

LABS_FIDS = [
    "glucose",
    "sodium",
    "potassium",
    "chloride",
    "bun",
    "creatinine",
    "co2",
    "anion_gap",
    "hemoglobin",
    "hematocrit",
    "platelets",
    "wbc",
    "calcium",
    "magnesium",
    "albumin",
    "bilirubin",
    "alt_liver_enzymes",
    "ast_liver_enzymes",
    "alkaline_phosphatase",
    "lactate",
    "inr",
    "neut",
]

# modality membership for each feature (consumed by the DataConnector/model)
MODALITY_OF = {fid: "vitals" for fid in VITALS_FIDS}
MODALITY_OF.update({fid: "labs" for fid in LABS_FIDS})

# canonical, ordered feature list (vitals first, then labs)
FEATURE_FIDS = VITALS_FIDS + LABS_FIDS

# Snowflake-safe column names (FIDs are already valid identifiers, kept uppercase
# in tables). value-array column = <FID>, observed-mask column = <FID>_MASK.
def col_value(fid):
    return fid.upper()

def col_mask(fid):
    return f"{fid.upper()}_MASK"

# ---------------------------------------------------------------------------
# Split (deterministic, by encounter)
# ---------------------------------------------------------------------------
SPLIT_SEED = 42
TRAIN_PCT = 70   # [0,70)  -> train
VAL_PCT = 15     # [70,85) -> val ; [85,100) -> test
