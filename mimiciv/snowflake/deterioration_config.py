SRC_DB = "MIMICIV"
HOSP = f"{SRC_DB}.HOSP"
ICU = f"{SRC_DB}.ICU"

OUT_DB = "TEST"
OUT_SCHEMA_NAME = "SILVER"
OUT_SCHEMA = f"{OUT_DB}.{OUT_SCHEMA_NAME}"

SRC_TRANSFERS = (
    '(SELECT "subject_id" AS SUBJECT_ID, "hadm_id" AS HADM_ID, "careunit" AS CAREUNIT, '
    '"intime" AS INTIME, "outtime" AS OUTTIME FROM MIMICIV.HOSP.TRANSFERS)'
)
SRC_ADMISSIONS = (
    '(SELECT "subject_id" AS SUBJECT_ID, "hadm_id" AS HADM_ID, "dischtime" AS DISCHTIME, '
    '"hospital_expire_flag" AS HOSPITAL_EXPIRE_FLAG FROM MIMICIV.HOSP.ADMISSIONS)'
)
SRC_D_LABITEMS = '(SELECT "itemid" AS ITEMID, "label" AS LABEL FROM MIMICIV.HOSP.D_LABITEMS)'
SRC_D_ITEMS = '(SELECT "itemid" AS ITEMID, "label" AS LABEL FROM MIMICIV.ICU.D_ITEMS)'
SRC_LABEVENTS = (
    '(SELECT SUBJECT_ID, HADM_ID, CHARTTIME, ITEMID, VALUENUM FROM MIMICIV.HOSP.LABEVENTS)'
)
SRC_CHARTEVENTS = (
    '(SELECT SUBJECT_ID, HADM_ID, CHARTTIME, ITEMID, VALUENUM FROM MIMICIV.ICU.CHARTEVENTS)'
)

TS_TABLE = f"{OUT_SCHEMA}.DETERIORATION_TRAINING_DATA"
NOTES_TABLE = f"{OUT_SCHEMA}.DETERIORATION_NOTE_EMBEDDINGS"
MULTIMODAL_TABLE = f"{OUT_SCHEMA}.DETERIORATION_MULTIMODAL"
RADIOLOGY_TABLE = f"{OUT_SCHEMA}.DETERIORATION_RADIOLOGY"
SCALER_TABLE = f"{OUT_SCHEMA}.DETERIORATION_SCALER"

SOURCE_STAGE = '@"TEST"."SILVER"."UDTF_FEATURE_STAGE"'
RADIOLOGY_FILE = f"{SOURCE_STAGE}/radiology.csv.gz"
BIOBERT_STAGE_DIR = f"{SOURCE_STAGE}/biobert-v1.1"

ICU_UNITS = [
    "Medical Intensive Care Unit (MICU)",
    "Surgical Intensive Care Unit (SICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Coronary Care Unit (CCU)",
    "Trauma SICU (TSICU)",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)",
]

OBSERVATION_UNITS = [
    "Emergency Department Observation",
    "Observation",
]

STEPDOWN_UNITS = [
    "Hematology/Oncology Intermediate",
    "Medicine/Cardiology Intermediate",
    "Cardiology Surgery Intermediate",
    "Neuro Intermediate",
    "Surgery/Vascular/Intermediate",
    "Neuro Stepdown",
    "Surgical Intermediate",
]

GENERAL_UNITS = [
    "Medicine", "Med/Surg", "Medicine/Cardiology", "Neurology",
    "Transplant", "Hematology/Oncology", "Vascular", "Med/Surg/GYN",
    "Surgery/Trauma", "Med/Surg/Trauma", "Surgery", "Cardiac Surgery",
    "Obstetrics (Postpartum & Antepartum)", "Psychiatry",
    "Medical/Surgical (Gynecology)", "Surgery/Pancreatic/Biliary/Bariatric",
    "Obstetrics Postpartum", "Cardiology", "Thoracic Surgery",
    "Obstetrics Antepartum", "Oncology",
]

ELIGIBLE_LEVELS = ("general", "stepdown", "observation")

# --- The same 30 labs+vitals as the MERGE IHM pipeline ---
LAB_EVENT_LIST = [
    'Glucose', 'Potassium', 'Sodium', 'Chloride', 'Creatinine',
    'Urea Nitrogen', 'Bicarbonate', 'Anion Gap', 'Hemoglobin', 'Hematocrit',
    'Magnesium', 'Platelet Count', 'Phosphate', 'White Blood Cells',
    'Calcium, Total', 'MCH', 'Red Blood Cells', 'MCHC', 'MCV', 'RDW',
    'Neutrophils', 'Vancomycin',
]

VITAL_EVENT_LIST = [
    'Heart Rate', 'Non Invasive Blood Pressure systolic',
    'Non Invasive Blood Pressure diastolic', 'Non Invasive Blood Pressure mean',
    'Respiratory Rate', 'O2 saturation pulseoxymetry',
    'GCS - Verbal Response', 'GCS - Eye Opening', 'GCS - Motor Response',
]

VITAL_RENAME = {
    'Non Invasive Blood Pressure systolic': 'systolic_bp',
    'Non Invasive Blood Pressure diastolic': 'diastolic_bp',
    'Non Invasive Blood Pressure mean': 'mean_bp',
    'O2 saturation pulseoxymetry': 'o2_saturation',
    'Heart Rate': 'heart_rate',
    'Respiratory Rate': 'respiratory_rate',
    'GCS - Verbal Response': 'gcs_verbal',
    'GCS - Eye Opening': 'gcs_eye',
    'GCS - Motor Response': 'gcs_motor',
}

LAB_RENAME = {name: name.lower().replace(' ', '_').replace(',', '') for name in LAB_EVENT_LIST}
RENAME_MAP = {**LAB_RENAME, **VITAL_RENAME}

# Canonical sorted feature order — identical to deterioration_dataloader.FEATURE_NAMES
FEATURE_NAMES = sorted(list(LAB_RENAME.values()) + list(VITAL_RENAME.values()))

SEQ_LEN = 48
NOTE_EMB_DIM = 768

POS_DECAY_RATE = 1e-3
NEG_DECAY_RATE = 5e-4
MIN_WEIGHT_FLOOR = 0.016
WASHOUT_HOURS = 48

LOOKBACK_HOURS = 48
HORIZON_HOURS = 24

TRAIN_PCT = 70
VAL_PCT = 15 


def flat_col(feature: str, hour: int) -> str:
    """Column name for a feature at a given hour slot (0=oldest, 47=newest)."""
    return f"{feature}__h{hour:02d}"


def all_flat_columns():
    """All 1488 flattened feature column names in (feature, hour) order."""
    return [flat_col(f, h) for f in FEATURE_NAMES for h in range(SEQ_LEN)]


def sql_str_list(values):
    """Render a python list of strings as a SQL IN-list literal body."""
    escaped = [v.replace("'", "''") for v in values]
    return ", ".join(f"'{v}'" for v in escaped)
