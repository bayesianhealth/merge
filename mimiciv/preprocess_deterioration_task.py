# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
# DBTITLE 1,Deterioration Task Preprocessing
# MAGIC %md
# MAGIC # Deterioration Task Preprocessing
# MAGIC
# MAGIC Generates training data for the **deterioration prediction task** on MIMIC-IV:
# MAGIC - **Population**: Patients on general/stepdown/observation wards
# MAGIC - **Example trigger**: Each new lab or vital measurement while on eligible ward
# MAGIC - **Features**: 30 labs+vitals, 48h lookback resampled to 1h → array[48] per feature
# MAGIC - **Label**: Death OR unplanned ICU transfer within 24h
# MAGIC - **Sample weights**: Exponential decay + 48h washout
# MAGIC - **Output**: Delta table with one row per (hadm_id, tsp), array columns for features

# COMMAND ----------

# DBTITLE 1,Configuration & Imports
from pyspark.sql import functions as F, Window
from pyspark.sql.types import *
import numpy as np

# --- Configuration ---
CATALOG = "mimiciv"
OUTPUT_TABLE = "mimiciv.hosp.deterioration_training_data"

# Careunit -> Level of Care mapping
ICU_UNITS = [
    "Medical Intensive Care Unit (MICU)",
    "Surgical Intensive Care Unit (SICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Coronary Care Unit (CCU)",
    "Trauma SICU (TSICU)",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)",
    "Neuro Intermediate",  # sometimes classified as ICU in MIMIC
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

ELIGIBLE_UNITS = OBSERVATION_UNITS + STEPDOWN_UNITS + GENERAL_UNITS

# Same 30 features as IHM pipeline
LAB_EVENT_LIST = [
    'Glucose', 'Potassium', 'Sodium', 'Chloride', 'Creatinine',
    'Urea Nitrogen', 'Bicarbonate', 'Anion Gap', 'Hemoglobin', 'Hematocrit',
    'Magnesium', 'Platelet Count', 'Phosphate', 'White Blood Cells',
    'Calcium, Total', 'MCH', 'Red Blood Cells', 'MCHC', 'MCV', 'RDW',
    'Neutrophils', 'Vancomycin'
]

VITAL_EVENT_LIST = [
    'Heart Rate', 'Non Invasive Blood Pressure systolic',
    'Non Invasive Blood Pressure diastolic', 'Non Invasive Blood Pressure mean',
    'Respiratory Rate', 'O2 saturation pulseoxymetry',
    'GCS - Verbal Response', 'GCS - Eye Opening', 'GCS - Motor Response'
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

ALL_FEATURE_NAMES = sorted(list(LAB_RENAME.values()) + list(VITAL_RENAME.values()))
print(f"{len(ALL_FEATURE_NAMES)} features: {ALL_FEATURE_NAMES[:5]}...")

# COMMAND ----------

# DBTITLE 1,Step 1: Build event table (ICU transfers + deaths)
# Load transfers table and identify ICU transfer events
transfers = spark.table(f"{CATALOG}.hosp.transfers")

# Tag each transfer row with level of care
transfers_loc = transfers.withColumn(
    "level_of_care",
    F.when(F.col("careunit").isin(ICU_UNITS), "icu")
     .when(F.col("careunit").isin(STEPDOWN_UNITS), "stepdown")
     .when(F.col("careunit").isin(GENERAL_UNITS), "general")
     .when(F.col("careunit").isin(OBSERVATION_UNITS), "observation")
     .otherwise("other")
)

# ICU transfer events: rows where a patient enters ICU from an eligible ward
w_transfer = Window.partitionBy("hadm_id").orderBy("intime")

transfers_with_prev = transfers_loc.withColumn(
    "prev_level_of_care", F.lag("level_of_care").over(w_transfer)
)

icu_transfer_events = (
    transfers_with_prev
    .filter(
        (F.col("level_of_care") == "icu") &
        (F.col("prev_level_of_care").isin("general", "stepdown", "observation"))
    )
    .select(
        F.col("subject_id"),
        F.col("hadm_id"),
        F.col("intime").alias("event_tsp"),
        F.lit("icu_transfer").alias("event_type")
    )
)

# Death events: hospital_expire_flag = 1, event time = dischtime
admissions = spark.table(f"{CATALOG}.hosp.admissions")

death_events = (
    admissions
    .filter(F.col("hospital_expire_flag") == 1)
    .select(
        F.col("subject_id"),
        F.col("hadm_id"),
        F.col("dischtime").alias("event_tsp"),
        F.lit("death").alias("event_type")
    )
)

# Combine all deterioration events
events = icu_transfer_events.unionByName(death_events)

print(f"ICU transfer events: {icu_transfer_events.count()}")
print(f"Death events: {death_events.count()}")
print(f"Total events: {events.count()}")

# COMMAND ----------

# DBTITLE 1,Step 2: Identify eligible ward periods
# Eligible ward periods: time intervals when patient is on general/stepdown/observation
ward_stays = (
    transfers_loc
    .filter(F.col("level_of_care").isin("general", "stepdown", "observation"))
    .select("subject_id", "hadm_id", "intime", "outtime", "careunit", "level_of_care")
)

print(f"Eligible ward stay segments: {ward_stays.count()}")
ward_stays.groupBy("level_of_care").count().show()

# COMMAND ----------

# DBTITLE 1,Step 3: Resolve lab & vital itemids
# Resolve lab itemids (pick most populated per label, same as IHM pipeline)
d_labitems = spark.table(f"{CATALOG}.hosp.d_labitems")
labevents = spark.table(f"{CATALOG}.hosp.labevents")

# For each lab label, find the most-populated itemid
lab_itemid_counts = (
    d_labitems.filter(F.col("label").isin(LAB_EVENT_LIST))
    .join(labevents.groupBy("itemid").count(), on="itemid", how="inner")
    .withColumn("rn", F.row_number().over(Window.partitionBy("label").orderBy(F.desc("count"))))
    .filter(F.col("rn") == 1)
    .select("itemid", "label")
)

# Resolve vital itemids from d_items
d_items = spark.table(f"{CATALOG}.icu.d_items")
chartevents = spark.table(f"{CATALOG}.icu.chartevents")

vital_itemid_counts = (
    d_items.filter(F.col("label").isin(VITAL_EVENT_LIST))
    .join(chartevents.groupBy("itemid").count(), on="itemid", how="inner")
    .withColumn("rn", F.row_number().over(Window.partitionBy("label").orderBy(F.desc("count"))))
    .filter(F.col("rn") == 1)
    .select("itemid", "label")
)

print("Lab itemids:")
lab_itemid_counts.show(truncate=False)
print("Vital itemids:")
vital_itemid_counts.show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Step 4: Collect measurements during eligible ward stays
# Collect lab measurements during eligible ward stays
lab_measurements = (
    labevents
    .join(lab_itemid_counts, on="itemid", how="inner")
    .select(
        F.col("subject_id"), F.col("hadm_id"),
        F.col("charttime").alias("tsp"),
        F.col("label").alias("feature_label"),
        F.col("valuenum").alias("value")
    )
    .filter(F.col("value").isNotNull())
)

# Collect vital measurements during eligible ward stays
# Note: vitals in MIMIC-IV come from chartevents (ICU) but also exist for
# ward patients. We join on hadm_id to the ward_stays time windows.
vital_measurements = (
    chartevents
    .join(vital_itemid_counts, on="itemid", how="inner")
    .select(
        F.col("subject_id"), F.col("hadm_id"),
        F.col("charttime").alias("tsp"),
        F.col("label").alias("feature_label"),
        F.col("valuenum").alias("value")
    )
    .filter(F.col("value").isNotNull())
)

# Combine labs + vitals
all_measurements = lab_measurements.unionByName(vital_measurements)

# Filter to only measurements that fall within eligible ward stay periods
# (interval join: tsp BETWEEN ward_stay.intime AND ward_stay.outtime)
measurements_on_ward = (
    all_measurements.alias("m")
    .join(
        ward_stays.alias("w"),
        on=[
            F.col("m.hadm_id") == F.col("w.hadm_id"),
            F.col("m.tsp").between(F.col("w.intime"), F.col("w.outtime"))
        ],
        how="inner"
    )
    .select(
        F.col("m.subject_id"), F.col("m.hadm_id"),
        F.col("m.tsp"), F.col("m.feature_label"), F.col("m.value")
    )
)

# Rename features to snake_case column names
rename_map = {**LAB_RENAME, **VITAL_RENAME}
measurements_on_ward = measurements_on_ward.withColumn(
    "feature_name",
    F.create_map(*[item for pair in rename_map.items() for item in (F.lit(pair[0]), F.lit(pair[1]))])[F.col("feature_label")]
)

print(f"Total measurements on eligible wards: {measurements_on_ward.count()}")
measurements_on_ward.groupBy("feature_name").count().orderBy(F.desc("count")).show(5)

# COMMAND ----------

# DBTITLE 1,Step 5: Generate observation timestamps (example triggers)
# Each unique (hadm_id, tsp) where a new measurement arrives = one potential example
observation_times = (
    measurements_on_ward
    .select("subject_id", "hadm_id", "tsp")
    .distinct()
)

print(f"Total observation timestamps (potential examples): {observation_times.count()}")

# COMMAND ----------

# DBTITLE 1,Step 6: Build 48h lookback arrays per observation
from pyspark.sql.functions import pandas_udf
import pandas as pd

# For each observation time, collect all measurements in the 48h window before it,
# resample to 1-hour intervals (forward-fill), and produce an array[48] per feature.

# First, join each observation time with all measurements in its 48h lookback window
obs_with_measurements = (
    observation_times.alias("o")
    .join(
        measurements_on_ward.alias("m"),
        on=[
            F.col("o.hadm_id") == F.col("m.hadm_id"),
            F.col("m.tsp").between(
                F.col("o.tsp") - F.expr("INTERVAL 48 HOURS"),
                F.col("o.tsp")
            )
        ],
        how="inner"
    )
    .select(
        F.col("o.subject_id").alias("subject_id"),
        F.col("o.hadm_id").alias("hadm_id"),
        F.col("o.tsp").alias("obs_tsp"),
        F.col("m.tsp").alias("meas_tsp"),
        F.col("m.feature_name"),
        F.col("m.value")
    )
    # Calculate hours-before-observation (0 = observation time, 47 = 47h before)
    .withColumn(
        "hours_before",
        F.floor(
            (F.unix_timestamp("obs_tsp") - F.unix_timestamp("meas_tsp")) / 3600
        ).cast("int")
    )
    .filter((F.col("hours_before") >= 0) & (F.col("hours_before") < 48))
)

print("Observation-measurement join complete")
obs_with_measurements.show(5, truncate=False)

# COMMAND ----------

# DBTITLE 1,Step 7: Pivot and resample into array columns
# For each (hadm_id, obs_tsp, feature_name), take the last value per hour bucket,
# then collect into an array[48] ordered from oldest (index 0) to newest (index 47).

# Deduplicate: if multiple values in same hour bucket, take the last one
w_dedup = Window.partitionBy("hadm_id", "obs_tsp", "feature_name", "hours_before").orderBy(F.desc("meas_tsp"))

deduped = (
    obs_with_measurements
    .withColumn("rn", F.row_number().over(w_dedup))
    .filter(F.col("rn") == 1)
    .drop("rn")
)

# Pivot: one row per (hadm_id, obs_tsp, hours_before), one column per feature
pivoted = (
    deduped
    .groupBy("subject_id", "hadm_id", "obs_tsp", "hours_before")
    .pivot("feature_name", values=ALL_FEATURE_NAMES)
    .agg(F.first("value"))
)

# Now for each (hadm_id, obs_tsp), collect the 48 hour-slots into arrays
# First ensure all 48 slots exist (even if empty), then forward-fill, then collect

# Create a cross join of all observations x all 48 hour slots
obs_keys = observation_times.select("subject_id", "hadm_id", F.col("tsp").alias("obs_tsp"))
hour_slots = spark.range(0, 48).withColumnRenamed("id", "hours_before")

full_grid = obs_keys.crossJoin(hour_slots)

# Left join the pivoted measurements
full_ts = (
    full_grid.alias("g")
    .join(
        pivoted.alias("p"),
        on=[
            F.col("g.hadm_id") == F.col("p.hadm_id"),
            F.col("g.obs_tsp") == F.col("p.obs_tsp"),
            F.col("g.hours_before") == F.col("p.hours_before")
        ],
        how="left"
    )
    .select(
        F.col("g.subject_id"), F.col("g.hadm_id"), F.col("g.obs_tsp"),
        F.col("g.hours_before"),
        *[F.col(f"p.{feat}").alias(feat) for feat in ALL_FEATURE_NAMES]
    )
)

# Forward-fill within each (hadm_id, obs_tsp) ordered by hours_before DESC
# (hours_before=47 is oldest, 0 is newest — fill forward in time)
w_ffill = Window.partitionBy("hadm_id", "obs_tsp").orderBy(F.desc("hours_before")).rowsBetween(Window.unboundedPreceding, Window.currentRow)

for feat in ALL_FEATURE_NAMES:
    full_ts = full_ts.withColumn(feat, F.last(feat, ignorenulls=True).over(w_ffill))

print("Forward-fill complete. Collecting into arrays...")

# COMMAND ----------

# DBTITLE 1,Step 8: Collect arrays per observation
# Collect the 48 values into arrays, ordered from oldest (hour 47) to newest (hour 0)
# So index 0 = 48h ago, index 47 = current observation time
#
# IMPORTANT: collect_list drops nulls, producing empty/short arrays.
# We replace nulls with NaN before collecting to preserve all 48 positions.

array_df = (
    full_ts
    .orderBy("hadm_id", "obs_tsp", F.desc("hours_before"))
    .groupBy("subject_id", "hadm_id", "obs_tsp")
    .agg(
        *[
            F.collect_list(
                F.coalesce(F.col(feat), F.lit(float('nan')))
            ).alias(feat)
            for feat in ALL_FEATURE_NAMES
        ]
    )
)

# Verify array lengths are all 48
sample_lens = array_df.select(
    F.size(F.col(ALL_FEATURE_NAMES[0])).alias("arr_len")
).groupBy("arr_len").count().orderBy("arr_len")
sample_lens.show()

print(f"Examples with array features: {array_df.count()}")

# COMMAND ----------

# DBTITLE 1,Step 9: Assign labels (event within 24h)
# Label = 1 if any deterioration event occurs within 24h of obs_tsp
# Use a left join with events + filter on time window

labeled_df = (
    array_df.alias("a")
    .join(
        events.alias("e"),
        on=[
            F.col("a.hadm_id") == F.col("e.hadm_id"),
            F.col("e.event_tsp").between(
                F.col("a.obs_tsp"),
                F.col("a.obs_tsp") + F.expr("INTERVAL 24 HOURS")
            )
        ],
        how="left"
    )
    .groupBy(
        F.col("a.subject_id"), F.col("a.hadm_id"), F.col("a.obs_tsp"),
        *[F.col(f"a.{feat}") for feat in ALL_FEATURE_NAMES]
    )
    .agg(
        F.max(F.when(F.col("e.event_tsp").isNotNull(), 1).otherwise(0)).alias("label"),
        F.min("e.event_tsp").alias("next_event_tsp")
    )
)

print(f"Label distribution:")
labeled_df.groupBy("label").count().show()

# COMMAND ----------

# DBTITLE 1,Step 10: Compute sample weights
# Sample weight logic from templatized notebook:
# - Positive window (event coming): weight = exp(-pos_decay_rate * t2e_mins)
# - Negative window (no event): weight = exp(-neg_decay_rate * t2end_mins)
# - 48h washout after a prior event: weight = 0

POS_DECAY_RATE = 1e-3
NEG_DECAY_RATE = 5e-4
MIN_WEIGHT_FLOOR = 0.016
WASHOUT_HOURS = 48

# Add time-to-event (minutes) and time-to-end-of-stay
admissions_times = admissions.select("hadm_id", "dischtime")

weighted_df = (
    labeled_df
    .join(admissions_times, on="hadm_id", how="left")
    .withColumn(
        "t2e_mins",
        F.when(
            F.col("next_event_tsp").isNotNull(),
            (F.unix_timestamp("next_event_tsp") - F.unix_timestamp("obs_tsp")) / 60
        )
    )
    .withColumn(
        "t2end_mins",
        (F.unix_timestamp("dischtime") - F.unix_timestamp("obs_tsp")) / 60
    )
)

# Compute raw weight
weighted_df = weighted_df.withColumn(
    "raw_weight",
    F.when(
        F.col("t2e_mins").isNotNull(),
        F.greatest(F.exp(-F.lit(POS_DECAY_RATE) * F.greatest(F.col("t2e_mins"), F.lit(-120))), F.lit(MIN_WEIGHT_FLOOR))
    ).otherwise(
        F.greatest(F.exp(-F.lit(NEG_DECAY_RATE) * F.greatest(F.col("t2end_mins"), F.lit(-120))), F.lit(MIN_WEIGHT_FLOOR))
    )
)

# Apply 48h washout after prior events
# Find the most recent prior event for each observation
w_prev_event = Window.partitionBy("hadm_id").orderBy("obs_tsp")

# Self-join with events to find prior event timestamps
with_prev_event = (
    weighted_df.alias("w")
    .join(
        events.select("hadm_id", F.col("event_tsp").alias("prev_event_tsp")).alias("pe"),
        on=[
            F.col("w.hadm_id") == F.col("pe.hadm_id"),
            F.col("pe.prev_event_tsp") < F.col("w.obs_tsp")
        ],
        how="left"
    )
    .groupBy("w.subject_id", "w.hadm_id", "w.obs_tsp", "w.label", "w.raw_weight",
             *[f"w.{feat}" for feat in ALL_FEATURE_NAMES])
    .agg(F.max("pe.prev_event_tsp").alias("prev_event_tsp"))
)

# Calculate minutes since prior event and apply washout
final_weighted = (
    with_prev_event
    .withColumn(
        "mins_after_event",
        F.when(
            F.col("prev_event_tsp").isNotNull(),
            (F.unix_timestamp(F.col("w.obs_tsp")) - F.unix_timestamp("prev_event_tsp")) / 60
        ).otherwise(F.lit(float('inf')))
    )
    .withColumn(
        "sample_weight",
        F.when(
            F.col("mins_after_event") < WASHOUT_HOURS * 60,
            F.lit(0.0)
        ).otherwise(F.col("w.raw_weight"))
    )
    .select(
        F.col("w.subject_id").alias("subject_id"),
        F.col("w.hadm_id").alias("hadm_id"),
        F.col("w.obs_tsp").alias("tsp"),
        *[F.col(f"w.{feat}").alias(feat) for feat in ALL_FEATURE_NAMES],
        F.col("w.label").alias("label"),
        F.col("sample_weight")
    )
)

print(f"Final examples: {final_weighted.count()}")
final_weighted.select("label", "sample_weight").describe().show()

# COMMAND ----------

# DBTITLE 1,Step 11: Train/Val/Test split by hadm_id
# Split by hadm_id, stratified on whether the admission has any positive label
from pyspark.sql import DataFrame

# Get per-admission label indicator for stratification
admission_labels = (
    final_weighted
    .groupBy("hadm_id")
    .agg(F.max("label").alias("has_event"))
)

# Deterministic split: hash-based on hadm_id
# train=70%, val=15%, test=15%
split_df = admission_labels.withColumn(
    "hash_val", F.abs(F.hash(F.col("hadm_id"))) % 100
).withColumn(
    "split",
    F.when(F.col("hash_val") < 70, "train")
     .when(F.col("hash_val") < 85, "val")
     .otherwise("test")
).select("hadm_id", "split")

# Join split assignment back
final_with_split = final_weighted.join(split_df, on="hadm_id", how="inner")

print("Split distribution:")
final_with_split.groupBy("split").agg(
    F.count("*").alias("n_examples"),
    F.sum("label").alias("n_positive"),
    F.mean("label").alias("prevalence")
).show()

# COMMAND ----------

# DBTITLE 1,Step 12: Normalize features (fit on train)
# Fit StandardScaler on train split (per-feature mean and std across all array elements)
# Then apply to all splits

train_data = final_with_split.filter(F.col("split") == "train")

# Compute element-wise stats by exploding each feature's array
# Filter out NaN values (missing) so they don't skew stats
stats_rows = []
for feat in ALL_FEATURE_NAMES:
    feat_stats = (
        train_data
        .select(F.explode(F.col(feat)).alias("val"))
        .filter(F.col("val").isNotNull() & ~F.isnan(F.col("val")))
        .agg(
            F.mean("val").alias("mean_val"),
            F.stddev("val").alias("std_val")
        )
        .collect()[0]
    )
    stats_rows.append((feat, feat_stats["mean_val"] or 0.0, feat_stats["std_val"] or 1.0))

stats_df = spark.createDataFrame(stats_rows, ["feature", "mean_val", "std_val"])
stats_df.show(truncate=False)

# Collect stats as dict
stats_dict = {row["feature"]: (row["mean_val"], row["std_val"]) for row in stats_df.collect()}

# Normalize each array column: (x - mean) / std
# NaN values (missing) -> 0.0 in normalized space (mean imputation)
normalized_df = final_with_split
for feat in ALL_FEATURE_NAMES:
    mean_val, std_val = stats_dict[feat]
    std_val = std_val if std_val > 0 else 1.0
    normalized_df = normalized_df.withColumn(
        feat,
        F.transform(
            F.col(feat),
            lambda x: F.when(F.isnan(x), F.lit(0.0))
                       .otherwise((x - F.lit(mean_val)) / F.lit(std_val))
        )
    )

print("Normalization complete.")

# COMMAND ----------

# DBTITLE 1,Step 13: Write to Delta table
# Write the final table
normalized_df.write.format("delta").mode("overwrite").saveAsTable(OUTPUT_TABLE)

print(f"Written to {OUTPUT_TABLE}")
print(f"Schema:")
spark.table(OUTPUT_TABLE).printSchema()
print(f"\nRow counts by split:")
spark.table(OUTPUT_TABLE).groupBy("split").count().show()

# COMMAND ----------

OUTPUT_TABLE

# COMMAND ----------

# DBTITLE 1,Step 14: Save scaler for inference
# Save the normalization stats for use during inference
import json

scaler_path = "/Workspace/Users/patrick.kasl@bayesianhealth.com/merge/mimiciv/data/deterioration_scaler.json"
with open(scaler_path, "w") as f:
    json.dump(stats_dict, f, indent=2)

print(f"Scaler saved to {scaler_path}")
print(f"\nDone! Table {OUTPUT_TABLE} is ready for training.")
print(f"Each row has {len(ALL_FEATURE_NAMES)} array[48] feature columns + label + sample_weight + split")

# COMMAND ----------

# DBTITLE 1,Step 15: Build note embeddings table
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import gc

NOTES_OUTPUT_TABLE = "mimiciv.hosp.deterioration_note_embeddings"

# Process embeddings in chunks, writing each chunk to parquet incrementally
emb_path = "/Workspace/Users/patrick.kasl@bayesianhealth.com/merge/mimiciv/data/rad_notes_text_embeddings.parquet"
temp_emb_dir = "/Workspace/Users/patrick.kasl@bayesianhealth.com/merge/mimiciv/data/note_embeddings_flat/"

import os, shutil
if os.path.exists(temp_emb_dir):
    shutil.rmtree(temp_emb_dir)
os.makedirs(temp_emb_dir, exist_ok=True)

CHUNK_SIZE = 15_000
pf = pq.ParquetFile(emb_path)
total_rows = pf.metadata.num_rows
print(f"Total notes in file: {total_rows}")

processed = 0
for i, batch in enumerate(pf.iter_batches(batch_size=CHUNK_SIZE, columns=['hadm_id', 'charttime', 'biobert_embeddings'])):
    chunk_df = batch.to_pandas()
    embeddings = []
    for emb in chunk_df['biobert_embeddings']:
        if emb is not None and len(emb) > 0:
            arr = np.stack([np.array(e, dtype=np.float64) for e in emb])
            embeddings.append((arr.sum(axis=0) / len(emb)).astype(np.float32).tolist())
        else:
            embeddings.append([0.0] * 768)
    
    out_df = pd.DataFrame({
        'hadm_id': chunk_df['hadm_id'].astype('Int64'),
        'charttime': chunk_df['charttime'].dt.floor('us'),  # microsecond precision for Spark
        'note_embedding': embeddings,
    }).dropna(subset=['hadm_id'])
    
    out_df.to_parquet(os.path.join(temp_emb_dir, f"part_{i:04d}.parquet"), index=False, coerce_timestamps='us')
    processed += len(out_df)
    print(f"  Chunk {i}: wrote {len(out_df)} rows (total: {processed})")
    del chunk_df, embeddings, out_df
    gc.collect()

print(f"\nDone. {processed} notes written to {temp_emb_dir}")

# Read all chunks with Spark
notes_spark = spark.read.parquet(temp_emb_dir)
print(f"Notes Spark DF: {notes_spark.count()} rows")
notes_spark.printSchema()

# COMMAND ----------

# DBTITLE 1,Step 16: Join notes to observation windows
# For each (hadm_id, obs_tsp) in our training data, find all notes
# within the 48h lookback window that also fall during ward stay periods.
#
# Join logic:
# 1. note.charttime BETWEEN (obs_tsp - 48h) AND obs_tsp
# 2. note.charttime falls within an eligible ward stay period

# First, get observation times from the training table
obs_from_table = spark.table("training_data").select("hadm_id", F.col("tsp").alias("obs_tsp")).distinct()

# Join notes to observations within 48h lookback
notes_in_window = (
    obs_from_table.alias("o")
    .join(
        notes_spark.alias("n"),
        on=[
            F.col("o.hadm_id") == F.col("n.hadm_id"),
            F.col("n.charttime").between(
                F.col("o.obs_tsp") - F.expr("INTERVAL 48 HOURS"),
                F.col("o.obs_tsp")
            )
        ],
        how="inner"
    )
    .select(
        F.col("o.hadm_id").alias("hadm_id"),
        F.col("o.obs_tsp").alias("obs_tsp"),
        F.col("n.charttime").alias("note_charttime"),
        F.floor(
            (F.unix_timestamp("o.obs_tsp") - F.unix_timestamp("n.charttime")) / 3600
        ).cast("int").alias("hours_before_obs"),
        F.col("n.note_embedding")
    )
)

# Also filter to notes that fall within ward stay periods
notes_on_ward = (
    notes_in_window.alias("ni")
    .join(
        ward_stays.alias("w"),
        on=[
            F.col("ni.hadm_id") == F.col("w.hadm_id"),
            F.col("ni.note_charttime").between(F.col("w.intime"), F.col("w.outtime"))
        ],
        how="inner"
    )
    .select(
        F.col("ni.hadm_id"),
        F.col("ni.obs_tsp"),
        F.col("ni.note_charttime"),
        F.col("ni.hours_before_obs"),
        F.col("ni.note_embedding")
    )
)

print(f"Notes matched to observation windows (on ward): {notes_on_ward.count()}")
notes_on_ward.groupBy("hours_before_obs").count().orderBy("hours_before_obs").show(10)

# COMMAND ----------

# DBTITLE 1,Step 17: Write notes table to Delta
# Write the notes embeddings table
notes_on_ward.write.format("delta").mode("overwrite").saveAsTable(NOTES_OUTPUT_TABLE)

print(f"Written to {NOTES_OUTPUT_TABLE}")
print(f"Schema:")
spark.table(NOTES_OUTPUT_TABLE).printSchema()

# Stats
total_notes = spark.table(NOTES_OUTPUT_TABLE).count()
total_obs = obs_from_table.count()
obs_with_notes = spark.table(NOTES_OUTPUT_TABLE).select("hadm_id", "obs_tsp").distinct().count()

print(f"\nTotal note rows: {total_notes}")
print(f"Total observation timestamps: {total_obs}")
print(f"Observations with at least 1 note: {obs_with_notes} ({obs_with_notes/total_obs*100:.1f}%)")
print(f"Avg notes per observation (when present): {total_notes/obs_with_notes:.1f}")
print(f"\nNote: Only ICU-matched notes have embeddings. Ward-only notes need GPU embedding (follow-up step).")
