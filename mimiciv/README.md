# MIMIC-IV Experiments

Experiments for the **In-Hospital Mortality (IHM)** and **Length-of-Stay (LOS)** prediction tasks on MIMIC-IV, using the MERGE multimodal MoE framework.

Three modalities are used:
- **labs\_vitals** — 30-dimensional regular hourly time series (labs + vital signs)
- **cxr** — 1024-dimensional chest X-ray features (DenseNet121), irregularly timed
- **notes** — 768-dimensional radiology note embeddings (BioBERT), irregularly timed

All scripts are designed to be run from the `mimiciv/` directory.

---

## Prerequisites

**Databricks environment:**
- A Databricks workspace with Unity Catalog enabled
- MIMIC-IV tables loaded into a Unity Catalog catalog (default: `mimiciv`) with schemas `hosp` and `icu`
- Radiology notes CSV stored in a UC Volume (e.g. `/Volumes/mimiciv/note/data/radiology.csv.gz`)
- BioBERT model files either in a UC Volume or downloadable from HuggingFace Hub (`dmis-lab/biobert-v1.1`)
- A GPU-enabled cluster for embedding steps (Steps 4 and 6)

**Expected Unity Catalog layout:**
```
mimiciv.hosp.admissions
mimiciv.hosp.labevents
mimiciv.hosp.d_labitems
mimiciv.icu.icustays
mimiciv.icu.chartevents
mimiciv.icu.d_items
```

**UC Volume for files** (path format: `/Volumes/<catalog>/<schema>/<volume>/<file>`):
```
/Volumes/mimiciv/note/data/radiology.csv.gz
```

Override the catalog name via the `MIMICIV_CATALOG` environment variable if your layout differs.

---

## Step 1 — Data Preprocessing

Run the full preprocessing pipeline from the `mimiciv/` directory. This produces the
train/val/test pkl files for both tasks.

```bash
bash data_preprocess/preprocess_mimic.sh \
    /Volumes/mimiciv/note/data/radiology.csv.gz \
    <gpu> \
    [batch_size]
```

**Arguments:**
| Argument | Description |
|---|---|
| `notes_file_path` | Path to radiology notes CSV (UC Volume path, e.g. `/Volumes/mimiciv/note/data/radiology.csv.gz`) |
| `gpu` | GPU device ID for the embedding steps (default: `0`) |
| `batch_size` | Admissions per batch for Step 1 (default: `40000`) |

The pipeline runs 6 steps in sequence:

| Step | Script | Output |
|---|---|---|
| 1 | `preprocess_irg_time_series.py` | `data/ts_labs_vitals.parquet` |
| 2 | `preprocess_imputed_time_series.py` | `data/imputed_ts_labs_vitals.parquet` |
| 3 | `preprocess_notes.py` | `data/rad_notes_text.parquet` |
| 4 | `preprocess_notes_embeddings.py` *(GPU)* | `data/rad_notes_text_embeddings.parquet` |
| 5 | `create_ihm_task.py` | `data/ihm/{train,val,test}_ihm-48-notes-missingInd-standardized_stays.pkl` |
| 6 | `create_los_task.py` | `data/los/{train,val,test}_los-notes-missingInd-standardized_stays.pkl` |

Steps 4 is GPU-intensive (BioBERT inference).


---

## Step 2 — RUS Computation

Compute pairwise temporal Redundancy/Uniqueness/Synergy (RUS) values between all modality pairs. This only needs to be done once per task; the results are saved and reused for training.

**IHM:**
```bash
bash run_rus_ihm.sh <gpu>
```
Output: `results/ihm/rus_multimodal_all_seq48_lags8_meanpool.npy`

**LOS:**
```bash
bash run_rus_los.sh <gpu>
```
Output: `results/los/rus_multimodal_all_seq48_lags8_meanpool.npy`

RUS computation uses the sequence-labeled batch estimator
(`pid/temporal_pid_multi_sequence.py`) with `--seq_len 48`, `--num_lags 8`, and
`--sequence_pooling mean` (mean-pooling over valid timesteps per stay before estimating).

---

## Step 3 — Training

Train the multimodal TRUS-MoE model. Both scripts accept four positional arguments:
`<lambda_rus> <lambda_load> <gpu> <seed>`.

`lambda_rus` controls the weight of all three RUS auxiliary losses (uniqueness,
redundancy, synergy) simultaneously.

**IHM** (best found: `lambda_rus=1.0`, `lambda_load=0.02`):
```bash
bash train_mimiciv_ihm.sh 1.0 0.02 <gpu> <seed>
```

**LOS** (best found: `lambda_rus=0.5`, `lambda_load=0.02`):
```bash
bash train_mimiciv_los.sh 0.5 0.02 <gpu> <seed>
```

Checkpoints are saved to:
```
results/{ihm,los}/checkpoints/<run_name>/best_multimodal_model_mimiciv.pth
```

---

## Step 4 — Testing

Evaluate a saved checkpoint on the test set (and optionally train/val sets).

**IHM:**
```bash
bash test_mimiciv_ihm.sh <checkpoint_path> <gpu>
```

**LOS:**
```bash
bash test_mimiciv_los.sh <checkpoint_path> <gpu>
```

**Example:**
```bash
bash test_mimiciv_ihm.sh \
    results/ihm/checkpoints/mimiciv_ihm_lambdarus1.0_lambdaload0.02_seed42/best_multimodal_model_mimiciv.pth \
    0
```

The test scripts evaluate on the test set and also run evaluation on the training and
validation sets (`--eval_train`, `--eval_val`). Metrics (accuracy, AU-ROC, F1, precision,
recall, confusion matrix) are saved as JSON alongside the checkpoint via `--save_metrics`.
Expert activation plots are generated with `--plot_expert_activations`.

---

## Results Directory Structure

```
mimiciv/
├── data/                          # gitignored — created by preprocessing
│   ├── *.parquet                  # intermediate files
│   ├── ihm/                       # IHM task pkl files + scalers
│   └── los/                       # LOS task pkl files + scalers
└── results/                       # gitignored — created at runtime
    ├── ihm/
    │   ├── rus_multimodal_all_seq48_lags8_meanpool.npy
    │   └── checkpoints/<run_name>/
    │       ├── best_multimodal_model_mimiciv.pth
    │       └── *.json             # evaluation metrics
    └── los/
        ├── rus_multimodal_all_seq48_lags8_meanpool.npy
        └── checkpoints/<run_name>/
            ├── best_multimodal_model_mimiciv.pth
            └── *.json
```
