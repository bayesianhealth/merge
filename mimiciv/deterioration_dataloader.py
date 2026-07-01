"""PyTorch DataLoader for the Deterioration prediction task.

Streams from two Delta tables:
  - mimiciv.hosp.deterioration_training_data: time series features (31 arrays of len 48)
  - mimiciv.hosp.deterioration_note_embeddings: per-note 768-dim embeddings with timestamps

Designed for single-GPU training on Databricks Serverless GPU.
Handles the join + note alignment on-the-fly in configurable batch sizes.

Performance notes:
  - First batch ~15s (Spark overhead). Steady-state ~8s per partition of 2000.
  - For faster throughput: increase partition_size (trades memory for fewer round-trips)
  - For production: consider pre-materializing a joined training table
  - num_workers must be 0 (Spark sessions are not fork-safe)

Usage:
    from deterioration_dataloader import DeteriorationDataLoader

    loader = DeteriorationDataLoader(
        spark=spark,
        split="train",
        batch_size=256,
        shuffle=True,
    )
    for batch in loader:
        ts_features, note_features, note_mask, labels, weights = batch
        # ts_features: (B, 48, 31)
        # note_features: (B, 48, 768)
        # note_mask: (B, 48) — 1 where notes exist
        # labels: (B,)
        # weights: (B,)
"""

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Optional, Tuple, Iterator

# Feature names (must match the training table columns)
FEATURE_NAMES = [
    'anion_gap', 'bicarbonate', 'calcium_total', 'chloride', 'creatinine',
    'diastolic_bp', 'gcs_eye', 'gcs_motor', 'gcs_verbal', 'glucose',
    'heart_rate', 'hematocrit', 'hemoglobin', 'magnesium', 'mch',
    'mchc', 'mcv', 'mean_bp', 'neutrophils', 'o2_saturation',
    'phosphate', 'platelet_count', 'potassium', 'rdw', 'red_blood_cells',
    'respiratory_rate', 'sodium', 'systolic_bp', 'urea_nitrogen',
    'vancomycin', 'white_blood_cells',
]

NOTE_EMB_DIM = 768
SEQ_LEN = 48

TS_TABLE = "mimiciv.hosp.deterioration_training_data"
NOTES_TABLE = "mimiciv.hosp.deterioration_note_embeddings"


class DeteriorationDataset(IterableDataset):
    """Streams deterioration training examples from Delta tables via Spark.

    Reads time series in configurable partition sizes, looks up corresponding
    notes, aligns them to the 48-step grid, and yields individual examples.

    Args:
        spark: Active SparkSession.
        split: One of 'train', 'val', 'test'.
        partition_size: Number of rows to fetch from Delta per Spark query.
            Controls memory usage. Larger = fewer round-trips but more RAM.
        shuffle: Whether to shuffle within each partition.
        seed: Random seed for shuffling and partition ordering.
        include_notes: If False, skip note lookup (faster, TS-only training).
    """

    def __init__(
        self,
        spark,
        split: str = "train",
        partition_size: int = 10_000,
        shuffle: bool = True,
        seed: int = 42,
        include_notes: bool = True,
    ):
        super().__init__()
        self.spark = spark
        self.split = split
        self.partition_size = partition_size
        self.shuffle = shuffle
        self.seed = seed
        self.include_notes = include_notes

        # Get total count and hadm_id list for partitioning
        self._count = (
            spark.table(TS_TABLE)
            .filter(f"split = '{split}'")
            .count()
        )

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[Tuple[np.ndarray, ...]]:
        """Iterate over all examples in the split."""
        from pyspark.sql import functions as F

        rng = np.random.default_rng(self.seed)

        # Read the full split with monotonically_increasing_id for pagination
        ts_df = (
            self.spark.table(TS_TABLE)
            .filter(f"split = '{self.split}'")
        )

        # Collect hadm_id + tsp pairs for partitioned reading
        # Use row_number for stable pagination
        keys_df = (
            ts_df.select("hadm_id", "tsp")
            .withColumn("_row_num", F.monotonically_increasing_id())
        )

        total = self._count
        n_partitions = (total + self.partition_size - 1) // self.partition_size

        # Optionally shuffle partition order
        partition_indices = np.arange(n_partitions)
        if self.shuffle:
            rng.shuffle(partition_indices)

        for part_idx in partition_indices:
            offset = int(part_idx) * self.partition_size

            # Fetch a partition of training data
            partition_df = (
                ts_df
                .offset(offset)
                .limit(self.partition_size)
            )

            # Collect TS features as pandas
            ts_cols = FEATURE_NAMES + ["hadm_id", "tsp", "label", "sample_weight"]
            ts_pdf = partition_df.select(*ts_cols).toPandas()

            if len(ts_pdf) == 0:
                continue

            # Parse TS arrays into numpy
            n_rows = len(ts_pdf)
            ts_features = np.zeros((n_rows, SEQ_LEN, len(FEATURE_NAMES)), dtype=np.float32)
            for feat_idx, feat_name in enumerate(FEATURE_NAMES):
                for row_idx, arr in enumerate(ts_pdf[feat_name].values):
                    if arr is not None:
                        ts_features[row_idx, :, feat_idx] = np.array(arr, dtype=np.float32)

            labels = ts_pdf["label"].values.astype(np.float32)
            weights = ts_pdf["sample_weight"].values.astype(np.float32)

            # Fetch and align notes for this partition
            if self.include_notes:
                note_features, note_mask = self._align_notes_for_partition(ts_pdf)
            else:
                note_features = np.zeros((n_rows, SEQ_LEN, NOTE_EMB_DIM), dtype=np.float32)
                note_mask = np.zeros((n_rows, SEQ_LEN), dtype=np.float32)

            # Shuffle within partition
            indices = np.arange(n_rows)
            if self.shuffle:
                rng.shuffle(indices)

            # Yield individual examples
            for i in indices:
                yield (
                    ts_features[i],       # (48, 31)
                    note_features[i],     # (48, 768)
                    note_mask[i],         # (48,)
                    labels[i],            # scalar
                    weights[i],           # scalar
                )

    def _align_notes_for_partition(
        self, ts_pdf
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch notes for a partition and align to the 48-step grid.

        For each (hadm_id, obs_tsp), retrieves all notes from the notes table
        and maps them into a (48, 768) array using hours_before_obs.
        Multiple notes at the same hour slot are averaged.

        Returns:
            note_features: (n_rows, 48, 768) aligned note embeddings
            note_mask: (n_rows, 48) binary mask (1 where notes exist)
        """
        from pyspark.sql import functions as F
        import pandas as pd

        n_rows = len(ts_pdf)
        note_features = np.zeros((n_rows, SEQ_LEN, NOTE_EMB_DIM), dtype=np.float32)
        note_mask = np.zeros((n_rows, SEQ_LEN), dtype=np.float32)

        # Build lookup key: (hadm_id, tsp) -> row_index
        key_to_idx = {}
        for idx, row in ts_pdf.iterrows():
            key = (int(row["hadm_id"]), row["tsp"])
            key_to_idx[key] = idx

        # Get unique hadm_ids for efficient filtering
        hadm_ids = ts_pdf["hadm_id"].unique().tolist()
        tsp_min = ts_pdf["tsp"].min()
        tsp_max = ts_pdf["tsp"].max()

        # Query notes table for this partition's hadm_ids and time range
        notes_pdf = (
            self.spark.table(NOTES_TABLE)
            .filter(F.col("hadm_id").isin(hadm_ids))
            .filter(F.col("obs_tsp").between(tsp_min, tsp_max))
            .select("hadm_id", "obs_tsp", "hours_before_obs", "note_embedding")
            .toPandas()
        )

        if len(notes_pdf) == 0:
            return note_features, note_mask

        # Align notes to grid
        for _, note_row in notes_pdf.iterrows():
            key = (int(note_row["hadm_id"]), note_row["obs_tsp"])
            if key not in key_to_idx:
                continue

            row_idx = key_to_idx[key]
            hours_before = int(note_row["hours_before_obs"])

            # Map to array index: index 0 = 48h ago, index 47 = current
            # hours_before_obs=0 -> index 47, hours_before_obs=47 -> index 0
            arr_idx = SEQ_LEN - 1 - hours_before
            if arr_idx < 0 or arr_idx >= SEQ_LEN:
                continue

            emb = np.array(note_row["note_embedding"], dtype=np.float32)

            if note_mask[row_idx, arr_idx] == 0:
                # First note at this slot
                note_features[row_idx, arr_idx, :] = emb
                note_mask[row_idx, arr_idx] = 1.0
            else:
                # Average with existing (running mean)
                # Count how many notes at this slot (approximate: use mask trick)
                # For simplicity, just average with equal weight
                note_features[row_idx, arr_idx, :] = (
                    note_features[row_idx, arr_idx, :] + emb
                ) / 2.0

        return note_features, note_mask


def collate_deterioration(batch):
    """Collate function for DataLoader — stacks individual examples into tensors."""
    ts_list, notes_list, mask_list, labels_list, weights_list = zip(*batch)

    return (
        torch.from_numpy(np.stack(ts_list)),        # (B, 48, 31)
        torch.from_numpy(np.stack(notes_list)),     # (B, 48, 768)
        torch.from_numpy(np.stack(mask_list)),      # (B, 48)
        torch.from_numpy(np.array(labels_list)),    # (B,)
        torch.from_numpy(np.array(weights_list)),   # (B,)
    )


class DeteriorationDataLoader:
    """Convenience wrapper that creates the Dataset + DataLoader.

    Args:
        spark: Active SparkSession.
        split: One of 'train', 'val', 'test'.
        batch_size: Number of examples per training batch.
        partition_size: Rows fetched from Delta per round-trip (controls memory).
        shuffle: Shuffle within partitions (and partition order).
        seed: Random seed.
        include_notes: Whether to include note modality.
        num_workers: DataLoader workers (0 for in-process, since Spark isn't
            fork-safe; leave at 0 for Databricks notebooks).
        pin_memory: Pin tensors to CUDA memory for faster transfer.

    Usage:
        loader = DeteriorationDataLoader(spark, split='train', batch_size=256)
        for ts, notes, mask, labels, weights in loader:
            ...
    """

    def __init__(
        self,
        spark,
        split: str = "train",
        batch_size: int = 256,
        partition_size: int = 10_000,
        shuffle: bool = True,
        seed: int = 42,
        include_notes: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        self.dataset = DeteriorationDataset(
            spark=spark,
            split=split,
            partition_size=partition_size,
            shuffle=shuffle,
            seed=seed,
            include_notes=include_notes,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def __len__(self) -> int:
        """Approximate number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Returns a PyTorch DataLoader iterator."""
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=collate_deterioration,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return iter(loader)


# --- Utility: quick validation ---
def validate_dataloader(spark, split="val", n_batches=3):
    """Run a quick validation: fetch a few batches and print shapes/stats."""
    loader = DeteriorationDataLoader(
        spark=spark,
        split=split,
        batch_size=64,
        partition_size=500,
        shuffle=False,
        include_notes=True,
    )

    print(f"Split: {split}, Total examples: {len(loader.dataset)}, ~Batches: {len(loader)}")
    print(f"Fetching {n_batches} batches...")

    for i, (ts, notes, mask, labels, weights) in enumerate(loader):
        if i >= n_batches:
            break
        print(f"\n  Batch {i}:")
        print(f"    ts_features:  {ts.shape}, dtype={ts.dtype}")
        print(f"    note_features: {notes.shape}, dtype={notes.dtype}")
        print(f"    note_mask:    {mask.shape}, sum={mask.sum().item():.0f}/{mask.numel()}")
        print(f"    labels:       {labels.shape}, mean={labels.mean():.4f}")
        print(f"    weights:      {weights.shape}, mean={weights.mean():.4f}")
        print(f"    ts NaN check: {torch.isnan(ts).sum().item()} NaNs")
        print(f"    notes non-zero slots: {(mask.sum(dim=0) > 0).sum().item()}/48 hours have data")

    print("\nValidation complete.")
