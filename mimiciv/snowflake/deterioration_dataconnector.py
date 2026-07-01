import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

from deterioration_config import (
    MULTIMODAL_TABLE, FEATURE_NAMES, SEQ_LEN, NOTE_EMB_DIM, flat_col, all_flat_columns,
)

FLAT_COLUMNS = [c.upper() for c in all_flat_columns()]
LABEL_COL = "LABEL"
WEIGHT_COL = "SAMPLE_WEIGHT"
NOTES_COL = "NOTES_PACKED"


def flat_to_ts(flat: np.ndarray) -> np.ndarray:
    """(B, 1488) flattened columns -> (B, 48, 31) tensor in FEATURE_NAMES order.

    flat columns are ordered [f0_h00..f0_h47, f1_h00..f1_h47, ...] so a reshape to
    (B, 31, 48) then transpose gives (B, 48, 31).
    """
    b = flat.shape[0]
    return flat.reshape(b, len(FEATURE_NAMES), SEQ_LEN).transpose(0, 2, 1).astype(np.float32)


def scatter_notes(packed_list):
    """List (len B) of packed-note arrays -> (B, 48, 768) features + (B, 48) mask.

    Each packed entry is a list of dicts {'slot': int, 'emb': [768 floats]} (or None).
    Multiple notes in the same slot are averaged (matches the Databricks loader).
    """
    b = len(packed_list)
    notes = np.zeros((b, SEQ_LEN, NOTE_EMB_DIM), dtype=np.float32)
    mask = np.zeros((b, SEQ_LEN), dtype=np.float32)
    counts = np.zeros((b, SEQ_LEN), dtype=np.float32)
    for i, packed in enumerate(packed_list):
        if not packed:
            continue
        for obj in packed:
            slot = int(obj["slot"])
            if slot < 0 or slot >= SEQ_LEN:
                continue
            emb = np.asarray(obj["emb"], dtype=np.float32)
            notes[i, slot, :] += emb
            counts[i, slot] += 1.0
            mask[i, slot] = 1.0
    nz = counts > 0
    notes[nz] /= counts[nz][:, None]
    return notes, mask


def _shard_predicate(world_size, rank):
    """Snowflake-side row-level shard filter, or None when not sharding.

    Hashing the row identity (HADM_ID, TSP) gives a uniform, deterministic
    partition that is pushed down to Snowflake, so each rank scans and transfers
    only its ~1/world_size slice of rows.
    """
    if world_size and int(world_size) > 1:
        return f"MOD(ABS(HASH(HADM_ID, TSP)), {int(world_size)}) = {int(rank)}"
    return None


def _split_df(session, split, include_notes, rank=0, world_size=1):
    cols = ["HADM_ID", "TSP", LABEL_COL, WEIGHT_COL] + FLAT_COLUMNS
    if include_notes:
        cols.append(NOTES_COL)
    df = session.table(MULTIMODAL_TABLE).filter(f"split = '{split}'")
    pred = _shard_predicate(world_size, rank)
    if pred is not None:
        df = df.filter(pred)
    return df.select(*cols)


def make_ts_data_connector(session, split="train"):
    """DataConnector over the numeric TS columns (+ label + weight). TS-only fast path."""
    from snowflake.ml.data.data_connector import DataConnector
    cols = [LABEL_COL, WEIGHT_COL] + FLAT_COLUMNS
    df = session.table(MULTIMODAL_TABLE).filter(f"split = '{split}'").select(*cols)
    return DataConnector.from_dataframe(df)


def make_sharded_ts_connector(session, split="train"):
    """ShardedDataConnector for PyTorchDistributor multi-node training."""
    from snowflake.ml.data.sharded_data_connector import ShardedDataConnector
    cols = [LABEL_COL, WEIGHT_COL] + FLAT_COLUMNS
    df = session.table(MULTIMODAL_TABLE).filter(f"split = '{split}'").select(*cols)
    return ShardedDataConnector.from_dataframe(df)


def ts_torch_loader(data_connector, batch_size=256, shuffle=True, drop_last=False):
    """Wrap a DataConnector.to_torch_dataset into (ts, zero-notes, mask, label, weight)
    batches so the TS-only path is drop-in compatible with the multimodal train loop.
    """
    ds = data_connector.to_torch_dataset(
        batch_size=batch_size, shuffle=shuffle
    )
    for batch in ds:
        flat = np.stack([np.asarray(batch[c]).reshape(-1) for c in FLAT_COLUMNS], axis=1)
        ts = torch.from_numpy(flat_to_ts(flat))
        b = ts.shape[0]
        notes = torch.zeros((b, SEQ_LEN, NOTE_EMB_DIM), dtype=torch.float32)
        mask = torch.zeros((b, SEQ_LEN), dtype=torch.float32)
        labels = torch.from_numpy(np.asarray(batch[LABEL_COL]).reshape(-1)).float()
        weights = torch.from_numpy(np.asarray(batch[WEIGHT_COL]).reshape(-1)).float()
        yield ts, notes, mask, labels, weights


class MultimodalDeteriorationDataset(IterableDataset):
    """Streams (ts, notes, note_mask, label, weight) from the multimodal table.

    Uses Snowpark DataFrame.to_pandas_batches() so the ARRAY notes column is
    preserved and memory stays bounded by `chunk_rows`. Rows are accumulated into a
    shuffle buffer of up to `chunk_rows` samples that spans multiple pandas batches,
    then shuffled and emitted - this breaks the within-admission ordering left by the
    table's CLUSTER BY (split). Larger `chunk_rows` = stronger shuffling, more memory.

    For distributed training, pass `rank`/`world_size` so each worker reads a
    disjoint, Snowflake-side ~1/world_size shard (hashed on the row identity),
    making the notes path scale with GPU count like the TS-only ShardedDataConnector.
    """

    def __init__(self, session, split="train", chunk_rows=20000, shuffle=True,
                 seed=42, include_notes=True, rank=0, world_size=1):
        super().__init__()
        self.session = session
        self.split = split
        self.chunk_rows = max(int(chunk_rows), 1)
        self.shuffle = shuffle
        self.seed = seed
        self.include_notes = include_notes
        self.rank = int(rank)
        self.world_size = max(int(world_size), 1)
        base = session.table(MULTIMODAL_TABLE).filter(f"split = '{split}'")
        pred = _shard_predicate(self.world_size, self.rank)
        if pred is not None:
            base = base.filter(pred)
        self._count = base.count()

    def __len__(self):
        return self._count

    def _emit(self, buf_flat, buf_label, buf_weight, buf_packed, rng):
        order = np.arange(len(buf_flat))
        if self.shuffle:
            rng.shuffle(order)
        for i in order:
            ts_i = flat_to_ts(buf_flat[i][None, :])[0]
            if self.include_notes:
                notes_i, mask_i = scatter_notes([buf_packed[i]])
                notes_i, mask_i = notes_i[0], mask_i[0]
            else:
                notes_i = np.zeros((SEQ_LEN, NOTE_EMB_DIM), dtype=np.float32)
                mask_i = np.zeros((SEQ_LEN,), dtype=np.float32)
            yield ts_i, notes_i, mask_i, buf_label[i], buf_weight[i]

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        df = _split_df(self.session, self.split, self.include_notes,
                       rank=self.rank, world_size=self.world_size)

        # Lightweight shuffle buffer: keep raw flat rows + packed notes (sparse) and
        # only materialize the dense (48, 768) note tensor per sample on emit, so the
        # buffer stays ~chunk_rows x 1488 floats rather than holding dense note arrays.
        buf_flat, buf_label, buf_weight, buf_packed = [], [], [], []
        for pdf in df.to_pandas_batches():
            if len(pdf) == 0:
                continue
            flat = pdf[FLAT_COLUMNS].to_numpy(dtype=np.float32)
            labels = pdf[LABEL_COL].to_numpy(dtype=np.float32)
            weights = pdf[WEIGHT_COL].to_numpy(dtype=np.float32)
            packed = ([_parse_packed(x) for x in pdf[NOTES_COL].tolist()]
                      if self.include_notes else [None] * len(pdf))
            for j in range(len(pdf)):
                buf_flat.append(flat[j]); buf_label.append(labels[j])
                buf_weight.append(weights[j]); buf_packed.append(packed[j])
                if len(buf_flat) >= self.chunk_rows:
                    yield from self._emit(buf_flat, buf_label, buf_weight, buf_packed, rng)
                    buf_flat, buf_label, buf_weight, buf_packed = [], [], [], []

        if buf_flat:
            yield from self._emit(buf_flat, buf_label, buf_weight, buf_packed, rng)


def _parse_packed(x):
    """Snowpark returns an ARRAY column as a JSON string (or already a list)."""
    if x is None:
        return None
    if isinstance(x, str):
        import json
        return json.loads(x)
    return x


def collate_multimodal(batch):
    ts_l, notes_l, mask_l, lab_l, w_l = zip(*batch)
    return (
        torch.from_numpy(np.stack(ts_l)),
        torch.from_numpy(np.stack(notes_l)),
        torch.from_numpy(np.stack(mask_l)),
        torch.from_numpy(np.array(lab_l, dtype=np.float32)),
        torch.from_numpy(np.array(w_l, dtype=np.float32)),
    )


def make_multimodal_loader(session, split="train", batch_size=256, chunk_rows=20000,
                           shuffle=True, seed=42, include_notes=True,
                           pin_memory=True, rank=0, world_size=1):
    ds = MultimodalDeteriorationDataset(
        session, split=split, chunk_rows=chunk_rows, shuffle=shuffle,
        seed=seed, include_notes=include_notes, rank=rank, world_size=world_size,
    )
    return DataLoader(
        ds, batch_size=batch_size, collate_fn=collate_multimodal,
        num_workers=0, pin_memory=pin_memory,
    )
