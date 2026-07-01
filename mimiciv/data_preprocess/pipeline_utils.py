import os
import json
import time
import pickle
import tempfile

MARKER_DIR = ".markers"

# Canonical step names (also used by check_pipeline.py).
STEP1_IRG = "step1_irg"
STEP2_IMPUTED = "step2_imputed"
STEP3_NOTES = "step3_notes"
STEP4_NOTES_EMB = "step4_notes_emb"
STEP5_IHM = "step5_ihm"
STEP6_LOS = "step6_los"


def _marker_dir(output_dir):
    return os.path.join(output_dir, MARKER_DIR)


def marker_path(output_dir, step):
    return os.path.join(_marker_dir(output_dir), f"{step}.json")


def is_done(output_dir, step):
    return os.path.exists(marker_path(output_dir, step))


def read_marker(output_dir, step):
    p = marker_path(output_dir, step)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def write_marker(output_dir, step, meta=None):
    """Atomically write a completion marker. Call only after a step fully succeeds."""
    d = _marker_dir(output_dir)
    os.makedirs(d, exist_ok=True)
    payload = {"step": step, "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "completed_epoch": time.time()}
    if meta:
        payload.update(meta)
    _atomic_write(marker_path(output_dir, step), json.dumps(payload, indent=2).encode())


def clear_marker(output_dir, step):
    p = marker_path(output_dir, step)
    if os.path.exists(p):
        os.remove(p)


def _best_effort_fsync(fileno):
    # Some (FUSE / stage-backed) filesystems don't support fsync; ignore those.
    try:
        os.fsync(fileno)
    except OSError:
        pass


def _atomic_replace(tmp, path, retries=10, delay=0.1):
    # FUSE / stage-backed filesystems can transiently fail rename with EAGAIN
    # (BlockingIOError, errno 11). Retry with backoff before giving up.
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except BlockingIOError:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))


def _atomic_write(path, data_bytes):
    """Write bytes to a temp file in the same dir, then atomically replace."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data_bytes)
            f.flush()
            _best_effort_fsync(f.fileno())
        _atomic_replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_to_parquet(df, path, **kwargs):
    """Write a DataFrame to parquet atomically (no half-written / 0-byte files)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".parquet.tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, **kwargs)
        _atomic_replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_pickle_dump(obj, path):
    """Pickle an object to disk atomically."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".pkl.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f)
            f.flush()
            _best_effort_fsync(f.fileno())
        _atomic_replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
