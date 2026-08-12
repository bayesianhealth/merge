import os
import json
import time
import tempfile

MARKER_DIR = ".markers"

STEP1_LOC = "step1_loc"
STEP2_EVENTS = "step2_events"
STEP3_ANCHORS = "step3_anchors"
STEP4_SCALER = "step4_scaler"
STEP5_EXAMPLES = "step5_examples"
STEP6_SPLITS = "step6_splits"


def _marker_dir(d):
    return os.path.join(d, MARKER_DIR)


def marker_path(d, step):
    return os.path.join(_marker_dir(d), f"{step}.json")


def is_done(d, step):
    return os.path.exists(marker_path(d, step))


def read_marker(d, step):
    p = marker_path(d, step)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def write_marker(d, step, meta=None):
    md = _marker_dir(d)
    os.makedirs(md, exist_ok=True)
    payload = {"step": step, "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "completed_epoch": time.time()}
    if meta:
        payload.update(meta)
    p = marker_path(d, step)
    fd, tmp = tempfile.mkstemp(dir=md, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(json.dumps(payload, indent=2).encode())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def clear_marker(d, step):
    p = marker_path(d, step)
    if os.path.exists(p):
        os.remove(p)
