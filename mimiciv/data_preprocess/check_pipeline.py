import os
import glob
import json
import pickle
import argparse

import pandas as pd
import pyarrow.parquet as pq

import pipeline_utils as pu

ID_COLS = ["subject_id", "hadm_id", "stay_id"]

# (step, human label, [output specs]) where each spec is (kind, relpath_or_glob, id_cols)
STAGES = [
    (pu.STEP1_IRG, "Step 1: irregular TS (labs + vitals)", [
        ("parquet", "ts_labs_icu.parquet", ID_COLS),
        ("parquet", "ts_vitals_icu.parquet", ID_COLS),
        ("parquet", "ts_labs_vitals.parquet", ID_COLS),
        ("parquet", "uom_labs_icu.parquet", []),
        ("parquet", "uom_vitals_icu.parquet", []),
        ("parquet", "uom_labs_vitals.parquet", []),
    ]),
    (pu.STEP2_IMPUTED, "Step 2: imputed regular TS", [
        ("parquet", "imputed_ts_labs_vitals.parquet", ID_COLS),
    ]),
    (pu.STEP3_NOTES, "Step 3: radiology notes text", [
        ("parquet", "rad_notes_text.parquet", ID_COLS),
    ]),
    (pu.STEP4_NOTES_EMB, "Step 4: BioBERT note embeddings", [
        ("parquet", "rad_notes_text_embeddings.parquet", ID_COLS),
    ]),
    (pu.STEP5_IHM, "Step 5: IHM task pkl", [
        ("pkl_glob", os.path.join("ihm", "*_stays.pkl"), None),
    ]),
    (pu.STEP6_LOS, "Step 6: LOS task pkl", [
        ("pkl_glob", os.path.join("los", "*_stays.pkl"), None),
    ]),
]


def _fmt(n):
    return f"{n:,}" if isinstance(n, int) else str(n)


def check_parquet(path, id_cols):
    if not os.path.exists(path):
        return {"status": "MISSING"}
    if os.path.getsize(path) == 0:
        return {"status": "INVALID", "detail": "0 bytes"}
    try:
        md = pq.read_metadata(path)
        names = md.schema.names
        info = {"status": "OK", "rows": md.num_rows}
        wanted = [c for c in id_cols if c in names]
        if wanted:
            df = pd.read_parquet(path, columns=wanted)
            info["ids"] = {c: int(df[c].nunique(dropna=True)) for c in wanted}
        return info
    except Exception as e:
        return {"status": "INVALID", "detail": f"{type(e).__name__}: {e}"}


def check_pkl(path, load):
    if not os.path.exists(path):
        return {"status": "MISSING"}
    size = os.path.getsize(path)
    if size == 0:
        return {"status": "INVALID", "detail": "0 bytes"}
    if not load:
        return {"status": "OK", "size_mb": round(size / 1e6, 1), "detail": "not loaded (--no-pkl-load)"}
    try:
        with open(path, "rb") as f:
            stays = pickle.load(f)
        n = len(stays)
        stay_ids = {s.get("stay_id") for s in stays if isinstance(s, dict)}
        hadm_ids = {s.get("hadm_id") for s in stays if isinstance(s, dict)}
        with_notes = sum(1 for s in stays if isinstance(s, dict) and not s.get("text_missing", 1))
        return {"status": "OK", "size_mb": round(size / 1e6, 1), "stays": n,
                "distinct_stay_id": len(stay_ids), "distinct_hadm_id": len(hadm_ids),
                "stays_with_notes": with_notes}
    except Exception as e:
        return {"status": "INVALID", "detail": f"{type(e).__name__}: {e}"}


def step1_part_progress(output_dir):
    parts_dir = os.path.join(output_dir, "_step1_parts")
    plan_path = os.path.join(parts_dir, "plan.json")
    if not os.path.exists(plan_path):
        return None
    try:
        with open(plan_path) as f:
            plan = json.load(f)
        n_batches = len(plan.get("batches", []))
    except Exception:
        return {"detail": "plan.json present but unreadable"}
    completed = 0
    comp_path = os.path.join(parts_dir, "completed.txt")
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            completed = len({x for x in f.read().split() if x.strip()})
    return {"batch_size": plan.get("batch_size"), "n_hadm": plan.get("n_hadm"),
            "completed": completed, "n_batches": n_batches}


def gather(output_dir, spec, load_pkl):
    kind, rel, id_cols = spec
    if kind == "parquet":
        return rel, check_parquet(os.path.join(output_dir, rel), id_cols or [])
    if kind == "pkl_glob":
        matches = sorted(glob.glob(os.path.join(output_dir, rel)))
        if not matches:
            return rel, {"status": "MISSING"}
        return rel, {"status": "MULTI", "files": {os.path.relpath(m, output_dir): check_pkl(m, load_pkl) for m in matches}}
    return rel, {"status": "UNKNOWN"}


def print_result(rel, res, indent="    "):
    s = res.get("status")
    if s == "MULTI":
        print(f"{indent}{rel}:")
        for fp, r in res["files"].items():
            print_result(fp, r, indent + "    ")
        return
    line = f"{indent}{rel}: {s}"
    if "rows" in res:
        line += f"  rows={_fmt(res['rows'])}"
    if "stays" in res:
        line += f"  stays={_fmt(res['stays'])} distinct_stay_id={_fmt(res['distinct_stay_id'])} distinct_hadm_id={_fmt(res['distinct_hadm_id'])} with_notes={_fmt(res['stays_with_notes'])}"
    if "ids" in res:
        line += "  " + " ".join(f"{c}={_fmt(v)}" for c, v in res["ids"].items())
    if "size_mb" in res:
        line += f"  size={res['size_mb']}MB"
    if "detail" in res:
        line += f"  ({res['detail']})"
    print(line)


def all_outputs_valid(output_dir, specs, load_pkl):
    for spec in specs:
        _, res = gather(output_dir, spec, load_pkl)
        statuses = []
        if res.get("status") == "MULTI":
            statuses = [r.get("status") for r in res["files"].values()]
        else:
            statuses = [res.get("status")]
        if any(st not in ("OK",) for st in statuses):
            return False
    return True


def main(args):
    out = args.output_dir
    print(f"Pipeline status for: {os.path.abspath(out)}\n")

    for step, label, specs in STAGES:
        done = pu.is_done(out, step)
        marker = pu.read_marker(out, step)
        flag = "DONE" if done else "no marker"
        when = f" @ {marker['completed_at']}" if marker and marker.get("completed_at") else ""
        print(f"[{flag}{when}] {label}  ({step})")

        for spec in specs:
            rel, res = gather(out, spec, not args.no_pkl_load)
            print_result(rel, res)

        if step == pu.STEP1_IRG:
            prog = step1_part_progress(out)
            if prog and "completed" in prog:
                print(f"    in-progress parts: {prog['completed']}/{prog['n_batches']} batches "
                      f"(batch_size={prog['batch_size']}, n_hadm={_fmt(prog['n_hadm'] or 0)})")
        print()

    if args.bless:
        step = args.bless
        valid_steps = {s for s, _, _ in STAGES}
        if step not in valid_steps:
            print(f"--bless: unknown step '{step}'. Valid: {sorted(valid_steps)}")
            return
        specs = next(sp for s, _, sp in STAGES if s == step)
        if all_outputs_valid(out, specs, not args.no_pkl_load):
            pu.write_marker(out, step, {"blessed": True, "note": "manually verified by check_pipeline"})
            print(f"--bless: all outputs for '{step}' are valid; wrote completion marker. "
                  f"Reruns will now skip this step.")
        else:
            print(f"--bless: REFUSED -- one or more outputs for '{step}' are missing/invalid "
                  f"(see report above). Fix or regenerate before blessing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data", help="Pipeline output directory")
    parser.add_argument("--no-pkl-load", action="store_true",
                        help="Do not load large .pkl files (report size/existence only)")
    parser.add_argument("--bless", type=str, default=None,
                        help="Write a completion marker for STEP if all its outputs are valid "
                             "(e.g. step1_irg). Use to adopt outputs from a prior run.")
    args = parser.parse_args()
    main(args)
