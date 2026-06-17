"""XGBoost baseline for the MIMIC-IV 48h in-hospital-mortality (IHM) task.

Reuses the already-compiled task pickles so the train/val/test split is byte-for-byte
identical to the multimodal model. No notes are used (labs + vitals only).

Per stay, features are built from the compiled `reg_ts` matrix (regularly-sampled,
imputed, standardized labs+vitals over ICU hours 0-48):
  - hour-48 snapshot  = reg_ts[-1]      (the value at the 48h timepoint)
  - per-feature mean / min / max over 0-48h
giving 4 x 31 = 124 features per stay.
"""
import os
import argparse
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import xgboost as xgb

DEFAULT_TASK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ihm")
DEFAULT_BASE = "ihm-48-notes-missingInd-standardized"


def load_split(task_dir, base, split):
    path = os.path.join(task_dir, f"{split}_{base}_stays.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_features(stays):
    X, y, feat_names = [], [], None
    for s in stays:
        reg = np.asarray(s["reg_ts"], dtype=float)
        if reg.ndim != 2 or reg.shape[0] == 0:
            continue
        reg = np.nan_to_num(reg, nan=0.0, posinf=0.0, neginf=0.0)
        snap = reg[-1]                 # hour-48 snapshot (== last timestep)
        feat = np.concatenate([snap, reg.mean(0), reg.min(0), reg.max(0)])
        X.append(feat)
        y.append(int(s["label"]))
        if feat_names is None:
            base = list(s.get("feature_names", [f"f{i}" for i in range(reg.shape[1])]))
            feat_names = ([f"{n}_h48" for n in base] + [f"{n}_mean" for n in base]
                          + [f"{n}_min" for n in base] + [f"{n}_max" for n in base])
    return np.asarray(X), np.asarray(y), feat_names


def main(args):
    print("Loading compiled IHM-48 splits (reusing multimodal cohort/split)...")
    train = load_split(args.task_dir, args.base, "train")
    val = load_split(args.task_dir, args.base, "val")
    test = load_split(args.task_dir, args.base, "test")

    X_tr, y_tr, feat_names = build_features(train)
    X_va, y_va, _ = build_features(val)
    X_te, y_te, _ = build_features(test)

    def bal(y):
        return f"{len(y)} stays, {int(y.sum())} deaths ({100*y.mean():.1f}%)"
    print(f"  train: {bal(y_tr)}")
    print(f"  val:   {bal(y_va)}")
    print(f"  test:  {bal(y_te)}")
    print(f"  features per stay: {X_tr.shape[1]}")

    spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)  # negative/positive
    clf = xgb.XGBClassifier(
        n_estimators=2000, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, objective="binary:logistic", eval_metric="auc",
        scale_pos_weight=spw, early_stopping_rounds=50, n_jobs=-1,
        tree_method="hist", random_state=args.seed,
    )
    print("Training XGBoost (early stopping on val AUROC)...")
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    print(f"  best_iteration: {clf.best_iteration}")

    p_te = clf.predict_proba(X_te)[:, 1]
    auroc = roc_auc_score(y_te, p_te)
    auprc = average_precision_score(y_te, p_te)
    acc = accuracy_score(y_te, (p_te >= 0.5).astype(int))

    print("\n" + "=" * 44)
    print("XGBoost IHM-48 baseline (labs+vitals, no notes)")
    print("=" * 44)
    print(f"  Test AUROC : {auroc:.4f}")
    print(f"  Test AUPRC : {auprc:.4f}")
    print(f"  Test Acc   : {acc:.4f}  (threshold 0.5)")
    print("=" * 44)

    if feat_names is not None:
        imp = clf.feature_importances_
        order = np.argsort(imp)[::-1][:15]
        print("Top 15 features by gain importance:")
        for i in order:
            print(f"  {feat_names[i]:<28} {imp[i]:.4f}")

    return clf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_dir", default=DEFAULT_TASK_DIR)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
