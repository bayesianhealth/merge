import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import xgboost as xgb

from deterioration_config import MULTIMODAL_TABLE, FEATURE_NAMES, SEQ_LEN, flat_col

LABEL_COL = "LABEL"
SPLIT_COL = "SPLIT"


def _feature_exprs():
    """Build the server-side [h47, mean, min, max] SQL expressions + names.

    Columns are COALESCEd to 0.0 (already the imputation convention) so LEAST /
    GREATEST never collapse a whole feature to NULL.
    """
    select_cols, feat_names = [], []
    last = SEQ_LEN - 1
    for f in FEATURE_NAMES:
        cc = [f"COALESCE({flat_col(f, h).upper()}, 0.0)" for h in range(SEQ_LEN)]
        select_cols.append(cc[last])                                  # h47 snapshot
        select_cols.append("(" + " + ".join(cc) + f") / {float(SEQ_LEN)}")  # mean
        select_cols.append("LEAST(" + ", ".join(cc) + ")")           # min
        select_cols.append("GREATEST(" + ", ".join(cc) + ")")        # max
        feat_names += [f"{f}_h47", f"{f}_mean", f"{f}_min", f"{f}_max"]
    return select_cols, feat_names


def load_features(session, split):
    """Stream the reduced 124-feature matrix for one split (bounded memory)."""
    select_cols, feat_names = _feature_exprs()
    sql = (f"SELECT {LABEL_COL}, " + ", ".join(select_cols)
           + f" FROM {MULTIMODAL_TABLE} WHERE {SPLIT_COL} = '{split}'")
    Xs, ys, n = [], [], 0
    for pdf in session.sql(sql).to_pandas_batches():
        if len(pdf) == 0:
            continue
        arr = pdf.to_numpy(dtype=np.float32)
        ys.append(arr[:, 0].astype(int))
        Xs.append(np.ascontiguousarray(arr[:, 1:]))
        n += len(pdf)
        print(f"    [{split}] streamed {n:,} rows", flush=True)
    if not Xs:
        return np.empty((0, len(feat_names)), np.float32), np.empty((0,), int), feat_names
    return np.concatenate(Xs), np.concatenate(ys), feat_names


def main(session, seed=42, experiment_name="DETERIORATION_XGB_BASELINE"):
    from snowflake.ml.experiment import ExperimentTracking

    print("Loading DETERIORATION_MULTIMODAL splits (server-side aggregation)...", flush=True)
    X_tr, y_tr, feat_names = load_features(session, "train")
    X_va, y_va, _ = load_features(session, "val")
    X_te, y_te, _ = load_features(session, "test")

    def bal(y):
        return f"{len(y):,} rows, {int(y.sum()):,} pos ({100*y.mean():.2f}%)"
    print(f"  train: {bal(y_tr)}")
    print(f"  val:   {bal(y_va)}")
    print(f"  test:  {bal(y_te)}")
    print(f"  features per row: {X_tr.shape[1]}")

    spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)  # negative/positive
    params = dict(
        n_estimators=2000, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, objective="binary:logistic", eval_metric="auc",
        scale_pos_weight=float(spw), early_stopping_rounds=50, n_jobs=-1,
        tree_method="hist", random_state=seed,
    )
    clf = xgb.XGBClassifier(**params)

    exp = ExperimentTracking(session=session)
    exp.set_experiment(experiment_name)

    run_name = f"XGB_D{params['max_depth']}_LR003_N{params['n_estimators']}"
    with exp.start_run(run_name=run_name):
        exp.log_params({k: str(v) for k, v in params.items()})
        exp.log_params({
            "n_features": str(X_tr.shape[1]),
            "n_train": str(len(y_tr)),
            "n_val": str(len(y_va)),
            "n_test": str(len(y_te)),
            "pos_rate_train": f"{y_tr.mean():.4f}",
        })

        print("Training XGBoost (early stopping on val AUROC)...", flush=True)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        print(f"  best_iteration: {clf.best_iteration}")

        p_te = clf.predict_proba(X_te)[:, 1]
        auroc = roc_auc_score(y_te, p_te)
        auprc = average_precision_score(y_te, p_te)
        acc = accuracy_score(y_te, (p_te >= 0.5).astype(int))

        p_va = clf.predict_proba(X_va)[:, 1]
        val_auroc = roc_auc_score(y_va, p_va)
        val_auprc = average_precision_score(y_va, p_va)

        exp.log_metrics({
            "best_iteration": clf.best_iteration,
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
            "test_auroc": auroc,
            "test_auprc": auprc,
            "test_acc": acc,
        })

        try:
            exp.log_model(clf, model_name="deterioration_xgb_baseline")
            print("  model logged to experiment")
        except Exception as e:
            print(f"  log_model skipped: {e}")

    print("\n" + "=" * 48)
    print("XGBoost deterioration baseline (labs+vitals, no notes)")
    print("=" * 48)
    print(f"  Test AUROC : {auroc:.4f}")
    print(f"  Test AUPRC : {auprc:.4f}")
    print(f"  Test Acc   : {acc:.4f}  (threshold 0.5)")
    print("=" * 48)

    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1][:15]
    print("Top 15 features by gain importance:")
    for i in order:
        print(f"  {feat_names[i]:<28} {imp[i]:.4f}")

    return clf


if __name__ == "__main__":
    from snowflake_utils import get_snowpark_session
    main(get_snowpark_session())
