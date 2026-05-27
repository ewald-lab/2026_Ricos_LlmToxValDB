import traceback
import os
import argparse
import joblib
import pandas as pd
import polars as pl
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm.contrib.concurrent import thread_map
from xgboost import XGBClassifier
import numpy as np
import optuna

def precision_recall_gain_curve(y_true, y_scores):
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = np.array(y_true)[sorted_indices]
    baseline_precision = np.mean(y_true)
    baseline_recall = np.sum(y_true) / len(y_true)
    precision_gains = []
    recall_gains = []

    for threshold in range(1, len(y_true) + 1):
        predicted_positive = y_true_sorted[:threshold]
        true_positives = np.sum(predicted_positive)
        precision = true_positives / threshold
        recall = true_positives / np.sum(y_true)
        precision_gains.append(precision - baseline_precision)
        recall_gains.append(recall - baseline_recall)
    return precision_gains, recall_gains

def get_auprg(y_true, y_scores):
    try:
        pgs, rgs = precision_recall_gain_curve(y_true, y_scores)
        area = 0.0
        for i in range(1, len(pgs)):
            area += (pgs[i] + pgs[i-1]) * (rgs[i] - rgs[i-1]) / 2.0
        return float(area)
    except:
        return 0.0

def tune_and_classify(
    dat, meta, label_column, n_splits, gpu_id, n_repeats, model_dir
):
    feature_cols = [c for c in dat.columns if c.startswith("Feature_")]
    x = dat[feature_cols].to_numpy()
    y = dat[label_column].astype(int).to_numpy()

    device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"
    tree_method = "hist" if gpu_id >= 0 else "auto"
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0

    # 1. Baseline Performance
    baseline_params = {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 6,
        "scale_pos_weight": pos_weight,
        "tree_method": tree_method,
        "device": device,
        "random_state": 42
    }

    def evaluate(params):
        cv_auprg = []
        for train_idx, val_idx in rskf.split(x, y):
            model = XGBClassifier(**params, objective="binary:logistic")
            model.fit(x[train_idx], y[train_idx])
            probs = model.predict_proba(x[val_idx])[:, 1]
            cv_auprg.append(get_auprg(y[val_idx], probs))
        return np.mean(cv_auprg)

    baseline_score = evaluate(baseline_params)

    # 2. Tuning with Optuna
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "scale_pos_weight": pos_weight,
            "tree_method": tree_method,
            "device": device,
            "random_state": 42
        }
        return evaluate(params)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15) # Fast but effective

    best_score = study.best_value
    best_params = study.best_params
    best_params.update({"scale_pos_weight": pos_weight, "tree_method": tree_method, "device": device, "random_state": 42})

    # 3. Decision: Use Tuned if better than Baseline
    final_params = best_params if best_score > baseline_score else baseline_params
    is_tuned = best_score > baseline_score

    # Final cross-val results with winner
    all_preds = []
    fold = 1
    for train_idx, val_idx in rskf.split(x, y):
        model = XGBClassifier(**final_params, objective="binary:logistic")
        model.fit(x[train_idx], y[train_idx])
        probs = model.predict_proba(x[val_idx])[:, 1]
        preds = model.predict(x[val_idx])

        repeat_num = (fold - 1) // n_splits + 1
        fold_num = (fold - 1) % n_splits + 1

        all_preds.append(pl.DataFrame({
            "DTXSID": list(meta.iloc[val_idx]["DTXSID"]),
            "y_prob": list(probs),
            "y_pred": list(preds),
            "y_actual": list(y[val_idx]),
            "k_fold": fold_num,
            "repeat": repeat_num,
        }))
        fold += 1

    # Full training for persistence
    final_model = XGBClassifier(**final_params, objective="binary:logistic")
    final_model.fit(x, y)

    safe_label = label_column.replace("/", "_").replace("\\", "_").replace(" ", "_")
    joblib.dump(final_model, os.path.join(model_dir, f"{safe_label}_model.joblib"))

    res_df = pl.concat(all_preds, how="vertical").with_columns(
        pl.lit(label_column).alias("Metadata_Label"),
        pl.lit(int((y==1).sum())).alias("Metadata_Count_1"),
        pl.lit(int((y==0).sum())).alias("Metadata_Count_0"),
        pl.lit(is_tuned).alias("Metadata_Is_Tuned")
    )
    return res_df

def process_label_tuned(dat, meta, label_column, n_splits, gpu_id, n_repeats, model_dir):
    try:
        num_1 = (dat[label_column] == 1).sum()
        num_0 = (dat[label_column] == 0).sum()
        if num_1 >= 5: # Same threshold we used for dataset creation
            return tune_and_classify(dat, meta, label_column, n_splits, gpu_id, n_repeats, model_dir)
        return None
    except Exception as e:
        print(f"Error in {label_column}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", required=True)
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--mapping_path", required=False)
    args = parser.parse_args()

    features = pd.read_parquet(args.features_path)
    labels = pd.read_csv(args.labels_path)

    if "DTXSID" not in features.columns:
        mapping = pd.read_csv(args.mapping_path)
        features = pd.merge(features, mapping, left_on="Metadata_compound", right_on="NAME", how="inner")

    merged = pd.merge(features, labels, on="DTXSID", how="inner")
    label_cols = [c for c in labels.columns if c != "DTXSID"]

    # Simple GPU detection
    gpu_id = 0
    model_dir = os.path.join(os.path.dirname(args.output), "models_tuned")
    os.makedirs(model_dir, exist_ok=True)

    print(f"Starting Winner-Stays Tuning for {len(label_cols)} labels...")
    results = thread_map(
        lambda col: process_label_tuned(merged, merged[["DTXSID"]], col, args.n_splits, gpu_id, 1, model_dir),
        label_cols,
        max_workers=2, # Tuning is heavy, keep it low for stability
        desc="Tuning labels"
    )

    results = [r for r in results if r is not None]
    if results:
        final_df = pl.concat(results, how="vertical")
        final_df.write_parquet(args.output)
        print(f"Tuning complete. Results saved to {args.output}")
