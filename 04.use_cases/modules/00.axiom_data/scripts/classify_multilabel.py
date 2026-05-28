from __future__ import annotations
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

try:
    import cupy as cp
except ImportError:
    import numpy as cp

def multi_label_classifier(
    dat: pd.DataFrame,
    meta: pd.DataFrame,
    n_splits: int,
    gpu_id: int,
    label_column: str,
    *,
    shuffle: bool = False,
    n_repeats: int = 1,
) -> tuple[XGBClassifier, pl.DataFrame]:
    """
    Description:
    Performs binary classification for a specific label in a multi-label context using RepeatedStratifiedKFold
    and XGBoost. It handles class imbalance and returns the trained final model alongside validation predictions.

    Args:
        dat (pd.DataFrame): The DataFrame containing features and target labels.
        meta (pd.DataFrame): The metadata DataFrame corresponding to `dat`.
        n_splits (int): Number of folds for stratified cross-validation.
        gpu_id (int): GPU device ID (negative value to run on CPU).
        label_column (str): Name of the target label column.
        shuffle (bool): If True, shuffles labels to establish a random control. Defaults to False.
        n_repeats (int): Number of cross-validation repeats. Defaults to 1.

    Returns:
        tuple[XGBClassifier, pl.DataFrame]: A tuple containing the trained final XGBClassifier and a Polars DataFrame of cross-validation predictions.
    """
    # Features are 'Feature_' columns
    feature_cols = [c for c in dat.columns if c.startswith("Feature_")]
    x = dat[feature_cols]
    y = dat[label_column].astype(int)

    if shuffle:
        y = y.sample(frac=1, random_state=42).reset_index(drop=True)

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    pred_df = []
    fold = 1

    device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"
    tree_method = "hist" if gpu_id >= 0 else "auto"

    # We use CPU or GPU based on availability
    # Convert features to numpy for XGBoost consistency
    x_np = x.to_numpy()
    y_np = y.to_numpy()

    for train_index, val_index in rskf.split(x_np, y_np):
        x_fold_train, x_fold_val = x_np[train_index], x_np[val_index]
        y_fold_train, y_fold_val = y_np[train_index], y_np[val_index]

        meta_fold_val = meta.iloc[val_index]

        # Handle class imbalance
        pos_count = (y_fold_train == 1).sum()
        neg_count = (y_fold_train == 0).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        # Initialize the model
        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=150,
            tree_method=tree_method,
            device=device,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )

        # Train
        model.fit(x_fold_train, y_fold_train)

        # Predict
        y_fold_prob = model.predict_proba(x_fold_val)[:, 1]
        y_fold_pred = model.predict(x_fold_val)

        repeat_num = (fold - 1) // n_splits + 1
        fold_num = (fold - 1) % n_splits + 1

        pred_df.append(
            pl.DataFrame({
                "DTXSID": list(meta_fold_val["DTXSID"]),
                "y_prob": list(y_fold_prob),
                "y_pred": list(y_fold_pred),
                "y_actual": list(y_fold_val),
                "k_fold": fold_num,
                "repeat": repeat_num,
            }),
        )
        fold += 1

    # Train final model on all data
    final_model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=150,
        tree_method=tree_method,
        device=device,
        learning_rate=0.05,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0,
        random_state=42
    )
    final_model.fit(x_np, y_np)

    return final_model, pl.concat(pred_df, how="vertical")

def process_label(
    dat: pd.DataFrame,
    meta: pd.DataFrame,
    label_column: str,
    n_splits: int,
    gpu_id: int,
    n_repeats: int,
    shuffle: bool,
    output_dir: str,
) -> pl.DataFrame | None:
    """
    Description:
    Wrapper method to process classification for a single target label, verifying sample counts,
    saving the trained model, and adding label metadata to predictions.

    Args:
        dat (pd.DataFrame): The features and labels DataFrame.
        meta (pd.DataFrame): The metadata DataFrame.
        label_column (str): Name of the target label column.
        n_splits (int): Number of splits for cross-validation.
        gpu_id (int): GPU device ID to use.
        n_repeats (int): Number of cross-validation repeats.
        shuffle (bool): Whether to shuffle labels for random control.
        output_dir (str): Directory where models will be saved.

    Returns:
        pl.DataFrame | None: Predictions DataFrame if processed successfully, or None if skipped or on error.
    """
    try:
        num_1 = (dat[label_column] == 1).sum()
        num_0 = (dat[label_column] == 0).sum()

        if (num_0 >= n_splits) and (num_1 >= n_splits):
            model, class_res = multi_label_classifier(
                dat,
                meta,
                n_splits=n_splits,
                gpu_id=gpu_id,
                label_column=label_column,
                shuffle=shuffle,
                n_repeats=n_repeats,
            )

            if output_dir and model is not None:
                model_type = "random" if shuffle else "actual"
                safe_label = label_column.replace("/", "_").replace("\\", "_").replace(" ", "_")
                model_path = os.path.join(output_dir, f"{safe_label}_{model_type}_model.joblib")
                joblib.dump(model, model_path)

            class_res = class_res.with_columns(
                pl.lit(label_column).alias("Metadata_Label"),
                pl.lit(int(num_0)).alias("Metadata_Count_0"),
                pl.lit(int(num_1)).alias("Metadata_Count_1"),
            )
            return class_res
        else:
            print(f"Skipping {label_column}: not enough samples (0s: {num_0}, 1s: {num_1})")
            return None
    except Exception as e:
        print(f"Error in {label_column}: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", required=True)
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_repeats", type=int, default=1)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--mapping_path", required=False, help="Path to name-to-DTXSID mapping CSV")
    args = parser.parse_args()

    # Load data
    print(f"Loading features from {args.features_path}...")
    features = pd.read_parquet(args.features_path)
    print(f"Loading labels from {args.labels_path}...")
    labels = pd.read_csv(args.labels_path)

    # Resolve DTXSID if not present in features
    if "DTXSID" not in features.columns:
        if args.mapping_path and os.path.exists(args.mapping_path):
            print(f"Mapping DTXSID using {args.mapping_path}...")
            mapping = pd.read_csv(args.mapping_path)
            # Use Metadata_compound to map to DTXSID
            features = pd.merge(features, mapping, left_on="Metadata_compound", right_on="NAME", how="inner")
            print(f"Features after mapping: {len(features)}")
        else:
            print("Error: 'DTXSID' not in features and no mapping file provided.")
            exit(1)

    # Merge on DTXSID
    print("Merging features and labels...")
    merged = pd.merge(features, labels, on="DTXSID", how="inner")
    print(f"Total merged samples: {len(merged)}")

    # Identify label columns (everything not Feature_ or DTXSID)
    label_cols = [c for c in labels.columns if c != "DTXSID"]
    meta_cols = ["DTXSID"]

    num_gpus = 0
    try:
        import torch
        num_gpus = torch.cuda.device_count()
    except:
        try:
            num_gpus = cp.cuda.runtime.getDeviceCount()
        except:
            num_gpus = 0

    if num_gpus == 0:
        print("No GPU detected, using CPU")
        num_workers = 8
    else:
        print(f"Using {num_gpus} GPUs")
        num_workers = num_gpus

    model_dir = os.path.join(os.path.dirname(args.output), "models")
    os.makedirs(model_dir, exist_ok=True)

    tasks = [
        (merged, merged[meta_cols], col, args.n_splits, i % max(1, num_gpus), args.n_repeats, False, model_dir)
        for i, col in enumerate(label_cols)
    ]

    print(f"Starting classification for {len(label_cols)} labels...")
    results = thread_map(
        lambda t: process_label(*t),
        tasks,
        max_workers=num_workers,
        desc="Classifying labels"
    )

    results = [r for r in results if r is not None]
    if results:
        final_df = pl.concat(results, how="vertical")
        final_df.write_parquet(args.output)
        print(f"Results saved to {args.output}")
    else:
        print("No labels were processed.")
