import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import mannwhitneyu
import os
import argparse

def precision_recall_gain_curve(y_true, y_scores):
    # Sort predictions and true labels by descending score
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = np.array(y_true)[sorted_indices]

    # Calculate baseline precision and recall
    baseline_precision = np.mean(y_true)
    baseline_recall = np.sum(y_true) / len(y_true)

    precisions = []
    recalls = []
    precision_gains = []
    recall_gains = []

    for threshold in range(1, len(y_true) + 1):
        # Predictions above threshold considered positive
        predicted_positive = y_true_sorted[:threshold]

        # Calculate precision and recall
        true_positives = np.sum(predicted_positive)
        precision = true_positives / threshold
        recall = true_positives / np.sum(y_true)

        # Calculate precision gain and recall gain
        precision_gain = precision - baseline_precision
        recall_gain = recall - baseline_recall

        precisions.append(precision)
        recalls.append(recall)
        precision_gains.append(precision_gain)
        recall_gains.append(recall_gain)

    return precisions, recalls, precision_gains, recall_gains

def prg_auc(y_true, y_scores):
    try:
        _, _, pgs, rgs = precision_recall_gain_curve(y_true, y_scores)
        # Manual trapezoidal integration to avoid Numpy 2.0 np.trapz removal
        pgs = np.array(pgs)
        rgs = np.array(rgs)
        area = 0.0
        for i in range(1, len(pgs)):
            area += (pgs[i] + pgs[i-1]) * (rgs[i] - rgs[i-1]) / 2.0
        return float(area)
    except:
        return 0.0

def analyze_results(input_path, output_dir, suffix=""):
    # Load data
    df = pl.read_parquet(input_path).to_pandas()

    # Create directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    metrics_list = []
    file_suffix = f"_{suffix}" if suffix else ""

    labels = df['Metadata_Label'].unique()
    print(f"Analyzing {len(labels)} labels with AUROC, PR-AUC and AUPRG...")

    for label in labels:
        label_df = df[df['Metadata_Label'] == label]
        y_true = label_df['y_actual'].values
        y_prob = label_df['y_prob'].values

        if len(np.unique(y_true)) < 2:
            continue

        # ROC AUC
        try:
            auroc = roc_auc_score(y_true, y_prob)
        except:
            auroc = 0.5

        # PR AUC
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        prauc = auc(recall, precision)

        # AUPRG
        auprg_val = prg_auc(y_true, y_prob)

        # P-Value (Mann-Whitney U test)
        try:
            pos_scores = y_prob[y_true == 1]
            neg_scores = y_prob[y_true == 0]
            _, p_val = mannwhitneyu(pos_scores, neg_scores, alternative='greater')
        except:
            p_val = 1.0

        metrics_list.append({
            'Label': label,
            'Total': len(label_df),
            'Positives': np.sum(y_true),
            'AUROC': auroc,
            'PR-AUC': prauc,
            'AUPRG': auprg_val,
            'P-Value': p_val
        })

    metrics_df = pd.DataFrame(metrics_list)

    # Decision Labels
    def get_decision(row):
        if row['AUROC'] > 0.7 and row['AUPRG'] > 0.3:
            return 'GOLD'
        elif row['AUROC'] > 0.5 and row['AUPRG'] > 0:
            return 'SILVER'
        else:
            return 'NOISE'

    metrics_df['Decision'] = metrics_df.apply(get_decision, axis=1)
    metrics_df = metrics_df.sort_values('AUPRG', ascending=False)
    metrics_df.to_csv(os.path.join(output_dir, f"summary_metrics{file_suffix}.csv"), index=False)

    # Manual Heatmap / Table with custom colors
    height = max(15, len(metrics_df) * 0.3)
    fig, ax = plt.subplots(figsize=(16, height))

    # Define columns to show
    show_cols = ['N Compounds', 'AUROC', 'AUPRG', 'P-Value', 'Decision']
    data = metrics_df.rename(columns={'Positives': 'N Compounds'}).set_index('Label')[show_cols]

    # Create color matrix (mapped to 0-1 for heatmap)
    color_matrix = np.zeros(data.shape)
    # Create annotation matrix (strings)
    annot_matrix = []

    for i, (idx, row) in enumerate(data.iterrows()):
        row_annots = []
        # N Compounds
        color_matrix[i, 0] = 1 if row['N Compounds'] > 10 else 0
        row_annots.append(str(int(row['N Compounds'])))

        # AUROC
        if row['AUROC'] > 0.7: color_matrix[i, 1] = 1
        elif row['AUROC'] > 0.5: color_matrix[i, 1] = 0.5
        else: color_matrix[i, 1] = 0
        row_annots.append(f"{row['AUROC']:.3f}")

        # AUPRG
        if row['AUPRG'] > 0.3: color_matrix[i, 2] = 1
        elif row['AUPRG'] > 0: color_matrix[i, 2] = 0.5
        else: color_matrix[i, 2] = 0
        row_annots.append(f"{row['AUPRG']:.3f}")

        # P-Value
        if row['P-Value'] < 0.05: color_matrix[i, 3] = 1
        elif row['P-Value'] < 0.1: color_matrix[i, 3] = 0.5
        else: color_matrix[i, 3] = 0
        row_annots.append(f"{row['P-Value']:.2e}")

        # Decision
        if row['Decision'] == 'GOLD': color_matrix[i, 4] = 1
        elif row['Decision'] == 'SILVER': color_matrix[i, 4] = 0.5
        else: color_matrix[i, 4] = 0
        row_annots.append(row['Decision'])

        annot_matrix.append(row_annots)

    from matplotlib.colors import ListedColormap
    # Colors: Red (#ff9999), Yellow (#ffffb2), Green (#91cf60)
    custom_cmap = ListedColormap(['#ff9999', '#ffffb2', '#91cf60'])

    sns.heatmap(color_matrix, annot=np.array(annot_matrix), fmt="", cmap=custom_cmap,
                cbar=False, linewidths=.5, annot_kws={"size": 10},
                xticklabels=show_cols, yticklabels=data.index)

    plt.title(f'Tox Classification Dashboard (Custom Thresholds)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"metrics_table{file_suffix}.png"))

    print(f"Analysis complete. Unified dashboard saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    analyze_results(args.input, args.output_dir, args.suffix)
