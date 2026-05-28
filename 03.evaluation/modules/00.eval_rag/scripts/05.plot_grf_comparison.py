from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

def get_model_file_name(model_name: str) -> str:
    """
    Description:
    Get standard prefix file name for friendly model names.

    Args:
        model_name (str): The friendly name of the model.

    Returns:
        str: The lowercase filename prefix.
    """
    if "Harrier" in model_name: return "harrier"
    if "KaLM" in model_name: return "kalm"
    if "Nemotron" in model_name: return "nemotron"
    if "Qwen" in model_name: return "qwen"
    return model_name.replace(" ", "_").lower()

def plot_grf_comparison(summary_csv: str, output_name: str) -> None:
    """
    Description:
    Plot Recall Comparison of GRF 0.0 vs GRF 1.0 using top configs summary.

    Args:
        summary_csv (str): Path to the summary CSV file.
        output_name (str): Path to save the scatter comparison plot.

    Returns:
        None
    """
    df = pd.read_csv(summary_csv)

    grf0_recalls = []
    grf1_recalls = []
    labels = []
    colors = []

    color_map = {
        'Harrier 27B': '#1f77b4',
        'KaLM 12B': '#ff7f0e',
        'Nemotron 8B': '#2ca02c',
        'Qwen 8B': '#d62728',
        'Qwen3 8B': '#d62728'
    }

    for _, row in df.iterrows():
        model = row['Model']
        model_file = get_model_file_name(model)
        reranker = row.get('Reranker', 'None')

        possible_paths = []
        if reranker != 'None':
            r_short = "qwen3_8b"
            if "0.6B" in str(reranker):
                r_short = "qwen3_0.6b"
            elif "4B" in str(reranker):
                r_short = "qwen3_4b"
            possible_paths.append(f"../outputs/results_tune/tuning_results_{model_file}_{r_short}.json")
        else:
            for r_short in ["qwen3_0.6b", "qwen3_4b", "qwen3_8b"]:
                possible_paths.append(f"../outputs/results_tune/tuning_results_{model_file}_{r_short}.json")
            possible_paths.append(f"../outputs/results_tune/tuning_results_{model_file}.json")

        json_path = None
        for p in possible_paths:
            if os.path.exists(p):
                json_path = p
                break

        if not json_path:
            print(f"Could not find a valid tuning results file for {model} (reranker: {reranker})")
            continue

        with open(json_path, 'r') as f:
            results = json.load(f)

        # Target parameters
        top_k_rag = row['top_k_rag']
        thresh_rag = row['thresh_rag']
        top_k_rerank = int(row['top_k_rerank'])
        thresh_rerank = row['thresh_rerank']
        use_reranker = top_k_rerank > 0 or thresh_rerank > 0.0

        recall_0 = None
        recall_1 = None

        for r in results:
            p = r['params']
            if (p['top_k_rag'] == top_k_rag and
                abs(p['threshold_rag'] - thresh_rag) < 1e-5 and
                p['use_reranker'] == use_reranker and
                (not use_reranker or p['top_k_reranker'] == top_k_rerank) and
                (not use_reranker or abs(p['threshold_reranker'] - thresh_rerank) < 1e-5)):

                if p['GRF'] == 0.0:
                    recall_0 = r['Mean Recall']
                elif p['GRF'] == 1.0:
                    recall_1 = r['Mean Recall']

        if recall_0 is not None and recall_1 is not None:
            grf0_recalls.append(recall_0)
            grf1_recalls.append(recall_1)
            labels.append(f"{model} ({row['Burden Range']})")
            colors.append(color_map.get(model, 'black'))
        else:
            print(f"Could not find matching configs for {model} {row['Burden Range']} in {json_path}")

    if not grf0_recalls:
        print("No data to plot.")
        return

    plt.figure(figsize=(8, 8))

    # Draw points
    for i in range(len(grf0_recalls)):
        plt.scatter(grf0_recalls[i], grf1_recalls[i], color=colors[i], s=100, alpha=0.8, edgecolors='white')

    # Draw y=x reference line
    min_val = min(min(grf0_recalls), min(grf1_recalls)) - 0.02
    max_val = max(max(grf0_recalls), max(grf1_recalls)) + 0.02
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y = x (Tie)')

    # Legend for models
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=m) for m, c in color_map.items()]
    handles.append(plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.5, label='Tie (GRF 0 = GRF 1)'))
    plt.legend(handles=handles, loc='upper left')

    plt.title('Recall Comparison: GRF 0.0 (Standardized Only) vs GRF 1.0 (Cleaned Only)', pad=20)
    plt.xlabel('Recall with GRF 0.0', fontsize=12)
    plt.ylabel('Recall with GRF 1.0', fontsize=12)

    # Set equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle=':', alpha=0.7)

    # Add text to explain regions
    plt.text(min_val + 0.01, max_val - 0.01, 'GRF 1.0 (Cleaned) is better',
             fontsize=10, color='green', alpha=0.7, ha='left', va='top')
    plt.text(max_val - 0.01, min_val + 0.01, 'GRF 0.0 (Standardized) is better',
             fontsize=10, color='blue', alpha=0.7, ha='right', va='bottom')

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"Plot saved to {output_name}")

if __name__ == "__main__":
    input_csv = "../outputs/tables/top_configs_summary.csv"
    output_path = "../outputs/figures/grf_comparison_scatter.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_grf_comparison(input_csv, output_path)
