from __future__ import annotations
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def load_data(res_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(res_dir, "tuning_results_*.json"))
    all_configs = []
    
    for f_path in files:
        try:
            with open(f_path, "r") as f:
                data = json.load(f)
                for item in data:
                    p = item["params"]
                    all_configs.append({
                        "embedding_model": p.get("embedding_model"),
                        "reranker_model": p.get("reranker_model"),
                        "top_k_rag": p.get("top_k_rag"),
                        "threshold_rag": p.get("threshold_rag"),
                        "threshold_reranker": p.get("threshold_reranker"),
                        "top_k_reranker": p.get("top_k_reranker"),
                        "GRF": p.get("GRF"),
                        "data_variant": p.get("data_variant", "base"),
                        "use_reranker": p.get("use_reranker", False),
                        "Recall": item.get("Mean Recall"),
                        "MAP": item.get("MAP"),
                        "Burden": item.get("Context Burden")
                    })
        except Exception as e:
            print(f"Error reading {f_path}: {e}")
            
    return pd.DataFrame(all_configs)

def format_p_value(p_val: float) -> str:
    if np.isnan(p_val):
        return "N/A"
    if p_val == 0.0 or p_val < 1e-300:
        return "< 1e-300"
    return f"{p_val:.2e}"

def process_and_plot(df: pd.DataFrame, metric: str, output_path: str) -> None:
    # 1. Map Definitions & Hierarchy
    df["Definitions"] = df["data_variant"].apply(lambda x: "With" if x in ["def", "def_hier"] else "Without")
    df["Hierarchy"] = df["data_variant"].apply(lambda x: "With" if x in ["hier", "def_hier"] else "Without")
    
    # Perform Paired t-tests
    # A. Definitions
    p_val_def = np.nan
    try:
        pivot_def = df.pivot_table(
            index=["embedding_model", "reranker_model", "top_k_rag", "threshold_rag", "threshold_reranker", "top_k_reranker", "GRF", "Hierarchy"],
            columns="Definitions",
            values=metric
        ).dropna()
        if len(pivot_def) >= 2:
            stat_res = stats.ttest_rel(pivot_def["With"], pivot_def["Without"])
            p_val_def = stat_res.pvalue
    except Exception as e:
        print(f"Error calculating t-test for Definitions: {e}")

    # B. Hierarchy
    p_val_hier = np.nan
    try:
        pivot_hier = df.pivot_table(
            index=["embedding_model", "reranker_model", "top_k_rag", "threshold_rag", "threshold_reranker", "top_k_reranker", "GRF", "Definitions"],
            columns="Hierarchy",
            values=metric
        ).dropna()
        if len(pivot_hier) >= 2:
            stat_res = stats.ttest_rel(pivot_hier["With"], pivot_hier["Without"])
            p_val_hier = stat_res.pvalue
    except Exception as e:
        print(f"Error calculating t-test for Hierarchy: {e}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    palette_colors = {"Without": "#ff7f0e", "With": "#1f77b4"}
    
    # 1. Definitions Plot
    sns.boxplot(
        data=df,
        x="Definitions",
        y=metric,
        ax=axes[0],
        palette=palette_colors,
        width=0.4,
        boxprops=dict(alpha=0.6),
        showfliers=False
    )
    title_def = f"Definitions\np-value: {format_p_value(p_val_def)}"
    axes[0].set_title(title_def, fontsize=12, fontweight="bold")
    axes[0].set_ylabel(metric, fontsize=12)
    axes[0].set_xlabel("Reranker Context", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # 2. Hierarchy Plot
    sns.boxplot(
        data=df,
        x="Hierarchy",
        y=metric,
        ax=axes[1],
        palette=palette_colors,
        width=0.4,
        boxprops=dict(alpha=0.6),
        showfliers=False
    )
    title_hier = f"Hierarchy\np-value: {format_p_value(p_val_hier)}"
    axes[1].set_title(title_hier, fontsize=12, fontweight="bold")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Reranker Context", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(f"Impact of RAG Architecture Decisions on {metric}", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved decision impact plot to: {output_path}")

if __name__ == "__main__":
    res_dir = "../outputs/results_tune"
    df = load_data(res_dir)
    if not df.empty:
        # Plot Recall
        process_and_plot(df, "Recall", "../outputs/figures/decisions/combined_impact_recall.png")
        # Plot MAP
        process_and_plot(df, "MAP", "../outputs/figures/decisions/combined_impact_map.png")
    else:
        print("No tuning results data found.")
