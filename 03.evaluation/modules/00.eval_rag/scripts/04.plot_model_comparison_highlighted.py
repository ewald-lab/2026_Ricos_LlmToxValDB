from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison_with_highlights(summary_csv: str, output_name: str) -> None:
    """
    Description:
    Plot model recall comparison across different max context burdens and add a highlight for the best KaLM 12B config at context burden 20.

    Args:
        summary_csv (str): Path to the summary CSV file containing configurations.
        output_name (str): Path to save the generated comparison plot.

    Returns:
        None
    """
    df = pd.read_csv(summary_csv)

    def get_x(range_str: str) -> float:
        """
        Description:
        Extract the upper bound of a range string as a float.

        Args:
            range_str (str): The range string (e.g., '15-20').

        Returns:
            float: The upper bound value.
        """
        return float(range_str.split('-')[1])

    df['x'] = df['Burden Range'].apply(get_x)

    df['Recall'] = df['Recall'].astype(float)
    df['MAP'] = df['MAP'].astype(float)

    df = df.sort_values(['Model', 'Reranker', 'x'])

    plt.figure(figsize=(14, 8))

    models = df['Model'].unique()
    colors = {
        "Harrier 27B": "#1f77b4",
        "KaLM 12B": "#ff7f0e",
        "Nemotron 8B": "#2ca02c",
        "Qwen 8B": "#d62728"
    }

    reranker_styles = {
        "Qwen/Qwen3-Reranker-0.6B": {"linestyle": ":", "marker": "v", "label_suffix": "(0.6B Reranker)"},
        "Qwen/Qwen3-Reranker-4B": {"linestyle": "--", "marker": "s", "label_suffix": "(4B Reranker)"},
        "Qwen/Qwen3-Reranker-8B": {"linestyle": "-", "marker": "o", "label_suffix": "(8B Reranker)"}
    }

    ax = plt.gca()

    for model in models:
        model_df = df[df['Model'] == model]
        color = colors.get(model, '#7f7f7f')

        for reranker_model, style in reranker_styles.items():
            sub_df = model_df[model_df['Reranker'] == reranker_model]
            if sub_df.empty:
                continue

            # Sort by x to draw the line correctly
            sub_df = sub_df.sort_values('x').drop_duplicates('x')

            label_name = f"{model} {style['label_suffix']}"
            ax.plot(sub_df['x'], sub_df['Recall'],
                     linestyle=style['linestyle'],
                     marker=style['marker'],
                     label=label_name,
                     linewidth=1.8,
                     markersize=6,
                     color=color)

    # Highlight the Qwen3-Embedding-8B, Qwen3-Reranker-0.6B configuration at Context 20
    qwen_df = df[(df['Model'] == "Qwen 8B") & (df['Reranker'] == "Qwen/Qwen3-Reranker-0.6B")]
    qwen_20 = qwen_df[qwen_df['x'] == 20]
    if not qwen_20.empty:
        best_row = qwen_20.iloc[0]
        
        # Draw the highlight circle
        ax.scatter(best_row['x'], best_row['Recall'], s=350, facecolors='none', edgecolors='black', linewidth=1.5, zorder=10)

        # Get the reranker name
        reranker_name = best_row.get('Reranker', 'None')
        if reranker_name != 'None' and '/' in str(reranker_name):
            reranker_name = str(reranker_name).split('/')[-1]

        # Parameters text with MathText to selectively bold keys and equals sign
        params_str = (r"$\bf{Model\ =}$ Qwen3-Embedding-8B" + "\n"
                      r"$\bf{Top\ K\ RAG\ =}$ " + f"{best_row['top_k_rag']}\n"
                      r"$\bf{Threshold\ RAG\ =}$ " + f"{best_row['thresh_rag']}\n"
                      r"$\bf{Reranker\ =}$ " + f"{reranker_name}\n"
                      r"$\bf{Top\ K\ Rerank\ =}$ " + f"{best_row['top_k_rerank']}\n"
                      r"$\bf{Threshold\ Rerank\ =}$ " + f"{best_row['thresh_rerank']}")

        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1)

        # Annotation with arrow pointing to the circle
        ax.annotate(params_str,
                    xy=(best_row['x'], best_row['Recall']),
                    xytext=(best_row['x'] + 5, 0.92), # Move box slightly higher
                    bbox=props,
                    arrowprops=dict(arrowstyle='->',
                                    connectionstyle='arc3,rad=.1',
                                    color='black',
                                    linewidth=1.5,
                                    shrinkB=12),
                    fontsize=8,
                    fontweight='normal',
                    zorder=20)

    plt.xlabel('Max Context Burden', fontsize=12)
    plt.ylabel('Mean Recall', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right', framealpha=1.0, fontsize=8, ncol=2) # two columns legend to fit all 12 lines nicely

    plt.ylim(0.90, 1.0)
    plt.xticks([5, 10, 15, 20, 30, 40, 50])
    plt.yticks([0.90, 0.95, 1.00])

    plt.xlim(0, 55)

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"Perfected comparison plot saved to {output_name}")

if __name__ == "__main__":
    input_csv = "../outputs/tables/top_configs_summary.csv"
    output_path = "../outputs/figures/model_comparison_highlighted.png"
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_comparison_with_highlights(input_csv, output_path)
