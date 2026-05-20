import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison_with_highlights(summary_csv, output_name):
    df = pd.read_csv(summary_csv)
    
    def get_x(range_str):
        return float(range_str.split('-')[1])
    
    df['x'] = df['Burden Range'].apply(get_x)
    df = df.sort_values(['Model', 'x'])
    
    plt.figure(figsize=(12, 7))
    
    models = df['Model'].unique()
    markers = ['o', 's', '^', 'D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    ax = plt.gca()
    
    for i, model in enumerate(models):
        model_df = df[df['Model'] == model]
        # Only plot unique x values to avoid duplicate points
        model_df = model_df.drop_duplicates('x')
        
        ax.plot(model_df['x'], model_df['Recall'], 
                 marker=markers[i % len(markers)], 
                 label=model, 
                 linewidth=2, 
                 markersize=8,
                 color=colors[i % len(colors)])
        
        # Highlight KaLM 12B at Context 20
        if model == "KaLM 12B":
            match = model_df[model_df['x'] == 20]
            if not match.empty:
                row = match.iloc[0]
                # Draw the highlight circle
                ax.scatter(row['x'], row['Recall'], s=350, facecolors='none', edgecolors='black', linewidth=1.5, zorder=10)
                
                # Parameters text with Title Case and no underscores
                params_str = (f"Best Configuration (KaLM 12B):\n"
                             f"Top K RAG = {row['top_k_rag']}\n"
                             f"Threshold RAG = {row['thresh_rag']}\n"
                             f"Top K Rerank = {row['top_k_rerank']}\n"
                             f"Threshold Rerank = {row['thresh_rerank']}")
                
                props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1)
                
                # Annotation with arrow pointing to the circle
                ax.annotate(params_str, 
                            xy=(row['x'], row['Recall']), 
                            xytext=(row['x'] + 5, 0.985), # Nudged slightly higher
                            bbox=props,
                            arrowprops=dict(arrowstyle='->', 
                                            connectionstyle='arc3,rad=.1', 
                                            color='black', 
                                            linewidth=1.5,
                                            shrinkB=12), 
                            fontsize=8,
                            fontweight='bold',
                            zorder=20)

    plt.xlabel('Max Context Burden', fontsize=12)
    plt.ylabel('Mean Recall', fontsize=12)
    # No title as requested
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right', framealpha=1.0, fontsize=10)
    
    plt.ylim(0.90, 1.0)
    # Exact requested ticks
    plt.xticks([5, 10, 15, 20, 30, 40, 50])
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
