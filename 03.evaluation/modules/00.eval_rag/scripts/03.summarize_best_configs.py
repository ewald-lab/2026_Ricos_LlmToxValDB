import json
import os
import polars as pl

def get_top_configs(json_path, model_name):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # EXACT brackets requested by user
    brackets = [
        (0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 50)
    ]
    results = []

    for low, high in brackets:
        # Find configs in this burden range
        # Use a small epsilon for float comparison if needed, but here we use <= target
        # We want the best Mean Recall within the budget 'high'
        valid_configs = [c for c in data if c["Context Burden"] <= high]
        if not valid_configs:
            continue

        # Pick the one with best Recall (and then best MAP)
        best = max(valid_configs, key=lambda x: (x["Mean Recall"], x["MAP"]))

        results.append({
            "Model": model_name,
            "Burden Range": f"{low}-{high}",
            "Actual Burden": f"{best['Context Burden']:.2f}",
            "Recall": f"{best['Mean Recall']:.4f}",
            "MAP": f"{best['MAP']:.4f}",
            "top_k_rag": best["params"]["top_k_rag"],
            "thresh_rag": best["params"]["threshold_rag"],
            "top_k_rerank": best["params"]["top_k_reranker"],
            "thresh_rerank": best["params"]["threshold_reranker"],
            "GRF": best["params"]["GRF"]
        })
    return results

if __name__ == "__main__":
    res_dir = "/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/03.evaluation/modules/00.eval_rag/outputs/results_tune"
    all_summary = []

    files = {
        "tuning_results_harrier.json": "Harrier 27B",
        "tuning_results_kalm.json": "KaLM 12B",
        "tuning_results_nemotron.json": "Nemotron 8B",
        "tuning_results_qwen.json": "Qwen 8B"
    }

    for f, name in files.items():
        path = os.path.join(res_dir, f)
        if os.path.exists(path):
            all_summary.extend(get_top_configs(path, name))

    df = pl.DataFrame(all_summary)
    output_path = "../outputs/tables/top_configs_summary.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.write_csv(output_path)
