import json
import os
import polars as pl

def get_top_configs(json_path, model_name):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Map friendly model name to its embedding_model identifier in JSON
    model_mapping = {
        "Harrier 27B": "microsoft/harrier-oss-v1-27b",
        "KaLM 12B": "tencent/KaLM-Embedding-Gemma3-12B-2511",
        "Nemotron 8B": "nvidia/llama-embed-nemotron-8b",
        "Qwen 8B": "Qwen/Qwen3-Embedding-8B"
    }
    target_model = model_mapping.get(model_name)
    if target_model:
        # Check both top-level and param-level embedding_model key
        data = [
            c for c in data 
            if c.get("embedding_model") == target_model or c.get("params", {}).get("embedding_model") == target_model
        ]

    # EXACT brackets requested by user
    brackets = [
        (0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 50)
    ]
    results = []
    best_at_20 = None  # best config within burden <= 20 (same as plot highlight)

    for low, high in brackets:
        # Find configs in this burden range
        valid_configs = [c for c in data if c["Context Burden"] <= high]
        if not valid_configs:
            continue

        # Pick the one with best Recall (prioritizing GRF == 1.0 in case of a tie, then MAP)
        best = max(valid_configs, key=lambda x: (x["Mean Recall"], x["params"].get("GRF", 0.0) == 1.0, x["MAP"]))

        if high == 20:
            best_at_20 = best

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
    return results, best_at_20


def make_best_config_json(best, model_name):
    """Builds a best_config.json dict from the best entry (same as plot bracket 15-20)."""
    p = best["params"]
    model_mapping = {
        "Harrier 27B": "microsoft/harrier-oss-v1-27b",
        "KaLM 12B": "tencent/KaLM-Embedding-Gemma3-12B-2511",
        "Nemotron 8B": "nvidia/llama-embed-nemotron-8b",
        "Qwen 8B": "Qwen/Qwen3-Embedding-8B"
    }
    return {
        "embedding_model": model_mapping.get(model_name, model_name),
        "Mean Recall": best["Mean Recall"],
        "MAP": best["MAP"],
        "Context Burden": best["Context Burden"],
        "top_k_rag": p["top_k_rag"],
        "threshold_rag": p["threshold_rag"],
        "reranker_model": "Qwen/Qwen3-Reranker-8B",
        "top_k_reranker": p.get("top_k_reranker", p.get("top_k_rerank", 10)),
        "threshold_reranker": p["threshold_reranker"],
        "reranker_batch_size": p.get("reranker_batch_size", 32),
        "reranker_prompt": p.get("reranker_prompt", ""),
        "grf": p.get("GRF", p.get("grf", 1.0))
    }


if __name__ == "__main__":
    res_dir = "/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/03.evaluation/modules/00.eval_rag/outputs/results_tune"
    all_summary = []

    files = {
        "tuning_results_harrier.json": "Harrier 27B",
        "tuning_results_kalm.json": "KaLM 12B",
        "tuning_results_nemotron.json": "Nemotron 8B",
        "tuning_results_qwen.json": "Qwen 8B"
    }

    json_output_dir = "../outputs/best_configs"
    os.makedirs(json_output_dir, exist_ok=True)

    for f, name in files.items():
        path = os.path.join(res_dir, f)
        if os.path.exists(path):
            summary_rows, best_at_20 = get_top_configs(path, name)
            all_summary.extend(summary_rows)

            # Save best_config.json only for KaLM (same point highlighted in the plot)
            if name == "KaLM 12B" and best_at_20 is not None:
                best_cfg = make_best_config_json(best_at_20, name)
                json_out_path = os.path.join(json_output_dir, "best_config.json")
                with open(json_out_path, 'w') as jf:
                    json.dump(best_cfg, jf, indent=4)
                print(f"Saved {json_out_path}")

    df = pl.DataFrame(all_summary)
    output_path = "../outputs/tables/top_configs_summary.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.write_csv(output_path)
