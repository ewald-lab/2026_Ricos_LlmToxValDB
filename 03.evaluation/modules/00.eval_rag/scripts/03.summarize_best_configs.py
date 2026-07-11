from __future__ import annotations
import json
import os
import polars as pl

def get_top_configs(data: list[dict], model_name: str, target_reranker: str) -> tuple[list[dict], dict | None]:
    """
    Description:
    Get top configurations for the specified embedding model and target reranker across different burden brackets.

    Args:
        data (list[dict]): The list of configuration evaluation results.
        model_name (str): The embedding model name.
        target_reranker (str): The target reranker model name.

    Returns:
        tuple[list[dict], dict | None]: A tuple containing the summary rows list and the best config at burden 20.
    """
    # Map friendly model name to its embedding_model identifier in JSON
    model_mapping = {
        "Harrier 27B": "microsoft/harrier-oss-v1-27b",
        "KaLM 12B": "tencent/KaLM-Embedding-Gemma3-12B-2511",
        "Nemotron 8B": "nvidia/llama-embed-nemotron-8b",
        "Qwen 8B": "Qwen/Qwen3-Embedding-8B"
    }
    target_model = model_mapping.get(model_name)
    if target_model:
        # Check both top-level and param-level embedding_model key, and only keep target reranker or non-reranker
        data = [
            c for c in data 
            if (c.get("embedding_model") == target_model or c.get("params", {}).get("embedding_model") == target_model)
            and (not c.get("params", {}).get("use_reranker", c.get("use_reranker", False)) or 
                 c.get("reranker_model", c.get("params", {}).get("reranker_model")) == target_reranker)
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
            "Reranker": target_reranker,
            "Burden Range": f"{low}-{high}",
            "Actual Burden": f"{best['Context Burden']:.2f}",
            "Recall": f"{best['Mean Recall']:.4f}",
            "MAP": f"{best['MAP']:.4f}",
            "top_k_rag": best["params"]["top_k_rag"],
            "thresh_rag": best["params"]["threshold_rag"],
            "top_k_rerank": best["params"]["top_k_reranker"],
            "thresh_rerank": best["params"]["threshold_reranker"],
            "GRF": best["params"]["GRF"],
            "Used Reranker": best["params"].get("use_reranker", False),
            "Variant": best["params"].get("data_variant", "base")
        })
    return results, best_at_20


def make_best_config_json(best: dict, model_name: str) -> dict:
    """
    Description:
    Builds a best_config.json dict from the best entry (same as plot bracket 15-20).

    Args:
        best (dict): The best configuration entry.
        model_name (str): The embedding model name.

    Returns:
        dict: A dictionary of the best configuration formatted for JSON export.
    """
    p = best["params"]
    model_mapping = {
        "Harrier 27B": "microsoft/harrier-oss-v1-27b",
        "KaLM 12B": "tencent/KaLM-Embedding-Gemma3-12B-2511",
        "Nemotron 8B": "nvidia/llama-embed-nemotron-8b",
        "Qwen 8B": "Qwen/Qwen3-Embedding-8B"
    }
    return {
        "embedding_model": model_mapping.get(model_name, model_name),
        "Mean Recall": round(best["Mean Recall"], 2),
        "MAP": round(best["MAP"], 2),
        "Context Burden": round(best["Context Burden"], 2),
        "top_k_rag": p["top_k_rag"],
        "threshold_rag": p["threshold_rag"],
        "reranker_model": p.get("reranker_model", "Qwen/Qwen3-Reranker-8B"),
        "top_k_reranker": p.get("top_k_reranker", p.get("top_k_rerank", 10)),
        "threshold_reranker": p["threshold_reranker"],
        "reranker_batch_size": p.get("reranker_batch_size", 32),
        "reranker_prompt": p.get("reranker_prompt", ""),
        "grf": p.get("GRF", p.get("grf", 1.0)),
        "data_variant": p.get("data_variant", "base")
    }


if __name__ == "__main__":
    res_dir = "/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/03.evaluation/modules/00.eval_rag/outputs/results_tune"
    all_summary = []

    models_info = {
        "harrier": "Harrier 27B",
        "kalm": "KaLM 12B",
        "nemotron": "Nemotron 8B",
        "qwen": "Qwen 8B"
    }
    rerankers_info = {
        "Qwen/Qwen3-Reranker-0.6B": "qwen3_0.6b",
        "Qwen/Qwen3-Reranker-4B": "qwen3_4b",
        "Qwen/Qwen3-Reranker-8B": "qwen3_8b"
    }
    variants = ["base", "def", "hier", "def_hier"]

    json_output_dir = "../outputs/best_configs"
    os.makedirs(json_output_dir, exist_ok=True)

    for prefix, name in models_info.items():
        for reranker_full, r_short in rerankers_info.items():
            combined_data = []
            
            for var in variants:
                path = os.path.join(res_dir, f"tuning_results_{prefix}_{r_short}_{var}.json")
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            combined_data.extend(json.load(f))
                    except Exception as e:
                        print(f"Error reading {path}: {e}")

            if combined_data:
                summary_rows, best_at_20 = get_top_configs(combined_data, name, reranker_full)
                all_summary.extend(summary_rows)

                # Save best_config.json only for Qwen 8B + Qwen3-Reranker-0.6B
                if name == "Qwen 8B" and reranker_full == "Qwen/Qwen3-Reranker-0.6B" and best_at_20 is not None:
                    best_cfg = make_best_config_json(best_at_20, name)
                    json_out_path = "../config/best_config.json"
                    os.makedirs(os.path.dirname(json_out_path), exist_ok=True)
                    with open(json_out_path, 'w') as jf:
                        json.dump(best_cfg, jf, indent=4)
                    print(f"Saved {json_out_path}")

    df = pl.DataFrame(all_summary)
    output_path = "../outputs/tables/top_configs_summary.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.write_csv(output_path)
