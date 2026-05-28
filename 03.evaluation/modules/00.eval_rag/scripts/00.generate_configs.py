import json
import itertools
import argparse

def generate_grid_combinations(
    output_path: str,
    embedding_models: list[str],
    top_k_rags: list[int],
    threshold_rags: list[float],
    threshold_reranker: list[float],
    top_k_reranks: list[int],
    grf_alphas: list[float],
    reranker_models: list[str]
) -> None:
    """
    Description:
    Generate a grid of tuning combinations for evaluation.

    Args:
        output_path (str): Path to write the combinations JSON file.
        embedding_models (list[str]): List of embedding model paths/names.
        top_k_rags (list[int]): List of top K RAG candidate retrieval values.
        threshold_rags (list[float]): List of threshold scores for RAG.
        threshold_reranker (list[float]): List of threshold scores for the reranker.
        top_k_reranks (list[int]): List of top K candidates to keep after reranking.
        grf_alphas (list[float]): List of alpha values for graph regulation filtering.
        reranker_models (list[str]): List of reranker model paths/names.

    Returns:
        None
    """


    # Pathway 1: No reranker
    no_rerank_combos = list(itertools.product(
        embedding_models, top_k_rags, threshold_rags, grf_alphas
    ))

    # Pathway 2: With reranker
    rerank_combos = list(itertools.product(
        embedding_models, top_k_rags, threshold_rags, grf_alphas,
        threshold_reranker, top_k_reranks, reranker_models
    ))

    unique_combos = []

    for c in no_rerank_combos:
        unique_combos.append({
            "embedding_model": c[0],
            "top_k_rag": c[1],
            "threshold_rag": c[2],
            "use_reranker": False,
            "threshold_reranker": 0.0,
            "top_k_reranker": 0,
            "GRF": c[3],
            "reranker_model": "None"
        })

    for c in rerank_combos:
        unique_combos.append({
            "embedding_model": c[0],
            "top_k_rag": c[1],
            "threshold_rag": c[2],
            "use_reranker": True,
            "threshold_reranker": c[4],
            "top_k_reranker": c[5],
            "GRF": c[3],
            "reranker_model": c[6]
        })

    print(f"Generated {len(unique_combos)} unique grid combinations.")

    with open(output_path, 'w') as f:
        json.dump(unique_combos, f, indent=4)

    print(f"Saved combinations to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate grid RAG configurations for tuning.")
    parser.add_argument("--output", default="../config/eval_final_tune.json", help="Path to save the generated JSON.")

    parser.add_argument("--models", nargs="+",
                        default=["microsoft/harrier-oss-v1-27b", "tencent/KaLM-Embedding-Gemma3-12B-2511", "nvidia/llama-embed-nemotron-8b", "Qwen/Qwen3-Embedding-8B"],
                        help="List of embedding models.")
    parser.add_argument("--top_k", type=int, nargs="+", default=[20, 50, 100, 150], help="List of top_k_rag values.")
    parser.add_argument("--threshold_rag", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], help="List of threshold_rag values.")
    parser.add_argument("--threshold_rerank", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4], help="List of threshold_reranker values.")
    parser.add_argument("--top_k_rerank", type=int, nargs="+", default=[5, 10, 15, 20], help="List of top_k_reranker values.")
    parser.add_argument("--grf", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0], help="List of GRF alpha values.")
    parser.add_argument("--reranker_models", nargs="+",
                        default=["Qwen/Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-4B", "Qwen/Qwen3-Reranker-8B"],
                        help="List of reranker models.")

    args = parser.parse_args()

    generate_grid_combinations(
        output_path=args.output,
        embedding_models=args.models,
        top_k_rags=args.top_k,
        threshold_rags=args.threshold_rag,
        threshold_reranker=args.threshold_rerank,
        top_k_reranks=args.top_k_rerank,
        grf_alphas=args.grf,
        reranker_models=args.reranker_models
    )
