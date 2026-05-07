import json
import random
import argparse

def generate_random_combinations(n, output_path, 
                                 embedding_models, top_k_rags, 
                                 threshold_rags, threshold_reranker, 
                                 grf_alphas):
    
    skip_def = [True, False]
    use_rerank = [True, False]
    
    combos = []
    
    # Try to generate unique combinations
    for _ in range(n * 10): 
        use_reranker_choice = random.choice(use_rerank)
        grf = random.choice(grf_alphas)
        
        c = {
            "embedding_model": random.choice(embedding_models),
            "top_k_rag": random.choice(top_k_rags),
            "threshold_rag": random.choice(threshold_rags),
            "use_reranker": use_reranker_choice,
            "threshold_reranker": random.choice(threshold_reranker) if use_reranker_choice else 0.0,
            "skip_definition_in_embedding": random.choice(skip_def),
            "GRF": grf
        }
        combos.append(c)
        
    unique_combos = []
    seen = set()
    for d in combos:
        items = tuple(sorted(d.items()))
        if items not in seen:
            unique_combos.append(d)
            seen.add(items)
            if len(unique_combos) >= n:
                break
                
    print(f"Generated {len(unique_combos)} unique random combinations.")
    
    with open(output_path, 'w') as f:
        json.dump(unique_combos, f, indent=4)
        
    print(f"Saved combinations to {output_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random RAG configurations for tuning.")
    parser.add_argument("--n", type=int, default=125, help="Number of unique combinations to generate.")
    parser.add_argument("--output", default="../config/eval_final_tune.json", help="Path to save the generated JSON.")
    
    parser.add_argument("--models", nargs="+", 
                        default=["microsoft/harrier-oss-v1-27b", "tencent/KaLM-Embedding-Gemma3-12B-2511", "nvidia/llama-embed-nemotron-8b", "Qwen/Qwen3-Embedding-8B"],
                        help="List of embedding models.")
    parser.add_argument("--top_k", type=int, nargs="+", default=[10, 20, 50, 100], help="List of top_k_rag values.")
    parser.add_argument("--threshold_rag", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5], help="List of threshold_rag values.")
    parser.add_argument("--threshold_rerank", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.8], help="List of threshold_reranker values.")
    parser.add_argument("--grf", type=float, nargs="+", default=[0.4, 0.6, 0.8], help="List of GRF alpha values.")
    
    args = parser.parse_args()
    
    generate_random_combinations(
        n=args.n, 
        output_path=args.output,
        embedding_models=args.models,
        top_k_rags=args.top_k,
        threshold_rags=args.threshold_rag,
        threshold_reranker=args.threshold_rerank,
        grf_alphas=args.grf
    )
