import json
import os
import glob

results_dir = "../outputs/results_final_tune"
all_results = []

for fpath in glob.glob(os.path.join(results_dir, "tuning_results_*.json")):
    try:
        with open(fpath) as f:
            data = json.load(f)
        if isinstance(data, list):
            all_results.extend(data)
        elif isinstance(data, dict):
            all_results.append(data)
    except Exception:
        continue

# Filter by Context Burden < 60
filtered = [r for r in all_results if r.get("Context Burden", float("inf")) < 60]

# Sort by Mean Recall descending
filtered.sort(key=lambda x: x.get("Mean Recall", 0), reverse=True)

print(f"Total results loaded: {len(all_results)}")
print(f"Results with Context Burden < 60: {len(filtered)}")
print("\n" + "="*80)
print("TOP 5 COMBINATIONS BY MEAN RECALL (Context Burden < 60)")
print("="*80)

for i, r in enumerate(filtered[:5], 1):
    p = r.get("params", {})
    model = p.get("embedding_model", "?").split("/")[-1]
    print(f"\n#{i}")
    print(f"  Mean Recall    : {r['Mean Recall']:.4f}")
    print(f"  MAP            : {r.get('MAP', 0):.4f}")
    print(f"  Context Burden : {r['Context Burden']:.2f}")
    print(f"  Params:")
    print(f"    embedding_model    : {p.get('embedding_model', '?')}")
    print(f"    top_k_rag          : {p.get('top_k_rag', '?')}")
    print(f"    threshold_rag      : {p.get('threshold_rag', '?')}")
    print(f"    use_reranker       : {p.get('use_reranker', '?')}")
    print(f"    threshold_reranker : {p.get('threshold_reranker', '?')}")
    print(f"    GRF                : {p.get('GRF', '?')}")
