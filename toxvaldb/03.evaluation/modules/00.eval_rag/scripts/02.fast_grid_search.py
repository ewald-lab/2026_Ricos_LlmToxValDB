from __future__ import annotations
import json
import os
import argparse
from tqdm import tqdm

def calculate_metrics(
    retrieved_candidates: list[dict],
    ground_truth_matches: list[str] | set[str],
    parent_map: dict[str, list[str]] | None = None
) -> tuple[float, float, int]:
    """
    Description:
    Calculate precision/recall metrics for retrieved candidates compared to ground truth.

    Args:
        retrieved_candidates (list[dict]): The list of candidates retrieved.
        ground_truth_matches (list[str] | set[str]): The ground truth matches.
        parent_map (dict[str, list[str]] | None): A map of node names to parent names. Defaults to None.

    Returns:
        tuple[float, float, int]: A tuple of (recall, ap, context_burden).
    """
    retrieved_names = []
    seen = set()
    for c in retrieved_candidates:
        name = c['name']
        if name not in seen:
            retrieved_names.append(name)
            seen.add(name)
        if parent_map:
            queue = [name]
            while queue:
                curr = queue.pop(0)
                parents = parent_map.get(curr, [])
                for p in parents:
                    if p and p not in seen:
                        retrieved_names.append(p)
                        seen.add(p)
                        queue.append(p)

    retrieved_set = set(retrieved_names)
    ground_truth_matches = set(ground_truth_matches)
    correct_retrieved = len(retrieved_set.intersection(ground_truth_matches))
    recall = correct_retrieved / len(ground_truth_matches) if ground_truth_matches else 0.0

    ap = 0.0
    hits = 0
    for i, name in enumerate(retrieved_names):
        if name in ground_truth_matches:
            hits += 1
            ap += (hits / (i + 1))
    ap = ap / len(ground_truth_matches) if ground_truth_matches else 0.0
    context_burden = len(retrieved_candidates)

    return recall, ap, context_burden

def run_evaluation(precomputed_data: dict, config: dict, parent_map: dict[str, list[str]]) -> dict:
    """
    Description:
    Run evaluation on all queries for a given configuration using precomputed candidates.

    Args:
        precomputed_data (dict): The precomputed candidate data dictionary.
        config (dict): The configuration dictionary containing evaluation parameters.
        parent_map (dict[str, list[str]]): A map of node names to parent names.

    Returns:
        dict: A dictionary of evaluation metrics and the configuration parameters.
    """
    gt_data = precomputed_data["ground_truth"]
    all_precomputed_by_grf = precomputed_data["precomputed_results_by_grf"]

    top_k_rag = config["top_k_rag"]
    thresh_rag = config["threshold_rag"]
    use_reranker = config["use_reranker"]
    top_k_rerank = config["top_k_reranker"]
    thresh_rerank = config["threshold_reranker"]
    grf_alpha = config.get("GRF", 0.0)
    all_precomputed = all_precomputed_by_grf[str(grf_alpha)]

    total_recall, total_map, total_context = 0.0, 0.0, 0.0

    for qi in range(len(gt_data)):
        candidates = all_precomputed[qi]
        matches = gt_data[qi]["matches"]

        # 1. Filter by RAG params (candidates are already sorted by ann_score from precompute)
        filtered = [c for c in candidates[:top_k_rag] if c['ann_score'] >= thresh_rag]

        # 2. Filter by Reranker if used
        if use_reranker:
            final = [c for c in filtered if c.get('rerank_score', 0.0) >= thresh_rerank]
            if top_k_rerank > 0:
                final = final[:top_k_rerank]
        else:
            final = filtered

        recall, ap, context = calculate_metrics(final, matches, parent_map=parent_map)
        total_recall += recall
        total_map += ap
        total_context += context

    n = len(gt_data)
    return {
        "Mean Recall": total_recall / n,
        "MAP": total_map / n,
        "Context Burden": total_context / n,
        "params": config
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precomputed_data", required=True)
    parser.add_argument("--config_grid", required=True)
    parser.add_argument("--ontology_db", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_variant", default=None, help="Filter configs to this data variant (base, def, hier, def_hier).")
    args = parser.parse_args()

    # Load precomputed data
    with open(args.precomputed_data, 'r') as f:
        precomputed_data = json.load(f)

    # Load grid configurations
    with open(args.config_grid, 'r') as f:
        all_configs = json.load(f)

    # Load parent map from ontology DB for hierarchical recall
    import sqlite3
    parent_map = {}
    import json
    with sqlite3.connect(args.ontology_db) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT name, parent FROM embeddings")
        except sqlite3.OperationalError:
            cursor.execute("SELECT name, parents FROM embeddings")
        for name, parents_json in cursor:
            if parents_json:
                if parents_json.startswith('['):
                    try:
                        parents_list = json.loads(parents_json)
                    except Exception:
                        parents_list = [parents_json]
                else:
                    parents_list = [parents_json]
                if parents_list:
                    parent_map[name] = parents_list

    target_embedding = precomputed_data.get("embedding_model")
    target_reranker = precomputed_data.get("reranker_model")
    target_variant = args.data_variant or precomputed_data.get("data_variant")

    results = []
    for cfg in tqdm(all_configs, desc="Evaluating Configurations"):
        # Filter configs that match the model used for precomputation
        if target_embedding and cfg.get("embedding_model") != target_embedding:
            continue
        # Filter configs by reranker model
        if cfg.get("use_reranker"):
            if target_reranker and cfg.get("reranker_model") != target_reranker:
                continue
        # Filter configs by data variant
        if target_variant and cfg.get("data_variant") != target_variant:
            continue

        res = run_evaluation(precomputed_data, cfg, parent_map)
        results.append(res)

    # Sort and save
    results_sorted = sorted(results, key=lambda x: x["MAP"], reverse=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, indent=4)

    print(f"Finished. Evaluated {len(results)} configurations in seconds.")
