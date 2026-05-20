import sys
import os
import json
import sqlite3
import torch
import torch.nn.functional as F

# Paths
DEBUG_DIR = "/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/01.text_to_onto/modules/04.debug"
SCRIPTS_DIR = "/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/01.text_to_onto/modules/00.tox_effect_to_onto/scripts"
sys.path.append(SCRIPTS_DIR)

from model.embeddings import EmbeddingModel
from src.utils import load_ontology_embeddings
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set environments
os.environ["HF_HOME"] = "/nfs/research/ewald/ricos/huggingface_cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def main():
    config_path = os.path.join(DEBUG_DIR, "rag", "best_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    embedding_model_name = config["embedding_model"]
    top_k_rag = config["top_k_rag"]
    threshold_rag = config["threshold_rag"]
    reranker_model_name = config["reranker_model"]
    threshold_reranker = config["threshold_reranker"]
    reranker_batch_size = config.get("reranker_batch_size", 32)
    reranker_prompt_path = config["reranker_prompt"]

    with open(reranker_prompt_path, "r", encoding="utf-8") as f:
        reranker_instruction = f.read().strip()

    db_path = "/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/00.data_preparation/modules/02.prepare_ontologies/outputs/kalm/mp.db"
    print(f"Loading ontology embeddings from {db_path}...")
    ontology_embeddings = load_ontology_embeddings(db_path)
    print(f"Loaded {len(ontology_embeddings)} nodes.")

    # Build hierarchy maps
    nodes = {}
    parent_map = {}
    for item in ontology_embeddings:
        name = item['name']
        nodes[name] = {
            'sanitized_name': item['sanitized_name'],
            'parent': item.get('parent')
        }
        parent_name = item.get('parent')
        if parent_name:
            parent_map[name] = parent_name

    def get_breadcrumbs(name: str) -> str:
        path = []
        curr = name
        visited = set()
        while curr and curr not in visited:
            visited.add(curr)
            label = nodes.get(curr, {}).get('sanitized_name', curr)
            path.append(label)
            curr = parent_map.get(curr)
        return " > ".join(reversed(path))

    ontology_matrix = torch.stack([item['vector'] for item in ontology_embeddings])

    # Load embedding model
    print(f"Loading embedding model: {embedding_model_name}")
    embedder = EmbeddingModel(embedding_model_name)
    embedder.load()

    # Load reranker model
    print(f"Loading reranker model: {reranker_model_name}")
    reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name, padding_side='left')
    reranker_model = AutoModelForCausalLM.from_pretrained(
        reranker_model_name, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    reranker_device = reranker_model.device
    token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
    token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    reranker_prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
    reranker_suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)
    reranker_max_content_len = 1024 - len(reranker_prefix_tokens) - len(reranker_suffix_tokens)

    queries = ["total hormones", "lipids", "glucose", "phosphorus", "LH", "sleep"]
    
    results = {}

    for q in queries:
        print(f"\n==================================================")
        print(f"Query: {q}")
        print(f"==================================================")
        
        # Get query embedding
        vec_bytes = embedder.get_embedding(q)
        query_vec = torch.frombuffer(bytearray(vec_bytes), dtype=torch.float32)

        # RAG similarity
        scores = torch.matmul(ontology_matrix, query_vec)
        _, top_indices = torch.topk(scores, k=min(top_k_rag, len(scores)))
        top_indices = top_indices.tolist()

        rag_candidates = []
        for idx in top_indices:
            item = ontology_embeddings[idx]
            rag_candidates.append({
                'name': item['name'],
                'sanitized_name': item['sanitized_name'],
                'ann_score': scores[idx].item()
            })

        rag_candidates.sort(key=lambda x: x['ann_score'], reverse=True)
        rag_candidates_filtered = [c for c in rag_candidates if c['ann_score'] >= threshold_rag]

        print(f"\nRAG Candidates (top_k={top_k_rag}, threshold={threshold_rag}): Found {len(rag_candidates_filtered)}")
        for idx_c, c in enumerate(rag_candidates_filtered):
            print(f"  {idx_c+1}. {c['name']} : Score: {c['ann_score']:.4f}")

        # Reranking
        print(f"\nRunning Reranker on {len(rag_candidates_filtered)} candidates...")
        pairs = []
        for c in rag_candidates_filtered:
            doc = c['sanitized_name']
            hierarchy = get_breadcrumbs(c['name'])
            doc = f"{doc}\n<Hierarchy>: {hierarchy}"
            pair_text = f"<Instruct>: {reranker_instruction}\n<Query>: {q}\n<Document>: {doc}"
            pairs.append(pair_text)

        reranker_scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), reranker_batch_size):
                batch_pairs = pairs[i:i+reranker_batch_size]
                inputs = reranker_tokenizer(
                    batch_pairs, padding=False, truncation='longest_first',
                    return_attention_mask=False, max_length=reranker_max_content_len
                )
                for j, ele in enumerate(inputs['input_ids']):
                    inputs['input_ids'][j] = reranker_prefix_tokens + ele + reranker_suffix_tokens
                inputs = reranker_tokenizer.pad(
                    inputs, padding=True, return_tensors="pt"
                )
                inputs = {k: v.to(reranker_device) for k, v in inputs.items()}
                batch_logits = reranker_model(**inputs).logits[:, -1, :]
                true_vector = batch_logits[:, token_true_id]
                false_vector = batch_logits[:, token_false_id]
                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                batch_probs = batch_scores[:, 1].exp().tolist()
                reranker_scores.extend(batch_probs)

        passed_candidates = []
        for score, cand in zip(reranker_scores, rag_candidates_filtered):
            cand['rerank_score'] = score
            if score >= threshold_reranker:
                passed_candidates.append(cand)

        passed_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

        print(f"\nReranked Candidates (threshold={threshold_reranker}): Found {len(passed_candidates)}")
        for idx_c, c in enumerate(passed_candidates):
            print(f"  {idx_c+1}. {c['name']} : Rerank Score: {c['rerank_score']:.4f} | RAG Score: {c['ann_score']:.4f}")

        results[q] = {
            "rag": rag_candidates_filtered,
            "reranked": passed_candidates
        }

    # Save outputs to file
    out_file = os.path.join(DEBUG_DIR, "debug_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, default=str, indent=2)
    print(f"\nSaved debug results to {out_file}")

    # Generate a nice markdown table report
    report_file = os.path.join(DEBUG_DIR, "debug_results_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# RAG & Reranker Debug Results (MP Ontology)\n\n")
        for q in queries:
            f.write(f"## Query: `{q}`\n\n")
            f.write(f"### All Reranked Candidates (threshold={threshold_reranker})\n\n")
            f.write("| Rank | MP ID | Term Name | Rerank Score | RAG Score | Hierarchy |\n")
            f.write("|---|---|---|---|---|---|\n")
            for idx, c in enumerate(results[q]["reranked"]):
                hier = get_breadcrumbs(c['name'])
                f.write(f"| {idx+1} | `{c['name']}` | {c['sanitized_name']} | **{c['rerank_score']:.4f}** | {c['ann_score']:.4f} | {hier} |\n")
            if not results[q]["reranked"]:
                f.write("| | | No candidates passed threshold | | | |\n")
            
            f.write(f"\n### All RAG Candidates (top_k={top_k_rag}, threshold={threshold_rag})\n\n")
            f.write("| Rank | MP ID | Term Name | RAG Score | Hierarchy |\n")
            f.write("|---|---|---|---|---|\n")
            for idx, c in enumerate(results[q]["rag"]):
                hier = get_breadcrumbs(c['name'])
                f.write(f"| {idx+1} | `{c['name']}` | {c['sanitized_name']} | {c['ann_score']:.4f} | {hier} |\n")
            f.write("\n---\n\n")
    print(f"Generated Markdown report at {report_file}")

if __name__ == "__main__":
    main()
