import sys
import os
import argparse
import json
import torch
import polars as pl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import load_ontology_embeddings, setup_logging
from model.embeddings import EmbeddingModel
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = setup_logging()

def load_ground_truth(csv_path: str):
    """
    Expects a CSV with:
    - TOXICOLOGICAL_EFFECT (Original Text)
    - TOXICOLOGICAL_EFFECT_STANDARDIZED (LLM Cleaned Text)
    - ONTOLOGY_MATCHES (pipe-separated ground truth node names)
    """
    df = pl.read_csv(csv_path)
    gt_data = []
    
    for row in df.iter_rows(named=True):
        orig = row.get("TOXICOLOGICAL_EFFECT", "").strip()
        std = row.get("TOXICOLOGICAL_EFFECT_STANDARDIZED", "").strip()
        matches_str = row.get("ONTOLOGY_MATCHES", "")
        # Handle NA and empty
        if not orig:
            continue
            
        matches = set([m.strip() for m in str(matches_str).split("|") if m.strip()])
        if len(matches) == 0:
            continue
            
        gt_data.append({
            "original": orig,
            "standardized": std if std else orig,
            "matches": matches
        })
        
    return gt_data

def calculate_metrics(retrieved_candidates, ground_truth_matches, parent_map=None):
    """
    Calculates Mean Recall, MAP, and Context Burden for a single query.
    retrieved_candidates: list of dicts with 'name' key, sorted by rank.
    """
    retrieved_names = []
    seen = set()
    for c in retrieved_candidates:
        name = c['name']
        if name not in seen:
            retrieved_names.append(name)
            seen.add(name)
        
        # Expand with ancestors if parent_map is provided
        if parent_map:
            curr = name
            while curr in parent_map and parent_map[curr]:
                curr = parent_map[curr]
                if curr not in seen:
                    retrieved_names.append(curr)
                    seen.add(curr)

    retrieved_set = set(retrieved_names)
    
    # 1. Recall
    correct_retrieved = len(retrieved_set.intersection(ground_truth_matches))
    recall = correct_retrieved / len(ground_truth_matches) if ground_truth_matches else 0.0
    
    # 2. Average Precision
    ap = 0.0
    hits = 0
    # In this expanded version, if we retrieve a child, we treat all its ancestors 
    # as being retrieved at the same rank for AP purposes? 
    # Or just count the first hit among the expanded set.
    # The most common approach for hierarchical recall is to just expand the sets.
    # For AP, let's stick to the expanded retrieved_names order.
    for i, name in enumerate(retrieved_names):
        if name in ground_truth_matches:
            hits += 1
            precision_at_i = hits / (i + 1)
            ap += precision_at_i
            
    ap = ap / len(ground_truth_matches) if ground_truth_matches else 0.0
    
    # 3. Context Burden (original number of candidates retrieved, not expanded)
    context_burden = len(retrieved_candidates)
    
    return recall, ap, context_burden

class Evaluator:
    def __init__(self, ground_truth, config, ontology_db=None):
        self.ground_truth = ground_truth
        self.config = config
        self.config_db_path = ontology_db
        self.current_embedding_model = None
        self.embedder = None
        
        self.current_ontology_db = None
        self.ontology_embeddings = []
        self._ontology_matrix = None
        self.parent_map = {}
        
        self.reranker_model = None
        self.reranker_tokenizer = None
        
        # Extract Reranker setup
        first_cfg = config[0] if isinstance(config, list) and len(config) > 0 else {}
        self.reranker_model_name = first_cfg.get("reranker_model")
        self.reranker_prompt_path = first_cfg.get("reranker_config")
        
    def _load_embedder(self, model_name):
        if self.current_embedding_model != model_name:
            if self.embedder is not None:
                del self.embedder
                torch.cuda.empty_cache()
            logger.info(f"Loading embedding model: {model_name}")
            self.embedder = EmbeddingModel(model_name)
            self.current_embedding_model = model_name

    def _load_ontology(self, embedding_model):
        if self.config_db_path:
            db_path = self.config_db_path
        else:
            model_to_dir = {
                "microsoft/harrier-oss-v1-27b": "harrier",
                "tencent/KaLM-Embedding-Gemma3-12B-2511": "kalm",
                "nvidia/llama-embed-nemotron-8b": "nemotron",
                "Qwen/Qwen3-Embedding-8B": "qwen"
            }
            dir_name = model_to_dir.get(embedding_model, embedding_model.replace("/", "_"))
            db_path = f"data/ontologies/databases/{dir_name}/mp.db"
        
        if self.current_ontology_db != db_path:
            if not os.path.exists(db_path):
                logger.warning(f"Ontology DB {db_path} not found!")
                return False
                
            logger.info(f"Loading ontology matrix from {db_path}")
            self.ontology_embeddings = load_ontology_embeddings(db_path)
            self._ontology_matrix = torch.stack([item['vector'] for item in self.ontology_embeddings])
            self.parent_map = {item['name']: item['parent'] for item in self.ontology_embeddings if item['parent']}
            self.current_ontology_db = db_path
            
        return True

    def _load_reranker(self):
        if self.reranker_model is None:
            logger.info(f"Loading Reranker Model: {self.reranker_model_name}")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name, padding_side='left')
            self.reranker_model = AutoModelForCausalLM.from_pretrained(
                self.reranker_model_name, torch_dtype=torch.bfloat16, device_map="auto"
            ).eval()
            self.token_false_id = self.reranker_tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.reranker_tokenizer.convert_tokens_to_ids("yes")
            prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.reranker_prefix_tokens = self.reranker_tokenizer.encode(prefix, add_special_tokens=False)
            self.reranker_suffix_tokens = self.reranker_tokenizer.encode(suffix, add_special_tokens=False)
            self._reranker_max_content_len = 1024 - len(self.reranker_prefix_tokens) - len(self.reranker_suffix_tokens)
            
            if not self.reranker_prompt_path or not os.path.exists(self.reranker_prompt_path):
                logger.error(f"Reranker prompt path {self.reranker_prompt_path} not found!")
                raise FileNotFoundError(f"Reranker prompt path {self.reranker_prompt_path} not found!")

            with open(self.reranker_prompt_path, 'r', encoding='utf-8') as f:
                self.reranker_instruction = f.read().strip()
                
    def _get_embeddings_batch(self, texts):
        vecs = []
        for text in texts:
            vec_bytes = self.embedder.get_embedding(text)
            vecs.append(torch.frombuffer(bytearray(vec_bytes), dtype=torch.float32))
        return torch.stack(vecs)

    def evaluate_config(self, params):
        # Unpack parameters
        emb_model = params.get("embedding_model", "Qwen/Qwen3-Embedding-8B")
        top_k = params.get("top_k_rag", 15)
        thresh_rag = params.get("threshold_rag", 0.5)
        use_reranker = params.get("use_reranker", False)
        thresh_rerank = params.get("threshold_reranker", 0.0)
        grf_alpha = params.get("GRF", 0.5)
        
        # Reranker info
        reranker_model = params.get("reranker_model", "Qwen/Qwen3-Reranker-8B")
        reranker_config = params.get("reranker_config", "config/reranker_prompt.txt")

        logger.info(f"\n--- Evaluating Config: {params} ---")

        if not self._load_ontology(emb_model):
            return None

        self._load_embedder(emb_model)

        if use_reranker:
            self._load_reranker()

        gt = self.ground_truth
        orig_texts = [row["original"] for row in gt]
        std_texts  = [row["standardized"] for row in gt]
        all_matches = [row["matches"] for row in gt]

        # --- 1. Query embeddings ---
        q0_mat = self._get_embeddings_batch(orig_texts)
        qf_mat = self._get_embeddings_batch(std_texts)
        query_mat = grf_alpha * q0_mat + (1.0 - grf_alpha) * qf_mat

        # --- 2. Retrieval ---
        ont = self._ontology_matrix
        scores_mat = torch.matmul(query_mat, ont.T)

        all_rag_candidates = []
        for qi in range(len(gt)):
            scores = scores_mat[qi]
            _, top_indices = torch.topk(scores, k=min(top_k, len(scores)))
            rag_candidates = []
            for idx in top_indices.tolist():
                score = scores[idx].item()
                if score >= thresh_rag:
                    item = self.ontology_embeddings[idx]
                    rag_candidates.append({
                        'name': item['name'],
                        'sanitized_name': item['sanitized_name'],
                        'ann_score': score
                    })
            rag_candidates.sort(key=lambda x: x['ann_score'], reverse=True)
            all_rag_candidates.append(rag_candidates)

        # --- 3. Reranker ---
        if use_reranker:
            query_texts_for_reranker = orig_texts

            flat_pairs = []
            flat_query_idxs = []
            for qi, candidates in enumerate(all_rag_candidates):
                query_text = query_texts_for_reranker[qi]
                for c in candidates:
                    pair_text = f"<Instruct>: {self.reranker_instruction}\n<Query>: {query_text}\n<Document>: {c['sanitized_name']}"
                    flat_pairs.append(pair_text)
                    flat_query_idxs.append(qi)

            rerank_scores_flat = []
            batch_size = 64
            with torch.no_grad():
                for i in range(0, len(flat_pairs), batch_size):
                    batch_pairs = flat_pairs[i:i + batch_size]
                    inputs = self.reranker_tokenizer(
                        batch_pairs, padding=False, truncation='longest_first',
                        return_attention_mask=False, max_length=self._reranker_max_content_len
                    )
                    for j, ele in enumerate(inputs['input_ids']):
                        inputs['input_ids'][j] = self.reranker_prefix_tokens + ele + self.reranker_suffix_tokens
                    inputs = self.reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt")
                    inputs = {k: v.to(self.reranker_model.device) for k, v in inputs.items()}

                    batch_logits = self.reranker_model(**inputs).logits[:, -1, :]
                    batch_scores = torch.stack([batch_logits[:, self.token_false_id], batch_logits[:, self.token_true_id]], dim=1)
                    batch_probs = torch.nn.functional.log_softmax(batch_scores, dim=1)[:, 1].exp().tolist()
                    rerank_scores_flat.extend(batch_probs)

            pair_cursor = 0
            all_candidates_final = []
            for qi, candidates in enumerate(all_rag_candidates):
                n = len(candidates)
                scores_for_q = rerank_scores_flat[pair_cursor:pair_cursor + n]
                pair_cursor += n
                passed = []
                for score, cand in zip(scores_for_q, candidates):
                    if score >= thresh_rerank:
                        cand['rerank_score'] = score
                        passed.append(cand)
                passed.sort(key=lambda x: x['rerank_score'], reverse=True)
                all_candidates_final.append(passed)
        else:
            all_candidates_final = all_rag_candidates

        # --- 4. Metrics ---
        total_recall = 0.0
        total_map = 0.0
        total_context = 0.0
        valid_queries = 0
        per_query_details = []

        for qi, (candidates, matches) in enumerate(zip(all_candidates_final, all_matches)):
            recall, ap, context = calculate_metrics(candidates, matches, parent_map=self.parent_map)
            total_recall += recall
            total_map += ap
            total_context += context
            valid_queries += 1

            retrieved_names = set()
            for c in candidates:
                retrieved_names.add(c['name'])
                curr = c['name']
                while curr in self.parent_map and self.parent_map[curr]:
                    curr = self.parent_map[curr]
                    retrieved_names.add(curr)

            hits = sorted(retrieved_names & matches)
            misses = sorted(matches - retrieved_names)

            per_query_details.append({
                "effect": gt[qi]["original"],
                "standardized": gt[qi]["standardized"],
                "recall": round(recall, 4),
                "ap": round(ap, 4),
                "hits": hits,
                "misses": misses,
                "n_candidates": len(candidates),
            })

        if valid_queries == 0:
            return None

        return {
            "Mean Recall": total_recall / valid_queries,
            "MAP": total_map / valid_queries,
            "Context Burden": total_context / valid_queries,
            "Valid Queries": valid_queries,
            "per_query": per_query_details,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth", required=True, help="Path to ground truth CSV file")
    parser.add_argument("--config", required=True, help="JSON file with parameter grids/combinations")
    parser.add_argument("--output", default="eval_results.json", help="Output JSON results")
    parser.add_argument("--config_index", type=int, default=None, help="Index of the configuration to evaluate (0-based)")
    parser.add_argument("--ontology_db", help="Path to the ontology SQLite database (overrides default)")
    parser.add_argument("--reranker_model", default="Qwen/Qwen3-Reranker-8B", help="Reranker model name")
    parser.add_argument("--reranker_config", default="config/reranker_prompt.txt", help="Path to reranker prompt config")
    args = parser.parse_args()
    
    gt_data = load_ground_truth(args.ground_truth)
    logger.info(f"Loaded {len(gt_data)} ground truth examples.")
    
    with open(args.config, 'r') as f:
        configs = json.load(f)
        
    if args.config_index is not None:
        if args.config_index < 0 or args.config_index >= len(configs):
            logger.error(f"Config index {args.config_index} out of range (0-{len(configs)-1})")
            sys.exit(1)
        configs_to_run = [configs[args.config_index]]
        logger.info(f"Running single config at index {args.config_index}")
    else:
        # Sort configs by model so each model is loaded only once on the GPU
        configs_to_run = sorted(configs, key=lambda c: c.get("embedding_model", ""))
        logger.info(f"Running all {len(configs_to_run)} configs")

    # Update config with command line args if they are not the defaults or if needed
    for cfg in configs_to_run:
        cfg["reranker_model"] = args.reranker_model
        cfg["reranker_config"] = args.reranker_config

    evaluator = Evaluator(gt_data, configs_to_run, ontology_db=args.ontology_db)
    
    results = []

    for i, params in enumerate(configs_to_run):
        res = evaluator.evaluate_config(params)
        if res:
            res["params"] = params
            results.append(res)
            logger.info(f"Results for config {i}: Recall={res['Mean Recall']:.4f}, MAP={res['MAP']:.4f}, Context={res['Context Burden']:.2f}")

    # Save results sorted by MAP descending
    results.sort(key=lambda x: x["MAP"], reverse=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    logger.info(f"Saved {len(results)} results to {args.output}")
