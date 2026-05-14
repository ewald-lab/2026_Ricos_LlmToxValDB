import sys
import os
import argparse
import json
import torch
import polars as pl
from tqdm import tqdm
from collections import defaultdict
import sqlite3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import load_ontology_embeddings, setup_logging
from model.embeddings import EmbeddingModel
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = setup_logging()

def load_ground_truth(csv_path: str):
    df = pl.read_csv(csv_path)
    gt_data = []
    for row in df.iter_rows(named=True):
        orig = row.get("TOXICOLOGICAL_EFFECT", "").strip()
        matches_str = row.get("ONTOLOGY_MATCHES", "")
        if not orig: continue
        matches = set([m.strip() for m in str(matches_str).split("|") if m.strip()])
        if len(matches) == 0: continue
        gt_data.append({
            "original": orig,
            "matches": list(matches) # Convert to list for JSON serialization
        })
    return gt_data

class Precomputer:
    def __init__(self, ground_truth, ontology_db, target_db):
        self.ground_truth = ground_truth
        self.ontology_db = ontology_db
        self.target_db = target_db
        self.ontology_embeddings = []
        self._ontology_matrix = None
        self.q0_mat = None
        
        # Load precomputed query embeddings (q0 only for now, as ground truth only has one column)
        self._load_query_embeddings()
        # Load ontology
        self._load_ontology()

    def _load_query_embeddings(self):
        logger.info(f"Loading query embeddings from {self.target_db}...")
        text_to_vec = {}
        with sqlite3.connect(self.target_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT text, vector FROM input_embeddings")
            for row in cursor:
                text_to_vec[row[0]] = torch.frombuffer(bytearray(row[1]), dtype=torch.float32)
        
        orig_texts = [row["original"] for row in self.ground_truth]
        missing = [t for t in orig_texts if t not in text_to_vec]
        if missing:
            raise ValueError(f"Error: {len(missing)} phrases still missing from {self.target_db}. Run population script first.")
        
        self.q0_mat = torch.stack([text_to_vec[t] for t in orig_texts]).to("cuda" if torch.cuda.is_available() else "cpu")

    def _load_ontology(self):
        logger.info(f"Loading ontology matrix from {self.ontology_db}")
        self.ontology_embeddings = load_ontology_embeddings(self.ontology_db)
        self._ontology_matrix = torch.stack([item['vector'] for item in self.ontology_embeddings]).to(self.q0_mat.device)

    def get_top_k_candidates(self, top_k=150):
        """Computes top K candidates for each query using cosine similarity."""
        logger.info(f"Computing Top {top_k} candidates for {len(self.ground_truth)} queries...")
        # Simple cosine similarity (assuming normalized vectors, or we can normalize them)
        # Assuming vectors are already normalized from DB
        scores_mat = torch.matmul(self.q0_mat, self._ontology_matrix.T)
        
        results = []
        for qi in range(len(self.ground_truth)):
            scores = scores_mat[qi]
            val, idxs = torch.topk(scores, k=min(top_k, len(scores)))
            
            candidates = []
            for v, i in zip(val.tolist(), idxs.tolist()):
                item = self.ontology_embeddings[i]
                candidates.append({
                    "name": item["name"],
                    "sanitized_name": item["sanitized_name"],
                    "ann_score": v
                })
            results.append(candidates)
        return results

    def compute_reranker_scores(self, all_candidates, reranker_model_name, reranker_config_path):
        logger.info(f"Loading Reranker Model: {reranker_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(reranker_model_name, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(
            reranker_model_name, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        max_content_len = 1024 - len(prefix_tokens) - len(suffix_tokens)
        
        with open(reranker_config_path, 'r', encoding='utf-8') as f:
            instruction = f.read().strip()

        flat_pairs = []
        for qi, candidates in enumerate(all_candidates):
            query_text = self.ground_truth[qi]["original"]
            for c in candidates:
                pair_text = f"<Instruct>: {instruction}\n<Query>: {query_text}\n<Document>: {c['sanitized_name']}"
                flat_pairs.append(pair_text)

        logger.info(f"Computing {len(flat_pairs)} reranker scores...")
        rerank_scores_flat = []
        batch_size = 32
        with torch.no_grad():
            for i in tqdm(range(0, len(flat_pairs), batch_size)):
                batch_pairs = flat_pairs[i:i + batch_size]
                inputs = tokenizer(batch_pairs, padding=False, truncation='longest_first', 
                                              return_attention_mask=False, max_length=max_content_len)
                for j, ele in enumerate(inputs['input_ids']):
                    inputs['input_ids'][j] = prefix_tokens + ele + suffix_tokens
                inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                batch_logits = model(**inputs).logits[:, -1, :]
                batch_scores = torch.stack([batch_logits[:, token_false_id], batch_logits[:, token_true_id]], dim=1)
                batch_probs = torch.nn.functional.log_softmax(batch_scores, dim=1)[:, 1].exp().tolist()
                rerank_scores_flat.extend(batch_probs)

        # Map back to structured format
        idx = 0
        for qi in range(len(all_candidates)):
            for ci in range(len(all_candidates[qi])):
                all_candidates[qi][ci]["rerank_score"] = rerank_scores_flat[idx]
                idx += 1
        
        return all_candidates

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--ontology_db", required=True)
    parser.add_argument("--target_embedding_db", required=True)
    parser.add_argument("--reranker_model", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--reranker_config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gt_data = load_ground_truth(args.ground_truth)
    precomputer = Precomputer(gt_data, args.ontology_db, args.target_embedding_db)
    
    # 1. Get Top 150 candidates (max possible in current grid search)
    all_candidates = precomputer.get_top_k_candidates(top_k=150)
    
    # 2. Get Reranker scores for all of them
    full_data = precomputer.compute_reranker_scores(all_candidates, args.reranker_model, args.reranker_config)
    
    # 3. Save to Master Cache
    output_data = {
        "ground_truth": gt_data,
        "precomputed_results": full_data
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    
    logger.info(f"Precomputation finished. Saved to {args.output}")
