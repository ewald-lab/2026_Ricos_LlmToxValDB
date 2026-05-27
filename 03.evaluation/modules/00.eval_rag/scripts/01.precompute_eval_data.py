import sys
import os
import argparse
import json
import torch
import polars as pl
from tqdm import tqdm
import sqlite3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import load_ontology_embeddings, setup_logging
from model.embeddings import EmbeddingModel
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = setup_logging()

def load_ground_truth(csv_path: str, toxval_path: str):
    # Load mapping from ToxValDB
    tox_df = pl.read_csv(toxval_path, separator='\t', null_values=["", "NA", "null"])
    # Create mapping: original -> {cleaned, standardized}
    tox_map = {}
    for row in tox_df.iter_rows(named=True):
        orig = row.get("TOXICOLOGICAL_EFFECT")
        if orig:
            orig = orig.strip()
            tox_map[orig] = {
                "cleaned": row.get("TOXICOLOGICAL_EFFECT_LLM_CLEANED", "").strip() if row.get("TOXICOLOGICAL_EFFECT_LLM_CLEANED") else "",
                "standardized": row.get("TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED", "").strip() if row.get("TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED") else ""
            }

    df = pl.read_csv(csv_path)
    gt_data = []
    for row in df.iter_rows(named=True):
        orig = row.get("TOXICOLOGICAL_EFFECT", "").strip()
        matches_str = row.get("ONTOLOGY_MATCHES", "")
        if not orig: continue
        matches = set([m.strip() for m in str(matches_str).split("|") if m.strip()])
        if len(matches) == 0: continue

        mapping = tox_map.get(orig, {"cleaned": "", "standardized": ""})
        # If mapping not found or empty, fallback to orig for cleaned
        cleaned = mapping["cleaned"] if mapping["cleaned"] else orig
        std = mapping["standardized"] if mapping["standardized"] else ""

        gt_data.append({
            "original": orig,
            "cleaned": cleaned,
            "standardized": std,
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

        # Determine model mapping (hacky heuristic based on db path if needed, or pass explicitly)
        # We need embedding model to generate on the fly if missing
        embed_model = None

        def get_vec(text):
            if not text: return None
            if text in text_to_vec:
                return text_to_vec[text]
            nonlocal embed_model
            if embed_model is None:
                # Get model id from target db name or just load what we can
                model_name = "Qwen/Qwen3-Embedding-8B"
                if "harrier" in self.target_db: model_name = "microsoft/harrier-oss-v1-27b"
                elif "nemotron" in self.target_db: model_name = "nvidia/llama-embed-nemotron-8b"
                elif "kalm" in self.target_db: model_name = "tencent/KaLM-Embedding-Gemma3-12B-2511"
                logger.info(f"Loading fallback EmbeddingModel: {model_name}")
                embed_model = EmbeddingModel(model_name)
                embed_model.load()
            logger.info(f"Embedding on the fly: {text}")
            vec_bytes = embed_model.get_embedding(text)
            vec = torch.frombuffer(bytearray(vec_bytes), dtype=torch.float32)
            text_to_vec[text] = vec
            return vec

        q_cleaned = []
        q_std = []

        for row in self.ground_truth:
            cleaned_vec = get_vec(row["cleaned"])
            std_vec = get_vec(row["standardized"])
            q_cleaned.append(cleaned_vec)
            q_std.append(std_vec)

        self.q_cleaned_mat = torch.stack(q_cleaned).to("cuda" if torch.cuda.is_available() else "cpu")
        # For std, some might be None if no standardized text. Fallback to cleaned if None.
        q_std_fixed = [v if v is not None else q_cleaned[i] for i, v in enumerate(q_std)]
        self.q_std_mat = torch.stack(q_std_fixed).to(self.q_cleaned_mat.device)

    def _load_ontology(self):
        logger.info(f"Loading ontology matrix from {self.ontology_db}")
        self.ontology_embeddings = load_ontology_embeddings(self.ontology_db)
        self._ontology_matrix = torch.stack([item['vector'] for item in self.ontology_embeddings]).to(self.q_cleaned_mat.device)

    def get_top_k_candidates(self, top_k=150):
        """Computes top K candidates for each query using cosine similarity, for each GRF alpha."""
        logger.info(f"Computing Top {top_k} candidates for {len(self.ground_truth)} queries for each GRF...")

        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        results_by_grf = {str(a): [] for a in alphas}

        for alpha in alphas:
            # query_vec = alpha * cleaned + (1 - alpha) * std
            q_mat = alpha * self.q_cleaned_mat + (1.0 - alpha) * self.q_std_mat
            scores_mat = torch.matmul(q_mat, self._ontology_matrix.T)

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
                results_by_grf[str(alpha)].append(candidates)

        return results_by_grf

    def compute_reranker_scores(self, results_by_grf, reranker_model_name, reranker_config_path):
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

        # To avoid re-computing reranker for same candidate-query pair across GRF alphas,
        # we compute it once per unique pair.
        pair_to_idx = {}
        flat_pairs = []

        for alpha_str, all_candidates in results_by_grf.items():
            for qi, candidates in enumerate(all_candidates):
                query_text = self.ground_truth[qi]["cleaned"]
                for c in candidates:
                    doc_text = c['sanitized_name']
                    pair_key = (query_text, doc_text)
                    if pair_key not in pair_to_idx:
                        pair_text = f"<Instruct>: {instruction}\n<Query>: {query_text}\n<Document>: {doc_text}"
                        pair_to_idx[pair_key] = len(flat_pairs)
                        flat_pairs.append(pair_text)

        logger.info(f"Computing {len(flat_pairs)} UNIQUE reranker scores...")
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
        for alpha_str, all_candidates in results_by_grf.items():
            for qi in range(len(all_candidates)):
                query_text = self.ground_truth[qi]["cleaned"]
                for ci in range(len(all_candidates[qi])):
                    doc_text = all_candidates[qi][ci]["sanitized_name"]
                    idx = pair_to_idx[(query_text, doc_text)]
                    all_candidates[qi][ci]["rerank_score"] = rerank_scores_flat[idx]

        return results_by_grf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--ontology_db", required=True)
    parser.add_argument("--target_embedding_db", required=True)
    parser.add_argument("--reranker_model", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--reranker_config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--toxval_db", required=True)
    args = parser.parse_args()

    gt_data = load_ground_truth(args.ground_truth, args.toxval_db)
    precomputer = Precomputer(gt_data, args.ontology_db, args.target_embedding_db)

    # 1. Get Top 150 candidates for each GRF value
    results_by_grf = precomputer.get_top_k_candidates(top_k=150)

    # 2. Get Reranker scores for all unique candidates across all GRF lists
    full_data_by_grf = precomputer.compute_reranker_scores(results_by_grf, args.reranker_model, args.reranker_config)

    # 3. Save to Master Cache
    output_data = {
        "ground_truth": gt_data,
        "precomputed_results_by_grf": full_data_by_grf
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    logger.info(f"Precomputation finished. Saved to {args.output}")
