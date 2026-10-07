from __future__ import annotations
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

def load_ground_truth(csv_path: str, toxval_path: str) -> list[dict]:
    """
    Description:
    Load ground truth data from the specified CSV file and join it with ToxValDB metadata.

    Args:
        csv_path (str): Path to the ground truth CSV file.
        toxval_path (str): Path to the ToxValDB database/TSV file.

    Returns:
        list[dict]: A list of dictionaries containing original, cleaned, standardized text and matches.
    """
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
            }

    df = pl.read_csv(csv_path)
    gt_data = []
    for row in df.iter_rows(named=True):
        orig = row.get("TOXICOLOGICAL_EFFECT", "").strip()
        matches_str = row.get("ONTOLOGY_MATCHES", "")
        if not orig: continue
        matches = set([m.strip() for m in str(matches_str).split("|") if m.strip()])
        if len(matches) == 0: continue

        mapping = tox_map.get(orig, {"cleaned": ""})
        # If mapping not found or empty, fallback to orig for cleaned
        cleaned = mapping["cleaned"] if mapping["cleaned"] else orig

        gt_data.append({
            "original": orig,
            "cleaned": cleaned,
            "matches": list(matches) # Convert to list for JSON serialization
        })
    return gt_data

class Precomputer:
    def __init__(self, ground_truth: list[dict], ontology_db: str, target_db: str,
                 global_definitions_db: str = None, data_variant: str = "base") -> None:
        """
        Description:
        Initialize the Precomputer class.

        Args:
            ground_truth (list[dict]): The ground truth list of dictionaries.
            ontology_db (str): Path to the ontology database file.
            target_db (str): Path to the target embeddings database file.
            global_definitions_db (str): Path to the global definitions database file. Defaults to None.
            data_variant (str): Reranker context variant (base, def, hier, def_hier). Defaults to "base".

        Returns:
            None
        """
        self.ground_truth = ground_truth
        self.ontology_db = ontology_db
        self.target_db = target_db
        self.data_variant = data_variant
        self.ontology_embeddings = []
        self._ontology_matrix = None
        self.q0_mat = None

        # Load parent map and definitions for reranker context
        self.parent_map = {}   # name -> list[parent_name]
        self.definitions = {}  # name -> definition text
        self._load_parent_map()
        if global_definitions_db:
            self._load_definitions(global_definitions_db)

        # Load precomputed query embeddings (q0 only for now, as ground truth only has one column)
        self._load_query_embeddings()
        # Load ontology
        self._load_ontology()

    def _load_query_embeddings(self) -> None:
        """
        Description:
        Load or generate query embeddings for cleaned and standardized texts.

        Args:
            None

        Returns:
            None
        """
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

        def get_vec(text: str) -> torch.Tensor | None:
            """
            Description:
            Get the embedding vector for the given text, generating it on the fly if missing.

            Args:
                text (str): The input text to embed.

            Returns:
                torch.Tensor | None: The embedding vector as a torch Tensor, or None if text is empty.
            """
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

        for row in self.ground_truth:
            cleaned_vec = get_vec(row["cleaned"])
            q_cleaned.append(cleaned_vec)

        self.q_cleaned_mat = torch.stack(q_cleaned).to("cuda" if torch.cuda.is_available() else "cpu")
        self.q_std_mat = self.q_cleaned_mat

    def _load_parent_map(self) -> None:
        """
        Description:
        Load parent mapping from ontology DB for hierarchy breadcrumbs.

        Args:
            None

        Returns:
            None
        """
        logger.info(f"Loading parent map from {self.ontology_db}...")
        with sqlite3.connect(self.ontology_db) as conn:
            cursor = conn.cursor()
            # Determine parent column name
            cursor.execute("PRAGMA table_info(embeddings)")
            cols = [col[1] for col in cursor.fetchall()]
            parent_col = "parents" if "parents" in cols else "parent" if "parent" in cols else None
            if not parent_col:
                logger.warning("No parent column found in ontology DB.")
                return
            cursor.execute(f"SELECT name, {parent_col} FROM embeddings")
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
                        self.parent_map[name] = parents_list
        logger.info(f"Loaded parent map for {len(self.parent_map)} nodes.")

    def _load_definitions(self, definitions_db: str) -> None:
        """
        Description:
        Load definitions from a global definitions DB, mapping URI to definition text.

        Args:
            definitions_db (str): Path to the global definitions SQLite database.

        Returns:
            None
        """
        if not os.path.exists(definitions_db):
            logger.warning(f"Definitions DB not found: {definitions_db}")
            return

        # First build URI -> name map from ontology DB
        uri_to_name = {}
        with sqlite3.connect(self.ontology_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT uri, name FROM embeddings")
            for uri, name in cursor:
                uri_to_name[str(uri)] = name

        # Load definitions
        with sqlite3.connect(definitions_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT uri, definition FROM global_definitions")
            count = 0
            for uri, definition in cursor:
                name = uri_to_name.get(str(uri))
                if name and definition and len(definition.strip()) > 5:
                    self.definitions[name] = definition.strip()
                    count += 1
        logger.info(f"Loaded {count} definitions from {definitions_db}.")

    def _get_breadcrumbs(self, name: str) -> str:
        """
        Description:
        Build hierarchy breadcrumbs for a node by traversing its parent chain.

        Args:
            name (str): The node name to get breadcrumbs for.

        Returns:
            str: Formatted breadcrumb string(s).
        """
        paths = []
        def dfs(curr, current_path):
            if not curr or curr in current_path:
                paths.append(current_path)
                return
            parents = self.parent_map.get(curr, [])
            if not parents:
                paths.append(current_path + [curr])
                return
            for p in parents:
                dfs(p, current_path + [curr])
        dfs(name, [])
        if not paths:
            return name
        formatted = []
        for path in paths:
            formatted.append(" > ".join(reversed(path)))
        if len(formatted) > 1:
            return "\n".join([f"Path {i+1}: {p}" for i, p in enumerate(formatted)])
        return formatted[0]

    def _build_doc_text(self, candidate: dict) -> str:
        """
        Description:
        Build the document text for the reranker based on the data variant.

        Args:
            candidate (dict): Candidate dictionary with 'sanitized_name' and 'name' keys.

        Returns:
            str: The document text for the reranker.
        """
        doc = candidate['sanitized_name']
        use_def = self.data_variant in ("def", "def_hier")
        use_hier = self.data_variant in ("hier", "def_hier")

        if use_hier:
            hierarchy = self._get_breadcrumbs(candidate['name'])
            doc = f"{doc}\n<Hierarchy>: {hierarchy}"
        if use_def:
            definition = self.definitions.get(candidate['name'], "")
            if definition:
                doc = f"{doc}\n<Definition>: {definition}"
        return doc

    def _load_ontology(self) -> None:
        """
        Description:
        Load ontology embeddings and construct the ontology matrix.

        Args:
            None

        Returns:
            None
        """
        logger.info(f"Loading ontology matrix from {self.ontology_db}")
        self.ontology_embeddings = load_ontology_embeddings(self.ontology_db)
        self._ontology_matrix = torch.stack([item['vector'] for item in self.ontology_embeddings]).to(self.q_cleaned_mat.device)

    def get_top_k_candidates(self, top_k: int = 150) -> dict[str, list]:
        """
        Description:
        Computes top K candidates for each query using cosine similarity, for each GRF alpha.

        Args:
            top_k (int): Number of top candidates to retrieve. Defaults to 150.

        Returns:
            dict[str, list]: A dictionary mapping each GRF alpha value to the list of candidates.
        """
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

    def compute_reranker_scores(
        self,
        results_by_grf: dict[str, list],
        reranker_model_name: str,
        reranker_config_path: str
    ) -> dict[str, list]:
        """
        Description:
        Compute reranker scores for all unique candidate-query pairs across all GRF lists.

        Args:
            results_by_grf (dict[str, list]): The dictionary mapping GRF alpha to candidate lists.
            reranker_model_name (str): The name/path of the reranker model.
            reranker_config_path (str): The path to the reranker instruction file.

        Returns:
            dict[str, list]: The updated dictionary with reranker scores filled in.
        """
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
                    doc_text = self._build_doc_text(c)
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
                    doc_text = self._build_doc_text(all_candidates[qi][ci])
                    idx = pair_to_idx[(query_text, doc_text)]
                    all_candidates[qi][ci]["rerank_score"] = rerank_scores_flat[idx]

        return results_by_grf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--ontology_db", required=True)
    parser.add_argument("--target_embedding_db", required=True)
    parser.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--reranker_model", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--reranker_config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--toxval_db", required=True)
    parser.add_argument("--global_definitions_db", default=None, help="Path to global definitions DB for reranker context.")
    parser.add_argument("--data_variant", default="base", choices=["base", "def", "hier", "def_hier"],
                        help="Reranker context variant: base, def, hier, def_hier.")
    args = parser.parse_args()

    gt_data = load_ground_truth(args.ground_truth, args.toxval_db)
    precomputer = Precomputer(gt_data, args.ontology_db, args.target_embedding_db,
                              global_definitions_db=args.global_definitions_db,
                              data_variant=args.data_variant)

    # 1. Get Top 150 candidates for each GRF value
    results_by_grf = precomputer.get_top_k_candidates(top_k=150)

    # 2. Get Reranker scores for all unique candidates across all GRF lists
    full_data_by_grf = precomputer.compute_reranker_scores(results_by_grf, args.reranker_model, args.reranker_config)

    # 3. Save to Master Cache
    output_data = {
        "embedding_model": args.embedding_model,
        "reranker_model": args.reranker_model,
        "data_variant": args.data_variant,
        "ground_truth": gt_data,
        "precomputed_results_by_grf": full_data_by_grf
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    logger.info(f"Precomputation finished. Saved to {args.output}")
