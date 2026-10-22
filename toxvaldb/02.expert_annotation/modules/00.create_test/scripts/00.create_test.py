import argparse
import json
import os
import sys
from typing import Any, Literal

sys.modules['torchvision'] = None

import matplotlib.pyplot as plt
import polars as pl
import torch
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.llm import get_model, test_model
from src.utils import get_or_create_embeddings, is_valid_description, load_ontology_embeddings, setup_logging

DEFAULT_HF_HOME = os.environ.get("HF_HOME", "/nfs/research/ewald/ricos/huggingface_cache")
DEFAULT_HF_TRANSFER = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "1")

logger = setup_logging()

def arg_parser() -> argparse.Namespace:
    """
    Description:
    Parse command line arguments for generating the test dataset with candidate pools.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate Test Dataset with Candidate Pools")

    parser.add_argument("--input_file", required=True, help="Path to input TSV file.")
    parser.add_argument("--output_file", required=True, help="Path to output TXT file for selected phrases.")
    parser.add_argument("--output_json", help="Path to output JSON file for candidate pool (required for expert_test.py).")
    parser.add_argument("--output_visualization", help="Path to save the PCA visualization PNG.")
    parser.add_argument("--target_column", default="TOXICOLOGICAL_EFFECT", help="Column name containing the text to process.")

    parser.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-8B", help="HuggingFace model for embeddings.")
    parser.add_argument("--embeddings_model_id", default=None, help="Identifier for the model configuration in RAG config (e.g., qwen, nemotron)")
    parser.add_argument("--target_embedding_db", required=True, help="Path to input embeddings database.")
    parser.add_argument("--ontology_embedding_db", required=True, help="Path to ontology embeddings database.")

    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to select via K-Means.")
    parser.add_argument("--top_k_rag", type=int, default=15, help="Number of candidates from RAG.")
    parser.add_argument("--threshold_rag", type=float, default=0.0, help="Score threshold for RAG retrieval.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for PCA and K-Means.")

    parser.add_argument("--reranker_model", default="Qwen/Qwen3-Reranker-8B", help="HuggingFace model for reranking.")
    parser.add_argument("--threshold_reranker", type=float, default=0.0, help="Score threshold for reranker. If > 0.0, reranker is active.")
    parser.add_argument("--reranker_batch_size", type=int, default=16, help="Batch size for reranker.")
    parser.add_argument("--reranker_prompt_file", default=None, help="Path to reranker prompt instruction text file.")

    # --- LLM Configuration ---
    parser.add_argument("--model_config", help="Path to model config JSON file.")
    parser.add_argument("--base_url", help="Base URL for the LLM API. Overrides value in model_config if provided.")
    parser.add_argument("--system_prompt_file", default=None, help="Path to the system prompt text file.")
    parser.add_argument("--llm_model", help="LLM model name (optional override).")
    parser.add_argument("--consensus_runs", type=int, default=1, help="Number of LLM runs for consensus.")
    parser.add_argument("--max_concurrency", type=int, default=10, help="Max concurrent LLM calls.")
    parser.add_argument("--rag_config", help="Path to RAG configuration JSON file.")
    parser.add_argument("--debug", action="store_true", help="Enable LLM reasoning evaluation for candidates.")
    parser.add_argument("--liver", action="store_true", help="Skip PCA/K-Means and select phrases containing 'liver' or 'hepa'.")
    parser.add_argument("--histopathology", action="store_true", help="Skip PCA/K-Means and select phrases containing histopathology-related keywords (e.g. necrosis, hyperplasia).")

    # --- Environment & Orchestration ---
    parser.add_argument("--hf_home", default=None, help="HuggingFace cache directory.")
    parser.add_argument("--hf_transfer", default=None, help="Enable/disable HF transfer (0 or 1).")

    return parser.parse_args()

def clean_llm_output(message: Any) -> str:
    """
    Description:
    Removes <think> tags and markdown JSON formatting from LLM output.

    Args:
        message (Any): The message or string to clean.

    Returns:
        str: The cleaned message content.
    """
    text = message.content if hasattr(message, "content") else str(message)
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if text.startswith("```json"):
        text = text.replace("```json", "", 1)
    elif text.startswith("```"):
        text = text.replace("```", "", 1)
    if text.endswith("```"):
        text = text[::-1].replace("```", "", 1)[::-1]
    return text.strip()

def clean_reason(reason: str) -> str:
    """
    Description:
    Cleans candidate evaluation reasoning string by removing leading ACCEPTED/REJECTED.

    Args:
        reason (str): The reasoning string.

    Returns:
        str: The cleaned reasoning string.
    """
    r = reason.strip()
    if r.upper().startswith("ACCEPTED"):
        return r[8:].lstrip(" .-:")
    elif r.upper().startswith("REJECTED"):
        return r[8:].lstrip(" .-:")
    return r

class CandidateEvaluation(BaseModel):
    """Structure for evaluating a single candidate."""
    candidate_name: str = Field(description="The exact name of the candidate being evaluated.")
    reason: str = Field(description="Think step-by-step. Provide a self-contained biological or phenotypic justification for the decision. DO NOT cite rules by number (e.g., avoid 'Rule 1'); instead, explain the underlying logic so it is understandable without the prompt. DO NOT write 'ACCEPTED' or 'REJECTED' in this field.")
    decision: Literal["ACCEPTED", "REJECTED"] = Field(description="Strictly 'ACCEPTED' or 'REJECTED' based on the reasoning above.")

def main() -> None:
    """
    Description:
    Main execution function to load descriptions and embeddings, run PCA and K-Means,
    retrieve and evaluate candidates, and output the selected phrases.

    Args:
        None

    Returns:
        None
    """
    args = arg_parser()

    import numpy as np
    import random

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load RAG config if provided
    if args.rag_config:
        logger.info(f"Loading RAG configuration from: {args.rag_config}")
        with open(args.rag_config, 'r', encoding='utf-8') as f:
            rag_config = json.load(f)

        # If it's a nested dictionary (not flat with embedding_model directly), look up block
        if isinstance(rag_config, dict) and "embedding_model" not in rag_config:
            key = args.embeddings_model_id or args.embedding_model
            model_config_block = rag_config.get(key)
            if model_config_block:
                logger.info(f"Using RAG configuration block for: {key}")
                rag_config = model_config_block
            else:
                logger.warning(f"No specific RAG configuration found for '{key}'. Using default values.")

        # Override CLI defaults with config values
        args.embedding_model = rag_config.get("embedding_model", args.embedding_model)
        args.top_k_rag = rag_config.get("top_k_rag", args.top_k_rag)
        args.threshold_rag = rag_config.get("threshold_rag", args.threshold_rag)
        args.reranker_model = rag_config.get("reranker_model", args.reranker_model)
        args.threshold_reranker = rag_config.get("threshold_reranker", args.threshold_reranker)
        args.reranker_batch_size = rag_config.get("reranker_batch_size", getattr(args, 'reranker_batch_size', 16))
        
        reranker_prompt_file = rag_config.get("reranker_prompt", getattr(args, 'reranker_prompt_file', None))
        if reranker_prompt_file and not os.path.isabs(reranker_prompt_file):
            reranker_prompt_file = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(args.rag_config)), reranker_prompt_file))
        args.reranker_prompt_file = reranker_prompt_file

        # Centralized path construction for target embeddings if placeholder or missing
        if not args.target_embedding_db or "tox_effects_embeddings.db" in args.target_embedding_db:
             # Use embeddings_model_id as dir name if available, else infer from model path
            embed_dir_name = args.embeddings_model_id or args.embedding_model.split("/")[-1].split("-")[0].lower()
            args.target_embedding_db = os.path.join(os.path.dirname(__file__), '..', '..', '00.data_preparation', 'outputs', f'tox_effects_embeddings_{embed_dir_name}.db')
            logger.info(f"Using target embedding db: {args.target_embedding_db}")

    # --- Set Environment Variables ---
    hf_home = args.hf_home or os.environ.get("HF_HOME") or DEFAULT_HF_HOME
    hf_transfer = args.hf_transfer or os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") or DEFAULT_HF_TRANSFER

    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = hf_transfer

    logger.info(f"HuggingFace Home: {hf_home}")
    logger.info(f"HF Transfer: {hf_transfer}")

    logger.info(f"Loading input file: {args.input_file}")
    df = pl.read_csv(args.input_file, separator='\t', ignore_errors=True, quote_char=None)

    descriptions = df.select(pl.col(args.target_column)).unique(maintain_order=True).drop_nulls().to_series().to_list()
    descriptions.sort()
    descriptions = [d for d in descriptions if is_valid_description(d)]

    logger.info(f"Found {len(descriptions)} unique descriptions.")

    if not descriptions:
        logger.error("No valid descriptions found.")
        return

    logger.info("Loading/Generating input embeddings...")
    text_to_vec = get_or_create_embeddings(descriptions, args.embedding_model, args.target_embedding_db)

    valid_descs = [d for d in descriptions if d in text_to_vec]
    if not valid_descs:
        logger.error("No embeddings available.")
        return

    logger.info("Loading ontology embeddings for selection validation...")
    ontology_data = load_ontology_embeddings(args.ontology_embedding_db)
    if not ontology_data:
        logger.error("Ontology data is empty.")
        return
    ontology_matrix = torch.stack([item['vector'] for item in ontology_data])

    llm = None
    max_concurrency = args.max_concurrency
    if (args.debug or args.histopathology) and args.model_config:
        logger.info(f"Initializing LLM from: {args.model_config}")
        with open(args.model_config, 'r', encoding='utf-8') as f:
            model_config = json.load(f)

        model_config.pop("_comment", None)
        max_concurrency = model_config.pop("max_concurrency", args.max_concurrency)
        if args.base_url:
            model_config["base_url"] = args.base_url
        if args.llm_model:
            model_config["model"] = args.llm_model

        llm = get_model(**model_config)
        test_model(llm, max_retries=model_config.get("max_retries", 3))

    if args.histopathology:
        if not llm:
            logger.error("--model_config is required when using --histopathology.")
            return

        logger.info("Filtering for histopathology-related phrases using LLM...")

        class HistopathologyCheck(BaseModel):
            decision: Literal["YES", "NO"] = Field(description="Strictly 'YES' if the description is a histopathology outcome in a broad sense, otherwise 'NO'.")

        parser_histo = PydanticOutputParser(pydantic_object=HistopathologyCheck)
        prompt_histo = ChatPromptTemplate.from_messages([
            ("system", "You are a toxicological expert. Evaluate if the following biological or phenotypic description is a histopathology outcome in a broad sense.\n\n{format_instructions}"),
            ("user", "Description: {description}")
        ])

        llm_chain_histo = prompt_histo | llm | RunnableLambda(clean_llm_output) | parser_histo

        logger.info(f"Evaluating {len(valid_descs)} descriptions for histopathology...")
        batch_inputs = [{"description": d, "format_instructions": llm_chain_histo.last.get_format_instructions()} for d in valid_descs]

        try:
            batch_results = []
            chunk_size = max_concurrency * 5  # chunk size to update the progress bar regularly

            with tqdm(total=len(batch_inputs), desc="Filtering Histopathology", smoothing=0.0) as pbar:
                for i in range(0, len(batch_inputs), chunk_size):
                    chunk = batch_inputs[i:i+chunk_size]
                    chunk_res = llm_chain_histo.batch(chunk, config={"max_concurrency": max_concurrency})
                    batch_results.extend(chunk_res)
                    pbar.update(len(chunk))

            histo_descs = [d for d, res in zip(valid_descs, batch_results) if res.decision == "YES"]
            logger.info(f"Found {len(histo_descs)} histopathology outcomes out of {len(valid_descs)}.")
            valid_descs = histo_descs
            if not valid_descs:
                logger.error("No valid histopathology descriptions found.")
                return
        except Exception as e:
            logger.error(f"LLM histopathology evaluation failed: {e}")
            return


    # --- RERANKER AND RAG HELPERS ---
    # Initialize Reranker if requested
    reranker_model = None
    reranker_tokenizer = None
    reranker_device = None
    token_false_id = None
    token_true_id = None
    reranker_prefix_tokens = None
    reranker_suffix_tokens = None
    reranker_instruction = ""
    if args.threshold_reranker > 0.0:
        logger.info(f"Loading Reranker Model: {args.reranker_model} (Threshold: {args.threshold_reranker})")
        reranker_tokenizer = AutoTokenizer.from_pretrained(args.reranker_model, padding_side='left')
        reranker_model = AutoModelForCausalLM.from_pretrained(
            args.reranker_model, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        reranker_device = reranker_model.device
        token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
        token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        reranker_prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
        reranker_suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)
        reranker_prompt_file = getattr(args, 'reranker_prompt_file', None)
        if reranker_prompt_file and os.path.exists(reranker_prompt_file):
            with open(reranker_prompt_file, 'r', encoding='utf-8') as f:
                reranker_instruction = f.read().strip()
            logger.info(f"Loaded reranker instruction from: {reranker_prompt_file}")
        elif os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "reranker_prompt.txt")):
            default_prompt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "reranker_prompt.txt")
            with open(default_prompt, 'r', encoding='utf-8') as f:
                reranker_instruction = f.read().strip()
            logger.info(f"Loaded reranker instruction from module config: {default_prompt}")
        else:
            reranker_instruction = "Given a biological phenotype description observed after chemical exposure, retrieve the identical ontology term.\nCRITICAL RULES FOR SELECTION:\n1. STRICT SPECIFICITY ALIGNMENT: The scope of the description must match the biological scope of the candidate. A broad, system-level description MUST ONLY be mapped to equally broad, system-level ontology terms.\n2. NO SUB-SYSTEMS OR PROCESSES: If the input describes an entire physiological system, you MUST REJECT candidates that refer to specific sub-functions, localized mechanisms, or pathways.\nMatch them based strictly on whether the underlying biological or phenotypic target is identical."
            logger.warning("reranker_prompt file not found or not set — using fallback hardcoded instruction.")

    # Pre-compute nodes map for faster breadcrumb generation
    nodes_map = {item['name']: item for item in ontology_data}

    # Helper for hierarchy breadcrumbs
    def get_all_paths(name: str, nodes_map: dict) -> list[list[str]]:
        paths = []
        def dfs(curr, current_path):
            if not curr or curr in current_path:
                paths.append(current_path)
                return
            parents = nodes_map.get(curr, {}).get('parents', [])
            if not parents:
                paths.append(current_path + [curr])
                return
            for p in parents:
                dfs(p, current_path + [curr])
        dfs(name, [])
        return paths

    def get_breadcrumbs(name: str, nodes_map: dict) -> str:
        """
        Description:
        Generate breadcrumbs path for a given node name.

        Args:
            name (str): The node name.
            nodes_map (dict): A map from node names to node dictionaries.

        Returns:
            str: The breadcrumbs path string.
        """
        paths = get_all_paths(name, nodes_map)
        if not paths: return name
        formatted = []
        for i, path in enumerate(paths):
            labels = [nodes_map.get(n, {}).get('sanitized_name', n) for n in path]
            formatted.append(" > ".join(reversed(labels)))
        return "\n".join([f"Path {i+1}: {p}" for i, p in enumerate(formatted)]) if len(formatted) > 1 else formatted[0]

    def evaluate_candidates(phrase: str, phrase_vec: torch.Tensor) -> list[dict]:
        """
        Description:
        Evaluate and rerank candidates for a given phrase and its vector.

        Args:
            phrase (str): The input phrase.
            phrase_vec (torch.Tensor): The embedding vector of the phrase.

        Returns:
            list[dict]: A list of dictionary candidates with scores and hierarchy.
        """
        scores = torch.matmul(ontology_matrix, phrase_vec)
        rag_vals, rag_idxs = torch.topk(scores, k=min(args.top_k_rag, len(ontology_data)))

        final_pool = []
        for idx, score in zip(rag_idxs.tolist(), rag_vals.tolist()):
            if score < args.threshold_rag:
                continue
            item = ontology_data[idx]
            final_pool.append({
                "uri": item.get('uri'),
                "name": item['name'],
                "sanitized_name": item['sanitized_name'],
                "score": round(float(score), 2),
                "definition": item['def'],
                "is_llm_generated": item['is_llm_generated']
            })

        if args.threshold_reranker > 0.0 and len(final_pool) > 0:
            pairs = []
            reranker_query = phrase # Always use original for reranker query
            for c in final_pool:
                h = get_breadcrumbs(c['name'], nodes_map)
                c['hierarchy'] = h # Cache it for later use
                doc = c['sanitized_name']
                doc = f"{doc}\n<Hierarchy>: {h}"
                pair_text = f"<Instruct>: {reranker_instruction}\n<Query>: {reranker_query}\n<Document>: {doc}"
                pairs.append(pair_text)

            batch_size = getattr(args, 'reranker_batch_size', 16)
            scores_out = []

            with torch.no_grad():
                for i in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[i:i+batch_size]

                    inputs = reranker_tokenizer(
                        batch_pairs, padding=False, truncation='longest_first',
                        return_attention_mask=False, max_length=1024 - len(reranker_prefix_tokens) - len(reranker_suffix_tokens)
                    )

                    for j, ele in enumerate(inputs['input_ids']):
                        inputs['input_ids'][j] = reranker_prefix_tokens + ele + reranker_suffix_tokens

                    inputs = reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt")

                    for key in inputs:
                        inputs[key] = inputs[key].to(reranker_device)

                    batch_logits = reranker_model(**inputs).logits[:, -1, :]
                    true_vector = batch_logits[:, token_true_id]
                    false_vector = batch_logits[:, token_false_id]
                    batch_scores = torch.stack([false_vector, true_vector], dim=1)
                    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                    batch_probs = batch_scores[:, 1].exp().tolist()
                    scores_out.extend(batch_probs)

            passed_candidates = []
            for score, cand in zip(scores_out, final_pool):
                if score >= args.threshold_reranker:
                    cand['rerank_score'] = score
                    passed_candidates.append(cand)

            final_pool = passed_candidates

        for i, cand in enumerate(final_pool):
            cand['original_order'] = i + 1
            if 'hierarchy' not in cand:
                cand['hierarchy'] = get_breadcrumbs(cand['name'], nodes_map)

        return final_pool

    precomputed_candidates = {}

    if args.liver:
        logger.info(f"Filtering for liver-related phrases (--liver flag is active). Searching up to {args.num_samples} samples.")
        keywords = ["liver", "hepa"]
        selected_phrases = []
        for d in valid_descs:
            desc_lower = d.lower()
            if any(k in desc_lower for k in keywords):
                selected_phrases.append(d)
                if len(selected_phrases) >= args.num_samples:
                    break

        # If we didn't find enough, we just use what we found
        logger.info(f"Found {len(selected_phrases)} phrases matching keywords: {keywords}")
        closest_indices = [valid_descs.index(phrase) for phrase in selected_phrases]
        embeddings_2d = None # No PCA needed
    else:
        query_vecs = []
        for d in valid_descs:
            qvec = text_to_vec[d].detach().clone()
            query_vecs.append(qvec)

        embeddings_matrix = torch.stack(query_vecs).numpy()

        logger.info(f"Performing PCA and K-Means selection ({args.num_samples} samples)...")

        pca = PCA(n_components=2, svd_solver='randomized', random_state=args.seed)
        embeddings_2d = pca.fit_transform(embeddings_matrix)

        num_clusters = min(args.num_samples, len(valid_descs))
        kmeans = KMeans(n_clusters=num_clusters, random_state=args.seed, n_init=10)
        kmeans.fit(embeddings_2d)

        selected_phrases = []
        closest_indices = []
        cluster_labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        for c_idx in range(num_clusters):
            in_cluster = np.where(cluster_labels == c_idx)[0]
            if len(in_cluster) == 0: continue

            # Sort points in cluster by distance to its centroid
            dists = np.linalg.norm(embeddings_2d[in_cluster] - centroids[c_idx], axis=1)
            sorted_in_cluster = in_cluster[np.argsort(dists)]

            found_idx = -1
            if args.histopathology:
                # Iterate through candidates in cluster until we find one with >= 2 RAG results
                for idx in sorted_in_cluster:
                    qvec = query_vecs[idx]
                    phrase = valid_descs[idx]
                    # Full evaluation including reranker
                    final_pool = evaluate_candidates(phrase, qvec)
                    if len(final_pool) >= 2:
                        found_idx = idx
                        precomputed_candidates[phrase] = final_pool
                        break

                if found_idx == -1:
                    logger.warning(f"Cluster {c_idx} has no phrase with >= 2 candidates. Falling back to closest.")

            if found_idx == -1:
                found_idx = sorted_in_cluster[0]

            closest_indices.append(found_idx)
            selected_phrases.append(valid_descs[found_idx])

    if args.output_visualization and embeddings_2d is not None:
        logger.info(f"Saving visualization to {args.output_visualization}")
        plt.figure(figsize=(10, 7))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='lightgrey', label='All descriptions', s=10, alpha=0.5)
        plt.scatter(embeddings_2d[closest_indices, 0], embeddings_2d[closest_indices, 1], c='red', label='Selected samples', s=20)
        plt.title(f"PCA Selection: {len(selected_phrases)} samples from {len(valid_descs)} unique phrases")
        plt.legend()
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.savefig(args.output_visualization, dpi=300)
        plt.close()

    results = []


    # Initialize LLM Evaluation chain if requested
    llm_chain = None
    if args.debug and llm:
        system_prompt_path = args.system_prompt_file
        if system_prompt_path is None:
             system_prompt_path = "config/prompts/system_prompt_1.txt" # Default path

        if not os.path.exists(system_prompt_path):
            logger.warning(f"System prompt file not found at {system_prompt_path}")
            system_prompt_content = "You are a biological phenotype classifier. Match the description to the most appropriate ontology term."
        else:
            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                system_prompt_content = f.read()

        parser = PydanticOutputParser(pydantic_object=CandidateEvaluation)

        user_prompt = "Description: {description}\nCandidate to evaluate: {candidate_name}"
        user_prompt += "\nCandidate Hierarchy: {candidate_hierarchy}"

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt_content + "\n\n{format_instructions}"),
            ("user", user_prompt)
        ])
        llm_chain = prompt_template | llm | RunnableLambda(clean_llm_output) | parser


    logger.info("Generating candidate pools...")
    for phrase in tqdm(selected_phrases, desc="Processing Phrases", smoothing=0.0):
        if phrase in precomputed_candidates:
            final_pool = precomputed_candidates[phrase]
        else:
            phrase_vec = text_to_vec[phrase].detach().clone()
            final_pool = evaluate_candidates(phrase, phrase_vec)

        if not args.debug:
            random.shuffle(final_pool)

        if args.debug and llm_chain and final_pool:
            logger.info(f"Evaluating {len(final_pool)} candidates with LLM for phrase: {phrase}")
            batch_inputs = []
            for cand in final_pool:
                for _ in range(args.consensus_runs):
                    ctx = {
                        "description": phrase,
                        "candidate_name": cand['sanitized_name'],
                        "format_instructions": llm_chain.last.get_format_instructions()
                    }
                    ctx["candidate_hierarchy"] = cand['hierarchy']
                    batch_inputs.append(ctx)

            try:
                batch_results = llm_chain.batch(batch_inputs, config={"max_concurrency": max_concurrency})

                threshold = (args.consensus_runs // 2) + 1
                for i, cand in enumerate(final_pool):
                    cand_results = batch_results[i * args.consensus_runs : (i + 1) * args.consensus_runs]
                    yes_votes = sum(1 for r in cand_results if r.decision == 'ACCEPTED')

                    cand['llm_decision'] = "ACCEPTED" if yes_votes >= threshold else "REJECTED"
                    cand['llm_reasoning'] = clean_reason(cand_results[0].reason) if cand_results else "No reasoning provided."
                    if args.consensus_runs > 1:
                        cand['llm_reasoning'] = f"({yes_votes}/{args.consensus_runs} votes) " + cand['llm_reasoning']
            except Exception as e:
                logger.error(f"LLM evaluation failed for phrase '{phrase}': {e}")
                for cand in final_pool:
                    cand['llm_decision'] = "UNKNOWN"
                    cand['llm_reasoning'] = f"Error: {e}"

        results.append({
            "phrase": phrase,
            "candidates": final_pool
        })

    logger.info(f"Saving selected phrases to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(f"{item['phrase']}\n")

    if args.output_json:
        logger.info(f"Saving candidate pool to {args.output_json}")
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Done.")

if __name__ == "__main__":
    main()
