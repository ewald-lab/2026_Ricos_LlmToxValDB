import argparse
import json
import os
import polars as pl
import logging
import time
import torch

# Environment defaults (can be overridden by CLI or environment variables)
DEFAULT_HF_HOME = "/nfs/research/ewald/ricos/huggingface_cache"
DEFAULT_HF_TRANSFER = "1"

from tqdm import tqdm

from src.utils import (
    setup_logging, is_valid_description, get_or_create_embeddings, 
    load_ontology_nodes, get_debug_rows, get_unique_descriptions,
    save_backup_chunk, merge_backups
)
from model.llm import get_model, test_model
from src.graph import TextToOntoGraph

logger = setup_logging()

def arg_parser():
    parser = argparse.ArgumentParser(description="Text to Onto LLM Classifier")
    
    # --- File I/O & Columns ---
    parser.add_argument("--input_file", required=True, help="Path to input TSV file.")
    parser.add_argument("--output_file", required=True, help="Path to output CSV file.")
    parser.add_argument("--reasoning_file", required=True, help="Path to output JSON file for reasoning.")
    parser.add_argument("--target_column", default="TOXICOLOGICAL_EFFECT", help="Column name containing the text to process.")
    parser.add_argument("--std_column", default="TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED", help="Column name containing the standardized text.")
    
    # --- LLM Configuration ---
    parser.add_argument("--model_config", required=True, help="Path to model config JSON file.")
    parser.add_argument("--base_url", help="Base URL for the LLM API. Overrides value in model_config if provided.")
    parser.add_argument("--system_prompt_file", default=None, help="Path to the system prompt text file.")
    
    # --- Execution Logic & Consensus ---
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2], help="Execution mode: 1 for individual verification, 2 for batch selection.")
    parser.add_argument("--consensus_runs", type=int, default=3, help="Number of runs for majority voting consensus.")
    parser.add_argument("--num_backups", type=int, default=4, help="Number of backups to save during processing (0 to disable).")
    parser.add_argument("--ontology_definitions", action="store_true", help="Use node definitions from DB in prompts.")
    parser.add_argument("--context_ontology", default="", help="Context information about the ontology to use in prompts.")
    
    # --- Embedding & Database Configuration ---
    parser.add_argument("--rag_config", required=True, help="Path to RAG configuration JSON file.")
    parser.add_argument("--embeddings_model_id", default="qwen", help="Identifier for the model configuration in RAG config (e.g., qwen, nemotron)")
    parser.add_argument("--ontology_embedding_db", required=True, help="Path to ontology embeddings database.")
    parser.add_argument("--target_embedding_db", default="", help="Path to save embedding database for target column.")
    parser.add_argument("--global_definitions_db", default=None, help="Path to global definitions database (URI to definition).")
    
    # --- Debugging ---
    parser.add_argument("--debug_rows", type=int, nargs='+', help="Specific row indices to process for debugging.")
    parser.add_argument("--random_debug_rows", type=int, nargs='?', const=10, default=None, help="Number of random rows to process for debugging (default 10 if flag used without value).")
    parser.add_argument("--seed", type=int, default=43, help="Random seed for reproducibility.")
    
    # --- Environment & Orchestration ---
    parser.add_argument("--hf_home", default=None, help="HuggingFace cache directory.")
    parser.add_argument("--hf_transfer", default=None, help="Enable/disable HF transfer (0 or 1).")
    
    return parser.parse_args()

def main():
    args = arg_parser()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    start_time = time.time()
    
    if args.system_prompt_file is None:
        args.system_prompt_file = f"config/prompts/system_prompt_{args.mode}.txt"
    
    # --- Set Environment Variables ---
    hf_home = args.hf_home or os.environ.get("HF_HOME") or DEFAULT_HF_HOME
    hf_transfer = args.hf_transfer or os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") or DEFAULT_HF_TRANSFER
    
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = hf_transfer
    
    logger.info(f"HuggingFace Home: {hf_home}")

    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        logger.error(f"Current working directory: {os.getcwd()}")
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Load RAG config (REQUIRED)
    logger.info(f"Loading RAG configuration from: {args.rag_config}")
    with open(args.rag_config, 'r', encoding='utf-8') as f:
        rag_config = json.load(f)
    
    # If it's a dictionary, look up by embeddings_model_id or embedding_model string
    if isinstance(rag_config, dict):
        key = args.embeddings_model_id or rag_config.get("embedding_model", "")
        # Case-insensitive key lookup
        rag_config_lower = {k.lower(): v for k, v in rag_config.items()}
        model_config_block = rag_config_lower.get(key.lower())
        if not model_config_block:
            # Fallback scan for embedding_model match if key was not found
            model_config_block = next((v for v in rag_config.values() if v.get("embedding_model") == key), None)
        
        if model_config_block:
            logger.info(f"Using RAG configuration block for: {key}")
            rag_config = model_config_block
        else:
            logger.warning(f"No specific RAG configuration found for '{key}'. Using default values.")

    # Extract RAG parameters
    embedding_model = rag_config.get("embedding_model", "Qwen/Qwen3-Embedding-8B")
    top_k_rag = rag_config.get("top_k_rag", 15)
    threshold_rag = rag_config.get("threshold_rag", 0.5)
    reranker_model = rag_config.get("reranker_model", "Qwen/Qwen3-Reranker-8B")
    threshold_reranker = rag_config.get("threshold_reranker", 0.0)
    reranker_batch_size = rag_config.get("reranker_batch_size", 32)
    reranker_prompt_file = rag_config.get("reranker_prompt")
    grf_alpha = rag_config.get("grf")
    
    with open(args.model_config, 'r', encoding='utf-8') as f:
        model_config = json.load(f)

    max_concurrency = model_config.pop("max_concurrency", 10)

    if args.base_url:
        model_config["base_url"] = args.base_url
        logger.info(f"Overriding base_url from CLI: {args.base_url}")

    logger.info(f"Loading model configuration from: {args.model_config}")
    logger.info(f"  max_concurrency: {max_concurrency}")
    for key, value in model_config.items():
        if key != "api_key":
            logger.info(f"  {key}: {value}")

    llm = get_model(**model_config)
    
    test_retries = model_config.get('max_retries', 3)
    if not test_model(llm, max_retries=test_retries):
        raise RuntimeError("Model connectivity test failed. Please check your configuration.")

    ontology_db_path = args.ontology_embedding_db
    
    if not os.path.exists(ontology_db_path):
        logger.warning(f"Ontology DB not found at {ontology_db_path}. Please ensure you ran process_ontology or check the path.")
    
    # Derive prefix from ontology DB name (e.g., mp.db -> MP:)
    db_filename = os.path.basename(ontology_db_path)
    onto_name = db_filename.split('.')[0].upper()
    prefix = f"{onto_name}:"
    logger.info(f"Using ontology prefix: {prefix}")

    nodes, categories = load_ontology_nodes(ontology_db_path, prefix=prefix)

    db_already_exists = os.path.exists(args.target_embedding_db)

    graph = TextToOntoGraph(
        categories,
        ontology_db=ontology_db_path, embedding_model_name=embedding_model,
        input_db=args.target_embedding_db, top_k_rag=top_k_rag,
        system_prompt_path=args.system_prompt_file,
        threshold_rag=threshold_rag,
        reranker_model_name=reranker_model, threshold_reranker=threshold_reranker,
        mode=args.mode,
        consensus_runs=args.consensus_runs,
        ontology_definitions=args.ontology_definitions,
        max_concurrency=max_concurrency,
        max_retries=test_retries,
        context_ontology=args.context_ontology,
        reranker_prompt_file=reranker_prompt_file,
        reranker_batch_size=reranker_batch_size,
        global_definitions_db=args.global_definitions_db
    )

    try:
        logger.info(f"Loading data from {args.input_file}")
        df = pl.read_csv(args.input_file, separator='\t', ignore_errors=True, quote_char=None)
        
        # Determine the exact schema for chunks to avoid any read/write/concat inference errors 
        target_schema = dict(df.schema)
        for cat in categories:
            # If the category column already exists (from _03 file), keep its current type or ensure it's compatible
            if cat in target_schema:
                # We might have floats (NaN) or ints (0/1). Float64 is safer for NaN support.
                target_schema[cat] = pl.Float64
            else:
                target_schema[cat] = pl.Float64
        
        unique_descs = df.select(args.target_column).unique().drop_nulls().to_series().to_list()
        unique_descs = [d for d in unique_descs if is_valid_description(d)]
        get_or_create_embeddings(unique_descs, embedding_model, args.target_embedding_db, return_vectors=False)
        
        if grf_alpha is not None and args.std_column in df.columns:
            unique_stds = df.select(args.std_column).unique().drop_nulls().to_series().to_list()
            unique_stds = [d for d in unique_stds if is_valid_description(d)]
            get_or_create_embeddings(unique_stds, embedding_model, args.target_embedding_db, return_vectors=False)
        
        args.debug_rows = get_debug_rows(df, args.target_column, args.random_debug_rows, args.debug_rows, args.seed)
        

        gpu_id_str = f"GPU:{os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}"
        unique_descriptions, seen_descriptions = get_unique_descriptions(df, args.target_column, args.debug_rows)
        
        desc_to_std = {}
        if grf_alpha is not None and args.std_column in df.columns:
            # Create a mapping from target_column to std_column
            mapping_df = df.select([args.target_column, args.std_column]).drop_nulls().group_by(args.target_column, maintain_order=True).agg(pl.col(args.std_column).first())
            desc_to_std = dict(zip(mapping_df[args.target_column], mapping_df[args.std_column]))

        description_cache = {}
        total_descs = len(unique_descriptions)
        num_backups = args.num_backups
        backup_interval = (total_descs + num_backups - 1) // num_backups if num_backups > 0 else total_descs + 1
        backup_count = 1

        for idx, desc in enumerate(tqdm(unique_descriptions, desc=f"Worker {gpu_id_str} (Unique)", smoothing=0.0)):
            try:
                logger.info(f"\n--------------------------------------------------------")
                logger.info(f"{gpu_id_str} | Processing unique description")
                
                std_desc = desc_to_std.get(desc) if grf_alpha is not None else None
                row_result_values, reasoning, selected = graph.execute(llm, desc, std_desc=std_desc, alpha=grf_alpha)
                
                description_cache[desc] = (row_result_values, reasoning, selected)
            except Exception as e:
                logger.error(f"Error processing description '{desc[:50]}...': {e}")
                
            if num_backups > 0:
                is_fin = (idx + 1) == total_descs
                if (idx + 1) % backup_interval == 0 or is_fin:
                    save_backup_chunk(df, description_cache, backup_count, 
                                    categories, args.target_column, 
                                    args.output_file, args.reasoning_file, 
                                    target_schema, is_final=is_fin, 
                                    debug_rows=args.debug_rows, seen_descriptions=seen_descriptions)
                    description_cache.clear()
                    if not is_fin: 
                        backup_count = backup_count + 1

        if num_backups > 0:
            merge_backups(args.output_file, args.reasoning_file, backup_count, 
                        categories, args.target_column, target_schema)
        else:
            logger.info("Applying results to rows and saving...")
            # Fallback for when no backups are used
            save_backup_chunk(df, description_cache, 0, categories, args.target_column, 
                             args.output_file, args.reasoning_file, target_schema, 
                             is_final=True, debug_rows=args.debug_rows, seen_descriptions=seen_descriptions)
            # Rename backup 0 to final
            os.rename(args.output_file.replace('.csv', '_0.csv'), args.output_file)
            os.rename(args.reasoning_file.replace('.json', '_0.json'), args.reasoning_file)
        
        logger.info(f"Total execution time: {time.time() - start_time:.2f}s")
    finally:
        graph.close()

if __name__ == "__main__":
    main()
