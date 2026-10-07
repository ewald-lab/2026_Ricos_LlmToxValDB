import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl
import torch
from tqdm import tqdm

from model.llm import get_model, test_model
from src.graph import TextToOntoGraph
from src.utils import (
    get_or_create_embeddings,
    get_unique_descriptions,
    is_valid_description,
    load_ontology_nodes,
    merge_backups,
    save_backup_chunk,
    setup_logging,
)

# --- Global Constants ---
DEFAULT_HF_HOME = "/nfs/research/ewald/ricos/huggingface_cache"
DEFAULT_HF_TRANSFER = "1"
MODEL_CONFIG_TEMPLATE = "/nfs/research/ewald/ricos/code/playground/models/config/{model}.json"

MODEL_PORTS = {
    "GLM-5.1": "8000",
    "gpt-oss-20b": "8001",
    "NVIDIA-Nemotron-3-Super-120B-A12B-FP8": "8002",
    "Qwen3.5-122B-A10B-FP8": "8003",
    "gpt-oss-120b": "8004",
    "Qwen3.5-35B-A3B-FP8": "8005",
    "Qwen3.6-35B-A3B-FP8": "8052",
    "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8": "8007",
    "Qwen3.5-27B-FP8": "8008",
    "Qwen3.6-27B-FP8": "8009"
}

logger = setup_logging()

def replace_placeholders(text: str, key: str, value: str) -> str:
    """
    Description:
    Helper to replace placeholders in format {key}, ${key}, {KEY}, ${KEY}.

    Args:
        text (str): The input text containing placeholders.
        key (str): The placeholder key to match (e.g. 'model').
        value (str): The value to substitute the placeholder with.

    Returns:
        str: The text with all matched placeholders replaced by the value.
    """
    for placeholder in (f"{{{key}}}", f"${{{key}}}", f"{{{key.upper()}}}", f"${{{key.upper()}}}"):
        text = text.replace(placeholder, str(value))
    return text

def arg_parser() -> argparse.Namespace:
    """
    Description:
    Parses command-line arguments for the Text to Onto LLM Classifier.

    Args:
        None

    Returns:
        argparse.Namespace: The parsed command-line arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Text to Onto LLM Classifier")

    # --- File I/O & Columns ---
    parser.add_argument("--input_file", required=True, help="Path to input TSV file.")
    parser.add_argument("--output_file", required=True, help="Path to output CSV file.")
    parser.add_argument("--reasoning_file", required=True, help="Path to output JSON file for reasoning.")
    parser.add_argument("--target_column", default="TOXICOLOGICAL_EFFECT", help="Column name containing the text to process.")

    # --- LLM Configuration ---
    parser.add_argument("--model_config", required=True, help="Path to model config JSON file.")
    parser.add_argument("--base_url", help="Base URL for the LLM API. Overrides value in model_config if provided.")
    parser.add_argument("--system_prompt_file", default=None, help="Path to the system prompt text file.")

    # --- Execution Logic & Consensus ---
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2], help="Execution mode: 1 for individual verification, 2 for batch selection.")
    parser.add_argument("--consensus_runs", type=int, default=3, help="Number of runs for majority voting consensus.")
    parser.add_argument("--num_backups", type=int, default=4, help="Number of backups to save during processing (0 to disable).")
    parser.add_argument("--ontology_definitions", dest="ontology_definitions", action="store_true", default=None, help="Use node definitions from DB in prompts.")
    parser.add_argument("--no_ontology_definitions", dest="ontology_definitions", action="store_false", help="Do not use node definitions from DB in prompts.")
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
    parser.add_argument("--llm_config", default=None, help="Path to llm_config.json configuration file.")
    parser.add_argument("--descriptions_concurrency", type=int, default=5, help="Number of descriptions to process in parallel.")

    return parser.parse_args()

def load_llm_config(args: argparse.Namespace) -> None:
    """
    Description:
    Loads configuration from llm_config and resolves dynamic model, mode, consensus, output paths, and ports.

    Args:
        args (argparse.Namespace): The parsed command-line arguments to be modified in-place.

    Returns:
        None
    """
    if not args.llm_config or not os.path.exists(args.llm_config):
        if args.ontology_definitions is None:
            args.ontology_definitions = False
        return

    logger.info(f"Loading configuration from llm_config: {args.llm_config}")
    with open(args.llm_config, 'r', encoding='utf-8') as jf:
        best_config = json.load(jf)
    
    model = best_config["model"]
    args.mode = best_config["mode"]
    args.consensus_runs = best_config["consensus_runs"]
    if args.ontology_definitions is None:
        args.ontology_definitions = best_config.get("ontology_definitions", False)
    args.model_config = MODEL_CONFIG_TEMPLATE.format(model=model)
    
    # Replace placeholders in filenames
    for attr in ('output_file', 'reasoning_file'):
        val = getattr(args, attr)
        if val:
            setattr(args, attr, replace_placeholders(val, "model", model))

    port = MODEL_PORTS.get(model, "8003")
    if args.base_url:
        if "PORT" in args.base_url:
            args.base_url = args.base_url.replace("PORT", port)

    if args.system_prompt_file:
        args.system_prompt_file = replace_placeholders(args.system_prompt_file, "mode", str(args.mode))

    if args.ontology_definitions is None:
        args.ontology_definitions = False

def load_rag_config(args: argparse.Namespace) -> dict:
    """
    Description:
    Loads RAG configuration from a JSON file and resolves EMBED_DIR placeholders in args.

    Args:
        args (argparse.Namespace): The command-line arguments namespace to be modified in-place.

    Returns:
        dict: The loaded RAG configuration dictionary.
    """
    if not os.path.exists(args.rag_config):
        raise FileNotFoundError(f"RAG config file not found: {args.rag_config}")
    logger.info(f"Loading RAG configuration from: {args.rag_config}")
    with open(args.rag_config, 'r', encoding='utf-8') as f:
        rag_config = json.load(f)

    embedding_model_name = rag_config.get("embedding_model", "").lower()
    if "kalm" in embedding_model_name:
        embed_dir = "kalm"
    elif "qwen" in embedding_model_name:
        embed_dir = "qwen"
    elif "nemotron" in embedding_model_name:
        embed_dir = "nemotron"
    elif "harrier" in embedding_model_name:
        embed_dir = "harrier"
    else:
        embed_dir = args.embeddings_model_id or "qwen"

    for attr in ('ontology_embedding_db', 'target_embedding_db'):
        val = getattr(args, attr)
        if val:
            resolved_val = replace_placeholders(val, "embed_dir", embed_dir)
            setattr(args, attr, resolved_val)
            logger.info(f"Resolved {attr} path: {resolved_val}")

    return rag_config

def init_env(args: argparse.Namespace) -> None:
    """
    Description:
    Initializes environment variables, random seeds, and logging configuration.

    Args:
        args (argparse.Namespace): The command-line arguments containing configuration parameters.

    Returns:
        None
    """
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.system_prompt_file is None:
        args.system_prompt_file = f"config/prompts/system_prompt_{args.mode}.txt"

    # --- Set Environment Variables ---
    hf_home = args.hf_home or os.environ.get("HF_HOME") or DEFAULT_HF_HOME
    hf_transfer = args.hf_transfer or os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") or DEFAULT_HF_TRANSFER

    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = hf_transfer

def load_input_data(input_file: str) -> pl.DataFrame:
    """
    Description:
    Loads input TSV file using Polars, raising an error if it doesn't exist.

    Args:
        input_file (str): Path to the input TSV file.

    Returns:
        pl.DataFrame: The loaded dataset as a Polars DataFrame.
    """
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        logger.error(f"Current working directory: {os.getcwd()}")
        raise FileNotFoundError(f"Input file not found: {input_file}")
    return pl.read_csv(input_file, separator='\t', ignore_errors=True, quote_char=None)

def init_llm(args: argparse.Namespace) -> Tuple[Any, int]:
    """
    Description:
    Initializes and returns the LLM model and test retries config.

    Args:
        args (argparse.Namespace): The command-line arguments containing configuration.

    Returns:
        Tuple[Any, int]: A tuple containing the initialized LLM client object and the maximum number of retries.
    """
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

    args.max_concurrency = max_concurrency
    return llm, test_retries

def extract_rag_params(rag_config: dict) -> dict:
    """
    Description:
    Extracts RAG and Reranker parameters from the config block.

    Args:
        rag_config (dict): The loaded RAG configuration dictionary.

    Returns:
        dict: A dictionary of RAG parameters including embedding and reranker model configurations.
    """
    return {
        "embedding_model": rag_config.get("embedding_model", "Qwen/Qwen3-Embedding-8B"),
        "top_k_rag": rag_config.get("top_k_rag", 15),
        "threshold_rag": rag_config.get("threshold_rag", 0.5),
        "reranker_model": rag_config.get("reranker_model", "Qwen/Qwen3-Reranker-8B"),
        "threshold_reranker": rag_config.get("threshold_reranker", 0.0),
        "top_k_reranker": rag_config.get("top_k_reranker"),
        "reranker_batch_size": rag_config.get("reranker_batch_size", 32),
        "reranker_prompt_file": rag_config.get("reranker_prompt"),
        "grf_alpha": rag_config.get("grf")
    }

def init_graph(args: argparse.Namespace, categories: list, test_retries: int, rag_params: dict) -> TextToOntoGraph:
    """
    Description:
    Initializes the TextToOntoGraph pipeline object.

    Args:
        args (argparse.Namespace): The command-line arguments configuration.
        categories (list): List of category names from the ontology.
        test_retries (int): Number of retries for testing connectivity.
        rag_params (dict): Dictionary containing the extracted RAG parameters.

    Returns:
        TextToOntoGraph: The initialized TextToOntoGraph pipeline instance.
    """
    ontology_db_path = args.ontology_embedding_db
    if not os.path.exists(ontology_db_path):
        logger.warning(f"Ontology DB not found at {ontology_db_path}. Please ensure you ran process_ontology or check the path.")

    return TextToOntoGraph(
        categories,
        ontology_db=ontology_db_path,
        embedding_model_name=rag_params["embedding_model"],
        input_db=args.target_embedding_db,
        top_k_rag=rag_params["top_k_rag"],
        system_prompt_path=args.system_prompt_file,
        threshold_rag=rag_params["threshold_rag"],
        reranker_model_name=rag_params["reranker_model"],
        threshold_reranker=rag_params["threshold_reranker"],
        mode=args.mode,
        consensus_runs=args.consensus_runs,
        ontology_definitions=args.ontology_definitions,
        max_concurrency=args.max_concurrency,
        max_retries=test_retries,
        context_ontology=args.context_ontology,
        reranker_prompt_file=rag_params["reranker_prompt_file"],
        reranker_batch_size=rag_params["reranker_batch_size"],
        global_definitions_db=args.global_definitions_db,
        top_k_reranker=rag_params["top_k_reranker"]
    )


def setup_target_schema_and_embeddings(
    df: pl.DataFrame,
    categories: list,
    target_column: str,
    embedding_model: str,
    target_embedding_db: str
) -> dict:
    """
    Description:
    Determines target schema and ensures embeddings exist for the target column.

    Args:
        df (pl.DataFrame): Input dataset DataFrame.
        categories (list): List of ontology categories.
        target_column (str): Name of the column containing the description.
        embedding_model (str): Name or identifier of the embedding model.
        target_embedding_db (str): Path to the target embedding database.

    Returns:
        dict: A dictionary mapping column names to their Polars types representing the output schema.
    """
    target_schema = dict(df.schema)
    for cat in categories:
        target_schema[cat] = pl.Float64

    unique_descs = df.select(target_column).unique().drop_nulls().to_series().to_list()
    unique_descs = [d for d in unique_descs if is_valid_description(d)]
    logger.info(f"Checking embeddings for target column '{target_column}' ({len(unique_descs)} unique texts)...")
    get_or_create_embeddings(unique_descs, embedding_model, target_embedding_db, return_vectors=False)

    return target_schema


def run_inference(
    args: argparse.Namespace,
    df: pl.DataFrame,
    categories: list,
    target_schema: dict,
    graph: TextToOntoGraph,
    llm: Any,
    grf_alpha: Optional[float]
) -> None:
    """
    Description:
    Executes the concurrent description classification flow.

    Args:
        args (argparse.Namespace): Command-line arguments namespace.
        df (pl.DataFrame): Input dataset DataFrame.
        categories (list): List of ontology categories.
        target_schema (dict): The output schema mapping.
        graph (TextToOntoGraph): The ontology graph pipeline helper.
        llm (Any): The initialized LLM client instance.
        grf_alpha (Optional[float]): Optional alpha parameter for GRF.

    Returns:
        None
    """
    unique_descriptions, seen_descriptions = get_unique_descriptions(df, args.target_column, args.debug_rows)

    description_cache = {}
    total_descs = len(unique_descriptions)
    num_backups = args.num_backups
    backup_interval = (total_descs + num_backups - 1) // num_backups if num_backups > 0 else total_descs + 1
    backup_count = 1

    def process_description(desc: str) -> Tuple[str, dict, str, list]:
        """
        Description:
        Processes a single description through the graph pipeline.

        Args:
            desc (str): The description text to process.

        Returns:
            Tuple[str, dict, str, list]: A tuple of (description, row result values, reasoning, selected categories).
        """
        max_attempts = 3
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                row_result_values, reasoning, selected = graph.execute(llm, desc, alpha=grf_alpha)
                return desc, row_result_values, reasoning, selected
            except Exception as e:
                last_err = e
                logger.warning(f"Attempt {attempt}/{max_attempts} failed for description '{desc}': {e}")
                if attempt < max_attempts:
                    time.sleep(1.0)
        raise RuntimeError(f"All {max_attempts} attempts failed for description '{desc}'. Last error: {last_err}")


    with ThreadPoolExecutor(max_workers=args.descriptions_concurrency) as executor:
        futures = {executor.submit(process_description, desc): desc for desc in unique_descriptions}

        for idx, future in enumerate(tqdm(as_completed(futures), total=total_descs, smoothing=0.0)):
            desc = futures[future]
            try:
                desc_res, row_result_values, reasoning, selected = future.result()
                description_cache[desc_res] = (row_result_values, reasoning, selected)
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
                        backup_count += 1

    if num_backups > 0:
        merge_backups(args.output_file, args.reasoning_file, backup_count,
                    categories, args.target_column, target_schema)
    else:
        logger.info("Applying results to rows and saving...")
        save_backup_chunk(df, description_cache, 0, categories, args.target_column,
                         args.output_file, args.reasoning_file, target_schema,
                         is_final=True, debug_rows=args.debug_rows, seen_descriptions=seen_descriptions)
        # Rename backup 0 to final safely using Path
        out_path = Path(args.output_file)
        reason_path = Path(args.reasoning_file)
        
        backup_out_file = out_path.with_name(f"{out_path.stem}_0{out_path.suffix}")
        backup_reason_file = reason_path.with_name(f"{reason_path.stem}_0{reason_path.suffix}")
        
        os.rename(backup_out_file, out_path)
        os.rename(backup_reason_file, reason_path)

def main() -> None:
    """
    Description:
    The main execution entry point for the text-to-ontology script.

    Args:
        None

    Returns:
        None
    """
    args = arg_parser()

    rag_config = load_rag_config(args)

    load_llm_config(args)

    init_env(args)

    start_time = time.time()

    df = load_input_data(args.input_file)

    rag_params = extract_rag_params(rag_config)

    llm, test_retries = init_llm(args)

    prefix = f"{Path(args.ontology_embedding_db).name.split('.')[0].upper()}:"
    logger.info(f"Using ontology prefix: {prefix}")
    nodes, categories = load_ontology_nodes(args.ontology_embedding_db, prefix=prefix)

    target_schema = setup_target_schema_and_embeddings(
        df, categories, args.target_column, 
        rag_params["embedding_model"], args.target_embedding_db)

    graph = init_graph(args, categories, test_retries, rag_params)

    try:
        run_inference(args, df, categories, target_schema, graph, llm, rag_params["grf_alpha"])
    finally:
        graph.close()

    logger.info(f"Total execution time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
