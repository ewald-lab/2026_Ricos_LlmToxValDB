from __future__ import annotations
import argparse
import json
import os
import polars as pl

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import (setup_logging, is_valid_description, get_or_create_embeddings)

logger = setup_logging()

def arg_parser() -> argparse.Namespace:
    """
    Description:
    Parses command line arguments for the target embeddings preparation script.

    Args:
        None

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Prepare Target Embeddings")

    # --- File I/O & Columns ---
    parser.add_argument("--input_file", required=True, help="Path to input TSV file.")
    parser.add_argument("--target_column", default="TOXICOLOGICAL_EFFECT", help="Column name containing the text to process.")
    parser.add_argument("--std_column", default="TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED", help="Column name containing the standardized text.")

    # --- Embedding Configuration ---
    parser.add_argument("--rag_config", required=True, help="Path to RAG configuration JSON file.")
    parser.add_argument("--embeddings_model_id", required=True, help="Identifier for the model configuration in RAG config (e.g., qwen, nemotron)")
    parser.add_argument("--target_embedding_db", required=True, help="Path to save embedding database for target column.")

    # --- Environment ---
    parser.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), help="HuggingFace cache directory.")

    return parser.parse_args()

def main() -> None:
    """
    Description:
    Main entry point for generating embeddings of unique descriptions and standardized descriptions
    using a model specified in the RAG configuration, and saving them to the embeddings database.

    Args:
        None

    Returns:
        None
    """
    args = arg_parser()

    # Set environment variables
    os.environ["HF_HOME"] = args.hf_home
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Load RAG config
    logger.info(f"Loading RAG configuration from: {args.rag_config}")
    with open(args.rag_config, 'r', encoding='utf-8') as f:
        rag_config = json.load(f)

    # If it's a dictionary, look up by embeddings_model_id
    if isinstance(rag_config, dict):
        key = args.embeddings_model_id
        # Case-insensitive key lookup
        rag_config_lower = {k.lower(): v for k, v in rag_config.items()}
        model_config_block = rag_config_lower.get(key.lower())

        if model_config_block:
            logger.info(f"Using RAG configuration block for: {key}")
            rag_config = model_config_block
        else:
            logger.warning(f"No specific RAG configuration found for '{key}'. Using default values.")

    # Extract embedding model
    embedding_model = rag_config.get("embedding_model", "Qwen/Qwen3-Embedding-8B")

    logger.info(f"Loading data from {args.input_file}")
    # Read TSV with polars
    df = pl.read_csv(args.input_file, separator='\t', ignore_errors=True, quote_char=None)

    # Process Target Column
    logger.info(f"Processing column: {args.target_column}")
    unique_descs = df.select(args.target_column).unique().drop_nulls().to_series().to_list()
    unique_descs = [d for d in unique_descs if is_valid_description(d)]

    logger.info(f"Found {len(unique_descs)} unique descriptions to embed.")
    get_or_create_embeddings(unique_descs, embedding_model, args.target_embedding_db, return_vectors=False)

    # Process Standardized Column if it exists
    if args.std_column in df.columns:
        logger.info(f"Processing standardized column: {args.std_column}")
        unique_stds = df.select(args.std_column).unique().drop_nulls().to_series().to_list()
        unique_stds = [d for d in unique_stds if is_valid_description(d)]

        logger.info(f"Found {len(unique_stds)} unique standardized descriptions to embed.")
        get_or_create_embeddings(unique_stds, embedding_model, args.target_embedding_db, return_vectors=False)
    else:
        logger.info(f"Standardized column '{args.std_column}' not found in input file.")

if __name__ == "__main__":
    main()
