import argparse
import json
import os
import sys
import time
import polars as pl
from tqdm import tqdm

# Add scripts directory to sys.path to resolve local packages
sys.path.insert(0, os.path.dirname(__file__))

from src.utils import (
    setup_logging, is_valid_description, load_ontology_nodes
)
from model.llm import get_model, test_model
from src.graph import TextToOntoGraph

logger = setup_logging()

def arg_parser():
    parser = argparse.ArgumentParser(description="LLM Evaluation Pipeline for Average Human toxicological effects")
    
    # --- File I/O & Columns ---
    parser.add_argument("--input_file", required=True, help="Path to Expanded_average_human.csv input file.")
    parser.add_argument("--output_file", required=True, help="Path to output CSV file in expert response format.")
    parser.add_argument("--reasoning_file", required=True, help="Path to output JSON file for reasoning.")
    parser.add_argument("--target_column", default="Phrase", help="Column name containing the text phrases to process.")
    parser.add_argument("--candidate_pool_file", default=None, help="Path to the JSON file containing the candidate pool presented to the humans.")
    
    # --- LLM Configuration ---
    parser.add_argument("--model_config", required=True, help="Path to model config JSON file.")
    parser.add_argument("--base_url", help="Base URL for the LLM API. Overrides value in model_config if provided.")
    parser.add_argument("--system_prompt_file", default=None, help="Path to the system prompt text file.")
    
    # --- Execution Logic & Consensus ---
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2], help="Execution mode: 1 for individual verification, 2 for batch selection.")
    parser.add_argument("--consensus_runs", type=int, default=3, help="Number of runs for majority voting consensus.")
    parser.add_argument("--ontology_definitions", action="store_true", help="Use node definitions from DB in prompts.")
    parser.add_argument("--context_ontology", default="", help="Context information about the ontology to use in prompts.")
    
    # --- Embedding & Database Configuration ---
    parser.add_argument("--rag_config", required=False, help="Path to RAG configuration JSON file.")
    parser.add_argument("--embeddings_model_id", default=None, help="Identifier for the model configuration in RAG config (e.g., qwen, nemotron)")
    parser.add_argument("--ontology_embedding_db", required=True, help="Path to ontology embeddings database.")
    parser.add_argument("--target_embedding_db", default="", help="Path to save/load embedding database for input phrases.")
    parser.add_argument("--global_definitions_db", default=None, help="Path to global definitions database (URI to definition).")
    
    # --- Debugging & Seed ---
    parser.add_argument("--seed", type=int, default=43, help="Random seed for reproducibility.")
    
    return parser.parse_args()

def load_candidates_from_html(html_path):
    """Loads candidates from an expert annotation HTML file."""
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    start_marker = "const rawPoolData = "
    start_idx = content.find(start_marker)
    if start_idx == -1:
        raise ValueError(f"Could not find '{start_marker}' in {html_path}")
    
    start_idx += len(start_marker)
    end_idx = content.find(";\n        const ontologyNodes", start_idx)
    if end_idx == -1:
        end_idx = content.find(";const ontologyNodes", start_idx)
    if end_idx == -1:
        end_idx = content.find("const ontologyNodes", start_idx)
        while end_idx > start_idx and content[end_idx] != ";":
            end_idx -= 1
            
    if end_idx == -1:
        raise ValueError(f"Could not find ending of rawPoolData in {html_path}")
        
    raw_js = content[start_idx:end_idx].strip()
    return json.loads(raw_js)

def main():
    args = arg_parser()
    
    start_time = time.time()
    
    if args.system_prompt_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        module_dir = os.path.abspath(os.path.join(script_dir, ".."))
        args.system_prompt_file = os.path.join(
            module_dir, "config", f"system_prompt_{args.mode}.txt"
        )
        
    os.environ["HF_HOME"] = "/nfs/research/ewald/ricos/huggingface_cache"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
        
    with open(args.model_config, 'r', encoding='utf-8') as f:
        model_config = json.load(f)
        
    max_concurrency = model_config.pop("max_concurrency", 10)
    if args.base_url:
        model_config["base_url"] = args.base_url
        logger.info(f"Overriding base_url from CLI: {args.base_url}")
        
    logger.info(f"Loading model configuration from: {args.model_config}")
    llm = get_model(**model_config)
    
    test_retries = model_config.get('max_retries', 3)
    if not test_model(llm, max_retries=test_retries):
        raise RuntimeError("Model connectivity test failed. Please check your configuration.")
        
    ontology_db_path = args.ontology_embedding_db
    nodes, categories = load_ontology_nodes(ontology_db_path)
    
    try:
        logger.info(f"Loading evaluation data from {args.input_file}")
        df = pl.read_csv(args.input_file, ignore_errors=True)
        
        phrase_to_candidates = {}
        if args.candidate_pool_file:
            logger.info(f"Loading candidate pool from {args.candidate_pool_file}")
            if args.candidate_pool_file.endswith(".html"):
                pool_data = load_candidates_from_html(args.candidate_pool_file)
            else:
                with open(args.candidate_pool_file, 'r', encoding='utf-8') as f:
                    pool_data = json.load(f)
            
            for item in pool_data:
                p_text = item.get("phrase")
                if p_text:
                    cands = item.get("candidates", [])
                    phrase_to_candidates[p_text.lower().strip()] = [c["name"] for c in cands]
        
        graph = TextToOntoGraph(
            categories,
            ontology_db=ontology_db_path,
            system_prompt_path=args.system_prompt_file,
            mode=args.mode,
            consensus_runs=args.consensus_runs,
            ontology_definitions=args.ontology_definitions,
            max_concurrency=max_concurrency,
            max_retries=test_retries,
            context_ontology=args.context_ontology,
            global_definitions_db=args.global_definitions_db
        )
        
        gpu_id_str = f"GPU:{os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}"
        
        results_list = []
        reasoning_cache = {}
        phrase_avg_tokens = []
        
        # Iterate over each row and match precisely
        for row in tqdm(df.iter_rows(named=True), total=df.height, desc="Worker"):
            index = row.get("Index")
            phrase = row.get(args.target_column)
            
            if not is_valid_description(phrase):
                # Empty or invalid phrase
                results_list.append({
                    "Index": index,
                    "Phrase": phrase,
                    "Selected_Candidates": "None",
                    "Selected_URIs": "None",
                    "Manual_Terms": "None",
                    "Toxicological_Effect_Does_Not_Make_Sense": "0",
                    "Not_My_Expertise": "0",
                    "Comments": ""
                })
                continue
                
            try:
                logger.info(f"\n--------------------------------------------------------")
                logger.info(f"Evaluating Phrase: {phrase}")
                
                # Execute graph logic
                candidates_list = None
                if args.candidate_pool_file:
                    candidates_list = phrase_to_candidates.get(phrase.lower().strip())
                    if candidates_list is None:
                        logger.warning(f"Phrase '{phrase}' not found in candidate pool!")
                
                row_result_values, reasoning, selected, token_usage = graph.execute(llm, phrase, direct_candidates=candidates_list)
                
                # Extract URIs
                selected_uris = []
                for term in selected:
                    node_info = graph.nodes.get(term, {})
                    uri = node_info.get('uri', 'Unknown')
                    selected_uris.append(str(uri))
                    
                selected_candidates_str = "|".join(selected) if selected else "None"
                selected_uris_str = "|".join(selected_uris) if selected_uris else "None"
                
                results_list.append({
                    "Index": index,
                    "Phrase": phrase,
                    "Selected_Candidates": selected_candidates_str,
                    "Selected_URIs": selected_uris_str,
                    "Manual_Terms": "None",
                    "Toxicological_Effect_Does_Not_Make_Sense": "0",
                    "Not_My_Expertise": "0",
                    "Comments": ""
                })
                
                reasoning_cache[phrase] = {
                    "reasoning": reasoning,
                    "selected": selected
                }
                phrase_avg_tokens.append(token_usage.get("avg_completion_tokens", 0.0))
                
            except Exception as e:
                logger.error(f"Error processing phrase '{phrase}': {e}")
                results_list.append({
                    "Index": index,
                    "Phrase": phrase,
                    "Selected_Candidates": "None",
                    "Selected_URIs": "None",
                    "Manual_Terms": "None",
                    "Toxicological_Effect_Does_Not_Make_Sense": "0",
                    "Not_My_Expertise": "0",
                    "Comments": f"Error during processing: {str(e)}"
                })
                
        # Save output CSV in expert format
        output_df = pl.DataFrame(results_list)
        # Ensure column order matches the expert file
        cols_order = [
            "Index", "Phrase", "Selected_Candidates", "Selected_URIs", 
            "Manual_Terms", "Toxicological_Effect_Does_Not_Make_Sense", 
            "Not_My_Expertise", "Comments"
        ]
        output_df = output_df.select(cols_order)
        
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        output_df.write_csv(args.output_file)
        logger.info(f"Saved expert format LLM responses to {args.output_file}")
        
        # Save reasoning JSON
        os.makedirs(os.path.dirname(args.reasoning_file), exist_ok=True)
        with open(args.reasoning_file, 'w', encoding='utf-8') as f:
            json.dump(reasoning_cache, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved reasoning to {args.reasoning_file}")
        
        # Calculate and save single mean and std to a separate txt file with 2 decimal places
        if phrase_avg_tokens:
            import numpy as np
            mean_tokens = np.mean(phrase_avg_tokens)
            std_tokens = np.std(phrase_avg_tokens)
        else:
            mean_tokens = 0.0
            std_tokens = 0.0

        token_output_file = args.reasoning_file.replace("_reasoning.json", "_tokens.txt")
        with open(token_output_file, 'w', encoding='utf-8') as tf:
            tf.write(f"Mean: {mean_tokens:.2f}\n")
            tf.write(f"Std: {std_tokens:.2f}\n")
        logger.info(f"Saved token stats (Mean: {mean_tokens:.2f}, Std: {std_tokens:.2f}) to {token_output_file}")
        
        logger.info(f"Total execution time: {time.time() - start_time:.2f}s")
        
    finally:
        if 'graph' in locals():
            graph.close()

if __name__ == "__main__":
    main()
