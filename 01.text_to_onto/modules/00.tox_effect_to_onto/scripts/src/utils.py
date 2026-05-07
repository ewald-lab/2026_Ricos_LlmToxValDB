import logging
import re
import sys
import os
import sqlite3
import torch
import polars as pl
import json
import random
from typing import List, Dict, Optional, Set, Tuple

from pathlib import Path
from tqdm import tqdm

def setup_logging(level=logging.INFO, stream=sys.stdout, fmt="%(message)s"):
    """Configures global logging and suppresses noisy third-party loggers."""
    logging.basicConfig(stream=stream, level=level, format=fmt)

    for name, lvl in {
        "transformers": logging.ERROR,
        "huggingface_hub": logging.ERROR,
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "urllib3": logging.WARNING,
    }.items():
        logging.getLogger(name).setLevel(lvl)

    return logging.getLogger(__name__)

def sanitize_name(name: str) -> str:
    """Creates a safe identifier for nodes by sanitizing a given name."""
    name = re.sub(r'\([^)]*\)|\[[^]]*\]', '', name)
    clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', name.lower())
    return re.sub(r'\s+', ' ', clean).strip()

def is_valid_description(description: str) -> bool:
    """Checks if a string is a valid, non-empty description."""
    return description is not None and isinstance(description, str) and len(description.strip()) > 0

def load_ontology_embeddings(db_path: str) -> List[Dict]:
    """Loads all ontology embeddings into memory."""
    if not os.path.exists(db_path):
        logging.getLogger(__name__).error(f"Ontology embeddings DB not found at {db_path}")
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT uri, name, sanitized_name, definition, parent, is_llm_generated, vector FROM embeddings")
    rows = cursor.fetchall()
    conn.close()
    
    return [{'uri': r[0], 'name': r[1], 'sanitized_name': r[2], 'def': r[3], 'parent': r[4], 'is_llm_generated': r[5], 'vector': torch.frombuffer(bytearray(r[6]), dtype=torch.float32)} for r in rows]

def load_ontology_nodes(db_path: str, prefix: str = "") -> tuple:
    """Loads ontology node metadata from the embeddings DB.
    
    Returns:
        nodes (dict): {name: {'sanitized_name': ..., 'parent': ..., 'definition': ...}}
        categories (list): list of all node names (possibly prefixed)
    """
    if not os.path.exists(db_path):
        logging.getLogger(__name__).error(f"Ontology DB not found at {db_path}")
        return {}, []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, sanitized_name, definition, parent FROM embeddings")
    rows = cursor.fetchall()
    conn.close()
    
    nodes = {}
    categories = []
    for name, sanitized_name, definition, parent in rows:
        prefixed_name = f"{prefix}{name}"
        nodes[name] = {
            'sanitized_name': sanitized_name,
            'parent': parent,
            'definition': definition or '',
            'prefixed_name': prefixed_name
        }
        categories.append(prefixed_name)
    
    return nodes, categories

def get_or_create_embeddings(texts: List[str], model_name: str, db_path: str, return_vectors: bool = True) -> Dict[str, torch.Tensor]:
    """Ensures embeddings exist in the DB for the given texts. Returns {text: vector} if return_vectors is True."""
    from model.embeddings import EmbeddingModel
    
    logger = setup_logging()
    path_obj = Path(db_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS input_embeddings (text TEXT PRIMARY KEY, vector BLOB)')
    conn.commit()
    
    text_to_vec = {}
    needed_texts = set(texts)
    
    if return_vectors:
        cursor.execute("SELECT text, vector FROM input_embeddings")
        for row in cursor:
            if row[0] in needed_texts:
                text_to_vec[row[0]] = torch.frombuffer(bytearray(row[1]), dtype=torch.float32)
        existing_texts = set(text_to_vec.keys())
    else:
        cursor.execute("SELECT text FROM input_embeddings")
        existing_texts = set(row[0] for row in cursor)
        
    missing = [t for t in texts if t not in existing_texts]
    
    if missing:
        logger.info(f"Generating embeddings for {len(missing)} missing texts...")
        embedder = EmbeddingModel(model_name)
        for text in tqdm(missing, desc="Embedding", smoothing=0.0):
            vec_bytes = embedder.get_embedding(text)
            if return_vectors:
                text_to_vec[text] = torch.frombuffer(bytearray(vec_bytes), dtype=torch.float32)
            cursor.execute("INSERT OR REPLACE INTO input_embeddings (text, vector) VALUES (?, ?)", (text, vec_bytes))
        conn.commit()
        
        del embedder.model
        del embedder.tokenizer
        del embedder
        torch.cuda.empty_cache()
    else:
        logger.info(f"All {len(texts)} texts already have embeddings in the DB.")
    
    conn.close()
    return text_to_vec if return_vectors else {}

def get_debug_rows(df: pl.DataFrame, target_column: str, random_debug_rows: Optional[int] = None, debug_rows: Optional[List[int]] = None, seed: int = 43) -> List[int]:
    """Handles the logic of selecting unique toxic effects for debugging."""
    logger = logging.getLogger(__name__)
    random.seed(seed)
    
    # 1. First, select rows based on random count if requested
    if random_debug_rows is not None and debug_rows is None:
        valid_df = df.with_row_index("orig_index").filter(
            pl.col(target_column).map_elements(is_valid_description, return_dtype=pl.Boolean)
        )
        unique_effects_df = valid_df.group_by(target_column, maintain_order=True).agg(pl.col("orig_index").first())
        
        total_unique = unique_effects_df.height
        if random_debug_rows >= total_unique:
            result_rows = sorted(unique_effects_df["orig_index"].to_list())
        else:
            result_rows = sorted(random.sample(unique_effects_df["orig_index"].to_list(), random_debug_rows))
        total_sampled = len(result_rows)
        if random_debug_rows:
            logger.info(f"Selected {total_sampled} unique random toxic effects for debugging.")
    else:
        result_rows = debug_rows or []

    # 2. Ensure the final list only includes the FIRST occurrence of each unique toxic effect among candidates
    if result_rows:
        debug_df = df.select(target_column).with_row_index("orig_index").filter(pl.col("orig_index").is_in(result_rows))
        final_rows = sorted(debug_df.group_by(target_column, maintain_order=True).agg(pl.col("orig_index").first())["orig_index"].to_list())
        
        # If filtering for uniqueness changed the count, we log it, otherwise one log is enough
        if len(final_rows) != len(result_rows) or not random_debug_rows:
            logger.info(f"Final debug selection: {len(final_rows)} unique toxic effects.")
        return final_rows
        
    return []

def get_unique_descriptions(df: pl.DataFrame, target_column: str, debug_rows: Optional[List[int]] = None) -> Tuple[List[str], Set[str]]:
    """Extracts unique valid descriptions to process based on debug selection."""
    logger = logging.getLogger(__name__)
    logger.info("Collecting unique descriptions...")
    unique_descriptions = []
    seen_descriptions = set()

    if debug_rows:
        for i, row in enumerate(df.iter_rows(named=True)):
            if i in debug_rows:
                desc = row.get(target_column)
                if is_valid_description(desc) and desc not in seen_descriptions:
                    unique_descriptions.append(desc)
                    seen_descriptions.add(desc)
    else:
        try:
            descs = df.select(pl.col(target_column)).unique(maintain_order=True).drop_nulls()
            for desc in descs[target_column].to_list():
                if is_valid_description(desc) and desc not in seen_descriptions:
                    unique_descriptions.append(desc)
                    seen_descriptions.add(desc)
        except Exception as e:
            logger.warning(f"Polars unique extraction failed, falling back to iteration: {e}")
            for row in df.iter_rows(named=True):
                desc = row.get(target_column)
                if is_valid_description(desc) and desc not in seen_descriptions:
                    unique_descriptions.append(desc)
                    seen_descriptions.add(desc)

    logger.info(f"Found {len(unique_descriptions)} unique descriptions to process.")
    return unique_descriptions, seen_descriptions

def save_backup_chunk(df: pl.DataFrame, cache_chunk: Dict, backup_idx: int, 
                    categories: List[str], target_column: str, 
                    output_file: str, reasoning_file: str, 
                    schema: Dict, is_final: bool = False, debug_rows: Optional[List[int]] = None, seen_descriptions: Optional[Set[str]] = None):
    """Saves a processed chunk of descriptions and its reasoning to disk using efficient Polars joins."""
    if not cache_chunk and not is_final: return
    logger = logging.getLogger(__name__)
    
    # 1. Create a Results DataFrame from the unique descriptions processed
    # Format: {description: (hits_dict, trace)}
    results_list = []
    chunk_reasoning = {}
    
    for desc, (hits, trace, selected) in cache_chunk.items():
        row_data = {target_column: desc}
        row_data.update(hits)
        results_list.append(row_data)
        # Store reasoning and selected nodes once per unique description
        chunk_reasoning[desc] = {"reasoning": trace, "selected": selected}

    if not results_list and not is_final: return

    # 2. Add row index to original DF to track order and filter relevant rows
    work_df = df.with_row_index("orig_index")
    if debug_rows:
        work_df = work_df.filter(pl.col("orig_index").is_in(debug_rows))

    if results_list:
        # Create a tiny DataFrame for joining
        results_df = pl.DataFrame(results_list)
        
        # Ensure results_df uses Float64 for category columns
        results_df = results_df.with_columns([
            pl.col(cat).cast(pl.Float64) for cat in categories if cat in results_df.columns
        ])

        # Join with the work_df
        # Use left join logic to identify overlaps
        chunk_df = work_df.join(results_df, on=target_column, how="inner")
        
        # Update category columns: 
        # If LLM found it (1), set to 1. 
        # If LLM didn't find it, keep original value from work_df.
        # If column didn't exist in work_df, fill with NaN based on LLM result.
        update_exprs = []
        for cat in categories:
            right_cat = f"{cat}_right"
            if right_cat in chunk_df.columns:
                # Exists in both: 1 from LLM overrides anything. 
                # Otherwise keep original (NaN)
                update_exprs.append(
                    pl.when(pl.col(right_cat).is_not_null())
                      .then(pl.col(right_cat))
                      .otherwise(pl.col(cat))
                      .alias(cat)
                )
            elif cat in work_df.columns:
                # Only in work_df: keep as is
                continue
            elif cat in results_df.columns:
                # Only in results_df (wasn't in work_df): already named 'cat' in chunk_df
                # Keep nulls as is (NaN) instead of filling with 0
                update_exprs.append(pl.col(cat).cast(pl.Float64))
            else:
                # In neither: initialize to NaN (since it's a new column and no hits)
                update_exprs.append(pl.lit(None).alias(cat).cast(pl.Float64))
        
        if update_exprs:
            chunk_df = chunk_df.with_columns(update_exprs)
    else:
        # Standard final fallback or empty chunk
        if is_final:
            # Initialize missing categories to NaN if they don't exist
            new_cols = [pl.lit(None).alias(cat).cast(pl.Float64) for cat in categories if cat not in work_df.columns]
            chunk_df = work_df.with_columns(new_cols) if new_cols else work_df
        else:
            return

    # Sort by original index to preserve order
    chunk_df = chunk_df.sort("orig_index")
    
    # Ensure correct column order and schema
    # We KEEP orig_index for the merge later
    final_cols = [col for col in schema.keys() if col in chunk_df.columns]
    if "orig_index" in chunk_df.columns and "orig_index" not in final_cols:
        final_cols.append("orig_index")
    chunk_df = chunk_df.select(final_cols)

    # Optional aggregation
    non_group_cols = set(categories) | {target_column}
    group_cols = [c for c in chunk_df.columns if c not in non_group_cols]
    if group_cols:
        chunk_df = chunk_df.group_by(group_cols, maintain_order=True).agg(
            [pl.col(target_column).str.join("|")] +
            [pl.col(c).max() for c in categories]
        )
    
    out_csv = output_file.replace('.csv', f'_{backup_idx}.csv')
    chunk_df.write_csv(out_csv)
    logger.info(f"Saved backup chunk {backup_idx} to {out_csv}")
    
    out_json = reasoning_file.replace('.json', f'_{backup_idx}.json')
    if chunk_reasoning:
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(chunk_reasoning, f, indent=2, ensure_ascii=False)

def merge_backups(output_file: str, reasoning_file: str, 
                  backup_count: int, categories: List[str], 
                  target_column: str, schema: Dict):
    """Merges all backup CSV and JSON files into the final outputs."""
    logger = logging.getLogger(__name__)
    logger.info("Merging backups to create final results...")
    
    # Merge CSVs
    backup_csv_files = [output_file.replace('.csv', f'_{i}.csv') for i in range(1, backup_count + 1)]
    dfs = []
    for f in backup_csv_files:
        if os.path.exists(f):
            # Read only relevant columns to save memory if needed, but schema_overrides is good
            dfs.append(pl.read_csv(f, schema_overrides=schema, ignore_errors=True))
    
    if dfs:
        final_df = pl.concat(dfs)
        
        # Sort ENTIRE dataframe by original index to restore strict input order
        if "orig_index" in final_df.columns:
            final_df = final_df.sort("orig_index")
            
        non_group_cols = set(categories) | {target_column} | {"orig_index"}
        group_cols = [c for c in final_df.columns if c not in non_group_cols]
        if group_cols:
            final_df = final_df.group_by(group_cols, maintain_order=True).agg(
                [pl.col(target_column).str.join("|")] +
                [pl.col(c).max() for c in categories]
            )
            
        # Drop orig_index before final write
        if "orig_index" in final_df.columns:
            final_df = final_df.drop("orig_index")
            
        logger.info(f"Saving final merged results to {output_file}")
        final_df.write_csv(output_file)
        
        # Cleanup backups
        for f in backup_csv_files if os.path.exists(f) else []: 
            try: os.remove(f) 
            except: pass
    
    # Merge JSON reasoning
    merged_reasoning = {}
    json_backups = [reasoning_file.replace('.json', f'_{i}.json') for i in range(1, backup_count + 1)]
    for jf in json_backups:
        if os.path.exists(jf):
            with open(jf, 'r', encoding='utf-8') as f:
                merged_reasoning.update(json.load(f))
            try: os.remove(jf)
            except: pass
    
    logger.info(f"Saving final merged reasoning to {reasoning_file}")
    with open(reasoning_file, 'w', encoding='utf-8') as f:
        json.dump(merged_reasoning, f, indent=2, ensure_ascii=False)
