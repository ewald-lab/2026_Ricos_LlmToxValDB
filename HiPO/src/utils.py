from __future__ import annotations

import logging
import sys
import os
import sqlite3
import torch
import polars as pl
import json
from typing import Any, Dict, List, Optional, Set, Tuple

from pathlib import Path
from tqdm import tqdm

def setup_logging(level: int = logging.INFO, stream: Any = sys.stdout, fmt: str = "%(message)s") -> logging.Logger:
    """
    Description:
    Configures global logging and suppresses noisy third-party loggers.

    Args:
        level (int): The logging level to set (e.g. logging.INFO). Defaults to logging.INFO.
        stream (Any): The output stream to log to. Defaults to sys.stdout.
        fmt (str): The logging format string. Defaults to "%(message)s".

    Returns:
        logging.Logger: The configured root-level or module-specific Logger instance.
    """
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

def is_valid_description(description: str) -> bool:
    """
    Description:
    Checks if a string is a valid, non-empty description.

    Args:
        description (str): The text description to validate.

    Returns:
        bool: True if the description is a valid non-empty string, False otherwise.
    """
    return description is not None and isinstance(description, str) and len(description.strip()) > 0

def load_ontology_embeddings(db_path: str) -> List[Dict]:
    """
    Description:
    Loads all ontology embeddings into memory from the database.

    Args:
        db_path (str): The path to the SQLite embeddings database.

    Returns:
        List[Dict]: A list of dictionaries representing the loaded ontology node embeddings.
    """
    if not os.path.exists(db_path):
        logging.getLogger(__name__).error(f"Ontology embeddings DB not found at {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT uri, name, sanitized_name, parent, vector FROM embeddings")
    rows = cursor.fetchall()
    conn.close()

    ontology_nodes = []
    for uri, name, sanitized_name, parent, vector in rows:
        parents = json.loads(parent) if parent and parent.startswith('[') else ([parent] if parent else [])
        ontology_nodes.append({
            'uri': uri,
            'name': name,
            'sanitized_name': sanitized_name,
            'def': '',
            'parents': parents,
            'is_llm_generated': 0,
            'vector': torch.frombuffer(bytearray(vector), dtype=torch.float32),
        })
    return ontology_nodes

def load_ontology_nodes(db_path: str, prefix: str = "") -> Tuple[dict, list]:
    """
    Description:
    Loads ontology node metadata from the embeddings DB.

    Args:
        db_path (str): The path to the SQLite embeddings database.
        prefix (str): Prefix to prepend to node names. Defaults to "".

    Returns:
        Tuple[dict, list]: A tuple containing a dict of ontology node metadata and a list of category names.
    """
    if not os.path.exists(db_path):
        logging.getLogger(__name__).error(f"Ontology DB not found at {db_path}")
        return {}, []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, sanitized_name, parent FROM embeddings")
    rows = cursor.fetchall()
    conn.close()

    nodes = {}
    categories = []
    for name, sanitized_name, parent in rows:
        parents = json.loads(parent) if parent and parent.startswith('[') else ([parent] if parent else [])
        prefixed_name = f"{prefix}{name}"
        nodes[name] = {
            'sanitized_name': sanitized_name,
            'parents': parents,
            'definition': '',
            'prefixed_name': prefixed_name
        }
        categories.append(prefixed_name)

    return nodes, categories

def get_or_create_embeddings(texts: List[str], model_name: str, db_path: str, return_vectors: bool = True) -> Dict[str, torch.Tensor]:
    """
    Description:
    Ensures embeddings exist in the DB for the given texts. Returns {text: vector} if return_vectors is True.

    Args:
        texts (List[str]): List of texts to get or generate embeddings for.
        model_name (str): Name/path of the embedding model to use.
        db_path (str): Path to the SQLite embeddings database.
        return_vectors (bool): If True, returns a dictionary mapping text to its embedding vector. Defaults to True.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping each text to its embedding tensor, or empty dict if return_vectors is False.
    """
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

def get_unique_descriptions(df: pl.DataFrame, target_column: str, debug_rows: Optional[List[int]] = None) -> Tuple[List[str], Set[str]]:
    """
    Description:
    Extracts unique valid descriptions to process based on debug selection.

    Args:
        df (pl.DataFrame): Input dataset DataFrame.
        target_column (str): Name of the column containing descriptions.
        debug_rows (Optional[List[int]]): Selected row indices for debugging. Defaults to None.

    Returns:
        Tuple[List[str], Set[str]]: A tuple containing the list of unique descriptions and a set of seen descriptions.
    """
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
                    schema: Dict, is_final: bool = False, debug_rows: Optional[List[int]] = None, seen_descriptions: Optional[Set[str]] = None) -> None:
    """
    Description:
    Saves a processed chunk of descriptions and its reasoning to disk using efficient Polars joins.

    Args:
        df (pl.DataFrame): Input dataset DataFrame.
        cache_chunk (Dict): Dictionary cache of processed results mapping description to (hits, trace, selected).
        backup_idx (int): The index of the backup chunk file to save.
        categories (List[str]): List of category columns.
        target_column (str): Column name containing raw text descriptions.
        output_file (str): Base output CSV file path.
        reasoning_file (str): Base reasoning JSON file path.
        schema (Dict): Output DataFrame schema dictionary mapping column names to types.
        is_final (bool): Whether this is the final chunk save. Defaults to False.
        debug_rows (Optional[List[int]]): Row indices being processed for debugging. Defaults to None.
        seen_descriptions (Optional[Set[str]]): Set of seen description strings. Defaults to None.

    Returns:
        None
    """
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

        update_exprs = []
        for cat in categories:
            right_cat = f"{cat}_right"
            if right_cat in chunk_df.columns:
                update_exprs.append(
                    pl.when(pl.col(right_cat).is_not_null())
                      .then(pl.col(right_cat))
                      .otherwise(pl.col(cat))
                      .alias(cat)
                )
            elif cat in work_df.columns:
                continue
            elif cat in results_df.columns:
                update_exprs.append(pl.col(cat).cast(pl.Float64))
            else:
                update_exprs.append(pl.lit(None).alias(cat).cast(pl.Float64))

        if update_exprs:
            chunk_df = chunk_df.with_columns(update_exprs)
    else:
        if is_final:
            new_cols = [pl.lit(None).alias(cat).cast(pl.Float64) for cat in categories if cat not in work_df.columns]
            chunk_df = work_df.with_columns(new_cols) if new_cols else work_df
        else:
            return

    # Sort by original index to preserve order
    chunk_df = chunk_df.sort("orig_index")

    # Ensure correct column order and schema
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
                  target_column: str, schema: Dict) -> None:
    """
    Description:
    Merges all backup CSV and JSON files into the final outputs.

    Args:
        output_file (str): The final merged output CSV file path.
        reasoning_file (str): The final merged reasoning JSON file path.
        backup_count (int): Total number of backup chunk files saved.
        categories (List[str]): List of ontology categories.
        target_column (str): Name of the raw target column.
        schema (Dict): Target DataFrame schema mapping.

    Returns:
        None
    """
    import csv
    import heapq
    logger = logging.getLogger(__name__)
    logger.info("Merging backups to create final results using streaming K-way merge...")

    # Merge CSVs
    backup_csv_files = [output_file.replace('.csv', f'_{i}.csv') for i in range(1, backup_count + 1)]
    files = []
    readers = []
    for f_path in backup_csv_files:
        if os.path.exists(f_path):
            f = open(f_path, 'r', encoding='utf-8')
            files.append(f)
            readers.append(csv.reader(f))

    if readers:
        headers = []
        for r in readers:
            headers.append(next(r))

        # Find the index of orig_index in the header
        try:
            orig_idx_col = headers[0].index("orig_index")
        except ValueError:
            logger.error("Could not find 'orig_index' in backup CSV header.")
            orig_idx_col = -1

        # The final header should exclude orig_index
        if orig_idx_col == -1 or orig_idx_col == len(headers[0]) - 1:
            final_header = headers[0][:-1]
        else:
            final_header = headers[0][:orig_idx_col] + headers[0][orig_idx_col+1:]

        logger.info(f"Creating final output file: {output_file}")
        out_f = open(output_file, 'w', encoding='utf-8', newline='')
        writer = csv.writer(out_f)
        writer.writerow(final_header)

        def get_next_valid_row(reader, file_idx):
            while True:
                try:
                    row = next(reader)
                    if not row or len(row) == 0:
                        continue
                    if all(val.strip() == '' for val in row):
                        continue
                    
                    orig_idx_str = row[orig_idx_col].strip()
                    if not orig_idx_str:
                        continue
                    try:
                        return int(orig_idx_str), row
                    except ValueError:
                        continue
                except StopIteration:
                    return None

        # Initialize heap
        heap = []
        for file_idx, reader in enumerate(readers):
            res = get_next_valid_row(reader, file_idx)
            if res is not None:
                orig_index, row = res
                heapq.heappush(heap, (orig_index, file_idx, row))

        count = 0
        while heap:
            orig_index, file_idx, row = heapq.heappop(heap)
            
            # Reconstruct the row excluding orig_index
            if orig_idx_col == -1 or orig_idx_col == len(row) - 1:
                cleaned_row = row[:-1]
            else:
                cleaned_row = row[:orig_idx_col] + row[orig_idx_col+1:]
                
            writer.writerow(cleaned_row)
            count += 1
            if count % 50000 == 0:
                logger.info(f"Merged {count} rows...")
                
            res = get_next_valid_row(readers[file_idx], file_idx)
            if res is not None:
                next_orig_index, next_row = res
                heapq.heappush(heap, (next_orig_index, file_idx, next_row))

        for f in files:
            f.close()
        out_f.close()
        logger.info(f"Finished merging {count} CSV rows.")

        # Cleanup backups
        for f in backup_csv_files:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
    else:
        logger.warning("No CSV backup files found to merge.")

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
