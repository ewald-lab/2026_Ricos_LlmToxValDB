from __future__ import annotations

import logging
import re
import sys
import os
import asyncio
import sqlite3
import torch
import polars as pl
import json
import random
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


def sanitize_name(name: str) -> str:
    """
    Description:
    Creates a safe identifier for nodes by sanitizing a given name.

    Args:
        name (str): The node name to sanitize.

    Returns:
        str: The sanitized name with special characters removed and spaces normalized.
    """
    name = re.sub(r'\([^)]*\)|\[[^]]*\]', '', name)
    clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', name.lower())
    return re.sub(r'\s+', ' ', clean).strip()


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

    return [{'uri': r[0], 'name': r[1], 'sanitized_name': r[2], 'def': '', 'parents': json.loads(r[3]) if r[3] and r[3].startswith('[') else ([r[3]] if r[3] else []), 'is_llm_generated': 0, 'vector': torch.frombuffer(bytearray(r[4]), dtype=torch.float32)} for r in rows]


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
        prefixed_name = f"{prefix}{name}"
        nodes[name] = {
            'sanitized_name': sanitized_name,
            'parents': json.loads(parent) if parent and parent.startswith('[') else ([parent] if parent else []),
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


def get_debug_rows(df: pl.DataFrame, target_column: str, random_debug_rows: Optional[int] = None, debug_rows: Optional[List[int]] = None, seed: int = 43) -> List[int]:
    """
    Description:
    Handles the logic of selecting unique toxic effects for debugging.

    Args:
        df (pl.DataFrame): Input dataset DataFrame.
        target_column (str): Name of the raw target column.
        random_debug_rows (Optional[int]): Number of random rows to select for debugging. Defaults to None.
        debug_rows (Optional[List[int]]): Explicit row indices to process. Defaults to None.
        seed (int): Random seed for reproducibility. Defaults to 43.

    Returns:
        List[int]: A list of selected row indices to be processed for debugging.
    """
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


def load_pipeline_prompts(directory_path: str) -> list[tuple[str, str]]:
    """
    Description:
    Loads all .txt prompt files from a directory, sorted by file name.

    Args:
        directory_path (str): Path to the directory containing prompt text files.

    Returns:
        list[tuple[str, str]]: A list of tuples containing (filename, file_content).
    """
    logger = logging.getLogger(__name__)
    prompts = []
    p_dir = Path(directory_path)
    if not p_dir.is_dir():
        logger.warning(f"Pipeline directory {directory_path} not found.")
        return []

    # Get all .txt files, sorted by name
    files = sorted([f for f in p_dir.glob("*.txt") if f.is_file()])
    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            prompts.append((f.name, file.read().strip()))
    logger.info(f"Loaded {len(prompts)} pipeline prompts from {directory_path}")
    return prompts


def load_system_prompt(path: str) -> str:
    """
    Description:
    Reads and returns the contents of a system prompt file.

    Args:
        path (str): The absolute or relative path to the prompt file.

    Returns:
        str: The content of the prompt file, stripped of leading/trailing whitespace.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def clean_llm_response(raw: str) -> str:
    """
    Description:
    Cleans raw LLM response by stripping out reasoning/thinking tags and extracting the raw JSON string.

    Args:
        raw (str): The raw response string from the LLM.

    Returns:
        str: The extracted JSON string or cleaned text.
    """
    text = raw
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    # Try to extract JSON from markdown code blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Try to extract raw JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def parse_llm_result(raw_content: str) -> dict[str, Any]:
    """
    Description:
    Parses the cleaned LLM response into a structured dictionary result, handling JSON parsing errors gracefully.

    Args:
        raw_content (str): The raw content string from the LLM.

    Returns:
        dict[str, Any]: A dictionary containing parsed effects/conclusions or error information.
    """
    logger = logging.getLogger(__name__)
    cleaned = clean_llm_response(raw_content)
    try:
        result = json.loads(cleaned)
        return result
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM response: {e}\nRaw: {raw_content[:200]}")
        return {"effects": [], "reasoning": f"Parse error: {e}", "parse_error": True}


async def process_standardize_batch(
    llm: Any, system_prompt: str, effects: list[str], max_concurrency: int
) -> dict[str, Any]:
    """
    Description:
    Processes standardization for a batch of cleaned toxicological effects.

    Args:
        llm (Any): The LLM engine wrapper.
        system_prompt (str): The standardization system prompt instructions.
        effects (list[str]): List of cleaned effect terms to standardize.
        max_concurrency (int): Maximum concurrent LLM requests.

    Returns:
        dict[str, Any]: A dictionary mapping cleaned effect terms to standardized names.
    """
    logger = logging.getLogger(__name__)
    semaphore = asyncio.Semaphore(max_concurrency)
    cache = {}
    counter = 0

    async def process_one(effect: str) -> None:
        nonlocal counter
        async with semaphore:
            if effect == "-":
                cache[effect] = {"standard_name": "-", "reasoning": "Preserved '-'"}
                counter += 1
                return

            user_message = f'Provide a standard name for this toxicological effect:\n\n"{effect}"'
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            for attempt in range(5):
                try:
                    response = await llm.ainvoke(messages)
                    cleaned = clean_llm_response(response.content)
                    try:
                        result = json.loads(cleaned)
                    except json.JSONDecodeError:
                        raise ValueError("Invalid JSON returned by LLM")

                    if "standard_name" not in result:
                        raise ValueError("Missing 'standard_name' in JSON")

                    cache[effect] = result

                    counter += 1
                    is_changed = result['standard_name'] != effect
                    status = "Changed" if is_changed else "Not Changed"
                    print(f"{counter}: LLM Std | Orig: '{effect}' -> ({status}): {result['standard_name']}", flush=True)

                    return
                except Exception as e:
                    if attempt == 4:
                        logger.error(f"LLM standardize error for '{effect[:50]}...' after 5 attempts: {e}")
                        cache[effect] = {
                            "standard_name": effect,
                            "reasoning": f"LLM error: {e}",
                        }
                    else:
                        await asyncio.sleep(1)

    tasks = [process_one(e) for e in effects]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM Standardizing", smoothing=0.0):
        await f

    return cache


async def process_clean_exceptions_batch(
    llm: Any, system_prompt: str, effects: list[str], max_concurrency: int
) -> dict[str, Any]:
    """
    Description:
    Processes structural cleaning corrections for a batch of effects (e.g. multiple colons, redundancies).

    Args:
        llm (Any): The LLM engine wrapper.
        system_prompt (str): The exceptions cleaning system prompt instructions.
        effects (list[str]): List of effect terms needing structural cleanup.
        max_concurrency (int): Maximum concurrent LLM requests.

    Returns:
        dict[str, Any]: A dictionary mapping original terms to corrected terms.
    """
    logger = logging.getLogger(__name__)
    semaphore = asyncio.Semaphore(max_concurrency)
    cache = {}
    counter = 0

    async def process_one(effect: str) -> None:
        nonlocal counter
        async with semaphore:
            if effect == "-":
                cache[effect] = {"corrected": "-", "reasoning": "Preserved '-'"}
                counter += 1
                return

            user_message = f'Fix and simplify this effect:\n\n"{effect}"'
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            for attempt in range(5):
                try:
                    response = await llm.ainvoke(messages)
                    cleaned = clean_llm_response(response.content)
                    try:
                        result = json.loads(cleaned)
                    except json.JSONDecodeError:
                        raise ValueError("Invalid JSON returned by LLM")

                    if "corrected" not in result:
                        raise ValueError("Missing 'corrected' in JSON")

                    corrected = result["corrected"]
                    reasoning = result.get("reasoning", "")
                    cache[effect] = {"corrected": corrected, "reasoning": reasoning}

                    counter += 1
                    print(f"{counter}: Exceptions | '{effect}' -> '{corrected}'", flush=True)
                    return
                except Exception as e:
                    if attempt == 4:
                        logger.error(f"Exceptions error for '{effect[:50]}...' after 5 attempts: {e}")
                        cache[effect] = {"corrected": effect, "reasoning": f"Error: {e}"}
                    else:
                        await asyncio.sleep(1)

    tasks = [process_one(e) for e in effects]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Cleaning Exceptions", smoothing=0.0):
        await f

    return cache


def needs_cleanup(effect: str) -> bool:
    """
    Description:
    Checks if a toxicological effect term requires structural exceptions cleanup.

    Args:
        effect (str): The term to check.

    Returns:
        bool: True if cleanup is needed, False otherwise.
    """
    if not isinstance(effect, str) or effect == "-":
        return False
    # 1. Multiple colons
    if effect.count(":") > 1:
        return True
    # 2. Redundancy context: context
    if ":" in effect:
        parts = effect.split(":", 1)
        ctx = parts[0].strip().lower()
        eff = parts[1].strip().lower()
        # Check if ctx word is in eff words or vice versa
        ctx_words = set(ctx.replace("/", " ").replace("-", " ").split())
        eff_words = set(eff.replace("/", " ").replace("-", " ").split())
        if any(len(w) > 3 and w in eff_words for w in ctx_words):
            return True
    return False
