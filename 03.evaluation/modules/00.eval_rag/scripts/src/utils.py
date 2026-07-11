from __future__ import annotations

import logging
import os
import sqlite3
import sys
from typing import Any, Dict, List

import torch

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

    # Dynamically determine columns
    cursor.execute("PRAGMA table_info(embeddings)")
    cols = [col[1] for col in cursor.fetchall()]

    query_cols = []
    for c in ["uri", "name", "sanitized_name"]:
        if c in cols:
            query_cols.append(c)

    has_def = "definition" in cols
    if has_def:
        query_cols.append("definition")

    has_parents = "parents" in cols
    has_parent = "parent" in cols
    if has_parents:
        query_cols.append("parents")
    elif has_parent:
        query_cols.append("parent")

    has_llm = "is_llm_generated" in cols
    if has_llm:
        query_cols.append("is_llm_generated")

    query_cols.append("vector")

    col_str = ", ".join(query_cols)
    cursor.execute(f"SELECT {col_str} FROM embeddings")
    rows = cursor.fetchall()
    conn.close()

    # Map rows to dict
    results = []
    import json
    for r in rows:
        d = {}
        idx = 0
        d['uri'] = r[idx]
        idx += 1
        d['name'] = r[idx]
        idx += 1
        d['sanitized_name'] = r[idx]
        idx += 1

        if has_def:
            d['def'] = r[idx]
            idx += 1
        else:
            d['def'] = ""

        p_val = r[idx]
        idx += 1
        parents_list = []
        if p_val:
            if isinstance(p_val, list):
                parents_list = p_val
            elif isinstance(p_val, str):
                if p_val.startswith('['):
                    try:
                        parents_list = json.loads(p_val)
                    except Exception:
                        parents_list = [p_val]
                else:
                    parents_list = [p_val]
        d['parents'] = parents_list

        if has_llm:
            d['is_llm_generated'] = r[idx]
            idx += 1
        else:
            d['is_llm_generated'] = 0

        d['vector'] = torch.frombuffer(bytearray(r[idx]), dtype=torch.float32)
        results.append(d)

    return results
