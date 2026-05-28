from __future__ import annotations

import logging
import sys
import os
import sqlite3
import torch
from typing import Any, Dict, List, Tuple

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
    cursor.execute("SELECT uri, name, sanitized_name, definition, parent, is_llm_generated, vector FROM embeddings")
    rows = cursor.fetchall()
    conn.close()

    return [{'uri': r[0], 'name': r[1], 'sanitized_name': r[2], 'def': r[3], 'parent': r[4], 'is_llm_generated': r[5], 'vector': torch.frombuffer(bytearray(r[6]), dtype=torch.float32)} for r in rows]

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
