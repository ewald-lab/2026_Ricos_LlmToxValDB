from __future__ import annotations

import logging
import sys
import os
import sqlite3
import torch
from typing import Any, Dict, List

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

    import json
    return [{'uri': r[0], 'name': r[1], 'sanitized_name': r[2], 'def': '', 'parents': json.loads(r[3]) if r[3] and r[3].startswith('[') else ([r[3]] if r[3] else []), 'is_llm_generated': 0, 'vector': torch.frombuffer(bytearray(r[4]), dtype=torch.float32)} for r in rows]

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
