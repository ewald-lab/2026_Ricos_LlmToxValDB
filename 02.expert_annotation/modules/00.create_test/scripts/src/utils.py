import logging
import re
import sys
import os
import sqlite3
import torch
from typing import List, Dict
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
