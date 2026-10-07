from __future__ import annotations

import logging
import re
import sys
import os
import asyncio
import sqlite3
import torch
import json
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
