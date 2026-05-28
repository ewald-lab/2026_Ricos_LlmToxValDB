from __future__ import annotations

import logging
import re
import sys
from typing import Any

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
