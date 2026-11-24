"""Python accessor for toxvaldb/config/ontologies.json, the single source of
truth for the ontologies used across every module. The bash equivalent is
toxvaldb/config/ontologies.sh; keep the two in step.

Deliberately minimal: it exposes the ontology lists, which is what Python
callers need in order to stop hardcoding them. Everything else (source
resolution, checksums, DB roots) is driven from job scripts via ontologies.sh,
so it lives there rather than being mirrored here speculatively.

Usage, from any script that already has the repository root on sys.path (the
job scripts export it as PYTHONPATH, and the stage scripts insert REPO_ROOT, to
import HiPO):

    from toxvaldb.config.ontology_config import ontology_ids

    parser.add_argument("--ontology", choices=ontology_ids("guidelines"))

The table is located from this file's own path, so no caller passes one in.
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "ontologies.json"


def _load() -> dict:
    """Read and parse the ontology table.

    Returns:
        dict: The decoded contents of toxvaldb/config/ontologies.json.
    """
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def ontology_ids(group: str = "all") -> list[str]:
    """List the ontology ids in a group, in canonical order.

    "all" is every configured ontology; any other name is looked up in the
    table's `groups`. Guideline runs deliberately exclude ontologies added after
    the original five, which is why the subsets are named rather than assumed.

    Args:
        group (str): Group name. Defaults to "all".

    Returns:
        list[str]: Ontology ids, ordered to match Slurm array task indices.

    Raises:
        KeyError: If the group is not defined.
    """
    table = _load()
    if group == "all":
        return list(table["order"])

    groups = table.get("groups", {})
    if group not in groups:
        known = ", ".join(["all", *groups])
        raise KeyError(f"unknown ontology group '{group}' (known: {known})")
    return list(groups[group])
