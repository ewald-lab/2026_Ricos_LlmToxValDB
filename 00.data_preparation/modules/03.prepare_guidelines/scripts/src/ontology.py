from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OntologyNode:
    """
    Description:
    Single ontology term as stored in the processed ontology SQLite DB.

    Attributes:
        uri (str): The unique URI of the ontology term.
        name (str): The name of the ontology term.
        sanitized_name (str): The sanitized name of the ontology term.
        definition (str): The definition of the ontology term.
        parent (str | None): The parent ontology term name, or None if it is a root.
        vector (bytes | None): Optional vector embedding representation. Defaults to None.
    """

    uri: str
    name: str
    sanitized_name: str
    definition: str
    parent: str | None
    vector: bytes | None = None


class OntologyTree:
    """
    Description:
    In-memory ontology tree used to walk from broad terms to specific filters.
    """

    def __init__(self, nodes: dict[str, OntologyNode]) -> None:
        """
        Description:
        Initializes the in-memory ontology tree and parent-to-child mapping.

        Args:
            nodes (dict[str, OntologyNode]): Dictionary mapping node names to OntologyNode objects.
        """
        self.nodes = nodes
        self.children: dict[str | None, list[str]] = defaultdict(list)

        # Build a parent -> children index. Terms whose parent is missing from the
        # loaded DB are treated as roots so traversal can still start cleanly.
        for name, node in self.nodes.items():
            parent = node.parent if node.parent in self.nodes else None
            self.children[parent].append(name)

        # Stable ordering makes prompts and outputs reproducible across runs.
        for child_names in self.children.values():
            child_names.sort(key=lambda item: self.nodes[item].sanitized_name.lower())

        self._descendant_counts: dict[str, int] = {}

    @classmethod
    def from_db(cls, ontology_db: str, global_definitions_db: str | None = None) -> OntologyTree:
        """
        Description:
        Loads ontology nodes from the SQLite database.

        Args:
            ontology_db (str): Path to the SQLite ontology database.
            global_definitions_db (str | None): Path to the SQLite global definitions database. Defaults to None.

        Returns:
            OntologyTree: An instance of OntologyTree containing the loaded nodes.
        """
        db_path = Path(ontology_db)
        if not db_path.exists():
            raise FileNotFoundError(f"Ontology DB not found: {ontology_db}")

        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT uri, name, sanitized_name, definition, parent, vector
                FROM embeddings
                """
            )
            rows = cursor.fetchall()

        nodes = {
            row[1]: OntologyNode(
                uri=row[0] or "",
                name=row[1] or "",
                sanitized_name=row[2] or row[1] or "",
                definition=row[3] or "",
                parent=row[4] or None,
                vector=row[5] if len(row) > 5 else None,
            )
            for row in rows
            if row[1]
        }

        # Prefer curated/LLM-generated global definitions when available. These
        # definitions are indexed by URI, so they can replace sparse DB definitions.
        if global_definitions_db:
            cls._apply_global_definitions(nodes, global_definitions_db)

        return cls(nodes)

    @staticmethod
    def _apply_global_definitions(nodes: dict[str, OntologyNode], global_definitions_db: str) -> None:
        """
        Description:
        Replaces node definitions using the global definitions database.

        Args:
            nodes (dict[str, OntologyNode]): Dictionary of ontology nodes.
            global_definitions_db (str): Path to the global definitions database.
        """
        db_path = Path(global_definitions_db)
        if not db_path.exists():
            return

        uri_to_name = {node.uri: name for name, node in nodes.items() if node.uri}
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT uri, definition FROM global_definitions")
                for uri, definition in cursor.fetchall():
                    name = uri_to_name.get(uri)
                    if name and definition and len(definition.strip()) > 5:
                        nodes[name].definition = definition
        except sqlite3.Error:
            return

    def roots(self, root_name: str | None = None) -> list[str]:
        """
        Description:
        Returns traversal roots, optionally constrained to one configured term.

        Args:
            root_name (str | None): Configured root name to filter, or None. Defaults to None.

        Returns:
            list[str]: A list of root term names.
        """
        if root_name:
            if root_name not in self.nodes:
                raise ValueError(f"Configured root not found in ontology DB: {root_name}")
            return [root_name]
        return list(self.children.get(None, []))

    def get_children(self, name: str) -> list[str]:
        """
        Description:
        Returns direct children for an ontology term name.

        Args:
            name (str): The name of the ontology term.

        Returns:
            list[str]: A list of child term names.
        """
        return self.children.get(name, [])

    def descendant_count(self, name: str) -> int:
        """
        Description:
        Counts all descendants under a term, caching results for prompt reuse.

        Args:
            name (str): The name of the ontology term.

        Returns:
            int: The total count of descendant terms.
        """
        if name in self._descendant_counts:
            return self._descendant_counts[name]

        count = 0
        stack = list(self.get_children(name))
        while stack:
            child = stack.pop()
            count += 1
            stack.extend(self.get_children(child))

        self._descendant_counts[name] = count
        return count

    def breadcrumbs(self, name: str) -> str:
        """
        Description:
        Returns a readable root-to-term path (separated by ' > ') for prompts and CSV outputs.

        Args:
            name (str): The name of the ontology term.

        Returns:
            str: A breadcrumb string representing the hierarchy path.
        """
        path = []
        current = name
        visited = set()
        while current and current not in visited:
            visited.add(current)
            node = self.nodes.get(current)
            if not node:
                break
            path.append(node.sanitized_name or node.name)
            current = node.parent
        return " > ".join(reversed(path))

    def resolve_child_names(self, selected_terms: list[str], child_names: list[str]) -> list[str]:
        """
        Description:
        Matches selected child labels (which could be names, sanitized names, or URIs)
        back to exact ontology child names.

        Args:
            selected_terms (list[str]): Terms selected by the LLM.
            child_names (list[str]): List of valid candidate child term names.

        Returns:
            list[str]: Matched candidate child term names in order of appearance.
        """
        by_name = {name.lower(): name for name in child_names}
        by_sanitized = {
            self.nodes[name].sanitized_name.lower(): name
            for name in child_names
        }
        by_uri = {
            self.nodes[name].uri.lower(): name
            for name in child_names
            if self.nodes[name].uri
        }

        resolved = []
        seen = set()
        for term in selected_terms or []:
            key = str(term).strip().lower()
            name = by_name.get(key) or by_sanitized.get(key) or by_uri.get(key)
            if name and name not in seen:
                resolved.append(name)
                seen.add(name)
        return resolved
