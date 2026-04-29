import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OntologyNode:
    """Single ontology term as stored in the processed ontology SQLite DB."""

    uri: str
    name: str
    sanitized_name: str
    definition: str
    parent: str | None


class OntologyTree:
    """In-memory ontology tree used to walk from broad terms to specific filters."""

    def __init__(self, nodes: dict[str, OntologyNode]):
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
    def from_db(cls, ontology_db: str, global_definitions_db: str | None = None):
        """Load ontology nodes from the existing embeddings DB format."""

        db_path = Path(ontology_db)
        if not db_path.exists():
            raise FileNotFoundError(f"Ontology DB not found: {ontology_db}")

        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT uri, name, sanitized_name, definition, parent
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
    def _apply_global_definitions(nodes: dict[str, OntologyNode], global_definitions_db: str):
        """Replace node definitions using the optional URI-level definitions DB."""

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
        """Return traversal roots, optionally constrained to one configured term."""

        if root_name:
            if root_name not in self.nodes:
                raise ValueError(f"Configured root not found in ontology DB: {root_name}")
            return [root_name]
        return list(self.children.get(None, []))

    def get_children(self, name: str) -> list[str]:
        """Return direct children for an ontology term name."""

        return self.children.get(name, [])

    def descendant_count(self, name: str) -> int:
        """Count all descendants under a term, caching results for prompt reuse."""

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
        """Return a readable root-to-term path for prompts and CSV outputs."""

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
        """Match LLM-selected child labels back to exact ontology child names."""

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
