import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class OntologyNode:
    """Single ontology term as stored in the processed ontology SQLite DB."""

    uri: str
    name: str
    sanitized_name: str
    definition: str
    parents: List[str]
    vector: Optional[bytes] = None


class OntologyTree:
    """In-memory ontology tree used to walk from broad terms to specific filters."""

    def __init__(self, nodes: Dict[str, OntologyNode]) -> None:
        """
        Description:
        Initializes the OntologyTree with in-memory nodes.

        Args:
            nodes (Dict[str, OntologyNode]): Mapping of term names to OntologyNode instances.

        Returns:
            None
        """
        self.nodes = nodes
        self.children: dict[Optional[str], List[str]] = defaultdict(list)

        # Build a parent -> children index. Terms whose parents are missing from the
        # loaded DB are treated as roots so traversal can still start cleanly.
        for name, node in self.nodes.items():
            if not node.parents:
                self.children[None].append(name)
            else:
                for p in node.parents:
                    parent_key = p if p in self.nodes else None
                    self.children[parent_key].append(name)

        # Remove duplicates from children lists (since multiple unknown parents could map to None)
        for key, child_names in self.children.items():
            self.children[key] = list(set(child_names))

        # Stable ordering makes prompts and outputs reproducible across runs.
        for child_names in self.children.values():
            child_names.sort(key=lambda item: self.nodes[item].sanitized_name.lower())

        self._descendant_counts: Dict[str, int] = {}

    @classmethod
    def from_db(cls, ontology_db: str, global_definitions_db: Optional[str] = None) -> "OntologyTree":
        """
        Description:
        Load ontology nodes from the existing embeddings DB format.

        Args:
            ontology_db (str): Path to SQLite database of ontology nodes.
            global_definitions_db (Optional[str]): Optional SQLite DB path with external node definitions. Defaults to None.

        Returns:
            OntologyTree: An instance of the loaded OntologyTree.
        """
        db_path = Path(ontology_db)
        if not db_path.exists():
            raise FileNotFoundError(f"Ontology DB not found: {ontology_db}")

        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT uri, name, sanitized_name, parent, vector
                FROM embeddings
                """
            )
            rows = cursor.fetchall()

        import json
        nodes = {
            row[1]: OntologyNode(
                uri=row[0] or "",
                name=row[1] or "",
                sanitized_name=row[2] or row[1] or "",
                definition="",
                parents=json.loads(row[3]) if row[3] and row[3].startswith('[') else ([row[3]] if row[3] else []),
                vector=row[4] if len(row) > 4 else None,
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
    def _apply_global_definitions(nodes: Dict[str, OntologyNode], global_definitions_db: str) -> None:
        """
        Description:
        Replace node definitions using the optional URI-level definitions DB.

        Args:
            nodes (Dict[str, OntologyNode]): Mapping of term name to node nodes to modify in-place.
            global_definitions_db (str): SQLite database path containing external definitions.

        Returns:
            None
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

    def roots(self, root_name: Optional[str] = None) -> List[str]:
        """
        Description:
        Return traversal roots, optionally constrained to one configured term.

        Args:
            root_name (Optional[str]): The configured name of a term to use as root. Defaults to None.

        Returns:
            List[str]: List of root term name strings.
        """
        if root_name:
            if root_name not in self.nodes:
                raise ValueError(f"Configured root not found in ontology DB: {root_name}")
            return [root_name]
        return list(self.children.get(None, []))

    def get_children(self, name: str) -> List[str]:
        """
        Description:
        Return direct children for an ontology term name.

        Args:
            name (str): Parent term name to look up.

        Returns:
            List[str]: Direct child term name strings list.
        """
        return self.children.get(name, [])

    def descendant_count(self, name: str) -> int:
        """
        Description:
        Count all descendants under a term, caching results for prompt reuse.

        Args:
            name (str): Node term name to count descendants for.

        Returns:
            int: The total count of descendants.
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
        Return a readable root-to-term path for prompts and CSV outputs.

        Args:
            name (str): Term name to build a hierarchical breadcrumb path for.

        Returns:
            str: The joined path string.
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
            current = node.parents[0] if node.parents else None
        return " > ".join(reversed(path))

    def resolve_child_names(self, selected_terms: List[str], child_names: List[str]) -> List[str]:
        """
        Description:
        Match LLM-selected child labels back to exact ontology child names.

        Args:
            selected_terms (List[str]): List of labels/URIs/names selected by LLM.
            child_names (List[str]): Exact ontology child names candidate list.

        Returns:
            List[str]: Resolved exact child names list.
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
