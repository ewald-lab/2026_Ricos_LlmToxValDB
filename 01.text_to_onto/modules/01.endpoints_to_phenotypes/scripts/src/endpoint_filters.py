import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .ontology import OntologyTree

VALID_DECISIONS = {"NOT_RELEVANT", "TOO_BROAD", "CONTINUE", "OPTIMAL_FILTER"}

# CSV schemas are centralized here so writer functions and downstream consumers
# use stable column names.
ENDPOINT_FILTER_FIELDS = [
    "ontology",
    "normalized_category",
    "normalized_endpoint",
    "original_endpoints",
    "guideline_numbers",
    "occurrence_count",
    "term_name",
    "term_uri",
    "sanitized_name",
    "breadcrumb",
    "descendant_count",
    "status",
    "confidence",
    "reason",
]

GUIDELINE_FILTER_FIELDS = [
    "ontology",
    "guideline_number",
    "term_name",
    "term_uri",
    "sanitized_name",
    "breadcrumb",
    "source_endpoints",
    "endpoint_count",
    "max_confidence",
]

UNIQUE_FILTER_FIELDS = [
    "ontology",
    "term_name",
    "term_uri",
    "sanitized_name",
    "breadcrumb",
    "descendant_count",
    "source_endpoints",
    "guideline_numbers",
    "endpoint_count",
    "max_confidence",
]


def clean_llm_output(message) -> str:
    """Normalize LLM responses before JSON parsing."""

    text = message.content if hasattr(message, "content") else str(message)

    # Some reasoning models emit a hidden reasoning block before the final answer.
    # Keep only the final response section requested by the prompt.
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Remove common Markdown code fences without changing the JSON body.
    text = text.strip()
    if text.startswith("```json"):
        text = text.replace("```json", "", 1)
    elif text.startswith("```"):
        text = text.replace("```", "", 1)
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_json_response(raw: str) -> dict[str, Any]:
    """Parse and validate the strict JSON object requested from the LLM."""

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Be tolerant of models that add a short preamble around the JSON.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))

    # Invalid decisions become rejections so malformed output cannot accidentally
    # create broad ontology filters.
    decision = str(payload.get("decision", "")).strip().upper()
    if decision not in VALID_DECISIONS:
        decision = "NOT_RELEVANT"

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    selected = payload.get("selected_child_terms", [])
    if isinstance(selected, str):
        selected = [selected]
    if not isinstance(selected, list):
        selected = []

    return {
        "decision": decision,
        "confidence": confidence,
        "selected_child_terms": [str(item).strip() for item in selected if str(item).strip()],
        "reason": str(payload.get("reason", "")).strip(),
    }


def truncate_text(text: str, limit: int = 700) -> str:
    """Shorten long definitions so prompts stay compact."""

    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def load_endpoint_rows(path: Path, debug_limit: int | None = None) -> list[dict[str, str]]:
    """Load normalized endpoint rows, skipping incomplete category/endpoint rows."""

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            endpoint = row.get("normalized_endpoint", "").strip()
            category = row.get("normalized_category", "").strip()
            if not endpoint or not category:
                continue
            rows.append(row)
            if debug_limit and len(rows) >= debug_limit:
                break
    return rows


class EndpointOntologyFilterBuilder:
    """Build endpoint-derived ontology subtree filters using LLM-guided traversal."""

    def __init__(
        self,
        ontology: str,
        tree: OntologyTree,
        system_prompt: str,
        context_ontology: str = "",
        max_paths_per_endpoint: int = 5,
        max_depth: int = 12,
        max_concurrency: int = 10,
        root_name: str | None = None,
    ):
        # Imports stay local so this helper module can be inspected or tested
        # without importing LangChain unless a builder is actually created.
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnableLambda

        self.ontology = ontology
        self.tree = tree
        self.max_paths_per_endpoint = max_paths_per_endpoint
        self.max_depth = max_depth
        self.max_concurrency = max_concurrency
        self.root_name = root_name

        prompt = system_prompt
        if context_ontology:
            prompt = f"{prompt}\n\nOntology context:\n{context_ontology.strip()}"

        # Each prompt presents one endpoint and one ontology candidate term,
        # plus only that candidate's immediate children. The model can stop at
        # the candidate or choose which children to explore next.
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                (
                    "user",
                    "Endpoint category: {endpoint_category}\n"
                    "Endpoint: {endpoint}\n"
                    "Original endpoint labels: {original_endpoints}\n"
                    "Guideline numbers: {guideline_numbers}\n\n"
                    "Current ontology candidate:\n"
                    "- name: {candidate_name}\n"
                    "- uri: {candidate_uri}\n"
                    "- breadcrumb: {candidate_breadcrumb}\n"
                    "- descendant_count: {candidate_descendant_count}\n"
                    "- definition: {candidate_definition}\n\n"
                    "Immediate children to consider:\n{children}\n\n"
                    "Return the JSON object now.",
                ),
            ]
        )
        self._cleaner = RunnableLambda(clean_llm_output)

    def build_for_endpoints(self, llm, endpoints: list[dict[str, str]]):
        """Run traversal for all endpoints and aggregate output tables."""

        endpoint_rows = []
        reasoning = {}

        for index, endpoint in enumerate(endpoints, start=1):
            endpoint_key = self.endpoint_key(endpoint)
            result = self.traverse_endpoint(llm, endpoint)
            reasoning[endpoint_key] = result["reasoning"]
            endpoint_rows.extend(self.filter_rows_for_endpoint(endpoint, result["filters"]))
            print(
                f"{index}/{len(endpoints)} | {self.ontology} | "
                f"{endpoint.get('normalized_category')} :: {endpoint.get('normalized_endpoint')} "
                f"-> {len(result['filters'])} filters",
                flush=True,
            )

        guideline_rows = build_guideline_rows(endpoint_rows)
        unique_rows = build_unique_rows(endpoint_rows)
        return endpoint_rows, guideline_rows, unique_rows, reasoning

    @staticmethod
    def endpoint_key(endpoint: dict[str, str]) -> str:
        """Create a stable key for the reasoning JSON."""

        return f"{endpoint.get('normalized_category', '')}::{endpoint.get('normalized_endpoint', '')}"

    def traverse_endpoint(self, llm, endpoint: dict[str, str]) -> dict[str, Any]:
        """Walk the ontology for one endpoint until optimal filter roots are found."""

        roots = self.tree.roots(self.root_name)
        queue = [{"name": name, "depth": 0, "path": []} for name in roots]
        visited = set()
        filters = []
        traversal_steps = []

        chain = self._prompt | llm | self._cleaner

        while queue and len(filters) < self.max_paths_per_endpoint:
            batch_items = []
            batch_context = []
            next_queue = []

            # Convert the current frontier into prompt payloads. A frontier can
            # contain several ontology branches, so they are evaluated in a batch.
            for item in queue:
                name = item["name"]
                if name in visited or name not in self.tree.nodes:
                    continue
                visited.add(name)

                node = self.tree.nodes[name]
                child_names = self.tree.get_children(name)
                batch_items.append(self.prompt_context(endpoint, name, child_names))
                batch_context.append({"queue_item": item, "node": node, "child_names": child_names})

            if not batch_items:
                break

            # LangChain handles concurrent LLM calls here. The surrounding code
            # keeps enough context to map each response back to its ontology node.
            raw_results = chain.batch(
                batch_items,
                config={"max_concurrency": self.max_concurrency},
                return_exceptions=True,
            )

            for raw_result, context in zip(raw_results, batch_context):
                item = context["queue_item"]
                node = context["node"]
                child_names = context["child_names"]

                evaluation = self.parse_or_reject(raw_result)
                selected_children = self.tree.resolve_child_names(
                    evaluation["selected_child_terms"],
                    child_names,
                )

                step = {
                    "candidate": node.name,
                    "candidate_uri": node.uri,
                    "breadcrumb": self.tree.breadcrumbs(node.name),
                    "decision": evaluation["decision"],
                    "confidence": evaluation["confidence"],
                    "selected_child_terms": selected_children,
                    "reason": evaluation["reason"],
                    "depth": item["depth"],
                }
                traversal_steps.append(step)

                # A selected optimal term becomes a final filter for this endpoint.
                if evaluation["decision"] == "OPTIMAL_FILTER":
                    filters.append(
                        {
                            "node": node,
                            "status": evaluation["decision"],
                            "confidence": evaluation["confidence"],
                            "reason": evaluation["reason"],
                        }
                    )
                    continue

                # Broad/relevant terms are expanded only through child terms that
                # the model explicitly selected from the immediate child list.
                if evaluation["decision"] in {"TOO_BROAD", "CONTINUE"}:
                    if item["depth"] + 1 >= self.max_depth:
                        # If traversal hits the depth limit, keep the current term
                        # as a conservative low-confidence fallback rather than
                        # silently dropping a relevant branch.
                        filters.append(
                            {
                                "node": node,
                                "status": "OPTIMAL_FILTER",
                                "confidence": min(evaluation["confidence"], 0.5),
                                "reason": (
                                    evaluation["reason"]
                                    or "Reached maximum traversal depth before a more specific child was selected."
                                ),
                            }
                        )
                        continue

                    for child in selected_children:
                        next_queue.append(
                            {
                                "name": child,
                                "depth": item["depth"] + 1,
                                "path": item["path"] + [node.name],
                            }
                        )

            queue = next_queue

        return {
            "filters": filters[: self.max_paths_per_endpoint],
            "reasoning": {
                "endpoint": endpoint,
                "final_filters": [
                    {
                        "term_name": item["node"].name,
                        "term_uri": item["node"].uri,
                        "status": item["status"],
                        "confidence": item["confidence"],
                        "reason": item["reason"],
                    }
                    for item in filters[: self.max_paths_per_endpoint]
                ],
                "traversal": traversal_steps,
            },
        }

    def prompt_context(self, endpoint: dict[str, str], name: str, child_names: list[str]):
        """Build the dictionary used to fill one LLM prompt."""

        node = self.tree.nodes[name]
        return {
            "endpoint_category": endpoint.get("normalized_category", ""),
            "endpoint": endpoint.get("normalized_endpoint", ""),
            "original_endpoints": endpoint.get("original_endpoints", ""),
            "guideline_numbers": endpoint.get("guideline_numbers", ""),
            "candidate_name": node.name,
            "candidate_uri": node.uri,
            "candidate_breadcrumb": self.tree.breadcrumbs(name),
            "candidate_descendant_count": self.tree.descendant_count(name),
            "candidate_definition": truncate_text(node.definition),
            "children": self.format_children(child_names),
        }

    def format_children(self, child_names: list[str]) -> str:
        """Format immediate children with enough metadata for LLM selection."""

        if not child_names:
            return "None"

        lines = []
        for index, child_name in enumerate(child_names, start=1):
            child = self.tree.nodes[child_name]
            lines.append(
                f"{index}. name: {child.name} | sanitized: {child.sanitized_name} | "
                f"uri: {child.uri} | descendants: {self.tree.descendant_count(child.name)} | "
                f"definition: {truncate_text(child.definition, 350)}"
            )
        return "\n".join(lines)

    @staticmethod
    def parse_or_reject(raw_result):
        """Convert failed calls or malformed responses into safe rejections."""

        if isinstance(raw_result, Exception):
            return {
                "decision": "NOT_RELEVANT",
                "confidence": 0.0,
                "selected_child_terms": [],
                "reason": f"LLM call failed: {raw_result}",
            }
        try:
            return parse_json_response(str(raw_result))
        except Exception as exc:
            return {
                "decision": "NOT_RELEVANT",
                "confidence": 0.0,
                "selected_child_terms": [],
                "reason": f"Could not parse LLM response: {exc}; raw={str(raw_result)[:300]}",
            }

    def filter_rows_for_endpoint(self, endpoint: dict[str, str], filters: list[dict[str, Any]]):
        """Convert final filters for one endpoint into CSV-ready rows."""

        rows = []
        for item in filters:
            node = item["node"]
            rows.append(
                {
                    "ontology": self.ontology,
                    "normalized_category": endpoint.get("normalized_category", ""),
                    "normalized_endpoint": endpoint.get("normalized_endpoint", ""),
                    "original_endpoints": endpoint.get("original_endpoints", ""),
                    "guideline_numbers": endpoint.get("guideline_numbers", ""),
                    "occurrence_count": endpoint.get("occurrence_count", ""),
                    "term_name": node.name,
                    "term_uri": node.uri,
                    "sanitized_name": node.sanitized_name,
                    "breadcrumb": self.tree.breadcrumbs(node.name),
                    "descendant_count": str(self.tree.descendant_count(node.name)),
                    "status": item["status"],
                    "confidence": f"{item['confidence']:.3f}",
                    "reason": item["reason"],
                }
            )
        return rows


def split_pipe_values(value: str) -> list[str]:
    """Split repository-style pipe-separated provenance fields."""

    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def sort_guidelines(values: set[str]) -> list[str]:
    """Sort numeric guideline identifiers before any non-numeric labels."""

    return sorted(values, key=lambda item: (not item.isdigit(), int(item) if item.isdigit() else item))


def build_guideline_rows(endpoint_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Aggregate endpoint filters into guideline-level filter unions."""

    grouped = {}
    for row in endpoint_rows:
        for guideline in split_pipe_values(row["guideline_numbers"]):
            key = (row["ontology"], guideline, row["term_name"])
            if key not in grouped:
                grouped[key] = {
                    "ontology": row["ontology"],
                    "guideline_number": guideline,
                    "term_name": row["term_name"],
                    "term_uri": row["term_uri"],
                    "sanitized_name": row["sanitized_name"],
                    "breadcrumb": row["breadcrumb"],
                    "source_endpoints": set(),
                    "max_confidence": 0.0,
                }
            grouped[key]["source_endpoints"].add(row["normalized_endpoint"])
            grouped[key]["max_confidence"] = max(
                grouped[key]["max_confidence"],
                float(row["confidence"] or 0.0),
            )

    rows = []
    for payload in grouped.values():
        endpoints = sorted(payload["source_endpoints"])
        rows.append(
            {
                "ontology": payload["ontology"],
                "guideline_number": payload["guideline_number"],
                "term_name": payload["term_name"],
                "term_uri": payload["term_uri"],
                "sanitized_name": payload["sanitized_name"],
                "breadcrumb": payload["breadcrumb"],
                "source_endpoints": "|".join(endpoints),
                "endpoint_count": str(len(endpoints)),
                "max_confidence": f"{payload['max_confidence']:.3f}",
            }
        )
    return sorted(rows, key=lambda row: (row["ontology"], row["guideline_number"], row["term_name"].lower()))


def build_unique_rows(endpoint_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Aggregate endpoint filters into unique ontology terms with provenance."""

    grouped = {}
    for row in endpoint_rows:
        key = (row["ontology"], row["term_name"])
        if key not in grouped:
            grouped[key] = {
                "ontology": row["ontology"],
                "term_name": row["term_name"],
                "term_uri": row["term_uri"],
                "sanitized_name": row["sanitized_name"],
                "breadcrumb": row["breadcrumb"],
                "descendant_count": row["descendant_count"],
                "source_endpoints": set(),
                "guideline_numbers": set(),
                "max_confidence": 0.0,
            }
        grouped[key]["source_endpoints"].add(row["normalized_endpoint"])
        grouped[key]["guideline_numbers"].update(split_pipe_values(row["guideline_numbers"]))
        grouped[key]["max_confidence"] = max(
            grouped[key]["max_confidence"],
            float(row["confidence"] or 0.0),
        )

    rows = []
    for payload in grouped.values():
        endpoints = sorted(payload["source_endpoints"])
        rows.append(
            {
                "ontology": payload["ontology"],
                "term_name": payload["term_name"],
                "term_uri": payload["term_uri"],
                "sanitized_name": payload["sanitized_name"],
                "breadcrumb": payload["breadcrumb"],
                "descendant_count": payload["descendant_count"],
                "source_endpoints": "|".join(endpoints),
                "guideline_numbers": "|".join(sort_guidelines(payload["guideline_numbers"])),
                "endpoint_count": str(len(endpoints)),
                "max_confidence": f"{payload['max_confidence']:.3f}",
            }
        )
    return sorted(rows, key=lambda row: (row["ontology"], row["term_name"].lower()))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> Path:
    """Write rows to CSV, creating the output directory when needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON artifact, creating the output directory when needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path
