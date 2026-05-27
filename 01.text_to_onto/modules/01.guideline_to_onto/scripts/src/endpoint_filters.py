import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Literal
import torch

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM

from .ontology import OntologyTree

def clean_llm_output(message) -> str:
    text = message.content if hasattr(message, "content") else str(message)
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = text.strip()
    if text.startswith("```json"):
        text = text.replace("```json", "", 1)
    elif text.startswith("```"):
        text = text.replace("```", "", 1)
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def clean_reason(reason: str) -> str:
    r = reason.strip()
    if r.upper().startswith("ACCEPTED"):
        return r[8:].lstrip(" .-:")
    elif r.upper().startswith("REJECTED"):
        return r[8:].lstrip(" .-:")
    return r

class CandidateEvaluation(BaseModel):
    candidate_name: str = Field(description="The exact name of the candidate being evaluated.")
    reason: str = Field(description="Think step-by-step. Provide the explicit reason for the decision based on the critical rules. DO NOT write 'ACCEPTED' or 'REJECTED' in this field.")
    decision: Literal["ACCEPTED", "REJECTED"] = Field(description="Strictly 'ACCEPTED' or 'REJECTED' based on the reasoning above.")

class BatchVerificationResult(BaseModel):
    analysis: str = Field(description="Step 1: Analyze the input description and determine its exact level of specificity.")
    evaluation: List[CandidateEvaluation] = Field(description="Step 2: Evaluate EVERY candidate individually. You MUST provide an evaluation object for ALL candidates in the list.")
    selected_candidates: List[str] = Field(description="Step 3: The final list of exact candidate names that pass the evaluation. Return an empty list [] if none match.")

ENDPOINT_FILTER_FIELDS = [
    "ontology", "normalized_category", "normalized_endpoint", "original_endpoints", "guideline_numbers",
    "occurrence_count", "term_name", "term_uri", "sanitized_name", "breadcrumb", "descendant_count",
    "confidence", "reason"
]

GUIDELINE_FILTER_FIELDS = [
    "ontology", "guideline_number", "term_name", "term_uri", "sanitized_name", "breadcrumb",
    "source_endpoints", "endpoint_count", "max_confidence"
]

UNIQUE_FILTER_FIELDS = [
    "ontology", "term_name", "term_uri", "sanitized_name", "breadcrumb", "descendant_count",
    "source_endpoints", "guideline_numbers", "endpoint_count", "max_confidence"
]

def truncate_text(text: str, limit: int = 700) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."

def load_endpoint_rows(path: Path, debug_limit: int | None = None) -> list[dict[str, str]]:
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
    def __init__(
        self, ontology: str, tree: OntologyTree, system_prompt: str, context_ontology: str = "",
        max_paths_per_endpoint: int = 5, max_depth: int = 12, max_concurrency: int = 10,
        root_name: str | None = None, embedding_model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        top_k_rag: int = 20, threshold_rag: float = 0.4,
        reranker_model: str = "Qwen/Qwen3-Reranker-8B", threshold_reranker: float = 0.0,
        reranker_batch_size: int = 32, reranker_prompt_file: str | None = None,
        mode: int = 1, consensus_runs: int = 3, max_retries: int = 3,
        top_k_reranker: int = None
    ):
        from model.embeddings import EmbeddingModel

        self.ontology = ontology
        self.tree = tree
        self.max_paths_per_endpoint = max_paths_per_endpoint
        self.max_concurrency = max_concurrency
        self.top_k_rag = top_k_rag
        self.threshold_rag = threshold_rag

        self.mode = mode
        self.consensus_runs = consensus_runs
        self.max_retries = max_retries
        self.consensus_threshold = 1 if self.consensus_runs == 2 else (self.consensus_runs // 2) + 1

        self.threshold_reranker = threshold_reranker
        self.top_k_reranker = top_k_reranker
        self.reranker_batch_size = reranker_batch_size
        self.reranker_model = None
        self.reranker_tokenizer = None

        if self.threshold_reranker > 0.0:
            print(f"Loading Reranker Model: {reranker_model}")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model, padding_side='left')
            self.reranker_model_hf = AutoModelForCausalLM.from_pretrained(
                reranker_model, torch_dtype=torch.bfloat16, device_map="auto"
            ).eval()
            self.reranker_device = self.reranker_model_hf.device
            self.token_false_id = self.reranker_tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.reranker_tokenizer.convert_tokens_to_ids("yes")
            prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.reranker_prefix_tokens = self.reranker_tokenizer.encode(prefix, add_special_tokens=False)
            self.reranker_suffix_tokens = self.reranker_tokenizer.encode(suffix, add_special_tokens=False)
            self._reranker_max_content_len = 1024 - len(self.reranker_prefix_tokens) - len(self.reranker_suffix_tokens)
            if reranker_prompt_file and Path(reranker_prompt_file).exists():
                with open(reranker_prompt_file, 'r', encoding='utf-8') as f:
                    self.reranker_instruction = f.read().strip()
            else:
                self.reranker_instruction = "Find candidates that perfectly match the described endpoint."

        self.embedder = EmbeddingModel(embedding_model_name)

        self.node_list = []
        vectors = []
        for name, node in self.tree.nodes.items():
            if getattr(node, "vector", None) is not None:
                self.node_list.append(node)
                vectors.append(torch.frombuffer(bytearray(node.vector), dtype=torch.float32))

        if vectors:
            self._ontology_matrix = torch.stack(vectors)
        else:
            self._ontology_matrix = None

        prompt = system_prompt
        if context_ontology:
            prompt += f"\n\nContext about the ontology:\n{context_ontology.strip()}"

        self.parser1 = PydanticOutputParser(pydantic_object=CandidateEvaluation)
        self.parser2 = PydanticOutputParser(pydantic_object=BatchVerificationResult)

        self._mode1_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt + "\n\n{format_instructions}"),
            ("user", "Endpoint category: {endpoint_category}\nEndpoint: {endpoint}\nOriginal endpoint labels: {original_endpoints}\nGuideline numbers: {guideline_numbers}\n\nCandidate to evaluate: {candidate_name}\nCandidate Hierarchy: {candidate_hierarchy}\nCandidate Definition: {candidate_definition}")
        ])

        self._mode2_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt + "\n\n{format_instructions}"),
            ("user", "Endpoint category: {endpoint_category}\nEndpoint: {endpoint}\nOriginal endpoint labels: {original_endpoints}\nGuideline numbers: {guideline_numbers}\n\nCandidates:\n{candidates}\n\nDescription to evaluate: {endpoint}")
        ])

        self._clean_llm_runner = RunnableLambda(clean_llm_output)

    def _rerank_candidates(self, description: str, candidates: list) -> list:
        if self.threshold_reranker <= 0.0 or not candidates or self.reranker_model_hf is None:
            return candidates

        pairs = []
        for c in candidates:
            doc = c.sanitized_name
            hierarchy = self.tree.breadcrumbs(c.name)
            doc = f"{doc}\n<Hierarchy>: {hierarchy}"
            pair_text = f"<Instruct>: {self.reranker_instruction}\n<Query>: {description}\n<Document>: {doc}"
            pairs.append(pair_text)

        batch_size = self.reranker_batch_size
        scores = []

        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                inputs = self.reranker_tokenizer(
                    batch_pairs, padding=False, truncation='longest_first',
                    return_attention_mask=False, max_length=self._reranker_max_content_len
                )
                for j, ele in enumerate(inputs['input_ids']):
                    inputs['input_ids'][j] = self.reranker_prefix_tokens + ele + self.reranker_suffix_tokens
                inputs = self.reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt")
                inputs = {k: v.to(self.reranker_device) for k, v in inputs.items()}

                batch_logits = self.reranker_model_hf(**inputs).logits[:, -1, :]
                true_vector = batch_logits[:, self.token_true_id]
                false_vector = batch_logits[:, self.token_false_id]
                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                batch_probs = batch_scores[:, 1].exp().tolist()
                scores.extend(batch_probs)
        passed_candidates = []
        for score, cand in zip(scores, candidates):
            if score >= self.threshold_reranker:
                setattr(cand, 'rerank_score', score)
                passed_candidates.append(cand)

        # Apply top_k_reranker limit if specified
        if self.top_k_reranker is not None and self.top_k_reranker > 0:
            passed_candidates.sort(key=lambda x: getattr(x, 'rerank_score', 0.0), reverse=True)
            passed_candidates = passed_candidates[:self.top_k_reranker]

        return passed_candidates

    def build_for_endpoints(self, llm, endpoints: list[dict[str, str]]):
        from tqdm import tqdm
        endpoint_rows = []
        reasoning = {}

        # Use tqdm for a proper progress bar
        pbar = tqdm(endpoints, desc=f"Endpoints ({self.ontology})", smoothing=0.0)
        for index, endpoint in enumerate(pbar, start=1):
            endpoint_key = f"{endpoint.get('normalized_category', '')}::{endpoint.get('normalized_endpoint', '')}"
            result = self.traverse_endpoint(llm, endpoint)
            reasoning[endpoint_key] = result["reasoning"]
            endpoint_rows.extend(self.filter_rows_for_endpoint(endpoint, result["filters"]))

            # Update tqdm postfix with current status
            pbar.set_postfix({"filters": len(result['filters']), "last": endpoint.get('normalized_endpoint', '')[:20]})

        guideline_rows = build_guideline_rows(endpoint_rows)
        unique_rows = build_unique_rows(endpoint_rows)
        return endpoint_rows, guideline_rows, unique_rows, reasoning

    def traverse_endpoint(self, llm, endpoint: dict[str, str]) -> dict[str, Any]:
        if self.mode == 2:
            return self._traverse_mode_2(llm, endpoint)
        return self._traverse_mode_1(llm, endpoint)

    def _traverse_mode_1(self, llm, endpoint: dict[str, str]) -> dict[str, Any]:
        filters = []
        traversal_steps = []
        if self._ontology_matrix is None:
            return {"filters": filters, "reasoning": {"endpoint": endpoint, "final_filters": [], "traversal": []}}

        desc = endpoint.get("normalized_endpoint", "")
        if not desc:
            return {"filters": filters, "reasoning": {"endpoint": endpoint, "final_filters": [], "traversal": []}}

        vec_bytes = self.embedder.get_embedding(desc)
        query_vec = torch.frombuffer(bytearray(vec_bytes), dtype=torch.float32)
        scores = torch.matmul(self._ontology_matrix, query_vec)
        _, top_indices = torch.topk(scores, k=min(self.top_k_rag, len(scores)))

        candidates = []
        for idx in top_indices.tolist():
            if scores[idx].item() >= self.threshold_rag:
                candidates.append(self.node_list[idx])

        candidates = self._rerank_candidates(desc, candidates)

        chain = self._mode1_prompt | llm | self._clean_llm_runner | self.parser1

        verified_nodes = set()
        evaluated_nodes = set()
        current_candidates = candidates
        iteration = 1

        while current_candidates:
            batch_inputs = []
            batch_metadata = []
            valid_candidates = []

            for idx, cand in enumerate(current_candidates):
                name = cand.name
                if name not in self.tree.nodes or name in verified_nodes or name in evaluated_nodes:
                    continue
                evaluated_nodes.add(name)
                valid_candidates.append(cand)

                for run_i in range(self.consensus_runs):
                    batch_inputs.append({
                        "endpoint_category": endpoint.get("normalized_category", ""),
                        "endpoint": desc,
                        "original_endpoints": endpoint.get("original_endpoints", ""),
                        "guideline_numbers": endpoint.get("guideline_numbers", ""),
                        "candidate_name": cand.sanitized_name,
                        "candidate_hierarchy": self.tree.breadcrumbs(name),
                        "candidate_definition": truncate_text(cand.definition),
                        "format_instructions": self.parser1.get_format_instructions()
                    })
                    batch_metadata.append({"name": name, "run_i": run_i, "cand": cand})

            if not valid_candidates:
                break

            batch_results = chain.batch(batch_inputs, config={"max_concurrency": self.max_concurrency}, return_exceptions=True)
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    for retry in range(self.max_retries):
                        try:
                            batch_results[i] = chain.invoke(batch_inputs[i])
                            break
                        except Exception:
                            pass

            candidate_votes = defaultdict(int)
            candidate_reasoning = defaultdict(list)
            for i, meta in enumerate(batch_metadata):
                name = meta["name"]
                result = batch_results[i]
                if isinstance(result, Exception):
                    candidate_reasoning[name].append("Parse error")
                elif result is not None:
                    if result.decision == 'ACCEPTED':
                        candidate_votes[name] += 1
                    candidate_reasoning[name].append(f"{result.decision} - {clean_reason(result.reason)}")

            next_round_candidates = []
            for cand in valid_candidates:
                name = cand.name
                votes = candidate_votes[name]
                status = "ACCEPTED" if votes >= self.consensus_threshold else "REJECTED"

                traversal_steps.append({
                    "candidate": name,
                    "candidate_uri": cand.uri,
                    "breadcrumb": self.tree.breadcrumbs(name),
                    "decision": status,
                    "votes": f"{votes}/{self.consensus_runs}",
                    "runs": candidate_reasoning[name]
                })

                if status == "ACCEPTED":
                    curr = name
                    while curr:
                        if curr in verified_nodes:
                            break
                        verified_nodes.add(curr)
                        curr_node = self.tree.nodes.get(curr)
                        if curr_node:
                            filters.append({
                                "node": curr_node,
                                "confidence": float(votes) / self.consensus_runs,
                                "reason": f"Selected via RAG matching {name}. {candidate_reasoning[name][0] if candidate_reasoning[name] else ''}"
                            })
                        curr = curr_node.parent if curr_node else None
                    # Enqueue unasked direct children of the accepted node
                    for child_name in self.tree.get_children(name):
                        if child_name not in verified_nodes and child_name not in evaluated_nodes:
                            child_info = self.tree.nodes.get(child_name)
                            if child_info and child_info not in next_round_candidates:
                                next_round_candidates.append(child_info)
                else:
                    parent_name = cand.parent
                    if parent_name and parent_name not in verified_nodes and parent_name not in evaluated_nodes:
                        node_info = self.tree.nodes.get(parent_name)
                        if node_info and node_info not in next_round_candidates:
                            next_round_candidates.append(node_info)

            current_candidates = self._rerank_candidates(desc, next_round_candidates)
            iteration += 1

        return {
            "filters": filters,
            "reasoning": {
                "endpoint": endpoint,
                "final_filters": [{"term_name": f["node"].name, "reason": f["reason"]} for f in filters],
                "traversal": traversal_steps,
            }
        }

    def _traverse_mode_2(self, llm, endpoint: dict[str, str]) -> dict[str, Any]:
        filters = []
        traversal_steps = []
        if self._ontology_matrix is None:
            return {"filters": filters, "reasoning": {"endpoint": endpoint, "final_filters": [], "traversal": []}}

        desc = endpoint.get("normalized_endpoint", "")
        if not desc:
            return {"filters": filters, "reasoning": {"endpoint": endpoint, "final_filters": [], "traversal": []}}

        vec_bytes = self.embedder.get_embedding(desc)
        query_vec = torch.frombuffer(bytearray(vec_bytes), dtype=torch.float32)
        scores = torch.matmul(self._ontology_matrix, query_vec)
        _, top_indices = torch.topk(scores, k=min(self.top_k_rag, len(scores)))

        candidates = []
        for idx in top_indices.tolist():
            if scores[idx].item() >= self.threshold_rag:
                candidates.append(self.node_list[idx])

        candidates = self._rerank_candidates(desc, candidates)
        chain = self._mode2_prompt | llm | self._clean_llm_runner | self.parser2

        verified_nodes = set()
        evaluated_nodes = set()
        current_candidates = candidates
        iteration = 1

        while current_candidates:
            valid_candidates = [c for c in current_candidates if c.name not in evaluated_nodes and c.name not in verified_nodes]
            if not valid_candidates:
                break

            for c in valid_candidates:
                evaluated_nodes.add(c.name)

            s2n = {c.sanitized_name: c.name for c in valid_candidates}
            candidates_str = "\n".join([f"- {c.sanitized_name} (Hierarchy: {self.tree.breadcrumbs(c.name)}): {truncate_text(c.definition)}" for c in valid_candidates])

            invoke_inputs = [{
                "endpoint_category": endpoint.get("normalized_category", ""),
                "endpoint": desc,
                "original_endpoints": endpoint.get("original_endpoints", ""),
                "guideline_numbers": endpoint.get("guideline_numbers", ""),
                "candidates": candidates_str,
                "format_instructions": self.parser2.get_format_instructions()
            } for _ in range(self.consensus_runs)]

            batch_results = chain.batch(invoke_inputs, config={"max_concurrency": self.max_concurrency}, return_exceptions=True)
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    for retry in range(self.max_retries):
                        try:
                            batch_results[i] = chain.invoke(invoke_inputs[i])
                            break
                        except Exception:
                            pass

            candidate_votes = defaultdict(int)
            for run_i, result in enumerate(batch_results):
                if isinstance(result, Exception): continue
                for selected in result.selected_candidates:
                    clean_sel = selected.strip()
                    if clean_sel in s2n:
                        candidate_votes[s2n[clean_sel]] += 1
                    elif clean_sel in [c.name for c in valid_candidates]:
                        candidate_votes[clean_sel] += 1

            next_round_candidates = []
            for cand in valid_candidates:
                name = cand.name
                votes = candidate_votes[name]
                status = "ACCEPTED" if votes >= self.consensus_threshold else "REJECTED"

                traversal_steps.append({
                    "candidate": name,
                    "decision": status,
                    "votes": f"{votes}/{self.consensus_runs}",
                })

                if status == "ACCEPTED":
                    curr = name
                    while curr:
                        if curr in verified_nodes: break
                        verified_nodes.add(curr)
                        curr_node = self.tree.nodes.get(curr)
                        if curr_node:
                            filters.append({
                                "node": curr_node,
                                "confidence": float(votes) / self.consensus_runs,
                                "reason": f"Selected via Batch RAG matching {name}"
                            })
                        curr = curr_node.parent if curr_node else None
                    # Enqueue unasked direct children of the accepted node
                    for child_name in self.tree.get_children(name):
                        if child_name not in verified_nodes and child_name not in evaluated_nodes:
                            child_info = self.tree.nodes.get(child_name)
                            if child_info and child_info not in next_round_candidates:
                                next_round_candidates.append(child_info)
                else:
                    parent_name = cand.parent
                    if parent_name and parent_name not in verified_nodes and parent_name not in evaluated_nodes:
                        node_info = self.tree.nodes.get(parent_name)
                        if node_info and node_info not in next_round_candidates:
                            next_round_candidates.append(node_info)

            current_candidates = self._rerank_candidates(desc, next_round_candidates)
            iteration += 1

        return {
            "filters": filters,
            "reasoning": {
                "endpoint": endpoint,
                "final_filters": [{"term_name": f["node"].name, "reason": f["reason"]} for f in filters],
                "traversal": traversal_steps,
            }
        }

    def filter_rows_for_endpoint(self, endpoint: dict[str, str], filters: list[dict[str, Any]]):
        rows = []
        for item in filters:
            node = item["node"]
            rows.append({
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
                "confidence": f"{item['confidence']:.3f}",
                "reason": item["reason"],
            })
        return rows

def split_pipe_values(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]

def sort_guidelines(values: set[str]) -> list[str]:
    return sorted(values, key=lambda item: (not item.isdigit(), int(item) if item.isdigit() else item))

def build_guideline_rows(endpoint_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped = {}
    for row in endpoint_rows:
        for guideline in split_pipe_values(row["guideline_numbers"]):
            key = (row["ontology"], guideline, row["term_name"])
            if key not in grouped:
                grouped[key] = {
                    "ontology": row["ontology"], "guideline_number": guideline, "term_name": row["term_name"],
                    "term_uri": row["term_uri"], "sanitized_name": row["sanitized_name"], "breadcrumb": row["breadcrumb"],
                    "source_endpoints": set(), "max_confidence": 0.0,
                }
            grouped[key]["source_endpoints"].add(row["normalized_endpoint"])
            grouped[key]["max_confidence"] = max(grouped[key]["max_confidence"], float(row["confidence"] or 0.0))
    rows = []
    for payload in grouped.values():
        endpoints = sorted(payload["source_endpoints"])
        rows.append({
            "ontology": payload["ontology"], "guideline_number": payload["guideline_number"],
            "term_name": payload["term_name"], "term_uri": payload["term_uri"],
            "sanitized_name": payload["sanitized_name"], "breadcrumb": payload["breadcrumb"],
            "source_endpoints": "|".join(endpoints), "endpoint_count": str(len(endpoints)),
            "max_confidence": f"{payload['max_confidence']:.3f}",
        })
    return sorted(rows, key=lambda row: (row["ontology"], row["guideline_number"], row["term_name"].lower()))

def build_unique_rows(endpoint_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped = {}
    for row in endpoint_rows:
        key = (row["ontology"], row["term_name"])
        if key not in grouped:
            grouped[key] = {
                "ontology": row["ontology"], "term_name": row["term_name"], "term_uri": row["term_uri"],
                "sanitized_name": row["sanitized_name"], "breadcrumb": row["breadcrumb"],
                "descendant_count": row["descendant_count"], "source_endpoints": set(),
                "guideline_numbers": set(), "max_confidence": 0.0,
            }
        grouped[key]["source_endpoints"].add(row["normalized_endpoint"])
        grouped[key]["guideline_numbers"].update(split_pipe_values(row["guideline_numbers"]))
        grouped[key]["max_confidence"] = max(grouped[key]["max_confidence"], float(row["confidence"] or 0.0))
    rows = []
    for payload in grouped.values():
        endpoints = sorted(payload["source_endpoints"])
        rows.append({
            "ontology": payload["ontology"], "term_name": payload["term_name"], "term_uri": payload["term_uri"],
            "sanitized_name": payload["sanitized_name"], "breadcrumb": payload["breadcrumb"],
            "descendant_count": payload["descendant_count"], "source_endpoints": "|".join(endpoints),
            "guideline_numbers": "|".join(sort_guidelines(payload["guideline_numbers"])),
            "endpoint_count": str(len(endpoints)), "max_confidence": f"{payload['max_confidence']:.3f}",
        })
    return sorted(rows, key=lambda row: (row["ontology"], row["term_name"].lower()))

def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path

def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path
