import sqlite3
import json
import re
import torch
import os
import threading
from functools import lru_cache

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from HiPO.model.embeddings import EmbeddingModel
from HiPO.model.llm import (
    _repair_unescaped_quotes,
    clean_llm_output,
    extract_reasoning_and_content,
)
from HiPO.model.reranker import CausalReranker
from .utils import setup_logging, load_ontology_embeddings

logger = setup_logging()

_CONTEXT_LEN_ERR_RE = re.compile(
    r"maximum context length is (\d+) tokens.*?(\d+) in the messages",
    flags=re.IGNORECASE | re.DOTALL,
)




def clean_reason(reason: str) -> str:
    """
    Description:
    Strips prefix words like ACCEPTED or REJECTED from the reasoning text.

    Args:
        reason (str): The raw reasoning text.

    Returns:
        str: The cleaned reasoning text.
    """
    r = reason.strip()
    if r.upper().startswith("ACCEPTED"):
        return r[8:].lstrip(" .-:")
    elif r.upper().startswith("REJECTED"):
        return r[8:].lstrip(" .-:")
    return r


def load_analysis_instruction() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "..", "..", "config", "analysis_instruction.txt")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "Step 1: Analyze the input description and determine its exact level of specificity (i.e., whether it is a broad categorical system or a highly specific mechanism)."


ANALYSIS_INSTRUCTION = load_analysis_instruction()


class CandidateEvaluation(BaseModel):
    """Structure for evaluating a single candidate."""
    candidate_name: str = Field(description="The exact name of the candidate being evaluated.")
    reason: str = Field(description="Think step-by-step. Provide the explicit reason for the decision based on the critical rules. DO NOT write 'ACCEPTED' or 'REJECTED' in this field.")
    decision: Literal["ACCEPTED", "REJECTED"] = Field(description="Strictly 'ACCEPTED' or 'REJECTED' based on the reasoning above.")


class BatchVerificationResult(BaseModel):
    """Structure for batch evaluation (Mode 2)."""
    analysis: str = Field(description=ANALYSIS_INSTRUCTION)
    evaluation: List[CandidateEvaluation] = Field(description="Step 2: Evaluate EVERY candidate individually. You MUST provide an evaluation object for ALL candidates in the list.")
    selected_candidates: List[str] = Field(description="Step 3: The final list of exact candidate names that pass the evaluation. Return an empty list [] if none match.")


class TextToOntoGraph:
    def __init__(self, categories: List[str],
                ontology_db: str, embedding_model_name: str,
                input_db: str, top_k_rag: int, system_prompt_path: str, threshold_rag: float,
                reranker_model_name: str, threshold_reranker: float,
                mode: int, consensus_runs: int, ontology_definitions: bool, max_concurrency: int, max_retries: int, context_ontology: str,
                reranker_prompt_file: str, reranker_batch_size: int,
                global_definitions_db: Optional[str] = None, top_k_reranker: Optional[int] = None) -> None:
        """
        Description:
        Initializes the TextToOntoGraph pipeline orchestration.

        Args:
            categories (List[str]): List of prefixed ontology category names.
            ontology_db (str): Path to ontology SQLite database.
            embedding_model_name (str): Embedding model identifier.
            input_db (str): Path to input/target embedding cache database.
            top_k_rag (int): Number of neighbors retrieved via ANN.
            system_prompt_path (str): Path to system prompt file.
            threshold_rag (float): Score threshold for ANN retrieval candidates.
            reranker_model_name (str): Reranker model identifier.
            threshold_reranker (float): Score threshold for reranking candidates.
            mode (int): Inference mode (1 for individual, 2 for batch).
            consensus_runs (int): Number of LLM verification runs per candidate.
            ontology_definitions (bool): Whether to include node definitions in prompt.
            max_concurrency (int): Maximum threads/calls in parallel.
            max_retries (int): Retry limit for API exceptions.
            context_ontology (str): Additional context about the ontology.
            reranker_prompt_file (str): Path to instruction file for the reranker.
            reranker_batch_size (int): Reranker evaluation batch size.
            global_definitions_db (Optional[str]): Optional database of global node definitions. Defaults to None.
            top_k_reranker (Optional[int]): Optional limit of top candidates kept after reranking. Defaults to None.

        Returns:
            None
        """
        self.categories = categories
        self.prefix = ""
        if categories:
            first_cat = categories[0]
            if ":" in first_cat:
                self.prefix = first_cat.split(":")[0] + ":"
        self.ontology_definitions = ontology_definitions
        self.include_superiors = True
        self.max_retries = max_retries
        self.context_ontology = context_ontology
        self._description_counter = 0
        self.db_lock = threading.Lock()
        self.counter_lock = threading.Lock()
        self.reranker_lock = threading.Lock()
        self.embedder_lock = threading.Lock()

        self.ontology_db = ontology_db
        self.input_db = input_db
        self._input_db_conn = sqlite3.connect(self.input_db, check_same_thread=False) if self.input_db else None

        self.embedder = EmbeddingModel(embedding_model_name)
        self.ontology_embeddings = load_ontology_embeddings(self.ontology_db)
        self.top_k_rag = top_k_rag
        self.threshold_rag = threshold_rag
        self.reranker_model_name = reranker_model_name
        self.threshold_reranker = threshold_reranker
        self.top_k_reranker = top_k_reranker
        self.reranker_batch_size = reranker_batch_size
        self.mode = mode
        self.consensus_runs = consensus_runs
        self.max_concurrency = max_concurrency

        self.reranker = None
        self.reranker_model = None
        self.reranker_tokenizer = None
        self.reranker_instruction = ""
        if self.threshold_reranker > 0.0:
            with open(reranker_prompt_file, 'r', encoding='utf-8') as f:
                self.reranker_instruction = f.read().strip()
            self.reranker = CausalReranker(
                model_name=self.reranker_model_name,
                instruction=self.reranker_instruction,
                batch_size=self.reranker_batch_size,
            )
            self.reranker_model = self.reranker.model
            self.reranker_tokenizer = self.reranker.tokenizer

        self.nodes = {}
        self.parent_map = {}
        self.children_map = {}  # name -> list[child_name]
        self.prompt_map = {}

        # URI to name map for global definitions
        self._uri_to_name = {}

        for item in self.ontology_embeddings:
            name = item['name']
            uri = item.get('uri')
            if uri:
                self._uri_to_name[str(uri)] = name

            parents = item.get('parents', [])
            if not parents and item.get('parent'):
                parents = [item.get('parent')]

            self.nodes[name] = {
                'sanitized_name': item['sanitized_name'],
                'parents': parents,
                'uri': uri
            }
            self.prompt_map[name] = item.get('def', '')
            self.parent_map[name] = parents
            for p in parents:
                if p:
                    c_list = self.children_map.setdefault(p, [])
                    if name not in c_list:
                        c_list.append(name)

        # Optionally load global definitions (overrides default ones)
        if global_definitions_db and os.path.exists(global_definitions_db):
            logger.info(f"Loading external definitions from {global_definitions_db}...")
            try:
                with sqlite3.connect(global_definitions_db) as conn_def:
                    cursor_def = conn_def.cursor()
                    cursor_def.execute("CREATE TABLE IF NOT EXISTS global_definitions (uri TEXT PRIMARY KEY, definition TEXT)")
                    cursor_def.execute("SELECT uri, definition FROM global_definitions")
                    count = 0
                    for r_uri, r_def in cursor_def:
                        name = self._uri_to_name.get(str(r_uri))
                        if name:
                            # Only overwrite if r_def is more complete or not empty
                            if r_def and len(r_def.strip()) > 5:
                                self.prompt_map[name] = r_def
                                count += 1
                    logger.info(f"Successfully loaded and matched {count} global definitions.")
            except Exception as e:
                logger.warning(f"Error loading global definitions DB: {e}")

        # Pre-compute ontology matrix once (never changes)
        self._ontology_matrix = torch.stack([item['vector'] for item in self.ontology_embeddings])

        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            self.system_prompt_content = f.read()

        if self.context_ontology:
            self.system_prompt_content += f"\n\nContext about the ontology:\n{self.context_ontology}"

        self.parser = PydanticOutputParser(pydantic_object=CandidateEvaluation)

        self.description_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt_content),
            ("user", f"{ANALYSIS_INSTRUCTION}\n\nDescription: {{description}}")
        ])

        user_prompt = "Description: {description}\nDescription Analysis: {description_analysis}\nCandidate to evaluate: {candidate_name}"
        user_prompt += "\nCandidate Hierarchy: {candidate_hierarchy}"
        if self.ontology_definitions:
            user_prompt += "\nCandidate Definition: {candidate_definition}"

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt_content + "\n\n{format_instructions}"),
            ("user", user_prompt)
        ])

        # Mode 2: store prompt template once, compose with llm at call time
        self._mode2_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt_content + "\n\n{format_instructions}"),
            ("user", "Candidates:\n{candidates}\n\nDescription to evaluate: {description}")
        ])

        # Mode 1: Prebuild the prompt template
        self._mode1_prompt = self.prompt_template
        self._mode1_format_instructions = self.parser.get_format_instructions()

        self._mode2_parser = PydanticOutputParser(pydantic_object=BatchVerificationResult)
        self._mode2_format_instructions = self._mode2_parser.get_format_instructions()

        # Precompute constants and runners
        self.consensus_threshold = 1 if self.consensus_runs == 2 else (self.consensus_runs // 2) + 1
        self._clean_llm_runner = RunnableLambda(clean_llm_output)

    def close(self) -> None:
        """
        Description:
        Clean up persistent resources.

        Args:
            None

        Returns:
            None
        """
        if self._input_db_conn:
            self._input_db_conn.close()
        if getattr(self, "reranker", None) is not None:
            self.reranker.close()

    def _build_retry_llm(self, llm: Any, error: Exception, retry_idx: int) -> Any:
        """
        Adapts sampling or completion limits on retry when thinking hits context bounds or loops.
        """
        err_str = str(error)
        bind_kwargs: Dict[str, Any] = {}

        ctx_match = _CONTEXT_LEN_ERR_RE.search(err_str)
        if ctx_match:
            ctx_limit = int(ctx_match.group(1))
            prompt_tokens = int(ctx_match.group(2))
            safe_completion = max(512, ctx_limit - prompt_tokens - 64)
            bind_kwargs["max_tokens"] = safe_completion
            logger.warning(
                "Context length %d exceeded by prompt (%d tokens); retrying with max_tokens=%d.",
                ctx_limit, prompt_tokens, safe_completion,
            )

        is_truncated_thinking = "finish_reason='length'" in err_str or "exhausted max_tokens" in err_str
        cur_temp = getattr(llm, "temperature", None)
        if cur_temp is not None and cur_temp < 0.3:
            bind_kwargs["temperature"] = min(0.8, 0.6 + 0.1 * retry_idx)
        elif is_truncated_thinking:
            bind_kwargs["presence_penalty"] = 1.5

        if is_truncated_thinking and retry_idx == self.max_retries - 1 and retry_idx >= 1:
            existing_extra = dict(getattr(llm, "extra_body", None) or {})
            chat_kwargs = dict(existing_extra.get("chat_template_kwargs") or {})
            if chat_kwargs.get("enable_thinking") is True:
                chat_kwargs["enable_thinking"] = False
                existing_extra["chat_template_kwargs"] = chat_kwargs
                bind_kwargs["extra_body"] = existing_extra
                logger.warning(
                    "Thinking repeatedly exhausted token budget on one item; disabling thinking on final retry."
                )

        return llm.bind(**bind_kwargs) if bind_kwargs else llm

    def _parse_message_with_thinking(self, message: Any, parser: Any = None) -> Tuple[Any, str]:
        """Extracts `(parsed_or_text, thinking_text)` from a raw LLM message."""
        thinking, _ = extract_reasoning_and_content(message)
        cleaned = clean_llm_output(message)
        parsed = parser.parse(cleaned) if parser is not None else cleaned
        return parsed, thinking

    def _invoke_batch_with_thinking(
        self,
        prompt_template: ChatPromptTemplate,
        llm: Any,
        inputs: List[Dict[str, Any]],
        parser: Any = None,
        label_fn: Optional[Callable[[int], str]] = None,
    ) -> List[Any]:
        """
        Invokes `prompt_template | llm` concurrently, preserves reasoning traces, and retries
        parse/truncation/context-length failures per item.
        """
        if not inputs:
            return []

        raw_chain = prompt_template | llm
        raw_results = raw_chain.batch(
            inputs,
            config={"max_concurrency": self.max_concurrency},
            return_exceptions=True,
        )

        outputs: List[Any] = [None] * len(inputs)
        for i, raw_res in enumerate(raw_results):
            if not isinstance(raw_res, Exception):
                try:
                    outputs[i] = self._parse_message_with_thinking(raw_res, parser=parser)
                    continue
                except Exception as parse_err:
                    last_err: Exception = parse_err
            else:
                last_err = raw_res

            item_label = label_fn(i) if label_fn else f"Item {i + 1}"
            for retry in range(self.max_retries):
                try:
                    retry_llm = self._build_retry_llm(llm, last_err, retry)
                    retry_msg = (prompt_template | retry_llm).invoke(inputs[i])
                    outputs[i] = self._parse_message_with_thinking(retry_msg, parser=parser)
                    last_err = None
                    break
                except Exception as retry_err:
                    last_err = retry_err
                    if retry == self.max_retries - 1:
                        logger.warning(f"{item_label}: All {self.max_retries} retries failed - {retry_err}")

            if last_err is not None:
                outputs[i] = last_err

        return outputs

    def _rerank_candidates(self, description: str, candidates: list) -> list:
        """
        Description:
        Applies the reranker model to filter candidates by score threshold. Returns filtered list.

        Args:
            description (str): The raw text description.
            candidates (list): A list of dictionary candidates to rerank.

        Returns:
            list: The filtered list of candidates.
        """
        if self.threshold_reranker <= 0.0 or not candidates or self.reranker is None:
            return candidates

        pairs = [
            (description, f"{c['sanitized_name']}\n<Hierarchy>: {self.get_breadcrumbs(c['name'])}")
            for c in candidates
        ]
        scores = self.reranker.score_pairs(pairs, instruction=self.reranker_instruction)

        passed_candidates = []
        for score, cand in zip(scores, candidates):
            if score >= self.threshold_reranker:
                cand['rerank_score'] = score
                passed_candidates.append(cand)

        # Apply top_k_reranker limit if specified
        if self.top_k_reranker is not None and self.top_k_reranker > 0:
            passed_candidates.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)
            passed_candidates = passed_candidates[:self.top_k_reranker]

        return passed_candidates

    def get_breadcrumbs(self, name: str, inline: bool = False) -> str:
        """
        Description:
        Constructs the hierarchical path(s) (breadcrumbs) for a node across all parents.
        Cached per unique name.

        Args:
            name (str): Node name to build the breadcrumb path(s) for.
            inline (bool): If True, join multiple paths on a single line with ' | ';
                otherwise separate them with newlines. Defaults to False.

        Returns:
            str: The joined breadcrumb hierarchical string path(s).
        """
        paths = self._get_breadcrumbs_cached(name)
        if len(paths) <= 1:
            return paths[0] if paths else name
        sep = " | " if inline else "\n"
        return sep.join(f"Path {i + 1}: {p}" for i, p in enumerate(paths))

    @lru_cache(maxsize=None)
    def _get_breadcrumbs_cached(self, name: str) -> Tuple[str, ...]:
        """
        Description:
        LRU cached internal builder returning all unique root-to-node breadcrumb paths.

        Args:
            name (str): Node name to build the breadcrumb paths for.

        Returns:
            Tuple[str, ...]: Ordered unique root-to-node path strings.
        """
        max_paths = 32
        raw_paths: List[List[str]] = []

        def dfs(curr: str, current_path: List[str], active: set[str]) -> None:
            if len(raw_paths) >= max_paths:
                return
            if not curr or curr in active:
                if current_path:
                    raw_paths.append(current_path)
                return
            active.add(curr)
            next_path = current_path + [curr]
            parents = [p for p in self.parent_map.get(curr, []) if p]
            if not parents:
                raw_paths.append(next_path)
            else:
                for p in parents:
                    dfs(p, next_path, active)
                    if len(raw_paths) >= max_paths:
                        break
            active.remove(curr)

        dfs(name, [], set())
        if not raw_paths:
            return (self.nodes.get(name, {}).get("sanitized_name", name),)

        formatted: List[str] = []
        seen: set[str] = set()
        for path in raw_paths:
            labels = [self.nodes.get(n, {}).get("sanitized_name", n) for n in reversed(path)]
            joined = " > ".join(labels)
            if joined not in seen:
                seen.add(joined)
                formatted.append(joined)
        return tuple(formatted)

    def execute(self, llm: Any, description: str, std_desc: Optional[str] = None, alpha: Optional[float] = None) -> Tuple[dict, list, list]:
        """
        Description:
        Executes the retrieval, reranking, and verification flow.

        Args:
            llm (Any): The LLM client model instance.
            description (str): Raw text description string.
            std_desc (Optional[str]): Optional standardized description string. Defaults to None.
            alpha (Optional[float]): Interpolation factor between query and std embeddings. Defaults to None.

        Returns:
            Tuple[dict, list, list]: (results dictionary mapping nodes to 1, reasoning trace log list, sorted list of hits)
        """
        with self.counter_lock:
            self._description_counter += 1
            logger.info(f"\n{self._description_counter}. Description: {description}")

        description_analysis = ""
        if self.mode == 1:
            try:
                analysis_outcomes = self._invoke_batch_with_thinking(
                    self.description_analysis_prompt,
                    llm,
                    [{"description": description}],
                    parser=None,
                    label_fn=lambda _: f"Mode 1 description_analysis ('{description}')",
                )
                analysis_outcome = analysis_outcomes[0]
                if isinstance(analysis_outcome, Exception):
                    raise analysis_outcome
                description_analysis = str(analysis_outcome[0]).strip()
            except Exception as e:
                logger.warning(f"Failed to generate specificity analysis: {e}")
                description_analysis = "Analysis failed or unavailable."

        results = {}
        reasoning_trace = []

        def get_vec_for_text(text: str) -> torch.Tensor:
            """
            Description:
            Fetches embedding vector for text from DB cache or embedder model.

            Args:
                text (str): The text string to embed.

            Returns:
                torch.Tensor: The float32 embedding tensor.
            """
            vec = None
            if self._input_db_conn:
                try:
                    with self.db_lock:
                        cursor_in = self._input_db_conn.cursor()
                        cursor_in.execute("SELECT vector FROM input_embeddings WHERE text = ?", (text,))
                        row = cursor_in.fetchone()
                    if row:
                        vec = torch.frombuffer(bytearray(row[0]), dtype=torch.float32)
                except Exception as e:
                    logger.warning(f"Could not fetch cached embedding for {text}: {e}")
            if vec is None:
                with self.embedder_lock:
                    self.embedder.load()
                    vec_bytes = self.embedder.get_embedding(text)
                vec = torch.frombuffer(bytearray(vec_bytes), dtype=torch.float32)
            return vec

        query_vec = get_vec_for_text(description)

        if std_desc and alpha is not None:
            std_vec = get_vec_for_text(std_desc)
            query_vec = alpha * query_vec + (1.0 - alpha) * std_vec

        scores = torch.matmul(self._ontology_matrix, query_vec)

        _, top_indices = torch.topk(scores, k=min(self.top_k_rag, len(scores)))
        top_indices = top_indices.tolist()

        rag_candidates = []
        for idx in top_indices:
            item = self.ontology_embeddings[idx]
            rag_candidates.append({
                'name': item['name'],
                'sanitized_name': item['sanitized_name'],
                'def': item['def'],
                'ann_score': scores[idx].item(),
                'source': 'rag'
            })

        rag_candidates.sort(key=lambda x: x['ann_score'], reverse=True)
        rag_candidates = [c for c in rag_candidates if c['ann_score'] >= self.threshold_rag]

        pool_map = {}
        for i, c in enumerate(rag_candidates):
            c['source'] = f"rag {i+1}/{len(rag_candidates)}"
            pool_map[c['name']] = c

        top_k_candidates = list(pool_map.values())

        top_k_candidates = self._rerank_candidates(description, top_k_candidates)

        if self.mode == 2:
            return self._execute_mode_2(llm, description, top_k_candidates)

        verified_nodes = set()
        evaluated_nodes = set()
        threshold = self.consensus_threshold

        current_candidates = top_k_candidates
        iteration = 1

        while current_candidates:
            batch_inputs = []
            batch_metadata = []
            valid_candidates = []

            for idx, cand in enumerate(current_candidates):
                name = cand['name']
                if name not in self.nodes or name in verified_nodes or name in evaluated_nodes:
                    continue

                evaluated_nodes.add(name)
                node_data = self.nodes[name]
                node_sanitized_name = node_data.get('sanitized_name', name)

                if self.ontology_definitions and name not in self.prompt_map:
                    logger.warning(f"Definition for node '{name}' not found")

                valid_candidates.append((idx, cand))

                for run_i in range(self.consensus_runs):
                    ctx = {
                        "description": description,
                        "description_analysis": description_analysis,
                        "candidate_name": node_sanitized_name,
                        "format_instructions": self._mode1_format_instructions
                    }
                    ctx["candidate_hierarchy"] = self.get_breadcrumbs(name)
                    if self.ontology_definitions:
                        ctx["candidate_definition"] = self.prompt_map.get(name, "No definition available.")
                    batch_inputs.append(ctx)
                    batch_metadata.append({"name": name, "run_i": run_i, "idx": idx, "cand": cand})

            if not valid_candidates:
                break

            batch_results_all = []
            if batch_inputs:
                batch_results_all = self._invoke_batch_with_thinking(
                    self._mode1_prompt,
                    llm,
                    batch_inputs,
                    parser=self.parser,
                    label_fn=lambda idx: f"Node {batch_metadata[idx]['name']} Run {batch_metadata[idx]['run_i']+1}",
                )

            for result in batch_results_all:
                if isinstance(result, Exception):
                    raise RuntimeError(f"Failed to parse LLM response in Mode 1 after retries: {result}")

            candidate_votes = {}
            candidate_reasoning = {}
            candidate_thinking = {}
            for i, meta in enumerate(batch_metadata):
                name = meta["name"]
                run_i = meta["run_i"]

                if name not in candidate_votes:
                    candidate_votes[name] = 0
                    candidate_reasoning[name] = []
                    candidate_thinking[name] = []

                outcome = batch_results_all[i]
                if isinstance(outcome, Exception):
                    candidate_reasoning[name].append(f"Run {run_i+1}: Parse error")
                elif outcome is not None:
                    result, thinking_text = outcome
                    if result.decision == 'ACCEPTED':
                        candidate_votes[name] += 1
                    reason_clean = clean_reason(result.reason)
                    candidate_reasoning[name].append(f"Run {run_i+1}: {result.decision} - {reason_clean}")
                    candidate_thinking[name].append(thinking_text)
                else:
                    candidate_reasoning[name].append(f"Run {run_i+1}: Error")

            next_round_candidates_map = {}

            for idx, cand in valid_candidates:
                name = cand['name']
                yes_votes = candidate_votes.get(name, 0)
                accumulated_reasoning = candidate_reasoning.get(name, [])
                accumulated_thinking = candidate_thinking.get(name, [])
                status = "ACCEPTED" if yes_votes >= threshold else "REJECTED"

                trace_entry = {
                    "candidate": name,
                    "decision": status,
                    "votes": f"{yes_votes}/{self.consensus_runs}",
                    "source": cand['source'],
                    "runs": accumulated_reasoning
                }
                if any(accumulated_thinking):
                    trace_entry["thinking_runs"] = accumulated_thinking
                reasoning_trace.append(trace_entry)

                if yes_votes >= threshold:
                    queue = [name]
                    while queue:
                        c = queue.pop(0)
                        if c in verified_nodes:
                            continue
                        verified_nodes.add(c)
                        prefixed_c = f"{self.prefix}{c}"
                        if prefixed_c in self.categories:
                            results[prefixed_c] = 1 # Store only hits
                        for p in self.parent_map.get(c, []):
                            if p and p not in verified_nodes:
                                queue.append(p)
                    # Enqueue unasked children of the accepted node
                    for child_name in self.children_map.get(name, []):
                        if child_name not in verified_nodes and child_name not in evaluated_nodes and child_name not in next_round_candidates_map:
                            child_info = self.nodes.get(child_name)
                            if child_info:
                                next_round_candidates_map[child_name] = {
                                    'name': child_name,
                                    'sanitized_name': child_info.get('sanitized_name', child_name),
                                    'def': self.prompt_map.get(child_name, ''),
                                    'source': f'inferior of: {name}',
                                    'ann_score': 0.0
                                }
                else:
                    for parent_name in self.parent_map.get(name, []):
                        if parent_name and parent_name not in verified_nodes and parent_name not in evaluated_nodes and parent_name not in next_round_candidates_map:
                            node_info = self.nodes.get(parent_name)
                            if node_info:
                                next_round_candidates_map[parent_name] = {
                                    'name': parent_name,
                                    'sanitized_name': node_info.get('sanitized_name', parent_name),
                                    'def': self.prompt_map.get(parent_name, ''),
                                    'source': f'superior of: {name}',
                                    'ann_score': 0.0
                                }

            current_candidates = list(next_round_candidates_map.values())
            # Pre-filter inferior/superior candidates with the reranker before sending to LLM
            if current_candidates:
                current_candidates = self._rerank_candidates(description, current_candidates)
            iteration += 1

        return results, reasoning_trace, sorted(results.keys())

    def _execute_mode_2(self, llm: Any, description: str, top_k_candidates: List[Dict]) -> Tuple[dict, list, list]:
        """
        Description:
        Executes batch verification using the BatchVerificationResult model.

        Args:
            llm (Any): The LLM client model instance.
            description (str): Raw text description string.
            top_k_candidates (List[Dict]): Candidates retrieved from RAG retrieval.

        Returns:
            Tuple[dict, list, list]: (results, reasoning trace, final verified hits)
        """
        results = {}
        reasoning_trace = []

        verified_nodes = set()
        evaluated_nodes = set()

        current_candidates = top_k_candidates
        iteration = 1

        final_selection_with_info = []

        while current_candidates:
            filtered_candidates = []
            for c in current_candidates:
                if c['name'] not in evaluated_nodes and c['name'] not in verified_nodes:
                    filtered_candidates.append(c)

            if not filtered_candidates:
                break

            candidates = filtered_candidates
            for c in candidates:
                evaluated_nodes.add(c['name'])

            candidate_names = [c['name'] for c in candidates]
            # O(1) position lookup instead of O(n) list.index() in the logging loop
            candidate_pos = {name: i + 1 for i, name in enumerate(candidate_names)}
            total_cands = len(candidate_names)

            s2n = {}
            s_names = []
            for c in candidates:
                name = c['name']
                node_data = self.nodes.get(name, {})
                s_name = node_data.get('sanitized_name', name)
                s2n[s_name] = name
                s_names.append(s_name)

            candidate_strings = []
            for s_name, c in zip(s_names, candidates):
                cand_str = f"- {s_name}"
                cand_str += f" (Hierarchy: {self.get_breadcrumbs(c['name'], inline=True)})"
                if self.ontology_definitions:
                    cand_str += f": {self.prompt_map.get(c['name'], c.get('def', ''))}"
                candidate_strings.append(cand_str)

            candidates_str = "\n".join(candidate_strings)

            threshold = self.consensus_threshold
            candidate_votes = {name: 0 for name in candidate_names}

            invoke_inputs = [
                {
                    "description": description,
                    "candidates": candidates_str,
                    "format_instructions": self._mode2_format_instructions
                }
                for _ in range(self.consensus_runs)
            ]

            try:
                batch_results = self._invoke_batch_with_thinking(
                    self._mode2_prompt,
                    llm,
                    invoke_inputs,
                    parser=self._mode2_parser,
                    label_fn=lambda idx: f"Mode 2 Iter {iteration} Run {idx+1}",
                )

                for result in batch_results:
                    if isinstance(result, Exception):
                        raise RuntimeError(f"Failed to parse LLM response in Mode 2 after retries: {result}")

                runs_details = []
                for run_i, outcome in enumerate(batch_results):
                    if isinstance(outcome, Exception) or outcome is None:
                        runs_details.append({
                            "run": run_i + 1,
                            "error": str(outcome) if outcome is not None else "Unknown error"
                        })
                        continue

                    result, thinking_text = outcome
                    evals_details = []
                    for ev in result.evaluation:
                        evals_details.append({
                            "candidate": ev.candidate_name,
                            "decision": ev.decision,
                            "reason": ev.reason
                        })

                    run_entry = {
                        "run": run_i + 1,
                        "analysis": result.analysis,
                        "evaluations": evals_details,
                        "selected_candidates": result.selected_candidates
                    }
                    if thinking_text:
                        run_entry["thinking"] = thinking_text
                    runs_details.append(run_entry)

                    candidate_name_map = {}
                    for c in candidates:
                        candidate_name_map[c['name'].strip().rstrip(':')] = c
                        if 'sanitized_name' in c:
                            candidate_name_map[c['sanitized_name'].strip().rstrip(':')] = c

                    for selected in result.selected_candidates:
                        clean_selected = selected.strip().rstrip(':')
                        matched_s_name = s2n.get(clean_selected)
                        if matched_s_name:
                            candidate_votes[matched_s_name] += 1
                        elif clean_selected in candidate_votes:
                            candidate_votes[clean_selected] += 1
            except Exception as e:
                logger.error(f"Mode 2 Iter {iteration} batch execution failed: {e}")
                raise e

            final_selected_names = [name for name, votes in candidate_votes.items() if votes >= threshold]
            rejected_names = [name for name, votes in candidate_votes.items() if votes < threshold]

            iteration_summary = []
            for name, votes in candidate_votes.items():
                pos = candidate_pos[name]
                status = "ACCEPTED" if votes >= threshold else "REJECTED"
                iteration_summary.append({
                    "candidate": name,
                    "position": f"{pos}/{total_cands}",
                    "votes": f"{votes}/{self.consensus_runs}",
                    "decision": status
                })

            reasoning_trace.append({
                "iteration": iteration,
                "candidates_sent": candidate_names,
                "runs": runs_details,
                "summary": iteration_summary
            })

            for selected in final_selected_names:
                queue = [selected]
                while queue:
                    c = queue.pop(0)
                    if c in verified_nodes:
                        continue
                    verified_nodes.add(c)
                    prefixed_c = f"{self.prefix}{c}"
                    if prefixed_c in self.categories:
                        results[prefixed_c] = 1 # Store only hits
                    for p in self.parent_map.get(c, []):
                        if p and p not in verified_nodes:
                            queue.append(p)

            for name in final_selected_names:
                 cand_obj = next((c for c in candidates if c['name'] == name), None)
                 source = cand_obj['source'] if cand_obj else "unknown"
                 final_selection_with_info.append(f"{name} ({source})")

            next_round_candidates_map = {}
            # Enqueue unasked children of every accepted node
            for name in final_selected_names:
                for child_name in self.children_map.get(name, []):
                    if child_name not in verified_nodes and child_name not in evaluated_nodes and child_name not in next_round_candidates_map:
                        child_info = self.nodes.get(child_name)
                        if child_info:
                            next_round_candidates_map[child_name] = {
                                'name': child_name,
                                'sanitized_name': child_info.get('sanitized_name', child_name),
                                'def': self.prompt_map.get(child_name, ''),
                                'source': f'inferior_of_{name}',
                                'ann_score': 0.0
                            }
            if self.include_superiors:
                for name in rejected_names:
                    for parent_name in self.parent_map.get(name, []):
                        if parent_name and parent_name not in verified_nodes and parent_name not in evaluated_nodes and parent_name not in next_round_candidates_map:
                            node_info = self.nodes.get(parent_name)
                            if node_info:
                                next_round_candidates_map[parent_name] = {
                                    'name': parent_name,
                                    'sanitized_name': node_info.get('sanitized_name', parent_name),
                                    'def': self.prompt_map.get(parent_name, ''),
                                    'source': f'superior_of_{name}',
                                    'ann_score': 0.0
                                }

            current_candidates = list(next_round_candidates_map.values())
            # Pre-filter inferior/superior candidates with the reranker before sending to LLM
            if current_candidates:
                current_candidates = self._rerank_candidates(description, current_candidates)
            iteration += 1

        return results, reasoning_trace, sorted(results.keys())
