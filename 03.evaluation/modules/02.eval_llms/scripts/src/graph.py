from __future__ import annotations
import sqlite3
import os
from functools import lru_cache

from typing import List, Dict, Literal, Any, Optional, Union, Tuple, Set
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from langchain_core.callbacks import BaseCallbackHandler

from .utils import setup_logging, load_ontology_embeddings

logger = setup_logging()

class TokenUsageCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        """
        Description:
        Initializes the TokenUsageCallbackHandler instance.

        Args:
            None

        Returns:
            None
        """
        super().__init__()
        self.completion_tokens = 0
        self.num_calls = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """
        Description:
        Callback method triggered when the LLM finishes execution, extracting and accumulating completion tokens.

        Args:
            response (Any): The LLM response object.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            None
        """
        try:
            self.num_calls += 1
            # Check llm_output first
            if hasattr(response, "llm_output") and response.llm_output and "token_usage" in response.llm_output:
                usage = response.llm_output["token_usage"]
                if usage:
                    self.completion_tokens += usage.get("completion_tokens", usage.get("output_tokens", 0))
                    return

            # Fallback to generations check
            if hasattr(response, "generations") and response.generations:
                for gen_list in response.generations:
                    for gen in gen_list:
                        if hasattr(gen, "message") and hasattr(gen.message, "response_metadata"):
                            metadata = gen.message.response_metadata
                            if "token_usage" in metadata:
                                usage = metadata["token_usage"]
                                if usage:
                                    self.completion_tokens += usage.get("completion_tokens", usage.get("output_tokens", 0))
        except Exception as e:
            pass

    @property
    def avg_completion_tokens(self) -> float:
        """
        Description:
        Calculates the average completion tokens consumed per LLM invocation.

        Args:
            None

        Returns:
            float: The average completion tokens count.
        """
        if self.num_calls == 0:
            return 0.0
        return self.completion_tokens / self.num_calls

def clean_llm_output(message: Any) -> str:
    """
    Description:
    Removes <think> tags, markdown code blocks, and fixes formatting in the LLM's raw response to ensure a valid JSON output.

    Args:
        message (Any): The raw response message from the LLM.

    Returns:
        str: The sanitized and formatted JSON string.
    """
    import re
    import json
    text = message.content if hasattr(message, "content") else str(message)

    # Remove <think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>")[0].strip()

    text = text.strip()

    # Extract JSON object if surrounded by text
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx+1]

    # Remove code block markers
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    # Remove trailing commas inside JSON objects/arrays
    text = re.sub(r',\s*([\]}])', r'\1', text)

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # Check Mode 2 BatchVerificationResult
            if "evaluation" in data and isinstance(data["evaluation"], list):
                for item in data["evaluation"]:
                    if isinstance(item, dict):
                        decision = item.get("decision")
                        reason = item.get("reason", "")
                        if not decision or str(decision).upper() not in ["ACCEPTED", "REJECTED"]:
                            reason_upper = str(reason).upper().strip()
                            if reason_upper.startswith("ACCEPTED"):
                                item["decision"] = "ACCEPTED"
                                item["reason"] = re.sub(r'^accepted[\s.,:-]*', '', reason, flags=re.IGNORECASE).strip()
                            elif reason_upper.startswith("REJECTED"):
                                item["decision"] = "REJECTED"
                                item["reason"] = re.sub(r'^rejected[\s.,:-]*', '', reason, flags=re.IGNORECASE).strip()

            # Check Mode 1 CandidateEvaluation
            elif "candidate_name" in data:
                decision = data.get("decision")
                reason = data.get("reason", "")
                if not decision or str(decision).upper() not in ["ACCEPTED", "REJECTED"]:
                    reason_upper = str(reason).upper().strip()
                    if reason_upper.startswith("ACCEPTED"):
                        data["decision"] = "ACCEPTED"
                        data["reason"] = re.sub(r'^accepted[\s.,:-]*', '', reason, flags=re.IGNORECASE).strip()
                    elif reason_upper.startswith("REJECTED"):
                        data["decision"] = "REJECTED"
                        data["reason"] = re.sub(r'^rejected[\s.,:-]*', '', reason, flags=re.IGNORECASE).strip()

        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        # Fallback to the raw string if JSON parsing/fixing fails
        return text.strip()


def clean_reason(reason: str) -> str:
    """
    Description:
    Cleans a reasoning string by removing leading 'ACCEPTED' or 'REJECTED' indicators.

    Args:
        reason (str): The raw reasoning string.

    Returns:
        str: The cleaned reasoning string.
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
    def __init__(
        self,
        categories: List[str],
        ontology_db: str,
        system_prompt_path: str,
        mode: int,
        consensus_runs: int,
        ontology_definitions: bool,
        max_concurrency: int,
        max_retries: int,
        context_ontology: str,
        global_definitions_db: Optional[str] = None,
    ) -> None:
        """
        Description:
        Initializes the TextToOntoGraph verification processor.

        Args:
            categories (List[str]): List of categories to extract ontology terms for.
            ontology_db (str): Path to the SQLite ontology/embeddings database.
            system_prompt_path (str): Path to the text file containing the system prompt template.
            mode (int): Evaluation mode (1 for individual term verification, 2 for batch verification).
            consensus_runs (int): Number of consensus voting runs to perform.
            ontology_definitions (bool): Whether to include definitions of terms in LLM context prompts.
            max_concurrency (int): Maximum concurrent LLM requests.
            max_retries (int): Maximum formatting/parsing retry attempts for LLM errors.
            context_ontology (str): Additional domain context to append to the system prompt.
            global_definitions_db (Optional[str]): Optional path to an external DB overriding standard definitions. Defaults to None.

        Returns:
            None
        """
        self.categories = categories
        self.ontology_definitions = ontology_definitions
        self.include_superiors = True
        self.max_retries = max_retries
        self.context_ontology = context_ontology
        self._description_counter = 0

        self.ontology_db = ontology_db

        self.ontology_embeddings = load_ontology_embeddings(self.ontology_db)
        self.mode = mode
        self.consensus_runs = consensus_runs
        self.max_concurrency = max_concurrency

        self.nodes = {}
        self.parent_map = {}
        self.prompt_map = {}

        # URI to name map for global definitions
        self._uri_to_name = {}

        for item in self.ontology_embeddings:
            name = item['name']
            uri = item.get('uri')
            if uri:
                self._uri_to_name[str(uri)] = name

            self.nodes[name] = {
                'sanitized_name': item['sanitized_name'],
                'parent': item.get('parent'),
                'uri': uri
            }
            self.prompt_map[name] = item.get('def', '')
            parent_name = item.get('parent')
            if parent_name:
                self.parent_map[name] = parent_name

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

        system_suffix_mode1 = (
            "\n\n{format_instructions}\n\n"
            "*** CRITICAL INSTRUCTION ***\n"
            "Do NOT output the JSON schema itself (do NOT include keys like 'properties', 'type', 'title', 'description', or 'required' at the root level of your response).\n"
            "You must output a single, flat JSON object containing ONLY the keys 'candidate_name', 'reason', and 'decision' with their respective values filled in.\n"
            "Expected output format example:\n"
            "{{\n"
            "  \"candidate_name\": \"exact name of the candidate\",\n"
            "  \"reason\": \"your step-by-step biological reason\",\n"
            "  \"decision\": \"ACCEPTED\"\n"
            "}}"
        )

        system_suffix_mode2 = (
            "\n\n{format_instructions}\n\n"
            "*** CRITICAL INSTRUCTION ***\n"
            "Do NOT output the JSON schema itself (do NOT include keys like 'properties', 'type', 'title', 'description', or 'required' at the root level of your response).\n"
            "You must output a single JSON object containing ONLY the keys 'analysis', 'evaluation', and 'selected_candidates' with their respective values filled in.\n"
            "Expected output format example:\n"
            "{{\n"
            "  \"analysis\": \"your step-by-step biological analysis of specificity\",\n"
            "  \"evaluation\": [\n"
            "    {{\n"
            "      \"candidate_name\": \"exact name of the candidate\",\n"
            "      \"reason\": \"your step-by-step reason\",\n"
            "      \"decision\": \"REJECTED\"\n"
            "    }}\n"
            "  ],\n"
            "  \"selected_candidates\": []\n"
            "}}"
        )

        # Standard clean prompts (default)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt_content + "\n\n{format_instructions}"),
            ("user", user_prompt)
        ])

        self._mode2_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt_content + "\n\n{format_instructions}"),
            ("user", "Candidates:\n{candidates}\n\nDescription to evaluate: {description}")
        ])

        # Clarified prompts (only used on retry fallback)
        self.prompt_template_clarified = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt_content + system_suffix_mode1),
            ("user", user_prompt)
        ])

        self._mode2_prompt_clarified = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt_content + system_suffix_mode2),
            ("user", "Candidates:\n{candidates}\n\nDescription to evaluate: {description}")
        ])

        # Prebuild the prompt templates
        self._mode1_prompt = self.prompt_template
        self._mode1_prompt_clarified = self.prompt_template_clarified
        self._mode1_format_instructions = self.parser.get_format_instructions()

        self._mode2_parser = PydanticOutputParser(pydantic_object=BatchVerificationResult)
        self._mode2_format_instructions = self._mode2_parser.get_format_instructions()

        # Precompute constants and runners
        self.consensus_threshold = 1 if self.consensus_runs == 2 else (self.consensus_runs // 2) + 1
        self._clean_llm_runner = RunnableLambda(clean_llm_output)

    def close(self) -> None:
        """
        Description:
        Cleans up any open resources/handlers.

        Args:
            None

        Returns:
            None
        """
        pass



    def get_breadcrumbs(self, name: str) -> str:
        """
        Description:
        Retrieves the hierarchical path (breadcrumbs string) for a given term name, utilizing a cache.

        Args:
            name (str): The term name.

        Returns:
            str: The breadcrumbs string (e.g. 'root > parent > child').
        """
        return self._get_breadcrumbs_cached(name)

    @lru_cache(maxsize=None)
    def _get_breadcrumbs_cached(self, name: str) -> str:
        """
        Description:
        Helper method that builds and caches the breadcrumbs string for a term.

        Args:
            name (str): The term name.

        Returns:
            str: The breadcrumbs string.
        """
        path = []
        curr = name
        visited = set()
        while curr and curr not in visited:
            visited.add(curr)
            label = self.nodes.get(curr, {}).get('sanitized_name', curr)
            path.append(label)
            curr = self.parent_map.get(curr)
        return " > ".join(reversed(path))

    def execute(
        self,
        llm: Any,
        description: str,
        std_desc: Optional[str] = None,
        alpha: Optional[float] = None,
        direct_candidates: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, int], List[Any], List[str], Dict[str, float]]:
        """
        Description:
        Executes the retrieval, reranking, and verification pipeline for a given toxic effect description.

        Args:
            llm (Any): The LLM engine or wrapper to invoke.
            description (str): The raw toxic effect description.
            std_desc (Optional[str]): The standardized description, if available. Defaults to None.
            alpha (Optional[float]): The alpha retrieval weight parameter, if applicable. Defaults to None.
            direct_candidates (Optional[List[str]]): List of candidate names to evaluate directly.

        Returns:
            Tuple[Dict[str, int], List[Any], List[str], Dict[str, float]]: A tuple containing (hits map, reasoning trace, sorted list of verified term names, token usage summary).
        """
        self._description_counter += 1
        logger.info(f"{self._description_counter}. Description: {description}")

        if direct_candidates is None:
            raise ValueError("direct_candidates must be provided in this evaluation pipeline.")

        # Optimization: return only hits to avoid massive dictionaries in memory
        results = {}
        reasoning_trace = []
        selected_candidates = []

        # Instantiate token usage tracker
        handler = TokenUsageCallbackHandler()

        description_analysis = ""
        if self.mode == 1:
            analysis_chain = self.description_analysis_prompt | llm | self._clean_llm_runner
            try:
                description_analysis = analysis_chain.invoke({"description": description}, config={"callbacks": [handler]})
                if hasattr(description_analysis, 'content'):
                    description_analysis = description_analysis.content
                description_analysis = str(description_analysis).strip()
            except Exception as e:
                logger.warning(f"Failed to generate specificity analysis: {e}")
                description_analysis = "Analysis failed or unavailable."

        # Build the evaluation chains (LangChain) dynamically with the provided llm
        chain = self._mode1_prompt | llm | self._clean_llm_runner | self.parser
        chain_clarified = self._mode1_prompt_clarified | llm | self._clean_llm_runner | self.parser

        # Bypass RAG and Reranker
        top_k_candidates = []
        for name in direct_candidates:
            node_info = self.nodes.get(name)
            if not node_info:
                # Case-insensitive lookup
                matched_name = next((k for k in self.nodes if k.lower().strip() == name.lower().strip()), None)
                if matched_name:
                    name = matched_name
                    node_info = self.nodes.get(name)

            if node_info:
                top_k_candidates.append({
                    'name': name,
                    'sanitized_name': node_info['sanitized_name'],
                    'def': self.prompt_map.get(name, ''),
                    'ann_score': 1.0,
                    'source': 'direct'
                })
            else:
                # Fallback for manually added terms or unknown ones
                top_k_candidates.append({
                    'name': name,
                    'sanitized_name': name.lower().replace('_', ' ').replace('-', ' '),
                    'def': '',
                    'ann_score': 1.0,
                    'source': 'direct_fallback'
                })

        if self.mode == 2:
            return self._execute_mode_2(llm, description, top_k_candidates, direct_candidates=direct_candidates, handler=handler)

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

            batch_results_all = [None] * len(batch_inputs)
            if batch_inputs:
                batch_results_all = chain.batch(batch_inputs, config={"max_concurrency": self.max_concurrency, "callbacks": [handler]}, return_exceptions=True)

            for i, result in enumerate(batch_results_all):
                if isinstance(result, Exception):
                    meta = batch_metadata[i]
                    for retry in range(self.max_retries):
                        try:
                            # Retry using the clarified chain with explicit formatting warnings
                            batch_results_all[i] = chain_clarified.invoke(batch_inputs[i], config={"callbacks": [handler]})
                            break
                        except Exception as retry_e:
                            if retry == self.max_retries - 1:
                                logger.warning(f"Node {meta['name']} Run {meta['run_i']+1}: All {self.max_retries} retries failed - {retry_e}")

            candidate_votes = {}
            candidate_reasoning = {}
            for i, meta in enumerate(batch_metadata):
                name = meta["name"]
                run_i = meta["run_i"]

                if name not in candidate_votes:
                    candidate_votes[name] = 0
                    candidate_reasoning[name] = []

                result = batch_results_all[i]
                if isinstance(result, Exception):
                    candidate_reasoning[name].append(f"Run {run_i+1}: Parse error")
                elif result is not None:
                    if result.decision == 'ACCEPTED':
                        candidate_votes[name] += 1
                    reason_clean = clean_reason(result.reason)
                    hierarchy_str = self.get_breadcrumbs(name)
                    candidate_reasoning[name].append(f"Run {run_i+1}: {result.decision} - {reason_clean}")
                else:
                    candidate_reasoning[name].append(f"Run {run_i+1}: Error")

            next_round_candidates_map = {}

            for idx, cand in valid_candidates:
                name = cand['name']
                yes_votes = candidate_votes.get(name, 0)
                accumulated_reasoning = candidate_reasoning.get(name, [])
                status = "ACCEPTED" if yes_votes >= threshold else "REJECTED"
                iter_prefix = f"[Iter {iteration}] " if iteration > 1 else ""

                reasoning_trace.append({
                    "candidate": name,
                    "decision": status,
                    "votes": f"{yes_votes}/{self.consensus_runs}",
                    "source": cand['source'],
                    "runs": accumulated_reasoning
                })

                if yes_votes >= threshold:
                    curr = name
                    while curr:
                        if curr in verified_nodes:
                            break
                        verified_nodes.add(curr)
                        if curr in self.categories:
                            results[curr] = 1 # Store only hits
                        curr = self.parent_map.get(curr)
                else:
                    if direct_candidates is None:
                        parent_name = self.parent_map.get(name)
                        if parent_name and parent_name not in verified_nodes and parent_name not in evaluated_nodes and parent_name not in next_round_candidates_map:
                            node_info = self.nodes.get(parent_name)
                            if node_info:
                                next_round_candidates_map[parent_name] = {
                                    'name': parent_name,
                                    'sanitized_name': node_info.get('sanitized_name', parent_name),
                                    'def': self.prompt_map.get(parent_name, ''),
                                    'source': f'superior of : {name}',
                                    'ann_score': 0.0
                                }

            current_candidates = list(next_round_candidates_map.values())
            iteration += 1

        token_usage = {
            "avg_completion_tokens": handler.avg_completion_tokens
        }
        return results, reasoning_trace, sorted(results.keys()), token_usage

    def _execute_mode_2(
        self,
        llm: Any,
        description: str,
        top_k_candidates: List[Dict[str, Any]],
        direct_candidates: Optional[List[str]] = None,
        handler: Optional[Any] = None,
    ) -> Tuple[Dict[str, int], List[str], List[str], Dict[str, float]]:
        """
        Description:
        Executes batch verification (Mode 2) for the given candidates and description.

        Args:
            llm (Any): The LLM engine or wrapper to invoke.
            description (str): The raw toxic effect description.
            top_k_candidates (List[Dict[str, Any]]): List of candidate dictionary objects.
            direct_candidates (Optional[List[str]]): Optional list of candidates. Defaults to None.
            handler (Optional[Any]): Optional token usage callback handler. Defaults to None.

        Returns:
            Tuple[Dict[str, int], List[str], List[str], Dict[str, float]]: A tuple containing (hits map, reasoning trace logs, sorted list of verified term names, token usage summary).
        """
        results = {}
        reasoning_trace = []

        verified_nodes = set()
        evaluated_nodes = set()

        current_candidates = top_k_candidates
        iteration = 1

        if handler is None:
            handler = TokenUsageCallbackHandler()

        # Build the chains once
        base_chain = self._mode2_prompt | llm | self._clean_llm_runner | self._mode2_parser
        base_chain_clarified = self._mode2_prompt_clarified | llm | self._clean_llm_runner | self._mode2_parser

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
                cand_str += f" (Hierarchy: {self.get_breadcrumbs(c['name'])})"
                if self.ontology_definitions:
                    cand_str += f": {self.prompt_map.get(c['name'], c.get('def', ''))}"
                candidate_strings.append(cand_str)

            candidates_str = "\n".join(candidate_strings)

            chain = base_chain
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
                batch_results = chain.batch(invoke_inputs, config={"max_concurrency": self.max_concurrency, "callbacks": [handler]}, return_exceptions=True)

                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        for retry in range(self.max_retries):
                            try:
                                # Retry using the clarified chain with explicit formatting warnings
                                batch_results[i] = base_chain_clarified.invoke(invoke_inputs[i], config={"callbacks": [handler]})
                                break
                            except Exception as retry_e:
                                if retry == self.max_retries - 1:
                                    logger.warning(f"Mode 2 Iter {iteration} Run {i+1}: All {self.max_retries} retries failed - {retry_e}")

                for run_i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        continue
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

            final_selected_names = [name for name, votes in candidate_votes.items() if votes >= threshold]
            rejected_names = [name for name, votes in candidate_votes.items() if votes < threshold]

            iter_prefix = f"[Iter {iteration}] " if iteration > 1 else ""
            for name, votes in candidate_votes.items():
                pos = candidate_pos[name]
                status = "ACCEPTED" if votes >= threshold else "REJECTED"
                reasoning_trace.append(f"{iter_prefix}- {name} ({pos}/{total_cands}): {votes} votes -> {status}")

            for selected in final_selected_names:
                curr = selected
                while curr:
                    if curr in verified_nodes:
                        break
                    verified_nodes.add(curr)
                    if curr in self.categories:
                        results[curr] = 1 # Store only hits
                    curr = self.parent_map.get(curr)

            for name in final_selected_names:
                 cand_obj = next((c for c in candidates if c['name'] == name), None)
                 source = cand_obj['source'] if cand_obj else "unknown"
                 final_selection_with_info.append(f"{name} ({source})")

            next_round_candidates_map = {}
            if self.include_superiors and direct_candidates is None:
                for name in rejected_names:
                    parent_name = self.parent_map.get(name)
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
            iteration += 1

        token_usage = {
            "avg_completion_tokens": handler.avg_completion_tokens
        }
        return results, reasoning_trace, sorted(results.keys()), token_usage