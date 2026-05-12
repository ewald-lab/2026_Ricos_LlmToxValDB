import sqlite3
import torch
import os
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM

from typing import List, Dict, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from model.embeddings import EmbeddingModel
from .utils import setup_logging, load_ontology_embeddings

logger = setup_logging()

def clean_llm_output(message) -> str:
    """Removes <think> tags and markdown JSON formatting from LLM output."""
    text = message.content if hasattr(message, "content") else str(message)
    
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
        
    if text.startswith("```json"):
        text = text.replace("```json", "", 1)
    elif text.startswith("```"):
        text = text.replace("```", "", 1)
    if text.endswith("```"):
        text = text[::-1].replace("```", "", 1)[::-1] 
        
    return text.strip()

def clean_reason(reason: str) -> str:
    r = reason.strip()
    if r.upper().startswith("ACCEPTED"):
        return r[8:].lstrip(" .-:")
    elif r.upper().startswith("REJECTED"):
        return r[8:].lstrip(" .-:")
    return r

class CandidateEvaluation(BaseModel):
    """Structure for evaluating a single candidate."""
    candidate_name: str = Field(description="The exact name of the candidate being evaluated.")
    reason: str = Field(description="Think step-by-step. Provide the explicit reason for the decision based on the critical rules. DO NOT write 'ACCEPTED' or 'REJECTED' in this field.")
    decision: Literal["ACCEPTED", "REJECTED"] = Field(description="Strictly 'ACCEPTED' or 'REJECTED' based on the reasoning above.")

class BatchVerificationResult(BaseModel):
    """Structure for batch evaluation (Mode 2)."""
    analysis: str = Field(description="Step 1: Analyze the input description and determine its exact level of specificity (i.e., whether it is a broad categorical system or a highly specific mechanism).")
    evaluation: List[CandidateEvaluation] = Field(description="Step 2: Evaluate EVERY candidate individually. You MUST provide an evaluation object for ALL candidates in the list.")
    selected_candidates: List[str] = Field(description="Step 3: The final list of exact candidate names that pass the evaluation. Return an empty list [] if none match.")

class TextToOntoGraph:
    def __init__(self, categories: List[str], 
                ontology_db: str, embedding_model_name: str, 
                input_db: str, top_k_rag: int, system_prompt_path: str, threshold_rag: float,
                reranker_model_name: str, threshold_reranker: float,
                mode: int, consensus_runs: int, ontology_definitions: bool, max_concurrency: int, max_retries: int, context_ontology: str,
                reranker_prompt_file: str, reranker_batch_size: int,
                global_definitions_db: str = None):
        self.categories = categories
        self.ontology_definitions = ontology_definitions
        self.include_superiors = True
        self.max_retries = max_retries
        self.context_ontology = context_ontology
        self._description_counter = 0
        
        self.ontology_db = ontology_db
        self.input_db = input_db
        self._input_db_conn = sqlite3.connect(self.input_db) if self.input_db else None
        
        self.embedder = EmbeddingModel(embedding_model_name)
        self.ontology_embeddings = load_ontology_embeddings(self.ontology_db)
        self.top_k_rag = top_k_rag
        self.threshold_rag = threshold_rag
        self.reranker_model_name = reranker_model_name
        self.threshold_reranker = threshold_reranker
        self.reranker_batch_size = reranker_batch_size
        self.mode = mode
        self.consensus_runs = consensus_runs
        self.max_concurrency = max_concurrency
        
        self.reranker_model = None
        self.reranker_tokenizer = None
        if self.threshold_reranker > 0.0:
            logger.info(f"Loading Reranker Model: {self.reranker_model_name}")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name, padding_side='left')
            self.reranker_model = AutoModelForCausalLM.from_pretrained(
                self.reranker_model_name, torch_dtype=torch.bfloat16, device_map="auto"
            ).eval()
            self.reranker_device = self.reranker_model.device
            self.token_false_id = self.reranker_tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.reranker_tokenizer.convert_tokens_to_ids("yes")
            prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.reranker_prefix_tokens = self.reranker_tokenizer.encode(prefix, add_special_tokens=False)
            self.reranker_suffix_tokens = self.reranker_tokenizer.encode(suffix, add_special_tokens=False)
            # Precompute available space for actual content tokens
            self._reranker_max_content_len = 1024 - len(self.reranker_prefix_tokens) - len(self.reranker_suffix_tokens)
            with open(reranker_prompt_file, 'r', encoding='utf-8') as f:
                self.reranker_instruction = f.read().strip()
        
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
                
            self.nodes[name] = {
                'sanitized_name': item['sanitized_name'],
                'parent': item.get('parent'),
                'uri': uri
            }
            self.prompt_map[name] = item.get('def', '')
            parent_name = item.get('parent')
            if parent_name:
                self.parent_map[name] = parent_name
                self.children_map.setdefault(parent_name, []).append(name)

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
        
        user_prompt = "Description: {description}\nCandidate to evaluate: {candidate_name}"
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

    def close(self):
        """Clean up persistent resources."""
        if self._input_db_conn:
            self._input_db_conn.close()

    def _rerank_candidates(self, description: str, candidates: list) -> list:
        """Applies the reranker model to filter candidates by score threshold. Returns filtered list."""
        if self.threshold_reranker <= 0.0 or not candidates or self.reranker_model is None:
            return candidates
        
        logger.info(f"Applying Reranker (Threshold: {self.threshold_reranker}) to {len(candidates)} candidates...")
        
        pairs = []
        for c in candidates:
            doc = c['sanitized_name']
            hierarchy = self.get_breadcrumbs(c['name'])
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
                
                inputs = self.reranker_tokenizer.pad(
                    inputs, padding=True, return_tensors="pt"
                )
                inputs = {k: v.to(self.reranker_device) for k, v in inputs.items()}
                
                batch_logits = self.reranker_model(**inputs).logits[:, -1, :]
                true_vector = batch_logits[:, self.token_true_id]
                false_vector = batch_logits[:, self.token_false_id]
                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                batch_probs = batch_scores[:, 1].exp().tolist()
                scores.extend(batch_probs)
        
        passed_candidates = []
        rejected_count = 0
        for score, cand in zip(scores, candidates):
            if score >= self.threshold_reranker:
                cand['rerank_score'] = score
                passed_candidates.append(cand)
            else:
                rejected_count += 1
        
        logger.info(f"Reranker complete: kept {len(passed_candidates)} candidates, rejected {rejected_count}.")
        
        rerank_list = [f"{c['sanitized_name']} ({c['rerank_score']:.3f})" for c in passed_candidates]
        logger.info(f"Reranker Candidates (>= {self.threshold_reranker}): {rerank_list}")
        
        return passed_candidates

    def get_breadcrumbs(self, name: str) -> str:
        """Constructs the hierarchical path (breadcrumbs) for a node. Cached per unique name."""
        return self._get_breadcrumbs_cached(name)

    @lru_cache(maxsize=None)
    def _get_breadcrumbs_cached(self, name: str) -> str:
        path = []
        curr = name
        visited = set()
        while curr and curr not in visited:
            visited.add(curr)
            label = self.nodes.get(curr, {}).get('sanitized_name', curr)
            path.append(label)
            curr = self.parent_map.get(curr)
        return " > ".join(reversed(path))

    def execute(self, llm, description: str, std_desc: str = None, alpha: float = None):
        """Executes the retrieval, reranking, and verification flow."""
        self._description_counter += 1
        logger.info(f"{self._description_counter}. Description: {description}")

        # Optimization: return only hits to avoid massive dictionaries in memory
        results = {} 
        reasoning_trace = []
        selected_candidates = []
        
        # Build the evaluation chain (LangChain) dynamically with the provided llm
        chain = self._mode1_prompt | llm | self._clean_llm_runner | self.parser
        
        def get_vec_for_text(text: str) -> torch.Tensor:
            vec = None
            if self._input_db_conn:
                try:
                    cursor_in = self._input_db_conn.cursor()
                    cursor_in.execute("SELECT vector FROM input_embeddings WHERE text = ?", (text,))
                    row = cursor_in.fetchone()
                    if row:
                        vec = torch.frombuffer(bytearray(row[0]), dtype=torch.float32)
                except Exception as e:
                    logger.warning(f"Could not fetch cached embedding for {text}: {e}")
            if vec is None:
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
                batch_results_all = chain.batch(batch_inputs, config={"max_concurrency": self.max_concurrency}, return_exceptions=True)
            
            for i, result in enumerate(batch_results_all):
                if isinstance(result, Exception):
                    meta = batch_metadata[i]
                    for retry in range(self.max_retries):
                        try:
                            batch_results_all[i] = chain.invoke(batch_inputs[i])
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
                    parent_name = self.parent_map.get(name)
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
    
    def _execute_mode_2(self, llm, description: str, top_k_candidates: List[Dict]):
        """Executes batch verification using the BatchVerificationResult model."""
        results = {}
        reasoning_trace = []
        
        verified_nodes = set()
        evaluated_nodes = set()
        
        current_candidates = top_k_candidates
        iteration = 1

        # Build the chain once
        base_chain = self._mode2_prompt | llm | self._clean_llm_runner | self._mode2_parser
        
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
                batch_results = chain.batch(invoke_inputs, config={"max_concurrency": self.max_concurrency}, return_exceptions=True)
                
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        for retry in range(self.max_retries):
                            try:
                                batch_results[i] = chain.invoke(invoke_inputs[i])
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
            # Pre-filter inferior/superior candidates with the reranker before sending to LLM
            if current_candidates:
                current_candidates = self._rerank_candidates(description, current_candidates)
            iteration += 1

        return results, reasoning_trace, sorted(results.keys())