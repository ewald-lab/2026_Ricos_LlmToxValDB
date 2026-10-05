from __future__ import annotations
import argparse
import re
import sqlite3
import time
import torch
import rdflib
import json

from pathlib import Path
from typing import List, Optional, Any
from rdflib.namespace import RDF, RDFS, SKOS, OWL
from tqdm import tqdm
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from model.llm import get_model, test_model
from model.embeddings import EmbeddingModel
from src.utils import setup_logging, sanitize_name

DEFAULT_HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
DEFAULT_HF_TRANSFER = "1"

logger = setup_logging()

class OntologyToHierarchy:
    def __init__(self, ontology_path: str, root_uri: Optional[str] = None, prompt_template_path: Optional[str] = None, 
            global_definitions_db: str = "definitions.db", max_concurrency: int = 10, evaluate_prompt_template_path: Optional[str] = None,
            definitions_report_path: Optional[str] = None) -> None:
        """
        Description:
        Initializes the OntologyToHierarchy parser.

        Args:
            ontology_path (str): Path to the input ontology file.
            root_uri (Optional[str]): Optional root URI class for hierarchy traversal.
            prompt_template_path (Optional[str]): Optional path to custom prompt template.
            global_definitions_db (str): Central definition caching DB path. Defaults to "definitions.db".
            max_concurrency (int): Maximum concurrent LLM batch requests.

        Returns:
            None
        """
        self.graph = rdflib.Graph()
        logger.info(f"Loading ontology: {ontology_path}\n")
        self.graph.parse(ontology_path)
        self.root_uri = rdflib.URIRef(root_uri) if root_uri else None
        self.prompt_template_path = prompt_template_path
        self.evaluate_prompt_template_path = evaluate_prompt_template_path
        self.definitions_report_path = definitions_report_path
        if not self.evaluate_prompt_template_path and self.prompt_template_path:
            self.evaluate_prompt_template_path = str(Path(self.prompt_template_path).parent / "evaluate_description_prompt.txt")

        self.prompt_template_content = None
        if self.prompt_template_path:
            with open(self.prompt_template_path, "r", encoding="utf-8") as f:
                self.prompt_template_content = f.read()

        self.node_parents = {}
        self.visited_edges = set()
        self.prompt_cache = {}
        self.llm_generated_labels = set()
        self.visited = set()
        self.max_concurrency = max_concurrency

        self.global_definitions_db = global_definitions_db
        if self.global_definitions_db:
            Path(self.global_definitions_db).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.global_definitions_db) as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE IF NOT EXISTS global_definitions (uri TEXT PRIMARY KEY, definition TEXT, is_llm_generated INTEGER DEFAULT 0)")
                try:
                    cursor.execute("SELECT uri, definition, is_llm_generated FROM global_definitions")
                except sqlite3.OperationalError:
                    cursor.execute("SELECT uri, definition, 0 as is_llm_generated FROM global_definitions")
                count = 0
                for ruri, rdef, is_llm_gen in cursor:
                    self.prompt_cache[rdflib.URIRef(ruri)] = rdef
                    if is_llm_gen:
                        self.llm_generated_labels.add(rdflib.URIRef(ruri))
                    count += 1
            logger.info(f"Loaded {count} cached definitions from {self.global_definitions_db}")

    def get_label(self, uri: rdflib.URIRef) -> str:
        """
        Description:
        Extracts the best available English label or fragment name for a given class URI.

        Args:
            uri (rdflib.URIRef): The class URI.

        Returns:
            str: The extracted label.
        """
        for prop in [SKOS.prefLabel, RDFS.label]:
            for o in self.graph.objects(uri, prop):
                if o.language == 'en' or not o.language:
                    return str(o)
        return str(uri).split('#')[-1].split('/')[-1]

    def get_definition(self, uri: rdflib.URIRef) -> str:
        """
        Description:
        Extracts the definition/description of a class from common ontology properties.

        Args:
            uri (rdflib.URIRef): The class URI.

        Returns:
            str: The definition text, or empty string if not found.
        """
        props = [
            SKOS.definition,
            RDFS.comment,
            rdflib.URIRef("http://purl.obolibrary.org/obo/IAO_0000115"),
            rdflib.URIRef("http://aopwiki.org/ontology#has_description"),
            rdflib.URIRef("http://purl.org/dc/elements/1.1/description")
        ]
        for prop in props:
            for o in self.graph.objects(uri, prop):
                if o.language == 'en' or not o.language:
                    return str(o)
        return ""

    def get_children(self, parent_uri: rdflib.URIRef) -> list[rdflib.URIRef]:
        """
        Description:
        Returns sorted list of children URIs subclassing the specified parent URI.

        Args:
            parent_uri (rdflib.URIRef): The parent class URI.

        Returns:
            list[rdflib.URIRef]: Sorted list of subclass children URIs.
        """
        children = []
        for s in self.graph.subjects(RDFS.subClassOf, parent_uri):
            if isinstance(s, rdflib.URIRef):
                children.append(s)
        children.sort(key=lambda x: self.get_label(x).lower())
        return children

    def get_direct_parents(self, uri: rdflib.URIRef) -> list[rdflib.URIRef]:
        """
        Description:
        Returns direct parent URIs (rdfs:subClassOf) for a given class URI.

        Args:
            uri (rdflib.URIRef): The class URI.

        Returns:
            list[rdflib.URIRef]: List of parent URIs.
        """
        return [o for o in self.graph.objects(uri, RDFS.subClassOf) if isinstance(o, rdflib.URIRef)]

    def get_all_paths_to_root(self, uri: rdflib.URIRef, visited: Optional[set] = None) -> list[list[str]]:
        """
        Description:
        Returns all root-to-leaf paths leading to this URI as lists of human-readable labels.
        Handles multi-parent (DAG) structures by exploring every parent branch.

        Args:
            uri (rdflib.URIRef): The class URI to get paths for.
            visited (Optional[set]): Set of visited URIs to prevent infinite cycles.

        Returns:
            list[list[str]]: Each element is one full path from root to this node (list of labels).
        """
        if visited is None:
            visited = set()
        if uri in visited:
            return []
        visited = visited | {uri}

        label = self.get_label(uri)
        parents = self.get_direct_parents(uri)

        if not parents:
            return [[label]]

        all_paths = []
        for parent_uri in parents:
            parent_paths = self.get_all_paths_to_root(parent_uri, visited)
            for path in parent_paths:
                all_paths.append(path + [label])

        return all_paths if all_paths else [[label]]

    def count_total_nodes(self, current_classes: list[rdflib.URIRef]) -> int:
        """
        Description:
        Calculates the total number of unique nodes in the tree structure starting from the given classes.

        Args:
            current_classes (list[rdflib.URIRef]): Starting classes list.

        Returns:
            int: The total count of unique nodes.
        """
        seen = set()

        def _recursive_count(nodes: list[rdflib.URIRef]) -> int:
            """
            Description:
            Helper method to recursively count nodes.

            Args:
                nodes (list[rdflib.URIRef]): Nodes list to count.

            Returns:
                int: The node count.
            """
            local_c = 0
            for uri in nodes:
                if uri in seen: continue
                seen.add(uri)
                local_c += 1
                children = self.get_children(uri)
                if children:
                    local_c += _recursive_count(children)
            return local_c

        return _recursive_count(current_classes)

    def generate_definitions_batch(self, llm: Any, labels_to_generate: list[str], hierarchies_to_generate: list[str]) -> dict[str, str]:
        """
        Description:
        Uses the LLM in batch mode to generate short descriptions for a batch of node labels.

        Args:
            llm (Any): The LLM engine instance.
            labels_to_generate (list[str]): List of labels needing descriptions.
            hierarchies_to_generate (list[str]): List of hierarchy breadcrumbs for each label.

        Returns:
            dict[str, str]: A dictionary mapping labels to generated description strings.
        """
        if not labels_to_generate:
            return {}

        instruction_tmpl = self.prompt_template_content.strip()
        messages = [("user", instruction_tmpl)]

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()

        inputs = [{"label": label, "hierarchy": hierarchy} for label, hierarchy in zip(labels_to_generate, hierarchies_to_generate)]

        # Process in chunks to show progress via tqdm
        chunk_size = self.max_concurrency * 2
        generated_descriptions = []

        for i in tqdm(range(0, len(inputs), chunk_size), desc="LLM Generating Definitions", leave=False, smoothing=0.0):
            chunk = inputs[i:i + chunk_size]
            try:
                chunk_results = chain.batch(chunk, config={"max_concurrency": self.max_concurrency})
            except Exception as e:
                logger.warning(f"Batch generation failed with error: {e}. Retrying sequentially with progressive truncation...")
                chunk_results = []
                for item in chunk:
                    retries = 0
                    current_item = dict(item)
                    while retries < 4:
                        try:
                            res = chain.invoke(current_item)
                            chunk_results.append(res)
                            break
                        except Exception as inner_e:
                            retries += 1
                            logger.warning(f"Item '{current_item['label']}' failed (retry {retries}/4): {inner_e}. Truncating hierarchy by half...")
                            hier_words = current_item["hierarchy"].split()
                            current_item["hierarchy"] = " ".join(hier_words[:max(1, len(hier_words)//2)]) + "..."
                    else:
                        logger.error(f"Failed to generate for item after retries: {item['label']}")
                        chunk_results.append("")
            generated_descriptions.extend(chunk_results)

        results = {}
        for label, desc in zip(labels_to_generate, generated_descriptions):
            desc = desc.strip('" \n')
            pattern = re.compile(rf'^(\*\*)?{re.escape(label)}(\*\*)?(\s*\([^)]+\))?\s*(is\s+|:\s+|-?\s+)', re.IGNORECASE)
            desc = pattern.sub('', desc, count=1)
            if desc:
                desc = desc[0].upper() + desc[1:]
            results[label] = desc

        return results

    def evaluate_definitions_batch(self, llm: Any, batch_data: list[dict]) -> list[str]:
        """
        Description:
        Uses the LLM in batch mode to evaluate if existing descriptions are good or bad.

        Args:
            llm (Any): The LLM engine instance.
            batch_data (list[dict]): List of dicts with keys 'label', 'hierarchy', 'definition'.

        Returns:
            list[str]: List of decisions ('GOOD' or 'BAD') for each definition.
        """
        if not batch_data:
            return []

        with open(self.evaluate_prompt_template_path, "r", encoding="utf-8") as f:
            eval_tmpl = f.read().strip()

        messages = [("user", eval_tmpl)]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()

        # Process in chunks
        chunk_size = self.max_concurrency * 2
        generated_results = []

        for i in tqdm(range(0, len(batch_data), chunk_size), desc="LLM Evaluating Definitions", leave=False, smoothing=0.0):
            chunk = batch_data[i:i + chunk_size]
            try:
                chunk_results = chain.batch(chunk, config={"max_concurrency": self.max_concurrency})
            except Exception as e:
                logger.warning(f"Batch evaluation failed with error: {e}. Retrying sequentially with progressive truncation...")
                chunk_results = []
                for item in chunk:
                    retries = 0
                    current_item = dict(item)
                    while retries < 4:
                        try:
                            res = chain.invoke(current_item)
                            chunk_results.append(res)
                            break
                        except Exception as inner_e:
                            retries += 1
                            logger.warning(f"Item '{current_item['label']}' failed (retry {retries}/4): {inner_e}. Truncating input by half...")
                            if "definition" in current_item:
                                def_words = current_item["definition"].split()
                                current_item["definition"] = " ".join(def_words[:max(1, len(def_words)//2)]) + "..."
                            hier_words = current_item["hierarchy"].split()
                            current_item["hierarchy"] = " ".join(hier_words[:max(1, len(hier_words)//2)]) + "..."
                    else:
                        logger.error(f"Failed to evaluate item after retries: {item['label']}")
                        chunk_results.append('{"decision": "BAD"}')
            generated_results.extend(chunk_results)

        decisions = []
        for res in generated_results:
            text = res.strip()
            if "</think>" in text:
                text = text.split("</think>")[-1].strip()

            decision = "BAD"
            try:
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    decision = data.get("decision", "BAD").strip().upper()
                else:
                    if "GOOD" in text.upper():
                        decision = "GOOD"
            except Exception:
                if "GOOD" in text.upper() and "BAD" not in text.upper():
                    decision = "GOOD"
            
            if decision not in ("GOOD", "BAD"):
                decision = "BAD"
            decisions.append(decision)

        return decisions

    def pre_generate_missing_definitions(self, llm: Any, current_classes: list[rdflib.URIRef]) -> None:
        """
        Description:
        Traverses hierarchy to identify classes lacking definitions or having bad ones,
        evaluates/generates them using LLM, and caches them.

        Args:
            llm (Any): The LLM engine instance.
            current_classes (list[rdflib.URIRef]): Hierarchy entry classes.

        Returns:
            None
        """
        missing_to_generate = []  # list of (uri, label, hierarchy_str)
        to_evaluate = []          # list of (uri, label, hierarchy_str, definition)
        seen = set()

        def _recursive_collect(nodes: list[rdflib.URIRef], parent_path: list[str]) -> None:
            """
            Description:
            Helper method to recursively collect classes lacking definitions or needing evaluation.
            For nodes with multiple parents, all root-to-node paths are passed to the LLM as
            additional context, one per line.

            Args:
                nodes (list[rdflib.URIRef]): Nodes list to inspect.
                parent_path (list[str]): Hierarchy path labels leading to current nodes.

            Returns:
                None
            """
            for uri in nodes:
                if uri in seen: continue
                seen.add(uri)

                label = self.get_label(uri)
                definition = self.get_definition(uri)
                if definition:
                    words = definition.split()
                    if len(words) > 1000:
                        definition = " ".join(words[:1000]) + "..."

                # Build all root-to-node paths to give the LLM full multi-parent context
                all_paths = self.get_all_paths_to_root(uri)
                if len(all_paths) > 1:
                    hier_lines = [" > ".join(path) for path in all_paths[:5]]
                    if len(all_paths) > 5:
                        hier_lines.append("...")
                    hierarchy_str = "\n".join(hier_lines)
                else:
                    hierarchy_str = " > ".join(parent_path + [label])

                if uri not in self.prompt_cache:
                    if not definition or len(definition.split()) < 5:
                        missing_to_generate.append((uri, label, hierarchy_str))
                    else:
                        to_evaluate.append((uri, label, hierarchy_str, definition))

                children = self.get_children(uri)
                if children:
                    _recursive_collect(children, parent_path + [label])

        _recursive_collect(current_classes, [])
        total_terms = len(seen)

        bad_to_generate = []  # list of (uri, label, hierarchy_str)
        good_to_cache = []    # list of (uri, label, definition)

        if llm is None:
            logger.info("LLM is disabled. Using existing ontology definitions as fallbacks...")
            combined_to_generate = missing_to_generate
            num_missing_generated = 0
            num_bad_generated = 0
            
            for uri, label, hierarchy_str, definition in to_evaluate:
                good_to_cache.append((uri, label, definition))
                
            if good_to_cache:
                with sqlite3.connect(self.global_definitions_db) as conn:
                    cursor = conn.cursor()
                    for uri, label, definition in good_to_cache:
                        self.prompt_cache[uri] = definition
                        cursor.execute("INSERT OR REPLACE INTO global_definitions (uri, definition, is_llm_generated) VALUES (?, ?, ?)", (str(uri), definition, 0))
                    conn.commit()
                logger.info(f"Cached {len(good_to_cache)} existing definitions.")
                
            if combined_to_generate:
                logger.warning(f"No LLM provided. Using ontology fallback (label/definitions) for {len(combined_to_generate)} terms lacking cached definitions.")
                with sqlite3.connect(self.global_definitions_db) as conn:
                    cursor = conn.cursor()
                    for uri, label, hierarchy_str in combined_to_generate:
                        fallback_def = self.get_definition(uri) or f"Description for {label}."
                        self.prompt_cache[uri] = fallback_def
                        cursor.execute("INSERT OR REPLACE INTO global_definitions (uri, definition, is_llm_generated) VALUES (?, ?, ?)", (str(uri), fallback_def, 0))
                        num_missing_generated += 1
                    conn.commit()
            
            self.total_terms = total_terms
            self.num_missing_generated = num_missing_generated
            self.num_bad_generated = num_bad_generated
            self.total_created = num_missing_generated + num_bad_generated
            return

        if to_evaluate:
            logger.info(f"Evaluating {len(to_evaluate)} existing definitions using LLM...")
            eval_batch_data = [
                {"label": item[1], "hierarchy": item[2], "definition": item[3]}
                for item in to_evaluate
            ]
            decisions = self.evaluate_definitions_batch(llm, eval_batch_data)

            for (uri, label, hierarchy_str, definition), decision in zip(to_evaluate, decisions):
                if decision == "BAD":
                    logger.info(f"Definition evaluated as BAD for {label} (URI: {uri}). Existing: {definition}")
                    bad_to_generate.append((uri, label, hierarchy_str))
                else:
                    logger.info(f"Definition evaluated as GOOD for {label} (URI: {uri}). Existing: {definition}")
                    good_to_cache.append((uri, label, definition))

        # Cache the good definitions
        if good_to_cache:
            with sqlite3.connect(self.global_definitions_db) as conn:
                cursor = conn.cursor()
                for uri, label, definition in good_to_cache:
                    self.prompt_cache[uri] = definition
                    is_llm = 1 if uri in self.llm_generated_labels else 0
                    cursor.execute("INSERT OR REPLACE INTO global_definitions (uri, definition, is_llm_generated) VALUES (?, ?, ?)", (str(uri), definition, is_llm))
                conn.commit()
            logger.info(f"Cached {len(good_to_cache)} good ontology definitions.")

        # Combine all to generate
        combined_to_generate = missing_to_generate + bad_to_generate
        num_missing_generated = 0
        num_bad_generated = 0

        if combined_to_generate:
            logger.info(f"Pre-generating {len(combined_to_generate)} descriptions in parallel ({len(missing_to_generate)} missing, {len(bad_to_generate)} bad)...")
            
            labels_to_generate = [item[1] for item in combined_to_generate]
            hierarchies_to_generate = [item[2] for item in combined_to_generate]
            
            generated_defs = self.generate_definitions_batch(llm, labels_to_generate, hierarchies_to_generate)

            with sqlite3.connect(self.global_definitions_db) as conn:
                cursor = conn.cursor()
                
                # Process missing
                for uri, label, hierarchy_str in missing_to_generate:
                    desc = generated_defs.get(label)
                    if desc:
                        self.prompt_cache[uri] = desc
                        self.llm_generated_labels.add(uri)
                        cursor.execute("INSERT OR REPLACE INTO global_definitions (uri, definition, is_llm_generated) VALUES (?, ?, ?)", (str(uri), desc, 1))
                        logger.info(f"Definition generated (missing) for {uri} ({label}) -> {desc}")
                        num_missing_generated += 1
                        
                # Process bad
                for uri, label, hierarchy_str in bad_to_generate:
                    desc = generated_defs.get(label)
                    if desc:
                        self.prompt_cache[uri] = desc
                        self.llm_generated_labels.add(uri)
                        cursor.execute("INSERT OR REPLACE INTO global_definitions (uri, definition, is_llm_generated) VALUES (?, ?, ?)", (str(uri), desc, 1))
                        logger.info(f"Definition generated (bad original) for {uri} ({label}) -> {desc}")
                        num_bad_generated += 1
                        
                conn.commit()

        total_created = num_missing_generated + num_bad_generated
        self.total_terms = total_terms
        self.num_missing_generated = num_missing_generated
        self.num_bad_generated = num_bad_generated
        self.total_created = total_created



    def save_definitions_report(self, report_path: str) -> None:
        """
        Description:
        Generates and saves a JSON report showing every node, its original definition,
        whether it was GOOD, BAD, or MISSING, and the final/new definition.

        Args:
            report_path (str): File path to save the JSON report.

        Returns:
            None
        """
        report = []
        seen = set()

        def _traverse_report(nodes: list[rdflib.URIRef], parent_path: list[str]) -> None:
            for uri in nodes:
                if uri in seen: continue
                seen.add(uri)

                label = self.get_label(uri)
                ontology_definition = self.get_definition(uri)
                hierarchy_str = " > ".join(parent_path + [label])
                cached_definition = self.prompt_cache.get(uri, "")

                old_definition = ontology_definition if ontology_definition else None
                new_definition = None

                if not ontology_definition or len(ontology_definition.split()) < 5:
                    status = "MISSING"
                    new_definition = cached_definition if cached_definition else None
                else:
                    if cached_definition and cached_definition != ontology_definition:
                        status = "BAD"
                        new_definition = cached_definition
                    else:
                        status = "GOOD"

                report.append({
                    "uri": str(uri),
                    "label": label,
                    "hierarchy": hierarchy_str,
                    "old_definition": old_definition,
                    "status": status,
                    "new_definition": new_definition
                })

                children = self.get_children(uri)
                if children:
                    _traverse_report(children, parent_path + [label])

        if not self.root_uri:
            subjects = set(self.graph.subjects(RDF.type, OWL.Class))
            children_with_parents = set()
            for s, p, o in self.graph.triples((None, RDFS.subClassOf, None)):
                if isinstance(o, rdflib.URIRef) and o != OWL.Thing:
                    children_with_parents.add(s)
            roots = list(subjects - children_with_parents)
            roots = [r for r in roots if isinstance(r, rdflib.URIRef)]
            top_level_classes = sorted(roots, key=lambda x: self.get_label(x))
        else:
            top_level_classes = [self.root_uri]

        _traverse_report(top_level_classes, [])

        try:
            report_file = Path(report_path)
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Definitions report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save definitions report to {report_path}: {e}")

    def build_hierarchy(self, current_classes: list[rdflib.URIRef], llm: Any, pbar: Optional[tqdm] = None, parent_label: Optional[str] = None, ) -> None:
        """
        Description:
        Recursively traverses ontology classes to build the node-parent mapping.

        Args:
            current_classes (list[rdflib.URIRef]): Ontology classes list to process.
            llm (Any): The LLM engine.
            pbar (Optional[tqdm]): Progress bar logger. Defaults to None.
            parent_label (Optional[str]): Parent label string. Defaults to None.

        Returns:
            None
        """
        for i, current_uri in enumerate(current_classes):
            edge = (parent_label, current_uri)
            if edge in self.visited_edges:
                continue
            
            self.visited_edges.add(edge)

            if pbar:
                pbar.update(1)

            label = self.get_label(current_uri)
            logger.debug(f"Processing node: {label}")
            definition = self.get_definition(current_uri)

            children = self.get_children(current_uri)
            if children:
                self.build_hierarchy(children, llm, pbar=pbar, parent_label=label)

            if current_uri not in self.prompt_cache:
                self.prompt_cache[current_uri] = definition

            if current_uri not in self.node_parents:
                self.node_parents[current_uri] = []
            
            if parent_label and parent_label not in self.node_parents[current_uri]:
                self.node_parents[current_uri].append(parent_label)

    def generate_embeddings(self, model_name: str, ontology_db: str) -> None:
        """
        Description:
        Generates text embeddings for all processed nodes and writes them to an SQLite database.

        Args:
            model_name (str): HuggingFace embedding model ID.
            ontology_db (str): Output database path.

        Returns:
            None
        """
        db_path = Path(ontology_db)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if db_path.exists():
            db_path.unlink()
            logger.info(f"Removed existing database: {ontology_db}")

        try:
            embedder = EmbeddingModel(model_name)
            embedder.load()
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                uri TEXT PRIMARY KEY,
                name TEXT,
                sanitized_name TEXT,
                parent TEXT,
                vector BLOB
            )
        ''')

        logger.info(f"Generating embeddings for {len(self.node_parents)} nodes... \n")

        for uri, parents in tqdm(self.node_parents.items(), desc="Embedding nodes", smoothing=0.0):
            label = self.get_label(rdflib.URIRef(uri))
            label_safe = sanitize_name(label)

            embed_text = label_safe

            vector = embedder.get_embedding(embed_text)
            parent_json = json.dumps(parents)

            cursor.execute("INSERT OR REPLACE INTO embeddings (uri, name, sanitized_name, parent, vector) VALUES (?, ?, ?, ?, ?)",
                           (str(uri), label, label_safe, parent_json, vector))

        del embedder.model
        del embedder.tokenizer
        del embedder
        torch.cuda.empty_cache()

        conn.commit()
        conn.close()
        logger.info(f"Embeddings saved: {ontology_db}")

    def create(self, llm: Any) -> None:
        """
        Description:
        Main pipeline builder initiating root detection, definition pre-generation, and hierarchy traversal.

        Args:
            llm (Any): The LLM engine.

        Returns:
            None
        """
        if not self.root_uri:
            subjects = set(self.graph.subjects(RDF.type, OWL.Class))
            children_with_parents = set()
            for s, p, o in self.graph.triples((None, RDFS.subClassOf, None)):
                if isinstance(o, rdflib.URIRef) and o != OWL.Thing:
                    children_with_parents.add(s)

            roots = list(subjects - children_with_parents)

            if not roots:
                logger.error("No root class found or specified. Please specify --root_uri.")
                return

            roots = [r for r in roots if isinstance(r, rdflib.URIRef)]
            logger.info(f"\nAuto-detected {len(roots)} root classes.")
            top_level_classes = sorted(roots, key=lambda x: self.get_label(x))
        else:
            top_level_classes = [self.root_uri]

        total_nodes = self.count_total_nodes(top_level_classes)
        logger.info(f"Total nodes to process (tree structure): {total_nodes}")

        self.pre_generate_missing_definitions(llm, top_level_classes)

        with tqdm(total=total_nodes, desc="Building Hierarchy", unit="node", smoothing=0.0) as pbar:
            self.build_hierarchy(top_level_classes, llm, pbar=pbar)

        logger.info(f"Hierarchy built: {len(self.node_parents)} nodes processed.")

        if self.definitions_report_path:
            self.save_definitions_report(self.definitions_report_path)


def main() -> None:
    """
    Description:
    Main entry point for processing ontology files. It parses inputs, runs the hierarchy extraction,
    generates missing definitions, produces node embeddings, and stores them in a local SQLite DB.

    Args:
        None

    Returns:
        None
    """
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Generate Hierarchy and Prompts from Ontology using LLM")

    parser.add_argument("--ontology_file", required=True, help="Path to input .ttl ontology file")
    parser.add_argument("--root_uri", help="URI of the root class to start traversal (optional)")
    parser.add_argument("--prompt_template", help="Path to custom prompt template file (optional)")
    parser.add_argument("--evaluate_prompt_template", help="Path to evaluation prompt template file (optional)")
    parser.add_argument("--definitions_report", help="Path to save the JSON report of definitions (optional)")

    parser.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-8B", help="HuggingFace model for node embeddings")
    parser.add_argument("--embeddings_model_id", default=None, help="Identifier for the model configuration in RAG config (e.g., qwen, nemotron)")
    parser.add_argument("--ontology_db", default="embeddings_db/tox_effects.db", help="Output SQLite database for embeddings")
    parser.add_argument("--global_definitions_db", default="definitions.db", help="Central DB for caching LLM generated definitions")
    parser.add_argument("--skip_embeddings", action="store_true", help="Skip embedding generation")
    parser.add_argument("--rag_config", help="Path to RAG configuration JSON file.")

    parser.add_argument("--model_config", default=None, help="Path to model config JSON file.")
    parser.add_argument("--base_url", default=None, help="Base URL for the LLM API. Overrides value in model_config if provided.")

    parser.add_argument("--hf_home", default=None, help="HuggingFace cache directory.")
    parser.add_argument("--hf_transfer", default=None, help="Enable/disable HF transfer (0 or 1).")

    args = parser.parse_args()

    # Load RAG config if provided
    if args.rag_config:
        logger.info(f"Loading RAG configuration from: {args.rag_config}")
        with open(args.rag_config, 'r', encoding='utf-8') as f:
            rag_config = json.load(f)

        # If it's a dictionary, look up by embeddings_model_id or embedding_model string
        if isinstance(rag_config, dict):
            key = args.embeddings_model_id or args.embedding_model
            model_config = rag_config.get(key)
            if model_config:
                logger.info(f"Using configuration block for: {key}")
                rag_config = model_config
            else:
                logger.warning(f"No configuration found for '{key}' in {args.rag_config}. Using default.")
        elif isinstance(rag_config, list):
            # Fallback for old list format just in case
            matching_config = next((c for c in rag_config if c.get("embedding_model") == args.embedding_model), None)
            if matching_config:
                rag_config = matching_config
            else:
                rag_config = rag_config[0]

        args.embedding_model = rag_config.get("embedding_model", args.embedding_model)

    llm = None
    max_concurrency = 10

    if args.model_config:
        with open(args.model_config, 'r', encoding='utf-8') as f:
            model_config = json.load(f)

        max_concurrency = model_config.pop("max_concurrency", 10)

        if args.base_url:
            model_config["base_url"] = args.base_url

        logger.info(f"Loading model configuration from: {args.model_config}")
        logger.info(f"  max_concurrency: {max_concurrency}")
        for key, value in model_config.items():
            if key != "api_key":
                logger.info(f"  {key}: {value}")

        llm = get_model(**model_config)
    else:
        logger.info("No model config provided. LLM definition generation is disabled.")

    hf_home = args.hf_home or os.environ.get("HF_HOME") or DEFAULT_HF_HOME
    hf_transfer = args.hf_transfer or os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") or DEFAULT_HF_TRANSFER

    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = hf_transfer

    logger.info(f"HuggingFace Home: {hf_home}")
    logger.info(f"HF Transfer: {hf_transfer}")

    generator = OntologyToHierarchy(args.ontology_file, args.root_uri, args.prompt_template, global_definitions_db=args.global_definitions_db, max_concurrency=max_concurrency, evaluate_prompt_template_path=args.evaluate_prompt_template, definitions_report_path=args.definitions_report)

    if llm:
        if not test_model(llm):
            raise RuntimeError("Model connectivity test failed. Please check your configuration.")
    
    generator.create(llm)

    if not args.skip_embeddings:
        generator.generate_embeddings(args.embedding_model, args.ontology_db)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    logger.info(f"Total execution time: {minutes}m {seconds:.2f}s")

    if hasattr(generator, "total_terms"):
        pct_missing = (generator.num_missing_generated / generator.total_terms) * 100 if generator.total_terms > 0 else 0.0
        pct_bad = (generator.num_bad_generated / generator.total_terms) * 100 if generator.total_terms > 0 else 0.0
        pct_total = (generator.total_created / generator.total_terms) * 100 if generator.total_terms > 0 else 0.0
        
        logger.info("=========================================\n"
                    "FINAL SUMMARY OF DEFINITION GENERATION:\n"
                    f"Total ontology terms: {generator.total_terms}\n"
                    f"Created because missing: {generator.num_missing_generated} ({pct_missing:.2f}% of total)\n"
                    f"Created because bad: {generator.num_bad_generated} ({pct_bad:.2f}% of total)\n"
                    f"Total created: {generator.total_created} ({pct_total:.2f}% of total)\n"
                    "=========================================")


if __name__ == "__main__":
    main()
