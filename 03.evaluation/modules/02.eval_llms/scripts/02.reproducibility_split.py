from __future__ import annotations
import os
import re
import json
import glob
import rdflib
import hashlib
import random
import networkx as nx
import pandas as pd
from rdflib.namespace import RDFS

def build_graph(owl_file: str) -> tuple[nx.DiGraph, dict, dict]:
    print(f"Parsing ontology: {owl_file}")
    g = rdflib.Graph()
    g.parse(owl_file, format="turtle")

    G = nx.DiGraph()
    name_to_uri: dict = {}
    uri_to_name: dict = {}

    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        G.add_edge(s, o)

    for s, p, o in g.triples((None, RDFS.label, None)):
        if not isinstance(s, rdflib.term.URIRef):
            continue
        lang = getattr(o, "language", None)
        if lang is not None and lang != "en":
            continue
        name = str(o).lower().strip()
        name_to_uri[name] = s
        uri_to_name[s] = name

    return G, name_to_uri, uri_to_name

def get_ancestors_names(
    term_name: str,
    G: nx.DiGraph,
    name_to_uri: dict,
    uri_to_name: dict,
    root_uri: rdflib.term.URIRef | None
) -> set[str]:
    term_name = term_name.lower().strip()
    if term_name not in name_to_uri:
        return set()
    uri = name_to_uri[term_name]
    ancestors = nx.descendants(G, uri)
    ancestor_names = set()
    for anc in ancestors:
        if anc == root_uri:
            continue
        if anc in uri_to_name:
            ancestor_names.add(uri_to_name[anc])
    return ancestor_names

def get_run_indices(phrase: str, cand_name: str, votes: int) -> list[int]:
    if votes <= 0:
        return []
    return list(range(1, min(votes, 10) + 1))

def main() -> None:
    base_dir = "/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/03.evaluation/modules/02.eval_llms"
    outputs_dir = os.path.join(base_dir, "outputs")
    reproducibility_dir = os.path.join(outputs_dir, "reproducibility")
    os.makedirs(reproducibility_dir, exist_ok=True)

    # Regex to parse votes from reasoning JSON lines:
    # e.g., "- abnormal Kupffer cell morphology (1/2): 10 votes -> ACCEPTED"
    vote_pattern = re.compile(r'^-\s*(.*?)\s*\(\d+/\d+\):\s*(\d+)\s*votes\s*->\s*(ACCEPTED|REJECTED)$')

    reasoning_files = glob.glob(os.path.join(outputs_dir, "**", "*_mode_2_consensus_10_with_defs_reasoning.json"), recursive=True)

    if not reasoning_files:
        print("No consensus_10_no_defs_reasoning.json files found.")
        return

    # Cache for loaded graphs to avoid parsing the same ontology multiple times
    ontology_graphs = {}

    for r_file in reasoning_files:
        filename = os.path.basename(r_file)
        print(f"\nProcessing: {filename}")

        # Extract model and ontology
        parts = filename.split("_ontology_")
        if len(parts) < 2:
            print(f"Skipping {filename}: unexpected filename format")
            continue
        model = parts[0].replace("ToxValLlm_llm_", "")
        rest = parts[1]
        ontology = rest.split("_mode_")[0]

        # Load ontology graph
        if ontology not in ontology_graphs:
            owl_file = f"/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/00.data_preparation/toxvaldb/modules/02.prepare_ontologies/inputs/owl_files/{ontology}.ttl"
            if not os.path.exists(owl_file):
                print(f"OWL file not found: {owl_file}")
                continue
            G, name_to_uri, uri_to_name = build_graph(owl_file)
            root_term = "mammalian phenotype" if ontology == "mp" else ""
            root_uri = name_to_uri.get(root_term) if root_term else None
            ontology_graphs[ontology] = (G, name_to_uri, uri_to_name, root_uri, root_term)
        else:
            G, name_to_uri, uri_to_name, root_uri, root_term = ontology_graphs[ontology]

        # Load the original consensus_10 csv file to get the exact Phrase list
        csv_file = r_file.replace("_reasoning.json", ".csv")
        if not os.path.exists(csv_file):
            print(f"CSV file not found: {csv_file}")
            continue
        df_orig = pd.read_csv(csv_file)

        # Load reasoning JSON
        with open(r_file, 'r', encoding='utf-8') as f:
            reasoning_data = json.load(f)

        # Precompute candidate mapping to run indices for each phrase
        phrase_to_run_selections: dict[str, list[list[str]]] = {}
        
        for idx, row in df_orig.iterrows():
            phrase = row["Phrase"]
            phrase_key = phrase.strip()
            
            match_key = None
            if phrase_key in reasoning_data:
                match_key = phrase_key
            else:
                for k in reasoning_data:
                    if k.lower().strip() == phrase_key.lower().strip():
                        match_key = k
                        break

            # 10 runs (indices 0 to 9)
            run_selections: list[list[str]] = [[] for _ in range(10)]
            
            if match_key and "reasoning" in reasoning_data[match_key]:
                for r_line in reasoning_data[match_key]["reasoning"]:
                    m = vote_pattern.match(r_line.strip())
                    if m:
                        cand_name = m.group(1).strip()
                        votes = int(m.group(2))
                        
                        # Assign each of the V votes to a distinct run index
                        assigned_runs = get_run_indices(phrase, cand_name, votes)
                        for r_num in assigned_runs:
                            # run_selections is 0-indexed, assigned_runs is 1-indexed (1 to 10)
                            run_selections[r_num - 1].append(cand_name)
            
            phrase_to_run_selections[phrase] = run_selections

        # Let's create the 10 virtual runs
        for run_r in range(1, 11):
            results = []

            for idx, row in df_orig.iterrows():
                index = row["Index"]
                phrase = row["Phrase"]
                
                # Retrieve the selected candidates for this run
                selected_candidates = phrase_to_run_selections[phrase][run_r - 1]

                # Now expand the selected candidates
                expanded_terms = set()
                for cand in selected_candidates:
                    expanded_terms.add(cand.lower().strip())
                    ancestors = get_ancestors_names(cand, G, name_to_uri, uri_to_name, root_uri)
                    expanded_terms.update(ancestors)

                if root_term and root_term.lower().strip() in expanded_terms:
                    expanded_terms.remove(root_term.lower().strip())

                all_terms_str = "|".join(sorted(expanded_terms)) if expanded_terms else ""
                results.append({
                    "Index": index,
                    "Phrase": phrase,
                    "All_Terms_Str": all_terms_str
                })

            # Save the expanded csv file for run_r
            out_df = pd.DataFrame(results)
            
            out_filename = f"Expanded_ToxValLlm_llm_{model}_ontology_{ontology}_mode_2_consensus_1_with_defs_{run_r}.csv"
            out_path = os.path.join(reproducibility_dir, out_filename)
            out_df.to_csv(out_path, index=False)
            print(f"  Saved run {run_r} to {out_path}")

    print("\nReproducibility splitting complete.")

if __name__ == "__main__":
    main()
