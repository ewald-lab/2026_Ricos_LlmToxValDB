from __future__ import annotations
import argparse
import glob
import os
import sqlite3
import json
import textwrap
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np
import pandas as pd
import rdflib
import seaborn as sns
from rdflib.namespace import RDFS
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Any


def arg_parser() -> argparse.Namespace:
    """
    Description:
    Parses command-line arguments for the combined evaluation script.

    Args:
        None

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate combined evaluation plots comparing human experts and LLMs."
    )
    parser.add_argument(
        "--expert_expanded_dir",
        required=True,
        help="Directory containing human expert Expanded_*.csv files.",
    )
    parser.add_argument(
        "--expert_orig_dir",
        required=True,
        help="Directory containing original ToxValLlm_expert_*.csv files.",
    )
    parser.add_argument(
        "--llm_expanded_dir",
        required=True,
        help="Directory containing LLM Expanded_ToxValLlm_llm_*.csv files.",
    )
    parser.add_argument(
        "--llm_orig_dir",
        required=True,
        help="Directory containing original ToxValLlm_llm_*.csv files.",
    )
    parser.add_argument(
        "--owl_file",
        required=True,
        help="Path to the ontology TTL/OWL file (used for depth computation).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where combined plots will be saved.",
    )
    parser.add_argument(
        "--ontology",
        required=True,
        help="Ontology short name (e.g. mp). Used to match files.",
    )
    parser.add_argument(
        "--root_term",
        default="mammalian phenotype",
        help="Label of the ontology root node (e.g. 'mammalian phenotype').",
    )
    parser.add_argument(
        "--exclude_average_human",
        action="store_true",
        help="Exclude average human data from the combined plots.",
    )
    parser.add_argument(
        "--skip_inter_annotator_agreement_boxplot",
        action="store_true",
        help="Skip inter_annotator_agreement_boxplot.png generation.",
    )
    parser.add_argument(
        "--ontology_embedding_db",
        default=None,
        help="Path to the ontology embedding DB (e.g. kalm/mp.db) for PCA visualization.",
    )
    parser.add_argument(
        "--expert_equivalence_file",
        default=None,
        help="Path to the TXT file containing expert name equivalences.",
    )
    return parser.parse_args()


EXPERT_MAP = {
    "Alexander_Cole_Dynanet_Corporation": r"$\mathbf{Expert\ 1}$",
    "Chelsea_Weitekamp_ULRI_Chemical_Insights": r"$\mathbf{Expert\ 2}$",
    "Joshua_Harrill_Corteva": r"$\mathbf{Expert\ 3}$",
    "Liz_McInnes_Syngenta": r"$\mathbf{Expert\ 4}$",
    "Lucie_Ford_Recursion": r"$\mathbf{Expert\ 5}$",
    "Sean_Watford_ULRI": r"$\mathbf{Expert\ 6}$",
}


def load_expert_map(filepath: str) -> dict[str, str]:
    expert_map = {}
    if not filepath or not os.path.exists(filepath):
        return expert_map
    import re
    pattern = re.compile(r"Expert\s+(\d+):.*?\((.*?)\)")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                num = match.group(1)
                expert_id = match.group(2).strip()
                expert_map[expert_id] = f"$\\mathbf{{Expert\\ {num}}}$"
    return expert_map


def clean_expert_name(name: str) -> str:
    """
    Description:
    Cleans and formats an expert's name from their identifier.

    Args:
        name (str): The raw expert identifier.

    Returns:
        str: The cleaned, formatted expert name.
    """
    if name == "average_human":
        return "Average Human"
    if name in EXPERT_MAP:
        return EXPERT_MAP[name]
    for k, v in EXPERT_MAP.items():
        if k.lower() in name.lower() or name.lower() in k.lower():
            return v
    parts = name.split("_")
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return name.replace("_", " ")


def clean_llm_name(name: str) -> str:
    """
    Description:
    Cleans and formats an LLM configuration name from its identifier.

    Args:
        name (str): The raw LLM identifier.

    Returns:
        str: The formatted LLM configuration name including mode, consensus, and definitions settings.
    """
    name = name.replace("ToxValLlm_llm_", "")
    parts = name.split("_ontology_")
    if len(parts) < 2:
        return name.replace("_", " ")

    model_part = parts[0]
    rest = parts[1]

    mode = "M1" if "mode_1" in rest else "M2"

    consensus = "C1"
    for c in ["10", "5", "3", "1"]:
        if f"consensus_{c}" in rest:
            consensus = f"C{c}"
            break

    defs = "with_defs" in rest
    defs_str = "defs" if defs else "no defs"

    return f"{model_part} ({mode}, {consensus}, {defs_str})"


def parse_llm_metadata(llm_id: str) -> dict[str, Any] | None:
    """
    Description:
    Parses metadata (model, mode, consensus runs, use of definitions) from an LLM identifier.

    Args:
        llm_id (str): The raw LLM identifier.

    Returns:
        dict[str, Any] | None: A dictionary of metadata properties, or None if the identifier is invalid.
    """
    parts = llm_id.split("_ontology_")
    if len(parts) < 2:
        return None
    model = parts[0].replace("ToxValLlm_llm_", "")
    rest = parts[1]

    # Mode
    mode = None
    if "mode_1" in rest:
        mode = 1
    elif "mode_2" in rest:
        mode = 2

    # Consensus
    consensus = None
    for c in [10, 5, 3, 1]:
        if f"consensus_{c}" in rest:
            consensus = c
            break

    # Definitions
    defs = "with_defs" in rest

    return {
        "model": model,
        "mode": mode,
        "consensus": consensus,
        "definitions": defs
    }


def calc_metrics(pred_set: set[str], ref_set: set[str]) -> tuple[float, float, float]:
    """
    Description:
    Calculates precision, recall, and F1 score between a predicted set of terms and a reference set of terms.

    Args:
        pred_set (set[str]): The set of predicted terms.
        ref_set (set[str]): The set of reference/ground truth terms.

    Returns:
        tuple[float, float, float]: A tuple containing (precision, recall, F1 score).
    """
    if not pred_set and not ref_set:
        return 1.0, 1.0, 1.0  # True negative / complete agreement on empty
    if not pred_set or not ref_set:
        return 0.0, 0.0, 0.0
    intersection = pred_set.intersection(ref_set)
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(ref_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def load_repro_selected_sets(llm_id: str, expanded_dir: str) -> dict[str, list[set[str]]]:
    """
    Parses the consensus_10 reasoning JSON file for a given model to reconstruct
    the exact selected candidates (non-expanded) for each of the 10 virtual runs.
    """
    import re
    # llm_id is like ToxValLlm_llm_Qwen3.5-35B-A3B-FP8_ontology_mp_mode_2_consensus_1_no_defs
    parts = llm_id.split("_ontology_")
    if len(parts) < 2:
        return {}
    model = parts[0].replace("ToxValLlm_llm_", "")
    
    r_file = os.path.join(expanded_dir, model, llm_id.replace("consensus_1", "consensus_10") + "_reasoning.json")
    if not os.path.exists(r_file):
        r_file = os.path.join(expanded_dir, llm_id.replace("consensus_1", "consensus_10") + "_reasoning.json")
        if not os.path.exists(r_file):
            return {}

    try:
        with open(r_file, "r", encoding="utf-8") as f:
            reasoning_data = json.load(f)
    except Exception as e:
        print(f"  WARNING: Could not load reasoning JSON {r_file}: {e}")
        return {}

    vote_pattern = re.compile(r'^-\s*(.*?)\s*\(\d+/\d+\):\s*(\d+)\s*votes\s*->\s*(ACCEPTED|REJECTED)$')
    
    result = {}
    for phrase, p_data in reasoning_data.items():
        run_sets = [set() for _ in range(10)]
        if "reasoning" in p_data:
            for r_line in p_data["reasoning"]:
                m = vote_pattern.match(r_line.strip())
                if m:
                    cand_name = m.group(1).strip()
                    votes = int(m.group(2))
                    # Assign to runs 1 to votes
                    for r_idx in range(min(votes, 10)):
                        run_sets[r_idx].add(cand_name)
        result[phrase.lower().strip()] = run_sets
    return result


def build_depth_map(owl_file: str, root_term: str | None) -> dict[str, int]:
    """
    Description:
    Parses an ontology file and builds a mapping from lowercased term labels to their shortest path depth from the root term.

    Args:
        owl_file (str): The path to the ontology OWL/TTL file.
        root_term (str | None): The label of the ontology root term.

    Returns:
        dict[str, int]: A dictionary mapping term labels to their integer depth.
    """
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

    depth_map: dict = {}
    if root_term:
        root_name = root_term.lower().strip()
        root_uri = name_to_uri.get(root_name)
        if root_uri:
            G_rev = G.reverse()
            lengths = nx.single_source_shortest_path_length(G_rev, root_uri)
            for node, d in lengths.items():
                if node in uri_to_name:
                    depth_map[uri_to_name[node]] = d

    return depth_map


def get_max_depth(term_set: set[str], depth_map: dict[str, int]) -> int:
    """
    Description:
    Computes the maximum depth from the ontology root across a set of terms.

    Args:
        term_set (set[str]): The set of ontology terms.
        depth_map (dict[str, int]): A dictionary mapping terms to their depth.

    Returns:
        int: The maximum depth found, or 0 if the term set is empty.
    """
    if not term_set:
        return 0
    depths = [depth_map.get(t, 0) for t in term_set]
    return np.max(depths)


def count_valid_terms(term_str: Any) -> int:
    """
    Description:
    Counts the number of valid terms in a pipe-separated string, filtering out placeholders.

    Args:
        term_str (Any): The pipe-separated string or object representing the terms.

    Returns:
        int: The count of valid terms.
    """
    if pd.isna(term_str):
        return 0
    terms = [t.strip().lower() for t in str(term_str).split("|")]
    valid_t = [t for t in terms if t not in ["none", "-", "unknown", ""]]
    return len(valid_t)


def parse_term_set(term_str: Any) -> set[str]:
    """
    Description:
    Parses a pipe-separated term string into a normalized set.
    """
    if pd.isna(term_str):
        return set()
    terms = [t.strip().lower() for t in str(term_str).split("|")]
    return set(t for t in terms if t not in ["none", "-", "unknown", ""])


def wrap_plot_label(label: str, width: int = 34) -> str:
    """
    Description:
    Wraps long labels for horizontal bar plots.
    """
    return "\n".join(textwrap.wrap(str(label), width=width, break_long_words=False))





def main() -> None:
    """
    Description:
    Main entry point for generating combined evaluation plots and model comparison tables.

    Args:
        None

    Returns:
        None
    """
    args = arg_parser()

    if args.expert_equivalence_file:
        loaded_map = load_expert_map(args.expert_equivalence_file)
        if loaded_map:
            EXPERT_MAP.clear()
            EXPERT_MAP.update(loaded_map)

    os.makedirs(args.output_dir, exist_ok=True)

    depth_map = build_depth_map(args.owl_file, args.root_term)

    # 1. Load Human Expert Data
    exp_expert_files = sorted(glob.glob(os.path.join(args.expert_expanded_dir, "Expanded_*.csv")))
    if args.exclude_average_human:
        exp_expert_files = [f for f in exp_expert_files if "average_human" not in os.path.basename(f)]

    # 2. Load LLM Data recursively
    exp_llm_files = glob.glob(os.path.join(args.llm_expanded_dir, "**", "Expanded_ToxValLlm_llm_*.csv"), recursive=True)
    if not exp_llm_files:
        exp_llm_files = glob.glob(os.path.join(args.llm_expanded_dir, "**", "Expanded_*.csv"), recursive=True)
        # Exclude human ones if they are in the same folder
        exp_llm_files = [f for f in exp_llm_files if "ToxValLlm_llm_" in os.path.basename(f)]

    all_data: dict = {}
    all_metrics: list = []

    print("Processing expert data...")
    for f in exp_expert_files:
        fname = os.path.basename(f)
        expert_id = fname.replace("Expanded_", "").replace(".csv", "")
        df_exp = pd.read_csv(f)
        df_exp["All_Terms_Set"] = df_exp["All_Terms_Str"].apply(parse_term_set)
        all_data[expert_id] = df_exp

        # Load original expert files for counts
        if expert_id == "average_human":
            # For average human, there is no direct ToxValLlm_expert_average_human_ontology_mp.csv
            df_orig = None
        else:
            orig_path = os.path.join(
                args.expert_orig_dir,
                f"ToxValLlm_expert_{expert_id}_ontology_{args.ontology}.csv",
            )
            if not os.path.exists(orig_path):
                print(f"  WARNING: expert original file not found: {orig_path} — skipping flags.")
                df_orig = None
            else:
                df_orig = pd.read_csv(orig_path)

        # Add Not_My_Expertise column to df_exp (treating both Not_My_Expertise and Toxicological_Effect_Does_Not_Make_Sense as exclusion criteria)
        if df_orig is not None:
            not_exp = pd.to_numeric(df_orig.get("Not_My_Expertise", 0), errors="coerce").fillna(0)
            nonsense = pd.to_numeric(df_orig.get("Toxicological_Effect_Does_Not_Make_Sense", 0), errors="coerce").fillna(0)
            df_exp["Not_My_Expertise"] = ((not_exp == 1) | (nonsense == 1)).astype(int).values
            df_exp["Selected_Terms_Set"] = df_orig.get(
                "Selected_Candidates", pd.Series([None] * len(df_exp))
            ).apply(parse_term_set).values
        else:
            df_exp["Not_My_Expertise"] = 0
            df_exp["Selected_Terms_Set"] = [set() for _ in range(len(df_exp))]

        df_exp["Max_Depth"] = df_exp["All_Terms_Set"].apply(
            lambda x: get_max_depth(x, depth_map)
        )

        selected_count_total = (
            df_orig["Selected_Candidates"].apply(count_valid_terms).sum()
            if df_orig is not None and "Selected_Candidates" in df_orig.columns else 0
        )
        manual_count_total = 0
        nonsense_count = (
            pd.to_numeric(
                df_orig["Toxicological_Effect_Does_Not_Make_Sense"], errors="coerce"
            ).fillna(0).sum()
            if df_orig is not None and "Toxicological_Effect_Does_Not_Make_Sense" in df_orig.columns else 0
        )
        not_expertise_count = (
            pd.to_numeric(df_orig["Not_My_Expertise"], errors="coerce").fillna(0).sum()
            if df_orig is not None and "Not_My_Expertise" in df_orig.columns else 0
        )
        if df_orig is not None:
            if "Not_My_Expertise" in df_orig.columns:
                not_exp_mask = pd.to_numeric(df_orig["Not_My_Expertise"], errors="coerce").fillna(0) == 1
                filtered_df_exp = df_exp[~not_exp_mask]
            else:
                filtered_df_exp = df_exp
            avg_orig_terms = (
                filtered_df_exp["All_Terms_Set"].apply(len).mean()
                if not filtered_df_exp.empty else 0
            )
        else:
            avg_orig_terms = df_exp["All_Terms_Set"].apply(len).mean()

        filtered_for_depth = df_exp[df_exp["Not_My_Expertise"] != 1]
        mean_max_depth = filtered_for_depth["Max_Depth"].mean() if not filtered_for_depth.empty else 0

        all_metrics.append(
            {
                "Entity_ID": expert_id,
                "Entity": clean_expert_name(expert_id),
                "Type": "Human Expert",
                "Avg_Orig_Num_Terms": avg_orig_terms,
                "Mean_Max_Depth": mean_max_depth,
                "Selected_Terms": selected_count_total,
                "Manual_Terms": manual_count_total,
                "Nonsense_Flags": nonsense_count,
                "Not_Expertise_Flags": not_expertise_count,
            }
        )

    # Load average human response as ground truth reference for Precision/Recall calculation
    avg_human_file = os.path.join(args.expert_expanded_dir, "Expanded_average_human.csv")
    human_ref_map = None
    if os.path.exists(avg_human_file):
        print(f"Loading human average reference from: {avg_human_file}")
        df_avg_human = pd.read_csv(avg_human_file)
        df_avg_human["All_Terms_Set"] = df_avg_human["All_Terms_Str"].apply(parse_term_set)
        human_ref_map = {row["Index"]: row["All_Terms_Set"] for _, row in df_avg_human.iterrows()}
    else:
        print(f"WARNING: Expanded_average_human.csv not found at {avg_human_file}. Cannot compute LLM vs. Human metrics.")

    llm_perf_metrics = []

    print("Processing LLM response data...")
    for f in exp_llm_files:
        fname = os.path.basename(f)
        llm_id = fname.replace("Expanded_", "").replace(".csv", "")
        df_exp = pd.read_csv(f)
        df_exp["All_Terms_Set"] = df_exp["All_Terms_Str"].apply(parse_term_set)
        all_data[llm_id] = df_exp

        # Load original LLM files for counts (in the same subdirectory as the expanded file)
        orig_path = os.path.join(os.path.dirname(f), f"{llm_id}.csv")
        if not os.path.exists(orig_path):
            print(f"  WARNING: LLM original file not found: {orig_path} — skipping flags.")
            df_orig = None
        else:
            df_orig = pd.read_csv(orig_path)

        # Add Not_My_Expertise column to df_exp (treating both Not_My_Expertise and Toxicological_Effect_Does_Not_Make_Sense as exclusion criteria)
        if df_orig is not None:
            not_exp = pd.to_numeric(df_orig.get("Not_My_Expertise", 0), errors="coerce").fillna(0)
            nonsense = pd.to_numeric(df_orig.get("Toxicological_Effect_Does_Not_Make_Sense", 0), errors="coerce").fillna(0)
            df_exp["Not_My_Expertise"] = ((not_exp == 1) | (nonsense == 1)).astype(int).values
            df_exp["Selected_Terms_Set"] = df_orig.get(
                "Selected_Candidates", pd.Series([None] * len(df_exp))
            ).apply(parse_term_set).values
        else:
            df_exp["Not_My_Expertise"] = 0
            df_exp["Selected_Terms_Set"] = [set() for _ in range(len(df_exp))]

        df_exp["Max_Depth"] = df_exp["All_Terms_Set"].apply(
            lambda x: get_max_depth(x, depth_map)
        )

        selected_count_total = (
            df_orig["Selected_Candidates"].apply(count_valid_terms).sum()
            if df_orig is not None and "Selected_Candidates" in df_orig.columns else 0
        )
        manual_count_total = 0
        nonsense_count = (
            pd.to_numeric(
                df_orig["Toxicological_Effect_Does_Not_Make_Sense"], errors="coerce"
            ).fillna(0).sum()
            if df_orig is not None and "Toxicological_Effect_Does_Not_Make_Sense" in df_orig.columns else 0
        )
        not_expertise_count = (
            pd.to_numeric(df_orig["Not_My_Expertise"], errors="coerce").fillna(0).sum()
            if df_orig is not None and "Not_My_Expertise" in df_orig.columns else 0
        )
        # Check if it is a reproducibility run (ends with _1 to _10)
        is_repro_run = False
        parts_id = llm_id.split("_")
        if len(parts_id) > 1 and parts_id[-1].isdigit() and 1 <= int(parts_id[-1]) <= 10:
            is_repro_run = True

        avg_orig_terms = 0.0
        mean_max_depth = 0.0

        if not is_repro_run:
            # Find reproducibility runs
            df_runs = []
            for run_r in range(1, 11):
                run_file = os.path.join(os.path.dirname(f), "reproducibility", f"Expanded_{llm_id}_{run_r}.csv")
                if os.path.exists(run_file):
                    df_run = pd.read_csv(run_file)
                    df_run["All_Terms_Set"] = df_run["All_Terms_Str"].apply(parse_term_set)
                    df_runs.append(df_run)

            if not df_runs:
                df_runs = [df_exp]

            run_avg_orig_terms = []
            run_mean_max_depth = []

            for df_run in df_runs:
                # Calculate metrics for this run
                if df_orig is not None:
                    if "Not_My_Expertise" in df_orig.columns:
                        not_exp_mask = pd.to_numeric(df_orig["Not_My_Expertise"], errors="coerce").fillna(0) == 1
                        filtered_df_run = df_run[~not_exp_mask]
                    else:
                        filtered_df_run = df_run
                    run_avg = filtered_df_run["All_Terms_Set"].apply(len).mean() if not filtered_df_run.empty else 0
                else:
                    run_avg = df_run["All_Terms_Set"].apply(len).mean()
                
                df_run["Max_Depth"] = df_run["All_Terms_Set"].apply(lambda x: get_max_depth(x, depth_map))
                filtered_for_depth = df_run[df_run.get("Not_My_Expertise", 0) != 1]
                run_depth = filtered_for_depth["Max_Depth"].mean() if not filtered_for_depth.empty else 0

                run_avg_orig_terms.append(run_avg)
                run_mean_max_depth.append(run_depth)

            avg_orig_terms = np.mean(run_avg_orig_terms)
            mean_max_depth = np.mean(run_mean_max_depth)

        if not is_repro_run:
            all_metrics.append(
                {
                    "Entity_ID": llm_id,
                    "Entity": clean_llm_name(llm_id),
                    "Type": "LLM",
                    "Avg_Orig_Num_Terms": avg_orig_terms,
                    "Mean_Max_Depth": mean_max_depth,
                    "Selected_Terms": selected_count_total,
                    "Manual_Terms": manual_count_total,
                    "Nonsense_Flags": nonsense_count,
                    "Not_Expertise_Flags": not_expertise_count,
                }
            )

            metadata = parse_llm_metadata(llm_id)

            # Compute Precision & Recall vs. Human average reference
            mean_p, mean_r, mean_f1 = 0.0, 0.0, 0.0
            if human_ref_map is not None:
                total_intersection = 0
                total_pred = 0
                total_true = 0
                for idx, human_terms in human_ref_map.items():
                    llm_row = df_exp[df_exp["Index"] == idx]
                    if len(llm_row) == 0:
                        continue
                    # Skip evaluating this phrase if the human reference has no terms (no consensus or not_my_expertise)
                    if not human_terms:
                        continue
                    llm_terms = llm_row.iloc[0]["All_Terms_Set"]
                    
                    total_intersection += len(llm_terms.intersection(human_terms))
                    total_pred += len(llm_terms)
                    total_true += len(human_terms)

                mean_p = total_intersection / total_pred if total_pred > 0 else 0.0
                mean_r = total_intersection / total_true if total_true > 0 else 0.0
                mean_f1 = (2 * mean_p * mean_r) / (mean_p + mean_r) if (mean_p + mean_r) > 0 else 0.0

            # Extract token usage
            tokens_path = os.path.join(os.path.dirname(f), f"{llm_id}_tokens.txt")
            avg_completion_tokens = 0.0
            if os.path.exists(tokens_path):
                try:
                    with open(tokens_path, "r", encoding="utf-8") as tf:
                        for line in tf:
                            if line.startswith("Mean:"):
                                avg_completion_tokens = float(line.replace("Mean:", "").strip())
                except Exception as re:
                    print(f"  WARNING: Error reading tokens from {tokens_path}: {re}")

            if metadata:
                llm_perf_metrics.append({
                    "LLM_ID": llm_id,
                    "Model": metadata["model"].replace("-fp8", "").replace("-FP8", ""),
                    "Mode": f"Mode {metadata['mode']}" if metadata["mode"] else "Unknown",
                    "Consensus": metadata["consensus"],
                    "Definitions": "With Defs" if metadata["definitions"] else "No Defs",
                    "Precision": mean_p,
                    "Recall": mean_r,
                    "F1": mean_f1,
                    "Tokens": avg_completion_tokens
                })

    metrics_df = pd.DataFrame(all_metrics)
    entity_ids = list(all_data.keys())

    # Calculate actual clean names list mapping for indexing consistency
    id_to_clean = {}
    for eid in entity_ids:
        if eid.startswith("ToxValLlm_llm_"):
            id_to_clean[eid] = clean_llm_name(eid)
        else:
            id_to_clean[eid] = clean_expert_name(eid)

    # Find the single best configuration for each unique model (highest F1 score)
    best_config_per_model = {}
    best_llm_ids = set()
    llm_perf_df = pd.DataFrame()
    if llm_perf_metrics:
        llm_perf_df = pd.DataFrame(llm_perf_metrics)
        for m in sorted(llm_perf_df["Model"].unique()):
            model_df = llm_perf_df[llm_perf_df["Model"] == m]
            if not model_df.empty:
                # Select configuration: Consensus=1, Mode 2, No Defs
                target_rows = model_df[
                    (model_df["Mode"] == "Mode 2") & 
                    (model_df["Consensus"] == 1) & 
                    (model_df["Definitions"] == "No Defs")
                ]
                if not target_rows.empty:
                    best_row = target_rows.iloc[0]
                else:
                    best_row = model_df.sort_values(by="LLM_ID", ascending=True).iloc[0]
                best_config_per_model[m] = {
                    "Mode": best_row["Mode"],
                    "Definitions": best_row["Definitions"],
                    "Consensus": best_row["Consensus"],
                    "F1": best_row["F1"],
                    "LLM_ID": best_row["LLM_ID"]
                }
                best_llm_ids.add(best_row["LLM_ID"])

    # Pairwise overlap (Jaccard)
    # Filter entity IDs for pairwise similarity plots (Plot 1, Plot 6, Plot 15)
    # We include all human experts, and the best configuration for each LLM model
    pairwise_entity_ids = []
    for eid in entity_ids:
        if not eid.startswith("ToxValLlm_llm_"):
            # Human expert
            pairwise_entity_ids.append(eid)
        else:
            if eid in best_llm_ids:
                pairwise_entity_ids.append(eid)

    # Fallback in case no LLM matches (e.g. test runs)
    if not any(eid.startswith("ToxValLlm_llm_") for eid in pairwise_entity_ids):
        pairwise_entity_ids = entity_ids[:]

    expert_alias_by_id = {
        eid: clean_expert_name(eid)
        for eid in pairwise_entity_ids if not eid.startswith("ToxValLlm_llm_")
    }
    for eid, alias in expert_alias_by_id.items():
        id_to_clean[eid] = alias
        metrics_df.loc[metrics_df["Entity_ID"] == eid, "Entity"] = alias

    pairwise_cleaned_names = []
    for eid in pairwise_entity_ids:
        if eid.startswith("ToxValLlm_llm_"):
            meta = parse_llm_metadata(eid)
            name = meta["model"] if meta else clean_llm_name(eid)
            clean_name = name.replace("-fp8", "").replace("-FP8", "")
            pairwise_cleaned_names.append(clean_name)
            metrics_df.loc[metrics_df["Entity_ID"] == eid, "Entity"] = clean_name
        else:
            pairwise_cleaned_names.append(expert_alias_by_id[eid])


    print("Calculating pairwise overlap...")
    num_phrases = len(all_data[pairwise_entity_ids[0]])
    pairwise_overlap = np.zeros((len(pairwise_entity_ids), len(pairwise_entity_ids)))
    for i, id1 in enumerate(pairwise_entity_ids):
        for j, id2 in enumerate(pairwise_entity_ids):
            # Get run list for id1
            if id1.startswith("ToxValLlm_llm_"):
                df_runs1 = []
                for run_r in range(1, 11):
                    run_key = f"{id1}_{run_r}"
                    if run_key in all_data:
                        df_runs1.append(all_data[run_key])
                    else:
                        run_file = os.path.join(args.llm_expanded_dir, "reproducibility", f"Expanded_{id1}_{run_r}.csv")
                        if os.path.exists(run_file):
                            df_run = pd.read_csv(run_file)
                            df_run["All_Terms_Set"] = df_run["All_Terms_Str"].apply(parse_term_set)
                            df_runs1.append(df_run)
                if not df_runs1:
                    df_runs1 = [all_data[id1]]
            else:
                df_runs1 = [all_data[id1]]

            # Get run list for id2
            if id2.startswith("ToxValLlm_llm_"):
                df_runs2 = []
                for run_r in range(1, 11):
                    run_key = f"{id2}_{run_r}"
                    if run_key in all_data:
                        df_runs2.append(all_data[run_key])
                    else:
                        run_file = os.path.join(args.llm_expanded_dir, "reproducibility", f"Expanded_{id2}_{run_r}.csv")
                        if os.path.exists(run_file):
                            df_run = pd.read_csv(run_file)
                            df_run["All_Terms_Set"] = df_run["All_Terms_Str"].apply(parse_term_set)
                            df_runs2.append(df_run)
                if not df_runs2:
                    df_runs2 = [all_data[id2]]
            else:
                df_runs2 = [all_data[id2]]

            # Align run lists to same length
            max_len = max(len(df_runs1), len(df_runs2))
            if len(df_runs1) < max_len:
                df_runs1 = df_runs1 * max_len
            if len(df_runs2) < max_len:
                df_runs2 = df_runs2 * max_len

            # Calculate pairwise scores for each run pair, then average them
            pair_scores = []
            for df1, df2 in zip(df_runs1, df_runs2):
                scores = []
                for k in range(num_phrases):
                    not_exp1 = df1.iloc[k].get("Not_My_Expertise", 0) == 1
                    not_exp2 = df2.iloc[k].get("Not_My_Expertise", 0) == 1
                    if not_exp1 or not_exp2:
                        continue
                    s1 = df1.iloc[k]["All_Terms_Set"]
                    s2 = df2.iloc[k]["All_Terms_Set"]
                    u = s1.union(s2)
                    inter = s1.intersection(s2)
                    score = len(inter) / len(u) if len(u) > 0 else 1.0
                    scores.append(score)
                if scores:
                    pair_scores.append(np.mean(scores))
            
            pairwise_overlap[i, j] = np.mean(pair_scores) if pair_scores else 1.0

    print("Generating combined plots...")

    # Plot 1: Pairwise Heatmap / Clustermap
    num_entities_pw = len(pairwise_entity_ids)
    heatmap_width = max(12, num_entities_pw * 0.5 + 4)
    heatmap_height = max(10, num_entities_pw * 0.5 + 2)

    annot_font_size = max(5, 10 - num_entities_pw // 5)
    tick_font_size = max(6, 11 - num_entities_pw // 6)

    human_clean_names = {expert_alias_by_id[eid] for eid in pairwise_entity_ids if not eid.startswith("ToxValLlm_llm_")}

    if num_entities_pw > 1:
        df_overlap = pd.DataFrame(pairwise_overlap, index=pairwise_cleaned_names, columns=pairwise_cleaned_names)
        g = sns.clustermap(
            df_overlap,
            annot=True,
            annot_kws={"size": annot_font_size},
            cmap="Blues",
            fmt=".2f",
            figsize=(heatmap_width, heatmap_height),
        )
        # Position colorbar on the right, vertically aligned with the heatmap
        heatmap_pos = g.ax_heatmap.get_position()
        g.ax_cbar.set_position([1.02, heatmap_pos.y0, 0.03, heatmap_pos.height])

        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=tick_font_size)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=tick_font_size)

        # Color human expert names in red
        for label in g.ax_heatmap.get_xticklabels():
            if label.get_text() in human_clean_names:
                label.set_color("red")
        for label in g.ax_heatmap.get_yticklabels():
            if label.get_text() in human_clean_names:
                label.set_color("red")

        g.fig.suptitle("Combined Pairwise Overlap (Average Jaccard Similarity) - Humans & LLMs", fontsize=max(10, 14 - num_entities_pw // 8), fontweight="bold", y=1.02)
        g.savefig(os.path.join(args.output_dir, "combined_pairwise_overlap_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.figure(figsize=(heatmap_width, heatmap_height))
        sns.heatmap(
            pairwise_overlap,
            annot=True,
            annot_kws={"size": annot_font_size},
            xticklabels=pairwise_cleaned_names,
            yticklabels=pairwise_cleaned_names,
            cmap="Blues",
            fmt=".2f",
        )
        plt.title("Combined Pairwise Overlap (Average Jaccard Similarity) - Humans & LLMs", fontsize=max(10, 14 - num_entities_pw // 8), fontweight="bold")

        ax = plt.gca()
        plt.xticks(rotation=90, fontsize=tick_font_size)
        plt.yticks(rotation=0, fontsize=tick_font_size)

        # Color human expert names in red
        for label in ax.get_xticklabels():
            if label.get_text() in human_clean_names:
                label.set_color("red")
        for label in ax.get_yticklabels():
            if label.get_text() in human_clean_names:
                label.set_color("red")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "combined_pairwise_overlap_heatmap.png"), dpi=300)
        plt.close()

    # Filter metrics_df to only include the entities in pairwise_entity_ids (humans + best LLM configs)
    metrics_df = metrics_df[metrics_df["Entity_ID"].isin(pairwise_entity_ids)].copy()
    # Update Entity column for LLMs to use the clean model name
    for idx, row in metrics_df.iterrows():
        eid = row["Entity_ID"]
        if eid.startswith("ToxValLlm_llm_"):
            meta = parse_llm_metadata(eid)
            if meta:
                metrics_df.at[idx, "Entity"] = meta["model"].replace("-fp8", "").replace("-FP8", "")
    # Standalone Plot 2 & 3: Avg Terms and Max Depth plots removed as per user request



    # Plot 6: Dendrogram (Skipped as per user request)

    # 7. Generate Precision & Recall Plots vs. Human Average
    if human_ref_map is not None and not llm_perf_df.empty:
        pass

    human_indices = [i for i, eid in enumerate(pairwise_entity_ids) if not eid.startswith("ToxValLlm_llm_")]
    llm_indices = [i for i, eid in enumerate(pairwise_entity_ids) if eid.startswith("ToxValLlm_llm_")]

    # Plot 15: Inter-Annotator Agreement (Boxplot)
    if args.skip_inter_annotator_agreement_boxplot:
        pass
    else:
        try:
            print("Generating Inter-Annotator Agreement boxplot...")
            agreement_data = []

            # Human-Human
            for i in range(len(human_indices)):
                for j in range(i + 1, len(human_indices)):
                    idx1, idx2 = human_indices[i], human_indices[j]
                    agreement_data.append({"Type": "Human-Human", "Jaccard": pairwise_overlap[idx1, idx2]})

            # LLM-LLM
            for i in range(len(llm_indices)):
                for j in range(i + 1, len(llm_indices)):
                    idx1, idx2 = llm_indices[i], llm_indices[j]
                    agreement_data.append({"Type": "LLM-LLM", "Jaccard": pairwise_overlap[idx1, idx2]})

            # Human-LLM
            for h_idx in human_indices:
                for l_idx in llm_indices:
                    agreement_data.append({"Type": "Human-LLM", "Jaccard": pairwise_overlap[h_idx, l_idx]})

            if agreement_data:
                agreement_df = pd.DataFrame(agreement_data)
                plt.figure(figsize=(9, 6))
                sns.boxplot(
                    data=agreement_df,
                    x="Type",
                    y="Jaccard",
                    palette={"Human-Human": "skyblue", "LLM-LLM": "lightgreen", "Human-LLM": "salmon"}
                )
                plt.title("Inter-Annotator Agreement (Jaccard Similarity)", fontsize=14)
                plt.ylabel("Average Jaccard Similarity per Pair", fontsize=12)
                plt.xlabel("Agreement Type", fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, "inter_annotator_agreement_boxplot.png"), dpi=300)
                plt.close()
        except Exception as e:
            print(f"Error generating inter_annotator_agreement_boxplot.png: {e}")

    # Plot 16: Term-level Variability (Human, LLM, Human vs. LLM)
    try:
        print("Generating term-level variability comparison...")
        
        # Pre-load reproducibility selected sets for all LLMs in pairwise_entity_ids
        repro_selected_maps = {}
        for eid in pairwise_entity_ids:
            if eid.startswith("ToxValLlm_llm_"):
                repro_selected_maps[eid] = load_repro_selected_sets(eid, args.llm_expanded_dir)

        term_variability_data = []
        for k in range(num_phrases):
            phrase_name = all_data[pairwise_entity_ids[0]].iloc[k]["Phrase"]
            phrase_idx = all_data[pairwise_entity_ids[0]].iloc[k]["Index"]
            
            # Human sets (exclude Not_My_Expertise == 1)
            human_sets = []
            human_selected_sets = []
            for h_idx in human_indices:
                h_id = pairwise_entity_ids[h_idx]
                not_exp = all_data[h_id].iloc[k].get("Not_My_Expertise", 0) == 1
                if not not_exp:
                    human_sets.append(all_data[h_id].iloc[k]["All_Terms_Set"])
                    human_selected_sets.append(all_data[h_id].iloc[k].get("Selected_Terms_Set", set()))
            
            n_humans = len(human_sets)

            # LLM run sets
            llm_models_runs = []
            llm_models_selected_runs = []
            for l_idx in llm_indices:
                l_id = pairwise_entity_ids[l_idx]
                
                # Find reproducibility runs
                df_runs = []
                for run_r in range(1, 11):
                    run_key = f"{l_id}_{run_r}"
                    if run_key in all_data:
                        df_runs.append(all_data[run_key])
                    else:
                        run_file = os.path.join(args.llm_expanded_dir, "reproducibility", f"Expanded_{l_id}_{run_r}.csv")
                        if os.path.exists(run_file):
                            df_run = pd.read_csv(run_file)
                            df_run["All_Terms_Set"] = df_run["All_Terms_Str"].apply(parse_term_set)
                            df_runs.append(df_run)
                
                if not df_runs:
                    df_runs = [all_data[l_id]]

                model_sets = []
                model_selected_sets = []
                
                # Get the pre-loaded selected sets for this model
                sel_map = repro_selected_maps.get(l_id, {})
                phrase_key = phrase_name.lower().strip()
                phrase_run_sets = sel_map.get(phrase_key, [set() for _ in range(10)])

                for run_idx, df_run in enumerate(df_runs):
                    not_exp = df_run.iloc[k].get("Not_My_Expertise", 0) == 1
                    if not not_exp:
                        model_sets.append(df_run.iloc[k]["All_Terms_Set"])
                        # If we have the selections from the reasoning file, use them
                        if run_idx < len(phrase_run_sets):
                            model_selected_sets.append(phrase_run_sets[run_idx])
                        else:
                            model_selected_sets.append(df_run.iloc[k].get("Selected_Terms_Set", set()))
                
                if model_sets:
                    llm_models_runs.append(model_sets)
                    llm_models_selected_runs.append(model_selected_sets)
            
            # Human-Human
            hh_dists = []
            for i in range(len(human_sets)):
                for j in range(i + 1, len(human_sets)):
                    u = human_sets[i].union(human_sets[j])
                    jacc = len(human_sets[i].intersection(human_sets[j])) / len(u) if len(u) > 0 else 1.0
                    hh_dists.append(1.0 - jacc)
            avg_hh = np.mean(hh_dists) if hh_dists else np.nan
            
            # LLM-LLM
            # Average distance between different LLM models
            llm_dists = []
            for i in range(len(llm_models_runs)):
                for j in range(i + 1, len(llm_models_runs)):
                    runs_i = llm_models_runs[i]
                    runs_j = llm_models_runs[j]
                    max_runs = max(len(runs_i), len(runs_j))
                    if len(runs_i) < max_runs:
                        runs_i = runs_i * max_runs
                    if len(runs_j) < max_runs:
                        runs_j = runs_j * max_runs
                    
                    pair_dists = []
                    for r_i, r_j in zip(runs_i, runs_j):
                        u = r_i.union(r_j)
                        jacc = len(r_i.intersection(r_j)) / len(u) if len(u) > 0 else 1.0
                        pair_dists.append(1.0 - jacc)
                    if pair_dists:
                        llm_dists.append(np.mean(pair_dists))
            avg_llm = np.mean(llm_dists) if llm_dists else np.nan
            
            # Human-LLM
            hl_dists = []
            for h_set in human_sets:
                for model_runs in llm_models_runs:
                    model_dists = []
                    for l_set in model_runs:
                        u = h_set.union(l_set)
                        jacc = len(h_set.intersection(l_set)) / len(u) if len(u) > 0 else 1.0
                        model_dists.append(1.0 - jacc)
                    if model_dists:
                        hl_dists.append(np.mean(model_dists))
            avg_hl = np.mean(hl_dists) if hl_dists else np.nan

            # Human-LLM Ontology F1
            ontology_hl_f1s = []
            for h_selected_set in human_selected_sets:
                for model_selected_runs in llm_models_selected_runs:
                    model_f1s = []
                    for l_selected_set in model_selected_runs:
                        _, _, f1_val = calc_metrics(l_selected_set, h_selected_set)
                        model_f1s.append(f1_val)
                    if model_f1s:
                        ontology_hl_f1s.append(np.mean(model_f1s))
            avg_ontology_hl_f1 = np.mean(ontology_hl_f1s) if ontology_hl_f1s else np.nan

            avg_human_selected_terms = (
                np.mean([len(s) for s in human_selected_sets]) if human_selected_sets else np.nan
            )
            
            # Average number of selected terms for LLMs
            llm_term_counts = []
            for model_selected_runs in llm_models_selected_runs:
                model_counts = [len(s) for s in model_selected_runs]
                if model_counts:
                    llm_term_counts.append(np.mean(model_counts))
            avg_llm_selected_terms = np.mean(llm_term_counts) if llm_term_counts else np.nan

            llm_selected_term_excess = (
                avg_llm_selected_terms - avg_human_selected_terms
                if pd.notna(avg_human_selected_terms) and pd.notna(avg_llm_selected_terms)
                else np.nan
            )
            avg_human_terms = human_ref_map.get(phrase_idx, set()) if human_ref_map is not None else set()
            avg_human_f1s = []
            if avg_human_terms:
                for model_runs in llm_models_runs:
                    for l_set in model_runs:
                        _, _, f1_val = calc_metrics(l_set, avg_human_terms)
                        avg_human_f1s.append(f1_val)
            avg_human_ontology_f1 = np.mean(avg_human_f1s) if avg_human_f1s else np.nan
            
            phrase_display_name = f"{phrase_name} (n={n_humans})"
            
            term_variability_data.append({
                "Index": phrase_idx,
                "Raw_Phrase": phrase_name,
                "Phrase": phrase_display_name,
                "Human-Human Variability": avg_hh,
                "LLM-LLM Variability": avg_llm,
                "Human-LLM Variability": avg_hl,
                "Human Selected Terms": avg_human_selected_terms,
                "LLM Selected Terms": avg_llm_selected_terms,
                "LLM Selected Term Excess": llm_selected_term_excess,
                "Human-LLM Ontology F1": avg_ontology_hl_f1,
                "LLM-Average Human Ontology F1": avg_human_ontology_f1
            })
            
        var_df = pd.DataFrame(term_variability_data)
        var_df["Delta Human-LLM minus Human-Human"] = (
            var_df["Human-LLM Variability"] - var_df["Human-Human Variability"]
        )
        var_df["Delta LLM-LLM minus Human-Human"] = (
            var_df["LLM-LLM Variability"] - var_df["Human-Human Variability"]
        )
        
        # Generate Plot (3 stacked subplots sharing the x-axis)
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, sharey=True)
        
        # Clean phrases for display
        phrases_display = var_df["Phrase"].tolist()
        x_indices = np.arange(len(phrases_display))
        
        # Plot 1: Human-Human
        axes[0].bar(x_indices, var_df["Human-Human Variability"], color="skyblue", edgecolor="black", alpha=0.8)
        axes[0].set_ylabel("Variability")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_yticks([0.0, 0.5, 1.0])
        
        # Plot 2: LLM-LLM
        axes[1].bar(x_indices, var_df["LLM-LLM Variability"], color="lightgreen", edgecolor="black", alpha=0.8)
        axes[1].set_ylabel("Variability")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_yticks([0.0, 0.5, 1.0])
        
        # Plot 3: Human-LLM
        axes[2].bar(x_indices, var_df["Human-LLM Variability"], color="salmon", edgecolor="black", alpha=0.8)
        axes[2].set_ylabel("Variability")
        axes[2].set_ylim(0.0, 1.0)
        axes[2].set_yticks([0.0, 0.5, 1.0])
        
        # Format bottom x-axis with Phrase labels
        plt.xticks(x_indices, phrases_display, rotation=90, fontsize=8, ha="center")
        axes[2].set_xlabel("Toxicological Effect", fontsize=11, fontweight="bold")
        
        # Add legend to the right side
        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(facecolor="skyblue", edgecolor="black", label="Human-Human", alpha=0.8),
            mpatches.Patch(facecolor="lightgreen", edgecolor="black", label="LLM-LLM", alpha=0.8),
            mpatches.Patch(facecolor="salmon", edgecolor="black", label="Human-LLM", alpha=0.8)
        ]
        fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   title="Comparison Groups", title_fontsize=12, fontsize=11)
        
        fig.suptitle("Term-Level Variability Comparison (Humans & LLMs)", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "term_variability_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Term variability plot saved to: {os.path.join(args.output_dir, 'term_variability_comparison.png')}")

        # Paper-oriented summary: trend, direction of disagreement, and semantic pattern.
        paper_df = var_df.dropna(
            subset=[
                "Human-Human Variability",
                "LLM-LLM Variability",
                "Human-LLM Variability",
                "Delta Human-LLM minus Human-Human",
                "LLM Selected Term Excess",
                "LLM-Average Human Ontology F1",
            ]
        ).copy()

        if not paper_df.empty:
            print("Generating paper-ready term variability pattern summary (PNG only)...")
            delta_col = "Delta Human-LLM minus Human-Human"
            paper_df["Direction"] = np.where(
                paper_df[delta_col] > 0,
                "Human-LLM higher",
                "Human-Human higher",
            )
            plt.rcParams.update({
                "font.size": 14,
                "axes.titlesize": 17,
                "axes.labelsize": 15,
                "xtick.labelsize": 13,
                "ytick.labelsize": 13,
                "legend.fontsize": 13,
                "axes.spines.top": False,
                "axes.spines.right": False,
            })

            fig = plt.figure(figsize=(18, 18))
            gs = fig.add_gridspec(
                3,
                2,
                width_ratios=[1.0, 1.0],
                height_ratios=[1.0, 0.9, 1.8],
                hspace=0.35,
                wspace=0.20,
            )

            ax_scatter = fig.add_subplot(gs[0:2, 0])
            ax_dist = fig.add_subplot(gs[0, 1])
            import matplotlib.gridspec as gridspec
            gs_c = gridspec.GridSpecFromSubplotSpec(
                1,
                3,
                subplot_spec=gs[1, 1],
                width_ratios=[0.89, 0.03, 0.08],
                wspace=0.10,
            )
            ax_semantic = fig.add_subplot(gs_c[0, 0])
            ax_heatmap = fig.add_subplot(gs[2, 0])
            scatter_df = paper_df[paper_df[delta_col] != 0].copy()

            sns.scatterplot(
                data=scatter_df,
                x="Human-Human Variability",
                y="Human-LLM Variability",
                hue="Direction",
                hue_order=["Human-LLM higher", "Human-Human higher"],
                palette={
                    "Human-LLM higher": "salmon",
                    "Human-Human higher": "skyblue",
                },
                s=72,
                alpha=0.88,
                edgecolor="black",
                linewidth=0.45,
                ax=ax_scatter,
            )
            ax_scatter.plot([0, 1], [0, 1], color="0.35", linestyle="--", linewidth=1.1)
            ax_scatter.fill_between(
                [0, 1],
                [0, 1],
                [1, 1],
                color="salmon",
                alpha=0.06,
                linewidth=0,
            )
            ax_scatter.fill_between(
                [0, 1],
                [0, 0],
                [0, 1],
                color="skyblue",
                alpha=0.06,
                linewidth=0,
            )
            above_diag = int((paper_df["Delta Human-LLM minus Human-Human"] > 0).sum())
            below_diag = int((paper_df["Delta Human-LLM minus Human-Human"] < 0).sum())
            ax_scatter.set_xlim(0, 1)
            ax_scatter.set_ylim(0, 1)
            ax_scatter.set_xticks([0.0, 0.5, 1.0])
            ax_scatter.set_yticks([0.0, 0.5, 1.0])
            ax_scatter.set_xlabel("Human-Human variability")
            ax_scatter.set_ylabel("Human-LLM variability")
            ax_scatter.set_title("A. Disagreement trend", loc="left", fontweight="bold", fontsize=17, pad=24)
            ax_scatter.legend(
                loc="lower right",
                frameon=True,
                fontsize=11,
            )

            dist_df = paper_df.melt(
                value_vars=[
                    "Human-Human Variability",
                    "LLM-LLM Variability",
                    "Human-LLM Variability",
                ],
                var_name="Comparison",
                value_name="Variability",
            )
            dist_df["Comparison"] = dist_df["Comparison"].replace({
                "Human-Human Variability": "Human-Human",
                "LLM-LLM Variability": "LLM-LLM",
                "Human-LLM Variability": "Human-LLM",
            })
            comparison_order = ["Human-Human", "LLM-LLM", "Human-LLM"]
            comparison_palette = {
                "Human-Human": "skyblue",
                "LLM-LLM": "lightgreen",
                "Human-LLM": "salmon",
            }
            sns.boxplot(
                data=dist_df,
                x="Comparison",
                y="Variability",
                hue="Comparison",
                order=comparison_order,
                hue_order=comparison_order,
                palette=comparison_palette,
                showfliers=False,
                width=0.55,
                legend=False,
                ax=ax_dist,
            )
            sns.stripplot(
                data=dist_df,
                x="Comparison",
                y="Variability",
                order=comparison_order,
                color="0.25",
                alpha=0.45,
                size=3,
                jitter=0.22,
                ax=ax_dist,
            )
            ax_dist.set_ylim(0, 1.04)
            ax_dist.set_yticks([0.0, 0.5, 1.0])
            ax_dist.set_xlabel("")
            ax_dist.set_ylabel("Variability")
            ax_dist.set_title("B. Variability distributions", loc="left", fontweight="bold", fontsize=17, pad=24)

            correctness_df = paper_df[
                [
                    "LLM Selected Term Excess",
                    "LLM-Average Human Ontology F1",
                    "Human-LLM Variability",
                    "LLM Selected Terms",
                ]
            ].dropna().copy()
            correctness_df = correctness_df[correctness_df["LLM Selected Terms"] > 0]
            norm = mpl.colors.Normalize(
                vmin=0.0,
                vmax=1.0,
            )
            cmap = mpl.colors.LinearSegmentedColormap.from_list("custom", ["skyblue", "salmon"])
            ax_semantic.scatter(
                correctness_df["LLM Selected Term Excess"],
                correctness_df["LLM-Average Human Ontology F1"],
                c=correctness_df["Human-LLM Variability"],
                cmap=cmap,
                norm=norm,
                s=70,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.35,
            )
            cax = fig.add_subplot(gs_c[0, 1])
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cax,
            )
            cbar.set_label("Human-LLM variability", fontsize=13)
            cbar.set_ticks([0.0, 0.5, 1.0])
            x_min = correctness_df["LLM Selected Term Excess"].min()
            x_max = correctness_df["LLM Selected Term Excess"].max()
            x_pad = max(0.5, (x_max - x_min) * 0.08)
            ax_semantic.set_xlim(x_min - x_pad, x_max + x_pad)
            ax_semantic.set_ylim(0, 1.04)
            ax_semantic.set_yticks([0.0, 0.5, 1.0])
            ax_semantic.set_xlabel("LLM selected-term excess")
            ax_semantic.set_ylabel("F1 vs average human")
            ax_semantic.set_title("C. Consensus on extra LLM terms", loc="left", fontweight="bold", fontsize=17, pad=24)

            dist_matrix = 1 - pairwise_overlap
            np.fill_diagonal(dist_matrix, 0)
            dist_matrix = (dist_matrix + dist_matrix.T) / 2
            condensed_dist = squareform(dist_matrix)
            Z = linkage(condensed_dist, method="ward")

            dendrogram_tick_size = 13
            dendrogram(
                Z,
                labels=pairwise_cleaned_names,
                leaf_rotation=90,
                leaf_font_size=dendrogram_tick_size,
                color_threshold=0,
                above_threshold_color="skyblue",
                ax=ax_heatmap
            )
            ax_heatmap.set_ylabel("Distance")
            ax_heatmap.set_yticks([0.0, 0.5, 1.0])

            for label in ax_heatmap.get_xticklabels():
                if label.get_text() in human_clean_names:
                    label.set_color("black")
                    label.set_fontweight("bold")

            # Remove top and right spines from the dendrogram
            ax_heatmap.spines["top"].set_visible(False)
            ax_heatmap.spines["right"].set_visible(False)

            ax_heatmap.set_title(
                "D. Dendrogram of Human & LLM Similarity",
                loc="left",
                fontweight="bold",
                fontsize=17,
                pad=20,
            )

            import matplotlib.gridspec as gridspec
            gs_inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, 1], hspace=0.15)
            ax_terms = fig.add_subplot(gs_inner[0])
            ax_depth = fig.add_subplot(gs_inner[1])
            
            # Use same order as in Panel D dendrogram, if possible, or just default.
            sns.barplot(
                data=metrics_df,
                x="Entity",
                y="Avg_Orig_Num_Terms",
                hue="Type",
                palette={"Human Expert": "skyblue", "LLM": "salmon"},
                ax=ax_terms,
            )
            ax_terms.set_title("E. Number of Terms and Maximum Depth", loc="left", fontweight="bold", fontsize=17, pad=20)
            ax_terms.set_ylabel("Terms", fontsize=15)
            ax_terms.set_xlabel("")
            ax_terms.set_xticks([])
            
            if ax_terms.get_legend():
                ax_terms.get_legend().remove()
                
            sns.barplot(
                data=metrics_df,
                x="Entity",
                y="Mean_Max_Depth",
                hue="Type",
                palette={"Human Expert": "skyblue", "LLM": "salmon"},
                ax=ax_depth,
            )
            ax_depth.set_ylabel("Max Depth", fontsize=15)
            ax_depth.set_xlabel("")
            ax_depth.tick_params(axis='x', rotation=90, labelsize=13)
            
            if ax_depth.get_legend():
                ax_depth.get_legend().remove()
                
            for label in ax_depth.get_xticklabels():
                if label.get_text() in human_clean_names:
                    label.set_color("black")
                    label.set_fontweight("bold")

            fig.suptitle("")
            fig.savefig(
                os.path.join(args.output_dir, "term_variability_pattern_summary.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
            print(
                "Paper-ready term variability summary saved to: "
                f"{os.path.join(args.output_dir, 'term_variability_pattern_summary.png')}"
            )
        
    except Exception as e:
        print(f"Error generating term_variability_comparison.png: {e}")

    # Plot 17: 2D PCA of ontology term embeddings
    if args.ontology_embedding_db and os.path.exists(args.ontology_embedding_db):
        try:
            from sklearn.decomposition import PCA
            print("Generating PCA visualization of ontology term embeddings...")

            # Collect all terms selected by humans and/or LLMs (across all phrases)
            human_terms_all: set = set()
            llm_terms_all: set = set()

            for k in range(num_phrases):
                for h_idx in human_indices:
                    h_id = pairwise_entity_ids[h_idx]
                    not_exp = all_data[h_id].iloc[k].get("Not_My_Expertise", 0) == 1
                    if not not_exp:
                        human_terms_all.update(all_data[h_id].iloc[k]["All_Terms_Set"])
                for l_idx in llm_indices:
                    l_id = pairwise_entity_ids[l_idx]
                    not_exp = all_data[l_id].iloc[k].get("Not_My_Expertise", 0) == 1
                    if not not_exp:
                        llm_terms_all.update(all_data[l_id].iloc[k]["All_Terms_Set"])

            # Remove empty strings
            human_terms_all.discard("")
            llm_terms_all.discard("")
            terms_of_interest = human_terms_all | llm_terms_all

            # Load embeddings from the DB
            conn = sqlite3.connect(args.ontology_embedding_db)
            cursor = conn.cursor()
            cursor.execute("SELECT name, sanitized_name, vector FROM embeddings")
            rows = cursor.fetchall()
            conn.close()

            # Build name -> vector map (lowercased to match All_Terms_Set)
            term_vectors = {}
            term_display = {}  # name -> sanitized_name
            for (name, sanitized_name, blob) in rows:
                name_l = name.lower().strip()
                vec = np.frombuffer(bytearray(blob), dtype=np.float32).copy()
                term_vectors[name_l] = vec
                term_display[name_l] = sanitized_name or name

            # Filter to terms that appear in any selection AND have an embedding
            selected_terms = [t for t in terms_of_interest if t in term_vectors]
            if len(selected_terms) < 2:
                print("  Not enough terms with embeddings found for PCA, skipping.")
            else:
                matrix = np.stack([term_vectors[t] for t in selected_terms])
                pca = PCA(n_components=2, svd_solver="randomized", random_state=2011)
                coords_2d = pca.fit_transform(matrix)

                # Assign colors
                colors = []
                for t in selected_terms:
                    in_human = t in human_terms_all
                    in_llm = t in llm_terms_all
                    if in_human and in_llm:
                        colors.append("red")
                    elif in_llm:
                        colors.append("steelblue")
                    else:
                        colors.append("mediumseagreen")

                fig, ax = plt.subplots(figsize=(14, 10))
                scatter_blue = ax.scatter(
                    [c[0] for c, col in zip(coords_2d, colors) if col == "steelblue"],
                    [c[1] for c, col in zip(coords_2d, colors) if col == "steelblue"],
                    c="steelblue", s=18, alpha=0.7, label="LLM only", zorder=3
                )
                scatter_green = ax.scatter(
                    [c[0] for c, col in zip(coords_2d, colors) if col == "mediumseagreen"],
                    [c[1] for c, col in zip(coords_2d, colors) if col == "mediumseagreen"],
                    c="mediumseagreen", s=18, alpha=0.7, label="Human only", zorder=3
                )
                scatter_red = ax.scatter(
                    [c[0] for c, col in zip(coords_2d, colors) if col == "red"],
                    [c[1] for c, col in zip(coords_2d, colors) if col == "red"],
                    c="red", s=22, alpha=0.85, label="Both", zorder=4
                )

                ax.set_xlabel("PC1", fontsize=12)
                ax.set_ylabel("PC2", fontsize=12)
                ax.set_title("2D PCA of Selected Ontology Terms", fontsize=14, fontweight="bold")
                ax.legend(title="Selected by", title_fontsize=11, fontsize=10,
                          loc="upper right", framealpha=0.9)

                # Summary counts in corner
                n_both = colors.count("red")
                n_llm = colors.count("steelblue")
                n_human = colors.count("mediumseagreen")
                ax.text(0.01, 0.01,
                        f"LLM only: {n_llm}  |  Human only: {n_human}  |  Both: {n_both}",
                        transform=ax.transAxes, fontsize=9, color="#444444",
                        verticalalignment="bottom")

                plt.tight_layout()
                pca_out = os.path.join(args.output_dir, "pca_ontology_terms.png")
                plt.savefig(pca_out, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"PCA plot saved to: {pca_out}")

        except Exception as e:
            print(f"Error generating PCA ontology terms plot: {e}")
    else:
        print("Skipping PCA plot: --ontology_embedding_db not provided or file not found.")

    # --- Generate Model Comparison Table (previously in 06.summary.py) ---
    print("Generating Model Comparison Table...")
    try:
        model_metadata_lookup = {
            "gpt-oss-20b": {"total_params": "21B", "active_params": "3.6B", "dtype": "BF16", "gpu": "1x A100 / 1x H200"},
            "Qwen3.5-27B": {"total_params": "27B", "active_params": "27B", "dtype": "FP8", "gpu": "1x A100 / 1x H200"},
            "Qwen3.6-27B": {"total_params": "27B", "active_params": "27B", "dtype": "FP8", "gpu": "1x A100 / 1x H200"},
            "NVIDIA-Nemotron-3-Nano-30B-A3B": {"total_params": "30B", "active_params": "3.0B", "dtype": "FP8", "gpu": "1x A100 / 1x H200"},
            "Qwen3.5-35B-A3B": {"total_params": "35B", "active_params": "3.0B", "dtype": "FP8", "gpu": "1x A100 / 1x H200"},
            "Qwen3.6-35B-A3B": {"total_params": "35B", "active_params": "3.0B", "dtype": "FP8", "gpu": "1x A100 / 1x H200"},
            "Qwen3.5-122B-A10B": {"total_params": "122B", "active_params": "10.0B", "dtype": "FP8", "gpu": "4x A100 / 2x H200"},
            "NVIDIA-Nemotron-3-Super-120B-A12B": {"total_params": "120B", "active_params": "12.0B", "dtype": "FP8", "gpu": "4x A100 / 2x H200"},
            "gpt-oss-120b": {"total_params": "117B", "active_params": "5.1B", "dtype": "BF16", "gpu": "4x A100 / 2x H200"}
        }

        # Identify human experts (excluding average_human)
        human_expert_ids = [eid for eid in all_data if not eid.startswith("ToxValLlm_llm_") and eid != "average_human"]

        table_results = []
        for model_name, best_conf in sorted(best_config_per_model.items()):
            llm_id = best_conf["LLM_ID"]
            # Find reproducibility runs
            df_runs = []
            for run_r in range(1, 11):
                run_key = f"{llm_id}_{run_r}"
                if run_key in all_data:
                    df_runs.append(all_data[run_key])
                else:
                    run_file = os.path.join(args.llm_expanded_dir, "reproducibility", f"Expanded_{llm_id}_{run_r}.csv")
                    if os.path.exists(run_file):
                        df_run = pd.read_csv(run_file)
                        df_run["All_Terms_Set"] = df_run["All_Terms_Str"].apply(parse_term_set)
                        df_runs.append(df_run)

            if not df_runs:
                print(f"  WARNING: No reproducibility runs found for {llm_id}. Using original single run.")
                df_runs = [all_data[llm_id]]

            run_precisions_means = []
            run_recalls_means = []
            run_f1s_means = []
            run_jaccards_means = []

            for df_run in df_runs:
                expert_precisions = []
                expert_recalls = []
                expert_f1s = []
                expert_jaccards = []

                for exp_id in human_expert_ids:
                    df_expert = all_data[exp_id]
                    expert_map = {row["Index"]: row["All_Terms_Set"] for _, row in df_expert.iterrows()}

                    total_inter = 0
                    total_pred = 0
                    total_true = 0
                    jaccards = []

                    for idx, human_terms in expert_map.items():
                        row = df_run[df_run["Index"] == idx]
                        if len(row) == 0:
                            continue
                        not_exp_m = row.iloc[0].get("Not_My_Expertise", 0) == 1
                        not_exp_h = df_expert[df_expert["Index"] == idx].iloc[0].get("Not_My_Expertise", 0) == 1
                        if not_exp_m or not_exp_h:
                            continue
                        if not human_terms:
                            continue
                        llm_terms = row.iloc[0]["All_Terms_Set"]
                        
                        total_inter += len(llm_terms.intersection(human_terms))
                        total_pred += len(llm_terms)
                        total_true += len(human_terms)

                        u = llm_terms.union(human_terms)
                        inter = llm_terms.intersection(human_terms)
                        j_val = len(inter) / len(u) if len(u) > 0 else 1.0
                        jaccards.append(j_val)

                    if total_pred > 0 or total_true > 0:
                        ep = total_inter / total_pred if total_pred > 0 else 0.0
                        er = total_inter / total_true if total_true > 0 else 0.0
                        ef1 = (2 * ep * er) / (ep + er) if (ep + er) > 0 else 0.0
                        expert_precisions.append(ep)
                        expert_recalls.append(er)
                        expert_f1s.append(ef1)

                    if jaccards:
                        expert_jaccards.append(np.mean(jaccards))

                if expert_precisions:
                    run_precisions_means.append(np.mean(expert_precisions))
                if expert_recalls:
                    run_recalls_means.append(np.mean(expert_recalls))
                if expert_f1s:
                    run_f1s_means.append(np.mean(expert_f1s))
                if expert_jaccards:
                    run_jaccards_means.append(np.mean(expert_jaccards))

            mean_p = np.mean(run_precisions_means) if run_precisions_means else 0.0
            std_p = np.std(run_precisions_means) if run_precisions_means else 0.0
            mean_r = np.mean(run_recalls_means) if run_recalls_means else 0.0
            std_r = np.std(run_recalls_means) if run_recalls_means else 0.0
            mean_f1 = np.mean(run_f1s_means) if run_f1s_means else 0.0
            std_f1 = np.std(run_f1s_means) if run_f1s_means else 0.0
            mean_j = np.mean(run_jaccards_means) if run_jaccards_means else 0.0
            std_j = np.std(run_jaccards_means) if run_jaccards_means else 0.0

            # Get token count from the metadata/run
            # We already have Tokens in llm_perf_metrics, so let's match it by llm_id
            avg_completion_tokens = 0.0
            for item in llm_perf_metrics:
                if item["LLM_ID"] == llm_id:
                    avg_completion_tokens = item["Tokens"]
                    break

            meta_info = model_metadata_lookup.get(model_name, {"total_params": "N/A", "active_params": "N/A", "dtype": "N/A", "gpu": "N/A"})

            table_results.append({
                "Model": model_name,
                "Precision Mean": mean_p,
                "Precision Std": std_p,
                "Recall Mean": mean_r,
                "Recall Std": std_r,
                "F1 Mean": mean_f1,
                "F1 Std": std_f1,
                "Jaccard Mean": mean_j,
                "Jaccard Std": std_j,
                "Total Params": meta_info["total_params"],
                "Active Params": meta_info["active_params"],
                "Data Type": meta_info["dtype"],
                "Avg Tokens": int(avg_completion_tokens),
                "GPU Requirements": meta_info["gpu"]
            })

        if table_results:
            res_df = pd.DataFrame(table_results)
            fig, ax = plt.subplots(figsize=(24, 4.0))
            ax.axis('off')

            header_color   = '#1f4e79'
            row_even_color = '#f2f2f2'
            row_odd_color  = '#ffffff'

            table_data = []
            for idx, row in res_df.iterrows():
                table_data.append([
                    row["Model"],
                    f"{row['Precision Mean']:.2%} ± {row['Precision Std']:.2%}",
                    f"{row['Recall Mean']:.2%} ± {row['Recall Std']:.2%}",
                    f"{row['F1 Mean']:.2%} ± {row['F1 Std']:.2%}",
                    f"{row['Jaccard Mean']:.2%} ± {row['Jaccard Std']:.2%}",
                    row["Total Params"],
                    row["Active Params"],
                    row["Data Type"],
                    f"{row['Avg Tokens']:,}" if row['Avg Tokens'] > 0 else "N/A",
                    row["GPU Requirements"]
                ])

            col_labels = [
                "Model", "Precision (Mean ± Std)", "Recall (Mean ± Std)", "F1 Score (Mean ± Std)",
                "Jaccard Overlap (Mean ± Std)",
                "Total Params", "Active Params", "Data Type", "Avg Tokens", "GPU Requirements"
            ]

            col_widths = [0.20, 0.10, 0.10, 0.10, 0.15, 0.08, 0.08, 0.05, 0.06, 0.11]

            table = ax.table(
                cellText=table_data,
                colLabels=col_labels,
                colWidths=col_widths,
                cellLoc='center',
                loc='upper center'
            )

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.0, 1.8)

            for (row_idx, col_idx), cell in table.get_celld().items():
                if row_idx == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor(header_color)
                else:
                    if col_idx == 0:
                        cell.set_text_props(weight='bold')
                    if row_idx % 2 == 0:
                        cell.set_facecolor(row_even_color)
                    else:
                        cell.set_facecolor(row_odd_color)

            plt.title("LLM Evaluation Metrics & Hardware Requirements",
                      fontsize=14, fontweight='bold', pad=6, color='#1f4e79')

            output_path = os.path.join(args.output_dir, "model_comparison_table.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved summary table plot to: {output_path}")

            # --- Find best model on a single GPU and save llm_config.json ---
            single_gpu_results = [r for r in table_results if r["GPU Requirements"].startswith("1x")]
            if single_gpu_results:
                best_single_gpu = sorted(single_gpu_results, key=lambda x: x["F1 Mean"], reverse=True)[0]
                best_json_path = os.path.join(args.llm_expanded_dir, "llm_config.json")
                best_llm_data = {
                    "model": best_single_gpu["Model"],
                    "mode": 2,
                    "consensus_runs": 1,
                    "ontology_definitions": False,
                    "f1_score": round(best_single_gpu["F1 Mean"], 2)
                }
                with open(best_json_path, "w", encoding="utf-8") as jf:
                    json.dump(best_llm_data, jf, indent=4)
                print(f"Saved best single GPU model configuration to: {best_json_path}")
    except Exception as ex:
        print(f"Error generating model comparison table: {ex}")

    print(f"All combined plots saved successfully to: {args.output_dir}")


if __name__ == "__main__":
    main()
