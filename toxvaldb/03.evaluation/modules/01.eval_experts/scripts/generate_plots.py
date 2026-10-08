from __future__ import annotations
import argparse
import glob
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import rdflib
import seaborn as sns
from rdflib.namespace import RDFS
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


def arg_parser() -> argparse.Namespace:
    """
    Description:
    Parse command line arguments for the expert evaluation plot generator.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots from expanded expert annotation CSVs."
    )
    parser.add_argument(
        "--expanded_dir",
        required=True,
        help="Directory containing Expanded_*.csv files (output of run_analysis.py).",
    )
    parser.add_argument(
        "--orig_dir",
        required=True,
        help=(
            "Directory containing the original ToxValLlm_expert_*_ontology_<onto>.csv "
            "files used to compute flag counts."
        ),
    )
    parser.add_argument(
        "--owl_file",
        required=True,
        help="Path to the ontology TTL/OWL file (used for depth computation).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--ontology",
        required=True,
        help="Ontology short name (e.g. mp, hpath). Used to match original files.",
    )
    parser.add_argument(
        "--root_term",
        default=None,
        help="Label of the ontology root node (e.g. 'mammalian phenotype').",
    )
    parser.add_argument(
        "--expert_equivalence_file",
        default=None,
        help="Path to the TXT file containing expert name equivalences.",
    )
    return parser.parse_args()


EXPERT_MAP = {
    "expert1": r"$\mathbf{Expert\ 1}$",
    "expert2": r"$\mathbf{Expert\ 2}$",
    "expert3": r"$\mathbf{Expert\ 3}$",
    "expert4": r"$\mathbf{Expert\ 4}$",
    "expert5": r"$\mathbf{Expert\ 5}$",
    "expert6": r"$\mathbf{Expert\ 6}$",
    "expert_1": r"$\mathbf{Expert\ 1}$",
    "expert_2": r"$\mathbf{Expert\ 2}$",
    "expert_3": r"$\mathbf{Expert\ 3}$",
    "expert_4": r"$\mathbf{Expert\ 4}$",
    "expert_5": r"$\mathbf{Expert\ 5}$",
    "expert_6": r"$\mathbf{Expert\ 6}$",
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
    Clean and format the expert name for plot labels.

    Args:
        name (str): The raw expert identifier.

    Returns:
        str: The cleaned expert name.
    """
    if name in EXPERT_MAP:
        return EXPERT_MAP[name]
    import re
    m = re.match(r"^expert_?(\d+)$", name, re.IGNORECASE)
    if m:
        return f"$\mathbf{{Expert\ {m.group(1)}}}$"
    for k, v in EXPERT_MAP.items():
        if k.lower() in name.lower() or name.lower() in k.lower():
            return v
    parts = name.split("_")
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return name.replace("_", " ")


def build_depth_map(owl_file: str, root_term: str | None) -> dict:
    """
    Description:
    Parse the ontology OWL file and construct a dictionary mapping term names to hierarchy depth.

    Args:
        owl_file (str): Path to the owl/ttl ontology file.
        root_term (str | None): Label of the root node of the ontology.

    Returns:
        dict: A dictionary mapping term names to depth values.
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


def get_max_depth(term_set: set[str], depth_map: dict) -> int | float:
    """
    Description:
    Get the maximum hierarchy depth within a set of terms.

    Args:
        term_set (set[str]): A set of term names.
        depth_map (dict): A map from term names to depth integers.

    Returns:
        int | float: The maximum depth value found.
    """
    if not term_set:
        return 0
    depths = [depth_map.get(t, 0) for t in term_set]
    return np.max(depths)


def count_valid_terms(term_str: str | float) -> int:
    """
    Description:
    Count the number of valid terms (excluding none, -, unknown) in a pipe-delimited string.

    Args:
        term_str (str | float): A pipe-delimited string of terms.

    Returns:
        int: Number of valid terms.
    """
    if pd.isna(term_str):
        return 0
    terms = [t.strip().lower() for t in str(term_str).split("|")]
    valid_t = [t for t in terms if t not in ["none", "-", "unknown", ""]]
    return len(valid_t)




def main() -> None:
    """
    Description:
    Main execution function to build the depth map, calculate metrics, and generate 6 expert similarity plots.

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

    # Load expanded files (exclude the synthetic average_human file)
    exp_files = sorted([
        f for f in glob.glob(os.path.join(args.expanded_dir, "Expanded_*.csv"))
        if "average" not in os.path.basename(f).lower()
    ])
    if not exp_files:
        print(f"No Expanded_*.csv files found in: {args.expanded_dir}")
        return

    experts_data: dict = {}
    expert_metrics: list = []

    print("Processing expert data...")
    for f in exp_files:
        fname = os.path.basename(f)
        expert_id = fname.replace("Expanded_", "").replace(".csv", "")
        df_exp = pd.read_csv(f)
        df_exp["All_Terms_Set"] = df_exp["All_Terms_Str"].apply(
            lambda x: set(str(x).split("|")) if pd.notna(x) and str(x) != "" else set()
        )
        experts_data[expert_id] = df_exp

        # Load original expert files for counts
        orig_path = os.path.join(
            args.orig_dir,
            f"ToxValLlm_expert_{expert_id}_ontology_{args.ontology}.csv",
        )
        if not os.path.exists(orig_path):
            print(f"  WARNING: original file not found: {orig_path} — skipping flags.")
            df_orig = None
        else:
            df_orig = pd.read_csv(orig_path)

        # Add Not_My_Expertise column to df_exp (treating both Not_My_Expertise and Toxicological_Effect_Does_Not_Make_Sense as exclusion criteria)
        if df_orig is not None:
            not_exp = pd.to_numeric(df_orig.get("Not_My_Expertise", 0), errors="coerce").fillna(0)
            nonsense = pd.to_numeric(df_orig.get("Toxicological_Effect_Does_Not_Make_Sense", 0), errors="coerce").fillna(0)
            df_exp["Not_My_Expertise"] = ((not_exp == 1) | (nonsense == 1)).astype(int).values
        else:
            df_exp["Not_My_Expertise"] = 0

        # Depth metric
        df_exp["Max_Depth"] = df_exp["All_Terms_Set"].apply(
            lambda x: get_max_depth(x, depth_map)
        )

        selected_count_total = (
            df_orig["Selected_Candidates"].apply(count_valid_terms).sum()
            if df_orig is not None else 0
        )
        manual_count_total = 0
        nonsense_count = (
            pd.to_numeric(
                df_orig["Toxicological_Effect_Does_Not_Make_Sense"], errors="coerce"
            ).fillna(0).sum()
            if df_orig is not None else 0
        )
        not_expertise_count = (
            pd.to_numeric(df_orig["Not_My_Expertise"], errors="coerce")
            .fillna(0).sum()
            if df_orig is not None else 0
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

        expert_metrics.append(
            {
                "Expert_ID": expert_id,
                "Expert": clean_expert_name(expert_id),
                "Avg_Orig_Num_Terms": avg_orig_terms,
                "Mean_Max_Depth": mean_max_depth,
                "Selected_Terms": selected_count_total,
                "Manual_Terms": manual_count_total,
                "Nonsense_Flags": nonsense_count,
                "Not_Expertise_Flags": not_expertise_count,
            }
        )

    metrics_df = pd.DataFrame(expert_metrics)
    expert_ids = list(experts_data.keys())
    cleaned_names = [clean_expert_name(eid) for eid in expert_ids]

    # Pairwise overlap (Jaccard)
    print("Calculating pairwise overlap...")
    num_phrases = len(experts_data[expert_ids[0]])
    pairwise_overlap = np.zeros((len(expert_ids), len(expert_ids)))
    for i, id1 in enumerate(expert_ids):
        for j, id2 in enumerate(expert_ids):
            scores = []
            for k in range(num_phrases):
                # Check if either expert has Not_My_Expertise == 1 for this phrase
                not_exp1 = experts_data[id1].iloc[k].get("Not_My_Expertise", 0) == 1
                not_exp2 = experts_data[id2].iloc[k].get("Not_My_Expertise", 0) == 1
                if not_exp1 or not_exp2:
                    continue
                s1 = experts_data[id1].iloc[k]["All_Terms_Set"]
                s2 = experts_data[id2].iloc[k]["All_Terms_Set"]
                u = s1.union(s2)
                inter = s1.intersection(s2)
                score = len(inter) / len(u) if len(u) > 0 else 1.0
                scores.append(score)
            pairwise_overlap[i, j] = np.mean(scores) if scores else 1.0

    print("Generating plots...")

    # Plot 1: Pairwise Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pairwise_overlap,
        annot=True,
        xticklabels=cleaned_names,
        yticklabels=cleaned_names,
        cmap="Blues",
        fmt=".2f",
    )
    plt.title("Pairwise Overlap (Average Jaccard Similarity)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "pairwise_overlap_heatmap.png"), dpi=300)
    plt.close()

    # Plot 2: Avg Terms per Expert
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=metrics_df,
        x="Expert",
        y="Avg_Orig_Num_Terms",
        palette="viridis",
        hue="Expert",
        legend=False,
    )
    plt.title("Average Number of Terms per Phrase per Expert (including Hierarchy)")
    plt.ylabel("Average Number of Terms")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "avg_terms_per_expert.png"), dpi=300)
    plt.close()

    # Plot 3: Max Depth per Expert
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=metrics_df,
        x="Expert",
        y="Mean_Max_Depth",
        palette="magma",
        hue="Expert",
        legend=False,
    )
    plt.title("Average Maximum Depth of Terms per Phrase per Expert")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "max_depth_per_expert.png"), dpi=300)
    plt.close()

    # Plot 4: Selected candidates only (manual browse-ontology terms excluded)
    plt.figure(figsize=(10, 6))
    plt.bar(
        metrics_df["Expert"],
        metrics_df["Selected_Terms"],
        label="Selected Candidates",
        color="skyblue",
    )
    plt.xticks(rotation=90)
    plt.title("Usage of Selected Candidate Terms (Manual Terms Excluded)")
    plt.ylabel("Total Terms Used")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "manual_vs_selected_terms.png"), dpi=300)
    plt.close()

    # Plot 5: Expert Flags
    x = np.arange(len(metrics_df["Expert"]))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(
        x - width / 2,
        metrics_df["Nonsense_Flags"],
        width,
        label="Does Not Make Sense",
        color="orange",
    )
    plt.bar(
        x + width / 2,
        metrics_df["Not_Expertise_Flags"],
        width,
        label="Not My Expertise",
        color="purple",
    )
    plt.xticks(x, metrics_df["Expert"], rotation=90)
    plt.title("Phrases Flagged by Experts")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "expert_flags.png"), dpi=300)
    plt.close()

    # Plot 6: Dendrogram
    dist_matrix = 1 - pairwise_overlap
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    condensed_dist = squareform(dist_matrix)
    Z = linkage(condensed_dist, method="ward")

    plt.figure(figsize=(10, 8))
    dendrogram(Z, labels=cleaned_names, leaf_rotation=90, leaf_font_size=10)
    plt.title("Dendrogram of Expert Similarity (Ward Linkage)")
    plt.ylabel("Distance (1 - Jaccard)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "expert_dendrogram.png"), dpi=300)
    plt.close()

    print(f"All 6 plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
