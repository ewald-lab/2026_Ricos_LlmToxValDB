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
    Parse command line arguments for the LLM evaluation plot generator.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots from expanded LLM responses."
    )
    parser.add_argument(
        "--expanded_dir",
        required=True,
        help="Directory containing Expanded_ToxValLlm_llm_*.csv files.",
    )
    parser.add_argument(
        "--orig_dir",
        required=True,
        help="Directory containing the original ToxValLlm_llm_*.csv files.",
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
        help="Ontology short name (e.g. mp, hpath). Used to match files.",
    )
    parser.add_argument(
        "--root_term",
        default="mammalian phenotype",
        help="Label of the ontology root node (e.g. 'mammalian phenotype').",
    )
    parser.add_argument(
        "--expert_expanded_dir",
        default=None,
        help="Directory containing human expert Expanded_*.csv files.",
    )
    parser.add_argument(
        "--expert_orig_dir",
        default=None,
        help="Directory containing original ToxValLlm_expert_*.csv files.",
    )
    return parser.parse_args()


def clean_expert_name(name: str) -> str:
    """
    Description:
    Clean and format the human expert name for plot labels.

    Args:
        name (str): The raw expert identifier.

    Returns:
        str: The cleaned expert name.
    """
    if name == "average_human":
        return "Average Human"
    parts = name.split("_")
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return name.replace("_", " ")


def clean_llm_name(name: str) -> str:
    """
    Description:
    Clean and format the LLM configuration name for plot labels.

    Args:
        name (str): The raw LLM identifier.

    Returns:
        str: The formatted LLM configuration name with mode, consensus, and definitions.
    """
    # Example input: ToxValLlm_llm_Qwen3.6-35B-A3B-FP8_ontology_mp_mode_1_consensus_1_with_defs
    name = name.replace("ToxValLlm_llm_", "")
    parts = name.split("_ontology_")
    if len(parts) < 2:
        return name.replace("_", " ")

    model_part = parts[0]
    rest = parts[1]

    # Rest contains: mp_mode_1_consensus_1_with_defs or similar
    mode = "M1" if "mode_1" in rest else "M2"

    consensus = "C1"
    for c in ["10", "5", "3", "1"]:
        if f"consensus_{c}" in rest:
            consensus = f"C{c}"
            break

    defs = "with_defs" in rest
    defs_str = "defs" if defs else "no defs"

    return f"{model_part} ({mode}, {consensus}, {defs_str})"


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
    Main execution function to build the depth map, calculate metrics, and generate comparison plots for each LLM model.

    Args:
        None

    Returns:
        None
    """
    args = arg_parser()

    os.makedirs(args.output_dir, exist_ok=True)

    depth_map = build_depth_map(args.owl_file, args.root_term)

    # Load expert data if directories are provided
    expert_data = {}
    expert_metrics = []
    if args.expert_expanded_dir and args.expert_orig_dir:
        print("Loading expert data...")
        exp_expert_files = glob.glob(os.path.join(args.expert_expanded_dir, "Expanded_*.csv"))
        for f in exp_expert_files:
            fname = os.path.basename(f)
            expert_id = fname.replace("Expanded_", "").replace(".csv", "")
            if expert_id == "average_human":
                continue

            try:
                df_exp = pd.read_csv(f)
                df_exp["All_Terms_Set"] = df_exp["All_Terms_Str"].apply(
                    lambda x: set(str(x).split("|")) if pd.notna(x) and str(x) != "" else set()
                )
                expert_data[expert_id] = df_exp
            except Exception as re:
                print(f"  WARNING: Error reading expert expanded file {f}: {re}")
                continue

            # Load original expert files for counts
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
            else:
                df_exp["Not_My_Expertise"] = 0

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

            expert_metrics.append(
                {
                    "LLM_ID": expert_id,
                    "LLM": clean_expert_name(expert_id),
                    "Avg_Orig_Num_Terms": avg_orig_terms,
                    "Mean_Max_Depth": mean_max_depth,
                    "Selected_Terms": selected_count_total,
                    "Manual_Terms": manual_count_total,
                    "Nonsense_Flags": nonsense_count,
                    "Not_Expertise_Flags": not_expertise_count,
                    "Avg_Completion_Tokens": 0.0,
                }
            )

    # Load expanded files recursively
    exp_files = glob.glob(os.path.join(args.expanded_dir, "**", f"Expanded_ToxValLlm_llm_*_ontology_{args.ontology}_mode_*.csv"), recursive=True)
    if not exp_files:
        # Fallback to broader matching
        exp_files = glob.glob(os.path.join(args.expanded_dir, "**", "Expanded_ToxValLlm_llm_*.csv"), recursive=True)

    if not exp_files:
        print(f"No Expanded_ToxValLlm_llm_*.csv files found in: {args.expanded_dir}")
        return

    # Group files by model (based on the name of their containing subdirectory)
    model_groups = {}
    for f in exp_files:
        model_name = os.path.basename(os.path.dirname(f))
        # Exclude output subfolders that are not models
        if model_name in ["figures", "figures_llm_only", "figures_llm_onky"]:
            continue
        if model_name not in model_groups:
            model_groups[model_name] = []
        model_groups[model_name].append(f)

    for model_name, files in model_groups.items():
        print("\n========================================================")
        print(f"Generating plots for LLM: {model_name} ({len(files)} configurations)")
        print("========================================================")

        model_output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        llms_data: dict = {}
        llm_metrics: list = []

        # Include human expert data in this model's comparison
        if expert_data:
            llms_data.update(expert_data)
            llm_metrics.extend(expert_metrics)

        print(f"Processing data for {model_name}...")
        for f in files:
            fname = os.path.basename(f)
            llm_id = fname.replace("Expanded_", "").replace(".csv", "")
            try:
                df_exp = pd.read_csv(f)
                df_exp["All_Terms_Set"] = df_exp["All_Terms_Str"].apply(
                    lambda x: set(str(x).split("|")) if pd.notna(x) and str(x) != "" else set()
                )
                llms_data[llm_id] = df_exp
            except Exception as re:
                print(f"  WARNING: Error reading expanded file {f}: {re}")
                continue

            # Load original for counts (in the same subdirectory as the expanded file)
            orig_path = os.path.join(os.path.dirname(f), f"{llm_id}.csv")
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

            # Load token metrics from _tokens.txt file
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

            filtered_for_depth = df_exp[df_exp["Not_My_Expertise"] != 1]
            mean_max_depth = filtered_for_depth["Max_Depth"].mean() if not filtered_for_depth.empty else 0

            llm_metrics.append(
                {
                    "LLM_ID": llm_id,
                    "LLM": clean_llm_name(llm_id),
                    "Avg_Orig_Num_Terms": avg_orig_terms,
                    "Mean_Max_Depth": mean_max_depth,
                    "Selected_Terms": selected_count_total,
                    "Manual_Terms": manual_count_total,
                    "Nonsense_Flags": nonsense_count,
                    "Not_Expertise_Flags": not_expertise_count,
                    "Avg_Completion_Tokens": avg_completion_tokens,
                }
            )

        if not llm_metrics:
            print(f"No valid configuration files processed for {model_name}.")
            continue

        metrics_df = pd.DataFrame(llm_metrics)
        llm_ids = list(llms_data.keys())
        cleaned_names = [clean_expert_name(lid) if lid in expert_data else clean_llm_name(lid) for lid in llm_ids]

        # Pairwise overlap (Jaccard)
        print("Calculating pairwise overlap...")
        num_phrases = len(llms_data[llm_ids[0]])
        pairwise_overlap = np.zeros((len(llm_ids), len(llm_ids)))
        for i, id1 in enumerate(llm_ids):
            for j, id2 in enumerate(llm_ids):
                scores = []
                for k in range(num_phrases):
                    # Check if either entity has Not_My_Expertise == 1 for this phrase
                    not_exp1 = llms_data[id1].iloc[k].get("Not_My_Expertise", 0) == 1
                    not_exp2 = llms_data[id2].iloc[k].get("Not_My_Expertise", 0) == 1
                    if not_exp1 or not_exp2:
                        continue
                    s1 = llms_data[id1].iloc[k]["All_Terms_Set"]
                    s2 = llms_data[id2].iloc[k]["All_Terms_Set"]
                    u = s1.union(s2)
                    inter = s1.intersection(s2)
                    score = len(inter) / len(u) if len(u) > 0 else 1.0
                    scores.append(score)
                pairwise_overlap[i, j] = np.mean(scores) if scores else 1.0

        print("Generating plots...")

        # Plot 1: Pairwise Heatmap
        num_llms = len(llm_ids)
        heatmap_width = max(10, num_llms * 0.5 + 4)
        heatmap_height = max(8, num_llms * 0.5 + 2)
        plt.figure(figsize=(heatmap_width, heatmap_height))

        annot_font_size = max(5, 10 - num_llms // 5)
        tick_font_size = max(6, 11 - num_llms // 6)

        sns.heatmap(
            pairwise_overlap,
            annot=True,
            annot_kws={"size": annot_font_size},
            xticklabels=cleaned_names,
            yticklabels=cleaned_names,
            cmap="Blues",
            fmt=".2f",
        )
        plt.title(f"Pairwise Overlap (Average Jaccard Similarity) - {model_name}", fontsize=max(10, 14 - num_llms // 8))
        plt.xticks(rotation=90, fontsize=tick_font_size)
        plt.yticks(rotation=0, fontsize=tick_font_size)
        plt.tight_layout()
        plt.savefig(os.path.join(model_output_dir, "pairwise_overlap_heatmap.png"), dpi=300)
        plt.close()

        # Plot 2: Avg Terms per LLM
        bar_width = max(12, num_llms * 0.4 + 2)
        plt.figure(figsize=(bar_width, 6))
        sns.barplot(
            data=metrics_df,
            x="LLM",
            y="Avg_Orig_Num_Terms",
            palette="viridis",
            hue="LLM",
            legend=False,
        )
        plt.title(f"Average Number of Terms per Phrase (including Hierarchy) - {model_name}")
        plt.ylabel("Average Number of Terms")
        plt.xticks(rotation=90, fontsize=tick_font_size)
        plt.tight_layout()
        plt.savefig(os.path.join(model_output_dir, "avg_terms_per_llm.png"), dpi=300)
        plt.close()

        # Plot 3: Max Depth per LLM
        plt.figure(figsize=(bar_width, 6))
        sns.barplot(
            data=metrics_df,
            x="LLM",
            y="Mean_Max_Depth",
            palette="magma",
            hue="LLM",
            legend=False,
        )
        plt.title(f"Average Maximum Depth of Terms per Phrase - {model_name}")
        plt.xticks(rotation=90, fontsize=tick_font_size)
        plt.tight_layout()
        plt.savefig(os.path.join(model_output_dir, "max_depth_per_llm.png"), dpi=300)
        plt.close()



        # Plot 6: Dendrogram
        if len(llm_ids) > 1:
            try:
                dist_matrix = 1 - pairwise_overlap
                np.fill_diagonal(dist_matrix, 0)
                dist_matrix = (dist_matrix + dist_matrix.T) / 2
                condensed_dist = squareform(dist_matrix)
                Z = linkage(condensed_dist, method="ward")

                dendrogram_width = max(10, num_llms * 0.4 + 2)
                dendrogram_height = max(8, num_llms * 0.4 + 2)
                plt.figure(figsize=(dendrogram_width, dendrogram_height))
                dendrogram(Z, labels=cleaned_names, leaf_rotation=90, leaf_font_size=tick_font_size)
                plt.title(f"Dendrogram of Similarity - {model_name} (Ward Linkage)")
                plt.ylabel("Distance (1 - Jaccard)")
                plt.tight_layout()
                plt.savefig(os.path.join(model_output_dir, "llm_dendrogram.png"), dpi=300)
                plt.close()
            except Exception as de:
                print(f"  WARNING: Could not generate dendrogram for {model_name}: {de}")
                plt.close()
        else:
            print("Skipping dendrogram: less than 2 configurations found.")

        # Plot 7: Average Completion Tokens per LLM Invocation
        llm_only_metrics = metrics_df[~metrics_df["LLM_ID"].isin(expert_data.keys())]
        bar_width_llm_only = max(12, len(llm_only_metrics) * 0.4 + 2)
        plt.figure(figsize=(bar_width_llm_only, 6))
        sns.barplot(
            data=llm_only_metrics,
            x="LLM",
            y="Avg_Completion_Tokens",
            palette="viridis",
            hue="LLM",
            legend=False,
        )
        plt.xticks(rotation=90, fontsize=tick_font_size)
        plt.title(f"Average Completion Tokens per Candidate Evaluation - {model_name}")
        plt.ylabel("Average Completion Tokens")
        plt.tight_layout()
        plt.savefig(os.path.join(model_output_dir, "tokens_used_per_llm.png"), dpi=300)
        plt.close()

        print(f"All plots for {model_name} saved successfully to: {model_output_dir}")


if __name__ == "__main__":
    main()
