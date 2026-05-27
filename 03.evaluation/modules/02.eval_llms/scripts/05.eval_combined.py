import argparse
import glob
import os
import sqlite3
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import rdflib
import seaborn as sns
from rdflib.namespace import RDFS
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


def arg_parser():
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
        "--ontology_embedding_db",
        default=None,
        help="Path to the ontology embedding DB (e.g. kalm/mp.db) for PCA visualization.",
    )
    return parser.parse_args()


def clean_expert_name(name: str) -> str:
    if name == "average_human":
        return "Average Human"
    parts = name.split("_")
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return name.replace("_", " ")


def clean_llm_name(name: str) -> str:
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


def parse_llm_metadata(llm_id: str):
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


def calc_metrics(pred_set, ref_set):
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


def build_depth_map(owl_file: str, root_term: str | None):
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


def get_max_depth(term_set, depth_map):
    if not term_set:
        return 0
    depths = [depth_map.get(t, 0) for t in term_set]
    return np.max(depths)


def count_valid_terms(term_str) -> int:
    if pd.isna(term_str):
        return 0
    terms = [t.strip().lower() for t in str(term_str).split("|")]
    valid_t = [t for t in terms if t not in ["none", "-", "unknown", ""]]
    return len(valid_t)





def main():
    args = arg_parser()

    os.makedirs(args.output_dir, exist_ok=True)

    depth_map = build_depth_map(args.owl_file, args.root_term)

    # 1. Load Human Expert Data
    exp_expert_files = glob.glob(os.path.join(args.expert_expanded_dir, "Expanded_*.csv"))
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
        df_exp["All_Terms_Set"] = df_exp["All_Terms_Str"].apply(
            lambda x: set(str(x).split("|")) if pd.notna(x) and str(x) != "" else set()
        )
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

        # Add Not_My_Expertise column to df_exp for downstream use (depth & similarity)
        if df_orig is not None and "Not_My_Expertise" in df_orig.columns:
            df_exp["Not_My_Expertise"] = pd.to_numeric(df_orig["Not_My_Expertise"], errors="coerce").fillna(0).values
        else:
            df_exp["Not_My_Expertise"] = 0

        df_exp["Max_Depth"] = df_exp["All_Terms_Set"].apply(
            lambda x: get_max_depth(x, depth_map)
        )

        selected_count_total = (
            df_orig["Selected_Candidates"].apply(count_valid_terms).sum()
            if df_orig is not None and "Selected_Candidates" in df_orig.columns else 0
        )
        manual_count_total = (
            df_orig["Manual_Terms"].apply(count_valid_terms).sum()
            if df_orig is not None and "Manual_Terms" in df_orig.columns else 0
        )
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
        df_avg_human["All_Terms_Set"] = df_avg_human["All_Terms_Str"].apply(
            lambda x: set(str(x).split("|")) if pd.notna(x) and str(x) != "" else set()
        )
        human_ref_map = {row["Index"]: row["All_Terms_Set"] for _, row in df_avg_human.iterrows()}
    else:
        print(f"WARNING: Expanded_average_human.csv not found at {avg_human_file}. Cannot compute LLM vs. Human metrics.")

    llm_perf_metrics = []

    print("Processing LLM response data...")
    for f in exp_llm_files:
        fname = os.path.basename(f)
        llm_id = fname.replace("Expanded_", "").replace(".csv", "")
        df_exp = pd.read_csv(f)
        df_exp["All_Terms_Set"] = df_exp["All_Terms_Str"].apply(
            lambda x: set(str(x).split("|")) if pd.notna(x) and str(x) != "" else set()
        )
        all_data[llm_id] = df_exp

        # Load original LLM files for counts (in the same subdirectory as the expanded file)
        orig_path = os.path.join(os.path.dirname(f), f"{llm_id}.csv")
        if not os.path.exists(orig_path):
            print(f"  WARNING: LLM original file not found: {orig_path} — skipping flags.")
            df_orig = None
        else:
            df_orig = pd.read_csv(orig_path)

        # Add Not_My_Expertise column to df_exp for downstream use (depth & similarity)
        if df_orig is not None and "Not_My_Expertise" in df_orig.columns:
            df_exp["Not_My_Expertise"] = pd.to_numeric(df_orig["Not_My_Expertise"], errors="coerce").fillna(0).values
        else:
            df_exp["Not_My_Expertise"] = 0

        df_exp["Max_Depth"] = df_exp["All_Terms_Set"].apply(
            lambda x: get_max_depth(x, depth_map)
        )

        selected_count_total = (
            df_orig["Selected_Candidates"].apply(count_valid_terms).sum()
            if df_orig is not None and "Selected_Candidates" in df_orig.columns else 0
        )
        manual_count_total = (
            df_orig["Manual_Terms"].apply(count_valid_terms).sum()
            if df_orig is not None and "Manual_Terms" in df_orig.columns else 0
        )
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
            precisions = []
            recalls = []
            f1s = []
            for idx, human_terms in human_ref_map.items():
                llm_row = df_exp[df_exp["Index"] == idx]
                if len(llm_row) == 0:
                    continue
                # Skip evaluating this phrase if the human reference has no terms (no consensus or not_my_expertise)
                if not human_terms:
                    continue
                llm_terms = llm_row.iloc[0]["All_Terms_Set"]
                p_val, r_val, f1_val = calc_metrics(llm_terms, human_terms)
                precisions.append(p_val)
                recalls.append(r_val)
                f1s.append(f1_val)

            mean_p = np.mean(precisions) if precisions else 0.0
            mean_r = np.mean(recalls) if recalls else 0.0
            mean_f1 = np.mean(f1s) if f1s else 0.0

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
                "Model": metadata["model"],
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
                    best_row = model_df.sort_values(by="F1", ascending=False).iloc[0]
                best_config_per_model[m] = {
                    "Mode": best_row["Mode"],
                    "Definitions": best_row["Definitions"],
                    "Consensus": best_row["Consensus"],
                    "F1": best_row["F1"],
                    "LLM_ID": best_row["LLM_ID"]
                }
                best_llm_ids.add(best_row["LLM_ID"])
                print(f"Selected config for model {m}: Mode={best_row['Mode']}, Definitions={best_row['Definitions']}, Consensus={best_row['Consensus']}, F1={best_row['F1']:.4f} (ID: {best_row['LLM_ID']})")

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

    pairwise_cleaned_names = []
    for eid in pairwise_entity_ids:
        if eid.startswith("ToxValLlm_llm_"):
            meta = parse_llm_metadata(eid)
            pairwise_cleaned_names.append(meta["model"] if meta else clean_llm_name(eid))
        else:
            pairwise_cleaned_names.append(clean_expert_name(eid))


    print("Calculating pairwise overlap...")
    num_phrases = len(all_data[pairwise_entity_ids[0]])
    pairwise_overlap = np.zeros((len(pairwise_entity_ids), len(pairwise_entity_ids)))
    for i, id1 in enumerate(pairwise_entity_ids):
        for j, id2 in enumerate(pairwise_entity_ids):
            scores = []
            for k in range(num_phrases):
                # Check if either entity has Not_My_Expertise == 1 for this phrase
                not_exp1 = all_data[id1].iloc[k].get("Not_My_Expertise", 0) == 1
                not_exp2 = all_data[id2].iloc[k].get("Not_My_Expertise", 0) == 1
                if not_exp1 or not_exp2:
                    continue
                s1 = all_data[id1].iloc[k]["All_Terms_Set"]
                s2 = all_data[id2].iloc[k]["All_Terms_Set"]
                u = s1.union(s2)
                inter = s1.intersection(s2)
                score = len(inter) / len(u) if len(u) > 0 else 1.0
                scores.append(score)
            pairwise_overlap[i, j] = np.mean(scores) if scores else 1.0

    print("Generating combined plots...")

    # Plot 1: Pairwise Heatmap / Clustermap
    num_entities_pw = len(pairwise_entity_ids)
    heatmap_width = max(12, num_entities_pw * 0.5 + 4)
    heatmap_height = max(10, num_entities_pw * 0.5 + 2)

    annot_font_size = max(5, 10 - num_entities_pw // 5)
    tick_font_size = max(6, 11 - num_entities_pw // 6)

    human_clean_names = {id_to_clean[eid] for eid in pairwise_entity_ids if not eid.startswith("ToxValLlm_llm_")}

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
                metrics_df.at[idx, "Entity"] = meta["model"]

    # Plot 2: Avg Terms per Entity
    num_entities = len(pairwise_entity_ids)
    bar_width = max(14, num_entities * 0.4 + 2)
    plt.figure(figsize=(bar_width, 7))
    sns.barplot(
        data=metrics_df,
        x="Entity",
        y="Avg_Orig_Num_Terms",
        hue="Type",
        palette={"Human Expert": "skyblue", "LLM": "lightgreen"},
    )
    plt.title("Combined: Average Number of Terms per Phrase (including Hierarchy)")
    plt.ylabel("Average Number of Terms")
    plt.xticks(rotation=90, fontsize=tick_font_size)

    # Color human expert names in red
    ax = plt.gca()
    for label in ax.get_xticklabels():
        if label.get_text() in human_clean_names:
            label.set_color("red")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "combined_avg_terms.png"), dpi=300)
    plt.close()

    # Plot 3: Max Depth per Entity
    plt.figure(figsize=(bar_width, 7))
    sns.barplot(
        data=metrics_df,
        x="Entity",
        y="Mean_Max_Depth",
        hue="Type",
        palette={"Human Expert": "skyblue", "LLM": "lightgreen"},
    )
    plt.title("Combined: Average Maximum Depth of Terms per Phrase")
    plt.xticks(rotation=90, fontsize=tick_font_size)

    # Color human expert names in red
    ax = plt.gca()
    for label in ax.get_xticklabels():
        if label.get_text() in human_clean_names:
            label.set_color("red")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "combined_max_depth.png"), dpi=300)
    plt.close()



    # Plot 6: Dendrogram
    if len(pairwise_entity_ids) > 1:
        dist_matrix = 1 - pairwise_overlap
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        condensed_dist = squareform(dist_matrix)
        Z = linkage(condensed_dist, method="ward")

        dendrogram_width = max(12, len(pairwise_entity_ids) * 0.4 + 2)
        dendrogram_height = max(10, len(pairwise_entity_ids) * 0.4 + 2)
        plt.figure(figsize=(dendrogram_width, dendrogram_height))
        dendrogram(Z, labels=pairwise_cleaned_names, leaf_rotation=90, leaf_font_size=tick_font_size)
        plt.title("Combined Dendrogram of Human & LLM Similarity (Ward Linkage)")
        plt.ylabel("Distance (1 - Jaccard)")

        # Color human expert names in red
        ax = plt.gca()
        for label in ax.get_xticklabels():
            if label.get_text() in human_clean_names:
                label.set_color("red")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "combined_similarity_dendrogram.png"), dpi=300)
        plt.close()

    # 7. Generate Precision & Recall Plots vs. Human Average
    if human_ref_map is not None and not llm_perf_df.empty:
        print("Generating LLM vs. Human Precision/Recall analysis plots...")

        # Plot 12: Combined Hyperparameter Justification & Optimal Performance
        try:
            import matplotlib.patches as mpatches
            print("Generating combined hyperparameter justification plot (llm_decisions_justification.png)...")

            # 16-subplot 4x4 Grid for all parameter combinations
            # Row index maps to combinations of Mode and Definitions
            rows_configs = [
                ("Mode 1", "No Defs"),
                ("Mode 1", "With Defs"),
                ("Mode 2", "No Defs"),
                ("Mode 2", "With Defs"),
            ]
            # Column index maps to Consensus runs
            cols_configs = [1, 3, 5, 10]

            fig, axes = plt.subplots(4, 4, figsize=(24, 24), sharex=True, sharey=True, constrained_layout=True)

            # Palette for metrics to ensure consistency
            metric_palette = {"Precision": "#3182bd", "Recall": "#e6550d", "F1": "#31a354"}

            # Get sorted unique models to enforce consistent X-axis ordering across all subplots
            unique_models = sorted(llm_perf_df["Model"].unique())

            for r_idx, (mode_val, defs_val) in enumerate(rows_configs):
                for c_idx, cons_val in enumerate(cols_configs):
                    ax = axes[r_idx, c_idx]

                    # Filter dataframe for this specific combination
                    sub_df = llm_perf_df[
                        (llm_perf_df["Mode"] == mode_val) &
                        (llm_perf_df["Definitions"] == defs_val) &
                        (llm_perf_df["Consensus"] == cons_val)
                    ]

                    ax.set_axisbelow(True)
                    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)

                    if not sub_df.empty:
                        # Melt Precision, Recall, and F1 into a Metric column for sns.barplot
                        sub_melted = sub_df.melt(
                            id_vars=["Model"],
                            value_vars=["Precision", "Recall", "F1"],
                            var_name="Metric",
                            value_name="Score"
                        )

                        try:
                            sns.barplot(
                                data=sub_melted,
                                x="Model",
                                y="Score",
                                hue="Metric",
                                hue_order=["Precision", "Recall", "F1"],
                                order=unique_models,
                                palette=metric_palette,
                                ax=ax,
                                edgecolor="black",
                                linewidth=0.8,
                                width=0.85,
                                zorder=3
                            )
                        except TypeError:
                            sns.barplot(
                                data=sub_melted,
                                x="Model",
                                y="Score",
                                hue="Metric",
                                hue_order=["Precision", "Recall", "F1"],
                                order=unique_models,
                                palette=metric_palette,
                                ax=ax,
                                edgecolor="black",
                                linewidth=0.8,
                                zorder=3
                            )

                        # Remove individual legend
                        if ax.get_legend():
                            ax.get_legend().remove()

                        # Add small text labels on top of the bars, rotated perpendicularly
                        for metric_idx, container in enumerate(ax.containers):
                            metric_name = ["Precision", "Recall", "F1"][metric_idx]
                            for i, p in enumerate(container):
                                val = p.get_height()
                                if pd.notna(val) and val > 0:
                                    if 0 <= i < len(unique_models):
                                        model_name = unique_models[i]
                                        best_conf = best_config_per_model.get(model_name)
                                        if (metric_name == "F1" and
                                            best_conf and
                                            best_conf["Mode"] == mode_val and
                                            best_conf["Definitions"] == defs_val and
                                            best_conf["Consensus"] == cons_val):
                                            text_color = "blue"
                                        else:
                                            text_color = "black"
                                    else:
                                        text_color = "black"

                                    ax.annotate(
                                        f"{val:.0%}",
                                        (p.get_x() + p.get_width() / 2., val),
                                        ha='center',
                                        va='bottom',
                                        xytext=(0, 3),
                                        textcoords='offset points',
                                        fontsize=7.5,
                                        fontweight="bold",
                                        color=text_color,
                                        rotation=90,
                                        zorder=5
                                    )

                    # Title for subplot indicating parameter settings
                    ax.set_title(f"{mode_val}, {defs_val}\nConsensus = {cons_val}", fontsize=11, fontweight="bold")

                    # X and Y limits
                    ax.set_ylim(0, 1.05)

                    # Label and format only edge subplots to reduce redundancy
                    if r_idx == 3:
                        ax.set_xlabel("Model", fontsize=11)
                        ax.tick_params(axis='x', labelrotation=90)
                    else:
                        ax.set_xlabel("")

                    if c_idx == 0:
                        ax.set_ylabel("Score (vs. Human Average)", fontsize=11)
                    else:
                        ax.set_ylabel("")

            # Construct a unified legend for the metrics
            legend_handles = [
                mpatches.Patch(facecolor=metric_palette["Precision"], edgecolor="black", label="Precision", linewidth=0.8),
                mpatches.Patch(facecolor=metric_palette["Recall"], edgecolor="black", label="Recall", linewidth=0.8),
                mpatches.Patch(facecolor=metric_palette["F1"], edgecolor="black", label="F1 Score", linewidth=0.8)
            ]

            fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                       title="Performance Metric", title_fontsize=12, fontsize=11)
            fig.suptitle("LLM System Optimization Justification & Model Performance (Grid of 16 Parameter Combinations)", fontsize=18, fontweight="bold")

            plt.savefig(os.path.join(args.output_dir, "llm_decisions_justification.png"), dpi=300, bbox_inches="tight")
            plt.close()

            # --- 7b. Detailed Decision/Parameter Impact Analysis and Statistical Significance ---
            decisions_dir = os.path.join(args.output_dir, "decisions")
            os.makedirs(decisions_dir, exist_ok=True)
            import scipy.stats as stats

            def generate_decision_plots(df, target_dir, title_suffix=""):
                os.makedirs(target_dir, exist_ok=True)
                
                # 1. Definitions Impact Analysis
                try:
                    pivot_defs = df.pivot(
                        index=["Model", "Mode", "Consensus"], 
                        columns="Definitions", 
                        values="F1"
                    ).dropna()
                    
                    if len(pivot_defs) >= 2:
                        stat_res = stats.ttest_rel(pivot_defs["With Defs"], pivot_defs["No Defs"])
                        p_val = stat_res.pvalue
                        stat_name = "Paired t-test"
                    else:
                        p_val = np.nan
                        stat_name = "N/A"

                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.boxplot(
                        data=df, 
                        x="Definitions", 
                        y="F1", 
                        ax=ax, 
                        palette={"No Defs": "#ff7f0e", "With Defs": "#1f77b4"},
                        width=0.4,
                        boxprops=dict(alpha=0.6),
                        showfliers=False
                    )
                    sns.stripplot(
                        data=df,
                        x="Definitions",
                        y="F1",
                        ax=ax,
                        color="black",
                        alpha=0.5,
                        size=6,
                        jitter=0.1
                    )
                    
                    title = f"Impact of Definitions on F1 Score{title_suffix}"
                    if pd.notna(p_val):
                        title += f"\n({stat_name} p-value: {p_val:.4e})"
                    ax.set_title(title, fontsize=14, fontweight="bold")
                    ax.set_ylabel("F1 Score (vs. Human Average)", fontsize=12)
                    ax.set_xlabel("Definitions Configuration", fontsize=12)
                    ax.set_ylim(-0.05, 1.05)
                    ax.grid(True, linestyle="--", alpha=0.3)
                    
                    plt.savefig(os.path.join(target_dir, "definitions_impact.png"), dpi=300, bbox_inches="tight")
                    plt.close()
                except Exception as e:
                    print(f"Error generating definitions_impact.png in {target_dir}: {e}")

                # 2. Mode Impact Analysis
                try:
                    pivot_mode = df.pivot(
                        index=["Model", "Definitions", "Consensus"], 
                        columns="Mode", 
                        values="F1"
                    ).dropna()
                    
                    if len(pivot_mode) >= 2:
                        stat_res = stats.ttest_rel(pivot_mode["Mode 1"], pivot_mode["Mode 2"])
                        p_val = stat_res.pvalue
                        stat_name = "Paired t-test"
                    else:
                        p_val = np.nan
                        stat_name = "N/A"

                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.boxplot(
                        data=df, 
                        x="Mode", 
                        y="F1", 
                        ax=ax, 
                        palette={"Mode 1": "#ff7f0e", "Mode 2": "#1f77b4"},
                        width=0.4,
                        boxprops=dict(alpha=0.6),
                        showfliers=False
                    )
                    sns.stripplot(
                        data=df,
                        x="Mode",
                        y="F1",
                        ax=ax,
                        color="black",
                        alpha=0.5,
                        size=6,
                        jitter=0.1
                    )
                    
                    title = f"Impact of Mode (M1 vs M2) on F1 Score{title_suffix}"
                    if pd.notna(p_val):
                        title += f"\n({stat_name} p-value: {p_val:.4e})"
                    ax.set_title(title, fontsize=14, fontweight="bold")
                    ax.set_ylabel("F1 Score (vs. Human Average)", fontsize=12)
                    ax.set_xlabel("Mode Configuration", fontsize=12)
                    ax.set_ylim(-0.05, 1.05)
                    ax.grid(True, linestyle="--", alpha=0.3)
                    
                    plt.savefig(os.path.join(target_dir, "mode_impact.png"), dpi=300, bbox_inches="tight")
                    plt.close()
                except Exception as e:
                    print(f"Error generating mode_impact.png in {target_dir}: {e}")

                # 3. Consensus Impact Analysis
                try:
                    pivot_cons = df.pivot(
                        index=["Model", "Mode", "Definitions"], 
                        columns="Consensus", 
                        values="F1"
                    ).dropna()
                    
                    if len(pivot_cons) >= 2:
                        stat_res = stats.friedmanchisquare(
                            pivot_cons[1], 
                            pivot_cons[3], 
                            pivot_cons[5], 
                            pivot_cons[10]
                        )
                        p_val = stat_res.pvalue
                        stat_name = "Friedman test"
                    else:
                        p_val = np.nan
                        stat_name = "N/A"

                    fig, ax = plt.subplots(figsize=(9, 6))
                    sns.boxplot(
                        data=df, 
                        x="Consensus", 
                        y="F1", 
                        ax=ax, 
                        width=0.5,
                        boxprops=dict(alpha=0.6),
                        showfliers=False
                    )
                    sns.stripplot(
                        data=df,
                        x="Consensus",
                        y="F1",
                        ax=ax,
                        color="black",
                        alpha=0.5,
                        size=6,
                        jitter=0.1
                    )
                    
                    title = f"Impact of Consensus Runs on F1 Score{title_suffix}"
                    if pd.notna(p_val):
                        title += f"\n({stat_name} p-value: {p_val:.4e})"
                    ax.set_title(title, fontsize=14, fontweight="bold")
                    ax.set_ylabel("F1 Score (vs. Human Average)", fontsize=12)
                    ax.set_xlabel("Number of Consensus Runs", fontsize=12)
                    ax.set_ylim(-0.05, 1.05)
                    ax.grid(True, linestyle="--", alpha=0.3)
                    
                    plt.savefig(os.path.join(target_dir, "consensus_impact.png"), dpi=300, bbox_inches="tight")
                    plt.close()
                except Exception as e:
                    print(f"Error generating consensus_impact.png in {target_dir}: {e}")

            # Generate global plots
            print("Generating global decision impact plots...")
            generate_decision_plots(llm_perf_df, decisions_dir, title_suffix=" (All Models)")

            # Generate per-model plots in subfolders
            print("Generating per-model decision impact plots...")
            for model_name in sorted(llm_perf_df["Model"].unique()):
                model_df = llm_perf_df[llm_perf_df["Model"] == model_name]
                model_folder_name = model_name.replace("/", "_").replace(" ", "_").replace(":", "_")
                model_target_dir = os.path.join(decisions_dir, model_folder_name)
                generate_decision_plots(model_df, model_target_dir, title_suffix=f" ({model_name})")

        except Exception as e:
            print(f"Error generating llm_decisions_justification.png: {e}")

    # Plot 15: Inter-Annotator Agreement (Boxplot)
    try:
        print("Generating Inter-Annotator Agreement boxplot...")
        human_indices = [i for i, eid in enumerate(pairwise_entity_ids) if not eid.startswith("ToxValLlm_llm_")]
        llm_indices = [i for i, eid in enumerate(pairwise_entity_ids) if eid.startswith("ToxValLlm_llm_")]

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
        term_variability_data = []
        for k in range(num_phrases):
            phrase_name = all_data[pairwise_entity_ids[0]].iloc[k]["Phrase"]
            phrase_idx = all_data[pairwise_entity_ids[0]].iloc[k]["Index"]
            
            # Human sets (exclude Not_My_Expertise == 1)
            human_sets = []
            for h_idx in human_indices:
                h_id = pairwise_entity_ids[h_idx]
                not_exp = all_data[h_id].iloc[k].get("Not_My_Expertise", 0) == 1
                if not not_exp:
                    human_sets.append(all_data[h_id].iloc[k]["All_Terms_Set"])
            
            # LLM sets
            llm_sets = []
            for l_idx in llm_indices:
                l_id = pairwise_entity_ids[l_idx]
                not_exp = all_data[l_id].iloc[k].get("Not_My_Expertise", 0) == 1
                if not not_exp:
                    llm_sets.append(all_data[l_id].iloc[k]["All_Terms_Set"])
            
            # Human-Human
            hh_dists = []
            for i in range(len(human_sets)):
                for j in range(i + 1, len(human_sets)):
                    u = human_sets[i].union(human_sets[j])
                    jacc = len(human_sets[i].intersection(human_sets[j])) / len(u) if len(u) > 0 else 1.0
                    hh_dists.append(1.0 - jacc)
            avg_hh = np.mean(hh_dists) if hh_dists else np.nan
            
            # LLM-LLM
            llm_dists = []
            for i in range(len(llm_sets)):
                for j in range(i + 1, len(llm_sets)):
                    u = llm_sets[i].union(llm_sets[j])
                    jacc = len(llm_sets[i].intersection(llm_sets[j])) / len(u) if len(u) > 0 else 1.0
                    llm_dists.append(1.0 - jacc)
            avg_llm = np.mean(llm_dists) if llm_dists else np.nan
            
            # Human-LLM
            hl_dists = []
            for h_set in human_sets:
                for l_set in llm_sets:
                    u = h_set.union(l_set)
                    jacc = len(h_set.intersection(l_set)) / len(u) if len(u) > 0 else 1.0
                    hl_dists.append(1.0 - jacc)
            avg_hl = np.mean(hl_dists) if hl_dists else np.nan
            
            term_variability_data.append({
                "Index": phrase_idx,
                "Phrase": phrase_name,
                "Human-Human Variability": avg_hh,
                "LLM-LLM Variability": avg_llm,
                "Human-LLM Variability": avg_hl
            })
            
        var_df = pd.DataFrame(term_variability_data)
        
        # Generate Plot (3 stacked subplots sharing the x-axis)
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, sharey=True)
        
        # Clean phrases for display
        phrases_display = var_df["Phrase"].tolist()
        x_indices = np.arange(len(phrases_display))
        
        # Plot 1: Human-Human
        axes[0].bar(x_indices, var_df["Human-Human Variability"], color="skyblue", edgecolor="black", alpha=0.8)
        axes[0].grid(True, linestyle="--", alpha=0.3)
        axes[0].set_ylabel("Variability")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_yticks([0.0, 0.5, 1.0])
        
        # Plot 2: LLM-LLM
        axes[1].bar(x_indices, var_df["LLM-LLM Variability"], color="lightgreen", edgecolor="black", alpha=0.8)
        axes[1].grid(True, linestyle="--", alpha=0.3)
        axes[1].set_ylabel("Variability")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_yticks([0.0, 0.5, 1.0])
        
        # Plot 3: Human-LLM
        axes[2].bar(x_indices, var_df["Human-LLM Variability"], color="salmon", edgecolor="black", alpha=0.8)
        axes[2].grid(True, linestyle="--", alpha=0.3)
        axes[2].set_ylabel("Variability")
        axes[2].set_ylim(0.0, 1.0)
        axes[2].set_yticks([0.0, 0.5, 1.0])
        
        # Format bottom x-axis with Phrase labels
        plt.xticks(x_indices, phrases_display, rotation=90, fontsize=8, ha="right")
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
                ax.grid(True, linestyle="--", alpha=0.3)
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

    print(f"All combined plots saved successfully to: {args.output_dir}")


if __name__ == "__main__":
    main()
