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
    for c in [1, 3, 5, 10]:
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


def orig_count_per_row_expert(row) -> int:
    t1 = str(row["Selected_Candidates"]).split("|") if pd.notna(row["Selected_Candidates"]) else []
    t2 = str(row["Manual_Terms"]).split("|") if pd.notna(row["Manual_Terms"]) else []
    all_t = [t.strip().lower() for t in t1 + t2]
    valid_t = set(t for t in all_t if t not in ["none", "-", "unknown", ""])
    return len(valid_t)


def orig_count_per_row_llm(row) -> int:
    t1 = str(row["Selected_Candidates"]).split("|") if pd.notna(row["Selected_Candidates"]) else []
    t2 = str(row["Manual_Terms"]).split("|") if pd.notna(row["Manual_Terms"]) else []
    all_t = [t.strip().lower() for t in t1 + t2]
    valid_t = set(t for t in all_t if t not in ["none", "-", "unknown", ""])
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
        avg_orig_terms = (
            df_orig.apply(orig_count_per_row_expert, axis=1).mean()
            if df_orig is not None else 0
        )

        all_metrics.append(
            {
                "Entity_ID": expert_id,
                "Entity": clean_expert_name(expert_id),
                "Type": "Human Expert",
                "Avg_Orig_Num_Terms": avg_orig_terms,
                "Mean_Max_Depth": df_exp["Max_Depth"].mean(),
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
        avg_orig_terms = (
            df_orig.apply(orig_count_per_row_llm, axis=1).mean()
            if df_orig is not None else 0
        )

        all_metrics.append(
            {
                "Entity_ID": llm_id,
                "Entity": clean_llm_name(llm_id),
                "Type": "LLM",
                "Avg_Orig_Num_Terms": avg_orig_terms,
                "Mean_Max_Depth": df_exp["Max_Depth"].mean(),
                "Selected_Terms": selected_count_total,
                "Manual_Terms": manual_count_total,
                "Nonsense_Flags": nonsense_count,
                "Not_Expertise_Flags": not_expertise_count,
            }
        )

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
                llm_terms = llm_row.iloc[0]["All_Terms_Set"]
                p_val, r_val, f1_val = calc_metrics(llm_terms, human_terms)
                precisions.append(p_val)
                recalls.append(r_val)
                f1s.append(f1_val)
            mean_p = np.mean(precisions) if precisions else 0.0
            mean_r = np.mean(recalls) if recalls else 0.0
            mean_f1 = np.mean(f1s) if f1s else 0.0

        metadata = parse_llm_metadata(llm_id)
        if metadata:
            llm_perf_metrics.append({
                "LLM_ID": llm_id,
                "Model": metadata["model"],
                "Mode": f"Mode {metadata['mode']}" if metadata["mode"] else "Unknown",
                "Consensus": metadata["consensus"],
                "Definitions": "With Defs" if metadata["definitions"] else "No Defs",
                "Precision": mean_p,
                "Recall": mean_r,
                "F1": mean_f1
            })

    metrics_df = pd.DataFrame(all_metrics)
    entity_ids = list(all_data.keys())
    cleaned_names = [clean_expert_name(eid) if eid in exp_expert_files or not eid.startswith("ToxValLlm_llm_") else clean_llm_name(eid) for eid in entity_ids]

    # Calculate actual clean names list mapping for indexing consistency
    id_to_clean = {}
    for eid in entity_ids:
        if eid.startswith("ToxValLlm_llm_"):
            id_to_clean[eid] = clean_llm_name(eid)
        else:
            id_to_clean[eid] = clean_expert_name(eid)
    cleaned_names = [id_to_clean[eid] for eid in entity_ids]

    # Pairwise overlap (Jaccard)
    print("Calculating pairwise overlap...")
    num_phrases = len(all_data[entity_ids[0]])
    pairwise_overlap = np.zeros((len(entity_ids), len(entity_ids)))
    for i, id1 in enumerate(entity_ids):
        for j, id2 in enumerate(entity_ids):
            scores = []
            for k in range(num_phrases):
                s1 = all_data[id1].iloc[k]["All_Terms_Set"]
                s2 = all_data[id2].iloc[k]["All_Terms_Set"]
                u = s1.union(s2)
                inter = s1.intersection(s2)
                score = len(inter) / len(u) if len(u) > 0 else 1.0
                scores.append(score)
            pairwise_overlap[i, j] = np.mean(scores)

    print("Generating combined plots...")

    # Plot 1: Pairwise Heatmap
    num_entities = len(entity_ids)
    heatmap_width = max(12, num_entities * 0.5 + 4)
    heatmap_height = max(10, num_entities * 0.5 + 2)
    plt.figure(figsize=(heatmap_width, heatmap_height))
    
    annot_font_size = max(5, 10 - num_entities // 5)
    tick_font_size = max(6, 11 - num_entities // 6)

    sns.heatmap(
        pairwise_overlap,
        annot=True,
        annot_kws={"size": annot_font_size},
        xticklabels=cleaned_names,
        yticklabels=cleaned_names,
        cmap="Blues",
        fmt=".2f",
    )
    plt.title("Combined Pairwise Overlap (Average Jaccard Similarity) - Humans & LLMs", fontsize=max(10, 14 - num_entities // 8))
    plt.xticks(rotation=90, fontsize=tick_font_size)
    plt.yticks(rotation=0, fontsize=tick_font_size)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "combined_pairwise_overlap_heatmap.png"))
    plt.close()

    # Plot 2: Avg Terms per Entity
    bar_width = max(14, num_entities * 0.4 + 2)
    plt.figure(figsize=(bar_width, 7))
    sns.barplot(
        data=metrics_df,
        x="Entity",
        y="Avg_Orig_Num_Terms",
        hue="Type",
        palette={"Human Expert": "skyblue", "LLM": "lightgreen"},
    )
    plt.title("Combined: Average Number of Original Terms per Phrase")
    plt.xticks(rotation=90, fontsize=tick_font_size)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "combined_avg_terms.png"))
    plt.close()

    # Plot 3: Max Depth per Entity
    plt.figure(figsize=(bar_width, 7))
    sns.barplot(
        data=metrics_df,
        x="Entity",
        y="Mean_Max_Depth",
        hue="Type",
        palette={"Human Expert": "salmon", "LLM": "mediumpurple"},
    )
    plt.title("Combined: Average Maximum Depth of Terms per Phrase")
    plt.xticks(rotation=90, fontsize=tick_font_size)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "combined_max_depth.png"))
    plt.close()

    # Plot 4: Manual vs Selected
    plt.figure(figsize=(bar_width, 7))
    plt.bar(
        metrics_df["Entity"],
        metrics_df["Selected_Terms"],
        label="Selected Candidates",
        color="skyblue",
    )
    plt.bar(
        metrics_df["Entity"],
        metrics_df["Manual_Terms"],
        bottom=metrics_df["Selected_Terms"],
        label="Manual Terms",
        color="salmon",
    )
    plt.xticks(rotation=90, fontsize=tick_font_size)
    plt.title("Combined Usage of Selected Candidates vs Manual Terms")
    plt.ylabel("Total Terms Used")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "combined_manual_vs_selected_terms.png"))
    plt.close()

    # Plot 5: Entity Flags
    x = np.arange(len(metrics_df["Entity"]))
    width = 0.35
    plt.figure(figsize=(bar_width, 7))
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
    plt.xticks(x, metrics_df["Entity"], rotation=90, fontsize=tick_font_size)
    plt.title("Combined Phrases Flagged")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "combined_flags.png"))
    plt.close()

    # Plot 6: Dendrogram
    if len(entity_ids) > 1:
        dist_matrix = 1 - pairwise_overlap
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        condensed_dist = squareform(dist_matrix)
        Z = linkage(condensed_dist, method="ward")

        dendrogram_width = max(12, num_entities * 0.4 + 2)
        dendrogram_height = max(10, num_entities * 0.4 + 2)
        plt.figure(figsize=(dendrogram_width, dendrogram_height))
        dendrogram(Z, labels=cleaned_names, leaf_rotation=90, leaf_font_size=tick_font_size)
        plt.title("Combined Dendrogram of Human & LLM Similarity (Ward Linkage)")
        plt.ylabel("Distance (1 - Jaccard)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "combined_similarity_dendrogram.png"))
        plt.close()

    # 7. Generate Precision & Recall Plots vs. Human Average
    if human_ref_map is not None and llm_perf_metrics:
        print("Generating LLM vs. Human Precision/Recall analysis plots...")
        llm_perf_df = pd.DataFrame(llm_perf_metrics)
        
        # Plot 7: Precision & Recall vs. Consensus runs
        df_melted = llm_perf_df.melt(
            id_vars=["Model", "Mode", "Consensus", "Definitions"],
            value_vars=["Precision", "Recall"],
            var_name="Metric",
            value_name="Score"
        )
        # Ensure Consensus is plotted as numeric/categorical correctly
        df_melted["Consensus"] = pd.to_numeric(df_melted["Consensus"])
        
        try:
            g = sns.relplot(
                data=df_melted,
                x="Consensus",
                y="Score",
                hue="Mode",
                style="Definitions",
                col="Model",
                row="Metric",
                kind="line",
                markers=True,
                dashes=True,
                height=4,
                aspect=1.2,
                facet_kws={"sharex": True, "sharey": True}
            )
            g.set(xticks=[1, 3, 5, 10])
            g.fig.suptitle("LLM Precision & Recall vs. Human Response by Consensus Runs", y=1.02, fontsize=16)
            g.savefig(os.path.join(args.output_dir, "llm_perf_vs_consensus.png"), bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Error generating llm_perf_vs_consensus.png: {e}")

        # Plot 8: Mode 1 vs. Mode 2 comparison
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
            for ax, metric in zip(axes, ["Precision", "Recall"]):
                sns.barplot(
                    data=llm_perf_df,
                    x="Model",
                    y=metric,
                    hue="Mode",
                    ax=ax,
                    alpha=0.6,
                    errorbar=None
                )
                sns.stripplot(
                    data=llm_perf_df,
                    x="Model",
                    y=metric,
                    hue="Mode",
                    ax=ax,
                    dodge=True,
                    alpha=0.8,
                    linewidth=1,
                    edgecolor="black"
                )
                ax.set_title(f"LLM {metric} by Prompt Mode (Mode 1 vs Mode 2)")
                ax.set_ylabel(metric)
                ax.set_xlabel("Model")
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[:2], labels[:2], title="Prompt Mode")
                ax.tick_params(axis='x', rotation=45)
            plt.suptitle("Prompt Mode Comparison (Mode 1 vs Mode 2)", fontsize=16, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "llm_perf_vs_mode.png"), bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Error generating llm_perf_vs_mode.png: {e}")

        # Plot 9: Definitions usage comparison
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
            for ax, metric in zip(axes, ["Precision", "Recall"]):
                sns.barplot(
                    data=llm_perf_df,
                    x="Model",
                    y=metric,
                    hue="Definitions",
                    ax=ax,
                    alpha=0.6,
                    errorbar=None
                )
                sns.stripplot(
                    data=llm_perf_df,
                    x="Model",
                    y=metric,
                    hue="Definitions",
                    ax=ax,
                    dodge=True,
                    alpha=0.8,
                    linewidth=1,
                    edgecolor="black"
                )
                ax.set_title(f"LLM {metric} with vs. without Ontology Definitions")
                ax.set_ylabel(metric)
                ax.set_xlabel("Model")
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[:2], labels[:2], title="Definitions")
                ax.tick_params(axis='x', rotation=45)
            plt.suptitle("Ontology Definitions Usage Comparison", fontsize=16, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "llm_perf_vs_definitions.png"), bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Error generating llm_perf_vs_definitions.png: {e}")

        # Plot 10: Precision-Recall Trade-off Space
        try:
            plt.figure(figsize=(10, 8))
            plt.scatter(1.0, 1.0, marker="*", color="gold", s=300, label="Perfect Match", zorder=5, edgecolor="black")
            
            # Map size sizes dictionaries
            sizes_dict = {}
            for c in llm_perf_df["Consensus"].unique():
                sizes_dict[c] = int(c) * 30
            
            sns.scatterplot(
                data=llm_perf_df,
                x="Precision",
                y="Recall",
                hue="Model",
                style="Definitions",
                size="Consensus",
                sizes=sizes_dict,
                alpha=0.8,
                edgecolor="black",
                linewidth=1.2
            )
            plt.xlim(0.0, 1.05)
            plt.ylim(0.0, 1.05)
            plt.xlabel("Precision vs Human Response", fontsize=12)
            plt.ylabel("Recall vs Human Response", fontsize=12)
            plt.title("LLM Precision-Recall Trade-off Space (vs. Human Average)", fontsize=14)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "llm_precision_vs_recall_scatter.png"), bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Error generating llm_precision_vs_recall_scatter.png: {e}")

        # Plot 11: Definitions in Mode 1 vs Mode 2 Interaction Plot (Precision & Recall)
        try:
            n_models = llm_perf_df["Model"].nunique()
            fig, axes = plt.subplots(
                2, n_models,
                figsize=(max(6, n_models * 4), 10),
                sharey=False,
                sharex=True
            )
            # Ensure axes is always 2D
            if n_models == 1:
                axes = [[axes[0]], [axes[1]]]

            models = sorted(llm_perf_df["Model"].unique())
            palette = {"With Defs": "#2196F3", "No Defs": "#FF5722"}
            markers = {"With Defs": "o", "No Defs": "s"}

            for row_idx, metric in enumerate(["Precision", "Recall"]):
                for col_idx, model in enumerate(models):
                    ax = axes[row_idx][col_idx]
                    df_model = llm_perf_df[llm_perf_df["Model"] == model]
                    for defs_val, grp in df_model.groupby("Definitions"):
                        agg = grp.groupby("Mode")[metric].mean().reset_index()
                        ax.plot(
                            agg["Mode"], agg[metric],
                            marker=markers.get(defs_val, "o"),
                            linestyle="-" if defs_val == "With Defs" else "--",
                            color=palette.get(defs_val, "gray"),
                            label=defs_val,
                            linewidth=2,
                            markersize=9
                        )
                    ax.set_title(model if row_idx == 0 else "", fontsize=11)
                    ax.set_ylabel(f"{metric} (vs. Human Avg)", fontsize=10)
                    ax.set_xlabel("Prompt Mode" if row_idx == 1 else "")
                    ax.set_ylim(0, 1.0)
                    ax.grid(True, linestyle="--", alpha=0.5)
                    if col_idx == 0:
                        ax.legend(title="Definitions", fontsize=9)

            fig.suptitle(
                "Interaction: Effect of Ontology Definitions on Precision & Recall\n(Mode 1 vs. Mode 2)",
                fontsize=14, y=1.02
            )
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "llm_definitions_interaction.png"), bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Error generating llm_definitions_interaction.png: {e}")

    print(f"All combined plots saved successfully to: {args.output_dir}")


if __name__ == "__main__":
    main()
