import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate LLM performance summary table and plot."
    )
    parser.add_argument(
        "--expert_expanded_dir",
        default="/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/03.evaluation/modules/01.eval_experts/outputs",
        help="Directory containing human expert Expanded_*.csv files.",
    )
    parser.add_argument(
        "--llm_expanded_dir",
        default="/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/03.evaluation/modules/02.eval_llms/outputs",
        help="Directory containing LLM Expanded_ToxValLlm_llm_*.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        default="/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/03.evaluation/modules/02.eval_llms/outputs/figures",
        help="Directory where the output table plot will be saved.",
    )
    return parser.parse_args()

def calc_metrics(pred_set, ref_set):
    if not pred_set and not ref_set:
        return 1.0, 1.0, 1.0
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

def main():
    args = arg_parser()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load average human reference
    avg_human_file = os.path.join(args.expert_expanded_dir, "Expanded_average_human.csv")
    if not os.path.exists(avg_human_file):
        raise FileNotFoundError(f"Average human reference not found at: {avg_human_file}")

    df_avg_human = pd.read_csv(avg_human_file)
    df_avg_human["All_Terms_Set"] = df_avg_human["All_Terms_Str"].apply(
        lambda x: set(str(x).split("|")) if pd.notna(x) and str(x) != "" else set()
    )
    human_ref_map = {row["Index"]: row["All_Terms_Set"] for _, row in df_avg_human.iterrows()}

    # Load individual human experts
    expert_files = [
        "Expanded_Alexander_Cole_Dynanet_Corporation.csv",
        "Expanded_Chelsea_Weitekamp_ULRI_Chemical_Insights.csv",
        "Expanded_Joshua_Harrill_Corteva.csv",
        "Expanded_Lucie_Ford_Recursion.csv",
        "Expanded_Sean_Watford_ULRI.csv"
    ]
    experts_data = {}
    for ef in expert_files:
        ef_path = os.path.join(args.expert_expanded_dir, ef)
        if os.path.exists(ef_path):
            df_exp = pd.read_csv(ef_path)
            df_exp["All_Terms_Set"] = df_exp["All_Terms_Str"].apply(
                lambda x: set(str(x).split("|")) if pd.notna(x) and str(x) != "" else set()
            )
            expert_id = ef.replace("Expanded_", "").replace(".csv", "")
            orig_name = f"ToxValLlm_expert_{expert_id}_ontology_mp.csv"
            orig_path = os.path.join(args.expert_expanded_dir.replace("outputs", "inputs/mp"), orig_name)
            if os.path.exists(orig_path):
                df_orig = pd.read_csv(orig_path)
                df_exp["Not_My_Expertise"] = pd.to_numeric(df_orig["Not_My_Expertise"], errors="coerce").fillna(0).values
            else:
                df_exp["Not_My_Expertise"] = 0
            experts_data[expert_id] = df_exp

    # Model specifications and optimal config IDs
    models = [
        {
            "name": "gpt-oss-20b",
            "id": "ToxValLlm_llm_gpt-oss-20b_ontology_mp_mode_2_consensus_1_no_defs",
            "total_params": "21B", "active_params": "3.6B", "dtype": "BF16", "gpu": "1x A100 / 1x H200"
        },
        {
            "name": "Qwen3.5-27B-FP8",
            "id": "ToxValLlm_llm_Qwen3.5-27B-FP8_ontology_mp_mode_1_consensus_1_with_defs",
            "total_params": "27B", "active_params": "27B", "dtype": "FP8", "gpu": "1x A100 / 1x H200"
        },
        {
            "name": "Qwen3.6-27B-FP8",
            "id": "ToxValLlm_llm_Qwen3.6-27B-FP8_ontology_mp_mode_2_consensus_1_with_defs",
            "total_params": "27B", "active_params": "27B", "dtype": "FP8", "gpu": "1x A100 / 1x H200"
        },
        {
            "name": "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
            "id": "ToxValLlm_llm_NVIDIA-Nemotron-3-Nano-30B-A3B-FP8_ontology_mp_mode_2_consensus_1_no_defs",
            "total_params": "30B", "active_params": "3.0B", "dtype": "FP8", "gpu": "1x A100 / 1x H200"
        },
        {
            "name": "Qwen3.5-35B-A3B-FP8",
            "id": "ToxValLlm_llm_Qwen3.5-35B-A3B-FP8_ontology_mp_mode_2_consensus_1_no_defs",
            "total_params": "35B", "active_params": "3.0B", "dtype": "FP8", "gpu": "1x A100 / 1x H200"
        },
        {
            "name": "Qwen3.6-35B-A3B-FP8",
            "id": "ToxValLlm_llm_Qwen3.6-35B-A3B-FP8_ontology_mp_mode_2_consensus_1_no_defs",
            "total_params": "35B", "active_params": "3.0B", "dtype": "FP8", "gpu": "1x A100 / 1x H200"
        },
        {
            "name": "Qwen3.5-122B-A10B-FP8",
            "id": "ToxValLlm_llm_Qwen3.5-122B-A10B-FP8_ontology_mp_mode_2_consensus_1_no_defs",
            "total_params": "122B", "active_params": "10.0B", "dtype": "FP8", "gpu": "4x A100 / 2x H200"
        },
        {
            "name": "NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
            "id": "ToxValLlm_llm_NVIDIA-Nemotron-3-Super-120B-A12B-FP8_ontology_mp_mode_2_consensus_1_no_defs",
            "total_params": "120B", "active_params": "12.0B", "dtype": "FP8", "gpu": "4x A100 / 2x H200"
        },
        {
            "name": "gpt-oss-120b",
            "id": "ToxValLlm_llm_gpt-oss-120b_ontology_mp_mode_2_consensus_1_no_defs",
            "total_params": "117B", "active_params": "5.1B", "dtype": "BF16", "gpu": "4x A100 / 2x H200"
        }
    ]

    results = []
    for m in models:
        csv_path = os.path.join(args.llm_expanded_dir, m["name"], f"Expanded_{m['id']}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: File not found {csv_path}, skipping model {m['name']}")
            continue

        df = pd.read_csv(csv_path)
        df["All_Terms_Set"] = df["All_Terms_Str"].apply(
            lambda x: set(str(x).split("|")) if pd.notna(x) and str(x) != "" else set()
        )

        # Calculate Precision, Recall, F1, Jaccard vs each individual human expert
        expert_precisions = []
        expert_recalls = []
        expert_f1s = []
        expert_jaccards = []

        for expert_id, df_expert in experts_data.items():
            precisions = []
            recalls = []
            f1s = []
            jaccards = []

            # Map expert terms by index
            expert_map = {row["Index"]: row["All_Terms_Set"] for _, row in df_expert.iterrows()}

            for idx, human_terms in expert_map.items():
                row = df[df["Index"] == idx]
                if len(row) == 0:
                    continue
                # Skip if either marked Not_My_Expertise
                not_exp_m = row.iloc[0].get("Not_My_Expertise", 0) == 1
                not_exp_h = df_expert[df_expert["Index"] == idx].iloc[0].get("Not_My_Expertise", 0) == 1
                if not_exp_m or not_exp_h:
                    continue
                # Skip evaluating this phrase if the human expert has no terms
                if not human_terms:
                    continue
                llm_terms = row.iloc[0]["All_Terms_Set"]
                p_val, r_val, f1_val = calc_metrics(llm_terms, human_terms)
                precisions.append(p_val)
                recalls.append(r_val)
                f1s.append(f1_val)

                # Jaccard
                u = llm_terms.union(human_terms)
                inter = llm_terms.intersection(human_terms)
                j_val = len(inter) / len(u) if len(u) > 0 else 1.0
                jaccards.append(j_val)

            if precisions:
                expert_precisions.append(np.mean(precisions))
            if recalls:
                expert_recalls.append(np.mean(recalls))
            if f1s:
                expert_f1s.append(np.mean(f1s))
            if jaccards:
                expert_jaccards.append(np.mean(jaccards))

        mean_p = np.mean(expert_precisions) if expert_precisions else 0.0
        std_p = np.std(expert_precisions) if expert_precisions else 0.0
        mean_r = np.mean(expert_recalls) if expert_recalls else 0.0
        std_r = np.std(expert_recalls) if expert_recalls else 0.0
        mean_f1 = np.mean(expert_f1s) if expert_f1s else 0.0
        std_f1 = np.std(expert_f1s) if expert_f1s else 0.0
        mean_j = np.mean(expert_jaccards) if expert_jaccards else 0.0
        std_j = np.std(expert_jaccards) if expert_jaccards else 0.0

        # Extract tokens
        tokens_path = os.path.join(args.llm_expanded_dir, m["name"], f"{m['id']}_tokens.txt")
        avg_completion_tokens = 0
        if os.path.exists(tokens_path):
            try:
                with open(tokens_path, "r", encoding="utf-8") as tf:
                    for line in tf:
                        if line.startswith("Mean:"):
                            avg_completion_tokens = int(float(line.replace("Mean:", "").strip()))
            except Exception as re:
                print(f"  WARNING: Error reading tokens from {tokens_path}: {re}")

        results.append({
            "Model": m["name"],
            "Precision Mean": mean_p,
            "Precision Std": std_p,
            "Recall Mean": mean_r,
            "Recall Std": std_r,
            "F1 Mean": mean_f1,
            "F1 Std": std_f1,
            "Jaccard Mean": mean_j,
            "Jaccard Std": std_j,
            "Total Params": m["total_params"],
            "Active Params": m["active_params"],
            "Data Type": m["dtype"],
            "Avg Tokens": avg_completion_tokens,
            "GPU Requirements": m["gpu"]
        })

    if not results:
        print("No results processed.")
        return

    res_df = pd.DataFrame(results)

    # Plot Table
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
    print(f"Saved plot table to: {output_path}")

if __name__ == "__main__":
    main()
