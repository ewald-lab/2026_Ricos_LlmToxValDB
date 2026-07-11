import argparse
import glob
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_term_set(term_str):
    if pd.isna(term_str):
        return set()
    terms = [t.strip().lower() for t in str(term_str).split("|")]
    return set(t for t in terms if t not in ["none", "-", "unknown", ""])

def main():
    parser = argparse.ArgumentParser(description="Generate Qwen FP8 vs No Quantization Comparison Table")
    parser.add_argument("--expert_expanded_dir", required=True, help="Path to Expanded expert files")
    parser.add_argument("--expert_orig_dir", required=True, help="Path to original expert files")
    parser.add_argument("--llm_expanded_dir", required=True, help="Path to LLM expanded outputs (main outputs folder)")
    parser.add_argument("--output_dir", required=True, help="Output folder for the comparison table image")
    parser.add_argument("--ontology", default="mp", help="Ontology name (mp)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Human Expert Data
    exp_expert_files = glob.glob(os.path.join(args.expert_expanded_dir, "Expanded_*.csv"))
    exp_expert_files = [f for f in exp_expert_files if "average_human" not in os.path.basename(f)]

    all_data = {}
    human_expert_ids = []

    print("Loading expert data...")
    for f in exp_expert_files:
        fname = os.path.basename(f)
        expert_id = fname.replace("Expanded_", "").replace(".csv", "")
        df_exp = pd.read_csv(f)
        df_exp["All_Terms_Set"] = df_exp["All_Terms_Str"].apply(parse_term_set)
        
        orig_path = os.path.join(
            args.expert_orig_dir,
            f"ToxValLlm_expert_{expert_id}_ontology_{args.ontology}.csv",
        )
        if os.path.exists(orig_path):
            df_orig = pd.read_csv(orig_path)
            not_exp = pd.to_numeric(df_orig.get("Not_My_Expertise", 0), errors="coerce").fillna(0)
            nonsense = pd.to_numeric(df_orig.get("Toxicological_Effect_Does_Not_Make_Sense", 0), errors="coerce").fillna(0)
            df_exp["Not_My_Expertise"] = ((not_exp == 1) | (nonsense == 1)).astype(int).values
        else:
            df_exp["Not_My_Expertise"] = 0
            
        all_data[expert_id] = df_exp
        human_expert_ids.append(expert_id)

    # 2. Define the 4 target configurations
    configs = [
        {
            "name": "Qwen3.5-35B-A3B (FP8)",
            "model": "Qwen3.5-35B-A3B",
            "is_fp8": True,
            "total_params": "35B",
            "active_params": "3.0B",
            "dtype": "FP8",
            "gpu": "1x H200",
            "run_files_pattern": os.path.join(args.llm_expanded_dir, "reproducibility", "Expanded_ToxValLlm_llm_Qwen3.5-35B-A3B-FP8_ontology_mp_mode_2_consensus_1_no_defs_{run}.csv"),
            "tokens_path": os.path.join(args.llm_expanded_dir, "Qwen3.5-35B-A3B-FP8", "ToxValLlm_llm_Qwen3.5-35B-A3B-FP8_ontology_mp_mode_2_consensus_1_no_defs_tokens.txt"),
            "token_overwritten_reasoning": None
        },
        {
            "name": "Qwen3.5-35B-A3B (No Quantization)",
            "model": "Qwen3.5-35B-A3B",
            "is_fp8": False,
            "total_params": "35B",
            "active_params": "3.0B",
            "dtype": "BF16",
            "gpu": "2x H200",
            "run_files_pattern": os.path.join(args.llm_expanded_dir, "no_quantization", "Expanded_ToxValLlm_llm_Qwen3.5-35B-A3B_ontology_mp_mode_2_consensus_1_no_defs_run_{run}.csv"),
            "tokens_path": None,
            "token_overwritten_reasoning": os.path.join(args.llm_expanded_dir, "no_quantization", "ToxValLlm_llm_Qwen3.5-35B-A3B_ontology_mp_mode_2_consensus_1_no_defs_reasoning_run_{run}.json")
        },
        {
            "name": "Qwen3.6-35B-A3B (FP8)",
            "model": "Qwen3.6-35B-A3B",
            "is_fp8": True,
            "total_params": "35B",
            "active_params": "3.0B",
            "dtype": "FP8",
            "gpu": "1x H200",
            "run_files_pattern": os.path.join(args.llm_expanded_dir, "reproducibility", "Expanded_ToxValLlm_llm_Qwen3.6-35B-A3B-FP8_ontology_mp_mode_2_consensus_1_no_defs_{run}.csv"),
            "tokens_path": os.path.join(args.llm_expanded_dir, "Qwen3.6-35B-A3B-FP8", "ToxValLlm_llm_Qwen3.6-35B-A3B-FP8_ontology_mp_mode_2_consensus_1_no_defs_tokens.txt"),
            "token_overwritten_reasoning": None
        },
        {
            "name": "Qwen3.6-35B-A3B (No Quantization)",
            "model": "Qwen3.6-35B-A3B",
            "is_fp8": False,
            "total_params": "35B",
            "active_params": "3.0B",
            "dtype": "BF16",
            "gpu": "2x H200",
            "run_files_pattern": os.path.join(args.llm_expanded_dir, "no_quantization", "Expanded_ToxValLlm_llm_Qwen3.6-35B-A3B_ontology_mp_mode_2_consensus_1_no_defs_run_{run}.csv"),
            "tokens_path": None,
            "token_overwritten_reasoning": os.path.join(args.llm_expanded_dir, "no_quantization", "ToxValLlm_llm_Qwen3.6-35B-A3B_ontology_mp_mode_2_consensus_1_no_defs_reasoning_run_{run}.json")
        }
    ]

    table_results = []

    for conf in configs:
        print(f"Evaluating {conf['name']}...")
        df_runs = []
        for run_r in range(1, 11):
            filepath = conf["run_files_pattern"].format(run=run_r)
            if os.path.exists(filepath):
                df_run = pd.read_csv(filepath)
                df_run["All_Terms_Set"] = df_run["All_Terms_Str"].apply(parse_term_set)
                df_runs.append(df_run)
            else:
                print(f"  WARNING: File not found: {filepath}")

        if not df_runs:
            print(f"  ERROR: No runs found for {conf['name']}")
            continue

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

        # Retrieve avg completion tokens
        avg_completion_tokens = 0.0
        if conf["tokens_path"] and os.path.exists(conf["tokens_path"]):
            try:
                with open(conf["tokens_path"], "r", encoding="utf-8") as tf:
                    for line in tf:
                        if line.startswith("Mean:"):
                            avg_completion_tokens = float(line.replace("Mean:", "").strip())
            except Exception as re:
                print(f"  WARNING: Error reading tokens from {conf['tokens_path']}: {re}")
        elif conf["token_overwritten_reasoning"]:
            token_means = []
            for run_r in range(1, 11):
                reasoning_path = conf["token_overwritten_reasoning"].format(run=run_r)
                if os.path.exists(reasoning_path):
                    try:
                        with open(reasoning_path, "r", encoding="utf-8") as tf:
                            for line in tf:
                                if line.startswith("Mean:"):
                                    token_means.append(float(line.replace("Mean:", "").strip()))
                    except Exception as re:
                        pass
            if token_means:
                avg_completion_tokens = np.mean(token_means)

        table_results.append({
            "Model": conf["name"],
            "Precision Mean": mean_p,
            "Precision Std": std_p,
            "Recall Mean": mean_r,
            "Recall Std": std_r,
            "F1 Mean": mean_f1,
            "F1 Std": std_f1,
            "Jaccard Mean": mean_j,
            "Jaccard Std": std_j,
            "Total Params": conf["total_params"],
            "Active Params": conf["active_params"],
            "Data Type": conf["dtype"],
            "Avg Tokens": int(avg_completion_tokens),
            "GPU Requirements": conf["gpu"]
        })

    if table_results:
        res_df = pd.DataFrame(table_results)
        fig, ax = plt.subplots(figsize=(18, 3.2))
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
                f"{row['Avg Tokens']:,}" if row['Avg Tokens'] > 0 else "N/A"
            ])

        col_labels = [
            "Model", "Precision (Mean ± Std)", "Recall (Mean ± Std)", "F1 Score (Mean ± Std)",
            "Jaccard Overlap (Mean ± Std)", "Avg Tokens"
        ]

        col_widths = [0.30, 0.14, 0.14, 0.14, 0.18, 0.10]

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

        # plt.title("FP8 vs No Quantization Evaluation Metrics & Hardware Requirements",
        #           fontsize=14, fontweight='bold', pad=6, color='#1f4e79')

        output_path = os.path.join(args.output_dir, "quantization_comparison_table.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved summary table plot to: {output_path}")

if __name__ == "__main__":
    main()
