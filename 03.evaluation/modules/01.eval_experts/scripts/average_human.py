from __future__ import annotations
import argparse
import glob
import os
import pandas as pd
from collections import Counter


def arg_parser() -> argparse.Namespace:
    """
    Description:
    Parse command line arguments for the average human profile generator.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create an average human/expert profile from expanded expert CSV files."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing Expanded_*.csv files.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path where the average human profile CSV will be saved.",
    )
    parser.add_argument(
        "--min_experts",
        type=int,
        default=2,
        help="Minimum number of experts a term must appear in to be kept (default: 2).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Description:
    Main execution function to load expert expanded term CSV files, compute consensus terms, and save the averaged human profile.

    Args:
        None

    Returns:
        None
    """
    args = arg_parser()

    # Locate all Expanded_*.csv files in the input directory, excluding any average files
    pattern = os.path.join(args.input_dir, "Expanded_*.csv")
    files = [
        f for f in glob.glob(pattern)
        if "average" not in os.path.basename(f).lower()
    ]

    if not files:
        print(f"No expert Expanded_*.csv files found in {args.input_dir}")
        return

    print(f"Found {len(files)} expert files for averaging:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    # Read all dataframes
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Ensure consistent schema
        for col in ["Index", "Phrase"]:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing in {f}")
        dfs.append(df)

    # Let's align them by Index and Phrase
    # Since they should have the same structure, we can verify this
    ref_df = dfs[0][["Index", "Phrase"]].copy()
    for i, df in enumerate(dfs[1:], start=1):
        if not df[["Index", "Phrase"]].equals(ref_df):
            print(f"Warning: File {files[i]} does not perfectly match the first file in Index/Phrase order. Aligning on Index and Phrase.")

    # To be extremely safe and robust, let's use a dictionary-based aggregation:
    # key: (Index, Phrase), value: list of sets of terms
    data_dict = {}
    for df in dfs:
        for _, row in df.iterrows():
            idx = row["Index"]
            phrase = row["Phrase"]
            terms_str = row.get("All_Terms_Str", "")
            if pd.isna(terms_str) or not isinstance(terms_str, str):
                terms = set()
            else:
                terms = set(t.strip() for t in terms_str.split("|") if t.strip())

            key = (idx, phrase)
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append(terms)

    # Now compute the average human terms for each key
    aggregated_rows = []
    for (idx, phrase), term_sets in data_dict.items():
        # Count term frequencies across experts for this phrase
        term_counts = Counter()
        for t_set in term_sets:
            term_counts.update(t_set)

        # Select terms with count >= min_experts
        selected_terms = [
            term for term, count in term_counts.items()
            if count >= args.min_experts
        ]

        # Sort them for consistency and reproducibility
        selected_terms_sorted = sorted(selected_terms)
        all_terms_str = "|".join(selected_terms_sorted)

        aggregated_rows.append({
            "Index": idx,
            "Phrase": phrase,
            "All_Terms_Str": all_terms_str
        })

    # Create final DataFrame and sort by Index to maintain original order
    out_df = pd.DataFrame(aggregated_rows)
    out_df = out_df.sort_values(by="Index").reset_index(drop=True)

    # Make sure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    out_df.to_csv(args.output_file, index=False)
    print(f"Successfully created average human profile at: {args.output_file}")


if __name__ == "__main__":
    main()
