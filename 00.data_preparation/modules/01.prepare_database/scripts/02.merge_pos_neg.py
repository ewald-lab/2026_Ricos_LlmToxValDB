import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Merge positive and negative toxicological effect datasets.")
    parser.add_argument("--positives", required=True, help="Path to positives TSV.")
    parser.add_argument("--negatives", required=True, help="Path to negatives TSV.")
    parser.add_argument("--output", required=True, help="Path to output merged TSV.")
    args = parser.parse_args()

    print(f"Loading positives from {args.positives}...")
    df_pos = pd.read_csv(args.positives, sep="\t", low_memory=False)
    
    print(f"Loading negatives from {args.negatives}...")
    df_neg = pd.read_csv(args.negatives, sep="\t", low_memory=False)

    # Ensure NEGATIVE column exists in positives and set to False
    if 'NEGATIVE' not in df_pos.columns:
        df_pos['NEGATIVE'] = False
    else:
        # Just in case it existed but with other values
        df_pos['NEGATIVE'] = False

    # Standardize column order. 
    # We'll use the column order from the negatives file as the base, 
    # but ensure it includes all columns from positives too.
    neg_cols = list(df_neg.columns)
    pos_cols = list(df_pos.columns)
    
    # Combined list of columns, preserving order from neg_cols then adding missing from pos_cols
    all_cols = neg_cols + [c for c in pos_cols if c not in neg_cols]
    
    print(f"Merging datasets...")
    # Concatenate
    df_merged = pd.concat([df_pos, df_neg], ignore_index=True)
    
    # Fill NaN for NEGATIVE (shouldn't be any now, but just in case)
    df_merged['NEGATIVE'] = df_merged['NEGATIVE'].fillna(False)

    # Reorder columns
    df_merged = df_merged[all_cols]

    print(f"Saving merged dataset to {args.output} (Shape: {df_merged.shape})...")
    df_merged.to_csv(args.output, sep="\t", index=False)
    print("Done.")

if __name__ == "__main__":
    main()
