import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Finalize ToxVal dataset with 5% prevalence filtering")
    parser.add_argument("--input", required=True, help="Input grouped CSV file")
    parser.add_argument("--output", required=True, help="Output filtered CSV file")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    print(f"Reading {input_file} for final 5% prevalence filtering...")
    df = pd.read_csv(input_file, index_col="DTXSID", low_memory=False)
    n_rows = len(df)

    # Calculate absolute counts of positives
    positive_counts = (df == 1).sum()

    # Keep columns with >= 5 positive cases
    keep_cols = [col for col in df.columns if positive_counts[col] >= 5]

    print(f"Final Filtering Summary:")
    print(f"  Total Compounds: {n_rows}")
    print(f"  Total potential effects: {len(df.columns)}")
    print(f"  Effects meeting minimum 5 cases: {len(keep_cols)}")

    df_filtered = df[keep_cols]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df_filtered.to_csv(output_file)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
