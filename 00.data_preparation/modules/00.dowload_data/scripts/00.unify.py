import pandas as pd
import glob
import os
import argparse

def main():
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Unify ToxValDB Excel files into a single TSV.")
    parser.add_argument("output_path", help="Full path for the output TSV file (e.g., outputs/ToxValDB_Unified.tsv)")
    parser.add_argument("--input_folder", default="Data Excel Files", help="Folder containing the Excel files (default: 'Data Excel Files')")
    
    args = parser.parse_args()
    
    output_file = args.output_path
    input_folder = args.input_folder

    # --- CONFIGURATION ---
    # Pattern: Search for files starting with "toxval_all_" inside the input folder
    source_pattern = os.path.join(input_folder, "toxval_all_res_toxval_v97_0_*.xlsx")

    print(f"--> Searching for files in: {source_pattern}")
    files = glob.glob(source_pattern)
    files.sort()

    if not files:
        print(f"ERROR: No files found. Verify that '{input_folder}' folder exists and contains .xlsx files.")
        return

    print(f"--> Found {len(files)} Excel files to process.")

    all_data = []

    # --- PROCESSING ---
    for i, f in enumerate(files):
        # Print progress
        print(f"[{i+1}/{len(files)}] Reading: {os.path.basename(f)} ...")
        try:
            # Read Excel file
            # dtype=str ensures CAS numbers or special codes are not corrupted
            df = pd.read_excel(f, dtype=str) 
            
            # Add source filename column for traceability
            df['source_filename'] = os.path.basename(f)
            
            all_data.append(df)
        except Exception as e:
            print(f"⚠️ ERROR reading {f}: {e}")

    # --- SAVING ---
    if all_data:
        print("--> Concatenating all tables (this may take a while)...")
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            print(f"--> Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        print(f"--> Writing {len(final_df)} rows to {output_file}...")
        # Using TSV (tab-separated values) as it's safer for chemical data than CSV
        final_df.to_csv(output_file, sep='\t', index=False)
        
        print("SUCCESS! File created correctly.")
    else:
        print("No data could be processed.")

if __name__ == "__main__":
    main()