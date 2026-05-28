from __future__ import annotations
import argparse
import pandas as pd
import logging
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils import setup_logging

def main() -> None:
    """
    Description:
    Main entry point for merging positive and negative toxicological cleaned data files
    and saving the concatenated output.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Merge positive and negative toxicological cleaned data.")
    parser.add_argument("--positives", required=True, help="Path to the positives TSV file.")
    parser.add_argument("--negatives", required=True, help="Path to the negatives TSV file.")
    parser.add_argument("--output", required=True, help="Path to the output merged TSV file.")
    args = parser.parse_args()

    logger = setup_logging(level=logging.INFO, stream=sys.stdout, fmt="%(asctime)s - %(levelname)s - %(message)s")
    start_time = time.time()

    logger.info(f"Loading positives from: {args.positives}")
    df_pos = pd.read_csv(args.positives, sep="\t", low_memory=False)
    logger.info(f"Positives shape: {df_pos.shape}")

    logger.info(f"Loading negatives from: {args.negatives}")
    df_neg = pd.read_csv(args.negatives, sep="\t", low_memory=False)
    logger.info(f"Negatives shape: {df_neg.shape}")

    # Add NEGATIVE column to positives and set to False
    logger.info("Setting NEGATIVE=False for positive findings...")
    df_pos['NEGATIVE'] = False

    # Ensure NEGATIVE is present in negatives (it should be True)
    if 'NEGATIVE' not in df_neg.columns:
        logger.warning("NEGATIVE column missing in negatives file. Setting to True.")
        df_neg['NEGATIVE'] = True
    else:
        # Just to be sure, fill any NaNs with True
        df_neg['NEGATIVE'] = df_neg['NEGATIVE'].fillna(True)

    # Concatenate
    logger.info("Merging dataframes...")
    df_merged = pd.concat([df_pos, df_neg], ignore_index=True)
    logger.info(f"Merged shape: {df_merged.shape}")

    # Save
    logger.info(f"Saving merged data to: {args.output}")
    df_merged.to_csv(args.output, sep="\t", index=False)

    logger.info(f"Done in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
