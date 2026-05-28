from __future__ import annotations
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from src.endpoint_normalization import (
    ALL_ENDPOINT_FIELDS,
    CATEGORY_NORMALIZATION_FIELDS,
    DEFAULT_ENDPOINTS_DIR,
    DEFAULT_OUTPUT_DIR,
    UNIQUE_ENDPOINT_FIELDS,
    build_all_endpoint_list,
    build_category_normalization,
    build_normalized_rows,
    build_unique_endpoint_list,
    discover_endpoint_csvs,
    write_csv,
)


def arg_parser() -> argparse.Namespace:
    """
    Description:
    Parses command line arguments for the guidelines normalization script.

    Args:
        None

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare normalized OECD guideline endpoint tables."
    )
    parser.add_argument(
        "--input_dir",
        default=str(DEFAULT_ENDPOINTS_DIR),
        help="Directory containing OECD_*_endpoints.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where normalized endpoint outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Description:
    Main entry point for preparing OECD guideline endpoint tables. It reads endpoint CSV files,
    standardizes, groups them by required and category normalization lists, and writes normalized CSVs.

    Args:
        None

    Returns:
        None
    """
    args = arg_parser()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=== Prepare Guidelines ===")
    print(f"Input endpoint directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    csv_paths = discover_endpoint_csvs(input_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No OECD endpoint CSV files found in {input_dir}")

    normalized_rows = build_normalized_rows(csv_paths)
    unique_endpoints = build_unique_endpoint_list(normalized_rows)
    all_endpoints = build_all_endpoint_list(normalized_rows)
    category_normalization = build_category_normalization(normalized_rows)

    unique_path = write_csv(
        output_dir / "normalized_endpoint_list.csv",
        UNIQUE_ENDPOINT_FIELDS,
        unique_endpoints,
    )
    all_path = write_csv(
        output_dir / "all_endpoints_list.csv",
        ALL_ENDPOINT_FIELDS,
        all_endpoints,
    )
    categories_path = write_csv(
        output_dir / "category_normalization.csv",
        CATEGORY_NORMALIZATION_FIELDS,
        category_normalization,
    )

    normalized_categories = sorted(
        {row["normalized_category"] for row in normalized_rows},
        key=str.lower,
    )

    print(f"Read {len(csv_paths)} endpoint CSV files.")

    print(f"Wrote {len(unique_endpoints)} required endpoint rows: {unique_path}")
    print(f"Wrote {len(all_endpoints)} all endpoint rows (required + optional): {all_path}")
    print(f"Wrote {len(category_normalization)} category mappings: {categories_path}")
    print(
        f"Normalized categories ({len(normalized_categories)}): "
        f"{', '.join(normalized_categories)}"
    )


if __name__ == "__main__":
    main()
