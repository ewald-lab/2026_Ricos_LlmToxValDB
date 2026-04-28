import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from src.endpoint_normalization import (
    CATEGORY_NORMALIZATION_FIELDS,
    DEFAULT_ENDPOINTS_DIR,
    DEFAULT_OUTPUT_DIR,
    NORMALIZED_ROW_FIELDS,
    UNIQUE_ENDPOINT_FIELDS,
    build_category_normalization,
    build_normalized_rows,
    build_unique_endpoint_list,
    discover_endpoint_csvs,
    write_csv,
)


def arg_parser():
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


def main():
    args = arg_parser()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=== Prepare Guidelines ===")
    print(f"Input endpoint directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    csv_paths = discover_endpoint_csvs(input_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No OECD_*_endpoints.csv files found in {input_dir}")

    normalized_rows = build_normalized_rows(csv_paths)
    unique_endpoints = build_unique_endpoint_list(normalized_rows)
    category_normalization = build_category_normalization(normalized_rows)

    rows_path = write_csv(
        output_dir / "normalized_endpoint_rows.csv",
        NORMALIZED_ROW_FIELDS,
        normalized_rows,
    )
    unique_path = write_csv(
        output_dir / "normalized_endpoint_list.csv",
        UNIQUE_ENDPOINT_FIELDS,
        unique_endpoints,
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
    print(f"Wrote {len(normalized_rows)} normalized endpoint rows: {rows_path}")
    print(f"Wrote {len(unique_endpoints)} unique endpoint rows: {unique_path}")
    print(f"Wrote {len(category_normalization)} category mappings: {categories_path}")
    print(
        f"Normalized categories ({len(normalized_categories)}): "
        f"{', '.join(normalized_categories)}"
    )


if __name__ == "__main__":
    main()
