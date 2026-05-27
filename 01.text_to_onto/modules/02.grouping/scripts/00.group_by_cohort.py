"""
00.group_by_cohort.py

Groups the ToxValDB classified output by experiment cohort and computes
min/max concentration bounds for each ontology term (tox effect column).

Effect columns from 00.tox_effect_to_onto contain 1 (hit) or NaN (no hit).

The grouping includes TOXVAL_UNITS so that concentrations are only compared
within the same unit space.

Logic:
    1. Read the output CSV from 00.tox_effect_to_onto (rows with 1/NaN columns).
    2. Read the guideline mapping outputs from 01.guideline_to_onto (txt breadcrumb
       files) to build a set of "guideline-covered" ontology terms per guideline.
    3. Group rows by EXPERIMENT_COHORT columns (includes TOXVAL_UNITS).
    4. Within each cohort, split rows into N-family and L-family by TOXVAL_TYPE.
    5. For each tox-effect (ontology) column:
       - N-family: pick the row with the HIGHEST concentration that has a 1;
         if none has a 1, pick the highest concentration overall → N_value, N_conc.
       - L-family: pick the row with the LOWEST concentration that has a 1;
         if none has a 1, pick the lowest concentration overall → L_value, L_conc.
    6. Validate that N_conc < L_conc (NOAEL must be below LOAEL). If violated
       the pair is discarded (both set to NaN).
    7. Apply decision rules:
       - If N_value == 1 OR L_value == 1:
           min = L_conc (if L is NOT from a NEGATIVE row) else NaN
           max = N_conc
       - Elif neither is 1:
           If the tox effect is in the guideline set for this cohort's guideline:
               min = NaN
               max = N_conc
           Else:
               min = NaN, max = NaN
    8. Output a CSV grouped by experiment cohort with columns:
       EXPERIMENT_COHORT cols + tox_effect_min + tox_effect_max per ontology term.
"""

import argparse
import os
import re
import polars as pl
from pathlib import Path
from typing import Dict, Set, Tuple, Optional


# ---------------------------------------------------------------------------
# TOXVAL_TYPE family classification
# ---------------------------------------------------------------------------

FAM_N = [
    'NOAEL', 'NOEL', 'NOAEC', 'NOEC', 'NEL', 'LD0', 'LC0', 'NCEL',
    'NOAEL (HEC)', 'NOAEL (ADJ)', 'NOAEL (HED)', 'NOEL (TAD)', 'NOAEL (HEC ADJ)'
]

FAM_L = [
    'LOAEL', 'LEL', 'LOEL', 'LD50', 'LC50', 'LOAEC', 'LOEC', 'LD100', 'LC100',
    'LCLo', 'ALD', 'FEL', 'LOAEL (HEC)', 'LOAEL (ADJ)', 'LOAEL (HED)', 'LOEL (TAD)',
    'LOEL (HED)', 'LEC', 'AEL', 'LD40', 'LD10', 'LC83', 'LC10', 'LC67', 'LC17',
    'LC01', 'LC60', 'LD5', 'LC66', 'LC65', 'LD', 'FEL (ADJ)', 'FEL (HEC)', 'LEL (ADJ)'
]

FAM_N_SET = set(FAM_N)
FAM_L_SET = set(FAM_L)

# ---------------------------------------------------------------------------
# Experiment cohort definition
# ---------------------------------------------------------------------------

EXPERIMENT_COHORT = [
    'STUDY_GROUP', 'DTXSID', 'STUDY_TYPE', 'EXPOSURE_ROUTE',
    'SPECIES_COMMON', 'SEX', 'LIFESTAGE', 'GENERATION',
    'TOXVAL_UNITS',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Group ToxValDB classified data by experiment cohort."
    )
    parser.add_argument(
        "--input_file", required=True,
        help="Path to the classified CSV from 00.tox_effect_to_onto "
             "(e.g. ToxValDB_Classified_mp_*.csv)."
    )
    parser.add_argument(
        "--guidelines_dir", required=True,
        help="Path to the guideline mapping directory from 01.guideline_to_onto "
             "(e.g. outputs/guidelines_mapping/mp/guidelines)."
    )
    parser.add_argument(
        "--output_file", required=True,
        help="Path to the output CSV file."
    )
    parser.add_argument(
        "--concentration_col", default="TOXVAL_NUMERIC",
        help="Column name for the concentration/dose value."
    )
    parser.add_argument(
        "--toxval_type_col", default="TOXVAL_TYPE",
        help="Column name for the toxval type (used to classify N/L family)."
    )
    parser.add_argument(
        "--negative_col", default="NEGATIVE",
        help="Column name for the negative flag."
    )
    parser.add_argument(
        "--guideline_col", default="MATRIX_GUIDELINE",
        help="Column name for the OECD guideline number."
    )
    return parser.parse_args()


def load_guideline_terms(guidelines_dir: str) -> Dict[str, Set[str]]:
    """Load guideline → set of ontology term names from the breadcrumb txt files.

    Each file is named <guideline_number>.txt and contains breadcrumb paths like:
        mammalian phenotype > cellular phenotype > abnormal cell morphology
    The *last* element of each breadcrumb line is the leaf term name. We also
    store all intermediate nodes so that any ancestor covered by the guideline
    is also considered "in guideline".

    Returns:
        Dict mapping guideline number (str) to set of term names.
    """
    guideline_terms: Dict[str, Set[str]] = {}
    guidelines_path = Path(guidelines_dir)

    if not guidelines_path.exists():
        print(f"WARNING: Guidelines directory not found: {guidelines_path}")
        return guideline_terms

    for txt_file in sorted(guidelines_path.glob("*.txt")):
        g_num = txt_file.stem  # e.g. "408"
        terms: Set[str] = set()
        with open(txt_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                # Each element between " > " is a term in the hierarchy
                parts = [p.strip() for p in line.split(" > ")]
                terms.update(parts)
        guideline_terms[g_num] = terms

    return guideline_terms


def extract_guideline_number(guideline_str: str) -> str:
    """Extract the numeric guideline number from strings like 'OECD 408'."""
    if not guideline_str or guideline_str == "-":
        return ""
    match = re.search(r'(\d+)', str(guideline_str))
    return match.group(1) if match else str(guideline_str).strip()


def classify_family(toxval_type: str) -> Optional[str]:
    """Return 'N', 'L', or None depending on which family the type belongs to."""
    if toxval_type in FAM_N_SET:
        return "N"
    elif toxval_type in FAM_L_SET:
        return "L"
    return None


def pick_representative_row(
    sub_df: pl.DataFrame,
    effect_col: str,
    conc_col: str,
    family: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Pick the representative concentration and effect value for a family.

    Effect columns contain 1 (hit) or NaN (no hit).

    For N-family: highest concentration with a 1, else highest concentration.
    For L-family: lowest concentration with a 1, else lowest concentration.

    Returns (effect_value, concentration).
        effect_value is 1.0 when the row is a hit, None otherwise.
    """
    if sub_df.is_empty():
        return None, None

    # Only keep rows that have a non-null concentration
    valid = sub_df.filter(pl.col(conc_col).is_not_null())
    if valid.is_empty():
        return None, None

    # Rows where the effect column is 1 (NaN rows are excluded by is_not_null)
    hits = valid.filter(
        pl.col(effect_col).is_not_null() & (pl.col(effect_col) == 1.0)
    )

    if family == "N":
        # Prefer highest concentration with a 1
        if not hits.is_empty():
            best = hits.sort(conc_col, descending=True).row(0, named=True)
        else:
            best = valid.sort(conc_col, descending=True).row(0, named=True)
    else:  # family == "L"
        # Prefer lowest concentration with a 1
        if not hits.is_empty():
            best = hits.sort(conc_col, descending=False).row(0, named=True)
        else:
            best = valid.sort(conc_col, descending=False).row(0, named=True)

    return best.get(effect_col), best.get(conc_col)


def is_negative_row(row_val) -> bool:
    """Check if the NEGATIVE column indicates a negative finding."""
    if row_val is None:
        return False
    if isinstance(row_val, bool):
        return row_val
    s = str(row_val).strip().lower()
    return s in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Loading classified data from: {args.input_file}")
    df = pl.read_csv(args.input_file, ignore_errors=True)
    print(f"  Loaded {df.height} rows, {df.width} columns")

    # Identify ontology effect columns (everything that is not a metadata column)
    metadata_cols = set(EXPERIMENT_COHORT) | {
        args.concentration_col, args.toxval_type_col, args.negative_col,
        args.guideline_col,
        'CASRN', 'NAME', 'SOURCE', 'SUB_SOURCE', 'TOXVAL_SUBTYPE',
        'TOXVAL_TYPE_SUPERCATEGORY', 'QUALIFIER', 'TOXVAL_UNITS',
        'RISK_ASSESSMENT_CLASS', 'STUDY_DURATION_CLASS',
        'STUDY_DURATION_VALUE', 'STUDY_DURATION_UNITS', 'STRAIN',
        'LATIN_NAME', 'SPECIES_SUPERCATEGORY', 'EXPOSURE_METHOD',
        'EXPOSURE_FORM', 'MEDIA', 'TOXICOLOGICAL_EFFECT',
        'TOXICOLOGICAL_EFFECT_CATEGORY', 'YEAR', 'ORIGINAL_YEAR',
        'MATRIX_GUIDELINE', 'STUDY_DURATION_YEARS', 'TOXVAL_OUTLIER_FLAG',
        'TOXICOLOGICAL_EFFECT_LLM_CLEANED',
        'TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED', 'NEGATIVE',
    }
    effect_cols = [c for c in df.columns if c not in metadata_cols]
    print(f"  Found {len(effect_cols)} ontology effect columns")

    if not effect_cols:
        print("ERROR: No ontology effect columns found in the input file.")
        return

    # Load guideline term sets
    print(f"Loading guideline mappings from: {args.guidelines_dir}")
    guideline_terms = load_guideline_terms(args.guidelines_dir)
    print(f"  Loaded terms for {len(guideline_terms)} guidelines")

    # Classify each row into N or L family
    df = df.with_columns(
        pl.col(args.toxval_type_col)
        .map_elements(classify_family, return_dtype=pl.Utf8)
        .alias("_family")
    )

    # Filter to only N and L rows (drop BMD, BMC, etc. that are neither)
    df_nl = df.filter(pl.col("_family").is_not_null())
    print(f"  Rows in N/L families: {df_nl.height} (dropped {df.height - df_nl.height} non-N/L rows)")

    # Ensure concentration column is Float64
    df_nl = df_nl.with_columns(
        pl.col(args.concentration_col).cast(pl.Float64, strict=False)
    )

    # Ensure effect columns are Float64
    cast_exprs = []
    for ec in effect_cols:
        if ec in df_nl.columns:
            cast_exprs.append(pl.col(ec).cast(pl.Float64, strict=False))
    if cast_exprs:
        df_nl = df_nl.with_columns(cast_exprs)

    # Ensure cohort columns exist and fill nulls with "-"
    for col in EXPERIMENT_COHORT:
        if col not in df_nl.columns:
            df_nl = df_nl.with_columns(pl.lit("-").alias(col))
        else:
            df_nl = df_nl.with_columns(
                pl.col(col).cast(pl.Utf8).fill_null("-")
            )

    # Group by experiment cohort
    groups = df_nl.group_by(EXPERIMENT_COHORT, maintain_order=True)

    result_rows = []
    for cohort_key, cohort_df in groups:
        # Build cohort dict from the group key
        cohort_dict = {}
        for i, col_name in enumerate(EXPERIMENT_COHORT):
            cohort_dict[col_name] = cohort_key[i] if cohort_key[i] is not None else "-"

        # Determine the guideline number for this cohort
        guideline_raw = cohort_df[args.guideline_col].to_list()[0] if args.guideline_col in cohort_df.columns else ""
        g_num = extract_guideline_number(str(guideline_raw) if guideline_raw else "")
        g_terms = guideline_terms.get(g_num, set())

        # Split into N and L sub-dataframes
        n_df = cohort_df.filter(pl.col("_family") == "N")
        l_df = cohort_df.filter(pl.col("_family") == "L")

        for ec in effect_cols:
            if ec not in cohort_df.columns:
                cohort_dict[f"{ec}_min"] = None
                cohort_dict[f"{ec}_max"] = None
                continue

            # N-family: highest conc with 1, else highest conc
            n_val, n_conc = pick_representative_row(n_df, ec, args.concentration_col, "N")
            # L-family: lowest conc with 1, else lowest conc
            l_val, l_conc = pick_representative_row(l_df, ec, args.concentration_col, "L")

            # Check if L representative is from a NEGATIVE row
            l_is_negative = False
            if l_val is not None and not l_df.is_empty():
                # Find the actual row that was selected to check its NEGATIVE flag
                l_hits = l_df.filter(
                    pl.col(ec).is_not_null() & (pl.col(ec) == 1.0)
                )
                if not l_hits.is_empty():
                    l_sorted = l_hits.sort(args.concentration_col, descending=False)
                else:
                    l_sorted = l_df.filter(pl.col(args.concentration_col).is_not_null()).sort(
                        args.concentration_col, descending=False
                    )
                if not l_sorted.is_empty():
                    neg_val = l_sorted.row(0, named=True).get(args.negative_col)
                    l_is_negative = is_negative_row(neg_val)

            # Decision rules
            n_is_1 = (n_val is not None and n_val == 1.0)
            l_is_1 = (l_val is not None and l_val == 1.0)

            # Validate N_conc < L_conc (NOAEL must be below LOAEL).
            # If both concentrations exist and the constraint is violated,
            # discard the pair.
            conc_valid = True
            if (n_conc is not None and l_conc is not None
                    and n_conc >= l_conc):
                conc_valid = False

            if not conc_valid:
                min_val = None
                max_val = None
            elif n_is_1 or l_is_1:
                # At least one family detected a hit
                min_val = l_conc if (not l_is_negative) else None
                max_val = n_conc
            elif not n_is_1 and not l_is_1:
                # Extract the leaf term name from a prefixed column like "MP:term name"
                # to check against guideline terms
                term_name = ec.split(":", 1)[1] if ":" in ec else ec
                if term_name in g_terms:
                    min_val = None
                    max_val = n_conc
                else:
                    min_val = None
                    max_val = None
            else:
                min_val = None
                max_val = None

            cohort_dict[f"{ec}_min"] = min_val
            cohort_dict[f"{ec}_max"] = max_val

        result_rows.append(cohort_dict)

    # Build output DataFrame
    if result_rows:
        out_df = pl.DataFrame(result_rows)
    else:
        print("WARNING: No cohort groups produced. Output will be empty.")
        out_df = pl.DataFrame()

    # Write output
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    out_df.write_csv(args.output_file)
    print(f"Wrote {out_df.height} cohort rows to: {args.output_file}")
    print(f"  Columns: {out_df.width}")


if __name__ == "__main__":
    main()
