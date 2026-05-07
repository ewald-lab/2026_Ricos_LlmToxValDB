#!/usr/bin/env python3
"""Utilities for normalizing OECD guideline endpoint CSVs.

The guideline extraction step writes one CSV per guideline in
``outputs/endpoints``. This module keeps the downstream preparation limited to
normalized category and endpoint labels; ontology matching happens elsewhere.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

MODULE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENDPOINTS_DIR = MODULE_ROOT / "outputs" / "endpoints"
DEFAULT_OUTPUT_DIR = MODULE_ROOT / "outputs" / "endpoints"

def load_category_map() -> dict[str, str]:
    map_path = MODULE_ROOT / "config" / "category_normalization.csv"
    mapping = {}
    if map_path.exists():
        with open(map_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                original = row.get("original_category", "").strip()
                normalized = row.get("normalized_category", "").strip()
                if original:
                    mapping[original] = normalized
    return mapping

CATEGORY_MAP = load_category_map()

CATEGORY_SUFFIX = {
    "Organ Weights": "weight",
    "Histopathology": "histopathology",
}

ORGAN_SYNONYMS = {
    "testes": "testes",
    "testis": "testes",
    "ovaries": "ovaries",
    "ovary": "ovaries",
    "adrenals": "adrenal glands",
    "adrenal glands": "adrenal glands",
    "adrenal": "adrenal glands",
    "epididymides": "epididymides",
    "epididymis": "epididymides",
    "epididymides (total and cauda)": "epididymides",
    "kidneys": "kidneys",
    "kidney": "kidneys",
    "paired kidneys": "kidneys",
    "paired adrenals": "adrenal glands",
    "thyroid": "thyroid",
    "thyroid gland": "thyroid",
    "thyroid and parathyroid glands": "thyroid and parathyroid glands",
    "pituitary": "pituitary",
    "pituitary gland": "pituitary",
    "uterus": "uterus",
    "uterus, including cervix": "uterus with cervix",
    "uterus with cervix": "uterus with cervix",
    "mammary glands (female and male)": "mammary glands",
    "male mammary glands": "male mammary glands",
    "liver": "liver",
    "brain": "brain",
    "heart": "heart",
    "spleen": "spleen",
    "thymus": "thymus",
    "vagina": "vagina",
    "cervix": "cervix",
    "lung": "lung",
    "pancreas": "pancreas",
    "pancreatic islets": "pancreatic islets",
    "prostate": "prostate",
    "seminal vesicles": "seminal vesicles",
    "coagulating gland": "coagulating gland",
}

ENDPOINT_SYNONYMS = {
    "blood creatinine": "creatinine",
    "creatinine": "creatinine",
    "urea nitrogen": "urea nitrogen",
    "blood urea nitrogen": "urea nitrogen",
    "urea (urea nitrogen)": "urea nitrogen",
    "total serum protein": "total protein",
    "total protein": "total protein",
    "fasting glucose": "glucose (fasting)",
    "glucose": "glucose",
    "fasting triglycerides": "triglycerides",
    "triglycerides": "triglycerides",
    "osmolality or specific gravity": "specific gravity or osmolality",
    "specific gravity or osmolality": "specific gravity or osmolality",
    "cholinesterase": "cholinesterase",
    "cholinesterase activity": "cholinesterase",
    "cholinesterase inhibition": "cholinesterase",
    "haematocrit": "haematocrit",
    "haematocrit (packed cell volume)": "haematocrit",
    "total leucocyte count": "total leukocyte count",
    "differential leucocyte count": "differential leukocyte count",
    "total and differential leukocyte count": "total and differential leukocyte count",
    "total leukocyte count": "total leukocyte count",
    "differential leukocyte count": "differential leukocyte count",
    "alanine aminotransferase (alt)": "alanine aminotransferase",
    "alanine aminotransferase": "alanine aminotransferase",
    "aspartate aminotransferase (ast)": "aspartate aminotransferase",
    "aspartate aminotransferase": "aspartate aminotransferase",
    "thyroxine (t4)": "thyroxine (t4)",
    "serum t4": "thyroxine (t4)",
    "circulating levels of t3, t4": "circulating t3 and t4",
    "circulating levels of tsh": "tsh",
    "thyroid stimulating hormone (tsh)": "tsh",
    "tsh": "tsh",
    "t3": "triiodothyronine (t3)",
    "triiodothyronine (t3)": "triiodothyronine (t3)",
    "serum t3": "triiodothyronine (t3)",
    "serum lh (luteinising hormone)": "lh",
    "lh": "lh",
    "serum fsh (follicular stimulating hormone)": "fsh",
    "fsh": "fsh",
    "oestradiol": "oestradiol",
    "testosterone": "testosterone",
    "serum testosterone (t)": "testosterone",
    "testis weight": "testes weight",
    "epididymis weight": "epididymides weight",
    "epididymis histopathology": "epididymides histopathology",
    "methaemoglobin": "methaemoglobin",
    "methaemoglobin or heinz bodies": "methaemoglobin or heinz bodies",
    "gross necropsy": "gross necropsy",
    "gross necropsy (macroscopic examination)": "gross necropsy",
    "gross pathology (complete necropsy)": "gross necropsy",
    "gross pathological changes": "gross pathological changes",
    "histopathology": "histopathology (full)",
    "histopathological examination": "histopathology (full)",
    "histopathological findings": "histopathological findings",
    "histopathological examination of uterus": "uterus histopathology",
    "histopathology (high dose and control)": "histopathology (full)",
    "total protein (urine)": "total protein (urine)",
    "glucose (urine)": "glucose (urine)",
    "appearance (colour and turbidity)": "appearance",
    "appearance": "appearance",
    "vaginal smears": "vaginal smears",
    "vaginal smear (collected at necropsy) to determine stage of oestrus cycle": "vaginal smears",
}

COLS_WITH_SOURCE = [
    "guideline_number",
    "category",
    "endpoint_name",
    "requirement_level",
    "source",
    "footnotes",
]
COLS_WITHOUT_SOURCE = [
    "guideline_number",
    "category",
    "endpoint_name",
    "requirement_level",
    "footnotes",
]

NORMALIZED_ROW_FIELDS = [
    "endpoint_id",
    "guideline_number",
    "original_category",
    "normalized_category",
    "original_endpoint",
    "normalized_endpoint",
    "requirement_level",
    "source",
    "footnotes",
]

UNIQUE_ENDPOINT_FIELDS = [
    "normalized_category",
    "normalized_endpoint",
    "guideline_numbers",
    "occurrence_count",
    "original_categories",
    "original_endpoints",
]

CATEGORY_NORMALIZATION_FIELDS = ["original_category", "normalized_category"]


def guideline_sort_key(path: Path) -> tuple[int, str]:
    """Sort OECD endpoint files by guideline number, with a name fallback."""
    match = re.search(r"OECD_(\d+)_endpoints", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (999999, path.name)


def discover_endpoint_csvs(endpoints_dir: Path = DEFAULT_ENDPOINTS_DIR) -> list[Path]:
    """Find all OECD endpoint CSV files."""
    return sorted(endpoints_dir.glob("OECD_*_endpoints*.csv"), key=guideline_sort_key)


def load_endpoint_csv(path: Path) -> list[dict[str, str]]:
    """Load an endpoint CSV, accepting files with or without a source column."""
    rows: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return rows

        has_source = len(header) == 6 and "source" in header[4].lower()
        columns = COLS_WITH_SOURCE if has_source else COLS_WITHOUT_SOURCE

        for raw_row in reader:
            if not raw_row or not raw_row[0].strip():
                continue

            padded = raw_row + [""] * max(0, len(columns) - len(raw_row))
            row = dict(zip(columns, padded))
            if not has_source:
                row["source"] = "Annex"
            rows.append(row)

    return rows


def normalize_category(original: str) -> str:
    """Map an original category name to the transfer-normalized category."""
    original = original.strip()
    return CATEGORY_MAP.get(original, original)


def reclassify_pathology_endpoints(normalized_category: str, endpoint_name: str) -> str:
    """Move pathology sub-endpoints that are really weights or histopathology."""
    name_lower = endpoint_name.lower().strip()
    if normalized_category != "Pathology":
        return normalized_category
    if name_lower.startswith("organ weight"):
        return "Organ Weights"
    if name_lower.startswith("histopath"):
        return "Histopathology"
    return normalized_category


def normalize_endpoint_name(name: str, normalized_category: str) -> str:
    """Normalize endpoint names using synonym tables and category suffixes."""
    name_lower = name.strip().lower()
    normalized = ENDPOINT_SYNONYMS.get(name_lower, ORGAN_SYNONYMS.get(name_lower, name_lower))

    # Remove text within parentheses
    normalized = re.sub(r'\(.*?\)', '', normalized)
    # Remove 'with 95% ci and slope' or similar
    normalized = re.sub(r'\bwith\s+\d+%\s+ci\s+and\s+slope\b', '', normalized)
    # Remove any standalone numbers with %
    normalized = re.sub(r'\d+%', '', normalized)
    # Clean up any leftover 'with  ci and slope' if present
    normalized = re.sub(r'\bwith\s+ci\s+and\s+slope\b', '', normalized)
    
    # Remove extra whitespace
    normalized = " ".join(normalized.split())

    suffix = CATEGORY_SUFFIX.get(normalized_category)
    if suffix and not normalized.endswith(suffix):
        measurement_words = {
            "concentration",
            "count",
            "histology",
            "histopathological",
            "histopathology",
            "index",
            "level",
            "quantification",
            "score",
            "smear",
            "smears",
            "weight",
        }
        if not set(normalized.split()) & measurement_words:
            normalized = f"{normalized} {suffix}"

    return normalized


def build_normalized_rows(csv_paths: Iterable[Path]) -> list[dict[str, str]]:
    """Build row-level normalized endpoint records."""
    rows: list[dict[str, str]] = []
    seq_counters: dict[str, int] = defaultdict(int)

    for csv_path in csv_paths:
        file_guideline_match = re.search(r"OECD_(\d+)_endpoints", csv_path.name)
        file_guideline = file_guideline_match.group(1) if file_guideline_match else ""

        for endpoint in load_endpoint_csv(csv_path):
            guideline = endpoint.get("guideline_number", "").strip() or file_guideline
            original_category = endpoint.get("category", "").strip()
            original_endpoint = endpoint.get("endpoint_name", "").strip()

            normalized_category = normalize_category(original_category)
            normalized_category = reclassify_pathology_endpoints(
                normalized_category,
                original_endpoint,
            )
            normalized_endpoint = normalize_endpoint_name(original_endpoint, normalized_category)

            seq_counters[guideline] += 1
            rows.append(
                {
                    "endpoint_id": f"EP_{guideline}_{seq_counters[guideline]:03d}",
                    "guideline_number": guideline,
                    "original_category": original_category,
                    "normalized_category": normalized_category,
                    "original_endpoint": original_endpoint,
                    "normalized_endpoint": normalized_endpoint,
                    "requirement_level": endpoint.get("requirement_level", "").strip(),
                    "source": endpoint.get("source", "").strip(),
                    "footnotes": endpoint.get("footnotes", "").strip(),
                }
            )

    return rows


def sort_guideline_numbers(values: set[str]) -> list[str]:
    """Sort guideline numbers numerically where possible."""
    return sorted(values, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))


def build_unique_endpoint_list(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Collapse row-level records to one row per normalized category/endpoint pair."""
    grouped: dict[tuple[str, str], dict[str, set[str] | int]] = {}

    for row in rows:
        key = (row["normalized_category"], row["normalized_endpoint"])
        if key not in grouped:
            grouped[key] = {
                "guideline_numbers": set(),
                "occurrence_count": 0,
                "original_categories": set(),
                "original_endpoints": set(),
            }

        grouped_row = grouped[key]
        grouped_row["guideline_numbers"].add(row["guideline_number"])  # type: ignore[union-attr]
        grouped_row["occurrence_count"] = int(grouped_row["occurrence_count"]) + 1
        grouped_row["original_categories"].add(row["original_category"])  # type: ignore[union-attr]
        grouped_row["original_endpoints"].add(row["original_endpoint"])  # type: ignore[union-attr]

    unique_rows = []
    for (category, endpoint), payload in grouped.items():
        unique_rows.append(
            {
                "normalized_category": category,
                "normalized_endpoint": endpoint,
                "guideline_numbers": "|".join(sort_guideline_numbers(payload["guideline_numbers"])),  # type: ignore[arg-type]
                "occurrence_count": str(payload["occurrence_count"]),
                "original_categories": "|".join(sorted(payload["original_categories"])),  # type: ignore[arg-type]
                "original_endpoints": "|".join(sorted(payload["original_endpoints"])),  # type: ignore[arg-type]
            }
        )

    return sorted(
        unique_rows,
        key=lambda row: (row["normalized_category"].lower(), row["normalized_endpoint"].lower()),
    )


def build_category_normalization(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Return one row per original-to-normalized category mapping."""
    category_pairs = {
        (row["original_category"], row["normalized_category"])
        for row in rows
        if row["original_category"]
    }
    return [
        {"original_category": original, "normalized_category": normalized}
        for original, normalized in sorted(category_pairs, key=lambda item: item[0].lower())
    ]


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> Path:
    """Write dictionaries to CSV, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
