from __future__ import annotations

import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from src.endpoint_normalization import (  # noqa: E402
    build_normalization_result,
    build_unique_endpoint_list,
)


def write_endpoint_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "guideline_number",
        "category",
        "endpoint_name",
        "requirement_level",
        "source",
        "footnotes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def endpoint_rows(*rows: tuple[str, str, str]) -> list[dict[str, str]]:
    return [
        {
            "guideline_number": guideline,
            "category": category,
            "endpoint_name": endpoint,
            "requirement_level": "Required",
            "source": "Guideline text",
            "footnotes": "",
        }
        for guideline, category, endpoint in rows
    ]


def test_rejects_non_section4_and_method_level_rows(tmp_path: Path) -> None:
    path = tmp_path / "OECD_410_endpoints.csv"
    write_endpoint_csv(
        path,
        endpoint_rows(
            ("203", "Chemical Analysis", "Test chemical concentration"),
            ("408", "Clinical Biochemistry", "Clinical biochemistry tests"),
            ("441", "Assay Models", "Adult intact male assay (Hershberger)"),
            ("423", "Gross Pathology", "Gross necropsy findings"),
            ("452", "Clinical Biochemistry", "ALT"),
        ),
    )

    result = build_normalization_result([path])

    accepted = {row["normalized_endpoint"] for row in result.rows}
    excluded = {row["normalized_endpoint"]: row["exclusion_reason"] for row in result.excluded_rows}

    assert accepted == {"alanine aminotransferase"}
    assert excluded["test chemical concentration"] == "outside_oecd_section_4"
    assert excluded["clinical biochemistry tests"] == "method_or_activity_level_endpoint"
    assert excluded["adult intact male assay"] == "method_or_activity_level_endpoint"
    assert excluded["gross necropsy findings"] == "method_or_activity_level_endpoint"


def test_canonicalizes_abbreviations_spelling_context_and_composites(tmp_path: Path) -> None:
    path = tmp_path / "OECD_452_endpoints.csv"
    write_endpoint_csv(
        path,
        endpoint_rows(
            ("452", "Clinical Biochemistry", "ALT (Serum glutamic-pyruvic transaminase)"),
            ("452", "Clinical Biochemistry", "AST (Serum glutamic-oxaloacetic transaminase)"),
            ("452", "Clinical Biochemistry", "HDL"),
            ("452", "Clinical Biochemistry", "LDL"),
            ("452", "Clinical Biochemistry", "Glucose"),
            ("452", "Hematology", "Hemoglobin concentration"),
            ("452", "Hormones", "TSH"),
            ("452", "Urinalysis", "pH"),
            ("452", "Urinalysis", "Glucose"),
            ("452", "Urinalysis", "Total protein"),
            ("452", "Urinalysis", "Osmolality or specific gravity"),
        ),
    )

    result = build_normalization_result([path])
    endpoints = {row["normalized_endpoint"] for row in result.rows}
    categories = {row["normalized_endpoint"]: row["normalized_category"] for row in result.rows}

    assert "alanine aminotransferase" in endpoints
    assert "aspartate aminotransferase" in endpoints
    assert "high density lipoprotein cholesterol" in endpoints
    assert "low density lipoprotein cholesterol" in endpoints
    assert "blood glucose" in endpoints
    assert "haemoglobin concentration" in endpoints
    assert "thyroid stimulating hormone" in endpoints
    assert "urine ph" in endpoints
    assert "urine glucose" in endpoints
    assert "urine protein" in endpoints
    assert "urine osmolality" in endpoints
    assert "urine specific gravity" in endpoints
    assert categories["haemoglobin concentration"] == "Haematology"


def test_context_aware_duplicate_resolution(tmp_path: Path) -> None:
    path = tmp_path / "OECD_416_endpoints.csv"
    write_endpoint_csv(
        path,
        endpoint_rows(
            ("416", "Clinical Observations", "Food consumption"),
            ("416", "General Toxicity", "Food consumption"),
            ("416", "Reproductive", "Fertility index"),
            ("416", "Reproductive Indices", "Fertility index"),
            ("416", "Clinical Biochemistry", "Glucose"),
            ("416", "Urinalysis", "Glucose"),
            ("416", "Clinical Biochemistry", "Total protein"),
            ("416", "Urinalysis", "Total protein"),
        ),
    )

    result = build_normalization_result([path])
    unique_rows = build_unique_endpoint_list(result.rows)
    unique_endpoints = [row["normalized_endpoint"] for row in unique_rows]

    assert unique_endpoints.count("food consumption") == 1
    assert unique_endpoints.count("fertility index") == 1
    assert "blood glucose" in unique_endpoints
    assert "urine glucose" in unique_endpoints
    assert "serum total protein" in unique_endpoints
    assert "urine protein" in unique_endpoints
    assert len(unique_endpoints) == len(set(unique_endpoints))


def test_keeps_target_specific_ophthalmology_examinations(tmp_path: Path) -> None:
    path = tmp_path / "OECD_412_endpoints.csv"
    write_endpoint_csv(
        path,
        endpoint_rows(
            ("412", "Ophthalmology", "Iris examination"),
            ("412", "Ophthalmology", "Conjunctivae examination"),
            ("412", "Ophthalmology", "Fundus examination"),
            ("412", "Ophthalmology", "Refractive media examination"),
            ("452", "Ophthalmology", "Ophthalmological examination"),
        ),
    )

    result = build_normalization_result([path])
    accepted = {row["normalized_endpoint"] for row in result.rows}
    excluded = {row["normalized_endpoint"]: row["exclusion_reason"] for row in result.excluded_rows}

    assert "iris morphology" in accepted
    assert "conjunctiva morphology" in accepted
    assert "fundus morphology" in accepted
    assert "refractive media transparency" in accepted
    assert excluded["ophthalmological examination"] == "method_or_activity_level_endpoint"
