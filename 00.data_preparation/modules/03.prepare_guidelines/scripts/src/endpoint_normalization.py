#!/usr/bin/env python3
"""Utilities for normalizing OECD guideline endpoint CSVs.

The guideline extraction step writes one CSV per guideline. This module turns
those rows into stable endpoint labels, removes rows outside the Section 4
health-effects scope, and emits audit rows for rejected or suspicious records.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, NamedTuple

MODULE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENDPOINTS_DIR = MODULE_ROOT / "outputs" / "endpoints"
DEFAULT_OUTPUT_DIR = MODULE_ROOT / "outputs" / "endpoints"

SECTION_4_MIN = 400
SECTION_4_MAX = 499

CATEGORY_ALIASES = {
    "BALF": "Bronchoalveolar Lavage",
    "BALF Analysis": "Bronchoalveolar Lavage",
    "Body Weights": "Clinical Observations",
    "Clinical Signs": "Clinical Observations",
    "Developmental/Pup Data": "Developmental",
    "Fetal Alterations (External)": "Developmental",
    "Fetal Alterations (Skeletal)": "Developmental",
    "Fetal Alterations (Soft Tissue)": "Developmental",
    "Food Consumption": "Clinical Observations",
    "Food/Water": "Clinical Observations",
    "General Toxicity": "Clinical Observations",
    "Gross Pathology": "Gross Pathology",
    "Hematology": "Haematology",
    "Hormones (Dams)": "Hormones",
    "Mortality/Morbidity": "Clinical Observations",
    "Nutrition": "Clinical Observations",
    "Pathology (Gross)": "Gross Pathology",
    "Physiological Measures": "Reproductive",
    "Reproductive Indices": "Reproductive",
    "Uterine Contents": "Developmental",
    "Uterine Weight": "Organ Weights",
    "Vaginal Status": "Developmental",
}

KNOWN_NORMALIZED_CATEGORIES = {
    "Aerosol Characterization",
    "Bronchoalveolar Lavage",
    "Chemical Analysis",
    "Clinical Biochemistry",
    "Clinical Observations",
    "Clinical Pathology",
    "Dermal Assessment",
    "Developmental",
    "Environmental Parameters",
    "Exposure Protocol",
    "Gross Pathology",
    "Haematology",
    "Histopathology",
    "Hormones",
    "Lethality Estimates",
    "Mortality",
    "Necropsy",
    "Ophthalmology",
    "Organ Weights",
    "Other",
    "Pathology",
    "Reproductive",
    "Statistical Analysis",
    "Urinalysis",
}

CATEGORY_SUFFIX = {
    "Organ Weights": "weight",
    "Histopathology": "histopathology",
}

ORGAN_SYNONYMS = {
    "adrenal": "adrenal glands",
    "adrenal glands": "adrenal glands",
    "adrenals": "adrenal glands",
    "brain": "brain",
    "cervix": "cervix",
    "coagulating gland": "coagulating gland",
    "epididymides": "epididymides",
    "epididymides (total and cauda)": "epididymides",
    "epididymis": "epididymides",
    "heart": "heart",
    "kidney": "kidneys",
    "kidneys": "kidneys",
    "liver": "liver",
    "lung": "lung",
    "male mammary glands": "male mammary glands",
    "mammary glands (female and male)": "mammary glands",
    "ovaries": "ovaries",
    "ovary": "ovaries",
    "paired adrenals": "adrenal glands",
    "paired kidneys": "kidneys",
    "pancreas": "pancreas",
    "pancreatic islets": "pancreatic islets",
    "pituitary": "pituitary",
    "pituitary gland": "pituitary",
    "prostate": "prostate",
    "seminal vesicles": "seminal vesicles",
    "spleen": "spleen",
    "testes": "testes",
    "testis": "testes",
    "thymus": "thymus",
    "thyroid": "thyroid",
    "thyroid and parathyroid glands": "thyroid and parathyroid glands",
    "thyroid gland": "thyroid",
    "uterus": "uterus",
    "uterus with cervix": "uterus with cervix",
    "uterus, including cervix": "uterus with cervix",
    "vagina": "vagina",
}

ENDPOINT_SYNONYMS = {
    "5'-nucleotidase": "5 nucleotidase",
    "alanine aminotransferase": "alanine aminotransferase",
    "alanine aminotransferase (alt)": "alanine aminotransferase",
    "alp": "alkaline phosphatase",
    "alt": "alanine aminotransferase",
    "alt (serum glutamic-pyruvic transaminase)": "alanine aminotransferase",
    "aspartate aminotransferase": "aspartate aminotransferase",
    "aspartate aminotransferase (ast)": "aspartate aminotransferase",
    "ast": "aspartate aminotransferase",
    "ast (serum glutamic-oxaloacetic transaminase)": "aspartate aminotransferase",
    "blood creatinine": "creatinine",
    "blood urea nitrogen": "urea nitrogen",
    "circulating levels of t3": "triiodothyronine",
    "circulating levels of t3, t4": "triiodothyronine and thyroxine",
    "circulating levels of t4": "thyroxine",
    "circulating levels of tsh": "thyroid stimulating hormone",
    "creatinine": "creatinine",
    "degree of np agglomeration": "degree of nanoparticle agglomeration",
    "differential leucocyte count": "differential leukocyte count",
    "fasting glucose": "glucose",
    "fasting triglycerides": "triglycerides",
    "fsh": "follicle stimulating hormone",
    "gamma glutamyl transpeptidase": "gamma glutamyl transferase",
    "glucose": "glucose",
    "glucose (urine)": "urine glucose",
    "haematocrit": "haematocrit",
    "haematocrit (packed cell volume)": "haematocrit",
    "hematocrit": "haematocrit",
    "hemoglobin concentration": "haemoglobin concentration",
    "hdl": "high density lipoprotein cholesterol",
    "lc50": "lethal concentration 50",
    "lc50 value": "lethal concentration 50",
    "ldl": "low density lipoprotein cholesterol",
    "lh": "luteinising hormone",
    "mean corpuscular haemoglobin": "mean corpuscular haemoglobin",
    "mean corpuscular haemoglobin concentration": "mean corpuscular haemoglobin concentration",
    "mean corpuscular hemoglobin": "mean corpuscular haemoglobin",
    "mean corpuscular hemoglobin concentration": "mean corpuscular haemoglobin concentration",
    "mean corpuscular volume": "mean corpuscular volume",
    "osmolality or specific gravity": "specific gravity or osmolality",
    "organ/body weight ratios": "organ to body weight ratios",
    "particle fraction <100 nm": "particle fraction under 100 nm",
    "ph": "ph",
    "serum fsh (follicular stimulating hormone)": "follicle stimulating hormone",
    "serum lh (luteinising hormone)": "luteinising hormone",
    "serum t3": "triiodothyronine",
    "serum t4": "thyroxine",
    "serum testosterone (t)": "testosterone",
    "specific gravity or osmolality": "specific gravity or osmolality",
    "t3": "triiodothyronine",
    "t4": "thyroxine",
    "thyroid stimulating hormone (tsh)": "thyroid stimulating hormone",
    "thyroxine (t4)": "thyroxine",
    "total leucocyte count": "total leukocyte count",
    "total protein": "total protein",
    "total protein (urine)": "urine protein",
    "total serum protein": "total protein",
    "triiodothyronine (t3)": "triiodothyronine",
    "tsh": "thyroid stimulating hormone",
    "urea (urea nitrogen)": "urea nitrogen",
    "urea nitrogen": "urea nitrogen",
    "vaginal smear (collected at necropsy) to determine stage of oestrus cycle": "vaginal smear",
    "vaginal smears": "vaginal smear",
}

COMPOSITE_ENDPOINT_SPLITS = {
    "birth index / viability": ["birth index", "viability index"],
    "body weight and changes": ["body weight", "body weight gain"],
    "clinical signs of toxicity/distress": ["clinical signs of toxicity", "clinical signs of distress"],
    "clonic or tonic movements": ["clonic movements", "tonic movements"],
    "death / morbidity": ["mortality", "morbidity"],
    "embryonic/fetal deaths": ["embryonic deaths", "fetal deaths"],
    "external alterations/malformations": ["external alterations", "external malformations"],
    "eye and mucous membrane condition": ["eye condition", "mucous membrane condition"],
    "eyes and mucous membranes": ["eye condition", "mucous membrane condition"],
    "eyes and mucous membranes changes": ["eye changes", "mucous membrane changes"],
    "eyes/mucous membranes changes": ["eye changes", "mucous membrane changes"],
    "gait and posture": ["gait", "posture"],
    "morbidity and mortality check": ["morbidity", "mortality"],
    "morbidity and mortality checks": ["morbidity", "mortality"],
    "mortality/moribundity": ["mortality", "moribundity"],
    "number and sex of pups": ["number of pups", "pup sex"],
    "oestrous cycle length/normality": ["oestrous cycle length", "oestrous cycle normality"],
    "osmolality or specific gravity": ["osmolality", "specific gravity"],
    "particle shape/size": ["particle shape", "particle size"],
    "respiratory/circulatory/autonomic/cns effects": [
        "respiratory effects",
        "circulatory effects",
        "autonomic effects",
        "central nervous system effects",
    ],
    "secretions and excretions": ["secretions", "excretions"],
    "skin and fur changes": ["skin changes", "fur changes"],
    "skin and fur condition": ["skin condition", "fur condition"],
    "skin/fur changes": ["skin changes", "fur changes"],
    "skin/fur/eyes/mucous membrane changes": ["skin changes", "fur changes", "eye changes", "mucous membrane changes"],
    "somatomotor activity and behaviour": ["somatomotor activity", "behaviour"],
    "specific gravity or osmolality": ["specific gravity", "osmolality"],
    "stillbirths/live births": ["stillbirths", "live births"],
    "time of death / moribund status": ["time of death", "moribund status"],
    "total and differential leucocyte count": ["total leukocyte count", "differential leukocyte count"],
    "total and differential leukocyte count": ["total leukocyte count", "differential leukocyte count"],
    "total protein or albumin": ["total protein", "albumin"],
    "wet weight and total length": ["wet weight", "total length"],
}

COMBINED_ANATOMY_REPLACEMENTS = {
    "prostate + seminal vesicle with coagulating glands": "prostate and seminal vesicle with coagulating glands",
    "prostate + seminal vesicles with coagulating glands": "prostate and seminal vesicles with coagulating glands",
}

GENERIC_REJECT_ENDPOINTS = {
    "absolute organ weights",
    "absolute organ weights weight",
    "adult intact male assay",
    "androgenic / anti-androgenic response",
    "androgenic/anti-androgenic response",
    "all gross lesions histopathology",
    "bronchoalveolar lavage fluid analysis",
    "clinical biochemistry measurements",
    "clinical biochemistry tests",
    "clinical observations",
    "clinical pathology data",
    "conjunctivae examination",
    "detailed clinical examination",
    "detailed clinical observations",
    "female pubertal assay",
    "full histopathology",
    "fundus examination",
    "general clinical observation",
    "general clinical observations",
    "gross necropsy",
    "gross necropsy findings",
    "gross necropsy findings histopathology",
    "gross pathology findings",
    "haematological determinations",
    "haematological tests",
    "histopathological examination",
    "histopathology",
    "histopathology full",
    "iris examination",
    "male pubertal assay",
    "morbidity monitoring",
    "mortality monitoring",
    "ophthalmological examination",
    "organ weight determination",
    "organ weights",
    "organ weights weight",
    "ovarian stub examination",
    "refractive media examination",
    "relative organ weights",
    "relative organ weights weight",
    "respiratory tract gross pathology histopathology",
    "reversibility assessment histopathology",
    "target organs histopathology",
    "target organs weight",
    "weanling reproductive organ histopathology",
}

GENERIC_REJECT_PATTERNS = [
    re.compile(r"\b(pubertal|intact male)\s+assay\b"),
    re.compile(r"\b(generic|general|detailed)\s+clinical\s+observations?\b"),
    re.compile(r"\b(generic|general|detailed)\s+clinical\s+examination\b"),
    re.compile(r"\bclinical\s+biochemistry\s+(tests?|measurements?|determinations?)\b"),
    re.compile(r"\bha?ematological\s+(tests?|determinations?)\b"),
    re.compile(r"\borgan\s+weights?\s+(determination|measurement)s?\b"),
    re.compile(r"\b(gross\s+necropsy|gross\s+pathology)(\s+findings?)?\b"),
    re.compile(r"\bmonitoring\b"),
]

MATRIX_CONTEXT_BY_CATEGORY = {
    "Bronchoalveolar Lavage": {
        "albumin": "balf albumin",
        "lactate dehydrogenase": "balf lactate dehydrogenase",
        "osmolality": "balf osmolality",
        "specific gravity": "balf specific gravity",
        "total protein": "balf total protein",
    },
    "Clinical Biochemistry": {
        "glucose": "blood glucose",
        "total protein": "serum total protein",
    },
    "Urinalysis": {
        "appearance": "urine appearance",
        "glucose": "urine glucose",
        "osmolality": "urine osmolality",
        "ph": "urine ph",
        "protein": "urine protein",
        "specific gravity": "urine specific gravity",
        "total protein": "urine protein",
        "volume": "urine volume",
    },
}

PREFERRED_CATEGORY_BY_ENDPOINT = {
    "behaviour": "Clinical Observations",
    "body weight": "Clinical Observations",
    "body weight gain": "Clinical Observations",
    "fertility index": "Reproductive",
    "food consumption": "Clinical Observations",
    "gestation length": "Reproductive",
    "litter size": "Developmental",
    "mating index": "Reproductive",
    "morbidity": "Clinical Observations",
    "mortality": "Clinical Observations",
    "somatomotor activity": "Clinical Observations",
    "water consumption": "Clinical Observations",
}

UNEXPANDED_ACRONYM_RE = re.compile(r"\b(alt|ast|hdl|ldl|fsh|lh|tsh|t3|t4|lc50|np)\b")
OPERATOR_ARTIFACT_RE = re.compile(r"[/+<>]|\b(or)\b")
METHOD_WORD_RE = re.compile(
    r"\b(analysis|assay|determination|determinations|examination|findings|monitoring|necropsy|observations|tests)\b"
)

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
    "requirement_level",
    "guideline_numbers",
    "occurrence_count",
    "original_categories",
    "original_endpoints",
]

ALL_ENDPOINT_FIELDS = [
    "normalized_category",
    "normalized_endpoint",
    "requirement_level",
    "guideline_numbers",
    "occurrence_count",
    "original_categories",
    "original_endpoints",
]

CATEGORY_NORMALIZATION_FIELDS = ["original_category", "normalized_category"]
EXCLUDED_ENDPOINT_FIELDS = NORMALIZED_ROW_FIELDS + ["exclusion_reason"]
QC_REPORT_FIELDS = [
    "endpoint_id",
    "guideline_number",
    "normalized_category",
    "normalized_endpoint",
    "issue_type",
    "issue_detail",
    "suggested_action",
    "original_categories",
    "original_endpoints",
]


class NormalizationResult(NamedTuple):
    rows: list[dict[str, str]]
    excluded_rows: list[dict[str, str]]
    qc_rows: list[dict[str, str]]


def load_category_map() -> dict[str, str]:
    """Load the category normalization map from the configuration CSV file."""
    map_path = MODULE_ROOT / "config" / "category_normalization.csv"
    mapping = {}
    if map_path.exists():
        with open(map_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                original = row.get("original_category", "").strip()
                normalized = row.get("normalized_category", "").strip()
                if original and original not in mapping:
                    mapping[original] = normalized
    return mapping


CATEGORY_MAP = load_category_map()


def guideline_sort_key(path: Path) -> tuple[int, str]:
    """Generate a stable sort key for OECD endpoint files."""
    match = re.search(r"OECD_(\d+)_endpoints", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (999999, path.name)


def discover_endpoint_csvs(endpoints_dir: Path = DEFAULT_ENDPOINTS_DIR) -> list[Path]:
    """Find all OECD endpoint CSV files in the specified directory."""
    return sorted(endpoints_dir.glob("OECD_*_endpoints*.csv"), key=guideline_sort_key)


def load_endpoint_csv(path: Path) -> list[dict[str, str]]:
    """Load an endpoint CSV file, accepting files with or without a source column."""
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

            row = coerce_endpoint_row(raw_row, columns, has_source)
            if not has_source:
                row["source"] = "Annex"
            rows.append(row)

    return rows


def coerce_endpoint_row(raw_row: list[str], columns: list[str], has_source: bool) -> dict[str, str]:
    """Coerce malformed CSV rows where endpoint or footnote text has unquoted commas."""
    if len(raw_row) == len(columns):
        return dict(zip(columns, raw_row))

    level_index = None
    for index, value in enumerate(raw_row[2:], start=2):
        if value.strip() in {"Required", "Optional"}:
            level_index = index
            break

    if level_index is None:
        padded = raw_row + [""] * max(0, len(columns) - len(raw_row))
        return dict(zip(columns, padded))

    row = {
        "guideline_number": raw_row[0],
        "category": raw_row[1] if len(raw_row) > 1 else "",
        "endpoint_name": ",".join(raw_row[2:level_index]).strip(),
        "requirement_level": raw_row[level_index].strip(),
    }
    if has_source:
        row["source"] = raw_row[level_index + 1].strip() if len(raw_row) > level_index + 1 else ""
        row["footnotes"] = ",".join(raw_row[level_index + 2 :]).strip()
    else:
        row["footnotes"] = ",".join(raw_row[level_index + 1 :]).strip()
    return row


def normalize_category(original: str) -> str:
    """Map an original category name to the normalized category name."""
    original = original.strip()
    return CATEGORY_ALIASES.get(original, CATEGORY_MAP.get(original, original))


def is_section_4_guideline(guideline: str) -> bool:
    """Return True when a guideline number is in OECD Section 4 health effects."""
    if not guideline.isdigit():
        return False
    guideline_number = int(guideline)
    return SECTION_4_MIN <= guideline_number <= SECTION_4_MAX


def reclassify_pathology_endpoints(normalized_category: str, endpoint_name: str) -> str:
    """Reclassify Pathology sub-endpoints that are really weights or histopathology."""
    name_lower = endpoint_name.lower().strip()
    if normalized_category != "Pathology":
        return normalized_category
    if name_lower.startswith("organ weight"):
        return "Organ Weights"
    if name_lower.startswith("histopath"):
        return "Histopathology"
    return normalized_category


def strip_parenthetical_text(value: str) -> str:
    """Remove parenthetical qualifiers after exact synonym matching has had a chance."""
    return re.sub(r"\(.*?\)", "", value)


def clean_endpoint_text(name: str) -> str:
    """Lowercase and remove extraction-only qualifiers from an endpoint label."""
    value = " ".join(name.strip().lower().split())
    value = value.replace("anti androgenic", "anti-androgenic")
    value = ENDPOINT_SYNONYMS.get(value, ORGAN_SYNONYMS.get(value, value))
    value = strip_parenthetical_text(value)
    value = re.sub(r"\bwith\s+\d+%\s+ci\s+and\s+slope\b", "", value)
    value = re.sub(r"\bwith\s+\d+%\s+confidence\s+limits?\b", "", value)
    value = re.sub(r"\d+%", "", value)
    value = re.sub(r"\bwith\s+ci\s+and\s+slope\b", "", value)
    value = value.replace("+", " + ")
    value = value.replace("/", " / ")
    value = " ".join(value.split())
    value = value.strip(" ,;-")
    return ENDPOINT_SYNONYMS.get(value, ORGAN_SYNONYMS.get(value, value))


def split_composite_endpoint(name: str) -> list[str]:
    """Split known composite labels into individually measurable endpoints."""
    if name in COMPOSITE_ENDPOINT_SPLITS:
        return COMPOSITE_ENDPOINT_SPLITS[name]

    slash_compact = name.replace(" / ", "/")
    if slash_compact in COMPOSITE_ENDPOINT_SPLITS:
        return COMPOSITE_ENDPOINT_SPLITS[slash_compact]

    return [name]


def canonicalize_combined_anatomy(name: str) -> str:
    """Keep OECD-defined combined anatomy endpoints, but remove symbolic operators."""
    return COMBINED_ANATOMY_REPLACEMENTS.get(name, name)


def apply_category_context(name: str, normalized_category: str) -> str:
    """Add matrix context only where the same endpoint string would otherwise collide."""
    return MATRIX_CONTEXT_BY_CATEGORY.get(normalized_category, {}).get(name, name)


def apply_category_suffix(name: str, normalized_category: str) -> str:
    """Add measurement suffixes to tissue labels for organ weights and histopathology."""
    suffix = CATEGORY_SUFFIX.get(normalized_category)
    if not suffix or name.endswith(suffix):
        return name

    measurement_words = {
        "concentration",
        "count",
        "histology",
        "histopathological",
        "histopathology",
        "index",
        "level",
        "levels",
        "mass",
        "quantification",
        "ratio",
        "ratios",
        "score",
        "smear",
        "smears",
        "status",
        "weight",
        "weights",
    }
    if set(name.split()) & measurement_words:
        return name
    return f"{name} {suffix}"


def normalize_endpoint_names(name: str, normalized_category: str) -> list[str]:
    """Normalize, split, contextualize, and suffix an endpoint label."""
    cleaned = clean_endpoint_text(name)
    cleaned = canonicalize_combined_anatomy(cleaned)

    normalized_names = []
    for part in split_composite_endpoint(cleaned):
        part = clean_endpoint_text(part)
        part = canonicalize_combined_anatomy(part)
        part = apply_category_context(part, normalized_category)
        part = apply_category_suffix(part, normalized_category)
        normalized_names.append(part)

    return normalized_names


def rejection_reason(guideline: str, normalized_endpoint: str) -> str:
    """Return a rejection reason, or an empty string when the endpoint is accepted."""
    if not is_section_4_guideline(guideline):
        return "outside_oecd_section_4"

    if normalized_endpoint in GENERIC_REJECT_ENDPOINTS:
        return "method_or_activity_level_endpoint"

    for pattern in GENERIC_REJECT_PATTERNS:
        if pattern.search(normalized_endpoint):
            return "method_or_activity_level_endpoint"

    if normalized_endpoint in {"", "none", "n/a"}:
        return "empty_endpoint"

    return ""


def make_row(
    endpoint_id: str,
    guideline: str,
    original_category: str,
    normalized_category: str,
    original_endpoint: str,
    normalized_endpoint: str,
    endpoint: dict[str, str],
) -> dict[str, str]:
    """Create a normalized row with stable field names."""
    return {
        "endpoint_id": endpoint_id,
        "guideline_number": guideline,
        "original_category": original_category,
        "normalized_category": normalized_category,
        "original_endpoint": original_endpoint,
        "normalized_endpoint": normalized_endpoint,
        "requirement_level": endpoint.get("requirement_level", "").strip(),
        "source": endpoint.get("source", "").strip(),
        "footnotes": endpoint.get("footnotes", "").strip(),
    }


def build_normalization_result(csv_paths: Iterable[Path]) -> NormalizationResult:
    """Build accepted endpoint records plus exclusion and QC audit rows."""
    rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []
    seq_counters: dict[str, int] = defaultdict(int)

    for csv_path in csv_paths:
        file_guideline_match = re.search(r"OECD_(\d+)_endpoints", csv_path.name)
        file_guideline = file_guideline_match.group(1) if file_guideline_match else ""

        for endpoint in load_endpoint_csv(csv_path):
            guideline = endpoint.get("guideline_number", "").strip() or file_guideline
            original_category = endpoint.get("category", "").strip()
            original_endpoint = endpoint.get("endpoint_name", "").strip()
            normalized_category = normalize_category(original_category)
            normalized_category = reclassify_pathology_endpoints(normalized_category, original_endpoint)
            normalized_endpoints = normalize_endpoint_names(original_endpoint, normalized_category)

            if not normalized_endpoints:
                normalized_endpoints = [""]

            for normalized_endpoint in normalized_endpoints:
                seq_counters[guideline] += 1
                row = make_row(
                    f"EP_{guideline}_{seq_counters[guideline]:03d}",
                    guideline,
                    original_category,
                    normalized_category,
                    original_endpoint,
                    normalized_endpoint,
                    endpoint,
                )
                reason = rejection_reason(guideline, normalized_endpoint)
                if reason:
                    excluded_rows.append({**row, "exclusion_reason": reason})
                else:
                    rows.append(row)

    return NormalizationResult(rows, excluded_rows, build_qc_rows(rows, excluded_rows))


def build_normalized_rows(csv_paths: Iterable[Path]) -> list[dict[str, str]]:
    """Build row-level normalized endpoint records from a collection of CSV paths."""
    return build_normalization_result(csv_paths).rows


def sort_guideline_numbers(values: set[str]) -> list[str]:
    """Sort guideline numbers numerically where possible."""
    return sorted(values, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))


def choose_group_category(endpoint: str, categories: Iterable[str]) -> str:
    """Choose one category for a globally unique normalized endpoint string."""
    category_counts = Counter(categories)
    preferred = PREFERRED_CATEGORY_BY_ENDPOINT.get(endpoint)
    if preferred and preferred in category_counts:
        return preferred
    return sorted(category_counts.items(), key=lambda item: (-item[1], item[0].lower()))[0][0]


def build_endpoint_list(rows: Iterable[dict[str, str]], include_optional: bool) -> list[dict[str, str]]:
    """Collapse row-level records to one globally unique row per normalized endpoint."""
    grouped: dict[str, dict[str, set[str] | int]] = {}

    for row in rows:
        level = row.get("requirement_level", "").strip()
        if not include_optional and level.lower() != "required":
            continue

        endpoint = row["normalized_endpoint"]
        if endpoint not in grouped:
            grouped[endpoint] = {
                "guideline_numbers": set(),
                "occurrence_count": 0,
                "original_categories": set(),
                "normalized_categories": set(),
                "original_endpoints": set(),
                "requirement_levels": set(),
            }

        grouped_row = grouped[endpoint]
        grouped_row["guideline_numbers"].add(row["guideline_number"])  # type: ignore[union-attr]
        grouped_row["occurrence_count"] = int(grouped_row["occurrence_count"]) + 1
        grouped_row["original_categories"].add(row["original_category"])  # type: ignore[union-attr]
        grouped_row["normalized_categories"].add(row["normalized_category"])  # type: ignore[union-attr]
        grouped_row["original_endpoints"].add(row["original_endpoint"])  # type: ignore[union-attr]
        if level:
            grouped_row["requirement_levels"].add(level)  # type: ignore[union-attr]

    output_rows = []
    for endpoint, payload in grouped.items():
        levels = payload["requirement_levels"]
        if not include_optional:
            requirement_level = "Required"
        elif "Required" in levels and "Optional" in levels:  # type: ignore[operator]
            requirement_level = "Required/Optional"
        elif "Required" in levels:  # type: ignore[operator]
            requirement_level = "Required"
        elif "Optional" in levels:  # type: ignore[operator]
            requirement_level = "Optional"
        else:
            requirement_level = "|".join(sorted(levels))  # type: ignore[arg-type]

        output_rows.append(
            {
                "normalized_category": choose_group_category(endpoint, payload["normalized_categories"]),  # type: ignore[arg-type]
                "normalized_endpoint": endpoint,
                "requirement_level": requirement_level,
                "guideline_numbers": "|".join(sort_guideline_numbers(payload["guideline_numbers"])),  # type: ignore[arg-type]
                "occurrence_count": str(payload["occurrence_count"]),
                "original_categories": "|".join(sorted(payload["original_categories"])),  # type: ignore[arg-type]
                "original_endpoints": "|".join(sorted(payload["original_endpoints"])),  # type: ignore[arg-type]
            }
        )

    return sorted(
        output_rows,
        key=lambda row: (row["normalized_category"].lower(), row["normalized_endpoint"].lower()),
    )


def build_unique_endpoint_list(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Collapse row-level records to one row per required normalized endpoint."""
    return build_endpoint_list(rows, include_optional=False)


def build_all_endpoint_list(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Collapse row-level records to one row per required or optional normalized endpoint."""
    return build_endpoint_list(rows, include_optional=True)


def build_category_normalization(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Generate one row per original-to-normalized category mapping."""
    category_pairs = {
        (row["original_category"], row["normalized_category"])
        for row in rows
        if row["original_category"]
    }
    return [
        {"original_category": original, "normalized_category": normalized}
        for original, normalized in sorted(category_pairs, key=lambda item: item[0].lower())
    ]


def make_qc_row(row: dict[str, str], issue_type: str, issue_detail: str, suggested_action: str) -> dict[str, str]:
    """Create a normalization QC audit row."""
    return {
        "endpoint_id": row.get("endpoint_id", ""),
        "guideline_number": row.get("guideline_number", ""),
        "normalized_category": row.get("normalized_category", ""),
        "normalized_endpoint": row.get("normalized_endpoint", ""),
        "issue_type": issue_type,
        "issue_detail": issue_detail,
        "suggested_action": suggested_action,
        "original_categories": row.get("original_category", ""),
        "original_endpoints": row.get("original_endpoint", ""),
    }


def build_qc_rows(rows: list[dict[str, str]], excluded_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Flag unresolved normalization issues and document automatic duplicate merges."""
    qc_rows: list[dict[str, str]] = []

    for row in rows:
        endpoint = row["normalized_endpoint"]
        category = row["normalized_category"]

        if category not in KNOWN_NORMALIZED_CATEGORIES:
            qc_rows.append(
                make_qc_row(
                    row,
                    "unknown_category",
                    f"Category '{category}' is not in the normalization allow-list.",
                    "Add a category alias or confirm this is a new accepted category.",
                )
            )

        if UNEXPANDED_ACRONYM_RE.search(endpoint):
            qc_rows.append(
                make_qc_row(
                    row,
                    "unexpanded_acronym",
                    "Endpoint still contains a short acronym that should be expanded.",
                    "Add or update ENDPOINT_SYNONYMS.",
                )
            )

        if OPERATOR_ARTIFACT_RE.search(endpoint):
            qc_rows.append(
                make_qc_row(
                    row,
                    "operator_artifact",
                    "Endpoint still contains a slash, symbolic operator, or unresolved 'or'.",
                    "Add a composite split or combined-anatomy replacement.",
                )
            )

        if METHOD_WORD_RE.search(endpoint) and endpoint not in GENERIC_REJECT_ENDPOINTS:
            qc_rows.append(
                make_qc_row(
                    row,
                    "method_word",
                    "Endpoint contains a word commonly associated with methods or activities.",
                    "Confirm it is a parameter-level endpoint or add a rejection rule.",
                )
            )

    endpoint_to_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        endpoint_to_rows[row["normalized_endpoint"]].append(row)

    for endpoint, endpoint_rows in endpoint_to_rows.items():
        if len(endpoint_rows) <= 1:
            continue
        categories = sorted({row["normalized_category"] for row in endpoint_rows})
        original_endpoints = sorted({row["original_endpoint"] for row in endpoint_rows})
        qc_rows.append(
            {
                "endpoint_id": "|".join(row["endpoint_id"] for row in endpoint_rows),
                "guideline_number": "|".join(sort_guideline_numbers({row["guideline_number"] for row in endpoint_rows})),
                "normalized_category": choose_group_category(endpoint, categories),
                "normalized_endpoint": endpoint,
                "issue_type": "merged_duplicate_endpoint",
                "issue_detail": f"Merged from categories: {'|'.join(categories)}",
                "suggested_action": "Review only if this should be context-disambiguated rather than merged.",
                "original_categories": "|".join(sorted({row["original_category"] for row in endpoint_rows})),
                "original_endpoints": "|".join(original_endpoints),
            }
        )

    for row in excluded_rows:
        qc_rows.append(
            make_qc_row(
                row,
                "excluded_endpoint",
                row.get("exclusion_reason", ""),
                "Review excluded_endpoint_list.csv if this row should be rescued.",
            )
        )

    return sorted(
        qc_rows,
        key=lambda row: (
            row["issue_type"].lower(),
            row["normalized_category"].lower(),
            row["normalized_endpoint"].lower(),
            row["endpoint_id"],
        ),
    )


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> Path:
    """Write dictionaries to a CSV file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
