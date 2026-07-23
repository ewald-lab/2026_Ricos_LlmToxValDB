#!/usr/bin/env python3
"""
analyze_unmapped_phrases.py

Analyzes and categorizes toxicological effect descriptions in ToxValDB that were unmapped
(i.e. received no ontology term selections during LLM validation).

Outputs summary metrics, category distributions, and example lists to disk.
"""

import argparse
import csv
import json
import re
from pathlib import Path


def classify_unmapped_phrase(phrase: str) -> str:
    p_lower = phrase.strip().lower()
    
    # 1. Non-biological / Administrative / Report Artifacts
    if not p_lower or p_lower in {"-", "na", "n/a", "hd", "ld", "md", "p0:", "f1:", "f2:"}:
        return "Administrative / Text Noise"
    if len(phrase) > 300:
        return "Study Metadata / Text Block Artifact"
        
    # 2. Negative or Normal Clinical Findings (explicit absence of toxicity)
    neg_keywords = [
        "no adverse", "no discharge", "no effect", "no significant", "no change",
        "normal", "unremarkable", "comparable to control", "no ophthalmic", "no macroscopic"
    ]
    if any(kw in p_lower for kw in neg_keywords):
        return "Negative / Normal Clinical Finding"
        
    # 3. Pharmacokinetic (PK) & Systemic Exposure Parameters
    pk_keywords = ["auc", "cmax", "tmax", "plasma level", "systemic exposure", "clearance", "half-life", "half life"]
    if any(kw in p_lower for kw in pk_keywords):
        return "Pharmacokinetic / Exposure Parameter"
        
    # 4. Highly Generic / Vague Phrase Fragments (lacking organ or directionality)
    vague_phrases = [
        "effect on organ weight", "altered organ weight", "effect on gross pathology",
        "effect on epithelium", "histopathology: nonneoplastic", "histopathology: non-neoplastic",
        "regeneration", "degeneration", "nodule", "fibrotic", "hyperplastic", "premature sacrifice",
        "clinical signs: lesion", "histopathology: non-neoplastic"
    ]
    if p_lower in vague_phrases or p_lower.startswith("effect on ") or p_lower == "histopathology: non-neoplastic" or p_lower == "histopathology: nonneoplastic":
        return "Generic / Vague Description"
        
    # 5. Specific Molecular / Gene Expression / Enzyme Assays (outside MP scope)
    mol_keywords = ["mrna", "gene expression", "activity", "phosphatase", "ceruloplasmin", "transaminase", "receptor", "interleukin", "glucosaminidase"]
    if any(kw in p_lower for kw in mol_keywords):
        return "Molecular / Enzyme Assay Endpoint"

    return "Other Unmapped Endpoint"


def analyze_unmapped(json_file: Path, output_dir: Path, ontology_name: str = "mp"):
    print(f"Loading reasoning JSON: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_phrases = len(data)
    unmapped_records = []
    
    category_counts = {
        "Negative / Normal Clinical Finding": 0,
        "Pharmacokinetic / Exposure Parameter": 0,
        "Generic / Vague Description": 0,
        "Administrative / Text Noise": 0,
        "Study Metadata / Text Block Artifact": 0,
        "Molecular / Enzyme Assay Endpoint": 0,
        "Other Unmapped Endpoint": 0,
    }
    
    categorized_examples = {cat: [] for cat in category_counts}

    for phrase, info in data.items():
        selected = info.get("selected", [])
        if not selected:
            cat = classify_unmapped_phrase(phrase)
            category_counts[cat] = category_counts.get(cat, 0) + 1
            categorized_examples[cat].append(phrase)
            unmapped_records.append({
                "phrase": phrase,
                "category": cat,
                "reasoning_steps": len(info.get("reasoning", []))
            })

    unmapped_count = len(unmapped_records)
    mapped_count = total_phrases - unmapped_count
    coverage_pct = (mapped_count / total_phrases) * 100 if total_phrases > 0 else 0
    unmapped_pct = (unmapped_count / total_phrases) * 100 if total_phrases > 0 else 0

    print("\n==========================================")
    print(f" UNMAPPED PHRASE ANALYSIS: {ontology_name.upper()}")
    print("==========================================")
    print(f"Total phrases analyzed: {total_phrases:,}")
    print(f"Mapped phrases       : {mapped_count:,} ({coverage_pct:.2f}%)")
    print(f"Unmapped phrases     : {unmapped_count:,} ({unmapped_pct:.2f}%)")
    print("------------------------------------------")
    print("Breakdown of Unmapped Categories:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / unmapped_count * 100) if unmapped_count > 0 else 0
        print(f"  - {cat:<38}: {count:5d} ({pct:5.1f}%)")
    print("==========================================\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV summary of unmapped phrases
    csv_file = output_dir / f"unmapped_phrases_{ontology_name}.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phrase", "category", "reasoning_steps"])
        writer.writeheader()
        writer.writerows(unmapped_records)
    print(f"Saved CSV: {csv_file}")

    # Save detailed summary text file with examples per category
    txt_file = output_dir / f"unmapped_phrases_summary_{ontology_name}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"UNMAPPED PHRASE QUALITATIVE SUMMARY ({ontology_name.upper()})\n")
        f.write(f"Total phrases: {total_phrases} | Mapped: {mapped_count} ({coverage_pct:.2f}%) | Unmapped: {unmapped_count} ({unmapped_pct:.2f}%)\n\n")
        
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"=== Category: {cat} ({count} phrases, {count/unmapped_count*100:.1f}%) ===\n")
            examples = categorized_examples[cat][:20]
            for i, ex in enumerate(examples, 1):
                clean_ex = ex.replace("\n", " ")[:120]
                f.write(f"  {i:2d}. {clean_ex}\n")
            f.write("\n")
    print(f"Saved summary report: {txt_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze unmapped phrases in ToxValDB ontology mappings.")
    parser.add_argument("--json_file", type=Path, required=True, help="Path to ToxValDB_Reasoning_{onto}_*.json file.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save unmapped analysis outputs.")
    parser.add_argument("--ontology", type=str, default="mp", help="Ontology name (e.g. mp, hp, aop, txpo, hpath).")
    args = parser.parse_args()

    analyze_unmapped(args.json_file, args.output_dir, args.ontology)


if __name__ == "__main__":
    main()
