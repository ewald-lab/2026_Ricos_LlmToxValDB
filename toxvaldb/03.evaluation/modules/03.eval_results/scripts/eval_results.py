#!/usr/bin/env python3
"""
Evaluate results: generate all analysis plots + ontology tree PNG.

PART A – Depth Analysis (from CSV):
  - Distribution of YEAR vs max ontology depth of ontology assignments
  - Statistics on assignments per depth level

PART B – Phenotypic Landscape (from JSON):
  - Treemap of unique phrase distribution across top-level ontology branches

PART C – Ontology Tree (from JSON + OWL):
  - Propagates phrase counts through the ontology DAG
  - Prunes nodes below --threshold
  - Saves a single PNG (no .dot or .pdf)

Usage:
    python eval_results.py \
        --input_file  <ToxValDB_Classified_mp_*.csv> \
        --json_file   <ToxValDB_Reasoning_mp_*.json> \
        --owl_file    <mp.ttl> \
        --output_dir  <outputs/> \
        [--root_term  "mammalian phenotype"] \
        [--threshold  50]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
os.environ["POLARS_MAX_THREADS"] = "4"
import sys
import traceback
from pathlib import Path

# Ensure user site-packages are available (e.g. squarify)
sys.path.append(os.path.expanduser("~/.local/lib/python3.12/site-packages"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager
import networkx as nx
import numpy as np
import polars as pl
import rdflib
import seaborn as sns
from rdflib.namespace import RDFS

try:
    import squarify
    HAS_SQUARIFY = True
except ImportError:
    HAS_SQUARIFY = False
    print("WARNING: squarify not installed – treemap will be skipped.", file=sys.stderr)

# Keep a real print for progress messages; suppress the default one inside helpers
_print = print
print = lambda *a, **kw: None  # noqa: E731


def configure_fonts() -> None:
    font_dir = Path(__file__).resolve().parents[4] / "02.expert_annotation" / "modules" / "01.create_form" / "assets" / "fonts" / "ibm-plex-sans"
    for font_path in font_dir.glob("*.ttf"):
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"],
    })


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_file", required=True, type=Path,
                   help="Classified CSV (ToxValDB_Classified_mp_*.csv)")
    p.add_argument("--json_file", required=True, type=Path,
                   help="Reasoning JSON (ToxValDB_Reasoning_mp_*.json)")
    p.add_argument("--missing_tsv", type=Path, default=None,
                   help="Optional per-column missing-%% TSV")
    p.add_argument("--owl_file", required=True, type=Path,
                   help="MP ontology TTL file")
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--root_term", default="mammalian phenotype")
    p.add_argument("--min_year", type=int, default=None)
    p.add_argument("--max_year", type=int, default=None)
    p.add_argument("--threshold", type=int, default=500,
                   help="Min propagated count to keep a node in the ontology tree")
    p.add_argument("--depth_only", action="store_true",
                   help="Generate only CSV/depth plots, then exit before JSON-based plots.")
    return p.parse_args()


# ── Ontology helpers ──────────────────────────────────────────────────────────

def build_ontology_graph(owl_file: Path) -> tuple[nx.DiGraph, dict, dict]:
    _print(f"Parsing ontology: {owl_file}", flush=True)
    g = rdflib.Graph()
    g.parse(str(owl_file), format="turtle")

    G = nx.DiGraph()          # Edge: child → parent  (subClassOf)
    name_to_uri: dict = {}
    uri_to_name: dict = {}

    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, rdflib.term.URIRef) and isinstance(o, rdflib.term.URIRef):
            G.add_edge(s, o)
    for s, p, o in g.triples((None, RDFS.label, None)):
        if not isinstance(s, rdflib.term.URIRef):
            continue
        lang = getattr(o, "language", None)
        if lang is not None and lang != "en":
            continue
        name = str(o).lower().strip()
        name_to_uri[name] = s
        uri_to_name[s] = name

    for node in G.nodes():
        if node not in uri_to_name:
            fallback_name = str(node).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            fallback_name = fallback_name.replace("_", " ").lower().strip()
            if fallback_name:
                uri_to_name[node] = fallback_name
                name_to_uri.setdefault(fallback_name, node)

    return G, name_to_uri, uri_to_name


def ensure_root_node(G: nx.DiGraph, name_to_uri: dict,
                     uri_to_name: dict, root_term: str):
    root_name = root_term.lower().strip()
    root_uri = name_to_uri.get(root_name)
    if root_uri is not None:
        return root_uri

    root_uri = rdflib.URIRef(f"urn:synthetic-root:{root_name.replace(' ', '_')}")
    roots = [node for node in G.nodes() if G.out_degree(node) == 0]
    if len(roots) == 1 and uri_to_name.get(roots[0], "").lower() == "thing":
        roots = list(G.predecessors(roots[0]))
    for node in roots:
        G.add_edge(node, root_uri)

    name_to_uri[root_name] = root_uri
    uri_to_name[root_uri] = root_name
    _print(
        f"WARNING: root term '{root_term}' not found; created synthetic root "
        f"covering {len(roots)} top-level nodes.",
        file=sys.stderr,
        flush=True,
    )
    return root_uri


def build_depth_map(G: nx.DiGraph, name_to_uri: dict,
                    uri_to_name: dict, root_term: str) -> dict[str, int]:
    depth_map: dict = {}
    root_uri = ensure_root_node(G, name_to_uri, uri_to_name, root_term)
    G_rev = G.reverse()
    for node, d in nx.single_source_shortest_path_length(G_rev, root_uri).items():
        if node in uri_to_name:
            depth_map[uri_to_name[node]] = d
    _print(f"  Depth map: {len(depth_map):,} terms", flush=True)
    return depth_map


def map_to_top_branches(G: nx.DiGraph, name_to_uri: dict,
                        uri_to_name: dict, root_term: str) -> dict[str, str]:
    root_uri = ensure_root_node(G, name_to_uri, uri_to_name, root_term)
    top_branches = list(G.predecessors(root_uri))
    top_branch_names = {n: uri_to_name.get(n, str(n)) for n in top_branches}

    G_rev = G.reverse()
    term_to_top_branch: dict = {}
    for branch_uri, branch_name in top_branch_names.items():
        descendants = nx.descendants(G_rev, branch_uri)
        descendants.add(branch_uri)
        for desc in descendants:
            desc_name = uri_to_name.get(desc)
            if desc_name and desc_name not in term_to_top_branch:
                term_to_top_branch[desc_name] = branch_name
    return term_to_top_branch


# ── PART A: Depth analysis ────────────────────────────────────────────────────

METADATA_COLS = {
    "DTXSID", "CASRN", "NAME", "SOURCE", "SUB_SOURCE", "TOXVAL_TYPE",
    "TOXVAL_SUBTYPE", "TOXVAL_TYPE_SUPERCATEGORY", "QUALIFIER",
    "TOXVAL_NUMERIC", "TOXVAL_UNITS", "RISK_ASSESSMENT_CLASS", "STUDY_TYPE",
    "STUDY_DURATION_CLASS", "STUDY_DURATION_VALUE", "STUDY_DURATION_UNITS",
    "SPECIES_COMMON", "STRAIN", "LATIN_NAME", "SPECIES_SUPERCATEGORY",
    "SEX", "GENERATION", "LIFESTAGE", "EXPOSURE_ROUTE", "EXPOSURE_METHOD",
    "EXPOSURE_FORM", "MEDIA", "TOXICOLOGICAL_EFFECT",
    "TOXICOLOGICAL_EFFECT_CATEGORY", "STUDY_GROUP", "YEAR", "ORIGINAL_YEAR",
    "MATRIX_GUIDELINE", "STUDY_DURATION_YEARS", "TOXVAL_OUTLIER_FLAG",
    "NEGATIVE",
    "TOXICOLOGICAL_EFFECT_LLM_CLEANED",
}


def compute_year_depth_stats(
    input_file: Path,
    depth_map: dict[str, int],
    *,
    min_year: int | None = None,
    max_year: int | None = None,
    missing_tsv: Path | None = None,
):
    """Return per-year depth statistics used for the line plot."""
    if missing_tsv:
        schema = pl.read_csv(input_file, n_rows=0).schema
        mp_cols = [c for c in schema.keys() if c not in METADATA_COLS]
        tsv = pl.read_csv(missing_tsv, separator="\t")
        active_mp_cols = tsv.filter(pl.col("missing_pct") < 100.0).select("column").to_series().to_list()
    else:
        schema = pl.read_csv(input_file, n_rows=0).schema
        mp_cols = [c for c in schema.keys() if c not in METADATA_COLS]
        df_temp = pl.read_csv(
            input_file,
            columns=mp_cols,
            infer_schema_length=0,
            null_values=["", "None", "none", "NA", "N/A", "NaN", "null"],
        )
        total_rows = df_temp.height
        missing_counts = df_temp.null_count().row(0)
        active_mp_cols = [col for col, cnt in zip(mp_cols, missing_counts) if cnt < total_rows]

    cols_to_load = ["YEAR"] + active_mp_cols
    df = pl.read_csv(
        input_file,
        columns=cols_to_load,
        schema_overrides={col: pl.Utf8 for col in cols_to_load},
        null_values=["", "NA", "NaN", "null"],
        ignore_errors=True,
    )
    df = df.with_columns(
        pl.col("YEAR").cast(pl.Utf8).str.strip_chars().replace("-", None)
        .cast(pl.Float64, strict=False).cast(pl.Int32, strict=False)
    )
    if min_year:
        df = df.filter(pl.col("YEAR") >= min_year)
    if max_year:
        df = df.filter(pl.col("YEAR") <= max_year)
    df = df.filter(pl.col("YEAR").is_not_null() & (pl.col("YEAR") > 0))

    col_depths = []
    for c in active_mp_cols:
        clean_c = c.split(":", 1)[1] if ":" in c else c
        col_depths.append(depth_map.get(clean_c.lower().strip(), 0))

    depth_exprs = []
    for col, depth in zip(active_mp_cols, col_depths):
        is_assigned = (
            pl.col(col).is_not_null()
            & (pl.col(col).cast(pl.Utf8).str.strip_chars() != "")
            & (pl.col(col).cast(pl.Utf8).str.strip_chars() != "0")
        )
        depth_exprs.append(pl.when(is_assigned).then(depth).otherwise(None))

    df = df.with_columns([
        pl.max_horizontal(depth_exprs).alias("max_depth"),
    ])
    df_plot = df.filter(pl.col("max_depth").is_not_null()).select(["YEAR", "max_depth"]).to_pandas()
    df_plot["max_depth"] = df_plot["max_depth"].astype(int)
    return (
        df_plot.groupby("YEAR")["max_depth"]
        .agg(
            mean_depth="mean",
            median_depth="median",
            n_rows="count",
            std_depth="std",
        )
        .reset_index()
    )


def draw_year_vs_max_depth_line(
    ax,
    year_stats,
    *,
    fig=None,
    font_size: float = 36,
    min_rows: int = 5,
    marker_size: float = 120,
    cax=None,
) -> None:
    """Draw mean ± std depth by study year on the given axes."""
    ys = year_stats[year_stats["n_rows"] >= min_rows].copy()
    ax.fill_between(
        ys["YEAR"],
        ys["mean_depth"] - ys["std_depth"],
        ys["mean_depth"] + ys["std_depth"],
        alpha=0.2, color="steelblue", label="± std",
    )
    ax.plot(ys["YEAR"], ys["mean_depth"], "-", color="steelblue", linewidth=1.5, zorder=2)
    sc = ax.scatter(
        ys["YEAR"], ys["mean_depth"], c=ys["n_rows"], cmap="YlOrRd",
        s=marker_size, edgecolor="grey", linewidth=0.7, zorder=3,
    )
    ax.scatter([], [], facecolors="none", edgecolors="grey", s=marker_size, linewidth=0.7, label="Mean")
    if cax is not None:
        cbar = fig.colorbar(sc, cax=cax)
    else:
        colorbar_ax = ax if fig is None else fig
        cbar = colorbar_ax.colorbar(sc, ax=ax, pad=0.01)
    cbar.ax.tick_params(labelsize=font_size * 0.86)
    cbar.set_label("Number of studies", fontsize=font_size, fontweight="bold")
    ax.set_xlabel("Study Year", fontsize=font_size, fontweight="bold")
    ax.set_ylabel("Ontology Depth", fontsize=font_size, fontweight="bold")
    ax.set_ylim(bottom=max(0, ys["mean_depth"].min() - 0.5))
    year_padding = 0.25
    ax.set_xlim(left=ys["YEAR"].min() - year_padding, right=ys["YEAR"].max() + year_padding)
    ax.legend(fontsize=font_size * 0.86)
    ax.tick_params(axis="both", labelsize=font_size * 0.86)
    plt.setp(ax.get_xticklabels(), rotation=90)


def run_depth_analysis(args, depth_map: dict, output_dir: Path) -> None:
    _print("\n--- PART A: Depth Analysis ---", flush=True)

    if args.missing_tsv:
        _print("Reading missing TSV ...", flush=True)
        schema = pl.read_csv(args.input_file, n_rows=0).schema
        mp_cols = [c for c in schema.keys() if c not in METADATA_COLS]
        tsv = pl.read_csv(args.missing_tsv, separator="\t")
        active_mp_cols = tsv.filter(pl.col("missing_pct") < 100.0).select("column").to_series().to_list()
    else:
        _print("Identifying active ontology-term columns directly from CSV...", flush=True)
        schema = pl.read_csv(args.input_file, n_rows=0).schema
        mp_cols = [c for c in schema.keys() if c not in METADATA_COLS]
        df_temp = pl.read_csv(
            args.input_file,
            columns=mp_cols,
            infer_schema_length=0,
            null_values=["", "None", "none", "NA", "N/A", "NaN", "null"],
        )
        total_rows = df_temp.height
        missing_counts = df_temp.null_count().row(0)
        active_mp_cols = [col for col, cnt in zip(mp_cols, missing_counts) if cnt < total_rows]

    _print(f"  Active ontology-term columns (< 100% missing): {len(active_mp_cols):,}", flush=True)

    cols_to_load = ["YEAR"] + active_mp_cols
    _print(f"Loading CSV ({len(cols_to_load):,} columns) ...", flush=True)
    df = pl.read_csv(
        args.input_file,
        columns=cols_to_load,
        schema_overrides={col: pl.Utf8 for col in cols_to_load},
        null_values=["", "NA", "NaN", "null"],
        ignore_errors=True,
    )
    _print(f"  Shape loaded: {df.shape}", flush=True)

    df = df.with_columns(
        pl.col("YEAR").cast(pl.Utf8).str.strip_chars().replace("-", None)
        .cast(pl.Float64, strict=False).cast(pl.Int32, strict=False)
    )
    if args.min_year:
        df = df.filter(pl.col("YEAR") >= args.min_year)
    if args.max_year:
        df = df.filter(pl.col("YEAR") <= args.max_year)
    df = df.filter(pl.col("YEAR").is_not_null() & (pl.col("YEAR") > 0))

    col_depths = []
    for c in active_mp_cols:
        clean_c = c.split(":", 1)[1] if ":" in c else c
        col_depths.append(depth_map.get(clean_c.lower().strip(), 0))

    depth_exprs, assignment_exprs = [], []
    for col, depth in zip(active_mp_cols, col_depths):
        is_assigned = (
            pl.col(col).is_not_null()
            & (pl.col(col).cast(pl.Utf8).str.strip_chars() != "")
            & (pl.col(col).cast(pl.Utf8).str.strip_chars() != "0")
        )
        depth_exprs.append(pl.when(is_assigned).then(depth).otherwise(None))
        assignment_exprs.append(pl.when(is_assigned).then(1).otherwise(0))

    _print("  Computing max depth and active assignments per row ...", flush=True)
    df = df.with_columns([
        pl.max_horizontal(depth_exprs).alias("max_depth"),
        pl.sum_horizontal(assignment_exprs).alias("num_assignments"),
    ])

    df_plot_pl = df.filter(pl.col("max_depth").is_not_null())
    df_plot = df_plot_pl.select(["YEAR", "max_depth"]).to_pandas()
    df_plot["max_depth"] = df_plot["max_depth"].astype(int)

    year_stats = (
        df_plot.groupby("YEAR")["max_depth"]
        .agg(
            mean_depth="mean",
            median_depth="median",
            n_rows="count",
            std_depth="std",
        )
        .reset_index()
    )

    stats_csv = output_dir / "year_depth_stats.csv"
    year_stats.to_csv(stats_csv, index=False)
    _print(f"  Saved year stats cache: {stats_csv}", flush=True)

    # Compute term_counts (column-by-column)
    _print("  Computing term_counts (column-by-column) ...", flush=True)
    term_counts = []
    for col in active_mp_cols:
        cnt = df_plot_pl.select(
            (
                pl.col(col).is_not_null()
                & (pl.col(col).str.strip_chars() != "")
                & (pl.col(col).str.strip_chars() != "0")
                & (pl.col(col).str.strip_chars() != "0.0")
                & (pl.col(col).str.strip_chars().str.to_lowercase() != "false")
            ).sum()
        ).item()
        term_counts.append(cnt)
    term_df = pl.DataFrame({"term": active_mp_cols, "depth": col_depths, "assignments": term_counts})

    # Free heavy dataframes before PART B/C
    del df, df_plot, df_plot_pl
    gc.collect()
    _print("  [gc] Freed CSV dataframes inside run_depth_analysis.", flush=True)

    term_frequencies = {col: 0 for col in mp_cols}
    term_frequencies.update(zip(term_df["term"], term_df["assignments"]))
    return term_frequencies


# ── PART D: Extra metrics analysis and plotting helpers ───────────────────────

def analyze_json_metrics(data: dict, G: nx.DiGraph, name_to_uri: dict) -> dict:
    """Parses the reasoning JSON to extract iterations, mapped/unmapped details, and candidate origins for unique terms."""
    total_phrases = len(data)
    mapped_count = 0
    unmapped_count = 0
    unmapped_phrases = []
    rag_relative_levels_per_phrase = []
    mapping_iterations = []
    undirected_G = G.to_undirected(as_view=True)
    distance_cache: dict[tuple[str, tuple[str, ...]], int | None] = {}

    def min_distance_from_initial_rag(term: str, initial_terms: list[str]) -> int | None:
        term_uri = name_to_uri.get(term)
        if term_uri is None:
            return None

        initial_terms_key = tuple(sorted(initial_terms))
        cache_key = (term, initial_terms_key)
        if cache_key in distance_cache:
            return distance_cache[cache_key]

        distances = []
        for initial_term in initial_terms:
            initial_uri = name_to_uri.get(initial_term)
            if initial_uri is None:
                continue
            try:
                distances.append(nx.shortest_path_length(undirected_G, initial_uri, term_uri))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        distance = min(distances) if distances else None
        distance_cache[cache_key] = distance
        return distance

    for phrase, info in data.items():
        selected = info.get("selected", [])
        steps = info.get("reasoning", [])
        
        if selected:
            mapped_count += 1
            initial_rag_terms = []
            if steps:
                initial_rag_terms = [
                    c.lower().strip()
                    for c in steps[0].get("candidates_sent", [])
                ]
            selected_terms = {
                term.split(":", 1)[-1].lower().strip()
                for term in selected
            }
            explicitly_accepted_terms = set()
            for step in steps:
                for summary_item in step.get("summary", []):
                    if summary_item.get("decision") == "ACCEPTED":
                        explicitly_accepted_terms.add(
                            summary_item.get("candidate", "").lower().strip()
                        )
            accepted_final_terms = sorted(selected_terms & explicitly_accepted_terms)
            selected_relative_levels = []
            
            for clean_term in accepted_final_terms:
                distance_from_rag = min_distance_from_initial_rag(clean_term, initial_rag_terms)
                if distance_from_rag is not None:
                    selected_relative_levels.append(distance_from_rag)
                found_iter = None
                
                # Check summary lists in iterations
                for step in steps:
                    iter_num = step.get("iteration", 1)
                    for summary_item in step.get("summary", []):
                        cand = summary_item.get("candidate", "").lower().strip()
                        if cand == clean_term and summary_item.get("decision") == "ACCEPTED":
                            found_iter = iter_num
                            break
                    if found_iter is not None:
                        break
                
                # Fallback: check candidates_sent
                if found_iter is None:
                    for step in steps:
                        iter_num = step.get("iteration", 1)
                        cands = [c.lower().strip() for c in step.get("candidates_sent", [])]
                        if clean_term in cands:
                            found_iter = iter_num
                            break
                
                final_iter = found_iter if found_iter is not None else 1
                mapping_iterations.append(final_iter)

            if selected_relative_levels:
                rag_relative_levels_per_phrase.append(max(selected_relative_levels))
        else:
            unmapped_count += 1
            unmapped_phrases.append(phrase)

    return {
        "total_phrases": total_phrases,
        "mapped_count": mapped_count,
        "unmapped_count": unmapped_count,
        "unmapped_phrases": unmapped_phrases,
        "rag_relative_levels_per_phrase": rag_relative_levels_per_phrase,
        "unique_accepted_terms_iterations": mapping_iterations,
        "total_unique_accepted_terms": len(mapping_iterations)
    }


def plot_origin_and_density(unique_accepted_iters: list[int], term_frequencies: dict[str, int], output_dir: Path):
    """Generates Combined Plot 1: Origin of Accepted Mappings & Term Selection Frequency Distribution."""
    _print("Generating Combined Plot: Origin & Density Analysis...", flush=True)
    if not unique_accepted_iters:
        _print("  No accepted terms to plot.")
        return

    # Categorize mappings by iteration
    from collections import Counter
    iter_counts = Counter(unique_accepted_iters)
    iter_1_rag = iter_counts.get(1, 0)
    iter_2 = iter_counts.get(2, 0)
    iter_3 = iter_counts.get(3, 0)
    iter_4 = iter_counts.get(4, 0)
    iter_5 = iter_counts.get(5, 0)
    iter_other = sum(count for iter_num, count in iter_counts.items() if iter_num >= 6)
    
    total = iter_1_rag + iter_2 + iter_3 + iter_4 + iter_5 + iter_other
    graph_total = iter_2 + iter_3 + iter_4 + iter_5 + iter_other

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5)) # Slightly taller for legend space

    # --- Subplot 1: Nested Donut Chart ---
    outer_sizes = [iter_1_rag, graph_total]
    outer_labels = [f"RAG\n({iter_1_rag:,} mappings)", f"Ontology Hierarchy\n({graph_total:,} mappings)"]
    outer_colors = ["#1D3557", "#55A868"]

    inner_sizes = [iter_1_rag, iter_2, iter_3, iter_4, iter_5, iter_other]
    inner_colors = ["white", "#2A9D8F", "#52B788", "#74C69D", "#95D5B2", "#B7E4C7"]

    # Outer ring
    wedges_out, texts_out, autotexts_out = axes[0].pie(
        outer_sizes, labels=outer_labels, autopct="%1.1f%%", startangle=140,
        colors=outer_colors, textprops=dict(fontweight="bold", fontsize=13),
        pctdistance=0.90, radius=1.0,
        wedgeprops=dict(width=0.2, edgecolor='w', linewidth=1.5)
    )
    plt.setp(autotexts_out, size=12, weight="bold", color="white")

    # Inner ring (subgroups inside RAG and Hierarchical Scheme)
    # Do not put the percentages in the inner ring as they are already in the legend
    def inner_autopct(pct):
        return ""

    wedges_in, texts_in, autotexts_in = axes[0].pie(
        inner_sizes, autopct=inner_autopct, startangle=140,
        colors=inner_colors, textprops=dict(fontweight="bold", fontsize=11),
        radius=0.78,
        wedgeprops=dict(width=0.2, edgecolor='w', linewidth=1.5)
    )
    plt.setp(autotexts_in, size=11, weight="bold", color="white")

    # Legend for the detailed iterations (excluding Iter 1/RAG as requested)
    legend_labels = [
        f"Iter 2 ({iter_2/total*100:.1f}%)",
        f"Iter 3 ({iter_3/total*100:.1f}%)",
        f"Iter 4 ({iter_4/total*100:.1f}%)",
        f"Iter 5 ({iter_5/total*100:.1f}%)",
        f"Other ({iter_other/total*100:.1f}%)"
    ]
    legend_handles = list(wedges_in[1:])
    axes[0].legend(
        legend_handles, legend_labels,
        title="Iteration Breakdown",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=11
    )

    # --- Subplot 2: Selection Frequency Histogram ---
    term_counts = list(term_frequencies.values())
    total_terms = len(term_counts)
    
    # Filter selection counts to less than 100,000 as requested
    selected_counts = [c for c in term_counts if 0 < c < 100000]
    total_selected = len([c for c in term_counts if c > 0])
    never_selected = total_terms - total_selected
    
    if selected_counts:
        bins = np.logspace(0, 5, 30) # log space bins up to 100,000 (10^5)
        axes[1].hist(
            selected_counts, bins=bins, color="#1D3557", edgecolor="w", 
            linewidth=0.8, alpha=0.95
        )
        axes[1].set_xscale("log")
        axes[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        axes[1].set_xlim(1, 100000)
    else:
        axes[1].text(0.5, 0.5, "No terms were selected < 100,000.", ha="center", va="center", fontsize=14)
        
    stats_text = (
        f"Total unique ontology terms: {total_terms:,}\n"
        f"Selected at least once: {total_selected:,} ({total_selected/total_terms*100:.1f}%)\n"
        f"Never selected: {never_selected:,} ({never_selected/total_terms*100:.1f}%)"
    )
    axes[1].text(
        0.95, 0.75, stats_text, transform=axes[1].transAxes,
        verticalalignment='top', horizontalalignment='right', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='grey')
    )
    axes[1].set_xlabel("Selection Frequency in Database (Log Scale)")
    axes[1].set_ylabel("Selection Frequency")
    axes[1].grid(False)

    plt.tight_layout()
    out_path = output_dir / "accepted_terms_origin_distribution.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    _print(f"  Saved: {out_path}", flush=True)


def plot_mapping_success_and_depth(mapped_cnt: int, unmapped_cnt: int, relative_levels: list[int], output_dir: Path):
    """Generates Combined Plot 2: Mapping Rate & RAG-relative Ontology Distance."""
    _print("Generating Combined Plot: Success & RAG-relative Ontology Distance Analysis...", flush=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    # --- Subplot 1: Pie Chart ---
    labels = [f"Mapped to Ontology\n({mapped_cnt:,} phrases)", f"Unmapped\n({unmapped_cnt:,} phrases)"]
    sizes = [mapped_cnt, unmapped_cnt]
    colors = ["#34A853", "#EA4335"]  # Green and Red
    
    def my_autopct(pct):
        return f"{pct:.1f}%" if pct > 50.0 else ""

    wedges, texts, autotexts = axes[0].pie(
        sizes, labels=labels, autopct=my_autopct, startangle=140,
        colors=colors, textprops=dict(fontweight="bold", fontsize=13),
        pctdistance=0.8,
        wedgeprops=dict(width=0.4, edgecolor='w', linewidth=2)
    )
    plt.setp(autotexts, size=13, weight="bold", color="white")

    # --- Subplot 2: Maximum ontology distance from initial RAG terms per mapped phrase ---
    if relative_levels:
        from collections import Counter
        level_counts = Counter(relative_levels)
        
        sorted_counts = [(level, level_counts.get(level, 0)) for level in range(1, 8)]
        overflow_count = sum(count for level, count in level_counts.items() if level >= 8)
        if overflow_count:
            sorted_counts.append(("8+", overflow_count))
        level_labels = [f"{level}" for level, _ in sorted_counts]
        counts = [count for _, count in sorted_counts]
        
        axes[1].bar(level_labels, counts, color="#1D3557", edgecolor="grey", linewidth=0.7, width=0.6)
        axes[1].grid(False)
        
        axes[1].set_xlabel("Max Ontology Levels from Initial RAG Candidate", labelpad=10)
        axes[1].set_ylabel("Number of Phrases")
        axes[1].set_ylim(0, (max(counts) if counts else 1) * 1.15)
    else:
        axes[1].text(0.5, 0.5, "No RAG-relative levels to plot.", ha="center", va="center", fontsize=14)

    plt.tight_layout()
    out_path = output_dir / "mapping_success_and_unmapped_analysis.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    _print(f"  Saved: {out_path}", flush=True)


# ── PART B: Phenotypic landscape ──────────────────────────────────────────────

def run_phenotypic_landscape(data: dict, term_to_top_branch: dict,
                             output_dir: Path) -> None:
    _print("\n--- PART B: Phenotypic Landscape (Unique Phrases) ---", flush=True)
    if not HAS_SQUARIFY:
        _print("WARNING: Skipping treemap – squarify not installed.", file=sys.stderr)
        return

    branch_counts: dict = {}
    for phrase, info in data.items():
        selected = info.get("selected", [])
        if not selected:
            continue
        phrase_branches: set = set()
        for term_with_prefix in selected:
            term = term_with_prefix.split(":", 1)[-1].lower().strip()
            branch = term_to_top_branch.get(term, "Other/Unknown")
            if branch != "Other/Unknown":
                phrase_branches.add(branch)
        if not phrase_branches:
            phrase_branches.add("Other/Unknown")
        for branch in phrase_branches:
            branch_counts[branch] = branch_counts.get(branch, 0) + 1

    sorted_branches = sorted(branch_counts.items(), key=lambda x: x[1], reverse=True)
    branches = [x[0] for x in sorted_branches if x[0] != "Other/Unknown"]
    counts = [x[1] for x in sorted_branches if x[0] != "Other/Unknown"]

    max_c = max(counts)
    labels = []
    for b, c in zip(branches, counts):
        if c >= 234:
            b_wrapped = (b.replace(" phenotype", "")
                          .replace(" system", "\nsystem")
                          .replace("/", "/\n"))
            labels.append(f"{b_wrapped}\n({c:,})")
        else:
            labels.append("")

    fig, ax = plt.subplots(figsize=(16, 10))
    squarify.plot(sizes=counts, label=labels, alpha=0.85,
                  color=sns.color_palette("Spectral", len(counts)),
                  text_kwargs={"fontsize": 9, "weight": "bold"}, ax=ax)
    ax.axis("off")

    out5 = output_dir / "phenotypic_landscape_unique_treemap.png"
    fig.savefig(out5, dpi=300, bbox_inches="tight")
    plt.close(fig)
    _print(f"  Saved Plot 5: {out5}", flush=True)


# ── PART C: Ontology tree PNG (no .dot, no .pdf) ─────────────────────────────

def run_ontology_tree(data: dict, G: nx.DiGraph,
                      name_to_uri: dict, uri_to_name: dict,
                      root_term: str, threshold: int,
                      output_dir: Path) -> None:
    _print("\n--- PART C: Ontology Tree ---", flush=True)

    root_uri = name_to_uri.get(root_term.lower())
    if not root_uri:
        _print("ERROR: Root term not found in ontology – skipping tree.", file=sys.stderr)
        return

    # Reverse graph: parent → children
    G_rev = G.reverse()
    reachable = nx.descendants(G_rev, root_uri)
    reachable.add(root_uri)
    G_rev = G_rev.subgraph(reachable).copy()

    # Propagate counts through DAG
    dag_hits: dict = {n: 0 for n in G_rev.nodes()}
    for phrase, info in data.items():
        selected = info.get("selected", [])
        phrase_uris: set = set()
        for term_with_prefix in selected:
            term = term_with_prefix.split(":", 1)[-1].lower().strip()
            uri = name_to_uri.get(term)
            if uri and uri in G_rev:
                phrase_uris.add(uri)
        all_ancestors: set = set()
        for u in phrase_uris:
            all_ancestors.add(u)
            if u in G:
                all_ancestors.update(nx.descendants(G, u))
        for u in all_ancestors:
            if u in dag_hits:
                dag_hits[u] += 1

    # Prune
    _print(f"Pruning nodes with < {threshold} hits ...", flush=True)
    keep_nodes = {n for n, cnt in dag_hits.items() if cnt >= threshold}
    keep_nodes.add(root_uri)
    G_pruned = G_rev.subgraph(keep_nodes).copy()
    _print(f"  Pruned graph: {G_pruned.number_of_nodes()} nodes, "
           f"{G_pruned.number_of_edges()} edges", flush=True)

    # Build colormap
    counts_nonroot = [dag_hits[n] for n in G_pruned.nodes() if n != root_uri]
    max_hits = max(counts_nonroot) if counts_nonroot else threshold
    cmap = plt.get_cmap("YlOrRd")
    norm = mcolors.LogNorm(vmin=threshold, vmax=max_hits)

    def get_color(count: int) -> str:
        if count == 0:
            return "#ffffff"
        return mcolors.to_hex(cmap(norm(count)))

    # Build DOT string in memory (not written to disk)
    dot_lines = [
        "digraph G {",
        "  rankdir=TB;",
        "  nodesep=0.8;",
        "  ranksep=1.5;",
        '  node [shape=box, style="filled,rounded", fontname="Helvetica"];',
        '  edge [color="#666666"];',
        "",
    ]
    for n in G_pruned.nodes():
        count = dag_hits[n]
        name = uri_to_name.get(n, "Unknown")
        words = name.split()
        wrapped = "\\n".join(" ".join(words[i:i+3]) for i in range(0, len(words), 3))
        label = f"{wrapped}\\n({count})"
        fillcolor = get_color(count)
        penwidth = 3.0 if n == root_uri else 1.0
        dot_lines.append(
            f'  "{n}" [label="{label}", fillcolor="{fillcolor}", '
            f'fontcolor="#000000", penwidth={penwidth}];'
        )
    for u, v in G_pruned.edges():
        thickness = 1.0 + 4.0 * (dag_hits[v] / max_hits)
        dot_lines.append(f'  "{u}" -> "{v}" [penwidth={thickness:.2f}];')
    dot_lines.append("}")
    dot_source = "\n".join(dot_lines)

    # Render directly to PNG via Graphviz (pipe mode – no temp files on disk)
    import subprocess
    out_png = output_dir / "ontology_tree.png"
    _print(f"Rendering ontology tree PNG: {out_png}", flush=True)
    try:
        result = subprocess.run(
            ["dot", "-Tpng", "-o", str(out_png)],
            input=dot_source.encode(),
            capture_output=True,
            check=True,
        )
        _print(f"  Saved Plot 6: {out_png}", flush=True)
    except subprocess.CalledProcessError as e:
        _print(f"ERROR running Graphviz dot: {e.stderr.decode()}", file=sys.stderr)
    except FileNotFoundError:
        _print("ERROR: 'dot' (Graphviz) not found in PATH.", file=sys.stderr)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    configure_fonts()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse ontology once – shared by all parts
    G, name_to_uri, uri_to_name = build_ontology_graph(args.owl_file)
    depth_map = build_depth_map(G, name_to_uri, uri_to_name, args.root_term)
    term_to_top_branch = map_to_top_branches(G, name_to_uri, uri_to_name, args.root_term)

    # Run depth analysis (Part A) and get term frequencies while JSON is not yet in memory
    term_frequencies = run_depth_analysis(args, depth_map, args.output_dir)
    if args.depth_only:
        _print("Depth-only mode complete.", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    # Free CSV dataframes memory from RAM immediately
    gc.collect()
    _print("  [gc] Memory freed after PART A (CSV dataframes).", flush=True)

    # Load JSON once CSV memory is cleared
    _print(f"Loading reasoning JSON: {args.json_file} ...", flush=True)
    with open(args.json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Generate extra metrics plots (Part D)
    try:
        _print("\n--- PART D: Extra Metrics Plots ---", flush=True)
        json_metrics = analyze_json_metrics(json_data, G, name_to_uri)
        # plot_origin_and_density(
        #     json_metrics["unique_accepted_terms_iterations"],
        #     term_frequencies,
        #     args.output_dir
        # )
        plot_mapping_success_and_depth(
            json_metrics["mapped_count"],
            json_metrics["unmapped_count"],
            json_metrics["rag_relative_levels_per_phrase"],
            args.output_dir
        )
    except Exception as e:
        _print(f"ERROR in PART D (extra metrics plots): {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    # Free RAM from the large CSV before heavy graph/JSON operations
    gc.collect()
    _print("  [gc] Memory freed after PART A & D.", flush=True)

    try:
        run_phenotypic_landscape(json_data, term_to_top_branch, args.output_dir)
    except Exception as e:
        _print(f"ERROR in PART B (phenotypic landscape): {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    gc.collect()
    _print("  [gc] Memory freed after PART B.", flush=True)

    try:
        run_ontology_tree(
            json_data, G, name_to_uri, uri_to_name,
            args.root_term, args.threshold, args.output_dir,
        )
    except Exception as e:
        _print(f"ERROR in PART C (ontology tree): {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    _print("\nAll plots generated successfully!", flush=True)


if __name__ == "__main__":
    main()
