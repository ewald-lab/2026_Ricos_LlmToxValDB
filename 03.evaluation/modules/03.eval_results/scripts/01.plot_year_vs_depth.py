#!/usr/bin/env python3
"""
Plot: distribution of YEAR vs max ontology depth of MP assignments,
along with stats on assignments per depth level.

For each row in the classified CSV:
  - Reads only YEAR + active MP columns (those with <100% missing, from TSV) using Polars
  - Computes max depth and active assignments per row using Polars
  - Compiles statistics on ontology terms: mean assignments by depth level, 
    and distribution of terms' selection frequency by depth level.

Usage:
    python 02.plot_year_vs_depth.py \
        --input_file    <path/to/ToxValDB_Classified_mp_*.csv> \
        --missing_tsv   <path/to/missing_analysis_mp_*.tsv> \
        --owl_file      <path/to/mp.ttl> \
        --output_dir    <path/to/outputs/figures> \
        [--root_term    "mammalian phenotype"] \
        [--min_year     <int>] \
        [--max_year     <int>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
import rdflib
import seaborn as sns
from rdflib.namespace import RDFS

# Silence all print statements
print = lambda *args, **kwargs: None


# ── Args ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_file",  required=True, type=Path,
                   help="Classified CSV (ToxValDB_Classified_mp_*.csv)")
    p.add_argument("--missing_tsv", type=Path, default=None,
                   help="Optional per-column missing-% TSV produced by 01.analyse_missing.py")
    p.add_argument("--owl_file",    required=True, type=Path,
                   help="MP ontology TTL file")
    p.add_argument("--output_dir",  required=True, type=Path)
    p.add_argument("--root_term",   default="mammalian phenotype")
    p.add_argument("--min_year",    type=int, default=None)
    p.add_argument("--max_year",    type=int, default=None)
    return p.parse_args()


# ── Ontology depth map (same logic as 05.eval_combined.py) ───────────────────
def build_depth_map(owl_file: Path, root_term: str) -> dict[str, int]:
    print(f"Parsing ontology: {owl_file}", flush=True)
    g = rdflib.Graph()
    g.parse(str(owl_file), format="turtle")

    G = nx.DiGraph()
    name_to_uri: dict = {}
    uri_to_name: dict = {}

    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
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

    depth_map: dict = {}
    root_uri = name_to_uri.get(root_term.lower().strip())
    if root_uri is None:
        print(f"WARNING: root term '{root_term}' not found!", file=sys.stderr)
    else:
        G_rev = G.reverse()
        for node, d in nx.single_source_shortest_path_length(G_rev, root_uri).items():
            if node in uri_to_name:
                depth_map[uri_to_name[node]] = d

    print(f"  Depth map: {len(depth_map):,} terms", flush=True)
    return depth_map


# ── Main ─────────────────────────────────────────────────────────────────────
# Metadata columns – everything else is treated as an ontology (MP) column
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
    "TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED", "NEGATIVE",
    "TOXICOLOGICAL_EFFECT_LLM_CLEANED",
}


def main() -> None:
    args = parse_args()
    for f in (args.input_file, args.owl_file):
        if not f.exists():
            print(f"ERROR: not found: {f}", file=sys.stderr)
            sys.exit(1)
    if args.missing_tsv and not args.missing_tsv.exists():
        print(f"ERROR: not found: {args.missing_tsv}", file=sys.stderr)
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build depth map
    depth_map = build_depth_map(args.owl_file, args.root_term)

    # 2. Find active MP columns (missing < 100%)
    if args.missing_tsv:
        print("Reading missing TSV ...", flush=True)
        tsv = pl.read_csv(args.missing_tsv, separator="\t")
        active_mp_cols = tsv.filter(pl.col("missing_pct") < 100.0).select("column").to_series().to_list()
    else:
        print("Identifying active MP columns directly from CSV...", flush=True)
        schema = pl.read_csv(args.input_file, n_rows=0).schema
        all_cols = list(schema.keys())
        mp_cols = [c for c in all_cols if c not in METADATA_COLS]
        
        # Read only MP columns to compute null counts
        df_temp = pl.read_csv(
            args.input_file,
            columns=mp_cols,
            infer_schema_length=0,
            null_values=["", "None", "none", "NA", "N/A", "NaN", "null"]
        )
        total_rows = df_temp.height
        missing_counts = df_temp.null_count().row(0)
        active_mp_cols = [
            col for col, cnt in zip(mp_cols, missing_counts) if cnt < total_rows
        ]
        
    print(f"  Active MP columns (< 100% missing): {len(active_mp_cols):,}", flush=True)

    # 3. Load CSV with only YEAR + active MP columns
    cols_to_load = ["YEAR"] + active_mp_cols
    print(f"Loading CSV (only {len(cols_to_load):,} columns) ...", flush=True)
    
    # Read using polars
    df = pl.read_csv(
        args.input_file,
        columns=cols_to_load,
        infer_schema_length=10000,
        null_values=["", "NA", "NaN", "null"],
        ignore_errors=True
    )
    print(f"  Shape loaded: {df.shape}", flush=True)

    # 4. Filter YEAR
    # Replace non-numeric values like '-' with null, cast first to Float64 (since years are formatted as decimals like '1991.0'), then to Int32
    df = df.with_columns(
        pl.col("YEAR")
        .cast(pl.Utf8)
        .str.strip_chars()
        .replace("-", None)
        .cast(pl.Float64, strict=False)
        .cast(pl.Int32, strict=False)
    )
    if args.min_year:
        df = df.filter(pl.col("YEAR") >= args.min_year)
    if args.max_year:
        df = df.filter(pl.col("YEAR") <= args.max_year)
    df = df.filter(pl.col("YEAR").is_not_null() & (pl.col("YEAR") > 0))
    print(f"  Rows with valid YEAR: {len(df):,}", flush=True)

    if len(df) == 0:
        print("ERROR: No rows left after filtering YEAR.", file=sys.stderr)
        sys.exit(1)

    # 5. Map active columns to their ontology depth
    # Strip the 'MP:' prefix (if present) before looking up the term label in the depth map
    col_depths = []
    for c in active_mp_cols:
        clean_c = c.split(":", 1)[1] if ":" in c else c
        col_depths.append(depth_map.get(clean_c.lower().strip(), 0))
    
    # Create expressions to get depth when assigned, else null
    depth_exprs = []
    assignment_exprs = [] # 1 if assigned, 0 if not
    for col, depth in zip(active_mp_cols, col_depths):
        is_assigned = (
            pl.col(col).is_not_null() & 
            (pl.col(col).cast(pl.Utf8).str.strip_chars() != "") &
            (pl.col(col).cast(pl.Utf8).str.strip_chars() != "0")
        )
        depth_exprs.append(
            pl.when(is_assigned).then(depth).otherwise(None)
        )
        assignment_exprs.append(
            pl.when(is_assigned).then(1).otherwise(0)
        )

    print("  Computing max depth and active assignments per row ...", flush=True)
    df = df.with_columns([
        pl.max_horizontal(depth_exprs).alias("max_depth"),
        pl.sum_horizontal(assignment_exprs).alias("num_assignments")
    ])

    # Keep rows with ≥1 assignment
    df_plot_pl = df.filter(pl.col("max_depth").is_not_null())
    print(f"  Rows with ≥1 MP assignment: {len(df_plot_pl):,}", flush=True)

    if len(df_plot_pl) == 0:
        print("WARNING: No rows with valid MP assignments found.", file=sys.stderr)
        sys.exit(0)

    # Convert to pandas for primary year vs depth plots
    df_plot = df_plot_pl.select(["YEAR", "max_depth"]).to_pandas()
    df_plot["max_depth"] = df_plot["max_depth"].astype(int)

    # ── Aggregate stats by year ───────────────────────────────────────────────
    year_stats_pl = (
        df_plot_pl.group_by("YEAR")
        .agg([
            pl.col("max_depth").mean().alias("mean_depth"),
            pl.col("max_depth").median().alias("median_depth"),
            pl.col("max_depth").count().alias("n_rows"),
            pl.col("max_depth").std().alias("std_depth"),
        ])
        .sort("YEAR")
    )
    year_stats = year_stats_pl.to_pandas()

    years_sorted = sorted(df_plot["YEAR"].unique())
    fig_w = max(14, len(years_sorted) * 0.5 + 2)

    # ── Plot 2: Mean ± std line ───────────────────────────────────────────────
    ys = year_stats[year_stats["n_rows"] >= 5].copy()
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    ax.fill_between(
        ys["YEAR"],
        ys["mean_depth"] - ys["std_depth"],
        ys["mean_depth"] + ys["std_depth"],
        alpha=0.2, color="steelblue", label="± std",
    )
    # Draw line without markers
    ax.plot(ys["YEAR"], ys["mean_depth"], "-", color="steelblue", linewidth=1.5, zorder=2)
    # Scatter plot with colormap mapped to number of rows (study count) per year
    sc = ax.scatter(
        ys["YEAR"], ys["mean_depth"], 
        c=ys["n_rows"], cmap="YlOrRd", 
        s=80, edgecolor="grey", linewidth=0.7, 
        zorder=3, label="Mean"
    )
    # Add colorbar for study count scale
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Number of rows (studies)", fontsize=10)
    
    ax.set_title("Depth by Year",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Study Year (YEAR)", fontsize=12)
    ax.set_ylabel("Depth", fontsize=12)
    ax.legend()
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    out2 = args.output_dir / "year_vs_max_depth_line.png"
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out2}")

    # ── Plot 3: 3D Bar Plot (YEAR × max_depth × study count) ──────────────────
    pivot = df_plot.pivot_table(
        index="max_depth", columns="YEAR",
        values="max_depth", aggfunc="count", fill_value=0,
    )
    pivot = pivot[sorted(pivot.columns)]
    
    years = sorted(pivot.columns)
    depths = sorted(pivot.index)
    
    # Increase figure width and add space for labels
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    x_pos, y_pos = np.meshgrid(np.arange(len(years)), np.arange(len(depths)))
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(x_pos)
    
    dx = 0.5 * np.ones_like(z_pos)
    dy = 0.5 * np.ones_like(z_pos)
    
    dz = []
    for d in depths:
        for y in years:
            dz.append(pivot.at[d, y])
    dz = np.array(dz)
    
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color="steelblue", shade=True)
    
    # Prevent overlap by setting ticks for all years, but labeling only every 20th index
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels([str(y) if i % 20 == 0 else "" for i, y in enumerate(years)], 
                       rotation=45, horizontalalignment='right', fontsize=8)
    ax.set_yticks(np.arange(len(depths)))
    ax.set_yticklabels(depths, fontsize=8)
    
    ax.set_title("Study Count by Year × Depth", fontsize=14, fontweight="bold")
    ax.set_xlabel("Study Year (YEAR)", fontsize=11, labelpad=15)
    ax.set_ylabel("Depth", fontsize=11, labelpad=15)
    ax.set_zlabel("Row Count", fontsize=11, labelpad=10)
    
    # Custom viewpoint elevation and azimuth angle
    ax.view_init(elev=25, azim=-55)
    
    out3 = args.output_dir / "year_vs_max_depth_heatmap.png"
    fig.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 3D plot with legend: {out3}")

    # ── New Analysis 1: Mean number of assignments by ontology term depth ───
    print("Analyzing assignments by ontology term depth...", flush=True)
    # Sum frequency of selection for each active column
    term_counts = []
    for col in active_mp_cols:
        col_sum = df_plot_pl.select(
            pl.col(col).cast(pl.Int32, strict=False).fill_null(0).sum()
        ).item()
        term_counts.append(col_sum)

    # DataFrame representing each Term, its depth, and its selection count
    term_df = pl.DataFrame({
        "term": active_mp_cols,
        "depth": col_depths,
        "assignments": term_counts
    })

    # Average assignments per depth level (excluding depth 0)
    depth_assignment_stats = (
        term_df.filter(pl.col("depth") > 0)
        .group_by("depth")
        .agg([
            pl.col("assignments").mean().alias("mean_assignments"),
            pl.col("assignments").median().alias("median_assignments"),
            pl.col("assignments").sum().alias("total_assignments"),
            pl.col("term").count().alias("num_terms")
        ])
        .sort("depth")
    )
    print(depth_assignment_stats)

    # Plot: Mean assignments by depth level (log scale)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=depth_assignment_stats.to_pandas(),
        x="depth", y="mean_assignments",
        palette="viridis", ax=ax
    )
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_title("Mean Selection Count of MP Terms by Ontology Depth (Excluding Root)", fontsize=14, fontweight="bold")
    ax.set_xlabel("MP Ontology Depth", fontsize=12)
    ax.set_ylabel("Mean Assignment Frequency (Log Scale)", fontsize=12)
    plt.tight_layout()
    out4 = args.output_dir / "mean_assignments_by_depth.png"
    fig.savefig(out4, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out4}")

    # ── New Analysis 2: Term count distribution with threshold selection by depth ───
    # Bin terms per depth that were selected >= N times (e.g., >=1, >=5, >=10, >=50, >=100) (excluding depth 0)
    print("Compiling threshold selection frequencies by depth...", flush=True)
    bins = [5, 10, 50, 100]
    bin_exprs = []
    for val in bins:
        bin_exprs.append(
            pl.col("assignments").filter(pl.col("assignments") >= val).count().alias(f"selected_ge_{val}")
        )
    
    threshold_stats = (
        term_df.filter(pl.col("depth") > 0)
        .group_by("depth")
        .agg([pl.col("term").count().alias("total_active_terms")] + bin_exprs)
        .sort("depth")
    )
    print(threshold_stats)

    # Plot a stacked or grouped bar chart of the count of terms by depth for different selection counts (excluding total_active_terms)
    threshold_df_melt = (
        threshold_stats.select(["depth"] + [f"selected_ge_{val}" for val in bins])
        .to_pandas()
        .melt(id_vars="depth", var_name="Threshold", value_name="Term Count")
    )
    # Rename Threshold entries from e.g. selected_ge_5 to 5 selections
    threshold_df_melt["Threshold"] = threshold_df_melt["Threshold"].map(
        lambda x: f"{x.replace('selected_ge_', '')} selections"
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=threshold_df_melt,
        x="depth", y="Term Count", hue="Threshold",
        palette="muted", ax=ax
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    
    # Add percentages above the bars
    depths = sorted(threshold_stats["depth"].to_list())
    depth_totals = dict(zip(threshold_stats["depth"], threshold_stats["total_active_terms"]))
    for container in ax.containers:
        for i, bar in enumerate(container):
            if i >= len(depths):
                continue
            d = depths[i]
            total = depth_totals.get(d, 0)
            val = bar.get_height()
            if val > 0 and total > 0:
                pct = (val / total) * 100
                pct_str = f"{pct:.1f}%"
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                ax.text(
                    x, y * 1.15, pct_str,
                    ha='center', va='bottom', fontsize=7, rotation=90
                )
                
    # Give some extra room at the top for labels on log scale
    current_ymin, current_ymax = ax.get_ylim()
    ax.set_ylim(current_ymin, current_ymax * 5)

    ax.set_title("Number of MP Terms Meeting Selection Thresholds by Depth Level (Excluding Root)", fontsize=14, fontweight="bold")
    ax.set_xlabel("MP Ontology Depth", fontsize=12)
    ax.set_ylabel("Number of Terms (Log Scale)", fontsize=12)
    ax.legend(title="Min. Number of Selections")
    plt.tight_layout()
    out5 = args.output_dir / "term_thresholds_by_depth.png"
    fig.savefig(out5, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out5}")



    print("Done.", flush=True)


if __name__ == "__main__":
    main()
