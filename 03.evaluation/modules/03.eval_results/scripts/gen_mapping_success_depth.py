#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import networkx as nx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", required=True, type=Path)
    parser.add_argument("--ontology_db", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--root_term", required=True)
    parser.add_argument("--no_inner_donut", action="store_true", help="Exclude the inner donut chart showing depth levels")
    parser.add_argument("--no_inner_label_2", action="store_true", help="Hide the '2' label in the inner donut (show only '1')")
    return parser.parse_args()


def configure_fonts() -> None:
    font_dir = Path(__file__).resolve().parents[4] / "assets" / "fonts" / "ibm-plex-sans"
    for font_path in font_dir.glob("*.ttf"):
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"],
    })


def build_ontology_graph(ontology_db: Path) -> tuple[nx.DiGraph, dict[str, str], dict[str, str]]:
    G = nx.DiGraph()
    name_to_uri = {}
    uri_to_name = {}

    with sqlite3.connect(str(ontology_db)) as conn:
        rows = list(conn.execute("SELECT uri, name, sanitized_name, parent FROM embeddings"))

    for uri, name, sanitized_name, _parent_json in rows:
        G.add_node(uri)
        orig_name = (name or "").lower().strip()
        san_name = (sanitized_name or "").lower().strip()
        
        if orig_name:
            name_to_uri[orig_name] = uri
            # Use original name for display
            uri_to_name[uri] = orig_name
        if san_name:
            name_to_uri[san_name] = uri
            if uri not in uri_to_name:
                uri_to_name[uri] = san_name

    for uri, _name, _sanitized_name, parent_json in rows:
        try:
            parents = json.loads(parent_json) if parent_json else []
        except json.JSONDecodeError:
            parents = []
        for parent_name in parents:
            parent_uri = name_to_uri.get(parent_name.lower().strip())
            if parent_uri:
                G.add_edge(uri, parent_uri)

    return G, name_to_uri, uri_to_name


def ensure_root_node(G, name_to_uri, uri_to_name, root_term: str):
    root_name = root_term.lower().strip()
    root_uri = name_to_uri.get(root_name)
    if root_uri:
        return root_uri

    root_uri = f"urn:synthetic-root:{root_name.replace(' ', '_')}"
    roots = [node for node in G.nodes() if G.out_degree(node) == 0]
    if len(roots) == 1 and uri_to_name.get(roots[0], "").lower() == "thing":
        roots = list(G.predecessors(roots[0]))
    for node in roots:
        G.add_edge(node, root_uri)

    name_to_uri[root_name] = root_uri
    uri_to_name[root_uri] = root_name
    return root_uri


def build_depth_map(G: nx.DiGraph, name_to_uri: dict, uri_to_name: dict, root_term: str) -> dict[str, int]:
    depth_map: dict = {}
    root_uri = ensure_root_node(G, name_to_uri, uri_to_name, root_term)
    G_rev = G.reverse()
    for node, d in nx.single_source_shortest_path_length(G_rev, root_uri).items():
        if node in uri_to_name:
            depth_map[uri_to_name[node]] = d
    return depth_map


def iter_json_object_items(path: Path):
    decoder = json.JSONDecoder()
    buffer = ""
    with open(path, "r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            buffer += chunk
            buffer = buffer.lstrip()
            if buffer.startswith("{"):
                buffer = buffer[1:]
                break

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith("}"):
                return
            if not buffer:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    return
                buffer += chunk
                continue

            try:
                key, idx = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    raise
                buffer += chunk
                continue
            buffer = buffer[idx:].lstrip()
            if not buffer.startswith(":"):
                raise ValueError("Expected ':' after JSON object key")
            buffer = buffer[1:].lstrip()

            while True:
                try:
                    value, idx = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        raise
                    buffer += chunk

            yield key, value
            buffer = buffer[idx:].lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]
            elif buffer.startswith("}"):
                return


def analyze_mapping_metrics(json_file: Path, G: nx.DiGraph, name_to_uri: dict[str, str], depth_map: dict[str, int]) -> tuple[int, int, list[int], list[int], list[int], list[int]]:
    mapped_count = 0
    unmapped_count = 0
    relative_levels_per_phrase = []
    relative_levels_per_term = []
    accepted_term_iterations = []
    phrase_max_depths = []
    undirected_G = G.to_undirected(as_view=True)
    distance_cache = {}

    def min_distance_from_initial_rag(term: str, initial_terms: list[str]) -> int | None:
        term_uri = name_to_uri.get(term)
        if term_uri is None:
            return None

        cache_key = (term, tuple(sorted(initial_terms)))
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

    for _phrase, info in iter_json_object_items(json_file):
        selected = info.get("selected", [])
        steps = info.get("reasoning", [])

        if not selected:
            unmapped_count += 1
            continue

        mapped_count += 1
        initial_rag_terms = []
        if steps:
            initial_rag_terms = [c.lower().strip() for c in steps[0].get("candidates_sent", [])]

        selected_terms = {term.split(":", 1)[-1].lower().strip() for term in selected}
        explicitly_accepted_terms = set()
        for step in steps:
            for summary_item in step.get("summary", []):
                if summary_item.get("decision") == "ACCEPTED":
                    explicitly_accepted_terms.add(summary_item.get("candidate", "").lower().strip())

        for term in selected:
            clean_term = term.split(":", 1)[-1].lower().strip()
            found_iter = None
            for step in steps:
                iter_num = step.get("iteration", 1)
                for summary_item in step.get("summary", []):
                    candidate = summary_item.get("candidate", "").lower().strip()
                    if candidate == clean_term and summary_item.get("decision") == "ACCEPTED":
                        found_iter = iter_num
                        break
                if found_iter is not None:
                    break
            if found_iter is None:
                for step in steps:
                    iter_num = step.get("iteration", 1)
                    candidates = [c.lower().strip() for c in step.get("candidates_sent", [])]
                    if clean_term in candidates:
                        found_iter = iter_num
                        break
            accepted_term_iterations.append(found_iter if found_iter is not None else 1)

        explicit_final_terms = sorted(selected_terms & explicitly_accepted_terms)
        distances = [
            distance
            for term in explicit_final_terms
            if (distance := min_distance_from_initial_rag(term, initial_rag_terms)) is not None
        ]
        if distances:
            relative_levels_per_phrase.append(max(distances))
            relative_levels_per_term.extend(distances)

        phrase_depths = [depth_map.get(term, 0) for term in selected_terms]
        phrase_depths = [d for d in phrase_depths if d > 0]
        if phrase_depths:
            phrase_max_depths.append(max(phrase_depths))

    return mapped_count, unmapped_count, relative_levels_per_phrase, relative_levels_per_term, accepted_term_iterations, phrase_max_depths


def plot_combined_summary(
    mapped_count: int,
    unmapped_count: int,
    phrase_max_depths: list[int],
    relative_levels_per_term: list[int],
    accepted_term_iterations: list[int],
    output_dir: Path,
    json_file: Path,
    ontology_db: Path,
    root_term: str,
    no_inner_donut: bool = False,
    no_inner_label_2: bool = False,
) -> None:
    # 2-row grid layout using GridSpec
    # Row 1 has A (depth bar) and B (coverage bar), Row 2 has C (year vs depth line) spanning both columns.
    # Set left=0.06 to shift column 0 left, wspace=0.40 to separate columns, and width_ratios to make B narrower.
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0], width_ratios=[1.15, 1.0], hspace=0.35, wspace=0.40, left=0.06, right=0.94)
    
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_coverage = fig.add_subplot(gs[0, 1])
    ax_line = fig.add_subplot(gs[1, :])

    from collections import Counter
    import matplotlib.patches as mpatches
    import pandas as pd
    import numpy as np

    if mapped_count > 0:
        # Plot mapping depth distribution on ax_bar (based on phrase max absolute depth)
        total_mapped = len(phrase_max_depths)
        depth_categories = ["1", "2", "3", "4", "5", "6", "7", "8+"]
        if total_mapped > 0:
            phrase_depth_counts = Counter(phrase_max_depths)
            depth_counts = [
                phrase_depth_counts.get(1, 0),
                phrase_depth_counts.get(2, 0),
                phrase_depth_counts.get(3, 0),
                phrase_depth_counts.get(4, 0),
                phrase_depth_counts.get(5, 0),
                phrase_depth_counts.get(6, 0),
                phrase_depth_counts.get(7, 0),
                sum(count for depth, count in phrase_depth_counts.items() if depth >= 8),
            ]
            depth_pcts = [c / total_mapped * 100 for c in depth_counts]
        else:
            depth_pcts = [0] * 8

        bars = ax_bar.bar(depth_categories, depth_pcts, color="#4c72b0", edgecolor="black", linewidth=0.5, width=0.6)

        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height + 1.0, f"{height:.1f}%",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax_bar.set_ylabel("Percentage of Mapped Toxicological Effects (%)", fontsize=11, fontweight="bold")
        ax_bar.set_xlabel("Maximum Ontology Depth per Toxicological Effect", fontsize=11, fontweight="bold")
        ax_bar.set_ylim(0, max(depth_pcts) * 1.15 if any(depth_pcts) else 100)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)

        # Panel B: Ontology coverage bar plot
        from gen_coverage_bar import build_ontology_graph as c_build_graph, build_term_to_top_branch as c_build_branches, get_theme as c_get_theme
        G_cov, name_to_uri, uri_to_name = c_build_graph(ontology_db)
        term_to_top_branch, top_branches_list = c_build_branches(G_cov, name_to_uri, uri_to_name, root_term)

        branch_counts = {branch: 0 for branch in top_branches_list}
        for _phrase, info in iter_json_object_items(json_file):
            selected = info.get("selected", [])
            if not selected:
                continue

            phrase_branches = set()
            for term_with_prefix in selected:
                term = term_with_prefix.split(":", 1)[-1].lower().strip()
                branch = term_to_top_branch.get(term, "Other/Unknown")
                if branch != "Other/Unknown":
                    phrase_branches.add(branch)

            for branch in phrase_branches:
                if branch in branch_counts:
                    branch_counts[branch] += 1

        theme_order = {
            "Anatomy & Organ Systems": 0,
            "Physiology & Processes": 1,
            "Cellular & Molecular": 2,
            "Pathology & Outcomes": 3,
            "Other / General": 4
        }

        sorted_items = sorted(branch_counts.items(), key=lambda x: x[1])
        branches = [x[0].capitalize() for x in sorted_items]
        counts = [x[1] for x in sorted_items]
        themes = [c_get_theme(x[0]) for x in sorted_items]

        theme_colors = {
            "Anatomy & Organ Systems": "#4c72b0", # Blue
            "Physiology & Processes": "#55a868", # Green
            "Cellular & Molecular": "#8172b3",   # Purple
            "Pathology & Outcomes": "#c44e52",   # Red
            "Other / General": "#8c8c8c"         # Grey
        }

        colors = [theme_colors[t] for t in themes]
        bars_cov = ax_coverage.barh(branches, counts, color=colors, edgecolor="black", linewidth=0.5)

        max_cov = max(counts) if counts else 1
        for bar in bars_cov:
            width = bar.get_width()
            label_x_pos = width + max_cov * 0.01
            ax_coverage.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{int(width):,}',
                             va='center', ha='left', fontsize=8)

        ax_coverage.set_xlabel("Number of mapped toxicological effects", fontsize=11, fontweight="bold")
        x_max = 4000
        ax_coverage.set_xlim(0, x_max)
        ax_coverage.set_ylim(-0.5, len(branches) - 0.5)
        ticks = np.arange(0, x_max + 1, 1000)
        ax_coverage.set_xticks(ticks)
        ax_coverage.tick_params(axis='x', which='major', bottom=True, length=4, width=1.0, colors='black', labelsize=8)
        ax_coverage.tick_params(axis='y', which='major', labelsize=8)
        ax_coverage.spines['top'].set_visible(False)
        ax_coverage.spines['right'].set_visible(False)
        ax_coverage.spines['bottom'].set_visible(True)
        ax_coverage.spines['bottom'].set_color('#000000')
        ax_coverage.spines['bottom'].set_linewidth(1.0)
        ax_coverage.spines['left'].set_visible(True)
        ax_coverage.spines['left'].set_color('#000000')
        ax_coverage.spines['left'].set_linewidth(1.0)
        ax_coverage.spines['left'].set_bounds(-0.5, len(branches) - 0.5)

        present_themes = set(themes)
        legend_patches = [
            mpatches.Patch(facecolor=theme_colors[t], label=t, edgecolor="black", linewidth=0.5)
            for t in sorted(theme_order.keys(), key=lambda x: theme_order[x])
            if t in present_themes
        ]
        ax_coverage.legend(handles=legend_patches, loc='lower right', bbox_to_anchor=(0.95, 0.02), frameon=False, fontsize=8)

        # Panel C: Year vs max depth line plot (Puntos más pequeños: marker_size=60)
        from eval_results import draw_year_vs_max_depth_line
        stats_csv = output_dir / "year_depth_stats.csv"
        if stats_csv.exists():
            year_stats = pd.read_csv(stats_csv)
            # Align ax_line and cax positions with ax_bar and ax_coverage columns
            pos_bar_now = ax_bar.get_position()
            pos_cov_now = ax_coverage.get_position()
            pos_line_now = ax_line.get_position()

            total_x0 = pos_bar_now.x0 - 0.005
            total_x1 = pos_cov_now.x1 - 0.04
            total_width = total_x1 - total_x0

            line_width = total_width * 0.96
            gap = total_width * 0.015
            cax_width = total_width * 0.025

            ax_line.set_position([total_x0, pos_line_now.y0, line_width, pos_line_now.height])
            cax = fig.add_axes([total_x0 + line_width + gap, pos_line_now.y0, cax_width, pos_line_now.height])

            draw_year_vs_max_depth_line(ax_line, year_stats, fig=fig, font_size=11, marker_size=60, cax=cax)
        else:
            ax_line.text(0.5, 0.5, "year_depth_stats.csv not found", ha="center", va="center", fontsize=10)

    else:
        ax_bar.text(0.5, 0.5, "No accepted terms.", ha="center", va="center", fontsize=14)

    # Get final layout positions to align panel letters A, B, C
    pos_bar_updated = ax_bar.get_position()
    pos_cov_updated = ax_coverage.get_position()
    pos_line_updated = ax_line.get_position()

    # Perfect vertical alignment using dynamic offset coordinates
    fig.text(pos_bar_updated.x0 - 0.04, pos_bar_updated.y0 + pos_bar_updated.height + 0.01, "A", fontsize=14, fontweight="bold", va="bottom", ha="left")
    fig.text(pos_cov_updated.x0 - 0.14, pos_cov_updated.y0 + pos_cov_updated.height + 0.01, "B", fontsize=14, fontweight="bold", va="bottom", ha="left")
    fig.text(pos_bar_updated.x0 - 0.04, pos_line_updated.y0 + pos_line_updated.height + 0.01, "C", fontsize=14, fontweight="bold", va="bottom", ha="left")

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_name in ("accepted_terms_origin_distribution.png", "mapping_success_and_unmapped_analysis.png"):
        old_path = output_dir / old_name
        if old_path.exists():
            old_path.unlink()
    out_path = output_dir / "mapping_origin_success_summary.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}", flush=True)
    plt.close(fig)


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    configure_fonts()
    args = parse_args()
    G, name_to_uri, uri_to_name = build_ontology_graph(args.ontology_db)
    depth_map = build_depth_map(G, name_to_uri, uri_to_name, args.root_term)
    mapped_count, unmapped_count, relative_levels, relative_levels_per_term, accepted_term_iterations, phrase_max_depths = analyze_mapping_metrics(
        args.json_file, G, name_to_uri, depth_map
    )
    plot_combined_summary(
        mapped_count,
        unmapped_count,
        phrase_max_depths,
        relative_levels_per_term,
        accepted_term_iterations,
        args.output_dir,
        args.json_file,
        args.ontology_db,
        args.root_term,
        no_inner_donut=args.no_inner_donut,
        no_inner_label_2=args.no_inner_label_2,
    )


if __name__ == "__main__":
    main()
