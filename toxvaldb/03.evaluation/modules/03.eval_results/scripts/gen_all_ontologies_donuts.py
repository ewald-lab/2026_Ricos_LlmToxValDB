#!/usr/bin/env python3
"""
Generate a combined figure with:
  Panel A – origin + success donuts for ALL ontologies
  Panel B – MP year vs max ontology depth line plot
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import networkx as nx
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_results import draw_year_vs_max_depth_line


# ── Ontology configs ─────────────────────────────────────────────────────────
ONTOLOGIES = [
    {"name": "mp",    "display": "MP",    "no_inner_donut": False, "inner_labels": {"1", "2"}},
    {"name": "aop",   "display": "AOP",   "no_inner_donut": True,  "inner_labels": set()},
    {"name": "txpo",  "display": "TXPO",  "no_inner_donut": True,  "inner_labels": set()},
    {"name": "hpath", "display": "HPATH", "no_inner_donut": False, "inner_labels": {"1"}},
    {"name": "hp",    "display": "HP",    "no_inner_donut": False, "inner_labels": {"1", "2"}},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def configure_fonts(repo_root: Path) -> None:
    font_dir = repo_root / "assets" / "fonts" / "ibm-plex-sans"
    for font_path in font_dir.glob("*.ttf"):
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"],
    })


def build_ontology_graph(ontology_db: Path) -> tuple[nx.DiGraph, dict[str, str]]:
    G = nx.DiGraph()
    name_to_uri = {}
    with sqlite3.connect(str(ontology_db)) as conn:
        rows = list(conn.execute("SELECT uri, name, sanitized_name, parent FROM embeddings"))
        for uri, _name, sanitized_name, _parent_json in rows:
            G.add_node(uri)
            term_name = (sanitized_name or _name or "").lower().strip()
            if term_name:
                name_to_uri[term_name] = uri
        for uri, _name, _sanitized_name, parent_json in rows:
            try:
                parents = json.loads(parent_json) if parent_json else []
            except json.JSONDecodeError:
                parents = []
            for parent_name in parents:
                parent_uri = name_to_uri.get(parent_name.lower().strip())
                if parent_uri:
                    G.add_edge(uri, parent_uri)
    return G, name_to_uri


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


def analyze_mapping_metrics(json_file: Path, G: nx.DiGraph, name_to_uri: dict) -> tuple:
    mapped_count = 0
    unmapped_count = 0
    relative_levels_per_phrase = []
    relative_levels_per_term = []
    accepted_term_iterations = []
    undirected_G = G.to_undirected(as_view=True)
    distance_cache = {}

    def min_distance_from_initial_rag(term, initial_terms):
        term_uri = name_to_uri.get(term)
        if term_uri is None:
            return None
        cache_key = (term, tuple(sorted(initial_terms)))
        if cache_key in distance_cache:
            return distance_cache[cache_key]
        distances = []
        for it in initial_terms:
            iu = name_to_uri.get(it)
            if iu is None:
                continue
            try:
                distances.append(nx.shortest_path_length(undirected_G, iu, term_uri))
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

    return mapped_count, unmapped_count, relative_levels_per_phrase, relative_levels_per_term, accepted_term_iterations


def draw_origin_donut(ax, accepted_term_iterations, relative_levels_per_term, no_inner_donut, inner_labels_to_show=None):
    if inner_labels_to_show is None:
        inner_labels_to_show = {"1", "2"}
    iter_counts = Counter(accepted_term_iterations)
    iter_1_rag = iter_counts.get(1, 0)
    iter_2 = iter_counts.get(2, 0)
    iter_3 = iter_counts.get(3, 0)
    iter_4 = iter_counts.get(4, 0)
    iter_5 = iter_counts.get(5, 0)
    iter_other = sum(count for k, count in iter_counts.items() if k >= 6)
    total_accepted = iter_1_rag + iter_2 + iter_3 + iter_4 + iter_5 + iter_other
    graph_total = iter_2 + iter_3 + iter_4 + iter_5 + iter_other

    if not total_accepted:
        ax.text(0.5, 0.5, "No accepted terms.", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return

    outer_sizes = [iter_1_rag, graph_total]
    outer_colors = ["#4C78A8", "#55A868"]

    level_counts = Counter(relative_levels_per_term)
    level_labels = [str(level) for level in range(1, 8)] + ["8+"]
    level_sizes = [level_counts.get(level, 0) for level in range(1, 8)]
    level_sizes.append(sum(count for level, count in level_counts.items() if level >= 8))
    level_colors = ["#2A9D8F", "#3BAF91", "#52B788", "#74C69D", "#95D5B2", "#B7E4C7", "#D8F3DC", "#EAF7EC"]
    level_total = sum(level_sizes)
    level_scale = graph_total / level_total if level_total else 0
    level_sizes = [size * level_scale for size in level_sizes]
    inner_raw = [iter_1_rag] + level_sizes
    inner_raw_labels = ["RAG"] + level_labels
    inner_raw_colors = ["white"] + level_colors
    nonzero_levels = [
        (label, size, color)
        for label, size, color in zip(inner_raw_labels, inner_raw, inner_raw_colors)
        if size > 0
    ]
    inner_labels = [lbl for lbl, _s, _c in nonzero_levels]
    inner_sizes = [s for _l, s, _c in nonzero_levels]
    inner_colors = [c for _l, _s, c in nonzero_levels]

    def outer_autopct(pct):
        return f"{pct:.1f}%" if pct >= 5.0 else ""

    wedges_out, _texts_out, autotexts_out = ax.pie(
        outer_sizes, labels=None, autopct=outer_autopct, startangle=140,
        colors=outer_colors, textprops=dict(fontweight="bold", fontsize=9),
        pctdistance=0.80, radius=1.0,
        wedgeprops=dict(width=0.4, edgecolor="w", linewidth=2),
    )
    plt.setp(autotexts_out, size=9, weight="bold", color="white")
    for autotext, wedge in zip(autotexts_out, wedges_out):
        angle = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
        radius = 0.80
        autotext.set_position((radius * math.cos(angle), radius * math.sin(angle)))

    if not no_inner_donut:
        wedges_in, _texts_in = ax.pie(
            inner_sizes, startangle=140, colors=inner_colors, radius=0.60,
            wedgeprops=dict(width=0.4, edgecolor="w", linewidth=2),
        )
        for label, wedge in zip(inner_labels, wedges_in):
            if label not in inner_labels_to_show:
                continue
            angle = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
            r = 0.40
            ax.text(r * math.cos(angle), r * math.sin(angle), label,
                    ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    ax.axis("equal")
    ax.axis("off")


def add_panel_label(ax, label: str, x: float = -0.05, y: float = 1.04) -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def draw_success_donut(ax, mapped_count, unmapped_count):
    sizes = [mapped_count, unmapped_count]
    colors = ["#34A853", "#EA4335"]

    def autopct(pct):
        return f"{pct:.1f}%" if pct >= 5.0 else ""

    _wedges, _texts, autotexts = ax.pie(
        sizes, labels=None, autopct=autopct, startangle=140,
        colors=colors, textprops=dict(fontweight="bold", fontsize=9),
        pctdistance=0.8, wedgeprops=dict(width=0.4, edgecolor="w", linewidth=2),
    )
    plt.setp(autotexts, size=9, weight="bold", color="white")
    ax.axis("equal")
    ax.axis("off")


def load_mp_year_stats(output_dir: Path) -> pd.DataFrame:
    stats_csv = output_dir / "mp" / "year_depth_stats.csv"
    if not stats_csv.is_file():
        raise FileNotFoundError(
            f"Missing MP year stats cache: {stats_csv}\n"
            "Run eval_results.py --depth_only for MP first."
        )
    return pd.read_csv(stats_csv)


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    args = parse_args()
    configure_fonts(args.repo_root)

    n = len(ONTOLOGIES)
    MODEL = "Qwen3.6-35B-A3B-FP8"
    year_stats = load_mp_year_stats(args.output_dir)
    print(f"Loaded MP year stats for panel B ({len(year_stats):,} years)", flush=True)

    fig = plt.figure(figsize=(3.5 * n, 11.5))
    # GridSpec for donuts (first 2 rows) - shifted further to the left (left=0.04, right=0.88)
    gs_donuts = gridspec.GridSpec(
        2, n,
        figure=fig,
        hspace=0.02,
        wspace=0.02,
        left=0.04,
        right=0.88,
        bottom=0.41,
        top=0.92,
    )
    # GridSpec for depth plot (bottom row) - shifted further left, and raised higher (bottom=0.17, top=0.37)
    gs_depth = gridspec.GridSpec(
        1, n,
        figure=fig,
        left=0.04,
        right=0.88,
        bottom=0.17,
        top=0.37,
    )

    ax_origin_first = None
    ax_success_first = None
    ax_origin_last = None

    for col, ont in enumerate(ONTOLOGIES):
        name = ont["name"]
        display = ont["display"]
        no_inner = ont["no_inner_donut"]
        inner_labels = ont["inner_labels"]

        json_file = (args.repo_root / "01.text_to_onto" / "modules"
                     / "00.tox_effect_to_onto" / "outputs"
                     / f"ToxValDB_Reasoning_{name}_{MODEL}.json")
        db_file = (args.repo_root / "00.data_preparation" / "modules"
                   / "02.prepare_ontologies" / "outputs" / "qwen" / f"{name}.db")

        print(f"[{col+1}/{n}] Loading {display} ({name}) ...", flush=True)
        G, name_to_uri = build_ontology_graph(db_file)
        mapped_count, unmapped_count, _lvl_phrase, lvl_term, iterations = \
            analyze_mapping_metrics(json_file, G, name_to_uri)
        print(f"  mapped={mapped_count:,}  unmapped={unmapped_count:,}", flush=True)

        ax_origin = fig.add_subplot(gs_donuts[0, col])
        if col == 0:
            ax_origin_first = ax_origin
        if col == n - 1:
            ax_origin_last = ax_origin
            
        draw_origin_donut(ax_origin, iterations, lvl_term, no_inner_donut=no_inner, inner_labels_to_show=inner_labels)
        ax_origin.set_title(display, fontsize=14, fontweight="bold", pad=8)

        ax_success = fig.add_subplot(gs_donuts[1, col])
        if col == 0:
            ax_success_first = ax_success
            
        draw_success_donut(ax_success, mapped_count, unmapped_count)
        del G, name_to_uri, iterations, lvl_term
        gc.collect()

    ax_depth = fig.add_subplot(gs_depth[0, :])
    draw_year_vs_max_depth_line(ax_depth, year_stats, fig=fig, font_size=12)
    ax_depth.set_xlabel("Study Year", fontsize=12, fontweight="bold")
    ax_depth.set_ylabel("Ontology Depth", fontsize=12, fontweight="bold")
    ax_depth.legend(fontsize=12)
    del year_stats

    # ── Post-hoc Ajuste 1: Hacer los puntos de Panel B más pequeños ─────────
    for line in ax_depth.lines:
        line.set_markersize(5)  # Slightly larger marker size
    for coll in ax_depth.collections:
        if isinstance(coll, matplotlib.collections.PathCollection):
            coll.set_sizes([30])  # Slightly larger scatter circles

    # ── Post-hoc Ajuste 2: Centrar perfectamente el plot B con los donuts ────
    pos_left = ax_origin_first.get_position()
    pos_right = ax_origin_last.get_position()
    pos_depth = ax_depth.get_position()

    ideal_x0 = pos_left.x0
    ideal_x1 = pos_right.x1
    ideal_width = ideal_x1 - ideal_x0

    # Forzar al plot B a ocupar simétricamente el mismo ancho real que los donuts, pero desplazado 0.010 a la derecha para más espacio
    delta_x = 0.010
    ax_depth.set_position([ideal_x0 + delta_x, pos_depth.y0, ideal_width - delta_x, pos_depth.height])

    # Align Ontology Depth y-label with Mapping Origin/Success labels (which are at -0.08)
    # Using the exact mathematical alignment formula based on the shifted axis coordinates
    x_axes_depth = (-0.035 * pos_left.width - delta_x) / (ideal_width - delta_x)
    ax_depth.yaxis.set_label_coords(x_axes_depth, 0.5)

    # Si hay una colorbar generada en el panel B, moverla a la derecha alineada con la leyenda (un pelin mas cerca, 0.012 de offset)
    if len(fig.axes) > (2 * n + 1):
        cbar_ax = fig.axes[-1]
        cbar_ax.set_position([ideal_x1 + 0.012, pos_depth.y0, 0.015, pos_depth.height])

    # ── Leyenda unificada arriba a la derecha (junto a HP Origin) ────────────
    legend_elements = [
        mpatches.Patch(color="#4C78A8", label="Origin: RAG"),
        mpatches.Patch(color="#55A868", label="Origin: Hierarchy"),
        mpatches.Patch(color="#2A9D8F", label="Hierarchy Depth 1"),
        mpatches.Patch(color="#3BAF91", label="Hierarchy Depth 2"),
        mpatches.Patch(color="#34A853", label="Success: Mapped"),
        mpatches.Patch(color="#EA4335", label="Success: Unmapped"),
    ]

    ax_origin_last.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5), # Centered on the center of the top row donuts
        fontsize=12,
        title=None,
        frameon=True,
        shadow=False,
        facecolor="white",
        edgecolor="#CCCCCC",
    )

    # Etiquetas de fila perfectamente alineadas al margen izquierdo real (offset reduced to -0.08 for less distance)
    ax_origin_first.annotate(
        "Mapping Origin", xy=(0, 0.5), xytext=(-0.08, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        fontsize=12, fontweight="bold", rotation=90, va="center", ha="center",
    )
    ax_success_first.annotate(
        "Mapping Success", xy=(0, 0.5), xytext=(-0.08, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        fontsize=12, fontweight="bold", rotation=90, va="center", ha="center",
    )

    # Align panel labels A and B perfectly using the same multiplier (offset reduced to -0.12)
    add_panel_label(ax_origin_first, "A", x=-0.12, y=1.04)
    x_axes_B = (-0.12 * pos_left.width - delta_x) / (ideal_width - delta_x)
    add_panel_label(ax_depth, "B", x=x_axes_B, y=1.04)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "all_ontologies_donuts.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    print(f"\nSaved: {out_path}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()