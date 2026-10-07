#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.append(os.path.expanduser("~/.local/lib/python3.12/site-packages"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import networkx as nx
import seaborn as sns

try:
    import squarify
except ImportError:
    squarify = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", required=True, type=Path)
    parser.add_argument("--ontology_db", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--root_term", required=True)
    return parser.parse_args()


def configure_fonts() -> None:
    font_dir = Path(__file__).resolve().parents[4] / "02.expert_annotation" / "modules" / "01.create_form" / "assets" / "fonts" / "ibm-plex-sans"
    for font_path in font_dir.glob("*.ttf"):
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"],
    })


def build_ontology_graph(ontology_db: Path):
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
    print(
        f"WARNING: root term '{root_term}' not found; created synthetic root "
        f"covering {len(roots)} top-level nodes.",
        file=sys.stderr,
        flush=True,
    )
    return root_uri


def build_term_to_top_branch(G, name_to_uri, uri_to_name, root_term: str):
    root_uri = ensure_root_node(G, name_to_uri, uri_to_name, root_term)
    top_branches = list(G.predecessors(root_uri))
    G_rev = G.reverse()
    term_to_top_branch = {}

    for branch_uri in top_branches:
        branch_name = uri_to_name.get(branch_uri, str(branch_uri))
        descendants = nx.descendants(G_rev, branch_uri)
        descendants.add(branch_uri)
        for desc in descendants:
            desc_name = uri_to_name.get(desc)
            if desc_name and desc_name not in term_to_top_branch:
                term_to_top_branch[desc_name] = branch_name
    return term_to_top_branch


def iter_json_object_items(path: Path):
    decoder = json.JSONDecoder()
    buffer = ""
    with open(path, "r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return
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


def generate_treemap(json_file: Path, term_to_top_branch: dict, output_dir: Path):
    if squarify is None:
        print("WARNING: squarify not installed; skipping treemap.", file=sys.stderr)
        return

    branch_counts = {}
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
        if not phrase_branches:
            phrase_branches.add("Other/Unknown")

        for branch in phrase_branches:
            branch_counts[branch] = branch_counts.get(branch, 0) + 1

    sorted_branches = sorted(branch_counts.items(), key=lambda x: x[1], reverse=True)
    branches = [x[0] for x in sorted_branches if x[0] != "Other/Unknown"]
    counts = [x[1] for x in sorted_branches if x[0] != "Other/Unknown"]

    if not counts:
        print("No mapped top branches found; skipping treemap.", file=sys.stderr)
        return

    labels = []
    for branch, count in zip(branches, counts):
        if count >= 234:
            wrapped = (
                branch.replace(" phenotype", "")
                .replace(" system", "\nsystem")
                .replace("/", "/\n")
            )
            labels.append(f"{wrapped}\n({count:,})")
        else:
            labels.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 10))
    squarify.plot(
        sizes=counts,
        label=labels,
        alpha=0.85,
        color=sns.color_palette("Spectral", len(counts)),
        text_kwargs={"fontsize": 9, "weight": "bold"},
        ax=ax,
    )
    ax.axis("off")

    out_path = output_dir / "phenotypic_landscape_unique_treemap.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    configure_fonts()
    G, name_to_uri, uri_to_name = build_ontology_graph(args.ontology_db)
    term_to_top_branch = build_term_to_top_branch(
        G, name_to_uri, uri_to_name, args.root_term
    )
    generate_treemap(args.json_file, term_to_top_branch, args.output_dir)


if __name__ == "__main__":
    main()
