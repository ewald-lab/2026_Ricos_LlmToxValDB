#!/usr/bin/env python3
"""
Generate only the ontology tree PNG (PART C).
Standalone version to avoid memory issues from the full eval pipeline.

Usage:
    python gen_ontology_tree.py \
        --json_file  <ToxValDB_Reasoning_mp_*.json> \
        --owl_file   <mp.ttl> \
        --output_dir <outputs/> \
        --root_term  "mammalian phenotype" \
        --threshold  50
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import rdflib
from rdflib.namespace import RDFS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json_file", required=True, type=Path)
    p.add_argument("--owl_file", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--root_term", default="mammalian phenotype")
    p.add_argument(
        "--threshold",
        type=int,
        default=500,
        help="Minimum propagated number of cases required to keep a node in the ontology tree.",
    )
    return p.parse_args()


def build_ontology_graph(owl_file: Path):
    print(f"Parsing ontology: {owl_file}", flush=True)
    g = rdflib.Graph()
    g.parse(str(owl_file), format="turtle")

    G = nx.DiGraph()
    name_to_uri = {}
    uri_to_name = {}

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


def ensure_root_node(G, name_to_uri, uri_to_name, root_term):
    root_name = root_term.lower().strip()
    root_uri = name_to_uri.get(root_name)
    if root_uri:
        return root_uri

    root_uri = rdflib.URIRef(f"urn:synthetic-root:{root_name.replace(' ', '_')}")
    roots = [node for node in G.nodes() if G.out_degree(node) == 0]
    if len(roots) == 1 and uri_to_name.get(roots[0], "").lower() == "thing":
        roots = list(G.predecessors(roots[0]))
    for node in roots:
        G.add_edge(node, root_uri)

    name_to_uri[root_name] = root_uri
    uri_to_name[root_uri] = root_name
    print(
        f"WARNING: Root term '{root_term}' not found; created synthetic root "
        f"covering {len(roots)} top-level nodes.",
        file=sys.stderr,
        flush=True,
    )
    return root_uri


def run_ontology_tree(json_file, G, name_to_uri, uri_to_name,
                      root_term, threshold, output_dir):
    print("\n--- Ontology Tree ---", flush=True)

    root_uri = ensure_root_node(G, name_to_uri, uri_to_name, root_term)

    G_rev = G.reverse()
    reachable = nx.descendants(G_rev, root_uri)
    reachable.add(root_uri)
    G_rev = G_rev.subgraph(reachable).copy()

    print("Loading JSON ...", flush=True)
    with open(json_file) as f:
        data = json.load(f)

    dag_hits = {n: 0 for n in G_rev.nodes()}
    for phrase, info in data.items():
        selected = info.get("selected", [])
        phrase_uris = set()
        for term_with_prefix in selected:
            term = term_with_prefix.split(":", 1)[-1].lower().strip()
            uri = name_to_uri.get(term)
            if uri and uri in G_rev:
                phrase_uris.add(uri)
        all_ancestors = set()
        for u in phrase_uris:
            all_ancestors.add(u)
            if u in G:
                all_ancestors.update(nx.descendants(G, u))
        for u in all_ancestors:
            if u in dag_hits:
                dag_hits[u] += 1

    print(f"Pruning nodes with < {threshold} hits ...", flush=True)
    keep_nodes = {n for n, cnt in dag_hits.items() if cnt >= threshold}
    keep_nodes.add(root_uri)
    G_pruned = G_rev.subgraph(keep_nodes).copy()
    print(f"  Pruned graph: {G_pruned.number_of_nodes()} nodes, "
          f"{G_pruned.number_of_edges()} edges", flush=True)

    counts_nonroot = [dag_hits[n] for n in G_pruned.nodes() if n != root_uri]
    max_hits = max(counts_nonroot) if counts_nonroot else threshold
    cmap = plt.get_cmap("YlOrRd")
    norm = mcolors.LogNorm(vmin=threshold, vmax=max_hits)

    def get_color(count):
        if count == 0:
            return "#ffffff"
        return mcolors.to_hex(cmap(norm(count)))

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

    out_png = output_dir / "ontology_tree.png"
    print(f"Rendering ontology tree PNG: {out_png}", flush=True)
    try:
        render_source = dot_source.encode()
        try:
            unflattened = subprocess.run(
                ["unflatten", "-l", "4", "-c", "4"],
                input=render_source,
                capture_output=True,
                check=True,
            )
            render_source = unflattened.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        subprocess.run(
            ["dot", "-Tpng", "-o", str(out_png)],
            input=render_source,
            capture_output=True,
            check=True,
        )
        print(f"  Saved: {out_png}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR running Graphviz dot: {e.stderr.decode()}", file=sys.stderr)
    except FileNotFoundError:
        print("ERROR: 'dot' (Graphviz) not found in PATH.", file=sys.stderr)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    G, name_to_uri, uri_to_name = build_ontology_graph(args.owl_file)
    run_ontology_tree(args.json_file, G, name_to_uri, uri_to_name,
                      args.root_term, args.threshold, args.output_dir)
    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
