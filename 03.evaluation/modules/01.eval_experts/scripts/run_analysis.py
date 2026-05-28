from __future__ import annotations
import argparse
import glob
import os

import networkx as nx
import pandas as pd
import rdflib
from rdflib.namespace import RDFS


def arg_parser() -> argparse.Namespace:
    """
    Description:
    Parse command line arguments for the expert expansion analysis script.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Expand expert annotation terms using an ontology hierarchy."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing ToxValLlm_expert_*_ontology_<onto>.csv files.",
    )
    parser.add_argument(
        "--owl_file",
        required=True,
        help="Path to the ontology TTL/OWL file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where Expanded_*.csv files will be saved.",
    )
    parser.add_argument(
        "--ontology",
        required=True,
        help="Ontology short name (e.g. mp, hpath). Used to match input files.",
    )
    parser.add_argument(
        "--root_term",
        default=None,
        help=(
            "Label of the root node to exclude from expanded terms "
            "(e.g. 'mammalian phenotype'). If omitted no term is excluded."
        ),
    )
    return parser.parse_args()


def build_graph(owl_file: str) -> tuple[nx.DiGraph, dict, dict]:
    """
    Description:
    Parse an owl ontology file and build a subclass hierarchy directed graph.

    Args:
        owl_file (str): Path to the owl/ttl ontology file.

    Returns:
        tuple[nx.DiGraph, dict, dict]: (Hierarchy directed graph, name to URI dictionary, URI to name dictionary).
    """
    print(f"Parsing ontology: {owl_file}")
    g = rdflib.Graph()
    g.parse(owl_file, format="turtle")

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

    return G, name_to_uri, uri_to_name


def get_ancestors_names(
    term_name: str,
    G: nx.DiGraph,
    name_to_uri: dict,
    uri_to_name: dict,
    root_uri: rdflib.term.URIRef | None
) -> set[str]:
    """
    Description:
    Retrieve all ancestors of a term in the ontology graph.

    Args:
        term_name (str): The name of the term.
        G (nx.DiGraph): The subclass hierarchy ontology graph.
        name_to_uri (dict): Map of term name to URI.
        uri_to_name (dict): Map of URI to term name.
        root_uri (rdflib.term.URIRef | None): Optional URI of the root node to ignore.

    Returns:
        set[str]: Set of ancestor term names.
    """
    term_name = term_name.lower().strip()
    if term_name not in name_to_uri:
        return set()
    uri = name_to_uri[term_name]
    ancestors = nx.descendants(G, uri)
    ancestor_names = set()
    for anc in ancestors:
        if anc == root_uri:
            continue
        if anc in uri_to_name:
            ancestor_names.add(uri_to_name[anc])
    return ancestor_names


def process_terms(
    term_str: str | float,
    G: nx.DiGraph,
    name_to_uri: dict,
    uri_to_name: dict,
    root_uri: rdflib.term.URIRef | None
) -> set[str]:
    """
    Description:
    Process and expand terms from a pipe-delimited string by fetching their ancestors.

    Args:
        term_str (str | float): Pipe-delimited string of terms.
        G (nx.DiGraph): The subclass hierarchy ontology graph.
        name_to_uri (dict): Map of term name to URI.
        uri_to_name (dict): Map of URI to term name.
        root_uri (rdflib.term.URIRef | None): Optional URI of the root node to ignore.

    Returns:
        set[str]: Set of expanded term names.
    """
    if pd.isna(term_str):
        return set()
    terms = [t.strip().lower() for t in str(term_str).split("|")]
    valid_terms = set(t for t in terms if t not in ["none", "-", "unknown", ""])
    expanded = set(valid_terms)
    for t in valid_terms:
        expanded.update(get_ancestors_names(t, G, name_to_uri, uri_to_name, root_uri))
    return expanded


def main() -> None:
    """
    Description:
    Main execution function to load configurations, build ontology graph, and expand terms for all expert CSV files.

    Args:
        None

    Returns:
        None
    """
    args = arg_parser()

    os.makedirs(args.output_dir, exist_ok=True)

    G, name_to_uri, uri_to_name = build_graph(args.owl_file)

    root_uri = None
    root_name = None
    if args.root_term:
        root_name = args.root_term.lower().strip()
        root_uri = name_to_uri.get(root_name)

    pattern = os.path.join(
        args.input_dir, f"ToxValLlm_expert_*_ontology_{args.ontology}.csv"
    )
    files = glob.glob(pattern)

    if not files:
        print(f"No input files found matching: {pattern}")
        return

    for f in files:
        fname = os.path.basename(f)
        expert_name = fname.replace("ToxValLlm_expert_", "").replace(
            f"_ontology_{args.ontology}.csv", ""
        )
        print(f"Processing: {fname}")
        df = pd.read_csv(f)

        df["Selected_Candidates_Expanded"] = df["Selected_Candidates"].apply(
            lambda x: process_terms(x, G, name_to_uri, uri_to_name, root_uri)
        )
        df["Manual_Terms_Expanded"] = df["Manual_Terms"].apply(
            lambda x: process_terms(x, G, name_to_uri, uri_to_name, root_uri)
        )

        df["All_Terms"] = df.apply(
            lambda row: row["Selected_Candidates_Expanded"].union(
                row["Manual_Terms_Expanded"]
            ),
            axis=1,
        )
        if root_name:
            df["All_Terms"] = df["All_Terms"].apply(
                lambda x: x - {root_name} if root_name in x else x
            )

        df["All_Terms_Str"] = df["All_Terms"].apply(lambda x: "|".join(sorted(x)))

        out_df = df[["Index", "Phrase", "All_Terms_Str"]]
        out_path = os.path.join(args.output_dir, f"Expanded_{expert_name}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    print("Expansion complete.")


if __name__ == "__main__":
    main()
