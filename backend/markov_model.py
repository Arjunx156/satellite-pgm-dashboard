"""Sections 6-7 of the notebook: Markov network and its conditional independencies.

The Markov network is a structural, undirected view of the dependencies. It is not used to
compute the TAN anomaly probability.
"""

import itertools

import networkx as nx
import pandas as pd

from .tan_model import mutual_info


def learn_markov_tree(data, nodes):
    G = nx.Graph()
    for a, b in itertools.combinations(nodes, 2):
        G.add_edge(a, b, weight=mutual_info(data, a, b))
    mn = nx.maximum_spanning_tree(G)
    assert (not mn.is_directed()) and nx.is_tree(mn)
    return mn


def separated(graph, x, y, Z):
    H = graph.copy()
    H.remove_nodes_from(Z)
    return not nx.has_path(H, x, y)


def independence_table(mn, data, nodes, limit=3):
    triples = [(x, nx.shortest_path(mn, x, y)[1], y) for x, y in itertools.combinations(nodes, 2)
               if len(nx.shortest_path(mn, x, y)) == 3]
    triples.sort(key=lambda t: "anomaly" not in (t[0], t[2]))        # statements involving Anomaly first
    rows = [{"Independence statement": f"{x} ⊥ {y} | {z}",
             "Separated in graph": separated(mn, x, y, [z]),
             "I(X;Y)": mutual_info(data, x, y),
             "I(X;Y | Z)": mutual_info(data, x, y, z)} for x, z, y in triples[:limit]]
    return pd.DataFrame(rows)


def markov_blanket(mn, node="anomaly"):
    return sorted(mn.neighbors(node))
