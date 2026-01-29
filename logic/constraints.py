from collections import defaultdict
import numpy as np
import networkx as nx


## g = (v, e), e = co-occurrence
def build_graph(f):
    g = nx.Graph()
    for tokens in f:
        u = list(set(tokens))
        g.add_nodes_from(u)
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                g.add_edge(u[i], u[j])
    return g


def graph_stats(f):
    occ = defaultdict(set)
    for i, tokens in enumerate(f):
        for t in set(tokens):
            occ[t].add(i)

    deg_t = np.mean([len(v) for v in occ.values()]) if occ else 0
    deg_s = np.mean([len(set(tokens)) for tokens in f]) if f else 0

    g = build_graph(f)
    n_comp = nx.number_connected_components(g) if g.number_of_nodes() else 0

    try:
        tw, _ = nx.approximation.treewidth_min_fill_in(g)
    except:
        tw = None

    return {
        "n_comp": n_comp,
        "deg_t": float(deg_t),
        "deg_s": float(deg_s),
        "tw": tw,
        "v": g.number_of_nodes(),
        "e": g.number_of_edges(),
    }
