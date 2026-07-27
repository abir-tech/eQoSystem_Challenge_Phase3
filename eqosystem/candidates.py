"""Island candidate generation.

The challenge states candidates "need not be distinct", so we generate an
OVERLAPPING candidate pool by running spectral clustering at several
resolutions (k = 3, 4, 5) on the electrical-distance Laplacian. Overlap is
what makes the downstream islanding decision a genuinely coupled QUBO
instead of a set of independent yes/no choices.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.cluster.vq import kmeans2

from . import grid


def _spectral_partition(k: int, seed: int = 42):
    n = grid.N_BUSES
    adj = np.zeros((n, n))
    for (u, v, r, x, lim) in grid.LINES:
        w = 1.0 / (r + 1e-6)           # electrical proximity
        adj[u - 1, v - 1] = adj[v - 1, u - 1] = w
    lap = np.diag(adj.sum(axis=1)) - adj
    _, vecs = eigh(lap)
    coords = vecs[:, 1:k + 1]
    _, labels = kmeans2(coords, k, iter=100, seed=seed, minit="++")
    parts = {}
    for bus_idx, lab in enumerate(labels):
        parts.setdefault(int(lab), []).append(bus_idx + 1)
    # drop the trivial substation-only fragment, require connectivity
    g = grid.build_graph(include_ties=True)
    out = []
    for buses in parts.values():
        sub = g.subgraph(buses)
        if len(buses) >= 3 and __import__("networkx").is_connected(sub):
            out.append(sorted(buses))
    return out


def generate(resolutions=(3, 4, 5), seed: int = 42):
    """Return list of candidate dicts with load, critical flag, anchor bus."""
    pool, seen = [], set()
    for k in resolutions:
        for buses in _spectral_partition(k, seed):
            key = tuple(buses)
            if key in seen:
                continue
            seen.add(key)
            bus_arr = [b for b in buses if b != 1]  # substation not an island member
            load = float(sum(grid.LOAD_P[b - 1] for b in bus_arr))
            crit = sorted(set(bus_arr) & grid.CRITICAL_BUSES)
            anchor = max(bus_arr, key=lambda b: grid.LOAD_P[b - 1])
            # secondary DER hub: geographic spread for internal N-1 resilience.
            # Pick the bus maximizing load x (1 + hops from anchor) so an
            # internal line failure rarely severs ALL generation.
            g = _subgraph(bus_arr)
            dist = _hops_from(g, anchor)
            hub2 = max(bus_arr, key=lambda b: grid.LOAD_P[b - 1] * (1 + dist.get(b, 0)))
            pool.append({
                "id": len(pool),
                "buses": bus_arr,
                "load_kw": load,
                "critical": crit,
                "anchor": anchor,
                "hubs": sorted({anchor, hub2}),
                "customers": int(sum(grid.CUSTOMERS[b - 1] for b in bus_arr)),
            })
    # ---- feasibility pruning + dedup (keeps the design Hamiltonian on-budget)
    from . import hamiltonians as ham
    max_firm = sum(ham.asset_umax(k) * ham.ASSETS[k]["kw"] * ham.ASSETS[k]["firm"]
                   for k in ham.ASSET_KEYS)
    feas = [cd for cd in pool if ham.design_demand(cd) <= max_firm]
    feas.sort(key=lambda cd: cd["load_kw"])
    kept = []
    for cd in feas:                       # drop near-duplicates (Jaccard > 0.85)
        s1 = set(cd["buses"])
        if all(len(s1 & set(k2["buses"])) / len(s1 | set(k2["buses"])) <= 0.85
               for k2 in kept):
            kept.append(cd)
    for i, cd in enumerate(kept):
        cd["id"] = i
    return kept



def _subgraph(bus_arr):
    import networkx as nx
    g = nx.Graph()
    bs = set(bus_arr)
    g.add_nodes_from(bs)
    for (u, v, *_r) in grid.LINES:
        if u in bs and v in bs:
            g.add_edge(u, v)
    return g


def _hops_from(g, src):
    import networkx as nx
    try:
        return nx.single_source_shortest_path_length(g, src)
    except Exception:
        return {}


def overlap_matrix(pool):
    m = len(pool)
    O = np.zeros((m, m))
    for i in range(m):
        si = set(pool[i]["buses"])
        for j in range(i + 1, m):
            inter = si & set(pool[j]["buses"])
            O[i, j] = O[j, i] = sum(grid.LOAD_P[b - 1] for b in inter)
    return O
