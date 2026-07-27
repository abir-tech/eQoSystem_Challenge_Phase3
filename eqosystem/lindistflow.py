"""LinDistFlow electrical-feasibility validation (post-hoc layer).

The Hamiltonians decide WHAT to energize; this module verifies each energized
island is ELECTRICALLY feasible under the linearized DistFlow model
(Baran & Wu): voltage within +/-5% of nominal at every bus, and every line
below its thermal rating.

Model (standard LinDistFlow on a radial island):
    v_j = v_i - 2 (r_ij P_ij + x_ij Q_ij) / (V_base^2)     [v = |V|^2, per unit]
where P_ij, Q_ij are the real/reactive power flowing on line (i,j), equal to
the sum of net downstream injections. Loss terms are dropped (the LinDistFlow
approximation, accurate to ~1% on feeders of this class).

DER injection: generation is injected at the island's two hubs, split in
proportion to the demand of the post-fault fragment each hub serves (the same
stated capacity-sharing assumption used by the dispatch accounting). The
primary hub of each fragment is the slack bus (v = 1.0 pu).

This is a VALIDATION layer, not a constraint inside the Hamiltonians -- it
converts the limitation "voltage is not modeled" into the measured statement
"N% of island-hours are LinDistFlow-feasible".
"""
import numpy as np
import networkx as nx

from . import grid

V_BASE_KV = 12.66
# ZIP load model coefficients (constant-Z / constant-I / constant-P shares)
ZIP = (0.3, 0.3, 0.4)
INV_S_FACTOR = 1.10   # inverter apparent-power rating vs real-power rating
V_MIN, V_MAX = 0.95, 1.05          # per-unit voltage band
S_BASE_KVA = 1000.0

# W9 -- P-Q capability model. "piecewise" is the challenge's non-convex
# D-shaped capability curve; "circle" is the previous convex apparent-power
# limit, kept for the A/B.
PQ_MODE = "piecewise"


def q_capability(gen_p, s_inv, mode=None):
    """Reactive-power limit at real output gen_p for an s_inv-rated inverter.

    The convex model is the stator circle Q <= sqrt(S^2 - P^2). The piecewise
    model composes four limits:

      standby ("Q at night"), P <= 0.02 S:    Q <= 0.90 S
      low-power stability dip, P < 0.15 S:    Q <= 0.44 S
      field/thermal limit:                    Q <= 0.90 S - 0.25 P
      stator circle / end-region derate:      Q <= sqrt(S^2 - P^2), 4 (S - P)

    A NOTE ON WHAT MAKES THIS NON-CONVEX, because the first version got it
    wrong and a test caught it: a minimum of concave functions is still
    concave, so a boundary built purely from the circle plus linear derates
    leaves the feasible set {(P, Q): 0 <= Q <= cap(P)} CONVEX. The genuinely
    non-convex feature of inverter capability is the low-power dip: full var
    support in standby, REDUCED capability below ~15% loading (control
    stability, IEEE 1547-style var capability categories), full capability
    above. That makes cap(P) non-monotone, so a chord from the standby point
    to the mid-loading boundary passes above the dip -- the feasible set has a
    notch. Segment constants are representative and documented approximations.
    """
    mode = PQ_MODE if mode is None else mode
    circle = float(np.sqrt(max(s_inv ** 2 - gen_p ** 2, 0.0)))
    if mode == "circle":
        return circle
    if gen_p <= 0.02 * s_inv:            # standby: Q-at-night var support
        return min(0.90 * s_inv, circle)
    if gen_p < 0.15 * s_inv:             # low-power control-stability dip
        return min(0.44 * s_inv, circle)
    field = 0.90 * s_inv - 0.25 * gen_p
    cap = min(circle, max(field, 0.0))
    if gen_p > 0.85 * s_inv:             # end-region derate near rated output
        cap = min(cap, max(4.0 * (s_inv - gen_p), 0.0))
    return max(cap, 0.0)

_tables_cache = {}

def _tables():
    """Line R/X/limit lookup tables for the ACTIVE grid (rebuilt on switch)."""
    key = grid.ACTIVE
    if key not in _tables_cache:
        R, X, LIM = {}, {}, {}
        for (u, v, r, x, lim) in grid.LINES:
            R[(u, v)] = R[(v, u)] = r
            X[(u, v)] = X[(v, u)] = x
            LIM[(u, v)] = LIM[(v, u)] = lim
        for (u, v) in grid.TIE_SWITCHES:
            R.setdefault((u, v), 0.5); R.setdefault((v, u), 0.5)
            X.setdefault((u, v), 0.5); X.setdefault((v, u), 0.5)
            LIM.setdefault((u, v), 1000.0); LIM.setdefault((v, u), 1000.0)
        _tables_cache[key] = (R, X, LIM)
    return _tables_cache[key]


def _island_graph(cand, scen, closed_ties):
    g = nx.Graph()
    bs = set(cand["buses"])
    g.add_nodes_from(bs)
    for (u, v, *_r) in grid.LINES:
        if u in bs and v in bs and (u, v) != scen.failed_line and (v, u) != scen.failed_line:
            g.add_edge(u, v)
    for l in closed_ties:
        u, v = grid.TIE_SWITCHES[l]
        if u in bs and v in bs:
            g.add_edge(u, v)
    return g


def check_island(cand, scen, h, served_frac_nc, crit_served, supply_kw,
                 closed_ties=()):
    _R, _X, _LIMIT = _tables()
    """LinDistFlow check of one energized island at hour h.

    Returns dict(feasible, v_min, v_max, n_v_viol, worst_line_pct, n_l_viol).
    """
    lf = scen.load_factor_at(h)     # bucket-varying under scenario-tree mode
    prof = grid.LOAD_PROFILE[h]
    g = _island_graph(cand, scen, closed_ties)
    hubs = cand.get("hubs", [cand["anchor"]])

    # post-fault fragments that contain a hub are energized
    frags = []
    seen = set()
    for hub in hubs:
        if hub in seen or hub not in g:
            continue
        comp = set(nx.node_connected_component(g, hub))
        seen |= comp
        frags.append((hub, comp))

    # served P/Q per bus (critical always if crit_served; non-critical scaled)
    def load_pq(b):
        base_p = grid.LOAD_P[b - 1] * prof * lf
        base_q = grid.LOAD_Q[b - 1] * prof * lf
        if b in grid.CRITICAL_BUSES:
            s = 1.0 if crit_served else 0.0
        else:
            s = served_frac_nc
        return base_p * s, base_q * s

    total_p = sum(load_pq(b)[0] for _, comp in frags for b in comp)
    v_min_all, v_max_all = 1.0, 1.0
    n_v_viol = 0
    worst_line_pct = 0.0
    n_l_viol = 0

    for hub, comp in frags:
        # spanning tree of the fragment rooted at its hub
        tree = nx.bfs_tree(g.subgraph(comp), hub)
        frag_p = sum(load_pq(b)[0] for b in comp)
        # generation injected at the hub, share proportional to fragment demand
        gen_p = supply_kw * (frag_p / total_p) if total_p > 1e-9 else 0.0
        # DERs regulate reactive power: inject Q matching served Q demand
        frag_q = sum(load_pq(b)[1] for b in comp)

        # net injections (loads positive; hub generation negative)
        # inverter P-Q capability: reactive injection limited by sqrt(S^2-P^2)
        # inverter P-Q capability: piecewise non-convex D-curve (W9), see
        # q_capability; the convex circle remains available via PQ_MODE
        s_inv = INV_S_FACTOR * max(gen_p, 1e-9)
        q_cap = q_capability(gen_p, s_inv)
        gen_q = min(frag_q, q_cap)
        inj_p = {b: load_pq(b)[0] for b in comp}
        inj_q = {b: load_pq(b)[1] for b in comp}
        inj_p[hub] -= gen_p
        inj_q[hub] -= gen_q

        # downstream aggregation (post-order) for line flows
        order = list(nx.dfs_postorder_nodes(tree, hub))
        down_p = dict(inj_p)
        down_q = dict(inj_q)
        parent = {c: p for p, c in nx.bfs_edges(tree, hub)}
        for b in order:
            if b in parent:
                down_p[parent[b]] += down_p[b]
                down_q[parent[b]] += down_q[b]

        # voltages: v = |V|^2 pu, hub = 1.0; propagate root->leaf
        v = {hub: 1.0}
        vb2 = (V_BASE_KV * 1e3) ** 2
        for p, c in nx.bfs_edges(tree, hub):
            r = _R[(p, c)]; x = _X[(p, c)]
            P = down_p[c] * 1e3   # W
            Q = down_q[c] * 1e3   # var
            v[c] = v[p] - 2.0 * (r * P + x * Q) / vb2
            # line loading vs thermal limit (kVA approx, |S| = sqrt(P^2+Q^2))
            s_kva = float(np.hypot(down_p[c], down_q[c]))
            pct = 100.0 * s_kva / _LIMIT[(p, c)]
            worst_line_pct = max(worst_line_pct, pct)
            if s_kva > _LIMIT[(p, c)]:
                n_l_viol += 1

        vmags = {b: float(np.sqrt(max(val, 0.0))) for b, val in v.items()}
        # pass 2: voltage-dependent (ZIP) loads -- P(V)=P0(z*V^2 + i*V + p)
        zc, ic, pc = ZIP
        for b in comp:
            f = zc * vmags.get(b, 1.0) ** 2 + ic * vmags.get(b, 1.0) + pc
            inj_p[b] *= f; inj_q[b] *= f
        inj_p[hub] = inj_p[hub]  # generation unchanged
        down_p = dict(inj_p); down_q = dict(inj_q)
        for b in order:
            if b in parent:
                down_p[parent[b]] += down_p[b]; down_q[parent[b]] += down_q[b]
        v = {hub: 1.0}
        for p, c in nx.bfs_edges(tree, hub):
            r = _R[(p, c)]; x = _X[(p, c)]
            v[c] = v[p] - 2.0 * (r * down_p[c] * 1e3 + x * down_q[c] * 1e3) / vb2
            s_kva = float(np.hypot(down_p[c], down_q[c]))
            worst_line_pct = max(worst_line_pct, 100.0 * s_kva / _LIMIT[(p, c)])
            if s_kva > _LIMIT[(p, c)]:
                n_l_viol += 1
        vmags = {b: float(np.sqrt(max(val, 0.0))) for b, val in v.items()}
        v_min_all = min(v_min_all, min(vmags.values()))
        v_max_all = max(v_max_all, max(vmags.values()))
        n_v_viol += sum(1 for m in vmags.values() if m < V_MIN or m > V_MAX)

    return dict(feasible=(n_v_viol == 0 and n_l_viol == 0),
                v_min=v_min_all, v_max=v_max_all,
                n_v_viol=n_v_viol, n_l_viol=n_l_viol,
                worst_line_pct=worst_line_pct)
