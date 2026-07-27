"""Scalable exact certification for binary Hamiltonians (E2 beyond m ~ 20).

WHY THIS EXISTS
---------------
The submission's strongest claim is "solved to CERTIFIED global optimality":
every islanding QUBO is checked against an exact reference. That reference is
exhaustive enumeration, which is 2^m -- fine at m = 3, hopeless past m ~ 20.
Any scale-up (more candidates, bigger feeders) would silently lose the
certification claim the moment enumeration stops being feasible.

THE INSTRUMENT
--------------
The islanding stage is a BINARY polynomial of degree <= 3, and binary monomials
linearize EXACTLY (Fortet / Glover-Woolsey):

    y = x_i * x_j        <=>   y <= x_i,  y <= x_j,  y >= x_i + x_j - 1
    y = x_i * x_j * x_k  <=>   y <= each, y >= x_i + x_j + x_k - 2

with y in [0, 1]. No approximation is involved: the linearized MILP has the
same optimal set as the polynomial. HiGHS branch-and-bound then proves global
optimality with a zero relative gap -- the SAME certificate enumeration gives,
but at hundreds of binary variables instead of twenty.

Binary also means x^2 = x, so repeated indices in a term collapse to the linear
part before linearization.

This makes certification an O(poly) preprocessing step plus a MILP solve,
instead of an O(2^m) wall. It is the instrument that keeps "certified" honest
at scale.
"""
import time

import numpy as np


def milp_certify(H, time_limit=120.0):
    """Prove the global optimum of a binary Poly of degree <= 3 via HiGHS.

    Returns dict(energy, x, certified, wall, n_aux, status). `certified` is
    True only when the solver status is proven optimality at mip_rel_gap = 0;
    a time-limit hit returns certified=False rather than a guess.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds

    assert all(H.upper[v] == 1 for v in H.upper), "binary Hamiltonians only"
    vs = sorted(H.upper)
    idx = {v: i for i, v in enumerate(vs)}
    n = len(vs)

    # collapse repeated indices (x^2 = x for binary), gather product terms
    lin = np.zeros(n)
    products = {}                      # frozenset of columns -> coefficient
    for key, c in H.terms.items():
        uniq = tuple(sorted(set(key)))
        if len(uniq) == 1:
            lin[idx[uniq[0]]] += c
        else:
            cols = tuple(idx[v] for v in uniq)
            products[cols] = products.get(cols, 0.0) + c
    assert all(len(k) <= 3 for k in products), "degree > 3 not supported"

    n_aux = len(products)
    N = n + n_aux
    cost = np.zeros(N)
    cost[:n] = lin
    aux_col = {}
    for j, (cols, c) in enumerate(sorted(products.items())):
        aux_col[cols] = n + j
        cost[n + j] = c

    rows, lo, hi = [], [], []
    for cols, _c in sorted(products.items()):
        y = aux_col[cols]
        for xc in cols:                              # y <= x_i
            r = np.zeros(N)
            r[y] = 1.0
            r[xc] = -1.0
            rows.append(r); lo.append(-np.inf); hi.append(0.0)
        r = np.zeros(N)                              # y >= sum x_i - (k-1)
        r[y] = 1.0
        for xc in cols:
            r[xc] = -1.0
        rows.append(r); lo.append(-(len(cols) - 1)); hi.append(np.inf)

    t0 = time.time()
    res = milp(c=cost, integrality=np.ones(N),
               constraints=[LinearConstraint(np.array(rows), lo, hi)] if rows else [],
               bounds=Bounds(np.zeros(N), np.ones(N)),
               options=dict(mip_rel_gap=0.0, time_limit=time_limit))
    wall = time.time() - t0

    certified = bool(res.status == 0)
    x = np.zeros(H.n)
    if res.x is not None:
        for v in vs:
            x[v - 1] = int(round(res.x[idx[v]]))
    energy = H.evaluate(x) if res.x is not None else float("nan")
    return dict(energy=float(energy), x=x, certified=certified, wall=wall,
                n_aux=n_aux, status=int(res.status))
