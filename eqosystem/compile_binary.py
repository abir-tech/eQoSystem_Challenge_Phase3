"""Compile a bounded-integer Poly into an equivalent polynomial over BINARY
variables (the encoding any qubit-based machine needs).

Each integer x with upper bound U is expanded as x = sum_j w_j b_j with
weights 1,2,4,...,U-(2^{k-1}-1)  (standard bounded binary encoding).
Products of sums are expanded; repeated bits merge (b^2 = b).

This is used by experiment E5 to MEASURE, not assert, what Dirac-3's native
qudit encoding saves: variable count, term count, coefficient dynamic range,
and simulated-annealing success probability at equal compute budget.
"""
import itertools
import numpy as np
from .hamiltonians import Poly


def _weights(ub):
    k = max(1, int(np.ceil(np.log2(ub + 1))))
    w = [2 ** j for j in range(k - 1)]
    w.append(ub - (2 ** (k - 1) - 1))
    return [x for x in w if x > 0]


def binary_expand(H):
    """Return (Hb, groups): Hb is an equivalent Poly over binaries;
    groups[v] = [(bit_id, weight), ...] reconstructs each integer."""
    groups, nxt = {}, 1
    for v in sorted(H.upper):
        ws = _weights(max(1, H.upper[v]))
        groups[v] = [(nxt + j, w) for j, w in enumerate(ws)]
        nxt += len(ws)
    Hb = Poly()
    for bits in groups.values():
        for b, _ in bits:
            Hb.upper[b] = 1
    Hb.const = H.const
    for key, c in H.terms.items():
        # expand product over the bit decomposition of every factor
        for combo in itertools.product(*[groups[v] for v in key]):
            coef = c * float(np.prod([w for _, w in combo]))
            mono = tuple(sorted(set(b for b, _ in combo)))  # b^2 = b
            Hb.terms[mono] = Hb.terms.get(mono, 0.0) + coef
    Hb.terms = {k: v for k, v in Hb.terms.items() if abs(v) > 1e-12}
    return Hb, groups


def decode(xb, groups, n_full):
    x = np.zeros(n_full, dtype=int)
    for v, bits in groups.items():
        x[v - 1] = int(sum(w * xb[b - 1] for b, w in bits))
    return x
