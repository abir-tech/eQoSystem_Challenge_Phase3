"""W3a -- radix-16 digit decomposition for the Dirac-3 integer solver.

The integer solver caps every variable at 16 levels (upper bound 15). Any stage
with a larger bound is illegal as formulated. A variable u with bound U > 15 is
replaced by base-16 digits

    u = sum_k 16^k * d_k,     d_k in [0, min(15, U // 16^k)]

Substituting a SUM of digit variables for each occurrence of u in a monomial
cannot raise the monomial's total degree: expanding the product turns one
degree-d term into many degree-d terms over digits. Degree preservation is
asserted anyway, per the work order -- measured, not assumed.

HONEST CAVEAT, asserted and reported rather than hidden: the digit box can
represent values ABOVE the original bound (e.g. U = 20 -> digits reach 31), so
the decomposition is a RELAXATION of the feasible box, not a bijection. The
round-trip is exact wherever both are defined; hardware solutions are clamped
back into the original box by `radix_recompose` and re-polished classically, and
the clamp count is reported.
"""
import numpy as np

from .hamiltonians import Poly

BASE = 16
DIGIT_MAX = 15          # integer-solver per-variable cap: levels 0..15


def _digits_needed(U, base=BASE):
    n, cap = 1, base - 1
    while cap < U:
        n += 1
        cap = base ** n - 1
    return n


def radix_decompose(H, base=BASE, digit_max=DIGIT_MAX):
    """Return (H_radix, mapping). Variables with bound <= digit_max pass through.

    mapping[v] = [(digit_vid, base**k), ...] for decomposed v, else [(vid, 1)].
    """
    Hr = Poly()
    mapping = {}
    for v in sorted(H.upper):
        U = H.upper[v]
        name = H.names.get(v, f"v{v}")
        if U <= digit_max:
            nv = Hr.new_var(name, U)
            mapping[v] = [(nv, 1)]
        else:
            n = _digits_needed(U, base)
            parts = []
            for k in range(n):
                w = base ** k
                # top digit only needs to reach U // w
                ub = min(digit_max, U // w)
                parts.append((Hr.new_var(f"{name}.r{k}", ub), w))
            mapping[v] = parts

    Hr.const = H.const
    for key, c in H.terms.items():
        # expand prod_i (sum_j w_ij * d_ij) into monomials over digits
        expansions = [mapping[v] for v in key]
        stack = [((), c)]
        for parts in expansions:
            stack = [(vids + (dv,), coef * w)
                     for (vids, coef) in stack for (dv, w) in parts]
        for vids, coef in stack:
            Hr.add(coef, *vids)

    assert Hr.degree <= H.degree, (
        f"radix decomposition raised degree {H.degree} -> {Hr.degree}")
    assert max(Hr.upper.values(), default=0) <= digit_max
    return Hr, mapping


def radix_encode(x, mapping, H_original):
    """Original-variable vector -> digit vector (the inverse of recompose).

    Every value in the original box has an exact digit representation, so this
    is the direction that is always well defined. Round-trip tests should sample
    here rather than sampling digits directly: the digit box is a relaxation, so
    uniformly random digit vectors land outside the original box most of the
    time once several variables are decomposed.
    """
    n_digits = sum(len(parts) for parts in mapping.values())
    d = np.zeros(n_digits, dtype=int)
    for v, parts in mapping.items():
        val = int(x[v - 1])
        U = H_original.upper[v]
        if not 0 <= val <= U:
            raise ValueError(f"value {val} outside the box [0, {U}] for variable {v}")
        for (dv, w) in sorted(parts, key=lambda p: -p[1]):     # most significant first
            q, val = divmod(val, w)
            d[dv - 1] = q
        assert val == 0, "digit expansion left a remainder"
    return d


def radix_recompose(x_digits, mapping, H_original):
    """Digit vector -> original-variable vector, clamped into the original box.

    Returns (x, n_clamped): n_clamped counts variables whose digit value
    exceeded the original bound (the relaxation caveat above) -- report it.
    """
    x = np.zeros(len(H_original.upper), dtype=int)
    n_clamped = 0
    for v, parts in mapping.items():
        val = sum(int(x_digits[dv - 1]) * w for (dv, w) in parts)
        U = H_original.upper[v]
        if val > U:
            val = U
            n_clamped += 1
        x[v - 1] = val
    return x, n_clamped


def levels(H):
    return sum(H.upper[v] + 1 for v in H.upper if H.upper[v] >= 1)


# --------------------------------------------------------------------------
# Bounded-sum decomposition -- exact where radix is a relaxation.
#
# Base-16 positional digits OVER-represent: u = 40 becomes d1 in [0,2] and d0 in
# [0,15], spanning 0..47, so 41..47 are infeasible points the solver can reach
# and that recompose then clips. Measured on non-trivial dispatch instances that
# clipping is not a rounding detail -- it costs 1.640x against the MILP baseline
# versus 1.041x for the same Hamiltonian solved directly, with 39 clamped
# variable-instances.
#
# Splitting into parts whose bounds SUM to exactly u removes the failure mode
# rather than mitigating it:
#
#     u = sum_j d_j,    d_j in [0, b_j],    sum_j b_j = u
#
# The representable set is then exactly [0, u]. The cost is levels: a variable
# with bound u goes from u+1 levels to u+k for k parts, i.e. only k-1 extra,
# which is far cheaper than it first appears. The encoding is symmetric (many
# digit vectors give the same u), so it enlarges the search space -- that is
# measured rather than assumed.
def sum_decompose(H, part_max=DIGIT_MAX):
    """Return (H_sum, mapping) with every bound <= part_max and NO clamping."""
    Hs = Poly()
    mapping = {}
    for v in sorted(H.upper):
        U = H.upper[v]
        name = H.names.get(v, f"v{v}")
        if U <= part_max:
            mapping[v] = [Hs.new_var(name, U)]
            continue
        parts, rem, j = [], U, 0
        while rem > 0:
            b = min(part_max, rem)
            parts.append(Hs.new_var(f"{name}.s{j}", b))
            rem -= b
            j += 1
        mapping[v] = parts

    Hs.const = H.const
    for key, c in H.terms.items():
        stack = [((), c)]
        for v in key:
            stack = [(vids + (p,), coef) for (vids, coef) in stack
                     for p in mapping[v]]
        for vids, coef in stack:
            Hs.add(coef, *vids)

    assert Hs.degree <= H.degree, "sum decomposition raised the degree"
    assert max(Hs.upper.values(), default=0) <= part_max
    return Hs, mapping


def sum_recompose(x_parts, mapping, H_original):
    """Parts -> original variables. Cannot exceed the box, so never clamps."""
    x = np.zeros(len(H_original.upper), dtype=int)
    for v, parts in mapping.items():
        x[v - 1] = sum(int(x_parts[p - 1]) for p in parts)
        assert x[v - 1] <= H_original.upper[v], "sum decomposition over-represented"
    return x, 0
