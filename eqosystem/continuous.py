"""W3c -- Dirac-3 quasi-continuous solver path for stages that stay large.

Why this exists. The integer solver caps every variable at 16 levels, and the
multi-period dispatch stage does not fit: its largest bound is 43. W1b's
trust-region trick fixed the design stage the same way, but dispatch setpoints
are genuine physical quantities with a wide range, not corrections around a
seed, so the same trick does not apply. The continuous solver has
max_upper_bound = 10000 and no 16-level cap, which is the escape hatch.

The construction. Dirac3ContinuousCloudSolver enforces a SUM CONSTRAINT,
sum(x) = S, as an equality over every variable in the model. Setting S to the
sum of the upper bounds would therefore pin every variable AT its bound, which
is not a relaxation of anything -- it is a single point. A slack variable that
appears nowhere in the Hamiltonian absorbs the remainder,

    sum_i y_i + y_slack = S,   y_slack in [0, S]

so the constraint becomes sum_i y_i <= S, which is what we actually want.

The solution comes back continuous, so it is rounded to the integer lattice,
clamped to the original bounds, repaired by greedy descent on the ORIGINAL
Hamiltonian, and re-scored by our own evaluator. The rounding gap is measured
and reported rather than assumed small -- see `relax_round_repair`.
"""
import numpy as np

from .hamiltonians import Poly
from .solvers import greedy_polish

SLACK_NAME = "_simplex_slack"


def sum_constraint_for(H, headroom=1.0):
    """S for the simplex embedding: the box's own bound, times any headroom."""
    return float(headroom * sum(H.upper[v] for v in H.upper if H.upper[v] >= 1))


def to_simplex(H, headroom=1.0):
    """Return (H_ext, meta) where H_ext carries an inert slack variable.

    H_ext is identical to H on the original variables; the slack appears in no
    term, so it changes no energy. It exists only so the device's equality sum
    constraint does not pin the solution to a single point.
    """
    H_ext = Poly()
    H_ext.upper = dict(H.upper)
    H_ext.names = dict(H.names)
    H_ext.const = H.const
    for k, c in H.terms.items():
        H_ext.terms[k] = c
    S = sum_constraint_for(H, headroom)
    slack = H_ext.new_var(SLACK_NAME, int(np.ceil(S)))
    meta = dict(sum_constraint=S, slack_vid=slack, n_original=H.n)
    return H_ext, meta


def round_and_repair(y, H, meta=None):
    """Continuous vector -> feasible integer point on the original Hamiltonian.

    Returns (x, info) with the energy before and after repair so the rounding
    cost is visible instead of being folded into the headline number.
    """
    n = H.n
    x = np.zeros(n, dtype=int)
    for v in sorted(H.upper):
        val = float(y[v - 1]) if v - 1 < len(y) else 0.0
        x[v - 1] = int(np.clip(round(val), 0, H.upper[v]))
    e_rounded = H.evaluate(x.astype(float))
    x_rep, e_rep = greedy_polish(H, x)
    return np.asarray(x_rep, dtype=int), dict(
        energy_rounded=float(e_rounded), energy_repaired=float(e_rep),
        repair_gain=float(e_rounded - e_rep),
        n_moved=int(np.sum(np.asarray(x_rep) != x)))


def continuous_relaxation(H, meta=None, seed=0, restarts=4):
    """Classical stand-in for the device's continuous solve.

    Minimises H over the BOX relaxation intersected with sum(x) <= S, using
    SLSQP from several starts. This is what makes W3c testable and reportable
    with no hardware allocation: the rounding gap measured here is the same gap
    the device path incurs, because it is a property of the discretisation, not
    of the solver.
    """
    from scipy.optimize import minimize, LinearConstraint

    vs = sorted(H.upper)
    ub = np.array([H.upper[v] for v in vs], dtype=float)
    S = meta["sum_constraint"] if meta else sum_constraint_for(H)

    def f(z):
        return H.evaluate(z)

    cons = [LinearConstraint(np.ones(len(vs)), -np.inf, S)]
    rng = np.random.default_rng(seed)
    best, best_e = None, np.inf
    for r in range(restarts):
        z0 = ub * 0.5 if r == 0 else rng.random(len(vs)) * ub
        try:
            res = minimize(f, z0, method="SLSQP",
                           bounds=[(0.0, u) for u in ub], constraints=cons,
                           options=dict(maxiter=300, ftol=1e-9))
        except Exception:
            continue
        if res.x is not None and res.fun < best_e:
            best, best_e = res.x, float(res.fun)
    if best is None:
        best, best_e = ub * 0.5, f(ub * 0.5)
    return best, best_e


def relax_round_repair(H, seed=0, restarts=4, headroom=1.0):
    """Full W3c path, classically rehearsed: relax -> round -> repair -> score.

    Reports the continuous lower bound, the cost of rounding to the lattice, and
    what greedy repair recovers, so the discretisation penalty is explicit.
    """
    H_ext, meta = to_simplex(H, headroom=headroom)
    y, e_cont = continuous_relaxation(H, meta=meta, seed=seed, restarts=restarts)
    x, info = round_and_repair(y, H, meta)
    info.update(energy_continuous=float(e_cont),
                rounding_gap=float(info["energy_rounded"] - e_cont),
                final_gap=float(info["energy_repaired"] - e_cont),
                sum_constraint=meta["sum_constraint"],
                levels_with_slack=int(sum(H_ext.upper[v] + 1 for v in H_ext.upper
                                          if H_ext.upper[v] >= 1)))
    return x, info


class ContinuousRelaxSolver:
    """Classical rehearsal backend with the same interface as the others."""
    name = "continuous-relaxation"

    def __init__(self, seed=0, restarts=4):
        self.seed, self.restarts = seed, restarts

    def solve(self, H, warm_start=None):
        import time
        t0 = time.time()
        x, info = relax_round_repair(H, seed=self.seed, restarts=self.restarts)
        return dict(x=np.asarray(x, dtype=int), energy=H.evaluate(x.astype(float)),
                    wall=time.time() - t0, backend=self.name, samples=self.restarts,
                    continuous_info=info)
