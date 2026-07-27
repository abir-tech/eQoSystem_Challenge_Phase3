"""Solver backends behind one interface.

``Dirac3Solver``  -- real QCi hardware via eqc-models Dirac3IntegerCloudSolver
                     (activates when QCI_TOKEN is set, e.g. on qBraid).
``AnnealerSolver``-- discrete simulated annealing over bounded integers; the
                     local development stand-in and the classical reference.
``ExactSolver``   -- brute force for small problems; certifies optimality.

All backends consume the same ``Poly`` object, and the Dirac path converts it
to a *validated* eqc-models ``PolynomialModel`` (1-based, zero-padded, sorted
indices -- the format the Phase-2 code got wrong).
"""
import itertools
import os
import time
import numpy as np


# ------------------------------------------------------------ eqc conversion
def to_eqc_model(H, return_mapping=False):
    """Convert Poly -> eqc_models PolynomialModel with correct index format.

    Variables with upper bound 0 (e.g. PV curtailment at night) are fixed at
    zero and compressed out -- Dirac-3 rejects zero upper bounds, and dropping
    them also saves qudits. `mapping` re-expands hardware solutions."""
    from eqc_models.base import PolynomialModel
    keep = [v for v in sorted(H.upper) if H.upper[v] >= 1]
    remap = {v: i + 1 for i, v in enumerate(keep)}
    deg = max(H.degree, 2)
    rows, coefs = [], []
    for key, c in sorted(H.terms.items(), key=lambda kv: (len(kv[0]), kv[0])):
        if abs(c) < 1e-12 or any(v not in remap for v in key):
            continue                                # fixed-zero var kills the term
        nk = sorted(remap[v] for v in key)
        rows.append([0] * (deg - len(nk)) + nk)     # zero-pad to uniform width
        coefs.append(float(c))
    order = np.lexsort(tuple(np.array(rows).T[::-1]))
    indices = np.array(rows, dtype=np.int32)[order]
    coefficients = np.array(coefs, dtype=np.float64)[order]
    model = PolynomialModel(coefficients, indices)
    model.upper_bound = np.array([H.upper[v] for v in keep], dtype=np.int64)
    # self-check: eqc evaluation must match our own evaluator
    probe_full = np.zeros(H.n)
    for v in keep:
        probe_full[v - 1] = min(1, H.upper[v])
    probe = np.array([probe_full[v - 1] for v in keep], dtype=float)
    assert abs(model.evaluate(probe) - (H.evaluate(probe_full) - H.const)) < 1e-6 * max(
        1.0, abs(H.evaluate(probe_full))), "eqc-models polynomial mismatch"
    if return_mapping:
        return model, keep
    return model


def expand_solution(x_small, keep, n_full):
    x = np.zeros(n_full, dtype=int)
    for i, v in enumerate(keep):
        x[v - 1] = int(x_small[i])
    return x


# ----------------------------------------------------------------- polishing
def greedy_polish(H, x):
    """Deterministic +-1 coordinate descent; cheap classical repair applied to
    every hardware/heuristic sample."""
    x = np.array(x, dtype=int)
    best = H.evaluate(x)
    improved = True
    while improved:
        improved = False
        for v in range(1, H.n + 1):
            for dv in (+1, -1):
                nv = x[v - 1] + dv
                if 0 <= nv <= H.upper[v]:
                    x2 = x.copy()
                    x2[v - 1] = nv
                    e2 = H.evaluate(x2)
                    if e2 < best - 1e-12:
                        x, best, improved = x2, e2, True
    return x, best


# --------------------------------------------------------------------- SA
class AnnealerSolver:
    name = "simulated-annealing"

    def __init__(self, restarts=6, iters=4000, t0=2.5, t1=1e-3, seed=0):
        self.restarts, self.iters, self.t0, self.t1, self.seed = restarts, iters, t0, t1, seed

    def solve(self, H, warm_start=None):
        rng = np.random.default_rng(self.seed)
        vs = sorted(H.upper)
        ubs = np.array([H.upper[v] for v in vs])
        # ---- delta-evaluation structures: var -> terms containing it -------
        keys = [np.array([vi - 1 for vi in k]) for k in H.terms]
        coefs = np.array(list(H.terms.values()))
        var_terms = {i: [] for i in range(H.n)}
        for t, k in enumerate(keys):
            for u in set(k):
                var_terms[u].append(t)
        best_x, best_e = None, np.inf
        t_start = time.time()
        for r in range(self.restarts):
            if warm_start is not None and r == 0:
                x = np.array(warm_start, dtype=int)
            else:
                x = rng.integers(0, ubs + 1)
            e = H.evaluate(x)
            for it in range(self.iters):
                T = self.t0 * (self.t1 / self.t0) ** (it / self.iters)
                v = rng.integers(0, H.n)
                if ubs[v] == 0:
                    continue
                step = int(rng.choice([-1, 1])) if ubs[v] > 1 else (1 - 2 * x[v])
                nv = x[v] + step
                if nv < 0 or nv > ubs[v]:
                    continue
                delta, old = 0.0, x[v]
                for t in var_terms[v]:          # only terms touching v change
                    k = keys[t]
                    p_old = np.prod(x[k])
                    x[v] = nv
                    p_new = np.prod(x[k])
                    x[v] = old
                    delta += coefs[t] * (p_new - p_old)
                if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
                    x[v], e = nv, e + delta
                if e < best_e:
                    best_x, best_e = x.copy(), e
            # numerical-drift self-check: incremental energy must match full eval
            assert abs(e - H.evaluate(x)) < 1e-6 * max(1.0, abs(e)), "SA delta drift"
        best_x, best_e = greedy_polish(H, best_x)
        return dict(x=best_x, energy=best_e, wall=time.time() - t_start,
                    backend=self.name, samples=self.restarts)


# ------------------------------------------------------------------- exact
class ExactSolver:
    name = "exact"

    def solve(self, H, limit=2 ** 21):
        ubs = [H.upper[v] for v in sorted(H.upper)]
        space = 1
        for u in ubs:
            space *= (u + 1)
            if space > limit:
                raise ValueError(f"search space too large for exact solve ({space})")
        best_x, best_e = None, np.inf
        t0 = time.time()
        for x in itertools.product(*[range(u + 1) for u in ubs]):
            e = H.evaluate(np.array(x))
            if e < best_e:
                best_x, best_e = np.array(x), e
        return dict(x=best_x, energy=best_e, wall=time.time() - t0,
                    backend=self.name, samples=space)


# ----------------------------------------------------------------- Dirac-3
class Dirac3Solver:
    """Entropy-computing hardware path. Requires QCI_TOKEN (set on qBraid).

    Noise controls exposed: relaxation_schedule (1-4, higher = colder/slower),
    num_samples (best-of-N against shot noise), mean_photon_number and
    quantum_fluctuation_coefficient (device-level). Every returned sample is
    classically polished (greedy descent) -- a hybrid loop that removes
    residual analog noise at negligible cost.
    """
    name = "dirac-3"

    def __init__(self, relaxation_schedule=2, num_samples=10, polish=True):
        self.relaxation_schedule = relaxation_schedule
        self.num_samples = num_samples
        self.polish = polish

    def available(self):
        return bool(os.environ.get("QCI_TOKEN"))

    def solve(self, H, job_name="eqosystem"):
        from eqc_models.solvers import Dirac3IntegerCloudSolver
        model, keep = to_eqc_model(H, return_mapping=True)
        solver = Dirac3IntegerCloudSolver()
        t0 = time.time()
        resp = solver.solve(model, name=job_name,
                            relaxation_schedule=self.relaxation_schedule,
                            num_samples=self.num_samples)
        wall = time.time() - t0
        sols = [expand_solution(s, keep, H.n) for s in resp["results"]["solutions"]]
        energies = [H.evaluate(np.array(s)) for s in sols]
        k = int(np.argmin(energies))
        x = np.array(sols[k], dtype=int)
        if self.polish:
            x, _ = greedy_polish(H, x)
        return dict(x=x, energy=H.evaluate(x), wall=wall, backend=self.name,
                    samples=self.num_samples, raw_energies=energies,
                    dirac_metadata={kk: resp.get(kk) for kk in ("job_info",) if kk in resp})


def get_solver(backend: str, **kw):
    if backend == "dirac3":
        s = Dirac3Solver(**kw)
        if not s.available():
            raise RuntimeError(
                "QCI_TOKEN not set. Dirac-3 hardware requires a token.\n"
                "  * For the classical baseline, run: --backend sa (no token needed)\n"
                "  * On qBraid with hardware access, set QCI_TOKEN then use --backend dirac3")
            return AnnealerSolver()
        return s
    if backend == "exact":
        return ExactSolver()
    return AnnealerSolver(**kw)
