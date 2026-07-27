#!/usr/bin/env python
"""Dirac hardware test — QCi Free Tier (600 s cumulative, up to 100 vars).

Runs the eQoSystem ISLANDING QUBOs on real QCi hardware and checks each result
against the exact (brute-force) ground state. Binary QUBOs are the in-spec
target: small, degree-2, low dynamic range, and independently certifiable.

Auth:  export QCI_TOKEN=<your api key>   (and QCI_API_URL if QCi gave you one)

Backends:
  --backend dirac1   binary/discrete solver  (recommended for these QUBOs)
  --backend dirac3c  Dirac-3 continuous (sum-constrained) solver
  --backend sa       classical rehearsal (no token, no compute used)

Usage (rehearse first!):
  python hardware_test.py --backend sa            # dry run, free
  python hardware_test.py --backend dirac1 -n 5   # real hardware, 5 scenarios
"""
import argparse, os, time, json
import numpy as np

from eqosystem import grid, candidates, scenarios, hamiltonians as ham
from eqosystem.pipeline import run_design
from eqosystem.solvers import AnnealerSolver, ExactSolver, to_eqc_model


def qubo_to_quadratic_model(H):
    """Convert a binary Poly (islanding QUBO) to eqc-models QuadraticModel(C, J).
    Requires all variables binary (upper bound 1) and degree <= 2."""
    assert all(H.upper[v] == 1 for v in H.upper), "Dirac-1 path needs binary vars"
    assert H.degree <= 2, "QuadraticModel needs degree <= 2"
    keep = sorted(H.upper)
    idx = {v: i for i, v in enumerate(keep)}
    n = len(keep)
    C = np.zeros((n, 1)); J = np.zeros((n, n))
    for key, c in H.terms.items():
        if len(key) == 1:
            C[idx[key[0]], 0] += c
        elif len(key) == 2:
            a, b = idx[key[0]], idx[key[1]]
            J[a, b] += c / 2.0; J[b, a] += c / 2.0
    return C, J, keep


def build_quadratic_model(C, J):
    """QuadraticModel with upper_bound explicitly set (binary: 0/1 per var).
    Required -- eqc-models' Dirac1CloudSolver.checkModel() compares
    model.upper_bound against the device limit and errors if it's left None."""
    from eqc_models.base import QuadraticModel
    model = QuadraticModel(C, J)
    model.upper_bound = np.ones(C.shape[0], dtype=int)
    return model


def get_hw_solver(backend):
    if backend == "dirac1":
        from eqc_models.solvers import Dirac1CloudSolver
        return Dirac1CloudSolver()
    if backend == "dirac3":                      # Dirac-3 discrete/integer solver
        from eqc_models.solvers import Dirac3IntegerCloudSolver
        return Dirac3IntegerCloudSolver()
    if backend == "dirac3c":                     # Dirac-3 quasi-continuous solver
        from eqc_models.solvers import Dirac3ContinuousCloudSolver
        return Dirac3ContinuousCloudSolver()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sa",
                    choices=["sa", "dirac1", "dirac3", "dirac3c"])
    ap.add_argument("-n", "--num", type=int, default=3, help="how many islanding QUBOs")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--grid", default="ieee69", choices=["ieee33", "ieee69"])
    args = ap.parse_args()

    if args.backend != "sa" and not os.environ.get("QCI_TOKEN"):
        raise SystemExit("Set QCI_TOKEN first:  export QCI_TOKEN=<your key>")

    grid.select(args.grid)
    pool = candidates.generate()
    scens = scenarios.generate(20)
    design, _, _ = run_design(pool, AnnealerSolver())

    # collect non-trivial binary islanding QUBOs with known exact ground states
    tasks = []
    for sc in scens:
        H, meta = ham.build_island(design, pool, sc)
        if H.n >= 3 and H.degree <= 2 and all(H.upper[v] == 1 for v in H.upper):
            exact = ExactSolver().solve(H)
            tasks.append((sc.sid, H, exact["energy"], exact["x"]))
        if len(tasks) >= args.num:
            break

    print(f"Prepared {len(tasks)} binary islanding QUBOs "
          f"(vars: {[t[1].n for t in tasks]}, dyn-range dB: "
          f"{[round(t[1].dynamic_range_db(),1) for t in tasks]})")

    hw = get_hw_solver(args.backend)
    out = []
    total_wall = 0.0
    for sid, H, exact_E, exact_x in tasks:
        if args.backend == "sa":
            r = AnnealerSolver().solve(H)
            best_E, wall, ns = r["energy"], r["wall"], 6
            raw = None
        else:
            t0 = time.time()
            if args.backend == "dirac1":
                # discrete solver wants the QUBO (QuadraticModel) operator
                C, J, keep = qubo_to_quadratic_model(H)
                model = build_quadratic_model(C, J)
                resp = hw.solve(model, name=f"island_s{sid}",
                                num_samples=args.samples)
            else:
                # Dirac-3 (integer or continuous) wants the polynomial operator;
                # to_eqc_model already produces a validated PolynomialModel and
                # returns the kept-variable mapping for re-expansion.
                model, keep = to_eqc_model(H, return_mapping=True)
                if args.backend == "dirac3":
                    resp = hw.solve(model, name=f"island_s{sid}",
                                    relaxation_schedule=2, num_samples=args.samples)
                else:  # dirac3c: quasi-continuous, needs a sum constraint
                    resp = hw.solve(model, name=f"island_s{sid}",
                                    sum_constraint=float(max(H.n, 1)),
                                    relaxation_schedule=2, num_samples=args.samples)
            wall = time.time() - t0
            raw = resp
            # SolutionResults exposes .solutions (ndarray of vectors) and
            # .energies (ndarray). Extract without boolean-testing arrays.
            sols = None
            if hasattr(resp, "solutions") and resp.solutions is not None:
                sols = np.asarray(resp.solutions)
            elif isinstance(resp, dict):
                r = resp.get("results", resp)
                sols = np.asarray(r["solutions"])
            if sols is None or len(sols) == 0:
                raise RuntimeError(f"No solutions returned by hardware for s{sid}")
            sols = np.atleast_2d(sols)
            full = []
            for s in sols:
                x = np.zeros(H.n)
                for i, v in enumerate(keep):
                    x[v - 1] = int(round(float(s[i])))
                full.append(H.evaluate(x))
            best_E, ns = min(full), len(sols)
        total_wall += wall
        matched = abs(best_E - exact_E) < 1e-6
        out.append(dict(scenario=int(sid), n_vars=H.n, hw_energy=float(best_E),
                        exact_energy=float(exact_E), matched=bool(matched),
                        wall_s=round(wall, 3), samples=ns,
                        dyn_range_db=round(H.dynamic_range_db(), 1)))
        print(f"  s{sid:02d}: hw={best_E:8.3f}  exact={exact_E:8.3f}  "
              f"{'MATCH ✓' if matched else 'GAP ✗'}  {wall:.2f}s  {ns} samples")

    n_match = sum(o["matched"] for o in out)
    print(f"\n{n_match}/{len(out)} islanding QUBOs solved to the certified optimum "
          f"on '{args.backend}' | total wall-clock {total_wall:.1f}s")
    print("  (note: wall-clock includes queue + device evolution; billed "
          "allocation is far lower -- see the 'allocation balance' lines above)")
    os.makedirs("results", exist_ok=True)
    fn = f"results/hardware_{args.backend}.json"
    json.dump(dict(backend=args.backend, grid=args.grid, matched=n_match,
                   total=len(out), total_wall_s=total_wall, runs=out),
              open(fn, "w"), indent=2)
    print(f"Saved {fn}  <-- your hardware evidence for the write-up")


if __name__ == "__main__":
    main()
