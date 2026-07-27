#!/usr/bin/env python
"""Dirac hardware protocol for ALL THREE pipeline stages -- QCi free tier.

The challenge requires all three pipeline stages to be executed on a concrete
grid instance. This script runs each stage, checks it against a classical
reference on the identical instance, and records the resources it used.

  stage 1  design     H_design            vs HiGHS mixed-integer  (E4)
  stage 2  islanding  H_island(s)         vs exact enumeration    (E2)
  stage 3  dispatch   H_dispatch_mp(s,c)  vs HiGHS mixed-integer  (E12)

Every stage is checked against the device's limits BEFORE submission and the
run is refused rather than sent if it would be out of range -- an over-range
Hamiltonian is accepted by the device and answered incorrectly, with no error.

Auth:  export QCI_TOKEN=<your api key>     (and QCI_API_URL if QCi issued one)

Backends
  --backend sa        classical rehearsal, free, no token -- ALWAYS RUN THIS FIRST
  --backend dirac3    Dirac-3 integer solver
  --backend dirac3c   Dirac-3 quasi-continuous solver
  --backend dirac1    Dirac-1 (islanding QUBOs only)

Usage
  python hardware_test.py --backend sa --stage all -n 5        # free rehearsal
  python hardware_test.py --backend dirac3 --stage islanding -n 20
  cp results/hardware_dirac3_all.json results/hardware_dirac3_all_$(date +%F).json
"""
import argparse
import json
import os
import time

import numpy as np

from eqosystem import grid, candidates, scenarios, hamiltonians as ham
from eqosystem import conditioning as cond, radix
from eqosystem.pipeline import (run_design, milp_design_baseline,
                                milp_dispatch_baseline, simulate_plan)
from eqosystem.solvers import AnnealerSolver, ExactSolver, to_eqc_model

# The diesel heat-rate cubic is dropped for hardware submission. Its coefficient
# sits ~65 dB below c_max, far below the device's ~200:1 (23 dB) coefficient
# resolution, so the hardware cannot represent it in any case: submitting it
# means the device silently solves a truncated Hamiltonian instead of a known
# one. Measured, dropping it moves the stage 64.3 -> 28.3 dB while the unserved
# ratio against the mixed-integer baseline moves 0.999 -> 1.030 on non-trivial
# instances, about one standard deviation. The classical path keeps the cubic.
HW_DISPATCH_FUEL_CUBIC = 0.0

_design_meta = [None]


def qubo_to_quadratic_model(H):
    """Binary Poly (islanding QUBO) -> eqc-models QuadraticModel(C, J)."""
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
    from eqc_models.base import QuadraticModel
    model = QuadraticModel(C, J)
    model.upper_bound = np.ones(C.shape[0], dtype=int)
    return model


def solve_on(backend, H, name, samples, stage):
    """Solve H on the chosen backend, returning (x, wall, n_samples)."""
    if backend == "sa":
        r = AnnealerSolver().solve(H)
        return np.asarray(r["x"]), r["wall"], 6
    if not os.environ.get("QCI_TOKEN"):
        raise SystemExit("Set QCI_TOKEN first:  export QCI_TOKEN=<your key>")

    t0 = time.time()
    if backend == "dirac1":
        from eqc_models.solvers import Dirac1CloudSolver
        C, J, keep = qubo_to_quadratic_model(H)
        resp = Dirac1CloudSolver().solve(build_quadratic_model(C, J), name=name,
                                         num_samples=samples)
    elif backend == "dirac3":
        from eqc_models.solvers import Dirac3IntegerCloudSolver
        model, keep = to_eqc_model(H, return_mapping=True, stage=stage)
        resp = Dirac3IntegerCloudSolver().solve(model, name=name,
                                                relaxation_schedule=2,
                                                num_samples=samples)
    else:                                   # dirac3c
        from eqosystem import continuous as cont
        from eqc_models.solvers import Dirac3ContinuousCloudSolver
        H_ext, meta = cont.to_simplex(H)
        model, keep = to_eqc_model(H_ext, return_mapping=True, stage=stage)
        resp = Dirac3ContinuousCloudSolver().solve(
            model, name=name, sum_constraint=float(meta["sum_constraint"]),
            relaxation_schedule=2, num_samples=samples)
    wall = time.time() - t0

    sols = resp.solutions if hasattr(resp, "solutions") else \
        resp.get("results", resp)["solutions"]
    sols = np.atleast_2d(np.asarray(sols))
    best_x, best_e = None, np.inf
    for sm in sols:
        x = np.zeros(H.n)
        for i, v in enumerate(keep):
            if i < len(sm):
                x[v - 1] = int(round(float(sm[i])))
        x = np.clip(x, 0, [H.upper[v] for v in sorted(H.upper)])
        e = H.evaluate(x)
        if e < best_e:
            best_x, best_e = x, e
    return np.asarray(best_x, dtype=int), wall, len(sols)


def check_or_refuse(H, stage, allow_over_range=False):
    """Device-limit gate. Returns (certificate, reasons); raises if out of range."""
    _out, cert = cond.truncate_certified(H) if H.n <= 12 else (H, None)
    reasons = cond.hardware_legality(H, cert=cert)
    if reasons and not allow_over_range:
        raise SystemExit(
            f"REFUSED: stage '{stage}' is not legal on the integer solver:\n  - "
            + "\n  - ".join(reasons)
            + "\nUse --backend dirac3c (continuous) or fix the formulation.")
    return cert, reasons


def stage_design(args, pool, out):
    print("=" * 70, "\nSTAGE 1  Microgrid design\n" + "=" * 70)
    design, H, _res = run_design(pool, AnnealerSolver())
    check_or_refuse(H, "design",
                    allow_over_range=(args.backend == "dirac3c" or args.allow_over_range))
    milp = milp_design_baseline(pool)
    print(f"  {H.n} vars, {cond.total_levels(H)} levels, maxUB "
          f"{max(H.upper.values())}, {H.dynamic_range_db():.1f} dB, degree {H.degree}")
    x, wall, ns = solve_on(args.backend, H, "design", args.samples, "design")
    d = ham.decode_design(x, _design_meta[0])
    raw_capex = d["capex"]

    # A ratio below 1.0 against a CERTIFIED optimum is hidden infeasibility, not
    # a win: the raw penalty solution can sit a step short of the capacity gate.
    # Repair first, and report both numbers plus whether repair was needed, so
    # an under-built solution can never be read as a good result.
    n_repair = 0
    for c in d["selected"]:
        port = d["portfolio"][c]
        D = ham.design_demand(pool[c])
        firm = lambda: sum(port[k] * ham.ASSETS[k]["kw"] * ham.ASSETS[k]["firm"]
                           for k in ham.ASSET_KEYS)
        while firm() < D:
            kb = min((k for k in ham.ASSET_KEYS if port[k] < ham.asset_umax(k)),
                     key=lambda k: (ham.ASSETS[k]["cost"] + ham.ASSETS[k]["op"])
                     / (ham.ASSETS[k]["kw"] * ham.ASSETS[k]["firm"]), default=None)
            if kb is None:
                break
            port[kb] += 1
            d["capex"] += ham.ASSETS[kb]["cost"] + ham.ASSETS[kb]["op"]
            n_repair += 1
    covered = set().union(*[set(pool[c]["buses"]) for c in d["selected"]]) \
        if d["selected"] else set()
    feasible = set(range(2, grid.N_BUSES + 1)) <= covered
    ratio = d["capex"] / milp["capex"] if milp["capex"] else float("nan")

    print(f"  raw Hamiltonian capex {raw_capex:.1f}"
          + (f"  ->  {n_repair} repair unit(s) added" if n_repair
             else "  (feasible as solved)"))
    print(f"  capex {d['capex']:.1f} vs certified optimum {milp['capex']:.1f}"
          f"  ->  ratio {ratio:.3f}   ({wall:.2f}s, {ns} samples)")
    if ratio < 1.0:
        print("  !! ratio < 1.0 against a certified optimum -- investigate before "
              "reporting; this indicates infeasibility, not a better solution")
    if not feasible:
        print("  !! selected islands do not cover every load bus")
    out.append(dict(stage="design", n_vars=H.n, levels=cond.total_levels(H),
                    max_ub=int(max(H.upper.values())), degree=H.degree,
                    dyn_range_db=round(H.dynamic_range_db(), 1),
                    raw_capex=raw_capex, repair_units=n_repair,
                    coverage_complete=bool(feasible),
                    capex=d["capex"], milp_capex=milp["capex"], ratio=ratio,
                    wall_s=round(wall, 3), samples=ns))
    return design


def stage_islanding(args, pool, design, scens, out):
    print("=" * 70, "\nSTAGE 2  Contingency islanding\n" + "=" * 70)
    n_match, tasks = 0, []
    for sc in scens:
        H, _m = ham.build_island(design, pool, sc)
        if H.n >= 3 and H.degree <= 2 and all(H.upper[v] == 1 for v in H.upper):
            tasks.append((sc.sid, H, ExactSolver().solve(H)))
        if len(tasks) >= args.num:
            break
    for sid, H, exact in tasks:
        check_or_refuse(H, f"island_s{sid}")
        x, wall, ns = solve_on(args.backend, H, f"island_s{sid}", args.samples,
                               "islanding")
        e = H.evaluate(x)
        ok = abs(e - exact["energy"]) < 1e-6
        n_match += ok
        print(f"  s{sid:02d}: {e:9.3f} vs exact {exact['energy']:9.3f}  "
              f"{'MATCH' if ok else 'GAP  '}  {H.dynamic_range_db():4.1f} dB  {wall:.2f}s")
        out.append(dict(stage="islanding", scenario=int(sid), n_vars=H.n,
                        levels=cond.total_levels(H), degree=H.degree,
                        dyn_range_db=round(H.dynamic_range_db(), 1),
                        energy=float(e), exact_energy=float(exact["energy"]),
                        matched=bool(ok), wall_s=round(wall, 3), samples=ns))
    print(f"  {n_match}/{len(tasks)} solved to the certified optimum")
    return n_match, len(tasks)


def select_dispatch_instances(args, pool, design, scens):
    """Pick instances worth spending allocation on.

    Most dispatch instances are TRIVIAL: the island covers its load, so both the
    Hamiltonian and the mixed-integer reference shed nothing and the comparison
    is vacuously equal. Measured on ieee69 at seed 42, only 11 of 21 are
    non-trivial, and taking the first n in scenario order picks trivial ones --
    a hardware run on those would report a meaningless success.
    """
    cands = []
    for sc in scens:
        Hi, mi = ham.build_island(design, pool, sc)
        if not Hi.n:
            continue
        ri = AnnealerSolver().solve(Hi)
        for c in [k for k in mi["z"] if ri["x"][mi["z"][k] - 1] > 0.5]:
            reach = mi["info"][c]["reach"]
            if not reach:
                continue
            H, md = ham.build_dispatch_mp(design, pool, sc, c,
                                          fuel_cubic=HW_DISPATCH_FUEL_CUBIC)
            Hr, _m = radix.sum_decompose(H)
            legal = not cond.hardware_legality(Hr, cert=dict(certified=True))
            mb = milp_dispatch_baseline(design, pool, sc, c)
            shed = (simulate_plan(pool, sc, c, mb["plan"], md, reach)["unserved_kwh"]
                    if mb["feasible"] else 0.0)
            cands.append(dict(sc=sc, c=c, reach=reach, nontrivial=shed > 1.0,
                              legal=legal, milp_shed=shed))
        if len(cands) >= 4 * max(args.num, 1) + 8:
            break
    good = [k for k in cands if k["nontrivial"] and k["legal"]]
    rest = [k for k in cands if k not in good]
    good.sort(key=lambda k: -k["milp_shed"])
    print(f"  {len(cands)} candidate instances: "
          f"{sum(1 for k in cands if k['nontrivial'])} non-trivial, "
          f"{sum(1 for k in cands if k['legal'])} integer-legal, "
          f"{len(good)} both -> selecting {min(args.num, len(good + rest))}")
    if len(good) < args.num:
        print(f"  !! only {len(good)} instance(s) are both non-trivial and legal; "
              f"padding with the rest, which prove less")
    return (good + rest)[:args.num]


def stage_dispatch(args, pool, design, scens, out):
    print("=" * 70, "\nSTAGE 3  Islanded DER dispatch\n" + "=" * 70)
    done = 0
    for pick in select_dispatch_instances(args, pool, design, scens):
        sc, c, reach = pick["sc"], pick["c"], pick["reach"]
        H, md = ham.build_dispatch_mp(design, pool, sc, c,
                                      fuel_cubic=HW_DISPATCH_FUEL_CUBIC)
        # Decomposition is bounded-SUM, not base-16 radix. Radix over-represents
        # (u=40 spans 0..47) so recompose clips, and its 16^k digit weights
        # inflate dynamic range. Measured on non-trivial instances: radix 1.640x
        # against the mixed-integer baseline at 39.8 dB, bounded-sum 1.060x at
        # 31.1 dB, against 1.041x solving the same Hamiltonian directly but
        # illegally.
        Hr, mapping = radix.sum_decompose(H)
        _c, reasons = check_or_refuse(Hr, f"dispatch_s{sc.sid}_c{c}",
                                      allow_over_range=True)
        if reasons and args.allow_over_range:
            print(f"  s{sc.sid:02d}/C{c:02d}: OVER-RANGE, submitting anyway by "
                  f"request ({Hr.dynamic_range_db():.1f} dB) -- {reasons[0]}")
            reasons = []
        if reasons and args.backend in ("dirac3", "dirac1"):
            print(f"  s{sc.sid:02d}/C{c:02d}: ROUTED to the continuous solver "
                  f"({Hr.dynamic_range_db():.1f} dB) -- {reasons[0]}")
            out.append(dict(stage="dispatch", scenario=int(sc.sid), island=int(c),
                            n_vars=Hr.n, levels=cond.total_levels(Hr),
                            dyn_range_db=round(Hr.dynamic_range_db(), 1),
                            nontrivial=bool(pick["nontrivial"]),
                            routed_to_continuous=True, reason=reasons[0]))
            done += 1
            continue

        xd, wall, ns = solve_on(args.backend, Hr, f"disp_s{sc.sid}c{c}",
                                args.samples, "dispatch")
        x, n_clamped = radix.sum_recompose(xd, mapping, H)
        plan = {g: [int(x[md["vars"][(g, t)] - 1]) for t in range(md["nb"])]
                for g in ("pv", "dg", "dis", "chg", "nc")}
        mb = milp_dispatch_baseline(design, pool, sc, c)
        q = simulate_plan(pool, sc, c, plan, md, reach)
        m = simulate_plan(pool, sc, c, mb["plan"], md, reach) if mb["feasible"] else None
        ratio = (q["unserved_kwh"] / m["unserved_kwh"]
                 if m and m["unserved_kwh"] > 1e-9 else float("nan"))
        tag = "" if pick["nontrivial"] else "  [TRIVIAL: nothing to shed, proves little]"
        print(f"  s{sc.sid:02d}/C{c:02d}: {Hr.n} vars, {cond.total_levels(Hr)} levels, "
              f"maxUB {max(Hr.upper.values())}, {Hr.dynamic_range_db():.1f} dB | "
              f"unserved {q['unserved_kwh']:.0f} vs reference "
              f"{(m['unserved_kwh'] if m else float('nan')):.0f} kWh "
              f"(x{ratio:.2f})  {wall:.2f}s{tag}")
        out.append(dict(stage="dispatch", scenario=int(sc.sid), island=int(c),
                        n_vars=Hr.n, levels=cond.total_levels(Hr),
                        max_ub=int(max(Hr.upper.values())), degree=Hr.degree,
                        dyn_range_db=round(Hr.dynamic_range_db(), 1),
                        radix_clamped=int(n_clamped),
                        nontrivial=bool(pick["nontrivial"]),
                        unserved_kwh=q["unserved_kwh"],
                        milp_unserved_kwh=(m["unserved_kwh"] if m else None),
                        ratio=ratio, crit_short_hours=q["crit_short_hours"],
                        wall_s=round(wall, 3), samples=ns))
        done += 1
        if done >= args.num:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sa",
                    choices=["sa", "dirac1", "dirac3", "dirac3c"])
    ap.add_argument("--stage", default="islanding",
                    choices=["design", "islanding", "dispatch", "all"])
    ap.add_argument("-n", "--num", type=int, default=3, help="instances per stage")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--grid", default="ieee69", choices=["ieee33", "ieee69"])
    ap.add_argument("--n-scenarios", type=int,
                    default=scenarios.DEFAULT_N_SCENARIOS,
                    help="must match run_experiments.py so the two agree; the "
                         "design stage is scenario-independent either way")
    ap.add_argument("--fuel-cubic", type=float, default=None,
                    help="override the hardware dispatch fuel-curve weight. The "
                         "default drops the cubic (0.0) because its coefficient "
                         "sits ~65 dB below c_max, below the device's own "
                         "resolution. Pass 0.35 to submit it anyway and MEASURE "
                         "whether the device can use it; needs --allow-over-range.")
    ap.add_argument("--allow-over-range", action="store_true",
                    help="submit even if a stage exceeds the dynamic-range "
                         "trigger. Deliberate experiments only -- the device "
                         "answers an over-range Hamiltonian incorrectly and "
                         "without error.")
    args = ap.parse_args()

    if args.backend != "sa" and not os.environ.get("QCI_TOKEN"):
        raise SystemExit("Set QCI_TOKEN first:  export QCI_TOKEN=<your key>")
    if args.backend != "sa":
        print(f"!! about to spend non-renewable free-tier allocation on "
              f"'{args.backend}' -- rehearse with --backend sa first\n")

    global HW_DISPATCH_FUEL_CUBIC
    if args.fuel_cubic is not None:
        HW_DISPATCH_FUEL_CUBIC = args.fuel_cubic
        print(f"!! dispatch fuel-curve weight overridden to "
              f"{HW_DISPATCH_FUEL_CUBIC}. The cubic's coefficient sits ~65 dB "
              f"below c_max, below the device's own resolution, so this is a "
              f"deliberate experiment to MEASURE whether the device can use it "
              f"-- not the recommended profile.\n")

    grid.select(args.grid)
    pool = candidates.generate()
    scens = scenarios.generate(args.n_scenarios, seed=42)

    # the design is needed by every stage; keep its meta for decoding
    _H_d, meta_d = ham.build_design(
        pool, seed_units=ham.greedy_portfolio(pool)[1], radius=3, slack_max=4)
    _design_meta[0] = meta_d

    out, t0 = [], time.time()
    design, _H, _ = run_design(pool, AnnealerSolver())
    if args.stage in ("design", "all"):
        stage_design(args, pool, out)
    if args.stage in ("islanding", "all"):
        stage_islanding(args, pool, design, scens, out)
    if args.stage in ("dispatch", "all"):
        stage_dispatch(args, pool, design, scens, out)

    total = time.time() - t0
    print("=" * 70)
    print(f"stages run: {sorted({r['stage'] for r in out})} | "
          f"{len(out)} jobs | wall-clock {total:.1f}s")
    print("  (wall-clock includes queueing; billed allocation is far lower)")
    os.makedirs("results", exist_ok=True)
    fn = f"results/hardware_{args.backend}_{args.stage}.json"
    json.dump(dict(backend=args.backend, grid=args.grid, stage=args.stage,
                   n_scenarios=args.n_scenarios, total_wall_s=total,
                   hw_dispatch_fuel_cubic=HW_DISPATCH_FUEL_CUBIC, runs=out),
              open(fn, "w"), indent=2)
    print(f"Saved {fn}")


if __name__ == "__main__":
    main()
