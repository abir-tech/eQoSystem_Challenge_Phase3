#!/usr/bin/env python3
"""Phase-3 experiment battery for the eQoSystem microgrid framework.

Usage:
    python run_experiments.py --backend sa        # local (default)
    python run_experiments.py --backend dirac3    # on qBraid with QCI_TOKEN set

E1  Full 3-stage pipeline on 20 LHS contingency scenarios vs no-islanding baseline
E2  Optimality certification: backend vs exact enumeration on every islanding QUBO
E3  Analog-noise robustness: solution quality vs coefficient dynamic range
E4  Design stage vs exact MILP (classical baseline required by the rubric)
"""
import argparse
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eqosystem import grid, scenarios, candidates, hamiltonians as ham
from eqosystem.pipeline import run_design, run_scenario, milp_design_baseline
from eqosystem.solvers import get_solver, AnnealerSolver, ExactSolver, greedy_polish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sa", choices=["sa", "dirac3"])
    ap.add_argument("--n-scenarios", type=int, default=20)
    ap.add_argument("--grid", default="ieee69", choices=["ieee33","ieee69"],
                    help="test system (default: IEEE 69-bus)")
    ap.add_argument("--stress", action="store_true", help="beyond-design-basis scenarios (load to 1.5x, outages to 20 h)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    grid.select(args.grid)
    solver = get_solver(args.backend)
    rng = np.random.default_rng(args.seed)
    R = {"backend": solver.name, "grid": dict(buses=grid.N_BUSES,
         total_load_kw=grid.TOTAL_P, customers=grid.TOTAL_CUSTOMERS,
         critical_buses=sorted(grid.CRITICAL_BUSES))}

    pool = candidates.generate()
    scens = scenarios.generate(args.n_scenarios, seed=args.seed, stress=args.stress)

    # ---------------- E1: full pipeline ----------------
    print("=" * 62, "\nE1  Full pipeline\n" + "=" * 62)
    t0 = time.time()
    design, H_design, res_design = run_design(pool, solver)
    print(f"  Stage 1 | {H_design.n} vars, {len(H_design.terms)} terms, deg {H_design.degree}, "
          f"dyn-range {H_design.dynamic_range_db():.1f} dB | wall {res_design['wall']:.1f}s")
    print(f"  Selected islands: {design['selected']}  capex={design['capex']:.1f} (x10 k$)")
    for c in design["selected"]:
        print(f"    C{c:02d}: {design['portfolio'][c]}  demand={ham.design_demand(pool[c]):.0f} kW")
    print(f"  Tie switches closed: {[grid.TIE_SWITCHES[l] for l in design['switches']]}")

    scen_results, job_log = [], []
    for sc in scens:
        r = run_scenario(design, pool, sc, solver)
        scen_results.append(r)
        job_log.extend(r["jobs"])
        print(f"  s{sc.sid:02d} line{sc.failed_line} dead={r['n_dead']:2d} "
              f"active={r['active']} maxUns={r['max_unserved']:.1%} "
              f"(base {r['base_max_unserved']:.1%}) critH={r['crit_hours']} "
              f"LDF {r['ldf_feasible_hours']}/{r['ldf_island_hours']} Vmin={r['ldf_v_min']:.3f} "
              f"(base {r['base_crit_hours']})")

    M1 = max(r["max_unserved"] for r in scen_results)
    M1b = max(r["base_max_unserved"] for r in scen_results)
    M2 = sum(r["crit_hours"] for r in scen_results)
    M2b = sum(r["base_crit_hours"] for r in scen_results)
    E_uns = sum(r["unserved_energy"] for r in scen_results) / len(scen_results)
    E_unsb = sum(r["base_energy"] for r in scen_results) / len(scen_results)
    R["E1"] = dict(
        capex_units=design["capex"], selected=design["selected"],
        portfolio={str(k): v for k, v in design["portfolio"].items()},
        switches=design["switches"],
        M1_max_unserved=M1, M1_baseline=M1b,
        M2_crit_hours=M2, M2_baseline=M2b,
        expected_unserved_kwh=E_uns, expected_unserved_kwh_baseline=E_unsb,
        full_coverage_scenarios=sum(1 for r in scen_results if r["max_unserved"] < 0.01),
        n_scenarios=len(scens),
        design_stats=design["H_stats"], wall_total=time.time() - t0,
        scenario_detail=[{k: (v if not isinstance(v, set) else sorted(v))
                          for k, v in r.items() if k != "jobs"} for r in scen_results],
        job_log=job_log)
    print(f"\n  M1 max unserved customers/hour : {M1:.1%}   (baseline {M1b:.1%})")
    print(f"  M2 critical-infra hours lost   : {M2}      (baseline {M2b})")
    print(f"  Expected unserved energy       : {E_uns:.0f} kWh (baseline {E_unsb:.0f})")
    tot_ih = sum(r["ldf_island_hours"] for r in scen_results)
    tot_ok = sum(r["ldf_feasible_hours"] for r in scen_results)
    vmin_all = min((r["ldf_v_min"] for r in scen_results), default=1.0)
    wl = max((r["ldf_worst_line_pct"] for r in scen_results), default=0.0)
    print(f"  LinDistFlow validation (E6)    : {tot_ok}/{tot_ih} island-hours electrically feasible "
          f"({100*tot_ok/max(tot_ih,1):.1f}%), Vmin={vmin_all:.3f} pu, worst line {wl:.0f}% of rating")
    from eqosystem.pipeline import run_basecase
    bc = run_basecase(design, pool, solver)
    print("=" * 62, "\nE7  Grid-connected mode: PCC economics over 24 h (TOU prices)\n" + "=" * 62)
    print(f"  island C{bc['island']:02d}: {bc['n_vars']} vars ({bc['levels']} levels, deg {bc['degree']})")
    print(f"  daily energy cost with storage arbitrage: ${bc['cost']:.0f}  "
          f"vs no-storage baseline ${bc['baseline']:.0f}  ->  saving {bc['saving_pct']:.1f}%")
    print(f"  PCC plan (import): {bc['plan']['imp']}   (export): {bc['plan']['exp']}   SOC: {bc['plan']['soc']}")
    R["E7_grid_connected"] = {k: bc[k] for k in ("island","cost","baseline","saving_pct","n_vars","levels")}
    R["E6_lindistflow"] = dict(island_hours=tot_ih, feasible=tot_ok,
                               v_min=vmin_all, worst_line_pct=wl)

    # ---------------- E2: optimality certification ----------------
    print("=" * 62, "\nE2  Islanding QUBO: backend vs exact enumeration\n" + "=" * 62)
    gaps, certs = [], 0
    for sc in scens:
        Hi, mi = ham.build_island(design, pool, sc)
        if Hi.n == 0:
            continue
        rb = solver.solve(Hi)
        rx = ExactSolver().solve(Hi)
        gap = rb["energy"] - rx["energy"]
        gaps.append(gap)
        certs += int(abs(gap) < 1e-9)
    print(f"  {certs}/{len(gaps)} islanding problems solved to certified optimality "
          f"(max gap {max(gaps) if gaps else 0:.2e})")
    R["E2"] = dict(problems=len(gaps), optimal=certs,
                   max_gap=float(max(gaps)) if gaps else 0.0)

    # ---------------- E3: analog-noise robustness ----------------
    print("=" * 62, "\nE3  Coefficient dynamic range vs analog noise\n" + "=" * 62)
    sc = max(scens, key=lambda s: len(s.dead_buses))
    Hi, mi = ham.build_island(design, pool, sc)
    exact_e = ExactSolver().solve(Hi)["energy"]
    noise_rel = 0.01                     # 1% relative coefficient noise (analog)
    ratios = [1, 10, 100, 1e3, 1e4, 1e6, 1e8]
    e3 = []
    for ratio in ratios:
        degrade = []
        for trial in range(30):
            Hn = ham.Poly()
            Hn.upper = dict(Hi.upper)
            cmax = max(abs(c) for c in Hi.terms.values())
            for k, c in Hi.terms.items():
                # rescale feasibility penalties up to emulate the target range,
                # then corrupt with device noise proportional to the LARGEST coef
                c2 = c * (ratio if c > 0 and len(k) == 1 else 1.0)
                Hn.terms[k] = c2 + rng.normal(0, noise_rel * cmax * ratio)
            r = AnnealerSolver(restarts=3, iters=1500, seed=trial).solve(Hn)
            degrade.append(Hi.evaluate(r["x"]) - exact_e)
        e3.append(dict(ratio=ratio, mean_gap=float(np.mean(degrade)),
                       p_optimal=float(np.mean([d < 1e-9 for d in degrade]))))
        print(f"  range 1e{np.log10(ratio):>3.0f} : P(optimal)={e3[-1]['p_optimal']:.2f}  "
              f"mean quality gap={e3[-1]['mean_gap']:.3f}")
    R["E3"] = dict(noise_rel=noise_rel, scenario=sc.sid, sweep=e3)

    # ---------------- E4: design vs exact MILP ----------------
    print("=" * 62, "\nE4  Design stage vs classical MILP baseline\n" + "=" * 62)
    milp = milp_design_baseline(pool)
    ratio = design["capex"] / milp["capex"] if milp["capex"] else float("nan")
    print(f"  MILP optimum capex : {milp['capex']:.1f} (x10 k$) in {milp['wall']:.2f}s")
    print(f"  {solver.name} capex: {design['capex']:.1f}  ->  ratio {ratio:.3f}")
    R["E4"] = dict(milp_capex=milp["capex"], milp_wall=milp["wall"],
                   backend_capex=design["capex"], cost_ratio=ratio)

    # ---------------- E5: native qudits vs binary (qubit) compilation ----------------
    print("=" * 62, "\nE5  Native integer (qudit) encoding vs binary compilation\n" + "=" * 62)
    from eqosystem.compile_binary import binary_expand, decode
    Hb, groups = binary_expand(H_design)
    rngc = np.random.default_rng(3)
    for _ in range(30):  # equivalence check of the compilation itself
        xb = rngc.integers(0, 2, size=Hb.n).astype(float)
        xi = decode(xb, groups, H_design.n).astype(float)
        assert abs(Hb.evaluate(xb) - H_design.evaluate(xi)) < 1e-6 * max(1, abs(H_design.evaluate(xi)))
    print(f"  native : {H_design.n:4d} vars  {len(H_design.terms):6d} terms  "
          f"deg {H_design.degree}  dyn {H_design.dynamic_range_db():.1f} dB")
    print(f"  binary : {Hb.n:4d} vars  {len(Hb.terms):6d} terms  "
          f"deg {Hb.degree}  dyn {Hb.dynamic_range_db():.1f} dB")
    trials, budget = 12, 4000
    e_nat, e_bin, w_nat, w_bin = [], [], [], []
    for t in range(trials):
        tt = time.time()
        rn = AnnealerSolver(restarts=1, iters=budget, seed=300 + t).solve(H_design)
        w_nat.append(time.time() - tt)
        e_nat.append(rn["energy"])
        tt = time.time()
        rb = AnnealerSolver(restarts=1, iters=budget, seed=300 + t).solve(Hb)
        w_bin.append(time.time() - tt)
        e_bin.append(H_design.evaluate(decode(rb["x"], groups, H_design.n).astype(float)))
    tgt = min(min(e_nat), min(e_bin))
    tol = abs(tgt) * 0.01
    p_nat = float(np.mean([e <= tgt + tol for e in e_nat]))
    p_bin = float(np.mean([e <= tgt + tol for e in e_bin]))
    print(f"  equal-budget SA ({trials} trials x {budget} iters, same seeds):")
    print(f"    native : P(within 1% of best)={p_nat:.2f}  mean E={np.mean(e_nat):.2f}  wall {np.mean(w_nat):.2f}s")
    print(f"    binary : P(within 1% of best)={p_bin:.2f}  mean E={np.mean(e_bin):.2f}  wall {np.mean(w_bin):.2f}s")
    R["E5"] = dict(
        native=dict(vars=H_design.n, terms=len(H_design.terms), degree=H_design.degree,
                    dyn_db=H_design.dynamic_range_db(), p_success=p_nat,
                    mean_energy=float(np.mean(e_nat)), mean_wall_s=float(np.mean(w_nat))),
        binary=dict(vars=Hb.n, terms=len(Hb.terms), degree=Hb.degree,
                    dyn_db=Hb.dynamic_range_db(), p_success=p_bin,
                    mean_energy=float(np.mean(e_bin)), mean_wall_s=float(np.mean(w_bin))),
        trials=trials, iters=budget)

    # ---------------- resource accounting ----------------
    dis_jobs = [j for j in job_log if j["stage"] == "dispatch"]
    isl_jobs = [j for j in job_log if j["stage"] == "island"]
    qubit_equiv = sum(int(np.ceil(np.log2(u + 1))) for u in H_design.upper.values())
    R["resources"] = dict(
        total_quantum_jobs=1 + len(isl_jobs) + len(dis_jobs),
        design_vars=H_design.n, design_terms=len(H_design.terms),
        design_degree=H_design.degree,
        design_qudits=H_design.n, design_qubit_equivalent_binary=qubit_equiv,
        island_vars_mean=float(np.mean([j["n_vars"] for j in isl_jobs])) if isl_jobs else 0,
        dispatch_vars_mean=float(np.mean([j["n_vars"] for j in dis_jobs])) if dis_jobs else 0,
        wall_clock_total_s=time.time() - t0)
    print("\nResource summary:", json.dumps(R["resources"], indent=2))

    # ---------------- plots ----------------
    make_plots(R, scen_results, e3, args.outdir)
    import os as _os
    _os.makedirs(args.outdir, exist_ok=True)
    with open(f"{args.outdir}/results_{solver.name}.json", "w") as f:
        json.dump(R, f, indent=2, default=str)
    print(f"\nSaved {args.outdir}/results_{solver.name}.json and plots.")


def make_plots(R, scen_results, e3, outdir):
    import os
    os.makedirs(outdir, exist_ok=True)
    ids = [r["sid"] for r in scen_results]
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.2))
    axes = [axs[0, 0], axs[0, 1], axs[1, 0]]
    w = 0.4
    axes[0].bar([i - w / 2 for i in ids], [r["base_max_unserved"] * 100 for r in scen_results],
                width=w, label="Baseline (no islanding)", color="#B23A48")
    axes[0].bar([i + w / 2 for i in ids], [r["max_unserved"] * 100 for r in scen_results],
                width=w, label="eQoSystem", color="#2E86AB")
    axes[0].set_xlabel("Scenario"); axes[0].set_ylabel("Max unserved customers (%)")
    axes[0].set_title("M1: worst-hour unserved customers"); axes[0].legend()

    axes[1].bar([i - w / 2 for i in ids], [r["base_crit_hours"] for r in scen_results],
                width=w, label="Baseline", color="#B23A48")
    eqo_crit = [r["crit_hours"] for r in scen_results]
    axes[1].bar([i + w / 2 for i in ids], eqo_crit,
                width=w, label="eQoSystem", color="#2E86AB")
    if max(eqo_crit) == 0:  # all zero -> draw a visible marker line so it isn't "missing"
        axes[1].plot([min(ids) - 0.5, max(ids) + 0.5], [0, 0],
                     color="#2E86AB", lw=3, solid_capstyle="round")
        axes[1].annotate("eQoSystem = 0 in every scenario",
                         xy=(np.mean(ids), 0), xytext=(np.mean(ids), max(r["base_crit_hours"] for r in scen_results) * 0.5),
                         ha="center", fontsize=9, color="#2E86AB",
                         arrowprops=dict(arrowstyle="->", color="#2E86AB"))
    axes[1].set_xlabel("Scenario"); axes[1].set_ylabel("Critical bus-hours unserved")
    axes[1].set_title("M2: critical infrastructure outage"); axes[1].legend()

    xs = [np.log10(p["ratio"]) for p in e3]
    axes[2].plot(xs, [p["p_optimal"] for p in e3], "o-", color="#2E86AB", label="P(optimal)")
    axes[2].axvline(R["E1"]["design_stats"]["dyn_range_db"] / 10.0, ls="--", c="#3BB273",
                    label=f"this framework ({R['E1']['design_stats']['dyn_range_db']:.0f} dB)")
    axes[2].axvline(8, ls="--", c="#B23A48", label="Phase-2 code (80 dB)")
    axes[2].set_xlabel("Coefficient dynamic range (log10)")
    axes[2].set_ylabel("P(ground state) under 1% analog noise")
    axes[2].set_title("E3: why coefficient conditioning matters"); axes[2].legend(fontsize=8)

    ax4 = axs[1, 1]
    e5 = R.get("E5")
    if e5:
        cats = ["Variables", "Polynomial\nterms", "SA wall-clock\nper attempt"]
        ratios = [e5["binary"]["vars"] / e5["native"]["vars"],
                  e5["binary"]["terms"] / e5["native"]["terms"],
                  e5["binary"]["mean_wall_s"] / max(e5["native"]["mean_wall_s"], 1e-9)]
        bars = ax4.barh(cats, ratios, color="#B23A48", height=0.55)
        ax4.axvline(1.0, color="#2E86AB", lw=2)
        ax4.text(1.02, -0.42, "native qudit encoding = 1x", color="#2E86AB", fontsize=9, rotation=0)
        for b, r in zip(bars, ratios):
            ax4.text(b.get_width() + 0.08, b.get_y() + b.get_height() / 2,
                     f"{r:.1f}x", va="center", fontsize=10, color="#B23A48")
        ax4.set_xlim(0, max(ratios) * 1.25)
        ax4.set_xlabel("binary (qubit) compilation cost, relative to native")
        ax4.set_title("E5: cost of forgoing native integer encoding "
                      f"\nP(within 1% of best): native {e5['native']['p_success']:.2f} vs binary {e5['binary']['p_success']:.2f}",
                      fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{outdir}/experiments.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
