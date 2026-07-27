"""End-to-end pipeline: design -> per-scenario islanding -> per-island dispatch,
then honest 24-hour metric evaluation against the challenge criteria:

  M1  max fraction of customers unserved in any hour, over all contingencies
  M2  total critical-infrastructure hours unserved, over all contingencies
  M3  capital cost of grid upgrades

Baselines: (a) legacy grid, no islanding -- every de-energized bus stays dark
for the repair window; (b) exact MILP for the design stage (classical
reference demanded by the Phase-3 rubric).
"""
import numpy as np

from . import grid
from . import lindistflow as ldf, hamiltonians as ham
from .solvers import get_solver


def greedy_seed(H, meta, pool):
    """Feasible classical warm start: greedy set cover, then buy the cheapest
    firm-capacity mix per selected island. Hybrid warm-starting is standard
    practice for annealing hardware as well (best-of-N with a seeded sample).

    Under the W1b trust-region encoding the decision variable is the correction
    d = n - base, so the seed is written in those coordinates."""
    x = np.zeros(H.n, dtype=int)
    chosen, units = ham.greedy_portfolio(pool)
    base = meta.get("base", {})
    for c in chosen:
        x[meta["b"][c] - 1] = 1
    for (c, k), u in units.items():
        vid = meta["n"][(c, k)]
        x[vid - 1] = int(np.clip(u - base.get((c, k), 0), 0, H.upper[vid]))
    return x


def run_design(pool, solver, delta=True, radius=3, slack_max=4):
    if delta:
        _chosen, seed_units = ham.greedy_portfolio(pool)
        H, meta = ham.build_design(pool, seed_units=seed_units, radius=radius,
                                   slack_max=slack_max)
    else:
        H, meta = ham.build_design(pool, slack_max=slack_max)
    seed = greedy_seed(H, meta, pool)
    try:
        res = solver.solve(H, warm_start=seed)
    except TypeError:            # Dirac-3 path takes no warm start; polish covers it
        res = solver.solve(H)
    design = ham.decode_design(res["x"], meta)
    design["energy"] = res["energy"]
    design["wall"] = res["wall"]
    design["H_stats"] = dict(n_vars=H.n, n_terms=len(H.terms), degree=H.degree,
                             dyn_range_db=H.dynamic_range_db())
    # feasibility repair: penalty solutions may sit a step short of the
    # capacity gate; add cheapest firm units until every island is feasible
    for c in design["selected"]:
        port = design["portfolio"][c]
        D = ham.design_demand(pool[c])
        firm = lambda: sum(port[k] * ham.ASSETS[k]["kw"] * ham.ASSETS[k]["firm"]
                           for k in ham.ASSET_KEYS)
        while firm() < D:
            k_best = min((k for k in ham.ASSET_KEYS if port[k] < ham.asset_umax(k)),
                         key=lambda k: (ham.ASSETS[k]["cost"] + ham.ASSETS[k]["op"])
                         / (ham.ASSETS[k]["kw"] * ham.ASSETS[k]["firm"]),
                         default=None)
            if k_best is None:
                break
            port[k_best] += 1
            design["capex"] += ham.ASSETS[k_best]["cost"] + ham.ASSETS[k_best]["op"]
    return design, H, res


def run_scenario(design, pool, scen, solver, voltage_aware=True,
                 export_aware=False):
    """Solve islanding QUBO + one dispatch per active island; return per-hour
    served/unserved accounting."""
    out = dict(sid=scen.sid, jobs=[])
    Hi, mi = ham.build_island(design, pool, scen, voltage_aware=voltage_aware,
                              export_aware=export_aware)
    if Hi.n == 0:
        active = []
    else:
        ri = solver.solve(Hi)
        active = [c for c in mi["z"] if ri["x"][mi["z"][c] - 1] > 0.5]
        out["jobs"].append(dict(stage="island", n_vars=Hi.n, degree=Hi.degree,
                                terms=len(Hi.terms), wall=ri["wall"], energy=ri["energy"]))
    worst_h = mi["worst_hour"] if Hi.n else scen.hours[0]

    # dead-bus -> island assignment (higher restored value claims ties)
    assign = {}
    for c in sorted(active, key=lambda c: -mi["info"][c]["value"]):
        for b in mi["info"][c]["reach"]:
            assign.setdefault(b, c)

    # W7 -- cross-island export: a bus stranded from its own island (no
    # DER hub on its side of the break) may be picked up by a tie-adjacent
    # island that IS energized, bounded by tie capacity and the exporter's
    # surplus. Only pairs the Hamiltonian rewarded, and only if both are on.
    exported = {}
    for (ca, cb), ex in mi.get("exports", {}).items():
        if ca not in active or cb not in active:
            continue
        donor = ca if mi["info"][ca]["cap"] >= mi["info"][cb]["cap"] else cb
        for b in ex["buses"]:
            if b not in assign:
                assign[b] = donor
                exported[b] = donor

    # ---- Stage 3: multi-period (SOC-coupled) dispatch per active island ----
    plans = {}
    for c in active:
        Hmp, md = ham.build_dispatch_mp(design, pool, scen, c)
        rd = solver.solve(Hmp)
        plan = {k: [int(rd["x"][md["vars"][(k, t)] - 1]) for t in range(md["nb"])]
                for k in ("pv", "dg", "dis", "chg", "nc")}
        plans[c] = dict(plan=plan, md=md)
        out["jobs"].append(dict(stage="dispatch_mp", island=c, n_vars=Hmp.n,
                                degree=Hmp.degree, terms=len(Hmp.terms),
                                levels=sum(Hmp.upper[v] + 1 for v in Hmp.upper
                                           if Hmp.upper[v] >= 1),
                                buckets=md["nb"], step_kw=md["P"], wall=rd["wall"]))

    # ---- hourly physical simulation: follow the MP plan, track SOC ----------
    soc = {c: plans[c]["md"]["soc0"] for c in active}
    hourly_unserved_cust, hourly_crit_unserved = [], []
    unserved_energy = 0.0
    ldf_checks = []
    crit_ok_h, nc_frac_h = {}, {}
    for hi, h in enumerate(scen.hours):
        lf = scen.load_factor_at(h)      # bucket-varying in tree mode
        for c in active:
            md = plans[c]["md"]; plan = plans[c]["plan"]
            P = md["P"]; caps = md["caps"]
            t = next(i for i, hb in enumerate(md["buckets"]) if h in hb)
            # W7: buses exported to c are SERVED BY c, so they must also
            # be CHARGED to c. Crediting them without adding their load
            # would let an island serve more than it generates -- the
            # supply-capped rule this pipeline holds everywhere else.
            reach = mi["info"][c]["reach"] | {b for b, d in exported.items()
                                              if d == c}
            L_cr_h = sum(grid.LOAD_P[b - 1] for b in reach if b in grid.CRITICAL_BUSES) \
                * grid.LOAD_PROFILE[h] * lf
            L_nc_h = sum(grid.LOAD_P[b - 1] for b in reach if b not in grid.CRITICAL_BUSES) \
                * grid.LOAD_PROFILE[h] * lf
            pv_h = min(plan["pv"][t] * P,
                       caps["pv_kw"] * grid.SOLAR_PROFILE[h] * scen.r_factor_at(h))
            dg_h = min(plan["dg"][t] * P, caps["dg_kw"])
            dis_h = min(plan["dis"][t] * P, caps["bess_kw"], soc[c])
            chg_h = min(plan["chg"][t] * P, caps["bess_kw"],
                        max(0.0, (caps["bess_kwh"] - soc[c]) / ham.BESS_ETA0),
                        max(0.0, pv_h + dg_h - L_cr_h))   # charge only from surplus
            # W9 physics truth: efficiency falls with charging power
            soc[c] = min(caps["bess_kwh"],
                         soc[c] + ham.eta_charge(chg_h, caps["bess_kw"]) * chg_h - dis_h)
            net = pv_h + dg_h + dis_h - chg_h
            crit_ok_h[(c, h)] = net >= L_cr_h - P
            nc_frac_h[(c, h)] = min(1.0, min(plan["nc"][t] * P,
                                             max(0.0, net - L_cr_h)) / L_nc_h) \
                if L_nc_h > 1 else 1.0
            ldf_checks.append(ldf.check_island(
                pool[c], scen, h, nc_frac_h[(c, h)], crit_ok_h[(c, h)],
                min(net, L_cr_h + L_nc_h), closed_ties=mi.get("ties_closed", ())))
        cust_uns, crit_uns = 0, 0
        for b in scen.dead_buses:
            if b == 1:
                continue
            c = assign.get(b)
            if c is None:
                cust_uns += grid.CUSTOMERS[b - 1]
                unserved_energy += grid.LOAD_P[b - 1] * grid.LOAD_PROFILE[h] * lf
                if b in grid.CRITICAL_BUSES:
                    crit_uns += 1
            elif b in grid.CRITICAL_BUSES:
                if not crit_ok_h[(c, h)]:
                    crit_uns += 1
                    cust_uns += grid.CUSTOMERS[b - 1]
                    unserved_energy += grid.LOAD_P[b - 1] * grid.LOAD_PROFILE[h] * lf
            else:
                miss = 1.0 - nc_frac_h[(c, h)]
                cust_uns += grid.CUSTOMERS[b - 1] * miss
                unserved_energy += grid.LOAD_P[b - 1] * grid.LOAD_PROFILE[h] * lf * miss
        hourly_unserved_cust.append(cust_uns / grid.TOTAL_CUSTOMERS)
        hourly_crit_unserved.append(crit_uns)

    # baseline: no islanding -- every dead bus dark all window
    base_hourly = []
    base_energy = 0.0
    for h in scen.hours:
        cu = sum(grid.CUSTOMERS[b - 1] for b in scen.dead_buses if b != 1)
        base_hourly.append(cu / grid.TOTAL_CUSTOMERS)
        base_energy += sum(grid.LOAD_P[b - 1] for b in scen.dead_buses if b != 1) \
            * grid.LOAD_PROFILE[h] * scen.load_factor_at(h)
    base_crit_hours = sum(len(scen.dead_buses & grid.CRITICAL_BUSES) for _ in scen.hours)

    ldf_pass = sum(1 for r in ldf_checks if r["feasible"])
    # W2/W7 telemetry
    v_preds = [mi["info"][c].get("v_min_pred") for c in mi["info"]
               if mi["info"][c].get("v_min_pred") is not None]
    out.update(
        v_min_predicted=(min(v_preds) if v_preds else None),
        n_candidates_v_at_risk=sum(1 for c in mi["info"]
                                   if mi["info"][c].get("v_violation", 0.0) > 0.0),
        v_penalty_total=sum(mi["info"][c].get("v_penalty", 0.0) for c in mi["info"]),
        active_v_at_risk=sum(1 for c in active
                             if mi["info"][c].get("v_violation", 0.0) > 0.0),
        n_exported_buses=len(exported),
        export_pairs=len(mi.get("exports", {})))
    out.update(active=active,
               max_unserved=max(hourly_unserved_cust, default=0.0),
               crit_hours=sum(hourly_crit_unserved),
               unserved_energy=unserved_energy,
               ldf_island_hours=len(ldf_checks),
               ldf_feasible_hours=ldf_pass,
               ldf_v_min=min((r["v_min"] for r in ldf_checks), default=1.0),
               ldf_worst_line_pct=max((r["worst_line_pct"] for r in ldf_checks), default=0.0),
               base_max_unserved=max(base_hourly, default=0.0),
               base_crit_hours=base_crit_hours,
               base_energy=base_energy,
               n_dead=len(scen.dead_buses))
    return out


def milp_design_baseline(pool):
    """Exact classical design (HiGHS branch-and-bound) -- the rubric's
    'classical baseline on the same problem instance'."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    import time
    m = len(pool)
    K = ham.ASSET_KEYS
    nb, nn, ny = m, m * len(K), len(grid.TIE_SWITCHES)
    N = nb + nn + ny
    cost = np.zeros(N)
    for c in range(m):
        for j, k in enumerate(K):
            cost[nb + c * len(K) + j] = ham.ASSETS[k]["cost"] + ham.ASSETS[k]["op"]
    cost[nb + nn:] = ham.SWITCH_COST

    A, lb, ub = [], [], []
    for bus in range(2, grid.N_BUSES + 1):        # coverage >= 1
        row = np.zeros(N)
        for c in range(m):
            if bus in pool[c]["buses"]:
                row[c] = 1
        A.append(row); lb.append(1); ub.append(np.inf)
    for c in range(m):                           # firm capacity >= D_c * b_c
        row = np.zeros(N)
        D = ham.design_demand(pool[c])
        row[c] = -D
        for j, k in enumerate(K):
            row[nb + c * len(K) + j] = ham.ASSETS[k]["kw"] * ham.ASSETS[k]["firm"]
        A.append(row); lb.append(0); ub.append(np.inf)
        for j, k in enumerate(K):                # n <= U * b
            row = np.zeros(N)
            row[nb + c * len(K) + j] = 1
            row[c] = -ham.asset_umax(k)
            A.append(row); lb.append(-np.inf); ub.append(0)

    ubs = np.ones(N)
    for c in range(m):
        for j, k in enumerate(K):
            ubs[nb + c * len(K) + j] = ham.asset_umax(k)
    t0 = time.time()
    res = milp(c=cost, integrality=np.ones(N),
               constraints=LinearConstraint(np.array(A), lb, ub),
               bounds=Bounds(np.zeros(N), ubs))
    return dict(capex=float(res.fun) if res.fun is not None else float("nan"), wall=time.time() - t0, status=res.message)


def simulate_plan(pool, scen, c, plan, md, reach):
    """Score a dispatch plan by the same hourly physics used for the metrics.

    Both the Hamiltonian plan and the mixed-integer plan go through this, so the
    E12 comparison is apples-to-apples even though the two objectives differ."""
    caps, P = md["caps"], md["P"]
    crit = [b for b in reach if b in grid.CRITICAL_BUSES]
    ncrit = [b for b in reach if b not in grid.CRITICAL_BUSES]
    soc = md["soc0"]
    crit_short_h, unserved_kwh, dg_kwh, nc_served = 0, 0.0, 0.0, []
    for h in scen.hours:
        t = next(i for i, hb in enumerate(md["buckets"]) if h in hb)
        lf = scen.load_factor_at(h) * grid.LOAD_PROFILE[h]
        L_cr = sum(grid.LOAD_P[b - 1] for b in crit) * lf
        L_nc = sum(grid.LOAD_P[b - 1] for b in ncrit) * lf
        pv = min(plan["pv"][t] * P, caps["pv_kw"] * grid.SOLAR_PROFILE[h]
                 * scen.r_factor_at(h))
        dg = min(plan["dg"][t] * P, caps["dg_kw"])
        dis = min(plan["dis"][t] * P, caps["bess_kw"], soc)
        chg = min(plan["chg"][t] * P, caps["bess_kw"],
                  max(0.0, (caps["bess_kwh"] - soc) / ham.BESS_ETA0),
                  max(0.0, pv + dg - L_cr))
        soc = min(caps["bess_kwh"], soc + ham.eta_charge(chg, caps["bess_kw"]) * chg - dis)
        net = pv + dg + dis - chg
        dg_kwh += dg
        if net < L_cr - P:
            crit_short_h += 1
            unserved_kwh += (L_cr - net)
            nc_served.append(0.0)
        else:
            served = min(plan["nc"][t] * P, max(0.0, net - L_cr))
            frac = min(1.0, served / L_nc) if L_nc > 1 else 1.0
            nc_served.append(frac)
            unserved_kwh += (1.0 - frac) * L_nc
    return dict(crit_short_hours=crit_short_h, unserved_kwh=unserved_kwh,
                dg_kwh=dg_kwh,
                nc_served_frac=float(np.mean(nc_served)) if nc_served else 1.0)


# ------------------- classical baseline for the DISPATCH stage (experiment E12)
def milp_dispatch_baseline(design, pool, scen, c, n_tangents=6):
    """Exact classical dispatch on the identical instance, via HiGHS.

    Stages 1 and 2 already had classical baselines (mixed-integer for design,
    exhaustive enumeration for islanding). Stage 3 had none, which left the
    rubric's "comparison against a non-quantum method on the same problem
    instance" unmet for a third of the pipeline.

    Same discrete decision space as the Hamiltonian: integer setpoints in units
    of P per time bucket, plus an integer state-of-charge level. The difference
    is that this enforces power balance and the SOC recursion as HARD
    constraints with an explicit unserved-critical slack, where the Hamiltonian
    uses quadratic penalties. The convex diesel cubic is carried exactly in
    epigraph form: kappa*p^3 is convex on p >= 0, so tangent cuts are a valid
    lower bound that tightens with n_tangents.

    Because the two objectives are not literally identical, E12 compares the
    PHYSICAL outcome both plans are scored by -- simulate_plan -- not objective
    values."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    import time

    _H, md = ham.build_dispatch_mp(design, pool, scen, c)
    nb, P, E = md["nb"], md["P"], md["E"]
    caps, scale = md["caps"], max(pool[c]["load_kw"], 1.0)
    L_cr, L_nc, pv_av = md["L_cr"], md["L_nc"], md["pv_av"]
    hrs = [len(b) for b in md["buckets"]]

    K = 6
    n = K * nb + nb + nb

    def ix(k, t):
        return K * t + k

    def IZ(t):
        return K * nb + t

    def IS(t):
        return K * nb + nb + t

    ub = np.zeros(n)
    integrality = np.zeros(n)
    for t in range(nb):
        caps_t = (pv_av[t] / P, caps["dg_kw"] / P, caps["bess_kw"] / P,
                  min(caps["bess_kw"], caps["pv_kw"]) / P,
                  L_nc[t] / P, caps["bess_kwh"] / E)
        for k, cap in enumerate(caps_t):
            ub[ix(k, t)] = max(int(cap), 0)
            integrality[ix(k, t)] = 1
        ub[IZ(t)] = np.inf
        ub[IS(t)] = max(L_cr[t], 1.0)

    cost = np.zeros(n)
    ub_dg = max(ub[ix(1, 0)], 1)
    kappa = 0.35 * (8.0 * P / scale) / (3.0 * ub_dg ** 2)
    for t in range(nb):
        h = hrs[t]
        cost[ix(4, t)] = -8.0 * h * P / scale
        cost[ix(2, t)] = 0.5 * h * P / scale
        cost[ix(3, t)] = 0.5 * h * P / scale
        cost[ix(1, t)] = 2.0 * h * P / scale
        cost[IZ(t)] = 1.0
        cost[IS(t)] = 1.0e3 / scale

    A, lo, hi = [], [], []
    for t in range(nb):
        r = np.zeros(n)                                   # hard power balance
        r[ix(0, t)] = r[ix(2, t)] = r[ix(1, t)] = P
        r[ix(3, t)] = r[ix(4, t)] = -P
        r[IS(t)] = 1.0
        A.append(r); lo.append(L_cr[t]); hi.append(L_cr[t])

        r = np.zeros(n)                                   # hard SOC recursion
        r[ix(5, t)] = E
        r[ix(2, t)] = P * hrs[t]
        r[ix(3, t)] = -ham.BESS_ETA * P * hrs[t]
        if t > 0:
            r[ix(5, t - 1)] = -E
        A.append(r)
        lo.append(md["soc0"] if t == 0 else 0.0)
        hi.append(md["soc0"] if t == 0 else 0.0)

        for j in range(n_tangents):                       # z >= tangent to kappa p^3
            p0 = ub_dg * j / max(n_tangents - 1, 1)
            r = np.zeros(n)
            r[IZ(t)] = 1.0
            r[ix(1, t)] = -kappa * 3.0 * p0 ** 2
            A.append(r); lo.append(-kappa * 2.0 * p0 ** 3); hi.append(np.inf)

    t0 = time.time()
    res = milp(c=cost, integrality=integrality,
               constraints=LinearConstraint(np.array(A), lo, hi),
               bounds=Bounds(np.zeros(n), ub))
    wall = time.time() - t0
    if res.x is None:
        return dict(feasible=False, wall=wall, status=res.message, plan=None, md=md)
    x = res.x
    plan = {k: [int(round(x[ix(i, t)])) for t in range(nb)]
            for i, k in enumerate(("pv", "dg", "dis", "chg", "nc"))}
    return dict(feasible=True, wall=wall, status=res.message, plan=plan, md=md,
                objective=float(res.fun),
                unserved_crit_kw=[float(x[IS(t)]) for t in range(nb)])


# ------------------------------------------- W10: VSS / EVPI (experiment E10)
VOLL_PER_KWH = 10.0      # value of lost load, $/kWh (conservative)
COST_UNIT = 1.0e4        # capex is carried in units of 10 k$
PROJECT_YEARS = 20.0     # straight-line amortization, no discounting
ANNUAL_EVENTS = 12.0     # significant N-1 events per year on a feeder of this class
MARGIN_GRID = (1.00, 1.10, 1.25, 1.40, 1.60)


def total_cost(capex_units, unserved_kwh):
    """Annualized two-stage objective, in units of 10 k$ per year.

    Capex is a one-off investment and unserved energy is per contingency, so the
    two cannot be added directly -- an earlier version did, and the recourse term
    came out at 0.2% of capex, which made the comparison meaningless."""
    return (capex_units / PROJECT_YEARS
            + ANNUAL_EVENTS * VOLL_PER_KWH * unserved_kwh / COST_UNIT)


def design_is_feasible(design, pool, margin):
    """Every selected island must cover its own demand, and all buses covered.

    A design that fails this is cheaper only because it is infeasible; including
    it would let the study "win" by under-building."""
    saved = ham.DESIGN_MARGIN
    ham.DESIGN_MARGIN = margin
    try:
        for c in design["selected"]:
            port = design["portfolio"][c]
            firm = sum(port[k] * ham.ASSETS[k]["kw"] * ham.ASSETS[k]["firm"]
                       for k in ham.ASSET_KEYS)
            if firm < ham.design_demand(pool[c]) - 1e-6:
                return False
    finally:
        ham.DESIGN_MARGIN = saved
    covered = set()
    for c in design["selected"]:
        covered |= set(pool[c]["buses"])
    return set(range(2, grid.N_BUSES + 1)) <= covered


def design_at_margin(pool, solver, margin):
    """First-stage design at a given sizing margin (the here-and-now decision)."""
    saved = ham.DESIGN_MARGIN
    ham.DESIGN_MARGIN = margin
    try:
        design, _H, _res = run_design(pool, solver)
    finally:
        ham.DESIGN_MARGIN = saved
    return design


def vss_evpi(pool, scens, solver, margins=MARGIN_GRID, voltage_aware=True):
    """Value of the Stochastic Solution and Expected Value of Perfect Information.

    SCOPE. The design Hamiltonian is scenario-independent: it sizes each island
    to a deterministic base-load requirement, so it is robust-by-construction
    rather than a scenario-coupled stochastic program, and classical VSS against
    it would be vacuous. What IS a here-and-now decision under uncertainty is
    the SIZING MARGIN, so the first-stage decision space is parameterized by it.
    A fully scenario-coupled design Hamiltonian is future work."""
    rows = {}
    for m in margins:
        d = design_at_margin(pool, solver, m)
        feasible = design_is_feasible(d, pool, m)
        costs, uns = [], []
        for sc in scens:
            r = run_scenario(d, pool, sc, solver, voltage_aware=voltage_aware)
            uns.append(r["unserved_energy"])
            costs.append(total_cost(d["capex"], r["unserved_energy"]))
        rows[m] = dict(capex=d["capex"], costs=costs, feasible=bool(feasible),
                       mean_cost=float(np.mean(costs)),
                       mean_unserved=float(np.mean(uns)))

    usable = [m for m in margins if rows[m]["feasible"]]
    if not usable:
        raise RuntimeError("no feasible design at any margin in the grid")
    m_star = min(usable, key=lambda m: rows[m]["mean_cost"])
    RP = rows[m_star]["mean_cost"]
    m_ev = min(usable)
    EEV = rows[m_ev]["mean_cost"]
    WS = float(np.mean([min(rows[m]["costs"][i] for m in usable)
                        for i in range(len(scens))]))

    sens = []
    for ev in (4.0, 12.0, 24.0, 52.0):
        for voll in (2.0, 5.0, 10.0, 25.0, 50.0):
            cost = {m: rows[m]["capex"] / PROJECT_YEARS
                    + ev * voll * rows[m]["mean_unserved"] / COST_UNIT
                    for m in usable}
            best = min(usable, key=lambda m: cost[m])
            sens.append(dict(annual_events=ev, voll=voll, best_margin=best,
                             VSS=cost[min(usable)] - cost[best]))

    return dict(
        margins=list(margins), usable_margins=usable,
        infeasible_margins=[m for m in margins if not rows[m]["feasible"]],
        voll_per_kwh=VOLL_PER_KWH, project_years=PROJECT_YEARS,
        annual_events=ANNUAL_EVENTS,
        per_margin={str(m): {k: v for k, v in rows[m].items() if k != "costs"}
                    for m in margins},
        margin_stochastic=m_star, margin_mean_value=m_ev,
        RP=RP, EEV=EEV, WS=WS, VSS=EEV - RP, EVPI=RP - WS,
        sensitivity=sens, n_scenarios=len(scens))


# ------------------------------------------------ grid-connected mode (PCC)
TOU_PRICE = [0.08]*6 + [0.12]*10 + [0.20]*5 + [0.08]*3     # $/kWh import
EXPORT_PRICE = 0.06                                        # $/kWh export


def run_basecase(design, pool, solver, island=None):
    """Grid-connected economic operation over 24 h (challenge: 'grid-connected
    mode with PCC power export'). One Hamiltonian: 6x4h buckets with PV,
    battery charge/discharge, SOC recursion, and PCC import/export at
    time-of-use prices. Returns cost vs a no-storage baseline."""
    from . import hamiltonians as ham
    import numpy as np
    c = island if island is not None else max(
        design["selected"], key=lambda k: design["portfolio"][k]["BESS"])
    port = design["portfolio"][c]
    cand = pool[c]
    pv_kw = port["PV"] * ham.ASSETS["PV"]["kw"]
    bess_kw = port["BESS"] * ham.ASSETS["BESS"]["kw"]
    bess_kwh = port["BESS"] * ham.BESS_KWH_PER_UNIT
    P, E = 25.0, ham.MP_ESTEP
    buckets = [list(range(t*4, t*4+4)) for t in range(6)]
    load = [np.mean([cand["load_kw"]*grid.LOAD_PROFILE[h] for h in hb]) for hb in buckets]
    pv_av = [pv_kw*np.mean([grid.SOLAR_PROFILE[h] for h in hb]) for hb in buckets]
    price = [np.mean([TOU_PRICE[h] for h in hb]) for hb in buckets]
    H = ham.Poly(); V = {}
    scale = max(cand["load_kw"], 1.0)
    for t in range(6):
        V[("pv",t)]  = H.new_var(f"pv{t}",  max(int(pv_av[t]//P),0))
        V[("dis",t)] = H.new_var(f"dis{t}", max(int(bess_kw//P),0))
        V[("chg",t)] = H.new_var(f"chg{t}", max(int(bess_kw//P),0))
        V[("imp",t)] = H.new_var(f"imp{t}", max(int(1.2*max(load)//P),0))
        V[("exp",t)] = H.new_var(f"exp{t}", max(int((pv_kw+bess_kw)//P),0))
        V[("soc",t)] = H.new_var(f"soc{t}", max(int(bess_kwh//E),0))
    W_ECON = 40.0            # economics weight vs balance penalty
    for t in range(6):
        h_t = 4
        H.add( W_ECON*price[t]*h_t*P/scale, V[("imp",t)])
        H.add(-W_ECON*EXPORT_PRICE*h_t*P/scale, V[("exp",t)])
        H.add( W_ECON*0.02*h_t*P/scale, V[("dis",t)])
        H.add( W_ECON*0.02*h_t*P/scale, V[("chg",t)])
        lin = {V[("pv",t)]: P/scale, V[("dis",t)]: P/scale, V[("imp",t)]: P/scale,
               V[("chg",t)]: -P/scale, V[("exp",t)]: -P/scale}
        H.add_square(60.0, lin, -load[t]/scale)
        lin2 = {V[("soc",t)]: E/bess_kwh, V[("chg",t)]: -ham.BESS_ETA*P*h_t/bess_kwh,
                V[("dis",t)]: P*h_t/bess_kwh}
        if t > 0: lin2[V[("soc",t-1)]] = -E/bess_kwh
        H.add_square(60.0, lin2, -(0.5*bess_kwh/bess_kwh) if t == 0 else 0.0)
    r = solver.solve(H)
    x = r["x"]
    cost = sum(price[t]*4*x[V[("imp",t)]-1]*P - EXPORT_PRICE*4*x[V[("exp",t)]-1]*P
               for t in range(6))
    base = sum(price[t]*4*max(0.0, load[t]-pv_av[t]) - EXPORT_PRICE*4*max(0.0, pv_av[t]-load[t])
               for t in range(6))
    plan = {k: [int(x[V[(k,t)]-1]) for t in range(6)] for k in ("pv","dis","chg","imp","exp","soc")}
    lv = sum(H.upper[v]+1 for v in H.upper if H.upper[v] >= 1)
    return dict(island=c, cost=cost, baseline=base,
                saving_pct=100*(base-cost)/max(base,1e-9), plan=plan,
                n_vars=H.n, levels=lv, degree=H.degree, wall=r["wall"])
