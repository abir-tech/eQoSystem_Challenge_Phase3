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
    practice for annealing hardware as well (best-of-N with a seeded sample)."""
    x = np.zeros(H.n, dtype=int)
    uncovered = set(range(2, grid.N_BUSES + 1))
    order = sorted(range(len(pool)),
                   key=lambda c: -len(set(pool[c]["buses"])))
    chosen = []
    while uncovered:
        best = max(range(len(pool)),
                   key=lambda c: len(set(pool[c]["buses"]) & uncovered)
                   / (1 + ham.design_demand(pool[c]) / 500.0))
        gain = set(pool[best]["buses"]) & uncovered
        if not gain:
            break
        chosen.append(best)
        uncovered -= gain
    for c in chosen:
        x[meta["b"][c] - 1] = 1
        need = ham.design_demand(pool[c])
        # cheapest firm kW first: DG, then BESS, then PV
        for k in ("DG", "BESS", "PV"):
            a = ham.ASSETS[k]
            while need > 0 and x[meta["n"][(c, k)] - 1] < ham.asset_umax(k):
                x[meta["n"][(c, k)] - 1] += 1
                need -= a["kw"] * a["firm"]
    return x


def run_design(pool, solver):
    H, meta = ham.build_design(pool)
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


def run_scenario(design, pool, scen, solver):
    """Solve islanding QUBO + one dispatch per active island; return per-hour
    served/unserved accounting."""
    out = dict(sid=scen.sid, jobs=[])
    Hi, mi = ham.build_island(design, pool, scen)
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
    lf = scen.load_factor
    soc = {c: plans[c]["md"]["soc0"] for c in active}
    hourly_unserved_cust, hourly_crit_unserved = [], []
    unserved_energy = 0.0
    ldf_checks = []
    crit_ok_h, nc_frac_h = {}, {}
    for hi, h in enumerate(scen.hours):
        for c in active:
            md = plans[c]["md"]; plan = plans[c]["plan"]
            P = md["P"]; caps = md["caps"]
            t = next(i for i, hb in enumerate(md["buckets"]) if h in hb)
            reach = mi["info"][c]["reach"]
            L_cr_h = sum(grid.LOAD_P[b - 1] for b in reach if b in grid.CRITICAL_BUSES) \
                * grid.LOAD_PROFILE[h] * lf
            L_nc_h = sum(grid.LOAD_P[b - 1] for b in reach if b not in grid.CRITICAL_BUSES) \
                * grid.LOAD_PROFILE[h] * lf
            pv_h = min(plan["pv"][t] * P,
                       caps["pv_kw"] * grid.SOLAR_PROFILE[h] * scen.r_factor)
            dg_h = min(plan["dg"][t] * P, caps["dg_kw"])
            dis_h = min(plan["dis"][t] * P, caps["bess_kw"], soc[c])
            chg_h = min(plan["chg"][t] * P, caps["bess_kw"],
                        max(0.0, (caps["bess_kwh"] - soc[c]) / ham.BESS_ETA),
                        max(0.0, pv_h + dg_h - L_cr_h))   # charge only from surplus
            soc[c] = min(caps["bess_kwh"], soc[c] + ham.BESS_ETA * chg_h - dis_h)
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
            * grid.LOAD_PROFILE[h] * lf
    base_crit_hours = sum(len(scen.dead_buses & grid.CRITICAL_BUSES) for _ in scen.hours)

    ldf_pass = sum(1 for r in ldf_checks if r["feasible"])
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
