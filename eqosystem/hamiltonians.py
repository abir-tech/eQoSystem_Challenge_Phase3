"""Polynomial Hamiltonian builders for the three pipeline stages.

Design principles (these are the Phase-3 fixes):
  * Native integer (qudit) variables wherever a quantity is a level/count --
    no binary expansion, which is the main qubit-count reduction on Dirac-3.
  * Maximum polynomial degree 3 (capacity activation and diesel fuel curve),
    exploiting Dirac-3's native support for degree <= 5 with NO auxiliary
    variables. A gate-model or QUBO device would need quadratization.
  * Conditioned coefficients: every Hamiltonian is normalized so that the
    penalty-to-objective dynamic range stays ~1e2-1e3. Analog hardware has
    finite coefficient precision; the Phase-2 code used a 1e8 range, which
    is below the noise floor of any physical device.

A polynomial is represented as ``Poly``: {tuple(sorted 1-based var ids): coef}
plus a constant offset, with per-variable integer upper bounds.
"""
from collections import defaultdict
import numpy as np

from . import grid

# ---------------------------------------------------------------- asset data
ASSETS = {
    #        unit kW  firm-credit  capex  op-adder  max units
    # op = expected lifetime operating cost adder (fuel/cycling) minus energy
    # credit, in the same 10 k$ unit -- prevents an all-diesel design.
    "PV":   dict(kw=50.0, firm=0.25, cost=4.0, op=-1.0, umax=8),
    "BESS": dict(kw=50.0, firm=0.50, cost=5.5, op=0.5, umax=6),   # 4h storage
    "DG":   dict(kw=100.0, firm=1.00, cost=6.0, op=2.5, umax=6),
}
CAP_REF = 100.0            # kW -- capacity violations measured in DG units
ASSET_KEYS = ["PV", "BESS", "DG"]
SWITCH_COST = 0.8

def asset_umax(k):
    """Unit-count cap per asset, scaled for the larger 69-bus system (whose
    largest critical load, bus 61 at 1244 kW, needs ~1.6 MW firm with margin)."""
    base = ASSETS[k]["umax"]
    mult = {"PV": 2.0, "BESS": 2.0, "DG": 2.5}[k] if grid.ACTIVE == "ieee69" else 1.0
    return int(round(base * mult))

def slack_ub():
    return 12 if grid.ACTIVE == "ieee69" else 20          # tie-switch deployment (same cost unit: 10 k$)
SLACK_STEP = 50.0          # kW per slack unit
SLACK_MAX = 20
BASE_FRACTION = 0.40       # non-critical base load an island must carry
DESIGN_MARGIN = 1.25       # sizing margin covering the load-factor tail (<=1.3)
DISPATCH_STEP = 10.0       # kW per dispatch unit


class Poly:
    def __init__(self):
        self.terms = defaultdict(float)
        self.const = 0.0
        self.upper = {}          # var id -> integer upper bound
        self.names = {}          # var id -> label

    def add(self, coef, *vars_):
        if not vars_:
            self.const += coef
            return
        self.terms[tuple(sorted(vars_))] += coef

    def new_var(self, name, ub):
        vid = len(self.upper) + 1
        self.upper[vid] = int(ub)
        self.names[vid] = name
        return vid

    @property
    def n(self):
        return len(self.upper)

    @property
    def degree(self):
        return max((len(k) for k in self.terms), default=1)

    def evaluate(self, x):
        e = self.const
        for k, c in self.terms.items():
            t = c
            for v in k:
                t *= x[v - 1]
            e += t
        return e

    def dynamic_range_db(self):
        cs = np.abs(np.array([c for c in self.terms.values() if abs(c) > 1e-12]))
        return float(10 * np.log10(cs.max() / cs.min())) if len(cs) else 0.0

    def add_square(self, coef, lin_terms, offset):
        """Add coef * (sum_i a_i x_i + offset)^2, expanded (degree 2)."""
        items = list(lin_terms.items())
        self.add(coef * offset * offset)
        for v, a in items:
            self.add(coef * 2 * a * offset, v)
            self.add(coef * a * a, v, v)
        for i in range(len(items)):
            vi, ai = items[i]
            for j in range(i + 1, len(items)):
                vj, aj = items[j]
                self.add(coef * 2 * ai * aj, vi, vj)

    def add_square_gated(self, coef, gate_var, lin_terms, offset):
        """Add coef * b * (sum a_i x_i + offset)^2 -- degree 3, native on Dirac-3."""
        items = list(lin_terms.items())
        self.add(coef * offset * offset, gate_var)
        for v, a in items:
            self.add(coef * 2 * a * offset, gate_var, v)
            self.add(coef * a * a, gate_var, v, v)
        for i in range(len(items)):
            vi, ai = items[i]
            for j in range(i + 1, len(items)):
                vj, aj = items[j]
                self.add(coef * 2 * ai * aj, gate_var, vi, vj)


# ------------------------------------------------------------------ Stage 1
def design_demand(cand):
    """Base load an island must sustain: full critical + 40% of the rest."""
    crit = sum(grid.LOAD_P[b - 1] for b in cand["critical"])
    rest = cand["load_kw"] - crit
    return DESIGN_MARGIN * (crit + BASE_FRACTION * rest)


def build_design(pool, lam_cov=25.0, lam_link=12.0, lam_cap=40.0):
    """H_design: select islands (b_c), size DER portfolios (n_ck, integer),
    deploy tie switches (y_l). Capacity feasibility is a *gated* quadratic
    (degree-3) -- only selected islands pay for capacity shortfall."""
    H = Poly()
    m = len(pool)
    b = {c: H.new_var(f"b[{c}]", 1) for c in range(m)}
    nvar = {(c, k): H.new_var(f"n[{c},{k}]", asset_umax(k))
            for c in range(m) for k in ASSET_KEYS}
    s = {c: H.new_var(f"s[{c}]", SLACK_MAX) for c in range(m)}
    y = {l: H.new_var(f"y[{l}]", 1) for l in range(len(grid.TIE_SWITCHES))}

    # capital + expected operating cost (objective)
    for c in range(m):
        for k in ASSET_KEYS:
            H.add(ASSETS[k]["cost"] + ASSETS[k]["op"], nvar[(c, k)])
    for l in y:
        H.add(SWITCH_COST, y[l])

    # coverage: every load bus in >= 1 selected island (near-partition penalty)
    member = {bus: [c for c in range(m) if bus in pool[c]["buses"]]
              for bus in range(2, grid.N_BUSES + 1)}
    for bus, cs in member.items():
        w = lam_cov * (2.0 if bus in grid.CRITICAL_BUSES else 1.0)
        H.add_square(w, {b[c]: 1.0 for c in cs}, -1.0)

    # linking: assets/slack only where an island is selected
    for c in range(m):
        for k in ASSET_KEYS:
            H.add(lam_link, nvar[(c, k)])
            H.add(-lam_link, nvar[(c, k)], b[c])
        H.add(0.1 * lam_link, s[c])
        H.add(-0.1 * lam_link, s[c], b[c])

    # gated capacity: b_c * ((firm_capacity - demand - slack) / CAP_REF)^2
    # Normalizing by the largest asset unit (100 kW) instead of the demand
    # makes one missing unit cost lam_cap >> its purchase price, so the
    # optimizer buys capacity instead of eating the penalty. It also keeps
    # coefficients O(1)-O(10), shrinking the coefficient dynamic range.
    for c in range(m):
        D = design_demand(pool[c])
        lin = {nvar[(c, k)]: ASSETS[k]["kw"] * ASSETS[k]["firm"] / CAP_REF
               for k in ASSET_KEYS}
        lin[s[c]] = -SLACK_STEP / CAP_REF
        H.add_square_gated(lam_cap, b[c], lin, -D / CAP_REF)

    # ring redundancy: reward closing a tie switch fully inside a selected island
    for l, (u, v) in enumerate(grid.TIE_SWITCHES):
        for c in range(m):
            bs = set(pool[c]["buses"])
            if u in bs and v in bs:
                H.add(-1.5 * SWITCH_COST, b[c], y[l])

    meta = dict(b=b, n=nvar, s=s, y=y, pool=pool)
    return H, meta


def decode_design(x, meta):
    pool = meta["pool"]
    sel = [c for c in meta["b"] if x[meta["b"][c] - 1] > 0.5]
    portfolio = {c: {k: int(x[meta["n"][(c, k)] - 1]) for k in ASSET_KEYS}
                 for c in sel}
    switches = [l for l in meta["y"] if x[meta["y"][l] - 1] > 0.5]
    capex = sum((ASSETS[k]["cost"] + ASSETS[k]["op"]) * portfolio[c][k] for c in sel for k in ASSET_KEYS) \
        + SWITCH_COST * len(switches)
    covered = set()
    for c in sel:
        covered |= set(pool[c]["buses"])
    return dict(selected=sel, portfolio=portfolio, switches=switches,
                capex=capex, covered=covered)


# ------------------------------------------------------------------ Stage 2
def island_capacity(portfolio_c, scen, hour):
    """Available island supply (kW) at a given hour under a scenario."""
    pv = portfolio_c["PV"] * ASSETS["PV"]["kw"] * scen.r_factor * grid.SOLAR_PROFILE[hour]
    bess = portfolio_c["BESS"] * ASSETS["BESS"]["kw"] * ASSETS["BESS"]["firm"] * 2  # power-limited
    bess = min(bess, portfolio_c["BESS"] * 200.0 / max(scen.duration, 1))           # energy-limited
    dg = portfolio_c["DG"] * ASSETS["DG"]["kw"]
    return pv + bess + dg


def reachable_dead(cand, scen, switches_closed):
    """Dead buses of candidate `cand` still connected to at least one of its
    DER hubs after the line failure, allowing closed tie switches inside the
    island. Dual hub siting means an internal failure that splits the island
    can leave BOTH fragments energizable (each by the hub it retains)."""
    import networkx as nx
    g = nx.Graph()
    bs = set(cand["buses"])
    for (u, v, *_r) in grid.LINES:
        if u in bs and v in bs and (u, v) != scen.failed_line and (v, u) != scen.failed_line:
            g.add_edge(u, v)
    for l in switches_closed:
        u, v = grid.TIE_SWITCHES[l]
        if u in bs and v in bs:
            g.add_edge(u, v)
    g.add_nodes_from(bs)
    reach = set()
    for hub in cand.get("hubs", [cand["anchor"]]):
        if hub in g:
            reach |= set(nx.node_connected_component(g, hub))
    return reach & scen.dead_buses


def build_island(design, pool, scen, lam_ov=6.0, lam_feas=30.0, w_crit=20.0):
    """H_island(s): QUBO over built islands. Coupled through load-weighted
    overlap on de-energized buses -- two islands cannot both claim a bus."""
    H = Poly()
    sel = design["selected"]
    z = {c: H.new_var(f"z[{c}]", 1) for c in sel}
    info = {}
    worst_h = max(scen.hours, key=lambda h: grid.LOAD_PROFILE[h] * scen.load_factor
                  - grid.SOLAR_PROFILE[h] * scen.r_factor)
    for c in sel:
        cand = pool[c]
        reach = reachable_dead(cand, scen, design["switches"])
        val = sum(grid.LOAD_P[bb - 1] * (w_crit if bb in grid.CRITICAL_BUSES else 1.0)
                  for bb in reach) * grid.LOAD_PROFILE[worst_h] * scen.load_factor / 1000.0
        # island must carry base load of ALL its buses once separated
        need = design_demand(cand) * grid.LOAD_PROFILE[worst_h] * scen.load_factor
        cap = island_capacity(design["portfolio"][c], scen, worst_h)
        phi = min(1.0, cap / max(need, 1.0))    # serve-capability fraction
        val *= 5.0 * phi                        # restorable value, capacity-aware
        H.add(-val, z[c])                       # reward restoring load
        H.add(0.1, z[c])                        # switching/operating cost
        H.add(lam_feas * 0.1 * (1 - phi) ** 2, z[c])   # stress on tight islands
        info[c] = dict(reach=reach, value=val, phi=phi, cap=cap, need=need)
    # overlap coupling on dead buses
    for i, ci in enumerate(sel):
        for cj in sel[i + 1:]:
            ov = info[ci]["reach"] & info[cj]["reach"]
            if ov:
                w = lam_ov * sum(grid.LOAD_P[b - 1] for b in ov) / 1000.0
                H.add(w, z[ci], z[cj])
    return H, dict(z=z, info=info, worst_hour=worst_h)


# ------------------------------------------------------------------ Stage 3
def build_dispatch(design, pool, scen, c, worst_h,
                   gamma=150.0, w_serve=8.0, fuel_cubic=0.35):
    """H_dispatch(s, c): integer setpoints (10 kW steps) for PV/BESS/DG plus a
    served-load level for non-critical demand. Diesel fuel curve keeps its
    CUBIC term -- solved natively on Dirac-3, no quadratization."""
    H = Poly()
    port = design["portfolio"][c]
    cand = pool[c]
    lf = grid.LOAD_PROFILE[worst_h] * scen.load_factor
    L_cr = sum(grid.LOAD_P[b - 1] for b in cand["critical"]) * lf
    L_nc = (cand["load_kw"] - L_cr / lf) * lf if lf > 0 else 0.0

    pv_cap = port["PV"] * ASSETS["PV"]["kw"] * scen.r_factor * grid.SOLAR_PROFILE[worst_h]
    bess_cap = min(port["BESS"] * ASSETS["BESS"]["kw"],
                   port["BESS"] * 200.0 / max(scen.duration, 1))
    dg_cap = port["DG"] * ASSETS["DG"]["kw"]

    D = DISPATCH_STEP
    p_pv = H.new_var("p_pv", max(int(pv_cap // D), 0))
    p_be = H.new_var("p_bess", max(int(bess_cap // D), 0))
    p_dg = H.new_var("p_dg", max(int(dg_cap // D), 0))
    l_nc = H.new_var("l_nc", max(int(L_nc // D), 0))

    scale = max(cand["load_kw"] * lf, 1.0)
    # value of served non-critical load
    H.add(-w_serve * D / scale, l_nc)
    # marginal costs: PV ~ 0, BESS cycling, DG fuel with cubic heat-rate term.
    # The cubic is calibrated so that at FULL diesel loading its marginal cost
    # equals `fuel_cubic` x the serve reward: convex heat-rate shaping that
    # never makes serving load unprofitable. Degree-3, native on Dirac-3.
    H.add(0.5 * D / scale, p_be)
    H.add(2.0 * D / scale, p_dg)
    ub_dg = max(H.upper[p_dg], 1)
    kappa = fuel_cubic * (w_serve * D / scale) / (3.0 * ub_dg ** 2)
    H.add(kappa, p_dg, p_dg, p_dg)
    # power balance: (pv + bess + dg - l_nc - L_cr)^2 normalized
    lin = {p_pv: D / scale, p_be: D / scale, p_dg: D / scale, l_nc: -D / scale}
    H.add_square(gamma, lin, -L_cr / scale)
    return H, dict(vars=(p_pv, p_be, p_dg, l_nc), L_cr=L_cr, L_nc=L_nc, scale=scale)


# ------------------------------------------------------- multi-period dispatch
MP_BUCKETS = 4          # time buckets per outage window
MP_ESTEP = 50.0         # kWh per SOC level
BESS_KWH_PER_UNIT = 200.0
BESS_ETA = 0.9          # charge efficiency


def _mp_step(pv_kw, dg_kw, bess_kw, bess_kwh, L_nc_max, n_buckets):
    """Choose the coarsest-necessary power step so total qudit levels fit the
    Dirac-3 budget (954) with margin. Documented approximation: coarser
    setpoints on larger islands."""
    for P in (20.0, 25.0, 40.0, 50.0):
        per_bucket = (int(pv_kw // P) + int(dg_kw // P) + 2 * int(bess_kw // P)
                      + int(L_nc_max // P) + 5)
        soc = int(bess_kwh // MP_ESTEP) + 1
        if n_buckets * per_bucket + n_buckets * soc <= 900:
            return P
    return 80.0


def build_dispatch_mp(design, pool, scen, c, gamma=150.0, w_serve=8.0,
                      fuel_cubic=0.35, lam_soc=60.0):
    """H_dispatch_mp(s, c): TIME-COUPLED dispatch over the outage window.

    The window is split into MP_BUCKETS contiguous buckets; each bucket t has
    integer setpoints  p_pv[t], p_dg[t], p_dis[t] (battery discharge),
    p_chg[t] (battery charge), l_nc[t] (served non-critical), plus a state-of-
    charge qudit soc[t]. Inter-temporal coupling is the SOC recursion penalty

        lam_soc * ( soc[t]*E - soc[t-1]*E - (eta*p_chg[t] - p_dis[t])*P*h_t )^2

    which is native degree-2; the diesel heat-rate cubic keeps degree 3.
    This realizes the challenge's multi-hour horizon (storage arbitrage:
    charge from midday PV, discharge in the evening) INSIDE the Hamiltonian,
    on hardware, rather than only in evaluation.
    """
    lf = scen.load_factor
    port = design["portfolio"][c]
    cand = pool[c]
    pv_kw = port["PV"] * ASSETS["PV"]["kw"]
    dg_kw = port["DG"] * ASSETS["DG"]["kw"]
    bess_kw = port["BESS"] * ASSETS["BESS"]["kw"]
    bess_kwh = port["BESS"] * BESS_KWH_PER_UNIT

    hours = list(scen.hours)
    nb = min(MP_BUCKETS, max(1, len(hours) // 2))
    buckets = [hours[i * len(hours) // nb:(i + 1) * len(hours) // nb]
               for i in range(nb)]

    crit = [b for b in cand["buses"] if b in grid.CRITICAL_BUSES]
    ncrit = [b for b in cand["buses"] if b not in grid.CRITICAL_BUSES]
    L_cr = [np.mean([sum(grid.LOAD_P[b - 1] for b in crit) * grid.LOAD_PROFILE[h] * lf
                     for h in hb]) for hb in buckets]
    L_nc = [np.mean([sum(grid.LOAD_P[b - 1] for b in ncrit) * grid.LOAD_PROFILE[h] * lf
                     for h in hb]) for hb in buckets]
    pv_av = [pv_kw * np.mean([grid.SOLAR_PROFILE[h] for h in hb]) * scen.r_factor
             for hb in buckets]

    P = _mp_step(pv_kw, dg_kw, bess_kw, bess_kwh, max(L_nc, default=0), nb)
    E = MP_ESTEP
    scale = max(cand["load_kw"] * lf, 1.0)
    e_scale = max(bess_kwh, 4 * E)

    H = Poly()
    V = {}
    for t in range(nb):
        V[("pv", t)] = H.new_var(f"pv[{t}]", max(int(pv_av[t] // P), 0))
        V[("dg", t)] = H.new_var(f"dg[{t}]", max(int(dg_kw // P), 0))
        V[("dis", t)] = H.new_var(f"dis[{t}]", max(int(bess_kw // P), 0))
        V[("chg", t)] = H.new_var(f"chg[{t}]", max(int(min(bess_kw, pv_kw) // P), 0))
        V[("nc", t)] = H.new_var(f"nc[{t}]", max(int(L_nc[t] // P), 0))
        V[("soc", t)] = H.new_var(f"soc[{t}]", max(int(bess_kwh // E), 0))

    soc0 = bess_kwh  # start fully charged
    for t in range(nb):
        h_t = len(buckets[t])
        # serve reward + operating costs (bucket-hours weighted)
        H.add(-w_serve * h_t * P / scale, V[("nc", t)])
        H.add(0.5 * h_t * P / scale, V[("dis", t)])
        H.add(0.5 * h_t * P / scale, V[("chg", t)])
        H.add(2.0 * h_t * P / scale, V[("dg", t)])
        ub_dg = max(H.upper[V[("dg", t)]], 1)
        kappa = fuel_cubic * (w_serve * h_t * P / scale) / (3.0 * ub_dg ** 2)
        H.add(kappa, V[("dg", t)], V[("dg", t)], V[("dg", t)])
        # power balance: pv + dis + dg - chg - nc = L_cr[t]
        lin = {V[("pv", t)]: P / scale, V[("dis", t)]: P / scale,
               V[("dg", t)]: P / scale, V[("chg", t)]: -P / scale,
               V[("nc", t)]: -P / scale}
        H.add_square(gamma, lin, -L_cr[t] / scale)
        # SOC recursion: soc[t]*E = soc[t-1]*E + (eta*chg - dis)*P*h_t
        lin2 = {V[("soc", t)]: E / e_scale,
                V[("chg", t)]: -BESS_ETA * P * h_t / e_scale,
                V[("dis", t)]: P * h_t / e_scale}
        const = -(soc0 / e_scale) if t == 0 else 0.0
        if t > 0:
            lin2[V[("soc", t - 1)]] = -E / e_scale
        H.add_square(lam_soc, lin2, const)

    meta = dict(vars=V, buckets=buckets, P=P, E=E, L_cr=L_cr, L_nc=L_nc,
                pv_av=pv_av, soc0=soc0, nb=nb,
                caps=dict(pv_kw=pv_kw, dg_kw=dg_kw, bess_kw=bess_kw,
                          bess_kwh=bess_kwh))
    return H, meta
