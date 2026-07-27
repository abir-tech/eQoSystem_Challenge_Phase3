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


def greedy_portfolio(pool):
    """Classical feasible starting portfolio: greedy set cover, then buy the
    cheapest firm capacity per selected island. Returned as plain dicts so it
    can seed the trust-region encoding before any Poly exists."""
    uncovered = set(range(2, grid.N_BUSES + 1))
    chosen = []
    while uncovered:
        best = max(range(len(pool)),
                   key=lambda c: len(set(pool[c]["buses"]) & uncovered)
                   / (1 + design_demand(pool[c]) / 500.0))
        gain = set(pool[best]["buses"]) & uncovered
        if not gain:
            break
        chosen.append(best)
        uncovered -= gain
    # Size EVERY candidate to its own demand, not only the greedy-chosen ones.
    # The b_c variable still decides what gets built; this only sets the centre
    # of each trust region. Leaving unchosen candidates at zero left their gate
    # offset at the full -D/CAP_REF and they then dominated the dynamic range:
    # measured, unseeded c7 (D = 1089 kW) contributed 40*10.89^2 = 4746 against
    # 40-228 for the seeded islands, holding the design stage at 36.4 dB.
    units = {(c, k): 0 for c in range(len(pool)) for k in ASSET_KEYS}
    for c in range(len(pool)):
        need = design_demand(pool[c])
        for k in ("DG", "BESS", "PV"):          # cheapest firm kW first
            a = ASSETS[k]
            while need > 0 and units[(c, k)] < asset_umax(k):
                units[(c, k)] += 1
                need -= a["kw"] * a["firm"]
    return set(chosen), units


def cheapest_capacity_price():
    """Cost of one CAP_REF unit of firm capacity from the cheapest real asset."""
    return min((ASSETS[k]["cost"] + ASSETS[k]["op"])
               / (ASSETS[k]["kw"] * ASSETS[k]["firm"] / CAP_REF)
               for k in ASSET_KEYS)


def slack_unit_price(factor=1.0):
    """Price a slack unit at the real capacity it substitutes for.

    W14: at the original 0.1*lam_link = 1.20 a slack unit covered
    SLACK_STEP/CAP_REF = 0.5 gate-units, i.e. 2.40 per unit against 8.50 for the
    cheapest real capacity -- a 3.5x arbitrage in FICTITIOUS capacity, which the
    repair loop then has to buy for real. Measured: repricing does not move the
    cost ratio, so this was not the binding defect, but leaving a known
    arbitrage in the model would be wrong."""
    return factor * (SLACK_STEP / CAP_REF) * cheapest_capacity_price()


def build_design(pool, lam_cov=25.0, lam_link=12.0, lam_cap=40.0,
                 seed_units=None, radius=3, slack_max=None, slack_price=None):
    """H_design: select islands (b_c), size DER portfolios (n_ck, integer),
    deploy tie switches (y_l). Capacity feasibility is a *gated* quadratic
    (degree-3) -- only selected islands pay for capacity shortfall.

    W1b -- trust-region (delta) encoding. Passing `seed_units` re-expresses each
    unit count as a bounded correction around a classical seed,

        n_ck = base_ck + d_ck,   base_ck = max(0, seed_ck - radius),
                                 d_ck in [0, min(umax, seed_ck + radius) - base]

    an exact affine substitution, not an approximation of the encoding. It
    attacks the design stage's dynamic range at its actual source. Measured:
    c_max is the gated offset lam_cap*(D/CAP_REF)^2 = 40*(1764/100)^2 = 11947
    and c_min the PV-squared term at 0.625, so the range is
    (D_max / PV_firm_unit)^2 ~ 19900 -- intrinsic to encoding ABSOLUTE capacity
    when one PV unit is 141x smaller than the largest island's demand.

    Rescaling cannot fix that: per-island normalisation by D_c was tried and
    measured WORSE (42.8 -> 59.9 dB), because island demands span 20-1764 kW so
    dividing by D moves the spread between term families instead of removing it.
    Centring on a feasible seed gives 631 -> 202 levels, maximum bound 20 -> 6,
    and 42.8 -> 34.5 dB with the mixed-integer cost ratio unchanged at 1.019x."""
    H = Poly()
    m = len(pool)
    smax = SLACK_MAX if slack_max is None else slack_max
    slack_price = slack_unit_price() if slack_price is None else slack_price
    delta = seed_units is not None

    b = {c: H.new_var(f"b[{c}]", 1) for c in range(m)}
    base, nvar = {}, {}
    for c in range(m):
        for k in ASSET_KEYS:
            if delta:
                sd = int(seed_units.get((c, k), 0))
                lo = max(0, sd - radius)
                hi = min(asset_umax(k), sd + radius)
                base[(c, k)] = lo
                nvar[(c, k)] = H.new_var(f"d[{c},{k}]", max(hi - lo, 0))
            else:
                base[(c, k)] = 0
                nvar[(c, k)] = H.new_var(f"n[{c},{k}]", asset_umax(k))
    s = {c: H.new_var(f"s[{c}]", smax) for c in range(m)}
    y = {l: H.new_var(f"y[{l}]", 1) for l in range(len(grid.TIE_SWITCHES))}

    # capital + expected operating cost (objective); the base contributes a
    # constant, which does not affect the argmin but keeps energies comparable
    for c in range(m):
        for k in ASSET_KEYS:
            ck = ASSETS[k]["cost"] + ASSETS[k]["op"]
            H.add(ck, nvar[(c, k)])
            if base[(c, k)]:
                H.add(ck * base[(c, k)])
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
            if base[(c, k)]:
                H.add(lam_link * base[(c, k)])
                H.add(-lam_link * base[(c, k)], b[c])
        # W14: slack is FICTITIOUS capacity -- it cancels shortfall in the
        # gate but buys no hardware, so the repair loop purchases the real
        # units afterwards. Price it at the capacity it substitutes for.
        H.add(slack_price, s[c])
        H.add(-slack_price, s[c], b[c])

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
        # the seed's firm capacity absorbs most of the demand, so what
        # remains in the offset is a small residual, not the full D/CAP_REF
        base_firm = sum(base[(c, k)] * ASSETS[k]["kw"] * ASSETS[k]["firm"]
                        for k in ASSET_KEYS) / CAP_REF
        H.add_square_gated(lam_cap, b[c], lin, base_firm - D / CAP_REF)

    # ring redundancy: reward closing a tie switch fully inside a selected island
    for l, (u, v) in enumerate(grid.TIE_SWITCHES):
        for c in range(m):
            bs = set(pool[c]["buses"])
            if u in bs and v in bs:
                H.add(-1.5 * SWITCH_COST, b[c], y[l])

    meta = dict(b=b, n=nvar, s=s, y=y, pool=pool, base=base, delta=delta,
                radius=radius if delta else None)
    return H, meta


def decode_design(x, meta):
    pool = meta["pool"]
    base = meta.get("base", {})
    sel = [c for c in meta["b"] if x[meta["b"][c] - 1] > 0.5]
    portfolio = {c: {k: int(x[meta["n"][(c, k)] - 1])
                     + int(base.get((c, k), 0)) for k in ASSET_KEYS}
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
    pv = portfolio_c["PV"] * ASSETS["PV"]["kw"] * scen.r_factor_at(hour) * grid.SOLAR_PROFILE[hour]
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


LAMBDA_V = 8.0     # voltage-risk weight, in units of the non-critical reward
V_TOL = 0.05       # pu band width below V_MIN used to normalize the violation

# W7 -- cross-island export. Tie switches carry the same 1000 kVA default the
# LinDistFlow checker assumes for ties; exports are additionally capped by the
# exporting island's own surplus, so an island can never export capacity it
# does not have.
TIE_CAPACITY_KW = 1000.0


def tie_adjacency(pool):
    """{(ci, cj): [tie indices]} for candidate pairs a tie switch can bridge."""
    adj = {}
    for l, (u, v) in enumerate(grid.TIE_SWITCHES):
        for i, ci in enumerate(pool):
            si = set(ci["buses"])
            for cj in pool[i + 1:]:
                sj = set(cj["buses"])
                if (u in si and v in sj) or (v in si and u in sj):
                    adj.setdefault((ci["id"], cj["id"]), []).append(l)
    return adj


def export_opportunity(design, pool, scen, ci, cj, reach_i, reach_j, worst_h,
                       ties, switches):
    """Dead load that becomes reachable ONLY because a tie bridges ci and cj.

    This must be an honest post-fault reachability computation, not a set
    difference. An earlier version took `orphan = dead & dst_buses - reach_i -
    reach_j`, which credits the pair with every dead bus whenever neither island
    reaches anything -- so a tie between two specific buses appeared to
    resurrect an entire fragment. Measured, that spurious reward turned 19
    scenarios from 0 active islands to 3 and produced a 13862 kWh "improvement"
    of which exactly 0 came from export.

    The correct set is the difference between reachability on the UNION graph
    (both islands' intact lines, plus the tie edges, from both islands' hubs)
    and reachability each island already has alone. Returns (kW, buses) bounded
    by tie capacity and the donor's surplus."""
    import networkx as nx
    bs = set(pool[ci]["buses"]) | set(pool[cj]["buses"])
    g = nx.Graph()
    g.add_nodes_from(bs)
    for (u, v, *_r) in grid.LINES:
        if u in bs and v in bs and (u, v) != scen.failed_line \
                and (v, u) != scen.failed_line:
            g.add_edge(u, v)
    for l in switches:                       # ties the design actually closed
        u, v = grid.TIE_SWITCHES[l]
        if u in bs and v in bs:
            g.add_edge(u, v)
    bridged = False
    for l in ties:                           # the tie(s) bridging this pair
        u, v = grid.TIE_SWITCHES[l]
        if u in bs and v in bs and not g.has_edge(u, v):
            g.add_edge(u, v)
            bridged = True
    if not bridged:
        return 0.0, set()

    hubs = set(pool[ci].get("hubs", [pool[ci]["anchor"]])) | \
        set(pool[cj].get("hubs", [pool[cj]["anchor"]]))
    joint = set()
    for hb in hubs:
        if hb in g:
            joint |= set(nx.node_connected_component(g, hb))
    gained = (joint & scen.dead_buses) - reach_i - reach_j
    if not gained:
        return 0.0, set()

    lf_h = grid.LOAD_PROFILE[worst_h] * scen.load_factor_at(worst_h)
    need = sum(grid.LOAD_P[b - 1] for b in gained) * lf_h
    best = (0.0, set())
    for src, reach_src in ((ci, reach_i), (cj, reach_j)):
        cap = island_capacity(design["portfolio"][src], scen, worst_h)
        own = sum(grid.LOAD_P[b - 1] for b in reach_src) * lf_h
        served = min(need, max(0.0, cap - own), TIE_CAPACITY_KW)
        if served > best[0]:
            if served >= need - 1e-9:
                best = (served, set(gained))
            else:                            # partial: take the cheapest buses
                take, acc = set(), 0.0
                for b in sorted(gained, key=lambda b: grid.LOAD_P[b - 1]):
                    inc = grid.LOAD_P[b - 1] * lf_h
                    if acc + inc > served:
                        break
                    take.add(b)
                    acc += inc
                best = (acc, take)
    return best


def build_island(design, pool, scen, lam_ov=6.0, lam_feas=30.0, w_crit=20.0,
                 voltage_aware=True, lam_v=LAMBDA_V, export_aware=False):
    """H_island(s): QUBO over built islands. Coupled through load-weighted
    overlap on de-energized buses -- two islands cannot both claim a bus.

    Voltage awareness (W2). LinDistFlow is linear, so for a fixed candidate
    under a fixed scenario the predicted worst-bus voltage at its expected
    dispatch is a CONSTANT computable classically before the solve. The penalty
    is therefore linear in z_c -- degree 1, no new variables.

    Calibration is structural rather than tuned: the penalty is expressed in
    units of the island's own NON-CRITICAL restoration reward and capped there,
    so a voltage-risky island can lose at most what it earns from restoring
    non-critical load, and restoring critical load can never be outweighed.

    Cross-island export (W7) is DEFAULT OFF. Its reward adds degree-2 terms,
    which changes the very islanding QUBOs behind the recorded 20/20 Dirac-3
    result; leaving it on would make that hardware evidence unreproducible from
    the submitted code. Enable it via --export-ab to see the measured arm."""
    from . import lindistflow as ldf

    H = Poly()
    sel = design["selected"]
    z = {c: H.new_var(f"z[{c}]", 1) for c in sel}
    info = {}
    worst_h = max(scen.hours, key=lambda h: grid.LOAD_PROFILE[h] * scen.load_factor_at(h)
                  - grid.SOLAR_PROFILE[h] * scen.r_factor_at(h))
    lf_h = grid.LOAD_PROFILE[worst_h] * scen.load_factor_at(worst_h)
    for c in sel:
        cand = pool[c]
        reach = reachable_dead(cand, scen, design["switches"])
        # split the restorable value so critical dominance can be guaranteed
        val_cr = sum(grid.LOAD_P[bb - 1] * w_crit
                     for bb in reach if bb in grid.CRITICAL_BUSES) * lf_h / 1000.0
        val_nc = sum(grid.LOAD_P[bb - 1]
                     for bb in reach if bb not in grid.CRITICAL_BUSES) * lf_h / 1000.0
        # island must carry base load of ALL its buses once separated
        need = design_demand(cand) * lf_h
        cap = island_capacity(design["portfolio"][c], scen, worst_h)
        phi = min(1.0, cap / max(need, 1.0))    # serve-capability fraction
        val_cr *= 5.0 * phi
        val_nc *= 5.0 * phi
        val = val_cr + val_nc                   # restorable value, capacity-aware
        H.add(-val, z[c])                       # reward restoring load
        H.add(0.1, z[c])                        # switching/operating cost
        H.add(lam_feas * 0.1 * (1 - phi) ** 2, z[c])   # stress on tight islands

        # --- voltage risk, precomputed classically (degree 1 in z_c) ---
        v_pred, viol, v_pen = None, 0.0, 0.0
        if voltage_aware and reach:
            chk = ldf.check_island(cand, scen, worst_h, min(1.0, phi), True,
                                   min(cap, need), closed_ties=design["switches"])
            v_pred = chk["v_min"]
            viol = max(0.0, ldf.V_MIN - v_pred)
            if viol > 0.0:
                v_pen = min(lam_v * (viol / V_TOL) ** 2 * val_nc, val_nc)
                H.add(v_pen, z[c])
        info[c] = dict(reach=reach, value=val, phi=phi, cap=cap, need=need,
                       value_crit=val_cr, value_noncrit=val_nc,
                       v_min_pred=v_pred, v_violation=viol, v_penalty=v_pen)
    # overlap coupling on dead buses
    for i, ci in enumerate(sel):
        for cj in sel[i + 1:]:
            ov = info[ci]["reach"] & info[cj]["reach"]
            if ov:
                w = lam_ov * sum(grid.LOAD_P[b - 1] for b in ov) / 1000.0
                H.add(w, z[ci], z[cj])

    # W7 -- cross-island export over closed ties. A fault can strand buses whose
    # own island retains no DER hub on their side of the break while a
    # tie-adjacent island still reaches the rest of the fragment. Energizing
    # BOTH islands is what makes the transfer possible, so the reward is a
    # degree-2 term on z_ci * z_cj: no new variables, still binary, still
    # degree 2, and the pair only earns it when both are actually on.
    exports = {}
    if export_aware:
        adj = tie_adjacency(pool)
        for i, ci in enumerate(sel):
            for cj in sel[i + 1:]:
                key = (min(ci, cj), max(ci, cj))
                if key not in adj:
                    continue
                kw, buses = export_opportunity(design, pool, scen, ci, cj,
                                               info[ci]["reach"],
                                               info[cj]["reach"], worst_h,
                                               adj[key], design["switches"])
                if kw <= 0.0:
                    continue
                val = 5.0 * sum(grid.LOAD_P[b - 1]
                                * (w_crit if b in grid.CRITICAL_BUSES else 1.0)
                                for b in buses) * lf_h / 1000.0
                H.add(-val, z[ci], z[cj])
                exports[key] = dict(kw=kw, buses=buses, value=val,
                                    ties=adj[key])
    return H, dict(z=z, info=info, worst_hour=worst_h, exports=exports)


# ------------------------------------------------------------------ Stage 3
def build_dispatch(design, pool, scen, c, worst_h,
                   gamma=150.0, w_serve=8.0, fuel_cubic=0.35):
    """H_dispatch(s, c): integer setpoints (10 kW steps) for PV/BESS/DG plus a
    served-load level for non-critical demand. Diesel fuel curve keeps its
    CUBIC term -- solved natively on Dirac-3, no quadratization."""
    H = Poly()
    port = design["portfolio"][c]
    cand = pool[c]
    lf = grid.LOAD_PROFILE[worst_h] * scen.load_factor_at(worst_h)
    L_cr = sum(grid.LOAD_P[b - 1] for b in cand["critical"]) * lf
    L_nc = (cand["load_kw"] - L_cr / lf) * lf if lf > 0 else 0.0

    pv_cap = port["PV"] * ASSETS["PV"]["kw"] * scen.r_factor_at(worst_h) * grid.SOLAR_PROFILE[worst_h]
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
MP_CRIT_PEAK = True     # plan critical load at the intra-bucket peak, not the mean
MP_ESTEP = 50.0         # kWh per SOC level
BESS_KWH_PER_UNIT = 200.0
BESS_ETA = 0.9          # constant charge efficiency (hardware planning profile)

# W9 -- power-dependent charge efficiency, the challenge's "battery
# charge/discharge efficiency curves" non-convex element. eta falls linearly
# with loading: 0.95 near idle to 0.85 at rated power, mean ~0.90, so the old
# constant is the curve's midpoint rather than a different battery.
BESS_ETA0 = 0.95
BESS_ETA_SLOPE = 0.10


def eta_charge(p_kw, bess_kw):
    """Charge efficiency at charging power p_kw for a bess_kw-rated battery."""
    if bess_kw <= 0:
        return BESS_ETA0
    return max(0.7, BESS_ETA0 - BESS_ETA_SLOPE * min(p_kw / bess_kw, 1.0))


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


def build_dispatch_mp(design, pool, scen, c, gamma=None, gamma_k=0.20,
                      lam_cd=1.0, serve_bias_correct=True, eta_curve=False,
                      w_serve=8.0, fuel_cubic=0.35, lam_soc=300.0):
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

    CALIBRATION, all measured against the E12 mixed-integer dispatch baseline.

    lam_soc: the SOC recursion is a conservation law, not a preference. At the
    original 60 the solver profitably planned a 1520 kWh discharge from 1260 kWh
    of stored energy -- cheaper than the diesel it displaced -- and the hourly
    simulation then clipped it and shed critical load in the last hour of a 13 h
    outage. Sweep at 50 scenarios: M2 was 1 at 60 and 150, and 0 from 300 up;
    300 is the smallest weight at which the constraint binds.

    gamma: a FIXED balance weight does not transfer between feeders, because the
    penalty scales as gamma*(P/scale)^2 while the serve reward scales as
    w_serve*h*P/scale, so the trade-point moves with scale/P. gamma is therefore
    DERIVED per bucket as gamma_k*w_serve*h_t*scale/P, leaving gamma_k
    dimensionless. Calibrated over 4 seeds on both feeders, gamma_k = 0.20 was
    best on each.

    serve_bias_correct: the serve reward is linear in nc while the balance
    penalty is quadratic in the residual, so the stationary point sits at
    r = -w_serve*h*scale/(2*gamma), not r = 0. The plan therefore overshoots
    deliverable service by exactly 1/(2*gamma_k) units, independent of the
    instance, and the simulation delivers only min(plan, generation - L_cr).
    That is why extra annealing did not help -- it converged accurately onto a
    biased optimum. The bias is closed-form, so it is cancelled exactly by
    shifting the balance target. Measured: 1.213 -> 1.045 against the baseline.

    lam_cd: nothing forbade charging and discharging in the same bucket, which
    round-trips energy through the efficiency twice for no benefit. It happened
    in 21% of bucket-instances, wasting 5280 kWh. A degree-2 product term rules
    it out with no new variables. A correctness fix, not a performance one: the
    metric differences are inside run-to-run spread.
    """
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
    # Critical load is planned at the intra-bucket PEAK, not the mean.
    # Bucketing approximates an hourly problem and critical load must be
    # served in every hour rather than on average: planning to the mean
    # under-provisions by the peak-to-mean ratio, measured at up to 1.24x,
    # which is precisely how single critical bus-hours were lost late in
    # long outages. Non-critical stays at the mean, being curtailable.
    _agg_cr = max if MP_CRIT_PEAK else np.mean
    L_cr = [float(_agg_cr([sum(grid.LOAD_P[b - 1] for b in crit)
                           * grid.LOAD_PROFILE[h] * scen.load_factor_at(h)
                           for h in hb])) for hb in buckets]
    L_nc = [np.mean([sum(grid.LOAD_P[b - 1] for b in ncrit)
                     * grid.LOAD_PROFILE[h] * scen.load_factor_at(h)
                     for h in hb]) for hb in buckets]
    pv_av = [pv_kw * np.mean([grid.SOLAR_PROFILE[h] * scen.r_factor_at(h)
                              for h in hb]) for hb in buckets]

    P = _mp_step(pv_kw, dg_kw, bess_kw, bess_kwh, max(L_nc, default=0), nb)
    E = MP_ESTEP
    # window-mean multiplier: identical to scen.load_factor in flat mode,
    # and the right normalizer when it varies per bucket in tree mode
    lf_mean = float(np.mean([scen.load_factor_at(h) for h in hours]))
    scale = max(cand["load_kw"] * lf_mean, 1.0)
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
        # complementarity: charging and discharging in the same bucket is
        # physically pointless -- see the calibration note above
        H.add(lam_cd * w_serve * h_t * P / scale,
              V[("dis", t)], V[("chg", t)])
        H.add(2.0 * h_t * P / scale, V[("dg", t)])
        ub_dg = max(H.upper[V[("dg", t)]], 1)
        kappa = fuel_cubic * (w_serve * h_t * P / scale) / (3.0 * ub_dg ** 2)
        H.add(kappa, V[("dg", t)], V[("dg", t)], V[("dg", t)])
        # power balance: pv + dis + dg - chg - nc = L_cr[t]
        lin = {V[("pv", t)]: P / scale, V[("dis", t)]: P / scale,
               V[("dg", t)]: P / scale, V[("chg", t)]: -P / scale,
               V[("nc", t)]: -P / scale}
        # self-normalising balance weight and closed-form bias correction;
        # see the calibration note in the docstring
        g_t = (gamma_k * w_serve * h_t * scale / P) if gamma is None else gamma
        bias = (w_serve * h_t * scale / (2.0 * g_t)) if serve_bias_correct else 0.0
        H.add_square(g_t, lin, -(L_cr[t] + bias) / scale)
        # SOC recursion: soc[t]*E = soc[t-1]*E + (eta*chg - dis)*P*h_t
        # W9: with the efficiency curve the charged energy gains a chg^2
        # term, so squaring the residual yields degree-4 monomials --
        # native on Dirac-3, impossible on a QUBO device without
        # auxiliaries. Measured, the degree-4 arm is slightly worse
        # (1.033 vs 0.979), so constant-eta remains the planning default
        # while the curve is implemented, flagged and measured.
        eta_c = BESS_ETA0 if eta_curve else BESS_ETA
        lin2 = {V[("soc", t)]: E / e_scale,
                V[("chg", t)]: -eta_c * P * h_t / e_scale,
                V[("dis", t)]: P * h_t / e_scale}
        const = -(soc0 / e_scale) if t == 0 else 0.0
        if t > 0:
            lin2[V[("soc", t - 1)]] = -E / e_scale
        H.add_square(lam_soc, lin2, const)
        if eta_curve and bess_kw > 0:
            q = BESS_ETA_SLOPE * (P ** 2) * h_t / (bess_kw * e_scale)
            cv = V[("chg", t)]
            H.add(lam_soc * 2.0 * q * const, cv, cv)
            for vv, aa in lin2.items():
                H.add(lam_soc * 2.0 * q * aa, cv, cv, vv)
            H.add(lam_soc * q * q, cv, cv, cv, cv)

    meta = dict(vars=V, buckets=buckets, P=P, E=E, L_cr=L_cr, L_nc=L_nc,
                pv_av=pv_av, soc0=soc0, nb=nb,
                caps=dict(pv_kw=pv_kw, dg_kw=dg_kw, bess_kw=bess_kw,
                          bess_kwh=bess_kwh))
    return H, meta
