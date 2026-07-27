"""W2 -- voltage-aware islanding Hamiltonian.

At seed 42 no candidate island is anywhere near the 0.95 pu band, so the penalty
is identically zero and the A/B is a measured null result. These tests therefore
do two separate jobs:

  * prove the null result is real and that the two arms are byte-identical, and
  * prove the mechanism is LIVE rather than dead code, by tightening the band
    until candidates are genuinely at risk and showing the term then appears,
    respects critical-load dominance, and can flip an islanding decision.
"""
import numpy as np
import pytest

from eqosystem import grid, candidates, scenarios, hamiltonians as ham
from eqosystem import lindistflow as ldf
from eqosystem.pipeline import run_design
from eqosystem.solvers import AnnealerSolver, ExactSolver


@pytest.fixture(scope="module")
def rig():
    grid.select("ieee69")
    pool = candidates.generate()
    scens = scenarios.generate(20, seed=42)
    design, _H, _ = run_design(pool, AnnealerSolver())
    return dict(pool=pool, scens=scens, design=design)


@pytest.fixture
def restore_vmin():
    saved = ldf.V_MIN
    yield
    ldf.V_MIN = saved


# ------------------------------------------------------- structure of the term
def test_voltage_term_adds_no_variables_and_no_degree(rig):
    for sc in rig["scens"]:
        Hb, _ = ham.build_island(rig["design"], rig["pool"], sc, voltage_aware=False)
        Ha, _ = ham.build_island(rig["design"], rig["pool"], sc, voltage_aware=True)
        assert Ha.n == Hb.n, "voltage awareness must not add variables"
        assert Ha.degree <= 2, "islanding QUBO must stay degree 2"
        assert Ha.degree == Hb.degree


def test_predicted_voltage_is_recorded(rig):
    """The penalty is precomputed classically, so the prediction must be stored."""
    seen = 0
    for sc in rig["scens"]:
        _H, mi = ham.build_island(rig["design"], rig["pool"], sc, voltage_aware=True)
        for c, info in mi["info"].items():
            if info["reach"]:
                assert info["v_min_pred"] is not None
                assert 0.0 < info["v_min_pred"] <= 1.05
                seen += 1
    assert seen > 0


# ------------------------------------------------------- the measured null result
def test_no_candidate_is_at_voltage_risk_at_seed_42(rig):
    """Documents WHY the A/B is null: islands are fed from hubs inside them."""
    worst = 1.0
    at_risk = 0
    for sc in rig["scens"]:
        _H, mi = ham.build_island(rig["design"], rig["pool"], sc, voltage_aware=True)
        for info in mi["info"].values():
            if info["v_min_pred"] is not None:
                worst = min(worst, info["v_min_pred"])
            at_risk += info["v_violation"] > 0
    assert at_risk == 0, "expected no predicted violations at seed 42"
    assert worst > ldf.V_MIN, f"worst predicted {worst:.4f} should clear the band"
    assert worst == pytest.approx(0.9907, abs=0.01)


def test_arms_are_identical_when_nothing_is_at_risk(rig):
    """With no violations the two arms must be the same Hamiltonian, exactly."""
    for sc in rig["scens"]:
        Hb, _ = ham.build_island(rig["design"], rig["pool"], sc, voltage_aware=False)
        Ha, _ = ham.build_island(rig["design"], rig["pool"], sc, voltage_aware=True)
        assert Ha.terms == Hb.terms, f"scenario {sc.sid}: arms diverged"
        assert Ha.dynamic_range_db() == pytest.approx(Hb.dynamic_range_db())
        if Ha.n:
            assert ExactSolver().solve(Ha)["energy"] == pytest.approx(
                ExactSolver().solve(Hb)["energy"], abs=1e-12)


# --------------------------------------------------- the mechanism is live
def test_tightened_band_activates_the_penalty(rig, restore_vmin):
    """Raise the band until candidates are at risk; the term must then appear."""
    ldf.V_MIN = 0.995
    activated = 0
    for sc in rig["scens"]:
        Hb, _ = ham.build_island(rig["design"], rig["pool"], sc, voltage_aware=False)
        Ha, mi = ham.build_island(rig["design"], rig["pool"], sc, voltage_aware=True)
        for info in mi["info"].values():
            if info["v_penalty"] > 0:
                activated += 1
        if activated and Ha.terms != Hb.terms:
            break
    assert activated > 0, "tightening to 0.995 pu should put candidates at risk"


def test_penalty_never_outweighs_critical_restoration(rig, restore_vmin):
    """Critical-load dominance is structural: penalty <= non-critical reward."""
    ldf.V_MIN = 1.02                       # force maximal violations everywhere
    checked = 0
    for sc in rig["scens"]:
        _H, mi = ham.build_island(rig["design"], rig["pool"], sc, voltage_aware=True)
        for info in mi["info"].values():
            if info["v_penalty"] <= 0:
                continue
            checked += 1
            assert info["v_penalty"] <= info["value_noncrit"] + 1e-12
            if info["value_crit"] > 0:
                assert info["v_penalty"] < info["value"], (
                    "penalty must never cancel the total reward of an island "
                    "that restores critical load")
    assert checked > 0, "expected the penalty to be active at a 1.02 pu band"


def test_penalty_cannot_flip_an_island_that_restores_critical_load(rig, restore_vmin):
    """Second-order measured finding, and it is the intended behaviour.

    Even with the band forced to 1.02 pu so every candidate violates maximally,
    no islanding decision changes. Every energized island in this design restores
    critical load (6.0 to 120.4 in reward units) against non-critical rewards of
    only 0.3 to 3.5, and the penalty is capped at the non-critical reward. So
    voltage risk provably cannot strand critical infrastructure -- which is the
    dominance requirement, and it is why the term is inert here rather than a
    sign that it is miswired.
    """
    ldf.V_MIN = 1.02
    active_pen, crit_only = 0, 0
    for sc in rig["scens"]:
        Hb, _mb = ham.build_island(rig["design"], rig["pool"], sc, voltage_aware=False)
        Ha, ma = ham.build_island(rig["design"], rig["pool"], sc, voltage_aware=True)
        if not Ha.n:
            continue
        for info in ma["info"].values():
            if info["reach"]:
                active_pen += info["v_penalty"] > 0
                crit_only += info["value_crit"] > 0
        assert np.array_equal(ExactSolver().solve(Hb)["x"],
                              ExactSolver().solve(Ha)["x"])
    assert active_pen > 0, "the penalty should be active at this band"
    assert crit_only == active_pen, (
        "finding assumes every energized island restores critical load")


def test_penalty_flips_a_marginal_island_with_no_critical_load(rig, restore_vmin,
                                                               monkeypatch):
    """Mechanism proof: strip criticality, and voltage risk does change the answer.

    Exercises the real build_island path rather than the arithmetic in isolation.
    """
    ldf.V_MIN = 1.02
    monkeypatch.setattr(grid, "CRITICAL_BUSES", set())
    flipped = False
    for sc in rig["scens"]:
        Hb, _ = ham.build_island(rig["design"], rig["pool"], sc, voltage_aware=False)
        Ha, ma = ham.build_island(rig["design"], rig["pool"], sc, voltage_aware=True)
        if not Ha.n or not any(i["v_penalty"] > 0 for i in ma["info"].values()):
            continue
        if not np.array_equal(ExactSolver().solve(Hb)["x"],
                              ExactSolver().solve(Ha)["x"]):
            flipped = True
            break
    assert flipped, ("with no critical load to protect, a maximally voltage-risky "
                     "island must be switched off by the penalty")


def test_penalty_is_monotone_in_violation(rig, restore_vmin):
    """Deeper predicted violation must never reduce the penalty."""
    pens = []
    for band in (0.96, 0.99, 1.02):
        ldf.V_MIN = band
        tot = 0.0
        for sc in rig["scens"]:
            _H, mi = ham.build_island(rig["design"], rig["pool"], sc, voltage_aware=True)
            tot += sum(i["v_penalty"] for i in mi["info"].values())
        pens.append(tot)
    assert pens[0] <= pens[1] <= pens[2], f"penalty not monotone: {pens}"
    assert pens[-1] > 0


# --------------------------------------------------- post-hoc checker retained
def test_post_hoc_checker_still_independent(rig):
    """The claim is 'voltage-aware in-Hamiltonian AND independently verified'."""
    sc = max(rig["scens"], key=lambda s: len(s.dead_buses))
    _H, mi = ham.build_island(rig["design"], rig["pool"], sc, voltage_aware=True)
    c = next(iter(mi["info"]))
    r = ldf.check_island(rig["pool"][c], sc, mi["worst_hour"], 1.0, True, 500.0,
                         closed_ties=rig["design"]["switches"])
    assert set(r) >= {"feasible", "v_min", "n_v_viol", "worst_line_pct"}


def test_checker_can_fail_on_the_intact_feeder():
    """Regression: a checker that cannot fail proves nothing (handoff 9.1)."""
    grid.select("ieee33")
    ldf._tables_cache.clear()
    n = grid.N_BUSES
    from eqosystem.scenarios import Scenario
    scen = Scenario(sid=-1, r_factor=1.0, load_factor=1.0, failed_line=(-1, -1),
                    start_hour=17, duration=8, dead_buses=set(range(2, n + 1)))
    cand = dict(buses=list(range(2, n + 1)), critical=[], anchor=2, hubs=[2],
                load_kw=float(sum(grid.LOAD_P)), customers=0)
    r = ldf.check_island(cand, scen, 17, 1.0, True, 0.0)
    assert not r["feasible"], "intact 33-bus feeder at peak with no DER must fail"
    assert r["n_v_viol"] > 0
    grid.select("ieee69")
    ldf._tables_cache.clear()
