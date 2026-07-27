"""E12 -- classical MILP baseline for the dispatch stage.

Stages 1 and 2 already had classical baselines (HiGHS MILP for design, exhaustive
enumeration for islanding). Stage 3 had none, leaving the rubric's "comparison
against a non-quantum method on the same problem instance" unmet for a third of
the pipeline.
"""
import numpy as np
import pytest

from eqosystem import grid, candidates, scenarios, hamiltonians as ham
from eqosystem import pipeline as pl
from eqosystem.solvers import AnnealerSolver


@pytest.fixture(scope="module")
def rig():
    grid.select("ieee69")
    pool = candidates.generate()
    scens = scenarios.generate(20, seed=42)
    design = pl.run_design(pool, AnnealerSolver())[0]
    # pick a scenario with an energized island that actually reaches dead buses
    for sc in scens:
        _Hi, mi = ham.build_island(design, pool, sc)
        for c, info in mi["info"].items():
            if info["reach"]:
                return dict(pool=pool, design=design, scen=sc, island=c,
                            reach=info["reach"])
    pytest.skip("no energized island found")


def test_milp_dispatch_is_feasible_and_fast(rig):
    r = pl.milp_dispatch_baseline(rig["design"], rig["pool"], rig["scen"],
                                  rig["island"])
    assert r["feasible"], r["status"]
    assert r["plan"] is not None
    assert r["wall"] < 30.0


def test_milp_plan_has_the_same_shape_as_the_hamiltonian_plan(rig):
    H, md = ham.build_dispatch_mp(rig["design"], rig["pool"], rig["scen"],
                                  rig["island"])
    r = pl.milp_dispatch_baseline(rig["design"], rig["pool"], rig["scen"],
                                  rig["island"])
    for k in ("pv", "dg", "dis", "chg", "nc"):
        assert len(r["plan"][k]) == md["nb"]
        assert all(isinstance(v, int) and v >= 0 for v in r["plan"][k])


def test_milp_respects_the_setpoint_bounds(rig):
    H, md = ham.build_dispatch_mp(rig["design"], rig["pool"], rig["scen"],
                                  rig["island"])
    r = pl.milp_dispatch_baseline(rig["design"], rig["pool"], rig["scen"],
                                  rig["island"])
    caps, P = md["caps"], md["P"]
    for t in range(md["nb"]):
        assert r["plan"]["dg"][t] * P <= caps["dg_kw"] + 1e-6
        assert r["plan"]["dis"][t] * P <= caps["bess_kw"] + 1e-6


def test_milp_soc_recursion_is_actually_feasible(rig):
    """Hard constraint: the MILP plan must never discharge more than is stored."""
    H, md = ham.build_dispatch_mp(rig["design"], rig["pool"], rig["scen"],
                                  rig["island"])
    r = pl.milp_dispatch_baseline(rig["design"], rig["pool"], rig["scen"],
                                  rig["island"])
    soc, P = md["soc0"], md["P"]
    for t in range(md["nb"]):
        h = len(md["buckets"][t])
        soc = soc + ham.BESS_ETA * r["plan"]["chg"][t] * P * h \
            - r["plan"]["dis"][t] * P * h
        assert soc >= -1e-6, f"bucket {t}: MILP plan drains SOC below zero ({soc})"
        assert soc <= md["caps"]["bess_kwh"] + 1e-6


def test_simulate_plan_scores_both_plans_identically(rig):
    """The comparison must use one yardstick, not two."""
    H, md = ham.build_dispatch_mp(rig["design"], rig["pool"], rig["scen"],
                                  rig["island"])
    rq = AnnealerSolver().solve(H)
    qplan = {k: [int(rq["x"][md["vars"][(k, t)] - 1]) for t in range(md["nb"])]
             for k in ("pv", "dg", "dis", "chg", "nc")}
    mb = pl.milp_dispatch_baseline(rig["design"], rig["pool"], rig["scen"],
                                   rig["island"])
    a = pl.simulate_plan(rig["pool"], rig["scen"], rig["island"], qplan, md,
                         rig["reach"])
    b = pl.simulate_plan(rig["pool"], rig["scen"], rig["island"], mb["plan"], md,
                         rig["reach"])
    for d in (a, b):
        assert set(d) == {"crit_short_hours", "unserved_kwh", "dg_kwh",
                          "nc_served_frac"}
        assert d["unserved_kwh"] >= -1e-9
        assert 0.0 <= d["nc_served_frac"] <= 1.0


def test_simulate_plan_is_deterministic(rig):
    H, md = ham.build_dispatch_mp(rig["design"], rig["pool"], rig["scen"],
                                  rig["island"])
    mb = pl.milp_dispatch_baseline(rig["design"], rig["pool"], rig["scen"],
                                   rig["island"])
    a = pl.simulate_plan(rig["pool"], rig["scen"], rig["island"], mb["plan"], md,
                         rig["reach"])
    b = pl.simulate_plan(rig["pool"], rig["scen"], rig["island"], mb["plan"], md,
                         rig["reach"])
    assert a == b


def test_zero_plan_sheds_load(rig):
    """Sanity: the scorer must be able to report failure, not always succeed."""
    H, md = ham.build_dispatch_mp(rig["design"], rig["pool"], rig["scen"],
                                  rig["island"])
    zero = {k: [0] * md["nb"] for k in ("pv", "dg", "dis", "chg", "nc")}
    s = pl.simulate_plan(rig["pool"], rig["scen"], rig["island"], zero, md,
                         rig["reach"])
    assert s["unserved_kwh"] > 0.0
    assert s["dg_kwh"] == 0.0
