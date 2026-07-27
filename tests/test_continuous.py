"""W3c -- continuous-solver path for stages that exceed the 16-level cap."""
import numpy as np
import pytest

from eqosystem import grid, candidates, scenarios, hamiltonians as ham
from eqosystem import continuous as cont, conditioning as cond
from eqosystem.hamiltonians import Poly
from eqosystem.pipeline import run_design
from eqosystem.solvers import AnnealerSolver, get_solver


@pytest.fixture(scope="module")
def rig():
    grid.select("ieee69")
    pool = candidates.generate()
    scens = scenarios.generate(20, seed=42)
    design, _H, _ = run_design(pool, AnnealerSolver())
    sc = max(scens, key=lambda s: len(s.dead_buses))
    H, md = ham.build_dispatch_mp(design, pool, sc, design["selected"][0])
    return dict(pool=pool, design=design, scen=sc, H=H, md=md)


# ------------------------------------------------------- the simplex embedding
def test_slack_makes_the_sum_constraint_non_binding():
    """Without slack, sum(x) = S with S = sum(u) pins every variable AT its bound."""
    H = Poly()
    a = H.new_var("a", 5)
    b = H.new_var("b", 5)
    H.add(-1.0, a)
    H.add(2.0, b)
    S = cont.sum_constraint_for(H)
    assert S == 10.0
    H_ext, meta = cont.to_simplex(H)
    assert H_ext.n == H.n + 1, "an inert slack variable must be added"
    assert meta["sum_constraint"] == S
    # the slack can absorb the entire budget, so sum over ORIGINAL vars can be 0
    assert H_ext.upper[meta["slack_vid"]] >= S


def test_slack_appears_in_no_term_so_energies_are_unchanged():
    H = Poly()
    a = H.new_var("a", 4)
    b = H.new_var("b", 4)
    H.add(1.5, a, b)
    H.add(-2.0, a)
    H_ext, meta = cont.to_simplex(H)
    assert H_ext.terms == H.terms, "slack must not enter the Hamiltonian"
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = np.array([rng.integers(0, 5), rng.integers(0, 5)], dtype=float)
        xe = np.concatenate([x, [rng.integers(0, 10)]])
        assert H.evaluate(x) == pytest.approx(H_ext.evaluate(xe)), (
            "slack value must never change the energy")


# --------------------------------------------------------- round and repair
def test_round_and_repair_returns_a_feasible_integer_point(rig):
    H = rig["H"]
    rng = np.random.default_rng(0)
    y = np.array([rng.random() * H.upper[v] for v in sorted(H.upper)])
    x, info = cont.round_and_repair(y, H)
    assert x.dtype.kind in "iu"
    for v in sorted(H.upper):
        assert 0 <= x[v - 1] <= H.upper[v], "rounding must respect the box"
    assert info["energy_repaired"] <= info["energy_rounded"] + 1e-9


def test_repair_never_worsens_energy(rig):
    H = rig["H"]
    rng = np.random.default_rng(3)
    for _ in range(5):
        y = np.array([rng.random() * H.upper[v] for v in sorted(H.upper)])
        _x, info = cont.round_and_repair(y, H)
        assert info["repair_gain"] >= -1e-9


def test_out_of_range_values_are_clamped(rig):
    """Wildly out-of-range continuous output must land inside the box.

    Checked on the CLAMPED point rather than the returned one: greedy repair is
    then free to move it further, which is the whole point of repair.
    """
    H = rig["H"]
    y = np.array([1e6] * H.n, dtype=float)
    clamped = np.array([min(round(y[v - 1]), H.upper[v]) for v in sorted(H.upper)])
    assert all(clamped[v - 1] == H.upper[v] for v in sorted(H.upper))
    x, info = cont.round_and_repair(y, H)
    for v in sorted(H.upper):
        assert 0 <= x[v - 1] <= H.upper[v], "repaired point must stay feasible"
    assert info["energy_rounded"] == pytest.approx(
        H.evaluate(clamped.astype(float))), "pre-repair energy is the clamped point"


# ------------------------------------------------------- the full W3c path
def test_relax_round_repair_reports_its_rounding_gap(rig):
    x, info = cont.relax_round_repair(rig["H"])
    for k in ("energy_continuous", "energy_rounded", "energy_repaired",
              "rounding_gap", "final_gap", "sum_constraint"):
        assert k in info
    assert info["rounding_gap"] >= -1e-6, "continuous value is a lower bound"
    assert info["energy_repaired"] <= info["energy_rounded"] + 1e-9


def test_continuous_relaxation_lower_bounds_the_integer_optimum(rig):
    """The relaxation must not be above a feasible integer energy."""
    H = rig["H"]
    _x, info = cont.relax_round_repair(H)
    e_sa = AnnealerSolver().solve(H)["energy"]
    assert info["energy_continuous"] <= e_sa + 1e-6, (
        "a relaxation cannot exceed a feasible integer solution")


def test_solver_interface_matches_the_others(rig):
    s = get_solver("contrelax")
    r = s.solve(rig["H"])
    assert set(r) >= {"x", "energy", "wall", "backend", "samples"}
    assert r["energy"] == pytest.approx(rig["H"].evaluate(r["x"].astype(float)))


# ---------------------------------------------------- what this actually buys
def test_dispatch_is_illegal_on_integer_but_fits_the_continuous_path(rig):
    """The point of W3c, pinned."""
    H = rig["H"]
    reasons = cond.hardware_legality(H, cert=dict(certified=True))
    assert any("16-level" in r for r in reasons), (
        "precondition: dispatch should exceed the integer cap")
    H_ext, meta = cont.to_simplex(H)
    assert cond.total_levels(H_ext) <= cond.MAX_TOTAL_LEVELS, (
        "continuous embedding must still fit the 954-level budget")
    assert meta["sum_constraint"] > 0


def test_hardware_continuous_solver_raises_without_a_token(monkeypatch):
    monkeypatch.delenv("QCI_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="QCI_TOKEN"):
        get_solver("dirac3c")
