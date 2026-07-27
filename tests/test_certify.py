"""The MILP linearization certifier -- E2's scalable replacement for 2^m."""
import itertools

import numpy as np
import pytest

from eqosystem import grid, candidates, scenarios, hamiltonians as ham
from eqosystem.certify import milp_certify
from eqosystem.hamiltonians import Poly
from eqosystem.pipeline import run_design
from eqosystem.solvers import AnnealerSolver, ExactSolver


@pytest.fixture(scope="module")
def rig():
    grid.select("ieee69")
    pool = candidates.generate()
    scens = scenarios.generate(20, seed=42)
    design, _H, _ = run_design(pool, AnnealerSolver())
    return dict(pool=pool, scens=scens, design=design)


def brute(H):
    vs = sorted(H.upper)
    best = np.inf
    for x in itertools.product((0, 1), repeat=len(vs)):
        best = min(best, H.evaluate(np.asarray(x, dtype=float)))
    return best


def random_cubic(n, seed, density=0.3):
    rng = np.random.default_rng(seed)
    H = Poly()
    v = [H.new_var(f"x{i}", 1) for i in range(n)]
    for i in range(n):
        H.add(float(rng.normal()), v[i])
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                H.add(float(rng.normal()), v[i], v[j])
    for _ in range(n):
        i, j, k = rng.choice(n, size=3, replace=False)
        H.add(float(rng.normal()), v[i], v[j], v[k])
    return H


def test_matches_enumeration_on_every_islanding_qubo(rig):
    """Same certificate as E2's enumeration, from an independent method."""
    checked = 0
    for sc in rig["scens"]:
        H, _m = ham.build_island(rig["design"], rig["pool"], sc)
        if not H.n:
            continue
        r = milp_certify(H)
        assert r["certified"], f"scenario {sc.sid}: HiGHS did not prove optimality"
        assert r["energy"] == pytest.approx(
            ExactSolver().solve(H)["energy"], abs=1e-9), f"scenario {sc.sid}"
        checked += 1
    assert checked == 20


def test_matches_brute_force_on_random_cubics():
    """Degree-3 with negative and positive couplings, 20 instances."""
    for seed in range(20):
        H = random_cubic(10, seed)
        r = milp_certify(H)
        assert r["certified"]
        assert r["energy"] == pytest.approx(brute(H), abs=1e-8), f"seed {seed}"


def test_repeated_indices_collapse():
    """Binary x^2 = x: (v, v) terms must fold into the linear part."""
    H = Poly()
    a = H.new_var("a", 1)
    b = H.new_var("b", 1)
    H.add(-3.0, a, a)          # == -3a for binary
    H.add(1.0, a, b)
    r = milp_certify(H)
    assert r["certified"]
    assert r["energy"] == pytest.approx(-3.0)   # a=1, b=0
    assert list(r["x"]) == [1, 0]


def test_scales_where_enumeration_cannot():
    """m = 40: enumeration is 2^40 ~ 10^12 states; the certifier proves the
    optimum in seconds. This is the whole point of the instrument."""
    H = random_cubic(40, seed=7, density=0.15)
    r = milp_certify(H)
    assert r["certified"], "HiGHS should prove optimality at m = 40"
    assert r["wall"] < 60.0
    # a strong heuristic must never beat a proven optimum
    sa = AnnealerSolver().solve(H)
    assert sa["energy"] >= r["energy"] - 1e-6
    # and the certified point must actually attain the certified energy
    assert H.evaluate(r["x"]) == pytest.approx(r["energy"])


def test_rejects_nonbinary():
    H = Poly()
    H.new_var("u", 5)
    with pytest.raises(AssertionError, match="binary"):
        milp_certify(H)
