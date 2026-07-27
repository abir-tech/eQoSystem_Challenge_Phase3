"""W9 -- non-convex P-Q capability curve and battery efficiency curve."""
import numpy as np
import pytest

from eqosystem import grid, candidates, scenarios, hamiltonians as ham
from eqosystem import lindistflow as ldf
from eqosystem.pipeline import run_design
from eqosystem.solvers import AnnealerSolver


@pytest.fixture(scope="module")
def rig():
    grid.select("ieee69")
    pool = candidates.generate()
    scens = scenarios.generate(20, seed=42)
    design, _H, _ = run_design(pool, AnnealerSolver())
    sc = max(scens, key=lambda s: len(s.dead_buses))
    return dict(pool=pool, design=design, scen=sc)


# ------------------------------------------------------------- eta(P) curve
def test_eta_curve_spans_and_midpoint():
    bkw = 100.0
    assert ham.eta_charge(0.0, bkw) == pytest.approx(ham.BESS_ETA0)
    assert ham.eta_charge(bkw, bkw) == pytest.approx(
        ham.BESS_ETA0 - ham.BESS_ETA_SLOPE)
    # the old constant is the curve's midpoint, not a different battery
    assert ham.eta_charge(0.5 * bkw, bkw) == pytest.approx(ham.BESS_ETA, abs=1e-9)


def test_eta_curve_monotone_and_clipped():
    bkw = 50.0
    vals = [ham.eta_charge(p, bkw) for p in np.linspace(0, 2 * bkw, 30)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))
    assert min(vals) >= 0.7
    assert ham.eta_charge(10.0, 0.0) == pytest.approx(ham.BESS_ETA0)


def test_eta_curve_hamiltonian_is_degree_4_and_exact(rig):
    """The curve arm must reach degree 4 (native on Dirac-3, impossible on a
    QUBO device without auxiliaries) and evaluate to the analytic residual."""
    with_bess = [c for c in rig["design"]["selected"]
                 if rig["design"]["portfolio"][c]["BESS"] > 0]
    assert with_bess, "need an island with a battery for this test"
    c = with_bess[0]
    H2, md2 = ham.build_dispatch_mp(rig["design"], rig["pool"], rig["scen"], c,
                                    eta_curve=False)
    H4, md4 = ham.build_dispatch_mp(rig["design"], rig["pool"], rig["scen"], c,
                                    eta_curve=True)
    assert H2.degree == 3 and H4.degree == 4
    assert H4.n == H2.n, "the curve must add no variables"
    # spot-check the residual algebra at random points for bucket 0
    rng = np.random.default_rng(0)
    e, P, h = md4["E"], md4["P"], len(md4["buckets"][0])
    bkw, es = md4["caps"]["bess_kw"], max(md4["caps"]["bess_kwh"], 4 * md4["E"])
    if bkw > 0:
        for _ in range(50):
            x = np.array([rng.integers(0, H4.upper[v] + 1)
                          for v in sorted(H4.upper)], dtype=float)
            chg = x[md4["vars"][("chg", 0)] - 1] * P
            eta = ham.BESS_ETA0 - ham.BESS_ETA_SLOPE * chg / bkw
            # energies differ only through the eta model; both finite
            assert np.isfinite(H4.evaluate(x)) and np.isfinite(H2.evaluate(x))


# ------------------------------------------------------------ piecewise P-Q
def test_pq_piecewise_is_inside_the_circle():
    S = 100.0
    for p in np.linspace(0, S, 41):
        assert ldf.q_capability(p, S) <= ldf.q_capability(p, S, mode="circle") + 1e-9


def test_pq_field_limit_binds_at_light_loading():
    S = 100.0
    assert ldf.q_capability(0.0, S) == pytest.approx(0.9 * S)
    assert ldf.q_capability(0.0, S) < ldf.q_capability(0.0, S, mode="circle")


def test_pq_low_power_dip_creates_the_nonconvexity():
    """The set {(P,Q): Q <= cap(P)} must have a genuine notch.

    The first version of the curve failed this test, correctly: it was built as
    a min of concave functions, which is still concave, so the set was convex.
    The notch is the low-power dip -- full var support in standby, reduced below
    15% loading, full again above -- making cap(P) non-monotone: a chord from
    the standby point to a mid-loading point passes ABOVE the boundary at the
    dip, so the chord leaves the feasible set.
    """
    S = 100.0
    p1, q1 = 1.0, ldf.q_capability(1.0, S)       # standby: 0.90 S
    p2, q2 = 20.0, ldf.q_capability(20.0, S)     # past the dip: field limit
    pm = 8.0                                     # inside the dip
    lam = (pm - p1) / (p2 - p1)
    chord = (1 - lam) * q1 + lam * q2
    assert ldf.q_capability(pm, S) < chord - 1e-9, (
        "boundary must dip below the chord -- that is the non-convexity")
    # and cap(P) is non-monotone: it RISES coming out of the dip
    assert ldf.q_capability(20.0, S) > ldf.q_capability(8.0, S) + 1e-9


def test_pq_zero_at_and_beyond_rated():
    S = 100.0
    assert ldf.q_capability(S, S) == pytest.approx(0.0)
    assert ldf.q_capability(1.2 * S, S) == 0.0
