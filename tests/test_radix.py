"""W3a -- radix-16 decomposition: round-trip exactness, degree, level budget."""
import numpy as np
import pytest

from eqosystem import grid, candidates, scenarios, hamiltonians as ham
from eqosystem import radix, conditioning as cond
from eqosystem.hamiltonians import Poly
from eqosystem.pipeline import run_design
from eqosystem.solvers import AnnealerSolver


@pytest.fixture(scope="module")
def rig():
    grid.select("ieee69")
    pool = candidates.generate()
    scens = scenarios.generate(20, seed=42)
    design, H_design, _ = run_design(pool, AnnealerSolver())
    # W1b's trust-region encoding brought the DEFAULT design stage under the
    # 16-level cap on its own (max upper bound 6), so radix has nothing to do
    # there any more. The absolute encoding still needs it, and remains the
    # honest test subject for the design-stage decomposition.
    _d_abs, H_abs, _ = run_design(pool, AnnealerSolver(), delta=False)
    return dict(pool=pool, scens=scens, design=design, H_design=H_design,
                H_design_absolute=H_abs)


def test_digits_needed():
    assert radix._digits_needed(15) == 1
    assert radix._digits_needed(16) == 2
    assert radix._digits_needed(20) == 2
    assert radix._digits_needed(50) == 2
    assert radix._digits_needed(255) == 2
    assert radix._digits_needed(256) == 3


def test_small_vars_pass_through_untouched():
    H = Poly()
    a = H.new_var("a", 1)
    b = H.new_var("b", 15)
    H.add(2.0, a, b)
    Hr, m = radix.radix_decompose(H)
    assert Hr.n == 2
    assert m[a] == [(1, 1)] and m[b] == [(2, 1)]
    assert Hr.terms == H.terms


def test_roundtrip_exact_at_random_points():
    """H(recompose(d)) == H_radix(d) wherever the digit value is in range."""
    H = Poly()
    u = H.new_var("u", 50)
    v = H.new_var("v", 20)
    w = H.new_var("w", 3)
    H.add(1.5, u)
    H.add(-2.0, u, v)
    H.add(0.25, u, v, w)
    H.add(3.0, w, w)
    H.const = 7.0
    Hr, m = radix.radix_decompose(H)
    rng = np.random.default_rng(42)
    checked = 0
    for _ in range(400):
        d = np.array([rng.integers(0, Hr.upper[k] + 1) for k in sorted(Hr.upper)],
                     dtype=float)
        x, clamped = radix.radix_recompose(d, m, H)
        if clamped:
            continue                     # digit box is a relaxation; skip out-of-box
        assert H.evaluate(x.astype(float)) == pytest.approx(Hr.evaluate(d), rel=1e-9)
        checked += 1
    assert checked >= 200, f"only {checked} in-box samples"


def test_degree_preserved():
    H = Poly()
    a = H.new_var("a", 40)
    b = H.new_var("b", 40)
    H.add(1.0, a, a, a)                  # cubic in a decomposed variable
    H.add(2.0, a, b)
    Hr, _m = radix.radix_decompose(H)
    assert H.degree == 3
    assert Hr.degree == 3, "cubic must stay cubic over digits"


def test_all_digit_bounds_within_the_integer_cap():
    H = Poly()
    for i, U in enumerate((16, 20, 50, 100, 300)):
        v = H.new_var(f"x{i}", U)
        H.add(1.0, v)
    Hr, _m = radix.radix_decompose(H)
    assert all(ub <= radix.DIGIT_MAX for ub in Hr.upper.values())


def test_level_saving_versus_naive_encoding():
    """U = 50 costs 51 levels natively; as digits it costs 20, not 32.

    The top digit only has to reach U // 16 = 3, so the pair is 16 + 4 levels
    rather than the 16 + 16 a fixed-width base-16 encoding would spend.
    """
    H = Poly()
    v = H.new_var("big", 50)
    H.add(1.0, v)
    Hr, m = radix.radix_decompose(H)
    assert radix.levels(H) == 51
    assert [Hr.upper[dv] for dv, _w in m[v]] == [15, 3]
    assert radix.levels(Hr) == 20
    assert radix.levels(Hr) < radix.levels(H)


def test_clamping_is_counted():
    H = Poly()
    v = H.new_var("u", 20)               # digits reach 31 > 20
    H.add(1.0, v)
    Hr, m = radix.radix_decompose(H)
    d = np.zeros(Hr.n)
    d[m[v][1][0] - 1] = 1                # digit1 = 1 -> value 16
    d[m[v][0][0] - 1] = 15               # digit0 = 15 -> value 31
    x, clamped = radix.radix_recompose(d, m, H)
    assert clamped == 1
    assert x[v - 1] == 20


# ------------------------------------------------- real pipeline Hamiltonians
def test_default_design_stage_no_longer_needs_radix(rig):
    """W1b outcome: the shipped design encoding is already within the cap."""
    H = rig["H_design"]
    assert max(H.upper.values()) <= radix.DIGIT_MAX, (
        "trust-region design should be integer-legal without decomposition")
    assert not any("16-level" in r for r in
                   cond.hardware_legality(H, cert=dict(certified=True)))


def test_design_stage_becomes_integer_legal(rig):
    H = rig["H_design_absolute"]
    assert any(ub > radix.DIGIT_MAX for ub in H.upper.values()), "precondition"
    Hr, _m = radix.radix_decompose(H)
    assert Hr.degree <= 3
    assert all(ub <= radix.DIGIT_MAX for ub in Hr.upper.values())
    reasons = cond.hardware_legality(Hr, cert=dict(certified=True))
    assert not any("16-level" in r for r in reasons), reasons


def test_dispatch_stage_becomes_integer_legal(rig):
    sc = max(rig["scens"], key=lambda s: len(s.dead_buses))
    H, _md = ham.build_dispatch_mp(rig["design"], rig["pool"], sc,
                                   rig["design"]["selected"][0])
    Hr, _m = radix.radix_decompose(H)
    assert Hr.degree <= 3
    assert all(ub <= radix.DIGIT_MAX for ub in Hr.upper.values())
    assert radix.levels(Hr) <= cond.MAX_TOTAL_LEVELS


def test_encode_is_the_inverse_of_recompose():
    H = Poly()
    u = H.new_var("u", 50)
    v = H.new_var("v", 20)
    H.add(1.0, u, v)
    Hr, m = radix.radix_decompose(H)
    for a in range(0, 51, 7):
        for b in range(0, 21, 3):
            x = np.array([a, b])
            d = radix.radix_encode(x, m, H)
            back, clamped = radix.radix_recompose(d, m, H)
            assert clamped == 0
            assert list(back) == [a, b]


def test_design_roundtrip_exact_on_real_hamiltonian(rig):
    """200 random points sampled IN THE ORIGINAL BOX, then encoded to digits.

    Sampling digits directly is the wrong direction here: with 18 decomposed
    variables the probability that a uniform digit vector lands in the original
    box is about 0.0009, so every sample gets clamped and nothing is compared.
    """
    H = rig["H_design"]
    Hr, m = radix.radix_decompose(H)
    rng = np.random.default_rng(7)
    vs = sorted(H.upper)
    for _ in range(200):
        x = np.zeros(H.n, dtype=int)
        for v in vs:
            x[v - 1] = rng.integers(0, H.upper[v] + 1)
        d = radix.radix_encode(x, m, H)
        back, clamped = radix.radix_recompose(d, m, H)
        assert clamped == 0
        assert np.array_equal(back, x)
        assert H.evaluate(x.astype(float)) == pytest.approx(
            Hr.evaluate(d.astype(float)), rel=1e-7)


def test_dispatch_roundtrip_exact_on_real_hamiltonian(rig):
    sc = max(rig["scens"], key=lambda s: len(s.dead_buses))
    H, _md = ham.build_dispatch_mp(rig["design"], rig["pool"], sc,
                                   rig["design"]["selected"][0])
    Hr, m = radix.radix_decompose(H)
    rng = np.random.default_rng(11)
    vs = sorted(H.upper)
    for _ in range(200):
        x = np.zeros(H.n, dtype=int)
        for v in vs:
            x[v - 1] = rng.integers(0, H.upper[v] + 1)
        d = radix.radix_encode(x, m, H)
        assert H.evaluate(x.astype(float)) == pytest.approx(
            Hr.evaluate(d.astype(float)), rel=1e-7)


def test_radix_reports_its_level_cost(rig):
    """Decomposition trades variables for levels, and the trade is not always
    favourable -- which the work order's "also a level saving" implies it is.

    Radix saves levels only when the bound is well above the base. Just over the
    cap it COSTS levels: U = 16 needs two digits spanning 0..31 (18 levels) to
    represent 0..16 (17 levels). Measured on the absolute design encoding, where
    asset_umax gives PV a bound of exactly 16, decomposition moves the stage from
    487 to 496 levels. Pinned so the trade is reported rather than assumed.
    """
    # the favourable regime, as claimed in the work order
    Hbig = Poly()
    v = Hbig.new_var("u", 50)
    Hbig.add(1.0, v)
    Hr_big, _ = radix.radix_decompose(Hbig)
    assert radix.levels(Hr_big) < radix.levels(Hbig), "U=50 should save levels"

    # the unfavourable regime, just above the cap
    Hsmall = Poly()
    w = Hsmall.new_var("u", 16)
    Hsmall.add(1.0, w)
    Hr_small, _ = radix.radix_decompose(Hsmall)
    assert radix.levels(Hr_small) > radix.levels(Hsmall), "U=16 should cost levels"

    # on the real stage: always more variables, and the cap is always met
    H = rig["H_design_absolute"]
    Hr, _m = radix.radix_decompose(H)
    assert Hr.n > H.n, "decomposition adds digit variables"
    assert all(ub <= radix.DIGIT_MAX for ub in Hr.upper.values())


# ===================== bounded-sum decomposition (exact alternative) =========
def test_sum_decompose_representable_set_is_exactly_the_box():
    """The property radix lacks: no value outside [0, u] is reachable."""
    H = Poly()
    v = H.new_var("u", 40)
    H.add(1.0, v)
    Hs, m = radix.sum_decompose(H)
    assert max(Hs.upper.values()) <= radix.DIGIT_MAX
    assert sum(Hs.upper[p] for p in m[v]) == 40, "part bounds must sum to u exactly"
    # radix, by contrast, over-represents
    Hr, mr = radix.radix_decompose(H)
    reach = sum(Hr.upper[d] * w for d, w in mr[v])
    assert reach > 40, "radix should over-represent (this is the defect)"


def test_sum_recompose_never_clamps():
    H = Poly()
    v = H.new_var("u", 43)
    H.add(-1.0, v)
    Hs, m = radix.sum_decompose(H)
    rng = np.random.default_rng(0)
    for _ in range(200):
        xp = np.array([rng.integers(0, Hs.upper[p] + 1) for p in sorted(Hs.upper)])
        x, n_clamped = radix.sum_recompose(xp, m, H)
        assert n_clamped == 0, "bounded-sum must never need clamping"
        assert 0 <= x[v - 1] <= 43


def test_sum_decompose_is_energy_exact():
    H = Poly()
    a = H.new_var("a", 20)
    b = H.new_var("b", 18)
    H.add(1.5, a, b)
    H.add(-2.0, a)
    H.add(0.25, b, b)
    Hs, m = radix.sum_decompose(H)
    rng = np.random.default_rng(1)
    for _ in range(200):
        xp = np.array([rng.integers(0, Hs.upper[p] + 1) for p in sorted(Hs.upper)],
                      dtype=float)
        x, _n = radix.sum_recompose(xp, m, H)
        assert Hs.evaluate(xp) == pytest.approx(H.evaluate(x.astype(float)))


def test_sum_decompose_preserves_degree_and_dynamic_range():
    """Weight-1 parts leave coefficients untouched; radix's 16^k weights do not."""
    H = Poly()
    a = H.new_var("a", 40)
    H.add(1.0, a)
    H.add(0.5, a, a)
    Hs, _m = radix.sum_decompose(H)
    Hr, _mr = radix.radix_decompose(H)
    assert Hs.degree <= H.degree
    assert Hs.dynamic_range_db() == pytest.approx(H.dynamic_range_db(), abs=1e-9)
    assert Hr.dynamic_range_db() > Hs.dynamic_range_db(), (
        "radix digit weights should inflate dynamic range")


def test_sum_decompose_makes_dispatch_integer_legal(rig):
    H, _md = ham.build_dispatch_mp(rig["design"], rig["pool"], rig["scens"][6],
                                   rig["design"]["selected"][0], fuel_cubic=0.0)
    Hs, _m = radix.sum_decompose(H)
    assert max(Hs.upper.values()) <= radix.DIGIT_MAX
    assert cond.total_levels(Hs) <= cond.MAX_TOTAL_LEVELS
    assert cond.hardware_legality(Hs, cert=dict(certified=True)) == []
