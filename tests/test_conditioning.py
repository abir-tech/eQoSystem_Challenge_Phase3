"""W1 -- Certified Coefficient Truncation.

The load-bearing test in this file is `test_adversarial_certificate_fails`: a
certificate that always passes proves nothing. It is paired with
`test_adversarial_truncation_really_moves_the_optimum`, which shows the instance
the certificate refuses is one where truncation genuinely would have changed the
answer.
"""
import itertools
import warnings

import numpy as np
import pytest

from eqosystem import conditioning as cond
from eqosystem.hamiltonians import Poly


# ------------------------------------------------------------------ helpers
def brute_force(H):
    """(argmin, min energy, n_minimizers) by enumeration."""
    vs = sorted(H.upper)
    ubs = [H.upper[v] for v in vs]
    best_x, best_e, n = None, np.inf, 0
    for x in itertools.product(*[range(u + 1) for u in ubs]):
        e = H.evaluate(np.asarray(x, dtype=float))
        if e < best_e - 1e-12:
            best_x, best_e, n = np.asarray(x), e, 1
        elif abs(e - best_e) < 1e-12:
            n += 1
    return best_x, best_e, n


def easy_case():
    """One dominant term, a unique well-separated minimum, and tiny noise terms.

    c_max = 100 -> floor = 0.5, so the 0.01 pair terms are dropped.
    Dynamic range 10*log10(100/0.01) = 40 dB, above the 35 dB trigger, so this
    case exercises an actual rewrite.
    """
    H = Poly()
    v = [H.new_var(f"x{i}", 1) for i in range(5)]
    H.add(100.0, v[0])                       # sets c_max
    H.add(-10.0, v[1])
    H.add(-8.0, v[2])
    H.add(-6.0, v[3])
    H.add(-4.0, v[4])
    for a, b in ((v[1], v[2]), (v[2], v[3]), (v[3], v[4])):
        H.add(0.01, a, b)                    # below floor -> dropped
    return H


def adversarial_case():
    """Many small terms that genuinely reorder the spectrum.

    Kept part has a UNIQUE minimum with a gap of only 1.0, while ten dropped
    pair terms of +0.29 sum to delta = 2.9. Since 2*delta = 5.8 > 1.0 the
    certificate must refuse -- and it is right to, because the true minimizer of
    the full Hamiltonian is not the minimizer of the kept part.
    """
    H = Poly()
    x = [H.new_var(f"x{i}", 1) for i in range(6)]
    H.add(60.0, x[0])                        # c_max = 60 -> floor = 0.3
    for i, c in enumerate((-5.0, -4.0, -3.0, -2.0, -1.0), start=1):
        H.add(c, x[i])                       # distinct -> unique minimizer, gap 1.0
    for i in range(1, 6):
        for j in range(i + 1, 6):
            H.add(0.29, x[i], x[j])          # below floor -> dropped, but decisive
    return H


# ------------------------------------------------------------------ easy case
def test_easy_case_fires_and_certifies():
    H = easy_case()
    H_out, cert = cond.truncate_certified(H)
    assert cert["fired"], "tiny terms should fall below c_max/200"
    assert cert["dropped_terms"] == 3
    assert cert["method"] == "exact"
    assert cert["certified"], f"expected a certificate, got margin={cert['margin']}"
    assert cert["n_ground_states"] == 1
    assert cert["margin"] > 0
    assert H_out is not H, "above the trigger and certified -> should be rewritten"
    assert cert["rewritten"]


def test_easy_case_optimum_unchanged():
    H = easy_case()
    H_out, cert = cond.truncate_certified(H)
    x_full, e_full, _ = brute_force(H)
    x_trunc, _, _ = brute_force(H_out)
    assert np.array_equal(x_full, x_trunc), "certified truncation moved the argmin"
    # and the certified minimizer really is optimal for the FULL Hamiltonian
    assert H.evaluate(x_trunc.astype(float)) == pytest.approx(e_full)


def test_easy_case_reduces_dynamic_range():
    H = easy_case()
    _, cert = cond.truncate_certified(H)
    assert cert["db_before"] > cert["db_after"]
    assert cert["db_before"] == pytest.approx(40.0, abs=0.1)


# ----------------------------------------------------------- adversarial case
def test_adversarial_certificate_fails():
    """THE essential test: the certificate must be capable of refusing."""
    H = adversarial_case()
    H_out, cert = cond.truncate_certified(H)
    assert cert["fired"], "the 0.29 terms are below the 0.3 floor"
    assert cert["dropped_terms"] == 10
    assert cert["method"] == "exact"
    assert cert["gap"] == pytest.approx(1.0)
    assert cert["delta"] == pytest.approx(2.9)
    assert not cert["certified"], "certificate passed on an unsafe truncation"
    assert cert["margin"] < 0
    assert not cert["rewritten"]
    assert H_out is H, "an uncertified Hamiltonian must be passed through unchanged"


def test_adversarial_truncation_really_moves_the_optimum():
    """Proves the refusal is not merely conservative -- the answer would change."""
    H = adversarial_case()
    kept, dropped, _floor, _cmax = cond._partition(H, cond.DEFAULT_RESOLUTION_RATIO)
    H_K = cond._rebuild(H, kept)
    x_full, _, _ = brute_force(H)
    x_kept, _, _ = brute_force(H_K)
    assert not np.array_equal(x_full, x_kept), (
        "adversarial case is vacuous: truncation did not move the optimum")


def test_tight_threshold_is_never_looser_than_spec():
    for H in (easy_case(), adversarial_case()):
        _, cert = cond.truncate_certified(H)
        assert cert["margin_tight"] >= cert["margin"] - 1e-12
        assert cert["dropped_max"] - cert["dropped_min"] == pytest.approx(cert["delta"])


# -------------------------------------------------------------- the box bound
def test_bound_holds_at_random_points():
    """|H(x) - H_K(x)| <= delta must hold everywhere in the box."""
    rng = np.random.default_rng(42)
    for H in (easy_case(), adversarial_case()):
        kept, dropped, _f, _c = cond._partition(H, cond.DEFAULT_RESOLUTION_RATIO)
        H_K = cond._rebuild(H, kept)
        _m, _M, delta = cond._dropped_interval(H, dropped)
        vs = sorted(H.upper)
        for _ in range(200):
            x = np.array([rng.integers(0, H.upper[v] + 1) for v in vs], dtype=float)
            assert abs(H.evaluate(x) - H_K.evaluate(x)) <= delta + 1e-9


# --------------------------------------------------------------- idempotence
def test_idempotent():
    H = easy_case()
    H1, c1 = cond.truncate_certified(H)
    H2, c2 = cond.truncate_certified(H1)
    assert not c2["fired"], "second pass should find nothing left to drop"
    assert H2 is H1
    assert c1["db_after"] == pytest.approx(c2["db_before"])


# ---------------------------------------------------------------- degeneracy
def test_degenerate_kept_part_is_not_certified():
    """A degenerate H_K has a big distinct-energy gap but no preserved argmin."""
    H = Poly()
    v = [H.new_var(f"x{i}", 1) for i in range(4)]
    H.add(100.0, v[0])                 # c_max -> floor 0.5
    for i in (1, 2, 3):
        H.add(-0.2, v[i])              # all dropped; they alone decide the answer
    H_out, cert = cond.truncate_certified(H)
    assert cert["fired"]
    assert cert["n_ground_states"] > 1, "kept part should be degenerate here"
    assert cert["certified_subset"], "containment still provable"
    assert not cert["certified"], "must not claim a preserved unique ground state"
    assert H_out is H


# ------------------------------------------------------- heuristic never certifies
def test_heuristic_method_never_certifies():
    H = easy_case()
    _, cert = cond.truncate_certified(H, verify_exact=False)
    assert cert["method"] == "heuristic"
    assert not cert["certified"]
    assert not cert["rewritten"]


def test_exact_limit_forces_heuristic():
    """Too large to enumerate -> heuristic, and therefore uncertified."""
    H = easy_case()
    _, cert = cond.truncate_certified(H, exact_limit=4)
    assert cert["method"] == "heuristic"
    assert not cert["certified"]


# ------------------------------------------------------- the calibrated trigger
def test_below_trigger_is_left_untouched_even_when_certified():
    """The whole point of the calibrated trigger: don't touch what already works."""
    H = easy_case()                                  # 40 dB, certifiable
    H_out, cert = cond.truncate_certified(H, trigger_db=45.0)
    assert cert["certified"], "still provable"
    assert not cert["rewritten"], "below trigger -> must not rewrite"
    assert H_out is H
    assert cert["spec_exceeded_23db"], "40 dB is over the nominal specification"
    assert not cert["over_calibrated_trigger"]


def test_nominal_and_calibrated_thresholds_both_recorded():
    H = easy_case()
    _, cert = cond.truncate_certified(H)
    assert cert["trigger_db"] == cond.CALIBRATED_TRIGGER_DB
    assert cond.NOMINAL_SPEC_DB == pytest.approx(23.0, abs=0.05)
    # 200:1 is 23 dB only in the 10*log10 convention this package uses
    assert 10 * np.log10(cond.DEFAULT_RESOLUTION_RATIO) == pytest.approx(23.0, abs=0.05)


# --------------------------------------------------------------- empty / trivial
def test_no_terms_is_safe():
    H = Poly()
    H.new_var("x", 1)
    H_out, cert = cond.truncate_certified(H)
    assert not cert["fired"]
    assert H_out is H


# ------------------------------------------------------------ hardware legality
def test_legality_flags_over_16_levels():
    H = Poly()
    v = H.new_var("big", 20)
    H.add(1.0, v)
    reasons = cond.hardware_legality(H)
    assert any("16-level" in r for r in reasons)
    assert any("big" in r for r in reasons)
    with pytest.raises(ValueError, match="not legal"):
        cond.assert_hardware_legal(H, stage="design")


def test_legality_flags_total_levels():
    H = Poly()
    for i in range(100):
        v = H.new_var(f"v{i}", 15)
        H.add(1.0, v)
    reasons = cond.hardware_legality(H)
    assert any("total levels" in r for r in reasons)


def test_legality_flags_uncertified_over_range():
    H = adversarial_case()
    _, cert = cond.truncate_certified(H, trigger_db=10.0)
    reasons = cond.hardware_legality(H, cert=cert, trigger_db=10.0)
    assert any("calibrated trigger" in r for r in reasons)


def test_legality_passes_a_clean_binary_qubo():
    H = Poly()
    a = H.new_var("a", 1)
    b = H.new_var("b", 1)
    H.add(-2.0, a)
    H.add(-1.0, b)
    H.add(0.5, a, b)
    assert cond.hardware_legality(H) == []
    cond.assert_hardware_legal(H, stage="islanding")


# ------------------------------------------------------------- opt-out warning
def test_condition_false_warns_with_the_db():
    from eqosystem.solvers import to_eqc_model
    H = easy_case()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        to_eqc_model(H, condition=False, stage="unit-test")
        msgs = [str(w.message) for w in caught if w.category is RuntimeWarning]
    assert msgs, "condition=False must emit a visible warning"
    assert "40.0 dB" in msgs[0]
    assert "unit-test" in msgs[0]
