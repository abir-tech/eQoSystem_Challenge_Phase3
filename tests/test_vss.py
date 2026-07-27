"""W10 -- VSS / EVPI.

Uses a small scenario set so the suite stays fast; the reported numbers come
from `run_experiments.py --vss` at the full scenario count.
"""
import pytest

from eqosystem import grid, candidates, scenarios, hamiltonians as ham
from eqosystem import pipeline as pl
from eqosystem.solvers import AnnealerSolver


@pytest.fixture(scope="module")
def small():
    grid.select("ieee69")
    pool = candidates.generate()
    scens = scenarios.generate(6, seed=42)
    return pool, scens


def test_cost_is_annualized_not_raw_sum():
    """Regression: adding one-off capex to per-event unserved energy is wrong.

    With raw addition the recourse term was ~0.2% of capex and the study just
    picked the cheapest design regardless of how much load it shed.
    """
    c = pl.total_cost(capex_units=340.0, unserved_kwh=700.0)
    assert c == pytest.approx(340.0 / pl.PROJECT_YEARS
                              + pl.ANNUAL_EVENTS * pl.VOLL_PER_KWH * 700.0 / pl.COST_UNIT)
    # the recourse term must be a materially sized share, not a rounding error
    recourse = pl.ANNUAL_EVENTS * pl.VOLL_PER_KWH * 700.0 / pl.COST_UNIT
    assert recourse / c > 0.15


def test_total_cost_monotone_in_both_arguments():
    base = pl.total_cost(300.0, 500.0)
    assert pl.total_cost(310.0, 500.0) > base
    assert pl.total_cost(300.0, 600.0) > base


def test_infeasible_design_is_detected(small):
    """A margin so large the asset caps cannot meet demand must be rejected."""
    pool, _scens = small
    d = pl.design_at_margin(pool, AnnealerSolver(), 1.60)
    assert not pl.design_is_feasible(d, pool, 1.60), (
        "margin 1.6 under-builds -- it must not be usable in the study")
    d125 = pl.design_at_margin(pool, AnnealerSolver(), 1.25)
    assert pl.design_is_feasible(d125, pool, 1.25)
    # Note: before W1b the infeasible design was also CHEAPER than the feasible
    # one (238.8 vs 338.8), which is what made it a trap the study fell into.
    # Centring each candidate on a portfolio that covers its own demand removed
    # that: margin 1.6 now costs more (418.3) as well as being infeasible. The
    # feasibility gate is what protects E10 either way, so it is what is pinned.


def test_feasible_designs_pass_the_gate(small):
    pool, _scens = small
    for m in (1.00, 1.10, 1.25):
        d = pl.design_at_margin(pool, AnnealerSolver(), m)
        assert pl.design_is_feasible(d, pool, m), f"margin {m} should be feasible"


def test_design_margin_is_restored_after_use(small):
    pool, _scens = small
    before = ham.DESIGN_MARGIN
    pl.design_at_margin(pool, AnnealerSolver(), 1.60)
    assert ham.DESIGN_MARGIN == before, "global margin leaked out of the study"


def test_vss_and_evpi_are_nonnegative(small):
    """VSS >= 0 and EVPI >= 0 are structural; a negative value means a bug."""
    pool, scens = small
    v = pl.vss_evpi(pool, scens, AnnealerSolver(), margins=(1.00, 1.10, 1.25))
    assert v["VSS"] >= -1e-9, f"VSS negative: {v['VSS']}"
    assert v["EVPI"] >= -1e-9, f"EVPI negative: {v['EVPI']}"
    assert v["WS"] <= v["RP"] + 1e-9, "perfect foresight cannot be worse than RP"
    assert v["RP"] <= v["EEV"] + 1e-9, "stochastic cannot be worse than mean-value"


def test_infeasible_margins_excluded_from_the_study(small):
    pool, scens = small
    v = pl.vss_evpi(pool, scens, AnnealerSolver(), margins=(1.00, 1.25, 1.60))
    assert 1.60 in v["infeasible_margins"]
    assert 1.60 not in v["usable_margins"]
    assert v["margin_stochastic"] != 1.60


def test_sensitivity_is_recorded_and_spans_regimes(small):
    """The conclusion depends on VOLL and event rate, so both must be reported."""
    pool, scens = small
    v = pl.vss_evpi(pool, scens, AnnealerSolver(), margins=(1.00, 1.10, 1.25))
    assert v["sensitivity"], "sensitivity sweep missing"
    assert all(s["VSS"] >= -1e-9 for s in v["sensitivity"])
    assert {s["voll"] for s in v["sensitivity"]} >= {2.0, 50.0}
