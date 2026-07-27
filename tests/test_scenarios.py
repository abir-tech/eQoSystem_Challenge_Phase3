"""W6 -- 50 scenarios by default, plus the scenario-tree reading."""
import numpy as np
import pytest

from eqosystem import grid, scenarios


@pytest.fixture(autouse=True)
def on_69():
    grid.select("ieee69")


# ------------------------------------------------------------------ flat mode
def test_default_is_fifty():
    assert scenarios.DEFAULT_N_SCENARIOS == 50
    assert len(scenarios.generate()) == 50


def test_seed_reproducible():
    a = scenarios.generate(20, seed=42)
    b = scenarios.generate(20, seed=42)
    assert [s.failed_line for s in a] == [s.failed_line for s in b]
    assert [s.load_factor for s in a] == [s.load_factor for s in b]


def test_durations_and_spread():
    s = scenarios.generate(50, seed=42)
    assert all(4 <= sc.duration <= 16 for sc in s)
    assert len({sc.failed_line for sc in s}) > 10, "failures should spread over lines"
    assert any(len(sc.dead_buses) >= 60 for sc in s), "expect a near-total blackout"


def test_first_twenty_are_not_a_prefix_of_fifty():
    """LHS is a joint design, so n=50 is a different sample, not an extension."""
    a = scenarios.generate(20, seed=42)
    b = scenarios.generate(50, seed=42)
    assert [s.failed_line for s in a] != [s.failed_line for s in b[:20]]


def test_flat_mode_has_no_paths():
    for sc in scenarios.generate(20, seed=42):
        assert not sc.is_tree
        assert sc.load_factor_at(sc.hours[0]) == sc.load_factor
        assert sc.r_factor_at(sc.hours[0]) == sc.r_factor


# ------------------------------------------------------------------ tree mode
def test_tree_mode_resamples_per_bucket():
    s = scenarios.generate(20, seed=42, tree=True)
    assert all(sc.is_tree for sc in s)
    varying = [sc for sc in s if len(set(sc.load_path)) > 1]
    assert varying, "load forecast error should vary across buckets"
    varying_r = [sc for sc in s if len(set(sc.r_path)) > 1]
    assert varying_r, "renewable forecast error should vary across buckets"


def test_tree_mode_keeps_the_contingency_fixed_within_a_branch():
    """The contingency is the branch; only forecast errors are resampled."""
    flat = scenarios.generate(20, seed=42)
    tree = scenarios.generate(20, seed=42, tree=True)
    assert [s.failed_line for s in flat] == [s.failed_line for s in tree]
    assert [s.duration for s in flat] == [s.duration for s in tree]
    assert [s.start_hour for s in flat] == [s.start_hour for s in tree]
    assert [s.dead_buses for s in flat] == [s.dead_buses for s in tree]


def test_tree_paths_stay_inside_the_sampling_ranges():
    for sc in scenarios.generate(50, seed=42, tree=True):
        assert all(0.2 <= r <= 1.0 for r in sc.r_path)
        assert all(0.7 <= l <= 1.3 for l in sc.load_path)


def test_tree_factors_track_the_branch_central_value():
    """Per-bucket values are forecast errors AROUND the branch value."""
    for sc in scenarios.generate(20, seed=42, tree=True):
        assert abs(float(np.mean(sc.load_path)) - sc.load_factor) < 0.16 * sc.load_factor
        assert abs(float(np.mean(sc.r_path)) - sc.r_factor) < 0.31 * sc.r_factor


def test_bucket_lookup_covers_every_outage_hour():
    for sc in scenarios.generate(20, seed=42, tree=True):
        for h in sc.hours:
            assert 0 <= sc._bucket(h) < len(sc.load_path)
            assert sc.load_factor_at(h) in sc.load_path
            assert sc.r_factor_at(h) in sc.r_path


def test_tree_mode_is_seed_reproducible():
    a = scenarios.generate(20, seed=42, tree=True)
    b = scenarios.generate(20, seed=42, tree=True)
    assert [s.load_path for s in a] == [s.load_path for s in b]
    assert [s.r_path for s in a] == [s.r_path for s in b]


# ------------------------------------------------------------------- stress
def test_stress_widens_the_envelope():
    n = scenarios.generate(20, seed=42)
    s = scenarios.generate(20, seed=42, stress=True)
    assert max(x.load_factor for x in s) > max(x.load_factor for x in n)
    assert max(x.duration for x in s) > max(x.duration for x in n)
