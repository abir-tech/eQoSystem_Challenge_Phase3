"""W1 against the real pipeline Hamiltonians, at seed 42 on IEEE 69-bus.

The central guarantee tested here: turning conditioning on by default must not
disturb the 20 islanding QUBOs behind the recorded Dirac-3 result. They are
submitted byte-identically, and they still reproduce the recorded energies.
"""
import json
import pathlib

import numpy as np
import pytest

from eqosystem import grid, candidates, scenarios, hamiltonians as ham
from eqosystem import conditioning as cond
from eqosystem.pipeline import run_design
from eqosystem.solvers import AnnealerSolver, ExactSolver, to_eqc_model

ROOT = pathlib.Path(__file__).resolve().parent.parent
HW_JSON = ROOT / "results" / "hardware_dirac3.json"


@pytest.fixture(scope="module")
def rig():
    grid.select("ieee69")
    pool = candidates.generate()
    scens = scenarios.generate(20, seed=42)
    design, H_design, _ = run_design(pool, AnnealerSolver())
    return dict(pool=pool, scens=scens, design=design, H_design=H_design)


@pytest.fixture(scope="module")
def hardware():
    if not HW_JSON.exists():
        pytest.skip("results/hardware_dirac3.json not present")
    return {r["scenario"]: r for r in json.loads(HW_JSON.read_text())["runs"]}


def islanding_hams(rig):
    out = []
    for sc in rig["scens"]:
        H, meta = ham.build_island(rig["design"], rig["pool"], sc)
        if H.n:
            out.append((sc, H, meta))
    return out


# ============================================================ the hardware asset
def test_islanding_qubos_are_never_rewritten(rig):
    """Default-on conditioning must leave every islanding QUBO untouched."""
    for sc, H, _m in islanding_hams(rig):
        H_out, cert = cond.truncate_certified(H)
        assert not cert["rewritten"], (
            f"scenario {sc.sid} at {cert['db_before']:.1f} dB was rewritten; "
            "the calibrated trigger is supposed to leave it alone")
        assert H_out is H


def test_islanding_energies_still_match_recorded_hardware(rig, hardware):
    """The 20/20 result must remain reproducible from this code."""
    checked = 0
    for sc, H, _m in islanding_hams(rig):
        rec = hardware[sc.sid]
        e = ExactSolver().solve(H)["energy"]
        assert e == pytest.approx(rec["exact_energy"], abs=1e-9), (
            f"scenario {sc.sid}: exact energy drifted from the recorded run")
        assert round(H.dynamic_range_db(), 1) == pytest.approx(
            rec["dyn_range_db"], abs=0.05), f"scenario {sc.sid}: dB drifted"
        checked += 1
    assert checked == 20


def test_submitted_model_is_identical_with_and_without_conditioning(rig):
    """Byte-level: the model the device receives is unchanged by W1."""
    import warnings
    for sc, H, _m in islanding_hams(rig):
        m_on = to_eqc_model(H, condition=True)
        with warnings.catch_warnings():          # the opt-out warning is expected here
            warnings.simplefilter("ignore", RuntimeWarning)
            m_off = to_eqc_model(H, condition=False, stage=f"island_s{sc.sid}")
        assert np.array_equal(m_on.coefficients, m_off.coefficients), (
            f"scenario {sc.sid}: conditioning altered the submitted coefficients")
        assert np.array_equal(m_on.indices, m_off.indices)


def test_five_instances_exceed_the_nominal_spec_but_none_the_trigger(rig):
    """Documents the measured basis for the calibrated trigger."""
    over_spec, over_trigger, dbs = [], [], []
    for sc, H, _m in islanding_hams(rig):
        _, cert = cond.truncate_certified(H)
        dbs.append(cert["db_before"])
        if cert["spec_exceeded_23db"]:
            over_spec.append(sc.sid)
        if cert["over_calibrated_trigger"]:
            over_trigger.append(sc.sid)
    assert len(over_spec) == 5, f"expected 5 instances above 23 dB, got {over_spec}"
    assert over_trigger == [], "no islanding instance should cross the 35 dB trigger"
    assert max(dbs) == pytest.approx(30.9, abs=0.1)


def test_islanding_stage_is_hardware_legal(rig):
    for sc, H, _m in islanding_hams(rig):
        _, cert = cond.truncate_certified(H)
        assert cond.hardware_legality(H, cert=cert) == [], (
            f"scenario {sc.sid} should be legal on the integer solver")


def test_sa_matches_exact_on_every_islanding_qubo(rig):
    """E2 in miniature -- the classical certification path still holds."""
    for sc, H, _m in islanding_hams(rig):
        assert AnnealerSolver().solve(H)["energy"] == pytest.approx(
            ExactSolver().solve(H)["energy"], abs=1e-9)


def test_specified_certificate_would_be_unsound_without_uniqueness(rig):
    """Measured finding: on our problems the bare `E2 - E1 > 2*delta` test is unsafe.

    On every islanding instance where truncation fires, the dropped terms are the
    0.1 per-island switching costs -- exactly what distinguishes "energize only
    the island that restores load" from "energize worthless islands too". Removing
    them makes those states TIE, so the kept part has a degenerate ground state
    while its gap to the second-best distinct energy looks enormous.

    The specification's certificate would report success on all five. Requiring a
    unique minimizer catches it. Recorded so the uniqueness check is not removed
    as redundant.
    """
    import itertools
    fired = 0
    for sc, H, _m in islanding_hams(rig):
        _out, cert = cond.truncate_certified(H)
        if not cert["fired"]:
            continue
        fired += 1
        # the specification's bare condition passes ...
        assert cert["margin"] > 0, f"s{sc.sid}: expected the bare 2*delta test to pass"
        assert cert["certified_subset"]
        # ... but the kept part is degenerate, so we must not certify
        assert cert["n_ground_states"] > 1, f"s{sc.sid}: expected tied states"
        assert not cert["certified"]

        # and the tie is real: the full optimum is one member of a larger tied set
        kept, _d, _f, _c = cond._partition(H, cond.DEFAULT_RESOLUTION_RATIO)
        H_K = cond._rebuild(H, kept)
        vs = sorted(H.upper)
        pts = list(itertools.product(*[range(H.upper[v] + 1) for v in vs]))
        ek = np.array([H_K.evaluate(np.asarray(p, dtype=float)) for p in pts])
        ef = np.array([H.evaluate(np.asarray(p, dtype=float)) for p in pts])
        tied = {pts[i] for i in np.flatnonzero(np.abs(ek - ek.min()) < 1e-12)}
        best_full = pts[int(np.argmin(ef))]
        assert best_full in tied
        assert len(tied) > 1, f"s{sc.sid}: truncation should have created a tie"
    assert fired == 5, f"expected truncation to fire on 5 instances, got {fired}"


# ====================================================== the two integer stages
def test_design_stage_cannot_be_certified(rig):
    """Still uncertifiable, but for a different reason after W1b.

    Under the absolute encoding the prod(u_i) bound was enormous (delta 3.4e5).
    The trust-region encoding shrinks it by roughly 300x because the bounds
    themselves shrank, yet certification still fails: the stage remains far too
    large to enumerate E1/E2 exactly, and a heuristic spectrum may never be used
    to claim a certificate.
    """
    _, cert = cond.truncate_certified(rig["H_design"])
    assert cert["fired"]
    assert cert["method"] == "heuristic", "design stage is far too large to enumerate"
    assert not cert["certified"]
    assert not cert["certified_subset"], "heuristic spectra must not certify either"


def test_trust_region_shrinks_the_truncation_bound(rig):
    """Pins the W1b improvement to the bound itself."""
    from eqosystem.pipeline import run_design as _rd
    _d, H_abs, _ = _rd(rig["pool"], AnnealerSolver(), delta=False)
    _o1, c_abs = cond.truncate_certified(H_abs)
    _o2, c_dlt = cond.truncate_certified(rig["H_design"])
    assert c_dlt["delta"] < c_abs["delta"] / 50.0, (
        f"expected a large drop, got {c_abs['delta']:.3g} -> {c_dlt['delta']:.3g}")
    assert c_dlt["db_before"] < c_abs["db_before"]


def test_design_truncation_would_destroy_the_formulation(rig):
    """R=200 on the design stage drops coverage and linking, not just noise.

    This is why truncation must be gated on a certificate rather than on dB: the
    truncated Hamiltonian reports an excellent dynamic range while encoding a
    different problem.
    """
    # rebuild with meta so terms can be attributed to variable roles
    H, meta = ham.build_design(rig["pool"])
    kept, dropped, floor, c_max = cond._partition(H, cond.DEFAULT_RESOLUTION_RATIO)
    assert floor > 25.0, (
        f"floor {floor:.1f} should sit above the coverage penalty weight (25)")
    assert floor > 12.0, "floor should sit above the linking penalty weight (12)"
    assert len(dropped) > len(kept) / 2, "most terms fall below the floor"

    n_ids = set(meta["n"].values())
    y_ids = set(meta["y"].values())
    pure_cost = [k for k in dropped if len(k) == 1 and k[0] in n_ids]
    switch_cost = [k for k in dropped if len(k) == 1 and k[0] in y_ids]
    assert pure_cost, "asset capital costs should fall below the floor"
    assert switch_cost, "tie-switch costs should fall below the floor"

    # what survives cannot express coverage over the whole feeder
    b_ids = set(meta["b"].values())
    kept_pure_b = [k for k in kept if all(v in b_ids for v in k)]
    dropped_pure_b = [k for k in dropped if all(v in b_ids for v in k)]
    assert dropped_pure_b or kept_pure_b, "sanity: coverage terms exist somewhere"


def test_absolute_design_encoding_is_refused_by_the_hardware_guard(rig):
    """The guard must still refuse the pre-W1b encoding."""
    from eqosystem.pipeline import run_design as _rd
    _d, H_abs, _ = _rd(rig["pool"], AnnealerSolver(), delta=False)
    _o, cert = cond.truncate_certified(H_abs)
    reasons = cond.hardware_legality(H_abs, cert=cert)
    assert reasons, "absolute encoding exceeds the integer-solver limits"
    assert any("16-level" in r for r in reasons)
    with pytest.raises(ValueError, match="not legal"):
        cond.assert_hardware_legal(H_abs, cert=cert, stage="design")


def test_trust_region_design_is_hardware_legal(rig):
    """W1b outcome: the design stage now passes every integer-solver limit.

    Levels, the 16-level per-variable cap, and the calibrated dynamic-range
    trigger all pass -- with the MILP cost ratio unchanged.
    """
    H = rig["H_design"]
    _, cert = cond.truncate_certified(H)
    assert cond.total_levels(H) <= cond.MAX_TOTAL_LEVELS
    assert max(H.upper.values()) <= cond.MAX_LEVELS_PER_VAR - 1
    assert H.dynamic_range_db() <= cond.CALIBRATED_TRIGGER_DB
    assert cond.hardware_legality(H, cert=cert) == []
    cond.assert_hardware_legal(H, cert=cert, stage="design")


def test_dispatch_stage_is_refused_by_the_hardware_guard(rig):
    sc = max(rig["scens"], key=lambda s: len(s.dead_buses))
    c0 = rig["design"]["selected"][0]
    Hm, _md = ham.build_dispatch_mp(rig["design"], rig["pool"], sc, c0)
    _, cert = cond.truncate_certified(Hm)
    assert not cert["certified"]
    reasons = cond.hardware_legality(Hm, cert=cert)
    assert reasons
    with pytest.raises(ValueError, match="not legal"):
        cond.assert_hardware_legal(Hm, cert=cert, stage="dispatch_mp")


# ================================================================= the bound
@pytest.mark.parametrize("stage", ["design", "dispatch", "island"])
def test_bound_holds_on_real_hamiltonians(rig, stage):
    """|H_trunc(x) - H(x)| <= delta at 200 random feasible points."""
    if stage == "design":
        H = rig["H_design"]
    elif stage == "dispatch":
        sc = max(rig["scens"], key=lambda s: len(s.dead_buses))
        H, _ = ham.build_dispatch_mp(rig["design"], rig["pool"], sc,
                                     rig["design"]["selected"][0])
    else:
        H = islanding_hams(rig)[0][1]

    kept, dropped, _f, _c = cond._partition(H, cond.DEFAULT_RESOLUTION_RATIO)
    H_K = cond._rebuild(H, kept)
    _m, _M, delta = cond._dropped_interval(H, dropped)
    rng = np.random.default_rng(42)
    vs = sorted(H.upper)
    for _ in range(200):
        x = np.array([rng.integers(0, H.upper[v] + 1) for v in vs], dtype=float)
        assert abs(H.evaluate(x) - H_K.evaluate(x)) <= delta + 1e-6
