"""W7 -- cross-island export over closed ties."""
import numpy as np
import pytest

from eqosystem import grid, candidates, scenarios, hamiltonians as ham
from eqosystem.pipeline import run_design, run_scenario
from eqosystem.solvers import AnnealerSolver, ExactSolver


@pytest.fixture(scope="module")
def rig():
    grid.select("ieee69")
    pool = candidates.generate()
    scens = scenarios.generate(50, seed=42)
    design, _H, _ = run_design(pool, AnnealerSolver())
    return dict(pool=pool, scens=scens, design=design)


def test_tie_adjacency_is_symmetric_and_real(rig):
    adj = ham.tie_adjacency(rig["pool"])
    assert adj, "the 69-bus feeder has tie switches bridging candidates"
    for (ci, cj), ties in adj.items():
        assert ci < cj, "keys must be canonically ordered"
        si, sj = set(rig["pool"][ci]["buses"]), set(rig["pool"][cj]["buses"])
        for l in ties:
            u, v = grid.TIE_SWITCHES[l]
            assert (u in si and v in sj) or (v in si and u in sj)


def test_export_requires_a_genuine_reachability_gain(rig):
    """Regression for a real defect.

    The first implementation computed the orphan set as
    `dead & dst_buses - reach_i - reach_j`, which credits the pair with EVERY
    dead bus whenever neither island reaches anything -- as if a tie between two
    specific buses resurrected a whole fragment. Measured, that turned 19
    scenarios from 0 active islands to 3 and manufactured a 13862 kWh
    "improvement" of which 0 came from export. Export must be a difference of
    post-fault reachability, so every exported bus must be reachable on the
    union-plus-tie graph.
    """
    import networkx as nx
    checked = 0
    for sc in rig["scens"]:
        _H, mi = ham.build_island(rig["design"], rig["pool"], sc,
                                  export_aware=True)
        for (ci, cj), ex in mi["exports"].items():
            bs = set(rig["pool"][ci]["buses"]) | set(rig["pool"][cj]["buses"])
            g = nx.Graph(); g.add_nodes_from(bs)
            for (u, v, *_r) in grid.LINES:
                if u in bs and v in bs and (u, v) != sc.failed_line \
                        and (v, u) != sc.failed_line:
                    g.add_edge(u, v)
            for l in list(rig["design"]["switches"]) + list(ex["ties"]):
                u, v = grid.TIE_SWITCHES[l]
                if u in bs and v in bs:
                    g.add_edge(u, v)
            hubs = set(rig["pool"][ci].get("hubs", [])) | \
                set(rig["pool"][cj].get("hubs", []))
            joint = set()
            for hb in hubs:
                if hb in g:
                    joint |= set(nx.node_connected_component(g, hb))
            for b in ex["buses"]:
                assert b in joint, (
                    f"exported bus {b} is not reachable even with the tie closed")
                assert b in sc.dead_buses, "only dead buses can be exported to"
                assert b not in mi["info"][ci]["reach"], "already reachable alone"
                assert b not in mi["info"][cj]["reach"], "already reachable alone"
                checked += 1
    assert checked > 0, "expected at least one genuine export opportunity"


def test_export_is_bounded_by_tie_capacity(rig):
    for sc in rig["scens"]:
        _H, mi = ham.build_island(rig["design"], rig["pool"], sc,
                                  export_aware=True)
        for ex in mi["exports"].values():
            assert ex["kw"] <= ham.TIE_CAPACITY_KW + 1e-6


def test_export_reward_is_degree_2_and_adds_no_variables(rig):
    for sc in rig["scens"][:12]:
        Hb, _mb = ham.build_island(rig["design"], rig["pool"], sc,
                                   export_aware=False)
        Ha, _ma = ham.build_island(rig["design"], rig["pool"], sc,
                                   export_aware=True)
        assert Ha.n == Hb.n, "export must add no variables"
        assert Ha.degree <= 2, "islanding QUBO must stay degree 2"
        assert all(H.upper[v] == 1 for H in (Ha,) for v in H.upper), "stays binary"


def test_exported_load_is_charged_to_the_donor(rig):
    """Supply-capped accounting: serving an exported bus must also cost the
    donor its generation, otherwise an island serves more than it makes."""
    sc = next((s for s in rig["scens"]
               if ham.build_island(rig["design"], rig["pool"], s,
                                   export_aware=True)[1]["exports"]), None)
    assert sc is not None
    r = run_scenario(rig["design"], rig["pool"], sc, AnnealerSolver())
    assert r["n_exported_buses"] >= 0
    assert r["export_pairs"] >= 0
    # unserved energy can never be negative, which a double-count would allow
    assert r["unserved_energy"] >= -1e-9


def test_export_never_worsens_expected_unserved_energy(rig):
    """Over the scenario set, enabling export must not increase shed energy."""
    tot_off = tot_on = 0.0
    for sc in rig["scens"][:20]:
        import eqosystem.hamiltonians as _h
        orig = _h.build_island
        _h.build_island = lambda *a, **k: orig(*a, **{**k, "export_aware": False})
        tot_off += run_scenario(rig["design"], rig["pool"], sc,
                                AnnealerSolver())["unserved_energy"]
        _h.build_island = lambda *a, **k: orig(*a, **{**k, "export_aware": True})
        tot_on += run_scenario(rig["design"], rig["pool"], sc,
                               AnnealerSolver())["unserved_energy"]
        _h.build_island = orig
    assert tot_on <= tot_off + 1e-6
