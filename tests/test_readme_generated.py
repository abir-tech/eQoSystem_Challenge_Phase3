"""W11 -- the README's numbers must be generated, never hand-entered.

This is the guard for the rule that broke the team's Phase 2 submission: a
reported figure that the submitted code does not produce.
"""
import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
GEN = ROOT / "tools" / "gen_readme.py"
RESULTS = ROOT / "results" / "results_simulated-annealing.json"
HW = ROOT / "results" / "hardware_dirac3.json"


def test_markers_present():
    text = README.read_text(encoding="utf-8")
    assert "<!-- BEGIN GENERATED RESULTS" in text
    assert "<!-- END GENERATED RESULTS -->" in text


@pytest.mark.skipif(not RESULTS.exists(), reason="results JSON not generated yet")
def test_readme_is_in_sync_with_results():
    """`gen_readme.py --check` is the reproducibility gate."""
    r = subprocess.run([sys.executable, str(GEN), "--check"],
                       cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, (
        f"README.md is stale -- run `python tools/gen_readme.py`\n{r.stdout}{r.stderr}")


def generated_block():
    text = README.read_text(encoding="utf-8")
    return text.split("<!-- BEGIN GENERATED RESULTS", 1)[1].split(
        "<!-- END GENERATED RESULTS -->", 1)[0]


@pytest.mark.skipif(not HW.exists(), reason="hardware JSON absent")
def test_hardware_claims_match_the_hardware_json():
    """Provenance guard.

    An earlier version of the generator sourced the "observed resolving
    correctly on the device" figures from the CLASSICAL run's certificates,
    which silently attributed simulated numbers to hardware. These must come
    from results/hardware_dirac3.json.
    """
    hw = json.loads(HW.read_text())
    runs = hw["runs"]
    block = generated_block()

    max_db = max(r["dyn_range_db"] for r in runs)
    n_over = sum(1 for r in runs if r["dyn_range_db"] > 23.0)
    n_nontrivial = sum(1 for r in runs if r["hw_energy"] != 0.0)

    assert f"{hw['matched']}/{hw['total']}" in block
    assert f"**{max_db:.1f} dB**" in block or f"0.0-{max_db:.1f} dB" in block
    assert f"**{n_over}**" in block or f"**{n_over} instances" in block
    assert f"**{n_nontrivial} required active islanding**" in block, (
        "the non-trivial split must be disclosed and must match the hardware run")


@pytest.mark.skipif(not RESULTS.exists(), reason="results JSON not generated yet")
def test_headline_numbers_match_results_json():
    R = json.loads(RESULTS.read_text())
    block = generated_block()
    assert f"{100 * R['E1']['M1_max_unserved']:.1f}%" in block
    assert f"**{R['E1']['M2_crit_hours']}**" in block
    assert f"{R['E4']['cost_ratio']:.3f}x" in block
    assert f"{R['E2']['optimal']}/{R['E2']['problems']}" in block


def test_no_speedup_claim_anywhere():
    """Part 7 claims register: no speedup over classical, in any file."""
    banned = re.compile(r"\b(faster than|speed-?up|outperform\w*)\b", re.I)
    offenders = []
    for p in list(ROOT.glob("*.md")) + list(ROOT.glob("*.py")) + \
            list((ROOT / "eqosystem").glob("*.py")) + list((ROOT / "tools").glob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if banned.search(line) and "No speedup" not in line and \
                    "no speedup" not in line.lower():
                offenders.append(f"{p.name}:{i}: {line.strip()}")
    assert not offenders, "speedup language found:\n" + "\n".join(offenders)


def test_data_provenance_statement_present():
    text = README.read_text(encoding="utf-8")
    assert "publicly released IEEE" in text
    assert "No proprietary" in text


def test_qbraid_badge_present():
    text = README.read_text(encoding="utf-8")
    assert "Launch_on_qBraid" in text
    assert "account.qbraid.com?gitHubUrl=" in text


# ===================== the write-up must also be generated ==================
REPORT = ROOT / "report" / "eQoSystem_Phase3.tex"
GENREP = ROOT / "tools" / "gen_report.py"


@pytest.mark.skipif(not RESULTS.exists(), reason="results JSON not generated yet")
def test_report_is_in_sync_with_results():
    """Same reproducibility gate as the README, for the submitted write-up."""
    r = subprocess.run([sys.executable, str(GENREP), "--check"],
                       cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, (
        f"report .tex is stale -- run `python tools/gen_report.py`\n"
        f"{r.stdout}{r.stderr}")


@pytest.mark.skipif(not REPORT.exists(), reason="report not generated yet")
def test_report_meets_the_submission_format():
    t = REPORT.read_text(encoding="utf-8")
    assert "11pt" in t, "challenge requires 11-point type"
    assert "mathptmx" in t, "challenge requires Times New Roman"
    assert r"\linespread{1.0}" in t, "challenge requires single spacing"
    assert r"\onecolumn" in t, "references must not eat body pages"


@pytest.mark.skipif(not REPORT.exists(), reason="report not generated yet")
def test_report_addresses_all_four_required_topics():
    t = REPORT.read_text(encoding="utf-8").lower()
    for topic in ("focus area and rationale", "quantum integration",
                  "stakeholder relevance", "results, findings and observations"):
        assert topic in t, f"submission must address: {topic}"


@pytest.mark.skipif(not REPORT.exists(), reason="report not generated yet")
def test_report_hardware_numbers_match_the_hardware_json():
    """Provenance: hardware claims in the paper come from the hardware run."""
    hw = json.loads(HW.read_text())
    t = REPORT.read_text(encoding="utf-8")
    assert f"{hw['matched']}/{hw['total']}" in t
    assert f"{max(r['dyn_range_db'] for r in hw['runs']):.1f}~dB" in t
    nt = sum(1 for r in hw["runs"] if r["hw_energy"] != 0.0)
    assert str(nt) in t, "the non-trivial split must be disclosed"


@pytest.mark.skipif(not REPORT.exists(), reason="report not generated yet")
def test_report_makes_no_speedup_claim():
    t = REPORT.read_text(encoding="utf-8")
    assert "no speedup" in t.lower(), "the claims register requires this disclaimer"
    banned = re.compile(r"\b(faster than|speed-?up over|outperform\w*)\b", re.I)
    for line in t.splitlines():
        if banned.search(line) and "no speedup" not in line.lower():
            raise AssertionError(f"speedup language in the write-up: {line.strip()}")
