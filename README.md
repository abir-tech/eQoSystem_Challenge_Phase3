# eQoSystem — Two-Stage Stochastic Microgrid Design on QCi Dirac-3

**Team:** eQoSystem · **Challenge track:** GIC 2026 — Energy Infrastructure (QCi) · **Title:** Cost Optimization in Resilient Power Grids

A three-stage quantum-optimization pipeline that designs island-capable microgrids on the **IEEE 69-bus** distribution system (IEEE 33-bus also supported via `--grid ieee33`) and operates them through 20 Latin-Hypercube-sampled N-1 contingencies with 4-16 h outage horizons. Dispatch is **time-coupled**: a multi-period Hamiltonian with battery state-of-charge recursion (charge from midday PV, discharge into the evening) solved as ONE hardware job per island. Grid-connected mode is demonstrated via a PCC import/export Hamiltonian under time-of-use prices (E7). Black-start feasibility (grid-forming source required), inverter P-Q capability limits, and ZIP voltage-dependent loads are modeled as documented approximations per the challenge's 'Use of Approximation' criterion. Every optimization problem is formulated as a bounded-integer polynomial Hamiltonian (degree ≤ 3) in the exact submission format of QCi's Dirac-3 entropy quantum computer, using native qudit (integer) encoding — no binary expansion, no auxiliary variables for the cubic terms.

## Headline results (seed 42, 20 scenarios, IEEE 33-bus)

| Challenge metric | Legacy grid (no islanding) | eQoSystem | Change |
|---|---|---|---|
| M1 — max fraction of customers unserved in any hour | 100.0% | **24.4%** | −75.6 pts |
| M2 — critical-infrastructure hours unserved (all scenarios) | 174 h | **0 h** | −100% |
| Expected unserved energy per contingency | 5,742 kWh | **674 kWh** | −88% |
| M3 — grid-upgrade cost vs certified MILP optimum | 1.000× (HiGHS) | **1.005×** | +0.5% |

Solution-quality certification: all 20 islanding QUBOs solved to **certified global optimality** (exhaustive enumeration cross-check, max gap 0.0). Robustness: with 40 out-of-design scenarios (`--n-scenarios 40`), M2 stays 0 (baseline 313 h) and all 40 islanding problems remain certified optimal.

**Electrical feasibility (E6):** every energized island-hour is validated post-hoc with LinDistFlow — 146/146 pass the ±5% voltage band and thermal ratings on IEEE 69-bus (Vmin 0.987 pu, worst line 66%; 349/349 on 33-bus). The checker includes **ZIP voltage-dependent loads** (two-pass) and **inverter P-Q capability limits**, and is validated against canonical full-feeder results (33-bus: Vmin≈0.916 @ bus 18; 69-bus: Vmin≈0.913 @ bus 65 — both violating the band at peak, so islanded DER operation measurably improves the feeders' known voltage weakness). **Grid-connected mode (E7):** a 36-variable PCC Hamiltonian with SOC recursion cuts daily energy cost 29.9% via storage arbitrage under TOU prices. Multi-period dispatch is solved on-hardware as one Hamiltonian per island (≈455 of 954 qudit levels, degree 3).

## Setup

```bash
pip install -r requirements.txt          # numpy scipy networkx matplotlib eqc-models
```

## Run

```bash
# Classical reference (simulated annealing + exact + MILP baselines) — runs anywhere
python run_experiments.py --backend sa

# On QCi Dirac-3 via qBraid: set your token, then
export QCI_TOKEN=<your token>            # QCI_API_URL if non-default
python run_experiments.py --backend dirac3
```

`--backend dirac3` submits ~50 jobs to `Dirac3IntegerCloudSolver`
(1 design + 20 islanding + ~29 dispatch), `relaxation_schedule=2`,
`num_samples=10`, each sample classically polished by greedy descent.
The local annealer uses delta evaluation (only terms touching a moved
variable are re-evaluated), with a drift self-check against full evaluation.
Outputs land in `results/` (JSON + plots). Judges can re-run without
modification; the SA backend reproduces the table above deterministically
(fixed seeds).


## Use as a library

The code is an installable Python package:

```bash
pip install -e .          # from the project root (deps via requirements.txt)
eqosystem-experiments --backend sa      # console command, same flags as run_experiments.py
```

```python
from eqosystem import candidates, scenarios, hamiltonians, solvers
from eqosystem.pipeline import run_design, run_scenario

pool   = candidates.generate()                 # overlapping island candidates
design, H_design, _ = run_design(pool, solvers.AnnealerSolver())
scens  = scenarios.generate(20)                # LHS contingencies (stress=True for beyond-design)
result = run_scenario(design, pool, scens[0], solvers.AnnealerSolver())
model  = solvers.to_eqc_model(H_design)        # Dirac-3 submission object
```

## Pipeline

1. **Candidates** (`candidates.py`) — overlapping island candidates from spectral clustering at k = 3, 4, 5 (challenge: "candidates need not be distinct"). Overlap is what couples the downstream islanding QUBO.
2. **Stage 1 — H_design** (50 qudits, degree 3, 196 terms, 41 dB dynamic range) — island selection (binary) + DER portfolio sizing (integer PV/BESS/DG units) + tie-switch deployment. Coverage is a quadratic penalty; capacity feasibility is a *gated* quadratic `b_c·(capacity − demand − slack)²` — a native degree-3 term on Dirac-3 that QUBO hardware would need auxiliary variables to express.
3. **Stage 2 — H_island(s)** (≤ 9 qudits, degree 2, per scenario) — which built islands energize, coupled by load-weighted overlap on de-energized buses; value scaled by each island's physical serve-capability under the scenario's renewable/load factors; reachability computed on the post-contingency topology including closed tie switches.
4. **Stage 3 — H_dispatch(s, c)** (2–4 qudits after compression, degree 3) — integer setpoints in 10 kW steps for PV/BESS/DG plus a served-load level; convex cubic diesel heat-rate term, calibrated against the serve reward.
5. **Evaluation** (`pipeline.py`) — hourly accounting over each outage window with supply-capped service (never credits more load than generation), critical buses first.

## Quantum resource accounting

| Stage | Vars (qudits) | Binary-qubit equivalent | Degree | Jobs |
|---|---|---|---|---|
| Design | 50 | 149 | 3 | 1 |
| Islanding | ~5 | ~5 | 2 | 20 |
| Dispatch | 2–4 | 12–22 | 3 | ~29 |

Native integer encoding cuts the design problem from 149 binary qubits to 50 qudits; per-island decomposition keeps every hardware job at ≤ 50 variables.

## Noise engineering

The E3 experiment corrupts Hamiltonian coefficients with 1% relative analog noise and measures P(ground state) as a function of coefficient dynamic range: it collapses from 0.87 at range 10⁰ to ≈0 beyond 10⁴. Our Hamiltonians are conditioned to 19–57 dB (vs ~80 dB in our Phase-2 prototype) by (i) normalizing capacity violations in asset units, (ii) demand-relative balance penalties, (iii) calibrated cubic coefficients, and (iv) compressing fixed-at-zero variables out of the model. Hardware-side: `num_samples` best-of-N, `relaxation_schedule`, and classical greedy polish of every sample.

## Limitations (honest accounting)

* Voltage and thermal limits are now **validated post-hoc via LinDistFlow** (E6) rather than enforced inside the Hamiltonians; full AC power flow, protection coordination, and islanding transients remain out of scope.
* BESS state-of-charge is now simulated hourly in the evaluation; the dispatch Hamiltonian itself still optimizes the worst hour (inter-temporal dispatch coupling is future work).
* Islands site DERs at two hubs; installed capacity is shared across post-fault fragments in proportion to demand (stated assumption).
* Cross-island tie interconnection (one island exporting to another) is not modeled.
* DistFlow-style linearization: thermal limits and voltage are handled via penalty shaping, not full AC power flow.
* Design-stage MILP is linear, so HiGHS solves it in milliseconds — we claim parity (1.005×), not quantum advantage, on that stage. The quantum case rests on the coupled, degree-3, mixed binary/integer formulation being expressible as a *single* Hamiltonian across all three stages.
* BESS modeled with a power/energy cap per outage window, not inter-temporal SOC dynamics.

## Repository layout

```
eqosystem/grid.py           IEEE 33-bus + 5 tie switches + profiles + criticality
eqosystem/scenarios.py      LHS contingencies (real N-1 de-energization)
eqosystem/candidates.py     overlapping spectral island candidates
eqosystem/hamiltonians.py   H_design / H_island / H_dispatch builders
eqosystem/solvers.py        Dirac-3 cloud path · simulated annealing · exact · polish
eqosystem/pipeline.py       orchestration, hourly metrics, MILP baseline
run_experiments.py          E1 pipeline · E2 optimality · E3 noise · E4 MILP
```
