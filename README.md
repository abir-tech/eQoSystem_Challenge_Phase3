# eQoSystem — Cost Optimization in Resilient Power Grids

**Team:** eQoSystem (Achraf Boussahi, Abir Chekroun, Zakaria Lourghi — ESI-SBA, Algeria)
**Challenge:** QCi Global Industry Challenge 2026, Phase 3 — Energy Infrastructure
**Platform:** QCi Dirac-3 (Entropy Quantum Computing) via qBraid

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/abir-tech/eQoSystem_Challenge_Phase3.git)

A three-stage quantum-optimization pipeline for resilient microgrid design on IEEE
distribution feeders. It (1) selects overlapping island candidates and sizes their DER
portfolios at minimum cost, (2) decides which islands to energize during each N-1
contingency, and (3) dispatches each island over a time-coupled, state-of-charge-aware
horizon. Every stage is a bounded-integer polynomial Hamiltonian expressed in the native
qudit encoding of QCi's Dirac-3 entropy quantum computer — no binary expansion and no
auxiliary variables for the high-degree terms.

**Hardware status, stated precisely.** Two of the three stages have *executed* on real
Dirac-3 hardware: **islanding** (20/20 certified optimal) and **design** (1.064× the
certified mixed-integer optimum). **Dispatch** is within every integer-solver limit —
levels, the 16-level per-variable cap, and the calibrated dynamic-range trigger — but has
**not yet been run on the device**, and that distinction between *legal* and *executed* is
kept everywhere in this repository. Design reaches legality through the trust-region
encoding; dispatch through bounded-sum decomposition (exact, unlike base-16 radix, which
both clips and inflates dynamic range) plus dropping the diesel heat-rate cubic, whose
coefficient sits ~65 dB below the largest and is therefore below the device's own
coefficient resolution in any case.

## Reproduce in one minute

```bash
pip install -r requirements.txt
python run_experiments.py --backend sa
```

This requires **no token and no configuration**, runs entirely classically, and
regenerates every number in the results block below into
`results/results_simulated-annealing.json`. The `sa` backend is the reproducible
classical engine (simulated annealing, exact enumeration, and HiGHS mixed-integer
baselines); judges can verify the headline results without hardware access.

Optional studies (slower, off by default):

```bash
python run_experiments.py --backend sa --vss --voltage-ab --export-ab --scaling --outdir results_extended
```

Run the test suite:

```bash
python -m pytest tests/ -q
```

## Run on Dirac-3 hardware

Always rehearse classically first — free, no token, and it exercises the exact same code
path including every device-limit check:

```bash
python hardware_test.py --backend sa --stage all -n 5
```

Then, on hardware:

```bash
export QCI_TOKEN=<your QCi API token>       # also QCI_API_URL if QCi issued one
python hardware_test.py --backend dirac3 --stage all -n 20 --samples 10
```

`--stage` selects `design`, `islanding`, `dispatch`, or `all`, so each pipeline stage can
be reproduced independently. Every stage is checked against the device's limits *before*
submission and refused rather than sent if out of range — an over-range Hamiltonian is
accepted by the device and answered incorrectly, with no error. Each stage is compared
against a classical reference on the identical instance.

The token is read from the environment at run time and is never written to any file.
Archive the output immediately — filenames are reused:

```bash
cp results/hardware_dirac3_all.json results/hardware_dirac3_all_$(date +%Y%m%d).json
```

<!-- BEGIN GENERATED RESULTS -- edit tools/gen_readme.py, not this block -->

## Headline results

IEEE 69-bus feeder, 50 Latin-Hypercube N-1 contingency scenarios, seed 42, classical backend. Full battery wall-clock 142 s.

| Challenge metric | No-microgrid reference | eQoSystem |
|---|---|---|
| **M1** max fraction of customers unserved in any hour | 100.0% | **23.6%** |
| **M2** critical-infrastructure bus-hours unserved | 397 | **0** |
| Expected unserved energy per contingency | 5292 kWh | **488 kWh** |
| **M3** grid-upgrade capex | 332.5 (certified MILP optimum) | **338.8** (x10 k$) |

The no-microgrid column is a *resilience reference*, not an algorithmic baseline. The algorithmic baselines are below.

### Classical baselines on the identical instances

- **HiGHS MILP**, design stage, same instance: optimum 332.5 in 0.03 s. Our Hamiltonian solution costs **1.019x** the certified optimum.
- **Exhaustive enumeration**, every islanding QUBO: **50/50** solved to certified global optimality (max gap 0.0e+00).
- **HiGHS MILP**, dispatch stage, same instances: over 12 island-scenario dispatch problems scored by the identical hourly simulation, our Hamiltonian sheds 1453 kWh against the MILP's 1472 kWh (**0.987x**) while burning 101260 kWh of diesel against 97780 (**1.036x**). Both reach 0 critical short-hours.
  The unserved ratio below 1.0 is **not** evidence of beating an optimum: the MILP minimises its own fuel-and-service objective, not unserved energy, so shedding 1.3% less while burning 3.6% more diesel is a different point on the same trade-off, and is reported as parity rather than advantage.
  The MILP enforces power balance and the SOC recursion as hard constraints where the Hamiltonian uses penalties, so the comparison is made on physical outcome rather than objective value.
- **Simulated annealing** on the identical Hamiltonians is the reproducible classical engine behind every number here.

All three pipeline stages have a classical baseline on the same instance: design **1.019x** the certified MILP optimum, islanding **exact** (50/50 certified), dispatch **0.987x** on unserved energy at 1.036x the diesel. Design is the only stage with a strict optimality gap; islanding is provably optimal; dispatch is at parity on a two-objective trade rather than dominating or being dominated.

No speedup over classical methods is claimed anywhere in this work.

### Dirac-3 hardware

- **20/20** islanding instances on the ieee69 feeder returned the certified global optimum (hardware energy equals exhaustive-enumeration energy on every one).
- Of the 20 scenarios, **9 required active islanding** (non-zero objective); the remaining 11 were trivial, with "do nothing" optimal. The 20/20 figure should be read with that split in mind.
- 3 variables per instance, 1 sample per job (free-tier behaviour), 405 s total wall-clock including queueing.
- Coefficient dynamic range 0.0-30.9 dB. **5 instances exceeded the nominal 23 dB specification** (26.1, 29.2, 29.4, 30.7, 30.9 dB) and still resolved correctly, consistent with the 30.78 dB operating point reported in published Dirac-3 work.

**Design stage on Dirac-3.** The design Hamiltonian has now also been executed on the device, not merely shown to fit it.

| quantity | Dirac-3 | classical SA |
|---|---|---|
| raw Hamiltonian capex | 324.8 | 329.8 |
| repair units to reach feasibility | 6 | 3 |
| capex after repair | 353.8 | 338.8 |
| ratio vs certified MILP optimum | **1.064x** | 1.019x |

- Submitted at **34.5 dB**, 202 levels, max upper bound 6, degree 3, 5 samples. The returned solution covers every load bus.
- Billed **19 s** of free-tier allocation for this single job (441 -> 422 s), against roughly 3 s for a 3-variable islanding job.
- The device returned a *cheaper but less feasible* raw solution than the classical annealer (324.8 vs 329.8) and needed 6 repair units against 3, ending 4.4% worse after repair. Reported as a measured quality gap, not parity.
- It does, however, resolve a Hamiltonian at 34.5 dB — above the 30.9 dB previously validated — which is direct evidence for the calibrated trigger rather than the nominal specification.

### Coefficient conditioning (E8)

Certified truncation is default-on. A Hamiltonian is rewritten only if its dynamic range exceeds the calibrated **35 dB** trigger *and* the truncation carries a proof that the ground state does not move.

- **0 of 52** Hamiltonians were rewritten. All 50 islanding QUBOs are submitted unmodified.
- The trigger is calibrated from measured hardware behaviour, not the 23 dB nominal specification: on Dirac-3, instances up to **30.9 dB** returned the certified optimum. Applying the nominal reading would have rewritten exactly the **5** highest-dynamic-range instances of that hardware run and nothing else — the strongest evidence in the submission.
- In this classical run the highest islanding dynamic range is 31.1 dB, and 13 of 50 instances exceed the nominal specification.
- `dispatch_mp` (65.0 dB, 289 levels) is **refused** by the hardware guard rather than submitted: 8 variable(s) exceed the 16-level integer cap (worst: 'nc[0]' upper bound 43).

### Physical validation and resources

- **LinDistFlow (E6):** 328/331 energized island-hours electrically feasible, Vmin 0.9871 pu, worst line 128% of thermal rating. ZIP voltage-dependent loads and inverter P-Q limits included.
- **Grid-connected PCC (E7):** 29.9% daily energy-cost saving via storage arbitrage under time-of-use prices (36 variables, 1083 levels).
- **Encoding economy (E5):** native integer encoding uses 50 variables and 197 terms against 104 and 611 for binary compilation of the same problem (2.1x variables, 3.1x terms), measured under an equal-budget solver.
- **Quantum resource accounting:** 51 jobs. Design stage 50 qudits / 197 terms / degree 3 / 34.5 dB, versus 104 binary qubits if expanded. Islanding 3 variables per instance.

### Stochastic-programming value (E10)

Annualized cost in x10 k$/yr, VOLL $10/kWh, capex over 20 yr, 12 events/yr:

| RP | EEV | WS | VSS | EVPI |
|---|---|---|---|---|
| 22.35 | 22.56 | 18.49 | **0.22** | **3.85** |

The design Hamiltonian is scenario-independent, so the here-and-now decision studied is the sizing margin; a scenario-coupled design Hamiltonian is future work. VSS is positive but **assumption-dependent**: across VOLL $2-50/kWh and 4-52 events/yr the cost-optimal margin ranges over {1.00, 1.10, 1.25, 1.40} and VSS ranges 0.0-97.1. It vanishes when outages are cheap, because the mean-value design is then already optimal.

### Voltage-aware islanding A/B (W2)

| metric | voltage-blind | voltage-aware |
|---|---|---|
| M1 | 23.6% | 23.6% |
| M2 | 0 | 0 |
| LinDistFlow-feasible island-hours | 328/331 | 328/331 |
| islanding decisions changed | - | 0 |

**Measured null result, reported as such.** The predicted worst-bus voltage across all candidates is 0.9904 pu against a 0.95 pu band, so the penalty is identically zero and both arms submit the same Hamiltonian. Islands are electrically short and fed from DER hubs sited inside them. The binding physical constraint at this design point is thermal loading, not voltage.

<!-- END GENERATED RESULTS -->

Every figure above is written by `tools/gen_readme.py` from the committed result JSONs.
No result is hand-entered anywhere in this repository. Verify with:

```bash
python tools/gen_readme.py --check
python tools/gen_report.py --check
```

## Repository map

```
eqosystem/grid.py           IEEE 33-bus and 69-bus datasets, topology, profiles, criticality
eqosystem/scenarios.py      Latin-Hypercube N-1 contingencies; flat and scenario-tree modes
eqosystem/candidates.py     overlapping island candidates via spectral clustering
eqosystem/hamiltonians.py   H_design / H_island / H_dispatch / H_dispatch_mp builders
eqosystem/conditioning.py   certified coefficient truncation and hardware-legality guards
eqosystem/radix.py          bounded-sum and base-16 decompositions for the level cap
eqosystem/continuous.py     quasi-continuous solver path and its classical rehearsal
eqosystem/certify.py        exact mixed-integer certification (scales past enumeration)
eqosystem/solvers.py        Dirac-3 cloud paths, simulated annealing, exact, greedy polish
eqosystem/pipeline.py       orchestration, hourly metrics, MILP baselines, VSS/EVPI
eqosystem/lindistflow.py    voltage and thermal validation, ZIP loads, non-convex P-Q
eqosystem/compile_binary.py binary/QUBO compilation used by the encoding study
run_experiments.py          experiment battery E1-E12 and the W2/W7 A/B arms
hardware_test.py            Dirac-1 / Dirac-3 device protocol, per stage
tools/gen_readme.py         regenerates the results block above from results/*.json
tools/gen_report.py         regenerates the LaTeX write-up from results/*.json
tests/                      pytest suite
```

## Approximations and limitations

Stated plainly, because the rubric rewards it and because each one bounds what the results
mean.

**Physics.**
- Power flow is **LinDistFlow**, the linearized DistFlow approximation, not full nonlinear
  AC. Losses are dropped; accurate to roughly 1% on radial feeders of this class.
- Voltage and thermal limits are validated post-hoc, and voltage is additionally encoded
  inside the islanding Hamiltonian as a precomputed linear penalty. At the measured
  operating point that penalty is identically zero, so it changes no decision. The
  constraint that actually binds here is thermal loading.
- Inverter capability uses a piecewise non-convex curve; battery charge efficiency is
  power-dependent in the physics simulation, with the constant-efficiency form retained as
  the planning default because the degree-4 arm measured slightly worse.

**Model scope.**
- We model a **distribution** system. Transmission coupling is not modeled and not faked.
- Cross-island export is implemented and measured but **off by default**, because enabling
  it changes the islanding Hamiltonian and would make the recorded hardware energies
  unreproducible from this code.
- Generator startup/shutdown costs and reserves are not yet in the objective.
- Outage windows are 4–16 h against the challenge's stated 24–72 h horizon.

**Stochastics.**
- The scenario set is a sample, not an enumeration of contingencies. Both readings of the
  challenge's "10–50 per time step" are implemented and reported.
- The design stage is **scenario-independent** — robust by construction rather than a
  scenario-coupled stochastic program. The VSS/EVPI study therefore treats the sizing
  margin as the here-and-now decision, and says so.

**Hardware.**
- The hardware instances are small (3 variables each for islanding); a laptop solves them
  in microseconds. The result establishes correctness on the device, not advantage.
- **No speedup over classical methods is claimed.** At this scale exact classical methods
  win, and we say so.

## Data provenance

All network data are **publicly released IEEE / MATPOWER test feeders**. The 33-bus system
is Baran & Wu (1989); the 69-bus system is extracted from the canonical MATPOWER `case69`
and validated against the literature voltage profile. **No proprietary, utility-sensitive,
or otherwise non-public data is used**, and no result depends on data a judge cannot
access.
