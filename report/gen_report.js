const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ImageRun, ShadingType,
} = require("docx");

const FONT = "Times New Roman";
const SZ = 22; // 11pt
const t = (text, opts = {}) => new TextRun({ text, font: FONT, size: SZ, ...opts });
const p = (children, opts = {}) => new Paragraph({
  children: Array.isArray(children) ? children : [t(children)],
  spacing: { after: 80, line: 252 }, alignment: AlignmentType.JUSTIFIED, ...opts,
});
const h1 = (text) => new Paragraph({
  children: [t(text, { bold: true, size: 24 })],
  spacing: { before: 160, after: 80 }, heading: HeadingLevel.HEADING_1,
});
const noBorder = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const cell = (text, { bold = false, width, shade = false } = {}) => new TableCell({
  width: { size: width, type: WidthType.DXA },
  shading: shade ? { type: ShadingType.CLEAR, fill: "EFEFEF" } : undefined,
  borders: { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder },
  margins: { top: 40, bottom: 40, left: 80, right: 80 },
  children: [new Paragraph({ children: [t(text, { bold, size: 20 })] })],
});

const resultsWidths = [3600, 1900, 1700, 1600];
const resultsTable = new Table({
  width: { size: 8800, type: WidthType.DXA }, columnWidths: resultsWidths,
  rows: [
    new TableRow({ children: ["Challenge metric", "Legacy grid", "eQoSystem", "Change"].map((x, i) => cell(x, { bold: true, width: resultsWidths[i], shade: true })) }),
    new TableRow({ children: ["M1 — max fraction of customers unserved in any hour", "100.0%", "15.8%", "−84.2 pts"].map((x, i) => cell(x, { width: resultsWidths[i] })) }),
    new TableRow({ children: ["M2 — critical-infrastructure bus-hours unserved (20 scenarios)", "174 h", "0 h", "−100%"].map((x, i) => cell(x, { width: resultsWidths[i] })) }),
    new TableRow({ children: ["Expected unserved energy per contingency", "4,877 kWh", "428 kWh", "−91%"].map((x, i) => cell(x, { width: resultsWidths[i] })) }),
    new TableRow({ children: ["M3 — upgrade cost vs certified MILP optimum (HiGHS)", "1.000×", "1.005×", "+0.5%"].map((x, i) => cell(x, { width: resultsWidths[i] })) }),
  ],
});

const e5Widths = [2400, 1500, 1700, 3200];
const e5Table = new Table({
  width: { size: 8800, type: WidthType.DXA }, columnWidths: e5Widths,
  rows: [
    new TableRow({ children: ["Encoding", "Variables", "Poly. terms", "Equal-budget SA outcome"].map((x, i) => cell(x, { bold: true, width: e5Widths[i], shade: true })) }),
    new TableRow({ children: ["Native qudit (Dirac-3)", "50", "196", "P(within 1% of best) = 0.08; 1.0× wall-clock"].map((x, i) => cell(x, { width: e5Widths[i] })) }),
    new TableRow({ children: ["Binary compilation (qubit hardware)", "149", "1,238", "P(within 1% of best) = 0.00; ≈4× wall-clock"].map((x, i) => cell(x, { width: e5Widths[i] })) }),
  ],
});

const refs = [
  "M. E. Baran and F. F. Wu, “Network reconfiguration in distribution systems for loss reduction and load balancing,” IEEE Trans. Power Delivery, 4(2):1401–1407, 1989.",
  "L. Nguyen et al., “Entropy computing: a paradigm for optimization in an open quantum system,” arXiv:2407.04512, 2024.",
  "Quantum Computing Inc., “Dirac-3 User Guide,” v0.0.4, 2025; and eqc-models documentation, quantumcomputinginc.com.",
  "A. Lucas, “Ising formulations of many NP problems,” Frontiers in Physics, 2:5, 2014.",
  "I. G. Rosenberg, “Reduction of bivalent maximization to the quadratic case,” Cahiers du Centre d'Études de Recherche Opérationnelle, 17:71–74, 1975.",
  "M. D. McKay, R. J. Beckman, and W. J. Conover, “A comparison of three methods for selecting values of input variables in the analysis of output from a computer code,” Technometrics, 21(2):239–245, 1979.",
  "T. Ding et al., “A resilient microgrid formation strategy for load restoration considering master-slave distributed generators and topology reconfiguration,” Applied Energy, 199:205–216, 2017.",
  "L. Che and M. Shahidehpour (framework); see also “An exact microgrid formation model for load restoration in resilient distribution systems,” Int. J. Electrical Power & Energy Systems, 116, 2020.",
  "K. P. Nguyen et al., “Stochastic optimal sizing of distributed energy resources for a cost-effective and resilient microgrid,” Energy, 198:117284, 2020.",
  "A. Abbas et al., “Challenges and opportunities in quantum optimization,” Nature Reviews Physics (arXiv:2312.02279), 2024 — Sec. on power-grid applications.",
  "“Qubit-efficient quantum annealing for stochastic unit commitment,” arXiv:2502.15917, 2025.",
  "W. Fu et al., “Coordinated post-disaster restoration for resilient urban distribution systems: a hybrid quantum-classical approach,” Energy, 284:129314, 2023.",
  "A. Zare et al., “A stochastic-robust approach for resilient microgrid investment planning under static and transient islanding security constraints,” arXiv:2007.03149, 2020.",
];

const img = fs.readFileSync("/home/claude/phase3/results/experiments.png");

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: SZ } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, bottom: 1080, left: 1152, right: 1152 },
      },
    },
    children: [
      new Paragraph({
        children: [t("Cost Optimization in Resilient Power Grids on QCi Dirac-3:", { bold: true, size: 28 })],
        alignment: AlignmentType.CENTER, spacing: { after: 40 },
      }),
      new Paragraph({
        children: [t("Two-Stage Stochastic Microgrid Design with Native Degree-3 Integer Hamiltonians", { bold: true, size: 28 })],
        alignment: AlignmentType.CENTER, spacing: { after: 80 },
      }),
      new Paragraph({
        children: [t("Team eQoSystem — [Your Name], Achraf Boussahi, Abir Chekroun, Zakaria Lourghi (ESI-SBA, Algeria) · QCi Global Industry Challenge 2026, Phase 3 — Energy Infrastructure", { italics: true, size: 20 })],
        alignment: AlignmentType.CENTER, spacing: { after: 160 },
      }),

      h1("1. Problem and model"),
      p("A radial distribution feeder loses every bus downstream of a failed line. Pre-installing distributed energy resources (DERs) and remotely operated tie switches lets sections island: disconnect from the dead feeder and self-supply until repair. We work on the IEEE 33-bus feeder of Baran and Wu [1] (3,715 kW peak, five standard tie switches), with four critical buses {8, 14, 24, 30} representing hospital-class loads, hourly load and solar profiles, and customer counts proportional to bus load. Contingency uncertainty is sampled by Latin Hypercube design [6] over five dimensions — failed line, load factor (0.7–1.3), renewable availability (0.2–1.0), outage start hour, and duration (4–12 h) — producing 20 scenarios in which the failed line genuinely de-energizes its downstream component. The challenge metrics are M1, the maximum fraction of customers unserved in any hour of any contingency; M2, total critical-infrastructure bus-hours unserved across contingencies; and M3, the capital cost of upgrades. The decision problem is two-stage stochastic [9, 13]: invest once (islands, DER portfolios, tie switches), then operate through every contingency (islanding and dispatch decisions)."),

      h1("2. Three-stage Hamiltonian formulation"),
      p("Every stage is a polynomial over bounded integer variables — the native input of QCi's Dirac-3 entropy quantum computer [2, 3], which supports qudit (integer) encodings and polynomial interactions up to degree five without auxiliary variables. Candidate islands come from spectral clustering of the feeder at k = 3, 4, 5, giving nine overlapping candidates; overlap is permitted by the challenge and is what couples the islanding stage into a genuine combinatorial problem."),
      p([t("Stage 1 — design H", {}), t("des", { superScript: true }), t(" (50 qudits, 196 terms, degree 3, 41 dB coefficient range). Binary b\u2091 selects island c; integer n\u2091\u2096 counts units of asset k ∈ {PV, BESS, DG} with unit capacities, firmness factors and costs; binary y\u2097 deploys tie switch l; integer s\u2091 is a sizing slack. The objective sums capital costs plus penalties: a near-partition coverage penalty (every bus in ≥1 built island, doubled on critical buses) and the gated capacity constraint b\u2091·((F\u2091 − D\u2091 − s\u2091)/C\u1D63)², where F\u2091 is firm capacity and D\u2091 the design demand — a native degree-3 term that qubit hardware would need quadratization [5] to express. Design demand includes a 1.25 sizing margin over the load-factor tail; a consequence we exploit in §4 is that critical service then survives load factors up to 1.5 by construction. DER units are sited at two hubs per island (highest-load bus, plus the bus maximizing load × distance from it), so no single internal line failure can sever all generation.")]),
      p([t("Stage 2 — islanding H"), t("isl", {}), t("(s) (≤9 qudits, degree 2, one QUBO per scenario). Binary z\u2091 energizes built island c; its value is the restorable de-energized load — computed on the post-fault topology including closed intra-island ties, from either DER hub — weighted ×5 on critical buses and scaled by a capability factor φ = min(1, capacity/need) under the scenario's load and renewable factors. Pairwise couplings penalize overlapping islands claiming the same de-energized buses; switching costs and stress terms complete the QUBO.")]),
      p([t("Stage 3 — dispatch H"), t("dsp", {}), t("(s, c) (2–4 qudits after compressing fixed-at-zero variables, degree 3). Integer setpoints in 10 kW steps for PV, BESS, and DG plus a served non-critical level l\u2099\u2091. A stiff quadratic balance penalty ties claimed service to physical supply; the diesel heat-rate curve enters as a cubic term — again native on Dirac-3 — calibrated so its full-load marginal cost is a fixed fraction of the serve reward. Metric accounting is supply-capped (service never exceeds generation, critical loads first), hourly across each outage window.")]),

      h1("3. Hardware-aware engineering"),
      p("Dirac-3 is an analog photonic solver: coefficients are realized with finite precision, so a Hamiltonian whose largest and smallest coefficients differ by many orders of magnitude cannot be resolved. Experiment E3 quantifies this: injecting 1% relative coefficient noise, the probability of recovering the true optimum falls from 0.87 at dynamic range 10⁰ to ≈0 beyond 10⁴ (Fig. 1, bottom-left). All our Hamiltonians are conditioned to 19–57 dB by demand-relative normalization, per-problem calibration of the cubic term, unit-scaled capacity violations, and compression of zero-bound variables; our Phase-2 prototype sat at ≈80 dB, which E3 shows is unusable. Execution is hybrid: greedy warm start, hardware sampling (num_samples = 10, relaxation_schedule = 2, via eqc-models' Dirac3IntegerCloudSolver), and classical greedy polish of every returned sample. Per-island decomposition keeps every hardware job at ≤50 variables, ≈50 jobs total."),

      h1("4. Experiments and results"),
      p("All results are deterministic (fixed seeds) and reproduce in under one minute with `python run_experiments.py --backend sa`; the identical pipeline runs on hardware with `--backend dirac3`. Table 1 gives the headline metrics against the no-islanding baseline."),
      resultsTable,
      new Paragraph({ children: [t("Table 1 — Challenge metrics, 20 LHS contingencies, IEEE 33-bus, seed 42.", { italics: true, size: 18 })], spacing: { before: 40, after: 120 } }),
      p("Certification (E2): every islanding QUBO is cross-checked against exhaustive enumeration — 20/20 solved to certified global optimality (gap 0.0); the check extends to 40/40 on out-of-design scenarios, where M2 remains 0 (baseline 313 h). Classical baseline (E4): the design stage is also solved exactly as a MILP with HiGHS; our Hamiltonian solution costs 1.005× the certified optimum. Electrical feasibility (E6): every energized island-hour is validated post-hoc with LinDistFlow — 273/273 pass the ±5% voltage band and line thermal ratings (V_min = 0.986 pu; worst line at 42% of rating; 517/517 at 40 scenarios; 481/481 under stress). The checker reproduces the canonical intact-feeder result (V_min ≈ 0.916 pu at bus 18, violating the band at 19 buses at peak), confirming both its correctness and that islanded DER operation measurably improves the feeder's known voltage weakness. Hourly evaluation simulates battery state-of-charge depletion (200 kWh per 50 kW unit): each hour, service is PV(h) + DG + min(BESS power, remaining energy), critical loads first — hence the honest reduction of headline service in long outages relative to a static worst-hour model. Robustness of the zero: M2 = 0 is not a tuning artifact but an arithmetic consequence of the design margin — sizing at 1.25·(L_crit + 0.4·L_rest) covers 1.5·L_crit whenever L_rest ≥ 0.5·L_crit, which holds at every bus. A beyond-design-basis stress test (load factor to 1.5, outages to 20 h, weakened renewables; `--stress`) confirms the intended failure ordering: non-critical service degrades sharply (M1 rises to 47.9%, unserved energy 4,875 kWh) while critical service is preserved (M2 = 0)."),
      p("Encoding advantage (E5, Table 2 and Fig. 1 bottom-right): compiling the same design Hamiltonian into binary variables — as any qubit-based device requires — triples the variable count, multiplies terms by 6.3×, and at equal optimization budget with identical seeds never reaches within 1% of the best-known solution, at ≈4× the wall-clock per attempt. The compilation is verified equivalent at 30 random points, so this isolates the encoding cost itself. The degree-3 terms would additionally require Rosenberg quadratization [5] on QUBO hardware, adding auxiliary variables and large penalties that worsen exactly the dynamic-range failure mode of E3."),
      e5Table,
      new Paragraph({ children: [t("Table 2 — E5: the same physics in two encodings; 12 trials × 4,000 iterations, same seeds.", { italics: true, size: 18 })], spacing: { before: 40, after: 120 } }),
      new Paragraph({
        children: [new ImageRun({ type: "png", data: img, transformation: { width: 620, height: 421 } })],
        alignment: AlignmentType.CENTER, spacing: { after: 40 },
      }),
      new Paragraph({ children: [t("Figure 1 — Top: M1 and M2 per scenario vs baseline. Bottom-left: E3 noise study justifying coefficient conditioning (green: this framework, 41 dB; red: Phase-2 prototype, 80 dB). Bottom-right: E5 relative cost of forgoing native integer encoding.", { italics: true, size: 18 })], spacing: { after: 120 } }),

      h1("5. Dirac-3 execution"),
      p("[TO COMPLETE AFTER THE qBRAID RUN — replace bracketed values.] The pipeline submitted [50] jobs to Dirac-3 via qBraid: 1 design job (50 qudits, degree 3), 20 islanding jobs (≤9 qudits), and [29] dispatch jobs (2–4 qudits), num_samples = 10 each ([500] samples total). Median job wall-clock was [X] s ([Y] s total). Hardware solutions matched the certified islanding optima in [Z]/20 problems before polish and [Z']/20 after greedy polish; the design solution reached [ratio] of the MILP optimum. Job IDs are listed in results/results_dirac3.json for verification."),

      h1("6. Advantage rationale, honestly stated"),
      p("We do not claim a quantum speedup over HiGHS on the design stage: that subproblem is linear, and mature MILP solvers dispatch it in milliseconds; we claim certified parity (1.005×). The measured case for the quantum-native approach is representational: one polynomial family expresses all three coupled stages — mixed binary/integer decisions, overlapping-island couplings, a gated degree-3 capacity constraint, and a cubic heat-rate curve — with 3× fewer variables and 6.3× fewer terms than any binary encoding (E5), inside the coefficient-range regime that analog hardware can resolve (E3). Prior quantum power-grid work concentrates on unit commitment via QUBO with auxiliary-variable overhead [10, 11, 12]; native qudit, degree-3 encodings remove that overhead by construction [2]."),

      h1("7. Limitations and validity"),
      p("Voltage and thermal limits are validated post-hoc via LinDistFlow (E6) rather than enforced inside the Hamiltonians; full AC power flow, protection coordination, and islanding transients remain out of scope. Battery state of charge is simulated hourly in the evaluation, while the dispatch Hamiltonian optimizes the worst hour (inter-temporal dispatch coupling is future work). Load and solar profiles are representative, not measured. Installed capacity is shared between an island's two DER hubs in proportion to post-fault fragment demand (stated assumption). Cross-island export over tie lines is not modeled. All numbers in this report regenerate from the submitted code in under one minute."),

      h1("References"),
      ...refs.map((r, i) => new Paragraph({
        children: [t(`[${i + 1}] ${r}`, { size: 18 })],
        spacing: { after: 30 }, alignment: AlignmentType.LEFT,
      })),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("/home/claude/phase3/report/eQoSystem_Phase3_Report.docx", buf);
  console.log("written");
});
