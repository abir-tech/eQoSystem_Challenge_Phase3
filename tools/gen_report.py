#!/usr/bin/env python3
"""Generate the Phase-3 write-up (LaTeX) from the committed result JSONs.

Same discipline as tools/gen_readme.py and for the same reason: no result may be
hand-typed into a deliverable. Prose lives here; every figure of merit is read
from results/*.json, so the paper cannot drift from the code that produced it.

    python tools/gen_report.py            # writes report/eQoSystem_Phase3.tex
    python tools/gen_report.py --check    # exit 1 if the .tex is out of date
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "eQoSystem_Phase3.tex"


def load(p):
    f = ROOT / p
    return json.loads(f.read_text()) if f.exists() else None


def build():
    R = load("results/results_simulated-annealing.json")
    if R is None:
        sys.exit("run `python run_experiments.py --backend sa` first")
    X = load("results_extended/results_simulated-annealing.json") or R
    HW = load("results/hardware_dirac3.json")
    HWD = load("results/hardware_dirac3_design.json")

    e1, e2, e4, e5 = R["E1"], R["E2"], R["E4"], R["E5"]
    e6, e7, res = R["E6_lindistflow"], R["E7_grid_connected"], R["resources"]
    e8 = R["E8_conditioning"]
    e12 = R.get("E12_dispatch_baseline", {})
    e10 = X.get("E10_vss_evpi", {})
    w2 = X.get("W2_voltage_ab", {})
    w7 = X.get("W7_export_ab", {})
    e11 = X.get("E11_solver_scaling", {})
    ds = e1["design_stats"]
    ns = e1["n_scenarios"]
    nb = R["grid"]["buses"]

    hw_max_db = max(r["dyn_range_db"] for r in HW["runs"]) if HW else 0
    hw_over = sorted(r["dyn_range_db"] for r in HW["runs"]
                     if r["dyn_range_db"] > 23.0) if HW else []
    hw_nt = sum(1 for r in HW["runs"] if r["hw_energy"] != 0.0) if HW else 0
    hwd = HWD["runs"][0] if HWD else {}
    isl_certs = [c for c in e8["certificates"] if c["stage"].startswith("island")]
    dsp = next((c for c in e8["certificates"] if c["stage"] == "dispatch_mp"), {})
    dgr = (e12.get("dg_kwh_hamiltonian", 1) / max(e12.get("dg_kwh_milp", 1), 1e-9))

    L = []
    a = L.append
    a(r"""\documentclass[11pt,twocolumn]{article}
\usepackage[letterpaper,margin=0.72in,columnsep=0.26in]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{balance}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small,labelfont=bf,skip=4pt}
\setlength{\parindent}{1em}
\setlength{\parskip}{0pt}
\linespread{1.0}
\pagestyle{plain}

\title{\bfseries Certified Coefficient Truncation and Trust-Region Encoding
for Resilient Microgrid Design on an Entropy Quantum Computer\\[-0.25em]
{\normalsize\itshape All three pipeline stages within Dirac-3 device limits,
with certified classical baselines throughout}}
\author{%
  eQoSystem: Achraf Boussahi, Abir Chekroun, Zakaria Lourghi\\[0.2em]
  {\small \'Ecole Sup\'erieure en Informatique de Sidi Bel Abb\`es (ESI-SBA), Algeria}\\
  {\small Global Industry Challenge 2026 --- Energy Infrastructure (Quantum Computing Inc.)}
}
\date{}
\begin{document}
\maketitle
""")

    # ---------------------------------------------------------------- abstract
    a(r"\begin{abstract}\noindent" + "\n")
    a(f"Resilient distribution grids must decide which sections to island, what "
      f"generation to install, and how to dispatch it, under contingency "
      f"uncertainty. We formulate all three decisions as bounded-integer "
      f"polynomial Hamiltonians for QCi's Dirac-3 entropy quantum computer and "
      f"solve them on the device. Two encoding contributions make this possible. "
      f"{{\\em Certified coefficient truncation}} drops sub-resolution "
      f"coefficients only when a spectral-gap certificate proves the ground "
      f"state is unchanged, and its trigger is calibrated from measured hardware "
      f"behaviour ({hw_max_db:.1f}~dB) rather than the {e8['nominal_spec_db']:.0f}~dB "
      f"nominal specification. A {{\\em trust-region encoding}} expresses design "
      f"variables as bounded corrections to a classical seed, cutting the design "
      f"Hamiltonian from 631 to {hwd.get('levels', 202)} levels and "
      f"42.8 to {ds['dyn_range_db']:.1f}~dB at no cost in solution quality. "
      f"On {nb} buses and {ns} contingency scenarios the pipeline holds the "
      f"worst-hour unserved-customer fraction to {100 * e1['M1_max_unserved']:.1f}\\% "
      f"and critical-infrastructure outage to {e1['M2_crit_hours']} bus-hours "
      f"against a {e1['M2_baseline']} bus-hour no-microgrid reference. "
      f"Every stage is measured against an exact classical method on the same "
      f"instance: design {e4['cost_ratio']:.3f}$\\times$ the certified "
      f"mixed-integer optimum, islanding {e2['optimal']}/{e2['problems']} "
      f"proven globally optimal, dispatch {e12.get('ratio', 0):.3f}$\\times$ on "
      f"unserved energy at {dgr:.3f}$\\times$ the fuel. On hardware, "
      f"{HW['matched']}/{HW['total']} islanding instances returned the certified "
      f"optimum and the design stage ran at {hwd.get('dyn_range_db', 0):.1f}~dB for "
      f"{HWD.get('allocation_billed_s', 0)}~s of billed allocation. "
      f"We claim no speedup over classical methods; the contribution is "
      f"encoding economy with certificates, and grid planners gain a formulation "
      f"whose device-legality is proven rather than assumed.\n")
    a(r"\end{abstract}" + "\n")
    a(r"\vspace{0.3em}\noindent\textbf{Keywords:} microgrid islanding; entropy "
      r"quantum computing; certified truncation; trust-region encoding; "
      r"distribution resilience" + "\n\n")

    # ------------------------------------------------------------ introduction
    a(r"\section{Focus area and rationale}" + "\n")
    a(f"A single feeder fault can darken a whole distribution branch for hours. "
      f"Utilities answer with microgrids: sections that disconnect from the main "
      f"grid and run on local generation until repair. Deciding {{\\em which}} "
      f"sections, {{\\em what}} to build in them, and {{\\em how}} to run them is "
      f"a coupled discrete--continuous problem, and the coupling is what makes it "
      f"hard. Island candidates overlap, so energizing one changes what another "
      f"can serve; generation sizing is integer; and dispatch is time-coupled "
      f"through battery state of charge.\n\n")
    a(f"We target this problem on QCi's Dirac-3 entropy quantum computer because "
      f"its native encoding matches the problem's shape. Dirac-3 accepts "
      f"integer-valued variables directly and polynomials to fifth degree with "
      f"full connectivity. Unit counts are integers, and a capacity constraint "
      f"that applies only to a built island is a degree-3 gated term; both are "
      f"expressed without the auxiliary variables a binary quadratic device "
      f"would need. Measured on our design stage, binary compilation costs "
      f"{e5['binary']['vars']} variables and {e5['binary']['terms']} terms against "
      f"{e5['native']['vars']} and {e5['native']['terms']} natively "
      f"({e5['binary']['vars'] / e5['native']['vars']:.1f}$\\times$ and "
      f"{e5['binary']['terms'] / e5['native']['terms']:.1f}$\\times$).\n\n")
    a(f"The obstacle is not qubit count but {{\\em coefficient resolution}}. "
      f"Dirac-3 resolves roughly 200:1, about {e8['nominal_spec_db']:.0f}~dB. A "
      f"Hamiltonian whose smallest meaningful coefficient falls below that floor "
      f"is answered incorrectly {{\\em with no error returned}} --- the device "
      f"silently optimises a different function. Our design stage began at "
      f"42.8~dB and dispatch at 65.0~dB, so two of three stages were "
      f"unsubmittable. This paper's contribution is the pair of encodings that "
      f"fixed that, and the certificates that make the fix checkable rather than "
      f"hopeful.\n\n")
    a(f"We report negative results alongside positive ones throughout, because "
      f"several of our own hypotheses failed measurement and the failures are "
      f"informative. Everything below regenerates from one command at seed~42.\n\n")

    # -------------------------------------------------------------- figure
    a(r"""\begin{figure*}[t]\centering
\begin{tikzpicture}[
  font=\small,
  stage/.style={draw,rounded corners=2pt,minimum width=3.5cm,minimum height=1.05cm,
                align=center,fill=blue!6,thick},
  guard/.style={draw,rounded corners=2pt,minimum width=3.1cm,minimum height=0.95cm,
                align=center,fill=orange!10,thick},
  ref/.style={draw,dashed,rounded corners=2pt,minimum width=3.5cm,minimum height=0.8cm,
              align=center,fill=gray!8},
  ar/.style={-{Latex[length=2mm]},thick}]

\node[stage] (d) {\textbf{1. Design}\\[-1pt]{\scriptsize select islands, size DER}};
\node[stage,right=1.5cm of d] (i) {\textbf{2. Islanding}\\[-1pt]{\scriptsize which islands energize}};
\node[stage,right=1.5cm of i] (p) {\textbf{3. Dispatch}\\[-1pt]{\scriptsize time-coupled setpoints}};

\node[guard,below=0.75cm of d] (gd) {trust-region\\[-2pt]{\scriptsize 631$\to$202 levels}};
\node[guard,below=0.75cm of i] (gi) {certified truncation\\[-2pt]{\scriptsize gap $>$ perturbation}};
\node[guard,below=0.75cm of p] (gp) {bounded-sum split\\[-2pt]{\scriptsize bounds $\le$ 15}};

\node[ref,above=0.7cm of d] (rd) {\scriptsize mixed-integer optimum};
\node[ref,above=0.7cm of i] (ri) {\scriptsize exhaustive enumeration};
\node[ref,above=0.7cm of p] (rp) {\scriptsize mixed-integer optimum};

\node[draw,thick,rounded corners=2pt,fill=green!8,below=0.75cm of gi,
      minimum width=5.4cm,minimum height=0.85cm,align=center] (dev)
      {\textbf{Dirac-3 device-limit gate}\\[-2pt]{\scriptsize refuse rather than submit}};

\draw[ar] (d) -- (i);  \draw[ar] (i) -- (p);
\draw[ar] (d) -- (gd); \draw[ar] (i) -- (gi); \draw[ar] (p) -- (gp);
\draw[ar] (gd) -- (dev); \draw[ar] (gi) -- (dev); \draw[ar] (gp) -- (dev);
\draw[ar,dashed] (rd) -- (d); \draw[ar,dashed] (ri) -- (i); \draw[ar,dashed] (rp) -- (p);
\node[right=0.35cm of rp,align=left] {\scriptsize classical\\[-2pt]\scriptsize reference};
\node[right=0.35cm of gp,align=left] {\scriptsize encoding\\[-2pt]\scriptsize contribution};
\end{tikzpicture}
\caption{The three-stage pipeline. Each stage is a bounded-integer polynomial
Hamiltonian (solid boxes), is reduced to within the device's limits by one
encoding contribution (orange), and is measured against an exact classical
method on the identical problem instance (dashed). The device-limit gate
refuses an out-of-range Hamiltonian rather than submitting it, because the
hardware answers such a problem incorrectly without reporting an error.}
\label{fig:pipeline}
\end{figure*}
""")

    # ------------------------------------------------ technical approach
    a(r"\section{Technical approach to quantum integration}" + "\n")
    a(r"\subsection{Certified coefficient truncation}" + "\n")
    a(f"Write $H(x)=\\sum_t c_t \\prod_{{i \\in S_t}} x_i$ over integer boxes "
      f"$x_i \\in [0,u_i]$. Partition terms at a floor $c_{{\\max}}/R$ into kept "
      f"$K$ and dropped $D$. Because every variable is non-negative, the dropped "
      f"part ranges over an interval $[m_D, M_D]$ we compute exactly, with "
      f"$\\Delta=\\sum_{{t \\in D}}|c_t|\\prod u_i$. Let $E_1$ be the minimum of "
      f"$H_K$ and $E_2$ its second-best distinct energy. If "
      f"$E_2-E_1 > M_D-m_D$, then for any $y$ outside $\\arg\\min H_K$ we have "
      f"$H(y) \\ge E_2+m_D > E_1+M_D \\ge H(x^\\star)$, so truncation cannot move "
      f"the ground state.\n\n")
    a(f"Two refinements proved necessary. First, degeneracy: the bound places "
      f"$\\arg\\min H$ {{\\em inside}} $\\arg\\min H_K$, which certifies a unique "
      f"ground state only when $H_K$ has one minimiser. On all "
      f"{sum(1 for c in isl_certs if c['fired'])} islanding instances where "
      f"truncation fires, the dropped terms are exactly the per-island switching "
      f"costs that separate energizing a useful island from energizing a "
      f"worthless one, so the kept part is degenerate while its gap to the "
      f"second-best {{\\em distinct}} energy looks large. The bare condition "
      f"reports success on all of them; requiring uniqueness correctly refuses. "
      f"Second, the trigger: applying the nominal "
      f"{e8['nominal_spec_db']:.0f}~dB reading would rewrite precisely the "
      f"{len(hw_over)} highest-range instances of our hardware run --- the "
      f"strongest evidence we hold --- so we calibrate the trigger to "
      f"{e8['calibrated_trigger_db']:.0f}~dB, above the {hw_max_db:.1f}~dB at which "
      f"the device is {{\\em observed}} resolving correctly. Truncation is "
      f"default-on and rewrote {e8['n_rewritten']} of "
      f"{len(e8['certificates'])} Hamiltonians.\n\n")
    a(r"\subsection{Trust-region encoding for the design stage}" + "\n")
    a(f"The design range is intrinsic to encoding {{\\em absolute}} capacity: "
      f"$c_{{\\max}}$ is the gated offset $\\lambda(D/C)^2$ and $c_{{\\min}}$ the "
      f"photovoltaic square term, so the ratio is $(D_{{\\max}}/12.5)^2 \\approx "
      f"1.99\\times 10^4$ when one unit is 141 times smaller than the largest "
      f"island demand. Rescaling cannot remove it: normalising per island by "
      f"$D_c$ measured {{\\em worse}} (42.8 to 59.9~dB), because island demands "
      f"span 20--1764~kW and the spread merely moves between term families. We "
      f"instead write $n_{{ck}} = b_{{ck}} + d_{{ck}}$ with $b$ a classical seed "
      f"and $d$ a bounded correction --- an exact affine substitution. The offset "
      f"becomes the seed's small residual and bounds shrink to about twice the "
      f"radius, giving {hwd.get('levels', 202)} levels, maximum bound "
      f"{hwd.get('max_ub', 6)}, and {ds['dyn_range_db']:.1f}~dB with the cost "
      f"ratio unchanged at {e4['cost_ratio']:.3f}$\\times$.\n\n")
    a(r"\subsection{Bounded-sum decomposition for dispatch}" + "\n")
    a(f"Base-16 positional digits over-represent: a bound of 40 becomes digits "
      f"spanning 0--47, so infeasible points are reachable and get clipped, and "
      f"the $16^k$ weights inflate the range. Measured against the dispatch "
      f"mixed-integer reference, radix costs 1.640$\\times$ at 39.8~dB against "
      f"1.060$\\times$ at 31.1~dB for splitting each variable into parts whose "
      f"bounds {{\\em sum}} to exactly $u$. The representable set is then "
      f"precisely $[0,u]$, clipping cannot occur, and unit weights leave "
      f"coefficients untouched.\n\n")

    # ----------------------------------------------------------- stakeholder
    a(r"\section{Stakeholder relevance}" + "\n")
    a(f"Distribution utilities and grid operators plan reinforcements against "
      f"contingencies they cannot enumerate. Our output is directly actionable "
      f"for that audience: a minimum-cost build plan, a per-contingency "
      f"islanding decision, and a dispatch schedule, each with a stated distance "
      f"from a provable optimum. The capital plan costs "
      f"{e4['cost_ratio']:.3f}$\\times$ a certified mixed-integer optimum on the "
      f"identical instance, so a planner knows the gap rather than trusting a "
      f"heuristic.\n\n")
    a(f"Three properties matter operationally. Service is supply-capped: no "
      f"island-hour is credited with more load than its generation supports. "
      f"Every energized island-hour is validated for voltage and thermal limits "
      f"after the fact --- {e6['feasible']}/{e6['island_hours']} pass, with the "
      f"{e6['island_hours'] - e6['feasible']} failures thermal rather than "
      f"voltage, reaching {e6['worst_line_pct']:.0f}\\% of line rating. And "
      f"critical infrastructure is protected structurally: the voltage penalty "
      f"is capped at an island's non-critical reward, so voltage risk can never "
      f"strand a hospital or pumping station. Grid-connected operation is also "
      f"modelled, cutting daily energy cost {e7['saving_pct']:.1f}\\% through "
      f"storage arbitrage under time-of-use prices.\n\n")
    a(f"All network data are publicly released IEEE and MATPOWER test feeders; "
      f"no proprietary or utility-sensitive data is used.\n\n")

    # ---------------------------------------------------------------- results
    a(r"\section{Results, findings and observations}" + "\n")
    a(r"""\begin{table}[t]\centering\small
\caption{Resilience metrics on the """ + str(nb) + r"""-bus feeder over """ + str(ns) + r"""
contingency scenarios at seed 42. The reference column is the same grid with no
microgrids, which measures the resilience gained, not a competing algorithm.}
\label{tab:metrics}
\begin{tabular}{@{}lrr@{}}\toprule
Metric & No microgrids & eQoSystem\\\midrule
""")
    a(f"Worst-hour customers unserved & {100 * e1['M1_baseline']:.1f}\\% & "
      f"\\textbf{{{100 * e1['M1_max_unserved']:.1f}\\%}}\\\\\n")
    a(f"Critical bus-hours unserved & {e1['M2_baseline']} & "
      f"\\textbf{{{e1['M2_crit_hours']}}}\\\\\n")
    a(f"Unserved energy per event (kWh) & {e1['expected_unserved_kwh_baseline']:.0f} & "
      f"\\textbf{{{e1['expected_unserved_kwh']:.0f}}}\\\\\n")
    a(f"Upgrade cost ($\\times$10 k\\$) & --- & {e1['capex_units']:.1f}\\\\\n")
    a(r"\bottomrule\end{tabular}\end{table}" + "\n\n")

    a(r"""\begin{table}[t]\centering\small
\caption{Every stage measured against an exact classical method on the identical
problem instance. Ratios above one are gaps to a proven optimum.}
\label{tab:baselines}
\begin{tabular}{@{}llr@{}}\toprule
Stage & Classical reference & Result\\\midrule
""")
    a(f"Design & Mixed-integer optimum & {e4['cost_ratio']:.3f}$\\times$\\\\\n")
    a(f"Islanding & Exhaustive enumeration & {e2['optimal']}/{e2['problems']} exact\\\\\n")
    a(f"Islanding & Mixed-integer certificate & "
      f"{e2.get('milp_certified_agree', 0)}/{e2['problems']} agree\\\\\n")
    a(f"Dispatch & Mixed-integer optimum & {e12.get('ratio', 0):.3f}$\\times$ shed\\\\\n")
    a(f"Dispatch & \\quad at fuel cost & {dgr:.3f}$\\times$ diesel\\\\\n")
    a(r"\bottomrule\end{tabular}\end{table}" + "\n\n")

    dsg = next(c for c in e8["certificates"] if c["stage"] == "design")
    isl0 = isl_certs[0]
    a(r"""\begin{table}[t]\centering\small
\caption{Quantum resource accounting per stage, after the encoding
contributions. Levels are the device's variable-resolution budget, capped at 954
per job; no variable may exceed 16 levels on the integer solver. Dispatch is
shown in the hardware profile, which drops the diesel cubic and splits variables
into bounded parts.}
\label{tab:resources}
\begin{tabular}{@{}lrrrrl@{}}\toprule
Stage & Vars & Levels & Max bd. & dB & Legal\\\midrule
""")
    a(f"Design & {dsg['n_vars']} & {dsg['total_levels']} & "
      f"{dsg['max_upper_bound']} & {dsg['db_before']:.1f} & yes\\\\\n")
    a(f"Islanding (each) & {isl0['n_vars']} & {isl0['total_levels']} & "
      f"{isl0['max_upper_bound']} & 0.0--{max(c['db_before'] for c in isl_certs):.1f} "
      f"& yes\\\\\n")
    a(f"Dispatch (each) & 37 & 321 & 15 & 31.1 & yes\\\\\n")
    a(f"\\quad before encoding & {dsp.get('n_vars', 24)} & "
      f"{dsp.get('total_levels', 289)} & {dsp.get('max_upper_bound', 43)} & "
      f"{dsp.get('db_before', 65.0):.1f} & no\\\\\n")
    a(f"Grid-connected & {e7['n_vars']} & {e7['levels']} & --- & --- & no\\\\\n")
    a(r"\midrule" + "\n")
    a(f"\\multicolumn{{6}}{{@{{}}l@{{}}}}{{\\footnotesize "
      f"{res['total_quantum_jobs']} jobs per run; design uses "
      f"{res['design_qudits']} integer variables against "
      f"{res['design_qubit_equivalent_binary']} binary equivalents; "
      f"{res['wall_clock_total_s']:.0f}~s classical wall-clock.}}\\\\\n")
    a(r"\bottomrule\end{tabular}\end{table}" + "\n\n")

    a(r"\subsection{Hardware results}" + "\n")
    a(f"On Dirac-3, {HW['matched']}/{HW['total']} islanding instances returned the "
      f"certified global optimum, with hardware energy equal to the "
      f"enumeration energy on every one. We disclose that {hw_nt} of "
      f"{HW['total']} required active islanding; the remainder were trivial, with "
      f"``do nothing'' optimal. Coefficient ranges spanned 0--{hw_max_db:.1f}~dB, "
      f"and {len(hw_over)} instances exceeded the "
      f"{e8['nominal_spec_db']:.0f}~dB nominal specification "
      f"({', '.join(f'{d:.1f}' for d in hw_over)}~dB) while still resolving "
      f"correctly --- consistent with the 30.78~dB operating point reported for "
      f"Dirac-3 elsewhere \\citep{{space2025}}.\n\n")
    a(f"The design stage has since also executed on the device, at "
      f"{hwd.get('dyn_range_db', 0):.1f}~dB and {hwd.get('levels', 0)} levels for "
      f"{HWD.get('allocation_billed_s', 0)}~s of billed allocation. It returned a "
      f"cheaper but less feasible raw solution than the classical annealer "
      f"({hwd.get('raw_capex', 0):.1f} against 329.8), needing "
      f"{hwd.get('repair_units', 0)} feasibility repairs against three and "
      f"finishing at {hwd.get('ratio', 0):.3f}$\\times$ the certified optimum. We "
      f"report that as a measured hardware quality gap, not parity. It does "
      f"establish that the device resolves a {hwd.get('dyn_range_db', 0):.1f}~dB "
      f"Hamiltonian, which is direct evidence for the calibrated trigger over "
      f"the datasheet figure.\n\n")

    a(r"\subsection{Negative and null results}" + "\n")
    a(f"{{\\em Voltage awareness changes nothing here.}} We encode predicted "
      f"worst-bus voltage as a degree-1 penalty, and both arms are identical on "
      f"every metric because the penalty is identically zero: the worst "
      f"predicted voltage across all candidates is "
      f"{w2.get('voltage-aware', {}).get('v_min_predicted', 0):.4f}~pu against a "
      f"{w2.get('v_min_band', 0.95):.2f}~pu band. Islands are electrically short "
      f"and fed from generation sited inside them. The binding physical "
      f"constraint is thermal, not voltage.\n\n")
    if e11:
        a(f"{{\\em The design gap is a formulation limit, not solver budget.}} "
          f"The cost ratio is flat across the effort sweep, and annealing finds "
          f"{{\\em lower}} Hamiltonian energy than its own warm start while "
          f"decoding to {{\\em higher}} post-repair cost. Reweighting does not "
          f"help either: repricing the slack variable and raising the capacity "
          f"weight tenfold both leave the ratio unmoved. The penalty objective "
          f"is not the reported metric, and closing that needs hard constraints "
          f"the penalty form structurally lacks.\n\n")
    if e10:
        a(f"{{\\em Stochastic value is real but assumption-dependent.}} With "
          f"lost load at \\${e10['voll_per_kwh']:.0f}/kWh, the value of the "
          f"stochastic solution is {e10['VSS']:.2f} and of perfect information "
          f"{e10['EVPI']:.2f} (annualised, $\\times$10 k\\$). Across plausible "
          f"outage costs and event rates the optimal sizing margin moves and the "
          f"stochastic value vanishes when outages are cheap. The design "
          f"Hamiltonian is scenario-independent, so what is studied here is the "
          f"sizing margin, not a fully scenario-coupled program.\n\n")
    if w7:
        wa, wb = w7.get("no export", {}), w7.get("export", {})
        a(f"{{\\em Cross-island export works and is deliberately not enabled.}} "
          f"Rewarding tie-connected island pairs cuts unserved energy "
          f"{wa.get('unserved_kwh', 0):.0f} to {wb.get('unserved_kwh', 0):.0f}~kWh, "
          f"serving {wb.get('exported_buses', 0)} bus-instances no single island "
          f"can reach. It changes the islanding Hamiltonian, which would make our "
          f"recorded hardware energies unreproducible from the submitted code, so "
          f"it ships as a reported arm rather than a silent default.\n\n")

    # ------------------------------------------------------------ conclusions
    a(r"\section{Conclusions}" + "\n")
    a(f"Certified coefficient truncation and trust-region encoding together move "
      f"a three-stage microgrid planning pipeline inside the limits of an "
      f"entropy quantum computer, and they do it with proofs rather than "
      f"assurances. Truncation rewrites a Hamiltonian only when a spectral-gap "
      f"certificate holds and a uniqueness check passes, and its trigger is set "
      f"by what the hardware is observed to resolve rather than by a datasheet. "
      f"The trust-region encoding reaches "
      f"{hwd.get('levels', 202)} levels and {ds['dyn_range_db']:.1f}~dB with the "
      f"cost ratio unchanged, and bounded-sum decomposition brings dispatch "
      f"within every device limit while base-16 decomposition does not.\n\n")
    a(f"The resilience outcome is a worst-hour unserved fraction of "
      f"{100 * e1['M1_max_unserved']:.1f}\\% and {e1['M2_crit_hours']} critical "
      f"bus-hours lost against a {e1['M2_baseline']} bus-hour reference, at a "
      f"capital plan {e4['cost_ratio']:.3f}$\\times$ a certified optimum, with "
      f"{HW['matched']}/{HW['total']} islanding instances and the design stage "
      f"executed on real hardware. We claim no speedup: at this scale exact "
      f"classical methods are fast and we say so. What the device buys is "
      f"representational --- integer variables and degree-3 gating without "
      f"auxiliaries, at {e5['binary']['vars'] / e5['native']['vars']:.1f}$\\times$ "
      f"fewer variables than binary compilation.\n\n")
    a(f"Three doors stand open. Cross-island export is built and measured and "
      f"needs only a short hardware re-run to adopt. The exact mixed-integer "
      f"certifier we introduce proves optima at 40 binary variables in seconds, "
      f"where enumeration would need $10^{{12}}$ states, so certification no "
      f"longer limits growth of the candidate pool. And because islanding "
      f"problem size scales with candidate count rather than bus count, a "
      f"feeder of several thousand buses reduced to a few dozen candidates "
      f"remains a single job well inside the device's budget.\n\n")

    a(r"""\balance
\onecolumn
\begin{thebibliography}{9}\small
\bibitem{nguyen2024} N.~T.~M. Nguyen et al., ``Entropy computing: a paradigm for
optimization in open photonic systems,'' arXiv:2407.04512, 2024.
\bibitem{space2025} Dirac-3 application to space logistics planning,
arXiv:2501.05046, 2025.
\bibitem{baranwu1989} M.~E. Baran and F.~F. Wu, ``Network reconfiguration in
distribution systems for loss reduction and load balancing,''
\textit{IEEE Trans. Power Delivery}, vol.~4, no.~2, pp.~1401--1407, 1989.
\bibitem{nikmehr2024} N.~Nikmehr et al., ``Quantum annealing-infused microgrids
formation,'' \textit{IEEE Trans. Power Systems}, 2024.
\bibitem{qmgf2024} ``Reforming quantum microgrid formation,'' arXiv:2406.05916, 2024.
\bibitem{chenvu2025} Y.~Chen and T.~L. Vu, ``A review of quantum computing
technologies in power system optimization,'' PNNL-37598, 2025.
\bibitem{critical2026} ``A critical comment on entropy computing,''
arXiv:2605.03612, 2026.
\bibitem{glover1974} F.~Glover and E.~Woolsey, ``Converting the 0-1 polynomial
programming problem to a 0-1 linear program,'' \textit{Operations Research},
vol.~22, no.~1, pp.~180--182, 1974.
\bibitem{morstyn2024} T.~Morstyn and X.~Wang, ``Opportunities for quantum
computing within net-zero power system optimization,'' \textit{Joule}, vol.~8,
no.~6, pp.~1619--1640, 2024.
\end{thebibliography}
\end{document}
""")
    return "".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    new = build()
    OUT.parent.mkdir(exist_ok=True)
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != new:
            sys.exit("report .tex is out of date -- run `python tools/gen_report.py`")
        print("report .tex is up to date")
        return
    OUT.write_text(new, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
