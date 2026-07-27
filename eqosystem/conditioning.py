"""Certified Coefficient Truncation for analog (Dirac-3) submission.

WHY
---
Dirac-3 resolves coefficients to roughly 200:1 (23.0 dB in the 10*log10
convention used throughout this package). Terms below ``c_max / 200`` sit under
the analog noise floor, so the device silently optimizes a *different*
Hamiltonian than the one submitted -- and returns a plausible wrong answer with
no error. Naively dropping the small terms is not safe either: many small terms
can sum to reorder the spectrum.

This module drops small terms only when it can *prove* the ground state does not
move, and reports a certificate either way.

THE BOUND AND THE CERTIFICATE
-----------------------------
Write ``H(x) = sum_t c_t * prod_{i in S_t} x_i`` over integer boxes
``x_i in [0, u_i]``. Partition terms at ``floor = c_max / R`` into kept ``K``
and dropped ``D``, so ``H = H_K + H_D``.

Because every variable is non-negative, each dropped monomial ranges over
``[0, c_t * prod u_i]`` when ``c_t > 0`` and ``[c_t * prod u_i, 0]`` when
``c_t < 0``. Summing gives the exact interval of the dropped part over the box:

    M_D = sum_{c_t > 0} c_t * prod u_i          (max of H_D)
    m_D = sum_{c_t < 0} c_t * prod u_i          (min of H_D)
    delta = sum_{t in D} |c_t| * prod u_i       (= M_D - m_D here)

so ``|H(x) - H_K(x)| <= delta`` for every feasible ``x``.

Let ``E1 = min H_K`` and ``E2`` its second-best *distinct* energy.

*Specification certificate* (as written in the work order): ``E2 - E1 > 2*delta``.

*Tight certificate*: ``E2 - E1 > M_D - m_D``. Proof: for any ``y`` with
``H_K(y) >= E2`` we have ``H(y) >= E2 + m_D``, while for a minimizer ``x*`` of
``H_K`` we have ``H(x*) <= E1 + M_D``. If ``E2 + m_D > E1 + M_D`` then
``H(y) > H(x*)`` for every such ``y``, so no point outside ``argmin H_K`` can
beat ``x*``; hence ``argmin H <= argmin H_K``. QED

The tight form is a factor-of-two improvement over the specification form and is
equally rigorous, so both are computed and reported. Certification is claimed
only from the specification form, to stay faithful to the work order.

Note on degeneracy: the argument bounds ``argmin H`` *inside* ``argmin H_K``, which
is a statement about a unique ground state only when ``H_K`` has one minimizer.
This matters, because degeneracy inflates the apparent safety margin: a
Hamiltonian whose kept part ties across many states has a large gap to its
second-best *distinct* energy, while the dropped terms are precisely what breaks
the tie. So ``certified`` requires uniqueness; the weaker containment statement is
reported separately as ``certified_subset``.

WHEN TRUNCATION ACTUALLY FIRES (the calibrated trigger)
-------------------------------------------------------
Rewriting is gated on TWO conditions, both required:

  (a) the dynamic range exceeds ``trigger_db``, and
  (b) the certificate holds.

Condition (b) makes default-on conditioning safe: an unprovable rewrite is never
submitted. Condition (a) protects Hamiltonians that are already known to work.

``trigger_db`` defaults to 35 dB rather than the nominal 23 dB specification.
That is calibrated from measured hardware behaviour recorded in
``results/hardware_dirac3.json``: islanding instances at 26.1, 29.2, 29.4, 30.7
and 30.9 dB each returned the certified global optimum on Dirac-3. The trigger is
set above the highest dynamic range at which correct resolution has been
*observed*, so truncation can never be the cause of a change in an instance
already known to resolve correctly. This is a deliberately narrow claim: five
three-variable instances are not a device characterisation, and nothing here
asserts the device is accurate to 35 dB in general. ``spec_exceeded_23db`` is
recorded separately on every certificate so both the nominal and the calibrated
view can be reported.

Dynamic range alone is a poor predictor in any case: of the five instances above,
two had spectral gaps near 85 energy units (no plausible analog noise closes
them) while another had a gap of 0.24. The certificate measures gap against
perturbation, which is the quantity that actually matters.
"""
import warnings

import numpy as np

from .hamiltonians import Poly

# ----------------------------------------------------------------- device facts
MAX_TOTAL_LEVELS = 954        # sum of num_levels per job
MAX_LEVELS_PER_VAR = 16       # integer solver: 0..15, i.e. upper bound <= 15
DEFAULT_RESOLUTION_RATIO = 200.0
NOMINAL_SPEC_DB = 23.0        # user-guide figure: 200:1 == 23.0 dB at 10*log10
CALIBRATED_TRIGGER_DB = 35.0  # see module docstring
EXACT_LIMIT = 2 ** 21         # enumeration guard for E1/E2


def terms_db(terms):
    """Dynamic range in dB (10*log10 max/min), matching Poly.dynamic_range_db."""
    cs = np.abs(np.array([c for c in terms.values() if abs(c) > 1e-12]))
    if len(cs) == 0:
        return 0.0
    lo = cs.min()
    return float(10 * np.log10(cs.max() / lo)) if lo > 0 else 0.0


def total_levels(H):
    """Sum of num_levels over variables the device will actually receive."""
    return sum(H.upper[v] + 1 for v in H.upper if H.upper[v] >= 1)


def _box_product(H, key):
    """prod u_i over the variables appearing in a term (with multiplicity)."""
    p = 1.0
    for v in key:
        p *= H.upper[v]
    return p


def _partition(H, ratio):
    live = {k: c for k, c in H.terms.items() if abs(c) > 1e-12}
    if not live:
        return {}, {}, 0.0, 0.0
    c_max = max(abs(c) for c in live.values())
    floor = c_max / ratio
    kept = {k: c for k, c in live.items() if abs(c) >= floor}
    dropped = {k: c for k, c in live.items() if abs(c) < floor}
    return kept, dropped, floor, c_max


def _rebuild(H, kept):
    Ht = Poly()
    Ht.upper = dict(H.upper)
    Ht.names = dict(H.names)
    Ht.const = H.const
    for k, c in kept.items():
        Ht.terms[k] = c
    return Ht


def _dropped_interval(H, dropped):
    """Exact (m_D, M_D, delta) of the dropped polynomial over the feasible box."""
    m_d = M_d = delta = 0.0
    for k, c in dropped.items():
        span = c * _box_product(H, k)
        if span > 0:
            M_d += span
        else:
            m_d += span
        delta += abs(span)
    return m_d, M_d, delta


def _two_best_exact(H, limit=EXACT_LIMIT):
    """(E1, E2, n_ground_states) by enumeration, or (None, None, None) if too big.

    E2 is the second-best DISTINCT energy; n_ground_states counts minimizers.
    """
    import itertools
    vs = sorted(H.upper)
    ubs = [H.upper[v] for v in vs]
    space = 1
    for u in ubs:
        space *= (u + 1)
        if space > limit:
            return None, None, None
    energies = []
    for x in itertools.product(*[range(u + 1) for u in ubs]):
        energies.append(H.evaluate(np.asarray(x, dtype=float)))
    arr = np.asarray(energies)
    e1 = float(arr.min())
    n_gs = int(np.sum(np.abs(arr - e1) < 1e-12))
    rest = arr[np.abs(arr - e1) >= 1e-12]
    e2 = float(rest.min()) if rest.size else None
    return e1, e2, n_gs


def _two_best_heuristic(H, restarts=8):
    """Best / second-best distinct energies from a multi-restart annealer.

    Never sufficient for certification -- only for reporting.
    """
    from .solvers import AnnealerSolver
    seen = []
    for s in range(restarts):
        r = AnnealerSolver(restarts=2, iters=3000, seed=1000 + s).solve(H)
        seen.append(float(r["energy"]))
    uniq = sorted(set(round(e, 10) for e in seen))
    e1 = uniq[0]
    e2 = uniq[1] if len(uniq) > 1 else None
    return e1, e2, None


def truncate_certified(H, resolution_ratio=DEFAULT_RESOLUTION_RATIO,
                       trigger_db=CALIBRATED_TRIGGER_DB, verify_exact=True,
                       exact_limit=EXACT_LIMIT):
    """Conditionally truncate ``H`` and return ``(H_out, cert)``.

    ``H_out`` is a truncated copy only when the dynamic range exceeds
    ``trigger_db`` AND the certificate holds; otherwise the ORIGINAL object is
    returned unchanged, so a Hamiltonian that already resolves correctly is
    submitted byte-identically.

    ``cert`` always describes what truncation *would* do, whether or not it fired.
    """
    db_before = terms_db(H.terms)
    kept, dropped, floor, c_max = _partition(H, resolution_ratio)
    H_trunc = _rebuild(H, kept) if dropped else H
    m_d, M_d, delta = _dropped_interval(H, dropped)

    method, e1, e2, n_gs = "none", None, None, None
    if dropped:
        if verify_exact:
            e1, e2, n_gs = _two_best_exact(H_trunc, limit=exact_limit)
            method = "exact" if e1 is not None else "heuristic"
        else:
            method = "heuristic"
        if method == "heuristic":
            e1, e2, n_gs = _two_best_heuristic(H_trunc)

    gap = (e2 - e1) if (e1 is not None and e2 is not None) else None
    margin_spec = (gap - 2.0 * delta) if gap is not None else None
    margin_tight = (gap - (M_d - m_d)) if gap is not None else None

    # `certified_subset` is what the bound literally proves: argmin H is contained
    # in argmin H_K. That is only a statement about a UNIQUE ground state when H_K
    # has one minimizer. Under degeneracy the gap to the second-best distinct
    # energy can be large while the dropped terms are exactly what decides which
    # of the tied states wins -- so uniqueness is required before claiming the
    # ground state is preserved, and before rewriting anything.
    certified_subset = bool(dropped and method == "exact" and margin_spec is not None
                            and margin_spec > 0.0)
    certified = bool(certified_subset and n_gs == 1)
    certified_tight = bool(dropped and method == "exact" and margin_tight is not None
                           and margin_tight > 0.0 and n_gs == 1)
    over_trigger = db_before > trigger_db
    rewritten = bool(dropped and over_trigger and certified)

    cert = dict(
        # what truncation would do
        fired=bool(dropped),
        kept_terms=len(kept),
        dropped_terms=len(dropped),
        total_terms=len(kept) + len(dropped),
        c_max=float(c_max),
        floor=float(floor),
        db_before=float(db_before),
        db_after=float(terms_db(H_trunc.terms)),
        # the bound
        delta=float(delta),
        dropped_min=float(m_d),
        dropped_max=float(M_d),
        # the spectrum of H_K
        e1=(float(e1) if e1 is not None else None),
        e2=(float(e2) if e2 is not None else None),
        gap=(float(gap) if gap is not None else None),
        margin=(float(margin_spec) if margin_spec is not None else None),
        margin_tight=(float(margin_tight) if margin_tight is not None else None),
        method=method,
        n_ground_states=n_gs,
        # the verdict
        certified=certified,                  # unique ground state provably preserved
        certified_subset=certified_subset,    # argmin H contained in argmin H_K
        certified_tight=certified_tight,      # same, via the tight (width) threshold
        rewritten=rewritten,
        # thresholds, so the report can show nominal vs calibrated
        resolution_ratio=float(resolution_ratio),
        trigger_db=float(trigger_db),
        spec_exceeded_23db=bool(db_before > NOMINAL_SPEC_DB),
        over_calibrated_trigger=bool(over_trigger),
        # resources
        n_vars=int(H.n),
        total_levels=int(total_levels(H)),
        max_upper_bound=int(max(H.upper.values())) if H.upper else 0,
    )
    return (H_trunc if rewritten else H), cert


def hardware_legality(H, cert=None, resolution_ratio=DEFAULT_RESOLUTION_RATIO,
                      trigger_db=CALIBRATED_TRIGGER_DB):
    """Return a list of reasons ``H`` may not go to the integer solver (empty = OK)."""
    reasons = []
    lv = total_levels(H)
    if lv > MAX_TOTAL_LEVELS:
        reasons.append(f"total levels {lv} exceeds the {MAX_TOTAL_LEVELS} limit")
    over = {v: H.upper[v] for v in H.upper if H.upper[v] > MAX_LEVELS_PER_VAR - 1}
    if over:
        worst = max(over, key=lambda v: over[v])
        reasons.append(
            f"{len(over)} variable(s) exceed the {MAX_LEVELS_PER_VAR}-level integer cap "
            f"(worst: '{H.names.get(worst, worst)}' upper bound {over[worst]})")
    db = terms_db(H.terms)
    if db > trigger_db and not (cert and cert.get("certified")):
        reasons.append(
            f"dynamic range {db:.1f} dB exceeds the calibrated trigger "
            f"{trigger_db:.0f} dB and truncation could not be certified "
            f"(nominal specification is {NOMINAL_SPEC_DB:.0f} dB)")
    return reasons


def assert_hardware_legal(H, cert=None, stage="unknown", **kw):
    """Raise rather than let an out-of-range Hamiltonian reach the device."""
    reasons = hardware_legality(H, cert=cert, **kw)
    if reasons:
        raise ValueError(
            f"Hamiltonian for stage '{stage}' is not legal on the Dirac-3 integer "
            f"solver:\n  - " + "\n  - ".join(reasons) +
            "\nRoute this stage to the continuous solver (W3c), apply radix-16 "
            "decomposition (W3a), or decompose by zone (W3d).")


def warn_unconditioned(H, stage="unknown"):
    """Visible warning naming the dB when conditioning is deliberately disabled."""
    warnings.warn(
        f"to_eqc_model(condition=False) on stage '{stage}': submitting "
        f"{terms_db(H.terms):.1f} dB, {total_levels(H)} levels un-conditioned. "
        f"Coefficients below c_max/{DEFAULT_RESOLUTION_RATIO:.0f} are under the "
        f"analog noise floor and the device may optimize a different Hamiltonian.",
        RuntimeWarning, stacklevel=3)
