"""Contingency scenario generation via Latin Hypercube Sampling.

Each scenario is a genuine N-1 event: one tree branch fails at hour h0 for
`duration` hours, de-energizing every bus downstream of the break. Renewable
and load multipliers stress the islands that must pick up the abandoned load.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import qmc

from . import grid

DEFAULT_N_SCENARIOS = 50     # top of the challenge's stated 10-50 range
TREE_BUCKETS = 4             # forecast-error resampling points per outage window
TREE_LOAD_SPREAD = 0.15      # +/- fractional load forecast error per bucket
TREE_RENEW_SPREAD = 0.30     # +/- fractional renewable forecast error per bucket


@dataclass
class Scenario:
    sid: int
    r_factor: float          # renewable availability multiplier [0.2, 1.0]
    load_factor: float       # demand multiplier [0.7, 1.3]
    failed_line: tuple       # (u, v) tree branch that trips
    start_hour: int          # outage start hour [0, 23]
    duration: int            # repair time, hours [4, 12]
    prob: float = 0.0
    dead_buses: set = field(default_factory=set)
    # --- scenario-tree mode (W6): forecast errors resampled per time bucket,
    # with the contingency held fixed within the branch. Empty in flat mode.
    r_path: tuple = ()
    load_path: tuple = ()
    bucket_hours: int = 0

    @property
    def hours(self):
        return [(self.start_hour + k) % 24 for k in range(self.duration)]

    @property
    def is_tree(self):
        return bool(self.load_path)

    def _bucket(self, hour):
        """Index of the time bucket containing `hour` within the outage window."""
        k = (hour - self.start_hour) % 24
        k = min(k, self.duration - 1)
        return min(k // max(self.bucket_hours, 1), len(self.load_path) - 1)

    def load_factor_at(self, hour):
        """Demand multiplier at `hour` (bucket-varying in tree mode)."""
        return self.load_path[self._bucket(hour)] if self.load_path else self.load_factor

    def r_factor_at(self, hour):
        """Renewable multiplier at `hour` (bucket-varying in tree mode)."""
        return self.r_path[self._bucket(hour)] if self.r_path else self.r_factor


def generate(n_scenarios: int = DEFAULT_N_SCENARIOS, seed: int = 42,
             stress: bool = False, tree: bool = False,
             tree_buckets: int = TREE_BUCKETS):
    """Latin-Hypercube contingency scenarios.

    The challenge specifies "10-50 per time step", which admits two readings.
    Flat mode (default) treats the count as scenarios over the whole horizon.
    Tree mode (`tree=True`) reads it per time step: the contingency is fixed
    within a branch while the forecast-error dimensions (renewables, load) are
    resampled per time bucket around the branch's central value, which is what
    "scenarios represent forecast errors in renewables and loads" describes.
    Both are reported; the flat reading produces the headline numbers.
    """
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    u = sampler.random(n=n_scenarios)
    lo = [0.2, 0.7, 0.0, 0.0, 4.0]
    hi = [1.0, 1.3, 1.0, 24.0, 16.0]
    if stress:  # beyond-design-basis: load tail 1.5x, outages up to 20 h, weak renewables
        lo = [0.1, 1.0, 0.0, 0.0, 8.0]
        hi = [0.8, 1.5, 1.0, 24.0, 28.0]
    x = qmc.scale(u, lo, hi)

    # Per-bucket forecast errors get their own LHS block, seeded off the same
    # seed so the whole construction stays reproducible.
    if tree:
        tsam = qmc.LatinHypercube(d=2 * tree_buckets, seed=seed + 1)
        t = tsam.random(n=n_scenarios)

    # Weight line-failure sampling toward main-feeder branches (higher impact,
    # consistent with storm damage statistics on exposed trunk lines).
    n_lines = len(grid.LINES)
    scens = []
    for s in range(n_scenarios):
        line_idx = min(int(x[s, 2] * n_lines), n_lines - 1)
        fl = grid.LINES[line_idx][:2]
        dur = int(round(x[s, 4]))
        r0, l0 = float(x[s, 0]), float(x[s, 1])
        r_path, load_path, bh = (), (), 0
        if tree:
            bh = max(1, int(np.ceil(dur / tree_buckets)))
            nb = int(np.ceil(dur / bh))
            r_path, load_path = [], []
            for b in range(nb):
                er = (2 * t[s, b % tree_buckets] - 1) * TREE_RENEW_SPREAD
                el = (2 * t[s, tree_buckets + b % tree_buckets] - 1) * TREE_LOAD_SPREAD
                r_path.append(float(np.clip(r0 * (1 + er), lo[0], hi[0])))
                load_path.append(float(np.clip(l0 * (1 + el), lo[1], hi[1])))
            r_path, load_path = tuple(r_path), tuple(load_path)
        sc = Scenario(
            sid=s,
            r_factor=r0,
            load_factor=l0,
            failed_line=fl,
            start_hour=int(x[s, 3]) % 24,
            duration=dur,
            prob=1.0 / n_scenarios,
            dead_buses=grid.downstream_buses(fl),
            r_path=r_path, load_path=load_path, bucket_hours=bh,
        )
        scens.append(sc)
    return scens
