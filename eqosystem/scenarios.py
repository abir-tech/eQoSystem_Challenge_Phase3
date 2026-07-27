"""Contingency scenario generation via Latin Hypercube Sampling.

Each scenario is a genuine N-1 event: one tree branch fails at hour h0 for
`duration` hours, de-energizing every bus downstream of the break. Renewable
and load multipliers stress the islands that must pick up the abandoned load.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import qmc

from . import grid


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

    @property
    def hours(self):
        return [(self.start_hour + k) % 24 for k in range(self.duration)]


def generate(n_scenarios: int = 20, seed: int = 42, stress: bool = False):
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    u = sampler.random(n=n_scenarios)
    lo = [0.2, 0.7, 0.0, 0.0, 4.0]
    hi = [1.0, 1.3, 1.0, 24.0, 16.0]
    if stress:  # beyond-design-basis: load tail 1.5x, outages up to 20 h, weak renewables
        lo = [0.1, 1.0, 0.0, 0.0, 8.0]
        hi = [0.8, 1.5, 1.0, 24.0, 28.0]
    x = qmc.scale(u, lo, hi)

    # Weight line-failure sampling toward main-feeder branches (higher impact,
    # consistent with storm damage statistics on exposed trunk lines).
    n_lines = len(grid.LINES)
    scens = []
    for s in range(n_scenarios):
        line_idx = min(int(x[s, 2] * n_lines), n_lines - 1)
        fl = grid.LINES[line_idx][:2]
        sc = Scenario(
            sid=s,
            r_factor=float(x[s, 0]),
            load_factor=float(x[s, 1]),
            failed_line=fl,
            start_hour=int(x[s, 3]) % 24,
            duration=int(round(x[s, 4])),
            prob=1.0 / n_scenarios,
            dead_buses=grid.downstream_buses(fl),
        )
        scens.append(sc)
    return scens
