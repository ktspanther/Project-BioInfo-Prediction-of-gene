"""
Uniform random search.

This is the dumbest possible baseline: draw N random points uniformly in
the bounds, evaluate the objective at each, return the best one.

Why it matters: if PSO or DE only narrowly beats random search, then the
optimization is not the part doing the heavy lifting — the gain probably
comes from the pattern-matching idea itself, not the optimizer. Including
random search makes the comparison HONEST instead of cherry-picked.

Total evaluations are kept comparable to PSO/DE so the comparison is fair
in terms of compute budget.
"""
from __future__ import annotations

from typing import Callable
import numpy as np

from optimizer import OptimizerResult


def random_maximize(
    fn: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_particles: int = 15,
    max_iter: int = 25,
    patience: int = 8,
    target_score: float | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> OptimizerResult:
    """Maximize `fn` by uniform random sampling.

    Budget: n_particles * max_iter total evaluations — matches what PSO
    would use with the same settings (one full swarm per iteration).
    """
    bounds = np.asarray(bounds, dtype=float)
    rng = np.random.default_rng(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]

    budget = n_particles * max_iter
    best_x = None
    best_score = -np.inf
    history: list[float] = []

    for i in range(budget):
        x = rng.uniform(lo, hi)
        score = float(fn(x))
        if score > best_score:
            best_score = score
            best_x = x
        history.append(best_score)
        if target_score is not None and best_score >= target_score:
            break

    return OptimizerResult(
        best_x=np.asarray(best_x),
        best_score=float(best_score),
        history=history,
        n_evaluations=len(history),
        name="random",
    )
