"""
Differential Evolution wrapper.

Differential Evolution (Storn & Price, 1997) is a population-based,
gradient-free optimizer like PSO, but with very different update rules:
each candidate's new position is built from differences between OTHER
candidates' positions, scaled by a mutation factor, then recombined with
the candidate's current position via crossover.

Why DE is a fair alternative to PSO:
- Same family: gradient-free, population-based, continuous search space
- Long track record in benchmark suites (often beats PSO on multimodal
  problems)
- Available in scipy with no extra installation (`scipy.optimize.differential_evolution`)
- Different mechanism: PSO is mostly attractor-based (pulled toward bests),
  DE is mostly diversity-based (random vector differences). If both give
  similar AUCs, that's strong evidence the optimization step isn't the
  bottleneck.

Reference: Storn & Price, "Differential Evolution — A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces", 1997.
"""
from __future__ import annotations

from typing import Callable
import numpy as np
from scipy.optimize import differential_evolution

from optimizer import OptimizerResult


def de_maximize(
    fn: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_particles: int = 15,      # called "popsize" in DE jargon
    max_iter: int = 25,         # called "maxiter" in DE jargon
    patience: int = 8,
    target_score: float | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> OptimizerResult:
    """Maximize a scalar objective using Differential Evolution.

    Signature matches `pso_maximize_unified` so the pipeline can swap
    optimizers without other code changes.

    Notes
    -----
    - scipy's DE minimizes by default. We pass `-fn(x)` and negate back.
    - `popsize` in scipy is a multiplier on the number of dimensions
      (total population = popsize * len(bounds)). We compute it so the
      effective population is comparable to PSO's n_particles.
    - The `tol` parameter is set loose so DE terminates on `maxiter`
      rather than premature convergence, matching how PSO usually exits.
    """
    bounds = np.asarray(bounds, dtype=float)
    n_dim = bounds.shape[0]

    # Track every evaluation so we can compare against PSO fairly.
    history: list[float] = []
    n_calls = [0]

    def neg_fn(x):
        n_calls[0] += 1
        score = float(fn(x))
        # store the best score so far (running max)
        if not history or score > history[-1]:
            history.append(score)
        else:
            history.append(history[-1])
        return -score

    # scipy popsize is a MULTIPLIER on n_dim, so total particles = popsize * n_dim
    # Choose popsize so total population is roughly n_particles.
    popsize = max(2, n_particles // max(n_dim, 1))

    bounds_list = list(map(tuple, bounds))

    if verbose:
        print(f"  [DE] popsize_multiplier={popsize}  "
              f"effective_pop={popsize * n_dim}  maxiter={max_iter}")

    result = differential_evolution(
        neg_fn,
        bounds=bounds_list,
        maxiter=max_iter,
        popsize=popsize,
        tol=1e-6,
        mutation=(0.5, 1.0),   # F in [0.5, 1.0] — moderate diversity
        recombination=0.7,     # CR — DE crossover probability
        seed=seed,
        polish=False,          # don't run L-BFGS-B at the end (apples to apples vs PSO)
        init="sobol",          # quasi-random initialisation (better than uniform)
        updating="deferred",   # synchronous updates, closer to PSO's swarm step
        workers=1,
    )

    return OptimizerResult(
        best_x=np.asarray(result.x),
        best_score=-float(result.fun),
        history=history,
        n_evaluations=n_calls[0],
        name="de",
    )
