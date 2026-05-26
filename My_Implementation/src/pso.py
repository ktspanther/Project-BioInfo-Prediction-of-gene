"""
Particle Swarm Optimization (PSO) for PatternChrome.

The paper uses R's psoptim() from the 'pso' package with:
  w = c(0.8, 0.4)  -- omega decreases linearly from 0.8 down to 0.4
  c.p = c.g = 2.05 -- cognitive and social weights

This file reproduces those same settings in Python.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


# PSO coefficients matching the paper's R psoptim call
_W_START = 0.8   # inertia at iteration 1
_W_END   = 0.4   # inertia at the last iteration (linear decrease)
_C_P     = 2.05  # cognitive component (c.p)
_C_G     = 2.05  # social component   (c.g)


@dataclass
class PSOResult:
    best_x: np.ndarray
    best_score: float
    history: list[float]       # global-best score per iteration
    all_scores: list[np.ndarray]


def pso_maximize(
    fn: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_particles: int = 20,
    max_iter: int = 20,
    patience: int = 3,
    target_score: float | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> PSOResult:
    """Maximize fn(x) using PSO with the paper's hyperparameters.

    bounds: (n_dim, 2) array of [lo, hi] per dimension
    patience: stop early if no improvement for this many iterations (maxit.stagnate in R)
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    n_dim = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo

    # Initialise positions uniformly, velocities as a small fraction of the range
    x = rng.uniform(lo, hi, size=(n_particles, n_dim))
    v = rng.uniform(-span, span, size=(n_particles, n_dim)) * 0.1

    scores = np.array([fn(xi) for xi in x])
    pbest_x = x.copy()
    pbest_s = scores.copy()

    g_idx = int(np.argmax(pbest_s))
    gbest_x = pbest_x[g_idx].copy()
    gbest_s = float(pbest_s[g_idx])

    history: list[float] = [gbest_s]
    all_scores: list[np.ndarray] = [scores.copy()]
    stale = 0

    for it in range(1, max_iter + 1):
        # Inertia decreases linearly from _W_START to _W_END (matches R w=c(0.8,0.4))
        omega = _W_START - (_W_START - _W_END) * it / max_iter

        # Sample random weights in [0, c_p] and [0, c_g] (equivalent to c * U(0,1))
        r1 = rng.uniform(0, _C_P, size=(n_particles, n_dim))
        r2 = rng.uniform(0, _C_G, size=(n_particles, n_dim))

        v = omega * v + r1 * (pbest_x - x) + r2 * (gbest_x[None, :] - x)
        x = x + v

        # Clamp to bounds; zero out velocity on clamped dimensions
        below = x < lo
        above = x > hi
        x = np.clip(x, lo, hi)
        v[below | above] = 0.0

        scores = np.array([fn(xi) for xi in x])
        all_scores.append(scores.copy())

        better = scores > pbest_s
        pbest_x[better] = x[better]
        pbest_s[better] = scores[better]

        g_idx = int(np.argmax(pbest_s))
        new_gbest = float(pbest_s[g_idx])
        if new_gbest > gbest_s + 1e-12:
            gbest_s = new_gbest
            gbest_x = pbest_x[g_idx].copy()
            stale = 0
        else:
            stale += 1

        history.append(gbest_s)
        if verbose:
            print(f"  PSO iter {it:3d}  gbest={gbest_s:.4f}  stale={stale}")

        if target_score is not None and gbest_s >= target_score:
            if verbose:
                print(f"  reached target {target_score}; stopping")
            break
        if stale >= patience:
            if verbose:
                print(f"  no improvement for {patience} iters; stopping")
            break

    return PSOResult(
        best_x=gbest_x,
        best_score=gbest_s,
        history=history,
        all_scores=all_scores,
    )


def pso_maximize_unified(
    fn,
    bounds,
    n_particles: int = 20,
    max_iter: int = 20,
    patience: int = 3,
    target_score: float | None = None,
    seed: int = 0,
    verbose: bool = False,
):
    """Wrapper that returns an OptimizerResult so PSO can be swapped with DE or random search."""
    from optimizer import OptimizerResult

    res = pso_maximize(
        fn=fn, bounds=bounds,
        n_particles=n_particles, max_iter=max_iter,
        patience=patience, target_score=target_score,
        seed=seed, verbose=verbose,
    )
    n_evals = n_particles * len(res.all_scores)
    return OptimizerResult(
        best_x=res.best_x,
        best_score=res.best_score,
        history=res.history,
        n_evaluations=n_evals,
        name="pso",
    )
