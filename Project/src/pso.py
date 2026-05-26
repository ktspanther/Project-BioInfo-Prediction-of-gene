"""
Minimal Particle Swarm Optimization (PSO) implementation.

PSO is a population-based metaheuristic. A "swarm" of `n_particles` candidate
solutions flies through the parameter space, where each particle's velocity is
pulled toward (a) its own best-found position and (b) the swarm's best-found
position. No gradients needed — perfect for our black-box objective
"train an XGBoost model with this proposed pattern and report AUC gain".

The paper uses the SPSO2007 standard. We use a global-best topology with
SPSO2007's recommended hyperparameters. This is slightly simpler than the
full SPSO2007 (which uses random K=3 informants per particle) but converges
similarly on this kind of low-dim continuous problem.

Reference: Clerc M., "Standard PSO 2007 (SPSO-07)", in *Innovations and
Developments of Swarm Intelligence Applications*, 2012.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math
import numpy as np


# SPSO2007 recommended values
_OMEGA = 1.0 / (2.0 * math.log(2.0))        # ≈ 0.721
_C = 0.5 + math.log(2.0)                    # ≈ 1.193


@dataclass
class PSOResult:
    best_x: np.ndarray
    best_score: float
    history: list[float]      # global-best score per iteration
    all_scores: list[np.ndarray]  # per-particle scores per iteration (for diagnostics)


def pso_maximize(
    fn: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_particles: int = 20,
    max_iter: int = 30,
    patience: int = 8,
    target_score: float | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> PSOResult:
    """Maximize a scalar objective `fn(x)` over a continuous box.

    Parameters
    ----------
    fn : (n_dim,) array -> float
        Objective to maximize. May be stochastic — particles are re-evaluated
        each iteration only on their *current* position; personal bests use
        the score recorded when they were set.
    bounds : (n_dim, 2) array
        Per-dimension [low, high] limits. Positions are clamped to this box;
        velocity on the offending axis is reset to 0 when clamped.
    n_particles, max_iter : ints
        Swarm size and iteration budget. Defaults are intentionally small so
        the demo runs in minutes; the paper uses bigger swarms for real runs.
    patience : int
        Stop early if the global best hasn't improved for this many iterations.
    target_score : float or None
        If set, stop early once any particle reaches this score.
    seed : int
    verbose : bool

    Returns
    -------
    PSOResult
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    n_dim = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo

    # Initialise positions uniformly, velocities small and random
    x = rng.uniform(lo, hi, size=(n_particles, n_dim))
    v = rng.uniform(-span, span, size=(n_particles, n_dim)) * 0.1

    # Evaluate initial swarm
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
        # Stochastic pulls toward personal and global best
        r1 = rng.uniform(0, _C, size=(n_particles, n_dim))
        r2 = rng.uniform(0, _C, size=(n_particles, n_dim))
        v = _OMEGA * v + r1 * (pbest_x - x) + r2 * (gbest_x[None, :] - x)
        x = x + v

        # Clamp to bounds and zero velocity on the clamped axis
        below = x < lo
        above = x > hi
        x = np.clip(x, lo, hi)
        v[below | above] = 0.0

        # Re-evaluate
        scores = np.array([fn(xi) for xi in x])
        all_scores.append(scores.copy())

        # Update personal bests
        better = scores > pbest_s
        pbest_x[better] = x[better]
        pbest_s[better] = scores[better]

        # Update global best
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
                print(f"  reached target_score {target_score}; stopping")
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


# ─────────────────────────────────────────────────────────────────────────────
# Unified-interface wrapper so PSO returns the same type as other optimizers.
# ─────────────────────────────────────────────────────────────────────────────

def pso_maximize_unified(
    fn,
    bounds,
    n_particles: int = 20,
    max_iter: int = 30,
    patience: int = 8,
    target_score: float | None = None,
    seed: int = 0,
    verbose: bool = False,
):
    """Same as `pso_maximize` but returns an `OptimizerResult`.

    Used when the pipeline calls PSO via the `get_optimizer("pso")` factory,
    so PSO and Differential Evolution (and random search) all return the
    same shape and can be swapped without other code changes.
    """
    from optimizer import OptimizerResult

    res = pso_maximize(
        fn=fn, bounds=bounds,
        n_particles=n_particles, max_iter=max_iter,
        patience=patience, target_score=target_score,
        seed=seed, verbose=verbose,
    )
    # Count evaluations: n_particles per iteration (PSO evaluates every
    # particle at each iteration including the initial one).
    n_evals = n_particles * len(res.all_scores)
    return OptimizerResult(
        best_x=res.best_x,
        best_score=res.best_score,
        history=res.history,
        n_evaluations=n_evals,
        name="pso",
    )

