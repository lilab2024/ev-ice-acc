"""Calibrate one configuration with differential evolution (global optimizer)."""
import time

import numpy as np
from scipy.optimize import differential_evolution

from src.calibration.objective import objective_batch, run_errors, split_spec, to_params


def bound_flags(free_names, bounds, values, rel_tol=0.01):                    # NEW
    """Names of parameters lying at a bound.

    Positive lower bound: tolerance is rel_tol of the range on a log scale, so a wide
    range (e.g. beta in [0.3, 100]) does not produce a large absolute tolerance.
    Zero lower bound: tolerance is rel_tol of the range on a linear scale.
    """
    flags = []
    for name, (lo, hi), x in zip(free_names, bounds, values):
        if lo > 0:
            span = np.log(hi / lo)
            near_low = np.log(x / lo) <= rel_tol * span
            near_high = np.log(hi / x) <= rel_tol * span
        else:
            span = hi - lo
            near_low = x - lo <= rel_tol * span
            near_high = hi - x <= rel_tol * span
        if near_low:
            flags.append(f"{name}=low")
        elif near_high:
            flags.append(f"{name}=high")
    return flags


def calibrate(model, spec, runs, seed=42, popsize=15, maxiter=200, tol=1e-3):
    """Calibrate `model` on `runs` with the parameter spec from experiment.yaml.

    Returns a dict with the full parameter set, objective value, per-run errors,
    parameters at their bounds, and optimizer diagnostics.
    """
    free_names, bounds, fixed = split_spec(spec)
    start = time.perf_counter()
    res = differential_evolution(
        objective_batch, bounds, args=(model, free_names, fixed, runs),
        vectorized=True, updating="deferred",       # evaluate the whole population at once
        popsize=popsize, maxiter=maxiter, tol=tol,
        seed=seed, polish=False)
    elapsed = time.perf_counter() - start

    params = to_params(res.x, free_names, fixed)
    per_run = {run["run_id"]: run_errors(model, params, run) for run in runs}
    return dict(params=params, objective=float(res.fun),
                at_bound=bound_flags(free_names, bounds, res.x),                # NEW
                per_run=per_run, free=free_names, fixed=fixed,
                nfev=int(res.nfev), nit=int(res.nit), success=bool(res.success),
                message=res.message, seconds=round(elapsed, 1))
