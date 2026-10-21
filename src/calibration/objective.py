"""Calibration objective: mean spacing RMSE over the runs of one configuration."""
import numpy as np

from src.models.simulate import run_model

COLLISION_PENALTY = 1e3      # objective value returned when a simulation collides


# ---------------------------------------------------------------------------
# Parameter specification: free (calibrated) vs. fixed parameters
# ---------------------------------------------------------------------------
def split_spec(spec):
    """Split a parameter spec from experiment.yaml into free and fixed parts.

    spec: {name: [low, high]} for calibrated parameters, {name: value} for fixed ones.
    Returns (free_names, bounds, fixed) where bounds is a list of (low, high).
    """
    free_names, bounds, fixed = [], [], {}
    for name, value in spec.items():
        if isinstance(value, (list, tuple)):
            free_names.append(name)
            bounds.append((float(value[0]), float(value[1])))
        else:
            fixed[name] = float(value)
    return free_names, bounds, fixed


def to_params(x, free_names, fixed):
    """Combine an optimizer vector x with the fixed parameters into a full dict."""
    p = dict(fixed)
    p.update(zip(free_names, map(float, x)))
    return p


# ---------------------------------------------------------------------------
# Runs and errors
# ---------------------------------------------------------------------------
def prepare_runs(traj, run_ids):
    """Extract the arrays needed for calibration, one dict per run."""
    runs = []
    for run_id in run_ids:
        d = traj[traj["run_id"] == run_id]
        runs.append(dict(run_id=run_id, t=d["t"].values, v_l=d["v_l"].values,
                         v_f=d["v_f"].values, s=d["s"].values))
    return runs


def run_errors(model, p, run):
    """Simulate one run and return spacing / speed errors."""
    out = run_model(model, p, run["t"], run["v_l"],
                            s_init=run["s"][0], v_init=run["v_f"][0])
    if out["collided"]:
        return dict(collided=True, spacing_rmse=np.nan, speed_rmse=np.nan,
                    spacing_mape=np.nan, speed_mape=np.nan)
    e_s = out["s"] - run["s"]
    e_v = out["v"] - run["v_f"]
    moving = run["v_f"] > 1.0                  # avoid dividing by ~0 at standstill
    return dict(collided=False,
                spacing_rmse=float(np.sqrt(np.mean(e_s ** 2))),
                speed_rmse=float(np.sqrt(np.mean(e_v ** 2))),
                spacing_mape=float(100 * np.mean(np.abs(e_s) / run["s"])),
                speed_mape=float(100 * np.mean(np.abs(e_v[moving]) / run["v_f"][moving])))


def objective(x, model, free_names, fixed, runs):
    """Mean spacing RMSE over runs; COLLISION_PENALTY if any run collides."""
    p = to_params(x, free_names, fixed)
    total = 0.0
    for run in runs:
        err = run_errors(model, p, run)
        if err["collided"] or not np.isfinite(err["spacing_rmse"]):
            return COLLISION_PENALTY
        total += err["spacing_rmse"]
    return total / len(runs)


# ---------------------------------------------------------------------------
# Vectorized objective (used by the optimizer for speed)
# ---------------------------------------------------------------------------
from src.models.simulate import run_model_batch


def objective_batch(X, model, free_names, fixed, runs):
    """Mean spacing RMSE for many parameter sets at once.

    X has shape (n_free, m): one column per candidate parameter set, which is the
    layout scipy's differential_evolution uses with vectorized=True.
    Returns an array of length m; collided candidates get COLLISION_PENALTY.
    """
    X = np.atleast_2d(X)
    m = X.shape[1]
    P = {name: np.full(m, value) for name, value in fixed.items()}
    P.update({name: X[i] for i, name in enumerate(free_names)})

    total = np.zeros(m)
    bad = np.zeros(m, dtype=bool)
    for run in runs:
        out = run_model_batch(model, P, run["t"], run["v_l"],
                                      s_init=run["s"][0], v_init=run["v_f"][0])
        rmse = np.sqrt(np.mean((out["s"] - run["s"][:, None]) ** 2, axis=0))
        bad |= out["collided"] | ~np.isfinite(rmse)
        total += np.where(np.isfinite(rmse), rmse, 0.0)
    result = total / len(runs)
    result[bad] = COLLISION_PENALTY
    return result
