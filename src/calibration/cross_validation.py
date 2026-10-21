"""Grouped cross-validation: calibrate on all groups but one, test on the held-out group."""
import numpy as np

from src.calibration.objective import prepare_runs, run_errors
from src.calibration.optimizer import calibrate


def cross_validate(model, spec, config_runs, traj, fold_col, opt):
    """Grouped cross-validation for one configuration.

    config_runs: rows of the run table for ONE configuration
    fold_col   : column defining the folds (e.g. 'profile' for ICE, 'v_reduced_mph' for EV)
    opt        : optimizer settings (seed, popsize, maxiter, tol)
    Returns a list with one dict per fold.
    """
    folds = []
    for held_out in sorted(config_runs[fold_col].unique()):
        is_test = config_runs[fold_col] == held_out
        train = prepare_runs(traj, config_runs.loc[~is_test, "run_id"].tolist())
        test = prepare_runs(traj, config_runs.loc[is_test, "run_id"].tolist())

        r = calibrate(model, spec, train, seed=opt["seed"], popsize=opt["popsize"],
                      maxiter=opt["maxiter"], tol=opt["tol"])
        test_err = [run_errors(model, r["params"], run) for run in test]
        collided = any(e["collided"] for e in test_err)

        folds.append(dict(
            held_out=str(held_out),
            n_train=len(train), n_test=len(test),
            train_spacing_rmse=r["objective"],
            test_spacing_rmse=np.nan if collided else
                float(np.mean([e["spacing_rmse"] for e in test_err])),
            test_speed_rmse=np.nan if collided else
                float(np.mean([e["speed_rmse"] for e in test_err])),
            test_collided=collided,
            params=r["params"], at_bound=r["at_bound"]))
    return folds
