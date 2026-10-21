"""Step 03: calibrate all model variants for every configuration (vehicle x gap setting).

Variants: IDM and OVRV (main and supplementary run selections), IDM and OVRV with an
actuation lag (EV-aware extension), and the Newell and Gipps baselines (see VARIANTS).

Each (variant, configuration) result is saved immediately as JSON, so an interrupted
run can be resumed: configurations that already have a result file are skipped.

Usage (from the repository root):
    python scripts/03_calibrate.py                           # main variants only
    python scripts/03_calibrate.py --variants supp_idm_free  # selected variants
    python scripts/03_calibrate.py --force                   # recompute everything
Outputs: results/calibration/<variant>/<config>.json and
         results/calibration/<variant>_params.csv, <variant>_errors.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.calibration.objective import prepare_runs, split_spec
from src.calibration.optimizer import bound_flags, calibrate
from src.calibration.selection import list_configs, select_runs
from src.calibration.variants import MODELS, VARIANTS
from src.config import REPO_ROOT, load_paths, load_yaml

DEFAULT_VARIANTS = ["main_idm", "main_ovrv"]


def summarize(variant, out_dir, spec):
    """Collect all JSON results of one variant into a parameter table and an error table.

    Bound flags are recomputed from the stored parameters with the current criterion.
    """
    free_names, bounds, _ = split_spec(spec)
    param_rows, error_rows = [], []
    for f in sorted((out_dir / variant).glob("*.json")):
        r = json.loads(f.read_text())
        flags = bound_flags(free_names, bounds,
                            [r["params"][n] for n in free_names])
        param_rows.append({"config": r["config"], "powertrain": r["powertrain"],
                           "n_runs": len(r["per_run"]), **r["params"],
                           "objective": r["objective"], "at_bound": ";".join(flags),
                           "success": r["success"], "seconds": r["seconds"]})
        for run_id, e in r["per_run"].items():
            error_rows.append({"config": r["config"], "powertrain": r["powertrain"],
                               "run_id": run_id, **e})
    pd.DataFrame(param_rows).to_csv(out_dir / f"{variant}_params.csv", index=False)
    pd.DataFrame(error_rows).to_csv(out_dir / f"{variant}_errors.csv", index=False)


def check_stored(out_file, spec, run_ids):
    """Raise if a stored result was computed with other settings than the current ones.

    Compares the free and fixed parameters and the calibration runs. Parameter bounds are
    not stored in the result files, so a change of the bounds alone is not detected.
    """
    r = json.loads(out_file.read_text())
    free_names, _, fixed = split_spec(spec)
    problems = []
    if r["free"] != free_names:
        problems.append(f"free parameters {r['free']} -> {free_names}")
    if r["fixed"] != fixed:
        problems.append(f"fixed parameters {r['fixed']} -> {fixed}")
    if sorted(r["per_run"]) != sorted(run_ids):
        problems.append("different calibration runs")
    if problems:
        raise RuntimeError(f"{out_file} was computed with other settings "
                           f"({'; '.join(problems)}). Use --force to recompute.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                        choices=list(VARIANTS))
    parser.add_argument("--force", action="store_true",
                        help="recompute configurations that already have results")
    args = parser.parse_args()

    paths = load_paths()
    cfg = load_yaml("experiment.yaml")
    opt = cfg["calibration"]["optimizer"]
    traj = pd.read_parquet(paths["processed"] / "trajectories_10hz.parquet")
    manifest = pd.read_csv(paths["processed"] / "runs_manifest.csv")
    qc = pd.read_csv(paths["processed"] / "runs_qc.csv")
    out_dir = REPO_ROOT / "results" / "calibration"

    for variant in args.variants:
        model_name, spec_key, sel = VARIANTS[variant]
        model = MODELS[model_name]
        spec = cfg["calibration"]["params"][spec_key]
        runs_table = select_runs(manifest, qc, cfg, sel)
        (out_dir / variant).mkdir(parents=True, exist_ok=True)
        print(f"\n===== {variant}: {model_name.upper()}, params '{spec_key}', runs '{sel}' =====")

        for config in list_configs(runs_table):
            out_file = out_dir / variant / f"{config.replace(' ', '_')}.json"
            sub = runs_table[runs_table["config"] == config]
            if out_file.exists() and not args.force:
                check_stored(out_file, spec, sub["run_id"].tolist())
                print(f"  {config:18s} already done, skipped")
                continue
            runs = prepare_runs(traj, sub["run_id"].tolist())
            r = calibrate(model, spec, runs, seed=opt["seed"], popsize=opt["popsize"],
                          maxiter=opt["maxiter"], tol=opt["tol"])
            r.update(config=config, powertrain=sub["powertrain"].iloc[0],
                     variant=variant, message=str(r["message"]))
            out_file.write_text(json.dumps(r, indent=2))
            print(f"  {config:18s} runs={len(runs):3d}  RMSE={r['objective']:.3f} m  "
                  f"at_bound={r['at_bound'] or '-'}  {r['seconds']:.0f} s", flush=True)

        summarize(variant, out_dir, spec)
        print(f"  summary written to {out_dir / (variant + '_params.csv')}")


if __name__ == "__main__":
    main()
