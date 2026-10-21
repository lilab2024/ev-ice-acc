"""Step 03b: grouped cross-validation of the main calibration variants.

ICE: folds by speed profile; EV: folds by reduced speed (see experiment.yaml).
Each (variant, configuration) result is saved immediately, so a run can be resumed.

Usage (from the repository root):
    python scripts/03b_cross_validate.py            # variants listed in experiment.yaml
    python scripts/03b_cross_validate.py --force    # recompute everything
Outputs: results/cross_validation/<variant>/<config>.json and
         results/cross_validation/<variant>_folds.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.calibration.cross_validation import cross_validate
from src.calibration.objective import split_spec
from src.calibration.selection import list_configs, select_runs
from src.calibration.variants import MODELS, VARIANTS
from src.config import REPO_ROOT, load_paths, load_yaml


def summarize(variant, out_dir):
    """One row per fold, with the fold parameters as columns."""
    rows = []
    for f in sorted((out_dir / variant).glob("*.json")):
        r = json.loads(f.read_text())
        for fold in r["folds"]:
            rows.append({"config": r["config"], "powertrain": r["powertrain"],
                         "fold_by": r["fold_by"],
                         **{k: v for k, v in fold.items() if k not in ("params", "at_bound")},
                         **fold["params"], "at_bound": ";".join(fold["at_bound"])})
    pd.DataFrame(rows).to_csv(out_dir / f"{variant}_folds.csv", index=False)


def check_stored(out_file, spec, config_runs, fold_col):
    """Raise if a stored result was computed with other settings than the current ones.

    The result files store the folds and the full parameter sets, but not which parameters
    were free. Checked: fold definition and run counts, parameter names, the values of fixed
    parameters, and that no free parameter has the same interior value in every fold (which
    only happens if it was fixed). Bounds are not stored, so a change of bounds alone is not
    detected.
    """
    r = json.loads(out_file.read_text())
    free_names, bounds, fixed = split_spec(spec)
    folds = r["folds"]
    groups = sorted(config_runs[fold_col].unique())
    expected = [(str(g), int((config_runs[fold_col] != g).sum()),
                 int((config_runs[fold_col] == g).sum())) for g in groups]
    problems = []
    if r["fold_by"] != fold_col:
        problems.append(f"folds by {r['fold_by']} -> {fold_col}")
    elif [(f["held_out"], f["n_train"], f["n_test"]) for f in folds] != expected:
        problems.append("different folds or calibration runs")
    if set(folds[0]["params"]) != set(spec):
        problems.append(f"parameters {sorted(folds[0]['params'])} -> {sorted(spec)}")
    else:
        for name, value in fixed.items():
            if any(f["params"][name] != value for f in folds):
                problems.append(f"{name} is now fixed at {value}")
        for name, (lo, hi) in zip(free_names, bounds):
            values = {f["params"][name] for f in folds}
            if len(folds) > 1 and len(values) == 1 and values.isdisjoint({lo, hi}):
                problems.append(f"{name} is now free but was fixed at {values.pop()}")
    if problems:
        raise RuntimeError(f"{out_file} was computed with other settings "
                           f"({'; '.join(problems)}). Use --force to recompute.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="recompute configurations that already have results")
    args = parser.parse_args()

    paths = load_paths()
    cfg = load_yaml("experiment.yaml")
    opt = cfg["calibration"]["optimizer"]
    fold_by = cfg["cross_validation"]["fold_by"]
    traj = pd.read_parquet(paths["processed"] / "trajectories_10hz.parquet")
    manifest = pd.read_csv(paths["processed"] / "runs_manifest.csv")
    qc = pd.read_csv(paths["processed"] / "runs_qc.csv")
    out_dir = REPO_ROOT / "results" / "cross_validation"

    for variant in cfg["cross_validation"]["variants"]:
        model_name, spec_key, sel = VARIANTS[variant]
        model = MODELS[model_name]
        spec = cfg["calibration"]["params"][spec_key]
        runs_table = select_runs(manifest, qc, cfg, sel)
        (out_dir / variant).mkdir(parents=True, exist_ok=True)
        print(f"\n===== {variant} =====")

        for config in list_configs(runs_table):
            out_file = out_dir / variant / f"{config.replace(' ', '_')}.json"
            sub = runs_table[runs_table["config"] == config]
            powertrain = sub["powertrain"].iloc[0]
            if out_file.exists() and not args.force:
                check_stored(out_file, spec, sub, fold_by[powertrain])
                print(f"  {config:18s} already done, skipped")
                continue
            folds = cross_validate(model, spec, sub, traj, fold_by[powertrain], opt)
            out_file.write_text(json.dumps(
                dict(config=config, powertrain=powertrain, variant=variant,
                     fold_by=fold_by[powertrain], folds=folds), indent=2))
            train = sum(f["train_spacing_rmse"] for f in folds) / len(folds)
            test = pd.Series([f["test_spacing_rmse"] for f in folds]).mean()
            print(f"  {config:18s} folds={len(folds)}  train={train:.3f} m  "
                  f"test={test:.3f} m  ({100 * (test / train - 1):+.0f} %)", flush=True)

        summarize(variant, out_dir)
        print(f"  summary written to {out_dir / (variant + '_folds.csv')}")


if __name__ == "__main__":
    main()
