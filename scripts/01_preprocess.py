"""Step 01: load all raw runs, preprocess them uniformly, and save to data/processed/.

Outputs:
    runs_manifest.csv          one row per run (including excluded runs)
    runs_qc.csv                quality-check metrics per run, with a qc_pass flag
    trajectories_10hz.parquet  all trajectories in a common format, keyed by run_id

Usage (from the repository root):  python scripts/01_preprocess.py
"""
import sys
from pathlib import Path

# Make "src" importable when this file is run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_paths, load_yaml
from src.data.loaders import build_manifest, load_raw_run
from src.data.preprocess import preprocess_run


def main():
    paths = load_paths()
    vehicles = load_yaml("vehicles.yaml")
    cfg = load_yaml("experiment.yaml")
    out_dir = paths["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Manifest
    manifest = build_manifest(paths, vehicles)
    manifest.to_csv(out_dir / "runs_manifest.csv", index=False)
    use = manifest[~manifest["exclude"]]
    print(f"Manifest: {len(manifest)} runs, {manifest['exclude'].sum()} excluded, "
          f"{len(use)} to process")

    # 2. Process every run
    trajs, qcs = [], []
    for i, row in enumerate(use.to_dict("records"), start=1):
        raw = load_raw_run(row)

        # Safety check: actual sampling interval must match the config
        dt_actual = np.median(np.diff(raw["time"].values))
        dt_expected = 1.0 / cfg["preprocess"]["native_hz"][row["powertrain"]]
        if abs(dt_actual - dt_expected) > 1e-3:
            raise ValueError(f"{row['run_id']}: sampling interval {dt_actual:.4f} s "
                             f"does not match config ({dt_expected} s)")

        df, qc = preprocess_run(raw, row["powertrain"], cfg, vehicle=row["vehicle"])
        df.insert(0, "run_id", row["run_id"])
        trajs.append(df)
        qcs.append({"run_id": row["run_id"], "n_raw": len(raw), **qc})
        if i % 50 == 0:
            print(f"  processed {i}/{len(use)}")

    # 3. Save
    traj = pd.concat(trajs, ignore_index=True)
    qc = pd.DataFrame(qcs)
    qc["qc_kinematic"] = qc["kin_resid_max"] <= cfg["qc"]["max_kin_resid"]
    # Runs without standstill have a NaN median gap; NaN < threshold is False,
    # so these runs pass the standstill check.
    qc["qc_standstill"] = ~(qc["standstill_gap_median"] < cfg["qc"]["min_standstill_gap"])
    qc["qc_pass"] = qc["qc_kinematic"] & qc["qc_standstill"]
    traj.to_parquet(out_dir / "trajectories_10hz.parquet", index=False)
    qc.to_csv(out_dir / "runs_qc.csv", index=False)
    # copies without local file paths, published with the results
    summary_dir = REPO_ROOT / "results" / "data_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest.drop(columns="path").to_csv(summary_dir / "runs_manifest.csv", index=False)
    qc.to_csv(summary_dir / "runs_qc.csv", index=False)

    # 4. Summary
    summary = (qc.merge(use[["run_id", "powertrain", "vehicle"]], on="run_id")
                 .groupby(["powertrain", "vehicle"])
                 .agg(runs=("run_id", "size"),
                      runs_qc_pass=("qc_pass", "sum"),
                      samples=("n_samples", "sum"),
                      kin_slope_min=("kin_slope", "min"),
                      kin_slope_max=("kin_slope", "max")))
    print()
    print(summary.round(3).to_string())

    failed = qc[~qc["qc_kinematic"]]
    bad_stop = qc[~qc["qc_standstill"]]
    print(f"\nRuns failing QC (median standstill net gap < "
          f"{cfg['qc']['min_standstill_gap']} m): {len(bad_stop)}")
    if len(bad_stop):
        print(bad_stop[["run_id", "standstill_gap_median"]].to_string(index=False))
    print(f"\nRuns failing QC (kin_resid_max > {cfg['qc']['max_kin_resid']} m/s): "
          f"{len(failed)}")
    if len(failed):
        print(failed[["run_id", "kin_slope", "kin_resid_max"]]
              .sort_values("kin_resid_max", ascending=False)
              .round(3).to_string(index=False))
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
