"""Step 04: statistical comparison of ICE and EV ACC platforms.

Requires the outputs of steps 01 (processed data) and 03 (calibrated parameters).
Usage (from the repository root):  python scripts/04_statistics.py
Outputs: results/statistics/*.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.analysis.comparison import (config_level_tests, describe_by_vehicle,
                                     extract_events, per_config_medians,
                                     pooled_comparison, steady_state_headway)
from src.analysis.counterfactual import same_leader_events
from src.config import REPO_ROOT, load_paths, load_yaml
from src.models import idm, ovrv

MODELS = {"idm": idm, "ovrv": ovrv}
META = ["powertrain", "vehicle", "gap", "profile", "desired_mph", "v_free_mph",
        "v_reduced_mph", "run"]
# Common leaders for the counterfactual test
ICE_LEADER_RUNS = ["ICE_VehA_osc_min", "ICE_VehA_dip_min"]
EV_LEADER_FILTER = dict(vehicle="IONIQ 5", gap="Short", desired_mph=0, run="1")


def load_data(paths):
    traj = pd.read_parquet(paths["processed"] / "trajectories_10hz.parquet")
    manifest = pd.read_csv(paths["processed"] / "runs_manifest.csv")
    qc = pd.read_csv(paths["processed"] / "runs_qc.csv")
    runs = manifest.merge(qc[["run_id", "qc_pass"]], on="run_id")
    runs = runs[~runs["exclude"] & runs["qc_pass"]].copy()
    runs["run"] = runs["run"].astype(str)
    return traj.merge(runs[["run_id"] + META], on="run_id"), runs


def main():
    paths = load_paths()
    cfg = load_yaml("experiment.yaml")
    out = REPO_ROOT / "results" / "statistics"
    out.mkdir(parents=True, exist_ok=True)
    data, runs = load_data(paths)
    main_sel = cfg["calibration"]["run_selection"]["main"]["ev_reduced_mph"]
    print(f"Data: {data['run_id'].nunique()} runs, {len(data)} samples")

    # 1. Descriptive statistics and the paper's pooled comparison
    describe_by_vehicle(data, "v_f").to_csv(out / "descriptive_speed.csv")
    describe_by_vehicle(data, "s").to_csv(out / "descriptive_spacing.csv")
    pd.concat([pooled_comparison(data, "Full dataset"),
               pooled_comparison(data[data["v_f"].between(15, 25)], "Overlap 15-25 m/s")]
              ).to_csv(out / "pooled_comparison.csv", index=False)
    print("1. descriptive statistics and pooled comparison written")

    # 2. Steady-state headway at matched speed and gap setting
    steady_state_headway(data).to_csv(out / "steady_headway.csv", index=False)
    print("2. steady-state headway written")

    # 3. Event-level analysis (main run selection)
    ev = extract_events(data, META)
    ev.to_csv(out / "events.csv", index=False)
    ev_main = ev[(ev["powertrain"] == "ICE") | ev["v_reduced_mph"].isin(main_sel)]
    medians = per_config_medians(ev_main)
    medians.to_csv(out / "event_config_medians.csv", index=False)
    tests = config_level_tests(medians)
    tests.to_csv(out / "event_tests.csv", index=False)
    (ev_main.groupby(["drop_bin", "powertrain"], observed=True)["peak_decel_leader"]
            .describe()).to_csv(out / "leader_decel.csv")
    print(f"3. {len(ev)} events ({len(ev_main)} in the main selection); tests:")
    print(tests.round(4).to_string(index=False))

    # 4. Counterfactual 'same leader' test, for IDM and OVRV
    f = EV_LEADER_FILTER
    ev_leader_ids = runs[(runs["vehicle"] == f["vehicle"]) & (runs["gap"] == f["gap"])
                         & (runs["desired_mph"] == f["desired_mph"]) & (runs["run"] == f["run"])
                         & runs["v_reduced_mph"].isin(main_sel)]["run_id"].tolist()

    def leader_arrays(ids):
        return {i: {c: data.loc[data["run_id"] == i, c].values for c in ["t", "v_l", "a_l"]}
                for i in ids}
    leaders = {"EV-experiment leader": leader_arrays(ev_leader_ids),
               "ICE-experiment leader": leader_arrays(ICE_LEADER_RUNS)}
    excluded = list(cfg["parameter_use"]["exclude_from_main"])

    for name, model in MODELS.items():
        params = pd.read_csv(REPO_ROOT / "results" / "calibration" / f"main_{name}_params.csv")
        params = params[~params["config"].isin(excluded)]
        cf = same_leader_events(model, params, leaders)
        cf.to_csv(out / f"counterfactual_{name}_events.csv", index=False)
        cf_med = per_config_medians(cf, ("leader_set", "powertrain", "config"))
        cf_med.to_csv(out / f"counterfactual_{name}_medians.csv", index=False)
        cf_tests = config_level_tests(cf_med, by="leader_set", levels=list(leaders))
        cf_tests.to_csv(out / f"counterfactual_{name}_tests.csv", index=False)
        print(f"4. counterfactual test with {name.upper()} "
              f"({len(ev_leader_ids)} EV + {len(ICE_LEADER_RUNS)} ICE leader runs):")
        print(cf_tests.round(4).to_string(index=False))

    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
