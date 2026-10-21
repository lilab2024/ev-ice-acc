"""Select the runs used to calibrate each configuration (vehicle x gap setting)."""
import pandas as pd


def select_runs(manifest, qc, cfg, selection="main"):
    """Return the runs for calibration, one row per run, with a 'config' column.

    selection: 'main' or 'supplementary' (see calibration.run_selection in experiment.yaml).
    ICE runs are always kept; EV runs are filtered by their reduced speed.
    """
    rule = cfg["calibration"]["run_selection"][selection]
    runs = manifest.merge(qc[["run_id", "qc_pass"]], on="run_id")
    runs = runs[~runs["exclude"] & runs["qc_pass"]]
    keep = (runs["powertrain"] == "ICE") | runs["v_reduced_mph"].isin(rule["ev_reduced_mph"])
    runs = runs[keep].copy()
    runs["config"] = runs["vehicle"] + " " + runs["gap"]
    return runs.sort_values(["powertrain", "config", "run_id"]).reset_index(drop=True)


def list_configs(runs):
    """Configurations in a fixed order: ICE first, then EV, alphabetically within each."""
    # Descending powertrain order puts "ICE" before "EV"
    return (runs[["powertrain", "config"]].drop_duplicates()
            .sort_values(["powertrain", "config"], ascending=[False, True])["config"].tolist())
