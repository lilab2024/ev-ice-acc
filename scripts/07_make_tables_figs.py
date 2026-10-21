"""Step 07: all tables and figures of the revised paper.

Requires the outputs of steps 03-06 and results/data_summary/. The example runs for
Figures 2 and 5 are extracted from the processed data once and stored in
results/data_summary/example_runs.csv, so the figures can be redrawn without raw data.
Writes results/paper/tables/*.csv|*.tex and results/paper/figures/*.pdf|*.png
(drawn at printed size, journal style of Digital Transportation and Safety).
Usage (from the repository root):  python scripts/07_make_tables_figs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")                       # no display needed
import matplotlib.pyplot as plt
import pandas as pd

from src.calibration.selection import select_runs
from src.config import REPO_ROOT, load_paths, load_yaml
from src.models import idm, ovrv
from src.models.simulate import simulate_follower
from src.reporting import figures, tables
from src.analysis import event_curves
from src.reporting import table_bounds
from src.simulation.fleet import IDM_KEYS, build_fleet
from src.simulation.ring import growth_metrics, simulate_ring

RES = REPO_ROOT / "results"
OUT_T = RES / "paper" / "tables"
OUT_F = RES / "paper" / "figures"
EXAMPLES = RES / "data_summary" / "example_runs.csv"


def save_table(df, name, latex=None):
    """CSV of the table; LaTeX either given (publication format) or a default export."""
    df.to_csv(OUT_T / f"{name}.csv", index=False)
    if latex is None:
        latex = df.to_latex(index=False, escape=True, float_format="%.2f")
    (OUT_T / f"{name}.tex").write_text(latex)
    print(f"  {name}: {len(df)} rows")


def save_fig(fig, name, letter=None):
    if letter:
        figures.add_panel_label(fig, letter)
    fig.savefig(OUT_F / f"{name}.pdf")
    fig.savefig(OUT_F / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  {name}")


def example_runs(cfg, paths):
    """The two example runs, extracted once from the processed data."""
    if not EXAMPLES.exists():
        traj = pd.read_parquet(paths["processed"] / "trajectories_10hz.parquet")
        parts = []
        for key, ex in cfg["examples"].items():
            d = traj[traj["run_id"] == ex["run_id"]].sort_values("t")
            d = d.assign(t=d["t"] - d["t"].iloc[0])
            if ex.get("window_s"):
                lo, hi = ex["window_s"]
                d = d[(d["t"] >= lo) & (d["t"] <= hi)]
                d = d.assign(t=d["t"] - lo)
            parts.append(d[["run_id", "t", "v_l", "v_f", "s"]].assign(example=key))
        pd.concat(parts).to_csv(EXAMPLES, index=False)
    ex = pd.read_csv(EXAMPLES)
    return {k: g.reset_index(drop=True) for k, g in ex.groupby("example")}


def model_fits(run, config):
    """Gaps simulated by the calibrated IDM and OVRV behind the measured leader."""
    out = {}
    for label, model, name in [("IDM", idm, "idm"), ("OVRV", ovrv, "ovrv")]:
        p = pd.read_csv(RES / "calibration" / f"main_{name}_params.csv").set_index("config")
        par = {k: p.loc[config, k] for k in model.PARAM_NAMES}
        out[label] = simulate_follower(model, par, run["t"].values, run["v_l"].values,
                                       run["s"].iloc[0], run["v_f"].iloc[0])["s"]
    return out


def main():
    OUT_T.mkdir(parents=True, exist_ok=True)
    OUT_F.mkdir(parents=True, exist_ok=True)
    for old in list(OUT_F.glob("fig*")) + list(OUT_T.glob("table*")):
        old.unlink()                            # remove files with outdated numbers
    paths, cfg = load_paths(), load_yaml("experiment.yaml")
    excluded = list(cfg["parameter_use"]["exclude_from_main"])
    v_ref = cfg["stability"]["v_ref"]

    # run manifest and QC table without local file paths (published with the results)
    manifest = pd.read_csv(RES / "data_summary" / "runs_manifest.csv")
    qc = pd.read_csv(RES / "data_summary" / "runs_qc.csv")
    stat = RES / "statistics"
    steady = pd.read_csv(stat / "steady_headway.csv")
    measured = pd.read_csv(stat / "event_config_medians.csv")
    cf_idm = pd.read_csv(stat / "counterfactual_idm_medians.csv")
    cf_ovrv = pd.read_csv(stat / "counterfactual_ovrv_medians.csv")
    params = pd.read_csv(RES / "calibration" / "main_idm_params.csv")
    folds = pd.read_csv(RES / "cross_validation" / "main_idm_folds.csv")
    hom = pd.read_csv(RES / "stability" / "homogeneous.csv")
    validation = pd.read_csv(RES / "stability" / "validation.csv")
    mixed = pd.read_csv(RES / "stability" / "mixed.csv")
    ring_a = pd.read_csv(RES / "ring" / "ring_A.csv")
    ring_b = pd.read_csv(RES / "ring" / "ring_B.csv")

    print("Tables")
    t1 = tables.dataset_overview(manifest, qc, select_runs(manifest, qc, cfg, "main"))
    save_table(t1, "table1_dataset", tables.dataset_overview_latex(t1))
    t2 = tables.steady_headway_table(steady)
    save_table(t2, "table2_steady_headway", tables.steady_headway_latex(t2))
    t3 = tables.event_comparison_table(measured, cf_idm, cf_ovrv)
    save_table(t3, "table3_events", tables.event_comparison_latex(t3))
    t4 = tables.calibration_table(params, folds, idm, v_ref, excluded)
    save_table(t4, "table4_calibration", tables.calibration_latex(t4, v_ref))
    t5a, t5b = tables.model_comparison_tables(RES / "cross_validation", excluded)
    save_table(t5a, "table5_model_rmse", tables.model_rmse_latex(t5a))
    save_table(t5b, "tableS2_model_pairs", tables.model_pairs_latex(t5b))   # supplementary
    cal = RES / "calibration"                                       # supplementary tables
    s3 = tables.identifiability_table(pd.read_csv(cal / "main_idm_params.csv"),
                                      pd.read_csv(cal / "supp_idm_free_params.csv"))
    save_table(s3, "tableS3_identifiability", tables.identifiability_latex(s3))
    s4 = tables.allruns_table(pd.read_csv(cal / "main_idm_params.csv"),
                              pd.read_csv(cal / "supp_idm_allruns_params.csv"),
                              pd.read_csv(cal / "main_ovrv_params.csv"),
                              pd.read_csv(cal / "supp_ovrv_allruns_params.csv"))
    save_table(s4, "tableS4_allruns", tables.allruns_latex(s4))
    t6 = tables.stability_table(hom, v_ref, excluded)
    save_table(t6, "table6_stability", tables.stability_latex(t6))
    sb = table_bounds.bounds_table(cfg)
    save_table(sb, "tableS1_bounds", table_bounds.bounds_latex(sb))

    print("Figures")
    save_fig(figures.fig_framework(), "fig1_framework")

    ex = example_runs(cfg, paths)                                       # Figure 2: data
    for key, (l1, l2) in [("ice", ("a", "b")), ("ev", ("c", "d"))]:
        save_fig(figures.fig_trajectory_speed(ex[key]), f"fig2{l1}_{key}_speed", l1)
        save_fig(figures.fig_trajectory_gap(ex[key]), f"fig2{l2}_{key}_gap", l2)

    for g, letter in [("shortest", "a"), ("longest", "b")]:           # Figure 3
        save_fig(figures.fig_steady_headway(steady, g), f"fig3{letter}_steady_{g}", letter)

    save_fig(figures.fig_event_comparison(measured, cf_idm, "amplification"),   # Figure 4
             "fig4a_amplification", "a")
    save_fig(figures.fig_event_comparison(measured, cf_idm, "delay_s"), "fig4b_delay", "b")

    for key, letter in [("ice", "a"), ("ev", "b")]:                    # Figure 5: model fit
        run = ex[key]
        save_fig(figures.fig_model_fit(run, model_fits(run, cfg["examples"][key]["config"])),
                 f"fig5{letter}_fit_{key}", letter)

    save_fig(figures.fig_validation(validation), "fig6_validation")
    save_fig(figures.fig_stability_margin(hom), "fig7_stability_margin")
    save_fig(figures.fig_mixed_acc(mixed, v_ref), "fig8a_mixed_acc", "a")
    save_fig(figures.fig_mixed_human(mixed, v_ref), "fig8b_mixed_human", "b")

    # Figure 9: space-time diagrams (25 veh/km, first seed, small perturbation)
    rc = cfg["ring"]
    acc = params[~params["config"].isin(excluded)]
    pools = {pt: [{k: r[k] for k in IDM_KEYS} for r in acc[acc["powertrain"] == pt].to_dict("records")]
             for pt in ["ICE", "EV"]}
    pools["HV"] = [dict(cfg["human_driver"]["base"], T_a=rc["human_lag"])]
    L = rc["n_vehicles"] / 25 * 1000
    for letter, (base, new, share) in zip("abc", [("ICE", "EV", 0.0), ("ICE", "EV", 1.0),
                                                   ("HV", "EV", 0.0)]):
        _, p = build_fleet(0, rc["n_vehicles"], base, new, share, pools)
        sim = simulate_ring(p, L, rc, decel=rc["growth"]["perturbation_decel"])
        m = growth_metrics(sim, rc)
        t_exit = rc["perturbation"]["start_s"] + m["time_to_exit"] if m["exited"] else None
        save_fig(figures.fig_space_time(sim["t"], sim["v"], rc["speed_cap"], t_exit),
                 f"fig9{letter}_space_time", letter)

    fa, fb = figures.fig_ring_growth(ring_a)                            # Figure 10
    save_fig(fa, "fig10a_ring_growth_acc", "a")
    save_fig(fb, "fig10b_ring_growth_human", "b")
    save_fig(figures.fig_flow_density(ring_b), "figS1_flow_density")
    # Added figures: gap definition (Data section) and event-aligned response (Section 3.2)
    save_fig(figures.fig_gap_definition(), "figA_gap_definition")
    curves = event_curves.load_or_build(
        paths, RES, cfg["calibration"]["run_selection"]["main"]["ev_reduced_mph"])
    delay = measured[measured["drop_bin"] == "3-5"].groupby("powertrain")["delay_s"].median()
    for pt, letter in [("ICE", "a"), ("EV", "b")]:
        save_fig(figures.fig_event_response(curves, pt, delay[pt]),
                 f"figB{letter}_response_{pt.lower()}", letter)
    print(f"\nOutputs written to {RES / 'paper'}")


if __name__ == "__main__":
    main()


# --- layout fixes for Tables 4, 7 and 10 (src/reporting/latex_fixes.py) ---
from pathlib import Path as _P
import sys as _sys
_root = _P(__file__).resolve().parents[1]
_sys.path.insert(0, str(_root))
from src.reporting.latex_fixes import apply_all as _apply_all
_apply_all(_root / "results" / "paper" / "tables")
