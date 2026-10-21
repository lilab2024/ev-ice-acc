"""Tables for the revised paper. Each function returns a DataFrame."""
import re

import pandas as pd

GAP_ORDER = {"min": 0, "max": 1, "Short": 0, "Medium": 1, "Long": 2, "XLong": 3}


def dataset_overview(manifest, qc, main_runs):
    """Table 1: vehicles, gap settings, runs, duration and speed range.

    manifest : runs_manifest.csv (all runs, including excluded vehicles)
    qc       : runs_qc.csv (one row per processed run, with qc_pass and duration_s)
    main_runs: run table of the main selection (select_runs(..., 'main'))
    """
    runs = manifest[~manifest["exclude"]].merge(
        qc[["run_id", "qc_pass", "duration_s", "v_f_min", "v_f_max"]], on="run_id")
    runs["main"] = runs["run_id"].isin(main_runs["run_id"])
    rows = []
    for (pt, veh), g in runs.groupby(["powertrain", "vehicle"]):
        m = g[g["main"]]
        gaps = sorted(g["gap"].unique(), key=lambda x: GAP_ORDER.get(x, 9))
        rows.append({
            "Powertrain": pt, "Vehicle": veh, "Gap settings": "/".join(gaps),
            "Runs": len(g), "Runs passing QC": int(g["qc_pass"].sum()),
            "Runs (main)": len(m),
            "Duration (h)": round(g["duration_s"].sum() / 3600, 2),
            "Duration main (h)": round(m["duration_s"].sum() / 3600, 2),
            "Speed range main (m/s)": f"{m['v_f_min'].min():.1f}-{m['v_f_max'].max():.1f}",
        })
    out = pd.DataFrame(rows).sort_values(["Powertrain", "Vehicle"], ascending=[False, True])
    return out.reset_index(drop=True)


def preprocessing_comparison(qc, manifest):
    """Table 2: data handling in the submitted paper vs. the revised analysis."""
    runs = manifest[~manifest["exclude"]].merge(qc, on="run_id")
    n = runs.groupby("powertrain")["n_samples"].sum()
    n_fail = int((~runs["qc_pass"]).sum())
    rows = [
        ("ICE file reading", "first row read as header (lost)",
         "header=None; all rows kept"),
        ("EV sampling rate", "stated as 25 Hz", "50 Hz as provided (verified)"),
        ("Common sampling rate", "none (ICE 10 Hz, EV 50 Hz)", "both resampled to 10 Hz"),
        ("Speed smoothing", "EV: provider-smoothed; ICE: raw",
         "raw speeds, identical Savitzky-Golay filter (1 s window)"),
        ("Acceleration", "point-to-point differences (forward / backward)",
         "derivative of the same filter"),
        ("Quality control", "none",
         f"kinematic check ds/dt vs v_l - v_f; {n_fail} runs removed"),
        ("Unit of analysis", "pooled samples", "runs, configurations, events"),
        ("Samples (10 Hz)", "ICE 287,523; EV 533,572 (50 Hz)",
         f"ICE {n.get('ICE', 0):,}; EV {n.get('EV', 0):,}"),
    ]
    return pd.DataFrame(rows, columns=["Item", "Submitted paper", "Revised analysis"])


def steady_headway_table(steady):
    """Table 3: median steady-state time headway (s) at matched speed and gap setting.

    steady: steady_headway.csv (powertrain, config, speed_bin, headway, n)
    ICE configurations of a gap group are shown as 'min-max (median)'; EV configurations
    individually, with a flag whether all their values lie inside the ICE range.
    """
    groups = {"Shortest": ("min", ["Short"]), "Longest": ("max", ["XLong"])}
    s = steady.dropna(subset=["headway"]).copy()
    s["gap"] = s["config"].str.split().str[-1]
    ev_bins = sorted(s.loc[s["powertrain"] == "EV", "speed_bin"].unique(),
                     key=lambda b: float(b.strip("([]").split(",")[0]))
    ice_bins = set(s.loc[s["powertrain"] == "ICE", "speed_bin"])
    ev_bins = [b for b in ev_bins if b in ice_bins]          # comparable bins only
    rows = []
    for group, (ice_gap, ev_gaps) in groups.items():
        ice = s[(s["powertrain"] == "ICE") & (s["gap"] == ice_gap)]
        rng = {b: (ice.loc[ice["speed_bin"] == b, "headway"].min(),
                   ice.loc[ice["speed_bin"] == b, "headway"].max(),
                   ice.loc[ice["speed_bin"] == b, "headway"].median()) for b in ev_bins}
        rows.append({"Gap group": group, "Configuration": f"ICE {ice_gap} (6 vehicles)",
                     **{b: f"{lo:.2f}-{hi:.2f} ({md:.2f})" for b, (lo, hi, md) in rng.items()},
                     "Inside ICE range": "", "Below / above": ""})
        for cfg_name in sorted(s.loc[(s["powertrain"] == "EV") & s["gap"].isin(ev_gaps),
                                     "config"].unique()):
            ev = s[s["config"] == cfg_name].set_index("speed_bin")["headway"]
            comp = [b for b in ev.index if b in rng]
            inside = sum(rng[b][0] <= ev[b] <= rng[b][1] for b in comp)
            below = sum(ev[b] < rng[b][0] for b in comp)
            above = sum(ev[b] > rng[b][1] for b in comp)
            rows.append({"Gap group": group, "Configuration": cfg_name,
                         **{b: f"{ev[b]:.2f}" if b in ev.index else "-" for b in ev_bins},
                         "Inside ICE range": f"{inside}/{len(comp)}",
                         "Below / above": f"{below} / {above}"})
    return pd.DataFrame(rows)


def event_comparison_table(measured, cf_idm, cf_ovrv):
    """Table 3: ICE vs EV response to leader decelerations, measured and counterfactual.

    measured: event_config_medians.csv (drop_bin, powertrain, config, metrics)
    cf_*    : counterfactual_<model>_medians.csv (leader_set, powertrain, config, metrics)
    Each configuration counts once. Difference = ICE median - EV median with a two-sample
    bootstrap 95 % CI; p = exact two-sided Mann-Whitney test.
    """
    from scipy import stats
    from src.analysis.uncertainty import two_sample_difference

    sources = [(f"Measured, leader drop {b} m/s", measured[measured["drop_bin"] == b])
               for b in ["3-5", "7-12"]]
    for model, cf in [("IDM", cf_idm), ("OVRV", cf_ovrv)]:
        for leader in ["EV-experiment leader", "ICE-experiment leader"]:
            sources.append((f"Same leader ({model}), {leader}", cf[cf["leader_set"] == leader]))

    rows = []
    for source, d in sources:
        for metric, label in [("amplification", "Speed amplification"),
                              ("overshoot", "Overshoot (m/s)"),
                              ("delay_s", "Response delay (s)")]:
            ice = d.loc[d["powertrain"] == "ICE", metric].dropna()
            ev = d.loc[d["powertrain"] == "EV", metric].dropna()
            est, lo, hi = two_sample_difference(ice, ev)
            p = stats.mannwhitneyu(ice, ev, alternative="two-sided", method="exact").pvalue
            rows.append({"Source": source, "Metric": label,
                         "ICE median": round(ice.median(), 2), "EV median": round(ev.median(), 2),
                         "ICE - EV [95% CI]": f"{est:.2f} [{lo:.2f}, {hi:.2f}]",
                         "n ICE / EV": f"{len(ice)} / {len(ev)}", "p (exact)": round(p, 4)})
    return pd.DataFrame(rows)


def calibration_table(params, folds, model, v_ref, excluded):
    """Table 4: calibrated IDM parameters, equilibrium headway and spacing RMSE.

    params: main_idm_params.csv; folds: main_idm_folds.csv (cross-validation)
    Equilibrium headway at v_ref with the range over the CV folds.
    """
    rows = []
    for r in params.to_dict("records"):
        p = {k: r[k] for k in model.PARAM_NAMES}
        f = folds[folds["config"] == r["config"]]
        h_folds = [model.equilibrium_spacing(v_ref, {k: q[k] for k in model.PARAM_NAMES}) / v_ref
                   for q in f.to_dict("records")]
        note = []
        if isinstance(r.get("at_bound"), str) and r["at_bound"]:
            note.append(f"at bound: {r['at_bound']}")
        if r["config"] in excluded:
            note.append("not used in main results")
        rows.append({"Powertrain": r["powertrain"], "Configuration": r["config"],
                     "s0 (m)": round(r["s0"], 2), "tau (s)": round(r["tau"], 2),
                     "alpha (m/s2)": round(r["alpha"], 2), "beta (m/s2)": round(r["beta"], 2),
                     f"h_eq at {v_ref:.0f} m/s (s)":
                         f"{model.equilibrium_spacing(v_ref, p) / v_ref:.2f} "
                         f"[{min(h_folds):.2f}, {max(h_folds):.2f}]",
                     "RMSE train (m)": round(r["objective"], 2),
                     "RMSE CV test (m)": round(f["test_spacing_rmse"].mean(), 2),
                     "Note": "; ".join(note)})
    out = pd.DataFrame(rows)
    out["_gap"] = out["Configuration"].str.split().str[-1].map(GAP_ORDER)
    out = out.sort_values(["Powertrain", "_gap", "Configuration"], ascending=[False, True, True])
    return out.drop(columns="_gap").reset_index(drop=True)


MODEL_VARIANTS = {"Newell": "base_newell", "Gipps": "base_gipps", "OVRV": "main_ovrv",
                  "OVRV + lag": "ext_ovrv_lag", "IDM": "main_idm", "IDM + lag": "ext_idm_lag"}
MODEL_PAIRS = [("IDM + lag", "IDM"), ("OVRV + lag", "OVRV"), ("IDM", "OVRV"),
               ("IDM", "Gipps"), ("IDM", "Newell")]


def model_comparison_tables(cv_dir, excluded):
    """Table 5: cross-validated test spacing RMSE of all models and paired comparisons.

    Returns (summary, pairs). Means and CIs are over configurations (bootstrap).
    """
    from src.analysis.uncertainty import bootstrap_ci, paired_difference

    test = {}
    for name, variant in MODEL_VARIANTS.items():
        f = pd.read_csv(cv_dir / f"{variant}_folds.csv")
        f = f[~f["config"].isin(excluded)]
        test[name] = f.groupby(["powertrain", "config"])["test_spacing_rmse"].mean()
    test = pd.DataFrame(test)

    groups = [("ICE", test.xs("ICE")), ("EV", test.xs("EV")), ("All", test)]
    summary = []
    for name in MODEL_VARIANTS:
        row = {"Model": name}
        for g, d in groups:
            m, lo, hi = bootstrap_ci(d[name])
            row[f"{g} (n={len(d)})"] = f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
        summary.append(row)

    pairs = []
    for a, b in MODEL_PAIRS:
        for g, d in groups:
            r = paired_difference(d[a], d[b])
            pairs.append({"Comparison": f"{a} - {b}", "Group": g,
                          "Mean difference (m) [95% CI]":
                              f"{r['mean_diff']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]",
                          "First model better": f"{r['a_smaller']}/{r['n']}",
                          "Wilcoxon p": round(r["wilcoxon_p"], 4)})
    return pd.DataFrame(summary), pd.DataFrame(pairs)


def stability_table(homogeneous, v_ref, excluded, unreliable_ovrv=("Polestar 2 Short",)):
    """Table 6: string stability at v_ref (lambda2, margin M with CV-fold range, max|G|)."""
    h = homogeneous[homogeneous["v_e"] == v_ref]
    idm_h = h[h["model"] == "idm"].set_index("config")
    ovrv_h = h[h["model"] == "ovrv"].set_index("config")
    rows = []
    for config, r in idm_h.iterrows():
        o = ovrv_h.loc[config]
        note = []
        if config in excluded:
            note.append("not used in main results")
        if config in unreliable_ovrv:
            note.append("OVRV unreliable (single steady speed)")
        rows.append({"Powertrain": r["powertrain"], "Configuration": config,
                     "lambda2 IDM (1/s)": round(r["lambda2"], 3),
                     "M IDM [fold range]":
                         f"{r['M']:.2f} [{r['M_fold_min']:.2f}, {r['M_fold_max']:.2f}]",
                     "M OVRV [fold range]":
                         f"{o['M']:.2f} [{o['M_fold_min']:.2f}, {o['M_fold_max']:.2f}]",
                     "max|G| IDM": round(r["max_gain"], 3),
                     "String stable": "yes" if r["stable"] else "no",
                     "Note": "; ".join(note)})
    out = pd.DataFrame(rows)
    out["_gap"] = out["Configuration"].str.split().str[-1].map(GAP_ORDER)
    out = out.sort_values(["Powertrain", "_gap", "Configuration"], ascending=[False, True, True])
    return out.drop(columns="_gap").reset_index(drop=True)


def steady_headway_latex(t):
    """Publication LaTeX for the steady-state time-gap table (booktabs, no vertical rules)."""
    bins = [c for c in t.columns if str(c).startswith("(")]
    labels = ["{:.0f}--{:.0f}".format(*(float(x) for x in b.strip("([]").split(","))) for b in bins]
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r" & & \multicolumn{3}{c}{Follower speed bin (m/s)} & \multicolumn{2}{c}{EV bins vs. ICE range} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
             r"Setting & Configuration & " + " & ".join(labels) + r" & Within & Below / above \\",
             r"\midrule"]
    prev = None
    for r in t.to_dict("records"):
        if prev is not None and r["Gap group"] != prev:
            lines.append(r"\midrule")
        setting = r["Gap group"] if r["Gap group"] != prev else ""
        prev = r["Gap group"]
        vals = [re.sub(r"(\d)-(\d)", r"\1--\2", str(r[b])) for b in bins]
        inside = r["Inside ICE range"] if pd.notna(r["Inside ICE range"]) else ""
        below = r["Below / above"] if pd.notna(r["Below / above"]) else ""
        lines.append(f"{setting} & {r['Configuration']} & " + " & ".join(vals) + f" & {inside} & {below} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def _fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def event_comparison_latex(t):
    """Publication LaTeX for Table 3: grouped by source, booktabs, no vertical rules."""
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Metric & ICE & EV & ICE $-$ EV [95\% CI] & $p$ \\", r"\midrule"]
    for i, (source, g) in enumerate(t.groupby("Source", sort=False)):
        if i:
            lines.append(r"\addlinespace")
        source = re.sub(r"(\d)-(\d)", r"\1--\2", source)
        source = re.sub(r"Same leader \((\w+)\), (EV|ICE)-experiment leader",
                        r"Model-based (\1), leaders of the \2 campaign", source)
        n_ice, n_ev = (x.strip() for x in g["n ICE / EV"].iloc[0].split("/"))
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{source}}} "
                     rf"($n_{{\mathrm{{ICE}}}} = {n_ice}$, $n_{{\mathrm{{EV}}}} = {n_ev}$)}} \\")
        for r in g.to_dict("records"):
            lines.append(rf"\quad {r['Metric']} & {r['ICE median']:.2f} & {r['EV median']:.2f} & "
                         rf"{r['ICE - EV [95% CI]']} & {_fmt_p(r['p (exact)'])} \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def calibration_latex(t, v_ref):
    """Publication LaTeX for Table 4 (IDM parameters, equilibrium time gap, RMSE)."""
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r" & & & & & & \multicolumn{2}{c}{RMSE (m)} \\", r"\cmidrule(lr){7-8}",
             rf"Configuration & $s_0$ (m) & $\tau$ (s) & $\alpha$ (m/s$^2$) & $\beta$ (m/s$^2$) & "
             rf"$h_{{\mathrm{{eq}}}}$ at {v_ref:.0f} m/s (s) & Train & Test \\", r"\midrule"]
    for i, (pt, g) in enumerate(t.groupby("Powertrain", sort=False)):
        if i:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{pt}}}}} \\")
        heq = [c for c in g.columns if c.startswith("h_eq")][0]
        for r in g.to_dict("records"):
            mark = "$^{a}$" if r["Note"] else ""
            lines.append(rf"\quad {r['Configuration']}{mark} & {r['s0 (m)']:.2f} & "
                         rf"{r['tau (s)']:.2f} & {r['alpha (m/s2)']:.2f} & {r['beta (m/s2)']:.2f} & "
                         rf"{r[heq]} & {r['RMSE train (m)']:.2f} & {r['RMSE CV test (m)']:.2f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def model_rmse_latex(t):
    """Publication LaTeX for Table 5 (cross-validated test RMSE of all models)."""
    d = t.rename(columns=lambda c: re.sub(r"\(n=(\d+)\)", r"($n = \1$)", c))
    return d.to_latex(index=False, escape=False, column_format="lccc")


def model_pairs_latex(t):
    """Publication LaTeX for Table S2 (paired model comparisons)."""
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Comparison & Group & Mean difference (m) [95\% CI] & First better & "
             r"Wilcoxon $p$ \\", r"\midrule"]
    for i, (comp, g) in enumerate(t.groupby("Comparison", sort=False)):
        if i:
            lines.append(r"\addlinespace")
        comp = comp.replace(" - ", r" $-$ ")
        for j, r in enumerate(g.to_dict("records")):
            p = _fmt_p(r["Wilcoxon p"]) if pd.notna(r["Wilcoxon p"]) else "--"
            lines.append(rf"{comp if j == 0 else ''} & {r['Group']} & "
                         rf"{r['Mean difference (m) [95% CI]']} & {r['First model better']} & {p} \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def stability_latex(t):
    """Publication LaTeX for Table 6 (lambda2, margin M with fold range, peak gain)."""
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Configuration & $\lambda_2$ (IDM, 1/s) & $M$ (IDM) & $M$ (OVRV) & "
             r"$\max_\omega |G|$ (IDM) \\", r"\midrule"]
    for i, (pt, g) in enumerate(t.groupby("Powertrain", sort=False)):
        if i:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{pt}}}}} \\")
        for r in g.to_dict("records"):
            note = str(r["Note"]) if pd.notna(r["Note"]) else ""
            mark = "$^{a}$" if "not used" in note else ("$^{b}$" if "OVRV" in note else "")
            lines.append(rf"\quad {r['Configuration']}{mark} & {r['lambda2 IDM (1/s)']:.2f} & "
                         rf"{r['M IDM [fold range]']} & {r['M OVRV [fold range]']} & "
                         rf"{r['max|G| IDM']:.3f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def _sorted_by_setting(df):
    df = df.copy()
    df["_gap"] = df["Configuration"].str.split().str[-1].map(GAP_ORDER)
    df = df.sort_values(["Powertrain", "_gap", "Configuration"], ascending=[False, True, True])
    return df.drop(columns="_gap").reset_index(drop=True)


def _bound_text(x):
    return x if isinstance(x, str) and x else ""


def identifiability_table(main, free):
    """Table S3: IDM with v0 and delta free versus fixed (v0 = 40 m/s, delta = 20)."""
    m = main.set_index("config")
    rows = []
    for r in free.to_dict("records"):
        fixed = m.loc[r["config"], "objective"]
        rows.append({"Powertrain": r["powertrain"], "Configuration": r["config"],
                     "v0": r["v0"], "delta": r["delta"], "tau": r["tau"],
                     "RMSE free": r["objective"], "RMSE fixed": fixed,
                     "Change (%)": 100 * (r["objective"] - fixed) / fixed,
                     "At bound": _bound_text(r["at_bound"])})
    return _sorted_by_setting(pd.DataFrame(rows))


def _bound_latex(s):
    """'delta=low;tau=low' -> '$\\delta$ lower; $\\tau$ lower'."""
    if not s:
        return "--"
    sym = {"v0": r"$v_0$", "delta": r"$\delta$", "tau": r"$\tau$", "s0": r"$s_0$",
           "alpha": r"$\alpha$", "beta": r"$\beta$"}
    parts = []
    for p in s.split(";"):
        k, v = p.split("=")
        parts.append(sym.get(k, k) + (" lower" if v == "low" else " upper"))
    return "; ".join(parts)


def _signed_percent(x):
    if abs(x) < 0.05:
        return "0.0"
    return "$-$" + f"{abs(x):.1f}" if x < 0 else f"+{x:.1f}"


def identifiability_latex(t):
    lines = [r"\begin{tabular}{lccccccl}", r"\toprule",
             r"Configuration & $v_0$ (m/s) & $\delta$ & $\tau$ (s) & RMSE free (m) & "
             r"RMSE fixed (m) & Change (\%) & At bound (free) \\", r"\midrule"]
    for i, (pt, g) in enumerate(t.groupby("Powertrain", sort=False)):
        if i:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{pt}}}}} \\")
        for r in g.to_dict("records"):
            lines.append(rf"\quad {r['Configuration']} & {r['v0']:.1f} & {r['delta']:.1f} & "
                         rf"{r['tau']:.2f} & {r['RMSE free']:.2f} & {r['RMSE fixed']:.2f} & "
                         rf"{_signed_percent(r['Change (%)'])} & {_bound_latex(r['At bound'])} \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def allruns_table(main_idm, all_idm, main_ovrv, all_ovrv, v_ref=20.0):
    """Table S4: calibration on all runs including stops (EV configurations)."""
    mi, mo = main_idm.set_index("config"), main_ovrv.set_index("config")
    ai, ao = all_idm.set_index("config"), all_ovrv.set_index("config")
    heq = lambda p: (p["s0"] + p["tau"] * v_ref) / v_ref
    rows = []
    for config in ai.index[ai["powertrain"] == "EV"]:
        rows.append({"Powertrain": "EV", "Configuration": config,
                     "Runs main": mi.loc[config, "n_runs"] if "n_runs" in mi.columns else float("nan"),
                     "Runs all": ai.loc[config, "n_runs"],
                     "IDM main": mi.loc[config, "objective"], "IDM all": ai.loc[config, "objective"],
                     "OVRV main": mo.loc[config, "objective"], "OVRV all": ao.loc[config, "objective"],
                     "s0 main": mi.loc[config, "s0"], "s0 all": ai.loc[config, "s0"],
                     "heq main": heq(mi.loc[config]), "heq all": heq(ai.loc[config]),
                     "At bound (IDM, all runs)": _bound_text(ai.loc[config, "at_bound"])})
    return _sorted_by_setting(pd.DataFrame(rows))


def allruns_latex(t):
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcccccccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Runs} & \multicolumn{2}{c}{IDM RMSE (m)} & "
             r"\multicolumn{2}{c}{OVRV RMSE (m)} & \multicolumn{2}{c}{$s_0$ (m)} & "
             r"\multicolumn{2}{c}{$h_{\mathrm{eq}}$ (s)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
             r"Configuration & Main & All & Main & All & Main & All & Main & All & Main & All \\",
             r"\midrule"]
    for r in t.to_dict("records"):
        n = "--" if pd.isna(r["Runs main"]) else f"{int(r['Runs main'])}"
        lines.append(rf"{r['Configuration']} & {n} & {int(r['Runs all'])} & {r['IDM main']:.2f} & "
                     rf"{r['IDM all']:.2f} & {r['OVRV main']:.2f} & {r['OVRV all']:.2f} & "
                     rf"{r['s0 main']:.2f} & {r['s0 all']:.2f} & {r['heq main']:.2f} & {r['heq all']:.2f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def dataset_overview_latex(t):
    """Publication LaTeX for Table 1 (booktabs, no vertical rules)."""
    abbr = {"Short": "S", "Medium": "M", "Long": "L", "XLong": "XL"}
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & & \multicolumn{3}{c}{Runs} & \multicolumn{2}{c}{Duration (h)} & \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
             r"Vehicle & Gap settings & All & QC & Main & All & Main & Speed (m/s) \\", r"\midrule"]
    for i, (pt, g) in enumerate(t.groupby("Powertrain", sort=False)):
        if i:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{pt}}}}} \\")
        for r in g.to_dict("records"):
            gaps = "/".join(abbr.get(x, x) for x in r["Gap settings"].split("/"))
            rng = r["Speed range main (m/s)"].replace("-", "--")
            lines.append(rf"\quad {r['Vehicle']} & {gaps} & {r['Runs']} & {r['Runs passing QC']} & "
                         rf"{r['Runs (main)']} & {r['Duration (h)']:.1f} & {r['Duration main (h)']:.1f} & {rng} \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)
