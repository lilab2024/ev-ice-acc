"""Statistical comparison of ICE and EV ACC platforms."""
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.events import run_events

DROP_BINS = [1.5, 3, 5, 7, 12]
DROP_LABELS = ["1.5-3", "3-5", "5-7", "7-12"]
EVENT_METRICS = ["amplification", "overshoot", "delay_s", "decel_ratio"]


def cohens_d(x, y):
    """Cohen's d with pooled standard deviation."""
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled


def describe_by_vehicle(df, col):
    """Mean, median, min, max, std, n and hours of `col` per vehicle and pooled."""
    def summary(g):
        out = pd.DataFrame({"mean": g.mean(), "median": g.median(), "min": g.min(),
                            "max": g.max(), "std": g.std(), "n": g.size()})
        out["hours"] = out["n"] / 10 / 3600
        return out
    per_vehicle = summary(df.groupby(["powertrain", "vehicle"])[col])
    pooled = summary(df.groupby("powertrain")[col])
    pooled.index = pd.MultiIndex.from_tuples([(p, "ALL") for p in pooled.index])
    return pd.concat([per_vehicle, pooled]).sort_index(ascending=[False, True])


def pooled_comparison(df, scope):
    """Sample-level ICE vs EV comparison as in the submitted paper (Table 6)."""
    ice, ev = df[df["powertrain"] == "ICE"], df[df["powertrain"] == "EV"]
    rows = []
    for col, name in [("s", "Spacing (m)"), ("v_f", "Follow speed (m/s)"),
                      ("headway", "Time headway (s)"), ("a_f", "Acceleration (m/s2)"),
                      ("dv", "Relative speed (m/s)")]:
        x, y = ice[col].dropna(), ev[col].dropna()
        rows.append(dict(scope=scope, variable=name, ICE_mean=x.mean(), EV_mean=y.mean(),
                         cohens_d=cohens_d(x, y), MW_p=stats.mannwhitneyu(x, y).pvalue,
                         n_ICE=len(x), n_EV=len(y)))
    return pd.DataFrame(rows)


def steady_state_headway(df, speed_range=(15, 27), bin_width=2.0, a_max=0.2, dv_max=0.3,
                         min_samples=50):
    """Median time headway per configuration and speed bin during steady following.

    Steady following: |a_f| < a_max and |v_l - v_f| < dv_max. Cells with fewer than
    min_samples samples are NaN. Returns a long table (config, speed_bin, headway, n).
    """
    d = df[(df["a_f"].abs() < a_max) & (df["dv"].abs() < dv_max)
           & df["v_f"].between(*speed_range)].copy()
    d["config"] = d["vehicle"] + " " + d["gap"]
    d["speed_bin"] = pd.cut(d["v_f"], bins=np.arange(speed_range[0],
                                                     speed_range[1] + bin_width, bin_width))
    out = (d.groupby(["powertrain", "config", "speed_bin"], observed=True)["headway"]
             .agg(headway="median", n="size").reset_index())
    out.loc[out["n"] < min_samples, "headway"] = np.nan
    return out


def extract_events(df, meta_cols):
    """Leader deceleration events of every run, with run metadata attached."""
    tables = []
    for run_id, d in df.groupby("run_id"):
        run = {c: d[c].values for c in ["t", "v_l", "v_f", "s", "a_l", "a_f"]}
        ev = run_events(run)
        if len(ev):
            ev.insert(0, "run_id", run_id)
            for c in meta_cols:
                ev[c] = d[c].iloc[0]
            tables.append(ev)
    ev = pd.concat(tables, ignore_index=True)
    ev["config"] = ev["vehicle"] + " " + ev["gap"]
    ev["overshoot"] = ev["drop_follower"] - ev["drop_leader"]
    ev["drop_bin"] = pd.cut(ev["drop_leader"], bins=DROP_BINS, labels=DROP_LABELS)
    return ev


def per_config_medians(ev, group_cols=("drop_bin", "powertrain", "config")):
    """Median of each event metric per configuration (each configuration counts once)."""
    return ev.groupby(list(group_cols), observed=True)[EVENT_METRICS].median().reset_index()


def config_level_tests(per_config, by="drop_bin", levels=("3-5", "7-12")):
    """Exact two-sided Mann-Whitney test, ICE vs EV configurations, per level of `by`."""
    rows = []
    for level in levels:
        sub = per_config[per_config[by] == level]
        for metric in EVENT_METRICS:
            ice = sub.loc[sub["powertrain"] == "ICE", metric].dropna()
            ev = sub.loc[sub["powertrain"] == "EV", metric].dropna()
            if len(ice) < 2 or len(ev) < 2:
                continue
            p = stats.mannwhitneyu(ice, ev, alternative="two-sided", method="exact").pvalue
            rows.append({by: level, "metric": metric, "ICE_median": ice.median(),
                         "EV_median": ev.median(), "n_ICE": len(ice), "n_EV": len(ev),
                         "p_exact": p})
    return pd.DataFrame(rows)
