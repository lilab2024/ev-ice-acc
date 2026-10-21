"""Event-aligned speed responses to leader decelerations (data for the event-response figure)."""
import numpy as np
import pandas as pd


def build_event_curves(traj, events, pre_s=3.0, window=(-3.0, 15.0), dt=0.1):
    """Normalized speed drop of leader and follower around each event.

    y(t) = (v_before - v(t)) / drop_leader, where v_before is the mean speed over the pre_s
    seconds before the event start (same definition as event_metrics) and t = 0 is the start
    of the leader deceleration phase. A follower value above one means that the follower
    slows down more than its leader.
    """
    grid = np.round(np.arange(window[0], window[1] + dt / 2, dt), 3)
    runs = traj[traj["run_id"].isin(events["run_id"].unique())]
    by_run = {rid: d.sort_values("t") for rid, d in runs.groupby("run_id")}
    parts = []
    for e in events.itertuples(index=False):
        d = by_run[e.run_id]
        t = d["t"].values
        i0 = int(np.argmin(np.abs(t - e.t_start)))
        i_pre = max(0, i0 - int(round(pre_s / dt)))
        v_l, v_f = d["v_l"].values, d["v_f"].values
        vl0, vf0 = v_l[i_pre:i0 + 1].mean(), v_f[i_pre:i0 + 1].mean()
        tt = t - t[i0]
        inside = (grid >= tt[0]) & (grid <= tt[-1])
        yl = np.full(len(grid), np.nan)
        yf = np.full(len(grid), np.nan)
        yl[inside] = (vl0 - np.interp(grid[inside], tt, v_l)) / e.drop_leader
        yf[inside] = (vf0 - np.interp(grid[inside], tt, v_f)) / e.drop_leader
        parts.append(pd.DataFrame({"run_id": e.run_id, "config": e.config,
                                   "powertrain": e.powertrain, "t": grid,
                                   "leader": yl, "follower": yf}))
    return pd.concat(parts, ignore_index=True)


def load_or_build(paths, res_dir, main_sel):
    """Curves of the main selection (leader drops of 3-5 m/s); cached in results/data_summary.

    The cache is rebuilt when results/statistics/events.csv is newer than the cache.
    """
    cache = res_dir / "data_summary" / "event_curves.csv"
    events_path = res_dir / "statistics" / "events.csv"
    if cache.exists() and cache.stat().st_mtime >= events_path.stat().st_mtime:
        return pd.read_csv(cache)
    ev = pd.read_csv(events_path)
    ev = ev[(ev["drop_bin"] == "3-5")
            & ((ev["powertrain"] == "ICE") | ev["v_reduced_mph"].isin(main_sel))]
    print("  events used (paper: 103 ICE, 54 EV):", ev.groupby("powertrain").size().to_dict())
    traj = pd.read_parquet(paths["processed"] / "trajectories_10hz.parquet",
                           columns=["run_id", "t", "v_l", "v_f"])
    curves = build_event_curves(traj, ev)
    curves.round(4).to_csv(cache, index=False)
    return curves
