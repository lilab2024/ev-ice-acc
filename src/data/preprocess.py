"""Uniform preprocessing: both datasets go through exactly the same steps."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def _odd_window(window_s, hz, polyorder, n):
    """Convert a window length in seconds to an odd number of samples."""
    w = int(round(window_s * hz))
    if w % 2 == 0:
        w += 1                                  # Savitzky-Golay needs an odd window
    w = max(w, polyorder + 2 if polyorder % 2 == 1 else polyorder + 1)
    if w > n:                                   # run shorter than the window
        w = n if n % 2 == 1 else n - 1
    return w


def smooth_and_resample(raw, native_hz, target_hz, window_s=1.0, polyorder=2):
    """Smooth at the native rate, differentiate, then resample to target_hz.

    Input : DataFrame[time, v_f, v_l, s] at native_hz
    Output: DataFrame[t, v_f, v_l, s, a_f, a_l] at target_hz
    """
    t = raw["time"].values
    w = _odd_window(window_s, native_hz, polyorder, len(t))
    if w <= polyorder:
        raise ValueError(f"Run too short to smooth ({len(t)} samples)")
    dt = 1.0 / native_hz

    smoothed = {c: savgol_filter(raw[c].values, w, polyorder, mode="interp")
                for c in ["v_f", "v_l", "s"]}
    smoothed["a_f"] = savgol_filter(raw["v_f"].values, w, polyorder,
                                    deriv=1, delta=dt, mode="interp")
    smoothed["a_l"] = savgol_filter(raw["v_l"].values, w, polyorder,
                                    deriv=1, delta=dt, mode="interp")

    t_new = np.round(np.arange(0.0, t[-1] + 1e-9, 1.0 / target_hz), 6)
    out = pd.DataFrame({"t": t_new})
    for name, values in smoothed.items():
        out[name] = np.interp(t_new, t, values)
    return out


def add_derived(df, spacing_offset_m=0.0, min_speed_for_headway=0.45):
    """Add relative speed, net gap and time headway.

    dv      = v_l - v_f              (positive: leader is faster)
    s       = net gap: measured spacing - spacing_offset_m (raw value kept as s_raw)
    s_net   = s (kept for compatibility)
    headway = s / v_f                (NaN when v_f < min_speed_for_headway)
    """
    df = df.copy()
    # Smoothing can produce tiny negative speeds at standstill
    df["v_f"] = df["v_f"].clip(lower=0.0)
    df["v_l"] = df["v_l"].clip(lower=0.0)
    df["dv"] = df["v_l"] - df["v_f"]
    df["s_raw"] = df["s"]
    df["s"] = df["s"] - spacing_offset_m      # all later steps use the net gap
    df["s_net"] = df["s"]
    df["headway"] = np.where(df["v_f"] >= min_speed_for_headway,
                             df["s"] / df["v_f"].where(df["v_f"] > 0), np.nan)
    return df


def run_qc(df):
    """Quality-check metrics for one processed run.

    kin_slope    : slope of ds/dt regressed on (v_l - v_f); should be close to 1.
    kin_resid_rms: RMS of (ds/dt - (v_l - v_f)); overall agreement (m/s).
    kin_resid_max: max |ds/dt - (v_l - v_f)|; catches local GPS glitches (m/s).
    """
    ds_dt = np.gradient(df["s"].values, df["t"].values)
    x = df["dv"].values
    ok = np.isfinite(ds_dt) & np.isfinite(x)
    if np.ptp(x[ok]) > 0:
        slope = np.polyfit(x[ok], ds_dt[ok], 1)[0]
        corr = np.corrcoef(x[ok], ds_dt[ok])[0, 1]
    else:
        slope, corr = np.nan, np.nan
    resid = ds_dt[ok] - x[ok]                                  # NEW
    stopped = df["v_f"] < 0.5
    stand_gap = float(df.loc[stopped, "s"].median()) if stopped.any() else np.nan
    return dict(n_samples=len(df),
                duration_s=float(df["t"].iloc[-1]),
                v_f_min=df["v_f"].min(), v_f_max=df["v_f"].max(),
                s_min=df["s"].min(), s_max=df["s"].max(),
                a_f_min=df["a_f"].min(), a_f_max=df["a_f"].max(),
                kin_slope=slope, kin_corr=corr,
                kin_resid_rms=float(np.sqrt(np.mean(resid ** 2))),   # NEW
                kin_resid_max=float(np.max(np.abs(resid))),          # NEW
                standstill_gap_median=stand_gap)


def preprocess_run(raw, powertrain, cfg, vehicle=None):
    """Full preprocessing of one run using settings from experiment.yaml.

    Returns (processed DataFrame, QC dict).
    """
    p = cfg["preprocess"]
    df = smooth_and_resample(raw,
                             native_hz=p["native_hz"][powertrain],
                             target_hz=p["target_hz"],
                             window_s=p["smoothing"]["window_s"],
                             polyorder=p["smoothing"]["polyorder"])
    offset = p["spacing_offset_m"][powertrain]
    if isinstance(offset, dict):          # per vehicle (EV: leader length)
        offset = offset[vehicle]
    df = add_derived(df,
                     spacing_offset_m=offset,
                     min_speed_for_headway=p["min_speed_for_headway"])
    return df, run_qc(df)
