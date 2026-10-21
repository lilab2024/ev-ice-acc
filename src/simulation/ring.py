"""Closed ring-road simulation with heterogeneous IDM vehicles."""
import numpy as np
from scipy.optimize import brentq

from src.models import idm


def equilibrium_state(params, L, v_cap, min_spacing=None):
    """Common speed v* and individual spacings with sum of equilibrium spacings = L.

    With min_spacing, every equilibrium spacing is at least min_spacing, so that the
    initial state already satisfies the minimum-spacing constraint.

    Returns (v_eq, spacings, regime) with regime 'jam' (vehicles cannot even fit at
    standstill: v = 0, equal spacings), 'capped' (all at the speed cap, surplus length
    shared equally) or 'interior'.
    """
    n = len(params["s0"])
    floor = 0.0 if min_spacing is None else min_spacing
    spacing = lambda v: np.maximum(idm.equilibrium_spacing(v, params), floor)
    total = lambda v: float(np.sum(spacing(v)))
    if total(0.0) >= L:
        return 0.0, np.full(n, L / n), "jam"
    if total(v_cap) <= L:
        s = spacing(v_cap)
        return v_cap, s + (L - s.sum()) / n, "capped"
    v = brentq(lambda x: total(x) - L, 0.0, v_cap)
    return v, spacing(v), "interior"


def simulate_ring(params, L, rc, duration=None, decel=None, min_spacing=None):     # NEW
    """Simulate the ring from equilibrium with one standard perturbation.

    params     : dict of arrays (IDM keys and T_a), one entry per vehicle; vehicle i
                 follows vehicle i-1 and vehicle 0 follows vehicle n-1.
    rc         : the 'ring' section of experiment.yaml
    duration   : overrides rc['duration_s']
    decel      : overrides the perturbation deceleration (m/s2)
    min_spacing: if given, a vehicle's speed is limited so that it does not come closer
                 than this to its leader (supplementary long-run analysis only)
    """
    n = len(params["s0"])
    dt, cap = rc["dt"], rc["speed_cap"]
    n_steps = int(round((duration or rc["duration_s"]) / dt)) + 1
    t = np.arange(n_steps) * dt
    # states are net gaps; the vehicle bodies occupy n * vehicle_length_m of the ring
    L_net = L - n * rc["vehicle_length_m"]
    v_eq, s_eq, regime = equilibrium_state(params, L_net, cap, min_spacing)

    v = np.empty((n_steps, n))
    s = np.empty((n_steps, n))
    v[0], s[0] = v_eq, s_eq
    lead = np.roll(np.arange(n), 1)
    with np.errstate(divide="ignore"):
        lag = 1.0 - np.exp(-dt / params["T_a"])          # 1 for T_a = 0 (no lag)
    a_act = np.zeros(n)

    pert = rc["perturbation"]
    b = pert["decel"] if decel is None else decel                                   # NEW
    k0 = int(round(pert["start_s"] / dt))
    k1 = k0 + int(round(pert["duration_s"] / dt))
    j = pert["vehicle"]

    collided, last = False, n_steps - 1
    with np.errstate(all="ignore"):
        for k in range(n_steps - 1):
            v_l = v[k][lead]
            a_cmd = idm.accel(s[k], v[k], v_l, params)
            a_act = a_act + lag * (a_cmd - a_act)
            if k0 <= k < k1:
                a_act[j] = -b                             # forced braking of one vehicle
            v_new = np.clip(v[k] + a_act * dt, 0.0, cap)
            s[k + 1] = s[k] + (v_l - v[k]) * dt
            if min_spacing is not None:
                # do not come closer than min_spacing to the leader in the next step;
                # repeat until no speed changes, so the limit propagates along a whole
                # queue of constrained vehicles (speeds only decrease: converges)
                for _ in range(n):
                    v_lim = np.clip(np.minimum(v_new, v_new[lead]
                                               + (s[k + 1] - min_spacing) / dt), 0.0, cap)
                    if np.array_equal(v_lim, v_new):
                        break
                    v_new = v_lim
            v[k + 1] = v_new
            if np.any(~(s[k + 1] > 0)):
                collided, last = True, k + 1
                break
    return dict(t=t[:last + 1], v=v[:last + 1], s=s[:last + 1],
                v_eq=v_eq, regime=regime, collided=collided)


def ring_metrics(sim, density, rc, warmup=None):
    """Traffic and wave indicators after the warm-up period."""
    out = dict(v_eq=sim["v_eq"], regime=sim["regime"], collided=sim["collided"])
    if sim["collided"]:
        return out | dict(mean_speed=np.nan, flow=np.nan, speed_std=np.nan,
                          stopped_share=np.nan, min_spacing=np.nan, speed_range=np.nan)
    keep = sim["t"] >= (rc["warmup_s"] if warmup is None else warmup)
    v, s = sim["v"][keep], sim["s"][keep]
    return out | dict(mean_speed=float(v.mean()),
                      flow=float(density * v.mean() * 3.6),               # veh/h
                      speed_std=float(v.std()),
                      stopped_share=float(np.mean(v < rc["stopped_speed"])),
                      min_spacing=float(s.min()),
                      speed_range=float(np.mean(v.max(axis=0) - v.min(axis=0))))


def growth_metrics(sim, rc):                                                         # NEW
    """Growth of the perturbation while all vehicles stay inside the calibrated range.

    time_to_exit: time from the perturbation start until any vehicle leaves the range
                  (NaN and exited = False if it never leaves)
    growth_rate : slope (1/s) of log(max_i |v_i - v_eq|) from the end of the perturbation
                  + fit_start_s until the exit (or the end of the run)
    """
    g, pert = rc["growth"], rc["perturbation"]
    lo, hi = g["calibrated_speed_range"]
    t, v, v_eq = sim["t"], sim["v"], sim["v_eq"]
    t0 = pert["start_s"]

    outside = np.any((v < lo) | (v > hi + 1e-9), axis=1) & (t >= t0)
    exited = bool(outside.any())
    t_exit = t[np.argmax(outside)] if exited else t[-1]

    dev = np.max(np.abs(v - v_eq), axis=1)
    fit = (t >= t0 + pert["duration_s"] + g["fit_start_s"]) & (t < t_exit) & (dev > 0)
    if fit.sum() * (t[1] - t[0]) >= g["min_fit_s"]:
        rate = float(np.polyfit(t[fit], np.log(dev[fit]), 1)[0])
    else:
        rate = np.nan
    return dict(v_eq=v_eq, regime=sim["regime"], collided=sim["collided"], exited=exited,
                time_to_exit=float(t_exit - t0) if exited else np.nan,
                growth_rate=rate, max_dev=float(dev[t >= t0].max()))
