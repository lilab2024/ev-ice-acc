"""Follower simulation driven by a recorded leader trajectory (forward Euler).

Used for calibration: the leader speed is taken from data, the follower is simulated
with a car-following model, and the simulated spacing is compared with the measured one.

Optional actuation lag (EV-aware extension): if the parameter dict contains "T_a",
the model output is treated as a commanded acceleration and the actual acceleration
follows it with a first-order lag, discretized exactly:
    a[k+1] = a[k] + alpha * (a_cmd[k] - a[k]),   alpha = 1 - exp(-dt / T_a)
T_a = 0 gives alpha = 1, i.e. the original model without lag.
"""
import numpy as np


def _lag_factor(T_a, dt):                                                    # NEW
    """alpha = 1 - exp(-dt / T_a); equals 1 for T_a = 0 (no lag)."""
    with np.errstate(divide="ignore"):
        return 1.0 - np.exp(-dt / np.asarray(T_a, dtype=float))


def simulate_follower(model, p, t, v_l, s_init, v_init):
    """Simulate the follower behind a given leader.

    Parameters
    ----------
    model  : module with accel(s, v, v_l, p), e.g. src.models.idm or src.models.ovrv
    p      : parameter dict of the model (optionally with "T_a", the actuation lag in s)
    t      : time stamps (s), uniformly spaced
    v_l    : leader speed at each time stamp (m/s)
    s_init : initial spacing (m), usually the first measured value
    v_init : initial follower speed (m/s), usually the first measured value

    Returns
    -------
    dict with arrays v, s, a (actual acceleration; same length as t; NaN after a
    collision) and a boolean 'collided'.
    """
    t = np.asarray(t, dtype=float)
    v_l = np.asarray(v_l, dtype=float)
    dt_all = np.diff(t)
    dt = dt_all[0]
    if not np.allclose(dt_all, dt):
        raise ValueError("Time stamps must be uniformly spaced")
    alpha = _lag_factor(p.get("T_a", 0.0), dt)                               # NEW

    n = len(t)
    v = np.full(n, np.nan)
    s = np.full(n, np.nan)
    a = np.full(n, np.nan)
    v[0], s[0] = v_init, s_init
    a_act = 0.0                                                              # NEW
    collided = False

    for k in range(n - 1):
        a_cmd = model.accel(s[k], v[k], v_l[k], p)
        a_act = a_act + alpha * (a_cmd - a_act)                              # NEW
        a[k] = a_act
        v[k + 1] = max(v[k] + a_act * dt, 0.0)          # vehicles cannot reverse
        s[k + 1] = s[k] + (v_l[k] - v[k]) * dt
        if s[k + 1] <= 0:                              # collision: stop, do not hide it
            collided = True
            break
    if not collided:
        a_cmd = model.accel(s[-1], v[-1], v_l[-1], p)
        a[-1] = a_act + alpha * (a_cmd - a_act)
    return dict(v=v, s=s, a=a, collided=collided)


def simulate_follower_batch(model, P, t, v_l, s_init, v_init):
    """Simulate many parameter sets at once (vectorized over parameter sets).

    Same numerical scheme as simulate_follower, but every value in P is an array of
    length m (one entry per parameter set). Used to speed up calibration.

    Returns
    -------
    dict with arrays v, s of shape (len(t), m) and a boolean array 'collided' of length m.
    Columns of collided parameter sets are filled with NaN.
    """
    t = np.asarray(t, dtype=float)
    v_l = np.asarray(v_l, dtype=float)
    dt_all = np.diff(t)
    dt = dt_all[0]
    if not np.allclose(dt_all, dt):
        raise ValueError("Time stamps must be uniformly spaced")

    m = len(np.atleast_1d(next(iter(P.values()))))
    alpha = _lag_factor(P.get("T_a", np.zeros(m)), dt)                       # NEW
    n = len(t)
    v = np.empty((n, m))
    s = np.empty((n, m))
    v[0], s[0] = v_init, s_init
    a_act = np.zeros(m)                                                      # NEW
    collided = np.zeros(m, dtype=bool)

    with np.errstate(all="ignore"):            # collided columns may produce inf/NaN
        for k in range(n - 1):
            a_cmd = model.accel(s[k], v[k], v_l[k], P)
            a_act = a_act + alpha * (a_cmd - a_act)                          # NEW
            v[k + 1] = np.maximum(v[k] + a_act * dt, 0.0)
            s[k + 1] = s[k] + (v_l[k] - v[k]) * dt
            collided |= ~(s[k + 1] > 0)        # also catches NaN
    v[:, collided] = np.nan
    s[:, collided] = np.nan
    return dict(v=v, s=s, collided=collided)


def run_model(model, p, t, v_l, s_init, v_init):
    """Simulate one parameter set with any model.

    Models with their own simulate() (Newell, Gipps) use it; models that only provide
    accel() (IDM, OVRV, optionally with actuation lag T_a) use the generic simulator.
    """
    if hasattr(model, "simulate"):
        return model.simulate(p, t, v_l, s_init, v_init)
    return simulate_follower(model, p, t, v_l, s_init, v_init)


def run_model_batch(model, P, t, v_l, s_init, v_init):
    """Batch counterpart of run_model."""
    if hasattr(model, "simulate_batch"):
        return model.simulate_batch(P, t, v_l, s_init, v_init)
    return simulate_follower_batch(model, P, t, v_l, s_init, v_init)
