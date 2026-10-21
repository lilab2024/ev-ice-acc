"""Gipps (1981) car-following model, continuous-time relaxation form.

Target speed from the current state:
    v_acc   = v + 2.5 a tau (1 - v/V) sqrt(0.025 + v/V)
    v_brake = -B tau + sqrt(B^2 tau^2 + B [2 (s - S) - v tau] + v_l^2)
    v*      = min(v_acc, v_brake)            (v_brake = 0 if the root is negative)
Speed update with time constant tau:
    v[k+1]  = v[k] + (dt / tau) (v*[k] - v[k])
For tau = dt this is exactly the original Gipps update v[k+1] = v*.
B > 0 is the braking magnitude; the estimate of the leader's braking equals B.
Equilibrium spacing: s_e = S + 1.5 tau v.

Parameters (dict): tau [s], a [m/s^2], B [m/s^2], S [m], V [m/s]
"""
import numpy as np

PARAM_NAMES = ["tau", "a", "B", "S", "V"]


def target_speed(s, v, v_l, P):
    """Gipps target speed v* (works with scalars or arrays)."""
    tau, a, B, S, V = (P[k] for k in PARAM_NAMES)
    v_acc = v + 2.5 * a * tau * (1 - v / V) * np.sqrt(np.maximum(0.025 + v / V, 0.0))
    root = B ** 2 * tau ** 2 + B * (2 * (s - S) - v * tau) + v_l ** 2
    v_brake = -B * tau + np.sqrt(np.maximum(root, 0.0))
    return np.maximum(np.minimum(v_acc, v_brake), 0.0)


def equilibrium_spacing(v_e, p):
    """Spacing at which both vehicles keep driving at v_e."""
    return p["S"] + 1.5 * p["tau"] * v_e


def simulate_batch(P, t, v_l, s_init, v_init):
    """Vectorized over parameter sets; same output format as simulate_follower_batch."""
    t = np.asarray(t, dtype=float)
    v_l = np.asarray(v_l, dtype=float)
    dt = t[1] - t[0]
    m = len(np.atleast_1d(P["tau"]))
    P = {k: np.broadcast_to(np.asarray(P[k], dtype=float), (m,)) for k in PARAM_NAMES}
    gain = np.minimum(dt / P["tau"], 1.0)

    n = len(t)
    v = np.empty((n, m))
    s = np.empty((n, m))
    v[0], s[0] = v_init, s_init
    collided = np.zeros(m, dtype=bool)
    with np.errstate(all="ignore"):
        for k in range(n - 1):
            v_star = target_speed(s[k], v[k], v_l[k], P)
            v[k + 1] = np.maximum(v[k] + gain * (v_star - v[k]), 0.0)
            s[k + 1] = s[k] + (v_l[k] - v[k]) * dt
            collided |= ~(s[k + 1] > 0)
    v[:, collided] = np.nan
    s[:, collided] = np.nan
    return dict(v=v, s=s, collided=collided)


def simulate(p, t, v_l, s_init, v_init):
    """Single parameter set; same output format as simulate_follower."""
    out = simulate_batch({k: np.array([p[k]]) for k in PARAM_NAMES}, t, v_l, s_init, v_init)
    v, s = out["v"][:, 0], out["s"][:, 0]
    a = np.gradient(v, t) if not out["collided"][0] else np.full(len(t), np.nan)
    return dict(v=v, s=s, a=a, collided=bool(out["collided"][0]))
