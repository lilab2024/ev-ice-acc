"""Newell (2002) simplified car-following model, speed form.

The follower repeats the leader's speed with a time delay tau:
    v_f(t) = v_l(t - tau)
    s(t)   = s(0) + integral_0^t [v_l(u) - v_f(u)] du
The jam spacing d of the position form is implied by the initial state, so the only
parameter is tau [s]. Before t = 0 the leader is assumed to drive at its initial speed.
By construction the speed amplification is exactly 1 (neither amplified nor damped).
"""
import numpy as np

PARAM_NAMES = ["tau"]


def _delayed(t, v_l, tau):
    """Leader speed delayed by tau (array of shape (n, m) for m values of tau)."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    dt = t[1] - t[0]
    pos = np.arange(len(t))[:, None] - tau[None, :] / dt      # fractional index
    pos = np.clip(pos, 0, len(t) - 1)                         # before t = 0: initial speed
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(t) - 1)
    w = pos - i0
    return (1 - w) * v_l[i0] + w * v_l[i1]


def simulate_batch(P, t, v_l, s_init, v_init):
    """Vectorized over parameter sets; same output format as simulate_follower_batch."""
    t = np.asarray(t, dtype=float)
    v_l = np.asarray(v_l, dtype=float)
    dt = t[1] - t[0]
    v = _delayed(t, v_l, P["tau"])
    # spacing by the same forward Euler scheme as the generic simulator
    ds = (v_l[:-1, None] - v[:-1]) * dt
    s = s_init + np.vstack([np.zeros((1, v.shape[1])), np.cumsum(ds, axis=0)])
    collided = ~(s > 0).all(axis=0)
    v[:, collided] = np.nan
    s[:, collided] = np.nan
    return dict(v=v, s=s, collided=collided)


def simulate(p, t, v_l, s_init, v_init):
    """Single parameter set; same output format as simulate_follower."""
    out = simulate_batch({k: np.array([v]) for k, v in p.items()}, t, v_l, s_init, v_init)
    v, s = out["v"][:, 0], out["s"][:, 0]
    a = np.gradient(v, t) if not out["collided"][0] else np.full(len(t), np.nan)
    return dict(v=v, s=s, a=a, collided=bool(out["collided"][0]))
