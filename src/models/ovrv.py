"""Optimal Velocity Relative Velocity (OVRV) car-following model.

    a = k1 * (s - eta - tau * v) + k2 * (v_l - v)

Parameters (dict): k1 [1/s^2], k2 [1/s], eta [m], tau [s]
"""
PARAM_NAMES = ["k1", "k2", "eta", "tau"]


def accel(s, v, v_l, p):
    """Acceleration of the follower given spacing s, own speed v and leader speed v_l."""
    return p["k1"] * (s - p["eta"] - p["tau"] * v) + p["k2"] * (v_l - v)


def equilibrium_spacing(v_e, p):
    """Spacing at which a = 0 when both vehicles drive at speed v_e."""
    return p["eta"] + p["tau"] * v_e


def partials(v_e, p):
    """Partial derivatives (f_s, f_ds, f_v) at the equilibrium (s_e(v_e), v_e, v_e).

    For OVRV they are constant and do not depend on v_e.
    """
    f_s = p["k1"]
    f_ds = p["k2"]
    f_v = -p["k1"] * p["tau"]
    return f_s, f_ds, f_v
