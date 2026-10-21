"""Intelligent Driver Model (IDM), exactly as written in the paper.

    a      = alpha * [1 - (v / v0)**delta - (s_star / s)**2]
    s_star = s0 + tau * v - v * (v_l - v) / (2 * sqrt(alpha * beta))

Parameters (dict): v0 [m/s], s0 [m], tau [s], delta [-], alpha [m/s^2], beta [m/s^2]
No clipping is applied here; the model only evaluates the formula.
"""
import numpy as np

PARAM_NAMES = ["v0", "s0", "tau", "delta", "alpha", "beta"]


def desired_gap(v, v_l, p):
    """s_star: the gap the driver wants given own speed v and leader speed v_l."""
    return p["s0"] + p["tau"] * v - v * (v_l - v) / (2 * np.sqrt(p["alpha"] * p["beta"]))


def accel(s, v, v_l, p):
    """Acceleration of the follower given spacing s, own speed v and leader speed v_l."""
    s_star = desired_gap(v, v_l, p)
    return p["alpha"] * (1 - (v / p["v0"]) ** p["delta"] - (s_star / s) ** 2)


def equilibrium_spacing(v_e, p):
    """Spacing at which a = 0 when both vehicles drive at speed v_e (requires v_e < v0)."""
    s_star = p["s0"] + p["tau"] * v_e
    return s_star / np.sqrt(1 - (v_e / p["v0"]) ** p["delta"])


def partials(v_e, p):
    """Partial derivatives (f_s, f_ds, f_v) at the equilibrium (s_e(v_e), v_e, v_e)."""
    a, b = p["alpha"], p["beta"]
    s_e = equilibrium_spacing(v_e, p)
    s_star = p["s0"] + p["tau"] * v_e            # relative speed is zero at equilibrium
    f_s = 2 * a * s_star ** 2 / s_e ** 3
    f_ds = a * s_star * v_e / (s_e ** 2 * np.sqrt(a * b))
    f_v = -a * (p["delta"] * v_e ** (p["delta"] - 1) / p["v0"] ** p["delta"]
                + 2 * s_star * p["tau"] / s_e ** 2)
    return f_s, f_ds, f_v
