"""Linear string stability of car-following models (Wilson & Ward, 2011).

For a model a = f(s, ds, v) with ds = v_l - v, evaluated at equilibrium:

    lambda2 = f_s / f_v**3 * (f_v**2 / 2 - f_ds * f_v - f_s)

String stable if lambda2 < 0. Because f_s > 0 and f_v < 0, the sign of lambda2
is set by the bracket alone ("core"): core > 0  <=>  lambda2 < 0  <=>  stable.

core has units (1/s^2) and scales with model stiffness, so it cannot be compared
across vehicles. The dimensionless margin

    M = 1 + core / f_s          (string stable if M >= 1)

can be compared; for OVRV it reduces to M = k1*tau^2/2 + k2*tau.
"""


def lambda2_from_partials(f_s, f_ds, f_v):
    """Return (lambda2, core) from the three partial derivatives."""
    core = f_v ** 2 / 2 - f_ds * f_v - f_s
    lambda2 = f_s / f_v ** 3 * core
    return lambda2, core


def string_stability(model, p, v_e):
    """String stability of `model` (a module with partials()) at equilibrium speed v_e."""
    f_s, f_ds, f_v = model.partials(v_e, p)
    lambda2, core = lambda2_from_partials(f_s, f_ds, f_v)
    margin = 1 + core / f_s                                          # NEW
    return dict(f_s=f_s, f_ds=f_ds, f_v=f_v,
                lambda2=lambda2, core=core, margin=margin,           # NEW
                stable=bool(lambda2 < 0))


def ovrv_margin(p):
    """OVRV stability margin M = k1*tau^2/2 + k2*tau; string stable if M >= 1.

    Closed form of the general margin 1 + core / f_s for OVRV.
    """
    return p["k1"] * p["tau"] ** 2 / 2 + p["k2"] * p["tau"]


# ---------------------------------------------------------------------------
# Frequency-domain analysis of mixed platoons
# ---------------------------------------------------------------------------
import numpy as np


def transfer_gain(f_s, f_ds, f_v, omega, T_a=0.0):
    """|G(j omega)| of the linearized follower: speed response to the leader's speed.

        G(s) = (f_ds s + f_s) / (s^2 + (f_ds - f_v) s + f_s)

    |G| <= 1 for all omega  <=>  core >= 0  <=>  M >= 1 (same criterion as lambda2).
    Optional first-order lag T_a (actuation or human reaction) between the commanded and
    the actual acceleration: the denominator becomes T_a s^3 + s^2 + (f_ds - f_v) s + f_s.
    """
    s = 1j * np.asarray(omega, dtype=float)
    return np.abs((f_ds * s + f_s) / (T_a * s ** 3 + s ** 2 + (f_ds - f_v) * s + f_s))


def platoon_amplification(partials, omega=None):
    """Head-to-tail amplification of a platoon of linearized followers.

    partials: list of (f_s, f_ds, f_v) or (f_s, f_ds, f_v, T_a), one per follower
              (order does not matter: the head-to-tail transfer function is a product).
    Returns (peak head-to-tail gain, per-vehicle geometric-mean gain at the peak,
             frequency of the peak in rad/s).
    """
    if omega is None:
        omega = np.logspace(-3, 1, 2000)
    log_gain = np.zeros_like(omega)
    for p in partials:
        log_gain += np.log(transfer_gain(*p[:3], omega, *p[3:]))
    i = int(np.argmax(log_gain))
    return float(np.exp(log_gain[i])), float(np.exp(log_gain[i] / len(partials))), float(omega[i])
