"""Monte Carlo string stability of mixed platoons in the frequency domain."""
import numpy as np

from src.stability.string_stability import transfer_gain


def default_omega(spec=None):
    """Log-spaced frequency grid (rad/s)."""
    spec = spec or {"min": 1e-3, "max": 10.0, "n": 2000}
    return np.logspace(np.log10(spec["min"]), np.log10(spec["max"]), int(spec["n"]))


def mixed_platoon_mc(pools, shares, n_vehicles=30, n_samples=1000, seed=0, omega=None):
    """Distribution of the per-vehicle amplification of random mixed platoons.

    pools : {vehicle_type: list of partial-derivative tuples (f_s, f_ds, f_v[, T_a])},
            one tuple per calibrated configuration of that type
    shares: {vehicle_type: fraction}; fractions sum to 1. The number of vehicles of each
            type is fixed (rounded); the configuration of each vehicle is drawn at random
            with replacement from its pool. The order of the vehicles does not matter,
            because the head-to-tail transfer function is a product.
    Returns an array (n_samples,) of per-vehicle geometric-mean peak gains
    (> 1: disturbances grow along the platoon).
    """
    omega = default_omega() if omega is None else omega
    rng = np.random.default_rng(seed)
    types = [t for t in shares if shares[t] > 0]
    counts = {t: int(round(shares[t] * n_vehicles)) for t in types}
    counts[types[-1]] = n_vehicles - sum(counts[t] for t in types[:-1])   # exact total

    # log|G| of every configuration, computed once and reused for all samples
    log_g = {t: np.array([np.log(transfer_gain(*p[:3], omega, *p[3:])) for p in pools[t]])
             for t in types}

    gains = np.empty(n_samples)
    for i in range(n_samples):
        total = np.zeros_like(omega)
        for t in types:
            idx = rng.integers(len(pools[t]), size=counts[t])
            total += log_g[t][idx].sum(axis=0)
        gains[i] = np.exp(total.max() / n_vehicles)
    return gains


def summarize(gains):
    """Median and 95 % interval of the per-vehicle gain, and the share of stable platoons."""
    return dict(median=float(np.median(gains)),
                low=float(np.percentile(gains, 2.5)),
                high=float(np.percentile(gains, 97.5)),
                p_stable=float(np.mean(gains <= 1.0 + 1e-9)))
