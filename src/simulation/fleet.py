"""Fleets for the ring-road simulation, with common random numbers across mixes."""
import numpy as np

IDM_KEYS = ["v0", "s0", "tau", "delta", "alpha", "beta"]


def build_fleet(seed, n_vehicles, base_type, new_type, share, pools):
    """Vehicle types and IDM parameters for a mix of two vehicle types.

    For a given seed, all random choices are fixed in advance: a ranking of the vehicle
    positions and, for every position, the configuration it uses for each type. A share x
    of `new_type` converts the first round(x * n) ranked positions; all other vehicles
    are unchanged, so different shares differ only in the converted vehicles.

    pools: {vehicle_type: list of parameter dicts (IDM keys, optionally T_a)}
    Returns (types: list of str, params: dict of arrays of length n_vehicles).
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_vehicles)                      # conversion ranking
    pick = {t: rng.integers(len(pools[t]), size=n_vehicles)  # configuration per position
            for t in sorted(pools)}
    n_new = int(round(share * n_vehicles))
    types = [base_type] * n_vehicles
    for i in order[:n_new]:
        types[i] = new_type

    chosen = [pools[t][pick[t][i]] for i, t in enumerate(types)]
    params = {k: np.array([c[k] for c in chosen], dtype=float) for k in IDM_KEYS}
    params["T_a"] = np.array([c.get("T_a", 0.0) for c in chosen], dtype=float)
    return types, params
