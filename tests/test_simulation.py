"""Tests for the ring-road simulation. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_yaml
from src.simulation.ring import equilibrium_state, simulate_ring

RC = load_yaml("experiment.yaml")["ring"]
N = 10


def fleet(s0):
    """Ten identical IDM vehicles with the given s0 (one with a very small s0)."""
    p = dict(v0=40.0, s0=10.0, tau=1.0, delta=20.0, alpha=1.5, beta=4.0, T_a=0.0)
    params = {k: np.full(N, v, dtype=float) for k, v in p.items()}
    params["s0"][0] = s0
    return params


def test_equilibrium_without_perturbation_is_stationary():
    params = fleet(10.0)
    rc = dict(RC, perturbation=dict(RC["perturbation"], decel=0.0))
    sim = simulate_ring(params, 300.0, rc, duration=60.0)
    assert np.abs(sim["v"] - sim["v_eq"]).max() < 1e-9
    L_net = 300.0 - N * RC["vehicle_length_m"]          # net gaps fill ring minus bodies
    assert np.abs(sim["s"].sum(axis=1) - L_net).max() < 1e-6


def test_initial_state_respects_min_spacing():
    """A vehicle with s0 below the minimum spacing must not start closer than it."""
    params = fleet(2.5)
    _, s, _ = equilibrium_state(params, 120.0, RC["speed_cap"], min_spacing=7.0)
    assert s.min() >= 7.0 - 1e-9


def test_min_spacing_holds_in_dense_traffic():
    """Dense ring with the standard perturbation: spacing never below the minimum."""
    params = fleet(2.5)
    sim = simulate_ring(params, 120.0, RC, duration=300.0, min_spacing=7.0)
    assert not sim["collided"]
    assert sim["s"].min() >= 7.0 - 1e-6
