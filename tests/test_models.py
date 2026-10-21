"""Tests for the model and stability layers. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import idm, ovrv
from src.models.simulate import simulate_follower
from src.stability.string_stability import ovrv_margin, string_stability

RNG = np.random.default_rng(0)


def random_ovrv():
    return dict(k1=RNG.uniform(0.01, 0.5), k2=RNG.uniform(0.05, 1.0),
                eta=RNG.uniform(2, 10), tau=RNG.uniform(0.5, 3.0))


def random_idm():
    return dict(v0=RNG.uniform(30, 45), s0=RNG.uniform(2, 10), tau=RNG.uniform(0.5, 3.0),
                delta=RNG.uniform(2, 20), alpha=RNG.uniform(0.5, 3.0), beta=RNG.uniform(1, 10))


def numerical_partials(model, v_e, p, h=1e-5):
    """Finite differences at equilibrium, holding the other two arguments fixed."""
    s_e = model.equilibrium_spacing(v_e, p)
    f = model.accel
    return ((f(s_e + h, v_e, v_e, p) - f(s_e - h, v_e, v_e, p)) / (2 * h),
            (f(s_e, v_e, v_e + h, p) - f(s_e, v_e, v_e - h, p)) / (2 * h),
            (f(s_e, v_e + h, v_e + h, p) - f(s_e, v_e - h, v_e - h, p)) / (2 * h))


@pytest.mark.parametrize("model, sampler", [(ovrv, random_ovrv), (idm, random_idm)])
def test_partials_match_finite_differences(model, sampler):
    for _ in range(50):
        p = sampler()
        for v_e in [5.0, 15.0, 25.0]:
            ana = model.partials(v_e, p)
            num = numerical_partials(model, v_e, p)
            assert np.allclose(ana, num, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("model, sampler", [(ovrv, random_ovrv), (idm, random_idm)])
def test_zero_acceleration_at_equilibrium(model, sampler):
    for _ in range(50):
        p = sampler()
        for v_e in [5.0, 15.0, 25.0]:
            s_e = model.equilibrium_spacing(v_e, p)
            assert abs(model.accel(s_e, v_e, v_e, p)) < 1e-10


def test_ovrv_lambda2_agrees_with_margin():
    """lambda2 < 0 must hold exactly when k1*tau^2/2 + k2*tau >= 1 (the Table 8 error)."""
    n_stable = 0
    for _ in range(500):
        p = random_ovrv()
        res = string_stability(ovrv, p, 25.0)
        assert res["stable"] == (ovrv_margin(p) >= 1)
        assert np.isclose(res["margin"], ovrv_margin(p))
        n_stable += res["stable"]
    assert 0 < n_stable < 500          # the sample must contain both cases


@pytest.mark.parametrize("model, sampler", [(ovrv, random_ovrv), (idm, random_idm)])
def test_simulation_stays_at_equilibrium(model, sampler):
    t = np.arange(0, 60, 0.1)
    for _ in range(10):
        p = sampler()
        s_e = model.equilibrium_spacing(20.0, p)
        out = simulate_follower(model, p, t, np.full_like(t, 20.0), s_e, 20.0)
        assert not out["collided"]
        assert np.allclose(out["v"], 20.0, atol=1e-8)
        assert np.allclose(out["s"], s_e, atol=1e-8)
