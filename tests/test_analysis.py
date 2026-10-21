"""Tests for the event analysis (synthetic data only). Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.events import METRIC_COLUMNS, find_leader_decelerations, run_events

DT = 0.1


def smooth_steps(t, levels, times, ramp_s=4.0):
    """Speed that moves between levels with linear ramps starting at the given times."""
    v = np.full_like(t, levels[0], dtype=float)
    for (v0, v1), t0 in zip(zip(levels[:-1], levels[1:]), times):
        frac = np.clip((t - t0) / ramp_s, 0, 1)
        v = np.where(t >= t0, v0 + (v1 - v0) * frac, v)
    return v


def make_run(v_l, delay_s=2.0, gain=1.2):
    """Follower = leader delayed by delay_s, with the speed drop amplified by `gain`."""
    t = np.arange(len(v_l)) * DT
    k = int(delay_s / DT)
    v_f = np.r_[np.full(k, v_l[0]), v_l[:-k]]
    v_f = v_l[0] - gain * (v_l[0] - v_f)
    return dict(t=t, v_l=v_l, v_f=v_f, s=np.full_like(t, 30.0),
                a_l=np.gradient(v_l, DT), a_f=np.gradient(v_f, DT))


def test_two_steps_with_plateau_are_two_events():
    """Two step decelerations separated by a plateau must not be merged (the stepH case)."""
    t = np.arange(0, 200, DT)
    v_l = smooth_steps(t, [33.0, 30.8, 28.6], [50.0, 120.0])
    events = find_leader_decelerations(t, v_l, np.gradient(v_l, DT))
    assert len(events) == 2


def test_metrics_of_a_known_response():
    """Delayed, amplified copy of the leader: delay and amplification must be recovered."""
    t = np.arange(0, 80, DT)
    v_l = smooth_steps(t, [24.0, 20.0], [20.0])
    table = run_events(make_run(v_l, delay_s=2.0, gain=1.2))
    assert len(table) == 1
    row = table.iloc[0]
    assert row["drop_leader"] == pytest.approx(4.0, abs=0.05)
    assert row["amplification"] == pytest.approx(1.2, abs=0.02)
    assert row["delay_s"] == pytest.approx(2.0, abs=0.15)


def test_no_event_returns_empty_table_with_columns():
    t = np.arange(0, 60, DT)
    v_l = np.full_like(t, 25.0)
    table = run_events(make_run(v_l))
    assert len(table) == 0
    assert list(table.columns) == METRIC_COLUMNS
