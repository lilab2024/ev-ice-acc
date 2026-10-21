"""Tests for the data layer. Run from the repository root: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_paths, load_yaml
from src.data.loaders import build_manifest, load_raw_run
from src.data.preprocess import preprocess_run

CFG = load_yaml("experiment.yaml")


def synthetic_run(hz, duration=60.0):
    """Leader with a sinusoidal speed; follower lags by 1 s; spacing from exact kinematics."""
    t = np.arange(0.0, duration, 1.0 / hz)
    w = 2 * np.pi / 20
    v_l = 20 + 3 * np.sin(w * t)
    v_f = 20 + 3 * np.sin(w * (t - 1))
    s = 30 + 3 / w * (np.cos(w * (t - 1)) - np.cos(w * t))   # integral of (v_l - v_f)
    return pd.DataFrame({"time": t, "v_f": v_f, "v_l": v_l, "s": s})


def test_ice_first_row_is_kept():
    """ICE files have no header: every line in the file must become a data row."""
    manifest = build_manifest(load_paths(), load_yaml("vehicles.yaml"))
    row = manifest[manifest["powertrain"] == "ICE"].iloc[0]
    with open(row["path"]) as f:
        n_lines = sum(1 for _ in f)
    assert len(load_raw_run(row)) == n_lines


def test_10hz_and_50hz_give_the_same_result():
    """The same signal sampled at 10 Hz (ICE) and 50 Hz (EV) must give the same output."""
    ice, _ = preprocess_run(synthetic_run(10), "ICE", CFG)
    ev, _ = preprocess_run(synthetic_run(50), "EV", CFG, vehicle="IONIQ 5")
    n = min(len(ice), len(ev))
    inner = slice(10, n - 10)                                 # ignore filter edge effects
    # compare the raw spacing: resampling and filtering must agree; the EV leader-length
    # offset is tested separately in test_spacing_offset_gives_net_gap
    for col in ["v_f", "s_raw", "a_f"]:
        diff = np.abs(ice[col].values[inner] - ev[col].values[inner]).max()
        assert diff < 0.05, f"{col} differs by {diff:.3f}"


def test_kinematic_check_on_exact_data():
    """On physically consistent data the QC must report slope ~1 and residual ~0."""
    _, qc = preprocess_run(synthetic_run(50), "EV", CFG, vehicle="IONIQ 5")
    assert abs(qc["kin_slope"] - 1) < 0.02
    assert qc["kin_resid_max"] < 0.1


def test_spacing_offset_gives_net_gap():
    """EV spacing (front bumper to front bumper) minus leader length; ICE unchanged."""
    ice, _ = preprocess_run(synthetic_run(10), "ICE", CFG)
    ev, _ = preprocess_run(synthetic_run(50), "EV", CFG, vehicle="IONIQ 5")
    offset = CFG["preprocess"]["spacing_offset_m"]["EV"]["IONIQ 5"]
    assert np.allclose(ice["s"], ice["s_raw"])
    assert np.allclose(ev["s"], ev["s_raw"] - offset)
