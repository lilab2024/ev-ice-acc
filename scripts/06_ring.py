"""Step 06: ring-road simulations of mixed traffic.

A (main): growth of a small perturbation while every vehicle stays inside the calibrated
          speed range. B (supplementary): long run to the final stop-and-go state with a
          common minimum spacing. Requires step 03 (calibrated parameters).
Each (analysis, scenario, share, density) is saved immediately; reruns skip finished parts.
Usage (from the repository root):  python scripts/06_ring.py [--analysis A B] [--force]
Outputs: results/ring/parts/*.csv and results/ring/ring_A.csv, ring_B.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import REPO_ROOT, load_yaml
from src.simulation.fleet import IDM_KEYS, build_fleet
from src.simulation.ring import (equilibrium_state, growth_metrics, ring_metrics,
                                 simulate_ring)

OUT = REPO_ROOT / "results" / "ring"
# (scenario name, base type, new type): the share is the fraction of the new type
SCENARIOS = [("ICE-ACC + EV-ACC", "ICE", "EV"),
             ("HV + ICE-ACC", "HV", "ICE"),
             ("HV + EV-ACC", "HV", "EV")]


def make_pools(cfg):
    acc = pd.read_csv(REPO_ROOT / "results" / "calibration" / "main_idm_params.csv")
    acc = acc[~acc["config"].isin(list(cfg["parameter_use"]["exclude_from_main"]))]
    pools = {pt: [{k: r[k] for k in IDM_KEYS}
                  for r in acc[acc["powertrain"] == pt].to_dict("records")]
             for pt in ["ICE", "EV"]}
    pools["HV"] = [dict(cfg["human_driver"]["base"], T_a=cfg["ring"]["human_lag"])]
    return pools


def run_part(analysis, scenario, base, new, share, density, pools, rc):
    n = rc["n_vehicles"]
    L = n / density * 1000.0
    g, lr = rc["growth"], rc["long_run"]
    dip = g["perturbation_decel"] * rc["perturbation"]["duration_s"]
    rows = []
    for seed in range(rc["n_seeds"]):
        types, p = build_fleet(seed, n, base, new, share, pools)
        if analysis == "A":
            # same geometry as simulate_ring: net gaps fill the ring minus the bodies
            v_eq = equilibrium_state(p, L - n * rc["vehicle_length_m"],
                                     rc["speed_cap"])[0]
            if v_eq - dip < g["calibrated_speed_range"][0]:
                rows.append(dict(v_eq=v_eq, in_range=False))   # starts outside the range
                continue
            sim = simulate_ring(p, L, rc, decel=g["perturbation_decel"])
            m = growth_metrics(sim, rc) | dict(in_range=True)
        else:
            sim = simulate_ring(p, L, rc, duration=lr["duration_s"],
                                min_spacing=lr["min_spacing"])
            m = ring_metrics(sim, density, rc, warmup=lr["warmup_s"])
        rows.append(m)
    df = pd.DataFrame(rows)
    df.insert(0, "seed", range(rc["n_seeds"]))
    for col, val in [("density", density), ("share", share), ("scenario", scenario)]:
        df.insert(0, col, val)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", nargs="+", default=["A", "B"], choices=["A", "B"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml("experiment.yaml")
    rc = cfg["ring"]
    pools = make_pools(cfg)
    (OUT / "parts").mkdir(parents=True, exist_ok=True)

    for analysis in args.analysis:
        print(f"\n===== analysis {analysis} =====", flush=True)
        for scenario, base, new in SCENARIOS:
            for share in rc["shares"]:
                for density in rc["densities_veh_per_km"]:
                    tag = f"{analysis}_{base}-{new}_{share:.1f}_{density}"
                    f = OUT / "parts" / f"{tag}.csv"
                    if f.exists() and not args.force:
                        continue
                    run_part(analysis, scenario, base, new, share, density,
                             pools, rc).to_csv(f, index=False)
                print(f"  {scenario:18s} share {share:.1f} done", flush=True)
        parts = sorted((OUT / "parts").glob(f"{analysis}_*.csv"))
        pd.concat([pd.read_csv(p) for p in parts]).to_csv(OUT / f"ring_{analysis}.csv",
                                                          index=False)
        print(f"  combined {len(parts)} parts into ring_{analysis}.csv")


if __name__ == "__main__":
    main()
