"""Step 05: string stability of homogeneous and mixed platoons.

Requires steps 03 (calibration), 03b (cross-validation) and 04 (event statistics).
Usage (from the repository root):  python scripts/05_stability.py
Outputs: results/stability/*.csv
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy import stats

from src.config import REPO_ROOT, load_yaml
from src.models import idm, ovrv
from src.stability.mixed import default_omega, mixed_platoon_mc, summarize
from src.stability.string_stability import string_stability, transfer_gain

MODELS = {"idm": idm, "ovrv": ovrv}
CAL = REPO_ROOT / "results" / "calibration"
CV = REPO_ROOT / "results" / "cross_validation"
OUT = REPO_ROOT / "results" / "stability"


def get_params(row, model):
    return {k: row[k] for k in model.PARAM_NAMES}


def homogeneous(cfg, omega, excluded):
    """lambda2, margin M (with range over CV folds) and peak gain per configuration."""
    rows = []
    for name, model in MODELS.items():
        full = pd.read_csv(CAL / f"main_{name}_params.csv")
        folds = pd.read_csv(CV / f"main_{name}_folds.csv")
        for r in full.to_dict("records"):
            p = get_params(r, model)
            fold_p = [get_params(f, model)
                      for f in folds[folds["config"] == r["config"]].to_dict("records")]
            for v in cfg["stability"]["speeds"]:
                res = string_stability(model, p, v)
                fold_m = [string_stability(model, q, v)["margin"] for q in fold_p]
                rows.append(dict(model=name, powertrain=r["powertrain"], config=r["config"],
                                 excluded_from_main=r["config"] in excluded, v_e=v,
                                 lambda2=res["lambda2"], M=res["margin"],
                                 M_fold_min=min(fold_m), M_fold_max=max(fold_m),
                                 max_gain=transfer_gain(*model.partials(v, p), omega).max(),
                                 stable=res["stable"]))
    return pd.DataFrame(rows)


def human_variants(hd):
    """Base parameters plus every combination of the sensitivity values."""
    keys = list(hd["sensitivity"])
    variants = {"base": dict(hd["base"])}
    for values in itertools.product(*(hd["sensitivity"][k] for k in keys)):
        p = dict(hd["base"], **dict(zip(keys, values)))
        if p != hd["base"]:
            variants[", ".join(f"{k}={v}" for k, v in zip(keys, values))] = p
    return variants


def pools_at(v, acc_params, hv_params, lags):
    """Partial derivatives at speed v for every vehicle type."""
    pools = {pt: [idm.partials(v, get_params(r, idm))
                  for r in acc_params[acc_params["powertrain"] == pt].to_dict("records")]
             for pt in ["ICE", "EV"]}
    for T in lags:
        pools[f"HV_T{T}"] = [(*idm.partials(v, hv_params), T)]
    return pools


def scenarios(lags, human=True, acc_mix=True):
    out = []
    if acc_mix:
        out.append(("ICE-ACC + EV-ACC", lambda x: {"ICE": 1 - x, "EV": x}))
    if human:
        for acc in ["ICE", "EV"]:
            for T in lags:
                out.append((f"HV (T_a={T} s) + {acc}-ACC",
                            lambda x, acc=acc, T=T: {f"HV_T{T}": 1 - x, acc: x}))
    return out


def run_mixed(analysis, hv_label, v, pools, scen, mcfg, omega):
    rows = []
    for name, make_shares in scen:
        for x in mcfg["shares"]:
            g = mixed_platoon_mc(pools, make_shares(x), mcfg["n_vehicles"],
                                 mcfg["n_samples"], mcfg["seed"], omega)
            rows.append(dict(analysis=analysis, human_variant=hv_label, v_e=v,
                             scenario=name, share=x, **summarize(g)))
    return rows


def main():
    cfg = load_yaml("experiment.yaml")
    scfg, mcfg, hd = cfg["stability"], cfg["stability"]["mixed"], cfg["human_driver"]
    omega = default_omega(scfg["omega"])
    excluded = list(cfg["parameter_use"]["exclude_from_main"])
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Homogeneous platoons
    hom = homogeneous(cfg, omega, excluded)
    hom.to_csv(OUT / "homogeneous.csv", index=False)
    print(f"1. homogeneous: {len(hom)} rows; any configuration stable: {bool(hom['stable'].any())}")

    # 2. Validation against the measured speed amplification (leader drop 3-5 m/s)
    med = pd.read_csv(REPO_ROOT / "results" / "statistics" / "event_config_medians.csv")
    med = med[med["drop_bin"] == "3-5"][["config", "amplification"]]
    val = hom[(hom["model"] == "idm") & (hom["v_e"] == scfg["v_ref"]) & ~hom["excluded_from_main"]]
    val = val.merge(med, on="config")
    rho = stats.spearmanr(val["max_gain"], val["amplification"])
    val.to_csv(OUT / "validation.csv", index=False)
    print(f"2. model max|G| vs measured amplification: Spearman rho = {rho.statistic:.2f}, "
          f"p = {rho.pvalue:.4f} (n = {len(val)})")

    # 3. Human-driven baseline
    variants = human_variants(hd)
    rows = []
    for label, p in variants.items():
        for v in scfg["speeds"]:
            res = string_stability(idm, p, v)
            for T in mcfg["human_lags"]:
                rows.append(dict(human_variant=label, v_e=v, T_a=T, M_without_lag=res["margin"],
                                 max_gain=transfer_gain(*idm.partials(v, p), omega, T_a=T).max()))
    pd.DataFrame(rows).to_csv(OUT / "human_driver.csv", index=False)
    print(f"3. human driver: {len(variants)} parameter variants")

    # 4. Mixed platoons
    acc_all = pd.read_csv(CAL / "main_idm_params.csv")
    acc_main = acc_all[~acc_all["config"].isin(excluded)]
    lags = mcfg["human_lags"]
    rows = []
    for v in mcfg["speeds"]:                                   # main analysis
        pools = pools_at(v, acc_main, hd["base"], lags)
        rows += run_mixed("main", "base", v, pools, scenarios(lags), mcfg, omega)
    v = scfg["v_ref"]                                          # sensitivity: with Model 3
    pools = pools_at(v, acc_all, hd["base"], lags)
    rows += run_mixed("with_excluded_configs", "base", v, pools, scenarios(lags), mcfg, omega)
    for label, p in variants.items():                          # sensitivity: human parameters
        if label == "base":
            continue
        pools = pools_at(v, acc_main, p, lags)
        rows += run_mixed("human_parameters", label, v, pools,
                          scenarios(lags, acc_mix=False), mcfg, omega)
    mixed = pd.DataFrame(rows)
    mixed.to_csv(OUT / "mixed.csv", index=False)
    print(f"4. mixed platoons: {len(mixed)} rows")

    main_ref = mixed[(mixed["analysis"] == "main") & (mixed["v_e"] == v)]
    print(f"\nMain result at v_e = {v} m/s: per-vehicle amplification, median")
    print(main_ref.pivot_table(index="scenario", columns="share", values="median")
          .round(3).to_string())
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
