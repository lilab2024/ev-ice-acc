"""Table of parameters and bounds of all calibrated models, generated from config/experiment.yaml."""
import pandas as pd

# label and unit of each parameter, in the notation of the paper
LABELS = {
    "idm": {"v0": ("desired speed $v_0$", "m/s"), "delta": ("acceleration exponent $\\delta$", ""),
            "s0": ("standstill gap $s_0$", "m"), "tau": ("time gap $\\tau$", "s"),
            "alpha": ("maximum acceleration $\\alpha$", "m/s$^2$"),
            "beta": ("comfortable deceleration $\\beta$", "m/s$^2$")},
    "ovrv": {"k1": ("gap gain $k_1$", "s$^{-2}$"), "k2": ("relative-speed gain $k_2$", "s$^{-1}$"),
             "eta": ("standstill gap $\\eta$", "m"), "tau": ("time gap $\\tau$", "s")},
    "newell": {"tau": ("time shift $\\tau$", "s")},
    "gipps": {"tau": ("reaction time $\\tau$", "s"), "a": ("maximum acceleration $a$", "m/s$^2$"),
              "B": ("braking rate $B$ (also used as estimate of the leader's)", "m/s$^2$"),
              "S": ("standstill distance $S$", "m"), "V": ("desired speed $V$", "m/s")},
}
T_A = ("actuation lag $T_a$", "s")


def _fmt(value, unit):
    u = f"~{unit}" if unit else ""
    if isinstance(value, (list, tuple)):
        return f"[{value[0]:g}, {value[1]:g}]{u}"
    return f"{value:g}{u} (fixed)"


def _rows(group, spec, keys=None):
    return [(label, _fmt(spec[k], unit)) for k, (label, unit) in LABELS[group].items()
            if k in spec and (keys is None or k in keys)]


def bounds_table(cfg):
    """DataFrame with columns Model, Parameter, Bounds (Model empty for continuation rows)."""
    P = cfg["calibration"]["params"]
    rows = []

    def add(model, items):
        for i, (par, val) in enumerate(items):
            rows.append((model if i == 0 else "", par, val))

    add("IDM (main)", _rows("idm", P["idm"]))

    rest = ["s0", "tau", "alpha", "beta"]
    free = _rows("idm", P["idm_free"], keys=["v0", "delta"])
    if all(P["idm_free"][k] == P["idm"][k] for k in rest):
        free.append(("$s_0$, $\\tau$, $\\alpha$, $\\beta$", "as IDM (main)"))
    else:
        free += _rows("idm", P["idm_free"], keys=rest)
    add("IDM (all parameters free)", free)

    add("OVRV", _rows("ovrv", P["ovrv"]))
    assert P["idm_lag"]["T_a"] == P["ovrv_lag"]["T_a"]
    add("IDM + lag, OVRV + lag", [("parameters of IDM (main) or OVRV", "as above"),
                                  (T_A[0], _fmt(P["idm_lag"]["T_a"], T_A[1]))])
    add("Newell", _rows("newell", P["newell"]))
    add("Gipps", _rows("gipps", P["gipps"]))
    return pd.DataFrame(rows, columns=["Model", "Parameter", "Bounds or fixed value"])


def bounds_latex(df):
    """Publication LaTeX (booktabs, no vertical rules); one block per model."""
    lines = [r"\begin{tabular}{lll}", r"\toprule",
             r"Model & Parameter & Bounds or fixed value \\", r"\midrule"]
    for i, r in enumerate(df.itertuples(index=False)):
        if r[0] and i > 0:
            lines.append(r"\midrule")
        lines.append(f"{r[0]} & {r[1]} & {r[2]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"
