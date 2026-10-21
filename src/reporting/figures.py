"""Figures for the revised paper. Each function returns ONE matplotlib Figure (one panel).

Figures are drawn at their final printed size (no scaling in LaTeX), so the font sizes
set here are the sizes the reader sees. acmsmall text width ~5.35 in; a half-width panel
(0.48 \\textwidth) is ~2.6 in. Every half-width panel uses the same fixed layout (axes on
top, legend area below), so panels placed side by side line up exactly.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7, "lines.linewidth": 1.6,
                     "lines.markersize": 5, "axes.linewidth": 0.8,
                     "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42})

PANEL_SIZE = (2.6, 2.6)                 # inches, half-width panel
AXES_RECT = [0.20, 0.36, 0.76, 0.60]    # left, bottom, width, height (fractions)
LEGEND_Y = 0.02                         # legend sits in the bottom area
EV_STYLES = {"IONIQ 5": ("tab:green", "o"), "Polestar 2": ("tab:olive", "s"),
             "Model 3": ("tab:purple", "^")}


def _panel():
    """Half-width figure with a fixed axes rectangle and space for a legend below."""
    fig = plt.figure(figsize=PANEL_SIZE)
    ax = fig.add_axes(AXES_RECT)
    ax.grid(alpha=0.3, lw=0.5)
    return fig, ax


def _legend_below(fig, ax, ncol=2):
    """Legend in the bottom area of the figure, outside the axes."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, LEGEND_Y),
               ncol=ncol, frameon=False, handlelength=1.8, columnspacing=1.0,
               borderaxespad=0.0)


def _pad_ylim(ax, frac=0.08):
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - frac * (hi - lo), hi + frac * (hi - lo))


def _bin_center(label):
    lo, hi = (float(x) for x in label.strip("([]").split(","))
    return (lo + hi) / 2


def fig_steady_headway(steady, gap_group):
    """Figure 2a/2b: steady-state headway vs speed for one gap group.

    gap_group: 'shortest' (ICE min vs EV Short) or 'longest' (ICE max vs EV XLong).
    ICE range over the six vehicles as a band; EV configurations as markers joined by thin
    dotted lines. Only speed bins that contain ICE data are shown.
    """
    ice_gap, ev_gaps = {"shortest": ("min", ["Short"]), "longest": ("max", ["XLong"])}[gap_group]
    s = steady.dropna(subset=["headway"]).copy()
    s["v"] = s["speed_bin"].map(_bin_center)
    s["gap"] = s["config"].str.split().str[-1]
    ice = s[(s["powertrain"] == "ICE") & (s["gap"] == ice_gap)]
    band = ice.groupby("v")["headway"].agg(["min", "max", "median"])

    fig, ax = _panel()
    ax.fill_between(band.index, band["min"], band["max"], color="0.82", lw=0, label="ICE range")
    ax.plot(band.index, band["median"], color="0.35", ls="--", lw=1.2, label="ICE median")
    for config in sorted(s.loc[(s["powertrain"] == "EV") & s["gap"].isin(ev_gaps), "config"].unique()):
        vehicle = " ".join(config.split()[:-1])
        color, marker = EV_STYLES.get(vehicle, ("tab:green", "o"))
        e = s[(s["config"] == config) & s["v"].isin(band.index)].sort_values("v")
        ax.plot(e["v"], e["headway"], color=color, marker=marker, lw=1.2, ls=":", label=vehicle)
    ax.set_xticks(band.index)
    ax.set_xticklabels([f"{v:.0f}" for v in band.index])
    ax.set_xlabel("Follower speed (m/s)")
    ax.set_ylabel("Time gap (s)")
    _pad_ylim(ax)
    _legend_below(fig, ax)
    return fig


GROUP_COLORS = {"ICE": "tab:red", "EV": "tab:green"}


def fig_event_comparison(measured, cf_idm, metric):
    """Figure 3a/3b: one point per configuration, measured and same-leader (IDM).

    metric: 'amplification' or 'delay_s'. Short horizontal bars mark the group medians.
    """
    sources = [("Measured", measured[measured["drop_bin"] == "3-5"]),
               ("Model:\nEV leaders", cf_idm[cf_idm["leader_set"] == "EV-experiment leader"]),
               ("Model:\nICE leaders", cf_idm[cf_idm["leader_set"] == "ICE-experiment leader"])]
    rng = np.random.default_rng(0)
    fig, ax = _panel()
    for i, (name, d) in enumerate(sources):
        for j, pt in enumerate(["ICE", "EV"]):
            x0 = i + (-0.17 if pt == "ICE" else 0.17)
            y = d.loc[d["powertrain"] == pt, metric].dropna().values
            x = x0 + rng.uniform(-0.06, 0.06, len(y))
            ax.scatter(x, y, s=14, color=GROUP_COLORS[pt], alpha=0.8, lw=0,
                       label=pt if i == 0 else None)
            ax.hlines(np.median(y), x0 - 0.12, x0 + 0.12, color="black", lw=1.4,
                      label="Median" if (i == 0 and j == 0) else None)
    if metric == "amplification":
        ax.axhline(1.0, color="0.5", ls="--", lw=0.8, label="No amplification")
        ax.set_ylabel("Speed amplification")
    else:
        ax.set_ylabel("Response delay (s)")
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([s[0] for s in sources])
    ax.set_xlim(-0.5, len(sources) - 0.5)
    ax.grid(axis="x", visible=False)
    _pad_ylim(ax)
    _legend_below(fig, ax)
    return fig


def fig_validation(validation):
    """Figure 4: model peak gain max|G| (IDM, v_ref) vs measured speed amplification.

    validation: validation.csv (config, powertrain, max_gain, amplification)
    """
    from scipy import stats
    fig, ax = _panel()
    for pt in ["ICE", "EV"]:
        d = validation[validation["powertrain"] == pt]
        ax.scatter(d["max_gain"], d["amplification"], s=18, color=GROUP_COLORS[pt],
                   alpha=0.85, lw=0, label=pt)
    lo = min(validation["max_gain"].min(), validation["amplification"].min()) - 0.03
    hi = max(validation["max_gain"].max(), validation["amplification"].max()) + 0.03
    ax.plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=0.8, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    rho = stats.spearmanr(validation["max_gain"], validation["amplification"])
    ax.text(0.04, 0.96, f"Spearman $\\rho$ = {rho.statistic:.2f}\np = {rho.pvalue:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.set_xlabel("Model max|G|")
    ax.set_ylabel("Measured amplification")
    _legend_below(fig, ax, ncol=3)
    return fig


GAP_GROUP = {"min": "shortest", "Short": "shortest", "max": "longest", "XLong": "longest",
             "Medium": "intermediate", "Long": "intermediate"}
GAP_LINESTYLE = {"shortest": "-", "intermediate": ":", "longest": "--"}


def fig_stability_margin(homogeneous):
    """Figure 5: stability margin M (IDM) vs equilibrium speed, one line per configuration.

    Colour = powertrain, line style = gap group. Configurations excluded from the main
    results are not shown. The CV-fold ranges are given in Table 6.
    """
    h = homogeneous[(homogeneous["model"] == "idm") & ~homogeneous["excluded_from_main"]].copy()
    h["gap_group"] = h["config"].str.split().str[-1].map(GAP_GROUP)
    fig, ax = _panel()
    shown = set()
    for config, d in h.groupby("config"):
        pt, gg = d["powertrain"].iloc[0], d["gap_group"].iloc[0]
        key = (pt, gg)
        label = f"{pt}, {gg}" if key not in shown else None
        shown.add(key)
        d = d.sort_values("v_e")
        ax.plot(d["v_e"], d["M"], color=GROUP_COLORS[pt], ls=GAP_LINESTYLE[gg], lw=1.1,
                marker="o", ms=3, label=label)
    ax.axhline(1.0, color="black", lw=0.9, label="M = 1 (stable above)")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(sorted(h["v_e"].unique()))
    ax.set_xticklabels([f"{v:.0f}" for v in sorted(h["v_e"].unique())])
    ax.set_xlabel("Equilibrium speed (m/s)")
    ax.set_ylabel("Stability margin M")
    # order the legend: ICE first, then EV, then the threshold
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)),
                   key=lambda i: (labels[i].startswith("M ="), not labels[i].startswith("ICE"),
                                  ["shortest", "intermediate", "longest"].index(labels[i].split(", ")[1])
                                  if ", " in labels[i] else 0))
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
               bbox_to_anchor=(0.5, LEGEND_Y), ncol=2, frameon=False, handlelength=2.2,
               columnspacing=1.0, borderaxespad=0.0)
    return fig


MIXED_YLIM = (0.97, 1.33)     # same y-range for Figures 6a and 6b


def fig_mixed_acc(mixed, v_ref):
    """Figure 6a: ICE-ACC + EV-ACC; per-vehicle amplification vs EV share (median, 95 %)."""
    d = mixed[(mixed["analysis"] == "main") & (mixed["v_e"] == v_ref)
              & (mixed["scenario"] == "ICE-ACC + EV-ACC")].sort_values("share")
    fig, ax = _panel()
    x = d["share"] * 100
    ax.fill_between(x, d["low"], d["high"], color="0.75", lw=0, label="95 % of platoons")
    ax.plot(x, d["median"], color="black", marker="o", ms=4, label="Median")
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8, label="No amplification")
    ax.set_xticks(x)
    ax.set_xlabel("EV share among ACC vehicles (%)")
    ax.set_ylabel("Per-vehicle amplification")
    ax.set_ylim(*MIXED_YLIM)
    _legend_below(fig, ax)
    return fig


def fig_mixed_human(mixed, v_ref, main_lag=0.5):
    """Figure 6b: human-driven + ACC; median at main_lag, band over the human lag range."""
    d = mixed[(mixed["analysis"] == "main") & (mixed["v_e"] == v_ref)
              & mixed["scenario"].str.startswith("HV")].copy()
    d["acc"] = d["scenario"].str.extract(r"\+ (\w+)-ACC")[0]
    d["lag"] = d["scenario"].str.extract(r"T_a=([\d.]+)")[0].astype(float)
    fig, ax = _panel()
    for acc in ["ICE", "EV"]:
        a = d[d["acc"] == acc]
        band = a.groupby("share")["median"].agg(["min", "max"])
        main = a[a["lag"] == main_lag].sort_values("share")
        ax.fill_between(band.index * 100, band["min"], band["max"], color=GROUP_COLORS[acc],
                        alpha=0.2, lw=0)
        ax.plot(main["share"] * 100, main["median"], color=GROUP_COLORS[acc], marker="o",
                ms=4, label=f"Human + {acc}-ACC")
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8, label="No amplification")
    ax.set_xticks(sorted(d["share"].unique() * 100))
    ax.fill_between([], [], [], color="0.6", alpha=0.35, lw=0, label="Human lag 0-1 s")
    ax.set_xlabel("ACC share (%), rest human-driven")
    ax.set_ylabel("Per-vehicle amplification")
    ax.set_ylim(*MIXED_YLIM)
    _legend_below(fig, ax)
    return fig


RING_SCENARIO_STYLE = {"ICE-ACC + EV-ACC": ("black", "ICE-ACC + EV-ACC"),
                       "HV + ICE-ACC": (GROUP_COLORS["ICE"], "Human + ICE-ACC"),
                       "HV + EV-ACC": (GROUP_COLORS["EV"], "Human + EV-ACC")}


def _ring_growth_panel(ring_a, density, scenarios, xlabel, ylim):
    d = ring_a[(ring_a["density"] == density) & ring_a["in_range"].astype(bool)]
    fig, ax = _panel()
    for scenario in scenarios:
        color, label = RING_SCENARIO_STYLE[scenario]
        # in 10^-3 1/s: short tick labels leave room for the y label in the fixed layout
        s = (d[d["scenario"] == scenario]
             .assign(g=lambda x: 1000 * x["growth_rate"]).groupby("share")["g"])
        stat = pd.DataFrame({"median": s.median(), "low": s.quantile(0.025),
                             "high": s.quantile(0.975)})
        x = stat.index * 100
        ax.fill_between(x, stat["low"], stat["high"], color=color, alpha=0.2, lw=0)
        ax.plot(x, stat["median"], color=color, marker="o", ms=4, label=label)
    ax.fill_between([], [], [], color="0.6", alpha=0.35, lw=0, label="95 % of fleets")
    ax.axhline(0.0, color="0.5", ls="--", lw=0.8, label="No growth")
    ax.set_xticks(sorted(d["share"].unique() * 100))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Growth rate ($10^{-3}$ 1/s)")
    ax.set_ylim(*(1000 * y for y in ylim))
    _legend_below(fig, ax)
    return fig


def fig_ring_growth(ring_a, density=25, ylim=(-0.03, 0.05)):
    """Figure 7a/7b: growth rate of a small perturbation on the ring (analysis A).

    Returns (fig_a: ICE-ACC + EV-ACC vs EV share, fig_b: human + ACC vs ACC share).
    Median over 30 seeds; band = 2.5-97.5 % of the seeds.
    """
    fig_a = _ring_growth_panel(ring_a, density, ["ICE-ACC + EV-ACC"],
                               "EV share among ACC vehicles (%)", ylim)
    fig_b = _ring_growth_panel(ring_a, density, ["HV + ICE-ACC", "HV + EV-ACC"],
                               "ACC share (%), rest human-driven", ylim)
    return fig_a, fig_b


def fig_flow_density(ring_b):
    """Figure 8 (supplementary): final-state flow vs density, ICE-ACC + EV-ACC by EV share,
    with the human-only fleet for reference. Median over seeds. Long runs with a common
    minimum spacing; stop-and-go relies on model extrapolation below the calibrated range.
    """
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ice_ev", [GROUP_COLORS["ICE"], GROUP_COLORS["EV"]])
    d = ring_b[ring_b["scenario"] == "ICE-ACC + EV-ACC"]
    fig, ax = _panel()
    for share, s in d.groupby("share"):
        m = s.groupby("density")["flow"].median()
        # hide densities where most fleets cannot move at all: the extrapolated s0 of
        # some ICE configurations exceeds the available net gap (not a traffic result)
        jam = s.groupby("density")["regime"].apply(lambda r: (r == "jam").mean())
        m = m[jam <= 0.5]
        ax.plot(m.index, m.values, color=cmap(share), marker="o", ms=3, lw=1.2,
                label=f"{share * 100:.0f} % EV")
    hv = ring_b[(ring_b["scenario"] == "HV + EV-ACC") & (ring_b["share"] == 0.0)]
    m = hv.groupby("density")["flow"].median()
    ax.plot(m.index, m.values, color="tab:blue", ls="--", lw=1.2, label="Human only")
    ax.set_xlabel("Density (veh/km)")
    ax.set_ylabel("Flow (veh/h)")
    ax.set_ylim(bottom=0)
    _legend_below(fig, ax, ncol=3)
    return fig





def fig_framework():
    """Figure 1: analysis framework (full width), with orthogonal arrows. Merges and
    splits share one vertical line and a single arrow head per box."""
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(5.3, 3.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fill = {"data": "#dbe8f5", "emp": "#fbe3cc", "model": "#d9efd9"}
    W, H = 0.185, 0.14
    X = [0.105, 0.365, 0.635, 0.895]                        # column centres
    GAP = [(X[i] + W / 2 + X[i + 1] - W / 2) / 2 for i in range(3)]
    LINE = dict(color="0.25", lw=0.8, solid_capstyle="butt", zorder=1)

    def box(col, y, text, kind):
        x = X[col]
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H,
                                    boxstyle="round,pad=0.006,rounding_size=0.015",
                                    fc=fill[kind], ec="0.3", lw=0.7, zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=8, linespacing=1.15,
                zorder=3)
        return (x, y)

    def line(points):
        xs, ys = zip(*points)
        ax.plot(xs, ys, **LINE)

    def head(p, q):
        """Straight segment from p to q with an arrow head at q."""
        ax.annotate("", xy=q, xytext=p, zorder=1,
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.25",
                                    shrinkA=0, shrinkB=1.5, mutation_scale=8))

    right = lambda c: (c[0] + W / 2, c[1])
    left = lambda c: (c[0] - W / 2, c[1])
    top = lambda c: (c[0], c[1] + H / 2)
    bottom = lambda c: (c[0], c[1] - H / 2)

    ice = box(0, 0.80, "ICE ACC data\n6 vehicles", "data")
    ev = box(0, 0.56, "EV ACC data\n3 vehicles", "data")
    pre = box(1, 0.68, "Harmonized\npreprocessing\nand QC", "data")
    steady = box(2, 0.90, "Matched\nsteady-state\ncomparison", "emp")
    events = box(2, 0.68, "Event-level\nresponse metrics\n(measured)", "emp")
    cal = box(2, 0.44, "Identifiability-\naware\ncalibration", "model")
    cv = box(3, 0.44, "Cross-validation,\nbaselines,\nlag test", "model")
    cf = box(1, 0.20, "Same-leader test\n(same event\nmetrics)", "model")
    stab = box(2, 0.20, "String stability\n(validated\nvs. data)", "model")
    ring = box(3, 0.20, "Mixed platoons,\nring-road\nsimulation", "model")

    # merge: both datasets -> one vertical line -> one arrow into preprocessing
    xm = GAP[0]
    for c in (ice, ev):
        line([right(c), (xm, c[1])])
    line([(xm, ice[1]), (xm, ev[1])])
    head((xm, pre[1]), left(pre))

    # split: preprocessing -> one vertical line -> one arrow into each analysis box
    xs = GAP[1]
    line([right(pre), (xs, pre[1])])
    line([(xs, steady[1]), (xs, cal[1])])
    for c in (steady, events, cal):
        head((xs, c[1]), left(c))

    head(right(cal), left(cv))
    y_mid = (bottom(cal)[1] + top(cf)[1]) / 2
    line([bottom(cal), (cal[0], y_mid), (cf[0], y_mid)])
    head((cf[0], y_mid), top(cf))
    head(right(cf), left(stab)); head(right(stab), left(ring))

    for i, (kind, label) in enumerate([("data", "Data"), ("emp", "Empirical analysis"),
                                       ("model", "Model-based analysis")]):
        x0 = 0.03 + i * 0.27
        ax.add_patch(FancyBboxPatch((x0, 0.02), 0.025, 0.035, boxstyle="round,pad=0.002",
                                    fc=fill[kind], ec="0.3", lw=0.6))
        ax.text(x0 + 0.035, 0.037, label, fontsize=8, va="center")
    return fig


def fig_space_time(t, v, v_max, t_exit=None):
    """Space-time diagram of one ring-road run: vehicle index vs time, colour = speed.

    t: time stamps (n_steps,); v: speeds (n_steps, n_vehicles); v_max: speed cap.
    Vehicle 1 is perturbed; vehicle i follows vehicle i-1, so waves run diagonally.
    The colour bar sits below the axes, so all panels share the same axes rectangle.
    t_exit: if given, a dashed line marks the first time a vehicle leaves the calibrated
    speed range; later parts rely on model extrapolation.
    """
    step = max(1, int(round(1.0 / (t[1] - t[0]))))          # one column per second
    fig, ax = _panel()
    im = ax.imshow(v[::step].T, origin="lower", aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=v_max, interpolation="nearest",
                   extent=[t[0], t[-1], 0.5, v.shape[1] + 0.5])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vehicle")
    ax.set_yticks([1, 10, 20, 30])
    ax.grid(False)
    if t_exit is not None and np.isfinite(t_exit):
        ax.axvline(t_exit, color="black", ls="--", lw=1.0)
    cax = fig.add_axes([AXES_RECT[0], 0.18, AXES_RECT[2], 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Speed (m/s)", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    return fig



# ---------------------------------------------------------------------------
# Journal style: Digital Transportation and Safety. Times New Roman (Liberation Serif
# has identical metrics and is used where Times New Roman is not installed), all text
# 8 pt (allowed: 8-10 pt), panels of composite figures labelled a, b, c (not bold).
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix",
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})


def add_panel_label(fig, letter):
    """Panel letter (a, b, c) in the top-left corner of a figure, not bold."""
    fig.text(0.02, 0.98, letter, ha="left", va="top", fontsize=9)
    return fig


def fig_trajectory_speed(run):
    """Example run: leader and follower speed."""
    fig, ax = _panel()
    t = run["t"] - run["t"].iloc[0]
    ax.plot(t, run["v_l"], color="0.45", lw=1.0, label="Leader")
    ax.plot(t, run["v_f"], color="black", lw=1.0, ls="--", label="Follower (ACC)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    _legend_below(fig, ax)
    return fig


def fig_trajectory_gap(run):
    """Example run: net gap (bumper to bumper)."""
    fig, ax = _panel()
    t = run["t"] - run["t"].iloc[0]
    ax.plot(t, run["s"], color="black", lw=1.0, label="Net gap")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Net gap (m)")
    ax.set_ylim(bottom=0)
    return fig


def fig_model_fit(run, sims):
    """Measured net gap and the gaps simulated by calibrated models.

    sims: {label: simulated gap array (same length as run)}
    """
    styles = {"IDM": dict(color="tab:blue", ls="--"), "OVRV": dict(color="tab:orange", ls=":")}
    fig, ax = _panel()
    t = run["t"] - run["t"].iloc[0]
    ax.plot(t, run["s"], color="black", lw=1.2, label="Measured")
    for name, s in sims.items():
        ax.plot(t, s, lw=1.2, label=name, **styles.get(name, {}))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Net gap (m)")
    ax.set_ylim(bottom=0)
    _legend_below(fig, ax, ncol=3)
    return fig


# ---------------------------------------------------------------------------
# Added figures: gap definition (Data section) and event-aligned response (Section 3.2)
# ---------------------------------------------------------------------------
def fig_gap_definition():
    """Schematic of the reported spacing in the two datasets and the conversion to the net gap.

    Full-width figure (6.3 in); panel letters are part of the drawing.
    """
    from matplotlib.patches import Circle, FancyBboxPatch
    grey, red, green = "#8c8c8c", "tab:red", "tab:green"
    L, GAP, H = 4.8, 7.0, 1.5            # schematic lengths (m); the gap is not to scale

    def car(ax, x0, color, label):
        ax.add_patch(FancyBboxPatch((x0, 0.35), L, H, boxstyle="round,pad=0,rounding_size=0.5",
                                    fc=color, ec="black", lw=0.6, alpha=0.9, zorder=2))
        for dx in (0.9, L - 0.9):
            ax.add_patch(Circle((x0 + dx, 0.35), 0.35, fc="white", ec="black", lw=0.6, zorder=3))
        ax.plot([x0 + L - 1.3] * 2, [0.35 + H * 0.55, 0.35 + H * 0.9], color="black", lw=0.6, zorder=3)
        ax.text(x0 + L / 2 - 0.4, 0.35 + H / 2, label, ha="center", va="center", fontsize=8, zorder=4)

    def dim(ax, x1, x2, y, text, color="black", ytxt=None, va="bottom"):
        ax.annotate("", xy=(x1, y), xytext=(x2, y),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=0.8, shrinkA=0, shrinkB=0))
        ax.text((x1 + x2) / 2, ytxt if ytxt is not None else y + 0.2, text,
                ha="center", va=va, fontsize=8, color=color)

    with plt.rc_context({"font.family": "serif",
                         "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix"}):
        fig, axes = plt.subplots(2, 1, figsize=(6.3, 3.1))
        xf, xl = 0.0, L + GAP
        for ax in axes:
            ax.set_xlim(-1.5, xl + L + 2.5)
            ax.set_ylim(-3.2, 3.9)
            ax.axis("off")
            ax.plot([-1.5, xl + L + 2.5], [0.35, 0.35], color="black", lw=0.6, zorder=1)
            car(ax, xf, red if ax is axes[0] else green, "follower")
            car(ax, xl, grey, "leader")
        a, b = axes
        a.annotate("", xy=(xl + L + 2.4, 3.0), xytext=(xl + L - 1.2, 3.0),
                   arrowprops=dict(arrowstyle="->", lw=0.8))
        a.text(xl + L - 1.4, 3.0, "driving direction", ha="right", va="center", fontsize=8)
        a.text(-1.5, 3.6, "a", fontsize=8, ha="left", va="center")
        a.text(-0.9, 3.6, "ICE data: reported spacing is the net gap", fontsize=8, ha="left", va="center")
        dim(a, xf + L, xl, -0.9, r"net gap $s$ (bumper to bumper)", ytxt=-1.2, va="top")
        b.text(-1.5, 3.6, "b", fontsize=8, ha="left", va="center")
        b.text(-0.9, 3.6, "EV data: reported spacing is front to front", fontsize=8, ha="left", va="center")
        dim(b, xf + L, xl, -0.9, r"net gap $s = s_{\mathrm{raw}} - L_l$", ytxt=-1.2, va="top")
        dim(b, xl, xl + L, -0.9, r"leader length $L_l$", color=grey, ytxt=-1.2, va="top")
        dim(b, xf + L, xl + L, 2.55, r"reported spacing $s_{\mathrm{raw}}$", ytxt=2.7, va="bottom")
        for x in (xf + L, xl + L):
            b.plot([x, x], [0.35 + H, 2.55], color="black", lw=0.5, ls=":", zorder=1)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.05)
    return fig


def fig_event_response(curves, powertrain, delay_median, xlim=(-3, 15), ylim=(-0.1, 1.6)):
    """Speed response around leader decelerations (leader drops of 3-5 m/s), one powertrain.

    Speed drop of leader and follower, normalized by the leader drop and aligned at the start
    of the leader deceleration. Thin lines: median curve of each configuration; thick lines:
    median over configurations. Values above one: the follower slows down more than its leader.
    delay_median: configuration-level median response delay (s), from event_config_medians.csv.
    """
    color = {"ICE": "tab:red", "EV": "tab:green"}[powertrain]
    c = curves[curves["powertrain"] == powertrain]
    per_cfg = c.groupby(["config", "t"])[["leader", "follower"]].median().reset_index()
    overall = per_cfg.groupby("t")[["leader", "follower"]].median()

    fig, ax = _panel()
    for i, (_, g) in enumerate(per_cfg.groupby("config")):
        ax.plot(g["t"], g["follower"], color=color, alpha=0.3, lw=0.7,
                label="Configurations" if i == 0 else None)
    ax.plot(overall.index, overall["leader"], color="0.35", ls="--", lw=1.4, label="Leader")
    ax.plot(overall.index, overall["follower"], color=color, lw=1.6, label="Follower")
    for y in (0.5, 1.0):
        ax.axhline(y, color="0.5", lw=0.6, ls=":")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Time from event start (s)")
    ax.set_ylabel("Normalized speed drop")
    ax.text(0.04, 0.94, f"Median delay: {delay_median:.2f} s", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5)
    _legend_below(fig, ax, ncol=2)
    return fig
