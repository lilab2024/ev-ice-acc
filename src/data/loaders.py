"""Raw data loading for the ICE and EV ACC datasets."""
import hashlib
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Pattern for ICE file names, e.g. VehA_osc_min.csv
ICE_NAME_RE = re.compile(
    r"^Veh(?P<veh>[A-Z])_(?P<profile>dip|osc|stepH|stepL)_(?P<setting>min|max)$")


def list_ice_runs(ice_dir, ice_cfg):
    """Return one row per ICE run, with metadata parsed from the file name."""
    rows = []
    for f in sorted(Path(ice_dir).glob("*.csv")):
        m = ICE_NAME_RE.match(f.stem)
        if m is None:
            warnings.warn(f"Unrecognized ICE file name, skipped: {f.name}")
            continue
        info = ice_cfg.get(m["veh"])
        if info is None:
            raise KeyError(f"ICE vehicle {m['veh']} not found in vehicles.yaml ({f.name})")
        # run_id keeps the raw vehicle code from the file name, which differs from
        # the paper name for C-G (e.g. ICE_VehD_* is VehC in the paper; raw C is
        # excluded). The paper name is in the column "vehicle".
        rows.append(dict(
            run_id=f"ICE_Veh{m['veh']}_{m['profile']}_{m['setting']}",
            powertrain="ICE",
            raw_vehicle=f"Veh{m['veh']}",
            vehicle=info.get("paper_name"),
            gap=m["setting"],
            profile=m["profile"],
            exclude=bool(info.get("exclude", False)),
            exclude_reason=info.get("reason", ""),
            path=str(f),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EV runs
# ---------------------------------------------------------------------------
# Folder name for the desired-speed offset, e.g. 0_desired, 5_desired
DESIRED_RE = re.compile(r"^(?P<x>\d+)_desired$")


def _md5(path):
    """Content hash of a file, used to detect duplicate copies."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def list_ev_runs(ev_dir, ev_cfg):
    """Return one row per EV run, with metadata parsed from the folder path.

    Expected path: <vehicle>/<gap>/<X>_desired/<v_free_mph>/<v_reduced_mph>/<run>.csv
    """
    ev_dir = Path(ev_dir)
    rows = []
    for f in ev_dir.rglob("*.csv"):
        parts = f.relative_to(ev_dir).parts
        if len(parts) != 6:
            warnings.warn(f"Unexpected folder depth, skipped: {f}")
            continue
        veh_folder, gap, desired_dir, v_free, v_red, _ = parts
        m = DESIRED_RE.match(desired_dir)
        info = ev_cfg.get(veh_folder)
        if m is None or info is None or not (v_free.isdigit() and v_red.isdigit()):
            warnings.warn(f"Unrecognized EV path, skipped: {f}")
            continue
        rows.append(dict(
            powertrain="EV",
            raw_vehicle=veh_folder,
            vehicle=info.get("paper_name"),
            gap=gap,
            profile="dip",
            desired_mph=int(m["x"]),
            v_free_mph=int(v_free),
            v_reduced_mph=int(v_red),
            run=f.stem,
            exclude=bool(info.get("exclude", False)),
            exclude_reason=info.get("reason", ""),
            path=str(f),
        ))
    df = pd.DataFrame(rows)

    # Remove duplicate copies (e.g. "1 (1).csv") created by Drive uploads.
    # Only files in the same folder with identical content count as duplicates;
    # the copy without parentheses in its name is kept.
    df["md5"] = df["path"].map(_md5)
    df["folder"] = df["path"].map(lambda p: str(Path(p).parent))
    df["is_copy"] = df["run"].str.contains(r"\(")
    df = df.sort_values(["folder", "md5", "is_copy"])
    dup = df.duplicated(["folder", "md5"], keep="first")
    for p in df.loc[dup, "path"]:
        warnings.warn(f"Duplicate file ignored: {p}")
    df = df.loc[~dup].drop(columns=["md5", "folder", "is_copy"])

    # run_id keeps the raw folder name (e.g. Tesla = Model 3, Polestar = Polestar 2);
    # the paper name is in the column "vehicle".
    df["run_id"] = ("EV_" + df["raw_vehicle"].str.replace(" ", "") + "_" + df["gap"]
                    + "_d" + df["desired_mph"].astype(str)
                    + "_" + df["v_free_mph"].astype(str) + "to" + df["v_reduced_mph"].astype(str)
                    + "_r" + df["run"])
    if df["run_id"].duplicated().any():
        raise ValueError("Duplicate run_id found: "
                         + ", ".join(df.loc[df["run_id"].duplicated(), "run_id"]))
    return df.sort_values("run_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Combined manifest
# ---------------------------------------------------------------------------
MANIFEST_COLUMNS = ["run_id", "powertrain", "raw_vehicle", "vehicle", "gap", "profile",
                    "desired_mph", "v_free_mph", "v_reduced_mph", "run",
                    "exclude", "exclude_reason", "path"]


def build_manifest(paths, vehicles_cfg):
    """Combine ICE and EV runs into one table with a fixed set of columns.

    Columns that do not apply to a dataset (e.g. desired_mph for ICE) are left empty.
    """
    ice = list_ice_runs(paths["ice"], vehicles_cfg["ice"])
    ice["run"] = "1"                      # each ICE file is a single run
    ev = list_ev_runs(paths["ev"], vehicles_cfg["ev"])
    manifest = pd.concat([ice, ev], ignore_index=True)
    return manifest.reindex(columns=MANIFEST_COLUMNS)


# ---------------------------------------------------------------------------
# Load a single run
# ---------------------------------------------------------------------------
KMH_TO_MS = 1 / 3.6


def load_raw_run(row):
    """Load one run and return DataFrame[time, v_f, v_l, s] in SI units.

    ICE files have NO header row. Column order (time, follower speed, leader speed,
    spacing) was verified with the kinematic check ds/dt = v_l - v_f.
    EV files: only the time, speed (km/h) and spacing columns are used; smoothing is
    applied later to both datasets in the same way.
    """
    if row["powertrain"] == "ICE":
        d = pd.read_csv(row["path"], header=None, names=["time", "v_f", "v_l", "s"])
    else:
        d = pd.read_csv(row["path"],
                        usecols=["Time", "Speed Follower", "Speed Leader", "Spacing"])
        d = d.rename(columns={"Time": "time", "Speed Follower": "v_f",
                              "Speed Leader": "v_l", "Spacing": "s"})
        d[["v_f", "v_l"]] *= KMH_TO_MS

    d = d.apply(pd.to_numeric, errors="coerce")
    n_bad = int(d.isna().any(axis=1).sum())
    if n_bad:
        warnings.warn(f"{n_bad} rows with missing or non-numeric values dropped: "
                      f"{row['path']}")
    d = d.dropna().reset_index(drop=True)

    dt = np.diff(d["time"].values)
    if not np.all(dt > 0):
        raise ValueError(f"Time stamps are not strictly increasing: {row['path']}")
    if dt.max() > 1.5 * np.median(dt):
        warnings.warn(f"Gap in time stamps (max {dt.max():.3f} s, median "
                      f"{np.median(dt):.3f} s): {row['path']}")
    d["time"] -= d["time"].iloc[0]          # start every run at t = 0
    return d
