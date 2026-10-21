"""Counterfactual 'same leader' test: every calibrated vehicle follows the same leaders.

Removes differences in the leader disturbances between the two experimental campaigns:
any remaining difference between ICE and EV configurations comes from the vehicles.
"""
import warnings

import numpy as np
import pandas as pd

from src.analysis.events import run_events
from src.models.simulate import run_model


def same_leader_events(model, params_table, leaders):
    """Simulate every configuration behind every leader and extract its events.

    params_table: calibrated parameters, one row per configuration (columns: config,
                  powertrain, and the model parameters)
    leaders     : {leader_set_name: {run_id: dict(t, v_l, a_l)}}
    Each simulation starts at the configuration's own equilibrium for the initial
    leader speed. Collided simulations are skipped with a warning.
    """
    tables, skipped = [], []
    for row in params_table.to_dict("records"):
        p = {k: row[k] for k in model.PARAM_NAMES}
        for set_name, runs in leaders.items():
            for run_id, lead in runs.items():
                t, v_l, a_l = lead["t"], lead["v_l"], lead["a_l"]
                s_init = model.equilibrium_spacing(v_l[0], p)
                sim = run_model(model, p, t, v_l, s_init, v_l[0])
                if sim["collided"]:
                    skipped.append(f"{row['config']} / {run_id}")
                    continue
                run = dict(t=t, v_l=v_l, a_l=a_l, v_f=sim["v"], s=sim["s"],
                           a_f=np.nan_to_num(sim["a"]))
                ev = run_events(run)
                ev["leader_run"] = run_id
                ev["leader_set"] = set_name
                ev["config"] = row["config"]
                ev["powertrain"] = row["powertrain"]
                tables.append(ev)
    if skipped:
        warnings.warn(f"{len(skipped)} collided simulations skipped: {', '.join(skipped)}")
    ev = pd.concat(tables, ignore_index=True)
    ev["overshoot"] = ev["drop_follower"] - ev["drop_leader"]
    return ev
