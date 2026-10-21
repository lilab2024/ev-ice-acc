"""Event-level analysis: how the follower responds to each leader deceleration."""
import warnings

import numpy as np
import pandas as pd

METRIC_COLUMNS = ["t_start", "v_l_before", "v_l_min", "drop_leader", "drop_follower",
                  "amplification", "peak_decel_leader", "peak_decel_follower",
                  "decel_ratio", "delay_s", "min_headway", "min_spacing"]


def find_leader_decelerations(t, v_l, a_l, min_drop=1.5, a_thr=0.3, merge_gap_s=1.0,
                              pre_s=3.0, max_settle_s=10.0):
    """Leader deceleration events, detected from the leader acceleration.

    A deceleration phase is a stretch where a_l < -a_thr; phases separated by less than
    merge_gap_s are merged. The leader minimum is searched from the phase start up to
    max_settle_s after the phase, stopping early if the leader starts to accelerate.
    A phase that starts before the previous event's minimum belongs to the same
    deceleration and is merged with it. The drop is measured from the mean speed over
    pre_s before the event. Returns a list of (i_start, i_min) index pairs.
    """
    dt = t[1] - t[0]
    braking = a_l < -a_thr
    starts = np.flatnonzero(braking & ~np.r_[False, braking[:-1]])
    ends = np.flatnonzero(braking & ~np.r_[braking[1:], False])

    phases = []
    for s, e in zip(starts, ends):
        if phases and (s - phases[-1][1]) * dt < merge_gap_s:
            phases[-1][1] = e
        else:
            phases.append([s, e])

    candidates = []
    for s, e in phases:
        stop = min(len(t) - 1, e + int(max_settle_s / dt))
        accelerating = np.flatnonzero(a_l[e:stop + 1] > a_thr)
        if len(accelerating):
            stop = e + accelerating[0]
        i_min = s + int(np.argmin(v_l[s:stop + 1]))
        if candidates and s <= candidates[-1][1]:
            # starts before the previous minimum: same deceleration, merge
            prev_s, prev_min = candidates[-1]
            candidates[-1] = (prev_s, i_min if v_l[i_min] < v_l[prev_min] else prev_min)
        else:
            candidates.append((s, i_min))

    events = []
    for s, i_min in candidates:
        v_before = v_l[max(0, s - int(pre_s / dt)):s + 1].mean()
        if v_before - v_l[i_min] >= min_drop:
            events.append((int(s), int(i_min)))
    return events


def _half_drop_time(t, v, v_before, drop):
    """First time the speed falls below v_before - drop / 2."""
    below = np.flatnonzero(v <= v_before - drop / 2)
    return t[below[0]] if len(below) else np.nan


def event_metrics(run, i_start, i_min, post_s=15.0, pre_s=3.0):
    """Metrics of the follower's response to one leader deceleration event."""
    t, v_l, v_f, s = run["t"], run["v_l"], run["v_f"], run["s"]
    a_l, a_f = run["a_l"], run["a_f"]
    dt = t[1] - t[0]
    i_pre = max(0, i_start - int(pre_s / dt))
    i_end = min(len(t) - 1, i_min + int(post_s / dt))
    win = slice(i_start, i_end + 1)

    v_l_before = v_l[i_pre:i_start + 1].mean()
    v_f_before = v_f[i_pre:i_start + 1].mean()
    drop_l = v_l_before - v_l[i_min]
    drop_f = v_f_before - v_f[win].min()

    # Response delay: time between leader and follower reaching half of their drop
    delay = (_half_drop_time(t[win], v_f[win], v_f_before, drop_f)
             - _half_drop_time(t[win], v_l[win], v_l_before, drop_l))

    # 0.45 m/s: same threshold as preprocess.min_speed_for_headway in experiment.yaml
    headway = s[win] / np.where(v_f[win] > 0.45, v_f[win], np.nan)
    return dict(t_start=t[i_start], v_l_before=v_l_before, v_l_min=v_l[i_min],
                drop_leader=drop_l, drop_follower=drop_f,
                amplification=drop_f / drop_l,
                peak_decel_leader=-a_l[win].min(), peak_decel_follower=-a_f[win].min(),
                decel_ratio=a_f[win].min() / a_l[win].min(),
                delay_s=delay, min_headway=np.nanmin(headway), min_spacing=s[win].min())


def run_events(run, min_drop=1.5, post_s=15.0):
    """All leader deceleration events of one run as a DataFrame.

    The follower response is evaluated up to post_s after the leader minimum. A warning is
    issued if this window reaches the start of the next event (it does not in the present
    data), since the follower minimum could then belong to the next event.
    """
    pairs = find_leader_decelerations(run["t"], run["v_l"], run["a_l"], min_drop)
    dt = run["t"][1] - run["t"][0]
    for (_, i_min), (next_start, _) in zip(pairs[:-1], pairs[1:]):
        if next_start <= i_min + int(post_s / dt):
            warnings.warn(f"Event window reaches the next event at "
                          f"t = {run['t'][next_start]:.1f} s")
    rows = [event_metrics(run, i0, i1, post_s=post_s) for i0, i1 in pairs]
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)
