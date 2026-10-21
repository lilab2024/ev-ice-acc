"""Uncertainty summaries used in the reporting stage."""
import numpy as np
from scipy import stats


def bootstrap_ci(x, stat="mean", n_boot=10000, level=0.95, seed=0):
    """Statistic of x and its percentile bootstrap confidence interval.

    stat: 'mean' or 'median'. Returns (estimate, low, high). NaN values are dropped.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    f = np.mean if stat == "mean" else np.median
    rng = np.random.default_rng(seed)
    boot = f(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    alpha = (1 - level) / 2
    return float(f(x)), float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


def paired_difference(a, b, n_boot=10000, seed=0):
    """Paired difference a - b: mean, bootstrap CI, count of a < b, Wilcoxon p-value.

    a and b are matched element by element (e.g. the same configurations).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    mean, low, high = bootstrap_ci(d, "mean", n_boot, seed=seed)
    p = stats.wilcoxon(a[ok], b[ok]).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan
    return dict(n=int(len(d)), mean_diff=mean, ci_low=low, ci_high=high,
                a_smaller=int((d < 0).sum()), wilcoxon_p=p)


def censored_median(times, limit):
    """Median of event times where NaN means 'not observed before limit'.

    Censored values are set to +inf, so the median is correct as long as fewer than half
    of the values are censored; otherwise the median is reported as > limit.
    Returns (median or np.inf, share censored, label for tables).
    """
    t = np.asarray(times, dtype=float)
    censored = ~np.isfinite(t)
    med = float(np.median(np.where(censored, np.inf, t)))
    label = f"> {limit:.0f}" if not np.isfinite(med) else f"{med:.0f}"
    return med, float(censored.mean()), label


def two_sample_difference(a, b, stat="median", n_boot=10000, level=0.95, seed=0):
    """Difference stat(a) - stat(b) of two independent groups and its bootstrap CI.

    Each group is resampled separately (no pairing). Returns (estimate, low, high).
    """
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    f = np.mean if stat == "mean" else np.median
    rng = np.random.default_rng(seed)
    da = f(rng.choice(a, size=(n_boot, len(a)), replace=True), axis=1)
    db = f(rng.choice(b, size=(n_boot, len(b)), replace=True), axis=1)
    alpha = (1 - level) / 2
    diff = da - db
    return float(f(a) - f(b)), float(np.quantile(diff, alpha)), float(np.quantile(diff, 1 - alpha))
