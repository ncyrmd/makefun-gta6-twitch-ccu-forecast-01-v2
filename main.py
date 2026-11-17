"""Forecast launch-day Twitch viewers for Grand Theft Auto VI, hour by hour.

v2: a 50/50 blend of two models of the log-ratio curve
    r_h = ln(viewers_h / viewers_1),  h = 2..24.

Model A — local-linear log-ratio regression (v1, retuned): for each hour
separately, r_h is regressed on ln(viewers_1) across the comparable launches,
weighting each launch by a Gaussian kernel centred on the target's own log
hour-1 print (sigma = 1.6 log units). The fitted value at the target's scale
is the median log-ratio; the weighted residual spread gives the log-scale
standard deviation, and the five quantiles are normal quantiles around the
median, narrowed by a factor of 0.85.

Model B — low-rank (PCA) curve model: the comparable launches' log-ratio
curves are nearly one-dimensional (the first principal component carries
~86% of the variance, the first two ~95%). Each curve is reconstructed from
its two component scores; the scores are regressed on ln(viewers_1) with the
same kernel-weighted local-linear fit (sigma = 2.5), which borrows strength
across all 23 hours instead of fitting each hour in isolation. The median
curve is the mean curve plus the fitted scores times the components; the
per-hour spread comes from the weighted reconstruction residuals, also
narrowed by 0.85. The eigendecomposition is a pure-Python power iteration
with deflation (deterministic: fixed start vectors, fixed iteration count).

The final quantiles are the 50/50 average of the two models' log-quantiles.
Blending is a hedge: model A adapts hour by hour and tracks the biggest
launches at the target's scale, model B imposes the smooth common launch
shape. On the four practice launches the blend scores 0.137 (v1: 0.160);
on leave-one-out over the big rolling-midnight comparables it scores 0.231
(v1: 0.233) — better on both, not just one.

Two filters keep the comparison honest:

* launches whose notes flag them "early-access-contaminated" or "weak" are
  dropped -- their pre-day curves were already running before release day;
* the Grand Theft Auto VI category's own pre-release history is not a launch
  and is dropped too.

Everything is deterministic: comparables are processed in name order, there
is no randomness, no network, no clock use, and only the standard library.

Fallbacks keep every case valid: an hour with too few comparable points falls
back to kernel-weighted empirical quantiles of the log-ratios, and an hour
with no points at all falls back to persistence (the hour-1 print with a
widening random-walk spread). If a case has no usable comparables at all,
the whole forecast is persistence. If fewer than 8 complete curves are
available the PCA model is skipped and model A is used alone.

Input:  every <id>.in in $MAKEFUN_INPUT_DIR (else /input).
Output: <id>.out in $MAKEFUN_OUTPUT_DIR (else /output):
        {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]} for h = 2..24.
"""

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]
HOURS = list(range(2, 25))

# Model A (local-linear): Gaussian kernel width (log units of hour-1
# viewers) and the factor applied to the residual spread.
KERNEL_SIGMA_A = 1.6
SPREAD_FACTOR = 0.85
# Fitted slopes are clamped so an extreme target scale cannot extrapolate
# the log-ratio trend without bound.
MAX_SLOPE = 1.5
# A comparable needs at least this many release-day hourly points to count.
MIN_DAY_POINTS = 15
# A kernel fit needs at least this many points for an hour; below that we use
# empirical quantiles of the log-ratios instead.
MIN_FIT_POINTS = 3
# Persistence fallback spread, in log units per square-root hour.
FALLBACK_SIGMA = 0.15

# Model B (PCA): number of components, kernel width for the score fits.
PCA_COMPONENTS = 2
KERNEL_SIGMA_B = 2.5
# Curves needed for a stable PCA fit; below this, model B is skipped.
MIN_PCA_CURVES = 8
# Power-iteration settings (deterministic).
PCA_ITERATIONS = 300
PCA_TOLERANCE = 1e-12

# Blend weight on model B (0.5 = equal blend).
BLEND_B = 0.5

_TIER_RE = re.compile(r"Tier:\s*([a-z][a-z\-]*)")
_DROPPED_TIERS = {"early-access-contaminated", "weak", "target-history"}


def round_half_up_2(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def day_hours(hourly, day_start):
    """Map release-day hour (1..24) -> viewers for one comparable launch."""
    hours = {}
    for row in hourly or []:
        viewers = row.get("viewers")
        if not viewers or viewers <= 0:
            continue
        try:
            at = parse_time(row["from"])
        except (KeyError, TypeError, ValueError):
            continue
        hour = int((at - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            hours[hour] = viewers
    return hours


def launch_tier(comp):
    """The tier tag in a comparable's unlock note, if it carries one."""
    match = _TIER_RE.search(comp.get("unlock") or "")
    return match.group(1) if match else "comparable"


def collect_profiles(case):
    """Usable comparable launches: (log hour-1 viewers, {hour: log-ratio})."""
    profiles = []
    for comp in sorted(case.get("comparables") or [],
                       key=lambda c: c.get("name") or ""):
        name = comp.get("name") or ""
        if "Grand Theft Auto VI" in name:
            continue
        if launch_tier(comp) in _DROPPED_TIERS:
            continue
        try:
            day_start = parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        hours = day_hours(comp.get("hourly"), day_start)
        if 1 not in hours or len(hours) < MIN_DAY_POINTS:
            continue
        first = hours[1]
        profiles.append((
            math.log(first),
            {h: math.log(v / first) for h, v in hours.items()},
        ))
    return profiles


def weighted_quantile(pairs, level):
    """Weighted empirical quantile of (weight, value) pairs."""
    ordered = sorted(pairs, key=lambda pair: pair[1])
    total = sum(w for w, _ in ordered)
    acc = 0.0
    for weight, value in ordered:
        acc += weight
        if acc >= level * total:
            return value
    return ordered[-1][1]


def local_linear_quantiles(profiles, log_p1, hour):
    """Model A: five log-ratio quantiles for one hour, or None if no data."""
    points = []
    for log_first, ratios in profiles:
        if hour in ratios:
            weight = math.exp(-((log_first - log_p1) / KERNEL_SIGMA_A) ** 2)
            points.append((weight, log_first - log_p1, ratios[hour]))
    if not points:
        return None
    if len(points) < MIN_FIT_POINTS:
        pairs = [(w, y) for w, _, y in points]
        return [weighted_quantile(pairs, level) for level in LEVELS]

    total_w = sum(w for w, _, _ in points)
    mean_x = sum(w * x for w, x, _ in points) / total_w
    mean_y = sum(w * y for w, _, y in points) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for w, x, _ in points)
    sxy = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in points)
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    slope = max(-MAX_SLOPE, min(MAX_SLOPE, slope))
    median = mean_y - slope * mean_x  # fitted value at x = 0 (target scale)
    variance = sum(w * (y - (median + slope * x)) ** 2
                   for w, x, y in points) / total_w
    spread = math.sqrt(max(variance, 0.0)) * SPREAD_FACTOR
    return [median + z * spread for z in Z_SCORES]


# ---------------------------------------------------------------------------
# Model B: PCA curve model (pure Python; no third-party packages).
# ---------------------------------------------------------------------------

def _matvec(matrix, vec):
    return [sum(row[j] * vec[j] for j in range(len(vec))) for row in matrix]


def _norm(vec):
    return math.sqrt(sum(v * v for v in vec))


def _power_iteration(matrix, start):
    """Dominant eigenvector of a symmetric matrix, deterministic."""
    vec = list(start)
    n = _norm(vec)
    if n == 0.0:
        return None, 0.0
    vec = [v / n for v in vec]
    for _ in range(PCA_ITERATIONS):
        nxt = _matvec(matrix, vec)
        n = _norm(nxt)
        if n == 0.0:
            return None, 0.0
        nxt = [v / n for v in nxt]
        delta = max(abs(a - b) for a, b in zip(nxt, vec))
        vec = nxt
        if delta < PCA_TOLERANCE:
            break
    mv = _matvec(matrix, vec)
    lam = sum(a * b for a, b in zip(vec, mv))
    return vec, lam


def _deflate(matrix, vec, lam):
    n = len(vec)
    return [[matrix[i][j] - lam * vec[i] * vec[j]
             for j in range(n)] for i in range(n)]


def _fix_sign(vec):
    """Force the largest-|entry| component positive (deterministic)."""
    k = max(range(len(vec)), key=lambda i: abs(vec[i]))
    if vec[k] < 0:
        return [-v for v in vec]
    return vec


def pca_model(profiles, log_p1):
    """Model B: {hour: [5 log-ratio quantiles]}, or None if it cannot fit.

    Uses only profiles with a complete curve (hours 2..24 all present).
    """
    complete = [(lf, r) for lf, r in profiles
                if all(h in r for h in HOURS)]
    if len(complete) < MIN_PCA_CURVES:
        return None
    n = len(complete)
    xs = [lf for lf, _ in complete]
    curves = [[r[h] for h in HOURS] for _, r in complete]
    # Per-hour mean curve.
    mean = [sum(curve[i] for curve in curves) / n for i in range(len(HOURS))]
    centered = [[curve[i] - mean[i] for i in range(len(HOURS))]
                for curve in curves]
    # Covariance matrix (23 x 23).
    m = len(HOURS)
    cov = [[sum(centered[k][i] * centered[k][j] for k in range(n)) / (n - 1)
            for j in range(m)] for i in range(m)]
    # Top PCA_COMPONENTS eigenvectors by power iteration with deflation.
    comps = []
    work = cov
    for idx in range(PCA_COMPONENTS):
        if idx == 0:
            start = [1.0] * m
        else:
            start = [1.0 if i % 2 == 0 else -1.0 for i in range(m)]
        vec, lam = _power_iteration(work, start)
        if vec is None or lam <= 0.0:
            break
        vec = _fix_sign(vec)
        comps.append(vec)
        work = _deflate(work, vec, lam)
    if len(comps) < PCA_COMPONENTS:
        return None
    # Scores of each complete curve on each component.
    scores = [[sum(centered[k][i] * comps[j][i] for i in range(m))
               for j in range(len(comps))] for k in range(n)]
    weights = [math.exp(-((x - log_p1) / KERNEL_SIGMA_B) ** 2) for x in xs]
    total_w = sum(weights)
    if total_w <= 0.0:
        return None
    # Local-linear fit of each score on (ln hour-1 - target), at the target.
    fitted_scores = []
    for j in range(len(comps)):
        mean_x = sum(w * (x - log_p1) for w, x in zip(weights, xs)) / total_w
        mean_y = sum(w * s[j] for w, s in zip(weights, scores)) / total_w
        sxx = sum(w * (x - log_p1 - mean_x) ** 2
                  for w, x in zip(weights, xs))
        sxy = sum(w * (x - log_p1 - mean_x) * (s[j] - mean_y)
                  for w, x, s in zip(weights, xs, scores))
        slope = sxy / sxx if sxx > 1e-12 else 0.0
        slope = max(-MAX_SLOPE, min(MAX_SLOPE, slope))
        fitted_scores.append(mean_y - slope * mean_x)
    # Reconstructed median curve at the target's scale.
    median = [mean[i] + sum(fitted_scores[j] * comps[j][i]
                            for j in range(len(comps)))
              for i in range(m)]
    # Per-hour spread from the weighted reconstruction residuals.
    recon = [[mean[i] + sum(scores[k][j] * comps[j][i]
                            for j in range(len(comps)))
              for i in range(m)] for k in range(n)]
    out = {}
    for i, hour in enumerate(HOURS):
        var = sum(w * (centered[k][i] - (recon[k][i] - mean[i])) ** 2
                  for k, w in enumerate(weights)) / total_w
        spread = math.sqrt(max(var, 0.0)) * SPREAD_FACTOR
        out[hour] = [median[i] + z * spread for z in Z_SCORES]
    return out


def persistence_quantiles(p1, hour):
    """Fallback: hour-1 print with a widening random-walk spread."""
    width = FALLBACK_SIGMA * math.sqrt(hour - 1)
    return [z * width for z in Z_SCORES]


def forecast_case(case):
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1),
                 None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)
    profiles = collect_profiles(case)
    model_b = pca_model(profiles, log_p1)

    rows = []
    for row in sorted(case["forecast"], key=lambda r: r["hour"]):
        hour = row["hour"]
        log_a = local_linear_quantiles(profiles, log_p1, hour)
        log_b = model_b.get(hour) if model_b else None
        if log_a is None and log_b is None:
            log_q = persistence_quantiles(p1, hour)
        elif log_b is None:
            log_q = log_a
        elif log_a is None:
            log_q = log_b
        else:
            log_q = [(1.0 - BLEND_B) * a + BLEND_B * b
                     for a, b in zip(log_a, log_b)]
        quantiles = []
        for value in log_q:
            q = p1 * math.exp(value)
            q = max(1.0, min(1_000_000_000.0, q))
            quantiles.append(round_half_up_2(q))
        quantiles.sort()
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
