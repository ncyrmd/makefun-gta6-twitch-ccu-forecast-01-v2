# GTA VI Twitch CCU Forecaster — v2

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ncyrmd/makefun-gta6-twitch-ccu-forecast-01-v2/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ncyrmd/makefun-gta6-twitch-ccu-forecast-01-v2/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](main.py)

Forecasts hour-by-hour Twitch concurrent viewers in the **Grand Theft Auto VI**
category on launch day, 19 November 2026 (UTC), hours 2–24, as five quantiles
(5 / 25 / 50 / 75 / 95%) — an entry for the
[make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
($250, top 20). This is v2, a remake of v1 (dev loss 0.1595, rank 48/950).

## Method

`main.py` (Python 3.13, standard library only) reads one case file per run and
forecasts from the case's own data alone — no network, no hardcoded answers.

**A 50/50 blend of two models of the log-ratio curve** `r_h =
ln(viewers_h / viewers_1)`:

- **Model A — local-linear log-ratio regression** (v1, retuned): for each hour
  separately, `r_h` is regressed on `ln(viewers_1)` across the comparable
  launches, weighting each launch by a Gaussian kernel centred on the target's
  own log hour-1 print (σ = 1.6 log units). The fitted value at the target's
  scale is the median log-ratio; the weighted residual spread gives the
  log-scale standard deviation, and the five quantiles are normal quantiles
  around the median, narrowed by a factor of 0.85.
- **Model B — low-rank (PCA) curve model:** the comparables' log-ratio curves
  are nearly one-dimensional — the first principal component carries ~86% of
  the variance, the first two ~95%. Each curve is reconstructed from its two
  component scores; the scores are regressed on `ln(viewers_1)` with the same
  kernel-weighted local-linear fit (σ = 2.5), borrowing strength across all 23
  hours instead of fitting each hour in isolation. The median curve is the
  mean curve plus the fitted scores times the components; the per-hour spread
  comes from the weighted reconstruction residuals, also narrowed by 0.85.
  The eigendecomposition is a pure-Python power iteration with deflation
  (verified against NumPy to ~1e-13 on the practice pool).

The blend is a hedge: model A adapts hour by hour and tracks the biggest
launches at the target's scale, model B imposes the smooth common launch
shape. Two filters keep the comparison honest:

- Launches flagged `early-access-contaminated` or `weak` in their notes are
  dropped — their pre-day curves were already running before release day.
- The GTA VI category's own pre-release history is not a launch and is dropped.

Fallbacks keep every case valid: an hour with too few comparable points uses
kernel-weighted empirical quantiles of the log-ratios, and an hour with no
points at all falls back to persistence (hour-1 print, widening random-walk
spread). If fewer than 8 complete curves are available the PCA model is
skipped and model A is used alone. Everything is deterministic — no
randomness, no clock, comparables processed in name order.

## Local dev score (practice cases p00–p03)

Ground truth for the practice answers was taken from the same public source
the kit documents (SullyGnome hourly category averages, via the chart API
named in the kit's `input/NOTICE.txt`), scored with the exact pinball loss
from the rules (mean over every scored hour and quantile, log scale),
independently re-derived — not the coordinator's numbers alone:

| case | launch | v2 loss | v1 loss |
| ---- | ------ | ------: | ------: |
| p00 | Marvel's Wolverine | 0.143 | 0.193 |
| p01 | Red Dead Redemption 2 | 0.130 | 0.140 |
| p02 | God of War Ragnarök | 0.093 | 0.108 |
| p03 | Marvel's Spider-Man 2 | 0.182 | 0.198 |
| **mean** | | **0.137** | **0.160** |

All four practice cases improve. The blend was also checked by
leave-one-out over the big rolling-midnight comparables (0.231 vs v1's 0.233)
and on a final-like pool with the practice launches added as comparables
(0.135 vs v1's 0.151) — better on both, not just one. Runtime is ~0.08 s per
case; two runs produce byte-identical output.

Practice scores are no guide to the final — the final case is a single new
launch day — but the method beats v1 on every practice case and on both
cross-checks.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Files

- `main.py` — the entrypoint (stdlib only)
- `LICENSE` — MIT
- `.makefun/owner.json` — payout wallet

## License

MIT — see [LICENSE](LICENSE).
