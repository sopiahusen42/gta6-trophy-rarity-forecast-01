# GTA VI Trophy Rarity Forecast — week-one earn rates

[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sopiahusen42/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sopiahusen42/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)

A forecasting entry for the make.fun bounty **gta6-trophy-rarity-forecast-01**:
predict, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate on
Exophase at `2026-11-26T00:00:00Z` (week one), as five honest quantiles
(5%, 25%, 50%, 75%, 95%). Scored by mean pinball loss on the logit scale.

## How it works

Exophase refreshes a game's counts on its own schedule, hours apart, so many
consecutive reads are identical. The program (`main.py`, Python 3.13 standard
library only, no network) first collapses runs of identical reads into
**distinct states**, then:

1. **Marginal propensity (alpha).** Between two states,
   `alpha_i = (E_i,cur - E_i,prev) / (P_cur - P_prev)` with `E_i = rate_i * P / 100`
   is the share of newly tracked players who earned trophy i. It is estimated
   with recency weighting, extrapolated with a modest population maturation
   trend toward a per-trophy ceiling (alpha grows as the player base matures,
   then saturates), and shrunk toward a rate-proportional prior when a trophy's
   earners are too few to measure (rounding noise).
2. **Players tracked projection.** The marginal growth rate is fitted with an
   exponential decay across the distinct states, converted to an instantaneous
   end rate, and integrated to the target — growth decelerates hard after launch.
   The projection is sanity-clamped.
3. **Forecast.** `E_i,target = E_i,now + alpha_i * (P_target - P_now)`,
   `rate_i,target = 100 * E_i,target / P_target`.
4. **Honest spread.** A logit-scale band combining a `sqrt(horizon)` time
   component, a model component for how well alpha and the player count are
   identified (few states / noisy alpha / long extrapolation widen it), and a
   player-count component scaled by `(1 - rate)` so trophies pinned at 100%
   stay tight. Quantiles are clamped to the scored domain and kept non-decreasing.

A single-state (launch-night) history falls back to a dilution prior with a wide
band, since launch rates are inflated by completionists and a refresh dilutes
them. See `NOTES.md` for the model rationale and measured backtest results.

## Run it

Python 3.13, standard library only. The program reads every `*.in` in
`$MAKEFUN_INPUT_DIR` (default `/input`) in name order and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (default `/output`).

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=gta6-trophy-rarity-forecast-01-kit/input \
MAKEFUN_OUTPUT_DIR=out \
python3 main.py
```

Each output is one JSON object:
`{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}` — one row per case
trophy, five non-decreasing finite percentages in 0..100, under 65,536 bytes.
Runs are byte-for-byte deterministic (no randomness, no clock reads).

## License

MIT — see `LICENSE`.
