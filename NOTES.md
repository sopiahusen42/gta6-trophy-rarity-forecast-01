# Notes — model and validation

## Model

The core signal is the **marginal propensity** `alpha_i` between distinct
Exophase states: the share of newly tracked players who have earned trophy i.
`E_i,target = E_i,now + alpha_i * (P_target - P_now)`. Three refinements matter:

- **Distinct-state collapse.** Identical consecutive reads carry no slope
  information; treating them as separate observations corrupts every estimate.
  We collapse runs keyed on `(players_tracked, trophies_earned, rates)`.
- **Alpha matures and saturates.** Measured on the real trajectory, alpha from
  launch→T1 is systematically lower than T1→T2 (ratios ~1.9–5.5 for the trophies
  that move). We extrapolate a population log-growth `beta` (median across
  trophies with two measurable intervals, clamped to 0.15–1.2/day) toward a
  per-trophy ceiling `5 * current_rate`, using a saturating exponential from now
  to the target. Trophies whose earner change is below the printed-rounding
  uncertainty are flagged "not identified" and blended toward a
  rate-proportional prior with a wider band. A trophy at 100% gets alpha = 1.
- **Players projection.** Marginal growth decays exponentially; we fit the decay
  from successive interval rates (recency weighted), convert the last interval's
  average to an instantaneous end rate, integrate to the target, and clamp
  (never shrink, never exceed zero-decay growth, absolute cap 500k).

Spread is an honest logit band: `sqrt(horizon)` time noise (0.20/√day, measured),
plus model uncertainty (1-state ≫ 2-state ≫ many-state; extra for noisy alpha),
plus player-count uncertainty scaled by `(1 - rate)` so the 100% trophy stays
pinned. Quantiles clamp to 0.01..99.99 (the scored logit domain) and are monotone.

Single-state launch fallback: launch rates are inflated by completionists, so a
refresh dilutes them hard. With no alpha information we use a dilution prior
(median ≈ 0.10 × last rate as P grows) and a wide band.

## What I validated (measured, not claimed)

Backtests on the real trajectory (L = launch 638 players; T1 = 42,228;
T2 = 44,161), scored with the contest pinball formula (logit, clamp 0.01..99.99,
five levels). Persistence = house baseline (last rate, 0.5/√day band):

| test | history → target | persistence loss | model loss | model cov90 | model miss (pp) |
|------|------------------|------------------|------------|-------------|-----------------|
| B1   | {L} → T1         | 2.9148           | **2.1655** | 0.291       | 1.121           |
| B2   | {L,T1} → T2      | 0.0633           | 0.0746     | 0.982       | 0.024           |

- **B1: the model beats persistence clearly** (2.17 vs 2.91, ~26% lower). This is
  the hard long-horizon dilution case.
- **B2: the model is slightly worse than persistence** (0.075 vs 0.063) but not
  catastrophic; its typical miss is better (0.024 vs 0.054 pp) and coverage is
  high. Over a 6.77 h interval little moves, so persistence is near-optimal and
  the model's honest extra spread costs a little. This satisfies "beat
  persistence on at least one, not catastrophic on the other."

Note: the task brief quoted persistence ≈1.27–1.32 on these tests; my measured
persistence losses are 2.91 (B1) and 0.063 (B2). I report what I actually
computed. The brief's figure may have used a different spread/horizon convention.

Practice cases p00/p01: outputs are **valid** (correct ids, five finite
non-decreasing values in 0..100, exact key shape, 2,661 bytes each < 65,536),
**byte-for-byte deterministic** across two runs (diff identical), and run in
**~0.06 s** (well under 8 s). The practice *loss* could not be computed locally:
the answering read (the 04:30/05:00 refresh) is not in the kit, and the server
grades it. The house baseline scores ~1.51 there per the rules.

Edge cases exercised without crashing and with valid output: single read; null
rates; rates array shorter than the trophy list; rates longer than the trophy
list; players_tracked unchanged between the last two distinct states (alpha
undefined → prior); empty history; and a 3-state long-horizon case (P_target
projects to ~47k, within the sanity clamp).

## Honest limitations

- B2 is marginally worse than persistence; the model is built for the final's
  multi-state, multi-day regime (f00), which neither practice nor these two
  backtests fully reproduce. The 2-state spread is deliberately conservative.
- The player-count projection is the largest correlated risk; its uncertainty is
  folded into every trophy's band, but a wrong P_target still shifts all rates.
- Alpha saturation ceiling (`5 × rate`) and maturation rate are calibrated to the
  two observed intervals; the true week-one asymptote is unknown.
- Single-state coverage on B1 is only 0.29 — launch-alone forecasting is genuinely
  hard; the wide band reduces loss versus persistence but cannot cover the full
  0.01%–100% outcome spread.
