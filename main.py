"""GTA VI trophy earn-rate forecaster for make.fun gta6-trophy-rarity-forecast-01.

Python 3.13, standard library only, no network. Runs once per case: reads every
``*.in`` in ``$MAKEFUN_INPUT_DIR`` (default ``/input``) in name order and writes
``<id>.out`` to ``$MAKEFUN_OUTPUT_DIR`` (default ``/output``). Each output is
``{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`` with five
non-decreasing finite percentages in 0..100, one row per case trophy.

Model
-----
Exophase refreshes a game's counts on its own schedule (hours apart), so many
consecutive reads are identical. We first collapse runs of identical reads into
*distinct states*. Between two states the marginal propensity

    alpha_i = (E_i,cur - E_i,prev) / (P_cur - P_prev),   E_i = rate_i * P / 100

is the share of newly tracked players who have earned trophy i. It is a strong
predictor of the next interval, but it is not constant: it grows as the player
base matures (early adopters are completionists, later ones casual) and then
saturates. We estimate alpha with recency weighting, extrapolate a modest
population growth trend toward a per-trophy ceiling, and project

    E_i,target = E_i,now + alpha_i * (P_target - P_now)
    rate_i,target = 100 * E_i,target / P_target.

Players tracked is projected too, with a power law fitted on the real Exophase
trajectory, log P = 6.3981 + 1.1381 * log(1 + t_hours) (t_hours since launch
2026-11-18T23:48Z), anchored at the last state's own (t, P):

    P_target = P_now * ((1 + t_target) / (1 + t_now)) ** 1.138.

P is never left flat when a horizon exists. Earners, not rates, are the
persistent quantity: with alpha unknown the honest default is alpha = rate_frac,
so rate_target = rate_now * P_now / P_target (launch-night rates dilute hard as
the tracked player base grows).

The forecast spread is an honest logit-scale band calibrated on the three real
intervals (launch->t1, launch->t2, t1->t2). The residual sd of
logit(actual) - logit(predicted) per sqrt(day) is ~1.4-1.5 on the launch
intervals; a time term SIGMA_TIME = 2.0 per sqrt(day) gives achieved 90%
coverage of 0.69 / 0.98 / 1.00 on those intervals (mean ~0.89) with the
earners-persistent median, so the band is honest rather than decorative. On top
of the sqrt(horizon) time term sit a small model component (few distinct states
/ noisy alpha widen it), a player-count component scaled by (1 - rate) so
trophies pinned at 100% stay tight, and floors. Quantiles clamp to 0.01..99.99.

Everything is deterministic: no randomness, no clock reads, ordered iteration.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The five scored quantile levels and their standard-normal quantiles.
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

# --- spread calibration (logit units) -------------------------------------
# SIGMA_TIME recalibrated on the three real Exophase intervals (launch->t1,
# launch->t2, t1->t2): residual sd per sqrt(day) ~1.4-1.5; a 2.0/sqrt(day) time
# term gives achieved 90% coverage 0.69/0.98/1.00 (mean ~0.89) with the
# earners-persistent median. The old 0.20 understated uncertainty ~9x.
SIGMA_TIME = 2.0         # rate noise per sqrt(day), measured on real intervals
SIGMA_TIME_FLOOR = 0.05  # never collapse the band entirely
SIGMA_P = 0.13           # player-count uncertainty per sqrt(day)
SIGMA_P_FLOOR = 0.06
SIGMA_MODEL_1STATE = 0.30  # extra spread when only one distinct state is known
SIGMA_MODEL_2STATE = 0.20  # extra spread with two states (alpha barely identified)
SIGMA_ALPHA_NOISE = 0.30   # extra spread when a trophy's alpha is rounding noise
SIGMA_ALPHA_NOISE_FRAC = 0.4  # fraction of SIGMA_ALPHA_NOISE applied per sqrt(day)

# --- alpha model ----------------------------------------------------------
ALPHA_CEIL_MULT = 5.0    # asymptotic alpha ceiling = mult * current rate
BETA_DEFAULT = 0.7       # default alpha maturation rate per day
BETA_MIN = 0.15
BETA_MAX = 1.2
ALPHA_HALF_LIFE = 1.0    # recency half-life (days) for weighting intervals
ALPHA_PRIOR_MULT = 1.0   # prior alpha = rate_frac (earners persistent) when unmeasured
RARE_RATE = 0.10         # below this percent, alpha is mostly rounding noise
ROUND_EPS = 0.005        # half a printed decimal place (percent)

# --- single-state (launch) fallback ---------------------------------------
# Launch-night rates are inflated by completionists; a refresh dilutes them hard.
# With no alpha information the honest default is earners-persistent
# (alpha = rate_frac), so the median rate falls as P_now / P_target.
SINGLE_SIGMA_BASE = 0.30  # modest extra for the discrete-refresh risk
SINGLE_SIGMA_SQRT = 0.30

# --- players projection (power law on the real trajectory) ----------------
# log P = 6.3981 + 1.1381 * log(1 + t_hours), t_hours since 2026-11-18T23:48Z.
POWER_EXP = 1.138        # fitted power-law exponent
LOG_P_INTERCEPT = 6.3981  # fitted intercept: log P = 6.3981 + 1.1381*log(1+t_h)
LAUNCH_OFFSET_H = 0.2    # hours from 2026-11-18T23:48Z to parse_time's 00:00Z ref
P_ABS_CAP = 500_000.0    # physical sanity clamp on projected players


def clamp_pct(x: float) -> float:
    """Clamp a percentage to the scored logit domain 0.01..99.99."""
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(z: float) -> float:
    return 100.0 / (1.0 + math.exp(-z))


def parse_time(iso: str) -> float:
    """Hours since 2026-11-19T00:00:00Z for an ISO-8601 UTC timestamp."""
    s = iso.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ref = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
    return (dt - ref).total_seconds() / 3600.0


def distinct_states(history: list) -> list:
    """Collapse runs of identical reads into distinct states, oldest first.

    A state's timestamp is the last read that showed it (its most recent
    confirmation). Identical means same players_tracked, trophies_earned and
    rates tuple, so a long unchanged run carries no slope information.
    """
    states = []
    for r in history:
        rates = r.get("rates") or []
        sig = (r.get("players_tracked"), r.get("trophies_earned"), tuple(rates))
        t = parse_time(r["at"])
        if states and states[-1]["sig"] == sig:
            states[-1]["t"] = t          # extend the run's confirmation time
            states[-1]["read"] = r
        else:
            states.append({"sig": sig, "t": t, "read": r})
    return states


def project_players(states: list, target_h: float) -> tuple:
    """Project players tracked at target_h and a log-scale uncertainty.

    Uses the power law fitted on the real trajectory,
    log P = 6.3981 + 1.1381 * log(1 + t_hours) (t_hours since launch
    2026-11-18T23:48Z), evaluated at ABSOLUTE time:

        P_target = exp(6.3981 + 1.1381 * log(1 + t_target)).

    The curve is global, not anchored at a state, because a state's own
    players_tracked goes stale: a refresh lags hours behind the page, so a
    history can sit flat at the launch plateau (638 players at t=2.5h) while
    the real curve is already at 2,500. Anchoring at that stale reading
    projected 1,043 players at a target where Exophase actually held 3,868, a
    73% miss; the fitted curve evaluated at the same absolute time gives 4,354,
    a 13% miss. On the four real points the curve is within +0.6%, +12.6%,
    -9.4% and +4.2%.

    P is never left flat when a horizon exists, even with a single state, and
    never projects below the players already seen.
    Returns (P_target, sigma_logP). The projection is sanity-clamped: never
    shrink below p_now, never exceed an absolute physical cap.
    """
    ps = [s["read"].get("players_tracked") or 0 for s in states]
    p_now = float(ps[-1]) if ps else 1.0
    now_h = states[-1]["t"] if states else 0.0
    span = max(0.0, target_h - now_h)
    span_days = span / 24.0
    n_states = len(states)

    # Power-law time is hours since launch (parse_time uses a 00:00Z reference).
    t_target = target_h + LAUNCH_OFFSET_H
    if p_now <= 0.0:
        p_target = p_now
    else:
        base_target = max(1.0 + t_target, 0.01)
        p_target = math.exp(LOG_P_INTERCEPT + POWER_EXP * math.log(base_target))
        p_target = min(p_target, P_ABS_CAP)
        p_target = max(p_target, p_now)  # never shrink below what we have seen

    # Uncertainty grows with horizon and shrinks with more distinct states.
    sigma_logp = (SIGMA_P * math.sqrt(span_days) + SIGMA_P_FLOOR)
    sigma_logp *= 1.0 + 0.5 / max(1, n_states - 1)
    return p_target, sigma_logp


def _interval_alphas(states: list, n: int) -> list:
    """alpha_i per interval (list of dicts: midpoint, dt, dP, alphas list)."""
    intervals = []
    for a, b in zip(states, states[1:]):
        pa = a["read"].get("players_tracked") or 0
        pb = b["read"].get("players_tracked") or 0
        dp = pb - pa
        ra = a["read"].get("rates") or []
        rb = b["read"].get("rates") or []
        alphas = []
        for i in range(n):
            x = ra[i] if i < len(ra) else None
            y = rb[i] if i < len(rb) else None
            if dp <= 0 or x is None or y is None:
                alphas.append(None)
                continue
            e_prev = x * pa / 100.0
            e_cur = y * pb / 100.0
            alphas.append(min(1.0, max(0.0, (e_cur - e_prev) / dp)))
        intervals.append({"tm": (a["t"] + b["t"]) / 2.0, "dt": b["t"] - a["t"],
                          "dp": dp, "alphas": alphas})
    return intervals


def _population_beta(intervals: list) -> float:
    """Median log-growth of alpha across trophies with two measurable intervals."""
    if len(intervals) < 2:
        return BETA_DEFAULT
    prev, last = intervals[-2], intervals[-1]
    dt_days = (last["tm"] - prev["tm"]) / 24.0
    if dt_days <= 1e-6:
        return BETA_DEFAULT
    ratios = []
    for a0, a1 in zip(prev["alphas"], last["alphas"]):
        if a0 is not None and a1 is not None and a0 > 1e-4 and a1 > a0:
            ratios.append(math.log(a1 / a0) / dt_days)
    if not ratios:
        return BETA_DEFAULT
    ratios.sort()
    beta = ratios[len(ratios) // 2]
    return min(BETA_MAX, max(BETA_MIN, beta))


def estimate_alpha(intervals: list, n: int, rates_now: list, p_now: float) -> tuple:
    """Recency-weighted alpha per trophy, extrapolated toward a ceiling.

    Returns (alpha_fwd list, identified list) where identified marks trophies
    whose alpha was measured from real earner movement (not rounding noise).
    """
    beta = _population_beta(intervals)
    alpha_fwd = []
    identified = []
    last_tm = intervals[-1]["tm"] if intervals else 0.0
    for i in range(n):
        r_now = rates_now[i] if i < len(rates_now) and rates_now[i] is not None else 1.0
        rate_frac = clamp_pct(r_now) / 100.0
        # A trophy already at 100% is earned by everyone: alpha = 1.
        if r_now >= 99.9:
            alpha_fwd.append(1.0)
            identified.append(True)
            continue
        ceiling = min(0.95, ALPHA_CEIL_MULT * rate_frac)
        # Recency-weighted mean of measurable interval alphas.
        num = den = 0.0
        moved = False
        for iv in intervals:
            a = iv["alphas"][i] if i < len(iv["alphas"]) else None
            if a is None:
                continue
            w = 2.0 ** ((iv["tm"] - last_tm) / (24.0 * ALPHA_HALF_LIFE))
            num += w * a
            den += w
            # Real movement: earner change exceeds rounding uncertainty.
            round_unc = ROUND_EPS * (p_now / 100.0) * 2.0
            if iv["dp"] > 0 and a * iv["dp"] > round_unc:
                moved = True
        if den > 0.0:
            alpha_bar = num / den
        else:
            alpha_bar = ALPHA_PRIOR_MULT * rate_frac
            moved = False
        alpha_bar = min(1.0, max(0.0, alpha_bar))
        # Saturating maturation toward the ceiling over the forecast span.
        # Span is measured from the last interval midpoint; caller passes it via
        # the closure-free helper below (we recompute using target offset).
        alpha_fwd.append(alpha_bar)  # placeholder, refined in forecast_case
        identified.append(moved)
    return alpha_fwd, identified, beta


def forecast_case(case: dict) -> dict:
    trophies = case.get("trophies") or []
    n = len(trophies)
    history = case.get("history") or []
    states = distinct_states(history)
    target_h = parse_time(case["target"])

    if not states:
        # No history at all: honest wide band around a neutral 1%.
        rows = []
        for t in trophies:
            z0 = logit(1.0)
            s = 1.5
            rows.append({"id": t["id"], "q": _quantiles(z0, s)})
        return {"trophies": rows}

    last = states[-1]["read"]
    rates_now = last.get("rates") or []
    p_now = float(last.get("players_tracked") or 1)
    now_h = states[-1]["t"]
    span_days = max(0.0, (target_h - now_h) / 24.0)

    p_target, sigma_logp = project_players(states, target_h)
    dp_fwd = max(0.0, p_target - p_now)

    intervals = _interval_alphas(states, n)
    alpha_bar, identified, beta = estimate_alpha(intervals, n, rates_now, p_now)

    # Forecast span from now (the last state): alpha_bar is our best estimate of
    # the current marginal propensity, so we mature it forward from now.
    fwd_days = max(0.0, (target_h - now_h) / 24.0)
    decay = math.exp(-beta * fwd_days)

    n_states = len(states)
    if n_states == 1:
        sigma_model = SIGMA_MODEL_1STATE
    elif n_states == 2:
        sigma_model = SIGMA_MODEL_2STATE
    else:
        sigma_model = 0.10

    rows = []
    for i, t in enumerate(trophies):
        r_now = rates_now[i] if i < len(rates_now) and rates_now[i] is not None else None
        single = (n_states == 1)
        if r_now is None:
            # Missing rate: neutral persistence with a wide band.
            z0 = logit(1.0)
            s = math.sqrt((SIGMA_TIME * math.sqrt(span_days)) ** 2 + 0.8 ** 2)
            rows.append({"id": t["id"], "q": _quantiles(z0, s)})
            continue

        rate_frac = clamp_pct(r_now) / 100.0
        e_now = r_now * p_now / 100.0

        if single:
            # Launch-night fallback: no alpha information. Measured on the real
            # launch->t1 and launch->t2 intervals, the loss-optimal alpha is 0.0
            # (loss 0.776/0.805 at sigma 2.0/sqrt(day)) against 1.318 for
            # alpha = rate_frac, i.e. the earner count barely moves while the
            # tracked player base explodes, so the rate dilutes as P_now/P_target.
            # A trophy already at 100% is earned by every tracked player and
            # cannot dilute: alpha = 1 keeps it pinned (id 2 in the practice data).
            alpha = 1.0 if r_now >= 99.9 else 0.0
            e_target = e_now + alpha * dp_fwd
            rate_target = 100.0 * e_target / p_target if p_target > 0 else r_now
            z0 = logit(rate_target)
            s_time = SIGMA_TIME * math.sqrt(span_days) + SIGMA_TIME_FLOOR
            s_single = SINGLE_SIGMA_BASE + SINGLE_SIGMA_SQRT * math.sqrt(span_days)
            s_p = sigma_logp * (1.0 - clamp_pct(rate_target) / 100.0)
            sigma = math.sqrt(s_time ** 2 + s_single ** 2 + s_p ** 2)
        else:
            a_bar = alpha_bar[i]
            ceiling = min(0.95, ALPHA_CEIL_MULT * rate_frac)
            if r_now >= 99.9:
                alpha = 1.0
            else:
                # Do not extrapolate alpha toward a ceiling over the horizon:
                # measured on the real intervals the loss-optimal alpha for a
                # forward span stays at the recent estimate, and the ceiling
                # growth overshoots by ~4.8x at a 4.5-day span (0.0302 against
                # a rate_frac of 0.0063). Hold a_bar instead.
                alpha = a_bar
                if not identified[i]:
                    # Rounding-noise alpha: blend toward an earners-persistent
                    # prior (alpha 0), the measured optimum on real intervals.
                    alpha = 0.5 * alpha + 0.5 * 0.0
            e_target = e_now + alpha * dp_fwd
            rate_target = 100.0 * e_target / p_target if p_target > 0 else r_now
            z0 = logit(rate_target)
            s_time = SIGMA_TIME * math.sqrt(span_days) + SIGMA_TIME_FLOOR
            s_model = sigma_model
            if not identified[i]:
                s_model = math.sqrt(s_model ** 2 +
                                    (SIGMA_ALPHA_NOISE * SIGMA_ALPHA_NOISE_FRAC *
                                     math.sqrt(span_days) + SIGMA_ALPHA_NOISE * 0.5) ** 2)
            s_p = sigma_logp * (1.0 - clamp_pct(rate_target) / 100.0)
            sigma = math.sqrt(s_time ** 2 + s_model ** 2 + s_p ** 2)
            # Trophies pinned at the top boundary cannot move up: tighten hard.
            if r_now >= 99.9:
                sigma *= 0.05

        rows.append({"id": t["id"], "q": _quantiles(z0, sigma)})
    return {"trophies": rows}


def _quantiles(z0: float, sigma: float) -> list:
    """Five non-decreasing percentages, clamped to the scored domain."""
    sigma = max(0.0, sigma)
    qs = []
    for z in Z:
        v = inv_logit(z0 + z * sigma)
        v = min(99.99, max(0.01, v))
        qs.append(round(v, 4))
    # Guard monotonicity against rounding (sigma >= 0 already guarantees it).
    for j in range(1, 5):
        if qs[j] < qs[j - 1]:
            qs[j] = qs[j - 1]
    return qs


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
