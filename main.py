# GTA VI Trophy Rarity Forecast — week-one Exophase earn rates.
#
# SPDX-License-Identifier: MIT
#
# Idea in one paragraph: every trophy rate is a ratio, earned-count over
# players-tracked.  Both sides move.  Earned counts only grow as players
# progress through the story and side content, while the player count jumps
# upward each time Exophase refreshes, which dilutes every rate.  So instead
# of extrapolating the rate itself, this program extrapolates the two counts
# separately (each against log-time since release, which is how launch-week
# progress saturates), divides them back into a rate, and averages that with
# a direct log-time trend of the logit rate.  A weak prior built from the
# trophy's tier and description (story chapter, grind, collectible, money,
# one-off) steadies trophies whose history is flat or noisy; its weight grows
# with the horizon.  When the history shows no refresh at all (flat player
# count), the median is shifted down by a dilution allowance, because a
# refresh the program has not seen yet is expected before the target.
# Everything is computed on the logit scale so rare trophies count as much
# as common ones, and the spread widens with the square root of the horizon
# plus a refresh-uncertainty term, so the final's 4.5-day spread is far wider
# than the practice cases' 90-minute spread.
#
# Standard library only, no network, fully deterministic (no randomness,
# no dict-order or time dependence in the output).

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]
# Launch midnight UTC; progress is measured in log-hours from here.
ORIGIN = "2026-11-19T00:00:00Z"
LO = 0.01
HI = 99.99


def parse_time(text):
    """ISO-8601 UTC timestamp -> epoch seconds (float)."""
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    from datetime import datetime
    return datetime.fromisoformat(text).timestamp()


ORIGIN_TS = parse_time(ORIGIN)


def clamp_pct(x):
    if x < LO:
        return LO
    if x > HI:
        return HI
    return x


def logit_of(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def pct_of(logit):
    if logit >= 0:
        inv = 1.0 / (1.0 + math.exp(-logit))
    else:
        e = math.exp(logit)
        inv = e / (1.0 + e)
    return 100.0 * inv


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_sd) or None."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        sd = math.sqrt(var) if var > 0 else 0.0
    else:
        sd = 0.0
    return (a, b, sd)


def hours_since_origin(ts):
    return max(0.0, (ts - ORIGIN_TS) / 3600.0)


def log_time(ts):
    """Log-time coordinate: ln(1 + hours since launch midnight)."""
    return math.log1p(hours_since_origin(ts))


# ---------------------------------------------------------------- priors
# Week-one (about 7 days after release) median earn rate, in percent, by
# what the trophy asks for.  These are deliberately coarse: they only steady
# the forecast, they never replace the data.  Order matters — first match
# wins.  Each entry is (substring to look for in name+description, prior %).

PRIOR_RULES = [
    ("Gone Fishing", 99.5),
    ("100% Completion", 0.4),
    ("Congratulations! You're a Leonida Legend", 0.05),
    ("Complete Chapter 1", 48.0),
    ("Complete Chapter 2", 38.0),
    ("Complete Chapter 3", 30.0),
    ("Complete Chapter 4", 24.0),
    ("Complete the final mission.", 9.0),
    ("Complete the final mission for", 18.0),
    ("Complete 250 Stunts", 1.5),
    ("gold medal in all Shooting Range", 1.0),
    ("Drone Race at every location", 2.0),
    ("Win 20 different races", 4.0),
    ("3 Collectible Card sets", 2.0),
    ("WAiNK It at 100 locations", 1.5),
    ("Complete 50 Challenges", 3.0),
    ("Complete 8 mission strands", 6.0),
    ("Try each type of activity once", 22.0),
    ("Complete an arcade game", 18.0),
    ("Discover 100 vehicles", 12.0),
    ("Meditate at every spot", 3.0),
    ("Hunt a unique animal", 8.0),
    ("Explore all of Leonida", 10.0),
    ("Earn $250,000 laundering", 3.0),
    ("Spend $100,000", 6.0),
    ("Purchase all garage properties", 4.0),
    ("Register 10 vehicles", 8.0),
    ("Rob 5 Bank Vaults", 5.0),
    ("Rob an Armored Truck", 12.0),
    ("Rob 3 shops back-to-back", 14.0),
    ("Rob an Illegal Operation belonging to each gang", 6.0),
    ("Rob a jewelry store", 12.0),
    ("Take down 5 enemies in a single use of Combat Focus", 10.0),
    ("Lose a 4 star Wanted Level", 12.0),
    ("Beat Cal in a fishing challenge", 15.0),
    ("Complete Breakfast Run", 26.0),
    ("Complete The Cartel Heist", 26.0),
    ("Complete Child's Play", 26.0),
    ("Complete Headliners", 26.0),
    ("Complete VCIA Takedown", 26.0),
    ("Complete Operation Savage Glory", 26.0),
]

TIER_PRIOR = {"platinum": 0.05, "gold": 2.0, "silver": 3.0, "bronze": 8.0}


def prior_pct(trophy):
    text = (trophy.get("name", "") + " " + trophy.get("description", ""))
    for needle, value in PRIOR_RULES:
        if needle in text:
            return value
    return TIER_PRIOR.get(trophy.get("tier", "bronze"), 8.0)


# ---------------------------------------------------------------- forecast

def forecast_trophy(trophy, series, horizon_days, target_ts, checkpoint_ts):
    """series: list of (read_ts, rate_pct_or_None, players) oldest first."""
    obs = [(t, r, n) for (t, r, n) in series if r is not None]
    if not obs:
        # No read ever listed this trophy: fall back to the prior alone.
        med = logit_of(prior_pct(trophy))
        sigma = 1.6
        return [round4(clamp_pct(pct_of(med + z * sigma))) for z in Z]

    last_rate = obs[-1][1]
    last_logit = logit_of(last_rate)

    # Saturated trophy: (nearly) every tracked player has it.  Keep the
    # median very high but leave a tail downward in case the player base
    # broadens toward casual players by the target.
    if last_rate >= 99.99:
        med = logit_of(99.7)
        sigma = 0.55 + 0.15 * math.sqrt(horizon_days)
        return [round4(clamp_pct(pct_of(med + z * sigma))) for z in Z]

    # --- window: last 24 hours of reads before the checkpoint.
    window_start = checkpoint_ts - 24.0 * 3600.0
    win = [(t, r, n) for (t, r, n) in obs if t >= window_start]
    if len(win) < 3:
        win = obs

    xs = [log_time(t) for (t, r, n) in win]
    x_target = log_time(target_ts)
    x_last = log_time(obs[-1][0])

    # --- (a) direct trend: logit(rate) against log-time.
    trend_logit = None
    resid_sd = 0.0
    if len(set(round(v, 6) for v in [r for (_, r, _) in win])) >= 2:
        fit = ols(xs, [logit_of(r) for (_, r, _) in win])
        if fit is not None:
            a, b, sd = fit
            resid_sd = sd
            # A trend fitted on a flat stretch has slope ~0; that is fine.
            # Shrink the trend's move: log-time lines overshoot saturating
            # curves (story progress flattens), and tiny earned counts
            # make the fitted slope mostly noise.  Shrink harder when
            # the last earned count is small.
            moved = a + b * x_target
            moved = max(last_logit - 2.5, min(last_logit + 2.5, moved))
            earned_last = last_rate / 100.0 * (obs[-1][2] or 0)
            small = 0.3 + 0.7 * min(1.0, earned_last / 20.0)
            moved = last_logit + 0.6 * small * (moved - last_logit)
            trend_logit = moved

    # --- (b) structural counts: earned-count and player-count separately.
    struct_logit = None
    players_varied = False
    if win:
        ns = [n for (_, _, n) in win if n and n > 0]
        if len(ns) >= 2 and max(ns) > 1.05 * min(ns):
            players_varied = True
    if players_varied or len(win) >= 3:
        counts = []
        pcounts = []
        cxs = []
        for (t, r, n) in win:
            if n and n > 0 and r is not None:
                earned = max(0.5, r / 100.0 * n)
                counts.append(math.log(earned))
                pcounts.append(math.log(n))
                cxs.append(log_time(t))
        fit_c = ols(cxs, counts) if len(counts) >= 3 else None
        fit_n = ols(cxs, pcounts) if len(pcounts) >= 3 else None
        if fit_c is not None and fit_n is not None:
            ac, bc, _ = fit_c
            an, bn, _ = fit_n
            # Earned counts never fall in truth; a fitted fall is noise.
            bc = max(bc, 0.0)
            log_c = ac + bc * x_target
            log_n = an + bn * x_target
            # Cap the extrapolated player growth to something sane: the
            # tracked base cannot plausibly grow more than 40x past the
            # last read over one forecast, nor shrink.
            log_n = max(math.log(win[-1][2]), min(math.log(win[-1][2]) + math.log(40.0), log_n))
            rate_struct = 100.0 * math.exp(log_c - log_n)
            # Same shrinkage as the trend: counts saturate and small
            # counts are noisy; do not let the extrapolation run away.
            s_logit = logit_of(rate_struct)
            s_logit = max(last_logit - 2.5, min(last_logit + 2.5, s_logit))
            earned_last = last_rate / 100.0 * (obs[-1][2] or 0)
            small = 0.3 + 0.7 * min(1.0, earned_last / 20.0)
            struct_logit = last_logit + 0.6 * small * (s_logit - last_logit)

    # --- combine the two data-driven medians.
    parts = []
    if trend_logit is not None:
        parts.append(trend_logit)
    if struct_logit is not None:
        parts.append(struct_logit)
    if parts:
        model_logit = sum(parts) / len(parts)
    else:
        model_logit = last_logit
    if not players_varied:
        # No refresh has been seen yet: the player count sat flat all
        # through the history.  A refresh before the target is still
        # expected — Exophase refreshes in jumps hours apart — and each
        # one dilutes rates as fresh players sync with fewer trophies.
        # Allow for it explicitly; the fitted counts above cannot see it
        # because the history contains no refresh to learn from.
        horizon_hours = horizon_days * 24.0
        dilution = -2.2 * math.log1p(horizon_hours / 2.0)
        dilution = max(dilution, -1.6)
        model_logit += dilution

    # --- weak prior pull, weight growing with horizon (negligible at
    # 90 minutes, about a quarter at 4.5 days).
    w_prior = 0.30 * horizon_days / (horizon_days + 1.5)
    med = (1.0 - w_prior) * model_logit + w_prior * logit_of(prior_pct(trophy))
    # Never let the prior drag a trophy below a floor the data already
    # passed: if the last rate is above the prior, the median keeps at
    # least half the data's logit lead over the prior.
    med = max(med, last_logit - 1.5)

    # --- spread: random walk + unseen-refresh uncertainty + fit residual.
    # Grows with sqrt(horizon) as required, with a floor so a refresh
    # shock is never scored as impossible, and a cap: a well-identified
    # median must not pay for a needlessly wide band.
    rw = 0.35 * math.sqrt(horizon_days)
    refresh = 0.75 + 0.10 * math.sqrt(horizon_days)
    resid = 0.5 * min(resid_sd, 0.6)
    sigma = math.sqrt(rw * rw + refresh * refresh + resid * resid)
    sigma = max(0.5, min(1.4, sigma))

    qs = [round4(clamp_pct(pct_of(med + z * sigma))) for z in Z]
    # Rounding can only tie neighbours, never invert them; enforce anyway.
    for k in range(1, 5):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return qs


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target_ts = parse_time(case["target"])
    checkpoint_ts = parse_time(case["at"])

    series_by_trophy = [[] for _ in trophies]
    for read in history:
        ts = parse_time(read["at"])
        n = read.get("players_tracked") or 0
        rates = read.get("rates") or []
        for i in range(len(trophies)):
            rate = rates[i] if i < len(rates) else None
            series_by_trophy[i].append((ts, rate, n))

    rows = []
    for i, trophy in enumerate(trophies):
        qs = forecast_trophy(trophy, series_by_trophy[i], horizon_days,
                             target_ts, checkpoint_ts)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            raise ValueError("forecast too large for " + path.name)
        out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
