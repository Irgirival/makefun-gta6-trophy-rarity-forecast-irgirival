"""GTA VI trophy rarity forecaster — gta6-trophy-rarity-forecast-01.

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate
on Exophase at the case's target, as five quantiles (5/25/50/75/95%), scored by
mean pinball loss on the logit scale.

Method — a structural count model, not a curve fit on rates alone
-----------------------------------------------------------------
An Exophase earn rate is a ratio:  r_i(t) = 100 * E_i(t) / P(t),  where P(t) is
the number of players Exophase tracks and E_i(t) the number of them who hold
trophy i.  Two forces move every rate, in opposite directions:

* dilution — new players join the tracked set holding almost nothing, so each
  Exophase refresh (every few hours) drags rates down.  Launch night proved
  it, and proved how extreme it is: the first refresh took players tracked
  638 -> 3,868 while every rate but the one at 100% fell.  The earned total
  rose 943 -> 4,276, but the 100% trophy (every tracked player holds it)
  absorbed 3,230 of those 3,333 new earns, leaving only 103 new earns
  across the other 54 trophies — about 0.03 trophies per new player.  A new
  cohort is, to first order, fresh players who have only finished the
  first mission;
* progression — the players already tracked keep playing, so story and
  easy-activity trophies climb as the week goes on.

The forecaster therefore works in counts, per trophy, and blends three
views of the median, all on the logit scale:

1. trend — a weighted least-squares line of logit(rate) on ln(1 + hours
   since release), fitted over the case history's distinct refresh states
   (weights sqrt(players tracked)), slope shrunk toward the trophy's
   difficulty-group median slope and capped, then read at the target;
2. structural — earned counts carried forward:  E_target = E_last +
   (P_target - P_last) * q_i,  where P_target comes from a log-time fit of
   players tracked and q_i is the new-cohort earn rate for the trophy,
   measured from the history's own refresh events.  Where no refresh exists
   in history, q_i falls back to a budget model: a new cohort earns
   ~1.03 trophies per player (launch night), of which the first-mission
   trophy takes ~1.00, so only ~0.03 per player is left for the other
   trophies, allocated by the launch cohort's earned shares with a
   progression skew (story x3, easy x1.5, grind x0.3, money x0.5,
   platinum/completion x0.2) — new players are at the start of the game;
3. prior — a difficulty-group week-one anchor, calibrated on GTA V's
   lifetime trophy rarity (PSNProfiles: first story trophy ~48% of owners,
   final mission ~19%, platinum ~5% of owners, 100% completion ~5-6%) scaled
   down to launch-week progression, so long-horizon medians cannot run away
   on a noisy early trend.  The prior's weight grows with the horizon and
   shrinks when the trend fit is confident.

Spread — the logit spread combines the trend fit's residual, a fixed
refresh-timing term (Exophase refreshes on its own schedule, so the answer
is the last refresh before the target), a refresh-event term for the
bimodality of a short window (a refresh in the window drops most rates
~1.5 logit; no refresh leaves them flat — the term peaks when a refresh
is a coin flip and vanishes when one is certain, as over a 4.5-day
horizon), and the house baseline's random walk, 0.5 * sqrt(horizon in
days).  Quantiles are the median logit plus the standard normal's
quantile at each level times the spread.

The program reads every <id>.in in the input directory and writes
<id>.out beside it in the output directory: {"trophies": [{"id": n,
"q": [q05, q25, q50, q75, q95]}, ...]}, one row per trophy, quantiles
non-decreasing, in [0.01, 99.99].  Pure standard library, deterministic
(no randomness, no network, no wall-clock dependence beyond the case
file), well inside the 8-second cap.

Calibration sources are documented in METHODOLOGY.md.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release day start, 2026-11-19T00:00:00Z: the clock every trend is fitted on.
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Exophase refreshes a game's counts every few hours, not continuously.
REFRESH_MEAN_HOURS = 6.0
# A single early refresh multiplied players tracked ~6x (638 -> 3,868).
REFRESH_EARLY_MULTIPLIER = 6.0

# Random-walk spread, logit units per square-root day (the house baseline's).
SIGMA_RW = 0.5
# Refresh-timing spread: the answer is the last refresh before the target.
SIGMA_REFRESH = 0.35


def clamp(x: float) -> float:
    """Clamp a percent into the range the logit is defined on."""
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100.0 - clamp(pct)))


def inv_logit(y: float) -> float:
    if y >= 20.0:
        return 99.99
    if y <= -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-y))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> float:
    """ISO 8601 UTC timestamp -> epoch seconds."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def hours_since_release(when: str) -> float:
    return (parse_time(when) - RELEASE.timestamp()) / 3600.0


def fit_line(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, resid_std, n).

    Falls back to a flat line at the weighted mean when the x values do not
    vary or there are fewer than two points.
    """
    n = len(xs)
    sw = sum(ws)
    if n == 0 or sw <= 0:
        return 0.0, 0.0, 0.5, 0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12 or n < 2:
        resid = math.sqrt(sum(w * (y - my) ** 2 for y, w in zip(ys, ws)) / sw) if n > 1 else 0.5
        return my, 0.0, resid, n
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw)
    return a, b, resid, n


# ---------------------------------------------------------------------------
# Trophy difficulty groups, from the trophy's tier and description.
# Each group carries a week-one prior median (percent) calibrated on GTA V's
# lifetime rarity (see METHODOLOGY.md), and a progression skew used when
# allocating a new cohort's earned budget: new players are at the start of
# the game, so their few non-first-mission earns skew toward story trophies
# and away from grinds.
# ---------------------------------------------------------------------------

# Progression skew by group, for allocating a new cohort's earned budget.
SKEW = {
    "platinum": 0.2, "completion": 0.2,
    "story-ch1": 3.0, "story-ch2": 3.0, "story-ch3": 3.0, "story-ch4": 3.0,
    "story-final": 3.0, "story-mission": 3.0,
    "grind": 0.3, "money": 0.5, "easy": 1.5,
}

# Launch-night new-cohort budget (trophies per new player) and the part of
# it left after the first-mission trophy takes its ~1.00: 3,333 new earns
# over 3,230 new players, 3,230 of them the 100% trophy, so ~0.03.
BUDGET_TOTAL = 1.03
BUDGET_NONFIRST = 0.03


def classify(trophy: dict):
    """Return (group, prior_median_pct) for one trophy."""
    desc = trophy["description"].lower()
    tier = trophy["tier"]
    if tier == "platinum":
        return ("platinum", 0.9)
    if "100% completion" in desc:
        return ("completion", 1.0)
    if "chapter 1" in desc:
        return ("story-ch1", 32.0)
    if "chapter 2" in desc:
        return ("story-ch2", 24.0)
    if "chapter 3" in desc:
        return ("story-ch3", 18.0)
    if "chapter 4" in desc:
        return ("story-ch4", 14.0)
    if "final mission" in desc:
        return ("story-final", 10.0)
    # Grind and money keywords are checked BEFORE the generic story-mission
    # rule: several grinds are phrased "Complete 250 Stunts." /
    # "Complete 50 Challenges." / "Complete 3 Collectible Card sets." and
    # would otherwise be misread as story missions with a 12% prior.
    grind_keys = (
        "250 stunts", "20 different races", "50 challenges",
        "collectible card sets", "100 locations", "every spot",
        "unique animal", "all of leonida", "5 bank vaults", "each gang",
        "gold medal in all", "every location", "mission strands",
        "100 vehicles",
    )
    if any(k in desc for k in grind_keys):
        return ("grind", 3.0)
    money_keys = ("earn $", "spend $", "purchase all garage")
    if any(k in desc for k in money_keys):
        return ("money", 5.0)
    if desc.startswith("complete "):
        return ("story-mission", 12.0)
    return ("easy", 15.0)


def budget_defaults(states, trophies, groups):
    """Fallback new-cohort earn rates q_i (absolute fractions, 0-1).

    A new cohort earns ~BUDGET_NONFIRST trophies per player outside the
    first-mission trophy (which takes ~1.00 each).  That budget is split
    by the last state's earned shares, skewed by group: a new cohort is at
    the start of the game, so its earns skew toward story trophies.  The
    first-mission trophy itself (rate >= 99.5, if any) gets q = 1.0.
    """
    n = len(trophies)
    _, last_p, last_rates = states[-1]
    first = None
    for i, r in enumerate(last_rates):
        if r is not None and r >= 99.5:
            first = i
            break
    earned = [0.0] * n
    for i, r in enumerate(last_rates):
        if r is not None:
            earned[i] = r * last_p / 100.0
    weights = [0.0] * n
    total_w = 0.0
    for i in range(n):
        if i == first:
            continue
        w = earned[i] * SKEW[groups[i][0]]
        weights[i] = w
        total_w += w
    out = []
    for i in range(n):
        if i == first:
            out.append(1.0)
        elif total_w > 0:
            out.append(BUDGET_NONFIRST * weights[i] / total_w)
        else:
            out.append(BUDGET_NONFIRST / max(1, n - 1))
    return out, first


def distinct_states(case: dict):
    """Collapse consecutive identical reads into refresh states.

    Exophase only changes a game's counts on its own refresh, so a run of
    identical reads is one state. Returns [(hours, players, rates)] with
    the first read's time for each state, oldest first.
    """
    states = []
    for read in case["history"]:
        players = read["players_tracked"]
        rates = tuple(read["rates"])
        if states and states[-1][1] == players and states[-1][2] == rates:
            continue
        states.append((hours_since_release(read["at"]), players, rates))
    return states


def refresh_cohort_rates(states, n_trophies: int, defaults):
    """New-cohort earn rate q_i per trophy, from the history's refreshes.

    For each refresh (players jump), q_i = (E_after - E_before) / (P_after -
    P_before), weighted by the player jump. Falls back to the difficulty
    defaults where no refresh exists, shrunk toward them when the measured
    jumps are small.
    """
    num = [0.0] * n_trophies
    den = [0.0] * n_trophies
    total_jump = 0.0
    for (h0, p0, r0), (h1, p1, r1) in zip(states, states[1:]):
        jump = p1 - p0
        if jump <= 0:
            continue
        total_jump += jump
        for i in range(n_trophies):
            if r0[i] is None or r1[i] is None:
                continue
            e0 = r0[i] * p0 / 100.0
            e1 = r1[i] * p1 / 100.0
            q = (e1 - e0) / jump
            num[i] += jump * min(1.0, max(0.0, q))
            den[i] += jump
    last_players = states[-1][1] if states else 0
    out = []
    for i in range(n_trophies):
        if den[i] > 0:
            measured = num[i] / den[i]
            # Shrink toward the default when the measured jumps are small.
            w = den[i] / (den[i] + 0.5 * max(1, last_players))
            out.append(w * measured + (1.0 - w) * defaults[i])
        else:
            out.append(defaults[i])
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    target_h = hours_since_release(case["target"])
    x_target = math.log1p(max(0.0, target_h))

    states = distinct_states(case)
    last_h, last_p, last_rates = states[-1]

    # --- players tracked at the target -------------------------------------
    xs = [math.log1p(max(0.0, h)) for h, _, _ in states]
    ys = [math.log(max(1, p)) for _, p, _ in states]
    # Weight a state by how long it lasted (its successor's gap, floored).
    ws = []
    for k, (h, _, _) in enumerate(states):
        nxt = states[k + 1][0] if k + 1 < len(states) else h + 0.5
        ws.append(max(0.5, nxt - h))
    a_p, b_p, _, _ = fit_line(xs, ys, ws)
    if len(states) >= 2 and len(set(p for _, p, _ in states)) >= 2:
        p_target = math.exp(a_p + b_p * x_target)
    else:
        # Flat history: no growth signal. Expect refreshes before the
        # target (Poisson, mean interval REFRESH_MEAN_HOURS); each early
        # refresh multiplies players tracked several-fold.
        p_refresh = 1.0 - math.exp(-horizon_hours / REFRESH_MEAN_HOURS)
        p_target = last_p * (1.0 + p_refresh * (REFRESH_EARLY_MULTIPLIER - 1.0))
    p_target = min(max(p_target, last_p), max(last_p * 20.0, 1_000_000.0))

    # --- per-trophy classification -----------------------------------------
    groups = [classify(t) for t in trophies]
    priors = [g[1] for g in groups]
    q_defaults, first_idx = budget_defaults(states, trophies, groups)
    cohort_q = refresh_cohort_rates(states, n, q_defaults)

    # --- per-trophy logit trend --------------------------------------------
    # First pass: slopes, so each trophy can shrink toward its group median.
    trends = []
    for i in range(n):
        tx, ty, tw = [], [], []
        for h, p, rates in states:
            r = rates[i]
            if r is None:
                continue
            tx.append(math.log1p(max(0.0, h)))
            ty.append(logit(r))
            tw.append(math.sqrt(max(1, p)))
        a, b, resid, cnt = fit_line(tx, ty, tw)
        trends.append((a, b, resid, cnt))
    group_slopes = {}
    for gname in set(g[0] for g in groups):
        slopes = [trends[i][1] for i in range(n) if groups[i][0] == gname]
        slopes.sort()
        group_slopes[gname] = slopes[len(slopes) // 2] if slopes else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        last_r = last_rates[i]
        if last_r is None:
            last_r = priors[i]
        # A trophy everyone already holds stays held.
        if last_r >= 99.5:
            med = 99.99
            spread = 0.12
            rows.append({"id": trophy["id"],
                         "q": [round4(inv_logit(logit(med) + z * spread)) for z in Z]})
            continue

        a, b, resid, cnt = trends[i]
        # Shrink the slope toward the group median; cap wild slopes.
        shrink = cnt / (cnt + 3.0) if cnt > 0 else 0.0
        b_shrunk = shrink * b + (1.0 - shrink) * group_slopes[groups[i][0]]
        b_shrunk = max(-3.0, min(3.0, b_shrunk))
        y_trend = a + b_shrunk * x_target
        r_trend = inv_logit(y_trend)

        # Structural forecast: earned counts carried forward, existing
        # players earning nothing new (a dilution floor), plus the new
        # cohort earning at q_i.
        e_last = last_r * last_p / 100.0
        e_struct = e_last + max(0.0, p_target - last_p) * cohort_q[i]
        r_struct = 100.0 * e_struct / p_target if p_target > 0 else last_r

        # Blend weights: the trend earns weight with fit confidence — a
        # flat history carries no trend signal at all, so the structural
        # (dilution) model takes over there — and the prior earns weight
        # with horizon length.
        conf = (cnt / (cnt + 3.0) if cnt > 0 else 0.0) * (1.0 - min(0.7, resid))
        conf = max(0.0, min(1.0, conf))
        w_trend = 0.20 + 0.45 * conf
        w_struct = 0.45 - 0.15 * conf
        w_prior = max(0.05, 1.0 - w_trend - w_struct)
        w_prior *= min(1.0, horizon_days / 1.5 + 0.1)
        total = w_trend + w_struct + w_prior
        w_trend, w_struct, w_prior = w_trend / total, w_struct / total, w_prior / total

        med_logit = (w_trend * logit(r_trend)
                     + w_struct * logit(r_struct)
                     + w_prior * logit(priors[i]))

        # Spread: fit residual, refresh timing, refresh-event bimodality,
        # random walk, prior blend.
        s_fit = min(1.0, max(0.18, resid)) if cnt >= 3 else 0.45
        p_refresh = 1.0 - math.exp(-horizon_hours / REFRESH_MEAN_HOURS)
        # A refresh in the window drops most rates ~1.5 logit; the spread
        # of that Bernoulli mixture is sqrt(p(1-p)) * 1.5. It vanishes
        # when a refresh is certain (long horizons).
        s_event = math.sqrt(p_refresh * (1.0 - p_refresh)) * 1.5
        s = math.sqrt(s_fit ** 2 + SIGMA_REFRESH ** 2 + s_event ** 2
                      + (SIGMA_RW * math.sqrt(horizon_days)) ** 2
                      + (0.30 * w_prior) ** 2)
        s = min(2.0, max(0.25, s))

        qs = [round4(clamp(inv_logit(med_logit + z * s))) for z in Z]
        # Enforce non-decreasing after rounding (paranoia; z ascends).
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
