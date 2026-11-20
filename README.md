# makefun-gta6-trophy-rarity-forecast-irgirival

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Irgirival/makefun-gta6-trophy-rarity-forecast-irgirival/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Irgirival/makefun-gta6-trophy-rarity-forecast-irgirival/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PS5 trophy's Exophase earn rate a week after launch with a structural count model. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## The model in one paragraph

An Exophase earn rate is a ratio — earned counts over players tracked — so
the forecaster works in **counts**, not rates. It blends three views of each
trophy's median on the logit scale: a weighted log-time trend of the rate,
a structural dilution forecast (earned counts carried forward with a
new-cohort earn rate measured from the history's own refreshes, falling
back to a budget model calibrated on launch night's first refresh, where
the 100% trophy absorbed 3,230 of 3,333 new earns and only 103 were left
for the other 54 trophies), and a difficulty-group week-one prior
calibrated on GTA V lifetime trophy rarity. The spread adds the trend
fit's residual, a refresh-timing term, a refresh-event bimodality term,
and the house baseline's random walk. Full writeup, calibration, validation
and sensitivity analysis: [METHODOLOGY.md](METHODOLOGY.md).

## Validation (practice cases)

Against practice answers reconstructed from the launch-night refresh
totals, the model scores **mean pinball loss 0.69 vs 1.42 for the house
persistence baseline**, with **87% coverage** of its 90% intervals (baseline:
0%). On a synthetic final-like case (2 days of history, 4.5-day horizon)
its medians recover the designed group rates, and they are insensitive to
every model constant — with ~20 refresh states in history, the data
carries the forecast.

## Run it

```sh
python3 main.py
```

Reads every `<id>.in` case file from `$MAKEFUN_INPUT_DIR` (default `/input`)
and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (default `/output`): one
row per trophy, five quantiles (5/25/50/75/95%), non-decreasing, in
[0.01, 99.99]. Python 3.13 standard library only — no dependencies, no
network, no randomness. Deterministic: two runs are byte-identical.
~0.08 s per case.

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Files

- `main.py` — the entry program (the whole model, ~450 lines).
- `METHODOLOGY.md` — model, calibration, validation, sensitivity, and the
  full per-trophy practice forecast table.
- `docs/forecast_chart.svg` — median forecast by difficulty group, rendered
  from the practice output.

## License

MIT. See [LICENSE](LICENSE).

<sub>Entered on [make.fun](https://make.fun/?ref=kfz6780b&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
