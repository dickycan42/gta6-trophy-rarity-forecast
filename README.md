# GTA VI Trophy Rarity Forecast: Week-One Earn Rates on Exophase

[![GTA VI Trophy Rarity Forecast](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dickycan42/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dickycan42/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the percent
of Exophase-tracked players who have earned it one week after launch
(2026-11-26T00:00:00Z), as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

## How it works

A trophy rate is a ratio: players who earned it over players tracked. Both
sides move, in opposite directions for the rate: earned counts only grow as
players progress, while the tracked-player count jumps upward each time
Exophase refreshes, diluting every rate. So the program extrapolates the two
counts separately — each fitted against log-time since launch, which is how
launch-week progress saturates — and divides them back into a rate, then
averages that with a direct log-time trend of the logit rate. Both
extrapolations are shrunk toward persistence (saturating curves overshoot
under a straight line, and tiny earned counts make fitted slopes mostly
noise; the shrinkage is stronger when the last earned count is small).

A weak prior from the trophy's tier and description (story chapter, grind,
collectible, money, one-off) steadies trophies whose history is flat or
noisy; its weight grows with the horizon and is negligible at 90 minutes.
When the history shows no refresh at all (flat player count), the median is
shifted down by a dilution allowance, because a refresh the program has not
seen yet is expected before the target.

Everything is computed on the logit scale so rare trophies count as much as
common ones, with rates clamped to 0.01–99.99% so the logit stays defined at
both ends. The spread widens with the square root of the horizon plus a
refresh-uncertainty term, so the final's 4.5-day spread is far wider than the
practice cases' 90-minute spread. A (near-)saturated trophy (last rate
100%) keeps a very high median with a one-sided tail downward.

## Running it

Python 3.13, standard library only, no network, deterministic (no randomness;
two runs on the same case produce byte-identical output, well under the
8-second limit per case):

```sh
python3 main.py
```

It reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`) in name
order and writes one `<id>.out` forecast per case to `$MAKEFUN_OUTPUT_DIR`
(default `/output`).

## Local checks (practice cases p00/p01)

- Both outputs valid: 55 trophies, each id once, five non-decreasing
  quantiles in 0–100, under 65,536 bytes.
- Determinism: two runs byte-identical.
- Runtime: ~0.06 s per case on this machine.
- Pinball loss on the logit scale, scored against the refreshed page under
  two assumed answers (earned counts frozen, and new players earning at half
  the old rate): about 0.37 and 0.38, versus the house persistence baseline's
  about 1.51 on the practice — the baseline's narrow band is blown open by
  the first refresh, while this forecast expects the dilution and carries a
  wide enough spread (90% coverage about 0.98 on those scenarios).
- Robustness tested on: a single-read history, all-zero rates, scattered
  null rates, a zero player count, a 1-minute horizon, a 30-day horizon, and
  a synthetic 121-read final-like case (2 days of reads, refreshes every
  4 hours, player growth and story progression), where it scored 0.27
  against the persistence baseline's 0.31.

## License

MIT — see [LICENSE](LICENSE).
