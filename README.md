# Who holds up under pressure?

Code for my SupeRR Selector Season 2, Round 1 submission: ranking the batters and bowlers who deliver most reliably in high-pressure moments of IPL and T20I matches (2022 onwards).

The competition data is not included, in line with the competition rules. Everything else, from raw CSVs to the final lists, is reproducible from this repository.

## The result

| # | Batters | Bowlers |
|---|---|---|
| 1 | Heinrich Klaasen | Rashid Khan |
| 2 | Suryakumar Yadav | Sunil Narine |
| 3 | Ashutosh Sharma | Kuldeep Yadav |
| 4 | Reeza Hendricks | Josh Hazlewood |
| 5 | Tim David | Lockie Ferguson |
| 6 | Shreyas Iyer | Mohammad Nawaz |
| 7 | Mohammad Nawaz | Patrick Cummins |
| 8 | Kamindu Mendis | Axar Patel |
| 9 | Daryl Mitchell | Mohsin Khan |
| 10 | Fakhar Zaman | Krunal Pandya |

These are the players the model rates as most reliable under pressure. They are not a verdict on overall ability. The bowler ranking is clearly more repeatable than the batter ranking (see "What the data can and cannot say").

## The idea in short

1. **Pressure** is how much the next ball can swing the match: the expected change in win probability, taken over the possible outcomes of the ball, then ranked as a percentile across all chase balls. The top 10% of chase balls count as high pressure.
2. **Performance** is value added: the actual change in win probability minus the change expected from the match state, compared with other balls in the same phase and pressure level.
3. **Rating** = 0.60 pressure level + 0.30 consistency + 0.10 uplift over the player's usual level. Small samples are shrunk toward the league average, with the strength estimated from the data.

Wides and no-balls count against the bowler only, byes and leg-byes against nobody. Full rules are in the stage 05 script and the report.

## Run it

```bash
pip install -r requirements.txt
# put the data-pack CSVs in ./data (ball_by_ball.csv is required)
python run_all.py
```

A full run takes roughly half an hour (the script prints the exact time at the end). Useful flags:

- `--fast` uses 100 bootstrap resamples instead of 500 and skips the stage 02 stress test. Final numbers come from the full run.
- `--core-only` runs the numbered stages and skips the `evidence/` experiments.

Everything is written to `output/`. The run stops with a clear message if a step fails. Cross-validation folds are `match_id mod 5` and the bootstrap seed is fixed, so repeated runs give identical results.

## What is in the repo

```
run_all.py          runs everything in order
src/                the pipeline, one numbered script per stage
  01  ball-by-ball data to a clean ball-state table
  02  win-probability model and pressure (chases, primary)
  03  first-innings pressure (secondary)
  04  input checks
  05  value added per ball, with phase-aware standardisation
  05b first-innings value added
  06  player ratings
  07  robustness: 30 alternative definitions, bootstrap by match
  08  final lists, evidence tables, charts, near-miss audit
  09  43 automated checks on the ratings
evidence/           experiments behind each design choice (E1 to E8)
output/             tables, logs and charts produced by the pipeline
```

Each log in `output/logs/` carries the number of the script that wrote it.

## Checks

Stage 09 runs 43 automated checks (value-added identities, attribution rules, standardisation cells, eligibility floors, determinism, and that every final player id exists in the mapping file). The current log reads 43 pass, 0 fail, and 1 warning. The warning is advisory: two different players are called Rashid Khan (ids 4722 and 20172). The rankings use 4722, the Afghanistan and Gujarat Titans player.

## What the data can and cannot say

- **Repeatability.** Correlation of the final rating between two random halves of the matches / between earlier and later matches: bowlers 0.31 / 0.35, batters 0.25 / 0.10. Bowler ratings carry real signal. Batter ratings carry much less, so batter ranks 6 to 10 are close to ties.
- **Resampling.** Across 500 resamples of the matches, about 4.5 of the 10 batters and 5.2 of the 10 bowlers stay in a top 10.
- **Rank 10.** Fakhar Zaman (batters) and Krunal Pandya (bowlers) lead rank 11 by about 0.03 rating points.
- **Eligibility.** At least 8 pressure appearances, a total pressure weight of 25, and 15 top-band balls, fixed in advance. Naman Dhir and Sufiyan Muqeem miss only the appearance floor (`output/tables/near_miss_*.csv`).
- **Not a clutch trait.** A general-ability prior did not improve held-out prediction, so I do not claim a separate clutch skill.
- **Estimated extras.** The data has no run values for wides, byes or leg-byes. I estimated them from 399 official win margins (average error 2.11 runs).

## Key numbers

- Chase win-probability model: log-loss 0.394, AUC 0.901, calibration error 0.008. The first-innings model is much weaker (AUC about 0.70), so it only adds a down-weighted layer (weight 0.254).
- Pressure is real in the data: strike rate rises from about 133 to 171 and wickets from 4.9 to 11.0 per 100 balls between the ordinary and the top pressure bands.
- Within the middle overs of competitive chases, high-pressure balls have strike rate 148 against 129 and 6.1 against 4.6 wickets per 100 balls.
- Stage 02 flags 8,346 high-pressure chase balls (2,059 extreme).
- Rejected: required-rate gap and ratio as pressure measures (96 to 99% of their extreme balls are in chases already lost).

## Report and slides

The methodology report and slide deck for the competition were written separately. This repository holds the code and outputs behind them.
