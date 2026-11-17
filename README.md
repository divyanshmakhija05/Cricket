# Pressure Index pipeline

Data provided is not available here to be in compliance with the competition rules

This folder holds the data pipeline and the pressure model

## Quick start
```
pip install -r requirements.txt
# put the 6 data-pack CSVs in ./data   (ball_by_ball.csv, matches.csv, teams.csv, competitions.csv, ...)
python run_all.py                  # about 1.5 minutes; add --fast to skip the stress test, --core-only to skip evidence/
```
Everything is written to `output/`. The run stops with a clear message if any step fails.

## Folder layout
```
superr_selector/
├── README.md
├── run_all.py                         one command, correct order
├── requirements.txt
├── data/                              <- put the 6 input CSVs here
├── src/                               pipeline stages (numbered = run order)
│   ├── 01_build_ball_state.py         raw data -> clean one-row-per-delivery table
│   ├── 02_pressure_chase.py           win-probability model + Pressure Index for chases  (PRIMARY)
│   ├── 03_pressure_first_innings.py   structural first-innings pressure                    (SECONDARY)
│   └── 04_audit_checks.py             29 automated PASS/WARN/FAIL checks
├── evidence/                          experiments that justify the design choices (not needed for rankings)
│   ├── E1_model_selection.py          logistic regression vs gradient boosting
│   ├── E2_runrate_alternatives.py     run-rate pressure definitions vs leverage
│   └── E3_venue_par_and_composite.py  venue par, first-innings bridge, composite index
└── output/
    ├── tables/   ball_state.csv.gz            every delivery with match state, venue par, tier
    │             pressure_chase.csv.gz        win prob, leverage, pressure percentile, flags (chase balls)
    │             pressure_first_innings.csv.gz  secondary first-innings pressure
    │             alt_runrate_pressure.csv.gz  run-rate alternatives (for the Day-5 robustness check)
    ├── logs/     one .log per script, same number as the script  <- cite these in the report
    └── charts/   02_chase_validation.png, E2_pressure_definitions.png
```
Naming rule: `0N_` = pipeline stage (run in order) · `E#_` = evidence for a design choice ·
log and chart names start with the number of the script that made them.

## Old name -> new name (for anyone following earlier messages)
| Old | New |
|---|---|
| 01_build_ball_state.py | src/01_build_ball_state.py |
| 02_pressure_index.py | src/02_pressure_chase.py |
| 02d_first_innings_pressure.py | src/03_pressure_first_innings.py |
| 00_audit_checks.py | src/04_audit_checks.py |
| 02a_model_selection.py | evidence/E1_model_selection.py |
| 02b_runrate_pressure.py | evidence/E2_runrate_alternatives.py |
| 02c_venue_par_experiments.py | evidence/E3_venue_par_and_composite.py |
| ball_state.csv.gz | output/tables/ball_state.csv.gz |
| pressure_state.csv.gz | output/tables/pressure_chase.csv.gz |
| pressure_state_inn1.csv.gz | output/tables/pressure_first_innings.csv.gz |
| runrate_pressure.csv.gz | output/tables/alt_runrate_pressure.csv.gz |
| day2_validation.png / day2b_definitions.png | output/charts/02_chase_validation.png / E2_pressure_definitions.png |
| day2_log / day2b_log / day2c_log / day2d_log / audit_log / audit_checks .txt | output/logs/02_… / E2_… / E3_… / 03_… / 01_… / 04_… .log |

## Stages 05-09 (ratings, robustness, final lists)
| Script | What it does | Main outputs |
|---|---|---|
| `src/02_pressure_chase.py` | Win probability + leverage. v3: the outcome probabilities p_k behind leverage and expected change are now a multinomial model of the match STATE (out-of-fold; `--static-mix` reproduces the old innings x phase average). | `tables/pressure_chase.csv.gz` |
| `src/05_value_added.py` | Value added per chase ball: actual minus expected change in win probability, centred and standardised per PHASE-AWARE cell (competition x phase x pressure bin; 13 bins per phase, sparse bins merged to >= 400 balls, `src/binning.py`; `--global-bins` = old pooled bins), computed on clean LEGAL balls only; wides/no-balls are scored against that baseline un-centred (z = SD units of its own cell). Explicit attribution: wides/no-balls to the bowler only; byes/leg-byes and run-outs on wides to nobody; run-outs never to bowlers, to the striker only if he was dismissed. z is split into a runs part and a wicket/residual part. | `tables/va_chase.csv.gz` |
| `src/05b_first_innings_va.py` | Same for innings 1 (structural win probability of stage 03). Layer weight lambda = ratio of the out-of-fold Brier skill scores of the two win-probability models (0.254). Ordinary first-innings balls (0.25 <= WP <= 0.75, structural PI <= 0.5) are pooled into the baseline. | `tables/va_first.csv.gz`, `layer_weight.csv` |
| `src/06_player_ratings.py` | Pressure level P: every chase ball weighted by PI^2 plus first-innings balls weighted lambda x PI1^2; runs-value and wicket/residual parts each shrunk with their OWN estimated reliability. Baseline B: usual runs value on pooled ordinary balls; uplift U shrunk directly. Consistency C: shrunk hit rate and disaster rate (disaster = total match impact in the league's worst 20%). Eligibility anchored on genuine high-pressure exposure: >= 8 pressure appearances (PI>0.5), pressure weight >= 25, >= 15 balls in the top-10% band. Rating = 0.60 zeta(P) + 0.30 zeta(C) + 0.10 zeta(U). | `tables/ratings_batters.csv`, `ratings_bowlers.csv` |
| `src/07_robustness.py` | 30 alternative definitions, early-vs-late and IPL-vs-T20I checks, bootstrap by match (500 resamples). | `tables/robustness_*.csv`, `bootstrap_ranks_*.csv` |
| `src/08_final_outputs.py` | Locks the top-10 lists (primary rating order), evidence tables, charts, dropdown-name check, near-miss eligibility audit (`near_miss_*.csv`). | `tables/final_top10_*.csv`, `evidence_*.csv`, `charts/08_*.png` |
| `src/09_audit_ratings.py` | 43 automated PASS/WARN/FAIL checks on stages 05-08. | `logs/09_audit_ratings.log` |
| `evidence/E4` | Double-counting check and repeatability of definitions. | `logs/E4_weighting_variants.log` |
| `evidence/E5` | Held-out test of the estimation window and a general-ability prior. | `logs/E5_out_of_sample_validation.log` |
| `evidence/E7` | Volume-weighted consistency (sqrt(W) votes, total-impact disaster). | `logs/E7_consistency_weighting.log` |
| `evidence/E8` | Phase-aware vs pooled standardisation cells: heterogeneity, repeatability, death-share bias. | `logs/E8_phase_aware_bins.log` |
| `evidence/E6` | Held-out tests of five proposed improvements (wicket estimators, pooled baseline, continuous consistency, state-aware outcome mix, first-innings weight). Adopted: pooled baseline, state-aware mix, calibrated first-innings weight. | `logs/E6_proposed_improvements.log` |

Run order is handled by `run_all.py` (about 25 minutes in full; `--fast` uses 100 bootstrap resamples and skips the stage-02 stress test; `--core-only` skips E1, E3-E6). Stage 07 needs the table written by E2, so E2 runs before it.

## Honest summary of what the data can and cannot say
- Repeatability (mean correlation between two random halves of the matches; early vs late matches) of the shrunk pressure level: bowlers 0.33; 0.37, batters 0.28; 0.07. Final rating: bowlers 0.31; 0.35, batters 0.25; 0.10.
  Bowler ratings carry real signal; batter pressure ratings carry little, so the batter top 10 is uncertain (about 4.5 of 10 stay in a resampled top 10; bowlers 5.2).
- Wickets: the wicket/residual part has little between-player spread (tau 0.02 vs 0.05-0.06 for runs); REML, Paule-Mandel and a Poisson-Gamma rate model give no better held-out prediction (E6).
- Consistency: continuous downside semi-variance and sqrt(W)-weighted votes are less repeatable than binary rates (E6, E7); defining a disaster by total match impact is neutral on the data (batters +0.06, bowlers -0.04, within noise) and kept for its volume logic; the previous definition is a robustness variant (`cons_mode="a"`). Batter consistency has little repeatable signal in any form.
- Phase-aware cells (E8): the pooled bins favoured death-over hitters and penalised death-over bowlers (change in rating vs death share: -0.42 SD per unit for batters, +0.36 for bowlers); repeatability is unchanged.
- Eligibility: floors are pre-specified; `near_miss_*.csv` lists players who missed a floor narrowly (batter Naman Dhir would rank 7th, bowler Sufiyan Muqeem 4th if the appearance floor were 6-7).
- Baselines: pooling ordinary first-innings balls raises baseline repeatability (batters 0.22 -> 0.40, bowlers 0.03 -> 0.23) and uplift repeatability, but uplift is still weighted only 0.10.
- 8,326 high-pressure balls (stage 02) counts legal balls only; with wides/no-balls the top 10% of chase balls is 8,889.

## Design decisions
1. **Data**: IPL + T20I. Excluded from modelling: no-result, DLS/rain-affected (38), tied/Super-Over (10), one-innings, "Other T20".
2. **Extras are estimated**: the file has no run values for wides/byes/leg-byes. Averages calibrated on 399 official "won by N runs" margins
   (wide 1.15, no-ball 1.0, bye 2.06, leg-bye 1.30, other 1.11); margin error 2.11 runs on average.
3. **Pressure** = how much one ball can swing the win probability (leverage). **Primary: chases**, high = top 10% (8,326 balls), extreme = top 2.5% (2,063).
   **Secondary: first innings** (structural model, 9,250 high-pressure balls): plausible but only weakly validated, so it is down-weighted
   and used as a sensitivity check.
4. **Win-probability model**: logistic regression + venue par (shrunk leave-one-match-out) + tier indicators.
   Cross-validation folds are DETERMINISTIC (match_id mod 5), so every machine gets identical numbers.
   Chase: log-loss 0.394, AUC 0.901, calibration error 0.008 (all tiers within 0.003). First innings: generic AUC 0.698; structural AUC 0.702.
   Differences below ~0.005 in log-loss are within fold-to-fold noise and should not be over-interpreted.

## Key results
- Chase wickets per 100 balls rise from 4.9 (ordinary) to 11.1 (extreme pressure); strike rate 135 -> 168.
- Within the middle overs: high vs ordinary pressure = strike rate 142 vs 133, wickets 5.7 vs 4.7 per 100 balls.
- Stress test (score errors twice the observed size): pressure ranking correlation 0.96; 82-84% of top-10% balls retained.
- Rejected: run-rate gap / ratio (96-99% of their top balls are already-lost chases); generic first-innings leverage (start-of-innings artifact).

## Limitations
- Extra run values are estimated, so individual innings can be off by several runs.
- First-innings pressure has a weaker behavioural signature than chase pressure (small strike-rate lift, no consistent wicket lift).
- Associate-nation T20Is are few (50 matches) and are handled with a tier indicator and later down-weighting.
