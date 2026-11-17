"""
Stage 05b: first-innings value-added layer (SECONDARY information, down-weighted by the calibrated skill ratio).
Usage: python src/05b_first_innings_va.py <out_dir>
Same construction as stage 05, for innings 1 with the STRUCTURAL win-probability of stage 03:
  VA = actual change in structural WP over the ball, centred and standardised per structural-pressure bin x competition (z in SD units);
  z split into a runs part (league mean z for the runs scored, same bin) and a wicket/residual part; same attribution rules as stage 05.
  The last ball of an innings has no next state and is dropped.
Two uses in stage 06:
  * pressure-level estimate: each innings-1 ball enters with weight  lambda * PI1^2 * opposition weight,
    lambda = BSS(innings 1) / BSS(chase) = ratio of out-of-fold Brier skill scores of the two win-probability models (about 0.25):
    the first-innings model carries about a quarter of the chase model's skill, so its balls carry about a quarter of the information.
  * baseline: ORDINARY first-innings balls (0.25 <= WP <= 0.75 and structural PI <= 0.5) are pooled with the ordinary competitive chase balls
    (about 3.5x as many balls), because 9,145 chase balls alone leave finishers with 1-6 baseline balls.
Output: tables/va_first.csv.gz, tables/layer_weight.csv, logs/05b_first_innings_va.log
"""
import sys, os, numpy as np, pandas as pd
from sklearn.metrics import brier_score_loss
O = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
T = os.path.join(O, "tables"); LOG = os.path.join(O, "logs"); log = []
def say(s=""): print(s); log.append(str(s))
b = pd.read_csv(os.path.join(T, "ball_state.csv.gz"), low_memory=False)
f1 = pd.read_csv(os.path.join(T, "pressure_first_innings.csv.gz")); pc = pd.read_csv(os.path.join(T, "pressure_chase.csv.gz"))
key = ["match_id", "innings", "innings_ball_num"]

# ---- calibrated layer weight = ratio of out-of-fold Brier skill scores
def bss(df, col, inn):
    d = df[(df.innings == inn) & df[col].notna()].merge(b[key + ["batting_won"]], on=key); y = d.batting_won.astype(float).values; p = d[col].values
    return 1 - brier_score_loss(y, p) / brier_score_loss(y, np.full(len(y), y.mean())), brier_score_loss(y, p)
b2, br2 = bss(pc, "wp", 2); b1, br1 = bss(f1, "wp_inn1", 1); LAM = b1 / b2
say(f"Brier skill score vs the no-skill forecast: chase {b2:.3f} (Brier {br2:.3f}) | innings 1 {b1:.3f} (Brier {br1:.3f}) | layer weight lambda = {LAM:.3f}")
pd.DataFrame({"lambda_first_innings": [LAM], "bss_chase": [b2], "bss_first": [b1]}).to_csv(os.path.join(T, "layer_weight.csv"), index=False)

# ---- value added, innings 1
d = f1[f1.innings == 1].merge(b, on=key, how="left").sort_values(["match_id", "innings_ball_num"]).reset_index(drop=True)
d["wp_next"] = d.groupby("match_id").wp_inn1.shift(-1); d["dwp"] = d.wp_next - d.wp_inn1
d = d[d.dwp.notna() & d.pressure_pct_inn1.notna()].reset_index(drop=True)
d["pbin"] = pd.cut(d.pressure_pct_inn1, [0, .25, .5, .6, .7, .8, .9, .95, 1.0], labels=False, include_lowest=True)
GROUP = ["comp_group", "pbin"]
PH = None if "--global-bins" in sys.argv else next((a for a in sys.argv if a.startswith("--phase-aware")), "--phase-aware")   # default: phase-aware bins; --global-bins = old pooled bins
if PH:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from binning import assign_bins
    d["pbin"] = assign_bins(d, ["comp_group", "phase"], d.pressure_pct_inn1.values, [0, .25, .5, .6, .7, .8, .9, .95, 1.0] if PH == "--phase-aware" else [0, .5, .75, .9, .95, 1.0], 400)
    GROUP = ["comp_group", "phase", "pbin"]; say(f"phase-aware standardisation (innings 1): {d.groupby(GROUP).ngroups} cells ({PH})")
nonb = d.wicket_fell.eq(1) & d.bowler_wicket.eq(0); wide = ~d.legal & ~d.ball_faced; nob = ~d.legal & d.ball_faced; normal = d.legal & ~(d.extra_runs > 0)
ref = d[normal & ~nonb].groupby(GROUP).dwp.agg(bin_mean="mean", bin_sd="std").reset_index(); d = d.merge(ref, on=GROUP, how="left")
d["VA_c"] = np.where(d.legal, d.dwp - d.bin_mean, d.dwp); d["z_bat"] = d.VA_c / d.bin_sd; d["z_bowl"] = -d.z_bat
d["bat_ok"] = (normal & ~nonb) | (nonb & ~d.dismissal_type.eq("Retire Out") & d.legal & (d.dismissed_player_id == d.striker_id)); d["bowl_ok"] = (normal & ~nonb) | ((wide | nob) & ~nonb)
d["rcls"] = d.bat_runs.where(d.bat_runs.isin([0, 1, 2, 3, 4, 6]), 4)
lk = d[normal & ~nonb].groupby(GROUP + ["rcls"]).z_bat.mean().rename("lk").reset_index(); d = d.merge(lk, on=GROUP + ["rcls"], how="left"); d["lk"] = d.lk.fillna(0.0)
d["zr_bat"] = np.where(d.bat_ok, d.lk, 0.0); d["zw_bat"] = d.z_bat - d.zr_bat
d["zr_bowl"] = np.where(d.legal, -d.lk, d.z_bowl); d["zw_bowl"] = d.z_bowl - d.zr_bowl
d["PI"] = d.pressure_pct_inn1; d["pw"] = 0.0; d["pe"] = LAM * d.PI ** 2                           # estimation weight only (never eligibility / consistency)
d["opp_w"] = np.where(d.tier == "T20I_Assoc", 0.7, 1.0)
d["baseline_ball"] = d.wp_inn1.between(0.25, 0.75) & (d.PI <= 0.5)
d["kind"] = "first"; d["wp"] = d.wp_inn1; d["li"] = d.leverage_inn1 / d.leverage_inn1.mean(); d["VA_bat"] = d.VA_c; d["VA_bowl"] = -d.VA_c
d["high_pressure"] = 0; d["extreme_pressure"] = 0; d["knock_out"] = d.knock_out if "knock_out" in d else np.nan
say(f"innings-1 balls with value added: {len(d):,} | with estimation weight > 0: {(d.pe > 0).sum():,} | ordinary (baseline) balls: {int(d.baseline_ball.sum()):,}")
say(f"centred VA mean by bin (max abs): {d.groupby(GROUP).VA_c.mean().abs().max():.4f} | attributable to batter {d.bat_ok.mean():.3f}, to bowler {d.bowl_ok.mean():.3f}")
keep = ["match_id", "innings_ball_num", "comp_group", "comp_name", "season", "tier", "match_date", "over_no", "phase", "striker_id", "striker_name", "bowler_id", "bowler_name",
        "batting_team_id", "bowling_team_id", "kind", "bat_ok", "bowl_ok", "bat_runs", "wicket_fell", "bowler_wicket", "wp", "leverage", "li", "PI", "pw", "pe", "opp_w",
        "baseline_ball", "high_pressure", "extreme_pressure", "VA_c", "z_bat", "z_bowl", "VA_bat", "VA_bowl", "zr_bat", "zw_bat", "zr_bowl", "zw_bowl", "batting_won", "knock_out"]
d["leverage"] = d.leverage_inn1
d[[c for c in keep if c in d.columns]].to_csv(os.path.join(T, "va_first.csv.gz"), index=False, float_format="%.6g")
open(os.path.join(LOG, "05b_first_innings_va.log"), "w").write("\n".join(log))
