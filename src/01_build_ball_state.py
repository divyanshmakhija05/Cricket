"""
Day 1: Data audit + ball-state table for SupeRR Selector S2 (Pressure problem).
Usage:  python 01_build_ball_state.py <data_dir> <out_dir>
Output: <out_dir>/tables/ball_state.csv.gz (one row per delivery, innings 1-2 only)
        <out_dir>/logs/01_build_ball_state.log

v2 changes (after full re-audit):
  * DLS / rain-affected detection also reads the result TEXT (4 matches had empty revised-target columns)
  * tied / Super-Over-decided matches flagged (`tied`) - their win label is ambiguous
  * no-result detection is case-insensitive
  * adds team names, competition `tier` (IPL / T20I_Full / T20I_Assoc / OTHER_T20) and a shrunk
    leave-one-match-out venue par (`venue_par`, `tier_mean`)
"""
import sys, numpy as np, pandas as pd
D = sys.argv[1] if len(sys.argv) > 1 else "data"
O = sys.argv[2] if len(sys.argv) > 2 else "output"
import os
os.makedirs(f"{O}/tables", exist_ok=True); os.makedirs(f"{O}/logs", exist_ok=True)
log = []
def say(s): print(s); log.append(str(s))

b = pd.read_csv(f"{D}/ball_by_ball.csv")
m = pd.read_csv(f"{D}/matches.csv")
c = pd.read_csv(f"{D}/competitions.csv")
say(f"raw rows: {len(b):,} | matches in ball-by-ball: {b.match_id.nunique()} of {len(m)} in matches.csv")

# ---- 1. Drop super overs (innings 3/4) ------------------------------------
say(f"super-over rows dropped: {(b.innings>2).sum()}")
b = b[b.innings <= 2].copy()

# ---- 2. Competition label -------------------------------------------------
m = m.merge(c[["comp_id","comp_name","match_format","season"]], on="comp_id", how="left")
m["comp_group"] = np.where(m.comp_name.str.startswith("IPL"), "IPL",
                   np.where(m.match_format == "T20I", "T20I", "OTHER_T20"))
m["match_date"] = pd.to_datetime(m.match_date)

# ---- 3. Match-level flags --------------------------------------------------
res = m.result.fillna("")
bad = res.str.contains(r"no result|abandon|washed", case=False, regex=True)
say(f"No-result/abandoned matches flagged: {bad.sum()} (flagged 'no_result')")
m["no_result"] = bad
dls_cols = m.revised_target.notna() | m.first_inn_revised_overs.notna() | m.second_inn_revised_overs.notna()
dls_txt = res.str.contains(r"DLS|D/L|Duckworth|VJD", case=False, regex=True)
m["dls"] = dls_cols | dls_txt
say(f"DLS/rain-affected flagged: {int(m.dls.sum())} ({int((dls_txt & ~dls_cols).sum())} found only through the result text)")
m["tied"] = res.str.contains(r"\btied?\b|super over|eliminator", case=False, regex=True)
say(f"Tied / Super-Over-decided matches flagged: {int(m.tied.sum())} (excluded from win-probability training)")

# ---- 3b. Team names, tier ---------------------------------------------------
import os
FULL = {"India", "Australia", "England", "South Africa", "New Zealand", "Pakistan", "Sri Lanka", "Bangladesh",
        "West Indies", "Afghanistan", "Ireland", "Zimbabwe"}                    # the 12 ICC full members
tm = pd.read_csv(f"{D}/teams.csv").drop_duplicates("team_id").set_index("team_id").team_name

# ---- 4. Delivery semantics ------------------------------------------------
# NOTE: `runs` = runs off the bat only. `extras1_runs` is a TYPE CODE (wide=2, legbye=3, no-ball=1,
# bye=4, other=5, none=6) and NOT a run count. Extra runs are rebuilt by convention below.
x = b.extras_type.fillna("none")
b["is_wide"]   = x.eq("wide")
b["is_noball"] = x.eq("Overstep")
b["is_bye"]    = x.isin(["byes", "legbyes"])
b["is_other_x"]= x.eq("other")
b["legal"]     = ~(b.is_wide | b.is_noball | b.is_other_x)      # 98.2% of overs then have exactly 6 legal balls
# The file records NO run values for wides / byes / leg-byes (all 0 or a type code), and the boundary flags
# carry no information about them either (checked), so extra runs are ESTIMATED. Unit values are calibrated by
# least squares on the 399 non-DLS, non-tied matches with an official "won by N runs" margin:
#   official margin = bat-run diff + sum_k unit_k * (count_k in inn1 - count_k in inn2)
# Estimates (+/- bootstrap SE): wide 1.15 (0.04), no-ball 0.89 (0.17 -> fixed at the rule-based 1.0),
# bye 2.06 (0.18), leg-bye 1.30 (0.08), other 1.11 (0.24), extras_2 column 1.04 (0.24 -> 1.0 run each).
# The same rule is applied to every match, so no outcome information leaks into the features.
b["bat_runs"] = b.runs
U = dict(wide=1.15, nb=1.00, bye=2.06, lb=1.30, other=1.11)
x_lb = x.eq("legbyes"); x_by = x.eq("byes")
b["extra_runs"] = (b.is_wide*U["wide"] + b.is_noball*U["nb"] + x_by*U["bye"] + x_lb*U["lb"]
                   + b.is_other_x*U["other"] + x.eq("Penalty")*5 + b.extras_2_runs)
b["total_runs"] = b.bat_runs + b.extra_runs
# runs charged to bowler: bat runs + wides + no-balls (+ extras_2) but NOT byes/leg-byes/penalty
b["bowler_runs"] = b.bat_runs + b.is_wide*U["wide"] + b.is_noball*U["nb"] + b.is_other_x*U["other"] + b.extras_2_runs
b["ball_faced"]  = ~b.is_wide                                   # batter faces no-balls but not wides

# ---- 5. Wickets -----------------------------------------------------------
b["wicket_fell"] = b.is_wicket.astype(int)                      # any fall of wicket (team state)
b["bowler_wicket"] = (b.is_wicket & ~b.dismissal_type.isin(["RunOut","Retire Out","Obstracting the field",
                                                           "Obstructing the field"])).astype(int)

# ---- 6. Sort + cumulative match state ------------------------------------
b = b.sort_values(["match_id","innings","innings_ball_num"]).reset_index(drop=True)
g = b.groupby(["match_id","innings"])
b["legal_balls_before"] = g.legal.cumsum() - b.legal.astype(int)
b["score_before"]       = g.total_runs.cumsum() - b.total_runs
b["wkts_before"]        = g.wicket_fell.cumsum() - b.wicket_fell
b["score_after"]        = b.score_before + b.total_runs
b["wkts_after"]         = b.wkts_before + b.wicket_fell

# ---- 7. Attach match info, innings length (DLS-aware), targets ------------
b = b.merge(m[["match_id","comp_id","comp_name","comp_group","season","match_date","stage","venue_id",
               "venue_country","win_team_id","result","no_result","dls","tied","knock_out",
               "first_inn_revised_overs","second_inn_revised_overs","revised_target"]], on="match_id", how="left")
rev = np.where(b.innings == 1, b.first_inn_revised_overs, b.second_inn_revised_overs)
# NOTE: *_revised_overs columns are already stored in BALLS (all multiples of 6), so no x6.
b["max_balls"]    = np.where(pd.isna(rev), 120, rev).astype(int)
b["balls_left"]   = (b.max_balls - b.legal_balls_before).clip(lower=0)
b["wkts_in_hand"] = 10 - b.wkts_before

inn1_tot = b[b.innings == 1].groupby("match_id").total_runs.sum()
b["inn1_total"]  = b.match_id.map(inn1_tot)
b["target"]      = np.where(b.innings == 2, np.where(b.revised_target.notna(), b.revised_target,
                                                     np.round(b.inn1_total) + 1), np.nan)
b["runs_needed"] = b.target - b.score_before
b["req_rr"]      = np.where((b.innings == 2) & (b.balls_left > 0), b.runs_needed / b.balls_left * 6, np.nan)
b["curr_rr"]     = np.where(b.legal_balls_before > 0, b.score_before / b.legal_balls_before * 6, np.nan)
b["rrr_minus_crr"] = b.req_rr - b.curr_rr
b["over_no"]     = b.overid                                      # 1..20 (1-based)
b["phase"]       = pd.cut(b.over_no, [0,6,15,20], labels=["powerplay","middle","death"])
b["batting_won"] = (b.batting_team_id == b.win_team_id).astype(float).where(b.win_team_id.notna() & ~b.no_result)

# ---- 7b. Team names, tier, shrunk leave-one-out venue par ---------------------
b["batting_team"] = b.batting_team_id.map(tm); b["bowling_team"] = b.bowling_team_id.map(tm)
mt = b.groupby("match_id").agg(venue=("venue_id", "first"), grp=("comp_group", "first"), t1=("batting_team", "first"),
                               t2=("bowling_team", "first"), inn1=("inn1_total", "first"), n_inn=("innings", "nunique"),
                               nr=("no_result", "first"), dls=("dls", "first"))
mt["tier"] = np.where(mt.grp == "IPL", "IPL", np.where(mt.grp == "OTHER_T20", "OTHER_T20",
             np.where(mt.t1.isin(FULL) & mt.t2.isin(FULL), "T20I_Full", "T20I_Assoc")))
mt["elig"] = mt.grp.isin(["IPL", "T20I"]) & (mt.n_inn == 2) & ~mt.nr & ~mt.dls   # complete 20-over first innings
KS = 7                                                                          # shrinkage strength (pseudo-matches)
mu = mt[mt.elig].groupby("tier").inn1.mean()
mt["tier_mean"] = mt.tier.map(mu).fillna(mt[mt.elig].inn1.mean())          # OTHER_T20 has no eligible matches -> overall mean
gs = mt[mt.elig].groupby(["tier", "venue"]).inn1.agg(["sum", "count"]).rename(columns={"sum": "vs", "count": "vn"})
mt = mt.join(gs, on=["tier", "venue"])
mt[["vs", "vn"]] = mt[["vs", "vn"]].fillna(0)
own = np.where(mt.elig, mt.inn1, 0.0); n_oth = mt.vn - mt.elig.astype(int)           # leave THIS match out
mean_oth = np.where(n_oth > 0, (mt.vs - own) / np.maximum(n_oth, 1), 0.0)
mt["venue_par"] = (n_oth * mean_oth + KS * mt.tier_mean) / (n_oth + KS)
mt["venue_n_other"] = n_oth
b = b.merge(mt[["tier", "tier_mean", "venue_par", "venue_n_other"]], left_on="match_id", right_index=True, how="left")
say(f"matches by tier: {mt.tier.value_counts().to_dict()} | tier means: {mu.round(1).to_dict()}")

# ---- 8. Sanity checks -----------------------------------------------------
say("\n--- sanity checks ---")
_o = b.groupby(["match_id", "innings", "over_no"]).agg(leg=("legal", "sum"), last=("innings_ball_num", "max"))
_o = _o.join(b.groupby(["match_id", "innings"]).innings_ball_num.max().rename("inn_last"), on=["match_id", "innings"])
_c = _o[_o["last"] != _o.inn_last]
say(f"complete (non-final) overs with exactly 6 legal balls: {(_c.leg == 6).mean():.4f} (n={len(_c):,})")
say(f"2nd-innings rows: {(b.innings==2).sum():,}; with valid req_rr: {b.req_rr.notna().sum():,}")
# reconcile reconstructed margin vs official "won by N runs" (case-insensitive, all spelling variants)
_r = m.result.fillna("")
_mr = _r.str.extract(r"(\d+)\s*runs?\b", flags=2)[0].astype(float)
_ok = (_r.str.contains(r"\bwon\b", case=False) & _mr.notna()
       & ~_r.str.contains(r"wkts?|wickets?|DLS|D/L|VJD|tied|super over", case=False)) & ~m.dls & ~m.tied & ~m.no_result
mm = m[_ok].copy(); mm["margin"] = _mr[_ok]
tot = b.groupby(["match_id","innings"]).total_runs.sum().unstack()
j = mm.set_index("match_id").join((tot[1]-tot[2]).rename("d"), how="inner")
err = (j.d.abs() - j.margin)
say(f"margin reconciliation (n={len(j)}): mean abs error {err.abs().mean():.2f} runs, within +-2 runs {(err.abs()<=2).mean():.1%}, mean error {err.mean():+.2f}")
say("  -> extra run values are not recorded in the file; calibrated averages used. Individual innings can still be off by several runs.")
say(f"single-innings matches (kept, no target): {int((b.groupby('match_id').innings.nunique()==1).sum())}")
say(f"rows by comp_group: {b.comp_group.value_counts().to_dict()}")

cols = ["match_id","comp_id","comp_name","comp_group","season","match_date","stage","venue_id","venue_country",
        "innings","over_no","over_ball_num","innings_ball_num","legal","batting_team_id","bowling_team_id",
        "striker_id","striker_name","non_striker_id","bowler_id","bowler_name","batting_order","batting_type",
        "bowling_type","pace_or_spin","bat_runs","extra_runs","total_runs","bowler_runs","ball_faced",
        "is_four","is_six","wicket_fell","bowler_wicket","dismissal_type","dismissed_player_id",
        "legal_balls_before","balls_left","max_balls","score_before","wkts_before","wkts_in_hand","phase",
        "inn1_total","target","runs_needed","req_rr","curr_rr","rrr_minus_crr","batting_won","dls","no_result","tied","knock_out",
        "batting_team","bowling_team","tier","tier_mean","venue_par","venue_n_other"]
b[cols].to_csv(f"{O}/tables/ball_state.csv.gz", index=False)
open(f"{O}/logs/01_build_ball_state.log", "w").write("\n".join(log))
say(f"\nsaved {len(b):,} rows x {len(cols)} cols -> {O}/tables/ball_state.csv.gz")
