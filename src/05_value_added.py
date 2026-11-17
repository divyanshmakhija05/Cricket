"""
Day 3 (stage 05): value added per ball (VA) for chase balls, with explicit attribution rules.

  VA  = actual change in batting-side win probability  -  expected change (exp_dwp from stage 02)
  Batter credit = +VA, bowler credit = -VA.

Attribution rules (who is allowed to be credited / charged for a ball):
  normal legal ball ................ batter and bowler both attributed
  bye / leg-bye (legal, extras>0) .. neither (the run was not earned by the batter or conceded by the bowler)
  wide ............................. bowler only, charged the full actual change in WP (a wide is a failed delivery; expected = 0)
  no-ball / 'other' illegal ball ... bowler only (batter excluded: free-hit / extra-run distortion)
  run-out / retire-out / obstruction (not a bowler wicket):
        bowler NOT credited or charged; batter attributed only if the DISMISSED player is the striker, the ball was legal
        and the dismissal is not a tactical retire-out. A run-out on a wide / no-ball is attributed to nobody.

Standardisation cells are PHASE-AWARE: competition x phase x pressure bin (13 bins per phase, sparse bins merged so every cell has >= 400 balls; src/binning.py),
so a middle-over spinner is never standardised against death-over hitting. Reference mean / SD / runs lookup are computed on CLEAN LEGAL balls only; wides and
no-balls are scored against that clean legal baseline and are not centred (a wide earns its full penalty). `--global-bins` reproduces the old pooled bins.
Centring: stage-02 expected change uses the league outcome mix per phase, so it is slightly biased in the most extreme states
(batters take more risks there: mean VA +0.02 in the top 2.5%). We subtract the league mean VA of each pressure bin
(per competition group), so a player is compared with the league AT THE SAME PRESSURE LEVEL.
Standardising: z = centred VA / (SD of centred VA in the same bin and competition group). z is in "SD units of its own pressure
band", so leverage is NOT counted twice when pressure weights are applied afterwards (raw-WP version kept as a sensitivity).
Output: output/tables/va_chase.csv.gz
"""
import sys, os, numpy as np, pandas as pd
O = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
T = os.path.join(O, "tables"); LOG = os.path.join(O, "logs"); os.makedirs(LOG, exist_ok=True)
log = []
def say(s=""): print(s); log.append(str(s))

b = pd.read_csv(os.path.join(T, "ball_state.csv.gz"), low_memory=False)
p = pd.read_csv(os.path.join(T, "pressure_chase.csv.gz"))
c = p[p.innings == 2].merge(b, on=["match_id", "innings", "innings_ball_num"], how="left").sort_values(["match_id", "innings_ball_num"])
assert c.striker_id.notna().all() and not c.duplicated(["match_id", "innings_ball_num"]).any()
c = c.reset_index(drop=True)
say(f"chase balls in modelling set: {len(c):,} from {c.match_id.nunique()} matches")

# ---- actual change in WP (win prob of the batting side before the next ball; last ball -> actual result)
c["wp_next"] = c.groupby("match_id").wp.shift(-1)
last = c.wp_next.isna(); c.loc[last, "wp_next"] = c.loc[last, "batting_won"]
c["dwp"] = c.wp_next - c.wp
c["VA_raw"] = c.dwp - c.exp_dwp
c["PI"] = c.pressure_pct

# ---- attribution
nonbowler_w = c.wicket_fell.eq(1) & c.bowler_wicket.eq(0)
retire = c.dismissal_type.eq("Retire Out")
wide = ~c.legal & ~c.ball_faced
noball = ~c.legal & c.ball_faced
byelb = c.legal & (c.extra_runs > 0)
normal = c.legal & ~byelb
c["kind"] = np.select([wide, noball, byelb, nonbowler_w & ~retire, retire], ["wide", "noball", "bye_lb", "runout_etc", "retire"], "normal")
c["bat_ok"] = (normal & ~nonbowler_w) | (nonbowler_w & ~retire & c.legal & (c.dismissed_player_id == c.striker_id))
c["bowl_ok"] = (normal & ~nonbowler_w) | ((wide | noball) & ~nonbowler_w)
say(f"run-outs / non-bowler wickets on wides, no-balls and byes (excluded for both players): {int((nonbowler_w & (wide | noball | byelb)).sum())}")
say("\nattribution of chase balls:\n" + c.kind.value_counts().to_string())
say(f"batter-attributed {c.bat_ok.mean():.3f} | bowler-attributed {c.bowl_ok.mean():.3f}")

# ---- centring + standardising by pressure bin x competition group (computed on clean legal balls only)
edges = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .975, .99, 1.0]
c["pbin"] = pd.cut(c.PI.fillna(0), edges, labels=False, include_lowest=True)
GROUP = ["comp_group", "pbin"]
PH = None if "--global-bins" in sys.argv else next((a for a in sys.argv if a.startswith("--phase-aware")), "--phase-aware")   # default: phase-aware bins; --global-bins = old pooled bins   # --phase-aware: 13 pressure bins per phase (sparse cells merged); --phase-aware=5: five bins per phase
if PH:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from binning import assign_bins
    c["pbin"] = assign_bins(c, ["comp_group", "phase"], c.PI.fillna(0).values, edges if PH == "--phase-aware" else [0, .5, .75, .9, .95, 1.0], 400)
    GROUP = ["comp_group", "phase", "pbin"]; say(f"phase-aware standardisation: {c.groupby(GROUP).ngroups} cells ({PH})")
clean = c[normal & ~nonbowler_w]
ref = clean.groupby(GROUP).VA_raw.agg(bin_mean="mean", bin_sd="std", bin_n="size").reset_index()
say("\nbin reference (mean and SD of VA on clean legal balls), IPL / T20I:")
say(ref.pivot(index="pbin", columns="comp_group", values=["bin_mean", "bin_sd", "bin_n"]).round(4).to_string() if not PH else ref.round(4).to_string())
c = c.merge(ref, on=GROUP, how="left")
leg_like = c.legal                                                   # legal balls are centred; wides/no-balls are not (expected = 0)
c["VA_c"] = np.where(leg_like, c.VA_raw - c.bin_mean, c.VA_raw)
c["z_bat"] = c.VA_c / c.bin_sd                                       # batter credit in SD units
c["z_bowl"] = -c.VA_c / c.bin_sd                                     # bowler credit
c["VA_bat"] = c.VA_c; c["VA_bowl"] = -c.VA_c                         # raw-WP versions (sensitivity)

# ---- components: z = runs component + wicket/residual component (each shrunk by its OWN estimated reliability in stage 06)
c["rcls"] = c.bat_runs.where(c.bat_runs.isin([0, 1, 2, 3, 4, 6]), 4)
base_nw = c[normal & ~nonbowler_w]
lk = base_nw.groupby(GROUP + ["rcls"]).z_bat.mean().rename("lk").reset_index()
lk2 = base_nw.groupby(["comp_group", "rcls"]).z_bat.mean().rename("lk2").reset_index()
c = c.merge(lk, on=GROUP + ["rcls"], how="left").merge(lk2, on=["comp_group", "rcls"], how="left")
c["lk"] = c.lk.fillna(c.lk2)
c["zr_bat"] = np.where(c.bat_ok, c.lk, 0.0); c["zw_bat"] = c.z_bat - c.zr_bat
c["zr_bowl"] = np.where(c.legal, -c.lk, c.z_bowl); c["zw_bowl"] = c.z_bowl - c.zr_bowl      # wides/no-balls: all of it is 'runs' (discipline)
say("\ncomponents: runs-value (league mean z for the runs scored, by pressure bin) + wicket/residual = z; SD of each component on pressure balls:")
pm = (c.PI > .5) & c.bat_ok
say("  batter: runs %.2f | wicket/residual %.2f | z %.2f" % (c[pm].zr_bat.std(), c[pm].zw_bat.std(), c[pm].z_bat.std()))

# ---- pressure weight, opposition weight, baseline flag
c["pw"] = np.clip((c.PI - 0.5) / 0.5, 0, None) ** 2                  # continuous weight from the percentile; 0 at/below the median
c["pe"] = np.where(c.PI.notna(), c.PI.fillna(0) ** 2, 0.0)                    # ESTIMATION weight of the pressure level: every chase ball, weighted by PI^2 (see stage 06)
c["opp_w"] = np.where(c.tier == "T20I_Assoc", 0.7, 1.0)             # scale the WEIGHT for associate-nation T20Is, never the VA value
c["baseline_ball"] = (c.PI < 0.5) & c.wp.between(0.10, 0.90)         # ordinary competitive chase balls (disjoint from pw>0 balls)
assert not (c.baseline_ball & (c.pw > 0)).any()
say(f"\nballs with pressure weight>0: {(c.pw>0).sum():,} | high-pressure (top 10%): {c.high_pressure.sum():,} | baseline balls: {c.baseline_ball.sum():,}")
say(f"associate-tier balls: {(c.tier=='T20I_Assoc').sum():,} ({(c.tier=='T20I_Assoc').mean():.1%}), weight 0.7")

# ---- checks: centred VA mean ~0 in every bin; z has unit SD
chk = c[normal & ~nonbowler_w].groupby(GROUP).agg(mean_VAc=("VA_c", "mean"), sd_z=("z_bat", "std"))
say("\ncheck after centring (clean legal balls): max |mean VA_c| = %.5f, SD of z in each bin = %.2f..%.2f" % (chk.mean_VAc.abs().max(), chk.sd_z.min(), chk.sd_z.max()))
say("mean z of wides/no-balls charged to bowlers by pressure band (should be strongly negative for the bowler, rising with pressure):")
w = c[wide | noball]; say(w.groupby(pd.cut(w.PI, [0, .5, .75, .9, 1.0])).z_bowl.agg(["size", "mean"]).round(2).to_string())
say("sanity (high-pressure legal balls): mean z_bat by runs scored / wicket")
h = c[c.bat_ok & (c.PI >= .9)]
say(h.groupby(np.where(h.wicket_fell == 1, "wicket", h.bat_runs.astype(int).astype(str))).z_bat.agg(["size", "mean"]).round(2).to_string())

keep = ["match_id", "innings_ball_num", "comp_group", "comp_name", "season", "tier", "match_date", "over_no", "phase", "striker_id", "striker_name",
        "bowler_id", "bowler_name", "batting_team_id", "bowling_team_id", "kind", "bat_ok", "bowl_ok", "bat_runs", "wicket_fell", "bowler_wicket",
        "pbin", "wp", "wp_next", "leverage", "li", "PI", "pw", "pe", "opp_w", "baseline_ball", "high_pressure", "extreme_pressure", "VA_raw", "VA_c", "z_bat", "z_bowl",
        "VA_bat", "VA_bowl", "zr_bat", "zw_bat", "zr_bowl", "zw_bowl", "batting_won", "knock_out"]
c[keep].to_csv(os.path.join(T, "va_chase.csv.gz"), index=False, float_format="%.6g")
open(os.path.join(LOG, "05_value_added.log"), "w").write("\n".join(log))
