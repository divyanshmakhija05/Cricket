"""
Day 2b: RUN-RATE pressure definitions (simple, transparent) and comparison with the leverage-based index.
Usage: python E2_runrate_alternatives.py <out_dir>   (reads tables/ball_state + tables/pressure_chase)
EVIDENCE E2: run-rate pressure definitions compared with the leverage index.

Definitions (chase balls only):
  RR-gap   : required rate minus current rate (smoothed)           -> "am I scoring fast enough?"
  RR-ratio : required rate / what a team in this position normally scores over the remaining balls
             (league table by balls-left x wickets-in-hand, from complete FIRST innings)
             -> 1.0 = par chase, >1 = harder than normal, <1 = comfortable
  RR-tight : closeness-aware: |log(RR-ratio)| small AND few balls left  (a close, late chase)
"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import os
OUT = sys.argv[1] if len(sys.argv) > 1 else "output"
BS = f"{OUT}/tables/ball_state.csv.gz"; PS = f"{OUT}/tables/pressure_chase.csv.gz"
for _d in ("tables", "logs", "charts"): os.makedirs(f"{OUT}/{_d}", exist_ok=True)
log = []
def say(s=""): print(s); log.append(str(s))

b = pd.read_csv(BS, low_memory=False)
p = pd.read_csv(PS)
n_inn = b.groupby("match_id").innings.nunique()
keep = (b.comp_group.isin(["IPL", "T20I"]) & ~b.no_result & ~b.dls & ~b.tied & b.match_id.isin(n_inn[n_inn == 2].index)
        & b.batting_won.notna())
k = b[keep].merge(p, on=["match_id", "innings", "innings_ball_num"], how="left").reset_index(drop=True)

# ---- expected remaining scoring rate table from complete first innings -------------
f = k[k.innings == 1].copy()
f["future"] = f.inn1_total - f.score_before
f["ob"] = (f.balls_left // 6).clip(0, 19); f["wk"] = f.wkts_in_hand.clip(1, 10)
tab = f.groupby(["ob", "wk"]).agg(fut=("future", "mean"), bl=("balls_left", "mean"), n=("future", "size"))
tab["rate"] = tab.fut / tab.bl * 6
coarse = f.groupby("ob").apply(lambda g: g.future.mean() / g.balls_left.mean() * 6, include_groups=False)
def exp_rate(bl, wk):
    ob = np.clip(bl // 6, 0, 19).astype(int); w = np.clip(wk, 1, 10).astype(int)
    out = np.empty(len(bl))
    for i, (o, ww) in enumerate(zip(ob, w)):
        r = tab.rate.get((o, ww), np.nan); n = tab.n.get((o, ww), 0)
        out[i] = r if (n >= 80 and not np.isnan(r)) else coarse.get(o, 8.0)
    return out

# ---- chase balls -------------------------------------------------------------------
c = k[(k.innings == 2) & k.legal & (k.balls_left > 0)].copy()
c["rrr"] = c.req_rr.clip(0, 36)
c["crr_s"] = (c.score_before + 8.0 * 12 / 6) / (c.legal_balls_before + 12) * 6
c["rr_gap"] = c.rrr - c.crr_s
c["exp_rate"] = exp_rate(c.balls_left.values, c.wkts_in_hand.values)
c["rr_ratio"] = c.rrr / c.exp_rate
# closeness + lateness: high when ratio is near 1 and the end is near
c["rr_tight"] = np.exp(-np.abs(np.log(c.rr_ratio.clip(0.05, 20))) / 0.35) * (1 - c.balls_left / 120) ** 1.0
for col in ("rr_gap", "rr_ratio", "rr_tight"):
    c[col + "_pct"] = c[col].rank(pct=True)
c["lev_pct"] = c.pressure_pct

say(f"chase balls analysed: {len(c):,}")
say("\nAGREEMENT with leverage index (Spearman rank correlation; overlap of top-10% sets):")
top = lambda s: set(c.index[s >= 0.9])
T = {"leverage": top(c.lev_pct), "rr_gap": top(c.rr_gap_pct), "rr_ratio": top(c.rr_ratio_pct), "rr_tight": top(c.rr_tight_pct)}
for nm in ("rr_gap", "rr_ratio", "rr_tight"):
    rho = spearmanr(c.lev_pct, c[nm + "_pct"]).correlation
    j = len(T["leverage"] & T[nm]) / len(T["leverage"] | T[nm])
    say(f"  {nm:9s} rho={rho:.3f} | Jaccard top10% overlap={j:.2f} | share of leverage-top10% also in {nm} top10%: "
        f"{len(T['leverage'] & T[nm]) / len(T['leverage']):.2f}")

def table(g):
    return pd.Series(dict(balls=len(g), SR=100 * g.bat_runs.sum() / max(g.ball_faced.sum(), 1),
        wkt_per_100=100 * g.wicket_fell.mean(), boundary_pct=100 * (g.is_four | g.is_six).mean(),
        dot_pct=100 * ((g.total_runs == 0) & (g.wicket_fell == 0)).mean(),
        economy=6 * g.bowler_runs.sum() / len(g)))
BANDS = ["0-50", "50-75", "75-90", "90-95", "95-97.5", "97.5-100"]
res = {}
for nm in ("lev", "rr_gap", "rr_ratio", "rr_tight"):
    c["bin"] = pd.cut(c[nm + "_pct"], [0, .5, .75, .9, .95, .975, 1.0], labels=BANDS)
    res[nm] = c.groupby("bin", observed=True).apply(table, include_groups=False)
    say(f"\nVALIDATION BY BAND: {nm}")
    say(res[nm].round(2).to_string())

say("\nWITHIN-PHASE, competitive states only (0.10<=WP<=0.90): top-10% vs 25th-75th percentile, per definition")
C = c[(c.wp >= .10) & (c.wp <= .90)]
rows = []
for nm in ("lev", "rr_gap", "rr_ratio", "rr_tight"):
    pc = C[nm + "_pct"]
    for ph in ("middle", "death"):
        h = C[(pc >= .9) & (C.phase == ph)]; o = C[(pc >= .25) & (pc < .75) & (C.phase == ph)]
        if len(o) < 300 or len(h) < 300:
            continue
        rows.append(dict(defn=nm, phase=ph, n_high=len(h), n_ord=len(o),
                         SR_high=100 * h.bat_runs.sum() / h.ball_faced.sum(), SR_ord=100 * o.bat_runs.sum() / o.ball_faced.sum(),
                         wk100_high=100 * h.wicket_fell.mean(), wk100_ord=100 * o.wicket_fell.mean(),
                         bnd_high=100 * (h.is_four | h.is_six).mean(), bnd_ord=100 * (o.is_four | o.is_six).mean()))
say(pd.DataFrame(rows).round(2).to_string(index=False))

# ---- key statistic: how well does each index separate wicket risk? (spread top band vs 50-75 band) ----
say("\nSEPARATION SCORE (top 2.5% band vs 50-75 band): higher = index isolates riskier cricket")
for nm, r in res.items():
    say(f"  {nm:9s} wickets/100: {r.wkt_per_100.iloc[-1]:.1f} vs {r.wkt_per_100.iloc[1]:.1f} "
        f"(x{r.wkt_per_100.iloc[-1] / r.wkt_per_100.iloc[1]:.2f}) | SR: {r.SR.iloc[-1]:.0f} vs {r.SR.iloc[1]:.0f}")

# ---- figure ------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
lab = {"lev": "leverage (win-prob)", "rr_gap": "RR gap (req - current)", "rr_ratio": "RR ratio (req / par)", "rr_tight": "RR tight (close & late)"}
for nm, r in res.items():
    ax[0].plot(range(6), r.wkt_per_100.values, "o-", label=lab[nm]); ax[1].plot(range(6), r.SR.values, "o-", label=lab[nm])
for a, yl in zip(ax, ("wickets per 100 balls", "strike rate")):
    a.set_xticks(range(6)); a.set_xticklabels(BANDS, rotation=20); a.set_xlabel("pressure percentile band"); a.set_ylabel(yl); a.legend(fontsize=8)
ax[0].set_title("Wicket rate by pressure band: four definitions"); ax[1].set_title("Strike rate by pressure band: four definitions")
plt.tight_layout(); plt.savefig(f"{OUT}/charts/E2_pressure_definitions.png", dpi=130)

c[["match_id", "innings_ball_num", "rrr", "rr_gap", "rr_ratio", "rr_tight", "rr_gap_pct", "rr_ratio_pct", "rr_tight_pct"]] \
    .to_csv(f"{OUT}/tables/alt_runrate_pressure.csv.gz", index=False)
open(f"{OUT}/logs/E2_runrate_alternatives.log", "w").write("\n".join(log))
say("\nsaved tables/alt_runrate_pressure.csv.gz, charts/E2_pressure_definitions.png, logs/E2_runrate_alternatives.log")
