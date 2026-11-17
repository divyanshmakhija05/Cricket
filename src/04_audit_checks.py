"""
Automated audit of the ball-state table and (if present) the pressure file.
Usage: python 04_audit_checks.py <data_dir> <out_dir>
Every check prints PASS / WARN / FAIL. Output is also saved to <out_dir>/logs/04_audit_checks.log
Checks the ball-state table always, and the chase / first-innings pressure files when they exist.
"""
import sys, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
D = sys.argv[1] if len(sys.argv) > 1 else "data"
O = sys.argv[2] if len(sys.argv) > 2 else "output"
log, tally = [], {"PASS": 0, "WARN": 0, "FAIL": 0}
def chk(name, ok, detail="", warn=False):
    st = "PASS" if ok else ("WARN" if warn else "FAIL"); tally[st] += 1
    line = f"[{st}] {name}" + (f" -- {detail}" if detail else ""); print(line); log.append(line)

b = pd.read_csv(f"{O}/tables/ball_state.csv.gz", low_memory=False)
m = pd.read_csv(f"{D}/matches.csv").set_index("match_id")
key = ["match_id", "innings", "innings_ball_num"]
print("=== DATA TABLE CHECKS ==="); log.append("=== DATA TABLE CHECKS ===")

chk("keys unique (match, innings, ball)", not b.duplicated(key).any())
g = b.groupby(["match_id", "innings"])
chk("score, wickets, legal-ball count never decrease within an innings",
    (g.score_before.diff().dropna() >= -1e-9).all() and (g.wkts_before.diff().dropna() >= 0).all()
    and (g.legal_balls_before.diff().dropna() >= 0).all())
a = g.agg(w=("wicket_fell", "sum"), lg=("legal", "sum"), mb=("max_balls", "first"), mo=("over_no", "max"))
chk("no innings has >10 wickets, >max legal balls, or >20 overs", ((a.w > 10) | (a.lg > a.mb) | (a.mo > 20)).sum() == 0)
chk("batting team != bowling team on every ball", (b.batting_team_id == b.bowling_team_id).sum() == 0)
n_ss = int((b.striker_id == b.non_striker_id).sum()); chk("striker != non-striker", n_ss == 0, f"{n_ss} anomalous row(s) in raw data", warn=True)
chk("each bowler belongs to one team per match", (b.groupby(["match_id", "bowler_id"]).bowling_team_id.nunique() > 1).sum() == 0)

o = b.groupby(["match_id", "innings", "over_no"]).agg(leg=("legal", "sum"), last=("innings_ball_num", "max"))
o = o.join(b.groupby(["match_id", "innings"]).innings_ball_num.max().rename("il"), on=["match_id", "innings"]); c = o[o["last"] != o.il]
chk("complete overs have exactly 6 legal balls (>=99.9%)", (c.leg == 6).mean() >= 0.999, f"{(c.leg == 6).mean():.4%} of {len(c):,} overs")

# labels vs result text
res = m.result.fillna("")
lab = b[(b.innings == 2) & b.batting_won.notna()].groupby("match_id").agg(won=("batting_won", "first"), dls=("dls", "first"), tied=("tied", "first"))
j = lab.join(res).query("not dls and not tied")
wk = j[j.result.str.contains(r"\d+\s*(?:wickets?|wkts?)\b", case=False)]
rn = j[j.result.str.contains(r"\bwon\b", case=False) & j.result.str.contains(r"\d+\s*runs?\b", case=False) & ~j.index.isin(wk.index)]
chk("chase label = 1 when result says won by wickets (>=99%)", (wk.won == 1).mean() >= 0.99, f"{(wk.won == 1).mean():.3%} of {len(wk)}")
chk("chase label = 0 when result says won by runs (>=99%)", (rn.won == 0).mean() >= 0.99, f"{(rn.won == 0).mean():.3%} of {len(rn)}")
l1 = b[(b.innings == 1) & b.batting_won.notna()].groupby("match_id").batting_won.first()
chk("innings-1 label is the complement of innings-2 label", ((l1 + lab.won.reindex(l1.index)).dropna() == 1).all())
chk("DLS-like text in result is always flagged dls", not (m.result.fillna("").str.contains(r"DLS|D/L|VJD", case=False)
    & ~m.index.isin(b[b.dls].match_id.unique()) & m.index.isin(b.match_id.unique())).any())

c2 = b[(b.innings == 2) & b.legal]
sh = (c2.runs_needed <= 0).mean()
chk("chase balls already past the target BEFORE the ball are rare (<0.5%)", sh < 0.005, f"{sh:.3%} -- extras-estimation noise", warn=True)
chk("venue par has no missing values", b.venue_par.notna().all())
mm = b.groupby("match_id").agg(inn1=("inn1_total", "first"), par=("venue_par", "first"), mu=("tier_mean", "first"), t=("tier", "first"))
mm = mm[mm.t.isin(["IPL", "T20I_Full", "T20I_Assoc"])]
r1 = np.sqrt(((mm.inn1 - mm.mu) ** 2).mean()); r2 = np.sqrt(((mm.inn1 - mm.par) ** 2).mean())
chk("shrunk venue par predicts first-innings total at least as well as the tier mean", r2 <= r1 + 0.05, f"RMSE {r2:.2f} vs {r1:.2f}")

# extras reconciliation
rr = m.result.fillna("")
mr = rr.str.extract(r"(\d+)\s*runs?\b", flags=2)[0].astype(float)
ok = (rr.str.contains(r"\bwon\b", case=False) & mr.notna() & ~rr.str.contains(r"wkts?|wickets?|DLS|D/L|VJD|tied|super over", case=False))
mm2 = pd.DataFrame({"margin": mr[ok]}).join(b.groupby("match_id").agg(d=("dls", "first"), t=("tied", "first"), n=("no_result", "first")), how="inner")
mm2 = mm2[~mm2.d & ~mm2.t & ~mm2.n]
tot = b.groupby(["match_id", "innings"]).total_runs.sum().unstack()
e = ((tot[1] - tot[2]).reindex(mm2.index).abs() - mm2.margin).dropna()
chk("reconstructed run margins match official margins (mean abs error < 3 runs)", e.abs().mean() < 3, f"MAE {e.abs().mean():.2f} runs over {len(e)} matches; within 2 runs: {(e.abs() <= 2).mean():.1%}")

pf = f"{O}/tables/pressure_chase.csv.gz"
if os.path.exists(pf):
    print("\n=== PRESSURE FILE CHECKS ==="); log.append("\n=== PRESSURE FILE CHECKS ===")
    p = pd.read_csv(pf)
    chk("pressure keys unique and present in ball_state", not p.duplicated(key).any() and len(p.merge(b[key], on=key)) == len(p))
    chk("win probability in [0,1] with no missing values", p.wp.between(0, 1).all() and p.wp.notna().all())
    chk("leverage non-negative with no missing values", (p.leverage >= 0).all() and p.leverage.notna().all())
    ch = p[p.innings == 2]
    chk("every chase ball has a pressure percentile", ch.pressure_pct.notna().all())
    chk("innings-1 balls carry no pressure score (by design)", p[p.innings == 1].pressure_pct.isna().all() and p[p.innings == 1].high_pressure.sum() == 0)
    chk("high-pressure share of chase balls is ~10%", abs(ch.high_pressure.mean() - 0.10) < 0.01, f"{ch.high_pressure.mean():.3%}")
    k = b.merge(p, on=key); k = k[(k.innings == 2) & k.batting_won.notna()]
    f1 = k.sort_values(key).groupby("match_id").head(1)
    chk("chase WP at ball 1 is close to the actual chase win rate (within 0.03)", abs(f1.wp.mean() - f1.batting_won.mean()) < 0.03,
        f"{f1.wp.mean():.3f} vs {f1.batting_won.mean():.3f}")
    cal = k.groupby(pd.cut(k.wp, np.linspace(0, 1, 11), include_lowest=True), observed=True).agg(p=("wp", "mean"), o=("batting_won", "mean"), n=("wp", "size"))
    ece = (cal.n * (cal.p - cal.o).abs()).sum() / cal.n.sum()
    chk("expected calibration error < 0.01", ece < 0.01, f"ECE {ece:.4f}")
    for t, d in k.groupby("tier"):
        chk(f"calibration within tier {t} (|pred - actual| < 0.03)", abs(d.wp.mean() - d.batting_won.mean()) < 0.03, f"{d.wp.mean():.3f} vs {d.batting_won.mean():.3f}")
else:
    print("\n(tables/pressure_chase.csv.gz not found: run 02_pressure_chase.py, then re-run this audit)")

p1f = f"{O}/tables/pressure_first_innings.csv.gz"
if os.path.exists(p1f):
    print("\n=== FIRST-INNINGS PRESSURE FILE CHECKS ==="); log.append("\n=== FIRST-INNINGS PRESSURE FILE CHECKS ===")
    q = pd.read_csv(p1f)
    chk("first-innings keys unique and present in ball_state", not q.duplicated(key).any() and len(q.merge(b[key], on=key)) == len(q))
    chk("first-innings file contains innings 1 only", (q.innings == 1).all())
    chk("first-innings win probability in [0,1]", q.wp_inn1.between(0, 1).all())
    lp = q.pressure_pct_inn1.dropna()
    chk("first-innings pressure percentile in [0,1]; high-pressure share ~10% of legal balls", lp.between(0, 1).all() and abs((lp >= .9).mean() - .10) < .01, f"{(lp >= .9).mean():.3%}")

s = f"\nSUMMARY: {tally['PASS']} PASS, {tally['WARN']} WARN, {tally['FAIL']} FAIL"; print(s); log.append(s)
open(f"{O}/logs/04_audit_checks.log", "w").write("\n".join(log))
