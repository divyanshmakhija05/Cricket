"""
Day 2c: test the 'v2' proposals against the data instead of assuming they help.
  (1) shrunk leave-one-match-out venue par     (2) innings-1 par-projection bridge + innings-2 RRR_adj
  (3) asymmetric composite Pressure Index  PI = .60 Pct(L) + .25 Pct(D) + .15 Pct(S)
Usage: python E3_venue_par_and_composite.py <out_dir>   (reads tables/ball_state, writes logs/E3_venue_par_and_composite.log)
EVIDENCE E3: tests venue par, the first-innings bridge and the composite index against the data.
The final chase pipeline is src/02_pressure_chase.py.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold   # (no longer used: see match_folds)
def match_folds(groups, n=5):
    """Deterministic, library-independent cross-validation folds: a match always goes to fold (match_id mod n),
    so every ball of a match stays together and results are identical on any machine / library version."""
    f = np.asarray(groups) % n
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(n)]
from sklearn.metrics import log_loss, roc_auc_score

import os
OUT = sys.argv[1] if len(sys.argv) > 1 else "output"; BS = f"{OUT}/tables/ball_state.csv.gz"
os.makedirs(f"{OUT}/logs", exist_ok=True)
log = []
def say(s=""): print(s); log.append(str(s))

b = pd.read_csv(BS, low_memory=False)
n_inn = b.groupby("match_id").innings.nunique()
keep = (b.comp_group.isin(["IPL", "T20I"]) & ~b.no_result & ~b.dls & ~b.tied & b.match_id.isin(n_inn[n_inn == 2].index)
        & b.batting_won.notna())
k = b[keep].reset_index(drop=True)
k["ipl"] = (k.comp_group == "IPL") * 1.0

# ------------------------------------------------------------------ tiers and venue par (built in 01)
k["par"] = k.venue_par; k["mu_c"] = k.tier_mean
mt = k.groupby("match_id").agg(venue=("venue_id", "first"), tier=("tier", "first"), inn1=("inn1_total", "first"),
                               par=("par", "first"), mu_c=("mu_c", "first"))
say("matches by tier: " + str(mt.tier.value_counts().to_dict()))
# (1) does shrinkage help?  compare predictors of a match's first-innings total
g = mt.groupby(["tier", "venue"]).inn1.agg(["sum", "count"]); mt = mt.join(g, on=["tier", "venue"])
nv = mt["count"] - 1
mt["par_raw"] = np.where(nv > 0, (mt["sum"] - mt.inn1) / nv.replace(0, np.nan), mt.mu_c)
rm = lambda x: np.sqrt(np.nanmean((mt.inn1 - x) ** 2))
say("(1) VENUE PAR: how well does each predict a match's first-innings total? (RMSE in runs, lower is better)")
say(f"  tier mean only        : {rm(mt.mu_c):.2f}")
say(f"  raw leave-one-out avg : {rm(mt.par_raw):.2f}   <- overfits small venues")
say(f"  shrunk (k=7) LOO par  : {rm(mt.par):.2f}")

# --------------------------------------------------- feature builders
def Rfit(d1):
    pf = PolynomialFeatures(3)
    m = LinearRegression().fit(pf.fit_transform(np.column_stack([d1.balls_left / 120, d1.wkts_in_hand / 10])),
                               d1.inn1_total - d1.score_before)
    return lambda bl, wk: np.clip(m.predict(pf.transform(np.column_stack([np.asarray(bl) / 120, np.asarray(wk) / 10]))), 0, None)

def F1(sc, lb, bl, wk, ipl, par, mu_c, R, new):
    crr = (sc + 8.0 * 12 / 6) / (lb + 12) * 6
    cols = [bl, wk, sc, crr, sc + bl * crr / 6, wk * bl, sc * wk, np.sqrt(bl), ipl]
    if new:
        shat = sc + R(bl, wk) * par / mu_c                    # venue-adjusted projected final score
        cols += [shat - par, par]
    return np.column_stack(cols)

def F2(rn, bl, wk, tg, ipl, par, new):
    rn = np.clip(rn, 1, None); rrr = np.clip(rn / np.maximum(bl, 1) * 6, 0, 36)
    cols = [bl, wk, rn, rrr, np.log(rn), np.sqrt(bl), bl * wk, rn / (wk + 1), rrr * wk, np.log1p(rrr), ipl, tg]
    if new: cols += [rrr - par / 20, par]                     # RRR_adj and venue par
    return np.column_stack(cols)

mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
OUTS = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (6, 0), (0, 1)]
leg = k[k.legal].copy()
leg["oc"] = np.where(leg.wicket_fell == 1, 6, leg.bat_runs.clip(0, 6).replace({5: 4}).map({0:0,1:1,2:2,3:3,4:4,6:5}).fillna(4)).astype(int)
W = {(i, p): np.bincount(g.oc, minlength=7) / len(g) for (i, p), g in leg.groupby(["innings", "phase"])}

# ------------------------------------- (2) CV comparison + new-model outputs
k["wp2"] = np.nan; k["lev2"] = np.nan; k["exp_dwp"] = np.nan; k["D_bat"] = np.nan; k["D_bowl"] = np.nan
res = {}
for inn in (1, 2):
    idx = np.where(k.innings == inn)[0]; d = k.iloc[idx]; y = d.batting_won.values
    bl, wk, sc, lb = d.balls_left.values, d.wkts_in_hand.values, d.score_before.values, d.legal_balls_before.values
    tg, ipl, par, muc = d.target.values, d.ipl.values, d.par.values, d.mu_c.values
    ps_old = np.zeros(len(d)); ps_new = np.zeros(len(d))
    wp = np.zeros(len(d)); lev = np.zeros(len(d)); ed = np.zeros(len(d)); dbat = np.zeros(len(d)); dbowl = np.zeros(len(d))
    wts = np.vstack([W[(inn, p)] for p in d.phase.values])
    d1_all = k[k.innings == 1]
    for tr, te in match_folds(d.match_id.values):
        tr_m = set(d.match_id.values[tr]); R = Rfit(d1_all[d1_all.match_id.isin(tr_m)])
        if inn == 1:
            f_old = lambda ix, s=sc, l=lb, b_=bl, w=wk: F1(s[ix], l[ix], b_[ix], w[ix], ipl[ix], par[ix], muc[ix], R, False)
            f_new = lambda ix: F1(sc[ix], lb[ix], bl[ix], wk[ix], ipl[ix], par[ix], muc[ix], R, True)
        else:
            rn = tg - sc
            f_old = lambda ix: F2(rn[ix], bl[ix], wk[ix], tg[ix], ipl[ix], par[ix], False)
            f_new = lambda ix: F2(rn[ix], bl[ix], wk[ix], tg[ix], ipl[ix], par[ix], True)
        m_old = mk().fit(f_old(tr), y[tr]); ps_old[te] = m_old.predict_proba(f_old(te))[:, 1]
        m_new = mk().fit(f_new(tr), y[tr]); ps_new[te] = m_new.predict_proba(f_new(te))[:, 1]
        wp[te] = ps_new[te]
        cf = np.zeros((len(te), 7))
        for j, (r, w) in enumerate(OUTS):
            bl2 = np.maximum(bl[te] - 1, 0); wk2 = wk[te] - w
            if inn == 2:
                rn2 = (tg[te] - sc[te]) - r
                p = m_new.predict_proba(F2(rn2, bl2, wk2, tg[te], ipl[te], par[te], True))[:, 1]
                p = np.where(rn2 <= 0, 1.0, np.where((wk2 <= 0) | (bl2 <= 0), 0.0, p))
            else:
                p = m_new.predict_proba(F1(sc[te] + r, lb[te] + 1, bl2, wk2, ipl[te], par[te], muc[te], R, True))[:, 1]
            cf[:, j] = p
        dlt = cf - wp[te][:, None]
        lev[te] = (wts[te] * np.abs(dlt)).sum(1); ed[te] = (wts[te] * dlt).sum(1)
        dbat[te] = wp[te] - np.minimum(cf[:, 0], cf[:, 6])          # drop caused by worst batter outcome (dot / wicket)
        dbowl[te] = np.maximum(cf[:, 4], cf[:, 5]) - wp[te]         # rise caused by worst bowler outcome (4 / 6)
    k.loc[k.index[idx], ["wp2", "lev2", "exp_dwp", "D_bat", "D_bowl"]] = np.column_stack([wp, lev, ed, dbat, dbowl])
    res[inn] = {nm: (log_loss(y, np.clip(p, 1e-4, 1 - 1e-4)), roc_auc_score(y, p)) for nm, p in (("old", ps_old), ("new", ps_new))}
say("\n(2) WIN-PROBABILITY MODELS, out-of-fold by match (logloss lower / AUC higher is better)")
for inn in (1, 2):
    o, n = res[inn]["old"], res[inn]["new"]
    say(f"  innings {inn}: current features logloss {o[0]:.4f} AUC {o[1]:.3f}  ->  +venue par features logloss {n[0]:.4f} AUC {n[1]:.3f}")

# innings-1: is leverage now meaningful?
a = k[(k.innings == 1) & k.legal].copy()
a["pct"] = a.lev2.rank(pct=True)
hi = a[a.pct >= .9]
say(f"\n    innings-1 leverage (new model): median {a.lev2.median():.4f}, p90 {a.lev2.quantile(.9):.4f}, max {a.lev2.max():.4f}")
say(f"    top-10% innings-1 balls: median balls_left {hi.balls_left.median():.0f}, median wkts in hand {hi.wkts_in_hand.median():.0f}, phase mix {hi.phase.value_counts(normalize=True).round(2).to_dict()}")
a["band"] = pd.cut(a.pct, [0, .5, .75, .9, .95, 1.0], labels=["0-50", "50-75", "75-90", "90-95", "95-100"])
tb = lambda g: pd.Series(dict(balls=len(g), SR=100 * g.bat_runs.sum() / max(g.ball_faced.sum(), 1), wk100=100 * g.wicket_fell.mean()))
say(a.groupby("band", observed=True).apply(tb, include_groups=False).round(2).to_string())

# ------------------------------------ (3) composite PI on chase balls
c = k[(k.innings == 2) & k.legal & (k.balls_left > 0)].copy()
rn = np.clip(c.target - c.score_before, 1, None); rrr = np.clip(rn / c.balls_left * 6, 0, 36)
c["rrr_adj"] = rrr - c.par / 20
c["S"] = np.maximum(0, c.rrr_adj) + 1.5 * (c.wkts_in_hand <= 4)
pct = lambda x: x.rank(pct=True)
pL, pDb, pDw, pS = pct(c.lev2), pct(c.D_bat), pct(c.D_bowl), pct(c.S)
c["PI_L"] = pL
c["PI_LD"] = .7 * pL + .3 * pDb
c["PI_LS"] = .85 * pL + .15 * pS
c["PI_full_bat"] = .60 * pL + .25 * pDb + .15 * pS
c["PI_full_bowl"] = .60 * pL + .25 * pDw + .15 * pS
say(f"\n(3) COMPOSITE PRESSURE INDEX (chase balls: {len(c):,}).  Correlations between components (Spearman):")
say(f"    L vs D_bat {spearmanr(c.lev2, c.D_bat).correlation:.2f} | L vs D_bowl {spearmanr(c.lev2, c.D_bowl).correlation:.2f} | L vs S {spearmanr(c.lev2, c.S).correlation:.2f}")
rows = []
for nm in ("PI_L", "PI_LD", "PI_LS", "PI_full_bat", "PI_full_bowl"):
    x = c[nm]; pr = x.rank(pct=True); top = c[pr >= .9]; ext = c[pr >= .975]; mid = c[(pr >= .5) & (pr < .75)]
    rows.append(dict(index=nm, rho_vs_L=round(spearmanr(c.lev2, x).correlation, 2),
                     top10_median_WP=round(top.wp2.median(), 2), top10_pct_WP_below_10=round(100 * (top.wp2 < .1).mean(), 1),
                     wk100_top2p5=round(100 * ext.wicket_fell.mean(), 1), wk100_mid=round(100 * mid.wicket_fell.mean(), 1),
                     SR_top2p5=round(100 * ext.bat_runs.sum() / ext.ball_faced.sum()),
                     econ_top2p5=round(6 * ext.bowler_runs.sum() / len(ext), 2), econ_mid=round(6 * mid.bowler_runs.sum() / len(mid), 2)))
say(pd.DataFrame(rows).to_string(index=False))
say("    (WP below 10% = already-lost chase. A good PRESSURE index should keep this near 0.)")

open(f"{OUT}/logs/E3_venue_par_and_composite.log", "w").write("\n".join(log))
say("\nsaved logs/E3_venue_par_and_composite.log")
