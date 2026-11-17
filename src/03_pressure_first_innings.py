"""
Stage 03: STRUCTURAL first-innings pressure (secondary layer).
Usage: python 03_pressure_first_innings.py <out_dir>      (reads <out_dir>/tables/ball_state.csv.gz)
Outputs: <out_dir>/tables/pressure_first_innings.csv.gz, <out_dir>/logs/03_pressure_first_innings.log

Why: a generic win-probability model has no target in innings 1 and overreacts to the first balls (see evidence/E3).
Here win probability is built from three interpretable pieces, all fitted out-of-fold by match:
  f(T)       : P(batting-first team wins | final total T minus venue par)        (logistic, 1 feature)
  R(b,w)     : expected remaining runs given balls left b and wickets in hand w   (cubic lookup, venue-scaled)
  s(b,w)     : uncertainty (sd) of the final total given the state                (shrinks as the innings runs out)
  WP(state)  = E[ f(T) ],  T ~ Normal( score + R , s )       (closed-form logistic-normal approximation)
Leverage = expected |change in WP| over the 7 possible outcomes of the next ball, exactly as for chases.
Pressure is then the percentile of leverage WITHIN innings 1.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GroupKFold   # (no longer used: see match_folds)
def match_folds(groups, n=5):
    """Deterministic, library-independent cross-validation folds: a match always goes to fold (match_id mod n),
    so every ball of a match stays together and results are identical on any machine / library version."""
    f = np.asarray(groups) % n
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(n)]
from sklearn.metrics import log_loss, roc_auc_score

import os
OUT = sys.argv[1] if len(sys.argv) > 1 else "output"
SRC = f"{OUT}/tables/ball_state.csv.gz"
for _d in ("tables", "logs"): os.makedirs(f"{OUT}/{_d}", exist_ok=True)
log = []
def say(s=""): print(s); log.append(str(s))

b = pd.read_csv(SRC, low_memory=False)
n_inn = b.groupby("match_id").innings.nunique()
keep = (b.comp_group.isin(["IPL", "T20I"]) & ~b.no_result & ~b.dls & ~b.tied
        & b.match_id.isin(n_inn[n_inn == 2].index) & b.batting_won.notna())
d = b[keep & (b.innings == 1)].reset_index(drop=True)
d["par"] = d.venue_par; d["mu"] = d.tier_mean
say(f"first-innings balls: {len(d):,} from {d.match_id.nunique()} matches")

OUTS = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (6, 0), (0, 1)]
leg = d[d.legal].copy()
leg["oc"] = np.where(leg.wicket_fell == 1, 6, leg.bat_runs.clip(0, 6).replace({5: 4})
                     .map({0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5}).fillna(4)).astype(int)
W = {p: np.bincount(g.oc, minlength=7) / len(g) for p, g in leg.groupby("phase")}

P3, P2 = PolynomialFeatures(3), PolynomialFeatures(2)
f3 = lambda bl, wk: P3.fit_transform(np.column_stack([np.asarray(bl) / 120, np.asarray(wk) / 10]))
f2 = lambda bl, wk: P2.fit_transform(np.column_stack([np.asarray(bl) / 120, np.asarray(wk) / 10]))

class Struct:
    def fit(self, dtr):
        sc_ = dtr.par / dtr.mu
        fut = dtr.inn1_total - dtr.score_before
        self.R = LinearRegression().fit(f3(dtr.balls_left, dtr.wkts_in_hand), fut / sc_)
        res = fut - np.clip(self.R.predict(f3(dtr.balls_left, dtr.wkts_in_hand)), 0, None) * sc_
        self.V = LinearRegression().fit(f2(dtr.balls_left, dtr.wkts_in_hand), res ** 2)
        mm = dtr.groupby("match_id").agg(tot=("inn1_total", "first"), par=("par", "first"), y=("batting_won", "first"))
        lr = LogisticRegression(C=1e6, max_iter=1000).fit((mm.tot - mm.par).values.reshape(-1, 1), mm.y.values)
        self.a, self.k = lr.intercept_[0], lr.coef_[0][0]
        return self
    def parts(self, sc, bl, wk, par, mu):
        end = (wk <= 0) | (bl <= 0)
        r = np.clip(self.R.predict(f3(bl, wk)), 0, None) * (par / mu)
        r = np.where(end, 0.0, r)
        s2 = np.where(end, 0.25, np.clip(self.V.predict(f2(bl, wk)), 4.0, None))
        return sc + r, s2
    def wp(self, sc, bl, wk, par, mu):
        m, s2 = self.parts(sc, bl, wk, par, mu)
        z = (self.a + self.k * (m - par)) / np.sqrt(1 + np.pi * self.k ** 2 * s2 / 8)
        return 1 / (1 + np.exp(-z))

y = d.batting_won.values
bl, wk, sc, par, mu = (d.balls_left.values, d.wkts_in_hand.values, d.score_before.values, d.par.values, d.mu.values)
wts = np.vstack([W[p] for p in d.phase.values])
wp = np.zeros(len(d)); lev = np.zeros(len(d)); proj_gap = np.zeros(len(d)); sd = np.zeros(len(d))
curves = []
for tr, te in match_folds(d.match_id.values):
    M = Struct().fit(d.iloc[tr]); curves.append((M.a, M.k))
    wp[te] = M.wp(sc[te], bl[te], wk[te], par[te], mu[te])
    m, s2 = M.parts(sc[te], bl[te], wk[te], par[te], mu[te]); proj_gap[te] = m - par[te]; sd[te] = np.sqrt(s2)
    cf = np.zeros((len(te), 7))
    for j, (r, w) in enumerate(OUTS):
        cf[:, j] = M.wp(sc[te] + r, np.maximum(bl[te] - 1, 0), wk[te] - w, par[te], mu[te])
    lev[te] = (wts[te] * np.abs(cf - wp[te][:, None])).sum(1)
a_, k_ = np.mean(curves, axis=0)
say(f"\nWIN-CURVE f(T): P(win) = logistic({a_:.2f} + {k_:.3f} x (total - venue par)); a 10-run beat of par is worth about {100 * (1 / (1 + np.exp(-(a_ + k_ * 10))) - 1 / (1 + np.exp(-a_))):.1f} points of win probability")
say(f"STRUCTURAL WIN-PROBABILITY, out-of-fold by match: logloss {log_loss(y, np.clip(wp, 1e-4, 1 - 1e-4)):.4f} | AUC {roc_auc_score(y, wp):.3f}"
    f"   (generic logistic model in 02: logloss 0.6318, AUC 0.696)")
d["wp"], d["lev"], d["gap"], d["sd"] = wp, lev, proj_gap, sd

say("\nSANITY: does leverage now behave like pressure? (legal balls)")
L = d[d.legal].copy(); L["pct"] = L.lev.rank(pct=True)
L["over_grp"] = pd.cut(L.over_no, [0, 4, 8, 12, 16, 18, 20], labels=["1-4", "5-8", "9-12", "13-16", "17-18", "19-20"])
say("mean leverage by over group:"); say(L.groupby("over_grp", observed=True).lev.mean().round(4).to_string())
L["near"] = pd.cut(L.gap.abs(), [-1, 5, 10, 20, 40, 400], labels=["<5", "5-10", "10-20", "20-40", ">40"])
say("mean leverage by |projected total - par| (runs) in overs 13-20:")
say(L[L.over_no >= 13].groupby("near", observed=True).lev.mean().round(4).to_string())

hi = L[L.pct >= .9]
say(f"\ntop-10% first-innings balls: {len(hi):,} | median balls_left {hi.balls_left.median():.0f} | median wickets in hand {hi.wkts_in_hand.median():.0f}"
    f" | median |proj - par| {hi.gap.abs().median():.1f} runs")
say("phase mix: " + str(hi.phase.value_counts(normalize=True).round(2).to_dict()))
say("by over (share of high-pressure balls): " + str(hi.over_no.value_counts(normalize=True).sort_index().round(3).to_dict()))

def tb(g):
    return pd.Series(dict(balls=len(g), SR=100 * g.bat_runs.sum() / max(g.ball_faced.sum(), 1), wkt_per_100=100 * g.wicket_fell.mean(),
                          boundary_pct=100 * (g.is_four | g.is_six).mean(), dot_pct=100 * ((g.total_runs == 0) & (g.wicket_fell == 0)).mean(),
                          economy=6 * g.bowler_runs.sum() / len(g)))
BANDS = ["0-50", "50-75", "75-90", "90-95", "95-97.5", "97.5-100"]
L["band"] = pd.cut(L.pct, [0, .5, .75, .9, .95, .975, 1.0], labels=BANDS)
say("\nVALIDATION: league-wide first-innings outcomes by pressure band")
say(L.groupby("band", observed=True).apply(tb, include_groups=False).round(2).to_string())

say("\nWITHIN PHASE, competitive states (0.25<=WP<=0.75): high (top 10%) vs ordinary (25th-75th pct)")
C = L[(L.wp >= .25) & (L.wp <= .75)].copy()
C["grp"] = np.where(C.pct >= .9, "high", np.where((C.pct >= .25) & (C.pct < .75), "ordinary", "other"))
say(C[C.grp != "other"].groupby(["phase", "grp"], observed=True).apply(tb, include_groups=False).round(2).to_string())

col = L[(L.phase == "powerplay") & (L.wkts_before >= 3)]
say(f"\nCOLLAPSE CHECK: powerplay balls with 3+ wickets already down: n={len(col):,}, mean pressure percentile {col.pct.mean():.2f} (0.50 = average)")
say(f"                 overs 16-20 with projection within 10 runs of par: mean pressure percentile {L[(L.over_no >= 16) & (L.gap.abs() <= 10)].pct.mean():.2f}")

d["pressure_pct_inn1"] = np.nan; d.loc[d.legal, "pressure_pct_inn1"] = L.pct.reindex(d.index[d.legal]).values
d["high_pressure_inn1"] = (d.pressure_pct_inn1 >= 0.9).astype(int)
d[["match_id", "innings", "innings_ball_num", "wp", "lev", "gap", "pressure_pct_inn1", "high_pressure_inn1"]] \
    .rename(columns={"wp": "wp_inn1", "lev": "leverage_inn1", "gap": "proj_minus_par"}) \
    .to_csv(f"{OUT}/tables/pressure_first_innings.csv.gz", index=False)
open(f"{OUT}/logs/03_pressure_first_innings.log", "w").write("\n".join(log))
say("\nsaved tables/pressure_first_innings.csv.gz, logs/03_pressure_first_innings.log")
