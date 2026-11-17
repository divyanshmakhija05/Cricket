"""
Day 2 (FINAL): win-probability model -> leverage-based Pressure Index -> validation -> robustness.
Usage:  python 02_pressure_chase.py <out_dir> [--no-robust]      (reads <out_dir>/tables/ball_state.csv.gz)
Outputs: <out_dir>/tables/pressure_chase.csv.gz   (the pressure file used downstream)
         <out_dir>/charts/02_chase_validation.png, <out_dir>/logs/02_pressure_chase.log

IDEA
  Pressure on a delivery = how much that ball can swing the match.
  1. Win-probability (WP) model per innings: logistic regression on engineered state features plus venue par
     (shrunk leave-one-match-out venue average). Out-of-fold BY MATCH, so no match is scored by a model that saw it.
  2. For every ball, simulate the 7 outcomes (0,1,2,3,4,6,wicket), re-score WP, weight by the league outcome mix
     of that innings/phase  ->  LEVERAGE = expected |change in WP|.
  3. Pressure Index = percentile of leverage among CHASE balls (innings 2). High pressure = top 10%, extreme = top 2.5%.
     LI = leverage / average chase leverage (continuous weight for the rating).
     Innings 1 is NOT scored as pressure: its WP model is weak (AUC ~0.69), so ball-level leverage there reflects
     model extrapolation, not real match pressure (tested, see evidence/E3). First-innings balls still feed baselines.
  The index uses only the state BEFORE the ball, so it cannot leak the ball's own result.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

args = [a for a in sys.argv[1:] if not a.startswith("--")]
import os
OUT = args[0] if len(args) > 0 else "output"
SRC = f"{OUT}/tables/ball_state.csv.gz"
for _d in ("tables", "logs", "charts"): os.makedirs(f"{OUT}/{_d}", exist_ok=True)
ROBUST = "--no-robust" not in sys.argv
STATE_AWARE = "--static-mix" not in sys.argv   # default: outcome probabilities p_k come from a multinomial model of the match STATE (out-of-fold); --static-mix = old innings x phase average
log = []
def say(s=""): print(s); log.append(str(s))

# ------------------------------------------------------------------ data
b = pd.read_csv(SRC, low_memory=False)
n_inn = b.groupby("match_id").innings.nunique()
keep = (b.comp_group.isin(["IPL", "T20I"]) & ~b.no_result & ~b.dls & ~b.tied
        & b.match_id.isin(n_inn[n_inn == 2].index) & b.batting_won.notna())
k = b[keep].reset_index(drop=True)
k["ipl"] = (k.comp_group == "IPL") * 1.0
k["assoc"] = (k.tier == "T20I_Assoc") * 1.0          # associate-nation T20Is score and chase differently
k["par"] = k.venue_par; k["mu_c"] = k.tier_mean
mt = k.groupby("match_id").tier.first()
say(f"modelling set: {len(k):,} balls, {k.match_id.nunique()} matches {mt.value_counts().to_dict()}")
say("(excluded: no-result, DLS/rain-affected, tied/Super-Over, one-innings, OTHER_T20)")

OUTS = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (6, 0), (0, 1)]       # (bat runs, wicket)

# ------------------------------------------------------ outcome weights
leg = k[k.legal].copy()
leg["oc"] = np.where(leg.wicket_fell == 1, 6, leg.bat_runs.clip(0, 6).replace({5: 4})
                     .map({0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5}).fillna(4)).astype(int)
W = {(i, p): np.bincount(g.oc, minlength=7) / len(g) for (i, p), g in leg.groupby(["innings", "phase"])}
k["oc"] = np.nan; k.loc[leg.index, "oc"] = leg.oc.values
say("\noutcome mix by innings/phase (0,1,2,3,4,6,W):")
for kk, v in W.items(): say(f"  inn{kk[0]} {kk[1]:<9} " + " ".join(f"{x:.3f}" for x in v))

# ---------------------------------------------------- feature builders
def Rfit(d1):
    """Expected remaining runs R(balls_left, wickets_in_hand) from first innings (smooth cubic lookup)."""
    pf = PolynomialFeatures(3)
    m = LinearRegression().fit(pf.fit_transform(np.column_stack([d1.balls_left / 120, d1.wkts_in_hand / 10])),
                               d1.inn1_total - d1.score_before)
    return lambda bl, wk: np.clip(m.predict(pf.transform(np.column_stack([np.asarray(bl) / 120, np.asarray(wk) / 10]))), 0, None)

def F1(sc, lb, bl, wk, ipl, par, mu_c, R):
    crr = (sc + 8.0 * 12 / 6) / (lb + 12) * 6                  # run rate shrunk toward 8 rpo: no ball-1 artifact
    shat = sc + R(bl, wk) * par / mu_c                         # venue-adjusted projected final score
    return np.column_stack([bl, wk, sc, crr, sc + bl * crr / 6, wk * bl, sc * wk, np.sqrt(bl), ipl, shat - par, par])

def F2(rn, bl, wk, tg, ipl, par):
    rn = np.clip(rn, 1, None); rrr = np.clip(rn / np.maximum(bl, 1) * 6, 0, 36)
    return np.column_stack([bl, wk, rn, rrr, np.log(rn), np.sqrt(bl), bl * wk, rn / (wk + 1), rrr * wk,
                            np.log1p(rrr), ipl, tg, rrr - par / 20, par])          # rrr - par/20 = RRR_adj

mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))

def FO(inn, rn, sc, lb, bl, wk, ipl, par):
    """Features of the OUTCOME model (state-aware outcome probabilities): what the batting side is trying to do and can afford."""
    if inn == 2:
        rrr = np.clip(np.clip(rn, 1, None) / np.maximum(bl, 1) * 6, 0, 36)
        return np.column_stack([bl / 120, wk / 10, rrr / 10, (rrr - par / 20) / 10, rrr * wk / 100, bl * wk / 1200, np.log1p(rrr), (rrr / 10) ** 2, ipl, par / 100])
    crr = (sc + 8.0 * 12 / 6) / (lb + 12) * 6
    return np.column_stack([bl / 120, wk / 10, crr / 10, bl * wk / 1200, (crr / 10) ** 2, sc / 100, ipl, par / 100])
mo = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1.0))

def run_models(k, innings=(1, 2), noise_seed=None, s_noise=3.0):
    out = pd.DataFrame(np.nan, index=k.index, columns=["wp", "leverage", "exp_dwp", "D_bat", "D_bowl"])
    metrics = {}
    if noise_seed is not None:                               # stress test: corrupt score estimates by ~s_noise runs
        rng = np.random.default_rng(noise_seed); mids = k.match_id.unique()
        e_t = pd.Series(rng.normal(0, s_noise, len(mids)), index=mids)    # error in first-innings total
        e_f = pd.Series(rng.normal(0, s_noise, len(mids)), index=mids)    # error in chase final score
    d1_all = k[k.innings == 1]
    for inn in innings:
        idx = np.where(k.innings == inn)[0]; d = k.iloc[idx]; y = d.batting_won.values
        bl, wk, sc, lb = d.balls_left.values, d.wkts_in_hand.values, d.score_before.values, d.legal_balls_before.values
        tg, ipl, par, muc = d.target.values, d[["ipl", "assoc"]].values, d.par.values, d.mu_c.values
        if noise_seed is not None:
            et = d.match_id.map(e_t).values; ef = d.match_id.map(e_f).values
            if inn == 2: tg = tg + et; sc = sc + ef * lb / 120          # errors accumulate through the innings
            else: sc = sc + et * lb / 120
        wp = np.zeros(len(d)); lev = np.zeros(len(d)); ed = np.zeros(len(d)); dbat = np.zeros(len(d)); dbowl = np.zeros(len(d))
        wts = np.vstack([W[(inn, p)] for p in d.phase.values]); oc = d.oc.values; wts_static = wts.copy()
        for tr, te in match_folds(d.match_id.values):
            if STATE_AWARE:                                           # out-of-fold, state-aware outcome probabilities p_k(state)
                fo = lambda ii: FO(inn, (tg[ii] - sc[ii]), sc[ii], lb[ii], bl[ii], wk[ii], ipl[ii], par[ii])
                trl = tr[~np.isnan(oc[tr])]; mdo = mo().fit(fo(trl), oc[trl].astype(int)); wts[te] = mdo.predict_proba(fo(te))
            R = Rfit(d1_all[d1_all.match_id.isin(set(d.match_id.values[tr]))]) if inn == 1 else None
            Xtr = F1(sc[tr], lb[tr], bl[tr], wk[tr], ipl[tr], par[tr], muc[tr], R) if inn == 1 else \
                  F2(tg[tr] - sc[tr], bl[tr], wk[tr], tg[tr], ipl[tr], par[tr])
            Xte = F1(sc[te], lb[te], bl[te], wk[te], ipl[te], par[te], muc[te], R) if inn == 1 else \
                  F2(tg[te] - sc[te], bl[te], wk[te], tg[te], ipl[te], par[te])
            mdl = mk().fit(Xtr, y[tr]); wp[te] = mdl.predict_proba(Xte)[:, 1]
            cf = np.zeros((len(te), 7))
            for j, (r, w) in enumerate(OUTS):
                bl2 = np.maximum(bl[te] - 1, 0); wk2 = wk[te] - w
                if inn == 2:
                    rn2 = (tg[te] - sc[te]) - r
                    p = mdl.predict_proba(F2(rn2, bl2, wk2, tg[te], ipl[te], par[te]))[:, 1]
                    p = np.where(rn2 <= 0, 1.0, np.where((wk2 <= 0) | (bl2 <= 0), 0.0, p))   # chase won / lost
                else:
                    p = mdl.predict_proba(F1(sc[te] + r, lb[te] + 1, bl2, wk2, ipl[te], par[te], muc[te], R))[:, 1]
                cf[:, j] = p
            dlt = cf - wp[te][:, None]
            lev[te] = (wts[te] * np.abs(dlt)).sum(1)                    # expected |change| = LEVERAGE
            ed[te] = (wts[te] * dlt).sum(1)                             # expected signed change (the 'expected' term of VA)
            dbat[te] = wp[te] - np.minimum(cf[:, 0], cf[:, 6])          # drop from the worst batter outcome (dot / wicket)
            dbowl[te] = np.maximum(cf[:, 4], cf[:, 5]) - wp[te]         # rise from the worst bowler outcome (4 / 6)
        out.iloc[idx] = np.column_stack([wp, lev, ed, dbat, dbowl])
        if STATE_AWARE:
            lg = ~np.isnan(oc); yl = oc[lg].astype(int)
            metrics[inn] = {}
            ll_static = -np.log(np.clip(wts_static[lg][np.arange(lg.sum()), yl], 1e-9, 1)).mean(); ll_state = -np.log(np.clip(wts[lg][np.arange(lg.sum()), yl], 1e-9, 1)).mean()
            say(f"  outcome model innings {inn} (out-of-fold, legal balls): log-loss static phase mix {ll_static:.4f} -> state-aware {ll_state:.4f}")
        metrics[inn] = dict(n=len(y), ll=log_loss(y, np.clip(wp, 1e-4, 1 - 1e-4)), auc=roc_auc_score(y, wp),
                            brier=brier_score_loss(y, wp), ll_base=log_loss(y, np.full(len(y), y.mean())))
    return out, metrics

# --------------------------------------------------------- base run
base, metrics = run_models(k)
for c_ in base.columns: k[c_] = base[c_].values
say("\nWIN-PROBABILITY MODEL (5-fold, grouped by match; every number is out-of-fold)")
for inn, m_ in metrics.items():
    say(f"  innings {inn}: n={m_['n']:,} | logloss {m_['ll']:.4f}, AUC {m_['auc']:.3f}, Brier {m_['brier']:.4f} | no-skill logloss {m_['ll_base']:.4f}")

# ---------------------------------------------- Pressure Index (chases)
ch = k.innings == 2
k["pressure_pct"] = np.nan; k["li"] = np.nan
k.loc[ch, "pressure_pct"] = k.loc[ch, "leverage"].rank(pct=True)
k.loc[ch, "li"] = k.loc[ch, "leverage"] / k.loc[ch, "leverage"].mean()
k["high_pressure"] = (k.pressure_pct >= 0.90).astype(int)
k["extreme_pressure"] = (k.pressure_pct >= 0.975).astype(int)
rn = np.clip(k.target - k.score_before, 1, None); rrr = np.clip(rn / k.balls_left.clip(lower=1) * 6, 0, 36)
k["rrr_adj"] = np.where(ch, rrr - k.par / 20, np.nan)
k["S"] = np.where(ch, np.maximum(0, k.rrr_adj) + 1.5 * (k.wkts_in_hand <= 4), np.nan)   # state stress (for sensitivity checks)
say("\nchase leverage thresholds: median %.4f | top10%% >= %.4f (LI %.1f) | top2.5%% >= %.4f (LI %.1f)" % (
    k.loc[ch, "leverage"].median(), k.loc[ch & (k.pressure_pct >= .9), "leverage"].min(),
    k.loc[ch & (k.pressure_pct >= .9), "li"].min(), k.loc[ch & (k.pressure_pct >= .975), "leverage"].min(),
    k.loc[ch & (k.pressure_pct >= .975), "li"].min()))
say("high-pressure share of chase balls by phase:"); say(k[ch].groupby("phase").high_pressure.mean().round(3).to_string())

# ---------------------------------------------------------- validation
L = k[k.legal & ch].copy()
BANDS = ["0-50", "50-75", "75-90", "90-95", "95-97.5", "97.5-100"]
L["bin"] = pd.cut(L.pressure_pct, [0, .5, .75, .9, .95, .975, 1.0], labels=BANDS)
def table(g):
    return pd.Series(dict(balls=len(g), SR=100 * g.bat_runs.sum() / max(g.ball_faced.sum(), 1),
        wkt_per_100=100 * g.wicket_fell.mean(), boundary_pct=100 * (g.is_four | g.is_six).mean(),
        dot_pct=100 * ((g.total_runs == 0) & (g.wicket_fell == 0)).mean(),
        economy=6 * g.bowler_runs.sum() / len(g), median_wp=g.wp.median()))
say("\nVALIDATION: league-wide chase outcomes by pressure band (percentile of leverage)")
vb = L.groupby("bin", observed=True).apply(table, include_groups=False); say(vb.round(2).to_string())
say("\nSAME CHECK HELD WITHIN PHASE, competitive states only (0.10<=WP<=0.90): high (top 10%) vs ordinary (25th-75th pct)")
C = L[(L.wp >= 0.10) & (L.wp <= 0.90)].copy()
C["grp"] = np.where(C.pressure_pct >= 0.90, "high", np.where((C.pressure_pct >= .25) & (C.pressure_pct < .75), "ordinary", "other"))
say(C[C.grp != "other"].groupby(["phase", "grp"], observed=True).apply(table, include_groups=False).round(2).to_string())
say("(death overs have almost no 'ordinary' balls in a live chase: late in a close game nearly every ball is high-leverage)")
hi = L[L.high_pressure == 1]
say("\nmedian state of high-pressure chase balls:"); say(hi[["balls_left", "wkts_in_hand", "req_rr", "wp", "leverage"]].median().round(3).to_string())
say("phase mix of high-pressure chase balls: " + str(hi.phase.value_counts(normalize=True).round(2).to_dict()))
say(f"high-pressure chase balls: {len(hi):,} | extreme: {int(L.extreme_pressure.sum()):,}")

say("\nCALIBRATION CHECKS (chase balls): mean predicted WP vs actual win rate")
c2 = k[ch].copy(); c2["wk_bucket"] = pd.cut(c2.wkts_in_hand, [-1, 3, 6, 10], labels=["<=3 in hand", "4-6 in hand", "7-10 in hand"])
c2["bl_bucket"] = pd.cut(c2.balls_left, [-1, 12, 36, 72, 120], labels=["<=12 left", "13-36", "37-72", "73-120"])
for col in ("tier", "phase", "wk_bucket", "bl_bucket"):
    t = c2.groupby(col, observed=True).apply(lambda d: pd.Series(dict(n=len(d), predicted=d.wp.mean(), actual=d.batting_won.mean())), include_groups=False)
    say(f"  by {col}:"); say(t.round(3).to_string())
cut = pd.cut(c2.wp, np.linspace(0, 1, 11), include_lowest=True)
cal = c2.groupby(cut, observed=True).agg(p=("wp", "mean"), o=("batting_won", "mean"), n=("wp", "size"))
say(f"  expected calibration error (10 bins): {(cal.n * (cal.p - cal.o).abs()).sum() / cal.n.sum():.4f}")

# ------------------------------------------- robustness to the extras limitation
if ROBUST:
    say("\nROBUSTNESS: corrupt score/target estimates (random error, sd 3 runs per innings = about 2x the observed")
    say("error) and re-run the whole pipeline. Does the pressure ranking survive?")
    cb = k.index[ch]; base_pct = k.loc[cb, "leverage"].rank(pct=True)
    for sd in (1, 2, 3):
        nz, _ = run_models(k, innings=(2,), noise_seed=sd)
        lv = nz.loc[cb, "leverage"]; pc = lv.rank(pct=True)
        a, bb = set(cb[base_pct >= .9]), set(cb[pc >= .9])
        say(f"  seed {sd}: Spearman(leverage) = {spearmanr(k.loc[cb, 'leverage'], lv).correlation:.3f} | "
            f"top-10% overlap (Jaccard) = {len(a & bb) / len(a | bb):.2f} | share of original top-10% kept = {len(a & bb) / len(a):.2f}")

# ------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0, 0].plot([0, 1], [0, 1], "--", c="grey"); ax[0, 0].plot(cal.p, cal.o, "o-")
ax[0, 0].set(title="Calibration, chase win-probability model (out-of-fold)", xlabel="predicted win prob", ylabel="actual win rate")
ex = k[(k.match_id == 14099) & ch]
ax[0, 1].plot(ex.legal_balls_before, ex.wp); ax[0, 1].set_ylabel("win probability"); ax[0, 1].set_xlabel("legal balls bowled")
a2 = ax[0, 1].twinx(); a2.fill_between(ex.legal_balls_before, ex.leverage, alpha=.3, color="C3"); a2.set_ylabel("leverage (red)")
ax[0, 1].set_title("Example: IPL 2022 RCB chase (match 14099)")
x = range(len(vb))
ax[1, 0].bar(x, vb.wkt_per_100, color="C3"); ax[1, 0].set(title="Wickets per 100 balls by pressure band (chases)", ylabel="wickets / 100 balls")
ax[1, 1].bar(x, vb.SR, color="C0"); ax[1, 1].set(title="Strike rate by pressure band (chases)", ylabel="strike rate")
for a in (ax[1, 0], ax[1, 1]):
    a.set_xticks(list(x)); a.set_xticklabels(BANDS, rotation=20); a.set_xlabel("Pressure Index percentile band")
plt.tight_layout(); plt.savefig(f"{OUT}/charts/02_chase_validation.png", dpi=130)

cols = ["match_id", "innings", "innings_ball_num", "wp", "leverage", "exp_dwp", "D_bat", "D_bowl", "S", "rrr_adj",
        "pressure_pct", "li", "high_pressure", "extreme_pressure"]
k[cols].to_csv(f"{OUT}/tables/pressure_chase.csv.gz", index=False)
open(f"{OUT}/logs/02_pressure_chase.log", "w").write("\n".join(log))
say(f"\nsaved tables/pressure_chase.csv.gz ({len(k):,} rows), charts/02_chase_validation.png, logs/02_pressure_chase.log")
