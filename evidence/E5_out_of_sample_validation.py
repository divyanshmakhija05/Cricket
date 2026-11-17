"""
E5: out-of-sample choice of the estimator (how should a player's pressure level be estimated?).
Design: split the matches at random in two halves (60 repeats). Estimate each player's pressure level on the TRAIN half with each candidate method, and score how well
the estimate predicts the same player's HIGH-pressure runs value (weight = pressure weight, PI>0.5) on the held-out TEST half.
Score = Pearson correlation between prediction and held-out value across players (ranking quality; a method that rescales values is not penalised),
reported as the mean and as the % of splits in which the method beats the previous primary (the high-pressure-only estimate).
Candidates: estimation window (which chase balls inform the estimate, weight PI^power starting at PI=lo) and a general-ability prior (Fay-Herriot regression on a
player's runs-above-expected on ~100k non-pressure balls). The first-innings layer is tested in E6 (calibrated weight).
"""
import sys, os, warnings, numpy as np, pandas as pd, importlib.util
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "..", "output"); T = os.path.join(O, "tables")
sp = importlib.util.spec_from_file_location("r06", os.path.join(HERE, "..", "src", "06_player_ratings.py")); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M)
log = []
def say(s=""): print(s); log.append(str(s))
v = pd.read_csv(os.path.join(T, "va_chase.csv.gz"), low_memory=False)
b = pd.read_csv(os.path.join(T, "ball_state.csv.gz"), low_memory=False)
f1 = pd.read_csv(os.path.join(T, "pressure_first_innings.csv.gz"), usecols=["match_id", "innings", "innings_ball_num"]); pc = pd.read_csv(os.path.join(T, "pressure_chase.csv.gz"), usecols=["match_id", "innings", "innings_ball_num"])
ROLE = {"bat": ("striker_id", "bat_ok", "zr_bat"), "bowl": ("bowler_id", "bowl_ok", "zr_bowl")}
pw_ = lambda pi, p=2.0, lo=.5: np.clip((pi - lo) / (1 - lo), 0, None) ** p

# --- general ability on non-pressure balls (runs above expected in the same competition x innings x phase x wickets-lost cell, venue-adjusted)
mset = pd.concat([f1[["match_id", "innings", "innings_ball_num"]], pc]).drop_duplicates()
d = b.merge(mset, on=["match_id", "innings", "innings_ball_num"]).merge(v[["match_id", "innings_ball_num", "PI", "wp"]].assign(innings=2), on=["match_id", "innings", "innings_ball_num"], how="left")
usual = (d.innings == 1) | ((d.innings == 2) & (d.PI <= .5) & d.wp.between(.10, .90))
d["wb"] = pd.cut(d.wkts_before, [-1, 1, 3, 5, 10], labels=False); vf = (d.venue_par / d.tier_mean).clip(.7, 1.4).fillna(1); d["normal"] = d.legal & (d.extra_runs == 0); cells = ["comp_group", "innings", "phase", "wb"]
d = d.join(d[usual & d.normal].groupby(cells).bat_runs.mean().rename("e_bat"), on=cells); d["e_bat"] *= vf
d = d.join(d[usual].groupby(cells).bowler_runs.mean().rename("e_bowl"), on=cells); d["e_bowl"] *= vf

def stats(P, idc, w, col): return M.layer_stats(P, idc, w, col)[0]
def fh(x, vv, use, cov):
    """Fay-Herriot: true level ~ a + b*cov + N(0, tau^2); REML for tau^2; returns posterior means."""
    X = np.column_stack([np.ones(len(x)), cov])
    def nll(th):
        t2 = np.exp(th[0]); w = 1 / (vv[use] + t2); Xu = X[use]; A = Xu.T @ (Xu * w[:, None]); be = np.linalg.solve(A, Xu.T @ (w * x[use])); r = x[use] - Xu @ be
        return 0.5 * (np.log(vv[use] + t2).sum() + (w * r * r).sum() + np.linalg.slogdet(A)[1])
    t2 = np.exp(minimize(nll, [np.log(.002)], method="Nelder-Mead").x[0]); w = 1 / (vv[use] + t2); Xu = X[use]
    be = np.linalg.solve(Xu.T @ (Xu * w[:, None]), Xu.T @ (w * x[use])); fit = X @ be; B = t2 / (t2 + vv); return fit + B * (x - fit)
def post(st, idx, cov=None):
    st = st.reindex(idx); use = ((st.m >= 4) & (st.W >= 12)).values & st.x.notna().values; x = st.x.fillna(0).values; vv = st.v.fillna(1e6).values
    return M.eb(x, vv, use)[0] if cov is None else fh(x, vv, use, cov)

CAND = [("previous primary: high-pressure balls only (PI>0.5, power 2)", .5, 2), ("window PI>0.4, power 2", .4, 2), ("window PI>0.25, power 2", .25, 2), ("window PI>0.1, power 2", .1, 2),
        ("ALL chase balls, weight PI^2  (adopted)", 0.0, 2), ("window PI>0.25, power 1", .25, 1), ("ALL chase balls, weight PI^3", 0.0, 3)]
mids = v.match_id.unique(); rng = np.random.default_rng(11); NS = 60
say(f"E5 OUT-OF-SAMPLE VALIDATION: {NS} random splits of the {len(mids)} matches; prediction of held-out high-pressure runs value; players with >=3 train matches and held-out weight >= 8\n")
for role, (idc, okc, zr) in ROLE.items():
    out = {}
    for s_ in range(NS):
        h = pd.Series(rng.integers(0, 2, len(mids)), index=mids); tr = v.match_id.map(h) == 0; te = ~tr
        Pt = v[te & v[okc] & (v.pw > 0)]; wt = Pt.pw * Pt.opp_w; yt = (wt * Pt[zr]).groupby(Pt[idc]).sum() / wt.groupby(Pt[idc]).sum(); Wt = wt.groupby(Pt[idc]).sum()
        Ptr = v[tr & v[okc] & (v.pw > 0)]; st0 = stats(Ptr, idc, Ptr.pw * Ptr.opp_w, zr); idx = st0.index; preds = {}
        for nm, lo, pp in CAND:
            Pe = v[tr & v[okc] & (v.PI > lo)]; preds[nm] = post(stats(Pe, idc, pw_(Pe.PI, pp, lo) * Pe.opp_w, zr), idx)
        dd = d[d.match_id.map(h) == 0]; us = (dd.innings == 1) | ((dd.innings == 2) & (dd.PI <= .5) & dd.wp.between(.1, .9))
        if role == "bat": U = dd[us & dd.normal]; Gx = (U.bat_runs - U.e_bat).groupby(U.striker_id).mean()
        else: U = dd[us]; Gx = -(U.bowler_runs - U.e_bowl).groupby(U.bowler_id).mean()
        cov = Gx.reindex(idx).fillna(Gx.mean()).values
        preds["general-ability prior (Fay-Herriot) on high-pressure balls only"] = post(st0, idx, cov)
        j = [i for i in idx.intersection(yt.index) if Wt[i] >= 8 and st0.loc[i, "m"] >= 3]; pos = pd.Index(idx).get_indexer(j); y = yt[j].values
        for k, p in preds.items(): out.setdefault(k, []).append(np.corrcoef(p[pos], y)[0, 1])
    base = np.array(out[CAND[0][0]]); say(f"{'BATTERS' if role=='bat' else 'BOWLERS'} ({len(j)} players in the last split)")
    rows = [(k, np.mean(vv_), np.mean(np.array(vv_) - base), np.mean(np.array(vv_) > base)) for k, vv_ in out.items()]
    say(pd.DataFrame(rows, columns=["estimator", "held-out corr", "gain vs previous", "better in % splits"]).assign(**{"better in % splits": lambda x: (100 * x["better in % splits"]).round(0).astype(int)}).round(3).to_string(index=False)); say()
say("Reading: widening the window (more chase balls, weighted by PI^power) helps consistently; a general-ability prior does not. (First-innings layer: see E6.)")
say("The window is chosen at the plateau of a simple rule (weight = PI^2 over all chase balls), not at a tuned maximum.")
open(os.path.join(O, "logs", "E5_out_of_sample_validation.log"), "w").write("\n".join(log))
