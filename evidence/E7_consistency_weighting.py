"""
E7: volume-weighted match consistency. A pressure appearance is a match in which a player's total high-pressure weight W_pm >= 1. Equal-vote hit / disaster rates let a 2-ball
cameo count as much as an 18-ball match-winning knock. Candidates (all shrunk by empirical Bayes with a data-estimated prior; variance from the effective number of votes):
  C0  current: unweighted hit (appearance mean z > 0) and disaster (appearance mean z in the league's worst 20%)
  C1  each appearance weighted by sqrt(W_pm)
  C2  disaster defined by TOTAL match impact S_pm = sum(w z) in the worst 20% of league appearances (unweighted), hit as C0
  C3  as C2, each appearance weighted by sqrt(W_pm)
Score: repeatability of the combined measure (hit - disaster) between random halves of the matches (players eligible in the full sample, >= 2 appearances per half).
"""
import sys, os, warnings, numpy as np, pandas as pd, importlib.util
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "..", "output"); T = os.path.join(O, "tables")
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
sp = importlib.util.spec_from_file_location("r06", os.path.join(HERE, "..", "src", "06_player_ratings.py")); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M)
log = []
def say(s=""): print(s); log.append(str(s))
va = M.load_ball_table(T); v = va[va.kind != "first"]; mids = v.match_id.unique(); rng = np.random.default_rng(21)
ROLE = {"bat": ("striker_id", "bat_ok", "z_bat"), "bowl": ("bowler_id", "bowl_ok", "z_bowl")}
def app_tab(vv, idc, okc, zc):
    P = vv[vv[okc] & (vv.pw > 0)]; w = P.pw * P.opp_w; g = P.assign(w_=w, s_=w * P[zc]).groupby([idc, "match_id"]).agg(W=("w_", "sum"), S=("s_", "sum")).reset_index()
    g = g[g.W >= 1.0].copy(); g["a"] = g.S / g.W; g["v"] = np.sqrt(g.W); return g
def wprop(g, idc, flag, weighted):
    wt = g.v if weighted else pd.Series(1.0, index=g.index); num = (wt * g[flag]).groupby(g[idc]).sum(); den = wt.groupby(g[idc]).sum(); neff = den ** 2 / (wt ** 2).groupby(g[idc]).sum()
    return num / den, neff
def measures(g, qa, qs, el, minapp):
    idc = g.columns[0]; g = g[g[idc].isin(el)]; n = g.groupby(idc).size(); g = g[g[idc].isin(n[n >= minapp].index)].copy(); out = {}
    g["hit"] = (g.a > 0) * 1.0; g["dis_a"] = (g.a <= qa) * 1.0; g["dis_s"] = (g.S <= qs) * 1.0
    def shr(flag, weighted):
        p, ne = wprop(g, idc, flag, weighted); p0 = g[flag].mean(); vv = p0 * (1 - p0) / ne; return pd.Series(M.eb(p.values, vv.values, np.ones(len(p), bool))[0], index=p.index)
    out["C0 current (equal votes)"] = shr("hit", False) - shr("dis_a", False)
    out["C1 sqrt(W)-weighted votes"] = shr("hit", True) - shr("dis_a", True)
    out["C2 disaster = total-impact bottom 20%"] = shr("hit", False) - shr("dis_s", False)
    out["C3 total-impact disaster + sqrt(W) votes"] = shr("hit", True) - shr("dis_s", True)
    return out
say(f"E7 VOLUME-WEIGHTED CONSISTENCY - repeatability across random match halves ({NS} splits)")
for role, (idc, okc, zc) in ROLE.items():
    full = M.build(va, role, verbose=False); el = set(full.index[full.elig]); rs = {}
    for s in range(NS):
        h = pd.Series(rng.integers(0, 2, len(mids)), index=mids); hs = []
        for k in (0, 1):
            g = app_tab(v[v.match_id.map(h) == k], idc, okc, zc); hs.append(measures(g, g.a.quantile(.2), g.S.quantile(.2), el, 2))
        for k in hs[0]:
            com = hs[0][k].index.intersection(hs[1][k].index)
            if len(com) >= 30 and hs[0][k].loc[com].std() > 0 and hs[1][k].loc[com].std() > 0: rs.setdefault(k, []).append(np.corrcoef(hs[0][k].loc[com], hs[1][k].loc[com])[0, 1])
    say(f"  {role}: " + " | ".join(f"{k.split(' ')[0]} {np.mean(a):+.2f}" for k, a in rs.items()))
    for k, a in rs.items(): say(f"      {k:<44} repeatability {np.mean(a):+.3f} (SD over splits {np.std(a):.2f}, {len(a)} usable splits)")
open(os.path.join(O, "logs", "E7_consistency_weighting.log"), "w").write("\n".join(log))
