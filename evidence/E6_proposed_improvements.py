"""
E6: out-of-sample tests of five proposed improvements. Every test uses random splits of the matches (train half -> predict / score on the held-out half).
 (1) WICKET MODELLING   : REML / Paule-Mandel instead of DerSimonian-Laird; Poisson-Gamma conjugate model for wicket rates. Target: held-out high-pressure TOTAL z.
 (2) POOLED BASELINE    : ordinary first-innings balls added to the ordinary chase balls. Score: repeatability of baseline and uplift across halves.
 (3) CONTINUOUS CONSISTENCY: downside semi-variance of appearance scores vs the binary hit / disaster rates. Score: repeatability across halves.
 (4) STATE-AWARE OUTCOME MIX: tested inside stage 02 (see its log: out-of-fold log-loss of the outcome model, bias by pressure bin).
 (5) FIRST-INNINGS WEIGHT: lambda = BSS ratio vs 0 / 0.5. Target: held-out high-pressure runs value.
Adopted: (2), (4), (5).  Not adopted: (1) no gain (the Poisson-Gamma prior collapses: no detectable spread in wicket rates), (3) not more repeatable.
"""
import sys, os, warnings, numpy as np, pandas as pd, importlib.util
from scipy.optimize import minimize
from scipy.special import gammaln
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "..", "output"); T = os.path.join(O, "tables")
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
sp = importlib.util.spec_from_file_location("r06", os.path.join(HERE, "..", "src", "06_player_ratings.py")); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M)
log = []
def say(s=""): print(s); log.append(str(s))
va = M.load_ball_table(T); v = va[va.kind != "first"].copy(); d1 = va[va.kind == "first"].copy()
LAM = float(pd.read_csv(os.path.join(T, "layer_weight.csv")).lambda_first_innings[0])
edges = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .975, .99, 1.0]; v["pbin"] = pd.cut(v.PI.fillna(0), edges, labels=False, include_lowest=True)
v["wk_bat"] = ((v.wicket_fell == 1) & v.bat_ok).astype(float); v["wk_bowl"] = ((v.bowler_wicket == 1) & v.bowl_ok).astype(float)
ROLE = {"bat": ("striker_id", "bat_ok", "zr_bat", "zw_bat", "z_bat"), "bowl": ("bowler_id", "bowl_ok", "zr_bowl", "zw_bowl", "z_bowl")}
mids = v.match_id.unique(); stats = lambda P, idc, w, col: M.layer_stats(P, idc, w, col)[0]

def eb_method(x, vv, use, method="DL"):
    x = np.asarray(x, float); vv = np.asarray(vv, float); xs, vs = x[use], vv[use]
    if method == "DL": return M.eb(x, vv, use)[0]
    if method == "REML":
        def nll(th):
            t2 = np.exp(th[0]); w = 1 / (vs + t2); mu = (w * xs).sum() / w.sum(); return 0.5 * (np.log(vs + t2).sum() + (w * (xs - mu) ** 2).sum() + np.log(w.sum()))
        t2 = float(np.exp(minimize(nll, [np.log(1e-3)], method="Nelder-Mead").x[0]))
    else:                                                                    # Paule-Mandel: solve Q(tau2) = k - 1
        k = len(xs); Q = lambda t2: ((1 / (vs + t2)) * (xs - ((1 / (vs + t2)) * xs).sum() / (1 / (vs + t2)).sum()) ** 2).sum() - (k - 1)
        if Q(0.0) <= 0: t2 = 0.0
        else:
            lo, hi = 0.0, 1.0
            while Q(hi) > 0 and hi < 1e3: hi *= 2
            for _ in range(60):
                mid = (lo + hi) / 2; lo, hi = (mid, hi) if Q(mid) > 0 else (lo, mid)
            t2 = (lo + hi) / 2
    w = 1 / (vs + t2); mu = (w * xs).sum() / w.sum(); return mu + (t2 / (t2 + vv) if t2 > 0 else 0 * vv) * (x - mu)

# ------------------------------------------------------------------ (1) wicket modelling
say("(1) WICKET MODELLING - held-out correlation with the high-pressure TOTAL z (runs + wickets), %d splits" % NS)
def pg_alpha(O_, E_):
    nll = lambda th: -(gammaln(np.exp(th[0]) + O_) - gammaln(np.exp(th[0])) + np.exp(th[0]) * th[0] - (np.exp(th[0]) + O_) * np.log(np.exp(th[0]) + E_)).sum()
    return float(np.exp(minimize(nll, [np.log(20.)], method="Nelder-Mead").x[0]))
rng = np.random.default_rng(5)
for role, (idc, okc, zr, zw, zt) in ROLE.items():
    wk = "wk_" + role; ok = v[v[okc]]
    ref = ok.groupby(["comp_group", "pbin"]).agg(e=(wk, "mean")); zc = ok.groupby(["comp_group", "pbin", wk])[zt].mean().unstack()
    ref = ref.join((zc[1.0] - zc[0.0]).rename("c")).reset_index(); v2 = v.merge(ref, on=["comp_group", "pbin"], how="left"); res = {}
    for s in range(NS):
        h = pd.Series(rng.integers(0, 2, len(mids)), index=mids); trm = v2.match_id.map(h) == 0
        Pt = v2[~trm & v2[okc] & (v2.pw > 0)]; wt = Pt.pw * Pt.opp_w; yT = (wt * Pt[zt]).groupby(Pt[idc]).sum() / wt.groupby(Pt[idc]).sum(); Wt = wt.groupby(Pt[idc]).sum()
        Pe = v2[trm & v2[okc] & (v2.pe > 0)]; we = Pe.pe * Pe.opp_w; sr = stats(Pe, idc, we, zr); sw = stats(Pe, idc, we, zw).reindex(sr.index); idx = sr.index
        use = ((sr.m >= 4) & (sr.W >= 12)).values; xr, vr = sr.x.values, sr.v.values; xw, vw = sw.x.fillna(0).values, sw.v.fillna(1e6).values; preds = {}
        for meth in ["DL", "REML", "PM"]:
            preds[f"runs + wicket parts, tau by {meth}"] = eb_method(xr, vr, use, meth) + eb_method(xw, vw, use, meth)
        preds["runs part only (DL)"] = eb_method(xr, vr, use, "DL")
        g = Pe.assign(o=we * Pe[wk], e_=we * Pe.e, ec=we * Pe.e * Pe.c, w2e=we * we * Pe.e, w_=we).groupby(idc).agg(O=("o", "sum"), E=("e_", "sum"), EC=("ec", "sum"), W=("w_", "sum"), W2E=("w2e", "sum")).reindex(idx)
        kap = (g.E / g.W2E).values; O2, E2 = g.O.values * kap, g.E.values * kap; ok2 = use & (E2 > 0); al = pg_alpha(O2[ok2], E2[ok2]); th = (al + O2) / (al + E2)
        preds["runs (DL) + Poisson-Gamma wicket rate"] = eb_method(xr, vr, use, "DL") + np.nan_to_num((th - 1) * (g.EC / g.W).values)
        j = [i for i in idx.intersection(yT.index) if Wt[i] >= 8 and sr.loc[i, "m"] >= 3]; pos = pd.Index(idx).get_indexer(j)
        for k_, p in preds.items(): res.setdefault(k_, []).append((np.corrcoef(p[pos], yT[j].values)[0, 1], np.log10(al) if "Poisson" in k_ else np.nan))
    base = np.array(res["runs + wicket parts, tau by DL"])[:, 0]; say(f"  {role}")
    for k_, a_ in res.items():
        a = np.array(a_); dd = a[:, 0] - base
        say(f"    {k_:<42} corr {a[:,0].mean():+.3f} | vs DL {dd.mean():+.3f} (better in {np.mean(dd > 0):.0%})" + (f" | Poisson-Gamma prior strength alpha ~ 1e{np.nanmedian(a[:,1]):.0f} (alpha -> infinity = no spread in wicket rates)" if "Poisson" in k_ else ""))

# ------------------------------------------------------------------ (2) pooled baseline
say("\n(2) POOLED BASELINE - ordinary competitive chase balls only vs + ordinary first-innings balls (%d splits)" % NS)
rng = np.random.default_rng(3)
def ub(vh, d1h, idc, okc, zr, pool):
    Pb = vh[vh[okc] & (vh.pw > 0)]; Bb = vh[vh[okc] & vh.baseline_ball]
    if pool: Bb = pd.concat([Bb[[idc, "match_id", zr, "opp_w"]], d1h[d1h[okc] & d1h.baseline_ball][[idc, "match_id", zr, "opp_w"]]], ignore_index=True)
    return stats(Pb, idc, Pb.pw * Pb.opp_w, zr), stats(Bb, idc, Bb.opp_w, zr)
say(f"  ordinary balls: chase {int(v.baseline_ball.sum()):,} | first innings {int(d1.baseline_ball.sum()):,}")
for role, (idc, okc, zr, zw, zt) in ROLE.items():
    full = M.build(va, role, verbose=False); el = full.index[full.elig]
    for pool in [False, True]:
        pr, br = ub(v, d1, idc, okc, zr, pool); j = el.intersection(br.index).intersection(pr.index)
        eu = M.eb((pr.x - br.x).loc[j].values, (pr.v + br.v).loc[j].values, np.ones(len(j), bool)); ebb = M.eb(br.x.loc[j].values, br.v.loc[j].values, np.ones(len(j), bool))
        rU, rB = [], []
        for s in range(NS):
            h = pd.Series(rng.integers(0, 2, len(mids)), index=mids); o = [ub(v[v.match_id.map(h) == k_], d1[d1.match_id.map(h) == k_], idc, okc, zr, pool) for k_ in (0, 1)]
            com = [i for i in j if all(i in x.index and x.loc[i, "m"] >= 3 for pair in o for x in pair)]
            if len(com) < 30: continue
            ue = [M.eb((a.x - b_.x).loc[com].values, (a.v + b_.v).loc[com].values, np.ones(len(com), bool))[0] for a, b_ in o]; be = [M.eb(b_.x.loc[com].values, b_.v.loc[com].values, np.ones(len(com), bool))[0] for a, b_ in o]
            rU.append(np.corrcoef(*ue)[0, 1]); rB.append(np.corrcoef(*be)[0, 1])
        say(f"  {role:<4} {'pooled    ' if pool else 'chase-only'}: players with a baseline {len(j):>3} | median baseline weight {br.W.loc[j].median():6.1f} | tau baseline {np.sqrt(ebb[2]):.3f} | uplift tau {np.sqrt(eu[2]):.3f}, weight on own data {np.median(eu[3]):.2f} | repeatability: baseline {np.nanmean(rB):+.2f}, uplift {np.nanmean(rU):+.2f}")

# ------------------------------------------------------------------ (3) continuous consistency
say("\n(3) CONSISTENCY - repeatability across random match halves of the shrunk measure (%d splits)" % NS)
rng = np.random.default_rng(9)
def app_tab(vv, idc, okc, zc):
    P = vv[vv[okc] & (vv.pw > 0)]; w = P.pw * P.opp_w; g = P.assign(w_=w, s_=w * P[zc], q_=w * w).groupby([idc, "match_id"]).agg(W=("w_", "sum"), S=("s_", "sum"), Q=("q_", "sum")).reset_index()
    g = g[g.W >= 1.0].copy(); g["a"] = g.S / g.W; g["s"] = g.S / np.sqrt(g.Q); return g
def cons(g, q20, el, minapp):
    idc = g.columns[0]; g = g[g[idc].isin(el)]; n = g.groupby(idc).size(); keep = n[n >= minapp].index; g = g[g[idc].isin(keep)]; n = n.loc[keep]; out = {}
    def ebs(x, v_): return pd.Series(M.eb(x.values, v_.values, np.ones(len(x), bool))[0], index=x.index)
    g = g.assign(hit=(g.a > 0) * 1.0, dis=(g.a <= q20) * 1.0, dn_s=np.clip(-g.s, 0, None) ** 2, dn_a=np.clip(-g.a, 0, None) ** 2)
    for nm in ["hit", "dis"]: m = g.groupby(idc)[nm].mean(); p0 = m.mean(); out[nm] = ebs(m, p0 * (1 - p0) / n.loc[m.index])
    out["binary hit/disaster (current)"] = out["hit"] - out["dis"]
    for nm, lab in [("dn_s", "downside semi-variance of standardised appearance score"), ("dn_a", "downside semi-variance of appearance mean z")]:
        m = g.groupby(idc)[nm].mean(); out[lab] = -ebs(m, pd.Series(g[nm].var() / n.loc[m.index], index=m.index))
    return out
for role, (idc, okc, zr, zw, zt) in ROLE.items():
    full = M.build(va, role, verbose=False); el = set(full.index[full.elig]); rs = {}
    for s in range(NS):
        h = pd.Series(rng.integers(0, 2, len(mids)), index=mids); hs = []
        for k_ in (0, 1):
            g = app_tab(v[v.match_id.map(h) == k_], idc, okc, zt); hs.append(cons(g, g.a.quantile(.2), el, 2))
        for k_ in ["binary hit/disaster (current)", "downside semi-variance of standardised appearance score", "downside semi-variance of appearance mean z"]:
            com = hs[0][k_].index.intersection(hs[1][k_].index)
            if len(com) >= 30 and hs[0][k_].loc[com].std() > 0 and hs[1][k_].loc[com].std() > 0: rs.setdefault(k_, []).append(np.corrcoef(hs[0][k_].loc[com], hs[1][k_].loc[com])[0, 1])
    say(f"  {role}: " + " | ".join(f"{k_} {np.mean(a):+.2f}" for k_, a in rs.items()))

# ------------------------------------------------------------------ (5) first-innings weight
say(f"\n(5) FIRST-INNINGS LAYER WEIGHT - lambda = BSS(inn 1) / BSS(chase) = {LAM:.3f}; held-out correlation with the high-pressure runs value ({NS} splits)")
rng = np.random.default_rng(13)
for role, (idc, okc, zr, zw, zt) in ROLE.items():
    res = {}
    for s in range(NS):
        h = pd.Series(rng.integers(0, 2, len(mids)), index=mids); trm = v.match_id.map(h) == 0
        Pt = v[~trm & v[okc] & (v.pw > 0)]; wt = Pt.pw * Pt.opp_w; yR = (wt * Pt[zr]).groupby(Pt[idc]).sum() / wt.groupby(Pt[idc]).sum(); Wt = wt.groupby(Pt[idc]).sum()
        Pe = v[trm & v[okc] & (v.pe > 0)]; we = Pe.pe * Pe.opp_w; st0 = stats(Pe, idc, we, zr); idx = st0.index; F = d1[(d1.match_id.map(h) == 0) & d1[okc]]
        for lam in [0.0, round(LAM, 3), 0.5]:
            if lam == 0: st = st0
            else:
                D_ = pd.concat([Pe[[idc, "match_id", zr]].assign(w=we), F[[idc, "match_id", zr]].assign(w=F.pe / LAM * lam * F.opp_w)]); st = stats(D_, idc, D_.w, zr).reindex(idx)
            use = ((st.m >= 4) & (st.W >= 12)).values & st.x.notna().values; p = M.eb(st.x.fillna(0).values, st.v.fillna(1e6).values, use)[0]
            j = [i for i in idx.intersection(yR.index) if Wt[i] >= 8 and st0.loc[i, "m"] >= 3]; pos = pd.Index(idx).get_indexer(j); res.setdefault(lam, []).append(np.corrcoef(p[pos], yR[j].values)[0, 1])
    base = np.array(res[0.0]); say(f"  {role}")
    for lam, a in res.items():
        a = np.array(a); dd = a - base; say(f"    lambda {lam:<6}: corr {a.mean():+.3f} | vs no layer {dd.mean():+.3f} (better in {np.mean(dd > 0):.0%})" + ("   <- BSS ratio (adopted)" if lam == round(LAM, 3) else ""))

# ------------------------------------------------------------------ (4) state-aware outcome mix: numbers come from stage 02's own log
say("\n(4) STATE-AWARE OUTCOME MIX (stage 02, out-of-fold) - from logs/02_pressure_chase.log:")
lp = os.path.join(O, "logs", "02_pressure_chase.log")
if os.path.exists(lp):
    seen = set()
    for ln in open(lp).read().splitlines():
        if "outcome model innings" in ln and ln.strip()[:40] not in seen: seen.add(ln.strip()[:40]); say("  " + ln.strip())      # first two lines = the real run (later ones are stress-test re-runs)
say("  (run `python src/02_pressure_chase.py <out> --static-mix` to reproduce the old static innings x phase mix)")
open(os.path.join(O, "logs", "E6_proposed_improvements.log"), "w").write("\n".join(log))
