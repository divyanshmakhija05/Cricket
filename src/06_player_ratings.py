"""
Stage 06: player ratings under pressure (chase balls = PRIMARY layer). Separate batter and bowler tables.

Per player (batter and bowler tables are built separately, same method):
  P  pressure level  weighted mean of z over ALL chase balls with weight PI^2 x opposition weight (pressure-intensity weighting; chosen by held-out
                     prediction of high-pressure performance, evidence/E5), plus first-innings balls with weight lambda x PI1^2 (lambda = ratio of the Brier
                     skill scores of the two win-probability models, stage 05b; evidence/E6). z = runs-value part + wicket/residual part; each part is empirical-Bayes
                     shrunk with its OWN estimated reliability, then added. Eligibility, consistency and uplift stay anchored on the HIGH-pressure set (PI>0.5).
  B  baseline        usual RUNS value on ORDINARY balls: competitive chase balls (PI<0.5, 0.10<=WP<=0.90) pooled with ordinary first-innings balls
                     (0.25<=WP<=0.75, structural PI<=0.5): about 4.5x the balls of the chase-only baseline (evidence/E6)
  U  uplift          runs value under pressure minus usual runs value, shrunk directly with its own estimated reliability (small secondary term)
  C  consistency     shrunk share of pressure appearances with positive value, and shrunk share of 'disaster' appearances (total match impact in the league's worst 20%).
All components are put on ONE scale, "SDs of true skill": zeta = (posterior mean - prior mean) / tau, where tau is the estimated between-player
SD. A component with no estimated between-player spread (tau = 0) therefore contributes exactly 0 instead of having noise stretched to unit variance.
  RATING = 0.60 zeta_P + 0.30 zeta_C + 0.10 zeta_U
Shrinkage strength (tau) is ESTIMATED from the data (DerSimonian-Laird, on the players being ranked), never hand-set.
Sampling variance of a player's mean = ball-level variance x (sum w^2 / (sum w)^2) x design effect; the design effect (balls in one match are
not independent) is estimated from the data. IPL and T20I are normalised per competition x pressure bin in stage 05 and then pooled, so each
competition is automatically weighted by the player's sample size there; per-competition P is also reported.
"""
import sys, os, numpy as np, pandas as pd
from scipy.stats import spearmanr
O = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
T = os.path.join(O, "tables"); LOG = os.path.join(O, "logs")
log = []
def say(s=""): print(s); log.append(str(s))

APP_MIN_W = 1.0           # a 'pressure appearance' = a match in which the player's total pressure weight is >= 1 (about 2-4 high-pressure balls)
MIN_APP, MIN_W, MIN_HP = 8, 25.0, 15  # eligibility: >= 8 pressure appearances AND total pressure weight >= 25 AND >= 15 balls in the top-10% pressure band (about 2.5 overs of genuine high pressure)
W_P, W_C, W_U = 0.60, 0.30, 0.10
ROLES = {"bat": dict(id="striker_id", name="striker_name", ok="bat_ok", z="z_bat", zr="zr_bat", zw="zw_bat", va="VA_bat"),
         "bowl": dict(id="bowler_id", name="bowler_name", ok="bowl_ok", z="z_bowl", zr="zr_bowl", zw="zw_bowl", va="VA_bowl")}

def load_ball_table(T):
    """Chase value-added table (stage 05) + first-innings layer (stage 05b), stacked. First-innings rows carry pw = 0 (they never count for eligibility, consistency
    or uplift), an estimation weight pe = lambda x PI1^2 (lambda = BSS ratio) and, for ordinary balls, baseline_ball = True."""
    v = pd.read_csv(os.path.join(T, "va_chase.csv.gz"), low_memory=False)
    fp = os.path.join(T, "va_first.csv.gz")
    if os.path.exists(fp): v = pd.concat([v, pd.read_csv(fp, low_memory=False)], ignore_index=True)
    return v

def eb(x, v, use, mult=1.0):
    """DerSimonian-Laird empirical Bayes. x estimates, v sampling variances, use = players that define the prior.
    mult scales the estimated between-player variance (1 = as estimated; 0 = full shrinkage; inf = no shrinkage)."""
    x = np.asarray(x, float); v = np.asarray(v, float); xs, vs = x[use], v[use]; w = 1 / vs; mu = (w * xs).sum() / w.sum()
    q = (w * (xs - mu) ** 2).sum(); tau2 = max(0.0, (q - (len(xs) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    for _ in range(50):
        w2 = 1 / (vs + tau2); mu = (w2 * xs).sum() / w2.sum()
    t2 = tau2 * mult if not np.isinf(mult) else np.inf
    B = np.ones_like(v) if np.isinf(mult) else (np.zeros_like(v) if t2 == 0 else t2 / (t2 + v))
    return mu + B * (x - mu), mu, tau2, B

def wvar(x, w): m = (w * x).sum() / w.sum(); return (w * (x - m) ** 2).sum() / w.sum()

def layer_stats(d, idc, w, col):
    """per-player weighted mean of `col` with clustered (by match) sampling variance."""
    d = d.assign(_w=w, _wz=w * d[col], _w2=w * w)
    cl = d.groupby([idc, "match_id"]).agg(W=("_w", "sum"), S=("_wz", "sum"), Q=("_w2", "sum")).reset_index()
    g = cl.groupby(idc); W, S, Q, m = g.W.sum(), g.S.sum(), g.Q.sum(), g.size()
    x = S / W; cl = cl.join(x.rename("xb"), on=idc)
    sand = ((cl.S - cl.xb * cl.W) ** 2).groupby(cl[idc]).sum() / W ** 2 * (m / (m - 1).clip(lower=1))
    indep = wvar(d[col], d._w) * Q / W ** 2
    st = pd.DataFrame({"x": x, "W": W, "m": m, "sand": sand, "indep": indep})
    deff = st[st.m >= 5].sand.sum() / st[st.m >= 5].indep.sum()
    st["v"] = deff * st.indep
    return st, cl, deff

def build(v, role, floor_app=MIN_APP, floor_w=MIN_W, verbose=True, floor_hp=MIN_HP, z_col=None, comp=True, shrink_mult=1.0, weights=(W_P, W_C, W_U), info=None, window="all", cons_mode="S"):
    """v: ball table (stage 05 columns). z_col overrides the value column (then no component split). comp=True (default) shrinks the runs-value and wicket/residual components separately, each with its own estimated reliability; comp=False shrinks z as one number (check)."""
    r = ROLES[role]; idc = r["id"]; ok = v[v[r["ok"]]]
    zc = z_col or r["z"]; split = comp and z_col is None
    cols = [r["zr"], r["zw"]] if split else [zc]
    Pb = ok[ok.pw > 0]; wP = Pb.pw * Pb.opp_w                       # HIGH-PRESSURE set (PI > 0.5): eligibility, consistency, uplift, evidence
    Pe, wE = (ok[ok.pe > 0], None) if window == "all" else (Pb, None)  # ESTIMATION set of the pressure level: all chase balls weighted by PI^2 (window="all") or only PI>0.5 (window="hi")
    wE = (Pe.pe if window == "all" else Pe.pw) * Pe.opp_w
    Bb = ok[ok.baseline_ball]; wB = Bb.opp_w
    Pl = {c_: layer_stats(Pe, idc, wE, c_) for c_ in cols + [zc]}
    Bl = {c_: layer_stats(Bb, idc, wB, c_) for c_ in cols + [zc]}
    stP = Pl[zc][0]; stHi, clP, _ = layer_stats(Pb, idc, wP, zc)
    t = pd.DataFrame(index=stP.index); t["W"] = stHi.W.reindex(t.index).fillna(0); t["m"] = stP.m; t["m_hi"] = stHi.m.reindex(t.index).fillna(0); t["W_est"] = stP.W; t["P_raw"] = stP.x
    t["app"] = clP[clP.W >= APP_MIN_W].groupby(idc).size().reindex(t.index).fillna(0)
    t["hp_balls"] = (Pb.PI >= 0.9).groupby(Pb[idc]).sum().reindex(t.index).fillna(0)
    t["elig"] = ((t.app >= floor_app) & (t.W >= floor_w) & (t.hp_balls >= floor_hp)).values; e = t.elig.values
    # ---- pressure level and baseline: shrink each component separately, then add
    def shrunk(L, tag):
        post = np.zeros(len(t)); musum = 0.0; sd = {}
        for c_ in cols:
            st = L[c_][0].reindex(t.index)
            x = st.x.fillna(0).values; vv = st.v.fillna(1e6).values
            p, mu, tau2, B = eb(x, vv, e & st.x.notna().values, shrink_mult)
            post += p; musum += mu; sd[c_] = np.sqrt(tau2)
            if verbose: say(f"    {tag} component {c_:<8}: mu {mu:+.4f}, between-player SD tau {np.sqrt(tau2):.4f}, median shrink factor (eligible) {np.median(B[e]):.2f}")
        tau_tot = np.sqrt(sum(x_ ** 2 for x_ in sd.values()))        # true-skill SD of the sum of independent components
        return post, musum, tau_tot
    t["P_eb"], muP, tauP = shrunk(Pl, "pressure")
    # ---- baseline and UPLIFT, in the RUNS currency. The whole-z baseline has no measurable between-player spread (too few ordinary competitive chase
    # balls, and wickets dominate the noise); the runs-value part does. So uplift = (runs value under pressure) - (runs value on ordinary balls),
    # shrunk DIRECTLY with its own estimated reliability (sampling variance = v_pressure + v_baseline; the two ball sets are disjoint).
    pr, br = layer_stats(Pb, idc, wP, r["zr"])[0].reindex(t.index), layer_stats(Bb, idc, wB, r["zr"])[0].reindex(t.index)
    t["Pr_raw"], t["Br_raw"], t["b_balls_w"] = pr.x, br.x, br.W.fillna(0)
    uraw = (pr.x - br.x); uv = (pr.v + br.v); has = uraw.notna().values
    u_post, muU, tauU2, BU = eb(uraw.fillna(0).values, uv.fillna(1e6).values, e & has, shrink_mult)
    u_post = np.where(has, u_post, muU)                      # no ordinary balls at all -> no evidence of uplift either way (prior mean)
    t["U_raw"] = uraw; t["U"] = u_post; tauU = np.sqrt(tauU2)
    bp, muB, tauB2, _ = eb(br.x.fillna(0).values, br.v.fillna(1e6).values, e & br.x.notna().values, shrink_mult)
    t["B_eb"] = np.where(br.x.notna(), bp, muB)              # shrunk usual runs-value (for the report tables)
    if verbose: say(f"    baseline (runs value, pooled ordinary chase + first-innings balls): tau {np.sqrt(tauB2):.4f}, median weight per eligible player {t.b_balls_w[e].median():.1f} vs pressure {t.W[e].median():.1f}; "
                    f"uplift tau {tauU:.4f}, median shrink factor {np.median(BU[e & has]):.2f}, eligible players with no baseline balls {int((e & ~has).sum())}")
    # ---- consistency (appearance level)
    ap = clP[clP.W >= APP_MIN_W].copy(); ap["a"] = ap.S / ap.W
    qc = "S" if cons_mode == "S" else "a"; q20 = ap[qc].quantile(0.20); ap["hit"] = (ap.a > 0).astype(float); ap["dis"] = (ap[qc] <= q20).astype(float)      # disaster = TOTAL match impact (sum of w z) in the league's worst 20%: long bad innings count, 2-ball cameos cannot
    cs = ap.groupby(idc).agg(n_app=("a", "size"), hit=("hit", "mean"), dis=("dis", "mean"), app_sd=("a", "std"), app_med=("a", "median"))
    t = t.join(cs); na = t.n_app.fillna(0).values; zeta = {}
    for col in ["hit", "dis"]:
        p0 = np.nansum(t[col].values * na * e) / (na * e).sum()
        vv = (p0 * (1 - p0) / t.n_app.replace(0, np.nan)).fillna(1e6).values
        post, mu, tau2, B = eb(t[col].fillna(p0).values, vv, e, shrink_mult)
        t[col + "_eb"] = post; zeta[col] = (post - mu) / np.sqrt(tau2) if tau2 > 0 else np.zeros(len(t))
        if verbose: say(f"    consistency {col}: league rate {p0:.3f}, between-player SD {np.sqrt(tau2):.3f}")
    # ---- common scale: SDs of true skill
    t["zeta_P"] = (t.P_eb - muP) / tauP if tauP > 0 else 0.0
    t["zeta_C"] = 0.5 * (zeta["hit"] - zeta["dis"])
    t["zeta_U"] = (t.U - muU) / tauU if tauU > 0 else 0.0
    t["C_raw"] = t.hit_eb - t.dis_eb
    t["rating"] = weights[0] * t.zeta_P + weights[1] * t.zeta_C + weights[2] * t.zeta_U
    for cg in ["IPL", "T20I"]:
        msk = (Pe.comp_group == cg).values; sub = Pe[msk]; ww = wE[msk]            # positional mask: safe when rows are duplicated (bootstrap)
        t["P_" + cg] = (ww * sub[zc]).groupby(sub[idc]).sum().div(ww.groupby(sub[idc]).sum()).reindex(t.index)
        t["W_" + cg] = ww.groupby(sub[idc]).sum().reindex(t.index).fillna(0)
    names = v.groupby(idc)[r["name"]].agg(lambda s: s.iloc[-1]); t.insert(0, "player", names.reindex(t.index)); t["role"] = role
    if verbose: say(f"    players with any pressure ball {len(t)} | eligible {int(e.sum())}")
    if info is not None: info.update(tauP=tauP, tauU=tauU)
    return t.sort_values("rating", ascending=False)

def reliability(v, role, col="P_eb", n=40, seed=7, **kw):
    """Mean Pearson r between two random halves of the MATCHES (n repeats; whole pipeline re-run on each half, floors halved) across players eligible in the
    full sample with >=3 appearances in each half; plus early-vs-late (split at the median match date). Halves in which the prior collapses (tau=0) give NaN and are skipped."""
    full = build(v, role, verbose=False, **kw); rng = np.random.default_rng(seed); mids = v.match_id.unique(); rs = []
    def pair(m0, m1):
        hs = [build(v[m], role, MIN_APP // 2, MIN_W / 2, verbose=False, floor_hp=MIN_HP // 2, **kw) for m in (m0, m1)]
        idx = [i for i in full.index[full.elig] if i in hs[0].index and i in hs[1].index and hs[0].loc[i, "m"] >= 3 and hs[1].loc[i, "m"] >= 3]
        a, b_ = hs[0].loc[idx, col], hs[1].loc[idx, col]
        return (np.corrcoef(a, b_)[0, 1] if a.std() > 0 and b_.std() > 0 else np.nan), len(idx)
    for _ in range(n):
        h = pd.Series(rng.integers(0, 2, len(mids)), index=mids); hv = v.match_id.map(h); rs.append(pair(hv == 0, hv == 1)[0])
    md = pd.to_datetime(v.match_date); cut = md.median(); tr, nt = pair(md <= cut, md > cut)
    return np.nanmean(rs), np.nanstd(rs), int(np.isnan(rs).sum()), tr

if __name__ == "__main__":
    v = load_ball_table(T)
    say(f"Eligibility: >= {MIN_APP} pressure appearances (match with total pressure weight >= {APP_MIN_W}), total pressure weight >= {MIN_W:.0f}, and >= {MIN_HP} balls in the top-10% pressure band")
    say(f"Rating = {W_P:.2f} zeta(P) + {W_C:.2f} zeta(Consistency) + {W_U:.2f} zeta(Uplift); zeta = posterior mean in SDs of true between-player skill\n")
    out = {}
    for role in ROLES:
        say(f"=== {role.upper()} ==="); t = build(v, role); out[role] = t; e = t[t.elig]
        for col, nm in [("P_raw", "P raw"), ("P_eb", "P shrunk"), ("rating", "RATING")]:
            m_, sd_, nn, tr = reliability(v, role, col)
            say(f"    reliability {nm:<9}: random match-halves r = {m_:+.2f} (SD {sd_:.2f} over 40 splits, {nn} degenerate) | early vs late matches r = {tr:+.2f}")
        m_, sd_, nn, tr = reliability(v, role, "P_eb", comp=False)
        say(f"    (check: z shrunk as ONE number) P shrunk: random halves r = {m_:+.2f}, {nn}/40 splits degenerate (tau collapsed to 0) | early vs late r = {tr:+.2f}")
        say("    top 15 by rating (eligible):")
        say(e.head(15)[["player", "app", "W", "P_raw", "P_eb", "zeta_P", "zeta_C", "zeta_U", "rating"]].round(3).to_string()); say()
    out["bat"].to_csv(os.path.join(T, "ratings_batters.csv"), float_format="%.5g"); out["bowl"].to_csv(os.path.join(T, "ratings_bowlers.csv"), float_format="%.5g")
    open(os.path.join(LOG, "06_player_ratings.log"), "w").write("\n".join(log))
