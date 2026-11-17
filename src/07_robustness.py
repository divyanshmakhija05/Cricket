"""
Stage 07 (Day 5): robustness of the ratings and rank confidence.
  (a) 30+ variants: pressure definition (leverage percentile / run-rate-tight / composite index), pressure weight shape, hard cut-offs, leverage weight,
      associate-nation weight, first-innings layer on/off, shrinkage strength, rating-component weights, component shrinkage, eligibility floors,
      opposition-adjusted two-way ridge model.   For each: Spearman vs primary and top-10 overlap.
  (b) temporal stability (early vs late matches) and IPL-vs-T20I agreement.
  (c) bootstrap by match (B resamples of matches): rank confidence, P(top 10).
(v = chase balls + first-innings layer, see 06 load_ball_table)
Writes output/tables/robustness_*.csv, bootstrap_*.csv and logs/07_robustness.log
"""
import sys, os, importlib.util, numpy as np, pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import spearmanr
O = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 500
T = os.path.join(O, "tables"); LOG = os.path.join(O, "logs")
sp = importlib.util.spec_from_file_location("r06", os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_player_ratings.py")); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M)
log = []
def say(s=""): print(s); log.append(str(s))
ROLES = ["bat", "bowl"]

v = M.load_ball_table(T)
pc = pd.read_csv(os.path.join(T, "pressure_chase.csv.gz")); pc = pc[pc.innings == 2]
rr = pd.read_csv(os.path.join(T, "alt_runrate_pressure.csv.gz"), usecols=["match_id", "innings_ball_num", "rr_tight_pct"])
v = v.merge(pc[["match_id", "innings_ball_num", "D_bat", "D_bowl", "S"]], on=["match_id", "innings_ball_num"], how="left").merge(rr, on=["match_id", "innings_ball_num"], how="left")
pw_of = lambda pi, p=2.0, lo=0.5: np.clip((pi - lo) / (1 - lo), 0, None) ** p

# ---------------------------------------------------------------- opposition-adjusted two-way ridge
def ridge_two_way(vv, tau):
    """y = z_bat = batter effect - bowler effect + noise, ridge-penalised with lambda = s2/tau^2 per side (tau from the primary EB fit)."""
    P = vv[(vv.pe > 0) & (vv.bat_ok | vv.bowl_ok)]; w = (P.pe * P.opp_w).values; y = P.z_bat.values
    bi = pd.Index(P.striker_id.unique()); oi = pd.Index(P.bowler_id.unique())
    n = len(P); rows = np.arange(n); nb, no = len(bi), len(oi)
    Xb = sparse.csr_matrix((P.bat_ok.values.astype(float), (rows, bi.get_indexer(P.striker_id))), shape=(n, nb))
    Xo = sparse.csr_matrix((-P.bowl_ok.values.astype(float), (rows, oi.get_indexer(P.bowler_id))), shape=(n, no))
    X = sparse.hstack([sparse.csr_matrix(np.ones((n, 1))), Xb, Xo]).tocsr(); W = sparse.diags(w)
    s2 = M.wvar(y, w); pen = np.concatenate([[0.0], np.full(nb, s2 / tau["bat"] ** 2), np.full(no, s2 / tau["bowl"] ** 2)])
    beta = spsolve((X.T @ W @ X + sparse.diags(pen)).tocsc(), X.T @ (W * y))
    return pd.Series(beta[1:1 + nb], index=bi), pd.Series(beta[1 + nb:], index=oi)

# ---------------------------------------------------------------- variant machinery
def tables(vv=None, **kw):
    vv = v if vv is None else vv
    return {r: M.build(vv, r, verbose=False, **kw) for r in ROLES}
def chase_only(f):
    """Variant weights change the CHASE balls only; first-innings rows keep their calibrated layer weight lambda x PI1^2."""
    return lambda d: np.where(d.kind == "first", d.pe, f(d))
def mod(**cols):
    d = v.copy()
    for k, f in cols.items(): d[k] = f(d)
    return d
tau = {}
prim = {}
for r in ROLES:
    info = {}; prim[r] = M.build(v, r, verbose=False, info=info); tau[r] = info["tauP"]

V = {}                                                    # name -> dict(role -> table)
V["PRIMARY"] = prim
V["previous primary: high-pressure balls only (PI>0.5)"] = tables(window="hi")
V["pressure weight linear (power 1)"] = tables(mod(pe=chase_only(lambda d: d.PI.fillna(0))))
V["pressure weight cubic (power 3)"] = tables(mod(pe=chase_only(lambda d: d.PI.fillna(0) ** 3)))
V["hard cut: top 25% balls, equal weight"] = tables(mod(pe=chase_only(lambda d: (d.PI >= .75) * 1.0)))
V["hard cut: top 10% balls, equal weight"] = tables(mod(pe=chase_only(lambda d: (d.PI >= .9) * 1.0)))
V["weight = leverage index"] = tables(mod(pe=chase_only(lambda d: d.li)))
V["weight = PI^2 x leverage index"] = tables(mod(pe=chase_only(lambda d: d.PI.fillna(0) ** 2 * d.li)))
V["pressure weight from PI>0.5 only (stricter start)"] = tables(mod(pe=chase_only(lambda d: pw_of(d.PI, 2.0, 0.5))))
V["run-rate-tight pressure definition"] = tables(mod(PI=lambda d: d.rr_tight_pct.fillna(0), pw=lambda d: pw_of(d.rr_tight_pct.fillna(0)), pe=chase_only(lambda d: d.rr_tight_pct.fillna(0) ** 2)))
def composite(role):
    L = v.leverage.where(v.kind.isin(["normal", "bye_lb", "runout_etc"]), np.nan)
    pL, pD, pS = L.rank(pct=True), v["D_" + role].where(L.notna()).rank(pct=True), v.S.where(L.notna()).rank(pct=True)
    return (.60 * pL + .25 * pD + .15 * pS).fillna(v.PI)
Vc = {}
for r in ROLES:
    pic = composite(r); pic = pic.rank(pct=True)           # re-percentile so that "top 10%" keeps its meaning
    Vc[r] = M.build(v.assign(PI=np.where(v.kind == "first", v.PI, pic), pw=np.where(v.kind == "first", 0.0, pw_of(pic)), pe=np.where(v.kind == "first", v.pe, pic ** 2)), r, verbose=False)
V["composite pressure index (leverage+damage+stress)"] = Vc
V["associate-nation T20Is weight 1.0"] = tables(mod(opp_w=lambda d: 1.0))
V["associate-nation T20Is excluded"] = tables(mod(opp_w=lambda d: np.where(d.tier == "T20I_Assoc", 0.0, 1.0)))
LAM = float(pd.read_csv(os.path.join(T, "layer_weight.csv")).lambda_first_innings[0]); fst = (v.kind == "first")
V["first-innings layer OFF (chase balls only)"] = tables(mod(pe=lambda d: np.where(fst, 0.0, d.pe)))
V["first-innings layer weight x2"] = tables(mod(pe=lambda d: np.where(fst, 2 * d.pe, d.pe)))
V["baseline from chase balls only"] = tables(mod(baseline_ball=lambda d: d.baseline_ball & ~fst))
V["shrinkage x0.5 (weaker)"] = tables(shrink_mult=0.5)
V["shrinkage x2 (weaker still)"] = tables(shrink_mult=2.0)
V["shrinkage x0.25 (stronger)"] = tables(shrink_mult=0.25)
V["no shrinkage (raw means)"] = tables(shrink_mult=np.inf)
V["consistency: disaster by appearance mean z (previous)"] = tables(cons_mode="a")
V["rating = pressure level only"] = tables(weights=(1, 0, 0))
V["rating = 0.7 P + 0.3 C"] = tables(weights=(.7, .3, 0))
V["rating = 0.5 P + 0.5 C"] = tables(weights=(.5, .5, 0))
V["rating = 0.4 P + 0.3 C + 0.3 U"] = tables(weights=(.4, .3, .3))
V["z shrunk as one number (no component split)"] = tables(comp=False)
V["eligibility stricter (>=12 app, W>=40, >=25 top-band balls)"] = tables(floor_app=12, floor_w=40.0, floor_hp=25)
V["eligibility looser (>=6 app, W>=15, >=8 top-band balls)"] = tables(floor_app=6, floor_w=15.0, floor_hp=8)
V["no top-band exposure floor"] = tables(floor_hp=0)
V["value = raw WP VA (no z)"] = {r: M.build(v, r, verbose=False, z_col=M.ROLES[r]["va"]) for r in ROLES}
adj = {}
for r in ROLES:
    a_b, a_o = ridge_two_way(v, tau)
    t = prim[r].copy(); eff = (a_b if r == "bat" else a_o).reindex(t.index); mu = eff[t.elig].mean()
    t["zeta_P"] = ((eff - mu) / tau[r]).fillna(0)
    t["rating"] = M.W_P * t.zeta_P + M.W_C * t.zeta_C + M.W_U * t.zeta_U; adj[r] = t.sort_values("rating", ascending=False)
V["opposition-adjusted (two-way ridge)"] = adj
V = {k: x for k, x in V.items() if x is not None}

# ---------------------------------------------------------------- compare variants with primary
say(f"ROBUSTNESS: {len(V)-1} variants of the primary rating. Spearman = rank correlation with PRIMARY among players eligible in both; overlap = size of top-10 intersection.")
rows = []
for nm, tb in V.items():
    row = {"variant": nm}
    for r in ROLES:
        a, b = prim[r], tb[r]; ea, eb_ = a[a.elig], b[b.elig]; com = ea.index.intersection(eb_.index)
        row[r + "_spearman"] = spearmanr(ea.loc[com, "rating"], eb_.loc[com, "rating"]).correlation
        row[r + "_top10_overlap"] = len(set(ea.head(10).index) & set(eb_.head(10).index))
    rows.append(row)
R = pd.DataFrame(rows); say(R.round(2).to_string(index=False))
R.to_csv(os.path.join(T, "robustness_variants.csv"), index=False)
say("\nmedian across variants (excluding primary): bat Spearman %.2f, top-10 overlap %.1f | bowl Spearman %.2f, top-10 overlap %.1f" %
    (R.iloc[1:].bat_spearman.median(), R.iloc[1:].bat_top10_overlap.median(), R.iloc[1:].bowl_spearman.median(), R.iloc[1:].bowl_top10_overlap.median()))

freq = {}
for r in ROLES:
    names = prim[r].player
    cnt = pd.Series(0, index=prim[r].index[prim[r].elig][:25])
    for nm, tb in V.items():
        if nm == "PRIMARY": continue
        top = tb[r][tb[r].elig].head(10).index
        cnt = cnt.add(pd.Series(1, index=top), fill_value=0)
    cnt = cnt.reindex(prim[r].index[prim[r].elig][:25]).fillna(0)
    freq[r] = pd.DataFrame({"player": names.reindex(cnt.index), "in_top10_of_variants": cnt.astype(int), "primary_rank": [list(prim[r][prim[r].elig].index).index(i) + 1 for i in cnt.index]}).sort_values("in_top10_of_variants", ascending=False)
    freq[r]["share_of_variants"] = (freq[r].in_top10_of_variants / (len(V) - 1)).round(2)
    say(f"\n{r.upper()}: how often each player is in the top 10 across the {len(V)-1} variants"); say(freq[r].head(18).to_string())
    freq[r].to_csv(os.path.join(T, f"robustness_top10_frequency_{r}.csv"))

# ---------------------------------------------------------------- temporal stability and IPL vs T20I
say("\nTEMPORAL STABILITY: ratings from the earlier half of matches vs the later half (split at the median match date)")
md = pd.to_datetime(v.match_date); cut = md.median(); say(f"  split date {cut.date()}")
for r in ROLES:
    hs = [M.build(v[(md <= cut) if k == 0 else (md > cut)], r, M.MIN_APP // 2, M.MIN_W / 2, verbose=False, floor_hp=M.MIN_HP // 2) for k in (0, 1)]
    idx = [i for i in prim[r].index[prim[r].elig] if i in hs[0].index and i in hs[1].index and hs[0].loc[i, "m"] >= 3 and hs[1].loc[i, "m"] >= 3]
    for col in ["P_eb", "rating"]:
        a, b = hs[0].loc[idx, col], hs[1].loc[idx, col]
        say(f"  {r:<4} {col:<7} n={len(idx):>3}: Pearson {np.corrcoef(a, b)[0,1]:.2f}, Spearman {spearmanr(a, b).correlation:.2f}")
say("\nIPL vs T20I AGREEMENT (players with total estimation weight >= 20 in BOTH competitions; raw pressure z)")
for r in ROLES:
    t = prim[r]; both = t[(t.W_IPL >= 20) & (t.W_T20I >= 20)]
    say(f"  {r:<4} n={len(both):>3}: Pearson {np.corrcoef(both.P_IPL, both.P_T20I)[0,1]:.2f}, Spearman {spearmanr(both.P_IPL, both.P_T20I).correlation:.2f}")

# ---------------------------------------------------------------- bootstrap by match
say(f"\nBOOTSTRAP BY MATCH: {NB} resamples of the {v.match_id.nunique()} matches (with replacement); the whole rating is rebuilt each time")
groups = v.groupby("match_id").indices; mids = np.array(list(groups.keys())); rng = np.random.default_rng(20261117)
ranks = {r: pd.DataFrame(np.nan, index=prim[r].index[prim[r].elig], columns=range(NB)) for r in ROLES}
for b_ in range(NB):
    pick = rng.choice(mids, len(mids), replace=True)
    idx = np.concatenate([groups[m_] for m_ in pick]); newid = np.repeat(np.arange(len(pick)), [len(groups[m_]) for m_ in pick])
    vb = v.iloc[idx].assign(match_id=newid).reset_index(drop=True)
    for r in ROLES:
        t = M.build(vb, r, verbose=False); e = t[t.elig]; rk = pd.Series(np.arange(1, len(e) + 1), index=e.index)
        common = ranks[r].index.intersection(rk.index); ranks[r].loc[common, b_] = rk.loc[common].values
for r in ROLES:
    rk = ranks[r]; out = pd.DataFrame({"player": prim[r].player.reindex(rk.index), "primary_rank": np.arange(1, len(rk) + 1),
        "p_top10": (rk <= 10).mean(axis=1), "p_top5": (rk <= 5).mean(axis=1), "p_eligible": rk.notna().mean(axis=1),
        "median_rank": rk.median(axis=1), "rank_p05": rk.quantile(.05, axis=1), "rank_p95": rk.quantile(.95, axis=1)})
    out.to_csv(os.path.join(T, f"bootstrap_ranks_{r}.csv"), float_format="%.3f")
    say(f"\n{r.upper()} (top 16 by primary rank)"); say(out.head(16).round(2).to_string())
    top = out.head(10); say(f"  expected number of the primary top-10 that stay in a resampled top-10: {top.p_top10.sum():.1f} of 10")
open(os.path.join(LOG, "07_robustness.log"), "w").write("\n".join(log))
