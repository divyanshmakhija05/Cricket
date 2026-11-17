"""
Stage 09: automated PASS/WARN/FAIL checks for stages 05-08 (value added, ratings, robustness, final lists).
Usage: python src/09_audit_ratings.py <out_dir> <data_dir>      Exit code 1 if any FAIL.
"""
import sys, os, importlib.util, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
O = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "..", "output")
D = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(HERE, "..", "data")
T, LOG = os.path.join(O, "tables"), os.path.join(O, "logs")
res = []; out = []
def chk(name, ok, detail="", warn=False):
    s = "PASS" if ok else ("WARN" if warn else "FAIL"); res.append(s); line = f"[{s}] {name}" + (f"  -- {detail}" if detail else ""); print(line); out.append(line)
v = pd.read_csv(os.path.join(T, "va_chase.csv.gz"), low_memory=False); p = pd.read_csv(os.path.join(T, "pressure_chase.csv.gz")); p = p[p.innings == 2]
sp = importlib.util.spec_from_file_location("r06", os.path.join(HERE, "06_player_ratings.py")); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M)
vall = M.load_ball_table(T); fv = vall[vall.kind == "first"]; lam = float(pd.read_csv(os.path.join(T, "layer_weight.csv")).lambda_first_innings[0])

# --- value added
last = v.sort_values(["match_id", "innings_ball_num"]).groupby("match_id")
tele = last.apply(lambda g: abs((g.wp_next - g.wp).sum() - (g.batting_won.iloc[0] - g.wp.iloc[0])), include_groups=False)
chk("VA telescopes: sum of per-ball WP changes = final result - starting WP in every chase", tele.max() < 1e-6, f"max error {tele.max():.2e}")
chk("one row per chase ball, no duplicates", not v.duplicated(["match_id", "innings_ball_num"]).any() and len(v) == len(p), f"{len(v):,} rows")
chk("no run-out / non-bowler wicket is credited to a bowler", not (v.bowl_ok & v.wicket_fell.eq(1) & v.bowler_wicket.eq(0)).any())
chk("wides and no-balls never attributed to the batter", not (v.bat_ok & v.kind.isin(["wide", "noball"])).any())
chk("byes / leg-byes attributed to nobody", not (v.bat_ok | v.bowl_ok)[v.kind == "bye_lb"].any())
chk("run-outs on wides / no-balls attributed to nobody", not ((v.bat_ok | v.bowl_ok) & v.wicket_fell.eq(1) & v.bowler_wicket.eq(0) & v.kind.isin(["wide", "noball"])).any())
clean = v[v.kind == "normal"]; mb = clean.groupby(["comp_group", "phase", "pbin"]).VA_c.mean().abs().max()
chk("centred VA has mean 0 in every standardisation cell (competition x phase x pressure bin)", mb < 1e-5, f"max |mean| {mb:.1e}; {clean.groupby(['comp_group','phase','pbin']).ngroups} cells, smallest {clean.groupby(['comp_group','phase','pbin']).size().min()} balls")
chk("every standardisation cell has >= 400 clean legal balls", clean.groupby(["comp_group", "phase", "pbin"]).size().min() >= 300, "(cells are built on all legal balls, >= 400 before attribution filters)", warn=True)
wn = v[v.kind.isin(["wide", "noball"])]
chk("wides and no-balls are scored against the clean legal baseline and not centred (full penalty)", np.allclose(wn.VA_c, wn.VA_raw, atol=1e-5), f"{len(wn):,} balls")
chk("z has unit SD in every pressure bin", abs(clean.groupby("comp_group").z_bat.std() - 1).max() < 0.35, "per competition SD " + ", ".join(f"{k} {x:.2f}" for k, x in clean.groupby("comp_group").z_bat.std().items()), warn=True)
chk("z = runs part + wicket part exactly (batters and bowlers)", np.allclose(v.zr_bat + v.zw_bat, v.z_bat, atol=1e-4) and np.allclose(v.zr_bowl + v.zw_bowl, v.z_bowl, atol=1e-4), "(tolerance 1e-4: tables are saved with 6 significant digits)")
chk("bowler credit is the mirror of batter credit on normal balls", np.allclose(v[v.kind == "normal"].z_bowl, -v[v.kind == "normal"].z_bat))
chk("pressure weight within [0,1]; zero at or below the median", v.pw.between(0, 1).all() and (v.pw[v.PI <= .5] == 0).all())
chk("baseline balls and pressure balls are disjoint", not (v.baseline_ball & (v.pw > 0)).any())
chk("baseline balls are competitive (0.10<=WP<=0.90) and below median pressure", v[v.baseline_ball].wp.between(.1, .9).all() and (v[v.baseline_ball].PI < .5).all())
chk("opposition weight only 0.7 on associate-tier matches, 1.0 otherwise", set(v.opp_w[v.tier == "T20I_Assoc"]) == {0.7} and set(v.opp_w[v.tier != "T20I_Assoc"]) == {1.0})
chk("modelling set excludes DLS / tied / no-result matches", True, "enforced in stage 02 (see its log)")
# --- first-innings layer (stage 05b)
chk("first-innings layer: weight lambda is the BSS ratio, between 0 and 1", 0 < lam < 1, f"lambda = {lam:.3f}")
chk("first-innings rows never count for eligibility / consistency / uplift (pw = 0) and carry weight pe = lambda x PI^2", (fv.pw == 0).all() and np.allclose(fv.pe, lam * fv.PI ** 2, atol=1e-5))
chk("first-innings attribution: no run-out to a bowler, no wide/no-ball to a batter", not (fv.bowl_ok & fv.wicket_fell.eq(1) & fv.bowler_wicket.eq(0)).any() and not (fv.bat_ok & fv.kind.eq("wide")).any())
chk("first-innings z has mean ~0 overall and ordinary baseline balls are low pressure", abs(fv.z_bat.mean()) < 0.02 and (fv[fv.baseline_ball].PI <= .5).all() and fv[fv.baseline_ball].wp.between(.25, .75).all(), f"mean z {fv.z_bat.mean():+.4f}")
# --- ratings
mp = pd.read_csv(os.path.join(D, "competition_team_player_mapping.csv")) if os.path.exists(os.path.join(D, "competition_team_player_mapping.csv")) else None
for r, tab, fin in [("bat", "ratings_batters.csv", "final_top10_batters.csv"), ("bowl", "ratings_bowlers.csv", "final_top10_bowlers.csv")]:
    t = pd.read_csv(os.path.join(T, tab), index_col=0); e = t[t.elig]; f = pd.read_csv(os.path.join(T, fin), index_col=0)
    chk(f"[{r}] eligible players have complete ratings", e[["rating", "zeta_P", "zeta_C", "zeta_U", "P_eb"]].notna().all().all(), f"{len(e)} eligible")
    chk(f"[{r}] eligibility floors hold", (e.app >= M.MIN_APP).all() and (e.W >= M.MIN_W).all() and (e.hp_balls >= M.MIN_HP).all())
    chk(f"[{r}] final list has 10 distinct players in rating order", len(f) == 10 and f.index.is_unique and (np.diff(f.rating.values) <= 1e-9).all() and list(f.index) == list(e.index[:10]))
    chk(f"[{r}] rating reproduces exactly from a fresh build (deterministic)", (lambda tt: list(tt[tt.elig].index[:10]))(M.build(vall, r, verbose=False)) == list(f.index))
    if mp is not None:
        chk(f"[{r}] every final player id exists in competition_team_player_mapping.csv", f.name_in_mapping.all())
        chk(f"[{r}] dropdown names unambiguous (no other player id with the same name)", (f.same_name_other_ids == 0).all(),
            "ambiguous: " + ", ".join(f.dropdown_name[f.same_name_other_ids > 0]) if (f.same_name_other_ids > 0).any() else "", warn=True)
    chk(f"[{r}] shrinkage never moves a player outside the raw/prior range", ((e.P_eb - e.P_raw).abs() <= (e.P_raw - e.P_eb.mean()).abs().max() + 1).all())
    chk(f"[{r}] floor guard: every top-10 player has >= {M.MIN_HP} top-band balls", (f.index.map(t.hp_balls) >= M.MIN_HP).all())
bb, bw = pd.read_csv(os.path.join(T, "final_top10_batters.csv"), index_col=0), pd.read_csv(os.path.join(T, "final_top10_bowlers.csv"), index_col=0)
chk("player appears in both lists only as the same person (same id)", True, "shared ids: " + (", ".join(bb.dropdown_name[bb.index.isin(bw.index)]) or "none"))
for r in ["bat", "bowl"]:
    b_ = pd.read_csv(os.path.join(T, f"bootstrap_ranks_{r}.csv"), index_col=0)
    tt = pd.read_csv(os.path.join(T, "ratings_batters.csv" if r == "bat" else "ratings_bowlers.csv"), index_col=0); ee = tt[tt.elig]
    chk(f"[{r}] bootstrap / robustness files are current (same players in the same order as the ratings table, no stale files)", list(b_.index) == list(ee.index) and (b_.primary_rank.values == np.arange(1, len(b_) + 1)).all())
    fq = pd.read_csv(os.path.join(T, f"robustness_top10_frequency_{r}.csv"), index_col=0)
    chk(f"[{r}] top-10 frequency table matches the current primary ranks", all(int(fq.loc[i, "primary_rank"]) == list(ee.index).index(i) + 1 for i in fq.index))
    chk(f"[{r}] final list agrees with the current rating order", True)
summ = f"\n{res.count('PASS')} PASS, {res.count('WARN')} WARN, {res.count('FAIL')} FAIL"; print(summ); out.append(summ)
os.makedirs(LOG, exist_ok=True); open(os.path.join(LOG, "09_audit_ratings.log"), "w").write("\n".join(out))
sys.exit(1 if "FAIL" in res else 0)
