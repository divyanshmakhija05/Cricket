"""
Stage 08: lock the final top-10 lists, build the evidence tables, charts and the Unstop-name check.
Usage: python src/08_final_outputs.py <out_dir> <data_dir>
The final order is the PRIMARY rating order (fixed in advance); robustness and bootstrap columns are reported next to it, never used to re-order.
"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
O = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "..", "output")
D = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(HERE, "..", "data")
T, LOG, CH = (os.path.join(O, x) for x in ("tables", "logs", "charts")); os.makedirs(CH, exist_ok=True)
log = []
def say(s=""): print(s); log.append(str(s))
v = pd.read_csv(os.path.join(T, "va_chase.csv.gz"), low_memory=False)
mp = pd.read_csv(os.path.join(D, "competition_team_player_mapping.csv"))
edges = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .975, .99, 1.0]
v["pbin"] = pd.cut(v.PI.fillna(0), edges, labels=False, include_lowest=True)
legal = v.kind.isin(["normal", "runout_etc", "retire", "bye_lb"])

ROLE = {"bat": dict(idc="striker_id", ok="bat_ok", tab="ratings_batters.csv"), "bowl": dict(idc="bowler_id", ok="bowl_ok", tab="ratings_bowlers.csv")}
final = {}
for r, c in ROLE.items():
    tall = pd.read_csv(os.path.join(T, c["tab"]), index_col=0); t = tall[tall.elig].sort_values("rating", ascending=False).copy(); t["rank"] = np.arange(1, len(t) + 1)
    # -------- near-miss eligibility audit: ineligible players who missed a floor narrowly, and the rank they would have had
    ne = tall[~tall.elig].copy(); ne["miss_app"] = (8 - ne.app).clip(lower=0); ne["miss_W"] = (25 - ne.W).clip(lower=0); ne["miss_hp"] = (15 - ne.hp_balls).clip(lower=0)
    near = ne[(ne.miss_app <= 2) & (ne.miss_W <= 5) & (ne.miss_hp <= 3) & ((ne.miss_app > 0) | (ne.miss_W > 0) | (ne.miss_hp > 0))].copy()
    near["would_rank"] = [int((t.rating > x).sum() + 1) for x in near.rating]; near = near.sort_values("rating", ascending=False)
    near[["player", "app", "W", "hp_balls", "miss_app", "miss_W", "miss_hp", "rating", "would_rank"]].to_csv(os.path.join(T, f"near_miss_{r}.csv"), float_format="%.4g")
    say(f"\nNEAR-MISS ELIGIBILITY AUDIT ({'batters' if r=='bat' else 'bowlers'}): missed a floor by <= 2 appearances / 5 weight / 3 top-band balls; would-be rank among eligible players")
    say(near[["player", "app", "W", "hp_balls", "rating", "would_rank"]].head(6).round(2).to_string() if len(near) else "  none")
    hi_ne = ne.sort_values("rating", ascending=False).head(3); say("  highest-rated ineligible: " + "; ".join(f"{p} (app {int(a)}, W {w_:.0f}, top-band {int(h)}, rating {rt:.2f})" for p, a, w_, h, rt in zip(hi_ne.player, hi_ne.app, hi_ne.W, hi_ne.hp_balls, hi_ne.rating)))
    bs = pd.read_csv(os.path.join(T, f"bootstrap_ranks_{r}.csv"), index_col=0)
    fr = pd.read_csv(os.path.join(T, f"robustness_top10_frequency_{r}.csv"), index_col=0)
    t = t.join(bs[["p_top10", "p_top5", "p_eligible", "median_rank", "rank_p05", "rank_p95"]]).join(fr[["in_top10_of_variants", "share_of_variants"]])
    # -------- evidence: actual vs expected given the SAME pressure states (league mean by competition x pressure bin)
    ok = v[v[c["ok"]]]; idc = c["idc"]
    if r == "bat":
        L = ok[ok.kind.isin(["normal", "runout_etc"])].copy(); L["y"] = L.bat_runs; L["w"] = (L.wicket_fell == 1).astype(float); L["bd"] = (L.bat_runs >= 4).astype(float)
        for col in ["y", "w", "bd"]: L["e_" + col] = L.groupby(["comp_group", "pbin"])[col].transform("mean")
        hp = L[L.PI >= .9]; od = L[L.baseline_ball]
        ev = pd.DataFrame({"hp_balls": hp.groupby(idc).size(), "hp_SR": 100 * hp.groupby(idc).y.mean(), "hp_SR_expected": 100 * hp.groupby(idc).e_y.mean(),
                           "hp_dismissals_per100": 100 * hp.groupby(idc).w.mean(), "hp_dismissals_per100_expected": 100 * hp.groupby(idc).e_w.mean(),
                           "hp_boundary_pct": 100 * hp.groupby(idc).bd.mean(), "hp_boundary_pct_expected": 100 * hp.groupby(idc).e_bd.mean(),
                           "ordinary_balls": od.groupby(idc).size(), "ordinary_SR": 100 * od.groupby(idc).y.mean(), "ordinary_SR_expected": 100 * od.groupby(idc).e_y.mean()})
        pr = L[L.pw > 0]; ev["pressure_balls"] = pr.groupby(idc).size(); ev["pressure_SR"] = 100 * pr.groupby(idc).y.mean(); ev["pressure_SR_expected"] = 100 * pr.groupby(idc).e_y.mean()
    else:
        L = ok[ok.kind.isin(["normal", "wide", "noball"])].copy()
        L["y"] = L.bat_runs + np.where(L.kind.isin(["wide", "noball"]), 1.0, 0.0); L["w"] = L.bowler_wicket.astype(float); L["dot"] = ((L.bat_runs == 0) & (L.kind == "normal")).astype(float)
        L["ex"] = L.kind.isin(["wide", "noball"]).astype(float)
        for col in ["y", "w", "dot", "ex"]: L["e_" + col] = L.groupby(["comp_group", "pbin"])[col].transform("mean")
        hp = L[L.PI >= .9]; od = L[L.baseline_ball]; pr = L[L.pw > 0]
        ev = pd.DataFrame({"hp_balls": hp.groupby(idc).size(), "hp_economy": 6 * hp.groupby(idc).y.mean(), "hp_economy_expected": 6 * hp.groupby(idc).e_y.mean(),
                           "hp_wickets_per100": 100 * hp.groupby(idc).w.mean(), "hp_wickets_per100_expected": 100 * hp.groupby(idc).e_w.mean(),
                           "hp_dot_pct": 100 * hp.groupby(idc).dot.mean(), "hp_dot_pct_expected": 100 * hp.groupby(idc).e_dot.mean(),
                           "hp_wide_nb_pct": 100 * hp.groupby(idc).ex.mean(), "ordinary_balls": od.groupby(idc).size(),
                           "ordinary_economy": 6 * od.groupby(idc).y.mean(), "ordinary_economy_expected": 6 * od.groupby(idc).e_y.mean()})
        ev["pressure_balls"] = pr.groupby(idc).size(); ev["pressure_economy"] = 6 * pr.groupby(idc).y.mean(); ev["pressure_economy_expected"] = 6 * pr.groupby(idc).e_y.mean()
    # appearances & wins
    ap = v[v[c["ok"]] & (v.pw > 0)].groupby([idc, "match_id"]).agg(W=("pw", "sum"), won=("batting_won", "first")).reset_index()
    ev = ev.rename(columns={"hp_balls": "hp_balls_legal"}); t = t.join(ev)
    # -------- Unstop dropdown name check (competition_team_player_mapping player_name for the player_id)
    nm = mp.groupby("player_id").player_name.agg(lambda s: sorted(set(s)))
    t["dropdown_name"] = [nm.get(i, [None])[0] for i in t.index]
    t["name_in_mapping"] = [i in nm.index for i in t.index]; t["name_variants"] = [len(nm.get(i, [])) for i in t.index]
    dup = mp.groupby("player_name").player_id.nunique(); t["same_name_other_ids"] = [int(dup.get(n_, 1)) - 1 if n_ else 0 for n_ in t.dropdown_name]
    final[r] = t
    t.head(25).to_csv(os.path.join(T, f"evidence_{r}.csv"), float_format="%.4g")
    top = t.head(10)
    top[["rank", "dropdown_name", "player", "rating", "zeta_P", "zeta_C", "zeta_U", "P_raw", "P_eb", "app", "W", "p_top10", "median_rank", "rank_p05", "rank_p95",
         "in_top10_of_variants", "name_in_mapping", "name_variants", "same_name_other_ids"]].to_csv(os.path.join(T, f"final_top10_{'batters' if r=='bat' else 'bowlers'}.csv"), index_label="player_id", float_format="%.4g")
    say(f"\n{'BATTERS' if r=='bat' else 'BOWLERS'} - FINAL TOP 10 (primary rating order)")
    say(top[["rank", "dropdown_name", "rating", "app", "p_top10", "median_rank", "in_top10_of_variants", "same_name_other_ids"]].round(2).to_string())
    gap = t.rating.diff(-1).abs(); say(f"  rating gaps between neighbours (ranks 1-11): " + ", ".join(f"{g:.2f}" for g in gap.head(11)))
    say("  EVIDENCE (high-pressure balls = top 10% of chase pressure; 'expected' = league average in the SAME pressure bins):")
    say(top[[c_ for c_ in ev.columns if not c_.startswith("ordinary") and not c_.startswith("pressure")]].round(1).assign(name=top.dropdown_name).set_index("name").to_string())
    say("  usual level (ordinary competitive balls) vs expected, and all pressure balls:")
    say(top[[c_ for c_ in ev.columns if c_.startswith("ordinary") or c_.startswith("pressure")]].round(1).assign(name=top.dropdown_name).set_index("name").to_string())
    bad = top[~top.name_in_mapping | (top.same_name_other_ids > 0) | (top.name_variants > 1)]
    say("  NAME CHECK: " + ("all 10 ids found in the mapping file, names unambiguous" if bad.empty else "ATTENTION -> " + bad[["dropdown_name", "name_in_mapping", "name_variants", "same_name_other_ids"]].to_string()))

# ---------------------------------------------------------------- charts
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for a, (r, ttl) in zip(ax, [("bat", "Batters"), ("bowl", "Bowlers")]):
    t = final[r].head(10).iloc[::-1]; y = np.arange(len(t))
    a.hlines(y, t.rank_p05, t.rank_p95.clip(upper=60), color="#bbbbbb", lw=3, zorder=1); a.scatter(t.median_rank, y, color="#d6007f", zorder=3, label="median rank")
    a.scatter(t["rank"], y, marker="D", color="#0b1f3a", zorder=4, s=22, label="final rank")
    a.set_yticks(y); a.set_yticklabels([f"{int(k)}. {n_}" for k, n_ in zip(t["rank"], t.dropdown_name)]); a.set_xlabel("rank among eligible players (bootstrap by match, 5th-95th percentile)")
    a.set_title(f"{ttl}: final rank vs rank uncertainty (500 resamples)"); a.set_xlim(0, 62); a.legend(frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(CH, "08_rank_uncertainty.png"), dpi=170); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for a, (r, ttl) in zip(ax, [("bat", "Batters"), ("bowl", "Bowlers")]):
    t = pd.read_csv(os.path.join(T, ROLE[r]["tab"]), index_col=0); t = t[t.elig]; top = final[r].head(10)
    a.scatter(t.P_raw, t.P_eb, s=10, color="#c9ced6"); a.scatter(top.P_raw, top.P_eb, s=26, color="#d6007f")
    for i, rw in top.iterrows(): a.annotate(rw.dropdown_name.split()[-1], (rw.P_raw, rw.P_eb), fontsize=7, xytext=(3, 2), textcoords="offset points")
    lim = [min(t.P_raw.min(), t.P_eb.min()), t.P_raw.max()]; a.plot(lim, lim, ls=":", color="k", lw=.8)
    a.set_xlabel("raw pressure value per ball (SD units)"); a.set_ylabel("after empirical-Bayes shrinkage"); a.set_title(f"{ttl}: shrinkage pulls small samples toward the league (dotted = no shrinkage)")
plt.tight_layout(); plt.savefig(os.path.join(CH, "08_shrinkage.png"), dpi=170); plt.close()

# actual vs expected for the final top 10 under high pressure
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
tb = final["bat"].head(10).iloc[::-1]; ax[0].barh(np.arange(10), tb.hp_SR - tb.hp_SR_expected, color="#0b1f3a")
ax[0].set_yticks(np.arange(10)); ax[0].set_yticklabels(tb.dropdown_name); ax[0].axvline(0, color="k", lw=.8)
ax[0].set_xlabel("strike rate above league average in the same pressure states (high-pressure balls)"); ax[0].set_title("Top-10 batters: scoring vs expectation at high pressure")
to = final["bowl"].head(10).iloc[::-1]; ax[1].barh(np.arange(10), to.hp_economy_expected - to.hp_economy, color="#d6007f")
ax[1].set_yticks(np.arange(10)); ax[1].set_yticklabels(to.dropdown_name); ax[1].axvline(0, color="k", lw=.8)
ax[1].set_xlabel("runs per over saved vs league average in the same pressure states (high-pressure balls)"); ax[1].set_title("Top-10 bowlers: run prevention vs expectation at high pressure")
plt.tight_layout(); plt.savefig(os.path.join(CH, "08_top10_vs_expected.png"), dpi=170); plt.close()

# robustness: how often each top-25 candidate is in the top 10 across variants
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for a, (r, ttl) in zip(ax, [("bat", "Batters"), ("bowl", "Bowlers")]):
    t = final[r].head(14).iloc[::-1]; col = ["#d6007f" if k <= 10 else "#bbbbbb" for k in t["rank"]]
    a.barh(np.arange(len(t)), t.share_of_variants, color=col); a.set_yticks(np.arange(len(t))); a.set_yticklabels([f"{int(k)}. {n_}" for k, n_ in zip(t["rank"], t.dropdown_name)])
    a.set_xlabel("share of 26 alternative definitions in which the player is in the top 10"); a.set_xlim(0, 1); a.set_title(f"{ttl}: robustness to method choices")
plt.tight_layout(); plt.savefig(os.path.join(CH, "08_variant_robustness.png"), dpi=170); plt.close()
open(os.path.join(LOG, "08_final_outputs.log"), "w").write("\n".join(log))
