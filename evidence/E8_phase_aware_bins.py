"""
E8: phase-aware standardisation cells (competition x phase x pressure bin) versus the old pooled bins (competition x pressure bin).
Rebuilds stages 05 / 05b with --global-bins in a temporary folder and compares with the adopted tables:
  (1) heterogeneity of phases INSIDE a common grid of pressure bins (mean and SD of z by phase),
  (2) repeatability of the shrunk pressure level (a bias fix is not expected to change it),
  (3) systematic bias: how the change in a player's pressure level relates to his share of death-over exposure,
  (4) rank agreement and top-10 overlap.
"""
import sys, os, shutil, tempfile, subprocess, warnings, numpy as np, pandas as pd, importlib.util
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "..", "output"); T = os.path.join(O, "tables")
sp = importlib.util.spec_from_file_location("r06", os.path.join(HERE, "..", "src", "06_player_ratings.py")); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M)
log = []
def say(s=""): print(s); log.append(str(s))
tmp = tempfile.mkdtemp(); os.makedirs(os.path.join(tmp, "tables")); os.makedirs(os.path.join(tmp, "logs"))
for f in ["ball_state", "pressure_chase", "pressure_first_innings"]: shutil.copy(os.path.join(T, f + ".csv.gz"), os.path.join(tmp, "tables"))
for stage in ["05_value_added.py", "05b_first_innings_va.py"]: subprocess.run([sys.executable, os.path.join(HERE, "..", "src", stage), tmp, "--global-bins"], check=True, capture_output=True)
adopted = M.load_ball_table(T); old = M.load_ball_table(os.path.join(tmp, "tables"))
edges = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .975, .99, 1.0]
say("E8 PHASE-AWARE STANDARDISATION vs POOLED BINS")
say("\n(1) phases inside a common grid of 13 pressure bins (clean legal chase balls, cells with >= 300 balls): spread across phases of the mean and SD of z")
for nm, vv in [("pooled bins (old)", old), ("phase-aware bins (adopted)", adopted)]:
    c = vv[vv.kind == "normal"].copy(); c["g"] = pd.cut(c.PI.fillna(0), edges, labels=False, include_lowest=True)
    g = c.groupby(["comp_group", "g", "phase"]).agg(n=("z_bat", "size"), m=("z_bat", "mean"), s=("z_bat", "std")).reset_index(); g = g[(g.n >= 300) & (g.g >= 5)]
    rr = g.groupby(["comp_group", "g"]).agg(k=("m", "size"), dm=("m", lambda x: x.max() - x.min()), ds=("s", lambda x: x.max() / x.min())); rr = rr[rr.k >= 2]
    say(f"  {nm:<28} bins with >=2 phases: {len(rr)} | mean-z range across phases: mean {rr.dm.mean():.3f}, max {rr.dm.max():.3f} | SD ratio across phases: mean {rr.ds.mean():.2f}, max {rr.ds.max():.2f}")
say("\n(2) repeatability of the shrunk pressure level (random match halves, 20 splits; early vs late)")
for role in ["bat", "bowl"]:
    for nm, vv in [("pooled bins (old)", old), ("phase-aware (adopted)", adopted)]:
        m_, sd_, nn, tr = M.reliability(vv, role, "P_eb", n=20); say(f"  {role:<4} {nm:<24} random halves {m_:+.2f} | early vs late {tr:+.2f}")
say("\n(3)-(4) change in ratings when moving from pooled to phase-aware bins (eligible players of the adopted model)")
for role, idc, okc in [("bat", "striker_id", "bat_ok"), ("bowl", "bowler_id", "bowl_ok")]:
    a = M.build(adopted, role, verbose=False); b = M.build(old, role, verbose=False); ea = a[a.elig]; b = b.loc[ea.index]
    P = adopted[adopted[okc] & (adopted.pe > 0)]; w = P.pe * P.opp_w; death = ((w * (P.phase == "death")).groupby(P[idc]).sum() / w.groupby(P[idc]).sum()).reindex(ea.index)
    d = ea.zeta_P - b.zeta_P
    t10a = set(ea.sort_values("rating", ascending=False).head(10).index); t10b = set(b.sort_values("rating", ascending=False).head(10).index)
    say(f"  {role:<4}: rating Spearman {spearmanr(ea.rating, b.rating).correlation:.3f} | top-10 overlap {len(t10a & t10b)}/10 | change in pressure level vs death-over share: correlation {np.corrcoef(d, death)[0,1]:+.2f}, slope {np.polyfit(death, d, 1)[0]:+.2f} SDs per unit of death share | mean |change| {d.abs().mean():.3f}, max {d.abs().max():.2f}")
say("\nReading: a negative slope for batters and a positive slope for bowlers means the pooled bins rewarded death-over hitting and penalised death-over bowling relative to the phase-aware cells.")
shutil.rmtree(tmp, ignore_errors=True)
open(os.path.join(O, "logs", "E8_phase_aware_bins.log"), "w").write("\n".join(log))
