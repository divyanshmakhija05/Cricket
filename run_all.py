"""
Run the whole pipeline in the right order with one command.

    python run_all.py [data_dir] [output_dir] [--fast] [--core-only]

  data_dir     folder with the 6 data-pack CSVs          (default: ./data)
  output_dir   where tables / logs / charts are written  (default: ./output)
  --fast       skip the 3-run stress test in stage 02 and use 100 (not 500) bootstrap resamples in stage 07 (final numbers use the full run)
  --core-only  run only the numbered pipeline stages (skip the evidence/ scripts)

Stages (src/):       01 ball-state -> 02 chase pressure -> 03 first-innings pressure -> 04 audit -> 05 value added -> 06 ratings -> 07 robustness/bootstrap -> 08 final lists + charts -> 09 audit
Evidence (evidence/): E1 model selection -> E2 run-rate alternatives -> E3 venue par & composite index -> E4 weighting variants
Stops immediately if a step fails.
"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
D = os.path.abspath(args[0]) if len(args) > 0 else os.path.join(HERE, "data")
O = os.path.abspath(args[1]) if len(args) > 1 else os.path.join(HERE, "output")
fast, core = "--fast" in sys.argv, "--core-only" in sys.argv
if not os.path.exists(os.path.join(D, "ball_by_ball.csv")):
    sys.exit(f"Cannot find ball_by_ball.csv in {D}. Put the 6 data CSVs there, or pass the folder as the first argument.")
os.makedirs(O, exist_ok=True)
steps = [
    ("01  build ball-state table",            "src/01_build_ball_state.py",            [D, O]),
    ("02  chase pressure index (final model)", "src/02_pressure_chase.py",              [O] + (["--no-robust"] if fast else [])),
    ("03  first-innings pressure (secondary)", "src/03_pressure_first_innings.py",      [O]),
    ("04  audit checks",                       "src/04_audit_checks.py",                [D, O]),
    ("05  value added per ball (chases)",     "src/05_value_added.py",                [O]),
    ("05b first-innings value-added layer",    "src/05b_first_innings_va.py",           [O]),
    ("06  player ratings (batters, bowlers)", "src/06_player_ratings.py",            [O]),
]
steps += [("E2  run-rate alternatives (also feeds the stage-07 robustness check)", "evidence/E2_runrate_alternatives.py", [O]),
    ("07  robustness + bootstrap by match",   "src/07_robustness.py",                 [O, "100" if fast else "500"]),
    ("08  final top-10 lists, evidence, charts", "src/08_final_outputs.py",          [O, D]),
    ("09  audit of stages 05-08",             "src/09_audit_ratings.py",              [O, D]),
]
if not core:
    steps += [
    ("E1  model selection evidence",           "evidence/E1_model_selection.py",        [O]),
    ("E3  venue par / composite evidence",     "evidence/E3_venue_par_and_composite.py",[O]),
    ("E4  repeatability of definitions",       "evidence/E4_weighting_variants.py",     [O]),
    ("E5  out-of-sample choice of the estimator", "evidence/E5_out_of_sample_validation.py", [O]),
    ("E7  volume-weighted consistency", "evidence/E7_consistency_weighting.py", [O]),
    ("E8  phase-aware standardisation cells", "evidence/E8_phase_aware_bins.py", [O]),
    ("E6  proposed improvements (wickets, baseline, consistency, first-innings weight)", "evidence/E6_proposed_improvements.py", [O]),
    ]
t0 = time.time()
for name, script, a in steps:
    t = time.time(); print(f"\n{'=' * 72}\n>>> {name}\n{'=' * 72}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, script)] + a, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"\nSTEP FAILED: {name} (exit {r.returncode}). Fix this before continuing.")
    print(f"<<< done in {time.time() - t:.0f}s", flush=True)
print(f"\nALL STEPS COMPLETE in {time.time() - t0:.0f}s.\nOutputs: {O}/tables  {O}/logs  {O}/charts")
