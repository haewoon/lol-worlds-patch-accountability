"""Reproduce the revised manuscript analyses in an explicitly chosen directory.

Example: .venv/bin/python analysis/worlds/run_revised_pipeline.py --out /tmp/lol-repro
Figures and manuscript builds are separate.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
out = args.out.resolve()
if out == root / "output/worlds":
    parser.error("output/worlds is a preserved historical archive")
out.mkdir(parents=True, exist_ok=True)
logs = out / "logs"
logs.mkdir(exist_ok=True)
env = dict(os.environ, LOL_WORLDS_OUT=str(out), MPLCONFIGDIR=str(out / "matplotlib_cache"))
scripts = ["w1_pre_worlds_ratings.py", "w2_exposure.py", "w8_patch_exposure.py",
           "w9_buff_signal_checks.py", "w10_stages.py", "w13_ban_pressure.py",
           "w14_strength_sensitivity.py", "w15_rating_validation.py",
           "w17_buff_persistence.py", "w21_permutation_calibration.py",
           "w16_detectable_effect.py", "w22_manuscript_summary.py", "w24_regional_champions.py",
           "w23_governance_descriptives.py", "w27_role_exposure.py", "w28_roster_scan.py", "w29_buff_sources.py"]
for script in scripts:
    print(f"Running {script}; log: {logs / (script + '.log')}", flush=True)
    with (logs / (script + ".log")).open("w") as log:
        subprocess.run([sys.executable, "-u", str(root / "analysis/worlds" / script)],
                       cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
print(f"Finished; revised analyses in {out}")
