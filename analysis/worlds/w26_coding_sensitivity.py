"""W26: do the manuscript results survive recoding toward the game data (w25)?

Three alternative codings of the pre-Worlds champion changes; rows without comparable game data (hotfixes, and
patches 10.17 and 10.18, whose own or preceding snapshot is missing) keep the LLM code in A and B.
  A  contradictions   rows whose numbers moved only up (down) but were not coded buff (nerf) become buff (nerf);
                      a flipped sign keeps its magnitude, a formerly unsigned code gets magnitude 1
  B  game-data        every comparable row takes its direction from the numbers: up -> buff, down -> nerf,
     directions       mixed / reshaped / no numeric change -> unsigned (adjust, or the original unsigned code);
                      signed rows keep the LLM magnitude (1 if newly signed); discards changes the files do not
                      record (e.g. passives before 2019), so a stress test, not an LLM-free coding
  C  sign only        LLM directions, every buff +1 and nerf -1
Each variant reruns the whole analysis pipeline (the scripts of run_revised_pipeline.py, then w23) in its own
directory under output/coding_sensitivity/, and the summary compares the manuscript numbers with the main run.
Usage: w26_coding_sensitivity.py [--summarize-only]   (about an hour on a laptop, three variants in parallel)
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from worlds_common import ROOT

MAIN = ROOT / "output/worlds_revision_2026-09-24"
DEST = ROOT / "output/coding_sensitivity"
SCRIPTS = ["w1_pre_worlds_ratings.py", "w2_exposure.py", "w8_patch_exposure.py", "w9_buff_signal_checks.py",
           "w10_stages.py", "w13_ban_pressure.py", "w14_strength_sensitivity.py", "w15_rating_validation.py",
           "w17_buff_persistence.py", "w21_permutation_calibration.py", "w16_detectable_effect.py",
           "w22_manuscript_summary.py", "w24_regional_champions.py", "w23_governance_descriptives.py",
           "w27_role_exposure.py"]
HERE = Path(__file__).resolve().parent
SIGNED = {"buff", "nerf"}

coded = pd.concat([pd.read_csv(f, dtype={"patch_oe": str}) for f in sorted((ROOT / "data/patch_notes").glob("coded_[A-E].csv"))],
                  ignore_index=True)
coded["direction"] = coded.direction.str.lower().str.strip()
gd = pd.read_csv(MAIN / "gamedata_direction_rows.csv", dtype={"patch": str})
gd = gd[gd.source != "unavailable"][["patch", "champion", "data_dir"]]
base = coded.merge(gd, left_on=["patch_oe", "champion"], right_on=["patch", "champion"], how="left").drop(columns="patch")
base.loc[base.hotfix == 1, "data_dir"] = np.nan
assert len(base) == len(coded)


def recode(df, rows, new_dir):
    old = df.loc[rows, "direction"]
    df.loc[rows, "magnitude"] = np.where(old.isin(SIGNED) & (df.loc[rows, "magnitude"] > 0), df.loc[rows, "magnitude"], 1)
    df.loc[rows, "direction"] = new_dir


def variant(name):
    df = base.copy()
    if name == "A_contradictions":
        recode(df, (df.data_dir == "up") & (df.direction != "buff"), "buff")
        recode(df, (df.data_dir == "down") & (df.direction != "nerf"), "nerf")
    elif name == "B_gamedata_directions":
        up, down = (df.data_dir == "up") & (df.direction != "buff"), (df.data_dir == "down") & (df.direction != "nerf")
        flat = df.data_dir.isin(["mixed", "reshaped", "none"]) & df.direction.isin(SIGNED)
        recode(df, up, "buff")
        recode(df, down, "nerf")
        df.loc[flat, "direction"], df.loc[flat, "magnitude"] = "adjust", 0
    elif name == "C_sign_only":
        df.loc[df.direction.isin(SIGNED), "magnitude"] = 1
    changed = int(((df.direction != base.direction) | (df.magnitude != base.magnitude)).sum())
    return df.drop(columns="data_dir"), changed


def run(name):
    out = DEST / name
    (out / "logs").mkdir(parents=True, exist_ok=True)
    table, changed = variant(name)
    table.to_csv(out / "coding.csv", index=False)
    env = dict(os.environ, LOL_WORLDS_OUT=str(out), LOL_CODING_OVERRIDE=str(out / "coding.csv"),
               MPLCONFIGDIR=str(out / "matplotlib_cache"))
    for s in SCRIPTS:
        with (out / "logs" / (s + ".log")).open("w") as log:
            subprocess.run([sys.executable, "-u", str(HERE / s)], cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    return name, changed


VARIANTS = ["A_contradictions", "B_gamedata_directions", "C_sign_only"]
if "--summarize-only" not in sys.argv:
    manifest = ROOT / "data/manifests/regional_champions.csv"
    before = manifest.read_bytes()
    with cf.ThreadPoolExecutor(max_workers=3) as ex:          # the work itself runs in subprocesses
        for name, changed in ex.map(run, VARIANTS):
            print(f"{name}: {changed} coded rows changed; outputs in {DEST / name}", flush=True)
    assert manifest.read_bytes() == before, "w24 rewrote the regional-champion manifest differently"


# ---------- compare the manuscript numbers ----------
def numbers(d):
    desc = json.loads((d / "manuscript_descriptives.json").read_text())
    mech = pd.read_csv(d / "buff_persistence_mechanism.csv").set_index(["dep", "term"])
    reg = json.loads((d / "regional_champion_dispersion.json").read_text())
    ban = pd.read_csv(d / "ban_pressure_targeting.csv")
    b1 = ban[ban.model == "B1 relief: other pool x buff exposure"].set_index("term")
    g1 = ban[ban.model == "B1g1 series game 1"].set_index("term")
    pm = pd.read_csv(d / "patch_models.csv").set_index(["model", "term"])
    gov = json.loads((d / "governance_framing/summary.json").read_text())
    hb = pd.read_csv(d / "governance_framing/highest_buff_quarterfinalists.csv")
    E = pd.read_csv(d / "team_patch_exposure.csv").dropna(subset=["intended"])
    low = gov["lowest_net_quarterfinalists_finishes"]
    lq = pd.read_csv(d / "governance_framing/lowest_net_quarterfinalists.csv")
    mde = pd.read_csv(d / "detectable_effect_summary.csv").set_index("scope")
    cal = pd.read_csv(d / "permutation_null_calibration.csv")
    cal = cal[(cal.statistic == "studentized") & (cal.n_sim == cal.n_sim.max())].iloc[0]
    st = pd.read_csv(d / "strength_sensitivity.csv")
    st = st[st.row == st.row.iloc[0]]                      # net exposure under each strength control
    return {
        "nerf share of pre-Worlds picks (%)": desc["nerf_share_mean_pct"],
        "buff share of pre-Worlds picks (%)": desc["buff_share_mean_pct"],
        "within-year nerf-share gap, median (pp)": desc["nerf_share_median_range_pp"],
        "within-year buff-share gap, median (pp)": desc["buff_share_median_range_pp"],
        "top-decile champions nerfed (%)": desc["top_decile_nerf_pct"],
        "top-decile champions buffed (%)": desc["top_decile_buff_pct"],
        "meta concentration -> net exposure (SD)": mech.loc[("intended_z", "meta_z"), "coef"],
        "  p": mech.loc[("intended_z", "meta_z"), "p"],
        "regional champions: median net-exposure range (SD)": reg["median_range_net_z"],
        "  random four teams, 90% band": "{:.2f}-{:.2f}".format(*reg["random4_median_range_90"]),
        "regional champions: meta-exposure r": reg["corr_meta_net_regional"],
        "ban: buffed champions x exposure (pp)": 100 * b1.loc["pool_buff_x_expo", "Estimate"],
        "  p ": b1.loc["pool_buff_x_expo", "Pr(>|t|)"],
        "ban: other champions x exposure (pp)": 100 * b1.loc["pool_other_x_expo", "Estimate"],
        "  p  ": b1.loc["pool_other_x_expo", "Pr(>|t|)"],
        "ban, series game 1: buffed x exposure (pp)": 100 * g1.loc["pool_buff_x_expo", "Estimate"],
        "win: net exposure (pp per SD)": pm.loc[("intended", "d_intended_z"), "winprob_pp_per_sd"],
        "  event-bootstrap interval (pp)": "{:+.1f} to {:+.1f}".format(25 * pm.loc[("intended", "d_intended_z"), "boot_lo"],
                                                                      25 * pm.loc[("intended", "d_intended_z"), "boot_hi"]),
        "  permutation p": pm.loc[("intended", "d_intended_z"), "p_perm"],
        "win: buff exposure, buff+nerf model (pp per SD)": pm.loc[("nerfs + buffs", "d_buffs_z"), "winprob_pp_per_sd"],
        "  permutation p ": pm.loc[("nerfs + buffs", "d_buffs_z"), "p_perm"],
        "most buff-exposed quarterfinalist: finals / titles": f"{desc['most_buff_exposed_finals']} / {desc['most_buff_exposed_titles']}",
        "  2022-25 most buff-exposed quarterfinalist": ", ".join(f"{r.year % 100:02d} {r.team}" for r in hb[hb.year >= 2022].itertuples()),
        "lowest-net quarterfinalist: finals / titles": f"{low.get('Runner-up', 0) + low.get('Champion', 0)} / {low.get('Champion', 0)}",
        "  2022-25 lowest-net quarterfinalist": ", ".join(f"{r.year % 100:02d} {r.team}" for r in lq[lq.year >= 2022].itertuples()),
        "win: studentized test, null rejection rate": cal.rate,
        "win: 80% detection threshold (pp per game)": mde.loc["game", "mde80"],
        "win: net exposure across strength controls (pp)": "{:+.1f} to {:+.1f}".format(st.winprob_pp.min(), st.winprob_pp.max()),
        "_E": E.set_index(["year", "teamid"]).intended_z,
    }


N = {"main (LLM coding)": numbers(MAIN), **{v: numbers(DEST / v) for v in VARIANTS}}
Z = {k: n.pop("_E") for k, n in N.items()}
corr = {k: float(np.corrcoef(*Z["main (LLM coding)"].align(z, join="inner"))[0, 1]) for k, z in Z.items()}
T = pd.DataFrame(N)
T.loc["team net exposure: r with main coding"] = corr
changed = {"main (LLM coding)": 0, **{v: variant(v)[1] for v in VARIANTS}}
T.loc["coded rows changed (of 1,093)"] = changed
T = T.loc[[T.index[-1]] + list(T.index[:-1])]
T.to_csv(DEST / "coding_sensitivity_summary.csv")
fmt = lambda x: f"{x:.3f}" if isinstance(x, float) and abs(x) < 1 else (f"{x:.1f}" if isinstance(x, float) else str(x))
print(T.map(fmt).to_string())
