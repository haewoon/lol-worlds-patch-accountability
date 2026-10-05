"""W9: is the champion 'buff' signal (w8) a patch effect or something else?

 1. Pool breadth: Riot tends to buff weak, rarely played champions, so teams with wide
    pools may catch more buffs mechanically. Breadth = mean over roles of the effective
    number of champions exp(entropy) in the 100 days before Worlds. Re-rank champions on
    buffs residualised on breadth.
 2. Use: did champions actually pick the buffed champions at Worlds, and win with them?
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
from worlds_common import OUT, coding_files, load_matches, sr_only
rng = np.random.default_rng(3)
E = pd.read_csv(OUT / "team_patch_exposure.csv")
K = pd.read_csv(OUT / "knockout_patch_exposure.csv")
ev = pd.read_csv(OUT / "worlds_events.csv", index_col=0, parse_dates=["start"], dtype={"main_patch": str}).dropna()
X = pd.read_csv(OUT / "team_exposure.csv", dtype={"last_patch": str})
W = pd.read_parquet(OUT / "worlds_games_pred.parquet")
P = load_matches(["gameid", "date", "position", "teamid", "champion", "result"])
P["date"] = pd.to_datetime(P.date, errors="coerce")
P = P[P.position.isin(["top", "jng", "mid", "bot", "sup"])]
pre = pd.concat([pd.read_csv(f, dtype={"patch_oe": str}) for f in coding_files()])
pre = sr_only(pre)
pre["direction"] = pre.direction.str.lower().str.strip()
pre["magnitude"] = pd.to_numeric(pre.magnitude, errors="coerce").fillna(0)
pre["score"] = np.where(pre.direction == "buff", pre.magnitude, np.where(pre.direction == "nerf", -pre.magnitude, 0))
pk = lambda p: int(p.split(".")[0]) * 100 + int(p.split(".")[1])
pre["pk"] = pre.patch_oe.map(pk)

rows = []
for _, t in E.iterrows():
    T = ev.loc[t.year, "start"]
    pr = P[(P.teamid == t.teamid) & (P.date < T) & (P.date >= T - pd.Timedelta(days=100))]
    if pr.gameid.nunique() < 10:
        continue
    eff = [np.exp(-(s * np.log(s)).sum()) for s in (g.champion.value_counts(normalize=True) for _, g in pr.groupby("position"))]
    rows.append({"year": t.year, "teamid": t.teamid, "breadth": np.mean(eff)})
E2 = E.merge(pd.DataFrame(rows), on=["year", "teamid"])
E2["breadth_z"] = E2.groupby("year").breadth.transform(lambda s: (s - s.mean()) / s.std())
m = smf.ols("buffs_z ~ breadth_z", data=E2).fit()
E2["buffs_resid"] = m.resid
print(f"corr(buffs_z, breadth_z) = {np.corrcoef(E2.buffs_z, E2.breadth_z)[0, 1]:.3f}")

K2 = K.merge(E2[["year", "teamid", "breadth", "buffs_resid"]], on=["year", "teamid"])
res = []
for col in ["buffs", "breadth", "buffs_resid"]:
    K2["r"] = K2.groupby("year")[col].rank(ascending=False)
    ch = K2[K2.finish == "Champion"]
    ranks = [g.r.to_numpy() for _, g in K2.groupby("year")]
    null = np.array([np.mean([r[rng.integers(len(r))] for r in ranks]) for _ in range(100000)])
    res.append({"measure": col, "champ_mean_rank": ch.r.mean(), "p_one_sided": np.mean(null <= ch.r.mean()),
                **{f"rank_{y}": r for y, r in zip(ch.year, ch.r)}})
res = pd.DataFrame(res)
res.to_csv(OUT / "buff_signal_breadth.csv", index=False)
print(res.round(3).to_string())

use = []
for _, r in K[K.finish == "Champion"].iterrows():
    y, t = r.year, r.teamid
    x = X[(X.year == y) & (X.teamid == t)].iloc[0]
    sy = pre[pre.year_group == y]
    coded = sorted(sy.pk.unique())
    delta = sy[sy.pk > max(pk(x.last_patch), coded[0] - 1)].groupby("champion").score.sum()
    wg = set(W[(W.year == y) & ((W.blue_id == t) | (W.red_id == t))].gameid)
    wp = P[P.gameid.isin(wg) & (P.teamid == t)].assign(d=lambda d: d.champion.map(delta).fillna(0))
    b, o = wp[wp.d > 0], wp[wp.d <= 0]
    use.append({"year": y, "team": r.team, "worlds_picks": len(wp), "buffed_picks": len(b),
                "share_buffed": len(b) / len(wp), "wr_buffed": b.result.mean() if len(b) else np.nan,
                "wr_other": o.result.mean(),
                "buffed_champions": ", ".join(f"{c}x{n}" for c, n in b.champion.value_counts().items())})
use = pd.DataFrame(use)
use.to_csv(OUT / "champion_buffed_pick_use.csv", index=False)
print(use.round(3).to_string())
