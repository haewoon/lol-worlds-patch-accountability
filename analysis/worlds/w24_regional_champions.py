"""W24: champions of the four major regional leagues and how differently the Worlds patch reached them.

Regional champion = winner of the last domestic playoff series before that year's Worlds start (manifest), in
LCK, LPL, LEC (EU LCS before 2019), and LCS (LTA in 2025): the consecutive final games between the two teams of the
last playoff game, whatever their UTC dates (the 2020 LCS final, TSM 3-2 FlyQuest, crossed midnight UTC). Games
Oracle's Elixir lacks come from the Leaguepedia supplement (w0), e.g. the 2017 LPL summer playoffs.
The derived list must equal data/manifests/regional_champions.csv; `--write-manifest` rewrites that file (runs with
another LOL_WORLDS_OUT never touch it). Writes OUT/regional_champion_dispersion.json.
Dispersion: within-year range of net exposure (z) among the four champions vs four randomly drawn entrants.
"""
import json
import sys

import numpy as np
import pandas as pd

from worlds_common import OUT, ROOT, event_manifest, load_matches

LEAGUE_REGION = {"LCK": "LCK", "LPL": "LPL", "LEC": "LEC", "EU LCS": "LEC", "LCS": "LCS", "NA LCS": "LCS", "LTA": "LCS"}
rng = np.random.default_rng(24)
ev = event_manifest()

rows = []
for y in ev.index:
    d = load_matches(["gameid", "league", "date", "position", "teamid", "teamname", "result", "playoffs", "datacompleteness"],
                     years=[y])
    d = d[(d.position == "team") & d.league.isin(LEAGUE_REGION) & (d.playoffs == 1)]
    d["date"] = pd.to_datetime(d.date)
    d = d[d.date < ev.loc[y, "start"]]
    for league, g in d.groupby("league"):
        games = g.sort_values("date").groupby("date", sort=True)             # one game = two team rows
        pairs = [(frozenset(x.teamid), x) for _, x in games]
        final, series = pairs[-1][0], []
        for pair, x in reversed(pairs):                                       # walk back through the final series
            if pair != final:
                break
            series.append(x)
        wins = pd.concat(series).groupby(["teamid", "teamname"]).result.sum()
        assert wins.max() > wins.min(), (y, league, wins)
        team_id, team = wins.idxmax()
        region = LEAGUE_REGION[league]
        src = "Leaguepedia supplement" if pd.concat(series).datacompleteness.eq("leaguepedia").any() else "Oracle's Elixir"
        source = f"{src}: winner of the last domestic playoff series before Worlds ({wins.max()}-{wins.min()})"
        rows.append({"year": y, "region": region, "league": league, "team": team, "teamid": team_id, "source": source})
R = pd.DataFrame(rows).sort_values(["year", "region"]).reset_index(drop=True)
assert R.groupby("year").size().eq(4).all(), R.groupby("year").size()
MANIFEST = ROOT / "data" / "manifests" / "regional_champions.csv"
if "--write-manifest" in sys.argv:
    R.to_csv(MANIFEST, index=False)
old = pd.read_csv(MANIFEST)
assert old[["year", "region", "teamid"]].equals(R[["year", "region", "teamid"]]), \
    "derived regional champions differ from the manifest; rerun with --write-manifest after checking"

E = pd.read_csv(OUT / "team_patch_exposure.csv").dropna(subset=["buff_share"])
M = R.merge(E[["year", "teamid", "intended_z", "buff_share", "nerf_share"]], on=["year", "teamid"], how="left")
assert M.intended_z.notna().all(), M[M.intended_z.isna()]
obs = M.groupby("year").intended_z.agg(np.ptp)
sims = np.array([np.median([np.ptp(rng.choice(g.intended_z.to_numpy(), 4, replace=False)) for _, g in E.groupby("year")])
                 for _ in range(10000)])
S = {"median_range_net_z": float(obs.median()), "range_net_z_by_year": obs.round(3).to_dict(),
     "random4_median_range_mean": float(sims.mean()), "random4_median_range_90": np.quantile(sims, [.05, .95]).tolist(),
     "p_random_le_observed": float((sims <= obs.median()).mean()),
     "nerf_share_range_pp_by_year": (M.groupby("year").nerf_share.agg(np.ptp) * 100).round(1).to_dict(),
     "extremes": M.loc[M.groupby("year").intended_z.idxmin().tolist() + M.groupby("year").intended_z.idxmax().tolist(),
                       ["year", "region", "team", "intended_z", "nerf_share"]].sort_values("year").round(3).to_dict("records")}
T = pd.read_csv(OUT / "buff_persistence_teams.csv")                            # meta concentration (w17)
MT = R.merge(T[["year", "teamid", "meta_z", "intended_z"]], on=["year", "teamid"])
assert len(MT) == len(R)
S.update({"meta_z_min": float(MT.meta_z.min()), "meta_z_max": float(MT.meta_z.max()),
          "corr_meta_net_regional": float(MT.meta_z.corr(MT.intended_z)), "n_regional": len(MT),
          "corr_meta_net_all": float(T.meta_z.corr(T.intended_z))})
(OUT / "regional_champion_dispersion.json").write_text(json.dumps(S, indent=2) + "\n")
print(json.dumps({k: v for k, v in S.items() if k not in ("extremes", "range_net_z_by_year", "nerf_share_range_pp_by_year")}, indent=2))
