"""W2: how exposed was each Worlds team to the Worlds patch/meta?

For team i at Worlds year Y (T = main-event start):
  pool_i(r, c)  share of team i's picks of champion c in role r, games in [T-100d, T)
  pre(r, c)     role pick shares in the home leagues of all Worlds teams, [T-45d, T)
  wm_-i(r, c)   role pick shares at the Worlds main event, excluding games of team i

  meta_tailwind  A_i = sum_r sum_c pool_i(r,c) * (wm_-i(r,c) - pre(r,c))
                 > 0: the Worlds meta moved TOWARD the champions team i had been playing
                 < 0: the patch/meta moved away from team i's comfort picks
  coverage_change dC_i = mean_r [ sum_c wm_-i(r,c) * 1{i played c in r >= 2 times}
                                - sum_c pre(r,c)  * 1{...} ]
                 share of the Worlds meta the team had practiced, minus the same for the old meta
  patch_gap      number of patches between team i's last official game before T and the Worlds patch
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
from worlds_common import OUT, load_matches
POOL_DAYS, PRE_META_DAYS = 100, 45

cols = ["gameid", "league", "date", "patch", "side", "position", "teamid", "champion"]
raw = load_matches(cols, dtype={"patch": str})
raw["date"] = pd.to_datetime(raw.date, errors="coerce")
role_map = {"top": "top", "jng": "jng", "mid": "mid", "bot": "bot", "sup": "sup",
            "jungle": "jng", "middle": "mid", "adc": "bot", "support": "sup"}
raw["role"] = raw.position.str.lower().map(role_map)
print("positions:", raw.position.value_counts().to_dict())
P = raw[raw.role.notna()].dropna(subset=["teamid", "champion", "date"])
games_team = raw[raw.position == "team"].dropna(subset=["teamid", "date"])

W = pd.read_parquet(OUT / "worlds_games_pred.parquet")
ev = pd.read_csv(OUT / "worlds_events.csv", index_col=0, parse_dates=["start"],
                 dtype={"main_patch": str}).dropna()


def pkey(p):
    a, b = str(p).split(".")
    return int(a) * 100 + int(b)


def shares(df):
    c = df.groupby(["role", "champion"]).size()
    return c / c.groupby(level="role").transform("sum")


rows = []
for y, e in ev.iterrows():
    if y not in set(W.year):
        continue
    T = e.start
    wy = W[W.year == y]
    wgames = set(wy.gameid)
    teams = sorted(set(wy.blue_id) | set(wy.red_id))
    homes = set(wy.blue_home) | set(wy.red_home)
    pre = shares(P[(P.date < T) & (P.date >= T - pd.Timedelta(days=PRE_META_DAYS)) & P.league.isin(homes)])
    wp = P[P.gameid.isin(wgames)]
    for t in teams:
        own_games = set(wy.gameid[(wy.blue_id == t) | (wy.red_id == t)])
        wm = shares(wp[~wp.gameid.isin(own_games)])
        pool_rows = P[(P.teamid == t) & (P.date < T) & (P.date >= T - pd.Timedelta(days=POOL_DAYS))]
        n_pool_games = pool_rows.gameid.nunique()
        tg = games_team[(games_team.teamid == t) & (games_team.date < T)]
        last_patch = tg.sort_values("date").patch.dropna().iloc[-1] if len(tg) else None
        # the same quantities from Oracle's Elixir alone (without games restored from Leaguepedia), for reporting
        oe = tg[~tg.gameid.astype(str).str.startswith("lp:")].sort_values("date").patch.dropna()
        row = {"year": y, "teamid": t, "n_pool_games": n_pool_games,
               "last_patch": last_patch, "worlds_patch": e.main_patch,
               "last_patch_oe_only": oe.iloc[-1] if len(oe) else None,
               "n_pool_games_oe_only": pool_rows[~pool_rows.gameid.astype(str).str.startswith("lp:")].gameid.nunique()}
        if n_pool_games >= 10:
            pool = shares(pool_rows)
            cnt = pool_rows.groupby(["role", "champion"]).size()
            idx = pool.index
            d_meta = wm.reindex(idx).fillna(0) - pre.reindex(idx).fillna(0)
            row["meta_tailwind"] = float((pool * d_meta).sum())
            row["overlap_worlds"] = float((pool * wm.reindex(idx).fillna(0)).sum())
            row["overlap_pre"] = float((pool * pre.reindex(idx).fillna(0)).sum())
            practiced = cnt[cnt >= 2].index
            cov_w = wm.reindex(practiced).fillna(0).sum() / 5
            cov_p = pre.reindex(practiced).fillna(0).sum() / 5
            row.update(coverage_worlds=cov_w, coverage_pre=cov_p, coverage_change=cov_w - cov_p)
        try:
            row["patch_gap"] = pkey(e.main_patch) - pkey(last_patch)
        except Exception:
            row["patch_gap"] = np.nan
        rows.append(row)

X = pd.DataFrame(rows)
names = pd.concat([W[["year", "blue_id", "blue_name", "blue_home"]].set_axis(["year", "teamid", "team", "home"], axis=1),
                   W[["year", "red_id", "red_name", "red_home"]].set_axis(["year", "teamid", "team", "home"], axis=1)]
                  ).drop_duplicates(["year", "teamid"])
X = X.merge(names, on=["year", "teamid"], how="left")
region = {"EU LCS": "LEC", "NA LCS": "LCS", "LTA N": "LCS", "LMS": "PCS", "LTA S": "CBLOL"}
X["region"] = X.home.replace(region)
X["major"] = X.region.where(X.region.isin(["LCK", "LPL", "LEC", "LCS"]), "Other")
X.to_csv(OUT / "team_exposure.csv", index=False)
print(X.describe().round(3).to_string())
print(X.groupby("year")[["meta_tailwind", "coverage_change", "patch_gap"]].agg(["mean", "std"]).round(3).to_string())
print(X[["meta_tailwind", "coverage_change", "patch_gap", "overlap_pre"]].corr().round(2).to_string())
