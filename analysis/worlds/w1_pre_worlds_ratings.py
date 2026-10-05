"""W1: pre-Worlds expected win probability for every Worlds main-event game.

For each Worlds year Y (2014-2025):
  T      = first day of the main event (games on the Worlds main patch)
  data   = every game in the dataset in [T - 365d, T), all leagues incl. international
  model  = Bradley-Terry logit with a blue-side term and
           strength(team) = league_effect(home league) + team_deviation
           league effects are linked across regions by international games
           (MSI, previous Worlds, IEM, EWC, First Stand ...).
           Games are down-weighted by age with half-life H.
  output = p(blue wins) for each Worlds main-event game, using pre-T data only.
Hyper-parameters (H, team penalty) are chosen by pooled Worlds log-loss.
Years 2014-15 are dropped: most Worlds teams have <10 prior games in the source data.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from worlds_common import INTL, OUT, event_manifest, load_matches, select_worlds

ROOT = Path(__file__).resolve().parents[2]
YEARS = range(2016, 2026)  # 2014-15: too little domestic data in the source

cols = ["gameid", "league", "year", "date", "patch", "game", "side", "position", "teamid", "teamname",
        "result", "golddiffat15"]
raw = load_matches(cols, dtype={"patch": str})
raw["date"] = pd.to_datetime(raw.date, errors="coerce")
tm = raw[raw.position == "team"].dropna(subset=["teamid", "date", "result"]).drop_duplicates(["gameid", "side"])
_sum = tm.groupby("gameid").result.agg(["sum", "size"])
assert ((_sum["size"] != 2) | (_sum["sum"] == 1)).all(), "a game without exactly one winner reached the ratings"
blue, red = tm[tm.side == "Blue"].set_index("gameid"), tm[tm.side == "Red"].set_index("gameid")
G = blue[["league", "year", "date", "patch", "game", "teamid", "teamname", "result", "golddiffat15"]].join(
    red[["teamid", "teamname"]].rename(columns={"teamid": "red_id", "teamname": "red_name"}), how="inner")
G = G.rename(columns={"teamid": "blue_id", "teamname": "blue_name", "golddiffat15": "gd15"}).reset_index()
G = G[G.blue_id != G.red_id]

# Explicit event bounds exclude the 2023 WQS from outcomes while leaving it in G.
W = select_worlds(G)
events = event_manifest()
T0 = events.start
events.to_csv(OUT / "worlds_events.csv")


def fit_year(y, half_life, lam_team, lam_league=0.05):
    T = T0[y]
    d = G[(G.date < T) & (G.date >= T - pd.Timedelta(days=365))]
    age = (T - d.date).dt.total_seconds() / 86400
    w = 0.5 ** (age / half_life)
    # home league = most frequent non-international league of the team in the window
    long = pd.concat([d[["blue_id", "league"]].rename(columns={"blue_id": "t"}),
                      d[["red_id", "league"]].rename(columns={"red_id": "t"})])
    dom = long[~long.league.isin(INTL)]
    home = dom.groupby("t").league.agg(lambda s: s.value_counts().index[0])
    teams = pd.Index(sorted(set(d.blue_id) | set(d.red_id) | set(W[W.year == y].blue_id) |
                            set(W[W.year == y].red_id)))
    home = home.reindex(teams).fillna("UNKNOWN")
    leagues = pd.Index(sorted(home.unique()))
    tl = leagues.get_indexer(home.to_numpy())
    bi, ri = teams.get_indexer(d.blue_id), teams.get_indexer(d.red_id)
    yv, wv = d.result.to_numpy(float), w.to_numpy()
    nT, nL = len(teams), len(leagues)

    def nll(th):
        a, L, t = th[0], th[1:1 + nL], th[1 + nL:]
        s = L[tl] + t
        z = a + s[bi] - s[ri]
        p = 1 / (1 + np.exp(-z))
        ll = np.sum(wv * (yv * z - np.logaddexp(0, z)))
        g = wv * (yv - p)
        gs = np.bincount(bi, g, nT) - np.bincount(ri, g, nT)
        gL = np.bincount(tl, gs, nL)
        f = -(ll - 0.5 * lam_team * t @ t - 0.5 * lam_league * L @ L)
        return f, -np.concatenate([[g.sum()], gL - lam_league * L, gs - lam_team * t])

    res = minimize(nll, np.zeros(1 + nL + nT), jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000})
    a, L, t = res.x[0], res.x[1:1 + nL], res.x[1 + nL:]
    s = pd.Series(L[tl] + t, index=teams)
    n_pre = pd.concat([d.blue_id, d.red_id]).value_counts().reindex(teams).fillna(0)
    return a, s, home, n_pre


def predict(half_life, lam_team):
    out = []
    for y in YEARS:
        a, s, home, n_pre = fit_year(y, half_life, lam_team)
        w = W[W.year == y].copy()
        w["pred_logit"] = a + s.reindex(w.blue_id).to_numpy() - s.reindex(w.red_id).to_numpy()
        w["blue_home"], w["red_home"] = home.reindex(w.blue_id).to_numpy(), home.reindex(w.red_id).to_numpy()
        w["blue_npre"], w["red_npre"] = n_pre.reindex(w.blue_id).to_numpy(), n_pre.reindex(w.red_id).to_numpy()
        out.append(w)
    w = pd.concat(out)
    w["p_blue"] = 1 / (1 + np.exp(-w.pred_logit))
    return w


grid = []
for H in [120, 240, 480]:
    for lt in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:           # same grid as w15
        w = predict(H, lt)
        ok = (w.blue_npre >= 10) & (w.red_npre >= 10)
        p = w.p_blue.clip(1e-6, 1 - 1e-6)
        ll = -np.mean(w.result * np.log(p) + (1 - w.result) * np.log(1 - p))
        ll_ok = -np.mean((w.result * np.log(p) + (1 - w.result) * np.log(1 - p))[ok])
        grid.append({"half_life": H, "lam_team": lt, "logloss": ll, "logloss_known_teams": ll_ok,
                     "acc": np.mean((w.p_blue > 0.5) == (w.result == 1))})
        print(grid[-1])
grid = pd.DataFrame(grid)
grid.to_csv(OUT / "rating_grid.csv", index=False)
best = grid.sort_values("logloss_known_teams").iloc[0]
w = predict(best.half_life, best.lam_team)
w.to_parquet(OUT / "worlds_games_pred.parquet")
base = -np.log(0.5)
print(f"best: H={best.half_life} lam_team={best.lam_team} logloss={best.logloss:.4f} (coin flip {base:.4f})")
print(w.groupby("year").apply(lambda d: pd.Series({
    "games": len(d), "acc": np.mean((d.p_blue > 0.5) == (d.result == 1)),
    "logloss": -np.mean(d.result * np.log(d.p_blue) + (1 - d.result) * np.log(1 - d.p_blue)),
    "teams_lt10_pre": int(((d.blue_npre < 10) | (d.red_npre < 10)).sum())})).round(3).to_string())
