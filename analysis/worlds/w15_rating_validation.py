"""W15: how accurate is the pre-Worlds rating (w1) on Worlds results?

Game level      accuracy / log loss / Brier vs coin flip, blue side only and region only (the model's own
                league effects); calibration by predicted-probability bin and calibration slope; per year.
Out of sample   w1 chose half-life and team penalty on pooled Worlds log loss (in-sample). Here the grid is
                extended to larger penalties and chosen leave-one-year-out.
Series level    series winners (Bo1 / Bo3 / Bo5) from the game probabilities.
Region bias     actual - predicted wins in cross-region games, by region (series-cluster bootstrap).
Tournament      rank correlation of rating with final placement; champion's rating rank.
Bradley-Terry fit copied from w1 (same window, home league, penalties).
"""
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
from worlds_common import INTL, OUT, assign_series, load_matches
rng = np.random.default_rng(15)
GRID_H, GRID_LAM = [120, 240, 480], [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
W1 = (240, 8.0)                                      # w1's in-sample choice

W = pd.read_parquet(OUT / "worlds_games_pred.parquet").sort_values("date").reset_index(drop=True)
ev = pd.read_csv(OUT / "worlds_events.csv", index_col=0, parse_dates=["start"]).dropna()
X = pd.read_csv(OUT / "team_exposure.csv")[["year", "teamid", "team", "major"]]
K = pd.read_csv(OUT / "knockout_patch_exposure.csv")[["year", "teamid", "finish"]]
raw = load_matches(["gameid", "league", "date", "side", "position", "teamid", "result"],
                   years=set(W.year) | {min(W.year) - 1})
raw["date"] = pd.to_datetime(raw.date, errors="coerce")
tm = raw[raw.position == "team"].dropna(subset=["teamid", "date", "result"]).drop_duplicates(["gameid", "side"])
_sum = tm.groupby("gameid").result.agg(["sum", "size"])
assert ((_sum["size"] != 2) | (_sum["sum"] == 1)).all(), "a game without exactly one winner reached the ratings"
blue, red = tm[tm.side == "Blue"].set_index("gameid"), tm[tm.side == "Red"].set_index("gameid")
G = blue[["league", "date", "teamid", "result"]].join(red[["teamid"]].rename(columns={"teamid": "red_id"}), how="inner")
G = G.rename(columns={"teamid": "blue_id"}).reset_index()
G = G[G.blue_id != G.red_id]


def fit_year(y, half_life, lam_team, lam_league=0.05):
    T = ev.loc[y, "start"]
    d = G[(G.date < T) & (G.date >= T - pd.Timedelta(days=365))]
    w = 0.5 ** ((T - d.date).dt.total_seconds().to_numpy() / 86400 / half_life)
    long = pd.concat([d[["blue_id", "league"]].rename(columns={"blue_id": "t"}), d[["red_id", "league"]].rename(columns={"red_id": "t"})])
    home = long[~long.league.isin(INTL)].groupby("t").league.agg(lambda s: s.value_counts().index[0])
    wy = W[W.year == y]
    teams = pd.Index(sorted(set(d.blue_id) | set(d.red_id) | set(wy.blue_id) | set(wy.red_id)))
    home = home.reindex(teams).fillna("UNKNOWN")
    leagues = pd.Index(sorted(home.unique()))
    tl, nL, nT = leagues.get_indexer(home.to_numpy()), len(leagues), len(teams)
    bi, ri, yv = teams.get_indexer(d.blue_id), teams.get_indexer(d.red_id), d.result.to_numpy(float)

    def nll(th):
        a, L, t = th[0], th[1:1 + nL], th[1 + nL:]
        s = L[tl] + t
        z = a + s[bi] - s[ri]
        g = w * (yv - 1 / (1 + np.exp(-z)))
        gs = np.bincount(bi, g, nT) - np.bincount(ri, g, nT)
        f = -(np.sum(w * (yv * z - np.logaddexp(0, z))) - 0.5 * lam_team * t @ t - 0.5 * lam_league * L @ L)
        return f, -np.concatenate([[g.sum()], np.bincount(tl, gs, nL) - lam_league * L, gs - lam_team * t])

    x = minimize(nll, np.zeros(1 + nL + nT), jac=True, method="L-BFGS-B", options={"maxiter": 5000}).x
    a, L, t = x[0], x[1:1 + nL], x[1 + nL:]
    return a, pd.Series(L[tl] + t, index=teams), pd.Series(L[tl], index=teams)


def metrics(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"n": len(y), "acc": np.mean((p > 0.5) == (y == 1)),
            "logloss": -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)), "brier": np.mean((p - y) ** 2)}


# ---------- grid of ratings, per year ----------
P, STRENGTH = {}, {}
for H in GRID_H:
    for lam in GRID_LAM:
        cols, full = [], []
        for y in sorted(W.year.unique()):
            a, s, lg = fit_year(y, H, lam)
            wy = W[W.year == y]
            cols.append(pd.DataFrame({"gameid": wy.gameid,
                                      "full": a + s.reindex(wy.blue_id).to_numpy() - s.reindex(wy.red_id).to_numpy(),
                                      "region": a + lg.reindex(wy.blue_id).to_numpy() - lg.reindex(wy.red_id).to_numpy(),
                                      "side": a}))
            full.append(pd.DataFrame({"year": y, "teamid": s.index, "s": s.to_numpy()}))
        P[(H, lam)] = pd.concat(cols).set_index("gameid").reindex(W.gameid)
        STRENGTH[(H, lam)] = pd.concat(full, ignore_index=True)
    print("half-life", H, "done")
sig = lambda z: 1 / (1 + np.exp(-z))
yv = W.result.to_numpy()
known = ((W.blue_npre >= 10) & (W.red_npre >= 10)).to_numpy()
print(f"w1 refit reproduces the pipeline: max |diff| = {np.abs(P[W1].full.to_numpy() - W.pred_logit.to_numpy()).max():.2e}")

# ---------- game level ----------
pw1 = sig(P[W1].full.to_numpy())
base = [{"model": "Coin flip", **metrics(np.full(len(W), 0.5), yv), "acc": 0.5},
        {"model": "Blue side only", **metrics(sig(P[W1].side.to_numpy()), yv)},
        {"model": "League effects only", **metrics(sig(P[W1].region.to_numpy()), yv)},
        {"model": "Rating (hyper-parameters chosen on all years)", **metrics(pw1, yv)}]
grid = pd.DataFrame([{"half_life": H, "lam_team": lam, **metrics(sig(P[(H, lam)].full.to_numpy())[known], yv[known])}
                     for H, lam in P])
# leave-one-year-out choice of (half-life, penalty)
loyo = np.empty(len(W))
chosen = {}
for y in sorted(W.year.unique()):
    tr, te = (W.year != y).to_numpy() & known, (W.year == y).to_numpy()
    best = min(P, key=lambda k: metrics(sig(P[k].full.to_numpy())[tr], yv[tr])["logloss"])
    chosen[y] = best
    loyo[te] = sig(P[best].full.to_numpy())[te]
base.append({"model": "Rating (hyper-parameters chosen leave-one-year-out)", **metrics(loyo, yv)})
bestall = min(P, key=lambda k: metrics(sig(P[k].full.to_numpy())[known], yv[known])["logloss"])
base.append({"model": f"Rating (best on the full grid, H={bestall[0]}, lambda={bestall[1]:g})", **metrics(sig(P[bestall].full.to_numpy()), yv)})
GL = pd.DataFrame(base)
GL.to_csv(OUT / "rating_validation_game.csv", index=False)
grid.to_csv(OUT / "rating_validation_grid.csv", index=False)
print("\n", GL.round(4).to_string())
print("\nleave-one-year-out choices:", {y: (int(h), l) for y, (h, l) in chosen.items()})
print("\ngrid (known teams):\n", grid.pivot(index="half_life", columns="lam_team", values="logloss").round(4).to_string())

m = sm.Logit(W.result, sm.add_constant(W.pred_logit)).fit(disp=0)
print(f"\ncalibration: intercept {m.params['const']:+.3f}, slope {m.params['pred_logit']:.3f} "
      f"(95% {m.conf_int().loc['pred_logit', 0]:.2f}–{m.conf_int().loc['pred_logit', 1]:.2f}; 1 = well calibrated, <1 = overconfident)")
fav_p = np.maximum(pw1, 1 - pw1)
fav_won = np.where(pw1 >= 0.5, yv, 1 - yv)
bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0]
cb = pd.DataFrame({"bin": pd.cut(fav_p, bins, right=False, include_lowest=True), "p": fav_p, "won": fav_won})
CAL = cb.groupby("bin", observed=True).agg(n=("won", "size"), predicted=("p", "mean"), actual=("won", "mean"))
z = 1.96
CAL["lo"] = (CAL.actual + z * z / (2 * CAL.n) - z * np.sqrt(CAL.actual * (1 - CAL.actual) / CAL.n + z * z / (4 * CAL.n ** 2))) / (1 + z * z / CAL.n)
CAL["hi"] = (CAL.actual + z * z / (2 * CAL.n) + z * np.sqrt(CAL.actual * (1 - CAL.actual) / CAL.n + z * z / (4 * CAL.n ** 2))) / (1 + z * z / CAL.n)
CAL.to_csv(OUT / "rating_validation_calibration.csv")
print("\n", CAL.round(3).to_string())
PY = pd.DataFrame([{"year": y, **metrics(pw1[(W.year == y).to_numpy()], yv[(W.year == y).to_numpy()])} for y in sorted(W.year.unique())])
print("\n", PY.round(3).to_string(index=False))

# ---------- series level ----------
W["p"] = pw1
W["pair"] = [tuple(sorted(p)) for p in zip(W.blue_id, W.red_id)]
W["series"], _ = assign_series(W)
W["a_team"] = [p[0] for p in W.pair]
W["p_a"] = np.where(W.blue_id == W.a_team, W.p, 1 - W.p)
W["a_won"] = np.where(W.blue_id == W.a_team, W.result, 1 - W.result)
SR = W.groupby("series").agg(year=("year", "first"), games=("a_won", "size"), a_wins=("a_won", "sum"), q=("p_a", "mean"))
SR["k"] = np.maximum(SR.a_wins, SR.games - SR.a_wins)                     # wins needed: 1 = Bo1, 2 = Bo3, 3 = Bo5
assert (SR.a_wins * 2 != SR.games).all(), "a completed series cannot end level"
SR["format"] = SR.k.map({1: "Bo1", 2: "Bo3", 3: "Bo5"})
SR["p_series"] = [sum(comb(k - 1 + j, j) * q ** k * (1 - q) ** j for j in range(k)) for q, k in zip(SR.q, SR.k)]
SR["a_series_won"] = (SR.a_wins == SR.k).astype(int)
last7 = SR.groupby("year").tail(7).index
SR["stage"] = np.where(SR.index.isin(last7), "knockouts", "before knockouts")
SER = pd.concat([pd.DataFrame([{"group": g, **metrics(d.p_series.to_numpy(), d.a_series_won.to_numpy())}])
                 for g, d in [("all", SR), *[(f, d) for f, d in SR.groupby("format")], *[(s, d) for s, d in SR.groupby("stage")]]])
SER.to_csv(OUT / "rating_validation_series.csv", index=False)
print("\n", SER.round(3).to_string(index=False))

# ---------- region bias in cross-region games ----------
reg = X.set_index(["year", "teamid"]).major
W["rb"] = reg.reindex(list(zip(W.year, W.blue_id))).to_numpy()
W["rr"] = reg.reindex(list(zip(W.year, W.red_id))).to_numpy()
C = W[W.rb != W.rr]
long = pd.concat([pd.DataFrame({"series": C.series, "region": C.rb, "res": C.result - C.p}),
                  pd.DataFrame({"series": C.series, "region": C.rr, "res": (1 - C.result) - (1 - C.p)})])
rows = []
for r, d in long.groupby("region"):
    ser = d.groupby("series").res.agg(["sum", "size"])
    bs = [ser.loc[rng.choice(ser.index, len(ser))].pipe(lambda b: b["sum"].sum() / b["size"].sum()) for _ in range(2000)]
    rows.append({"region": r, "games": len(d), "actual_minus_pred_pp": d.res.mean() * 100,
                 "lo": np.quantile(bs, .025) * 100, "hi": np.quantile(bs, .975) * 100})
REG = pd.DataFrame(rows).sort_values("actual_minus_pred_pp", ascending=False)
REG.to_csv(OUT / "rating_validation_region.csv", index=False)
print("\n", REG.round(2).to_string(index=False))

# ---------- tournament level ----------
ST = STRENGTH[W1]
tr_rows, sp = [], []
for y, wy in W.groupby("year"):
    teams = sorted(set(wy.blue_id) | set(wy.red_id))
    s = ST[ST.year == y].set_index("teamid").s.reindex(teams)
    fin = K[K.year == y].set_index("teamid").finish.reindex(teams).fillna("Before quarterfinals")
    tier = fin.map({"Champion": 1, "Runner-up": 2, "SF": 3, "QF": 4, "Before quarterfinals": 5})
    rk = s.rank(ascending=False)
    rho = spearmanr(-rk, -tier).statistic
    sp.append(rho)
    top = rk.idxmin()
    champ = fin[fin == "Champion"].index[0]
    tr_rows.append({"year": y, "teams": len(teams), "spearman_rating_vs_finish": rho,
                    "top_rated": X.set_index(["year", "teamid"]).team.get((y, top)), "top_rated_finish": fin[top],
                    "champion": X.set_index(["year", "teamid"]).team.get((y, champ)), "champion_rating_rank": rk[champ],
                    "qf_teams_in_top8_rating": int(((tier <= 4) & (rk <= 8)).sum())})
TR = pd.DataFrame(tr_rows)
TR.to_csv(OUT / "rating_validation_tournament.csv", index=False)
print("\n", TR.round(2).to_string(index=False))
print(f"\nmean Spearman {np.mean(sp):.2f}; champion rating rank: mean {TR.champion_rating_rank.mean():.1f}, "
      f"#1 in {int((TR.champion_rating_rank == 1).sum())}/10, top-3 in {int((TR.champion_rating_rank <= 3).sum())}/10; "
      f"QF teams among top-8 rated: {TR.qf_teams_in_top8_rating.mean():.1f}/8")

# ---------- does the region bias move the ③ estimates? add region-difference dummies to each model ----------
E = pd.read_csv(OUT / "team_patch_exposure.csv")
REGS = ["LCK", "LPL", "LEC", "LCS"]                                      # Other = reference


def region_diff(df, team_col, opp_col):
    r1 = reg.reindex(list(zip(df.year, df[team_col]))).to_numpy()
    r2 = reg.reindex(list(zip(df.year, df[opp_col]))).to_numpy()
    return pd.DataFrame({f"r_{g}": (r1 == g).astype(float) - (r2 == g).astype(float) for g in REGS}, index=df.index)


ei = E.set_index(["year", "teamid"])[["intended_z", "nerfs_z", "buffs_z"]]
D8 = W.copy()
for f in ei.columns:
    D8["d_" + f] = ei[f].reindex(list(zip(D8.year, D8.blue_id))).to_numpy() - ei[f].reindex(list(zip(D8.year, D8.red_id))).to_numpy()
D8 = D8.dropna(subset=["d_intended_z"]).reset_index(drop=True)
D8 = D8.join(region_diff(D8, "blue_id", "red_id"))
TG = pd.read_csv(OUT / "ban_pressure_team_games.csv")
F = TG[TG.has_pool & (TG.game_no == 1)]
BL = F.merge(F[["gameid", "T", "pull", "other_pull", "comfort"]].rename(columns={"T": "O"}), on=["gameid", "O"], suffixes=("", "_op"))
BL = BL.drop_duplicates("gameid").dropna(subset=["pull", "pull_op", "other_pull", "other_pull_op", "comfort", "comfort_op"]).reset_index(drop=True)
BL["d_pull"] = (BL.pull - BL.pull_op) / (BL.pull - BL.pull_op).std()
BL = BL.join(region_diff(BL, "T", "O"))
S10 = pd.read_csv(OUT / "stage_team_games.csv")
G10 = S10[S10.pred_logit.notna()].drop_duplicates("gameid").reset_index(drop=True)
G10 = G10.join(region_diff(G10, "teamid", "opp"))
RC = [f"r_{g}" for g in REGS]
out = []
for lab, df, yc, terms, key, scale in [("Net exposure", D8, "result", ["d_intended_z"], "d_intended_z", 1.0),
                                      ("Buff exposure", D8, "result", ["d_nerfs_z", "d_buffs_z"], "d_buffs_z", 1.0),
                                      ("Excess bans", BL, "win", ["d_pull"], "d_pull", 1.0),
                                      ("Pick advantage", G10, "win", ["pick_adv"], "pick_adv", S10.pick_adv.std())]:
    yrs = df.year.unique()
    for ctrl_lab, ctrl in [("Rating", ["pred_logit"]), ("Rating + region dummies", ["pred_logit"] + RC)]:
        f = lambda d: sm.Logit(d[yc], sm.add_constant(d[ctrl + terms])).fit(disp=0).params[key] * scale / 4 * 100
        bs = np.array([f(pd.concat([df[df.year == yy] for yy in rng.choice(yrs, len(yrs))])) for _ in range(1000)])
        out.append({"row": lab, "control": ctrl_lab, "winprob_pp": f(df), "lo": np.quantile(bs, .025), "hi": np.quantile(bs, .975)})
RB = pd.DataFrame(out)
RB.to_csv(OUT / "rating_validation_region_control.csv", index=False)
print("\n", RB.round(2).to_string(index=False))
