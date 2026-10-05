"""W14: does the result level (Figure 6 ③) depend on how pre-Worlds strength is measured?

w1 rates teams on every game in the 365 days before Worlds (half-life 240 d). Alternatives:
  season    league effects from the same 1-year fit (international games link the regions),
            team deviations re-estimated on the last 90 days only (summer split, playoffs, qualifiers)
  90d only  league effects AND team deviations from the last 90 days (almost no cross-region games
            in that window -> regions barely linked; kept to show why the hybrid is needed)
  none      no strength control
For each: prediction quality on the Worlds main-event games, correlation of the patch exposures with
strength / recent form, and the four ③ estimates re-fitted with that control (same specs as w8, w13, w10).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
from worlds_common import INTL, OUT, load_matches
rng = np.random.default_rng(14)
SEASON = 90
N_BOOT, N_PERM = 2000, 1000

W = pd.read_parquet(OUT / "worlds_games_pred.parquet").sort_values("date").reset_index(drop=True)
ev = pd.read_csv(OUT / "worlds_events.csv", index_col=0, parse_dates=["start"]).dropna()
best = pd.read_csv(OUT / "rating_grid.csv").sort_values("logloss_known_teams").iloc[0]
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


def fit(d, teams, tl, nL, w, lam_team, lam_league=0.05, L0=None):
    """Bradley-Terry as in w1; with L0 the league effects are held fixed and only side + team deviations move."""
    bi, ri = teams.get_indexer(d.blue_id), teams.get_indexer(d.red_id)
    yv, nT, free = d.result.to_numpy(float), len(teams), L0 is None

    def nll(th):
        a = th[0]
        L, t = (th[1:1 + nL], th[1 + nL:]) if free else (L0, th[1:])
        s = L[tl] + t
        z = a + s[bi] - s[ri]
        g = w * (yv - 1 / (1 + np.exp(-z)))
        gs = np.bincount(bi, g, nT) - np.bincount(ri, g, nT)
        f = -(np.sum(w * (yv * z - np.logaddexp(0, z))) - 0.5 * lam_team * t @ t - (0.5 * lam_league * L @ L if free else 0))
        gL = [np.bincount(tl, gs, nL) - lam_league * L] if free else []
        return f, -np.concatenate([[g.sum()], *gL, gs - lam_team * t])

    x = minimize(nll, np.zeros(1 + (nL if free else 0) + nT), jac=True, method="L-BFGS-B", options={"maxiter": 5000}).x
    L, t = (x[1:1 + nL], x[1 + nL:]) if free else (L0, x[1:])
    return x[0], pd.Series(L[tl] + t, index=teams), L


# ---------- per-year setup (same window, home league and team set as w1) ----------
SET = {}
for y in sorted(W.year.unique()):
    T = ev.loc[y, "start"]
    d = G[(G.date < T) & (G.date >= T - pd.Timedelta(days=365))]
    long = pd.concat([d[["blue_id", "league"]].rename(columns={"blue_id": "t"}), d[["red_id", "league"]].rename(columns={"red_id": "t"})])
    home = long[~long.league.isin(INTL)].groupby("t").league.agg(lambda s: s.value_counts().index[0])
    wy = W[W.year == y]
    teams = pd.Index(sorted(set(d.blue_id) | set(d.red_id) | set(wy.blue_id) | set(wy.red_id)))
    home = home.reindex(teams).fillna("UNKNOWN")
    leagues = pd.Index(sorted(home.unique()))
    d90 = d[d.date >= T - pd.Timedelta(days=SEASON)]
    hb, hr = home.reindex(d90.blue_id).to_numpy(), home.reindex(d90.red_id).to_numpy()
    SET[y] = dict(T=T, d=d, d90=d90, teams=teams, home=home, tl=leagues.get_indexer(home.to_numpy()), nL=len(leagues),
                  cross90=int((hb != hr).sum()), intl90=int(d90.league.isin(INTL).sum()))


def predict(kind, lam):
    out, strength = [], []
    for y, s in SET.items():
        age = (s["T"] - s["d"].date).dt.total_seconds().to_numpy() / 86400
        a1, s1, L1 = fit(s["d"], s["teams"], s["tl"], s["nL"], 0.5 ** (age / best.half_life), best.lam_team)
        if kind == "1y":
            a, st = a1, s1
        elif kind == "season":
            a, st, _ = fit(s["d90"], s["teams"], s["tl"], s["nL"], np.ones(len(s["d90"])), lam, L0=L1)
        else:
            a, st, _ = fit(s["d90"], s["teams"], s["tl"], s["nL"], np.ones(len(s["d90"])), lam)
        wy = W[W.year == y]
        out.append(pd.Series(a + st.reindex(wy.blue_id).to_numpy() - st.reindex(wy.red_id).to_numpy(), index=wy.gameid))
        strength.append(pd.DataFrame({"year": y, "teamid": st.index, "s": st.to_numpy(), "s_1y": s1.to_numpy()}))
    return pd.concat(out), pd.concat(strength, ignore_index=True)


def quality(pl, mask=None):
    d = W.assign(pl=W.gameid.map(pl))
    d = d[mask] if mask is not None else d
    p = (1 / (1 + np.exp(-d.pl))).clip(1e-6, 1 - 1e-6)
    return {"logloss": -np.mean(d.result * np.log(p) + (1 - d.result) * np.log(1 - p)),
            "acc": np.mean((p > 0.5) == (d.result == 1)), "brier": np.mean((p - d.result) ** 2), "n": len(d)}


known = ((W.blue_npre >= 10) & (W.red_npre >= 10)).to_numpy()
PRED, STR, rows = {}, {}, []
pl, st = predict("1y", None)
print(f"1-year refit reproduces w1: max |diff| = {np.abs(W.gameid.map(pl) - W.pred_logit).max():.2e}")
PRED["1y"], STR["1y"] = pl, st
rows.append({"control": "1y", "lam_team": best.lam_team, **quality(pl), "logloss_known": quality(pl, known)["logloss"]})
for kind in ["season", "90d"]:
    grid = []
    for lam in [0.5, 1.0, 2.0, 4.0, 8.0]:
        pl, st = predict(kind, lam)
        grid.append((quality(pl, known)["logloss"], lam, pl, st))
    _, lam, pl, st = min(grid, key=lambda g: g[0])
    PRED[kind], STR[kind] = pl, st
    rows.append({"control": kind, "lam_team": lam, **quality(pl), "logloss_known": quality(pl, known)["logloss"]})
rows.append({"control": "none (coin flip)", "logloss": np.log(2), "acc": np.nan, "brier": 0.25, "n": len(W)})
Q = pd.DataFrame(rows)
Q.to_csv(OUT / "strength_models.csv", index=False)
print("\ninternational / cross-region games in the last 90 days:",
      {y: (s["intl90"], s["cross90"]) for y, s in SET.items()})
print("\n", Q.round(4).to_string())
per_year = pd.DataFrame({k: W.assign(pl=W.gameid.map(v)).groupby("year").apply(
    lambda d: -np.mean(d.result * np.log(1 / (1 + np.exp(-d.pl))) + (1 - d.result) * np.log(1 - 1 / (1 + np.exp(-d.pl)))))
    for k, v in PRED.items()})
print("\nlog loss by year\n", per_year.round(3).to_string())
W["pred_season"] = W.gameid.map(PRED["season"])
W.to_parquet(OUT / "worlds_games_pred_season.parquet")

# ---------- are the patch exposures correlated with strength or recent form? ----------
E = pd.read_csv(OUT / "team_patch_exposure.csv")
S = STR["season"].rename(columns={"s": "s_season"}).merge(E[["year", "teamid", "intended_z", "buffs_z", "nerfs_z"]], on=["year", "teamid"])
z = lambda s: (s - s.mean()) / s.std()
S["s1_z"] = S.groupby("year").s_1y.transform(z)
S["ss_z"] = S.groupby("year").s_season.transform(z)
S["form_z"] = S.groupby("year").apply(lambda g: z(g.s_season - g.s_1y), include_groups=False).reset_index(level=0, drop=True)
cor = []
for e in ["intended_z", "buffs_z", "nerfs_z"]:
    for s_ in ["s1_z", "ss_z", "form_z"]:
        d = S.dropna(subset=[e, s_])
        r, p = pearsonr(d[e], d[s_])
        cor.append({"exposure": e, "strength": s_, "r": r, "p": p, "n": len(d)})
cor = pd.DataFrame(cor)
cor.to_csv(OUT / "strength_exposure_corr.csv", index=False)
S.to_csv(OUT / "strength_teams.csv", index=False)
print("\n", cor.round(3).to_string())

# ---------- the four ③ estimates under each control ----------
blue_of = W.set_index("gameid").blue_id
CONTROLS = {"none": None, "1y": PRED["1y"], "season": PRED["season"], "90d": PRED["90d"]}


def logit(y, X):
    return sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0)


def own_pred(gid, team, pl):
    """blue-perspective logit -> the given team's perspective"""
    b = pl.reindex(gid).to_numpy()
    return np.where(blue_of.reindex(gid).to_numpy() == np.asarray(team), b, -b)


# rows 1-2 (w8): team exposure differences, all main-event games, permutation p
ei = E.set_index(["year", "teamid"])[["intended_z", "nerfs_z", "buffs_z"]]
for f in ei.columns:
    W["d_" + f] = ei[f].reindex(list(zip(W.year, W.blue_id))).to_numpy() - ei[f].reindex(list(zip(W.year, W.red_id))).to_numpy()
D8 = W.dropna(subset=["d_intended_z"]).reset_index(drop=True)
EXP = E.dropna(subset=["intended_z"])


def perm8(df):
    out = df.copy()
    for yy, g in EXP.groupby("year"):
        mp = dict(zip(g.teamid, g[list(ei.columns)].to_numpy()[rng.permutation(len(g))]))
        s = out.year == yy
        b = np.array([mp.get(t, [np.nan] * 3) for t in out.loc[s, "blue_id"]])
        r = np.array([mp.get(t, [np.nan] * 3) for t in out.loc[s, "red_id"]])
        for k, f in enumerate(ei.columns):
            out.loc[s, "d_" + f] = b[:, k] - r[:, k]
    return out.dropna(subset=["d_intended_z"])


# row 3 (w13): targeted ban pull, first game of each series
TG = pd.read_csv(OUT / "ban_pressure_team_games.csv")
F = TG[TG.has_pool & (TG.game_no == 1)]
BL = F.merge(F[["gameid", "T", "pull"]].rename(columns={"T": "O"}), on=["gameid", "O"], suffixes=("", "_op"))
BL = BL.drop_duplicates("gameid").dropna(subset=["pull", "pull_op"]).reset_index(drop=True)
BL = BL.merge(TG[TG.has_pool & (TG.game_no == 1)][["gameid", "T", "other_pull", "comfort"]].rename(columns={"T": "O"}),
              on=["gameid", "O"], suffixes=("", "_op"))
BL = BL.dropna(subset=["other_pull", "other_pull_op", "comfort", "comfort_op"]).reset_index(drop=True)
BL["d_pull"] = (BL.pull - BL.pull_op) / (BL.pull - BL.pull_op).std()
# row 4 (w10): pick advantage, all games
ST = pd.read_csv(OUT / "stage_team_games.csv")
G10 = ST[ST.pred_logit.notna()].drop_duplicates("gameid").reset_index(drop=True)
SD_ADV = ST.pick_adv.std()

res = []
for cname, pl in CONTROLS.items():
    ctrl = [] if pl is None else ["ctrl"]
    D8["ctrl"] = 0.0 if pl is None else pl.reindex(D8.gameid).to_numpy()
    BL["ctrl"] = 0.0 if pl is None else own_pred(BL.gameid, BL["T"], pl)
    G10["ctrl"] = 0.0 if pl is None else own_pred(G10.gameid, G10.teamid, pl)
    yrs = D8.year.unique()
    boot_years = [rng.choice(yrs, len(yrs)) for _ in range(N_BOOT)]
    specs = [("Net exposure", D8, "result", ["d_intended_z"], "d_intended_z", 1.0),
             ("Buff exposure", D8, "result", ["d_nerfs_z", "d_buffs_z"], "d_buffs_z", 1.0),
             ("Excess bans", BL, "win", ["d_pull"], "d_pull", 1.0),
             ("Pick advantage", G10, "win", ["pick_adv"], "pick_adv", SD_ADV)]
    perms = [perm8(D8) for _ in range(N_PERM)] if True else []
    for lab, df, yv, terms, key, scale in specs:
        m = logit(df[yv], df[ctrl + terms])
        est = m.params[key] * scale / 4 * 100
        bs = np.array([logit(b[yv], b[ctrl + terms]).params[key] for b in
                       (pd.concat([df[df.year == yy] for yy in by]) for by in boot_years)]) * scale / 4 * 100
        if df is D8:
            null = np.array([logit(p[yv], p[ctrl + terms]).tvalues[key] for p in perms])
            p = (np.sum(np.abs(null) >= abs(m.tvalues[key])) + 1) / (N_PERM + 1)
            how = "studentized permutation"
        elif lab == "Excess bans":
            p, how = m.pvalues[key], "wald"
        else:
            p, how = 2 * min(np.mean(bs <= 0), np.mean(bs >= 0)), "bootstrap"
        res.append({"row": lab, "control": cname, "winprob_pp": est, "lo": np.quantile(bs, .025),
                    "hi": np.quantile(bs, .975), "p": p, "p_method": how, "n_games": len(df)})
    print(cname, "done")
R = pd.DataFrame(res)
R.to_csv(OUT / "strength_sensitivity.csv", index=False)
print("\n", R.pivot(index="row", columns="control", values="winprob_pp").round(2).to_string())
print("\n", R.round(3).to_string())
