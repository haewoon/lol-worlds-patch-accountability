"""W10: separate the chain "patch -> draft -> result", plus the indirect meta.

Stage 1  Did the patch favour the team?        (pre-event: pool x patch notes; w8)
           buffs  = sum pool * positive change, nerfs = sum pool * negative change
Stage 2  Did that become a draft advantage?     (per Worlds game)
           pick_adv  = sum over own 5 picks of net(c) - same for opponent's picks
           denied    = share of the team's buff exposure removed by the opponent's bans
Stage 3  Did the favourable draft win games?    result ~ pre-Worlds strength + pick_adv
Stage 4  Indirect meta (not in champion notes): presence change of champions the notes did
         NOT touch (items/runes/systems/discovery), weighted by the team's original pool.
           Worlds presence from the first 5 days only (less contaminated by who wins), excl. own games.

net_i(c) = sum of coded magnitudes in patches after team i's last played version.
Each stage: pooled test over all teams + the champion's rank among the 8 quarterfinalists.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
from worlds_common import OUT, team_delta, coding_files, load_matches, sr_only
rng = np.random.default_rng(10)

W = pd.read_parquet(OUT / "worlds_games_pred.parquet").sort_values("date").reset_index(drop=True)
E = pd.read_csv(OUT / "team_patch_exposure.csv")
LAST = pd.read_csv(OUT / "team_exposure.csv", dtype={"last_patch": str}).set_index(["year", "teamid"]).last_patch
K = pd.read_csv(OUT / "knockout_patch_exposure.csv")[["year", "teamid", "team", "finish"]]
ev = pd.read_csv(OUT / "worlds_events.csv", index_col=0, parse_dates=["start"], dtype={"main_patch": str}).dropna()
raw = load_matches(["gameid", "league", "date", "position", "teamid", "champion", "ban1", "ban2", "ban3", "ban4", "ban5"])
raw["date"] = pd.to_datetime(raw.date, errors="coerce")
P = raw[raw.position.isin(["top", "jng", "mid", "bot", "sup"])].dropna(subset=["teamid", "champion"])
BAN = raw.loc[raw.position == "team", ["gameid", "teamid", "ban1", "ban2", "ban3", "ban4", "ban5"]].melt(
    id_vars=["gameid", "teamid"], value_name="champion").dropna(subset=["champion"])[["gameid", "teamid", "champion"]]

pre = pd.concat([pd.read_csv(f, dtype={"patch_oe": str}) for f in coding_files()])
pre = sr_only(pre)
pre["direction"] = pre.direction.str.lower().str.strip()
pre["magnitude"] = pd.to_numeric(pre.magnitude, errors="coerce").fillna(0)
pre["score"] = np.where(pre.direction == "buff", pre.magnitude, np.where(pre.direction == "nerf", -pre.magnitude, 0))
NET = pre.groupby(["year_group", "champion"]).score.sum()
TOUCHED = set(zip(pre[pre.direction.isin(["buff", "nerf", "adjust", "rework"])].year_group,
                  pre[pre.direction.isin(["buff", "nerf", "adjust", "rework"])].champion))

m0 = sm.Logit(W.result, sm.add_constant(W[["pred_logit"]])).fit(disp=0)
W["p_cal"] = m0.predict(sm.add_constant(W[["pred_logit"]]))


def pool_shares(teamid, T):
    pr = P[(P.teamid == teamid) & (P.date < T) & (P.date >= T - pd.Timedelta(days=100))]
    return pr.champion.value_counts(normalize=True) if pr.gameid.nunique() >= 10 else None


def presence(gids):
    g = set(gids)
    n = len(g)
    return (P[P.gameid.isin(g)].champion.value_counts().add(BAN[BAN.gameid.isin(g)].champion.value_counts(), fill_value=0) / n)


# ---------- per team-game (stage 2, 3) and per team (stage 4) ----------
tg_rows, team_rows = [], []
for y, w in W.groupby("year"):
    T = ev.loc[y, "start"]
    teams = sorted(set(w.blue_id) | set(w.red_id))
    delta = {t: team_delta(pre, y, LAST.loc[(y, t)]) for t in teams}
    pools = {t: pool_shares(t, T) for t in teams}
    homes = set(w.blue_home) | set(w.red_home)
    pre_g = P[(P.date < T) & (P.date >= T - pd.Timedelta(days=45)) & P.league.isin(homes)].gameid.unique()
    pres_pre = presence(pre_g)
    early = w[w.date < w.date.min() + pd.Timedelta(days=5)]
    for _, g in w.iterrows():
        for me, op, sign in [(g.blue_id, g.red_id, 1), (g.red_id, g.blue_id, -1)]:
            net, net_op = delta[me], delta[op]
            buff, buff_op = net.clip(lower=0), net_op.clip(lower=0)
            my_picks = P[(P.gameid == g.gameid) & (P.teamid == me)].champion
            op_picks = P[(P.gameid == g.gameid) & (P.teamid == op)].champion
            op_bans = BAN[(BAN.gameid == g.gameid) & (BAN.teamid == op)].champion
            my_bans = BAN[(BAN.gameid == g.gameid) & (BAN.teamid == me)].champion
            pn_me = net.reindex(my_picks).fillna(0).sum()
            pn_op = net_op.reindex(op_picks).fillna(0).sum()
            pool_me, pool_op = pools[me], pools[op]
            den = den_by = np.nan
            if pool_me is not None:
                be = (pool_me * buff.reindex(pool_me.index).fillna(0)).sum()
                den = (pool_me.reindex(op_bans).fillna(0) * buff.reindex(op_bans).fillna(0)).sum() / be if be > 0 else np.nan
            if pool_op is not None:
                bo = (pool_op * buff_op.reindex(pool_op.index).fillna(0)).sum()
                den_by = (pool_op.reindex(my_bans).fillna(0) * buff_op.reindex(my_bans).fillna(0)).sum() / bo if bo > 0 else np.nan
            tg_rows.append({"year": y, "gameid": g.gameid, "teamid": me, "opp": op,
                            "win": g.result if sign == 1 else 1 - g.result,
                            "p": g.p_cal if sign == 1 else 1 - g.p_cal,
                            "pred_logit": sign * g.pred_logit,
                            "pick_net": pn_me, "pick_adv": pn_me - pn_op,
                            "n_buffed_picks": int((net.reindex(my_picks).fillna(0) > 0).sum()),
                            "denied": den, "denied_by": den_by})
    for t in teams:
        pool = pools[t]
        if pool is None:
            continue
        untouched = [c for c in pool.index if (y, c) not in TOUCHED]
        pu = pool.reindex(untouched)
        pres_early = presence(early[(early.blue_id != t) & (early.red_id != t)].gameid)
        pres_full = presence(w[(w.blue_id != t) & (w.red_id != t)].gameid)
        d_early = pres_early.reindex(untouched).fillna(0) - pres_pre.reindex(untouched).fillna(0)
        d_full = pres_full.reindex(untouched).fillna(0) - pres_pre.reindex(untouched).fillna(0)
        team_rows.append({"year": y, "teamid": t, "untouched_pool_share": pu.sum(),
                          "indirect_early": float((pu * d_early).sum()), "indirect_full": float((pu * d_full).sum())})

TG = pd.DataFrame(tg_rows)
TG.to_csv(OUT / "stage_team_games.csv", index=False)
S = TG.groupby(["year", "teamid"]).agg(games=("win", "size"), pick_adv=("pick_adv", "mean"),
                                       pick_net=("pick_net", "mean"), denied=("denied", "mean"),
                                       denied_by=("denied_by", "mean"), buffed_picks=("n_buffed_picks", "mean"))
# stage 3 per team: wins above expectation in favourable-draft games minus in the rest
conv = TG.assign(r=TG.win - TG.p).groupby(["year", "teamid"]).apply(
    lambda d: d[d.pick_adv > 0].r.mean() - d[d.pick_adv <= 0].r.mean() if (d.pick_adv > 0).any() and (d.pick_adv <= 0).any() else np.nan)
S["conversion"] = conv
S = S.reset_index().merge(pd.DataFrame(team_rows), on=["year", "teamid"], how="left")
S = S.merge(E[["year", "teamid", "team", "buffs", "nerfs", "buffs_z", "nerfs_z"]], on=["year", "teamid"], how="left")
S.to_csv(OUT / "stage_team_summary.csv", index=False)

# ---------- pooled tests ----------
z = lambda s: (s - s.mean()) / s.std()
for c in ["pick_adv", "denied", "indirect_early", "indirect_full"]:
    S[c + "_z"] = S.groupby("year")[c].transform(z)


def perm_corr(a, b, n=5000):
    d = S.dropna(subset=[a, b])
    obs = np.corrcoef(d[a], d[b])[0, 1]
    null = []
    for _ in range(n):
        dd = d.copy()
        dd[b] = dd.groupby("year")[b].transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).to_numpy())
        null.append(np.corrcoef(dd[a], dd[b])[0, 1])
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1), len(d)


pooled = []
for a, b, lab in [("buffs_z", "pick_adv_z", "S2 buff exposure -> pick advantage"),
                  ("nerfs_z", "pick_adv_z", "S2 nerf exposure -> pick advantage"),
                  ("buffs_z", "denied_z", "S2 buff exposure -> share denied by opponent bans")]:
    r, p, n = perm_corr(a, b)
    pooled.append({"test": lab, "stat": "corr (team level)", "value": r, "p": p, "n": n})

G = TG[TG.pred_logit.notna()].drop_duplicates("gameid")          # one row per game is enough (symmetric)
yrs = G.year.unique()
fit = lambda d: sm.Logit(d.win, sm.add_constant(d[["pred_logit", "pick_adv"]])).fit(disp=0).params["pick_adv"]
b3 = fit(G)
boots = [fit(pd.concat([G[G.year == yy] for yy in rng.choice(yrs, len(yrs))])) for _ in range(2000)]
sd_adv = TG.pick_adv.std()
pooled.append({"test": "S3 pick advantage -> win (controls pre-Worlds strength)",
               "stat": "logit per 1 SD of pick_adv", "value": b3 * sd_adv,
               "lo": np.quantile(boots, .025) * sd_adv, "hi": np.quantile(boots, .975) * sd_adv,
               "winprob_pp": b3 * sd_adv / 4 * 100, "p": 2 * min(np.mean(np.array(boots) <= 0), np.mean(np.array(boots) >= 0)),
               "n": len(G)})

for col in ["indirect_early", "indirect_full"]:
    si = S.set_index(["year", "teamid"])[col + "_z"]
    W["d"] = si.reindex(list(zip(W.year, W.blue_id))).to_numpy() - si.reindex(list(zip(W.year, W.red_id))).to_numpy()
    D = W.dropna(subset=["d"])
    f4 = lambda d: sm.Logit(d.result, sm.add_constant(d[["pred_logit", "d"]])).fit(disp=0).params["d"]
    est = f4(D)
    null = []
    for _ in range(1000):
        Dp = D.copy()
        for yy, g in S.dropna(subset=[col + "_z"]).groupby("year"):
            mp = dict(zip(g.teamid, g[col + "_z"].to_numpy()[rng.permutation(len(g))]))
            s = Dp.year == yy
            Dp.loc[s, "d"] = [mp.get(b, np.nan) - mp.get(r, np.nan) for b, r in zip(Dp.blue_id[s], Dp.red_id[s])]
        null.append(f4(Dp.dropna(subset=["d"])))
    pooled.append({"test": f"S4 indirect meta ({col.split('_')[1]}) -> win", "stat": "logit per SD",
                   "value": est, "winprob_pp": est / 4 * 100,
                   "p": (np.sum(np.abs(null) >= abs(est)) + 1) / 1001, "n": len(D)})
pooled = pd.DataFrame(pooled)
pooled.to_csv(OUT / "stage_pooled_tests.csv", index=False)
print(pooled.round(4).to_string())

# ---------- champion rank among quarterfinalists, per stage ----------
KS = K.merge(S, on=["year", "teamid"], how="left", suffixes=("", "_s"))
stages = [("S1 buffs", "buffs", False), ("S1 nerfs (fewer = better)", "nerfs", True),
          ("S2 pick advantage", "pick_adv", False), ("S2 buff denied by bans (less = better)", "denied", True),
          ("S3 conversion of favourable drafts", "conversion", False),
          ("S4 indirect meta, first 5 days", "indirect_early", False), ("S4 indirect meta, full event", "indirect_full", False)]
out, ranks_tab = [], []
for lab, col, low_better in stages:
    KS["r"] = KS.groupby("year")[col].rank(ascending=low_better, method="average")
    ch = KS[KS.finish == "Champion"].sort_values("year")
    rk = [g.r.dropna().to_numpy() for _, g in KS.groupby("year")]
    valid = ch.r.notna()
    null = np.array([np.mean([r[rng.integers(len(r))] for r, ok in zip(rk, valid) if ok and len(r)]) for _ in range(50000)])
    obs = ch.r[valid].mean()
    ru = KS[KS.finish == "Runner-up"].set_index("year")[col]
    cv = ch.set_index("year")[col]
    better = (cv < ru) if low_better else (cv > ru)
    out.append({"stage": lab, "champ_mean_rank": obs, "null_mean": null.mean(),
                "p_favoured": np.mean(null <= obs), "p_two_sided": np.mean(np.abs(null - null.mean()) >= abs(obs - null.mean())),
                "champ_better_than_runner_up": f"{int(better.sum())}/{int(cv.notna().sum())}"})
    ranks_tab.append(ch.set_index("year").r.rename(lab))
out = pd.DataFrame(out)
out.to_csv(OUT / "stage_champion_ranks.csv", index=False)
R = pd.concat(ranks_tab, axis=1)
R.insert(0, "champion", KS[KS.finish == "Champion"].sort_values("year").set_index("year").team)
R.to_csv(OUT / "stage_champion_rank_table.csv")
print("\n", out.round(3).to_string())
print("\n", R.to_string())
