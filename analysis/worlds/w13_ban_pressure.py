"""W13: the ban path — did buffed champions pull the opponents' bans, and did that free the rest of the draft?

w10 stage 2: teams with more buff exposure lost more of it to opponent bans (r=0.23). Two readings:
  global    buffed champions are banned against everyone; the team just loses its buffed options
  targeted  opponents spend bans on THIS team's buffed champions -> fewer bans left for its other
            champions -> easier draft ("ban pressure" as an advantage)

A  Targeting   champion x team-game LPM:  opponent banned c ~ pool_T(c) [x buffed / nerfed] + pool_O(c)
               | champion-year FE (global ban rate, incl. what the patch did to it) + banning-team-game FE
B  Relief      A + pool_T(c) x not-buffed x buff exposure_T: are the OTHER champions of buff-exposed teams banned less?
               team level: pool mass banned vs expected from leave-team-out global ban rates
C  Draft ease  comfort of the team's own picks (role pick share in the 100 days before Worlds)
D  Result      win ~ pre-Worlds strength + ban pull / relief / comfort; first game of each series only
               (later games' bans react to earlier games)

pool_T(c) = sum over roles of the team's role pick share of c, 100 days before Worlds (1 = one role, every game).
buffed / nerfed = team-specific net coded change since last played patch; exposure_T = w8 buffs_z.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
from worlds_common import OUT, team_delta, coding_files, sr_only, assign_series, champion_unavailable, load_matches
rng = np.random.default_rng(13)
CORE = 0.1          # "core" champion: some role picked it in >= 10% of the team's games
N_PERM = 5000

W = pd.read_parquet(OUT / "worlds_games_pred.parquet").sort_values("date").reset_index(drop=True)
E = pd.read_csv(OUT / "team_patch_exposure.csv")
LAST = pd.read_csv(OUT / "team_exposure.csv", dtype={"last_patch": str}).set_index(["year", "teamid"]).last_patch
K = pd.read_csv(OUT / "knockout_patch_exposure.csv")[["year", "teamid", "team", "finish"]]
ev = pd.read_csv(OUT / "worlds_events.csv", index_col=0, parse_dates=["start"], dtype={"main_patch": str}).dropna()
raw = load_matches(["gameid", "date", "position", "teamid", "champion", "ban1", "ban2", "ban3", "ban4", "ban5"],
                   years=set(W.year))
raw["date"] = pd.to_datetime(raw.date, errors="coerce")
P = raw[raw.position.isin(["top", "jng", "mid", "bot", "sup"])].dropna(subset=["teamid", "champion"]).rename(columns={"position": "role"})
BAN = raw.loc[raw.position == "team", ["gameid", "teamid", "ban1", "ban2", "ban3", "ban4", "ban5"]].melt(
    id_vars=["gameid", "teamid"], var_name="order", value_name="champion").dropna(subset=["champion"])
BAN["order"] = BAN.order.str[-1].astype(int)

pre = pd.concat([pd.read_csv(f, dtype={"patch_oe": str}) for f in coding_files()])
pre = sr_only(pre)
pre["direction"] = pre.direction.str.lower().str.strip()
pre["magnitude"] = pd.to_numeric(pre.magnitude, errors="coerce").fillna(0)
pre["score"] = np.where(pre.direction == "buff", pre.magnitude, np.where(pre.direction == "nerf", -pre.magnitude, 0))
NET = pre.groupby(["year_group", "champion"]).score.sum()

m0 = sm.Logit(W.result, sm.add_constant(W[["pred_logit"]])).fit(disp=0)
W["p_cal"] = m0.predict(sm.add_constant(W[["pred_logit"]]))
W["pair"] = [tuple(sorted(p)) for p in zip(W.blue_id, W.red_id)]
W["series"], W["game_no"] = assign_series(W)

# In 2025 best-of series, champions previously picked by either team cannot be
# picked/banned again. Remove those options, rather than coding them as choices
# the opponent could have banned but did not. Bo1 rows have an empty locked set.
LOCKED = {}
for _, series in W[W.year == 2025].groupby("series", sort=False):
    used = set()
    for gid in series.gameid:
        LOCKED[gid] = frozenset(used)
        used.update(P.loc[P.gameid == gid, "champion"])

UNAVAILABLE = champion_unavailable()           # disabled or not yet allowed at that Worlds (whole event or window)
GAME_TIME = dict(zip(W.gameid, W.date))
assert not any(UNAVAILABLE(y, c, GAME_TIME[g]) for y, g, c in
               zip(W.set_index("gameid").loc[P[P.gameid.isin(GAME_TIME)].gameid, "year"], P[P.gameid.isin(GAME_TIME)].gameid,
                   P[P.gameid.isin(GAME_TIME)].champion)), "Observed pick of an unavailable champion"

# ---------- champion x team-game grid ----------
grids, tgs, breadths = [], [], []
for y, w in W.groupby("year"):
    T0 = ev.loc[y, "start"]
    teams = set(w.blue_id) | set(w.red_id)
    net = pd.concat({t: team_delta(pre, y, LAST.loc[(y, t)]) for t in sorted(teams)},
                    names=["teamid", "champion"])
    win = P[(P.date < T0) & (P.date >= T0 - pd.Timedelta(days=100)) & P.teamid.isin(teams)]
    ng = win.groupby("teamid").gameid.nunique()
    win = win[win.teamid.isin(ng[ng >= 10].index)]
    rc = win.groupby(["teamid", "role", "champion"]).size().rename("rcount")      # integer pick counts
    rs = (rc / rc.groupby(level=["teamid", "role"]).transform("sum")).rename("rshare")
    pool = rs.groupby(level=["teamid", "champion"]).sum().rename("pool")
    # breadth as in w9: mean over roles of exp(entropy)
    br = rs.groupby(level=["teamid", "role"]).apply(lambda s: np.exp(-(s * np.log(s)).sum())).groupby(level="teamid").mean()
    breadths.append(pd.DataFrame({"year": y, "T": br.index, "breadth": br.to_numpy()}))

    tg = pd.concat([pd.DataFrame({"gameid": w.gameid, "T": w.blue_id, "O": w.red_id, "win": w.result,
                                  "p": w.p_cal, "pred_logit": w.pred_logit, "series": w.series, "game_no": w.game_no}),
                    pd.DataFrame({"gameid": w.gameid, "T": w.red_id, "O": w.blue_id, "win": 1 - w.result,
                                  "p": 1 - w.p_cal, "pred_logit": -w.pred_logit, "series": w.series, "game_no": w.game_no})])
    tg["year"] = y
    tg["has_pool"] = tg["T"].isin(ng[ng >= 10].index)

    bw = BAN[BAN.gameid.isin(set(w.gameid))]
    pw = P[P.gameid.isin(set(w.gameid))]
    champs = sorted(set(pool.index.get_level_values("champion")) | set(bw.champion) | set(pw.champion))

    # leave-team-out global ban rate: P(a banning team bans c) in this Worlds' games not involving T
    side = pd.concat([w[["gameid", "blue_id"]].set_axis(["gameid", "t"], axis=1), w[["gameid", "red_id"]].set_axis(["gameid", "t"], axis=1)])
    nb = bw.groupby("champion").size()
    bt = bw[["gameid", "champion"]].merge(side, on="gameid").groupby(["t", "champion"]).size()
    ngt = side.groupby("t").size()

    g = tg[["gameid", "T", "O", "year"]].merge(pd.DataFrame({"champion": champs}), how="cross")
    g["eligible"] = [c not in LOCKED.get(gid, ()) and not UNAVAILABLE(y, c, GAME_TIME[gid])
                     for gid, c in zip(g.gameid, g.champion)]
    g = g.merge(bw.rename(columns={"teamid": "O"})[["gameid", "O", "champion", "order"]], on=["gameid", "O", "champion"], how="left")
    g["banned"] = g.order.notna().astype(float)
    g["banned_p1"] = (g.order <= 3).astype(float)
    g["pool"] = pool.reindex(pd.MultiIndex.from_arrays([g["T"], g.champion])).fillna(0).to_numpy()
    g["pool_o"] = pool.reindex(pd.MultiIndex.from_arrays([g["O"], g.champion])).fillna(0).to_numpy()
    g["q"] = ((nb.reindex(g.champion).fillna(0).to_numpy()
               - bt.reindex(pd.MultiIndex.from_arrays([g["T"], g.champion])).fillna(0).to_numpy())
              / (2 * len(w) - 2 * ngt.reindex(g["T"]).to_numpy()))
    if not g.eligible.all():
        # Compare bans only with other games in which that champion was available (fearless locks in 2025,
        # disabled champions in any year).
        eligible_total = g.groupby("champion").eligible.sum()
        unavailable_denominator = g[["gameid", "champion", "eligible"]].drop_duplicates().merge(side, on="gameid")
        eligible_involving = unavailable_denominator.groupby(["t", "champion"]).eligible.sum() * 2
        denom = (eligible_total.reindex(g.champion).to_numpy()
                 - eligible_involving.reindex(pd.MultiIndex.from_arrays([g["T"], g.champion])).to_numpy())
        numer = (nb.reindex(g.champion).fillna(0).to_numpy()
                 - bt.reindex(pd.MultiIndex.from_arrays([g["T"], g.champion])).fillna(0).to_numpy())
        if np.any((denom <= 0) & g.eligible.to_numpy()):
            raise ValueError("No leave-team-out eligible comparison for a champion")
        g["q"] = np.where(denom > 0, numer / np.where(denom > 0, denom, 1), 0.0)   # never-available: set to 0 below
    assert not ((g.banned == 1) & ~g.eligible).any(), "Observed ban of a locked or unavailable champion"
    g.loc[~g.eligible, "q"] = 0.0
    g["net"] = net.reindex(pd.MultiIndex.from_arrays([g["T"], g.champion])).fillna(0).to_numpy()
    grids.append(g.drop(columns="order"))

    # own picks: comfort = role pick share over the 100 days before Worlds
    pk = pw[["gameid", "teamid", "role", "champion"]].reset_index(drop=True)
    pk["rshare"] = rs.reindex(pd.MultiIndex.from_arrays([pk.teamid, pk.role, pk.champion])).fillna(0).to_numpy()
    pk["buffed"] = net.reindex(pd.MultiIndex.from_arrays([pk.teamid, pk.champion])).fillna(0).to_numpy() > 0
    pk["familiar"] = rc.reindex(pd.MultiIndex.from_arrays([pk.teamid, pk.role, pk.champion])).fillna(0).to_numpy() >= 2  # played in this role >= 2 times
    cm = pk.groupby(["gameid", "teamid"]).agg(comfort=("rshare", "mean"), core_picks=("rshare", lambda s: (s >= CORE).sum()),
                                              familiar=("familiar", "mean"),
                                              buffed_picks=("buffed", "sum"),
                                              comfort_other=("rshare", lambda s: s[~pk.loc[s.index, "buffed"]].mean()))
    tgs.append(tg.merge(cm.rename_axis(["gameid", "T"]).reset_index(), on=["gameid", "T"], how="left"))

G = pd.concat(grids, ignore_index=True)
TG = pd.concat(tgs, ignore_index=True)
G["buffed"], G["nerfed"] = (G.net > 0).astype(float), (G.net < 0).astype(float)
G["bmag"] = G.net.clip(lower=0)
G["core"] = (G.pool >= CORE).astype(float)

# ---------- per team-game ban quantities ----------
G["pb"] = G.pool * G.bmag                        # buff exposure weight (w10's "denied" uses this)
G["po"] = G.pool * (1 - G.buffed)                # the rest of the pool
G["cb"] = G.core * G.buffed
G["co"] = G.core * (1 - G.buffed)
agg = {}
for k in ["pb", "po", "cb", "co", "pool"]:
    G[k + "_obs"], G[k + "_exp"] = G[k] * G.banned, G[k] * G.q
    agg.update({k: (k, "sum"), k + "_obs": (k + "_obs", "sum"), k + "_exp": (k + "_exp", "sum")})
A = G.groupby(["gameid", "T"]).agg(n_bans=("banned", "sum"), **agg).reset_index()
TG = TG.merge(A, on=["gameid", "T"], how="left")
TG.loc[~TG.has_pool, A.columns.drop(["gameid", "T"])] = np.nan
TG["denied"] = TG.pb_obs / TG.pb.where(TG.pb > 0)                      # = w10 "denied"
TG["denied_exp"] = TG.pb_exp / TG.pb.where(TG.pb > 0)
TG["other_denied"] = TG.po_obs / TG.po.where(TG.po > 0)
TG["other_denied_exp"] = TG.po_exp / TG.po.where(TG.po > 0)
TG["pull"] = TG.cb_obs - TG.cb_exp               # extra opponent bans on the team's buffed core champions
TG["other_pull"] = TG.co_obs - TG.co_exp         # extra opponent bans on its other core champions
TG = TG.merge(E[["year", "teamid", "buffs_z", "buffs"]].rename(columns={"teamid": "T"}), on=["year", "T"], how="left")
BR = pd.concat(breadths, ignore_index=True)
BR["breadth_z"] = BR.groupby("year").breadth.transform(lambda s: (s - s.mean()) / s.std())
TG = TG.merge(BR, on=["year", "T"], how="left")
TG.to_csv(OUT / "ban_pressure_team_games.csv", index=False)

w10 = pd.read_csv(OUT / "stage_team_games.csv")[["gameid", "teamid", "denied"]].rename(columns={"teamid": "T", "denied": "d10"})
chk = TG.merge(w10, on=["gameid", "T"]).dropna(subset=["denied", "d10"])
print(f"check vs w10 'denied': corr {np.corrcoef(chk.denied, chk.d10)[0, 1]:.3f} (n={len(chk)})")

# ---------- A/B: champion-level targeting ----------
G = G.merge(E[["year", "teamid", "buffs_z"]].rename(columns={"teamid": "T", "buffs_z": "expo"}), on=["year", "T"], how="left")
G = G.merge(TG[TG.has_pool][["year", "T"]].drop_duplicates(), on=["year", "T"])      # teams with a pre-Worlds pool
G["cy"] = G.year.astype(str) + "|" + G.champion
G["bg"] = G.gameid + "|" + G["O"]
G["clus"] = G.year.astype(str) + "|" + G["T"]
G["pool_buff"], G["pool_nerf"] = G.pool * G.buffed, G.pool * G.nerfed
G["pool_bmag"] = G.pool * G.bmag
G["pool_other_x_expo"] = G.pool * (1 - G.buffed) * G.expo
G["pool_buff_x_expo"] = G.pool * G.buffed * G.expo
G = G.merge(BR[["year", "T", "breadth_z"]], on=["year", "T"], how="left")
G["pool_other_x_breadth"] = G.pool * (1 - G.buffed) * G.breadth_z
G["pool_buff_x_breadth"] = G.pool * G.buffed * G.breadth_z
Gm = G[G.eligible].copy()


def lpm(fml, data, label):
    m = pf.feols(fml, data=data, vcov={"CRV1": "clus"})
    t = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
    t["model"] = label
    t["n"] = m._N
    return t[["model", "term", "Estimate", "Std. Error", "Pr(>|t|)", "2.5%", "97.5%", "n"]]


FE = " | cy + bg"
tabs = [lpm("banned ~ pool + pool_o" + FE, Gm, "A0 pool targeting"),
        lpm("banned ~ pool + pool_buff + pool_nerf + pool_o" + FE, Gm, "A1 targeting x patch"),
        lpm("banned ~ pool + pool_bmag + pool_nerf + pool_o" + FE, Gm, "A1m targeting x buff magnitude"),
        lpm("banned_p1 ~ pool + pool_buff + pool_nerf + pool_o" + FE, Gm, "A1p1 phase-1 bans only"),
        lpm("banned ~ pool + pool_buff + pool_nerf + pool_o" + FE, Gm[Gm.year < 2025], "A1x25 excl. 2025 (fearless)"),
        lpm("banned ~ pool + pool_buff + pool_nerf + pool_o + pool_other_x_expo + pool_buff_x_expo" + FE,
            Gm.dropna(subset=["expo"]), "B1 relief: other pool x buff exposure"),
        lpm("banned ~ pool + pool_buff + pool_nerf + pool_o + pool_other_x_expo + pool_buff_x_expo"
            " + pool_other_x_breadth + pool_buff_x_breadth" + FE,
            Gm.dropna(subset=["expo", "breadth_z"]), "B2 relief, controlling pool breadth")]
BFORM = "pool + pool_buff + pool_nerf + pool_o + pool_other_x_expo + pool_buff_x_expo"
first_games = set(W.loc[W.game_no == 1, "gameid"])
tabs.extend([
    lpm("banned ~ " + BFORM + FE, Gm[Gm.year < 2025].dropna(subset=["expo"]), "B1x25 excl. 2025"),
    lpm("banned ~ " + BFORM + FE, Gm[Gm.gameid.isin(first_games)].dropna(subset=["expo"]), "B1g1 series game 1"),
    lpm("banned_p1 ~ " + BFORM + FE, Gm.dropna(subset=["expo"]), "B1p1 phase-1 bans"),
])
# functional-form sensitivity (post hoc, after the 2026-09-29 review): the interaction multiplies a champion's pool
# share by team buff exposure, which itself contains that champion's pool share x buff magnitude
yr = E.groupby("year").buffs.agg(["mean", "std"])
assert np.allclose((E.buffs - E.year.map(yr["mean"])) / E.year.map(yr["std"]), E.buffs_z, equal_nan=True)
G = G.merge(E[["year", "teamid", "buffs"]].rename(columns={"teamid": "T"}), on=["year", "T"], how="left")
G["expo_loo"] = (G.buffs - G.pool * G.bmag - G.year.map(yr["mean"])) / G.year.map(yr["std"])
G["pool_buff_x_expo_loo"] = G.pool * G.buffed * G.expo_loo
G["pool_other_x_expo_loo"] = G.pool * (1 - G.buffed) * G.expo_loo
G["pool2_buff"], G["pool2_other"] = G.pool ** 2 * G.buffed, G.pool ** 2 * (1 - G.buffed)
G["buffed_x_expo"] = G.buffed * G.expo
Gm = G[G.eligible].copy()
tabs.extend([
    lpm("banned ~ pool + pool_buff + pool_nerf + pool_o + pool_other_x_expo_loo + pool_buff_x_expo_loo" + FE,
        Gm.dropna(subset=["expo"]), "B1s1 exposure excluding the champion's own contribution"),
    lpm("banned ~ " + BFORM + " + pool2_buff + pool2_other" + FE, Gm.dropna(subset=["expo"]), "B1s2 + squared pool share"),
    lpm("banned ~ " + BFORM + " + buffed + nerfed + buffed_x_expo" + FE, Gm.dropna(subset=["expo"]),
        "B1s3 + buffed, nerfed, buffed x exposure"),
])
LPM = pd.concat(tabs, ignore_index=True)
LPM.to_csv(OUT / "ban_pressure_targeting.csv", index=False)
print("\n", LPM.round(4).to_string())

# ---------- team level ----------
S = TG[TG.has_pool].groupby(["year", "T"]).agg(
    games=("win", "size"), buffs_z=("buffs_z", "first"),
    denied=("denied", "mean"), denied_exp=("denied_exp", "mean"),
    other_denied=("other_denied", "mean"), other_denied_exp=("other_denied_exp", "mean"),
    pull=("pull", "mean"), cb_obs=("cb_obs", "mean"), cb_exp=("cb_exp", "mean"),
    other_pull=("other_pull", "mean"), co_obs=("co_obs", "mean"), co_exp=("co_exp", "mean"),
    comfort=("comfort", "mean"), comfort_other=("comfort_other", "mean"), core_picks=("core_picks", "mean"),
    familiar=("familiar", "mean"), breadth_z=("breadth_z", "first"),
    resid=("win", "mean"), pexp=("p", "mean")).reset_index()
S["resid"] = S.resid - S.pexp
S["denied_excess"] = S.denied - S.denied_exp
S["other_excess"] = S.other_denied - S.other_denied_exp
# breadth-residualised versions (pooled OLS with year means, as in w9)
for c in ["buffs_z", "pull", "other_pull", "comfort", "familiar", "resid"]:
    d = S.dropna(subset=[c, "breadth_z"])
    fit_b = sm.OLS(d[c] - d.groupby("year")[c].transform("mean"), sm.add_constant(d[["breadth_z"]])).fit()
    S.loc[d.index, c + "_rb"] = fit_b.resid


def perm_corr(a, b, d=S, n=N_PERM):
    d = d.dropna(subset=[a, b])
    obs = np.corrcoef(d[a], d[b])[0, 1]
    grp = [np.flatnonzero(d.year.to_numpy() == yy) for yy in d.year.unique()]
    bv = d[b].to_numpy()
    null = np.empty(n)
    for i in range(n):
        x = bv.copy()
        for ix in grp:
            x[ix] = x[rng.permutation(ix)]
        null[i] = np.corrcoef(d[a], x)[0, 1]
    return {"a": a, "b": b, "r": obs, "p": (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1), "n": len(d)}


team_tests = pd.DataFrame([
    perm_corr("buffs_z", "denied"), perm_corr("buffs_z", "denied_exp"), perm_corr("buffs_z", "denied_excess"),
    perm_corr("buffs_z", "cb_obs"), perm_corr("buffs_z", "cb_exp"), perm_corr("buffs_z", "pull"),
    perm_corr("buffs_z", "other_denied"), perm_corr("buffs_z", "other_denied_exp"), perm_corr("buffs_z", "other_excess"),
    perm_corr("buffs_z", "other_pull"), perm_corr("pull", "other_pull"),
    perm_corr("buffs_z", "comfort"), perm_corr("buffs_z", "comfort_other"), perm_corr("pull", "comfort_other"),
    perm_corr("buffs_z", "familiar"), perm_corr("pull", "familiar"),
    perm_corr("pull", "resid"), perm_corr("other_pull", "resid"), perm_corr("comfort", "resid"), perm_corr("familiar", "resid"),
    perm_corr("breadth_z", "buffs_z"), perm_corr("breadth_z", "pull"), perm_corr("breadth_z", "other_pull"),
    perm_corr("breadth_z", "comfort"), perm_corr("breadth_z", "familiar"),
    perm_corr("buffs_z_rb", "pull_rb"), perm_corr("buffs_z_rb", "other_pull_rb"), perm_corr("pull_rb", "other_pull_rb"),
    perm_corr("pull_rb", "resid_rb")])
# prepared, not in-series reaction: same targeting measures from the first game of each series only
S1 = TG[TG.has_pool & (TG.game_no == 1)].groupby(["year", "T"]).agg(
    buffs_z=("buffs_z", "first"), pull=("pull", "mean"), other_pull=("other_pull", "mean")).reset_index()
g1 = pd.DataFrame([perm_corr("buffs_z", "pull", S1), perm_corr("buffs_z", "other_pull", S1)])
g1["a"] = g1.a + " (game 1 only)"
team_tests = pd.concat([team_tests, g1], ignore_index=True)
team_tests.to_csv(OUT / "ban_pressure_team_tests.csv", index=False)
print("\n", team_tests.round(4).to_string())

# ---------- game level: within team-year (does a game with more pull free the rest?) ----------
TGg = TG[TG.has_pool].copy()
TGg["ty"] = TGg.year.astype(str) + "|" + TGg["T"]
within = []
for dep, x in [("other_pull", "pull"), ("comfort_other", "pull"), ("core_picks", "pull"), ("familiar", "pull")]:
    m = pf.feols(f"{dep} ~ {x} + pred_logit | ty", data=TGg.dropna(subset=[dep, x]), vcov={"CRV1": "ty"})
    r = m.tidy().loc[x]
    within.append({"dep": dep, "x": x, "est": r.Estimate, "se": r["Std. Error"], "p": r["Pr(>|t|)"], "n": m._N})
within = pd.DataFrame(within)
within.to_csv(OUT / "ban_pressure_within_team.csv", index=False)
print("\n", within.round(4).to_string())

# ---------- D: result, first game of each series ----------
F = TG[TG.has_pool & (TG.game_no == 1)]
bl = F.merge(F[["gameid", "T", "pull", "other_pull", "comfort"]].rename(columns={"T": "O"}), on=["gameid", "O"], suffixes=("", "_op"))
bl = bl.drop_duplicates("gameid").dropna(subset=["pull", "pull_op", "other_pull", "other_pull_op", "comfort", "comfort_op"])
for c in ["pull", "other_pull", "comfort"]:
    bl["d_" + c] = (bl[c] - bl[c + "_op"]) / (bl[c] - bl[c + "_op"]).std()
yrs = bl.year.unique()
res = []
for c in ["d_pull", "d_other_pull", "d_comfort"]:
    fit = lambda d: sm.Logit(d.win, sm.add_constant(d[["pred_logit", c]])).fit(disp=0)
    m = fit(bl)
    boots = np.array([fit(pd.concat([bl[bl.year == yy] for yy in rng.choice(yrs, len(yrs))])).params[c] for _ in range(2000)])
    res.append({"term": c, "logit_per_sd": m.params[c], "winprob_pp": m.params[c] / 4 * 100, "p_wald": m.pvalues[c],
                "boot_lo_pp": np.quantile(boots, .025) / 4 * 100, "boot_hi_pp": np.quantile(boots, .975) / 4 * 100, "n_games": len(bl)})
res = pd.DataFrame(res)
res.to_csv(OUT / "ban_pressure_result_models.csv", index=False)
print("\n", res.round(4).to_string())

# ---------- champion among quarterfinalists ----------
KS = K.merge(S.rename(columns={"T": "teamid"}), on=["year", "teamid"], how="left")
out, tab = [], []
for lab, col, low_better in [("ban pull on buffed core (more = more ban pressure)", "pull", False),
                             ("denied excess (buff exposure, targeted part)", "denied_excess", False),
                             ("other core banned beyond expected (less = relief)", "other_pull", True),
                             ("  same, breadth-residualised", "other_pull_rb", True),
                             ("comfort of own picks", "comfort", False),
                             ("  same, breadth-residualised", "comfort_rb", False),
                             ("comfort of non-buffed picks", "comfort_other", False),
                             ("familiar picks (played >= 2x in role)", "familiar", False),
                             ("pool breadth", "breadth_z", False)]:
    KS["r"] = KS.groupby("year")[col].rank(ascending=low_better, method="average")
    ch = KS[KS.finish == "Champion"].sort_values("year")
    rk = [g.r.dropna().to_numpy() for _, g in KS.groupby("year")]
    null = np.array([np.mean([r[rng.integers(len(r))] for r in rk if len(r)]) for _ in range(50000)])
    obs = ch.r.mean()
    out.append({"measure": lab, "champ_mean_rank": obs, "null_mean": null.mean(), "p_favoured": np.mean(null <= obs),
                "p_two_sided": np.mean(np.abs(null - null.mean()) >= abs(obs - null.mean()))})
    tab.append(ch.set_index("year").r.rename(col))
out = pd.DataFrame(out)
out.to_csv(OUT / "ban_pressure_champion_ranks.csv", index=False)
R = pd.concat(tab, axis=1)
R.insert(0, "champion", KS[KS.finish == "Champion"].sort_values("year").set_index("year").team)
R.to_csv(OUT / "ban_pressure_champion_rank_table.csv")
print("\n", out.round(3).to_string())
print("\n", R.to_string())

# ---------- descriptives: bans per game on buffed core, by buff-exposure tercile ----------
S["tercile"] = S.groupby("year").buffs_z.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]))
desc = S.groupby("tercile", observed=True)[["cb_obs", "cb_exp", "pull", "co_obs", "co_exp", "other_pull", "comfort"]].mean()
desc.to_csv(OUT / "ban_pressure_terciles.csv")
print("\n", desc.round(3).to_string())
S.to_csv(OUT / "ban_pressure_team_summary.csv", index=False)
