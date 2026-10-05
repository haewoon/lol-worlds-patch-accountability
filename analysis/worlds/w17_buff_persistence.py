"""W17: are some teams consistently buff-favoured (e.g. T1 '23-'25 #1 among quarterfinalists), and why?

1 Persistence   correlation of a team's buff exposure (w8 buffs_z) with its exposure at its next Worlds
                appearance; within-year permutation null. Organisations are linked across documented renames
                (data/manifests/org_lineage.csv, e.g. SK Telecom T1 -> T1, Samsung Galaxy -> Gen.G).
2 Org effect    do organisations with >= 3 appearances differ more than chance? (variance of org means
                vs within-year permutation)
3 Mechanism     champion level: pro presence (pick+ban per game, all games in the 100 days before Worlds)
                of champions the Worlds patch window buffed / nerfed / left alone.
                team level: buffs_z / nerfs_z ~ pool breadth (w9 definition) + meta concentration
                (pick-share-weighted presence of the team's pool, own games excluded).
4 Residual persistence  does persistence survive breadth + meta concentration?
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from worlds_common import OUT, ROOT, coding_files, load_matches, sr_only
rng = np.random.default_rng(17)
N_PERM = 5000
# organisations across renames: data/manifests/org_lineage.csv (team IDs of one renamed team share a lineage id;
# sourced to Leaguepedia rename records). Team IDs not listed are their own organisation.
LINEAGE = pd.read_csv(ROOT / "data/manifests/org_lineage.csv").set_index("teamid").lineage_id

E = pd.read_csv(OUT / "team_patch_exposure.csv").dropna(subset=["buffs_z"])
E["org"] = E.teamid.map(LINEAGE).fillna(E.teamid)
K = pd.read_csv(OUT / "knockout_patch_exposure.csv")[["year", "teamid", "finish"]]
ev = pd.read_csv(OUT / "worlds_events.csv", index_col=0, parse_dates=["start"]).dropna()
W = pd.read_parquet(OUT / "worlds_games_pred.parquet")
raw = load_matches(["gameid", "date", "league", "position", "teamid", "champion", "ban1", "ban2", "ban3", "ban4", "ban5"],
                   years=set(E.year))
raw["date"] = pd.to_datetime(raw.date, errors="coerce")
P = raw[raw.position.isin(["top", "jng", "mid", "bot", "sup"])].dropna(subset=["teamid", "champion"]).rename(columns={"position": "role"})
BAN = raw.loc[raw.position == "team", ["gameid", "teamid", "ban1", "ban2", "ban3", "ban4", "ban5"]].melt(
    id_vars=["gameid", "teamid"], value_name="champion").dropna(subset=["champion"])
pre = pd.concat([pd.read_csv(f, dtype={"patch_oe": str}) for f in coding_files()])
pre = sr_only(pre)
pre["direction"] = pre.direction.str.lower().str.strip()
pre["magnitude"] = pd.to_numeric(pre.magnitude, errors="coerce").fillna(0)
pre["score"] = np.where(pre.direction == "buff", pre.magnitude, np.where(pre.direction == "nerf", -pre.magnitude, 0))
NET = pre.groupby(["year_group", "champion"]).score.sum()


def perm_within_year(df, col):
    x = df[col].to_numpy().copy()
    for ix in df.groupby("year").indices.values():
        x[ix] = x[rng.permutation(ix)]
    return x


# ---------- 3 mechanism inputs: presence, pools, breadth, meta concentration ----------
# Presence reference (R6-3): "all" = every game Oracle's Elixir (plus the Leaguepedia supplement) records in the
# 100 days before Worlds, all leagues including development leagues, each game weighted equally (main); "major" =
# the four major regional leagues only (LCK, LPL, LEC / EU LCS, LCS / NA LCS / LTA); "pre_change" = the 100 days
# before the year's first coded change, so no game played after any coded change enters presence. Team pools and
# breadth always come from the team's own games in the 100 days before Worlds. In 2025 the LTA (North, South and
# cross-conference games) is the fourth major region.
MAJOR = {"LCK", "LPL", "LEC", "EU LCS", "LCS", "NA LCS", "LTA", "LTA N", "LTA S"}
FIRST_CHANGE = pd.to_datetime(pre.release_date).groupby(pre.year_group).min()


def mechanism_inputs(ref="all"):
    champ_rows, team_rows = [], []
    for y in sorted(E.year.unique()):
        T0 = ev.loc[y, "start"]
        win = P[(P.date < T0) & (P.date >= T0 - pd.Timedelta(days=100))]
        end = FIRST_CHANGE.get(y, T0) if ref == "pre_change" else T0
        ref_games = P[(P.date < end) & (P.date >= end - pd.Timedelta(days=100))]
        if ref == "major":
            ref_games = ref_games[ref_games.league.isin(MAJOR)]
        bw = BAN[BAN.gameid.isin(set(ref_games.gameid))]
        ngames = ref_games.gameid.nunique()
        pres = ref_games.champion.value_counts().add(bw.champion.value_counts(), fill_value=0) / ngames
        net = NET.xs(y) if y in NET.index.get_level_values(0) else pd.Series(dtype=float)
        champ_rows.append(pd.DataFrame({"year": y, "champion": pres.index, "presence": pres.to_numpy(),
                                        "net": net.reindex(pres.index).fillna(0).to_numpy()}))
        for t in E[E.year == y].teamid:
            own = win[win.teamid == t]
            if own.gameid.nunique() < 10:
                continue
            rs = own.groupby(["role", "champion"]).size()
            rs = rs / rs.groupby(level="role").transform("sum")
            breadth = rs.groupby(level="role").apply(lambda s: np.exp(-(s * np.log(s)).sum())).mean()
            own_ref = set(own.gameid) | set(ref_games.loc[ref_games.teamid == t, "gameid"])
            others = ref_games[~ref_games.gameid.isin(own_ref)]                   # presence without the team's own games
            ob = bw[bw.gameid.isin(set(others.gameid))]
            pres_o = others.champion.value_counts().add(ob.champion.value_counts(), fill_value=0) / others.gameid.nunique()
            meta = float((rs * pres_o.reindex(rs.index.get_level_values("champion")).fillna(0).to_numpy()).sum() / 5)
            team_rows.append({"year": y, "teamid": t, "breadth": breadth, "meta": meta})
    CH = pd.concat(champ_rows, ignore_index=True)
    CH["class"] = np.where(CH.net > 0, "buffed", np.where(CH.net < 0, "nerfed", "untouched"))
    CH["pres_pct"] = CH.groupby("year").presence.rank(pct=True)
    return CH, pd.DataFrame(team_rows)


CH, TM = mechanism_inputs("all")
champ_tab = CH.groupby("class").agg(n=("champion", "size"), mean_presence=("presence", "mean"),
                                    median_presence_pct=("pres_pct", "median"))
r_champ = np.corrcoef(CH.pres_pct, np.sign(CH.net))[0, 1]
print(champ_tab.round(3).to_string())
print(f"corr(presence percentile, sign of coded change) = {r_champ:.3f} (n={len(CH)})")
CH.to_csv(OUT / "buff_persistence_champion_presence.csv", index=False)

D = E.merge(TM, on=["year", "teamid"])
z = lambda s: (s - s.mean()) / s.std()
for c in ["breadth", "meta"]:
    D[c + "_z"] = D.groupby("year")[c].transform(z)
mech = []
for dep in ["buffs_z", "nerfs_z", "intended_z"]:
    m = smf.ols(f"{dep} ~ breadth_z + meta_z", D).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(D.org)[0]})
    for term in ["breadth_z", "meta_z"]:
        mech.append({"dep": dep, "term": term, "coef": m.params[term], "p": m.pvalues[term], "r2": m.rsquared, "n": int(m.nobs)})
    D[dep + "_res"] = m.resid
MECH = pd.DataFrame(mech)
MECH.to_csv(OUT / "buff_persistence_mechanism.csv", index=False)
print("\n", MECH.round(3).to_string(index=False))

# ---------- inference allowing dependence within a Worlds ----------
# Teams of one Worlds share its patches and meta. Besides organisation clusters (main), report two-way clusters
# (organisation x Worlds) with t(9) reference, and a wild cluster bootstrap over the ten Worlds (Webb weights, null
# imposed, CR1 t statistic).
WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
rng_w = np.random.default_rng(1717)             # separate stream: the permutation tests below keep their draws


def cr1_t(X, y, g, k):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    S = np.zeros((X.shape[1], X.shape[1]))
    for j in range(g.max() + 1):
        s = X[g == j].T @ e[g == j]
        S += np.outer(s, s)
    n, q = X.shape
    G = g.max() + 1
    V = G / (G - 1) * (n - 1) / (n - q) * XtXi @ S @ XtXi
    return b[k] / np.sqrt(V[k, k])


def inference(D, dep, terms=("breadth_z", "meta_z"), B=9999):
    Xdf = sm.add_constant(D[list(terms)])
    X, y = Xdf.to_numpy(), D[dep].to_numpy()
    org, yr = pd.factorize(D.org)[0], pd.factorize(D.year)[0]
    main = sm.OLS(y, Xdf).fit(cov_type="cluster", cov_kwds={"groups": org})
    two = sm.OLS(y, Xdf).fit(cov_type="cluster", cov_kwds={"groups": np.column_stack([org, yr])})
    out = []
    for term in terms:
        k = list(Xdf.columns).index(term)
        Xr = np.delete(X, k, axis=1)
        fr = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
        ur = y - fr
        t0 = cr1_t(X, y, yr, k)
        ts = np.array([cr1_t(X, fr + WEBB[rng_w.integers(0, 6, yr.max() + 1)][yr] * ur, yr, k) for _ in range(B)])
        out.append({"dep": dep, "term": term, "coef": main.params[term], "p_org_cluster": main.pvalues[term],
                    "p_org_worlds_twoway_t9": 2 * stats.t.sf(abs(two.params[term] / two.bse[term]), df=yr.max()),
                    "p_wild_cluster_worlds": (np.sum(np.abs(ts) >= abs(t0)) + 1) / (B + 1), "n": len(D), "worlds": yr.max() + 1})
    return out


INF = pd.DataFrame([r for dep in ["buffs_z", "nerfs_z", "intended_z"] for r in inference(D, dep)])
INF.to_csv(OUT / "buff_persistence_mechanism_inference.csv", index=False)
print("\n", INF.round(4).to_string(index=False))

# ---------- presence reference: which games define the meta ----------
def decile_shares(C):
    dec = np.minimum(np.ceil(C.pres_pct * 10).astype(int), 10)
    top, rest = C[dec == 10], C[dec <= 8]
    return {"top_decile_n": len(top), "top_decile_nerf_pct": 100 * (top.net < 0).mean(),
            "top_decile_buff_pct": 100 * (top.net > 0).mean(), "bottom_eight_nerf_pct": 100 * (rest.net < 0).mean(),
            "bottom_eight_buff_pct": 100 * (rest.net > 0).mean()}


sens = []
for ref in ["all", "major", "pre_change"]:
    C, T_ = (CH, TM) if ref == "all" else mechanism_inputs(ref)
    Dx = E.merge(T_, on=["year", "teamid"])
    for c in ["breadth", "meta"]:
        Dx[c + "_z"] = Dx.groupby("year")[c].transform(z)
    row = {"reference": ref, "rows": len(C), **decile_shares(C)}
    for dep in ["intended_z", "buffs_z", "nerfs_z"]:
        m = smf.ols(f"{dep} ~ breadth_z + meta_z", Dx).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(Dx.org)[0]})
        row[f"{dep}_meta_coef"], row[f"{dep}_meta_p"] = m.params["meta_z"], m.pvalues["meta_z"]
    sens.append(row)
# decision by decision: presence in the 100 days before each coded release (hotfixes are separate releases) and
# whether that release nerfed or buffed the champion (champions present in the window or changed by the release)
dec_rows = []
for (y, patch, rd), g in pre.groupby(["year_group", "patch_oe", "release_date"]):
    rd = pd.Timestamp(rd)
    w = P[(P.date < rd) & (P.date >= rd - pd.Timedelta(days=100))]
    pres = w.champion.value_counts().add(BAN[BAN.gameid.isin(set(w.gameid))].champion.value_counts(), fill_value=0) / w.gameid.nunique()
    net = g.groupby("champion").score.sum()
    champs = pres.index.union(net.index)
    d = pd.DataFrame({"year": y, "release": f"{patch}@{rd.date()}", "champion": champs,
                      "presence": pres.reindex(champs).fillna(0).to_numpy(), "net": net.reindex(champs).fillna(0).to_numpy()})
    d["pres_pct"] = d.presence.rank(pct=True)
    dec_rows.append(d)
DEC = pd.concat(dec_rows, ignore_index=True)
DEC.to_csv(OUT / "presence_by_release.csv", index=False)
sens.append({"reference": "by_release", "rows": len(DEC), "releases": DEC.release.nunique(), **decile_shares(DEC)})
SENS = pd.DataFrame(sens)
SENS.to_csv(OUT / "presence_reference_sensitivity.csv", index=False)
print("\n", SENS.round(3).T.to_string())

# ---------- 1 persistence and 2 org effect (raw and after breadth + meta) ----------
D = D.sort_values(["org", "year"]).reset_index(drop=True)


def persistence(col, D=D, org="org"):
    nxt = D.groupby(org)[col].shift(-1)
    ok = nxt.notna()
    obs = np.corrcoef(D.loc[ok, col], nxt[ok])[0, 1]
    null = []
    for _ in range(N_PERM):
        x = pd.Series(perm_within_year(D, col), index=D.index)
        xn = x.groupby(D[org]).shift(-1)
        null.append(np.corrcoef(x[ok], xn[ok])[0, 1])
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1), int(ok.sum())


def org_effect(col, min_n=3, D=D, org="org"):
    keep = D[org].map(D[org].value_counts()) >= min_n
    d = D[keep].reset_index(drop=True)
    stat = lambda v: pd.Series(v).groupby(d[org].to_numpy()).mean().var()
    obs = stat(d[col].to_numpy())
    null = np.array([stat(perm_within_year(d, col)) for _ in range(N_PERM)])
    return obs, (np.sum(null >= obs) + 1) / (N_PERM + 1), d[org].nunique()


COLS = ["buffs_z", "nerfs_z", "intended_z", "breadth_z", "meta_z", "buffs_z_res", "nerfs_z_res"]
rows = []
for col in COLS:
    r, p, n = persistence(col)
    v, pv, no = org_effect(col)
    rows.append({"definition": "linked lineages", "measure": col, "next_appearance_r": r, "p": p, "pairs": n,
                 "org_var": v, "org_p": pv, "orgs_3plus": no})
# The result depends on what counts as one organisation: repeat with teams as recorded in the data (renamed teams
# separate), and drop each linked organisation with >= 3 appearances in turn.
DR = D.sort_values(["teamid", "year"]).reset_index(drop=True)
for col in COLS:
    r, p, n = persistence(col, DR, "teamid")
    v, pv, no = org_effect(col, D=DR, org="teamid")
    rows.append({"definition": "recorded team IDs", "measure": col, "next_appearance_r": r, "p": p, "pairs": n,
                 "org_var": v, "org_p": pv, "orgs_3plus": no})
PERS = pd.DataFrame(rows)
infl = []
for o in D.org.value_counts()[lambda c: c >= 3].index:
    Dx = D[D.org != o].reset_index(drop=True)
    for col in ["buffs_z", "buffs_z_res"]:
        v, pv, no = org_effect(col, D=Dx)
        infl.append({"dropped_org": o, "team": D.loc[D.org == o, "team"].iloc[-1], "appearances": int((D.org == o).sum()),
                     "mean_buffs_z": float(D.loc[D.org == o, "buffs_z"].mean()), "measure": col, "org_p": pv})
pd.DataFrame(infl).sort_values(["measure", "org_p"]).to_csv(OUT / "buff_persistence_org_influence.csv", index=False)
PERS.to_csv(OUT / "buff_persistence_tests.csv", index=False)
print("\n", PERS.round(3).to_string(index=False))

# ---------- organisations: mean buff exposure over appearances ----------
D["buff_pct"] = D.groupby("year").buffs_z.rank(pct=True)
D["buff_res_pct"] = D.groupby("year").buffs_z_res.rank(pct=True)
D = D.merge(K, on=["year", "teamid"], how="left")
name = D.groupby("org").team.last()
ORGS = D.groupby("org").agg(n=("year", "size"), years=("year", lambda s: ",".join(str(v)[2:] for v in s)),
                            buff_pct=("buff_pct", "mean"), buff_res_pct=("buff_res_pct", "mean"),
                            breadth_z=("breadth_z", "mean"), meta_z=("meta_z", "mean"))
ORGS.insert(0, "team", name)
ORGS = ORGS[ORGS.n >= 3].sort_values("buff_pct", ascending=False)
ORGS.to_csv(OUT / "buff_persistence_orgs.csv")
print("\n", ORGS.round(2).to_string(index=False))
t1 = D[D.org == "T1"][["year", "team", "finish", "buff_pct", "buff_res_pct", "breadth_z", "meta_z"]]
print("\n", t1.round(2).to_string(index=False))
D.to_csv(OUT / "buff_persistence_teams.csv", index=False)
