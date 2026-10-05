"""W8: patch-notes-based exposure — did the Worlds patch itself favour anyone?

Inputs (blind-coded; coders never saw teams or results):
  data/patch_notes/coded_{A..E}.csv  champion changes in the patches between teams'
                                     last domestic patch and the Worlds patch
  data/patch_notes/coded_R.csv       independent second coder (2019, 2022, 2025)
  data/patch_notes/post_{A..E}.csv   champion changes in the 5 patches after Worlds,
                                     with the stated reason for each change

  score(p, c)   = +magnitude (buff), -magnitude (nerf), 0 otherwise; summed with hotfixes
  unseen_i      = coded pre-Worlds patches newer than team i's last official game
  delta_i(c)    = sum of score(p, c) over unseen_i          (what the patch did, as intended)
  post(c)       = sum of post-Worlds scores, excluding changes whose reason is pro_play or
                  system_compensation (those can react to Worlds itself or to preseason)
  pool_i        = team i's pick shares by role, 100 days before Worlds

Exposures (higher = more favoured):
  intended   sum pool * delta                               patch notes taken at face value
  walkback   sum pool * (-post) * 1{sign(post) = -sign(delta)}
             changes Riot later reversed: an over-buff counts as an unintended advantage,
             an over-nerf (later compensated) as an unintended disadvantage
  touched    sum pool * (-post) * 1{delta != 0}             Worlds-state of the champions the
                                                           patch touched (later nerfed = was strong)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
from worlds_common import OUT, coding_files, team_delta, sr_only, assign_series, champion_unavailable, load_matches
PN = ROOT / "data" / "patch_notes"
rng = np.random.default_rng(8)
N_PERM = N_BOOT = 2000
EXCLUDE_POST = {"pro_play", "system_compensation"}


def pkey(p):
    a, b = str(p).split(".")
    return int(a) * 100 + int(b)


names = set(open(PN / "champion_names.txt").read().split("\n")) - {""}
DIRS = {"buff", "nerf", "adjust", "rework", "bugfix"}


def load(files):
    df = pd.concat([pd.read_csv(f, dtype={"patch_oe": str, "refers_to_patch": str}) for f in files], ignore_index=True)
    df["patch_oe"] = df.patch_oe.astype(str).str.strip()
    df["direction"] = df.direction.astype(str).str.strip().str.lower()
    df["champion"] = df.champion.astype(str).str.strip()
    df["magnitude"] = pd.to_numeric(df.magnitude, errors="coerce").fillna(0)
    df = sr_only(df)                                   # rows that applied only to other modes
    bad = df[~df.direction.isin(DIRS)]
    if len(bad):
        print("  unknown directions:", bad[["patch_oe", "champion", "direction"]].to_dict("records"))
    miss = sorted(set(df.champion) - names)
    if miss:
        print("  names not in list:", miss)
    df["score"] = np.where(df.direction == "buff", df.magnitude, np.where(df.direction == "nerf", -df.magnitude, 0.0))
    df["year_group"] = df.year_group.astype(int)
    return df


print("pre-Worlds coding:")
pre = load(coding_files())
print(" ", len(pre), "rows;", pre.groupby("year_group").patch_oe.unique().apply(sorted).to_dict())
post_files = sorted(PN.glob("post_[A-E].csv"))
post = load(post_files) if post_files else pd.DataFrame(columns=["year_group", "champion", "score", "reason"])
if len(post):
    post["reason"] = post.reason.astype(str).str.strip().str.lower()
    print("post-Worlds coding:", len(post), "rows; reasons", post.reason.value_counts().to_dict())

# ---------- inter-coder reliability ----------
if (PN / "coded_R.csv").exists():
    sec = load([PN / "coded_R.csv"])
    agg = lambda d: d.groupby(["patch_oe", "champion"]).agg(score=("score", "sum"),
                                                          direction=("direction", lambda s: "/".join(sorted(set(s)))))
    m = agg(pre[pre.patch_oe.isin(set(sec.patch_oe))]).join(agg(sec), how="outer", lsuffix="_1", rsuffix="_2")
    m[["score_1", "score_2"]] = m[["score_1", "score_2"]].fillna(0)
    s1, s2 = np.sign(m.score_1).astype(int), np.sign(m.score_2).astype(int)
    po = np.mean(s1 == s2)
    pe = sum(np.mean(s1 == k) * np.mean(s2 == k) for k in (-1, 0, 1))
    nz = (s1 != 0) | (s2 != 0)
    rel = {"champion_patch_rows": len(m), "only_coder1": int(m.direction_2.isna().sum()),
           "only_coder2": int(m.direction_1.isna().sum()), "sign_agreement": po, "kappa_sign": (po - pe) / (1 - pe),
           "sign_agreement_when_either_nonzero": np.mean(s1[nz] == s2[nz]),
           "score_corr": np.corrcoef(m.score_1, m.score_2)[0, 1]}
    print("reliability:", {k: round(v, 3) if isinstance(v, float) else v for k, v in rel.items()})
    m.reset_index().to_csv(OUT / "patch_coding_reliability_rows.csv", index=False)
    pd.DataFrame([rel]).to_csv(OUT / "patch_coding_reliability.csv", index=False)
    SEC = sec
else:
    SEC = None

# ---------- data ----------
cols = ["gameid", "league", "date", "position", "teamid", "champion", "ban1", "ban2", "ban3", "ban4", "ban5"]
raw = load_matches(cols)
raw["date"] = pd.to_datetime(raw.date, errors="coerce")
P = raw[raw.position.isin(["top", "jng", "mid", "bot", "sup"])].dropna(subset=["teamid", "champion", "date"]).rename(columns={"position": "role"})
BN = raw.loc[raw.position == "team", ["gameid", "ban1", "ban2", "ban3", "ban4", "ban5"]].melt(id_vars=["gameid"], value_vars=["ban1", "ban2", "ban3", "ban4", "ban5"],
                                      value_name="champion").dropna(subset=["champion"])
W = pd.read_parquet(OUT / "worlds_games_pred.parquet").sort_values("date").reset_index(drop=True)
ev = pd.read_csv(OUT / "worlds_events.csv", index_col=0, parse_dates=["start"], dtype={"main_patch": str}).dropna()
X = pd.read_csv(OUT / "team_exposure.csv", dtype={"last_patch": str, "worlds_patch": str})


def shares(df):
    c = df.groupby(["role", "champion"]).size()
    return c / c.groupby(level="role").transform("sum")


def presence(gids):
    g = set(gids)
    pk = P[P.gameid.isin(g)].groupby("champion").size()
    bn = BN[BN.gameid.isin(g)].groupby("champion").size()
    return pk.add(bn, fill_value=0) / len(g)


def exposures(code):
    """team exposures for a given pre-Worlds coding table"""
    sc = code.groupby(["year_group", "patch_oe", "champion"]).score.sum().reset_index()
    sc["pk"] = sc.patch_oe.map(pkey)
    pst = post[~post.reason.isin(EXCLUDE_POST)] if len(post) else post
    post_net = pst.groupby(["year_group", "champion"]).score.sum() if len(pst) else pd.Series(dtype=float)
    post_all = post.groupby(["year_group", "champion"]).score.sum() if len(post) else pd.Series(dtype=float)
    rows = []
    for y, e in ev.iterrows():
        if y not in set(W.year) or y not in set(sc.year_group):
            continue
        T, sy = e.start, sc[sc.year_group == y]
        coded = sorted(sy.pk.unique())
        pn = post_net.xs(y) if len(post_net) and y in post_net.index.get_level_values(0) else pd.Series(dtype=float)
        pa = post_all.xs(y) if len(post_all) and y in post_all.index.get_level_values(0) else pd.Series(dtype=float)
        for _, t in X[X.year == y].iterrows():
            last = pkey(t.last_patch) if isinstance(t.last_patch, str) else coded[0] - 1
            unseen = [p for p in coded if p > max(last, coded[0] - 1)]
            delta = team_delta(sc, y, t.last_patch)
            pool_rows = P[(P.teamid == t.teamid) & (P.date < T) & (P.date >= T - pd.Timedelta(days=100))]
            r = {"year": y, "teamid": t.teamid, "team": t.team, "major": t.major, "n_unseen_patches": len(unseen)}
            if pool_rows.gameid.nunique() >= 10:
                pool = shares(pool_rows)
                w = pool.to_numpy()
                ch = pool.index.get_level_values("champion")
                d = delta.reindex(ch).fillna(0).to_numpy()
                q = -pn.reindex(ch).fillna(0).to_numpy()          # later nerf -> +, later buff -> -
                qa = -pa.reindex(ch).fillna(0).to_numpy()
                walk = (np.sign(q) == np.sign(d)) & (d != 0)       # -post opposite to delta <=> sign(q) == sign(d)
                r.update(intended=float((w * d).sum()), nerfs=float((w * np.clip(-d, 0, None)).sum()),
                         buffs=float((w * np.clip(d, 0, None)).sum()),
                         walkback=float((w * q * walk).sum()),
                         walkback_all_reasons=float((w * qa * ((np.sign(qa) == np.sign(d)) & (d != 0))).sum()),
                         touched=float((w * q * (d != 0)).sum()),
                         pool_share_changed=float(w[d != 0].sum() / 5),
                         buff_share=float(w[d > 0].sum() / 5), nerf_share=float(w[d < 0].sum() / 5))
            rows.append(r)
    E = pd.DataFrame(rows)
    for c in ["intended", "nerfs", "buffs", "walkback", "walkback_all_reasons", "touched"]:
        if c in E:
            sd = E.groupby("year")[c].transform("std")
            E[c + "_z"] = ((E[c] - E.groupby("year")[c].transform("mean")) / sd).where(sd > 0, 0.0)
    return E


E = exposures(pre)
E.to_csv(OUT / "team_patch_exposure.csv", index=False)
print(E.groupby("year")[["intended", "walkback", "pool_share_changed", "n_unseen_patches"]].agg(["mean", "std"]).round(3).to_string())

# ---------- validation: do coded buffs/nerfs move Worlds presence? ----------
UNAVAILABLE = champion_unavailable()
chk = []
for y, e in ev.iterrows():
    if y not in set(W.year) or y not in set(pre.year_group):
        continue
    wy = W[W.year == y]
    homes = set(wy.blue_home) | set(wy.red_home)
    pre_g = P[(P.date < e.start) & (P.date >= e.start - pd.Timedelta(days=45)) & P.league.isin(homes)].gameid.unique()
    net = pre[pre.year_group == y].groupby("champion").score.sum()
    c = pd.DataFrame({"net": net}).join(presence(pre_g).rename("pres_pre")).join(presence(wy.gameid).rename("pres_w")).fillna(0)
    c["d_presence"], c["year"] = c.pres_w - c.pres_pre, y
    c = c[[(y, ch) not in UNAVAILABLE.champion_years for ch in c.index]]   # not selectable for all or part of Worlds
    chk.append(c.reset_index())
CC = pd.concat(chk)
CC.to_csv(OUT / "champion_patch_vs_presence.csv", index=False)
ch = CC[CC.net != 0]
print(f"validation: corr(net coded change, Worlds presence change) = {np.corrcoef(ch.net, ch.d_presence)[0, 1]:.3f} "
      f"(n={len(ch)}); mean presence change buffed {ch[ch.net > 0].d_presence.mean():+.3f}, nerfed {ch[ch.net < 0].d_presence.mean():+.3f}")

# ---------- pooled game-level test ----------
HAVE_POST = len(post) > 0
EXPO = [c for c in ["intended_z", "nerfs_z", "buffs_z", "walkback_z", "walkback_all_reasons_z", "touched_z"]
        if c in E and (HAVE_POST or not c.startswith(("walkback", "touched")))]
ei = E.set_index(["year", "teamid"])[EXPO]
Bv, Rv = ei.reindex(list(zip(W.year, W.blue_id))).to_numpy(), ei.reindex(list(zip(W.year, W.red_id))).to_numpy()
for k, f in enumerate(EXPO):
    W["d_" + f] = Bv[:, k] - Rv[:, k]
D = W.dropna(subset=["d_intended_z"]).reset_index(drop=True)


def fit(df, terms, y="result", studentized=False):
    if y == "result":
        m = sm.Logit(df.result, sm.add_constant(df[["pred_logit"] + terms])).fit(disp=0)
    else:
        d = df.dropna(subset=[y])
        m = sm.OLS(d[y], sm.add_constant(d[["pred_logit"] + terms])).fit()
    return m.tvalues if studentized else m.params


def perm(df):
    out = df.copy()
    for yy, g in E.dropna(subset=["intended_z"]).groupby("year"):
        mp = dict(zip(g.teamid, g[EXPO].to_numpy()[rng.permutation(len(g))]))
        s = out.year == yy
        b = np.array([mp.get(t, [np.nan] * len(EXPO)) for t in out.loc[s, "blue_id"]])
        r = np.array([mp.get(t, [np.nan] * len(EXPO)) for t in out.loc[s, "red_id"]])
        for k, f in enumerate(EXPO):
            out.loc[s, "d_" + f] = b[:, k] - r[:, k]
    return out.dropna(subset=["d_intended_z"])


models = {"intended": (["d_intended_z"], "result"), "nerfs + buffs": (["d_nerfs_z", "d_buffs_z"], "result"),
          "intended, gold@15": (["d_intended_z"], "gd15")}
if "walkback_z" in EXPO:
    models.update({"walkback": (["d_walkback_z"], "result"),
                   "walkback (all reasons)": (["d_walkback_all_reasons_z"], "result"),
                   "touched state": (["d_touched_z"], "result")})
obs = {k: fit(D, t, o) for k, (t, o) in models.items()}
obs_stat = {k: fit(D, t, o, studentized=True) for k, (t, o) in models.items()}
pp, bb = {k: [] for k in models}, {k: [] for k in models}
for _ in range(N_PERM):
    Dp = perm(D)
    for k, (t, o) in models.items():
        pp[k].append(fit(Dp, t, o, studentized=True))
yrs = D.year.unique()
for _ in range(N_BOOT):
    Db = pd.concat([D[D.year == yy] for yy in rng.choice(yrs, len(yrs))])
    for k, (t, o) in models.items():
        bb[k].append(fit(Db, t, o))
res = []
for k, (terms, o) in models.items():
    Pp, Bb = pd.DataFrame(pp[k]), pd.DataFrame(bb[k])
    for t in terms:
        est = obs[k][t]
        res.append({"model": k, "term": t, "estimate": est, "boot_lo": Bb[t].quantile(.025), "boot_hi": Bb[t].quantile(.975),
                    "p_perm": (np.sum(np.abs(Pp[t]) >= abs(obs_stat[k][t])) + 1) / (len(Pp) + 1),
                    "permutation_statistic": "absolute model-based coefficient / standard error",
                    "winprob_pp_per_sd": est / 4 * 100 if o == "result" else np.nan,
                    "pred_logit_coef": obs[k]["pred_logit"], "n_games": len(D)})
res = pd.DataFrame(res)
res.to_csv(OUT / "patch_models.csv", index=False)
print("\n", res.drop(columns=["pred_logit_coef"]).round(4).to_string())

# ---------- champion focus ----------
W["pair"] = W.apply(lambda r: tuple(sorted([r.blue_id, r.red_id])), axis=1)
W["series"], W["game_no"] = assign_series(W)
RAW_EXPO = [c[:-2] for c in EXPO]
ko_rows, crow = [], []
for y, w in W.groupby("year"):
    ser = w.groupby("series").agg(start=("date", "min"), pair=("pair", "first")).sort_values("start")
    ko = ser.tail(7)
    ko_teams = sorted({t for p in ko.pair for t in p})
    last = w[w.series == ko.index[-1]].iloc[-1]
    champ = last.blue_id if last.result == 1 else last.red_id
    runner = last.red_id if champ == last.blue_id else last.blue_id
    sf = {t for p in ko.pair.iloc[4:6] for t in p}
    ey = E[E.year == y].set_index("teamid")
    for t in ko_teams:
        ko_rows.append({"year": y, "teamid": t, "team": ey.team.get(t),
                        "finish": "Champion" if t == champ else "Runner-up" if t == runner else "SF" if t in sf else "QF",
                        **{c: ey[c].get(t) for c in RAW_EXPO + EXPO + ["pool_share_changed", "n_unseen_patches"]}})
    # title probability of the champion with actual vs year-average exposure (per exposure model)
    kg = w[w.series.isin(ko.index) & ((w.blue_id == champ) | (w.red_id == champ))]
    sign = np.where(kg.blue_id == champ, 1, -1)
    row = {"year": y, "champion": ey.team.get(champ), "runner_up": ey.team.get(runner)}
    for mname, col in [("intended", "intended_z"), ("walkback", "walkback_z")]:
        if mname not in obs:
            continue
        ez = E.set_index(["year", "teamid"])[col]
        zb, zr = ez.reindex(list(zip(kg.year, kg.blue_id))).to_numpy(), ez.reindex(list(zip(kg.year, kg.red_id))).to_numpy()
        zc = ez.get((y, champ), np.nan)
        rr = res[res.model == mname].iloc[0]
        for lab, g in [("est", rr.estimate), ("lo", rr.boot_lo), ("hi", rr.boot_hi)]:
            def title(neutral):
                L = sign * (rr.pred_logit_coef * kg.pred_logit.to_numpy() + g * (zb - zr)) - (g * zc if neutral else 0)
                p, out = 1 / (1 + np.exp(-L)), 1.0
                for _, gp in kg.assign(p=p).groupby("series"):
                    qq = gp.p.mean(); out *= qq ** 3 * (1 + 3 * (1 - qq) + 6 * (1 - qq) ** 2)
                return out
            row[f"title_pp_{mname}_{lab}"] = (title(False) - title(True)) * 100
    crow.append(row)
K = pd.DataFrame(ko_rows)
K.to_csv(OUT / "knockout_patch_exposure.csv", index=False)
C = pd.DataFrame(crow)
tests = []
for c in RAW_EXPO:
    K[f"rank_{c}"] = K.groupby("year")[c].rank(ascending=False, method="average")
    ch = K[K.finish == "Champion"].set_index("year")
    ru = K[K.finish == "Runner-up"].set_index("year")
    ranks = [g[f"rank_{c}"].to_numpy() for _, g in K.groupby("year")]
    obs_r = ch[f"rank_{c}"].mean()
    null = np.array([np.mean([r[rng.integers(len(r))] for r in ranks]) for _ in range(100000)])
    tests.append({"exposure": c, "champ_mean_rank_among_8": obs_r, "null_mean": null.mean(),
                  "p_favoured_one_sided": np.mean(null <= obs_r), "p_disfavoured_one_sided": np.mean(null >= obs_r),
                  "champ_gt_runner": int((ch[c] > ru[c]).sum()), "champ_lt_runner": int((ch[c] < ru[c]).sum()),
                  "ties": int((ch[c] == ru[c]).sum()),
                  **{f"mean_rank_{f}": K[K.finish == f][f"rank_{c}"].mean() for f in ["Champion", "Runner-up", "SF", "QF"]}})
    C = C.merge(ch[[c, f"rank_{c}"]].reset_index(), on="year")
T = pd.DataFrame(tests)
T.to_csv(OUT / "champion_patch_rank_test.csv", index=False)
C.to_csv(OUT / "champion_patch_exposure.csv", index=False)
print("\n", T.round(3).to_string())
print("\n", C.round(2).to_string())

# ---------- sensitivity: second coder's codes for 2019/2022/2025 ----------
if SEC is not None:
    alt = pd.concat([pre[~pre.year_group.isin(set(SEC.year_group))], SEC], ignore_index=True)
    E2 = exposures(alt)
    mm = E.merge(E2, on=["year", "teamid"], suffixes=("", "_c2"))
    mm = mm[mm.year.isin(set(SEC.year_group))].dropna(subset=["intended", "intended_c2"])
    print("\nteam exposure corr, coder 1 vs coder 2 (2019/2022/2025):",
          round(np.corrcoef(mm.intended, mm.intended_c2)[0, 1], 3),
          "| walkback:", round(np.corrcoef(mm.walkback, mm.walkback_c2)[0, 1], 3) if "walkback" in mm else "n/a")
