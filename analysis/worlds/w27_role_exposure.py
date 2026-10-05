"""W27: how the patch incidence divides across roles (descriptive).

Role exposure of team i in role r = sum_c w_i(r, c) * delta_i(c), with the same role pick shares (100 days before
Worlds) and team-specific net changes as the team exposure (w8); the five role exposures sum to the team's net
exposure. Reports, by role, the mean share of picks on subsequently nerfed / buffed champions with 95% intervals
from resampling team-years, and how concentrated a team's exposure is in its most affected role (a sole largest
role is counted by role; tied largest roles and teams with no exposure are counted separately).
Writes OUT/role_exposure.csv (team-year x role) and OUT/role_exposure_summary.csv / .json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from worlds_common import OUT, ROOT, coding_files, load_matches, team_delta, sr_only

ROLES = ["top", "jng", "mid", "bot", "sup"]
rng = np.random.default_rng(27)
E = pd.read_csv(OUT / "team_patch_exposure.csv").dropna(subset=["intended"])
LAST = pd.read_csv(OUT / "team_exposure.csv", dtype={"last_patch": str}).set_index(["year", "teamid"]).last_patch
ev = pd.read_csv(OUT / "worlds_events.csv", index_col=0, parse_dates=["start"], dtype={"main_patch": str}).dropna()
raw = load_matches(["gameid", "date", "position", "teamid", "champion"], years=set(E.year))
raw["date"] = pd.to_datetime(raw.date, errors="coerce")
P = raw[raw.position.isin(ROLES)].dropna(subset=["teamid", "champion"]).rename(columns={"position": "role"})

pre = pd.concat([pd.read_csv(f, dtype={"patch_oe": str}) for f in coding_files()])
pre = sr_only(pre)
pre["magnitude"] = pd.to_numeric(pre.magnitude, errors="coerce").fillna(0)
d = pre.direction.str.lower().str.strip()
pre["score"] = np.where(d == "buff", pre.magnitude, np.where(d == "nerf", -pre.magnitude, 0))

rows = []
for _, e in E.iterrows():
    y, t, T0 = e.year, e.teamid, ev.loc[e.year, "start"]
    picks = P[(P.teamid == t) & (P.date < T0) & (P.date >= T0 - pd.Timedelta(days=100))]
    share = picks.groupby(["role", "champion"]).size()
    share = share / share.groupby(level="role").transform("sum")
    delta = team_delta(pre, y, LAST.loc[(y, t)]).reindex(share.index.get_level_values("champion")).fillna(0).to_numpy()
    for role in ROLES:
        m = share.index.get_level_values("role") == role
        w, x = share.to_numpy()[m], delta[m]
        rows.append({"year": y, "teamid": t, "team": e.team, "role": role, "net": (w * x).sum(),
                     "buffs": (w * np.clip(x, 0, None)).sum(), "nerfs": (w * np.clip(-x, 0, None)).sum(),
                     "nerf_share": w[x < 0].sum(), "buff_share": w[x > 0].sum()})
R = pd.DataFrame(rows)
check = R.groupby(["year", "teamid"]).net.sum().sub(E.set_index(["year", "teamid"]).intended).abs().max()
assert check < 1e-9, check                                   # role exposures add up to the team's net exposure
R.to_csv(OUT / "role_exposure.csv", index=False)

wide = {k: R.pivot_table(index=["year", "teamid"], columns="role", values=k)[ROLES] for k in ["nerf_share", "buff_share", "net"]}
boot = rng.integers(0, len(wide["net"]), (4000, len(wide["net"])))
ci = lambda df: np.quantile(np.stack([df.to_numpy()[b].mean(0) for b in boot]), [.025, .975], axis=0)
S = pd.DataFrame({"role": ROLES})
for k in ["nerf_share", "buff_share"]:
    lo, hi = ci(wide[k])
    S[k + "_pct"], S[k + "_lo"], S[k + "_hi"] = 100 * wide[k].mean().to_numpy(), 100 * lo, 100 * hi
S["net_sd"] = wide["net"].std().to_numpy()
S.to_csv(OUT / "role_exposure_summary.csv", index=False)
absnet = wide["net"].abs()
zero = absnet.max(axis=1) <= 1e-10                                   # e.g. a team that had played the Worlds patch
top = absnet[~zero].apply(lambda r: tuple(r.index[np.isclose(r, r.max(), rtol=0, atol=1e-10)]), axis=1)
sole = top[top.map(len) == 1].map(lambda t: t[0])
top_share = (absnet.max(axis=1) / absnet.sum(axis=1))[~zero]
J = {"n_team_years": len(wide["net"]), "n_zero_exposure": int(zero.sum()), "n_tied_largest": int((top.map(len) > 1).sum()),
     "n_sole_largest": len(sole), "sole_largest_counts": sole.value_counts().reindex(ROLES, fill_value=0).to_dict(),
     "jng_or_bot_among_tied": int(top[top.map(len) > 1].map(lambda t: bool({"jng", "bot"} & set(t))).sum()),
     "n_share": len(top_share), "top_role_share_of_abs_net_median": float(top_share.median()),
     "top_role_share_iqr": top_share.quantile([.25, .75]).tolist(),
     "role_net_corr": wide["net"].corr().round(3).to_dict()}
(OUT / "role_exposure_summary.json").write_text(json.dumps(J, indent=2) + "\n")
print(S.round(1).to_string(index=False))
print(json.dumps({k: v for k, v in J.items() if k != "role_net_corr"}, indent=1))
