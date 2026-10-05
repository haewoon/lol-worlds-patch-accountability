"""W29: where quarterfinalists' buff exposure comes from (descriptive; used for the T1 case in Section 4.4).

For every quarterfinalist, the buff exposure sum_{r,c} w(r,c) max(delta(c), 0) of Section 3.3 is split by champion
(summing a champion's roles):
  shared      the part from champions that also carry a net buff in at least one other quarterfinalist's pool that
              year (each team with its own unplayed versions)
  presence    the part by the champion's pick-and-ban presence (picks plus bans per game, all recorded games) in the
              100 days before the year's first coded change, so that no game played after a change counts; bands
              [0, 5%), [5%, 20%), [20%, 100%]
  top decile  the part from champions in the most-contested presence decile of the 100 days before Worlds (the main
              presence definition of Figure 2a)
  top champion the champion(s) contributing most (ties within 1e-9 all listed), their share of the exposure, and the
              top champion's share of the team's picks in its main role
It also records each team's rank among that year's quarterfinalists in raw buff exposure, breadth, meta concentration,
and buff exposure after w17's descriptive linear adjustment for breadth and meta concentration. These are
descriptions of where exposure came from, not of why a team ranked where it did.
Writes OUT/buff_sources.csv (one row per quarterfinalist) and OUT/buff_sources_summary.json.
"""
import json

import numpy as np
import pandas as pd

from worlds_common import OUT, coding_files, event_manifest, load_matches, sr_only, team_delta

BANDS = [(0.0, 0.05, "below_5"), (0.05, 0.20, "5_to_20"), (0.20, np.inf, "20_plus")]      # [lo, hi)
TOL = 1e-9


def main():
    code = sr_only(pd.concat([pd.read_csv(f, dtype={"patch_oe": str}) for f in coding_files()]))
    code["score"] = np.where(code.direction == "buff", code.magnitude, np.where(code.direction == "nerf", -code.magnitude, 0))
    first_change = pd.to_datetime(code.release_date).groupby(code.year_group).min()
    sc = code.groupby(["year_group", "patch_oe", "champion"]).score.sum().reset_index()
    ev = event_manifest()
    X = pd.read_csv(OUT / "team_exposure.csv", dtype={"last_patch": str})
    E = pd.read_csv(OUT / "team_patch_exposure.csv")[["year", "teamid", "buffs"]]
    K = pd.read_csv(OUT / "knockout_patch_exposure.csv")[["year", "teamid", "team", "finish"]]
    S = pd.read_csv(OUT / "buff_persistence_teams.csv")[["year", "teamid", "breadth", "meta", "buffs_z", "buffs_z_res"]]
    assert not K.duplicated(["year", "teamid"]).any() and (K.groupby("year").size() == 8).all()
    CP = pd.read_csv(OUT / "buff_persistence_champion_presence.csv")
    CP["decile"] = np.minimum(np.ceil(CP.pres_pct * 10).astype(int), 10)
    raw = load_matches(["gameid", "date", "position", "teamid", "champion", "ban1", "ban2", "ban3", "ban4", "ban5"])
    raw["date"] = pd.to_datetime(raw.date, errors="coerce")
    P = raw[raw.position.isin(["top", "jng", "mid", "bot", "sup"])].dropna(subset=["teamid", "champion", "date"])
    B = raw.loc[raw.position == "team", ["gameid", "ban1", "ban2", "ban3", "ban4", "ban5"]].melt(
        id_vars=["gameid"], value_name="banned").dropna(subset=["banned"])

    parts = []
    for y in sorted(K.year.unique()):
        start, end = ev.loc[y, "start"], first_change[y]
        ref = P[(P.date < end) & (P.date >= end - pd.Timedelta(days=100))]
        pre = ref.champion.value_counts().add(B[B.gameid.isin(set(ref.gameid))].banned.value_counts(), fill_value=0) / ref.gameid.nunique()
        top = set(CP[(CP.year == y) & (CP.decile == 10)].champion)
        for t in K[K.year == y].itertuples():
            last = X.loc[(X.year == y) & (X.teamid == t.teamid), "last_patch"].iloc[0]
            d = team_delta(sc, y, last)
            own = P[(P.teamid == t.teamid) & (P.date < start) & (P.date >= start - pd.Timedelta(days=100))]
            c = own.groupby(["position", "champion"]).size()
            w = c / c.groupby(level="position").transform("sum")
            for (role, ch), share in w.items():
                if d.get(ch, 0) > 0:
                    parts.append({"year": y, "teamid": t.teamid, "role": role, "champion": ch, "role_share": share,
                                  "contrib": share * d[ch], "pre_presence": pre.get(ch, 0.0), "top_decile": ch in top})
    Q = pd.DataFrame(parts)
    buffed = Q.groupby(["year", "champion"]).teamid.nunique()
    Q["shared"] = [buffed[(y, ch)] > 1 for y, ch in zip(Q.year, Q.champion)]

    rows = []
    for t in K.itertuples():                       # every quarterfinalist, including any with no buff exposure
        g = Q[(Q.year == t.year) & (Q.teamid == t.teamid)]
        tot = g.contrib.sum()
        r = {"year": t.year, "teamid": t.teamid, "team": t.team, "finish": t.finish, "buff_exposure": tot}
        if tot > 0:
            by_ch = g.groupby("champion").contrib.sum().sort_values(ascending=False)
            tops = sorted(by_ch.index[by_ch >= by_ch.iloc[0] - TOL])
            main = g[g.champion == tops[0]].sort_values(["contrib", "role"], ascending=[False, True]).iloc[0]
            r.update(shared_pct=100 * g.loc[g.shared, "contrib"].sum() / tot,
                     top_decile_pct=100 * g.loc[g.top_decile, "contrib"].sum() / tot,
                     top_champions="; ".join(tops), top_champion_pct=100 * by_ch.iloc[0] / tot,
                     top_champion_role=main.role, top_champion_role_share_pct=100 * main.role_share)
            for lo, hi, lab in BANDS:
                r[f"pre_presence_{lab}_pct"] = 100 * g.loc[(g.pre_presence >= lo) & (g.pre_presence < hi), "contrib"].sum() / tot
            assert abs(sum(r[f"pre_presence_{lab}_pct"] for _, _, lab in BANDS) - 100) < 1e-9
        rows.append(r)
    R = pd.DataFrame(rows).merge(S, on=["year", "teamid"], how="left", validate="one_to_one")
    assert R[["breadth", "meta", "buffs_z_res"]].notna().all().all()
    chk = R.merge(E, on=["year", "teamid"], how="left", validate="one_to_one")
    assert np.allclose(chk.buff_exposure, chk.buffs, atol=1e-9), "decomposition does not add up to w8's buff exposure"
    R["buff_rank_qf"] = R.groupby("year").buff_exposure.rank(ascending=False, method="min").astype(int)
    R["buff_adjusted_rank_qf"] = R.groupby("year").buffs_z_res.rank(ascending=False, method="min").astype(int)
    R["breadth_rank_qf"] = R.groupby("year").breadth.rank(ascending=False, method="min").astype(int)
    R["meta_rank_from_least_qf"] = R.groupby("year").meta.rank(ascending=True, method="min").astype(int)
    R.sort_values(["year", "buff_rank_qf"]).to_csv(OUT / "buff_sources.csv", index=False)

    # the case in Section 4.4: the most buff-exposed quarterfinalist of 2022-2025 (T1 in each year)
    years = [2022, 2023, 2024, 2025]
    case = R[(R.buff_rank_qf == 1) & R.year.isin(years)].sort_values("year")
    assert list(case.year) == years and (case.team == "T1").all(), case[["year", "team"]]
    cq = Q.merge(case[["year", "teamid"]], on=["year", "teamid"])
    allq = Q[Q.year.isin(years)]
    band = lambda q, lo, hi: 100 * q.loc[(q.pre_presence >= lo) & (q.pre_presence < hi), "contrib"].sum() / q.contrib.sum()
    y22 = case[case.year == 2022].iloc[0]
    summary = {"case_team": "T1", "years": years,
               "pooled_pre_presence_below_5_pct": band(cq, 0, 0.05), "pooled_pre_presence_20_plus_pct": band(cq, 0.20, np.inf),
               "pre_presence_20_plus_pct_by_year": dict(zip(map(str, case.year), case.pre_presence_20_plus_pct)),
               "all_quarterfinalists_pre_presence_below_5_pct": band(allq, 0, 0.05),
               "all_quarterfinalists_pre_presence_20_plus_pct": band(allq, 0.20, np.inf),
               "top_decile_pct_max": float(case.top_decile_pct.max()),
               "shared_pct_by_year": dict(zip(map(str, case.year), case.shared_pct)),
               "adjusted_rank_by_year": dict(zip(map(str, case.year), case.buff_adjusted_rank_qf.astype(int))),
               "breadth_rank_by_year": dict(zip(map(str, case.year), case.breadth_rank_qf.astype(int))),
               "meta_rank_from_least_by_year": dict(zip(map(str, case.year), case.meta_rank_from_least_qf.astype(int))),
               "y2022_top_champions": y22.top_champions, "y2022_top_champion_pct": float(y22.top_champion_pct),
               "y2022_top_champion_role": y22.top_champion_role,
               "y2022_top_champion_role_share_pct": float(y22.top_champion_role_share_pct),
               "y2022_top_champion_other_quarterfinalists": int(Q[(Q.year == 2022) & (Q.champion == y22.top_champions)
                                                                  & (Q.teamid != y22.teamid)].teamid.nunique())}
    (OUT / "buff_sources_summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
