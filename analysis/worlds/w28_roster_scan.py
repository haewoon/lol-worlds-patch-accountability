"""W28: are participants' pre-Worlds games recorded under the right team?

For every Worlds participant (team-year) and every game in the 365 days before its Worlds, count how many of the
side's five players belong to the team's regular roster: players in at least 20% of the team's other games within
180 days of that game (the same definition as w0c's team-label check). Games with at most one regular player are
listed with the team Leaguepedia names for that side of the matched game (w0 match files). Games restored from
Leaguepedia carry Leaguepedia's own label and are counted separately: for them the comparison is not an independent
check. The scan flags lineups; it does not establish that no other team-label error remains (lineups with two to
four regular players, or Leaguepedia names that are not linked, are not examined).
Writes OUT/roster_scan.csv (one row per flagged team and game) and OUT/roster_scan_summary.json (distinct games).
"""
import json
import re
from pathlib import Path

import pandas as pd

from worlds_common import OUT, event_manifest, load_matches

ROOT = Path(__file__).resolve().parents[2]
SUP = ROOT / "data/supplement"            # slim extracts written by w0 build
ROLES = ["top", "jng", "mid", "bot", "sup"]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    ev = event_manifest()
    W = pd.read_parquet(OUT / "worlds_games_pred.parquet")
    teams = pd.concat([W[["year", "blue_id", "blue_name"]].set_axis(["year", "teamid", "team"], axis=1),
                       W[["year", "red_id", "red_name"]].set_axis(["year", "teamid", "team"], axis=1)]).drop_duplicates(["year", "teamid"])
    P = load_matches(["gameid", "league", "date", "position", "side", "teamid", "teamname", "playername"])
    P = P[P.position.isin(ROLES)].copy()
    P["date"] = pd.to_datetime(P.date, errors="coerce")
    P["player"] = P.playername.map(norm)
    links = pd.read_csv(SUP / "leaguepedia_name_links.csv", dtype={"year": str})
    M = pd.read_csv(SUP / "leaguepedia_oe_matches.csv", dtype=str, keep_default_na=False).set_index("oe_gameid")

    def link(name, year):   # year: the Worlds whose 365-day window the match file covers
        k = links[(links.lp == name) & (links.year == str(year))]
        k = k if len(k) else links[(links.lp == name) & (links.year == "all")]
        return k.teamid.iloc[0] if len(k) else None

    rows, seen = [], set()
    for t in teams.itertuples():
        start = ev.loc[t.year, "start"]
        team = P[P.teamid == t.teamid]
        g = team[(team.date < start) & (team.date >= start - pd.Timedelta(days=365))]
        seen |= set(g.gameid)
        for gid, x in g.groupby("gameid"):
            when = x.date.iloc[0]
            o = team[(team.gameid != gid) & ((team.date - when).abs() <= pd.Timedelta(days=180))]
            c = o.groupby("player").gameid.nunique()
            regular = set(c[c >= 0.2 * o.gameid.nunique()].index)
            k = len(set(x.player) & regular)
            if k > 1:
                continue
            side = x.side.iloc[0]
            if str(gid).startswith("lp:"):
                lp_team, same = x.teamname.iloc[0], None           # restored from Leaguepedia: its own label, not a check
            elif gid in M.index:
                m = M.loc[gid]
                team1_side = "Blue" if m.side == "team1_blue" else "Red"
                lp_team = m.Team1 if side == team1_side else m.Team2
                same = link(lp_team, m.year) == t.teamid or norm(lp_team) == norm(x.teamname.iloc[0])
            else:
                lp_team, same = "", None
            rows.append({"year": t.year, "team": t.team, "gameid": gid, "date": when, "league": x.league.iloc[0],
                         "days_before_worlds": (start - when).days, "regular_players": k,
                         "players": ", ".join(sorted(map(str, x.playername))), "leaguepedia_team": lp_team,
                         "leaguepedia_same_team": same})
    R = pd.DataFrame(rows).sort_values(["year", "team", "date"])
    R.to_csv(OUT / "roster_scan.csv", index=False)
    R["source"] = R.gameid.astype(str).str.startswith("lp:").map({True: "restored from Leaguepedia", False: "Oracle's Elixir"})
    oe = R[R.source == "Oracle's Elixir"]
    summary = {"participant_games": len(seen), "flagged_team_games": int(len(R)), "flagged": int(R.gameid.nunique()),
               "flagged_team_years": int(R[["year", "team"]].drop_duplicates().shape[0]),
               "flagged_oe_games": int(oe.gameid.nunique()), "flagged_oe_team_games": int(len(oe)),
               "flagged_restored_games": int(R.loc[R.source != "Oracle's Elixir", "gameid"].nunique()),
               "flagged_restored_team_games": int((R.source != "Oracle's Elixir").sum()),
               "oe_leaguepedia_same_team": int((oe.leaguepedia_same_team == True).sum()),  # noqa: E712
               "oe_leaguepedia_other_team": int((oe.leaguepedia_same_team == False).sum()),  # noqa: E712
               "oe_not_in_leaguepedia": int(oe.leaguepedia_same_team.isna().sum())}
    (OUT / "roster_scan_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(summary)
    print(oe[oe.leaguepedia_same_team != True].to_string(index=False))  # noqa: E712


if __name__ == "__main__":
    main()
