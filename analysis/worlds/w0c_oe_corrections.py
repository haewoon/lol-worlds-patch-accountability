"""W0c: corrections to Oracle's Elixir (OE) records, each supported by at least two independent lines of evidence.

Runs after w0a (record checks) and w0 build (Leaguepedia matching, which writes the match files and name links).
Sources of evidence for a game's winner (a change needs two sources that agree and none that contradicts):
  lp       the winner in the Leaguepedia game matched one-to-one to the OE record (by the ten picks)
  OE       the OE record: its team-row result, or, when no winner is recorded, its end-of-game statistics (the side
           with at least two more towers destroyed and more total gold; otherwise inconclusive). The result and the
           statistics come from the same record and count as one source; statistics that w0a found copied from
           another game (stats_copy) are not used
  external data/manifests/oe_external_evidence.csv: a Games of Legends game page, or a Liquipedia series score that
           forces the game's winner given the other games of the series (Liquipedia's order of maps is not used)
Decisions (data/manifests/oe_corrections.csv):
  no winner recorded        set the winner when two sources agree and none contradicts; otherwise exclude the game
  OE and Leaguepedia differ the sources contradict each other: exclude the game
  mixed record (w0a)        keep (ignoring the team-row bans) only when Leaguepedia, the OE result and the statistics
                            agree and the teams sit on the same sides in both sources; otherwise exclude
  duplicate with different team pairs (w0a)  keep the record whose team pair the Leaguepedia game matched to any
                            record of the group names (through the name links); drop the others, or all of them when
                            none is confirmed
  statistics copied (w0a)   keep the record only when Leaguepedia's matched game records the same winner; otherwise
                            exclude it
  team label                Leaguepedia names another (linked) team for a side: relabel the side when at least three of
                            its players belong to that team's regular roster (players in 20%+ of its games within 180
                            days) and at most one to the recorded team's; otherwise keep the record
Safeguards (the script stops if one fails): an OE record may match at most one Leaguepedia game, the copies of one
game at most one Leaguepedia game in total, and evidence
taken through another record of the same game requires the same champions on the same sides.
The analyses read the corrected data through worlds_common.load_matches(); the source files are never modified.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "data/manifests"
CACHE = ROOT / "data/leaguepedia_cache"
YEARS = range(2016, 2026)


def main():
    checks = pd.read_csv(MAN / "oe_record_checks.csv")
    ext = pd.read_csv(MAN / "oe_external_evidence.csv").set_index("gameid")
    links = pd.read_csv(CACHE / "name_links.csv", dtype={"year": str})
    M = pd.concat([pd.read_csv(CACHE / f"match_{y}.csv", dtype=str, keep_default_na=False) for y in YEARS])
    M = M[M.oe_gameid != ""]
    n_lp = M.groupby("oe_gameid").GameId.nunique()                 # a record may appear in two years' match files
    assert (n_lp <= 1).all(), ("OE records matched to several Leaguepedia games", list(n_lp[n_lp > 1].index))
    M = M.drop_duplicates("oe_gameid").set_index("oe_gameid")

    cols = ["gameid", "position", "side", "teamid", "teamname", "result", "towers", "totalgold", "league", "date"]
    files = sorted((ROOT / "data").glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
    T = pd.concat([pd.read_csv(f, usecols=cols, low_memory=False) for f in files])
    T = T[T.position == "team"].drop_duplicates(["gameid", "side"])
    W = T.pivot(index="gameid", columns="side", values=["teamid", "teamname", "result", "towers", "totalgold", "league", "date"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]

    def link(name, year):
        """year: the Worlds whose 365-day window the match file covers (as in w0's link table)"""
        k = links[(links.lp == name) & (links.year == str(year))]
        k = k if len(k) else links[(links.lp == name) & (links.year == "all")]
        return k.teamid.iloc[0] if len(k) else None

    copied = set(checks.loc[checks.check == "stats_copy", "gameid"])

    def evidence(gid, via=None):
        """via: another record of the same game (a duplicate) whose Leaguepedia match stands for this record's"""
        w = W.loc[gid]
        names = {"Blue": w.teamname_Blue, "Red": w.teamname_Red}
        oe = ("Blue" if w.result_Blue == 1 else "Red") if w.result_Blue + w.result_Red == 1 else None
        dt, dg = w.towers_Blue - w.towers_Red, w.totalgold_Blue - w.totalgold_Red
        stats = "Blue" if dt >= 2 and dg > 0 else "Red" if dt <= -2 and dg < 0 else None
        if gid in copied:                              # statistics copied from another game: not evidence
            stats = None
        lp = sides_ok = None
        via = gid if via is None else via
        if via in M.index and M.loc[via, "Winner"] in ("1", "2"):
            m = M.loc[via]
            team1_side = "Blue" if m.side == "team1_blue" else "Red"
            lp = team1_side if m.Winner == "1" else ("Red" if team1_side == "Blue" else "Blue")
            t1, t2 = link(m.Team1, m.year), link(m.Team2, m.year)
            other = "Red" if team1_side == "Blue" else "Blue"
            sides_ok = t1 == w[f"teamid_{team1_side}"] and t2 == w[f"teamid_{other}"]
        external = None
        if gid in ext.index:
            e = ext.loc[gid]
            external = "Blue" if e.winner == names["Blue"] else "Red" if e.winner == names["Red"] else "?"
        return {"blue": names["Blue"], "red": names["Red"], "league": w.league_Blue, "date": w.date_Blue,
                "oe": oe, "stats": stats, "lp": lp, "external": external, "sides_consistent": sides_ok,
                "external_source": ext.loc[gid, "source"] if gid in ext.index else "",
                "external_url": ext.loc[gid, "url"] if gid in ext.index else "",
                "external_basis": ext.loc[gid, "basis"] if gid in ext.index else ""}

    rows = []
    # 1 games without a recorded winner
    for gid in checks.loc[checks.check == "no_winner", "gameid"]:
        ev = evidence(gid)
        lines = [v for v in (ev["lp"], ev["stats"], ev["external"]) if v]
        mixed = gid in set(checks.loc[checks.check == "mixed", "gameid"])
        if mixed and not ev["sides_consistent"]:
            action, winner, why = "exclude", None, "mixed record whose teams sit on different sides in the two sources"
        elif len(lines) >= 2 and len(set(lines)) == 1:
            action, winner, why = "set_winner", lines[0], "two or more sources agree"
        else:
            action, winner, why = "exclude", None, "fewer than two agreeing sources, or sources contradict"
        rows.append({"gameid": gid, "issue": "no winner recorded", "action": action, "winner_side": winner, "reason": why, **ev})
    # 2 OE and Leaguepedia disagree on the winner (all matched games)
    done = {r["gameid"] for r in rows}
    for gid in M.index:
        if gid in done or gid not in W.index:
            continue
        w = W.loc[gid]
        if w.result_Blue + w.result_Red != 1 or M.loc[gid, "Winner"] not in ("1", "2"):
            continue
        ev = evidence(gid)
        if ev["lp"] == ev["oe"]:
            continue
        rows.append({"gameid": gid, "issue": "OE and Leaguepedia winners differ", "action": "exclude", "winner_side": None,
                     "reason": "the two sources contradict each other", **ev})
    # 3 mixed records that w0a kept for checking
    done = {r["gameid"] for r in rows}
    for gid in checks.loc[(checks.check == "mixed") & (checks.action != "drop"), "gameid"]:
        if gid in done:
            continue
        ev = evidence(gid)
        ok = ev["lp"] is not None and ev["lp"] == ev["oe"] == ev["stats"] and ev["sides_consistent"]
        rows.append({"gameid": gid, "issue": "team rows from another game", "action": "keep_ignore_team_bans" if ok else "exclude",
                     "winner_side": ev["oe"] if ok else None,
                     "reason": "Leaguepedia, OE result and statistics agree; teams on the same sides" if ok
                     else "result not confirmed by two independent lines", **ev})
    # 4 duplicates naming different team pairs: the records hold one game, so the Leaguepedia game matched to any of
    #   them names the pair. Every duplicate group may match at most one Leaguepedia game.
    for grp, g in checks[checks.check == "duplicate"].groupby("group"):
        lp_ids = {M.loc[gid, "GameId"] for gid in g.gameid if gid in M.index}
        assert len(lp_ids) <= 1, ("copies of one game matched to several Leaguepedia games", list(g.gameid), lp_ids)
    amb = checks[checks.action == "resolve teams in w0c"]
    for grp, g in amb.groupby("group"):
        lp_pairs = {frozenset([link(M.loc[gid, "Team1"], M.loc[gid, "year"]), link(M.loc[gid, "Team2"], M.loc[gid, "year"])])
                    for gid in g.gameid if gid in M.index}
        confirmed = [gid for gid in sorted(g.gameid)
                     if frozenset([W.loc[gid, "teamid_Blue"], W.loc[gid, "teamid_Red"]]) in lp_pairs]
        for gid in g.gameid:
            keep = bool(confirmed) and gid == confirmed[0]
            rows.append({"gameid": gid, "issue": "duplicate records naming different team pairs",
                         "action": "keep" if keep else "drop",
                         "reason": "team pair confirmed by the Leaguepedia game" if keep
                         else (f"copy of {confirmed[0]}, whose pair Leaguepedia confirms" if confirmed else "no record's pair confirmed"),
                         "blue": W.loc[gid, "teamname_Blue"], "red": W.loc[gid, "teamname_Red"],
                         "league": W.loc[gid, "league_Blue"], "date": W.loc[gid, "date_Blue"]})
    # 4b statistics copied from another game (w0a): the OE result travelled with the copied statistics, so the record is
    #   kept only when Leaguepedia's matched game records the same winner
    gone = {r["gameid"] for r in rows if r["action"] in ("drop", "exclude", "set_winner")} | set(checks.loc[checks.action == "drop", "gameid"])
    dup_group = checks[checks.check == "duplicate"].set_index("gameid").group
    ch = pd.concat([pd.read_csv(f, usecols=["gameid", "position", "side", "champion"], low_memory=False) for f in files])
    ch = ch[ch.position.isin(["top", "jng", "mid", "bot", "sup"]) & ch.gameid.isin(set(checks.gameid))]
    side_draft = ch.groupby("gameid").apply(lambda g: {sd: frozenset(x.champion) for sd, x in g.groupby("side")})
    for gid in sorted(copied - gone):
        same_game = checks.loc[(checks.check == "duplicate") & (checks.group == dup_group.get(gid)), "gameid"]
        via = next((o for o in sorted(same_game) if o in M.index), None) if gid not in M.index else None
        if via is not None:                            # the other record must hold the same draft on the same sides
            assert side_draft[gid] == side_draft[via], ("evidence through a record with another side draft", gid, via)
        ev = evidence(gid, via)
        ok = ev["lp"] is not None and ev["lp"] == ev["oe"]
        rows.append({"gameid": gid, "issue": "statistics copied from another game", "action": "keep" if ok else "exclude",
                     "winner_side": ev["oe"] if ok else None,
                     "reason": "Leaguepedia records the same winner" if ok else "result not confirmed by Leaguepedia", **ev})
    # 5 team labels: Leaguepedia names another team for a side, and the side's players belong to that team's regular
    #   roster rather than the recorded team's (e.g. an academy roster recorded under the main team's ID): relabel
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    pcols = ["gameid", "date", "position", "side", "teamid", "playername"]
    Pl = pd.concat([pd.read_csv(f, usecols=pcols, low_memory=False) for f in files])
    Pl = Pl[Pl.position.isin(["top", "jng", "mid", "bot", "sup"])].copy()
    Pl["date"] = pd.to_datetime(Pl.date, errors="coerce")
    done = {r["gameid"] for r in rows if r.get("action") in ("drop", "exclude")} | set(checks.loc[checks.action == "drop", "gameid"])

    def core(teamid, when, exclude):
        """players in at least 20% of the team's other games within 180 days"""
        g = Pl[(Pl.teamid == teamid) & (Pl.gameid != exclude) & ((Pl.date - when).abs() <= pd.Timedelta(days=180))]
        n = g.gameid.nunique()
        c = g.groupby("playername").gameid.nunique()
        return set(c[c >= 0.2 * n].index) if n else set()
    for gid, m in M.iterrows():
        if gid in done or gid not in W.index:
            continue
        w = W.loc[gid]
        team1_side = "Blue" if m.side == "team1_blue" else "Red"
        other = "Red" if team1_side == "Blue" else "Blue"
        for lpname, sd in ((m.Team1, team1_side), (m.Team2, other)):
            lid = link(lpname, m.year)
            oid = w[f"teamid_{sd}"]
            if lid is None or lid == oid or norm(lpname) == norm(w[f"teamname_{sd}"]):
                continue
            when = pd.Timestamp(w.date_Blue)
            players = set(Pl[(Pl.gameid == gid) & (Pl.side == sd)].playername)
            in_lp, in_oe = len(players & core(lid, when, gid)), len(players & core(oid, when, gid))
            relabel = in_lp >= 3 and in_oe <= 1
            newname = links.loc[(links.teamid == lid), "teamname"].iloc[0]
            rows.append({"gameid": gid, "issue": "team label", "action": "relabel" if relabel else "keep",
                         "side": sd, "old_teamid": oid, "new_teamid": lid if relabel else "", "new_teamname": newname if relabel else "",
                         "reason": f"{in_lp} of the players in the regular roster of Leaguepedia's team ({newname}), {in_oe} in the recorded team's",
                         "blue": w.teamname_Blue, "red": w.teamname_Red, "league": w.league_Blue, "date": w.date_Blue})
    # 6 plain duplicates and copies decided in w0a
    for r in checks[checks.action == "drop"].itertuples():
        rows.append({"gameid": r.gameid, "issue": f"{r.check}: {r.detail}", "action": "drop", "reason": f"copy of {r.group}",
                     "league": r.league, "date": r.date})
    C = pd.DataFrame(rows)
    C["winner_teamid"] = [W.loc[g, f"teamid_{s}"] if isinstance(s, str) and s in ("Blue", "Red") else ""
                          for g, s in zip(C.gameid, C.get("winner_side", pd.Series([None] * len(C))))]
    C.to_csv(MAN / "oe_corrections.csv", index=False)
    print(C.groupby(["issue", "action"]).size().to_string())
    # the statistics line checked against every OE game of 2016-2025 that records exactly one winner
    one = W[(W.result_Blue + W.result_Red == 1) & W.date_Blue.astype(str).str[:4].astype(int).isin(YEARS)]
    dt, dg = one.towers_Blue - one.towers_Red, one.totalgold_Blue - one.totalgold_Red
    stats = np.where((dt >= 2) & (dg > 0), "Blue", np.where((dt <= -2) & (dg < 0), "Red", ""))
    rec = np.where(one.result_Blue == 1, "Blue", "Red")
    ok = stats != ""
    summary = {"stats_rule_games_one_winner": int(len(one)), "stats_rule_conclusive": int(ok.sum()),
               "stats_rule_agree_pct": round(100 * float((stats[ok] == rec[ok]).mean()), 2),
               "actions": {f"{i} | {a}": int(n) for (i, a), n in
                           C.assign(kind=C.issue.str.split(":").str[0]).groupby(["kind", "action"]).size().items()}}
    (MAN / "oe_corrections_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print({k: v for k, v in summary.items() if k != "actions"})


if __name__ == "__main__":
    main()
