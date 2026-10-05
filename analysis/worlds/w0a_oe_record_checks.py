"""W0a: internal consistency checks of the Oracle's Elixir (OE) records, before any analysis.

Four kinds of records fail:
  duplicate    the same game stored under several game IDs: either identical player rows (players, champions, results),
               game length and start time, or the same draft (the two teams' champions) with the same game length and
               the same kills, deaths, assists, gold and result for every player (player names and a few seconds of
               start time may differ). One record is kept, preferring the one whose team-row picks agree with its
               player rows, then the lowest game number; the others are dropped (they occupy another game's slot, which
               the Leaguepedia supplement can then restore). When the records name different team pairs, OE alone
               cannot tell which teams played: w0c keeps the record whose pair Leaguepedia confirms, else none.
  stats_copy   records with the same game length and player statistics but different drafts: the statistics (and
               the result stored with them) were copied from another game. w0c keeps such a record only when
               Leaguepedia's matched game records the same winner.
  mixed        the team rows' picks disagree with the player rows' champions, so the team rows (picks, bans) belong to
               another game; the team-row bans are not used and w0c checks the winner. When the player rows also repeat
               another record of the same day, the record is a copy of that game and is dropped.
  no_winner    both team rows record a loss. The winner is decided in w0c from independent evidence, or the game is
               excluded from the analyses.
Safeguards (the script stops if one fails): statistics are compared only for records with ten distinct side-role rows and
complete values, and every duplicate group must hold one draft, start within one day, and have ten player rows per
record.
Writes data/manifests/oe_record_checks.csv (one row per flagged game ID and check). The source files are never modified.
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTFILE = ROOT / "data/manifests/oe_record_checks.csv"
ROLES = ["top", "jng", "mid", "bot", "sup"]


def norm(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def main():
    cols = ["gameid", "league", "year", "date", "game", "position", "side", "teamname", "result", "champion", "gamelength",
            "playername", "kills", "deaths", "assists", "totalgold", "pick1", "pick2", "pick3", "pick4", "pick5"]
    files = sorted((ROOT / "data").glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
    D = pd.concat([pd.read_csv(f, usecols=cols, low_memory=False) for f in files], ignore_index=True)
    P = D[D.position.isin(ROLES)]
    T = D[D.position == "team"].drop_duplicates(["gameid", "side"])

    # mixed: team-row picks (when all five are recorded) differ from the player rows' champions on either side
    pc = P.groupby(["gameid", "side"]).champion.apply(lambda x: frozenset(norm(c) for c in x))
    tp = T.set_index(["gameid", "side"])[[f"pick{i}" for i in range(1, 6)]].apply(
        lambda r: frozenset(norm(c) for c in r if pd.notna(c)), axis=1)
    both = pd.concat({"players": pc, "team": tp}, axis=1).dropna()
    both = both[both.team.map(len) == 5]
    mixed = set(both[both.players != both.team].reset_index().gameid)

    # duplicate: (a) identical player rows, game length and start time, or (b) the same draft, game length and player
    # statistics (records with complete statistics only); records linked by either are one game
    fp = P.sort_values(["gameid", "side", "position"]).groupby("gameid").apply(
        lambda g: (tuple(zip(g.playername.astype(str), g.champion.astype(str), g.result)), g.gamelength.iloc[0], str(g.date.iloc[0])))
    stat_cols = ["kills", "deaths", "assists", "totalgold", "result"]
    full = P.groupby("gameid").apply(lambda g: len(g) == 10 and len(g[["side", "position"]].drop_duplicates()) == 10
                                     and g[stat_cols].notna().all().all() and pd.notna(g.gamelength.iloc[0]))
    Ps = P[P.gameid.isin(full[full].index)]
    stat = Ps.groupby("gameid").apply(lambda g: (int(g.gamelength.iloc[0]), tuple(sorted(zip(*(g[c].astype(int) for c in stat_cols))))))
    draft = Ps.groupby("gameid").apply(lambda g: frozenset(frozenset(norm(c) for c in s.champion) for _, s in g.groupby("side")))
    parent = {g: g for g in fp.index}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for key_series in (fp, pd.Series(list(zip(stat, draft.loc[stat.index])), index=stat.index)):
        for _, ids in key_series.groupby(key_series).groups.items():
            ids = list(ids)
            for other in ids[1:]:
                parent[find(other)] = find(ids[0])
    comp = pd.Series({g: find(g) for g in fp.index})
    # safeguards: a merged group must be one game (same draft where known, start times within a day)
    when = pd.to_datetime(P.groupby("gameid").date.first())
    side_roles = P.groupby("gameid").apply(lambda g: len(g[["side", "position"]].drop_duplicates()))
    for key, ids in comp[comp.duplicated(keep=False)].groupby(comp).groups.items():
        ids = list(ids)
        assert all(side_roles.get(i, 0) == 10 and (P.gameid == i).sum() == 10 for i in ids), ("duplicate group with incomplete player rows", ids)
        drafts = {draft[i] for i in ids if i in draft.index}
        assert len(drafts) <= 1, ("duplicate group with several drafts", ids)
        assert (when[ids].max() - when[ids].min()) <= pd.Timedelta(days=1), ("duplicate group spans more than a day", ids)
    info = T.drop_duplicates("gameid").set_index("gameid").loc[fp.index, ["league", "year", "date", "game"]].assign(fp=comp)
    info["teams"] = T.groupby("gameid").teamname.apply(lambda x: " vs ".join(map(str, x)))
    rows = []
    for key, g in info[info.fp.duplicated(keep=False)].groupby("fp"):
        g = g.assign(consistent=~g.index.isin(mixed), gid=g.index).sort_values(["consistent", "game", "gid"], ascending=[False, True, True])
        keep = g.index[0]
        # records that name different team pairs cannot be resolved from OE alone: w0c decides with Leaguepedia
        pairs = {frozenset(x.split(" vs ")) for x in g.teams}
        for gid, r in g.iterrows():
            action = "resolve teams in w0c" if len(pairs) > 1 else ("keep" if gid == keep else "drop")
            rows.append({"gameid": gid, "check": "duplicate", "action": action, "group": keep,
                         "league": r.league, "year": r.year, "date": r.date, "game": r.game, "teams": r.teams,
                         "detail": f"{len(g)} records of one game (identical player rows, or the same draft, game length and "
                                   f"player statistics); team-row picks {'agree' if r.consistent else 'disagree'} with player rows"})
    dup_ids = {r["gameid"] for r in rows}
    # stats_copy: the same game length and player statistics under different drafts
    sd = pd.DataFrame({"stat": stat, "draft": draft.loc[stat.index]})
    multi = sd.groupby("stat").draft.transform("nunique") > 1
    for key, g in sd[multi].groupby("stat"):
        rep = sorted(g.index)[0]
        for gid in sorted(g.index):
            r = info.loc[gid]
            rows.append({"gameid": gid, "check": "stats_copy", "action": "decide in w0c", "group": rep,
                         "league": r.league, "year": r.year, "date": r.date, "game": r.game, "teams": r.teams,
                         "detail": f"{len(g)} records share game length and player statistics under {g.draft.nunique()} drafts"})
    # a mixed record whose player rows (players and champions) repeat another record of the same day is a copy of that
    # game joined to another game's team rows: dropped
    Pd = P.assign(day=pd.to_datetime(P.date).dt.date)
    pfp = Pd.sort_values(["gameid", "side", "position"]).groupby("gameid").apply(
        lambda g: hash(tuple(zip(g.playername.astype(str), g.champion.astype(str)))))
    pday = Pd.groupby("gameid").day.first()
    for gid in sorted(mixed - dup_ids):
        r = info.loc[gid] if gid in info.index else T[T.gameid == gid].iloc[0]
        twin = [o for o in pfp.index[(pfp == pfp[gid]) & (pday == pday[gid])] if o != gid]
        rows.append({"gameid": gid, "check": "mixed", "league": r.league, "year": r.year, "date": r.date, "game": r.game,
                     "teams": info.teams.get(gid, ""), "group": twin[0] if twin else "",
                     "action": "drop" if twin else "ignore team-row picks and bans; winner checked in w0c",
                     "detail": ("player rows repeat " + twin[0] + " of the same day; team rows belong to another game") if twin
                     else "team-row picks disagree with the player rows' champions"})
    res = T.groupby("gameid").result.agg(["sum", "size"])
    for gid in res[(res["size"] == 2) & (res["sum"] != 1)].index:
        r = info.loc[gid]
        rows.append({"gameid": gid, "check": "no_winner", "action": "decide in w0c", "group": "", "league": r.league,
                     "year": r.year, "date": r.date, "game": r.game, "teams": r.teams,
                     "detail": f"team-row results sum to {int(res.loc[gid, 'sum'])}"})
    R = pd.DataFrame(rows)
    R.to_csv(OUTFILE, index=False)
    print(R.groupby(["check", "action"]).size().to_string())


if __name__ == "__main__":
    main()
