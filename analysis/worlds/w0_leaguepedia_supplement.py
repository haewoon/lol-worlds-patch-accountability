"""W0: games of Worlds participants that Oracle's Elixir lacks, restored from Leaguepedia.

Oracle's Elixir (OE) has no record of some pre-Worlds games in 2016-2019 (e.g. Rift Rivals 2017-19, the 2016
International Wildcard Qualifier, the 2017 LPL summer playoffs and regional finals), so for some teams the last
official game, the 100-day champion pool and the rating history were wrong. This script restores them.

  fetch     Leaguepedia ScoreboardGames for the 365 days before each Worlds (the rating window) and, for the games
            to restore, ScoreboardPlayers; cached in data/leaguepedia_cache/ (not redistributed).
  build     1 match Leaguepedia games to OE games one-to-one by the set of ten picks (identical sets within 36 hours,
              the same game number of the series first, then nearest in time); link Leaguepedia team names to OE team IDs from these matches only, when at least 3
              games and 90% of the name's matches point to one ID (same year, else all years); then allow an
              unmatched game to match an unassigned OE game of the same two linked teams within 36 hours with 8+ of
              the 10 picks in common;
            2 restore an unmatched game in the 365 days before Worlds when a team of that year's Worlds played in it
              and the competition is of a kind OE records (regional leagues with their playoffs, qualifiers and cups;
              domestic and third-party cups such as the Demacia Cup, KeSPA Cup, NEST and IEM; Riot international
              events); national-team events, showmatches and one-off invitationals are not restored; where OE holds
              games of the same two teams within 36 hours, only the surplus is restored, with the series' OE patch;
              OE records that w0a found to be copies of another record are left out of the matching;
            3 write the games in OE's row layout (10 player rows + 2 team rows) to
              data/supplement/leaguepedia_games.csv, with linked OE team IDs (else lp:team:<name>);
            4 report agreement between the two sources on participants' games matched in both (bans, winner, patch)
              to data/supplement/leaguepedia_agreement.json.
Leaguepedia content is CC BY-SA 3.0 (https://lol.fandom.com); the supplement file carries that attribution.
"""
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/leaguepedia_cache"
SUP = ROOT / "data/supplement"
URL = "https://lol.fandom.com/wiki/Special:CargoExport"
HEAD = {"User-Agent": "lol-worlds-patch-accountability research (hwkwak@iu.edu)"}
EV = pd.read_csv(ROOT / "data/manifests/worlds_events.csv", index_col="year", parse_dates=["start"])
YEARS = range(2016, 2026)
GAME_FIELDS = ["OverviewPage", "Tournament", "DateTime_UTC", "Team1", "Team2", "Winner", "Patch", "Team1Picks",
               "Team2Picks", "Team1Bans", "Team2Bans", "N_GameInMatch", "GameId", "MatchId"]
PLAYER_FIELDS = ["GameId", "Team", "Side", "Role", "Champion", "Link", "Name"]


def cargo(table, fields, where, order):
    rows, off = [], 0
    while True:
        q = {"tables": table, "fields": ",".join(f"{table}.{f}" for f in fields), "where": where, "limit": "2000",
             "offset": str(off), "format": "csv", "order by": f"{table}.{order}"}
        for attempt in range(5):
            r = requests.get(URL, params=q, headers=HEAD, timeout=180)
            if r.status_code == 200 and r.content[:1] != b"<":
                break
            time.sleep(20 * (attempt + 1))
        r.raise_for_status()
        text = r.content.decode("utf-8")
        d = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False) if text.strip() else pd.DataFrame()
        rows.append(d)
        if len(d) < 2000:
            break
        off += 2000
        time.sleep(3)
    return pd.concat(rows, ignore_index=True)


def fetch_games():
    CACHE.mkdir(parents=True, exist_ok=True)
    for y in YEARS:
        out = CACHE / f"games_{y}.csv"
        if out.exists():
            continue
        s = EV.loc[y, "start"]
        a, b = (s - pd.Timedelta(days=366)).strftime("%Y-%m-%d"), (s + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        d = cargo("ScoreboardGames", GAME_FIELDS,
                  f"ScoreboardGames.DateTime_UTC >= '{a}' AND ScoreboardGames.DateTime_UTC < '{b}'", "DateTime_UTC")
        d.to_csv(out, index=False)
        print(y, len(d), flush=True)
        time.sleep(3)
    for y in YEARS:                     # the Worlds itself: links Leaguepedia team names to OE team IDs
        out = CACHE / f"worlds_{y}.csv"
        if out.exists():
            continue
        s = EV.loc[y, "start"]
        a, b = s.strftime("%Y-%m-%d"), (s + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        d = cargo("ScoreboardGames", GAME_FIELDS, f"ScoreboardGames.DateTime_UTC >= '{a}' AND ScoreboardGames.DateTime_UTC < '{b}'"
                  f" AND ScoreboardGames.OverviewPage LIKE '%World Championship%'", "DateTime_UTC")
        d.to_csv(out, index=False)
        print("worlds", y, len(d), flush=True)
        time.sleep(3)


if __name__ == "__main__" and sys.argv[1:2] == ["fetch"]:
    fetch_games()


# ---------------------------------------------------------------- build
ROLE = {"Top": "top", "Jungle": "jng", "Mid": "mid", "Bot": "bot", "Support": "sup"}
TOL = pd.Timedelta(hours=36)
WINDOW = pd.Timedelta(days=365)                 # the rating window; the fetch keeps one extra day of margin
LINK_MIN, LINK_SHARE = 3, 0.9                   # a name is linked to an OE team ID only on this much evidence


def norm(c):
    return re.sub(r"[^a-z0-9]", "", str(c).replace("&amp;", "&").lower())


def picks(v):
    return [x.strip().replace("&amp;", "&") for x in str(v).split(",") if x.strip() and x.strip() not in ("None", "Missing Data")]


def oe_games(y):
    """OE games in the window of Worlds y (366 days before to 60 days after the start): one row per game with the
    blue and red pick sets and team IDs, plus the raw rows for comparisons"""
    s = EV.loc[y, "start"]
    cols = ["gameid", "league", "date", "patch", "game", "side", "position", "teamid", "teamname", "champion",
            "ban1", "ban2", "ban3", "ban4", "ban5", "result"]
    raw = pd.concat([pd.read_csv(ROOT / f"data/{c}_LoL_esports_match_data_from_OraclesElixir.csv", usecols=cols,
                                 dtype={"patch": str}, low_memory=False) for c in (y - 1, y)])
    raw["date"] = pd.to_datetime(raw.date, errors="coerce")
    raw = raw[(raw.date >= s - WINDOW - pd.Timedelta(days=1)) & (raw.date < s + pd.Timedelta(days=60))]
    checks = pd.read_csv(ROOT / "data/manifests/oe_record_checks.csv")          # w0a: copies of another OE record
    raw = raw[~raw.gameid.isin(set(checks.loc[checks.action == "drop", "gameid"]))]
    pl = raw[raw.position.isin(ROLE.values())].dropna(subset=["champion"])
    sets = pl.groupby(["gameid", "side"]).champion.agg(lambda x: frozenset(norm(c) for c in x)).unstack()
    tm = raw[raw.position == "team"].drop_duplicates(["gameid", "side"]).pivot(index="gameid", columns="side", values="teamid")
    g = pl.groupby("gameid").date.first().to_frame().join(sets).join(tm.add_suffix("_id"))
    g = g.dropna(subset=["Blue", "Red"])
    g["key"] = [b | r for b, r in zip(g.Blue, g.Red)]
    g["game_no"] = raw[raw.position == "team"].drop_duplicates("gameid").set_index("gameid").game.reindex(g.index)
    g["pair"] = [frozenset([a, b]) if isinstance(a, str) and isinstance(b, str) else None for a, b in zip(g.Blue_id, g.Red_id)]
    return g, raw


def lp_games(y):
    lp = pd.concat([pd.read_csv(CACHE / f"games_{y}.csv", dtype=str, keep_default_na=False),
                    pd.read_csv(CACHE / f"worlds_{y}.csv", dtype=str, keep_default_na=False)]).drop_duplicates("GameId")
    lp = lp.rename(columns=lambda c: c.replace(" ", "_")).reset_index(drop=True)   # "DateTime UTC", "N GameInMatch"
    lp["when"] = pd.to_datetime(lp["DateTime_UTC"], errors="coerce")
    lp["p1"] = [frozenset(norm(c) for c in picks(v)) for v in lp.Team1Picks]
    lp["p2"] = [frozenset(norm(c) for c in picks(v)) for v in lp.Team2Picks]
    lp["key"] = [a | b for a, b in zip(lp.p1, lp.p2)]
    lp["game_no"] = pd.to_numeric(lp.N_GameInMatch, errors="coerce")
    return lp


def one_to_one(cand):
    """greedy one-to-one assignment from (priority, lp index, OE gameid) candidates, best priority first"""
    used_lp, used_oe, out = set(), set(), {}
    for _, i, gid in sorted(cand, key=lambda t: t[0]):
        if i not in used_lp and gid not in used_oe:
            used_lp.add(i); used_oe.add(gid); out[i] = gid
    return out


def exact_matches(lp, g):
    """stage 1: the same set of ten picks within 36 hours, one-to-one; the same game number in the series first (teams
    sometimes repeat a draft in the next game), then nearest in time"""
    by_key = {}
    for gid, key, d, n in zip(g.index, g.key, g.date, g.game_no):
        by_key.setdefault(key, []).append((gid, d, n))
    cand = [((not (n == k), abs(d - w)), i, gid) for i, key, w, k in zip(lp.index, lp.key, lp.when, lp.game_no)
            if len(key) == 10 and pd.notna(w) for gid, d, n in by_key.get(key, []) if abs(d - w) <= TOL]
    return one_to_one(cand)


def accept(counts):
    """(teamid, teamname, games, share) when one OE team ID has LINK_MIN+ games and LINK_SHARE+ of the name's games"""
    c = counts.groupby("teamid").size().sort_values()
    n, top = int(c.sum()), c.index[-1]
    if n >= LINK_MIN and c.iloc[-1] / n >= LINK_SHARE:
        name = counts.loc[counts.teamid == top, "teamname"].mode().iloc[0]
        return top, name, n, float(c.iloc[-1] / n)
    return None


class Links:
    """Leaguepedia name -> OE team ID, from stage-1 matches only; the same year first, then all years"""
    def __init__(self, pairs):
        self.pairs = pairs
        self.year = {k: accept(d) for k, d in pairs.groupby(["year", "lp"])}
        self.all = {k: accept(d) for k, d in pairs.groupby("lp")}

    def get(self, name, y):
        return self.year.get((y, name)) or self.all.get(name)

    def table(self):
        rows = [{"year": y, "lp": n, "teamid": v[0], "teamname": v[1], "games": v[2], "share": v[3]}
                for (y, n), v in self.year.items() if v]
        rows += [{"year": "all", "lp": n, "teamid": v[0], "teamname": v[1], "games": v[2], "share": v[3]}
                 for n, v in self.all.items() if v]
        return pd.DataFrame(rows)


def match_all():
    """stage 1 for every year, name links from it, then stage 2: an unmatched game may match an unassigned OE game
    of the same two linked teams within 36 hours with at least 8 of the 10 picks in common (a pick recorded
    differently in one source), one-to-one"""
    data, pairs = {}, []
    for y in YEARS:
        lp, (g, raw) = lp_games(y), oe_games(y)
        m1 = exact_matches(lp, g)
        lp["oe_gameid"], lp["how"] = lp.index.map(m1), np.where(lp.index.isin(list(m1)), "exact", None)
        for i, gid in m1.items():
            r, o = lp.loc[i], g.loc[gid]
            blue_is_1 = len(o.Blue & r.p1) >= len(o.Blue & r.p2)
            for name, side in ((r.Team1, "Blue" if blue_is_1 else "Red"), (r.Team2, "Red" if blue_is_1 else "Blue")):
                pairs.append({"year": y, "lp": name, "teamid": o[f"{side}_id"]})
        data[y] = (lp, g, raw)
    # team names for the link table (one lookup per team ID)
    P = pd.DataFrame(pairs)
    names = {}
    for y in YEARS:
        raw = data[y][2]
        t = raw[raw.position == "team"][["teamid", "teamname"]].dropna().drop_duplicates("teamid")
        names.update(dict(zip(t.teamid, t.teamname)))
    P["teamname"] = P.teamid.map(names)
    P = P.dropna(subset=["teamid"])
    L = Links(P)
    for y in YEARS:
        lp, g, raw = data[y]
        used = set(lp.oe_gameid.dropna())
        free = g[~g.index.isin(used) & g.pair.notna()]
        cand = []
        for i, r in lp[lp.oe_gameid.isna() & lp.when.notna()].iterrows():
            a, b = L.get(r.Team1, y), L.get(r.Team2, y)
            if not a or not b or a[0] == b[0]:
                continue
            near = free[(free.pair == frozenset([a[0], b[0]])) & ((free.date - r.when).abs() <= TOL)]
            for gid, o in near.iterrows():
                ov = len(o.key & r.key)
                if ov >= 8:
                    cand.append(((not (o.game_no == r.game_no), -ov, abs(o.date - r.when)), i, gid))
        m2 = one_to_one(cand)
        for i, gid in m2.items():
            lp.loc[i, ["oe_gameid", "how"]] = [gid, "pair8"]
        # sides; for stage 2 from the linked team IDs, for stage 1 from the picks
        side = []
        for r in lp.itertuples():
            if not isinstance(r.oe_gameid, str):
                side.append(None); continue
            o = g.loc[r.oe_gameid]
            if r.how == "pair8":
                side.append("team1_blue" if L.get(r.Team1, y)[0] == o.Blue_id else "team1_red")
            else:
                side.append("team1_blue" if len(o.Blue & r.p1) >= len(o.Blue & r.p2) else "team1_red")
        lp["side"] = side
        # a stage-1 match whose linked teams disagree with the OE pair (recorded, excluded from agreement)
        conflict = []
        for r in lp.itertuples():
            a, b = L.get(r.Team1, y), L.get(r.Team2, y)
            conflict.append(bool(isinstance(r.oe_gameid, str) and a and b and g.loc[r.oe_gameid, "pair"] is not None
                                 and g.loc[r.oe_gameid, "pair"] != frozenset([a[0], b[0]])))
        lp["team_conflict"] = conflict
        lp["year"] = y
        data[y] = (lp, g, raw)
    return data, L


# Leaguepedia page prefix -> OE league label, for competitions of the kinds OE records (regional leagues with their
# playoffs, qualifiers and cups; domestic and third-party cups; Riot international events). Rift Rivals and the 2016
# International Wildcard Qualifier never appear in OE but are Riot events; they get labels treated as international.
LEAGUE = [(r"^LPL/", "LPL"), (r"^LCK/", "LCK"), (r"^LMS/", "LMS"), (r"^PCS/", "PCS"), (r"^VCS/", "VCS"),
          (r"^LCL/", "LCL"), (r"^TCL/", "TCL"), (r"^NA LCS/", "NA LCS"), (r"^EU LCS/", "EU LCS"), (r"^LEC/", "LEC"),
          (r"^LCS/", "LCS"), (r"^CBLOL/", "CBLOL"), (r"^LJL/", "LJL"), (r"^OPL/", "OPL"), (r"^LCO/", "LCO"),
          (r"^LLA/", "LLA"), (r"^LLN/", "LLN"), (r"^CLS/", "CLS"), (r"^(GPL|SEA Tour)/", "GPL"), (r"^LST/", "LST"),
          (r"Demacia Cup", "DCup"), (r"KeSPA Cup", "KeSPA"), (r"^National Electronic Sports Tournament \d{4}$", "NEST"),
          (r"^IEM ", "IEM"), (r"^Superliga ABCDE", "SL ABCDE"),       # Brazilian cup, not Spain's LVP SuperLiga
          (r"^(LVP )?SuperLiga", "LVP SL"), (r"Mid-Season Invitational", "MSI"), (r"Mid-Season Cup", "MSC"),
          (r"^Rift Rivals", "RR"), (r"International Wildcard Qualifier", "IWCQ")]
CUPS = {"DCup", "KeSPA", "NEST", "IEM", "SL ABCDE"}


def league_of(page):
    for pat, lab in LEAGUE:
        if re.search(pat, page):
            return lab
    return None                             # national teams, showmatches, one-off invitationals: not restored


def oe_patch(p):
    """Leaguepedia patch text -> OE style ('7.1' -> '7.01', '6.24b' -> '6.24', 2025's '25.17' -> '15.17')"""
    m = re.match(r"(\d+)\.(\d+)", str(p))
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return f"{a - 10 if a >= 25 else a}.{b:02d}"


def fetch_players(pages):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = []
    for page in sorted(pages):
        f = CACHE / ("players_" + re.sub(r"[^A-Za-z0-9]+", "_", page) + ".csv")
        if not f.exists():
            safe = page.replace("'", "\\'")
            d = cargo("ScoreboardPlayers", PLAYER_FIELDS + ["OverviewPage"], f"ScoreboardPlayers.OverviewPage = '{safe}'", "GameId")
            d.to_csv(f, index=False)
            time.sleep(3)
        out.append(pd.read_csv(f, dtype=str, keep_default_na=False))
    return pd.concat(out, ignore_index=True)


def build():
    W = pd.read_parquet(ROOT / "output/worlds_revision_2026-09-24/worlds_games_pred.parquet")
    data, L = match_all()
    restore, agree = [], []
    for y in YEARS:
        lp, g, raw = data[y]
        lp.drop(columns=["p1", "p2", "key"]).to_csv(CACHE / f"match_{y}.csv", index=False)
        s = EV.loc[y, "start"]
        in_window = (lp.when >= s - WINDOW) & (lp.when < s)
        # participants: Leaguepedia names in the games matched to this year's OE Worlds games; all must be linked
        wg = lp[lp.oe_gameid.isin(set(W.loc[W.year == y, "gameid"]))]
        part = set(wg.Team1) | set(wg.Team2)
        assert all(L.get(n, y) for n in part), (y, [n for n in part if not L.get(n, y)])
        assert {L.get(n, y)[0] for n in part} == set(W.loc[W.year == y, "blue_id"]) | set(W.loc[W.year == y, "red_id"]), y
        has_part = lp.Team1.isin(part) | lp.Team2.isin(part)
        miss = lp[in_window & lp.oe_gameid.isna() & has_part].copy()
        miss["league"] = miss.OverviewPage.map(league_of)
        restore.append(miss)
        # agreement on games both sources hold (participants' games in the window, one-to-one, teams consistent)
        both = lp[in_window & lp.oe_gameid.notna() & has_part & ~lp.team_conflict]
        tmr = raw[raw.position == "team"].drop_duplicates(["gameid", "side"])
        for r in both.itertuples():
            rows = tmr[tmr.gameid == r.oe_gameid].set_index("side")
            if set(rows.index) != {"Blue", "Red"}:
                continue
            lb, lr = (r.Team1Bans, r.Team2Bans) if r.side == "team1_blue" else (r.Team2Bans, r.Team1Bans)
            ob = [set(norm(c) for c in rows.loc[sd, ["ban1", "ban2", "ban3", "ban4", "ban5"]].dropna()) for sd in ("Blue", "Red")]
            win_blue = (r.Winner == "1") == (r.side == "team1_blue")
            po, pl = rows.loc["Blue", "patch"], oe_patch(r.Patch)      # compared only where both sources give a patch
            agree.append({"year": y, "how": r.how, "oe_gameid": r.oe_gameid, "team1_blue": r.side == "team1_blue",
                          "bans_equal": ob[0] == set(norm(c) for c in picks(lb)) and ob[1] == set(norm(c) for c in picks(lr)),
                          "winner_equal": bool(rows.loc["Blue", "result"] == 1) == win_blue,
                          "patch_equal": (str(po) == pl) if pd.notna(po) and pl else None,
                          "patch_versions_apart": abs(int(str(po).split(".")[0]) * 100 + int(str(po).split(".")[1])
                                                      - int(pl.split(".")[0]) * 100 - int(pl.split(".")[1]))
                          if pd.notna(po) and pl else None})
    R = pd.concat(restore, ignore_index=True)
    A = pd.DataFrame(agree)
    stats = {"matched_exact": int(sum((data[y][0].how == "exact").sum() for y in YEARS)),
             "matched_pair8": int(sum((data[y][0].how == "pair8").sum() for y in YEARS)),
             "exact_matches_with_conflicting_linked_teams": int(sum(data[y][0].team_conflict.sum() for y in YEARS)),
             "names_linked": int(len({n for (_, n), v in L.year.items() if v} | {n for n, v in L.all.items() if v}))}
    return R, L, A, stats


def oe_pairs():
    """every OE game: date, the pair of team IDs and the patch (to count games per pair around a candidate)"""
    rows = []
    for y in range(min(YEARS) - 1, max(YEARS) + 1):
        d = pd.read_csv(ROOT / f"data/{y}_LoL_esports_match_data_from_OraclesElixir.csv",
                        usecols=["gameid", "date", "teamid", "position", "patch"], dtype={"patch": str}, low_memory=False)
        rows.append(d[d.position == "team"])
    d = pd.concat(rows)
    checks = pd.read_csv(ROOT / "data/manifests/oe_record_checks.csv")          # count valid records only
    d = d[~d.gameid.isin(set(checks.loc[checks.action == "drop", "gameid"]))]
    d["date"] = pd.to_datetime(d.date, errors="coerce")
    g = d.groupby("gameid").agg(date=("date", "first"), ids=("teamid", lambda x: frozenset(x.dropna())), patch=("patch", "first"))
    return g[g.ids.map(len) == 2]


def select(R, L):
    """in-scope candidates with team IDs from the accepted links (unlinked names get an `lp:team:` ID); where OE holds
    games of the same two teams within 36 hours, restore only the surplus (Leaguepedia games of the pair nearby minus
    OE games of the pair nearby), because OE may hold the same game with a different pick record. A restored game from
    a series that OE partly records takes that series' OE patch (a series is played on one patch; Leaguepedia's manual
    patch entries occasionally differ)."""
    R = R[R.league.notna()].copy()
    for i in (1, 2):
        t = [L.get(n, y) or (f"lp:team:{n}", n) for n, y in zip(R[f"Team{i}"], R.year)]
        R[f"id{i}"], R[f"name{i}"] = [a[0] for a in t], [a[1] for a in t]
    O = oe_pairs()
    LPall = {y: pd.read_csv(CACHE / f"match_{y}.csv", dtype=str, keep_default_na=False, parse_dates=["when"]) for y in YEARS}
    keep = []
    for r in R.itertuples():
        same = O[(O.ids == frozenset([r.id1, r.id2])) & ((O.date - r.when).abs() <= TOL)]
        if len(same) == 0:
            keep.append((True, 0, 0, None)); continue
        series_patch = same.patch.dropna().mode()
        series_patch = series_patch.iloc[0] if len(series_patch) else None
        M = LPall[r.year]
        near = M[((M.Team1 == r.Team1) & (M.Team2 == r.Team2)) | ((M.Team1 == r.Team2) & (M.Team2 == r.Team1))]
        near = near[(near.when - r.when).abs() <= TOL]
        unmatched = near[near.oe_gameid == ""].sort_values("when")
        surplus = len(near) - len(same)
        keep.append((surplus > 0 and r.GameId in set(unmatched.GameId.iloc[len(unmatched) - max(surplus, 0):]), len(near),
                     len(same), series_patch))
    R["restore"], R["lp_pair_games_36h"], R["oe_pair_games_36h"] = [k[0] for k in keep], [k[1] for k in keep], [k[2] for k in keep]
    R["oe_series_patch"] = [k[3] for k in keep]
    return R


def oe_rows(R):
    """Leaguepedia games -> OE row layout (10 player rows + 2 team rows per game)"""
    PL = fetch_players(set(R.OverviewPage))
    PL = PL[PL.GameId.isin(set(R.GameId))]
    out = []
    for r in R.itertuples():
        base = {"gameid": "lp:" + r.GameId, "datacompleteness": "leaguepedia", "league": r.league, "year": r.when.year,
                "split": "", "playoffs": int("Playoffs" in r.OverviewPage), "date": r.when.strftime("%Y-%m-%d %H:%M:%S"),
                "game": int(r.N_GameInMatch) if str(r.N_GameInMatch).isdigit() else np.nan,
                "patch": r.oe_series_patch if isinstance(r.oe_series_patch, str) else oe_patch(r.Patch), "golddiffat15": np.nan,
                "source": "Leaguepedia ScoreboardGames/ScoreboardPlayers (CC BY-SA 3.0)"}
        for i, side in ((1, "Blue"), (2, "Red")):
            won = int(r.Winner == str(i))
            bans = picks(getattr(r, f"Team{i}Bans")) + [np.nan] * 5
            team = {"side": side, "teamid": getattr(r, f"id{i}"), "teamname": getattr(r, f"name{i}"), "result": won,
                    **{f"ban{k + 1}": bans[k] for k in range(5)}}
            out.append({**base, **team, "position": "team", "playerid": np.nan, "playername": np.nan, "champion": np.nan})
            pl = PL[(PL.GameId == r.GameId) & (PL.Side == str(i))]
            for q in pl.itertuples():
                if q.Role in ROLE:
                    out.append({**base, **team, "position": ROLE[q.Role], "playerid": np.nan, "playername": q.Name,
                                "champion": q.Champion.replace("&amp;", "&")})
    return pd.DataFrame(out)


def write_matching_extract():
    """Slim copies of the matching results that w28 reads, so the pipeline runs without the raw cache."""
    SUP.mkdir(parents=True, exist_ok=True)
    M = pd.concat([pd.read_csv(CACHE / f"match_{y}.csv", dtype=str, keep_default_na=False) for y in YEARS])
    M = M[M.oe_gameid != ""].drop_duplicates("oe_gameid")
    M[["year", "oe_gameid", "Team1", "Team2", "side"]].to_csv(SUP / "leaguepedia_oe_matches.csv", index=False)
    pd.read_csv(CACHE / "name_links.csv", dtype=str, keep_default_na=False).to_csv(SUP / "leaguepedia_name_links.csv", index=False)


if __name__ == "__main__" and sys.argv[1:2] == ["build"]:
    R, L, A, stats = build()
    L.table().to_csv(CACHE / "name_links.csv", index=False)
    write_matching_extract()
    S = select(R, L)
    S.drop(columns=["p1", "p2", "key"], errors="ignore").to_csv(CACHE / "restore_candidates.csv", index=False)
    G = S[S.restore]
    rows = oe_rows(G)
    n_players = rows[rows.position != "team"].groupby("gameid").size()
    SUP.mkdir(parents=True, exist_ok=True)

    def write_if_changed(df, path):                  # other scripts may be reading the file
        text = df.to_csv(index=False)
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
            print("wrote", path)
    write_if_changed(rows, SUP / "leaguepedia_games.csv")
    write_if_changed(G[["year", "OverviewPage", "league", "when", "Team1", "Team2", "id1", "id2", "Patch", "oe_series_patch", "Winner",
                        "GameId", "lp_pair_games_36h", "oe_pair_games_36h"]], SUP / "leaguepedia_restored_games.csv")
    unlinked = sorted(set(G.loc[G.id1.str.startswith("lp:team:"), "Team1"]) | set(G.loc[G.id2.str.startswith("lp:team:"), "Team2"]))
    summary = {"games_restored": int(len(G)), "player_rows_complete": int((n_players == 10).sum()),
               "candidates_out_of_scope": int(R.league.isna().sum()),
               "candidates_same_pair_in_oe_not_restored": int((~S.restore).sum()),
               "restored_patch_taken_from_oe_series": int(G.oe_series_patch.notna().sum()),
               "restored_by_year": G.groupby("year").size().to_dict(),
               "restored_by_league": G.groupby("league").size().to_dict(),
               "restored_unlinked_team_names": unlinked,
               **stats,
               "shared_games_compared": int(len(A)), "shared_games_unique_oe": int(A.oe_gameid.nunique()),
               "agreement_team1_is_blue": float(A.team1_blue.mean()),
               "agreement_bans_same_sets": float(A.bans_equal.mean()),
               "agreement_winner": float(A.winner_equal.mean()),
               "agreement_patch": float(A.patch_equal.dropna().astype(bool).mean()),
               "patch_compared": int(A.patch_equal.notna().sum()),
               "patch_disagreements_one_version_apart": int((A.patch_versions_apart == 1).sum()),
               "patch_disagreements": int((A.patch_equal == False).sum()),
               "link_rule": f"name -> OE team ID from exact matches only, at least {LINK_MIN} games and {LINK_SHARE:.0%} on one ID "
                            "(same year, else all years); otherwise an lp:team: ID",
               "rule": "Leaguepedia games in the 365 days before each Worlds with a team of that Worlds, not matched to an OE "
                       "game (exact ten picks within 36 hours one-to-one, or 8+ of 10 picks with the same linked team pair), "
                       "from competitions of the kinds OE records; where OE holds games of the same pair within 36 hours, "
                       "only the surplus is restored, with the series' OE patch."}
    (SUP / "leaguepedia_agreement.json").write_text(json.dumps(summary, indent=2, default=int) + "\n")
    print(json.dumps(summary, indent=2, default=int))
