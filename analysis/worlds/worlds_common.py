"""Shared definitions for the revised manuscript pipeline.

The earlier output/worlds files remain a preserved historical analysis. Set
LOL_WORLDS_OUT to an alternate directory for an isolated reproduction.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("LOL_WORLDS_OUT", str(ROOT / "output/worlds_revision_2026-09-24")))
OUT.mkdir(parents=True, exist_ok=True)
SUPPLEMENT = ROOT / "data/supplement/leaguepedia_games.csv"
# international events (not a team's home league); RR = Rift Rivals and IWCQ = 2016 International Wildcard
# Qualifier appear only in the Leaguepedia supplement
INTL = {"WLDs", "MSI", "IEM", "EWC", "FST", "MSC", "IWCI", "WSCI", "ASCI", "Riot", "RR", "IWCQ"}


VOIDED = ROOT / "data/manifests/voided_games.csv"
CORRECTIONS = ROOT / "data/manifests/oe_corrections.csv"


def load_matches(usecols, years=None, dtype=None):
    """Oracle's Elixir rows plus the games of Worlds participants that Oracle's Elixir lacks, restored from
    Leaguepedia (data/supplement/leaguepedia_games.csv, built by w0_leaguepedia_supplement.py), without games the
    organiser voided and replayed (data/manifests/voided_games.csv; the replays are kept), and with the record
    corrections of data/manifests/oe_corrections.csv (w0a, w0c). `years` are calendar years
    (source file years); LOL_NO_SUPPLEMENT=1 reads Oracle's Elixir only (for comparisons)."""
    files = sorted((ROOT / "data").glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
    if years is not None:
        files = [f for f in files if int(f.name[:4]) in set(years)]
    cols = list(usecols) + (["side"] if "teamid" in usecols and "side" not in usecols else [])   # relabels are per side
    frames = [pd.read_csv(f, usecols=cols, dtype=dtype, low_memory=False) for f in files]
    if os.environ.get("LOL_NO_SUPPLEMENT") != "1":
        sup = pd.read_csv(SUPPLEMENT, dtype=dtype, low_memory=False)
        if years is not None:
            sup = sup[sup.year.isin(set(years))]
        frames.append(sup[cols])
    out = pd.concat(frames, ignore_index=True)
    if "gameid" in out.columns:
        out = out[~out.gameid.isin(set(pd.read_csv(VOIDED).gameid))]
        out = apply_oe_corrections(out)
    return out[list(usecols)].reset_index(drop=True)


def apply_oe_corrections(df):
    """data/manifests/oe_corrections.csv (w0a/w0c): drop copies and games whose result cannot be established, set
    winners supported by two independent lines of evidence, blank the bans of records whose team rows belong to
    another game, and relabel sides recorded under the wrong team"""
    C = pd.read_csv(CORRECTIONS)
    df = df[~df.gameid.isin(set(C.loc[C.action.isin(["drop", "exclude"]), "gameid"]))].copy()
    win = C[C.action == "set_winner"].set_index("gameid").winner_teamid
    if "result" in df.columns:
        if "teamid" not in df.columns:
            raise ValueError("load_matches: 'result' needs 'teamid' to apply the winner corrections")
        hit = df.gameid.isin(win.index)
        df.loc[hit, "result"] = (df.loc[hit, "teamid"] == df.loc[hit, "gameid"].map(win)).astype(int)
    bans = [c for c in df.columns if c.startswith("ban")]
    if bans:
        df.loc[df.gameid.isin(set(C.loc[C.action == "keep_ignore_team_bans", "gameid"])), bans] = np.nan
    rl = C[C.action == "relabel"]
    if len(rl) and "teamid" in df.columns:
        for r in rl.itertuples():                   # e.g. an academy roster recorded under the main team's ID
            game = df.gameid == r.gameid
            if not game.any():
                continue                            # game outside the rows loaded
            hit = game & (df.side == r.side)
            assert hit.any(), ("relabel not applied", r.gameid, r.side)
            df.loc[hit, "teamid"] = r.new_teamid
            if "teamname" in df.columns:
                df.loc[hit, "teamname"] = r.new_teamname
    return df


def coding_files():
    """Pre-Worlds champion coding tables. LOL_CODING_OVERRIDE substitutes one table (coding sensitivity, w26)."""
    alt = os.environ.get("LOL_CODING_OVERRIDE")
    return [Path(alt)] if alt else sorted((ROOT / "data/patch_notes").glob("coded_[A-E].csv"))


def sr_only(coding):
    """Drop coded changes that applied only to other modes (listed with reasons in data/patch_notes/non_sr_rows.csv).
    Matching on patch, champion and the exact summary keeps Summoner's Rift rows that merely mention other modes."""
    ex = pd.read_csv(ROOT / "data/patch_notes/non_sr_rows.csv", dtype={"patch_oe": str})
    drop = set(zip(ex.patch_oe, ex.champion, ex.summary))
    keep = [k not in drop for k in zip(coding.patch_oe.astype(str), coding.champion, coding.summary.astype(str))]
    return coding[keep].copy()


def assign_series(games):
    """(series id, game number within the series), aligned with `games` as given. A series starts when the pair of
    teams changes, more than 12 hours pass, the year changes, or the source numbers the game 1, so single-game
    tiebreakers between teams that have just met stay separate series."""
    W = games.sort_values("date", kind="stable")
    pair = pd.Series([tuple(sorted(p)) for p in zip(W.blue_id, W.red_id)], index=W.index)
    new = ((pair != pair.shift()) | (W.date.diff() > pd.Timedelta(hours=12)) | (W.year != W.year.shift())
           | (W["game"] == 1))
    series = new.cumsum()
    game_no = series.groupby(series).cumcount() + 1
    return series.reindex(games.index), game_no.reindex(games.index)


def champion_unavailable():
    """Champions that could not be picked or banned for all or part of a Worlds (data/manifests/
    champion_availability.csv: whole_event or partial with a UTC window). Returns a function (year, champion,
    game time) -> True when the champion was unavailable in that game."""
    A = pd.read_csv(ROOT / "data/manifests/champion_availability.csv")
    A = A[A.scope.isin(["whole_event", "partial"])]
    whole = set(zip(A.loc[A.scope == "whole_event", "year"], A.loc[A.scope == "whole_event", "champion"]))
    part = {}
    for r in A[A.scope == "partial"].itertuples():
        utc = lambda x: pd.Timestamp(x).tz_convert(None) if pd.Timestamp(x).tzinfo else pd.Timestamp(x)   # game times are naive UTC
        part.setdefault((r.year, r.champion), []).append((utc(r.unavailable_from_utc), utc(r.unavailable_to_utc)))

    def unavailable(year, champion, when):
        return (year, champion) in whole or any(a <= when < b for a, b in part.get((year, champion), []))
    unavailable.champion_years = whole | set(part)
    return unavailable


def event_manifest():
    return pd.read_csv(ROOT / "data/manifests/worlds_events.csv", index_col="year",
                       dtype={"main_patch": str}, parse_dates=["start", "end_exclusive"])


def select_worlds(games):
    """Select play-ins and main event; qualifiers remain in pre-event history."""
    frames = []
    for year, event in event_manifest().iterrows():
        keep = ((games.year == year) & (games.league == "WLDs")
                & (games.patch == event.main_patch)
                & (games.date >= event.start) & (games.date < event.end_exclusive))
        frames.append(games.loc[keep])
    return pd.concat(frames).sort_values("date").reset_index(drop=True)


def patch_key(patch):
    major, minor = str(patch).split(".")
    return int(major) * 100 + int(minor)


def team_delta(coding, year, last_patch):
    """Net coded change since this team's last played version, not release date.

    Empty windows are valid zero exposure (e.g. BDS after the 2023 WQS).
    """
    year_rows = coding[coding.year_group == year]
    unseen = year_rows[year_rows.patch_oe.map(patch_key) > patch_key(last_patch)]
    return unseen.groupby("champion").score.sum()
