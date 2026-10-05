"""Post-hoc incidence/timing summaries for governance framing; no new inference.

Reads the corrected outputs and coded release metadata. Does not read the hidden
human-coding key or overwrite the historical forecast or any fitted analysis.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from worlds_common import OUT, ROOT, event_manifest


def extremes(frame, metric, direction):
    """Retain every numerical tie; never select a case using its tournament result."""
    bound = frame.groupby("year")[metric].transform(direction)
    return frame[np.isclose(frame[metric], bound, rtol=0, atol=1e-10)].sort_values(
        ["year", "team"]
    ).copy()


def main():
    dest = OUT / "governance_framing"
    dest.mkdir(exist_ok=True)
    inputs = [OUT / name for name in (
        "team_patch_exposure.csv", "knockout_patch_exposure.csv", "worlds_games_pred.parquet"
    )]
    inputs += [ROOT / "data/manifests/worlds_events.csv"]
    inputs += [ROOT / f"data/patch_notes/coded_{letter}.csv" for letter in "ABCDE"]
    E = pd.read_csv(inputs[0]).dropna(subset=["intended"])
    K = pd.read_csv(inputs[1])
    G = pd.read_parquet(inputs[2])
    assert len(E) == 215 and len(K) == 80 and len(G) == 1097
    assert not E.duplicated(["year", "teamid"]).any()
    assert not K.duplicated(["year", "teamid"]).any()
    E = E.merge(K[["year", "teamid", "finish"]], on=["year", "teamid"],
                how="left", validate="one_to_one")
    E["finish"] = E.finish.fillna("Before quarterfinals")
    E["net_rank_lowest_first"] = E.groupby("year").intended.rank(method="min")
    E["nerf_rank_highest_first"] = E.groupby("year").nerfs.rank(ascending=False, method="min")
    E = E.sort_values(["year", "net_rank_lowest_first", "team"])
    E.to_csv(dest / "all_eligible_teams_ranked.csv", index=False)
    frames = {}
    for population, frame in [("all_eligible", E), ("quarterfinalists", E[E.finish != "Before quarterfinals"])]:
        for label, metric, direction in [("lowest_net", "intended", "min"),
                                          ("highest_nerf", "nerfs", "max"),
                                          ("highest_buff", "buffs", "max")]:
            selected = extremes(frame, metric, direction)
            frames[f"{label}_{population}"] = selected
            selected.to_csv(dest / f"{label}_{population}.csv", index=False)

    # Verify encounters against the actual games, not the finish-label ordering.
    encounters = []
    for row in frames["lowest_net_all_eligible"].itertuples():
        champion = K[(K.year == row.year) & (K.finish == "Champion")].iloc[0]
        games = G[(G.year == row.year) & (
            ((G.blue_id == row.teamid) & (G.red_id == champion.teamid)) |
            ((G.red_id == row.teamid) & (G.blue_id == champion.teamid))
        )]
        wins = np.where(games.blue_id == row.teamid, games.result, 1 - games.result)
        # Before 2018 a pair can meet in groups and knockouts. Export individual
        # game dates and results so an early encounter cannot be called elimination.
        for game, won in zip(games.itertuples(), wins):
            encounters.append({"year": row.year, "team": row.team,
                               "champion": champion.team, "finish": row.finish,
                               "gameid": game.gameid, "date": str(game.date),
                               "team_won": int(won)})
    pd.DataFrame(encounters).to_csv(dest / "lowest_net_vs_champion_games.csv", index=False)

    coding = pd.concat([pd.read_csv(p, dtype={"patch_oe": str}) for p in inputs[4:]])
    timing = []
    for year, event in event_manifest().iterrows():
        rows = coding[(coding.year_group == year) & (coding.patch_oe == event.main_patch)
                      & (coding.hotfix == 0)]
        dates = pd.to_datetime(rows.release_date).drop_duplicates()
        assert len(dates) == 1, f"Conflicting baseline release dates for {year}"
        released = dates.iloc[0]
        timing.append({"year": year, "worlds_patch_oe": event.main_patch,
                       "coded_release_date": str(released.date()),
                       "worlds_start": str(event.start.date()),
                       "calendar_days": int((event.start - released).days),
                       "source_urls": " | ".join(sorted(rows.source_url.dropna().unique()))})
    T = pd.DataFrame(timing)
    T.to_csv(dest / "worlds_patch_timing.csv", index=False)
    worst = frames["lowest_net_all_eligible"]
    worst_qf = frames["lowest_net_quarterfinalists"]
    highest_nerf = frames["highest_nerf_all_eligible"]
    assert worst.groupby("year").size().eq(1).all(), "Table must display tied extremes"
    assert worst_qf.groupby("year").size().eq(1).all()
    summary = {
        "analysis_status": "post-hoc descriptive extension; no causal loss ranking",
        "eligible_team_years": len(E),
        "lowest_net_all_eligible_finishes": worst.finish.value_counts().to_dict(),
        "lowest_net_quarterfinalists_finishes": worst_qf.finish.value_counts().to_dict(),
        "lowest_net_equals_highest_nerf_years": worst.merge(highest_nerf, on=["year", "teamid"]).year.tolist(),
        "lead_days_min": int(T.calendar_days.min()),
        "lead_days_max": int(T.calendar_days.max()),
        "lead_days_median": float(T.calendar_days.median()),
        "timing_scope": "coded baseline patch release dates; not first notice, private access, or hotfix timing",
        "extreme_tie_counts": {key: frame.groupby("year").size().to_dict() for key, frame in frames.items()},
    }
    (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    label = lambda p: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else "OUT/" + str(p.relative_to(OUT))
    provenance = {label(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    provenance[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (dest / "input_hashes.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(worst[["year", "team", "intended_z", "nerf_share", "n_unseen_patches", "finish"]].to_string(index=False))
    print(T[["year", "coded_release_date", "worlds_start", "calendar_days"]].to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "extreme_tie_counts"}, indent=2))


if __name__ == "__main__":
    main()
