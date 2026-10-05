# Analysis outputs

`analysis/worlds/run_revised_pipeline.py` writes `worlds_revision_2026-09-24/`, and `analysis/worlds/w26_coding_sensitivity.py` writes `coding_sensitivity/`, which holds one full rerun per alternative coding (A, B, C) and `coding_sensitivity_summary.csv` (Table 5).

## Read by the manuscript checks or the figures

`analysis/worlds/verify_manuscript_numbers.py` and `paper/figures.py` read these files (and `governance_framing/`, the tables of Section 4.4).

- `ban_pressure_result_models.csv`
- `ban_pressure_targeting.csv`
- `ban_pressure_team_summary.csv`
- `ban_pressure_team_tests.csv`
- `ban_pressure_terciles.csv`
- `ban_pressure_within_team.csv`
- `buff_persistence_champion_presence.csv`
- `buff_persistence_mechanism.csv`
- `buff_persistence_mechanism_inference.csv`
- `buff_persistence_org_influence.csv`
- `buff_persistence_teams.csv`
- `buff_persistence_tests.csv`
- `buff_sources.csv`
- `buff_sources_summary.json`
- `champion_patch_vs_presence.csv`
- `detectable_effect_power.csv`
- `detectable_effect_summary.csv`
- `gamedata_direction_crosstab.csv`
- `gamedata_direction_rows.csv`
- `gamedata_direction_summary.json`
- `historical_style_validation.csv`
- `historical_style_validation_by_year.csv`
- `knockout_patch_exposure.csv`
- `manuscript_descriptives.json`
- `patch_models.csv`
- `permutation_null_calibration.csv`
- `presence_reference_sensitivity.csv`
- `rating_grid.csv`
- `rating_validation_game.csv`
- `rating_validation_grid.csv`
- `rating_validation_region.csv`
- `rating_validation_region_control.csv`
- `rating_validation_series.csv`
- `rating_validation_tournament.csv`
- `regional_champion_dispersion.json`
- `role_exposure_summary.csv`
- `role_exposure_summary.json`
- `roster_scan_summary.json`
- `stage_pooled_tests.csv`
- `strength_models.csv`
- `strength_sensitivity.csv`
- `team_exposure.csv`
- `team_patch_exposure.csv`
- `worlds_events.csv`
- `worlds_games_pred.parquet`

## Intermediate and supporting files

Inputs to later scripts, row-level detail behind reported summaries, simulation draws, and protocols.

- `ban_pressure_team_games.csv`
- `buff_persistence_orgs.csv`
- `manuscript_numerical_checks.json`
- `patch_coding_reliability.csv`
- `patch_coding_reliability_rows.csv`
- `permutation_calibration_protocol.json`
- `permutation_null_calibration_draws.csv`
- `power_protocol.json`
- `presence_by_release.csv`
- `rating_validation_calibration.csv`
- `role_exposure.csv`
- `roster_scan.csv`
- `stage_team_games.csv`
- `strength_exposure_corr.csv`
- `strength_teams.csv`
- `worlds_games_pred_season.parquet`

## Exploratory outputs not used in the paper

These files come from early analyses centered on the eventual winner. The paper does not use them and does not test case counts against such benchmarks (Section 4.4).

- `ban_pressure_champion_rank_table.csv`: winner ranks on the ban measures
- `ban_pressure_champion_ranks.csv`: winner ranks on the ban measures
- `buff_signal_breadth.csv`: winner-rank test of buff exposure and breadth
- `champion_buffed_pick_use.csv`: per-team use of buffed champions at Worlds
- `champion_patch_exposure.csv`: title-probability columns (title_pp_*) from a discarded approach; their intervals are not valid
- `champion_patch_rank_test.csv`: rank of the eventual winner among quarterfinalists against a uniform random benchmark
- `stage_champion_rank_table.csv`: winner ranks on the draft-stage measures
- `stage_champion_ranks.csv`: winner ranks on the draft-stage measures
- `stage_team_summary.csv`: team averages of the draft-stage measures
