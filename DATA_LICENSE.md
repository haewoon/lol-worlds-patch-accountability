# Licenses and data sources

## This repository

| Files | License |
|---|---|
| Code: `analysis/`, `paper/figures.py` | MIT (`LICENSE`) |
| Coded patch notes and prompts (`data/patch_notes/`), manifests written for this study (`data/manifests/`), validation data (`paper/validation/`), analysis outputs (`output/`), manuscript and figures (`paper/`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Files derived from Leaguepedia (`data/supplement/`) | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), as required by the source |

Please cite the paper when using these files.

## Third-party sources

- **Oracle's Elixir** (Tim Sevenhuysen, <https://oracleselixir.com>). The match files are not redistributed. `analysis/download_oracles_elixir.py` downloads them from the public folder that Oracle's Elixir links to; use them under that site's terms and credit Oracle's Elixir. Output files contain statistics computed from these data, and `data/supplement/leaguepedia_games.csv` follows the Oracle's Elixir column format.
- **Leaguepedia** (<https://lol.fandom.com>), content licensed CC BY-SA 3.0. The games in `data/supplement/` were restored from Leaguepedia's ScoreboardGames and ScoreboardPlayers tables, and the matching extracts there record Leaguepedia team names and game identifiers. The downloaded tables themselves are not included.
- **Liquipedia** (<https://liquipedia.net/leagueoflegends/>), content licensed CC BY-SA 3.0, and **Games of Legends** (<https://gol.gg>). `data/manifests/oe_external_evidence.csv` records series scores and game results from these sites with links to the pages consulted. The pages themselves are not included.
- **Riot Games.** Official patch notes (<https://www.leagueoflegends.com>), the League of Legends Wiki, and Riot's Data Dragon files were read to code and check champion changes. The `summary` column of the coded files paraphrases the patch notes in at most 20 words. Data Dragon and CommunityDragon files are not included; `output/.../gamedata_direction_rows.csv` records only which values changed and in which direction.

This project isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.
